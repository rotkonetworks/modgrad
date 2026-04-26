//! Patch-aware cross-attention — the bridge between byte-level and
//! patch-level representations.
//!
//! Two flavours, mirroring BLT paper §3.2.2 (encoder, eqs. 5–8) and
//! §3.3.1 (decoder, eqs. 9–12):
//!
//! - **Encoder cross-attn (`bytes → patches`):** queries are patch
//!   reps, keys/values are byte reps. Each patch query attends only
//!   to the bytes within that patch. The initial query for the first
//!   layer is the max-pool of byte reps in the patch (paper §4.8).
//!
//! - **Decoder cross-attn (`patches → bytes`):** queries are byte
//!   reps, keys/values are patch reps. Each byte attends to its
//!   containing patch and any patches before it under causal masking.
//!
//! Both directions share the structure:
//!   - pre-RMSNorm on Q (and on K/V via the source-side norm),
//!   - multi-head Q/K/V projections via `LinearResident`,
//!   - softmax over a per-query mask (per-patch for encoder,
//!     causal-up-to-containing-patch for decoder),
//!   - O projection back to the destination dim,
//!   - **no positional embeddings** inside cross-attn (paper §3.2.2),
//!   - **no residual added inside this module** — the caller (encoder
//!     or decoder layer) wraps the residual the same way the rest of
//!     the SDK does (see [`modgrad_transformer::resident::AttentionResident::forward`]
//!     and its `out_dev` doc-comment).
//!
//! Owner: agent noah.
//!
//! ## Scope
//!
//! Forward only for this slice. Backward will route through here once
//! the encoder / decoder forward passes are wired and the trainer
//! (agent ronan) has a place to call `.backward(...)`. The forward
//! contract is what `sasha` and `ronan` build on; landing it
//! unblocks both.
//!
//! ## Direction at construction time
//!
//! A `CrossAttention` instance is built for one direction
//! ([`CrossAttnDirection`]). Q/K/V/O projections and the pre-LN
//! weights are sized accordingly:
//!
//! | direction | Q in | K/V in | O out  |
//! |-----------|------|--------|--------|
//! | Encoder   | byte | byte   | patch  |
//! | Decoder   | byte | patch  | byte   |
//!
//! Encoder Q-input is `byte_dim` because the initial query is the
//! max-pool of byte reps inside the patch (the paper's `pool(h_b in P_j)`
//! step). Calling [`CrossAttention::forward_decoder`] on an
//! Encoder-direction instance returns an error, and vice versa. This
//! keeps the per-instance weight footprint minimal while giving
//! callers (sasha) a single type to plumb.
//!
//! ## Masking strategy
//!
//! Per-patch loop. For each query position we softmax over only the
//! valid key/value range:
//!   - encoder: keys are bytes `[boundaries[i] .. boundaries[i+1])`,
//!   - decoder: keys are patches `[0 .. owning_patch+1)` for byte at
//!     position `b` whose owning patch index is the first `i` with
//!     `boundaries[i+1] > b`.
//!
//! No score-masking-via-large-negative-value is needed because we
//! never compute scores for invalid positions in the first place.
//! This is also the cheapest path on `gfx1102`: each per-query
//! softmax operates on `≤ patch_len` (encoder) or `≤ owning+1`
//! (decoder) elements, both of which are small.

#![cfg(feature = "rocm")]
#![allow(clippy::too_many_arguments)]

use modgrad_compute::backend::{GpuVec, ResidencyError};
use modgrad_compute::neuron::{Linear, LinearResident, SimpleRng};
use modgrad_device::backend::ops::{
    matvec_resident, rms_norm_resident, softmax_resident,
};
use modgrad_device::backend::{HipBatch, HipBuffer};

// ─── Config ─────────────────────────────────────────────────

/// Direction of the cross-attention block.
///
/// See the module docs for the per-direction projection shape table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossAttnDirection {
    /// `bytes → patches` (encoder side, paper §3.2.2).
    Encoder,
    /// `patches → bytes` (decoder side, paper §3.3.1).
    Decoder,
}

/// Configuration for a single [`CrossAttention`] instance.
///
/// `head_dim * num_heads` is the multi-head inner size; it does not
/// have to equal `byte_dim` or `patch_dim` (the O projection brings
/// it back to the destination dim). Following the rest of the
/// modgrad attention stack, `head_dim` is *per head*, so the effective
/// inner width is `num_heads * head_dim`.
#[derive(Debug, Clone)]
pub struct CrossAttnConfig {
    /// Model dim of the byte-side stream. Encoder Q-input dim and
    /// encoder K/V-input dim, plus decoder Q-input dim and decoder
    /// O-output dim.
    pub byte_dim: usize,
    /// Model dim of the patch-side stream. Encoder O-output dim and
    /// decoder K/V-input dim.
    pub patch_dim: usize,
    pub num_heads: usize,
    /// Per-head dim. Total inner width = `num_heads * head_dim`.
    pub head_dim: usize,
    /// RMSNorm eps for the pre-LN steps.
    pub norm_eps: f32,
    /// Direction this instance is built for.
    pub direction: CrossAttnDirection,
}

impl CrossAttnConfig {
    /// Inner Q/K/V multi-head width.
    #[inline]
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Q-side input dim for this direction.
    #[inline]
    pub fn q_in_dim(&self) -> usize {
        match self.direction {
            // Encoder Q is computed from `pool(byte reps in patch)`,
            // which is `byte_dim`-shaped.
            CrossAttnDirection::Encoder => self.byte_dim,
            CrossAttnDirection::Decoder => self.byte_dim,
        }
    }

    /// K/V-side input dim for this direction.
    #[inline]
    pub fn kv_in_dim(&self) -> usize {
        match self.direction {
            CrossAttnDirection::Encoder => self.byte_dim,
            CrossAttnDirection::Decoder => self.patch_dim,
        }
    }

    /// O-side output dim for this direction.
    #[inline]
    pub fn o_out_dim(&self) -> usize {
        match self.direction {
            CrossAttnDirection::Encoder => self.patch_dim,
            CrossAttnDirection::Decoder => self.byte_dim,
        }
    }
}

// ─── Module ─────────────────────────────────────────────────

/// Patch-aware cross-attention block — see module docs.
pub struct CrossAttention {
    /// Q projection: `q_in_dim → inner_dim`.
    pub q_proj: LinearResident,
    /// K projection: `kv_in_dim → inner_dim`.
    pub k_proj: LinearResident,
    /// V projection: `kv_in_dim → inner_dim`.
    pub v_proj: LinearResident,
    /// O projection: `inner_dim → o_out_dim`.
    pub o_proj: LinearResident,
    /// Pre-LN weight on the Q-side input: `[q_in_dim]` of `1.0`.
    /// We use a constant-1 RMSNorm scale so the pre-LN is a pure
    /// normalisation; if a learnable scale is wanted in a follow-up,
    /// store it as a `HipBuffer` of `[q_in_dim]` and re-upload on
    /// optimizer step.
    pub q_norm_weight_dev: HipBuffer,
    /// Pre-LN weight on the K/V-side input: `[kv_in_dim]` of `1.0`.
    pub kv_norm_weight_dev: HipBuffer,
    /// Zero `[inner_dim]` buffer used as the matvec bias pointer for
    /// the per-head score matvec. Same trick as
    /// [`modgrad_transformer::resident::AttentionResident::zero_bias_dev`].
    pub zero_bias_dev: HipBuffer,
    /// `1 / sqrt(head_dim)` scale buffer of length `head_dim`,
    /// pre-baked so we can fold it into the Q post-projection step
    /// via `op_tensor_resident(Mul)` in a follow-up. Stored as a host
    /// scalar for now and applied on the host softmax-input copy.
    pub qk_scale: f32,
    pub num_heads: usize,
    pub head_dim: usize,
    pub byte_dim: usize,
    pub patch_dim: usize,
    pub norm_eps: f32,
    pub direction: CrossAttnDirection,
}

impl CrossAttention {
    /// Allocate a fresh cross-attention block with weight init from a
    /// deterministic seed.
    ///
    /// Weights use Xavier-like normal init scaled by `sqrt(2 / fan_in)`
    /// (the same recipe `SuperLinear::new` uses). Bias is zero. Pre-LN
    /// scale is uploaded as a buffer of `1.0`s. The caller can replace
    /// any field via the public struct members or by reconstructing
    /// from a host `Linear` using the `LinearResident::sync_weights_from`
    /// pattern.
    pub fn new(config: &CrossAttnConfig, seed: u64) -> Result<Self, ResidencyError> {
        let inner = config.inner_dim();
        let q_in = config.q_in_dim();
        let kv_in = config.kv_in_dim();
        let o_out = config.o_out_dim();

        let mut rng = SimpleRng::new(seed);

        let q = init_linear(&mut rng, q_in, inner);
        let k = init_linear(&mut rng, kv_in, inner);
        let v = init_linear(&mut rng, kv_in, inner);
        let o = init_linear(&mut rng, inner, o_out);

        let q_norm_weight = vec![1.0f32; q_in];
        let kv_norm_weight = vec![1.0f32; kv_in];

        let q_norm_weight_dev = upload_f32(&q_norm_weight)?;
        let kv_norm_weight_dev = upload_f32(&kv_norm_weight)?;

        // Zero bias for the per-head score matvec. `out_dim = max(patch_len)`
        // per call, but since the matvec only reads `out_dim` floats, an
        // upper bound of `inner_dim` (≥ `head_dim`) is always sufficient
        // for the per-head bias path used in this slice. Per-call code
        // reuses this buffer for both score and weighted-V matvecs.
        let zero_bias_host = vec![0.0f32; inner.max(o_out).max(q_in).max(kv_in)];
        let zero_bias_dev = upload_f32(&zero_bias_host)?;

        Ok(Self {
            q_proj: LinearResident::from_linear(&q)?,
            k_proj: LinearResident::from_linear(&k)?,
            v_proj: LinearResident::from_linear(&v)?,
            o_proj: LinearResident::from_linear(&o)?,
            q_norm_weight_dev,
            kv_norm_weight_dev,
            zero_bias_dev,
            qk_scale: 1.0f32 / (config.head_dim as f32).sqrt(),
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            byte_dim: config.byte_dim,
            patch_dim: config.patch_dim,
            norm_eps: config.norm_eps,
            direction: config.direction,
        })
    }

    /// Encoder direction (`bytes → patches`). For each patch `i` the
    /// query is the max-pool of byte reps within that patch (paper
    /// §4.8); keys and values are the byte reps within the same patch.
    ///
    /// `patch_boundaries[i]` is the start byte index of patch `i`, with
    /// `patch_boundaries[n_patches]` equal to the total byte count.
    /// Patch `i` therefore covers bytes `[boundaries[i] .. boundaries[i+1])`.
    ///
    /// Output is the post-O-projection tensor — **no residual is added
    /// inside this method**. Caller adds the residual themselves
    /// (see module docs).
    pub fn forward_encoder(
        &self,
        batch: &HipBatch,
        byte_reps: &GpuVec,
        patch_boundaries: &[usize],
        scratch: &mut CrossAttnScratch,
        patch_reps_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        if self.direction != CrossAttnDirection::Encoder {
            return Err(ResidencyError::WrongVariant {
                expected: "Encoder direction",
                got: "Decoder direction (instance was built for the other side)",
            });
        }
        let n_bytes = patch_boundaries.last().copied().unwrap_or(0);
        let n_patches = patch_boundaries.len().saturating_sub(1);

        debug_assert_eq!(byte_reps.len(), n_bytes * self.byte_dim);
        debug_assert_eq!(patch_reps_out.len(), n_patches * self.patch_dim);
        debug_assert!(scratch.fits(self));

        // Pull bytes to host once. The per-patch pool + scratch fill
        // works on host slices; the per-call resident dispatches read
        // from device buffers we upload below. This is the same shape
        // as the host-RoPE bounce in `AttentionResident::forward`:
        // small per-call D2H/H2D for a stage that is awkward to keep
        // resident, big stages stay on device.
        if scratch.byte_reps_host.len() < n_bytes * self.byte_dim {
            scratch.byte_reps_host.resize(n_bytes * self.byte_dim, 0.0);
        }
        byte_reps.copy_to_host(&mut scratch.byte_reps_host[..n_bytes * self.byte_dim]);

        // Ensure host-side patch out is sized.
        if scratch.patch_out_host.len() < n_patches * self.patch_dim {
            scratch.patch_out_host.resize(n_patches * self.patch_dim, 0.0);
        }

        for i in 0..n_patches {
            let p_start = patch_boundaries[i];
            let p_end = patch_boundaries[i + 1];
            debug_assert!(p_end > p_start, "empty patch at index {i}");
            debug_assert!(p_end <= n_bytes);
            let p_len = p_end - p_start;

            // Stage 1: max-pool of byte reps within the patch → query
            // (size `byte_dim`).
            let pooled = max_pool_bytes(
                &scratch.byte_reps_host[p_start * self.byte_dim..p_end * self.byte_dim],
                self.byte_dim,
                p_len,
            );
            scratch.q_input.copy_from(&pooled);

            // Stage 2: pre-LN on Q.
            unsafe {
                rms_norm_resident(
                    hip_buf(&scratch.q_input)?.device_ptr() as *const f32,
                    self.q_norm_weight_dev.device_ptr() as *const f32,
                    hip_buf_mut(&mut scratch.q_normed)?.device_ptr() as *mut f32,
                    1, self.byte_dim, self.norm_eps,
                )?;
            }
            batch.note_dispatch()?;

            // Stage 3: Q projection: `q_normed [byte_dim] → q [inner]`.
            self.q_proj.forward(batch, &scratch.q_normed, &mut scratch.q_proj_out)?;

            // Stage 4: per-byte K and V projections (with pre-LN). We
            // upload the patch's bytes one at a time, norm + project,
            // and stash into `k_pack[h, j, :]` / `v_pack` slabs that
            // the per-head score matvec reads directly. The per-byte
            // path keeps each dispatch tiny — fine for a first
            // implementation; a follow-up can batch via matmul_nn.
            for j in 0..p_len {
                let byte_idx = p_start + j;
                let byte_slice = &scratch.byte_reps_host
                    [byte_idx * self.byte_dim..(byte_idx + 1) * self.byte_dim];
                scratch.kv_input.copy_from(byte_slice);

                unsafe {
                    rms_norm_resident(
                        hip_buf(&scratch.kv_input)?.device_ptr() as *const f32,
                        self.kv_norm_weight_dev.device_ptr() as *const f32,
                        hip_buf_mut(&mut scratch.kv_normed)?.device_ptr() as *mut f32,
                        1, self.byte_dim, self.norm_eps,
                    )?;
                }
                batch.note_dispatch()?;

                self.k_proj.forward(batch, &scratch.kv_normed, &mut scratch.k_step)?;
                self.v_proj.forward(batch, &scratch.kv_normed, &mut scratch.v_step)?;

                // Copy k_step / v_step into the packed per-step slabs
                // at offset `j * inner`. Reading host-side then re-
                // uploading keeps the buffer plumbing simple; the
                // per-step alternative would need a D2D memcpy. For
                // small-patch BLT (typical patch lengths ≤ 16), this
                // is a few hundred floats.
                let inner = self.inner_dim();
                scratch
                    .k_step
                    .copy_to_host(&mut scratch.k_v_host_pack[..inner]);
                scratch.k_pack_host[j * inner..(j + 1) * inner]
                    .copy_from_slice(&scratch.k_v_host_pack[..inner]);
                scratch
                    .v_step
                    .copy_to_host(&mut scratch.k_v_host_pack[..inner]);
                scratch.v_pack_host[j * inner..(j + 1) * inner]
                    .copy_from_slice(&scratch.k_v_host_pack[..inner]);
            }

            // Upload the packed K/V for this patch to device.
            scratch.k_pack_dev.copy_from(
                &scratch.k_pack_host[..p_len * self.inner_dim()],
            );
            scratch.v_pack_dev.copy_from(
                &scratch.v_pack_host[..p_len * self.inner_dim()],
            );

            // Stage 5: per-head scoring matvec.
            //   scores[h, j] = K_pack[j, h*head_dim..(h+1)*head_dim]
            //                  · Q[h*head_dim..(h+1)*head_dim]      (× qk_scale)
            //
            // For matvec_resident we need K-slab laid out as
            // `[p_len × head_dim]` per head. The packed buffer is
            // `[p_len × num_heads × head_dim]` row-major (one row per
            // byte, head-major within), so the per-head slab is strided
            // across rows. To keep this slice readable, we issue one
            // matvec per head and rebuild the slab as a contiguous
            // copy on the host — the same simplification AttentionResident
            // uses for the strided softmax-output gather. Total D2H/H2D
            // per patch is `num_heads × p_len × head_dim` floats.
            self.encoder_score_and_softmax(
                batch, p_len, scratch,
            )?;

            // Stage 6: per-head weighted V sum.
            //   head_out[h, :] = sum_j softmax[h, j] * V_pack[j, h*head_dim..]
            self.encoder_weighted_v(batch, p_len, scratch)?;

            // Stage 7: O projection.
            self.o_proj.forward(batch, &scratch.head_out, &mut scratch.o_out)?;

            // Stage 8: stash this patch's output into the host pack.
            scratch.o_out.copy_to_host(&mut scratch.o_out_host);
            scratch.patch_out_host[i * self.patch_dim..(i + 1) * self.patch_dim]
                .copy_from_slice(&scratch.o_out_host[..self.patch_dim]);
        }

        // Stage 9: H2D the assembled patch reps.
        patch_reps_out.copy_from(&scratch.patch_out_host[..n_patches * self.patch_dim]);
        Ok(())
    }

    /// Decoder direction (`patches → bytes`). For each byte `b` the
    /// query is the byte rep itself; keys and values are the patch
    /// reps for the patch containing `b` and any earlier patches
    /// (causal). The owning patch `i_b` for byte `b` is the smallest
    /// `i` with `boundaries[i+1] > b`.
    ///
    /// Same residual contract as [`forward_encoder`] — caller adds the
    /// residual themselves.
    pub fn forward_decoder(
        &self,
        batch: &HipBatch,
        byte_reps: &GpuVec,
        patch_reps: &GpuVec,
        patch_boundaries: &[usize],
        scratch: &mut CrossAttnScratch,
        byte_reps_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        if self.direction != CrossAttnDirection::Decoder {
            return Err(ResidencyError::WrongVariant {
                expected: "Decoder direction",
                got: "Encoder direction (instance was built for the other side)",
            });
        }
        let n_bytes = patch_boundaries.last().copied().unwrap_or(0);
        let n_patches = patch_boundaries.len().saturating_sub(1);

        debug_assert_eq!(byte_reps.len(), n_bytes * self.byte_dim);
        debug_assert_eq!(patch_reps.len(), n_patches * self.patch_dim);
        debug_assert_eq!(byte_reps_out.len(), n_bytes * self.byte_dim);
        debug_assert!(scratch.fits(self));

        // Stage 0: project ALL patches once into K/V (with pre-LN). The
        // decoder reads each patch's K/V multiple times (once per byte
        // whose owning patch is ≥ this one), so amortising the
        // projection across the whole sequence pays off. Pull patches
        // host-side and stream through the device per-patch projection.
        if scratch.byte_reps_host.len() < n_bytes * self.byte_dim {
            scratch.byte_reps_host.resize(n_bytes * self.byte_dim, 0.0);
        }
        byte_reps.copy_to_host(&mut scratch.byte_reps_host[..n_bytes * self.byte_dim]);

        let patch_dim = self.patch_dim;
        if scratch.patch_reps_host.len() < n_patches * patch_dim {
            scratch.patch_reps_host.resize(n_patches * patch_dim, 0.0);
        }
        patch_reps.copy_to_host(&mut scratch.patch_reps_host[..n_patches * patch_dim]);

        // Build the global K/V pack: `[n_patches × inner]`.
        if scratch.k_pack_host.len() < n_patches * self.inner_dim() {
            scratch.k_pack_host.resize(n_patches * self.inner_dim(), 0.0);
        }
        if scratch.v_pack_host.len() < n_patches * self.inner_dim() {
            scratch.v_pack_host.resize(n_patches * self.inner_dim(), 0.0);
        }
        for i in 0..n_patches {
            let patch_slice = &scratch.patch_reps_host
                [i * patch_dim..(i + 1) * patch_dim];
            scratch.kv_input.copy_from(patch_slice);

            unsafe {
                rms_norm_resident(
                    hip_buf(&scratch.kv_input)?.device_ptr() as *const f32,
                    self.kv_norm_weight_dev.device_ptr() as *const f32,
                    hip_buf_mut(&mut scratch.kv_normed)?.device_ptr() as *mut f32,
                    1, patch_dim, self.norm_eps,
                )?;
            }
            batch.note_dispatch()?;

            self.k_proj.forward(batch, &scratch.kv_normed, &mut scratch.k_step)?;
            self.v_proj.forward(batch, &scratch.kv_normed, &mut scratch.v_step)?;

            let inner = self.inner_dim();
            scratch
                .k_step
                .copy_to_host(&mut scratch.k_v_host_pack[..inner]);
            scratch.k_pack_host[i * inner..(i + 1) * inner]
                .copy_from_slice(&scratch.k_v_host_pack[..inner]);
            scratch
                .v_step
                .copy_to_host(&mut scratch.k_v_host_pack[..inner]);
            scratch.v_pack_host[i * inner..(i + 1) * inner]
                .copy_from_slice(&scratch.k_v_host_pack[..inner]);
        }

        // Stage 1: per-byte query, per-byte attention over `[0..owning+1]`.
        if scratch.byte_out_host.len() < n_bytes * self.byte_dim {
            scratch.byte_out_host.resize(n_bytes * self.byte_dim, 0.0);
        }

        let mut owning_patch = 0usize;
        for b in 0..n_bytes {
            // Advance `owning_patch` so that boundaries[owning_patch+1] > b.
            while owning_patch + 1 < n_patches
                && patch_boundaries[owning_patch + 1] <= b
            {
                owning_patch += 1;
            }
            let valid_kv = owning_patch + 1; // patches `[0..valid_kv)`.

            // Stage 1a: pre-LN + Q projection of byte b.
            let byte_slice = &scratch.byte_reps_host
                [b * self.byte_dim..(b + 1) * self.byte_dim];
            scratch.q_input.copy_from(byte_slice);

            unsafe {
                rms_norm_resident(
                    hip_buf(&scratch.q_input)?.device_ptr() as *const f32,
                    self.q_norm_weight_dev.device_ptr() as *const f32,
                    hip_buf_mut(&mut scratch.q_normed)?.device_ptr() as *mut f32,
                    1, self.byte_dim, self.norm_eps,
                )?;
            }
            batch.note_dispatch()?;

            self.q_proj.forward(batch, &scratch.q_normed, &mut scratch.q_proj_out)?;

            // Stage 1b: stage the K/V slab for `[0..valid_kv)` patches
            // into the device pack buffers (already packed across the
            // full sequence; just upload the prefix view).
            scratch.k_pack_dev.copy_from(
                &scratch.k_pack_host[..valid_kv * self.inner_dim()],
            );
            scratch.v_pack_dev.copy_from(
                &scratch.v_pack_host[..valid_kv * self.inner_dim()],
            );

            // Stage 1c: scores + softmax (per-head, length `valid_kv`).
            self.encoder_score_and_softmax(batch, valid_kv, scratch)?;

            // Stage 1d: weighted V sum.
            self.encoder_weighted_v(batch, valid_kv, scratch)?;

            // Stage 1e: O projection.
            self.o_proj.forward(batch, &scratch.head_out, &mut scratch.o_out)?;

            // Stage 1f: stash byte output.
            scratch.o_out.copy_to_host(&mut scratch.o_out_host);
            scratch.byte_out_host[b * self.byte_dim..(b + 1) * self.byte_dim]
                .copy_from_slice(&scratch.o_out_host[..self.byte_dim]);
        }

        byte_reps_out.copy_from(&scratch.byte_out_host[..n_bytes * self.byte_dim]);
        Ok(())
    }

    /// Inner `head_dim * num_heads`.
    #[inline]
    fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Shared score+softmax inner loop used by both directions. Reads
    /// `scratch.q_proj_out` (current Q, `[inner]`) and
    /// `scratch.k_pack_dev` (`[kv_len × inner]`). Writes
    /// `scratch.scores_dev` (`[num_heads × kv_len]`, softmaxed).
    fn encoder_score_and_softmax(
        &self,
        batch: &HipBatch,
        kv_len: usize,
        scratch: &mut CrossAttnScratch,
    ) -> Result<(), ResidencyError> {
        let inner = self.inner_dim();
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;

        // Pull Q and K-pack to host so we can rebuild contiguous per-
        // head slabs. K-pack is `[kv_len × num_heads × head_dim]`
        // row-major; the per-head slab `[kv_len × head_dim]` is
        // strided across rows. The score matvec needs it contiguous.
        scratch.q_proj_out.copy_to_host(&mut scratch.q_host);
        scratch.k_pack_dev.copy_to_host(&mut scratch.k_v_host_pack);

        for h in 0..num_heads {
            for j in 0..kv_len {
                let src_off = j * inner + h * head_dim;
                let dst_off = j * head_dim;
                scratch.k_head_slab_host[dst_off..dst_off + head_dim]
                    .copy_from_slice(
                        &scratch.k_v_host_pack[src_off..src_off + head_dim],
                    );
            }
            scratch.k_head_slab_dev.copy_from(
                &scratch.k_head_slab_host[..kv_len * head_dim],
            );

            // Q for this head, with qk_scale folded in.
            let q_slice = &scratch.q_host[h * head_dim..(h + 1) * head_dim];
            for i in 0..head_dim {
                scratch.q_head_host[i] = q_slice[i] * self.qk_scale;
            }
            scratch.q_head_dev.copy_from(&scratch.q_head_host[..head_dim]);

            // matvec: scores[h, :kv_len] = K_head_slab · q_head
            //   weight = K_head_slab [kv_len × head_dim]
            //   x      = q_head [head_dim]
            //   out    = scores[h, :kv_len]
            unsafe {
                matvec_resident(
                    hip_buf(&scratch.q_head_dev)?.device_ptr() as *const f32,
                    hip_buf(&scratch.k_head_slab_dev)?.device_ptr() as *const f32,
                    self.zero_bias_dev.device_ptr() as *const f32,
                    (hip_buf_mut(&mut scratch.scores_dev)?.device_ptr() as *mut f32)
                        .add(h * scratch.cap_kv_len),
                    kv_len,
                    head_dim,
                )?;
            }
            batch.note_dispatch()?;
        }

        // Softmax per-head, row-major contiguous: scores[h, :kv_len].
        // The buffer layout has stride `cap_kv_len` per head; we
        // softmax in place using row_len = kv_len, but each row sits
        // at offset `h * cap_kv_len`. To keep the softmax contiguous
        // we copy into a tight buffer first (D2D would need MIOpen
        // strided softmax, which we don't have), then copy back after.
        //
        // For simplicity, do this on host: D2H the scores, softmax
        // host-side per row, H2D back. Only `num_heads × kv_len`
        // floats — typically a few hundred at most.
        scratch.scores_dev.copy_to_host(&mut scratch.scores_host_strided);
        for h in 0..num_heads {
            let row = &mut scratch.scores_host_strided
                [h * scratch.cap_kv_len..h * scratch.cap_kv_len + kv_len];
            stable_softmax_inplace(row);
        }
        scratch.scores_dev.copy_from(&scratch.scores_host_strided);

        // Tight copy host-side too: build `scores_tight_host` with
        // contiguous `[num_heads × kv_len]` for the weighted-V step.
        for h in 0..num_heads {
            let src_off = h * scratch.cap_kv_len;
            let dst_off = h * kv_len;
            scratch.scores_tight_host[dst_off..dst_off + kv_len]
                .copy_from_slice(
                    &scratch.scores_host_strided[src_off..src_off + kv_len],
                );
        }
        Ok(())
    }

    /// Shared weighted-V step. Reads `scratch.scores_tight_host`
    /// (`[num_heads × kv_len]`) and `scratch.v_pack_dev`
    /// (`[kv_len × inner]`). Writes `scratch.head_out`
    /// (`[inner]` = num_heads × head_dim concatenated).
    fn encoder_weighted_v(
        &self,
        batch: &HipBatch,
        kv_len: usize,
        scratch: &mut CrossAttnScratch,
    ) -> Result<(), ResidencyError> {
        let inner = self.inner_dim();
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;

        scratch.v_pack_dev.copy_to_host(&mut scratch.k_v_host_pack);

        for h in 0..num_heads {
            // Build the per-head V slab: `[kv_len × head_dim]`.
            for j in 0..kv_len {
                let src_off = j * inner + h * head_dim;
                let dst_off = j * head_dim;
                scratch.v_head_slab_host[dst_off..dst_off + head_dim]
                    .copy_from_slice(
                        &scratch.k_v_host_pack[src_off..src_off + head_dim],
                    );
            }

            // head_out[h, i] = sum_j softmax[h, j] * V_head_slab[j, i]
            //   — host loop is fine here; the surrounding stages
            //   are the heavy hitters and this is `kv_len × head_dim`
            //   multiply-adds. Resident matmul_tn could replace this
            //   in a follow-up (same trick AttentionResident uses).
            let head_out =
                &mut scratch.head_out_host[h * head_dim..(h + 1) * head_dim];
            head_out.fill(0.0);
            let scores_row =
                &scratch.scores_tight_host[h * kv_len..(h + 1) * kv_len];
            for j in 0..kv_len {
                let s = scores_row[j];
                let v_row = &scratch.v_head_slab_host
                    [j * head_dim..(j + 1) * head_dim];
                for i in 0..head_dim {
                    head_out[i] += s * v_row[i];
                }
            }
        }

        // H2D the concatenated head_out for the O projection. The O
        // projection runs on device.
        scratch.head_out.copy_from(&scratch.head_out_host[..inner]);
        // No dispatch issued by the host loop above — the H2D copy
        // doesn't touch the HIP queue. The following o_proj.forward
        // already calls note_dispatch.
        let _ = batch;
        Ok(())
    }
}

// ─── Scratch ────────────────────────────────────────────────

/// Per-call scratch buffers for [`CrossAttention::forward_encoder`] /
/// [`CrossAttention::forward_decoder`]. Pre-allocate with
/// `CrossAttnScratch::new` for the worst-case dimensions and reuse
/// across calls — the same residency idiom as
/// [`modgrad_transformer::resident::AttentionScratch`].
pub struct CrossAttnScratch {
    // Q-side single-step device scratch.
    pub q_input: GpuVec,           // [q_in_dim]
    pub q_normed: GpuVec,          // [q_in_dim]
    pub q_proj_out: GpuVec,        // [inner]

    // K/V-side single-step device scratch.
    pub kv_input: GpuVec,          // [kv_in_dim]
    pub kv_normed: GpuVec,         // [kv_in_dim]
    pub k_step: GpuVec,            // [inner]
    pub v_step: GpuVec,            // [inner]

    // Packed K/V across the per-call kv set (encoder: per-patch bytes;
    // decoder: all patches up to causal limit).
    pub k_pack_dev: GpuVec,        // [cap_kv_len × inner]
    pub v_pack_dev: GpuVec,        // [cap_kv_len × inner]

    // Per-head slab buffers (built host-side then uploaded).
    pub k_head_slab_dev: GpuVec,   // [cap_kv_len × head_dim]
    pub q_head_dev: GpuVec,        // [head_dim]

    // Score / softmax buffers — strided `[num_heads × cap_kv_len]`.
    pub scores_dev: GpuVec,        // [num_heads × cap_kv_len]

    // O-projection step output and aggregate per-call output buffers.
    pub head_out: GpuVec,          // [inner]
    pub o_out: GpuVec,             // [o_out_dim]

    // Host scratch — for the small bounce stages (max-pool, per-head
    // slab gather, host softmax).
    pub byte_reps_host: Vec<f32>,
    pub patch_reps_host: Vec<f32>,
    pub k_pack_host: Vec<f32>,
    pub v_pack_host: Vec<f32>,
    pub k_v_host_pack: Vec<f32>,
    pub k_head_slab_host: Vec<f32>,
    pub v_head_slab_host: Vec<f32>,
    pub q_host: Vec<f32>,
    pub q_head_host: Vec<f32>,
    pub head_out_host: Vec<f32>,
    pub scores_host_strided: Vec<f32>,
    pub scores_tight_host: Vec<f32>,
    pub o_out_host: Vec<f32>,
    pub patch_out_host: Vec<f32>,
    pub byte_out_host: Vec<f32>,

    // Caps — fits() checks against these.
    cap_q_in: usize,
    cap_kv_in: usize,
    cap_inner: usize,
    cap_o_out: usize,
    cap_kv_len: usize,
    /// Reserved for stricter `fits()` once callers start sizing the
    /// scratch tightly to per-call dims rather than the worst-case.
    #[allow(dead_code)]
    cap_n_bytes: usize,
    /// See `cap_n_bytes`.
    #[allow(dead_code)]
    cap_n_patches: usize,
    cap_num_heads: usize,
    cap_head_dim: usize,
}

impl CrossAttnScratch {
    /// Allocate scratch for the given config plus call caps. `max_kv_len`
    /// is the longest run of contiguous keys/values the call will see
    /// (encoder: the longest patch's byte count; decoder: the largest
    /// causal prefix, which equals `n_patches`). `max_n_bytes` and
    /// `max_n_patches` size the host-side staging buffers for the
    /// per-call assembly.
    pub fn new(
        config: &CrossAttnConfig,
        max_kv_len: usize,
        max_n_bytes: usize,
        max_n_patches: usize,
    ) -> Result<Self, ResidencyError> {
        let inner = config.inner_dim();
        let q_in = config.q_in_dim();
        let kv_in = config.kv_in_dim();
        let o_out = config.o_out_dim();
        let nh = config.num_heads;
        let hd = config.head_dim;

        Ok(Self {
            q_input: GpuVec::try_hip(q_in)?,
            q_normed: GpuVec::try_hip(q_in)?,
            q_proj_out: GpuVec::try_hip(inner)?,
            kv_input: GpuVec::try_hip(kv_in)?,
            kv_normed: GpuVec::try_hip(kv_in)?,
            k_step: GpuVec::try_hip(inner)?,
            v_step: GpuVec::try_hip(inner)?,
            k_pack_dev: GpuVec::try_hip(max_kv_len * inner)?,
            v_pack_dev: GpuVec::try_hip(max_kv_len * inner)?,
            k_head_slab_dev: GpuVec::try_hip(max_kv_len * hd)?,
            q_head_dev: GpuVec::try_hip(hd)?,
            scores_dev: GpuVec::try_hip(nh * max_kv_len)?,
            head_out: GpuVec::try_hip(inner)?,
            o_out: GpuVec::try_hip(o_out)?,
            byte_reps_host: vec![0.0; max_n_bytes * config.byte_dim],
            patch_reps_host: vec![0.0; max_n_patches * config.patch_dim],
            k_pack_host: vec![0.0; max_kv_len.max(max_n_patches) * inner],
            v_pack_host: vec![0.0; max_kv_len.max(max_n_patches) * inner],
            k_v_host_pack: vec![0.0; inner.max(max_kv_len * inner)],
            k_head_slab_host: vec![0.0; max_kv_len * hd],
            v_head_slab_host: vec![0.0; max_kv_len * hd],
            q_host: vec![0.0; inner],
            q_head_host: vec![0.0; hd],
            head_out_host: vec![0.0; inner],
            scores_host_strided: vec![0.0; nh * max_kv_len],
            scores_tight_host: vec![0.0; nh * max_kv_len],
            o_out_host: vec![0.0; o_out],
            patch_out_host: vec![0.0; max_n_patches * config.patch_dim],
            byte_out_host: vec![0.0; max_n_bytes * config.byte_dim],
            cap_q_in: q_in,
            cap_kv_in: kv_in,
            cap_inner: inner,
            cap_o_out: o_out,
            cap_kv_len: max_kv_len,
            cap_n_bytes: max_n_bytes,
            cap_n_patches: max_n_patches,
            cap_num_heads: nh,
            cap_head_dim: hd,
        })
    }

    /// True if this scratch is sized for the given module's caps.
    pub fn fits(&self, attn: &CrossAttention) -> bool {
        self.cap_inner >= attn.inner_dim()
            && self.cap_num_heads >= attn.num_heads
            && self.cap_head_dim >= attn.head_dim
            && self.cap_q_in
                >= match attn.direction {
                    CrossAttnDirection::Encoder => attn.byte_dim,
                    CrossAttnDirection::Decoder => attn.byte_dim,
                }
            && self.cap_kv_in
                >= match attn.direction {
                    CrossAttnDirection::Encoder => attn.byte_dim,
                    CrossAttnDirection::Decoder => attn.patch_dim,
                }
            && self.cap_o_out
                >= match attn.direction {
                    CrossAttnDirection::Encoder => attn.patch_dim,
                    CrossAttnDirection::Decoder => attn.byte_dim,
                }
    }
}

// ─── Helpers ────────────────────────────────────────────────

fn upload_f32(vals: &[f32]) -> Result<HipBuffer, ResidencyError> {
    let buf = HipBuffer::new(vals.len() * 4)?;
    buf.copy_from_host(vals)?;
    Ok(buf)
}

fn hip_buf(g: &GpuVec) -> Result<&HipBuffer, ResidencyError> {
    match g {
        GpuVec::Hip(b) => Ok(b),
        other => Err(ResidencyError::WrongVariant {
            expected: "Hip",
            got: other.variant_name(),
        }),
    }
}

fn hip_buf_mut(g: &mut GpuVec) -> Result<&mut HipBuffer, ResidencyError> {
    // HipBuffer's pointer is shared via Cell; mutable access is the
    // same pattern as the immutable one.
    match g {
        GpuVec::Hip(b) => Ok(b),
        other => {
            let name = other.variant_name();
            Err(ResidencyError::WrongVariant { expected: "Hip", got: name })
        }
    }
}

/// Initialise a host `Linear` with He-style normal scaling.
fn init_linear(rng: &mut SimpleRng, in_dim: usize, out_dim: usize) -> Linear {
    let scale = (2.0f32 / in_dim as f32).sqrt();
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|_| rng.next_normal() * scale)
        .collect();
    let bias = vec![0.0f32; out_dim];
    Linear {
        weight,
        bias,
        in_dim,
        out_dim,
    }
}

/// Element-wise max-pool of `n_tokens × dim` rows down to `[dim]`.
fn max_pool_bytes(rows: &[f32], dim: usize, n_tokens: usize) -> Vec<f32> {
    debug_assert_eq!(rows.len(), n_tokens * dim);
    let mut out = rows[..dim].to_vec();
    for t in 1..n_tokens {
        let row = &rows[t * dim..(t + 1) * dim];
        for i in 0..dim {
            if row[i] > out[i] {
                out[i] = row[i];
            }
        }
    }
    out
}

/// Numerically stable softmax in place: x_i = exp(x_i - max) / sum.
fn stable_softmax_inplace(row: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

// `softmax_resident` is imported but currently unused — the per-call
// host softmax is the cheaper path while the strided-row layout
// awaits the bring-up of a `softmax_resident_strided` op. Kept in the
// import set so the follow-up that flips host→device is a one-line
// swap. Suppress the unused-import warning until that lands.
const _: fn() = || {
    let _ = softmax_resident;
};

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny CrossAttention, run the encoder forward, and
    /// compare against a host reference computation.
    ///
    /// Test config: 16 bytes / 4 patches / 2 heads / patch_dim=8 /
    /// byte_dim=8 / head_dim=4. Boundaries split the bytes into four
    /// patches of 4 bytes each.
    #[test]
    fn encoder_forward_matches_host_reference() {
        // Skip without a real HIP runtime — the test builds but only
        // runs when `--features rocm` *and* a device is available.
        if std::env::var("MODGRAD_SKIP_HIP_TESTS").is_ok() {
            return;
        }

        const BYTE_DIM: usize = 8;
        const PATCH_DIM: usize = 8;
        const NUM_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const N_BYTES: usize = 16;
        const N_PATCHES: usize = 4;

        let config = CrossAttnConfig {
            byte_dim: BYTE_DIM,
            patch_dim: PATCH_DIM,
            num_heads: NUM_HEADS,
            head_dim: HEAD_DIM,
            norm_eps: 1e-5,
            direction: CrossAttnDirection::Encoder,
        };

        let attn = match CrossAttention::new(&config, 0xC0DE_BEEF_CAFE_F00D) {
            Ok(a) => a,
            Err(_) => {
                // No device available — abort the test cleanly.
                return;
            }
        };

        let mut scratch = match CrossAttnScratch::new(&config, 4, N_BYTES, N_PATCHES) {
            Ok(s) => s,
            Err(_) => return,
        };

        let mut byte_rng = SimpleRng::new(0xFEED_CAFE_BABE_0001);
        let byte_reps_host: Vec<f32> = (0..N_BYTES * BYTE_DIM)
            .map(|_| byte_rng.next_normal() * 0.5)
            .collect();
        let mut byte_reps = match GpuVec::try_hip(N_BYTES * BYTE_DIM) {
            Ok(v) => v,
            Err(_) => return,
        };
        byte_reps.copy_from(&byte_reps_host);

        let boundaries: Vec<usize> = vec![0, 4, 8, 12, 16];

        let mut patch_reps_out = match GpuVec::try_hip(N_PATCHES * PATCH_DIM) {
            Ok(v) => v,
            Err(_) => return,
        };

        let batch = HipBatch::new();
        attn.forward_encoder(
            &batch,
            &byte_reps,
            &boundaries,
            &mut scratch,
            &mut patch_reps_out,
        )
        .expect("forward_encoder");
        batch.flush().expect("flush");

        let mut got = vec![0.0f32; N_PATCHES * PATCH_DIM];
        patch_reps_out.copy_to_host(&mut got);

        // ── Host reference ───────────────────────────────
        // Pull all weights down once.
        let mut wq = vec![0.0f32; attn.q_proj.out_dim * attn.q_proj.in_dim];
        attn.q_proj.weight_dev.copy_to_host(&mut wq).unwrap();
        let mut bq = vec![0.0f32; attn.q_proj.out_dim];
        attn.q_proj.bias_dev.copy_to_host(&mut bq).unwrap();
        let mut wk = vec![0.0f32; attn.k_proj.out_dim * attn.k_proj.in_dim];
        attn.k_proj.weight_dev.copy_to_host(&mut wk).unwrap();
        let mut bk = vec![0.0f32; attn.k_proj.out_dim];
        attn.k_proj.bias_dev.copy_to_host(&mut bk).unwrap();
        let mut wv = vec![0.0f32; attn.v_proj.out_dim * attn.v_proj.in_dim];
        attn.v_proj.weight_dev.copy_to_host(&mut wv).unwrap();
        let mut bv = vec![0.0f32; attn.v_proj.out_dim];
        attn.v_proj.bias_dev.copy_to_host(&mut bv).unwrap();
        let mut wo = vec![0.0f32; attn.o_proj.out_dim * attn.o_proj.in_dim];
        attn.o_proj.weight_dev.copy_to_host(&mut wo).unwrap();
        let mut bo = vec![0.0f32; attn.o_proj.out_dim];
        attn.o_proj.bias_dev.copy_to_host(&mut bo).unwrap();

        let mut expected = vec![0.0f32; N_PATCHES * PATCH_DIM];
        for i in 0..N_PATCHES {
            let p_start = boundaries[i];
            let p_end = boundaries[i + 1];
            let p_len = p_end - p_start;

            // Max-pool query.
            let pooled = max_pool_bytes(
                &byte_reps_host[p_start * BYTE_DIM..p_end * BYTE_DIM],
                BYTE_DIM,
                p_len,
            );
            let q_normed = host_rms_norm(&pooled, BYTE_DIM, attn.norm_eps);
            let q = host_linear(&q_normed, &wq, &bq, BYTE_DIM, NUM_HEADS * HEAD_DIM);

            // Per-byte K/V.
            let mut k_pack = vec![0.0f32; p_len * NUM_HEADS * HEAD_DIM];
            let mut v_pack = vec![0.0f32; p_len * NUM_HEADS * HEAD_DIM];
            for j in 0..p_len {
                let b_off = (p_start + j) * BYTE_DIM;
                let byte_slice = &byte_reps_host[b_off..b_off + BYTE_DIM];
                let kv_normed = host_rms_norm(byte_slice, BYTE_DIM, attn.norm_eps);
                let kj = host_linear(&kv_normed, &wk, &bk, BYTE_DIM, NUM_HEADS * HEAD_DIM);
                let vj = host_linear(&kv_normed, &wv, &bv, BYTE_DIM, NUM_HEADS * HEAD_DIM);
                let off = j * NUM_HEADS * HEAD_DIM;
                k_pack[off..off + NUM_HEADS * HEAD_DIM].copy_from_slice(&kj);
                v_pack[off..off + NUM_HEADS * HEAD_DIM].copy_from_slice(&vj);
            }

            // Per-head scores + softmax + weighted V.
            let mut head_out = vec![0.0f32; NUM_HEADS * HEAD_DIM];
            for h in 0..NUM_HEADS {
                let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                let mut scores = vec![0.0f32; p_len];
                for j in 0..p_len {
                    let kj = &k_pack[j * NUM_HEADS * HEAD_DIM + h * HEAD_DIM
                        ..j * NUM_HEADS * HEAD_DIM + (h + 1) * HEAD_DIM];
                    let mut s = 0.0f32;
                    for i in 0..HEAD_DIM {
                        s += qh[i] * kj[i];
                    }
                    scores[j] = s * attn.qk_scale;
                }
                stable_softmax_inplace(&mut scores);

                for j in 0..p_len {
                    let vj = &v_pack[j * NUM_HEADS * HEAD_DIM + h * HEAD_DIM
                        ..j * NUM_HEADS * HEAD_DIM + (h + 1) * HEAD_DIM];
                    let s = scores[j];
                    for i in 0..HEAD_DIM {
                        head_out[h * HEAD_DIM + i] += s * vj[i];
                    }
                }
            }

            let patch_out =
                host_linear(&head_out, &wo, &bo, NUM_HEADS * HEAD_DIM, PATCH_DIM);
            expected[i * PATCH_DIM..(i + 1) * PATCH_DIM].copy_from_slice(&patch_out);
        }

        let mut max_diff = 0.0f32;
        for (g, e) in got.iter().zip(expected.iter()) {
            let d = (g - e).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff < 1e-3,
            "encoder forward diverged from host reference; max abs diff = {max_diff}"
        );
    }

    fn host_rms_norm(x: &[f32], dim: usize, eps: f32) -> Vec<f32> {
        let mut sq_sum = 0.0f32;
        for &v in x.iter().take(dim) {
            sq_sum += v * v;
        }
        let inv_rms = 1.0 / (sq_sum / dim as f32 + eps).sqrt();
        x.iter().take(dim).map(|&v| v * inv_rms).collect()
    }

    fn host_linear(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        debug_assert_eq!(w.len(), out_dim * in_dim);
        let mut y = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut s = b[o];
            for i in 0..in_dim {
                s += w[o * in_dim + i] * x[i];
            }
            y[o] = s;
        }
        y
    }
}
