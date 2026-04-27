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
use modgrad_device::backend::op::BinaryOpKind;
use modgrad_device::backend::ops::{
    matvec_resident, op_tensor_resident, rms_norm_resident, softmax_resident,
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

// ─── Backward — grads, cache, and per-direction backward fns ────

/// Public alias for [`CrossAttention`] used by callers that talk
/// about "the weight bundle" rather than "the module instance". The
/// resident path keeps weights, scratch, and dispatch glue on the
/// same struct, so this is a literal alias rather than a separate
/// view type.
pub type CrossAttnWeights = CrossAttention;

/// Device-resident weight gradients for [`CrossAttention`].
///
/// Each grad mirrors the matching forward `LinearResident`. Sizes:
///   - `dweight_q`, `dbias_q`: `[inner × q_in_dim]` / `[inner]`
///   - `dweight_k`, `dbias_k`: `[inner × kv_in_dim]` / `[inner]`
///   - `dweight_v`, `dbias_v`: `[inner × kv_in_dim]` / `[inner]`
///   - `dweight_o`, `dbias_o`: `[o_out_dim × inner]` / `[o_out_dim]`
///
/// `inner = num_heads * head_dim`. The four projections accumulate
/// across all per-step backward calls in one cross-attn invocation.
pub struct CrossAttnGrads {
    pub dweight_q: GpuVec,
    pub dbias_q: GpuVec,
    pub dweight_k: GpuVec,
    pub dbias_k: GpuVec,
    pub dweight_v: GpuVec,
    pub dbias_v: GpuVec,
    pub dweight_o: GpuVec,
    pub dbias_o: GpuVec,
    inner: usize,
    q_in: usize,
    kv_in: usize,
    o_out: usize,
}

impl CrossAttnGrads {
    /// Allocate fresh zero grads sized for the given config (alias for
    /// [`zeros_for_config`]; kept under both names because the decoder
    /// path's contract uses `new`).
    pub fn new(config: &CrossAttnConfig) -> Result<Self, ResidencyError> {
        Self::zeros_for_config(config)
    }

    /// Allocate fresh zero grads sized for the given config.
    pub fn zeros_for_config(config: &CrossAttnConfig) -> Result<Self, ResidencyError> {
        let inner = config.inner_dim();
        let q_in = config.q_in_dim();
        let kv_in = config.kv_in_dim();
        let o_out = config.o_out_dim();
        let g = Self {
            dweight_q: GpuVec::try_hip(inner * q_in)?,
            dbias_q: GpuVec::try_hip(inner)?,
            dweight_k: GpuVec::try_hip(inner * kv_in)?,
            dbias_k: GpuVec::try_hip(inner)?,
            dweight_v: GpuVec::try_hip(inner * kv_in)?,
            dbias_v: GpuVec::try_hip(inner)?,
            dweight_o: GpuVec::try_hip(o_out * inner)?,
            dbias_o: GpuVec::try_hip(o_out)?,
            inner, q_in, kv_in, o_out,
        };
        Ok(g)
    }

    /// Convenience matching the spec signature `zeros(model_dim, kv_dim)`.
    /// Interprets `model_dim` as both `q_in_dim` and `o_out_dim`
    /// (decoder direction); `kv_dim` as `kv_in_dim`. `inner` defaults to
    /// `model_dim` (square inner). For asymmetric layouts (encoder
    /// direction with `o_out_dim != q_in_dim`) use [`zeros_for_config`].
    pub fn zeros(model_dim: usize, kv_dim: usize) -> Result<Self, ResidencyError> {
        let inner = model_dim;
        Ok(Self {
            dweight_q: GpuVec::try_hip(inner * model_dim)?,
            dbias_q: GpuVec::try_hip(inner)?,
            dweight_k: GpuVec::try_hip(inner * kv_dim)?,
            dbias_k: GpuVec::try_hip(inner)?,
            dweight_v: GpuVec::try_hip(inner * kv_dim)?,
            dbias_v: GpuVec::try_hip(inner)?,
            dweight_o: GpuVec::try_hip(model_dim * inner)?,
            dbias_o: GpuVec::try_hip(model_dim)?,
            inner, q_in: model_dim, kv_in: kv_dim, o_out: model_dim,
        })
    }

    /// Reset every accumulator to zero. The H2D path (`copy_from`)
    /// keeps the API uniform — there is no `hipMemset` op exposed by
    /// `modgrad-device` at the time of writing.
    pub fn zero_resident(&mut self, _batch: &HipBatch) -> Result<(), ResidencyError> {
        // TODO(blt-bwd): swap to a `memset_resident` once exposed; the
        // H2D zero-upload below is correct but not as cheap as a device
        // memset for the larger weight tensors.
        let z_inner_q = vec![0.0f32; self.inner * self.q_in];
        let z_inner_kv = vec![0.0f32; self.inner * self.kv_in];
        let z_o = vec![0.0f32; self.o_out * self.inner];
        let z_inner = vec![0.0f32; self.inner];
        let z_o_bias = vec![0.0f32; self.o_out];
        self.dweight_q.copy_from(&z_inner_q);
        self.dbias_q.copy_from(&z_inner);
        self.dweight_k.copy_from(&z_inner_kv);
        self.dbias_k.copy_from(&z_inner);
        self.dweight_v.copy_from(&z_inner_kv);
        self.dbias_v.copy_from(&z_inner);
        self.dweight_o.copy_from(&z_o);
        self.dbias_o.copy_from(&z_o_bias);
        Ok(())
    }
}

/// Per-call backward cache for [`CrossAttention`]. Populated by
/// [`CrossAttention::forward_encoder_for_backward`] /
/// [`CrossAttention::forward_decoder_for_backward`] and consumed by
/// [`cross_attn_encoder_backward`] / [`cross_attn_decoder_backward`].
///
/// Sizing convention:
///   - encoder direction: `n_queries = n_patches`, per-query KV length
///     ≤ `max_patch_len`, K/V slabs are indexed by per-patch byte j.
///   - decoder direction: `n_queries = n_bytes`, per-query KV length
///     ≤ `n_patches` (causal prefix). Decoder K/V projections are
///     amortised once over all patches at forward time, so the cache
///     stores `n_patches × inner` for K and V (one global slab) plus
///     per-byte Q outputs.
///
/// Layouts are host-side flat row-major because the forward already
/// bounces through host scratch for the per-step softmax and pack
/// reshuffles; saving an extra contiguous copy is cheaper than holding
/// strided device buffers across thousands of dispatches.
pub struct CrossAttnBwdCache {
    /// `[n_queries × inner]` — Q projection output per query.
    pub q_proj_per_query: Vec<f32>,
    /// `[n_queries × q_in_dim]` — pre-LN Q input per query (max-pool
    /// for encoder, raw byte rep for decoder). Needed for the q_proj
    /// bias outer product and the upstream byte-grad.
    pub q_input_per_query: Vec<f32>,
    /// `[n_queries × q_in_dim]` — post-LN Q (= input to q_proj). Used
    /// as `x` for `q_proj.backward`.
    pub q_normed_per_query: Vec<f32>,
    /// Encoder: `[n_patches × max_patch_len × inner]` per-step K outputs.
    /// Decoder: `[n_patches × inner]` (one per patch, shared across
    /// every byte query).
    pub k_pack: Vec<f32>,
    /// Same shape as `k_pack`, V projection.
    pub v_pack: Vec<f32>,
    /// Encoder: `[n_patches × max_patch_len × kv_in_dim]` post-LN K/V
    /// inputs (same vector goes into both k_proj and v_proj).
    /// Decoder: `[n_patches × kv_in_dim]`.
    pub kv_normed_pack: Vec<f32>,
    /// Encoder: `[n_patches × max_patch_len × kv_in_dim]` raw pre-LN
    /// K/V inputs (= source byte/patch rep).
    /// Decoder: `[n_patches × kv_in_dim]`.
    pub kv_input_pack: Vec<f32>,
    /// `[n_queries × num_heads × max_kv_len]` — softmaxed scores per
    /// query, head-major. Trailing slots past the per-query kv_len
    /// are zero.
    pub softmax_per_query: Vec<f32>,
    /// `[n_queries × num_heads × max_kv_len]` — pre-softmax scores
    /// (after qk_scale), kept for parity with downstream softmax-
    /// backward checks; not consumed by the per-step backward but
    /// retained as a checkpoint surface.
    pub scores_per_query: Vec<f32>,
    /// `[n_queries × inner]` — concatenated head_out per query
    /// (= input to o_proj).
    pub head_out_per_query: Vec<f32>,
    /// `[n_queries]` — per-query kv length actually used (encoder:
    /// `p_end - p_start`; decoder: `owning_patch + 1`).
    pub kv_len_per_query: Vec<usize>,
    /// Cache caps (so backward sanity-checks).
    n_queries: usize,
    max_kv_len: usize,
    inner: usize,
    #[allow(dead_code)]
    q_in: usize,
    #[allow(dead_code)]
    kv_in: usize,
    num_heads: usize,
    /// Direction the cache was sized for.
    direction: CrossAttnDirection,
}

impl CrossAttnBwdCache {
    /// Allocate a cache for up to `max_n_queries` queries with up to
    /// `max_kv_len` keys/values per query.
    ///
    /// Per the spec:
    ///   - encoder direction: `max_n_queries = max_n_patches`,
    ///     `max_kv_len = max_patch_length` (in bytes).
    ///   - decoder direction: `max_n_queries = max_n_bytes`,
    ///     `max_kv_len = max_n_patches`.
    pub fn new(
        config: &CrossAttnConfig,
        max_n_queries: usize,
        max_kv_len: usize,
    ) -> Result<Self, ResidencyError> {
        let inner = config.inner_dim();
        let q_in = config.q_in_dim();
        let kv_in = config.kv_in_dim();
        let nh = config.num_heads;
        // For decoder, K/V are amortised once across n_patches (=
        // max_kv_len); k_pack/v_pack only need [n_patches × inner].
        // For encoder we need the worst case [n_queries × max_kv_len ×
        // inner]. Take the max so one cache shape works for both.
        let kv_pack_cap = match config.direction {
            CrossAttnDirection::Encoder => max_n_queries * max_kv_len * inner,
            CrossAttnDirection::Decoder => max_kv_len * inner,
        };
        let kv_input_cap = match config.direction {
            CrossAttnDirection::Encoder => max_n_queries * max_kv_len * kv_in,
            CrossAttnDirection::Decoder => max_kv_len * kv_in,
        };
        Ok(Self {
            q_proj_per_query: vec![0.0; max_n_queries * inner],
            q_input_per_query: vec![0.0; max_n_queries * q_in],
            q_normed_per_query: vec![0.0; max_n_queries * q_in],
            k_pack: vec![0.0; kv_pack_cap],
            v_pack: vec![0.0; kv_pack_cap],
            kv_normed_pack: vec![0.0; kv_input_cap],
            kv_input_pack: vec![0.0; kv_input_cap],
            softmax_per_query: vec![0.0; max_n_queries * nh * max_kv_len],
            scores_per_query: vec![0.0; max_n_queries * nh * max_kv_len],
            head_out_per_query: vec![0.0; max_n_queries * inner],
            kv_len_per_query: vec![0; max_n_queries],
            n_queries: max_n_queries,
            max_kv_len,
            inner, q_in, kv_in, num_heads: nh,
            direction: config.direction,
        })
    }

    /// True if this cache fits the given module + call shape.
    pub fn fits(
        &self,
        attn: &CrossAttention,
        n_queries: usize,
        max_kv_len: usize,
    ) -> bool {
        self.direction == attn.direction
            && n_queries <= self.n_queries
            && max_kv_len <= self.max_kv_len
            && attn.inner_dim() <= self.inner
            && attn.num_heads <= self.num_heads
    }
}

impl CrossAttention {
    /// Encoder-direction forward that additionally populates `cache`.
    /// Otherwise identical to [`forward_encoder`]. Same arg shapes.
    pub fn forward_encoder_for_backward(
        &self,
        batch: &HipBatch,
        byte_reps: &GpuVec,
        patch_boundaries: &[usize],
        scratch: &mut CrossAttnScratch,
        cache: &mut CrossAttnBwdCache,
        patch_reps_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        // Run the canonical forward first, then populate the cache by
        // re-staging the same host-side bounce. The forward's existing
        // host scratch already holds the data we want — we just snapshot
        // each per-query record into `cache` slabs.
        //
        // Implementation: re-run the per-patch loop here so the cache
        // captures the activations as they're produced (the public
        // `forward_encoder` reuses scratch buffers across patches and
        // by the time it returns the per-query state has been
        // overwritten n_patches-1 times). This is a moderate code
        // duplication but the alternative is plumbing a `cache: Option`
        // through the forward proper; that reshuffles the public API.
        if self.direction != CrossAttnDirection::Encoder {
            return Err(ResidencyError::WrongVariant {
                expected: "Encoder direction",
                got: "Decoder direction (instance was built for the other side)",
            });
        }
        let n_bytes = patch_boundaries.last().copied().unwrap_or(0);
        let n_patches = patch_boundaries.len().saturating_sub(1);
        // Encoder K/V is per-patch — only the largest patch must fit
        // the cache, not the full byte stream.
        let max_patch_len = patch_boundaries.windows(2)
            .map(|w| w[1] - w[0]).max().unwrap_or(0);

        debug_assert!(byte_reps.len() >= n_bytes * self.byte_dim);
        debug_assert!(patch_reps_out.len() >= n_patches * self.patch_dim);
        debug_assert!(scratch.fits(self));
        debug_assert!(cache.fits(self, n_patches, max_patch_len));

        if scratch.byte_reps_host.len() < n_bytes * self.byte_dim {
            scratch.byte_reps_host.resize(n_bytes * self.byte_dim, 0.0);
        }
        byte_reps.copy_to_host(&mut scratch.byte_reps_host[..n_bytes * self.byte_dim]);
        if scratch.patch_out_host.len() < n_patches * self.patch_dim {
            scratch.patch_out_host.resize(n_patches * self.patch_dim, 0.0);
        }

        let inner = self.inner_dim();
        let max_kv = cache.max_kv_len;

        for i in 0..n_patches {
            let p_start = patch_boundaries[i];
            let p_end = patch_boundaries[i + 1];
            let p_len = p_end - p_start;

            // Stage 1+2: max-pool + RMS-norm Q.
            let pooled = max_pool_bytes(
                &scratch.byte_reps_host[p_start * self.byte_dim..p_end * self.byte_dim],
                self.byte_dim, p_len,
            );
            cache.q_input_per_query[i * self.byte_dim..(i + 1) * self.byte_dim]
                .copy_from_slice(&pooled);
            scratch.q_input.copy_from(&pooled);
            unsafe {
                rms_norm_resident(
                    hip_buf(&scratch.q_input)?.device_ptr() as *const f32,
                    self.q_norm_weight_dev.device_ptr() as *const f32,
                    hip_buf_mut(&mut scratch.q_normed)?.device_ptr() as *mut f32,
                    1, self.byte_dim, self.norm_eps,
                )?;
            }
            batch.note_dispatch()?;
            // Snapshot q_normed.
            let qn_off = i * self.byte_dim;
            scratch.q_normed.copy_to_host(&mut cache.q_normed_per_query
                [qn_off..qn_off + self.byte_dim]);

            // Stage 3: Q projection.
            self.q_proj.forward(batch, &scratch.q_normed, &mut scratch.q_proj_out)?;
            scratch.q_proj_out.copy_to_host(&mut cache.q_proj_per_query
                [i * inner..(i + 1) * inner]);

            // Stage 4: per-byte K/V (with pre-LN, per-step).
            for j in 0..p_len {
                let byte_idx = p_start + j;
                let byte_slice = &scratch.byte_reps_host
                    [byte_idx * self.byte_dim..(byte_idx + 1) * self.byte_dim];
                // Snapshot kv_input_pack [i, j, :].
                let kv_in_off = (i * max_kv + j) * self.byte_dim;
                cache.kv_input_pack[kv_in_off..kv_in_off + self.byte_dim]
                    .copy_from_slice(byte_slice);
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
                let kvn_off = (i * max_kv + j) * self.byte_dim;
                scratch.kv_normed.copy_to_host(&mut cache.kv_normed_pack
                    [kvn_off..kvn_off + self.byte_dim]);

                self.k_proj.forward(batch, &scratch.kv_normed, &mut scratch.k_step)?;
                self.v_proj.forward(batch, &scratch.kv_normed, &mut scratch.v_step)?;

                scratch.k_step.copy_to_host(&mut scratch.k_v_host_pack[..inner]);
                scratch.k_pack_host[j * inner..(j + 1) * inner]
                    .copy_from_slice(&scratch.k_v_host_pack[..inner]);
                let k_pack_off = (i * max_kv + j) * inner;
                cache.k_pack[k_pack_off..k_pack_off + inner]
                    .copy_from_slice(&scratch.k_v_host_pack[..inner]);
                scratch.v_step.copy_to_host(&mut scratch.k_v_host_pack[..inner]);
                scratch.v_pack_host[j * inner..(j + 1) * inner]
                    .copy_from_slice(&scratch.k_v_host_pack[..inner]);
                cache.v_pack[k_pack_off..k_pack_off + inner]
                    .copy_from_slice(&scratch.k_v_host_pack[..inner]);
            }

            scratch.k_pack_dev.copy_from(&scratch.k_pack_host[..p_len * inner]);
            scratch.v_pack_dev.copy_from(&scratch.v_pack_host[..p_len * inner]);

            self.encoder_score_and_softmax(batch, p_len, scratch)?;
            // Snapshot softmax: scores_tight_host is [num_heads × kv_len].
            let nh = self.num_heads;
            for h in 0..nh {
                let dst_off = i * nh * max_kv + h * max_kv;
                let src_off = h * p_len;
                cache.softmax_per_query[dst_off..dst_off + p_len]
                    .copy_from_slice(&scratch.scores_tight_host[src_off..src_off + p_len]);
                // Trailing slots stay zero from initial alloc; they'd
                // be overwritten only on a subsequent call with longer
                // p_len — explicit zeroing keeps fits() honest.
                for k in p_len..max_kv {
                    cache.softmax_per_query[dst_off + k] = 0.0;
                }
            }
            cache.kv_len_per_query[i] = p_len;

            self.encoder_weighted_v(batch, p_len, scratch)?;
            // Snapshot head_out.
            cache.head_out_per_query[i * inner..(i + 1) * inner]
                .copy_from_slice(&scratch.head_out_host[..inner]);

            self.o_proj.forward(batch, &scratch.head_out, &mut scratch.o_out)?;
            scratch.o_out.copy_to_host(&mut scratch.o_out_host);
            scratch.patch_out_host[i * self.patch_dim..(i + 1) * self.patch_dim]
                .copy_from_slice(&scratch.o_out_host[..self.patch_dim]);
        }

        patch_reps_out.copy_from(&scratch.patch_out_host[..n_patches * self.patch_dim]);
        Ok(())
    }

    /// Decoder-direction forward that additionally populates `cache`.
    /// Mirrors [`forward_decoder`].
    pub fn forward_decoder_for_backward(
        &self,
        batch: &HipBatch,
        byte_reps: &GpuVec,
        patch_reps: &GpuVec,
        patch_boundaries: &[usize],
        scratch: &mut CrossAttnScratch,
        cache: &mut CrossAttnBwdCache,
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

        debug_assert!(byte_reps.len() >= n_bytes * self.byte_dim);
        debug_assert!(patch_reps.len() >= n_patches * self.patch_dim);
        debug_assert!(byte_reps_out.len() >= n_bytes * self.byte_dim);
        debug_assert!(scratch.fits(self));
        debug_assert!(cache.fits(self, n_bytes, n_patches));

        let inner = self.inner_dim();
        let patch_dim = self.patch_dim;

        if scratch.byte_reps_host.len() < n_bytes * self.byte_dim {
            scratch.byte_reps_host.resize(n_bytes * self.byte_dim, 0.0);
        }
        byte_reps.copy_to_host(&mut scratch.byte_reps_host[..n_bytes * self.byte_dim]);
        if scratch.patch_reps_host.len() < n_patches * patch_dim {
            scratch.patch_reps_host.resize(n_patches * patch_dim, 0.0);
        }
        patch_reps.copy_to_host(&mut scratch.patch_reps_host[..n_patches * patch_dim]);

        if scratch.k_pack_host.len() < n_patches * inner {
            scratch.k_pack_host.resize(n_patches * inner, 0.0);
        }
        if scratch.v_pack_host.len() < n_patches * inner {
            scratch.v_pack_host.resize(n_patches * inner, 0.0);
        }
        for i in 0..n_patches {
            let patch_slice = &scratch.patch_reps_host[i * patch_dim..(i + 1) * patch_dim];
            cache.kv_input_pack[i * patch_dim..(i + 1) * patch_dim]
                .copy_from_slice(patch_slice);
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
            scratch.kv_normed.copy_to_host(&mut cache.kv_normed_pack
                [i * patch_dim..(i + 1) * patch_dim]);
            self.k_proj.forward(batch, &scratch.kv_normed, &mut scratch.k_step)?;
            self.v_proj.forward(batch, &scratch.kv_normed, &mut scratch.v_step)?;

            scratch.k_step.copy_to_host(&mut scratch.k_v_host_pack[..inner]);
            scratch.k_pack_host[i * inner..(i + 1) * inner]
                .copy_from_slice(&scratch.k_v_host_pack[..inner]);
            cache.k_pack[i * inner..(i + 1) * inner]
                .copy_from_slice(&scratch.k_v_host_pack[..inner]);
            scratch.v_step.copy_to_host(&mut scratch.k_v_host_pack[..inner]);
            scratch.v_pack_host[i * inner..(i + 1) * inner]
                .copy_from_slice(&scratch.k_v_host_pack[..inner]);
            cache.v_pack[i * inner..(i + 1) * inner]
                .copy_from_slice(&scratch.k_v_host_pack[..inner]);
        }

        if scratch.byte_out_host.len() < n_bytes * self.byte_dim {
            scratch.byte_out_host.resize(n_bytes * self.byte_dim, 0.0);
        }

        let max_kv = cache.max_kv_len;
        let nh = self.num_heads;
        let mut owning_patch = 0usize;
        for b in 0..n_bytes {
            while owning_patch + 1 < n_patches
                && patch_boundaries[owning_patch + 1] <= b
            {
                owning_patch += 1;
            }
            let valid_kv = owning_patch + 1;
            cache.kv_len_per_query[b] = valid_kv;

            let byte_slice = &scratch.byte_reps_host
                [b * self.byte_dim..(b + 1) * self.byte_dim];
            cache.q_input_per_query[b * self.byte_dim..(b + 1) * self.byte_dim]
                .copy_from_slice(byte_slice);
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
            scratch.q_normed.copy_to_host(&mut cache.q_normed_per_query
                [b * self.byte_dim..(b + 1) * self.byte_dim]);
            self.q_proj.forward(batch, &scratch.q_normed, &mut scratch.q_proj_out)?;
            scratch.q_proj_out.copy_to_host(&mut cache.q_proj_per_query
                [b * inner..(b + 1) * inner]);

            scratch.k_pack_dev.copy_from(&scratch.k_pack_host[..valid_kv * inner]);
            scratch.v_pack_dev.copy_from(&scratch.v_pack_host[..valid_kv * inner]);

            self.encoder_score_and_softmax(batch, valid_kv, scratch)?;
            for h in 0..nh {
                let dst_off = b * nh * max_kv + h * max_kv;
                let src_off = h * valid_kv;
                cache.softmax_per_query[dst_off..dst_off + valid_kv]
                    .copy_from_slice(&scratch.scores_tight_host[src_off..src_off + valid_kv]);
                for k in valid_kv..max_kv {
                    cache.softmax_per_query[dst_off + k] = 0.0;
                }
            }

            self.encoder_weighted_v(batch, valid_kv, scratch)?;
            cache.head_out_per_query[b * inner..(b + 1) * inner]
                .copy_from_slice(&scratch.head_out_host[..inner]);

            self.o_proj.forward(batch, &scratch.head_out, &mut scratch.o_out)?;
            scratch.o_out.copy_to_host(&mut scratch.o_out_host);
            scratch.byte_out_host[b * self.byte_dim..(b + 1) * self.byte_dim]
                .copy_from_slice(&scratch.o_out_host[..self.byte_dim]);
        }

        byte_reps_out.copy_from(&scratch.byte_out_host[..n_bytes * self.byte_dim]);
        Ok(())
    }
}

/// Backward for the encoder-direction cross-attention.
///
/// `byte_reps` is the same buffer that fed the matching forward;
/// `boundaries` is the same boundary vector. `cache` must have been
/// populated by [`CrossAttention::forward_encoder_for_backward`].
///
/// Accumulates Q/K/V/O weight gradients into `grads` (`+=` semantics:
/// the caller is responsible for zeroing `grads` once per outer
/// training step). Adds the byte-side gradient contribution into
/// `d_byte_reps_out` (also `+=`).
///
/// `d_patch_reps` is `[n_patches × patch_dim]` — the upstream
/// gradient on the encoder cross-attn's output.
pub fn cross_attn_encoder_backward(
    batch: &HipBatch,
    weights: &CrossAttention,
    _byte_reps: &GpuVec,
    boundaries: &[usize],
    cache: &CrossAttnBwdCache,
    d_patch_reps: &GpuVec,
    grads: &mut CrossAttnGrads,
    d_byte_reps_out: &mut GpuVec,
    scratch: &mut CrossAttnScratch,
) -> Result<(), ResidencyError> {
    if weights.direction != CrossAttnDirection::Encoder {
        return Err(ResidencyError::WrongVariant {
            expected: "Encoder direction",
            got: "Decoder (cross_attn_encoder_backward called on a decoder instance)",
        });
    }
    let n_patches = boundaries.len().saturating_sub(1);
    let n_bytes = boundaries.last().copied().unwrap_or(0);
    let inner = weights.inner_dim();
    let head_dim = weights.head_dim;
    let nh = weights.num_heads;
    let byte_dim = weights.byte_dim;
    let patch_dim = weights.patch_dim;
    let max_kv = cache.max_kv_len;

    debug_assert_eq!(d_patch_reps.len(), n_patches * patch_dim);
    debug_assert_eq!(d_byte_reps_out.len(), n_bytes * byte_dim);

    // Pull upstream gradient host-side once. Per-patch slices feed the
    // o_proj backward + per-head backward chain; we re-upload only the
    // small per-patch o-out slice rather than the full patch_reps grad.
    let mut d_patch_host = vec![0.0f32; n_patches * patch_dim];
    d_patch_reps.copy_to_host(&mut d_patch_host);

    // Per-step scratch (allocated once for the whole loop).
    let mut tmp_dweight_q = GpuVec::try_hip(inner * byte_dim)?;
    let mut tmp_dbias_q = GpuVec::try_hip(inner)?;
    let mut tmp_dweight_k = GpuVec::try_hip(inner * byte_dim)?;
    let mut tmp_dbias_k = GpuVec::try_hip(inner)?;
    let mut tmp_dweight_v = GpuVec::try_hip(inner * byte_dim)?;
    let mut tmp_dbias_v = GpuVec::try_hip(inner)?;
    let mut tmp_dweight_o = GpuVec::try_hip(patch_dim * inner)?;
    let mut tmp_dbias_o = GpuVec::try_hip(patch_dim)?;
    let mut tmp_dx_q = GpuVec::try_hip(byte_dim)?;
    let mut tmp_dx_k = GpuVec::try_hip(byte_dim)?;
    let mut tmp_dx_v = GpuVec::try_hip(byte_dim)?;
    let mut q_normed_dev = GpuVec::try_hip(byte_dim)?;
    let mut head_out_dev = GpuVec::try_hip(inner)?;
    let mut d_o_in_dev = GpuVec::try_hip(patch_dim)?;
    let mut d_head_out_dev = GpuVec::try_hip(inner)?;
    let mut d_q_proj_dev = GpuVec::try_hip(inner)?;
    let mut kv_normed_dev = GpuVec::try_hip(byte_dim)?;
    let mut d_k_proj_dev = GpuVec::try_hip(inner)?;
    let mut d_v_proj_dev = GpuVec::try_hip(inner)?;

    // Staging vec for the upstream byte-grad (host-side accumulator),
    // pulled to host once and uploaded once at the end.
    let mut d_byte_host_acc = vec![0.0f32; n_bytes * byte_dim];
    d_byte_reps_out.copy_to_host(&mut d_byte_host_acc);

    for i in 0..n_patches {
        let p_start = boundaries[i];
        let p_end = boundaries[i + 1];
        let p_len = p_end - p_start;

        // Stage 1: o_proj backward.
        // x = head_out_per_query[i], dy = d_patch_host[i, :].
        head_out_dev.copy_from(&cache.head_out_per_query[i * inner..(i + 1) * inner]);
        d_o_in_dev.copy_from(&d_patch_host[i * patch_dim..(i + 1) * patch_dim]);
        weights.o_proj.backward(
            batch, &head_out_dev, &d_o_in_dev,
            &mut d_head_out_dev,
            &mut tmp_dweight_o,
            &mut tmp_dbias_o,
        )?;
        accumulate_into(batch, &mut grads.dweight_o, &tmp_dweight_o, patch_dim * inner)?;
        accumulate_into(batch, &mut grads.dbias_o, &tmp_dbias_o, patch_dim)?;

        // Pull d_head_out host-side and pull this patch's saved
        // softmax / k_pack / v_pack into compact host vecs.
        let mut d_head_out_host = vec![0.0f32; inner];
        d_head_out_dev.copy_to_host(&mut d_head_out_host);

        // Stage 2 (host-side): per-head backprop through softmax and
        // through Q · K^T scaling. Cheaper than chaining device
        // dispatches for the small patch sizes typical in BLT.
        // d_softmax[h, j] = sum_i d_head_out[h, i] * V[j, h, i]
        // d_scores[h, j]  = softmax[h, j] * (d_softmax[h, j] - sum_k softmax[h, k] * d_softmax[h, k])
        // d_q_normed[h, i] = sum_j d_scores[h, j] * K[j, h, i] * qk_scale
        // d_k[j, h, i]    = d_scores[h, j] * q_normed[h, i] * qk_scale
        // d_v[j, h, i]    = softmax[h, j] * d_head_out[h, i]
        let q_proj_row = &cache.q_proj_per_query[i * inner..(i + 1) * inner];
        let mut d_q_proj_host = vec![0.0f32; inner];
        let mut d_k_pack = vec![0.0f32; p_len * inner];
        let mut d_v_pack = vec![0.0f32; p_len * inner];
        let qk = weights.qk_scale;
        for h in 0..nh {
            let sm_off = i * nh * max_kv + h * max_kv;
            let softmax_row = &cache.softmax_per_query[sm_off..sm_off + p_len];
            let q_h = &q_proj_row[h * head_dim..(h + 1) * head_dim];
            let dy_h = &d_head_out_host[h * head_dim..(h + 1) * head_dim];

            let mut d_softmax_row = vec![0.0f32; p_len];
            for j in 0..p_len {
                let v_off = (i * max_kv + j) * inner + h * head_dim;
                let v_jh = &cache.v_pack[v_off..v_off + head_dim];
                let mut s = 0.0f32;
                for k in 0..head_dim { s += dy_h[k] * v_jh[k]; }
                d_softmax_row[j] = s;
            }
            // d_scores via softmax-Jacobian (standard `(d_y - <d_y, y>) * y`).
            let mut dot = 0.0f32;
            for j in 0..p_len { dot += softmax_row[j] * d_softmax_row[j]; }
            let mut d_scores = vec![0.0f32; p_len];
            for j in 0..p_len {
                d_scores[j] = softmax_row[j] * (d_softmax_row[j] - dot);
            }

            // d_q_normed[h, i] = sum_j d_scores[h, j] * K[j, h, i] * qk_scale
            // (Q post-projection: the qk_scale multiply only happened on
            //  the score path, so it folds into d_q here.)
            let mut d_q_h = vec![0.0f32; head_dim];
            for j in 0..p_len {
                let k_off = (i * max_kv + j) * inner + h * head_dim;
                let k_jh = &cache.k_pack[k_off..k_off + head_dim];
                for k in 0..head_dim {
                    d_q_h[k] += d_scores[j] * k_jh[k] * qk;
                }
            }
            for k in 0..head_dim {
                d_q_proj_host[h * head_dim + k] += d_q_h[k];
            }
            // d_k[j, h, i] = d_scores[h, j] * q_h[i] * qk_scale
            // d_v[j, h, i] = softmax[h, j] * d_head_out[h, i]
            for j in 0..p_len {
                for k in 0..head_dim {
                    d_k_pack[j * inner + h * head_dim + k] +=
                        d_scores[j] * q_h[k] * qk;
                    d_v_pack[j * inner + h * head_dim + k] +=
                        softmax_row[j] * dy_h[k];
                }
            }
        }

        // Stage 3: q_proj backward. x = q_normed (post-LN), dy = d_q_proj.
        d_q_proj_dev.copy_from(&d_q_proj_host);
        q_normed_dev.copy_from(&cache.q_normed_per_query
            [i * byte_dim..(i + 1) * byte_dim]);
        weights.q_proj.backward(
            batch, &q_normed_dev, &d_q_proj_dev,
            &mut tmp_dx_q,
            &mut tmp_dweight_q, &mut tmp_dbias_q,
        )?;
        accumulate_into(batch, &mut grads.dweight_q, &tmp_dweight_q, inner * byte_dim)?;
        accumulate_into(batch, &mut grads.dbias_q, &tmp_dbias_q, inner)?;

        // q-side RMSNorm backward (host) → d_q_input (q_input is the
        // max-pool result on the byte slice).
        let mut d_q_normed_host = vec![0.0f32; byte_dim];
        tmp_dx_q.copy_to_host(&mut d_q_normed_host);
        let q_input_row = &cache.q_input_per_query[i * byte_dim..(i + 1) * byte_dim];
        let mut d_q_input_host = vec![0.0f32; byte_dim];
        rms_norm_backward_const(
            q_input_row, &d_q_normed_host, &mut d_q_input_host,
            byte_dim, weights.norm_eps,
        );
        // Backprop the max-pool: gradient flows to whichever byte
        // contributed the max along each dim. Compute argmax host-
        // side from the cached byte slice (recomputed from kv_input_pack).
        for k in 0..byte_dim {
            let mut argmax = 0usize;
            let mut best = f32::NEG_INFINITY;
            for j in 0..p_len {
                let off = (i * max_kv + j) * byte_dim + k;
                let v = cache.kv_input_pack[off];
                if v > best {
                    best = v; argmax = j;
                }
            }
            d_byte_host_acc[(p_start + argmax) * byte_dim + k] += d_q_input_host[k];
        }

        // Stage 4: per-byte k_proj + v_proj backward, accumulate weight
        // grads, and route the input grad through kv-side RMSNorm
        // backward into d_byte_reps_out at the source byte index.
        for j in 0..p_len {
            let kv_normed_row = &cache.kv_normed_pack
                [(i * max_kv + j) * byte_dim..(i * max_kv + j + 1) * byte_dim];
            let kv_input_row = &cache.kv_input_pack
                [(i * max_kv + j) * byte_dim..(i * max_kv + j + 1) * byte_dim];
            kv_normed_dev.copy_from(kv_normed_row);
            d_k_proj_dev.copy_from(&d_k_pack[j * inner..(j + 1) * inner]);
            d_v_proj_dev.copy_from(&d_v_pack[j * inner..(j + 1) * inner]);

            weights.k_proj.backward(
                batch, &kv_normed_dev, &d_k_proj_dev,
                &mut tmp_dx_k,
                &mut tmp_dweight_k, &mut tmp_dbias_k,
            )?;
            accumulate_into(batch, &mut grads.dweight_k, &tmp_dweight_k, inner * byte_dim)?;
            accumulate_into(batch, &mut grads.dbias_k, &tmp_dbias_k, inner)?;

            weights.v_proj.backward(
                batch, &kv_normed_dev, &d_v_proj_dev,
                &mut tmp_dx_v,
                &mut tmp_dweight_v, &mut tmp_dbias_v,
            )?;
            accumulate_into(batch, &mut grads.dweight_v, &tmp_dweight_v, inner * byte_dim)?;
            accumulate_into(batch, &mut grads.dbias_v, &tmp_dbias_v, inner)?;

            // dx_kv = dx_from_k + dx_from_v  (post-norm gradient).
            let mut d_kv_normed = vec![0.0f32; byte_dim];
            let mut tmp_h = vec![0.0f32; byte_dim];
            tmp_dx_k.copy_to_host(&mut tmp_h);
            for k in 0..byte_dim { d_kv_normed[k] += tmp_h[k]; }
            tmp_dx_v.copy_to_host(&mut tmp_h);
            for k in 0..byte_dim { d_kv_normed[k] += tmp_h[k]; }

            let mut d_kv_input = vec![0.0f32; byte_dim];
            rms_norm_backward_const(
                kv_input_row, &d_kv_normed, &mut d_kv_input,
                byte_dim, weights.norm_eps,
            );
            let byte_idx = p_start + j;
            for k in 0..byte_dim {
                d_byte_host_acc[byte_idx * byte_dim + k] += d_kv_input[k];
            }
        }
    }

    d_byte_reps_out.copy_from(&d_byte_host_acc);
    let _ = scratch;
    Ok(())
}

/// Backward for the decoder-direction cross-attention.
///
/// `byte_reps_q` is the per-byte Q source; `patch_reps_kv` is the
/// per-patch K/V source; both are the same buffers fed to the
/// matching forward. `cache` must have been populated by
/// [`CrossAttention::forward_decoder_for_backward`].
///
/// Adds Q-side byte gradient into `d_byte_reps_q_out` and the K/V-side
/// patch gradient into `d_patch_reps_out`. Both use `+=` semantics.
pub fn cross_attn_decoder_backward(
    batch: &HipBatch,
    weights: &CrossAttention,
    _byte_reps_q: &GpuVec,
    _patch_reps_kv: &GpuVec,
    boundaries: &[usize],
    cache: &CrossAttnBwdCache,
    d_byte_reps: &GpuVec,
    grads: &mut CrossAttnGrads,
    d_byte_reps_q_out: &mut GpuVec,
    d_patch_reps_out: &mut GpuVec,
    scratch: &mut CrossAttnScratch,
) -> Result<(), ResidencyError> {
    if weights.direction != CrossAttnDirection::Decoder {
        return Err(ResidencyError::WrongVariant {
            expected: "Decoder direction",
            got: "Encoder (cross_attn_decoder_backward called on encoder instance)",
        });
    }
    let n_patches = boundaries.len().saturating_sub(1);
    let n_bytes = boundaries.last().copied().unwrap_or(0);
    let inner = weights.inner_dim();
    let head_dim = weights.head_dim;
    let nh = weights.num_heads;
    let byte_dim = weights.byte_dim;
    let patch_dim = weights.patch_dim;
    let max_kv = cache.max_kv_len;
    let qk = weights.qk_scale;

    debug_assert_eq!(d_byte_reps.len(), n_bytes * byte_dim);
    debug_assert_eq!(d_byte_reps_q_out.len(), n_bytes * byte_dim);
    debug_assert_eq!(d_patch_reps_out.len(), n_patches * patch_dim);

    let mut d_byte_host = vec![0.0f32; n_bytes * byte_dim];
    d_byte_reps.copy_to_host(&mut d_byte_host);
    let mut d_byte_q_acc = vec![0.0f32; n_bytes * byte_dim];
    d_byte_reps_q_out.copy_to_host(&mut d_byte_q_acc);
    let mut d_patch_acc = vec![0.0f32; n_patches * patch_dim];
    d_patch_reps_out.copy_to_host(&mut d_patch_acc);

    // Per-patch K/V projection grad accumulators (across bytes).
    // K/V projections are amortised once per patch in the forward, so the
    // patch-side weight grads accept one combined backward call per patch.
    let mut d_k_pack_acc = vec![0.0f32; n_patches * inner];
    let mut d_v_pack_acc = vec![0.0f32; n_patches * inner];

    let mut tmp_dweight_q = GpuVec::try_hip(inner * byte_dim)?;
    let mut tmp_dbias_q = GpuVec::try_hip(inner)?;
    let mut tmp_dweight_k = GpuVec::try_hip(inner * patch_dim)?;
    let mut tmp_dbias_k = GpuVec::try_hip(inner)?;
    let mut tmp_dweight_v = GpuVec::try_hip(inner * patch_dim)?;
    let mut tmp_dbias_v = GpuVec::try_hip(inner)?;
    let mut tmp_dweight_o = GpuVec::try_hip(byte_dim * inner)?;
    let mut tmp_dbias_o = GpuVec::try_hip(byte_dim)?;
    let mut tmp_dx_q = GpuVec::try_hip(byte_dim)?;
    let mut tmp_dx_k = GpuVec::try_hip(patch_dim)?;
    let mut tmp_dx_v = GpuVec::try_hip(patch_dim)?;
    let mut head_out_dev = GpuVec::try_hip(inner)?;
    let mut d_o_in_dev = GpuVec::try_hip(byte_dim)?;
    let mut d_head_out_dev = GpuVec::try_hip(inner)?;
    let mut d_q_proj_dev = GpuVec::try_hip(inner)?;
    let mut q_normed_dev = GpuVec::try_hip(byte_dim)?;
    let mut d_k_proj_dev = GpuVec::try_hip(inner)?;
    let mut d_v_proj_dev = GpuVec::try_hip(inner)?;
    let mut kv_normed_dev = GpuVec::try_hip(patch_dim)?;

    for b in 0..n_bytes {
        let valid_kv = cache.kv_len_per_query[b];

        // Stage 1: o_proj backward.
        head_out_dev.copy_from(&cache.head_out_per_query[b * inner..(b + 1) * inner]);
        d_o_in_dev.copy_from(&d_byte_host[b * byte_dim..(b + 1) * byte_dim]);
        weights.o_proj.backward(
            batch, &head_out_dev, &d_o_in_dev,
            &mut d_head_out_dev,
            &mut tmp_dweight_o, &mut tmp_dbias_o,
        )?;
        accumulate_into(batch, &mut grads.dweight_o, &tmp_dweight_o, byte_dim * inner)?;
        accumulate_into(batch, &mut grads.dbias_o, &tmp_dbias_o, byte_dim)?;

        let mut d_head_out_host = vec![0.0f32; inner];
        d_head_out_dev.copy_to_host(&mut d_head_out_host);

        let q_proj_row = &cache.q_proj_per_query[b * inner..(b + 1) * inner];
        let mut d_q_proj_host = vec![0.0f32; inner];

        for h in 0..nh {
            let sm_off = b * nh * max_kv + h * max_kv;
            let softmax_row = &cache.softmax_per_query[sm_off..sm_off + valid_kv];
            let q_h = &q_proj_row[h * head_dim..(h + 1) * head_dim];
            let dy_h = &d_head_out_host[h * head_dim..(h + 1) * head_dim];

            let mut d_softmax_row = vec![0.0f32; valid_kv];
            for p in 0..valid_kv {
                let v_off = p * inner + h * head_dim;
                let v_ph = &cache.v_pack[v_off..v_off + head_dim];
                let mut s = 0.0f32;
                for k in 0..head_dim { s += dy_h[k] * v_ph[k]; }
                d_softmax_row[p] = s;
            }
            let mut dot = 0.0f32;
            for p in 0..valid_kv { dot += softmax_row[p] * d_softmax_row[p]; }
            let mut d_scores = vec![0.0f32; valid_kv];
            for p in 0..valid_kv {
                d_scores[p] = softmax_row[p] * (d_softmax_row[p] - dot);
            }

            // d_q (this byte) and accumulate d_k, d_v into per-patch slabs.
            for p in 0..valid_kv {
                let k_off = p * inner + h * head_dim;
                let k_ph = &cache.k_pack[k_off..k_off + head_dim];
                for k in 0..head_dim {
                    d_q_proj_host[h * head_dim + k] += d_scores[p] * k_ph[k] * qk;
                    d_k_pack_acc[p * inner + h * head_dim + k] +=
                        d_scores[p] * q_h[k] * qk;
                    d_v_pack_acc[p * inner + h * head_dim + k] +=
                        softmax_row[p] * dy_h[k];
                }
            }
        }

        // Stage 2: q_proj backward.
        d_q_proj_dev.copy_from(&d_q_proj_host);
        q_normed_dev.copy_from(&cache.q_normed_per_query
            [b * byte_dim..(b + 1) * byte_dim]);
        weights.q_proj.backward(
            batch, &q_normed_dev, &d_q_proj_dev,
            &mut tmp_dx_q,
            &mut tmp_dweight_q, &mut tmp_dbias_q,
        )?;
        accumulate_into(batch, &mut grads.dweight_q, &tmp_dweight_q, inner * byte_dim)?;
        accumulate_into(batch, &mut grads.dbias_q, &tmp_dbias_q, inner)?;

        // Q-side RMSNorm backward → d_byte_reps_q_out[b].
        let mut d_q_normed_host = vec![0.0f32; byte_dim];
        tmp_dx_q.copy_to_host(&mut d_q_normed_host);
        let q_input_row = &cache.q_input_per_query[b * byte_dim..(b + 1) * byte_dim];
        let mut d_q_input_host = vec![0.0f32; byte_dim];
        rms_norm_backward_const(
            q_input_row, &d_q_normed_host, &mut d_q_input_host,
            byte_dim, weights.norm_eps,
        );
        for k in 0..byte_dim {
            d_byte_q_acc[b * byte_dim + k] += d_q_input_host[k];
        }
    }

    // Per-patch K/V backward (one per patch, summed across all attending bytes).
    for p in 0..n_patches {
        let kv_input_row = &cache.kv_input_pack[p * patch_dim..(p + 1) * patch_dim];
        let kv_normed_row = &cache.kv_normed_pack[p * patch_dim..(p + 1) * patch_dim];
        kv_normed_dev.copy_from(kv_normed_row);
        d_k_proj_dev.copy_from(&d_k_pack_acc[p * inner..(p + 1) * inner]);
        d_v_proj_dev.copy_from(&d_v_pack_acc[p * inner..(p + 1) * inner]);

        weights.k_proj.backward(
            batch, &kv_normed_dev, &d_k_proj_dev,
            &mut tmp_dx_k,
            &mut tmp_dweight_k, &mut tmp_dbias_k,
        )?;
        accumulate_into(batch, &mut grads.dweight_k, &tmp_dweight_k, inner * patch_dim)?;
        accumulate_into(batch, &mut grads.dbias_k, &tmp_dbias_k, inner)?;
        weights.v_proj.backward(
            batch, &kv_normed_dev, &d_v_proj_dev,
            &mut tmp_dx_v,
            &mut tmp_dweight_v, &mut tmp_dbias_v,
        )?;
        accumulate_into(batch, &mut grads.dweight_v, &tmp_dweight_v, inner * patch_dim)?;
        accumulate_into(batch, &mut grads.dbias_v, &tmp_dbias_v, inner)?;

        let mut d_kv_normed = vec![0.0f32; patch_dim];
        let mut tmp_h = vec![0.0f32; patch_dim];
        tmp_dx_k.copy_to_host(&mut tmp_h);
        for k in 0..patch_dim { d_kv_normed[k] += tmp_h[k]; }
        tmp_dx_v.copy_to_host(&mut tmp_h);
        for k in 0..patch_dim { d_kv_normed[k] += tmp_h[k]; }

        let mut d_kv_input = vec![0.0f32; patch_dim];
        rms_norm_backward_const(
            kv_input_row, &d_kv_normed, &mut d_kv_input,
            patch_dim, weights.norm_eps,
        );
        for k in 0..patch_dim {
            d_patch_acc[p * patch_dim + k] += d_kv_input[k];
        }
    }

    d_byte_reps_q_out.copy_from(&d_byte_q_acc);
    d_patch_reps_out.copy_from(&d_patch_acc);
    let _ = scratch;
    Ok(())
}

/// Add `src[..n]` into `dst[..n]` on the device. Used to merge per-step
/// `LinearResident::backward` outputs (which overwrite their dweight /
/// dbias targets) into a running accumulator.
fn accumulate_into(
    batch: &HipBatch,
    dst: &mut GpuVec,
    src: &GpuVec,
    n: usize,
) -> Result<(), ResidencyError> {
    unsafe {
        op_tensor_resident(
            hip_buf(dst)?.device_ptr() as *const f32,
            hip_buf(src)?.device_ptr() as *const f32,
            hip_buf_mut(dst)?.device_ptr() as *mut f32,
            n, 1.0, 1.0, 0.0, BinaryOpKind::Add,
        )?;
    }
    batch.note_dispatch()?;
    Ok(())
}

/// RMSNorm backward with constant scale 1.0, no learnable gamma.
/// Forward: `y[i] = x[i] / rms(x)`, `rms = sqrt(mean(x²) + eps)`.
/// Same shape as `rms_norm_backward_per_head` in `modgrad-transformer`,
/// but kept local so this crate doesn't depend on that internal helper.
fn rms_norm_backward_const(
    x: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    dim: usize,
    eps: f32,
) {
    let n = dim as f32;
    let mean_sq: f32 = x.iter().take(dim).map(|&v| v * v).sum::<f32>() / n;
    let rms = (mean_sq + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_rms_sq = inv_rms * inv_rms;
    let acc: f32 = x.iter().take(dim).zip(dy.iter().take(dim))
        .map(|(&a, &b)| a * b).sum();
    for i in 0..dim {
        dx[i] = inv_rms * (dy[i] - x[i] * acc * inv_rms_sq / n);
    }
}

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

    /// Smoke test for the encoder backward path. Builds the same tiny
    /// CrossAttention used by `encoder_forward_matches_host_reference`,
    /// runs `forward_encoder_for_backward` to populate the cache, then
    /// drives `cross_attn_encoder_backward` and asserts:
    ///   - no panic / dispatch error
    ///   - all weight grads contain finite values
    ///   - the byte-side gradient is finite and non-zero somewhere
    ///
    /// Without a real HIP runtime this short-circuits via the same
    /// `try_hip` failure path the forward test uses.
    #[test]
    fn encoder_backward_smoke() {
        if std::env::var("MODGRAD_SKIP_HIP_TESTS").is_ok() { return; }
        const BYTE_DIM: usize = 8;
        const PATCH_DIM: usize = 8;
        const NUM_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const N_BYTES: usize = 16;
        const N_PATCHES: usize = 4;

        let config = CrossAttnConfig {
            byte_dim: BYTE_DIM, patch_dim: PATCH_DIM,
            num_heads: NUM_HEADS, head_dim: HEAD_DIM,
            norm_eps: 1e-5, direction: CrossAttnDirection::Encoder,
        };
        let attn = match CrossAttention::new(&config, 0xC0DE_BEEF_CAFE_F00D) {
            Ok(a) => a, Err(_) => return,
        };
        let mut scratch = match CrossAttnScratch::new(&config, 4, N_BYTES, N_PATCHES) {
            Ok(s) => s, Err(_) => return,
        };
        let mut cache = match CrossAttnBwdCache::new(&config, N_PATCHES, 4) {
            Ok(c) => c, Err(_) => return,
        };
        let mut grads = match CrossAttnGrads::zeros_for_config(&config) {
            Ok(g) => g, Err(_) => return,
        };

        let mut byte_rng = SimpleRng::new(0xFEED_CAFE_BABE_0001);
        let byte_reps_host: Vec<f32> =
            (0..N_BYTES * BYTE_DIM).map(|_| byte_rng.next_normal() * 0.5).collect();
        let mut byte_reps = match GpuVec::try_hip(N_BYTES * BYTE_DIM) {
            Ok(v) => v, Err(_) => return,
        };
        byte_reps.copy_from(&byte_reps_host);
        let boundaries: Vec<usize> = vec![0, 4, 8, 12, 16];
        let mut patch_reps_out = match GpuVec::try_hip(N_PATCHES * PATCH_DIM) {
            Ok(v) => v, Err(_) => return,
        };

        let batch = HipBatch::new();
        let _ = grads.zero_resident(&batch);
        attn.forward_encoder_for_backward(
            &batch, &byte_reps, &boundaries, &mut scratch, &mut cache,
            &mut patch_reps_out,
        ).expect("forward_for_backward");

        // Synthetic upstream gradient — random unit-scale floats.
        let d_patch_host: Vec<f32> =
            (0..N_PATCHES * PATCH_DIM).map(|_| byte_rng.next_normal()).collect();
        let mut d_patch_reps = match GpuVec::try_hip(N_PATCHES * PATCH_DIM) {
            Ok(v) => v, Err(_) => return,
        };
        d_patch_reps.copy_from(&d_patch_host);

        let mut d_byte_reps_out = match GpuVec::try_hip(N_BYTES * BYTE_DIM) {
            Ok(v) => v, Err(_) => return,
        };
        let zeros = vec![0.0f32; N_BYTES * BYTE_DIM];
        d_byte_reps_out.copy_from(&zeros);

        cross_attn_encoder_backward(
            &batch, &attn, &byte_reps, &boundaries, &cache,
            &d_patch_reps, &mut grads, &mut d_byte_reps_out, &mut scratch,
        ).expect("encoder backward");
        batch.flush().expect("flush");

        // Sanity-check: every weight grad finite, byte grad has non-zero norm.
        let mut dwq = vec![0.0f32; (NUM_HEADS * HEAD_DIM) * BYTE_DIM];
        grads.dweight_q.copy_to_host(&mut dwq);
        for &v in &dwq { assert!(v.is_finite(), "dweight_q non-finite"); }
        let mut dwo = vec![0.0f32; PATCH_DIM * (NUM_HEADS * HEAD_DIM)];
        grads.dweight_o.copy_to_host(&mut dwo);
        for &v in &dwo { assert!(v.is_finite(), "dweight_o non-finite"); }
        let mut dby = vec![0.0f32; N_BYTES * BYTE_DIM];
        d_byte_reps_out.copy_to_host(&mut dby);
        let mut sumsq = 0.0f32;
        for &v in &dby { assert!(v.is_finite(), "d_byte_reps non-finite"); sumsq += v * v; }
        assert!(sumsq > 0.0, "d_byte_reps_out should be non-zero somewhere");
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
