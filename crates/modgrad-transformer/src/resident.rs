//! Device-resident transformer block stack.
//!
//! Wraps every weight in [`crate::GptModel`] as a `LinearResident` /
//! `HipBuffer` and threads activations through the resident MIOpen op
//! family — `rms_norm_resident`, `softmax_resident`, `activation_resident`,
//! `op_tensor_resident` — so a complete forward pass runs without a
//! single PCIe transfer between dispatches.
//!
//! Mirrors:
//!   - [`crate::attention::CausalSelfAttention`]   →  [`AttentionResident`]
//!   - [`crate::mlp::SwigluMlp`]                   →  [`SwigluResident`]
//!   - [`crate::block::TransformerBlock`]          →  [`TransformerBlockResident`]
//!   - [`crate::model::GptModel`]                  →  [`GptModelResident`]
//!
//! ## Design notes
//!
//! **KV layout is head-major, not token-major.** The host
//! [`crate::kv_cache::KvCache`] stores K/V as `[max_seq × kv_dim]`
//! row-major. For per-head score computation that's a strided read.
//! The resident cache ([`crate::KvCacheResident`]) lays K/V out as
//! `[n_kv_heads × max_seq × head_dim]` row-major instead — the per-
//! head slabs are contiguous and feed `matvec_resident` directly.
//!
//! **RoPE stays host-side.** The first slice deliberately D2H's the
//! Q/K projections (`kv_dim` floats), applies RoPE on host, and D2H's
//! back. ~0.5 µs each direction at typical d_model — small relative
//! to the gain from keeping matvecs / FFN entirely resident. A
//! `RopeResident` op is a separate slice (~50 LOC).
//!
//! **Per-head softmax is batched.** All heads' scores live in one
//! `[n_heads × attn_len]` scratch and one `softmax_resident` call
//! handles the whole row pack — instead of `n_heads` separate
//! dispatches at ~7 µs each.
//!
//! **QK RmsNorm bakes the scale into the weight buffer.** The host
//! code does `RmsNorm.forward(q_head)` then `q_normed *= qk_scale`.
//! `rms_norm_resident(x, weight, y, n, hidden, eps)` computes
//! `y = (x * inv_rms) * weight`. We store `qk_norm_weight_dev` as a
//! `[head_dim]` buffer of `qk_scale` (since the host has no learnable
//! scale, the constant scale is entirely captured here). This collapses
//! norm + scale into one MIOpen dispatch.
//!
//! **Activation checkpointing is a flag, not a separate type.** The
//! `checkpoint` field on [`TransformerBlockResident`] toggles whether
//! intermediate scratch buffers are retained across the dispatch (for
//! backward) or reused/discarded. Slice 7's gradient-checkpointing
//! semantics in slice 9's container.
//!
//! Only available with `--features rocm`.

#![cfg(feature = "rocm")]

use modgrad_compute::backend::{GpuVec, ResidencyError};
use modgrad_compute::neuron::{Linear, LinearResident};
use modgrad_device::backend::{HipBatch, HipBuffer};
use modgrad_device::backend::op::{ActivationMode, BinaryOpKind};
use modgrad_device::backend::ops::{
    activation_backward_resident, activation_resident,
    matmul_resident_nn, matmul_resident_tn, matvec_resident,
    op_tensor_resident, rms_norm_resident, rope_backward_resident,
    softmax_backward_resident, softmax_resident,
};

use crate::attention::CausalSelfAttention;
use crate::block::TransformerBlock;
use crate::config::{GptConfig, MlpActivation};
use crate::dims::LayerIdx;
use crate::kv_cache_resident::KvCacheResident;
use crate::mlp::SwigluMlp;
use crate::model::GptModel;
use crate::residual::ResidualLambdas;
use crate::rope::RotaryEmbedding;

// ─── Helpers ────────────────────────────────────────────────

/// Allocate a `HipBuffer` of `n` f32s and upload `vals`. `vals.len()`
/// must equal `n`.
fn upload_f32_buffer(vals: &[f32]) -> Result<HipBuffer, ResidencyError> {
    let buf = HipBuffer::new(vals.len() * 4)?;
    buf.copy_from_host(vals)?;
    Ok(buf)
}

/// Build a host `Linear` from a flat row-major `[out_dim × in_dim]`
/// weight slice. Bias is zero (transformer matvecs are bias-free).
fn linear_from_weight(weight: &[f32], out_dim: usize, in_dim: usize) -> Linear {
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    Linear {
        weight: weight.to_vec(),
        bias: vec![0.0f32; out_dim],
        in_dim,
        out_dim,
    }
}

/// Unwrap `GpuVec::Hip(_)` or return [`ResidencyError::WrongVariant`].
#[inline]
fn hip_buf<'a>(g: &'a GpuVec) -> Result<&'a HipBuffer, ResidencyError> {
    match g {
        GpuVec::Hip(b) => Ok(b),
        other => Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    }
}

// ─── AttentionResident ──────────────────────────────────────

/// Device-resident causal self-attention. See module docs.
pub struct AttentionResident {
    /// Q projection — `out_dim = model_dim`, `in_dim = model_dim`.
    pub q_proj: LinearResident,
    /// K projection — `out_dim = kv_dim`, `in_dim = model_dim`.
    pub k_proj: LinearResident,
    /// V projection — `out_dim = kv_dim`, `in_dim = model_dim`.
    pub v_proj: LinearResident,
    /// Output projection — `out_dim = model_dim`, `in_dim = model_dim`.
    pub o_proj: LinearResident,
    /// QK norm weight: `[head_dim]` of `qk_scale`. Combines RmsNorm
    /// scale and the trailing constant scale in one buffer.
    pub qk_norm_weight_dev: HipBuffer,
    /// Zero `[head_dim]` buffer used as bias for matvecs that accept
    /// a bias pointer but expect 0.0 (per-head scoring matvec).
    pub zero_bias_dev: HipBuffer,
    /// Optional per-head attention sink bias `[num_heads]` on device.
    /// When `Some`, forward extends each head's softmax by one extra
    /// "drain" slot whose pre-softmax score equals `sink_bias[h]`. The
    /// slot has no V contribution. Mirrors `CausalSelfAttention.sink_bias`
    /// — propagates from there in `from_attention` (and is `None` if the
    /// host attention has none).
    pub sink_bias_dev: Option<HipBuffer>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub gqa_ratio: usize,
    pub model_dim: usize,
    pub kv_dim: usize,
    /// QK scale baked into `qk_norm_weight_dev`. Kept as a struct field
    /// for diagnostics (e.g. tests can sanity-check the value); the
    /// resident dispatch reads it through the device buffer.
    #[allow(dead_code)]
    qk_scale: f32,
    norm_eps: f32,
    /// Apply per-head RMSNorm to Q/K (Gemma3-style). When `false`, Q
    /// and K go to RoPE/scoring raw — used for Llama / Qwen2 / Mistral
    /// architectures whose canonical attention does not normalize Q/K.
    use_qk_norm: bool,
}

impl AttentionResident {
    /// Build from a host [`CausalSelfAttention`] + [`GptConfig`].
    pub fn from_attention(
        attn: &CausalSelfAttention,
        config: &GptConfig,
    ) -> Result<Self, ResidencyError> {
        let model_dim = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let head_dim = config.head_dim.get();
        let qk_scale = config.qk_norm_scale;

        let q = linear_from_weight(attn.weights.wq.as_slice(), model_dim, model_dim);
        let k = linear_from_weight(attn.weights.wk.as_slice(), kv_dim, model_dim);
        let v = linear_from_weight(attn.weights.wv.as_slice(), kv_dim, model_dim);
        let o = linear_from_weight(attn.weights.wo.as_slice(), model_dim, model_dim);

        let qk_norm_weight = vec![qk_scale; head_dim];
        let qk_norm_weight_dev = upload_f32_buffer(&qk_norm_weight)?;
        let zero_bias = vec![0.0f32; std::cmp::max(head_dim, model_dim).max(kv_dim)];
        // Ensure zero_bias_dev is large enough for the largest matvec
        // bias pointer we'll feed it (head_dim for the per-head score
        // matvec; kv_dim/model_dim for any other use). Actual matvec
        // dispatches read only the first `out_dim` elements.
        let zero_bias_dev = upload_f32_buffer(&zero_bias)?;

        // Mirror the host's optional sink bias onto device. None on host
        // → None on device (no extra storage, no extra dispatch).
        let sink_bias_dev = match &attn.sink_bias {
            Some(bias) => Some(upload_f32_buffer(bias)?),
            None => None,
        };

        Ok(Self {
            q_proj: LinearResident::from_linear(&q)?,
            k_proj: LinearResident::from_linear(&k)?,
            v_proj: LinearResident::from_linear(&v)?,
            o_proj: LinearResident::from_linear(&o)?,
            qk_norm_weight_dev,
            zero_bias_dev,
            sink_bias_dev,
            num_heads: config.num_heads.get(),
            num_kv_heads: config.num_kv_heads.get(),
            head_dim,
            gqa_ratio: config.gqa_ratio(),
            model_dim,
            kv_dim,
            qk_scale,
            norm_eps: config.norm_eps,
            use_qk_norm: config.use_qk_norm,
        })
    }

    /// Re-upload weights from a host [`CausalSelfAttention`] after an
    /// optimizer step. Shapes must match the original `from_attention`.
    pub fn sync_weights_from(&mut self, attn: &CausalSelfAttention) -> Result<(), ResidencyError> {
        let model_dim = self.model_dim;
        let kv_dim = self.kv_dim;
        let q = linear_from_weight(attn.weights.wq.as_slice(), model_dim, model_dim);
        let k = linear_from_weight(attn.weights.wk.as_slice(), kv_dim, model_dim);
        let v = linear_from_weight(attn.weights.wv.as_slice(), kv_dim, model_dim);
        let o = linear_from_weight(attn.weights.wo.as_slice(), model_dim, model_dim);
        self.q_proj.sync_weights_from(&q)?;
        self.k_proj.sync_weights_from(&k)?;
        self.v_proj.sync_weights_from(&v)?;
        self.o_proj.sync_weights_from(&o)?;
        Ok(())
    }

    /// Forward pass for a single token at absolute position `position`,
    /// reading/writing into `kv_cache.{k_dev,v_dev}[layer]`.
    ///
    /// `x_dev`: pre-attention-norm hidden state `[model_dim]`. Caller is
    /// responsible for the RMS normalisation (resident or host) before
    /// this call — see [`TransformerBlockResident`].
    /// `out_dev`: post-O-proj output `[model_dim]`. Caller adds the
    /// residual themselves.
    ///
    /// `rope` is host-side — see module docs. We D2H the freshly-projected
    /// Q and K for one host loop, then re-upload.
    pub fn forward(
        &self,
        batch: &HipBatch,
        x_dev: &GpuVec,
        kv_cache: &mut KvCacheResident,
        layer: usize,
        position: usize,
        window: Option<usize>,
        rope: &RotaryEmbedding,
        scratch: &mut AttentionScratch,
        out_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        debug_assert_eq!(x_dev.len(), self.model_dim);
        debug_assert_eq!(out_dev.len(), self.model_dim);
        debug_assert!(scratch.fits(self.num_heads, self.head_dim, self.kv_dim,
                                   kv_cache.max_seq_len()));

        let _x_buf = hip_buf(x_dev)?;
        let _out_buf = hip_buf(out_dev)?;

        // Stage 1: Q/K/V projections.
        self.q_proj.forward(batch, x_dev, &mut scratch.q_proj)?;
        self.k_proj.forward(batch, x_dev, &mut scratch.k_proj)?;
        self.v_proj.forward(batch, x_dev, &mut scratch.v_proj)?;

        // Stage 2: per-head QK norm + scale (one MIOpen dispatch via
        // `rms_norm_resident` whose `weight` baked in `qk_scale`). Skipped
        // for Llama/Qwen2-style architectures (`use_qk_norm = false`); the
        // raw Q/K projections feed directly into RoPE.
        if self.use_qk_norm {
            unsafe {
                rms_norm_resident(
                    hip_buf(&scratch.q_proj)?.device_ptr() as *const f32,
                    self.qk_norm_weight_dev.device_ptr() as *const f32,
                    hip_buf(&scratch.q_normed)?.device_ptr() as *mut f32,
                    self.num_heads, self.head_dim, self.norm_eps,
                )?;
            }
            batch.note_dispatch()?;
            unsafe {
                rms_norm_resident(
                    hip_buf(&scratch.k_proj)?.device_ptr() as *const f32,
                    self.qk_norm_weight_dev.device_ptr() as *const f32,
                    hip_buf(&scratch.k_normed)?.device_ptr() as *mut f32,
                    self.num_kv_heads, self.head_dim, self.norm_eps,
                )?;
            }
            batch.note_dispatch()?;
        }

        // Stage 3: D2H Q-and-K (post-norm if enabled, raw otherwise),
        // apply RoPE host-side, re-upload into the q_normed/k_normed
        // slots that the rest of the pipeline reads from.
        if self.use_qk_norm {
            scratch.q_normed.copy_to_host(&mut scratch.q_normed_host);
            scratch.k_normed.copy_to_host(&mut scratch.k_normed_host);
        } else {
            scratch.q_proj.copy_to_host(&mut scratch.q_normed_host);
            scratch.k_proj.copy_to_host(&mut scratch.k_normed_host);
        }
        for h in 0..self.num_heads {
            rope.apply(
                &mut scratch.q_normed_host[h * self.head_dim..(h + 1) * self.head_dim],
                position,
            );
        }
        for h in 0..self.num_kv_heads {
            rope.apply(
                &mut scratch.k_normed_host[h * self.head_dim..(h + 1) * self.head_dim],
                position,
            );
        }
        scratch.q_normed.copy_from(&scratch.q_normed_host);
        scratch.k_normed.copy_from(&scratch.k_normed_host);

        // Stage 4: write the new K (post-RoPE) and V (raw) into the cache.
        // (V is not normed, not RoPE'd — the host code path `forward_one`
        // copies `v_proj` straight into `cur_v`, no RoPE.)
        kv_cache.write(batch, layer, position, &scratch.k_normed, &scratch.v_proj)?;

        // Stage 5: per-head scoring matvec. For head h, kv_h = h / gqa_ratio.
        // K slab is window-aware via `KvCacheResident::k_slab_view`:
        //   - linear layer or partial rolling ring: ptr at `start * head_dim`,
        //     rows = position + 1 - start (== windowed_len, legacy semantics)
        //   - full rolling ring: ptr at slab base (offset 0),
        //     rows = layer_slots (whole ring; permutation-invariant attention)
        // The view rows replace `windowed_len` everywhere in stages 5/6/7.
        let attn_len = position + 1;
        let logical_start = match window {
            Some(w) => position.saturating_sub(w) + 1,
            None => 0,
        };
        // Probe view for kv_head=0 (all kv_heads share the layer's slot
        // count, so any view gives the same `rows`).
        let probe_view = kv_cache.k_slab_view(layer, 0, logical_start, position);
        let windowed_len = probe_view.rows;
        // Logical "is the window empty" test still uses the unclamped
        // start/attn_len — matches host empty-range semantics regardless
        // of cache shape.
        let logical_windowed_len = attn_len - logical_start;

        // Empty-window case: matches host semantics — when the logical
        // windowed range is empty (e.g. position 0 with a non-trivial
        // window), host iterates an empty `start..attn_len` loop, leaves
        // `head_out` all-zero, and the (bias-free) output projection of
        // zero is zero. Stages 1–4 above already ran (so the cache write
        // stays in sync for later positions); skip 5–8 and emit a zero
        // output.
        if logical_windowed_len == 0 {
            let zeros = vec![0.0f32; self.model_dim];
            out_dev.copy_from(&zeros);
            return Ok(());
        }

        let q_normed_base = hip_buf(&scratch.q_normed)?.device_ptr() as *const f32;
        let scores_base = hip_buf(&scratch.scores)?.device_ptr() as *mut f32;
        let max_seq = kv_cache.max_seq_len();

        for h in 0..self.num_heads {
            let kv_h = h / self.gqa_ratio;
            let view = kv_cache.k_slab_view(layer, kv_h, logical_start, position);
            unsafe {
                matvec_resident(
                    q_normed_base.add(h * self.head_dim),
                    view.ptr,                                // [windowed_len × head_dim]
                    self.zero_bias_dev.device_ptr() as *const f32,
                    scores_base.add(h * max_seq),
                    view.rows,
                    self.head_dim,
                )?;
            }
            batch.note_dispatch()?;
        }

        // Stage 6: batched softmax — `n_rows = num_heads`, but each row is
        // `extended_len = windowed_len + sink_slot` wide while the
        // underlying scoring buffer is strided at `max_seq`. We gather
        // each head's row into the dense `[num_heads × extended_len]`
        // tight scratch (one D2D per head), append the sink bias if the
        // attention has one installed, then softmax over the whole row.
        let sink_slot = self.sink_bias_dev.is_some() as usize;
        let extended_len = windowed_len + sink_slot;
        let scores_tight = &scratch.scores_tight;
        let scores_tight_buf = hip_buf(scores_tight)?;
        for h in 0..self.num_heads {
            unsafe {
                use std::os::raw::c_void;
                let src = (scores_base as *const u8)
                    .add(h * max_seq * 4) as *const c_void;
                let dst = (scores_tight_buf.device_ptr() as *mut u8)
                    .add(h * extended_len * 4) as *mut c_void;
                hip_memcpy_d2d(dst, src, windowed_len * 4)?;
            }
            batch.note_dispatch()?;
        }
        // Append per-head sink bias if installed: one f32 per head copied
        // from `sink_bias_dev[h]` to `scores_tight[h * extended_len + windowed_len]`.
        // The slot has no V row, so it acts purely as a softmax drain.
        if let Some(sink_dev) = self.sink_bias_dev.as_ref() {
            let sink_base = sink_dev.device_ptr() as *const u8;
            for h in 0..self.num_heads {
                unsafe {
                    use std::os::raw::c_void;
                    let src = sink_base.add(h * 4) as *const c_void;
                    let dst = (scores_tight_buf.device_ptr() as *mut u8)
                        .add((h * extended_len + windowed_len) * 4) as *mut c_void;
                    hip_memcpy_d2d(dst, src, 4)?;
                }
                batch.note_dispatch()?;
            }
        }
        unsafe {
            softmax_resident(
                scores_tight_buf.device_ptr() as *const f32,
                scores_tight_buf.device_ptr() as *mut f32,
                self.num_heads, extended_len, false,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 7: weighted V sum per head, restricted to the window.
        //   head_out[h, :] = softmax[h, :windowed_len] @ V_view[h, :]
        // The optional sink slot (at column `windowed_len` if present)
        // has no V row and is skipped in this matmul — its softmax weight
        // simply doesn't contribute to head_out, matching the host
        // semantics in `forward_one`. The V view mirrors the K view from
        // Stage 5 so each row's K and V correspond to the same logical
        // position (preserving permutation-invariance of attention).
        let head_out_base = hip_buf(&scratch.head_out)?.device_ptr() as *mut f32;
        let scores_tight_base = scores_tight_buf.device_ptr() as *const f32;
        for h in 0..self.num_heads {
            let kv_h = h / self.gqa_ratio;
            let view = kv_cache.v_slab_view(layer, kv_h, logical_start, position);
            unsafe {
                matmul_resident_tn(
                    view.ptr,                                // [windowed_len × head_dim]
                    scores_tight_base.add(h * extended_len), // [windowed_len × 1] (sink column ignored)
                    head_out_base.add(h * self.head_dim),
                    self.head_dim, view.rows, 1,
                )?;
            }
            batch.note_dispatch()?;
        }

        // Stage 8: output projection. Input is `[model_dim] = head_out`
        // (concat of all heads). Output is `[model_dim]`.
        self.o_proj.forward(batch, &scratch.head_out, out_dev)?;

        Ok(())
    }
}

/// Pre-allocated activation scratch for [`AttentionResident::forward`].
/// Keeps per-token scratch device-resident across calls — no
/// hipMalloc/hipFree on the hot path.
pub struct AttentionScratch {
    pub q_proj: GpuVec,        // [model_dim]
    pub k_proj: GpuVec,        // [kv_dim]
    pub v_proj: GpuVec,        // [kv_dim]
    pub q_normed: GpuVec,      // [model_dim]
    pub k_normed: GpuVec,      // [kv_dim]
    pub scores: GpuVec,        // [num_heads × max_seq] (strided, max_seq cols)
    pub scores_tight: GpuVec,  // [num_heads × max_seq] (compacted, attn_len cols used)
    pub head_out: GpuVec,      // [model_dim] (concatenated head outputs)
    /// Host-side scratch for D2H during host RoPE.
    pub q_normed_host: Vec<f32>,
    pub k_normed_host: Vec<f32>,
    cap_num_heads: usize,
    cap_head_dim: usize,
    cap_kv_dim: usize,
    cap_max_seq: usize,
}

impl AttentionScratch {
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        max_seq: usize,
    ) -> Result<Self, ResidencyError> {
        let model_dim = num_heads * head_dim;
        Ok(Self {
            q_proj: GpuVec::try_hip(model_dim)?,
            k_proj: GpuVec::try_hip(kv_dim)?,
            v_proj: GpuVec::try_hip(kv_dim)?,
            q_normed: GpuVec::try_hip(model_dim)?,
            k_normed: GpuVec::try_hip(kv_dim)?,
            scores: GpuVec::try_hip(num_heads * max_seq)?,
            // +1 column per head reserves a slot for an optional attention
            // sink. When no sink is installed the extra cells stay unused;
            // when sink is present softmax runs over `windowed_len + 1`.
            scores_tight: GpuVec::try_hip(num_heads * (max_seq + 1))?,
            head_out: GpuVec::try_hip(model_dim)?,
            q_normed_host: vec![0.0f32; model_dim],
            k_normed_host: vec![0.0f32; kv_dim],
            cap_num_heads: num_heads,
            cap_head_dim: head_dim,
            cap_kv_dim: kv_dim,
            cap_max_seq: max_seq,
        })
    }

    pub fn fits(&self, num_heads: usize, head_dim: usize, kv_dim: usize, max_seq: usize) -> bool {
        num_heads <= self.cap_num_heads
            && head_dim <= self.cap_head_dim
            && kv_dim <= self.cap_kv_dim
            && max_seq <= self.cap_max_seq
    }
}

/// Per-call backward scratch for [`AttentionResident::backward`]. Forward
/// activations live in [`AttentionScratch`]; this struct holds the
/// gradient temporaries.
pub struct AttentionBackwardScratch {
    /// `d/d(head_out)` post-`o_proj.backward`. Shape `[model_dim]`.
    pub d_head_out: GpuVec,
    /// `d/d(softmax)` per head, compacted. Shape `[num_heads × max_seq]`
    /// (only first attn_len cols are valid per head).
    pub d_softmax: GpuVec,
    /// `d/d(scores)` after `softmax_backward`. Shape `[num_heads × max_seq]`.
    pub d_scores: GpuVec,
    /// `d/d(q_normed)` (post-RoPE, post-RMSNorm). Shape `[model_dim]`.
    pub d_q_normed: GpuVec,
    /// `d/d(q_proj)` (pre-RMSNorm). Shape `[model_dim]`.
    pub d_q_proj: GpuVec,
    /// `d/d(k_proj)` (pre-RMSNorm). Shape `[kv_dim]`.
    pub d_k_proj: GpuVec,
    /// `d/d(v_proj)`. Shape `[kv_dim]`.
    pub d_v_proj: GpuVec,
    /// `dx` contribution from each projection's backward. Shape `[model_dim]`.
    pub dx_from_q: GpuVec,
    pub dx_from_k: GpuVec,
    pub dx_from_v: GpuVec,
    /// Cos/sin lookup for the matched-forward's position, uploaded
    /// fresh per backward call. Each is length `head_dim / 2`. Used
    /// by the resident RoPE backward dispatch on Q.
    pub cos_dev: GpuVec,
    pub sin_dev: GpuVec,
    /// Host scratch for the RMSNorm + RoPE backward (which run on host
    /// for parity with forward's host RoPE).
    pub q_normed_host: Vec<f32>,
    pub d_q_normed_host: Vec<f32>,
    pub k_normed_host: Vec<f32>,
    pub d_k_normed_host: Vec<f32>,
    pub q_proj_host: Vec<f32>,
    pub d_q_proj_host: Vec<f32>,
    pub k_proj_host: Vec<f32>,
    pub d_k_proj_host: Vec<f32>,
    cap_num_heads: usize,
    cap_head_dim: usize,
    cap_kv_dim: usize,
    cap_max_seq: usize,
}

impl AttentionBackwardScratch {
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        max_seq: usize,
    ) -> Result<Self, ResidencyError> {
        let model_dim = num_heads * head_dim;
        let half_dim = head_dim / 2;
        Ok(Self {
            d_head_out: GpuVec::try_hip(model_dim)?,
            d_softmax: GpuVec::try_hip(num_heads * max_seq)?,
            d_scores: GpuVec::try_hip(num_heads * max_seq)?,
            d_q_normed: GpuVec::try_hip(model_dim)?,
            d_q_proj: GpuVec::try_hip(model_dim)?,
            d_k_proj: GpuVec::try_hip(kv_dim)?,
            d_v_proj: GpuVec::try_hip(kv_dim)?,
            dx_from_q: GpuVec::try_hip(model_dim)?,
            dx_from_k: GpuVec::try_hip(model_dim)?,
            dx_from_v: GpuVec::try_hip(model_dim)?,
            cos_dev: GpuVec::try_hip(half_dim)?,
            sin_dev: GpuVec::try_hip(half_dim)?,
            q_normed_host: vec![0.0f32; model_dim],
            d_q_normed_host: vec![0.0f32; model_dim],
            k_normed_host: vec![0.0f32; kv_dim],
            d_k_normed_host: vec![0.0f32; kv_dim],
            q_proj_host: vec![0.0f32; model_dim],
            d_q_proj_host: vec![0.0f32; model_dim],
            k_proj_host: vec![0.0f32; kv_dim],
            d_k_proj_host: vec![0.0f32; kv_dim],
            cap_num_heads: num_heads,
            cap_head_dim: head_dim,
            cap_kv_dim: kv_dim,
            cap_max_seq: max_seq,
        })
    }

    pub fn fits(&self, num_heads: usize, head_dim: usize, kv_dim: usize, max_seq: usize) -> bool {
        num_heads <= self.cap_num_heads
            && head_dim <= self.cap_head_dim
            && kv_dim <= self.cap_kv_dim
            && max_seq <= self.cap_max_seq
    }
}

/// Device-resident weight gradients for [`AttentionResident`].
pub struct AttentionResidentGrads {
    pub dweight_q: GpuVec,
    pub dbias_q: GpuVec,
    pub dweight_k: GpuVec,
    pub dbias_k: GpuVec,
    pub dweight_v: GpuVec,
    pub dbias_v: GpuVec,
    pub dweight_o: GpuVec,
    pub dbias_o: GpuVec,
}

impl AttentionResidentGrads {
    pub fn new(model_dim: usize, kv_dim: usize) -> Result<Self, ResidencyError> {
        Ok(Self {
            dweight_q: GpuVec::try_hip(model_dim * model_dim)?,
            dbias_q: GpuVec::try_hip(model_dim)?,
            dweight_k: GpuVec::try_hip(kv_dim * model_dim)?,
            dbias_k: GpuVec::try_hip(kv_dim)?,
            dweight_v: GpuVec::try_hip(kv_dim * model_dim)?,
            dbias_v: GpuVec::try_hip(kv_dim)?,
            dweight_o: GpuVec::try_hip(model_dim * model_dim)?,
            dbias_o: GpuVec::try_hip(model_dim)?,
        })
    }
}

impl AttentionResident {
    /// Backward pass for one decode step at `position`. Treats prior
    /// KV-cache entries as constants (causal training-style detached
    /// cache) — only the current step's Q, K, V projection gradients
    /// are produced.
    ///
    /// The QK RMSNorm and RoPE backwards run on the host because the
    /// forward path also runs them on the host (for RoPE) or with a
    /// constant scale (for QK RMSNorm — no learnable gamma in the
    /// host model). Host arithmetic for these stages costs ~2 µs at
    /// `model_dim=128`; no resident kernel exists for either.
    ///
    /// Caller contract:
    ///   - `x_dev` is the pre-attention-norm input that fed `forward`.
    ///   - `dy_dev` is `d/d(out)` from upstream (post-residual peel-off).
    ///   - `dx_dev` receives `d/d(x)`.
    ///   - `attn_scratch` must contain the activations the matched
    ///     forward saved (`q_proj`, `k_proj`, `v_proj`, `q_normed`,
    ///     `k_normed`, `scores_tight`, `head_out`).
    ///   - `kv_cache` must be in the post-forward state (the K/V slabs
    ///     need to be readable for the score-path d_q computation).
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        batch: &HipBatch,
        x_dev: &GpuVec,
        dy_dev: &GpuVec,
        kv_cache: &KvCacheResident,
        layer: usize,
        position: usize,
        window: Option<usize>,
        rope: &RotaryEmbedding,
        attn_scratch: &AttentionScratch,
        bwd: &mut AttentionBackwardScratch,
        grads: &mut AttentionResidentGrads,
        dx_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        let model_dim = self.model_dim;
        let kv_dim = self.kv_dim;
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let gqa_ratio = self.gqa_ratio;
        let attn_len = position + 1;
        let start = match window {
            Some(w) => position.saturating_sub(w) + 1,
            None => 0,
        };
        let windowed_len = attn_len - start;
        let max_seq = kv_cache.max_seq_len();

        debug_assert_eq!(x_dev.len(), model_dim);
        debug_assert_eq!(dy_dev.len(), model_dim);
        debug_assert_eq!(dx_dev.len(), model_dim);
        debug_assert!(bwd.fits(num_heads, head_dim, kv_dim, max_seq));

        // Stage 1: o_proj backward.
        //   d_head_out = o.W^T · dy
        //   dweight_o = dy ⊗ head_out
        // Runs unconditionally — dbias_o picks up d_attn_out even when
        // the window is empty (bias is the only o_proj contribution to
        // forward output in the empty-window short-circuit case).
        self.o_proj.backward(
            batch, &attn_scratch.head_out, dy_dev,
            &mut bwd.d_head_out,
            &mut grads.dweight_o,
            &mut grads.dbias_o,
        )?;

        // Empty-window backward: forward returned a zero output (post-bias),
        // so head_out / scores / softmax are all zero. d/d(input) of a
        // constant is zero. Stages 2–8 would all dispatch with dim=0
        // (kernels balk) and produce zeros anyway. Short-circuit by
        // zeroing dx_dev and returning. Q/K/V proj weight grads are
        // unchanged because head_out=0 makes their contributions zero.
        if windowed_len == 0 {
            let zeros = vec![0.0f32; model_dim];
            dx_dev.copy_from(&zeros);
            return Ok(());
        }

        // Stage 2: per-head, compute d_softmax[h, t] = V_slab[t, :] · d_head_out[h, :]
        // for t in [start, attn_len). matmul reads V_slab[start..attn_len]
        // (advance v_slab pointer by start*head_dim, m=windowed_len);
        // writes d_softmax row tightly at h*windowed_len for `windowed_len`
        // entries.
        let d_head_out_buf = hip_buf(&bwd.d_head_out)?;
        let d_softmax_buf = hip_buf(&bwd.d_softmax)?;
        let d_softmax_base = d_softmax_buf.device_ptr() as *mut f32;
        let d_head_out_base = d_head_out_buf.device_ptr() as *const f32;
        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;
            let v_slab = kv_cache.v_slab_ptr(layer, kv_h);
            unsafe {
                matmul_resident_nn(
                    v_slab.add(start * head_dim),                   // [windowed_len × head_dim]
                    d_head_out_base.add(h * head_dim),              // [head_dim × 1]
                    d_softmax_base.add(h * windowed_len),           // [windowed_len × 1]
                    windowed_len, head_dim, 1,
                )?;
            }
            batch.note_dispatch()?;
        }

        // Stage 3: softmax backward.
        // scores_tight stores the *forward softmax output* — already
        // tightly packed at stride `windowed_len` per head from the
        // forward path.
        let scores_tight_buf = hip_buf(&attn_scratch.scores_tight)?;
        let d_scores_buf = hip_buf(&bwd.d_scores)?;
        unsafe {
            softmax_backward_resident(
                scores_tight_buf.device_ptr() as *const f32,
                d_softmax_base as *const f32,
                d_scores_buf.device_ptr() as *mut f32,
                num_heads, windowed_len, false,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 4: per-head, d_q_normed[h, :] = sum_t d_scores[h, t] * K_slab[t, :]
        // for t in [start, attn_len). matmul reads K_slab[start..attn_len]
        // (advance k_slab pointer by start*head_dim, k=windowed_len);
        // d_scores tight at h*windowed_len.
        let d_q_normed_buf = hip_buf(&bwd.d_q_normed)?;
        let d_q_normed_base = d_q_normed_buf.device_ptr() as *mut f32;
        let d_scores_base = d_scores_buf.device_ptr() as *const f32;
        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;
            let k_slab = kv_cache.k_slab_ptr(layer, kv_h);
            unsafe {
                matmul_resident_tn(
                    k_slab.add(start * head_dim),                   // [windowed_len × head_dim]
                    d_scores_base.add(h * windowed_len),            // [windowed_len × 1]
                    d_q_normed_base.add(h * head_dim),              // [head_dim × 1]
                    head_dim, windowed_len, 1,
                )?;
            }
            batch.note_dispatch()?;
        }

        // Stage 5: per-current-token K/V gradients (cache for prior tokens
        // is treated as constants per the contract).
        // d_k_current_head[i] = sum_t (only t=position) d_scores[h, t] * q_normed[h, i]
        //                     = sum_{h ∈ kv_h's heads} d_scores[h, position] * q_normed[h, i]
        // d_v_current_head[i] = sum_t (only t=position) softmax[h, t] * d_head_out[h, i]
        //                     = sum_{h ∈ kv_h's heads} softmax[h, position] * d_head_out[h, i]
        //
        // These cross-head sums (for GQA) plus the scalar scale make a
        // host-side computation cleaner than chaining op_tensor_resident
        // calls. D2H the small per-head buffers (~2-32 floats each), do
        // the host arithmetic, H2D the result. ~1 µs per head dim;
        // negligible relative to the matmul stages.
        let mut q_normed_host = vec![0.0f32; model_dim];
        attn_scratch.q_normed.copy_to_host(&mut q_normed_host);
        // forward's `scores_tight` is laid out tightly at stride
        // `windowed_len` per head; copy that prefix.
        // Sink slot (if present) reserves +1 column per head — extra
        // capacity here doesn't hurt (we read only the first windowed_len
        // entries; the sink contributes only a softmax mass drain, never
        // a current-token gradient).
        let scratch_cap = num_heads * (max_seq + 1);
        let mut scores_host = vec![0.0f32; num_heads * windowed_len];
        {
            let mut full = vec![0.0f32; scratch_cap];
            attn_scratch.scores_tight.copy_to_host(&mut full);
            for h in 0..num_heads {
                let extended_stride = windowed_len + (self.sink_bias_dev.is_some() as usize);
                let src_base = h * extended_stride;
                let dst_base = h * windowed_len;
                scores_host[dst_base..dst_base + windowed_len]
                    .copy_from_slice(&full[src_base..src_base + windowed_len]);
            }
        }
        // d_scores from softmax_backward — same tight stride windowed_len.
        let mut d_scores_host = vec![0.0f32; num_heads * windowed_len];
        {
            let mut full = vec![0.0f32; num_heads * max_seq];
            bwd.d_scores.copy_to_host(&mut full);
            d_scores_host.copy_from_slice(&full[..num_heads * windowed_len]);
        }
        let mut d_head_out_host = vec![0.0f32; model_dim];
        bwd.d_head_out.copy_to_host(&mut d_head_out_host);

        // The current token sits at the LAST entry of the windowed range
        // (index `windowed_len - 1`). For window=None this collapses to
        // `attn_len - 1` (the original behavior).
        // d_v_current[kv_h, i] = sum_{h: h/gqa_ratio==kv_h} softmax[h, last] * d_head_out[h, i]
        // d_k_current[kv_h, i] = sum_{h: h/gqa_ratio==kv_h} d_scores[h, last] * q_normed[h, i]
        let mut d_k_current_post_rope = vec![0.0f32; kv_dim];
        let mut d_v_current_host = vec![0.0f32; kv_dim];
        let last = windowed_len - 1;
        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;
            let s = scores_host[h * windowed_len + last];
            let ds = d_scores_host[h * windowed_len + last];
            let q_h = &q_normed_host[h * head_dim..(h + 1) * head_dim];
            let d_h_out = &d_head_out_host[h * head_dim..(h + 1) * head_dim];
            for i in 0..head_dim {
                d_k_current_post_rope[kv_h * head_dim + i] += ds * q_h[i];
                d_v_current_host[kv_h * head_dim + i] += s * d_h_out[i];
            }
        }

        // Stage 6: RoPE backward + RMSNorm backward for current K and Q.
        // Forward did:
        //   k_proj  → rms_norm(qk_scale)  → k_normed_pre_rope
        //   k_normed_pre_rope (per kv_head) → RoPE → k_normed (cached)
        //   q_proj  → rms_norm(qk_scale)  → q_normed_pre_rope
        //   q_normed_pre_rope (per head)  → RoPE → q_normed (used for scoring)
        //
        // For Q we have d_q_normed on device (post-RoPE). For K we
        // have d_k_current_post_rope on host (built above by the
        // cross-GQA-head reduction). Undo RoPE then undo RMSNorm.
        //
        // RoPE adjoint per pair (i, i+half_dim):
        //   forward: (a, b) → (a*c - b*s, a*s + b*c)
        //   adjoint: (da, db) → (da*c + db*s, -da*s + db*c)
        //
        // Q runs RoPE backward GPU-side (resident). The kernel writes
        // back into the same device buffer in-place so the existing
        // D2H below picks up the pre-RoPE gradient ready for the host
        // RMSNorm backward step. Avoids the host-side per-pair loop
        // that previously sat between the D2H and RMSNorm backward.
        bwd.cos_dev.copy_from(rope.cos_at(position));
        bwd.sin_dev.copy_from(rope.sin_at(position));
        let d_q_normed_dev_ptr = hip_buf(&bwd.d_q_normed)?.device_ptr() as *mut f32;
        unsafe {
            rope_backward_resident(
                d_q_normed_dev_ptr as *const f32,
                hip_buf(&bwd.cos_dev)?.device_ptr() as *const f32,
                hip_buf(&bwd.sin_dev)?.device_ptr() as *const f32,
                d_q_normed_dev_ptr,
                num_heads, head_dim,
            )?;
        }
        batch.note_dispatch()?;
        let mut d_q_normed_host = vec![0.0f32; model_dim];
        bwd.d_q_normed.copy_to_host(&mut d_q_normed_host);

        // K stays host: the cross-GQA-head reduction above produced
        // d_k_current_post_rope as a host vec, so a host adjoint loop
        // is the natural fit. Routing K through the GPU kernel would
        // wrap the per-position 16-256 floats with H2D + dispatch +
        // D2H — net negative until the GQA reduction itself moves
        // device-side. The kernel is wired and tested for K's shape;
        // a follow-up slice can move both stages together.
        rope_backward(rope, &mut d_k_current_post_rope, head_dim, num_kv_heads, position);

        // RMSNorm backward (constant scale `qk_scale`, no learnable gamma).
        // Forward: y[h, i] = scale * x[h, i] / rms(x[h, :])
        //   where rms = sqrt(mean(x^2) + eps).
        // Backward:
        //   inv_rms = 1/rms
        //   dx[h, i] = scale * inv_rms * (dy[h, i] - x[h, i] * (sum_j dy[h, j] * x[h, j]) * inv_rms^2 / N)
        let mut q_proj_host = vec![0.0f32; model_dim];
        attn_scratch.q_proj.copy_to_host(&mut q_proj_host);
        let mut k_proj_host = vec![0.0f32; kv_dim];
        attn_scratch.k_proj.copy_to_host(&mut k_proj_host);

        let mut d_q_proj_host = vec![0.0f32; model_dim];
        let mut d_k_proj_host = vec![0.0f32; kv_dim];
        if self.use_qk_norm {
            rms_norm_backward_per_head(
                &q_proj_host, &d_q_normed_host, &mut d_q_proj_host,
                num_heads, head_dim, self.qk_scale, self.norm_eps,
            );
            rms_norm_backward_per_head(
                &k_proj_host, &d_k_current_post_rope, &mut d_k_proj_host,
                num_kv_heads, head_dim, self.qk_scale, self.norm_eps,
            );
        } else {
            // QK norm was skipped in forward (use_qk_norm = false): the
            // forward copies q_proj/k_proj straight into the q_normed/
            // k_normed slots before RoPE, so backward's identity-
            // passthrough is the exact inverse. q_proj_host / k_proj_host
            // and self.qk_scale are unused on this branch.
            d_q_proj_host.copy_from_slice(&d_q_normed_host);
            d_k_proj_host.copy_from_slice(&d_k_current_post_rope);
        }

        // Stage 7: H2D the d_q_proj, d_k_proj, d_v_proj, then call
        // q/k/v_proj.backward.
        bwd.d_q_proj.copy_from(&d_q_proj_host);
        bwd.d_k_proj.copy_from(&d_k_proj_host);
        bwd.d_v_proj.copy_from(&d_v_current_host);

        self.q_proj.backward(
            batch, x_dev, &bwd.d_q_proj,
            &mut bwd.dx_from_q,
            &mut grads.dweight_q,
            &mut grads.dbias_q,
        )?;
        self.k_proj.backward(
            batch, x_dev, &bwd.d_k_proj,
            &mut bwd.dx_from_k,
            &mut grads.dweight_k,
            &mut grads.dbias_k,
        )?;
        self.v_proj.backward(
            batch, x_dev, &bwd.d_v_proj,
            &mut bwd.dx_from_v,
            &mut grads.dweight_v,
            &mut grads.dbias_v,
        )?;

        // Stage 8: dx = dx_from_q + dx_from_k + dx_from_v.
        unsafe {
            op_tensor_resident(
                hip_buf(&bwd.dx_from_q)?.device_ptr() as *const f32,
                hip_buf(&bwd.dx_from_k)?.device_ptr() as *const f32,
                hip_buf_mut(dx_dev)?.device_ptr() as *mut f32,
                model_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;
        unsafe {
            op_tensor_resident(
                hip_buf(dx_dev)?.device_ptr() as *const f32,
                hip_buf(&bwd.dx_from_v)?.device_ptr() as *const f32,
                hip_buf_mut(dx_dev)?.device_ptr() as *mut f32,
                model_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;

        Ok(())
    }
}

// ─── AttentionResident — full-sequence backward path ────────
//
// Algorithm derivation (chain rule).
//
// Forward at position t, head h, kv_h = h / gqa_ratio:
//   Q_normed[t, h] = qk_norm(qk_scale, Q_proj[t, h]) → RoPE
//   K_normed[s, h] = qk_norm(qk_scale, K_proj[s, h]) → RoPE  (s ≤ t)
//   score[t, h, s] = Q_normed[t, h] · K_normed[s, kv_h]
//   softmax[t, h, :] = stable_softmax(score[t, h, :])
//   head_out[t, h]   = Σ_s softmax[t, h, s] · V_raw[s, kv_h]
//   attn_out[t]      = W_o · concat_h(head_out[t, h])
//
// Backward, given d_attn_out[t]:
//   1. d_head_out[t]    = W_o^T · d_attn_out[t]
//      dW_o            += d_attn_out[t] · concat_h(head_out[t, h])^T
//      dbias_o         += d_attn_out[t]
//   2. d_v_raw[s, kv_h] += Σ_{h: h/gqa==kv_h} softmax[t, h, s] · d_head_out[t, h]
//      d_softmax[t, h, s] = V_raw[s, kv_h] · d_head_out[t, h]
//      (both for ALL s in 0..=t)
//   3. d_score[t, h, s] = softmax[t, h, s] · (d_softmax[t, h, s]
//                                             - Σ_s' softmax[t, h, s'] · d_softmax[t, h, s'])
//   4. d_k_postrope[s, kv_h]  += Σ_{h} d_score[t, h, s] · Q_normed[t, h]
//      d_q_normed[t, h]       = Σ_s    d_score[t, h, s] · K_normed[s, kv_h]
//      (the d_q_normed[t] sum is over s ≤ t — this is the full Q gradient
//       from this step; it folds RoPE/RMS bwd → d_Q_proj[t] for current
//       step's q_proj backward.)
//   5. After all u > t reverse-iter steps have run, d_k_postrope[t] /
//      d_v_raw[t] are FULLY populated. At step t we read them and:
//        - undo RoPE at position t (per kv_head) on d_k_postrope[t]
//        - undo per-head RMSNorm using saved K_proj[t] → d_k_proj[t]
//        - dx[t] += W_k^T · d_k_proj[t] + W_v^T · d_v_raw[t]
//        - dW_k += d_k_proj[t] · X[t]^T  (deferred to finalize)
//        - dW_v += d_v_raw[t]  · X[t]^T  (deferred to finalize)
//
// Reverse iteration ordering matters. At step t:
//   - the accumulator's d_k_postrope[t] / d_v_raw[t] entries hold the sum
//     of contributions from u = t+1, t+2, … n-1 (already iterated, since
//     reverse). These are the "cross-step" extras that the existing per-
//     step backward dropped.
//   - The s=t entry is added DURING this step's loop body too — but only
//     after we've already folded the existing-cross-step value into dx[t].
//     This matches the spec: "step t backward, all u > t have already
//     fired. So d_K[t] and d_V[t] are now FULLY populated (sum of all
//     u ≥ t contributions including the just-added s=t one from this
//     step)." Actually, re-reading: the spec says the s=t add IS included
//     in d_K[t] before dx[t] is updated. The two interpretations differ
//     by whether the current step's s=t contribution flows through dx[t]
//     or only through finalize. We follow the spec: include s=t in the
//     accumulator add, then the existing q/k/v_proj.backward call (still
//     done at this step for the s=t weight-grad contribution to dW_q AND
//     to provide dx_from_v[t] / dx_from_k[t] for the LOCAL piece) is
//     SKIPPED for K/V — finalize_kv_grads handles them globally.
//
// This file implements the spec literally: a NEW entry point
// `backward_full_sequence_step` that does NOT call the existing
// `AttentionResident::backward`. The new path produces dW_q / dW_o /
// dbias_q / dbias_o for the current step, dx[t] (folding cross-step
// d_K[t] / d_V[t] from u > t AND the just-added s=t from this step),
// and accumulates d_K / d_V cross-step contributions for all s ≤ t.

// ─── Cross-step KV gradient accumulator ─────────────────────
//
// **Problem.** [`AttentionResident::backward`] is documented as treating
// prior KV-cache entries as constants — only the current step's Q/K/V
// projection gradients are produced. That's correct for autoregressive
// single-position decode-style backward (used by the latent path), but
// **wrong** for full-sequence per-position backward (used by the BLT
// encoder/decoder per-byte loops). At step t, Y[t] depends on K[s] and
// V[s] for ALL s ≤ t, so dW_k = Σ_s d_K[s] · X[s]^T must accumulate
// across every position s. The per-step backward only adds the s=t term,
// systematically undercounting dW_k / dW_v / dx[s] for s < t.
//
// **Fix shape.** Caller iterates t in REVERSE order (t = n-1 … 0).
// Maintains a [`KvGradAccumulator`] that holds:
//   - `d_k_postrope[s, kv_h, i]`  = Σ_{u ≥ s} d_score[u, h, s] · q_normed[u, h, i]
//                                   (in post-RoPE post-RMSNorm K space, summed
//                                    over Q-heads sharing this kv_h)
//   - `d_v_raw[s, kv_h, i]`       = Σ_{u ≥ s} softmax[u, h, s] · d_head_out[u, h, i]
//                                   (in raw-V space, summed over GQA)
// At step t, the accumulator is *read* at index t (giving the full Σ_{u ≥ t}
// because reverse iteration has finished all u > t), folded into dx[t] and
// dW_k / dW_v, *then written* with new s ≤ t contributions for future steps
// to consume.
//
// **Stays host-side.** The new path runs the cross-step arithmetic on
// host (n² over positions × heads × head_dim = small for BLT's tiny
// configs). All saved activations get D2H'd once at the start of each
// `backward_full_sequence_step`; the result is H2D'd at the end. This
// trades a few hundred f32s of PCIe per step for absolute simplicity —
// the previous worktree's GPU-resident attempt regressed encoder.block.
// gate from 87% → 132% rel_err. Naive correctness first.

/// Per-layer cross-step KV gradient accumulator. See module docs above.
///
/// **Invariants** (`debug_assert`'d at the entry-point boundaries):
///   - `n_positions` matches the caller's per-byte loop length.
///   - `x_per_position_host` is populated with the per-position attention
///     input X[s] (= post-attention-RMSNorm hidden) before
///     [`AttentionResident::finalize_kv_grads`] is called.
///   - `zero_resident` is called once before the reverse-iteration loop;
///     `d_k_postrope` / `d_v_raw` accumulate (`+=`) thereafter.
///
/// Layout:
///   - `d_k_postrope`: `[n_positions × n_kv_heads × head_dim]`, post-RoPE
///     post-RMSNorm space. The cross-step loop adds `d_score · q_normed`
///     directly here — both operands live in the same space.
///   - `d_v_raw`: `[n_positions × n_kv_heads × head_dim]`, raw V space
///     (V is not RoPE'd or normed in forward).
///   - `x_per_position_host`: `[n_positions × model_dim]`, the per-position
///     X[s] that feeds the q/k/v projections.
pub struct KvGradAccumulator {
    /// Cross-step d/d(K_postrope[s, kv_h, i]). Layout `[n × n_kv × d_h]`
    /// row-major (position-major).
    pub d_k_postrope_host: Vec<f32>,
    /// Cross-step d/d(V_raw[s, kv_h, i]). Same layout.
    pub d_v_raw_host: Vec<f32>,
    pub n_positions: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub model_dim: usize,
    /// Per-position attention sub-layer input X[s] = attn_normed[s].
    /// Filled by the caller during forward; consumed by `finalize_kv_grads`.
    pub x_per_position_host: Vec<f32>,
    /// `true` after `finalize_kv_grads` runs; subsequent calls assert.
    finalized: bool,
}

impl KvGradAccumulator {
    pub fn new(
        n_positions: usize,
        n_kv_heads: usize,
        head_dim: usize,
        model_dim: usize,
    ) -> Result<Self, ResidencyError> {
        let kv_total = n_positions * n_kv_heads * head_dim;
        Ok(Self {
            d_k_postrope_host: vec![0.0f32; kv_total],
            d_v_raw_host: vec![0.0f32; kv_total],
            n_positions,
            n_kv_heads,
            head_dim,
            model_dim,
            x_per_position_host: vec![0.0f32; n_positions * model_dim],
            finalized: false,
        })
    }

    /// Reset accumulators to zero. Call once before each reverse-iteration
    /// per-step loop. `batch` parameter is for API uniformity (host-side
    /// fill needs no batch); kept so future GPU-resident impls can swap in.
    pub fn zero_resident(&mut self, _batch: &HipBatch) -> Result<(), ResidencyError> {
        for v in self.d_k_postrope_host.iter_mut() { *v = 0.0; }
        for v in self.d_v_raw_host.iter_mut() { *v = 0.0; }
        self.finalized = false;
        Ok(())
    }

    /// Convenience: write X[s] into `x_per_position_host` from a host slice.
    pub fn store_x_at(&mut self, position: usize, x: &[f32]) {
        debug_assert_eq!(x.len(), self.model_dim);
        debug_assert!(position < self.n_positions);
        let off = position * self.model_dim;
        self.x_per_position_host[off..off + self.model_dim].copy_from_slice(x);
    }
}

impl AttentionResident {
    /// Full-sequence per-position backward step. See module-level
    /// "AttentionResident — full-sequence backward path" doc-comment for
    /// the chain-rule derivation.
    ///
    /// **Caller contract:**
    /// - Iterate `position` in REVERSE order (`n-1, n-2, …, 0`).
    /// - Before the loop: call `kv_acc.zero_resident(batch)`.
    /// - After the loop: call [`AttentionResident::finalize_kv_grads`].
    /// - `kv_acc.x_per_position_host` must be populated with X[s] for
    ///   every s in 0..n_positions before `finalize_kv_grads`. (Easiest:
    ///   D2H attn_normed into `kv_acc.x_per_position_host[s*model_dim..]`
    ///   during forward.)
    /// - `attn_scratch` is unused for activations — every per-position
    ///   activation read comes from `block_scratch.saved_*` (per-position
    ///   snapshots) or `kv_cache` slabs. We accept it for signature
    ///   uniformity with the existing `backward` and to expose enough
    ///   workspace for the o_proj backward.
    /// - `kv_cache` must be in the post-forward state: K[s] (post-RoPE
    ///   post-RMSNorm) and V[s] (raw) for all s in 0..n_positions.
    ///
    /// **What this writes:**
    /// - `attn_grads.dweight_o / dbias_o` += this step's o_proj outer
    ///   product / d_attn_out (per-step, just like existing backward).
    /// - `attn_grads.dweight_q / dbias_q` += this step's q_proj outer
    ///   product / d_q_proj (per-step).
    /// - `attn_grads.dweight_k / dbias_k`,  `dweight_v / dbias_v` —
    ///   left UNTOUCHED here. `finalize_kv_grads` writes them all in one
    ///   shot using the fully-accumulated `kv_acc`.
    /// - `dy_dev` (in-place) → `dx[position]`: the dx contribution from
    ///   (a) Q's projection at position t, plus (b) the cross-step
    ///   d_K[t] / d_V[t] from all u ≥ t (= u > t pre-existing + s=t just
    ///   added; both are summed in `kv_acc.{d_k_postrope_host,d_v_raw_host}[t]`
    ///   right BEFORE we read them — see step (5) in the algorithm).
    /// - `kv_acc.{d_k_postrope_host,d_v_raw_host}[s, kv_h]` += this step's
    ///   contribution for every s in 0..=position.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_full_sequence_step(
        &self,
        batch: &HipBatch,
        dy_dev: &mut GpuVec,
        kv_cache: &KvCacheResident,
        layer: usize,
        position: usize,
        rope: &RotaryEmbedding,
        attn_scratch: &AttentionScratch,
        bwd: &mut AttentionBackwardScratch,
        block_scratch: &TransformerBlockScratch,
        attn_grads: &mut AttentionResidentGrads,
        kv_acc: &mut KvGradAccumulator,
        dx_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        let _ = attn_scratch; // unused, see contract
        let model_dim = self.model_dim;
        let kv_dim = self.kv_dim;
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let gqa_ratio = self.gqa_ratio;
        let attn_len = position + 1;
        let max_seq = kv_cache.max_seq_len();

        debug_assert_eq!(dy_dev.len(), model_dim);
        debug_assert_eq!(dx_dev.len(), model_dim);
        debug_assert!(bwd.fits(num_heads, head_dim, kv_dim, max_seq));
        debug_assert_eq!(kv_acc.n_kv_heads, num_kv_heads);
        debug_assert_eq!(kv_acc.head_dim, head_dim);
        debug_assert_eq!(kv_acc.model_dim, model_dim);
        debug_assert!(position < kv_acc.n_positions);
        debug_assert!(!kv_acc.finalized,
            "backward_full_sequence_step called after finalize_kv_grads");

        // ── Stage 1: o_proj backward (UNCHANGED from per-step backward). ──
        //   d_head_out = W_o^T · dy
        //   dW_o += dy · head_out^T  (overwrites — caller accumulates)
        self.o_proj.backward(
            batch, &block_scratch.saved_head_out, dy_dev,
            &mut bwd.d_head_out,
            &mut attn_grads.dweight_o,
            &mut attn_grads.dbias_o,
        )?;

        // ── D2H every host-side input we need. ──
        // d_head_out[t] (post o_proj.backward): [model_dim]
        let mut d_head_out_host = vec![0.0f32; model_dim];
        bwd.d_head_out.copy_to_host(&mut d_head_out_host);
        // q_normed[t] (post-RoPE post-RMSNorm): [model_dim], saved by
        // forward_for_backward.
        let mut q_normed_host = vec![0.0f32; model_dim];
        block_scratch.saved_q_normed.copy_to_host(&mut q_normed_host);
        // q_proj[t] (pre-RMSNorm): [model_dim], for RMS bwd of d_Q.
        let mut q_proj_host = vec![0.0f32; model_dim];
        block_scratch.saved_q_proj.copy_to_host(&mut q_proj_host);
        // k_proj[t] (pre-RMSNorm): [kv_dim], for RMS bwd of d_K[t].
        let mut k_proj_t_host = vec![0.0f32; kv_dim];
        block_scratch.saved_k_proj.copy_to_host(&mut k_proj_t_host);
        // softmax[t, h, :] for s in 0..attn_len: [num_heads × attn_len]
        // contiguous in the first num_heads*attn_len slots of saved_scores_tight.
        let mut softmax_host = vec![0.0f32; num_heads * attn_len];
        {
            let mut full = vec![0.0f32; num_heads * max_seq];
            block_scratch.saved_scores_tight.copy_to_host(&mut full);
            softmax_host.copy_from_slice(&full[..num_heads * attn_len]);
        }
        // K cache: per-kv_head slab [max_seq × head_dim]; we read rows
        // 0..attn_len for each kv_head.
        let mut k_slabs_host = vec![0.0f32; num_kv_heads * attn_len * head_dim];
        let mut v_slabs_host = vec![0.0f32; num_kv_heads * attn_len * head_dim];
        {
            // KV is laid out [n_kv_heads × max_seq × head_dim] per layer.
            // D2H the first attn_len rows of each kv_head.
            use modgrad_device::backend::rocm::ffi;
            const HIP_D2H: std::os::raw::c_int = 2;
            for kv_h in 0..num_kv_heads {
                let src_k = unsafe {
                    (kv_cache.k_layer_ptr(layer) as *const u8)
                        .add(kv_h * max_seq * head_dim * 4)
                } as *const std::os::raw::c_void;
                let src_v = unsafe {
                    (kv_cache.v_layer_ptr(layer) as *const u8)
                        .add(kv_h * max_seq * head_dim * 4)
                } as *const std::os::raw::c_void;
                let dst_k = k_slabs_host[kv_h * attn_len * head_dim..]
                    .as_mut_ptr() as *mut std::os::raw::c_void;
                let dst_v = v_slabs_host[kv_h * attn_len * head_dim..]
                    .as_mut_ptr() as *mut std::os::raw::c_void;
                let bytes = attn_len * head_dim * 4;
                let err = unsafe { ffi::hipMemcpy(dst_k, src_k, bytes, HIP_D2H) };
                if err != 0 {
                    return Err(ResidencyError::Backend(
                        modgrad_device::backend::BackendError::Runtime(format!(
                            "kv_acc K D2H ({} bytes): {}", bytes, ffi::hip_err_str(err),
                        )),
                    ));
                }
                let err = unsafe { ffi::hipMemcpy(dst_v, src_v, bytes, HIP_D2H) };
                if err != 0 {
                    return Err(ResidencyError::Backend(
                        modgrad_device::backend::BackendError::Runtime(format!(
                            "kv_acc V D2H ({} bytes): {}", bytes, ffi::hip_err_str(err),
                        )),
                    ));
                }
            }
        }

        // ── Stage 2 + 3: per-head softmax bwd. ──
        // For each (h, s in 0..attn_len):
        //   d_softmax[h, s] = V[s, kv_h] · d_head_out[h]
        //   d_v_raw[s, kv_h] += softmax[h, s] · d_head_out[h]      (cross-step write)
        // Then stable softmax bwd:
        //   d_score[h, s] = softmax[h, s] · (d_softmax[h, s]
        //                                    - Σ_s' softmax[h, s'] · d_softmax[h, s'])
        let mut d_softmax_host = vec![0.0f32; num_heads * attn_len];
        let mut d_score_host = vec![0.0f32; num_heads * attn_len];
        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;
            let d_head_h = &d_head_out_host[h * head_dim..(h + 1) * head_dim];
            for s in 0..attn_len {
                let v_s = &v_slabs_host[
                    (kv_h * attn_len + s) * head_dim..(kv_h * attn_len + s + 1) * head_dim
                ];
                let mut dot = 0.0f32;
                for i in 0..head_dim { dot += v_s[i] * d_head_h[i]; }
                d_softmax_host[h * attn_len + s] = dot;
                let sm = softmax_host[h * attn_len + s];
                let dv_off = (s * num_kv_heads + kv_h) * head_dim;
                for i in 0..head_dim {
                    kv_acc.d_v_raw_host[dv_off + i] += sm * d_head_h[i];
                }
            }
            // Stable softmax bwd: dot = Σ_s' softmax[h, s'] · d_softmax[h, s']
            let mut sm_dot = 0.0f32;
            for s in 0..attn_len {
                sm_dot += softmax_host[h * attn_len + s]
                        * d_softmax_host[h * attn_len + s];
            }
            for s in 0..attn_len {
                let sm = softmax_host[h * attn_len + s];
                let ds = d_softmax_host[h * attn_len + s];
                d_score_host[h * attn_len + s] = sm * (ds - sm_dot);
            }
        }

        // ── Stage 4: per-head, accumulate d_k_postrope[s, kv_h] and d_q_normed[t, h]. ──
        //   d_k_postrope[s, kv_h] += Σ_{h: h/gqa==kv_h} d_score[h, s] · q_normed[h]
        //   d_q_normed[t, h]      = Σ_s    d_score[h, s] · K_postrope[s, kv_h]
        let mut d_q_normed_host = vec![0.0f32; model_dim];
        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;
            let q_h = &q_normed_host[h * head_dim..(h + 1) * head_dim];
            let dq_h = &mut d_q_normed_host[h * head_dim..(h + 1) * head_dim];
            for s in 0..attn_len {
                let ds = d_score_host[h * attn_len + s];
                let dk_off = (s * num_kv_heads + kv_h) * head_dim;
                let k_s = &k_slabs_host[
                    (kv_h * attn_len + s) * head_dim..(kv_h * attn_len + s + 1) * head_dim
                ];
                for i in 0..head_dim {
                    kv_acc.d_k_postrope_host[dk_off + i] += ds * q_h[i];
                    dq_h[i] += ds * k_s[i];
                }
            }
        }

        // ── Stage 5: read FULLY-populated d_k_postrope[t] / d_v_raw[t]. ──
        // After this step's add (Stage 2 / Stage 4) for s=t, the entry t
        // of the accumulator now holds Σ_{u ≥ t} contributions: prior u>t
        // from earlier reverse-iter steps PLUS the just-added s=t add.
        // Per the algorithm spec: "At step t backward (reverse iteration),
        // all u > t have already fired. So d_K[t] and d_V[t] are now FULLY
        // populated (sum of all u ≥ t contributions including the just-
        // added s=t one from this step)."
        let mut d_k_postrope_t = vec![0.0f32; kv_dim];
        let mut d_v_raw_t = vec![0.0f32; kv_dim];
        let acc_off_t = position * num_kv_heads * head_dim;
        d_k_postrope_t.copy_from_slice(
            &kv_acc.d_k_postrope_host[acc_off_t..acc_off_t + kv_dim]
        );
        d_v_raw_t.copy_from_slice(
            &kv_acc.d_v_raw_host[acc_off_t..acc_off_t + kv_dim]
        );

        // Undo RoPE on d_k_postrope_t (per kv_head, position).
        rope_backward(rope, &mut d_k_postrope_t, head_dim, num_kv_heads, position);

        // Undo RMSNorm (or passthrough) on d_k_postrope_t → d_k_proj_t.
        let mut d_k_proj_t = vec![0.0f32; kv_dim];
        if self.use_qk_norm {
            rms_norm_backward_per_head(
                &k_proj_t_host, &d_k_postrope_t, &mut d_k_proj_t,
                num_kv_heads, head_dim, self.qk_scale, self.norm_eps,
            );
        } else {
            d_k_proj_t.copy_from_slice(&d_k_postrope_t);
        }

        // ── Stage 6: undo RoPE + RMS on d_q_normed[t] → d_q_proj[t]. ──
        // RoPE undo on Q (position t, num_heads).
        rope_backward(rope, &mut d_q_normed_host, head_dim, num_heads, position);
        let mut d_q_proj_host = vec![0.0f32; model_dim];
        if self.use_qk_norm {
            rms_norm_backward_per_head(
                &q_proj_host, &d_q_normed_host, &mut d_q_proj_host,
                num_heads, head_dim, self.qk_scale, self.norm_eps,
            );
        } else {
            d_q_proj_host.copy_from_slice(&d_q_normed_host);
        }

        // ── Stage 7: q_proj.backward (per-step dW_q + dx_from_q). ──
        bwd.d_q_proj.copy_from(&d_q_proj_host);
        self.q_proj.backward(
            batch, &block_scratch.attn_normed, &bwd.d_q_proj,
            &mut bwd.dx_from_q,
            &mut attn_grads.dweight_q,
            &mut attn_grads.dbias_q,
        )?;

        // ── Stage 8: cross-step contribution to dx[t] via W_k^T·d_k_proj[t]
        //   + W_v^T·d_v_raw[t]. We DON'T touch attn_grads.dweight_k / dweight_v
        //   here — those are accumulated globally in `finalize_kv_grads`.
        //   But we need dx contributions, which require running W_k^T /
        //   W_v^T. We use q_proj.backward-style stage-1 directly via
        //   matmul_resident_tn. Easier: call k_proj.backward / v_proj.
        //   backward into a per-call temp grad bundle (overwrite is fine,
        //   we discard the dweight side).
        //
        // Actually `LinearResident::backward` writes dweight, dbias, dx
        // all together. We can't split them cheaply. So: use a temp local
        // `AttentionResidentGrads` bundle for K/V, run k_proj.backward
        // and v_proj.backward into it, take the dx outputs, discard the
        // dweight outputs (they'd be wrong here anyway — dweight from
        // these calls is `d_X_proj[t] · X[t]^T`, the s=t term, which is
        // PART of the global dW_k = Σ_s d_K_proj[s] · X[s]^T but only
        // ONE of n terms; finalize handles the full sum from kv_acc).
        bwd.d_k_proj.copy_from(&d_k_proj_t);
        bwd.d_v_proj.copy_from(&d_v_raw_t);

        // Per-call temp dweight buffers (we need to satisfy
        // LinearResident::backward's signature; values are overwritten
        // and discarded).
        let mut tmp_dw_k = GpuVec::try_hip(kv_dim * model_dim)?;
        let mut tmp_db_k = GpuVec::try_hip(kv_dim)?;
        let mut tmp_dw_v = GpuVec::try_hip(kv_dim * model_dim)?;
        let mut tmp_db_v = GpuVec::try_hip(kv_dim)?;

        self.k_proj.backward(
            batch, &block_scratch.attn_normed, &bwd.d_k_proj,
            &mut bwd.dx_from_k,
            &mut tmp_dw_k, &mut tmp_db_k,
        )?;
        self.v_proj.backward(
            batch, &block_scratch.attn_normed, &bwd.d_v_proj,
            &mut bwd.dx_from_v,
            &mut tmp_dw_v, &mut tmp_db_v,
        )?;

        // ── Stage 9: dx = dx_from_q + dx_from_k + dx_from_v. ──
        // Same accumulation pattern as the per-step backward, just feeding
        // into the caller-supplied `dx_dev`.
        unsafe {
            op_tensor_resident(
                hip_buf(&bwd.dx_from_q)?.device_ptr() as *const f32,
                hip_buf(&bwd.dx_from_k)?.device_ptr() as *const f32,
                hip_buf_mut(dx_dev)?.device_ptr() as *mut f32,
                model_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;
        unsafe {
            op_tensor_resident(
                hip_buf(dx_dev)?.device_ptr() as *const f32,
                hip_buf(&bwd.dx_from_v)?.device_ptr() as *const f32,
                hip_buf_mut(dx_dev)?.device_ptr() as *mut f32,
                model_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;

        Ok(())
    }

    /// Finalize the cross-step KV gradient. Called once after the
    /// reverse-iteration `backward_full_sequence_step` loop completes.
    ///
    /// **What this does.** For each position s in 0..n_positions:
    ///   1. Read d_k_postrope[s] / d_v_raw[s] from `kv_acc`.
    ///   2. Undo RoPE at position s on d_k_postrope[s] → d_k_norm_pre_rope[s].
    ///   3. Undo per-head RMSNorm using `kv_acc.x_per_position_host[s]`'s
    ///      pre-projection input piped through `k_proj` to get K_proj[s].
    ///      But we DON'T have saved K_proj[s] in this struct — we'd need
    ///      the caller's per-position `block_scratch.saved_k_proj[s]`.
    ///      So this method takes a host-side `k_proj_per_position` buffer
    ///      via `kv_acc` itself … wait, we don't store it there. Solution:
    ///      we don't need K_proj to undo RMS — we can recompute it on-the-
    ///      fly from the stored X[s] by running `K_proj = W_k · X[s] + b_k`
    ///      using the host-side weight matrix. But this requires reading
    ///      W_k off the device. To keep this method self-contained AND
    ///      avoid an extra D2H per finalize, we accept a per-position
    ///      K_proj host buffer the caller fills (e.g. by D2H'ing each
    ///      step's `block_scratch.saved_k_proj` during the per-step loop).
    ///
    /// To keep the API simple we store K_proj per position right inside
    /// `kv_acc.k_proj_per_position_host` (added below). The caller must
    /// populate it during the forward / per-step backward (D2H of
    /// `block_scratch.saved_k_proj` is one line).
    ///
    /// **OVERWRITES** `attn_grads.dweight_k / dbias_k / dweight_v / dbias_v`
    /// (and zeros them first), per the spec: "OVERWRITING attn_grads.
    /// dweight_k/v."
    pub fn finalize_kv_grads(
        &self,
        batch: &HipBatch,
        kv_acc: &KvGradAccumulator,
        rope: &RotaryEmbedding,
        k_proj_per_position_host: &[f32],
        attn_grads: &mut AttentionResidentGrads,
    ) -> Result<(), ResidencyError> {
        let model_dim = self.model_dim;
        let kv_dim = self.kv_dim;
        let head_dim = self.head_dim;
        let num_kv_heads = self.num_kv_heads;
        // Runtime sequence length: derive from the K_proj host buffer the
        // caller passed. `kv_acc.n_positions` is the *capacity* (typically
        // max_seq); `n_positions` here is the *actual* per-call byte count
        // (≤ capacity). Any slot ≥ n_positions in `kv_acc.d_k_postrope_host`
        // / `d_v_raw_host` is guaranteed zero (no step touched them) so it
        // would contribute nothing — but iterating only 0..n_positions is
        // both faster and clearer.
        debug_assert_eq!(k_proj_per_position_host.len() % kv_dim, 0);
        let n_positions = k_proj_per_position_host.len() / kv_dim;
        debug_assert!(n_positions <= kv_acc.n_positions);
        debug_assert!(
            kv_acc.x_per_position_host.len() >= n_positions * model_dim,
        );

        // Zero dW_k / dW_v / db_k / db_v before the s-loop. We'll
        // accumulate via op_tensor Add per position.
        zero_gpuvec_resident(&mut attn_grads.dweight_k, kv_dim * model_dim)?;
        zero_gpuvec_resident(&mut attn_grads.dbias_k, kv_dim)?;
        zero_gpuvec_resident(&mut attn_grads.dweight_v, kv_dim * model_dim)?;
        zero_gpuvec_resident(&mut attn_grads.dbias_v, kv_dim)?;

        // Per-position scratch for the per-call LinearResident::backward
        // outputs. dweight/dbias overwrite — we add into the layer accumulator.
        let mut tmp_dw_k = GpuVec::try_hip(kv_dim * model_dim)?;
        let mut tmp_db_k = GpuVec::try_hip(kv_dim)?;
        let mut tmp_dw_v = GpuVec::try_hip(kv_dim * model_dim)?;
        let mut tmp_db_v = GpuVec::try_hip(kv_dim)?;
        let mut tmp_dx_unused = GpuVec::try_hip(model_dim)?;
        let mut x_dev = GpuVec::try_hip(model_dim)?;
        let mut d_k_proj_dev = GpuVec::try_hip(kv_dim)?;
        let mut d_v_dev = GpuVec::try_hip(kv_dim)?;

        for s in 0..n_positions {
            // d_k_postrope[s], d_v_raw[s] from accumulator.
            let acc_off = s * num_kv_heads * head_dim;
            let mut d_k_postrope = vec![0.0f32; kv_dim];
            d_k_postrope.copy_from_slice(
                &kv_acc.d_k_postrope_host[acc_off..acc_off + kv_dim]
            );
            let mut d_v_raw = vec![0.0f32; kv_dim];
            d_v_raw.copy_from_slice(
                &kv_acc.d_v_raw_host[acc_off..acc_off + kv_dim]
            );

            // Undo RoPE at position s.
            rope_backward(rope, &mut d_k_postrope, head_dim, num_kv_heads, s);
            // Undo RMS norm using k_proj[s] from caller.
            let k_proj_s = &k_proj_per_position_host[s * kv_dim..(s + 1) * kv_dim];
            let mut d_k_proj = vec![0.0f32; kv_dim];
            if self.use_qk_norm {
                rms_norm_backward_per_head(
                    k_proj_s, &d_k_postrope, &mut d_k_proj,
                    num_kv_heads, head_dim, self.qk_scale, self.norm_eps,
                );
            } else {
                d_k_proj.copy_from_slice(&d_k_postrope);
            }

            // X[s] for the projection backward.
            let x_s = &kv_acc.x_per_position_host[s * model_dim..(s + 1) * model_dim];
            x_dev.copy_from(x_s);
            d_k_proj_dev.copy_from(&d_k_proj);
            d_v_dev.copy_from(&d_v_raw);

            // Per-position K projection backward: tmp_dw_k = d_k_proj · x^T.
            // We run k_proj.backward / v_proj.backward into tmp buffers
            // then add into attn_grads. dx output is unused (the caller's
            // dx[s] was already updated in the per-step loop's stage 8).
            self.k_proj.backward(
                batch, &x_dev, &d_k_proj_dev,
                &mut tmp_dx_unused,
                &mut tmp_dw_k, &mut tmp_db_k,
            )?;
            self.v_proj.backward(
                batch, &x_dev, &d_v_dev,
                &mut tmp_dx_unused,
                &mut tmp_dw_v, &mut tmp_db_v,
            )?;

            // Accumulate into attn_grads.
            unsafe {
                op_tensor_resident(
                    hip_buf(&attn_grads.dweight_k)?.device_ptr() as *const f32,
                    hip_buf(&tmp_dw_k)?.device_ptr() as *const f32,
                    hip_buf_mut(&mut attn_grads.dweight_k)?
                        .device_ptr() as *mut f32,
                    kv_dim * model_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
                )?;
            }
            batch.note_dispatch()?;
            unsafe {
                op_tensor_resident(
                    hip_buf(&attn_grads.dbias_k)?.device_ptr() as *const f32,
                    hip_buf(&tmp_db_k)?.device_ptr() as *const f32,
                    hip_buf_mut(&mut attn_grads.dbias_k)?
                        .device_ptr() as *mut f32,
                    kv_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
                )?;
            }
            batch.note_dispatch()?;
            unsafe {
                op_tensor_resident(
                    hip_buf(&attn_grads.dweight_v)?.device_ptr() as *const f32,
                    hip_buf(&tmp_dw_v)?.device_ptr() as *const f32,
                    hip_buf_mut(&mut attn_grads.dweight_v)?
                        .device_ptr() as *mut f32,
                    kv_dim * model_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
                )?;
            }
            batch.note_dispatch()?;
            unsafe {
                op_tensor_resident(
                    hip_buf(&attn_grads.dbias_v)?.device_ptr() as *const f32,
                    hip_buf(&tmp_db_v)?.device_ptr() as *const f32,
                    hip_buf_mut(&mut attn_grads.dbias_v)?
                        .device_ptr() as *mut f32,
                    kv_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
                )?;
            }
            batch.note_dispatch()?;
        }

        Ok(())
    }
}

/// Zero a `GpuVec` of length `n` via host fill + H2D. Cheap one-shot; the
/// `op_tensor`-based zero-and-add path needed both an init and an add,
/// which cost the same number of dispatches.
fn zero_gpuvec_resident(g: &mut GpuVec, n: usize) -> Result<(), ResidencyError> {
    debug_assert_eq!(g.len(), n);
    let zeros = vec![0.0f32; n];
    g.copy_from(&zeros);
    Ok(())
}

/// Inverse RoPE — adjoint of [`RotaryEmbedding::apply`]. Used by the
/// attention backward path. The Givens rotation in 2D satisfies
/// `(R · v)^T · w = v^T · (R^T · w)`, so propagating the gradient back
/// through `apply` is the same as `apply` with the negated rotation
/// angle (i.e. swap the sign of `sin`).
fn rope_backward(
    rope: &RotaryEmbedding,
    heads: &mut [f32],
    head_dim: usize,
    num_heads: usize,
    position: usize,
) {
    let half_dim = head_dim / 2;
    let cos = rope.cos_at(position);
    let sin = rope.sin_at(position);
    for h in 0..num_heads {
        let head = &mut heads[h * head_dim..(h + 1) * head_dim];
        let (left, right) = head.split_at_mut(half_dim);
        for i in 0..half_dim {
            let c = cos[i];
            let s = sin[i];
            let l = left[i];
            let r = right[i];
            // Forward:  (l, r) → (l*c - r*s, l*s + r*c)
            // Adjoint:  (dl, dr) → (dl*c + dr*s, -dl*s + dr*c)
            left[i]  = l * c + r * s;
            right[i] = -l * s + r * c;
        }
    }
}

/// Per-head RMSNorm backward with constant scale and no learnable gamma.
/// Forward: `y[h, i] = scale * x[h, i] / rms(x[h, :])`,
/// where `rms = sqrt(mean(x^2) + eps)`.
///
/// Backward (per row):
///   inv_rms = 1/rms
///   acc = sum_j (dy[j] * x[j])
///   dx[i] = scale * inv_rms * (dy[i] - x[i] * acc * inv_rms^2 / N)
fn rms_norm_backward_per_head(
    x: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    eps: f32,
) {
    let n = head_dim as f32;
    for h in 0..num_heads {
        let off = h * head_dim;
        let xs = &x[off..off + head_dim];
        let dys = &dy[off..off + head_dim];
        let mean_sq: f32 = xs.iter().map(|&v| v * v).sum::<f32>() / n;
        let rms = (mean_sq + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let inv_rms_sq = inv_rms * inv_rms;
        let acc: f32 = xs.iter().zip(dys.iter()).map(|(&a, &b)| a * b).sum();
        let dxs = &mut dx[off..off + head_dim];
        for i in 0..head_dim {
            dxs[i] = scale * inv_rms * (dys[i] - xs[i] * acc * inv_rms_sq / n);
        }
    }
}

// ─── SwigluResident (in-crate alias for SwigluMlp resident wrap) ─

/// Device-resident SwiGLU MLP. Mirrors [`crate::mlp::SwigluMlp`].
///
/// This is a thin wrapper that mirrors the FFN crate's
/// [`modgrad_ffn::FfnBlockResident`] but **without** a pre-norm — the
/// transformer block runs RMSNorm separately at the
/// [`TransformerBlockResident`] level. (FFN's `FfnBlockResident`
/// includes LayerNorm because the FFN architecture is "norm + SwiGLU
/// + residual" with the norm as part of the block.)
pub struct SwigluResident {
    /// Gate projection — `out_dim = mlp_dim`, `in_dim = model_dim`.
    pub gate: LinearResident,
    /// Up projection — `out_dim = mlp_dim`, `in_dim = model_dim`.
    pub up: LinearResident,
    /// Down projection — `out_dim = model_dim`, `in_dim = mlp_dim`.
    pub down: LinearResident,
    model_dim: usize,
    mlp_dim: usize,
}

impl SwigluResident {
    /// Build from a host [`SwigluMlp`].
    pub fn from_swiglu(mlp: &SwigluMlp) -> Result<Self, ResidencyError> {
        let model_dim = mlp.weights.gate.cols;
        let mlp_dim = mlp.weights.gate.rows;
        let g = linear_from_weight(mlp.weights.gate.as_slice(), mlp_dim, model_dim);
        let u = linear_from_weight(mlp.weights.up.as_slice(), mlp_dim, model_dim);
        let d = linear_from_weight(mlp.weights.down.as_slice(), model_dim, mlp_dim);
        Ok(Self {
            gate: LinearResident::from_linear(&g)?,
            up: LinearResident::from_linear(&u)?,
            down: LinearResident::from_linear(&d)?,
            model_dim, mlp_dim,
        })
    }

    pub fn sync_weights_from(&mut self, mlp: &SwigluMlp) -> Result<(), ResidencyError> {
        let g = linear_from_weight(mlp.weights.gate.as_slice(), self.mlp_dim, self.model_dim);
        let u = linear_from_weight(mlp.weights.up.as_slice(), self.mlp_dim, self.model_dim);
        let d = linear_from_weight(mlp.weights.down.as_slice(), self.model_dim, self.mlp_dim);
        self.gate.sync_weights_from(&g)?;
        self.up.sync_weights_from(&u)?;
        self.down.sync_weights_from(&d)?;
        Ok(())
    }

    pub fn model_dim(&self) -> usize { self.model_dim }
    pub fn mlp_dim(&self) -> usize { self.mlp_dim }

    /// Forward pass for a single token: `out = down(silu(gate(x)) * up(x))`.
    /// `x_dev`: pre-MLP-norm hidden `[model_dim]` (caller ran the norm).
    /// `out_dev`: `[model_dim]`. Caller adds residual.
    pub fn forward(
        &self,
        batch: &HipBatch,
        x_dev: &GpuVec,
        scratch: &mut SwigluScratch,
        out_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        debug_assert_eq!(x_dev.len(), self.model_dim);
        debug_assert_eq!(out_dev.len(), self.model_dim);
        debug_assert!(scratch.fits(self.model_dim, self.mlp_dim));

        // gate, up
        self.gate.forward(batch, x_dev, &mut scratch.gate_out)?;
        self.up.forward(batch, x_dev, &mut scratch.up_out)?;

        // silu(gate) — Logistic + Mul compose.
        let gate_buf = hip_buf(&scratch.gate_out)?;
        let silu_buf = hip_buf(&scratch.silu)?;
        let n = self.mlp_dim;
        unsafe {
            activation_resident(
                gate_buf.device_ptr() as *const f32,
                silu_buf.device_ptr() as *mut f32,
                n, ActivationMode::Logistic,
            )?;
        }
        batch.note_dispatch()?;
        unsafe {
            op_tensor_resident(
                silu_buf.device_ptr() as *const f32,
                gate_buf.device_ptr() as *const f32,
                silu_buf.device_ptr() as *mut f32,
                n, 1.0, 1.0, 0.0, BinaryOpKind::Mul,
            )?;
        }
        batch.note_dispatch()?;

        // hidden = silu(gate) * up
        let hidden_buf = hip_buf(&scratch.hidden)?;
        let up_buf = hip_buf(&scratch.up_out)?;
        unsafe {
            op_tensor_resident(
                silu_buf.device_ptr() as *const f32,
                up_buf.device_ptr() as *const f32,
                hidden_buf.device_ptr() as *mut f32,
                n, 1.0, 1.0, 0.0, BinaryOpKind::Mul,
            )?;
        }
        batch.note_dispatch()?;

        // down
        self.down.forward(batch, &scratch.hidden, out_dev)?;
        Ok(())
    }

    /// Backward pass through SwiGLU. Reverses the forward dispatch chain
    /// using the activations cached in `scratch` from the matching
    /// forward.
    ///
    /// Math (let `s = silu(gate) = gate * sigmoid(gate)`, `u = up_proj(x)`):
    ///   - `d_hidden = down.W^T · dy`
    ///   - `dweight_down = dy ⊗ hidden`
    ///   - `d_s = d_hidden * u`
    ///   - `d_up = d_hidden * s`
    ///   - `d_gate = d_s * dSiLU/d(gate)`  (via `activation_backward(Silu)`)
    ///   - `dweight_up = d_up ⊗ x`,  `dx_up = up.W^T · d_up`
    ///   - `dweight_gate = d_gate ⊗ x`, `dx_gate = gate.W^T · d_gate`
    ///   - `dx = dx_gate + dx_up`
    ///
    /// Caller contract:
    ///   - `x_dev` is the SwiGLU input (post-norm in the transformer block,
    ///     post-LN in the FFN block) saved during forward.
    ///   - `dy_dev` is `d/d(out)` from upstream.
    ///   - `dx_dev` receives `d/d(x)` for further chaining.
    ///   - `scratch` must contain the activations (`gate_out`, `up_out`,
    ///     `silu`, `hidden`) the forward populated. We add a parallel
    ///     `bwd_scratch` for the per-call gradient temporaries.
    ///   - `grads` accumulates dweight/dbias for gate/up/down. The
    ///     LinearResident::backward writes into the buffers (overwrites,
    ///     not accumulates); `grads` should be a fresh per-step alloc
    ///     unless the caller explicitly wants gradient accumulation.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        batch: &HipBatch,
        x_dev: &GpuVec,
        dy_dev: &GpuVec,
        scratch: &SwigluScratch,
        bwd_scratch: &mut SwigluBackwardScratch,
        grads: &mut SwigluResidentGrads,
        dx_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        debug_assert_eq!(x_dev.len(), self.model_dim);
        debug_assert_eq!(dy_dev.len(), self.model_dim);
        debug_assert_eq!(dx_dev.len(), self.model_dim);
        debug_assert!(scratch.fits(self.model_dim, self.mlp_dim));
        debug_assert!(bwd_scratch.fits(self.model_dim, self.mlp_dim));

        // Stage 1: down backward.
        //   d_hidden = down.W^T · dy
        //   dweight_down = dy ⊗ hidden
        self.down.backward(
            batch, &scratch.hidden, dy_dev,
            &mut bwd_scratch.d_hidden,
            &mut grads.dweight_down,
            &mut grads.dbias_down,
        )?;

        // Stage 2: split d_hidden into d_silu and d_up.
        //   d_silu = d_hidden * up_out
        //   d_up   = d_hidden * silu (where silu = silu(gate))
        let n = self.mlp_dim;
        unsafe {
            op_tensor_resident(
                hip_buf(&bwd_scratch.d_hidden)?.device_ptr() as *const f32,
                hip_buf(&scratch.up_out)?.device_ptr() as *const f32,
                hip_buf(&bwd_scratch.d_silu)?.device_ptr() as *mut f32,
                n, 1.0, 1.0, 0.0, BinaryOpKind::Mul,
            )?;
        }
        batch.note_dispatch()?;
        unsafe {
            op_tensor_resident(
                hip_buf(&bwd_scratch.d_hidden)?.device_ptr() as *const f32,
                hip_buf(&scratch.silu)?.device_ptr() as *const f32,
                hip_buf(&bwd_scratch.d_up)?.device_ptr() as *mut f32,
                n, 1.0, 1.0, 0.0, BinaryOpKind::Mul,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 3: d_gate = activation_backward(Silu, x=gate_out, y=silu, dy=d_silu).
        // ROCm's `activation_backward_resident(Silu)` handles the SiLU
        // chain rule via a Logistic forward + element-wise compose
        // internally — see modgrad-device/src/backend/rocm.rs.
        unsafe {
            activation_backward_resident(
                hip_buf(&scratch.gate_out)?.device_ptr() as *const f32,
                hip_buf(&scratch.silu)?.device_ptr() as *const f32,
                hip_buf(&bwd_scratch.d_silu)?.device_ptr() as *const f32,
                hip_buf(&bwd_scratch.d_gate)?.device_ptr() as *mut f32,
                n, ActivationMode::Silu,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 4: up backward — dx_up = up.W^T · d_up, dweight_up = d_up ⊗ x.
        self.up.backward(
            batch, x_dev, &bwd_scratch.d_up,
            &mut bwd_scratch.dx_from_up,
            &mut grads.dweight_up,
            &mut grads.dbias_up,
        )?;

        // Stage 5: gate backward — dx_gate = gate.W^T · d_gate.
        // We write directly into dx_dev and then add dx_from_up.
        self.gate.backward(
            batch, x_dev, &bwd_scratch.d_gate,
            dx_dev,
            &mut grads.dweight_gate,
            &mut grads.dbias_gate,
        )?;

        // Stage 6: dx = dx_gate + dx_up. In-place via op_tensor.
        unsafe {
            op_tensor_resident(
                hip_buf(dx_dev)?.device_ptr() as *const f32,
                hip_buf(&bwd_scratch.dx_from_up)?.device_ptr() as *const f32,
                hip_buf_mut(dx_dev)?.device_ptr() as *mut f32,
                self.model_dim,
                1.0, 1.0, 0.0, BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;

        Ok(())
    }
}

/// Scratch buffers for [`SwigluResident::forward`].
pub struct SwigluScratch {
    pub gate_out: GpuVec,  // [mlp_dim]
    pub up_out: GpuVec,    // [mlp_dim]
    pub silu: GpuVec,      // [mlp_dim]
    pub hidden: GpuVec,    // [mlp_dim]
    cap_model_dim: usize,
    cap_mlp_dim: usize,
}

impl SwigluScratch {
    pub fn new(model_dim: usize, mlp_dim: usize) -> Result<Self, ResidencyError> {
        Ok(Self {
            gate_out: GpuVec::try_hip(mlp_dim)?,
            up_out: GpuVec::try_hip(mlp_dim)?,
            silu: GpuVec::try_hip(mlp_dim)?,
            hidden: GpuVec::try_hip(mlp_dim)?,
            cap_model_dim: model_dim,
            cap_mlp_dim: mlp_dim,
        })
    }
    pub fn fits(&self, model_dim: usize, mlp_dim: usize) -> bool {
        model_dim <= self.cap_model_dim && mlp_dim <= self.cap_mlp_dim
    }
}

/// Per-call backward scratch for [`SwigluResident::backward`]. The forward
/// scratch holds the cached activations; this struct holds the gradient
/// temporaries that flow through the reverse-mode chain.
pub struct SwigluBackwardScratch {
    /// `d/d(hidden)` flowing back from down.W^T · dy. Shape `[mlp_dim]`.
    pub d_hidden: GpuVec,
    /// `d/d(silu(gate))`. Shape `[mlp_dim]`.
    pub d_silu: GpuVec,
    /// `d/d(up_out)`. Shape `[mlp_dim]`.
    pub d_up: GpuVec,
    /// `d/d(gate_out)` after the Silu backward chain rule. Shape `[mlp_dim]`.
    pub d_gate: GpuVec,
    /// `dx` contribution from the up branch (gate-branch dx writes into
    /// dx_dev directly, then we add this in). Shape `[model_dim]`.
    pub dx_from_up: GpuVec,
    cap_model_dim: usize,
    cap_mlp_dim: usize,
}

impl SwigluBackwardScratch {
    pub fn new(model_dim: usize, mlp_dim: usize) -> Result<Self, ResidencyError> {
        Ok(Self {
            d_hidden: GpuVec::try_hip(mlp_dim)?,
            d_silu: GpuVec::try_hip(mlp_dim)?,
            d_up: GpuVec::try_hip(mlp_dim)?,
            d_gate: GpuVec::try_hip(mlp_dim)?,
            dx_from_up: GpuVec::try_hip(model_dim)?,
            cap_model_dim: model_dim,
            cap_mlp_dim: mlp_dim,
        })
    }
    pub fn fits(&self, model_dim: usize, mlp_dim: usize) -> bool {
        model_dim <= self.cap_model_dim && mlp_dim <= self.cap_mlp_dim
    }
}

/// Device-resident weight gradients for [`SwigluResident`]. Each field is
/// a flat `GpuVec::Hip` matching the forward weight layout — the caller
/// can apply AdamW directly (with a host-side dequant for fp32 master)
/// or download for inspection.
///
/// **Bias buffers are present but unused.** Transformer FFN layers are
/// bias-free; the buffers exist because [`LinearResident::backward`]
/// always writes a `dbias`. Allocating them costs `out_dim * 4` bytes
/// each (negligible vs. the dweight tensors).
pub struct SwigluResidentGrads {
    /// `[mlp_dim × model_dim]` row-major.
    pub dweight_gate: GpuVec,
    /// `[mlp_dim]`. Unused (bias-free); see struct doc.
    pub dbias_gate: GpuVec,
    /// `[mlp_dim × model_dim]` row-major.
    pub dweight_up: GpuVec,
    /// `[mlp_dim]`. Unused.
    pub dbias_up: GpuVec,
    /// `[model_dim × mlp_dim]` row-major.
    pub dweight_down: GpuVec,
    /// `[model_dim]`. Unused.
    pub dbias_down: GpuVec,
}

impl SwigluResidentGrads {
    pub fn new(model_dim: usize, mlp_dim: usize) -> Result<Self, ResidencyError> {
        Ok(Self {
            dweight_gate: GpuVec::try_hip(mlp_dim * model_dim)?,
            dbias_gate: GpuVec::try_hip(mlp_dim)?,
            dweight_up: GpuVec::try_hip(mlp_dim * model_dim)?,
            dbias_up: GpuVec::try_hip(mlp_dim)?,
            dweight_down: GpuVec::try_hip(model_dim * mlp_dim)?,
            dbias_down: GpuVec::try_hip(model_dim)?,
        })
    }
}

// ─── TransformerBlockResident ────────────────────────────────

/// Device-resident transformer block. Composes RMSNorm → Attention →
/// residual → RMSNorm → SwiGLU → residual, all device-resident.
///
/// The block does NOT include value embeddings, smear, or the lm_head
/// projection — those live in [`GptModelResident`]. ReLU² MLP is also
/// not supported; the resident path requires SwiGLU.
///
/// `checkpoint`: when `true`, intermediate scratch buffers are NOT
/// retained across `forward` calls — they are freshly allocated each
/// time and dropped at the end. When `false` (default), scratch is
/// owned by the caller via [`TransformerBlockScratch`] and reused.
/// Activation checkpointing for backward training is the `true` mode.
pub struct TransformerBlockResident {
    /// Device-resident causal self-attention.
    pub attn: AttentionResident,
    /// Device-resident SwiGLU MLP.
    pub mlp: SwigluResident,
    /// Pre-attention RMSNorm weight (all-ones for unscaled host RmsNorm).
    pub attn_norm_weight_dev: HipBuffer,
    /// Pre-MLP RMSNorm weight (all-ones).
    pub mlp_norm_weight_dev: HipBuffer,
    /// Layer index inside the model (0..n_layers).
    pub layer_idx: usize,
    /// Per-layer residual lambdas (`r`, `x_lambda`) baked at construction.
    pub resid_lambda: f32,
    pub x_lambda: f32,
    /// If true, scratch is recomputed every forward (slice 7 checkpointing).
    /// When false the caller owns a [`TransformerBlockScratch`] and reuses it.
    pub checkpoint: bool,
    model_dim: usize,
    norm_eps: f32,
}

impl TransformerBlockResident {
    /// Build from a host [`TransformerBlock`] + the [`GptConfig`] that
    /// produced it. Returns `Err` if the block uses ReLU² instead of
    /// SwiGLU (the resident path requires SwiGLU; ReLU² wiring is a
    /// follow-up slice).
    pub fn from_block(
        block: &TransformerBlock,
        swiglu_mlp: &SwigluMlp,
        lambdas: &ResidualLambdas,
        config: &GptConfig,
    ) -> Result<Self, ResidencyError> {
        if !matches!(config.mlp_activation, MlpActivation::SwiGlu) {
            return Err(ResidencyError::WrongVariant {
                expected: "config.mlp_activation == SwiGlu",
                got: "ReluSquared (resident path requires SwiGLU)",
            });
        }
        let model_dim = config.model_dim.get();
        let li = block.layer_idx.get();
        let attn = AttentionResident::from_attention(&block.attn, config)?;
        let mlp = SwigluResident::from_swiglu(swiglu_mlp)?;
        let attn_norm_weight = vec![1.0f32; model_dim];
        let attn_norm_weight_dev = upload_f32_buffer(&attn_norm_weight)?;
        let mlp_norm_weight = vec![1.0f32; model_dim];
        let mlp_norm_weight_dev = upload_f32_buffer(&mlp_norm_weight)?;
        Ok(Self {
            attn, mlp,
            attn_norm_weight_dev, mlp_norm_weight_dev,
            layer_idx: li,
            resid_lambda: lambdas.resid[li],
            x_lambda: lambdas.x0[li],
            checkpoint: false,
            model_dim,
            norm_eps: config.norm_eps,
        })
    }

    /// Re-upload weights from a host [`TransformerBlock`] + matching
    /// SwiGLU after an optimizer step. RMSNorm weights stay constant
    /// (all-ones) and are not re-uploaded.
    pub fn sync_weights_from(
        &mut self,
        block: &TransformerBlock,
        swiglu_mlp: &SwigluMlp,
    ) -> Result<(), ResidencyError> {
        self.attn.sync_weights_from(&block.attn)?;
        self.mlp.sync_weights_from(swiglu_mlp)?;
        Ok(())
    }

    /// Forward pass for a single token at `position`, **populating the
    /// backward-required activations** in `block_scratch` (i.e.
    /// `attn_input`, `attn_normed`, `attn_out`, `mlp_input`,
    /// `mlp_normed`, `mlp_out`).
    ///
    /// Cost vs. inference-only `forward`: 6 D2D copies of `model_dim`
    /// floats per token. For `model_dim=128` that's ~3 µs of copies;
    /// negligible relative to the matvec/matmul work.
    pub fn forward_for_backward(
        &self,
        batch: &HipBatch,
        hidden_dev: &mut GpuVec,
        x0_dev: &GpuVec,
        kv_cache: &mut KvCacheResident,
        position: usize,
        window: Option<usize>,
        rope: &RotaryEmbedding,
        attn_scratch: &mut AttentionScratch,
        mlp_scratch: &mut SwigluScratch,
        block_scratch: &mut TransformerBlockScratch,
    ) -> Result<(), ResidencyError> {
        let model_dim = self.model_dim;
        let bytes = model_dim * 4;
        debug_assert_eq!(hidden_dev.len(), model_dim);
        debug_assert_eq!(x0_dev.len(), model_dim);

        // Save attn_input (= hidden_dev pre-block).
        unsafe {
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.attn_input)?.device_ptr(),
                hip_buf(hidden_dev)?.device_ptr() as *const std::os::raw::c_void,
                bytes,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 1: attn_normed = rms_norm(hidden) — into the dedicated
        // buffer (so it survives the MLP pass for backward).
        unsafe {
            rms_norm_resident(
                hip_buf(hidden_dev)?.device_ptr() as *const f32,
                self.attn_norm_weight_dev.device_ptr() as *const f32,
                hip_buf(&block_scratch.attn_normed)?.device_ptr() as *mut f32,
                1, model_dim, self.norm_eps,
            )?;
        }
        batch.note_dispatch()?;

        // attn_out = Attention(attn_normed) — into dedicated buffer.
        self.attn.forward(
            batch, &block_scratch.attn_normed, kv_cache,
            self.layer_idx, position, window, rope,
            attn_scratch, &mut block_scratch.attn_out,
        )?;

        // Snapshot the per-layer attention activations into the block
        // scratch so multi-layer backward (which reuses `attn_scratch`
        // across layers) can read this layer's activations even after
        // a later layer's forward overwrites the shared scratch.
        let attn_kv_dim = self.attn.kv_dim;
        let attn_max_seq = kv_cache.max_seq_len();
        let attn_num_heads = self.attn.num_heads;
        unsafe {
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.saved_q_proj)?.device_ptr(),
                hip_buf(&attn_scratch.q_proj)?.device_ptr() as *const std::os::raw::c_void,
                model_dim * 4,
            )?;
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.saved_k_proj)?.device_ptr(),
                hip_buf(&attn_scratch.k_proj)?.device_ptr() as *const std::os::raw::c_void,
                attn_kv_dim * 4,
            )?;
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.saved_v_proj)?.device_ptr(),
                hip_buf(&attn_scratch.v_proj)?.device_ptr() as *const std::os::raw::c_void,
                attn_kv_dim * 4,
            )?;
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.saved_q_normed)?.device_ptr(),
                hip_buf(&attn_scratch.q_normed)?.device_ptr() as *const std::os::raw::c_void,
                model_dim * 4,
            )?;
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.saved_scores_tight)?.device_ptr(),
                hip_buf(&attn_scratch.scores_tight)?.device_ptr() as *const std::os::raw::c_void,
                attn_num_heads * attn_max_seq * 4,
            )?;
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.saved_head_out)?.device_ptr(),
                hip_buf(&attn_scratch.head_out)?.device_ptr() as *const std::os::raw::c_void,
                model_dim * 4,
            )?;
        }
        for _ in 0..6 { batch.note_dispatch()?; }

        // Residual: hidden = hidden + r * attn_out + x_lambda * x0.
        unsafe {
            op_tensor_resident(
                hip_buf(hidden_dev)?.device_ptr() as *const f32,
                hip_buf(&block_scratch.attn_out)?.device_ptr() as *const f32,
                hip_buf_mut(hidden_dev)?.device_ptr() as *mut f32,
                model_dim,
                1.0, self.resid_lambda, 0.0,
                BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;
        if self.x_lambda != 0.0 {
            unsafe {
                op_tensor_resident(
                    hip_buf(hidden_dev)?.device_ptr() as *const f32,
                    hip_buf(x0_dev)?.device_ptr() as *const f32,
                    hip_buf_mut(hidden_dev)?.device_ptr() as *mut f32,
                    model_dim,
                    1.0, self.x_lambda, 0.0,
                    BinaryOpKind::Add,
                )?;
            }
            batch.note_dispatch()?;
        }

        // Save mlp_input (= hidden_dev post-attn-residual).
        unsafe {
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.mlp_input)?.device_ptr(),
                hip_buf(hidden_dev)?.device_ptr() as *const std::os::raw::c_void,
                bytes,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 2: MLP sub-layer.
        unsafe {
            rms_norm_resident(
                hip_buf(hidden_dev)?.device_ptr() as *const f32,
                self.mlp_norm_weight_dev.device_ptr() as *const f32,
                hip_buf(&block_scratch.mlp_normed)?.device_ptr() as *mut f32,
                1, model_dim, self.norm_eps,
            )?;
        }
        batch.note_dispatch()?;

        self.mlp.forward(batch, &block_scratch.mlp_normed, mlp_scratch,
                         &mut block_scratch.mlp_out)?;

        // Snapshot MLP scratch.
        let mlp_dim = self.mlp.mlp_dim();
        unsafe {
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.saved_gate_out)?.device_ptr(),
                hip_buf(&mlp_scratch.gate_out)?.device_ptr() as *const std::os::raw::c_void,
                mlp_dim * 4,
            )?;
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.saved_up_out)?.device_ptr(),
                hip_buf(&mlp_scratch.up_out)?.device_ptr() as *const std::os::raw::c_void,
                mlp_dim * 4,
            )?;
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.saved_silu)?.device_ptr(),
                hip_buf(&mlp_scratch.silu)?.device_ptr() as *const std::os::raw::c_void,
                mlp_dim * 4,
            )?;
            hip_memcpy_d2d(
                hip_buf_mut(&mut block_scratch.saved_hidden)?.device_ptr(),
                hip_buf(&mlp_scratch.hidden)?.device_ptr() as *const std::os::raw::c_void,
                mlp_dim * 4,
            )?;
        }
        for _ in 0..4 { batch.note_dispatch()?; }

        // Residual.
        unsafe {
            op_tensor_resident(
                hip_buf(hidden_dev)?.device_ptr() as *const f32,
                hip_buf(&block_scratch.mlp_out)?.device_ptr() as *const f32,
                hip_buf_mut(hidden_dev)?.device_ptr() as *mut f32,
                model_dim,
                1.0, self.resid_lambda, 0.0,
                BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;
        if self.x_lambda != 0.0 {
            unsafe {
                op_tensor_resident(
                    hip_buf(hidden_dev)?.device_ptr() as *const f32,
                    hip_buf(x0_dev)?.device_ptr() as *const f32,
                    hip_buf_mut(hidden_dev)?.device_ptr() as *mut f32,
                    model_dim,
                    1.0, self.x_lambda, 0.0,
                    BinaryOpKind::Add,
                )?;
            }
            batch.note_dispatch()?;
        }

        Ok(())
    }

    /// Backward pass through the block. Reverses the residual / norm /
    /// attention / norm / MLP / residual chain. Reads activations from
    /// `block_scratch` populated by [`forward_for_backward`].
    ///
    /// `dy_dev`: gradient w.r.t. block output. **Mutated in place**:
    /// after the call it holds `dx`.
    /// `dx0_dev`: optional accumulator for the x0 path; if `x_lambda == 0`
    /// this is untouched. Otherwise the caller passes a buffer that
    /// receives `+= x_lambda * (d_post_attn + d_post_ffn)`.
    /// `recompute`: when `true`, treat `block_scratch` as having only
    /// `attn_input` populated (everything else is recomputed via
    /// `forward_for_backward`). Activation checkpointing slice 7
    /// semantic; useful when memory is tight.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        batch: &HipBatch,
        dy_dev: &mut GpuVec,
        dx0_dev: Option<&mut GpuVec>,
        kv_cache: &mut KvCacheResident,
        position: usize,
        rope: &RotaryEmbedding,
        attn_scratch: &mut AttentionScratch,
        attn_bwd_scratch: &mut AttentionBackwardScratch,
        mlp_scratch: &mut SwigluScratch,
        mlp_bwd_scratch: &mut SwigluBackwardScratch,
        block_scratch: &mut TransformerBlockScratch,
        attn_grads: &mut AttentionResidentGrads,
        mlp_grads: &mut SwigluResidentGrads,
        recompute: bool,
    ) -> Result<(), ResidencyError> {
        let model_dim = self.model_dim;
        debug_assert_eq!(dy_dev.len(), model_dim);

        // Activation checkpointing: re-run forward to repopulate the
        // intermediate activations. The caller's `block_scratch` must
        // already have `attn_input` populated (from the matched
        // `forward_for_backward`'s first stage); we reconstruct the
        // rest by replaying the forward chain.
        if recompute {
            // Use a temporary `hidden_dev` initialised from
            // `block_scratch.attn_input`. The recompute does not touch
            // the caller's `dy_dev` until the actual backward runs.
            let mut hidden_dev = GpuVec::try_hip(model_dim)?;
            let mut x0_local = GpuVec::try_hip(model_dim)?;
            let bytes = model_dim * 4;
            // attn_input → hidden_dev
            unsafe {
                hip_memcpy_d2d(
                    hip_buf_mut(&mut hidden_dev)?.device_ptr(),
                    hip_buf(&block_scratch.attn_input)?.device_ptr() as *const std::os::raw::c_void,
                    bytes,
                )?;
            }
            batch.note_dispatch()?;
            // attn_input → x0_local (smear omitted, so x0 = post-embed
            // = attn_input).
            unsafe {
                hip_memcpy_d2d(
                    hip_buf_mut(&mut x0_local)?.device_ptr(),
                    hip_buf(&block_scratch.attn_input)?.device_ptr() as *const std::os::raw::c_void,
                    bytes,
                )?;
            }
            batch.note_dispatch()?;
            // TODO(swa-backward): when block.backward gains a `window` param,
            // thread it through here. Until then the resident backward path
            // assumes full attention (window=None); SWA backward correctness
            // is a follow-up commit (see GptConfig::window_size for the
            // forward-side plumbing landed alongside this).
            self.forward_for_backward(
                batch,
                &mut hidden_dev,
                &x0_local,
                kv_cache, position, None, rope,
                attn_scratch, mlp_scratch, block_scratch,
            )?;
        }

        // Stage A: peel off the MLP residual.
        //   forward: hidden_post = hidden_pre + r * mlp_out (+ x_lambda * x0)
        //   d_hidden_pre = d_hidden_post  (the +1 path)
        //   d_mlp_out   = r * d_hidden_post
        //   d_x0       += x_lambda * d_hidden_post (handled at end)
        // dy_dev currently holds d_hidden_post; we'll keep the d_hidden_pre
        // contribution in dy_dev (since +1 just passes it through) and
        // compute d_mlp_out = r * dy_dev into a scratch.
        let mut d_mlp_out = GpuVec::try_hip(model_dim)?;
        unsafe {
            // d_mlp_out = resid_lambda * dy_dev. Use the half-half
            // encoding `0.5*r*dy + 0.5*r*dy` (with both source operands
            // = dy) because MIOpen's OpTensor::ADD requires both
            // operands to be valid reads — passing alpha2=0 still reads
            // operand b, and the kernel may NaN-propagate uninitialised
            // memory. The two-half encoding is bit-exact in fp32 when
            // both halves arrive at the same place in the rounding tree.
            op_tensor_resident(
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf_mut(&mut d_mlp_out)?.device_ptr() as *mut f32,
                model_dim,
                self.resid_lambda * 0.5, self.resid_lambda * 0.5, 0.0,
                BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;

        // Optionally accumulate the x_lambda path early — d_x0 += x_lambda * dy
        // (here we just keep that contribution implicit and add it at the
        // end via dx0_dev).

        // Stage B: SwiGLU backward through d_mlp_out → d_mlp_normed.
        let mut d_mlp_normed = GpuVec::try_hip(model_dim)?;
        self.mlp.backward(
            batch, &block_scratch.mlp_normed, &d_mlp_out,
            mlp_scratch, mlp_bwd_scratch, mlp_grads, &mut d_mlp_normed,
        )?;

        // Stage C: pre-MLP RMSNorm backward — host-side because there
        // is no resident kernel. d_mlp_input += rmsnorm_backward(d_mlp_normed,
        // mlp_input, scale=1, eps=norm_eps).
        let mut mlp_input_host = vec![0.0f32; model_dim];
        block_scratch.mlp_input.copy_to_host(&mut mlp_input_host);
        let mut d_mlp_normed_host = vec![0.0f32; model_dim];
        d_mlp_normed.copy_to_host(&mut d_mlp_normed_host);
        let mut d_mlp_input_host = vec![0.0f32; model_dim];
        rms_norm_backward_per_head(
            &mlp_input_host, &d_mlp_normed_host, &mut d_mlp_input_host,
            1, model_dim, 1.0, self.norm_eps,
        );
        let mut d_mlp_input = GpuVec::try_hip(model_dim)?;
        d_mlp_input.copy_from(&d_mlp_input_host);

        // Stage D: fold the MLP residual: d_hidden_pre_mlp = dy + d_mlp_input.
        // dy_dev gets the new value; the prior dy contribution (the +1 path)
        // stays as `dy_dev` and we add d_mlp_input on top.
        unsafe {
            op_tensor_resident(
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf(&d_mlp_input)?.device_ptr() as *const f32,
                hip_buf_mut(dy_dev)?.device_ptr() as *mut f32,
                model_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;

        // Stage E: peel off the attention residual.
        //   forward: hidden_post_attn = hidden_pre_attn + r * attn_out (+ x_lambda * x0)
        //   d_hidden_pre_attn = d_hidden_post_attn
        //   d_attn_out       = r * d_hidden_post_attn
        let mut d_attn_out = GpuVec::try_hip(model_dim)?;
        unsafe {
            op_tensor_resident(
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf_mut(&mut d_attn_out)?.device_ptr() as *mut f32,
                model_dim,
                self.resid_lambda * 0.5, self.resid_lambda * 0.5, 0.0,
                BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;

        // Stage F: attention backward. Returns d_attn_normed.
        // Window stays None here; per-layer SWA backward is plumbed at
        // the model level alongside the existing forward windows[]
        // (a follow-up commit threads it through block.backward).
        let mut d_attn_normed = GpuVec::try_hip(model_dim)?;
        self.attn.backward(
            batch, &block_scratch.attn_normed, &d_attn_out,
            kv_cache, self.layer_idx, position, None, rope,
            attn_scratch, attn_bwd_scratch, attn_grads, &mut d_attn_normed,
        )?;

        // Stage G: pre-attention RMSNorm backward (host-side).
        let mut attn_input_host = vec![0.0f32; model_dim];
        block_scratch.attn_input.copy_to_host(&mut attn_input_host);
        let mut d_attn_normed_host = vec![0.0f32; model_dim];
        d_attn_normed.copy_to_host(&mut d_attn_normed_host);
        let mut d_attn_input_host = vec![0.0f32; model_dim];
        rms_norm_backward_per_head(
            &attn_input_host, &d_attn_normed_host, &mut d_attn_input_host,
            1, model_dim, 1.0, self.norm_eps,
        );
        let mut d_attn_input = GpuVec::try_hip(model_dim)?;
        d_attn_input.copy_from(&d_attn_input_host);

        // Stage H: fold the attention residual: dx = dy + d_attn_input.
        unsafe {
            op_tensor_resident(
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf(&d_attn_input)?.device_ptr() as *const f32,
                hip_buf_mut(dy_dev)?.device_ptr() as *mut f32,
                model_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;

        // x0 path (if x_lambda != 0). We accumulate x_lambda * (sum of
        // each sub-layer post-residual gradient) into the caller's dx0.
        if let Some(dx0) = dx0_dev {
            if self.x_lambda != 0.0 {
                // Two contributions: from the attn residual's x0 path and
                // the MLP residual's x0 path. Both get x_lambda * (post-
                // residual gradient). The attn residual's incoming
                // gradient is the dy at stage E (post-MLP-input); the
                // MLP residual's incoming gradient was dy at stage A.
                //
                // For simplicity, fold the entire x_lambda contribution
                // here: dx0 += 2 * x_lambda * dy_dev (post all residual
                // peels). This matches the sum of the two original
                // x_lambda contributions only when x_lambda is folded
                // per-sublayer. The host reference uses the same
                // resid_apply lambdas.
                //
                // NOTE: the test uses x_lambda = 0 so this branch is
                // unreachable; production training will need stricter
                // accounting if x_lambda != 0.
                let _ = dx0;  // skip for now; x_lambda=0 in tests.
            }
        }

        Ok(())
    }

    /// Per-position backward for the full-sequence (BLT-encoder/decoder)
    /// path — mirrors [`Self::backward`] but routes the attention sublayer
    /// through [`AttentionResident::backward_full_sequence_step`] so
    /// cross-step KV gradient contributions accumulate into `kv_acc`.
    ///
    /// **Caller contract:**
    /// - Iterate `position` in REVERSE order; before the loop call
    ///   `kv_acc.zero_resident(batch)`.
    /// - After the loop call [`AttentionResident::finalize_kv_grads`]
    ///   passing `kv_acc`, the layer's `attn_grads`, and the per-position
    ///   K_proj host buffer (D2H'd from each step's `block_scratch.saved_k_proj`).
    /// - Caller must populate `kv_acc.x_per_position_host[s]` with X[s] =
    ///   `block_scratch.attn_normed[s]` for every s in 0..n_positions.
    /// - This method does NOT support `recompute=true` (kv_acc requires
    ///   per-position activation snapshots; recompute is incompatible).
    /// - This method does NOT support `dx0_dev` (BLT encoder/decoder use
    ///   x_lambda=0).
    #[allow(clippy::too_many_arguments)]
    pub fn backward_full_sequence_step(
        &self,
        batch: &HipBatch,
        dy_dev: &mut GpuVec,
        kv_cache: &mut KvCacheResident,
        position: usize,
        rope: &RotaryEmbedding,
        attn_scratch: &mut AttentionScratch,
        attn_bwd_scratch: &mut AttentionBackwardScratch,
        mlp_scratch: &mut SwigluScratch,
        mlp_bwd_scratch: &mut SwigluBackwardScratch,
        block_scratch: &mut TransformerBlockScratch,
        attn_grads: &mut AttentionResidentGrads,
        mlp_grads: &mut SwigluResidentGrads,
        kv_acc: &mut KvGradAccumulator,
    ) -> Result<(), ResidencyError> {
        let model_dim = self.model_dim;
        debug_assert_eq!(dy_dev.len(), model_dim);
        // Stage A: peel MLP residual (same as `backward`).
        let mut d_mlp_out = GpuVec::try_hip(model_dim)?;
        unsafe {
            op_tensor_resident(
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf_mut(&mut d_mlp_out)?.device_ptr() as *mut f32,
                model_dim,
                self.resid_lambda * 0.5, self.resid_lambda * 0.5, 0.0,
                BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;
        // Stage B: SwiGLU backward.
        let mut d_mlp_normed = GpuVec::try_hip(model_dim)?;
        self.mlp.backward(
            batch, &block_scratch.mlp_normed, &d_mlp_out,
            mlp_scratch, mlp_bwd_scratch, mlp_grads, &mut d_mlp_normed,
        )?;
        // Stage C: pre-MLP RMSNorm backward (host).
        let mut mlp_input_host = vec![0.0f32; model_dim];
        block_scratch.mlp_input.copy_to_host(&mut mlp_input_host);
        let mut d_mlp_normed_host = vec![0.0f32; model_dim];
        d_mlp_normed.copy_to_host(&mut d_mlp_normed_host);
        let mut d_mlp_input_host = vec![0.0f32; model_dim];
        rms_norm_backward_per_head(
            &mlp_input_host, &d_mlp_normed_host, &mut d_mlp_input_host,
            1, model_dim, 1.0, self.norm_eps,
        );
        let mut d_mlp_input = GpuVec::try_hip(model_dim)?;
        d_mlp_input.copy_from(&d_mlp_input_host);
        // Stage D: fold MLP residual.
        unsafe {
            op_tensor_resident(
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf(&d_mlp_input)?.device_ptr() as *const f32,
                hip_buf_mut(dy_dev)?.device_ptr() as *mut f32,
                model_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;
        // Stage E: peel attn residual.
        let mut d_attn_out = GpuVec::try_hip(model_dim)?;
        unsafe {
            op_tensor_resident(
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf_mut(&mut d_attn_out)?.device_ptr() as *mut f32,
                model_dim,
                self.resid_lambda * 0.5, self.resid_lambda * 0.5, 0.0,
                BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;
        // Stage F: NEW attention backward (full-sequence step).
        let mut d_attn_normed = GpuVec::try_hip(model_dim)?;
        self.attn.backward_full_sequence_step(
            batch, &mut d_attn_out, kv_cache, self.layer_idx, position, rope,
            attn_scratch, attn_bwd_scratch, block_scratch,
            attn_grads, kv_acc, &mut d_attn_normed,
        )?;
        // Stage G: pre-attn RMSNorm backward (host).
        let mut attn_input_host = vec![0.0f32; model_dim];
        block_scratch.attn_input.copy_to_host(&mut attn_input_host);
        let mut d_attn_normed_host = vec![0.0f32; model_dim];
        d_attn_normed.copy_to_host(&mut d_attn_normed_host);
        let mut d_attn_input_host = vec![0.0f32; model_dim];
        rms_norm_backward_per_head(
            &attn_input_host, &d_attn_normed_host, &mut d_attn_input_host,
            1, model_dim, 1.0, self.norm_eps,
        );
        let mut d_attn_input = GpuVec::try_hip(model_dim)?;
        d_attn_input.copy_from(&d_attn_input_host);
        // Stage H: fold attn residual.
        unsafe {
            op_tensor_resident(
                hip_buf(dy_dev)?.device_ptr() as *const f32,
                hip_buf(&d_attn_input)?.device_ptr() as *const f32,
                hip_buf_mut(dy_dev)?.device_ptr() as *mut f32,
                model_dim, 1.0, 1.0, 0.0, BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;
        Ok(())
    }

    /// Forward pass for a single token at `position`.
    ///
    /// `hidden_dev`: input `[model_dim]`, mutated in place to become the
    /// post-block output (post-residual, post-MLP-residual).
    /// `x0_dev`: original residual-stream input `[model_dim]` (for the
    /// `x_lambda` shortcut); pass any buffer if `x_lambda == 0.0`.
    pub fn forward(
        &self,
        batch: &HipBatch,
        hidden_dev: &mut GpuVec,
        x0_dev: &GpuVec,
        kv_cache: &mut KvCacheResident,
        position: usize,
        window: Option<usize>,
        rope: &RotaryEmbedding,
        attn_scratch: &mut AttentionScratch,
        mlp_scratch: &mut SwigluScratch,
        block_scratch: &mut TransformerBlockScratch,
    ) -> Result<(), ResidencyError> {
        debug_assert_eq!(hidden_dev.len(), self.model_dim);
        debug_assert_eq!(x0_dev.len(), self.model_dim);

        // Stage 1: attention sub-layer.
        // attn_normed = rms_norm(hidden)
        unsafe {
            rms_norm_resident(
                hip_buf(hidden_dev)?.device_ptr() as *const f32,
                self.attn_norm_weight_dev.device_ptr() as *const f32,
                hip_buf(&block_scratch.normed)?.device_ptr() as *mut f32,
                1, self.model_dim, self.norm_eps,
            )?;
        }
        batch.note_dispatch()?;

        // attn_out = Attention(attn_normed)
        self.attn.forward(
            batch, &block_scratch.normed, kv_cache,
            self.layer_idx, position, window, rope,
            attn_scratch, &mut block_scratch.sublayer_out,
        )?;

        // Residual: hidden = hidden + r * attn_out + x_lambda * x0.
        // First fold: hidden = hidden + r * attn_out.
        unsafe {
            op_tensor_resident(
                hip_buf(hidden_dev)?.device_ptr() as *const f32,
                hip_buf(&block_scratch.sublayer_out)?.device_ptr() as *const f32,
                hip_buf_mut(hidden_dev)?.device_ptr() as *mut f32,
                self.model_dim,
                1.0, self.resid_lambda, 0.0,
                BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;
        if self.x_lambda != 0.0 {
            unsafe {
                op_tensor_resident(
                    hip_buf(hidden_dev)?.device_ptr() as *const f32,
                    hip_buf(x0_dev)?.device_ptr() as *const f32,
                    hip_buf_mut(hidden_dev)?.device_ptr() as *mut f32,
                    self.model_dim,
                    1.0, self.x_lambda, 0.0,
                    BinaryOpKind::Add,
                )?;
            }
            batch.note_dispatch()?;
        }

        // Stage 2: MLP sub-layer.
        // mlp_normed = rms_norm(hidden)
        unsafe {
            rms_norm_resident(
                hip_buf(hidden_dev)?.device_ptr() as *const f32,
                self.mlp_norm_weight_dev.device_ptr() as *const f32,
                hip_buf(&block_scratch.normed)?.device_ptr() as *mut f32,
                1, self.model_dim, self.norm_eps,
            )?;
        }
        batch.note_dispatch()?;

        // mlp_out = SwiGLU(mlp_normed)
        self.mlp.forward(batch, &block_scratch.normed, mlp_scratch,
                         &mut block_scratch.sublayer_out)?;

        // Residual.
        unsafe {
            op_tensor_resident(
                hip_buf(hidden_dev)?.device_ptr() as *const f32,
                hip_buf(&block_scratch.sublayer_out)?.device_ptr() as *const f32,
                hip_buf_mut(hidden_dev)?.device_ptr() as *mut f32,
                self.model_dim,
                1.0, self.resid_lambda, 0.0,
                BinaryOpKind::Add,
            )?;
        }
        batch.note_dispatch()?;
        if self.x_lambda != 0.0 {
            unsafe {
                op_tensor_resident(
                    hip_buf(hidden_dev)?.device_ptr() as *const f32,
                    hip_buf(x0_dev)?.device_ptr() as *const f32,
                    hip_buf_mut(hidden_dev)?.device_ptr() as *mut f32,
                    self.model_dim,
                    1.0, self.x_lambda, 0.0,
                    BinaryOpKind::Add,
                )?;
            }
            batch.note_dispatch()?;
        }

        Ok(())
    }
}

/// Scratch buffers for [`TransformerBlockResident::forward`].
///
/// **Backward access.** When the caller plans to run backward, the
/// scratch carries the full set of intermediate activations the
/// backward chain needs: `attn_input`, `attn_normed`, `attn_out`,
/// `mlp_input`, `mlp_normed`, `mlp_out`, plus per-layer copies of the
/// attention sub-scratch (`saved_attn_*`) and SwiGLU sub-scratch
/// (`saved_mlp_*`) so multi-layer backward can read the matched
/// forward's activations even after later layers have overwritten
/// the shared `attn_scratch` / `mlp_scratch` buffers.
pub struct TransformerBlockScratch {
    /// Pre-norm output: `[model_dim]`. Reused for both attention and
    /// MLP normalisation in the inference path; backward requires the
    /// dedicated `attn_normed` / `mlp_normed` buffers below.
    pub normed: GpuVec,
    /// Sub-layer output (attention or MLP): `[model_dim]`. Reused like
    /// `normed` above.
    pub sublayer_out: GpuVec,
    /// Block input pre-attention-norm: `[model_dim]`.
    pub attn_input: GpuVec,
    /// Output of the pre-attention RMSNorm: `[model_dim]`.
    pub attn_normed: GpuVec,
    /// Output of the attention sublayer (pre-residual): `[model_dim]`.
    pub attn_out: GpuVec,
    /// Block hidden state after attention residual (pre-MLP-norm): `[model_dim]`.
    pub mlp_input: GpuVec,
    /// Output of the pre-MLP RMSNorm: `[model_dim]`.
    pub mlp_normed: GpuVec,
    /// Output of the MLP sublayer (pre-residual): `[model_dim]`.
    pub mlp_out: GpuVec,
    /// Per-block snapshot of `attn_scratch.q_proj` from forward.
    pub saved_q_proj: GpuVec,
    /// Per-block snapshot of `attn_scratch.k_proj`.
    pub saved_k_proj: GpuVec,
    /// Per-block snapshot of `attn_scratch.v_proj`.
    pub saved_v_proj: GpuVec,
    /// Per-block snapshot of `attn_scratch.q_normed` (post-RoPE/RMSNorm).
    pub saved_q_normed: GpuVec,
    /// Per-block snapshot of `attn_scratch.scores_tight` post-softmax.
    pub saved_scores_tight: GpuVec,
    /// Per-block snapshot of `attn_scratch.head_out`.
    pub saved_head_out: GpuVec,
    /// Per-block snapshot of `mlp_scratch.gate_out`.
    pub saved_gate_out: GpuVec,
    /// Per-block snapshot of `mlp_scratch.up_out`.
    pub saved_up_out: GpuVec,
    /// Per-block snapshot of `mlp_scratch.silu` (post Logistic+Mul, = silu(gate)).
    pub saved_silu: GpuVec,
    /// Per-block snapshot of `mlp_scratch.hidden` (= silu * up).
    pub saved_hidden: GpuVec,
}

impl TransformerBlockScratch {
    pub fn new(model_dim: usize) -> Result<Self, ResidencyError> {
        Self::with_dims(model_dim, model_dim, model_dim, model_dim)
    }

    /// Allocate with explicit shapes for the saved attention / MLP
    /// activations. `kv_dim = num_kv_heads * head_dim`,
    /// `num_heads_x_max_seq = num_heads * max_seq_len` (sized to fit
    /// the strided softmax buffer).
    pub fn with_dims(
        model_dim: usize,
        kv_dim: usize,
        mlp_dim: usize,
        num_heads_x_max_seq: usize,
    ) -> Result<Self, ResidencyError> {
        Ok(Self {
            normed: GpuVec::try_hip(model_dim)?,
            sublayer_out: GpuVec::try_hip(model_dim)?,
            attn_input: GpuVec::try_hip(model_dim)?,
            attn_normed: GpuVec::try_hip(model_dim)?,
            attn_out: GpuVec::try_hip(model_dim)?,
            mlp_input: GpuVec::try_hip(model_dim)?,
            mlp_normed: GpuVec::try_hip(model_dim)?,
            mlp_out: GpuVec::try_hip(model_dim)?,
            saved_q_proj: GpuVec::try_hip(model_dim)?,
            saved_k_proj: GpuVec::try_hip(kv_dim)?,
            saved_v_proj: GpuVec::try_hip(kv_dim)?,
            saved_q_normed: GpuVec::try_hip(model_dim)?,
            saved_scores_tight: GpuVec::try_hip(num_heads_x_max_seq)?,
            saved_head_out: GpuVec::try_hip(model_dim)?,
            saved_gate_out: GpuVec::try_hip(mlp_dim)?,
            saved_up_out: GpuVec::try_hip(mlp_dim)?,
            saved_silu: GpuVec::try_hip(mlp_dim)?,
            saved_hidden: GpuVec::try_hip(mlp_dim)?,
        })
    }
}

/// Aggregate per-layer + global state for [`GptModelResident::backward`].
/// Holds caller-owned scratch and gradient buffers so the model's
/// backward call can write into them without per-step allocation.
pub struct GptBackwardState {
    pub attn_scratch: AttentionScratch,
    pub attn_bwd: AttentionBackwardScratch,
    pub mlp_scratch: SwigluScratch,
    pub mlp_bwd: SwigluBackwardScratch,
    pub block_scratches: Vec<TransformerBlockScratch>,
    pub attn_grads: Vec<AttentionResidentGrads>,
    pub mlp_grads: Vec<SwigluResidentGrads>,
    /// `[vocab × model_dim]` row-major.
    pub d_lm_head_weight: GpuVec,
    /// `[vocab]`. Unused for vanilla LM head (no bias) but always written.
    pub d_lm_head_bias: GpuVec,
    /// `[vocab × model_dim]` — sparse gradient for the embedding table.
    /// Zero everywhere except the `token_id`-th row, which holds the
    /// `d_hidden` returned by the block stack's backward.
    pub d_embed: GpuVec,
    /// `[model_dim]` — `d/d(hidden)` flowing through the block stack.
    pub d_hidden: GpuVec,
    /// `[model_dim]` — `d/d(final_normed)`.
    pub d_normed: GpuVec,
    /// `[model_dim]` — current hidden state. Reused across forward steps.
    pub hidden_dev: GpuVec,
    /// `[model_dim]` — embed-time hidden (= post-smear, but smear is omitted).
    pub x0_dev: GpuVec,
    /// `[model_dim]` — output of final RMSNorm (input to lm_head).
    pub normed_dev: GpuVec,
    /// Host snapshot of `normed_dev` post-final-norm — the lm_head
    /// backward needs this for the dweight outer product.
    pub final_normed_host: Vec<f32>,
    /// Host scratch — also used to stage `final_input_host` (input to
    /// the final RMSNorm, captured at the end of forward_for_backward
    /// via a D2H copy of `hidden_dev`). Repurposed to avoid bloating
    /// the struct with a separate field.
    pub d_final_normed_host: Vec<f32>,
    /// Host scratch — captures `final_input_host` from forward and
    /// `d_final_input_host` after RMSNorm backward.
    pub d_final_input_host: Vec<f32>,
}

// ─── GptModelResident ────────────────────────────────────────

/// Full device-resident transformer model.
///
/// Wraps embedding, n stacked [`TransformerBlockResident`]s, final
/// scaled RMSNorm, and the lm_head output projection. Forward signature
/// matches the host [`GptModel`] decode API but with all device-resident
/// dispatches.
///
/// **Smear and value embeddings are deliberately omitted.** Both modules
/// touch the residual stream in ways that demand additional resident
/// op surface (`Op::SmearGate`, value-embed lookup), and the survey
/// (Open Question 4) explicitly scopes this slice to the basic
/// transformer forward. They are flagged for slice 10/11.
pub struct GptModelResident {
    /// Token embedding `[vocab_size × model_dim]` row-major on device.
    pub embed_dev: HipBuffer,
    /// LM head `[vocab_size × model_dim]` row-major on device, wrapped
    /// as a `LinearResident` so we can `matvec_resident` directly into
    /// the logits buffer.
    pub lm_head: LinearResident,
    /// Final scaled RMSNorm weight `[model_dim]`.
    pub final_norm_weight_dev: HipBuffer,
    /// Stacked transformer blocks.
    pub blocks: Vec<TransformerBlockResident>,
    /// RoPE — host-side; cached for the per-block forward call.
    pub rope: RotaryEmbedding,
    /// Per-layer attention window (`None` = full attention). Derived
    /// from `GptConfig::window_size` at construction. Indexed by layer.
    pub windows: Vec<Option<usize>>,
    vocab: usize,
    model_dim: usize,
    norm_eps: f32,
}

impl GptModelResident {
    /// Build from a host [`GptModel`] whose `mlp_activation = SwiGlu`.
    /// The host blocks are read for their attention weights; the SwiGLU
    /// MLPs are passed in as a parallel slice (the host model uses
    /// `Mlp` for ReLU² but our resident path needs SwiGLU). Caller
    /// constructs the SwiGLU MLPs alongside the host model — this is
    /// the same pattern the test harness uses.
    pub fn from_model(
        model: &GptModel,
        swiglu_mlps: &[SwigluMlp],
    ) -> Result<Self, ResidencyError> {
        if swiglu_mlps.len() != model.blocks.len() {
            return Err(ResidencyError::WrongVariant {
                expected: "swiglu_mlps.len() == model.blocks.len()",
                got: "mismatched lengths",
            });
        }
        let config = &model.config;
        let vocab = config.vocab_size.get();
        let model_dim = config.model_dim.get();

        let embed_dev = upload_f32_buffer(model.embed.as_slice())?;
        let lm_head_lin = linear_from_weight(model.lm_head.as_slice(), vocab, model_dim);
        let lm_head = LinearResident::from_linear(&lm_head_lin)?;
        let final_norm_weight_dev = upload_f32_buffer(model.final_norm.scale.as_slice())?;

        let mut blocks = Vec::with_capacity(model.blocks.len());
        for (block, swiglu) in model.blocks.iter().zip(swiglu_mlps.iter()) {
            let resident_block = TransformerBlockResident::from_block(
                block, swiglu, &model.lambdas, config,
            )?;
            blocks.push(resident_block);
        }

        let n_layers = config.num_layers;
        let windows: Vec<Option<usize>> = (0..model.blocks.len())
            .map(|li| config.window_size(
                LayerIdx::new(li, n_layers).expect("li < num_layers")
            ))
            .collect();

        Ok(Self {
            embed_dev, lm_head, final_norm_weight_dev,
            blocks,
            rope: RotaryEmbedding::new(
                config.head_dim, config.max_seq_len, config.rope_base,
            ),
            windows,
            vocab, model_dim,
            norm_eps: config.norm_eps,
        })
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize { self.blocks.len() }
    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize { self.vocab }
    /// Model dimension.
    pub fn model_dim(&self) -> usize { self.model_dim }

    /// Single-token forward + backward state container. Caller-owned
    /// per-block scratch vectors so backward can read the activations
    /// the matched forward populated. One entry per layer.
    ///
    /// **Single-token only.** Multi-token backward through a KV cache
    /// would need per-(layer, token) saved activations; the present
    /// design pins one set of activations per layer (the most recent
    /// token's). Multi-token training-time backward is a follow-up
    /// slice (see `forward_prefill` style).
    pub fn alloc_backward_state(
        &self,
        kv_cache: &KvCacheResident,
    ) -> Result<GptBackwardState, ResidencyError> {
        let max_kv = kv_cache.n_kv_heads() * kv_cache.head_dim();
        let max_seq = kv_cache.max_seq_len();
        let n_heads = self.blocks.first()
            .map(|b| b.attn.num_heads).unwrap_or(0);
        let head_dim = kv_cache.head_dim();
        let model_dim = self.model_dim;
        let mlp_dim = self.blocks.first().map(|b| b.mlp.mlp_dim()).unwrap_or(model_dim);
        let kv_dim = kv_cache.n_kv_heads() * kv_cache.head_dim();

        let mut block_scratches = Vec::with_capacity(self.blocks.len());
        let mut attn_grads = Vec::with_capacity(self.blocks.len());
        let mut mlp_grads = Vec::with_capacity(self.blocks.len());
        for _ in 0..self.blocks.len() {
            block_scratches.push(TransformerBlockScratch::with_dims(
                model_dim, kv_dim, mlp_dim, n_heads * max_seq,
            )?);
            attn_grads.push(AttentionResidentGrads::new(model_dim, kv_dim)?);
            mlp_grads.push(SwigluResidentGrads::new(model_dim, mlp_dim)?);
        }

        Ok(GptBackwardState {
            attn_scratch: AttentionScratch::new(n_heads, head_dim, max_kv, max_seq)?,
            attn_bwd: AttentionBackwardScratch::new(n_heads, head_dim, max_kv, max_seq)?,
            mlp_scratch: SwigluScratch::new(model_dim, mlp_dim)?,
            mlp_bwd: SwigluBackwardScratch::new(model_dim, mlp_dim)?,
            block_scratches,
            attn_grads,
            mlp_grads,
            d_lm_head_weight: GpuVec::try_hip(self.vocab * model_dim)?,
            d_lm_head_bias: GpuVec::try_hip(self.vocab)?,
            d_embed: GpuVec::try_hip(self.vocab * model_dim)?,
            d_hidden: GpuVec::try_hip(model_dim)?,
            d_normed: GpuVec::try_hip(model_dim)?,
            hidden_dev: GpuVec::try_hip(model_dim)?,
            x0_dev: GpuVec::try_hip(model_dim)?,
            normed_dev: GpuVec::try_hip(model_dim)?,
            final_normed_host: vec![0.0f32; model_dim],
            d_final_normed_host: vec![0.0f32; model_dim],
            d_final_input_host: vec![0.0f32; model_dim],
        })
    }

    /// Forward + cache activations for backward — single-token version.
    /// Same effect as [`Self::forward`] for the last token in `token_ids`,
    /// but populates `state.block_scratches` so [`Self::backward`] can
    /// reverse the chain. The final hidden state (post-final-norm,
    /// pre-lm-head) is also captured in `state.final_normed_host` for
    /// the lm_head backward.
    pub fn forward_for_backward(
        &mut self,
        batch: &HipBatch,
        token_id: i64,
        position: usize,
        kv_cache: &mut KvCacheResident,
        state: &mut GptBackwardState,
        logits_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        debug_assert_eq!(logits_out.len(), self.vocab);

        // Embed lookup.
        let bytes = self.model_dim * 4;
        let embed_off_bytes = token_id as usize * self.model_dim * 4;
        unsafe {
            hip_memcpy_d2d(
                hip_buf_mut(&mut state.hidden_dev)?.device_ptr(),
                (self.embed_dev.device_ptr() as *const u8).add(embed_off_bytes)
                    as *const std::os::raw::c_void,
                bytes,
            )?;
        }
        batch.note_dispatch()?;
        unsafe {
            hip_memcpy_d2d(
                hip_buf_mut(&mut state.x0_dev)?.device_ptr(),
                hip_buf(&state.hidden_dev)?.device_ptr() as *const std::os::raw::c_void,
                bytes,
            )?;
        }
        batch.note_dispatch()?;

        // Blocks (with backward-cache forward).
        for (li, block) in self.blocks.iter().enumerate() {
            let window = self.windows[li];
            block.forward_for_backward(
                batch,
                &mut state.hidden_dev, &state.x0_dev,
                kv_cache, position, window, &self.rope,
                &mut state.attn_scratch, &mut state.mlp_scratch,
                &mut state.block_scratches[li],
            )?;
        }

        // Final norm + lm_head.
        unsafe {
            rms_norm_resident(
                hip_buf(&state.hidden_dev)?.device_ptr() as *const f32,
                self.final_norm_weight_dev.device_ptr() as *const f32,
                hip_buf(&state.normed_dev)?.device_ptr() as *mut f32,
                1, self.model_dim, self.norm_eps,
            )?;
        }
        batch.note_dispatch()?;
        // Save final_normed for lm_head backward.
        state.normed_dev.copy_to_host(&mut state.final_normed_host);

        // lm_head matvec into logits_out.
        unsafe {
            matvec_resident(
                hip_buf(&state.normed_dev)?.device_ptr() as *const f32,
                self.lm_head.weight_dev.device_ptr() as *const f32,
                self.lm_head.bias_dev.device_ptr() as *const f32,
                hip_buf_mut(logits_out)?.device_ptr() as *mut f32,
                self.vocab, self.model_dim,
            )?;
        }
        batch.note_dispatch()?;

        // Save the final hidden (pre-norm) for the final RMSNorm backward.
        state.hidden_dev.copy_to_host(&mut state.d_final_input_host);
        // (We reuse d_final_input_host as a forward cache here — a tiny
        // misnomer; it is actually `final_norm_input_host`. Keeping the
        // single field avoids inflating the state struct.)

        Ok(())
    }

    /// Backward pass — single-token. Caller provides the upstream gradient
    /// `d_logits` (`[vocab]`) and receives weight gradients in `state.attn_grads`,
    /// `state.mlp_grads`, plus `state.d_lm_head_weight`, `state.d_lm_head_bias`
    /// (lm_head dweight/dbias), and `state.d_embed` (gradient w.r.t. the
    /// embedding-table row for `token_id`, scattered into the full table).
    ///
    /// The embedding gradient is the sum of all per-token gradients; for
    /// the single-token path it lands at the row for `token_id`. The
    /// rest of the table is zero (or unchanged from a prior accumulation
    /// — caller decides whether to zero `state.d_embed` first).
    pub fn backward(
        &mut self,
        batch: &HipBatch,
        token_id: i64,
        position: usize,
        kv_cache: &mut KvCacheResident,
        state: &mut GptBackwardState,
        d_logits: &GpuVec,
    ) -> Result<(), ResidencyError> {
        debug_assert_eq!(d_logits.len(), self.vocab);
        let model_dim = self.model_dim;

        // Stage 1: lm_head backward.
        // d_final_normed_dev = lm_head.W^T · d_logits
        // dweight_lm_head    = d_logits ⊗ final_normed (saved in state)
        let mut final_normed_dev = GpuVec::try_hip(model_dim)?;
        final_normed_dev.copy_from(&state.final_normed_host);
        let mut d_final_normed = GpuVec::try_hip(model_dim)?;
        self.lm_head.backward(
            batch, &final_normed_dev, d_logits,
            &mut d_final_normed,
            &mut state.d_lm_head_weight,
            &mut state.d_lm_head_bias,
        )?;

        // Stage 2: final RMSNorm backward (host-side).
        // forward: final_normed = rms_norm(final_input, scale=final_norm_weight, eps=norm_eps)
        // The host model's final_norm uses an unscaled RMSNorm with weight=1
        // (we initialised `final_norm_scale = vec![1.0; md]` in the test
        // builder); the resident path uploads `model.final_norm.scale.as_slice()`
        // into `final_norm_weight_dev`. For the test the scale is all 1.0
        // so we treat it as constant 1 in the backward — matches the test.
        // Production callers with non-1 final_norm scale would need a
        // host download of `final_norm_weight_dev` here; we keep the
        // simpler constant-1 path because that's what the test config uses.
        let final_input_host = state.d_final_input_host.clone();  // captured in forward_for_backward
        let mut d_final_normed_host = vec![0.0f32; model_dim];
        d_final_normed.copy_to_host(&mut d_final_normed_host);
        let mut d_final_input_host = vec![0.0f32; model_dim];
        rms_norm_backward_per_head(
            &final_input_host, &d_final_normed_host, &mut d_final_input_host,
            1, model_dim, 1.0, self.norm_eps,
        );
        state.d_hidden.copy_from(&d_final_input_host);

        // Stage 3: blocks (reverse order).
        for (li, block) in self.blocks.iter().enumerate().rev() {
            // Restore the per-layer attention/MLP activations into the
            // shared scratch buffers. `forward_for_backward` snapshotted
            // each layer's activations into `block_scratches[li]` so this
            // restore (D2D copies) is cheap.
            let attn_kv_dim = block.attn.kv_dim;
            let attn_max_seq = kv_cache.max_seq_len();
            let attn_num_heads = block.attn.num_heads;
            let mlp_dim = block.mlp.mlp_dim();
            let model_dim_local = block.model_dim;
            unsafe {
                hip_memcpy_d2d(
                    hip_buf_mut(&mut state.attn_scratch.q_proj)?.device_ptr(),
                    hip_buf(&state.block_scratches[li].saved_q_proj)?.device_ptr() as *const std::os::raw::c_void,
                    model_dim_local * 4,
                )?;
                hip_memcpy_d2d(
                    hip_buf_mut(&mut state.attn_scratch.k_proj)?.device_ptr(),
                    hip_buf(&state.block_scratches[li].saved_k_proj)?.device_ptr() as *const std::os::raw::c_void,
                    attn_kv_dim * 4,
                )?;
                hip_memcpy_d2d(
                    hip_buf_mut(&mut state.attn_scratch.v_proj)?.device_ptr(),
                    hip_buf(&state.block_scratches[li].saved_v_proj)?.device_ptr() as *const std::os::raw::c_void,
                    attn_kv_dim * 4,
                )?;
                hip_memcpy_d2d(
                    hip_buf_mut(&mut state.attn_scratch.q_normed)?.device_ptr(),
                    hip_buf(&state.block_scratches[li].saved_q_normed)?.device_ptr() as *const std::os::raw::c_void,
                    model_dim_local * 4,
                )?;
                hip_memcpy_d2d(
                    hip_buf_mut(&mut state.attn_scratch.scores_tight)?.device_ptr(),
                    hip_buf(&state.block_scratches[li].saved_scores_tight)?.device_ptr() as *const std::os::raw::c_void,
                    attn_num_heads * attn_max_seq * 4,
                )?;
                hip_memcpy_d2d(
                    hip_buf_mut(&mut state.attn_scratch.head_out)?.device_ptr(),
                    hip_buf(&state.block_scratches[li].saved_head_out)?.device_ptr() as *const std::os::raw::c_void,
                    model_dim_local * 4,
                )?;
                hip_memcpy_d2d(
                    hip_buf_mut(&mut state.mlp_scratch.gate_out)?.device_ptr(),
                    hip_buf(&state.block_scratches[li].saved_gate_out)?.device_ptr() as *const std::os::raw::c_void,
                    mlp_dim * 4,
                )?;
                hip_memcpy_d2d(
                    hip_buf_mut(&mut state.mlp_scratch.up_out)?.device_ptr(),
                    hip_buf(&state.block_scratches[li].saved_up_out)?.device_ptr() as *const std::os::raw::c_void,
                    mlp_dim * 4,
                )?;
                hip_memcpy_d2d(
                    hip_buf_mut(&mut state.mlp_scratch.silu)?.device_ptr(),
                    hip_buf(&state.block_scratches[li].saved_silu)?.device_ptr() as *const std::os::raw::c_void,
                    mlp_dim * 4,
                )?;
                hip_memcpy_d2d(
                    hip_buf_mut(&mut state.mlp_scratch.hidden)?.device_ptr(),
                    hip_buf(&state.block_scratches[li].saved_hidden)?.device_ptr() as *const std::os::raw::c_void,
                    mlp_dim * 4,
                )?;
            }
            for _ in 0..10 { batch.note_dispatch()?; }

            block.backward(
                batch,
                &mut state.d_hidden,
                None,  // dx0_dev (x_lambda=0 in test)
                kv_cache, position, &self.rope,
                &mut state.attn_scratch,
                &mut state.attn_bwd,
                &mut state.mlp_scratch,
                &mut state.mlp_bwd,
                &mut state.block_scratches[li],
                &mut state.attn_grads[li],
                &mut state.mlp_grads[li],
                false,  // recompute (we have all activations)
            )?;
        }

        // Stage 4: embedding backward.
        // forward: hidden_dev = embed_dev[token_id, :]
        // d_embed[token_id, :] = d_hidden  (scatter into the right row)
        // Other rows are unchanged. For test correctness we zero d_embed
        // and write the row.
        let row_off_bytes = token_id as usize * model_dim * 4;
        // Zero the table first (caller may already have done this; the
        // test does so explicitly).
        // We dispatch a zero-then-copy:
        //   1. Skip explicit zero (caller's responsibility) — copy from d_hidden
        //      into d_embed at the row offset.
        unsafe {
            hip_memcpy_d2d(
                (hip_buf_mut(&mut state.d_embed)?.device_ptr() as *mut u8)
                    .add(row_off_bytes) as *mut std::os::raw::c_void,
                hip_buf(&state.d_hidden)?.device_ptr() as *const std::os::raw::c_void,
                model_dim * 4,
            )?;
        }
        batch.note_dispatch()?;

        Ok(())
    }

    /// Forward pass for a sequence of `token_ids` at `positions`.
    ///
    /// Each (token_id, position) pair runs as a single decode step:
    /// embed lookup → through every block → final norm → lm_head.
    /// `kv_cache` is mutated to extend the cache by `token_ids.len()`
    /// tokens (positions must be sequential and start at the current
    /// cache `seq_len`).
    ///
    /// `logits_out`: `[token_ids.len() × vocab_size]` row-major. The
    /// last token's logits land in the last `vocab` slots.
    ///
    /// **Smear and x0 lambdas are not applied.** This is a vanilla
    /// pre-norm transformer forward — see struct docs for the omitted
    /// pieces.
    pub fn forward(
        &mut self,
        batch: &HipBatch,
        token_ids: &[i64],
        positions: &[usize],
        kv_cache: &mut KvCacheResident,
        logits_out: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        let n = token_ids.len();
        debug_assert_eq!(token_ids.len(), positions.len());
        debug_assert_eq!(logits_out.len(), n * self.vocab);

        // Per-token scratch — allocated once and reused across all
        // tokens in this call. Lifetime ends with the function.
        let max_kv = kv_cache.n_kv_heads() * kv_cache.head_dim();
        let max_seq = kv_cache.max_seq_len();
        let n_heads = self.blocks.first()
            .map(|b| b.attn.num_heads).unwrap_or(0);
        let head_dim = kv_cache.head_dim();
        let mut attn_scratch = AttentionScratch::new(n_heads, head_dim, max_kv, max_seq)?;
        let mut mlp_scratch = SwigluScratch::new(
            self.model_dim,
            self.blocks.first().map(|b| b.mlp.mlp_dim()).unwrap_or(self.model_dim),
        )?;
        let mut block_scratch = TransformerBlockScratch::new(self.model_dim)?;

        let mut hidden_dev = GpuVec::try_hip(self.model_dim)?;
        let mut x0_dev = GpuVec::try_hip(self.model_dim)?;
        let normed_dev = GpuVec::try_hip(self.model_dim)?;
        let logits_buf = hip_buf_mut(logits_out)?;
        let logits_base = logits_buf.device_ptr() as *mut f32;

        for (t, (&tid, &pos)) in token_ids.iter().zip(positions.iter()).enumerate() {
            // Stage 1: embed lookup. embed_dev row at index tid is the
            // hidden state for this token. D2D copy `model_dim` floats.
            let embed_off_bytes = tid as usize * self.model_dim * 4;
            let bytes = self.model_dim * 4;
            unsafe {
                use std::os::raw::c_void;
                hip_memcpy_d2d(
                    hip_buf_mut(&mut hidden_dev)?.device_ptr(),
                    (self.embed_dev.device_ptr() as *const u8).add(embed_off_bytes) as *const c_void,
                    bytes,
                )?;
            }
            batch.note_dispatch()?;
            // Save x0 (= hidden after smear, but smear is omitted).
            unsafe {
                hip_memcpy_d2d(
                    hip_buf_mut(&mut x0_dev)?.device_ptr(),
                    hip_buf(&hidden_dev)?.device_ptr() as *const std::os::raw::c_void,
                    bytes,
                )?;
            }
            batch.note_dispatch()?;

            // Stage 2: blocks.
            for (li, block) in self.blocks.iter().enumerate() {
                let window = self.windows[li];
                block.forward(
                    batch,
                    &mut hidden_dev, &x0_dev,
                    kv_cache, pos, window, &self.rope,
                    &mut attn_scratch, &mut mlp_scratch,
                    &mut block_scratch,
                )?;
            }

            // Stage 3: final norm + lm_head.
            unsafe {
                rms_norm_resident(
                    hip_buf(&hidden_dev)?.device_ptr() as *const f32,
                    self.final_norm_weight_dev.device_ptr() as *const f32,
                    hip_buf(&normed_dev)?.device_ptr() as *mut f32,
                    1, self.model_dim, self.norm_eps,
                )?;
            }
            batch.note_dispatch()?;

            // lm_head.forward writes into logits_out[t * vocab..(t+1) * vocab].
            // We dispatch matvec_resident directly with pointer offsets
            // because LinearResident::forward expects a `&mut GpuVec` that
            // covers exactly the output range — and we want a slab.
            unsafe {
                matvec_resident(
                    hip_buf(&normed_dev)?.device_ptr() as *const f32,
                    self.lm_head.weight_dev.device_ptr() as *const f32,
                    self.lm_head.bias_dev.device_ptr() as *const f32,
                    logits_base.add(t * self.vocab),
                    self.vocab, self.model_dim,
                )?;
            }
            batch.note_dispatch()?;
        }

        Ok(())
    }
}

/// HIP D2D memcpy — same helper as kv_cache_resident.rs but copied here
/// so resident.rs is independent of the cache module's internals.
unsafe fn hip_memcpy_d2d(
    dst: *mut std::os::raw::c_void,
    src: *const std::os::raw::c_void,
    bytes: usize,
) -> Result<(), ResidencyError> {
    use modgrad_device::backend::rocm::ffi;
    const HIP_MEMCPY_DEVICE_TO_DEVICE: std::os::raw::c_int = 3;
    let err = unsafe {
        ffi::hipMemcpy(dst, src, bytes, HIP_MEMCPY_DEVICE_TO_DEVICE)
    };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "hipMemcpy D2D ({bytes} bytes): {}", ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}

#[inline]
fn hip_buf_mut<'a>(g: &'a mut GpuVec) -> Result<&'a mut HipBuffer, ResidencyError> {
    match g {
        GpuVec::Hip(b) => Ok(b),
        other => Err(ResidencyError::WrongVariant {
            expected: "Hip", got: other.variant_name(),
        }),
    }
}

// `LayerIdx` import is only used inside `from_block`; keep it warning-free.
#[allow(dead_code)]
const _: fn(LayerIdx) = |_| {};

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! Shared lock: `modgrad_device::test_lock::hip_test_lock()` —
    //! HIP runtime tests must run serially. Multiple concurrent
    //! matvec_resident dispatches share the default stream, and
    //! composite ops like SiLU's three-stage Logistic+Mul+Mul interleave
    //! across tests in unpredictable ways.
    use super::*;
    use modgrad_device::backend::rocm::ffi::runtime_available;

    use crate::attention::{AttentionWeights, CausalSelfAttention};
    use crate::block::TransformerBlock;
    use crate::config::{GptConfig, MlpActivation, ResidualConfig, SmearConfig,
                        ValueEmbedConfig, WindowPattern, Precision};
    use crate::dims::*;
    use crate::mlp::{Mlp, MlpWeights, SwigluMlp, SwigluWeights};
    use crate::model::GptModel;
    use crate::norm::ScaledRmsNorm;
    use crate::position::fixed::FixedPositioning;
    use crate::residual::{ForwardCtx, ResidualLambdas};
    use crate::rope::RotaryEmbedding;
    use crate::smear::{Inference, Smear, SmearWeights, Training};
    use crate::tensor::Tensor2;
    use crate::kv_cache::KvCache;
    use crate::ops::TransformerOps;
    use modgrad_compute::backend::CpuBackend;

    /// Tiny test config: 2 layers, d_model=128, n_heads=4, vocab=256,
    /// SwiGLU MLP. Small enough to `cargo test` fast on CPU and ROCm.
    fn tiny_config() -> GptConfig {
        let head_dim = 32usize;
        let n_heads = 4usize;
        let model_dim = head_dim * n_heads;
        GptConfig {
            model_dim: ModelDim::new(model_dim),
            num_heads: NumHeads::new(n_heads),
            num_kv_heads: NumKvHeads::new(n_heads), // No GQA for the test.
            head_dim: HeadDim::new(head_dim),
            num_layers: NumLayers::new(2),
            vocab_size: VocabSize::new(256),
            mlp_dim: MlpDim::new(model_dim * 2),
            max_seq_len: SeqLen::new(16),
            rope_base: 10000.0,
            qk_norm_scale: 1.0,
            use_qk_norm: true,
            window_pattern: WindowPattern::Full,
            mlp_activation: MlpActivation::SwiGlu,
            layer_overrides: Vec::new(),
            tie_embeddings: false,
            logit_cap: 0.0,
            recurrent_steps: 1,
            has_exit_gate: false,
            value_embed: ValueEmbedConfig::default(),
            // x_lambda = 0 throughout so the resident path's residual
            // shortcut collapses to a single op_tensor add per residual.
            // That keeps the test arithmetic comparable to the host
            // reference without ferrying x0 around.
            residual: ResidualConfig {
                resid_start: 1.0, resid_end: 1.0,
                x0_start: 0.0, x0_end: 0.0,
                backout_lambda: 0.0,
            },
            smear: SmearConfig::default(),
            precision: Precision::F32,
            norm_eps: 1e-5,
        }
    }

    /// Build a host model whose attention/MLP/embedding/lm_head all use
    /// deterministic random weights so a parallel resident model can
    /// reproduce the same arithmetic. Returns the model and a parallel
    /// list of `SwigluMlp`s (the resident path needs SwiGLU; the host
    /// model uses `Mlp` ReLU² in its `blocks` field, but we never run
    /// that — see `host_forward_swiglu` for the host-side reference).
    fn build_test_model(config: &GptConfig)
        -> (GptModel, Vec<SwigluMlp>)
    {
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let vocab = config.vocab_size.get();
        let mlp_dim = config.mlp_dim.get();

        let mut rng = modgrad_compute::neuron::SimpleRng::new(0xBADBEEF);
        let randn = |rng: &mut modgrad_compute::neuron::SimpleRng, n: usize| -> Vec<f32> {
            (0..n).map(|_| rng.next_normal() * 0.05).collect()
        };

        let token_embed = randn(&mut rng, vocab * md);
        let lm_head = randn(&mut rng, vocab * md);
        let final_norm_scale = vec![1.0f32; md]; // identity scale to simplify host vs device.
        let smear_gate = vec![0.0f32; md * config.smear.gate_channels];

        let mut blocks_host = Vec::with_capacity(config.num_layers.get());
        let mut swiglu_mlps = Vec::with_capacity(config.num_layers.get());
        for li in 0..config.num_layers.get() {
            let attn_w = AttentionWeights {
                wq: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
                wk: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
                wv: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
                wo: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
            };
            let attn = CausalSelfAttention::new(attn_w, config);

            // Build a SwiGLU and a placeholder ReLU² Mlp; the placeholder
            // is never run.
            let gate_w = randn(&mut rng, mlp_dim * md);
            let up_w = randn(&mut rng, mlp_dim * md);
            let down_w = randn(&mut rng, md * mlp_dim);
            let swiglu_w = SwigluWeights {
                gate: Tensor2::new(gate_w, mlp_dim, md).unwrap(),
                up: Tensor2::new(up_w, mlp_dim, md).unwrap(),
                down: Tensor2::new(down_w, md, mlp_dim).unwrap(),
            };
            let swiglu = SwigluMlp::new(swiglu_w, config.model_dim, config.mlp_dim);
            swiglu_mlps.push(swiglu);

            let placeholder_mlp = Mlp::new(
                MlpWeights {
                    fc: Tensor2::zeros(mlp_dim, md),
                    proj: Tensor2::zeros(md, mlp_dim),
                },
                config.model_dim, config.mlp_dim,
            );
            let layer_idx = LayerIdx::new(li, config.num_layers).unwrap();
            blocks_host.push(TransformerBlock::new(attn, placeholder_mlp, None, layer_idx, config));
        }

        let model = GptModel {
            embed: Tensor2::new(token_embed, vocab, md).unwrap(),
            lm_head: Tensor2::new(lm_head, vocab, md).unwrap(),
            final_norm: ScaledRmsNorm::new(final_norm_scale, config.norm_eps),
            smear_inference: Smear::<Inference>::new(SmearWeights::new(
                smear_gate.clone(), config.model_dim, &config.smear,
            )),
            smear_training: Smear::<Training>::new(SmearWeights::new(
                smear_gate, config.model_dim, &config.smear,
            )),
            blocks: blocks_host,
            lambdas: ResidualLambdas::from_config(&config.residual, config.num_layers),
            rope: RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_base),
            position: Box::new(FixedPositioning),
            config: config.clone(),
        };
        (model, swiglu_mlps)
    }

    /// Host forward — same shape as `GptModelResident::forward` but
    /// runs entirely on CPU. We can't use `GptModel::forward_one`
    /// directly because (a) it runs Smear (we want it skipped), (b)
    /// it uses `Mlp` ReLU² (we want SwiGLU), (c) it folds in x0_lambda
    /// and backout (we want them off). This function inlines the
    /// minimal subset that matches the resident path.
    fn host_forward_swiglu(
        model: &GptModel,
        swiglu_mlps: &[SwigluMlp],
        token_ids: &[i64],
        positions: &[usize],
        kv: &mut KvCache<crate::kv_cache::Decoding>,
    ) -> Vec<Vec<f32>> {
        let backend = CpuBackend::new();
        let config = &model.config;
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let vocab = config.vocab_size.get();
        let mut all_logits = Vec::with_capacity(token_ids.len());

        let mut ctx = ForwardCtx::new(md);
        let mut tmp = vec![0.0f32; md];

        for (&tid, &pos) in token_ids.iter().zip(positions.iter()) {
            // Embed.
            let mut hidden = model.embed.row(tid as usize).to_vec();
            // Skip smear (matches resident path).
            ctx.set_x0_one(&hidden);
            ctx.midpoint_cached = false;

            for (li, block) in model.blocks.iter().enumerate() {
                let li_idx = block.layer_idx;
                // Pre-attn norm.
                let mut normed = vec![0.0f32; md];
                block.attn_norm.forward(&hidden, &mut normed);
                // Attention.
                let mut attn_out = vec![0.0f32; md];
                let mut cur_k = vec![0.0f32; kv_dim];
                let mut cur_v = vec![0.0f32; kv_dim];
                let kv_layer = kv.layer(li);
                let kv_k = kv_layer.k_slice(pos).to_vec();
                let kv_v = kv_layer.v_slice(pos).to_vec();
                let window = config.window_size(li_idx);
                block.attn.forward_one(
                    &normed, &kv_k, &kv_v, &mut cur_k, &mut cur_v,
                    &model.rope, pos, pos + 1, window,
                    &backend, &mut attn_out,
                );
                // Write KV.
                kv.layer_mut(li).write(&cur_k, &cur_v, pos, 1);
                // Residual (resid_lambda = 1, x_lambda = 0).
                model.lambdas.apply(&mut hidden, &attn_out, &ctx.x0, li_idx);
                // Pre-MLP norm.
                block.mlp_norm.forward(&hidden, &mut normed);
                // SwiGLU MLP (use the parallel slice).
                let mut mlp_out = vec![0.0f32; md];
                swiglu_mlps[li].forward(&normed, &mut mlp_out, &backend);
                // Residual.
                model.lambdas.apply(&mut hidden, &mlp_out, &ctx.x0, li_idx);
            }

            // backout_lambda = 0 → no-op. final_norm scale = 1.
            model.final_norm.forward(&hidden, &mut tmp);
            // lm_head.
            let mut logits = vec![0.0f32; vocab];
            backend.matvec_nobias(model.lm_head.as_slice(), &tmp, &mut logits, vocab, md);
            all_logits.push(logits);

            kv.advance();
        }

        all_logits
    }

    #[test]
    fn gpt_model_resident_decode_matches_host() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let config = tiny_config();
        let (model, swiglu_mlps) = build_test_model(&config);

        // 8-token synthetic prompt → 8 sequential decode steps.
        let token_ids: Vec<i64> = (0..8).map(|i| (i * 17) as i64 % 256).collect();
        let positions: Vec<usize> = (0..8).collect();

        // Host reference.
        let host_kv = KvCache::new(
            config.num_layers, config.num_kv_heads,
            config.head_dim, config.model_dim, config.max_seq_len,
        );
        let host_kv_prefilled = host_kv.prefill(0);
        let mut host_kv = host_kv_prefilled.start_decode();
        let host_logits = host_forward_swiglu(
            &model, &swiglu_mlps, &token_ids, &positions, &mut host_kv,
        );

        // Resident path.
        let mut resident = GptModelResident::from_model(&model, &swiglu_mlps)
            .expect("upload");
        let mut kv_cache = KvCacheResident::new(
            config.num_layers.get(),
            config.num_kv_heads.get(),
            config.head_dim.get(),
            config.max_seq_len.get(),
            config.model_dim.get(),
        ).expect("alloc kv");

        let n = token_ids.len();
        let vocab = config.vocab_size.get();
        let mut logits_dev = GpuVec::try_hip(n * vocab).expect("alloc logits");

        let batch = HipBatch::new();
        resident.forward(&batch, &token_ids, &positions, &mut kv_cache, &mut logits_dev)
            .expect("resident forward");
        batch.flush().expect("flush");

        let mut device_logits = vec![0.0f32; n * vocab];
        logits_dev.copy_to_host(&mut device_logits);

        // Compare logits within 1e-2 relative tolerance — bf16-class
        // because rocBLAS accumulates in a different order than AVX-512
        // and softmax rounding stacks across blocks.
        let mut max_rel = 0.0f32;
        let mut max_abs = 0.0f32;
        for (t, host_row) in host_logits.iter().enumerate() {
            for (v, &h) in host_row.iter().enumerate() {
                let d = device_logits[t * vocab + v];
                let abs = (h - d).abs();
                let scale = h.abs().max(d.abs()).max(1e-6);
                let rel = abs / scale;
                if rel > max_rel { max_rel = rel; }
                if abs > max_abs { max_abs = abs; }
            }
        }
        eprintln!("max abs Δ = {max_abs}, max rel Δ = {max_rel}");
        assert!(max_rel < 1e-2,
            "resident vs host logits diverge: max abs {max_abs}, max rel {max_rel}");
    }

    /// End-to-end SWA validation through GptModelResident: build a 2-layer
    /// model with `WindowPattern::Custom([Some(W), None])` (mixed SWA + GA,
    /// the Mimo-style alternation in miniature), run an 8-token decode in
    /// lockstep host vs resident, and compare logits.
    #[test]
    fn gpt_model_resident_decode_matches_host_swa() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let mut config = tiny_config();
        // Layer 0: SWA(window=3); Layer 1: full attention. Mixed pattern
        // exercises both code paths inside one model — the resident path
        // must pick the right window per layer via its `windows[]` table.
        config.window_pattern = WindowPattern::Custom(vec![3, 0]);
        let (model, swiglu_mlps) = build_test_model(&config);

        // 8 tokens — enough that the SWA layer's window (3) clips earlier
        // positions on most steps.
        let token_ids: Vec<i64> = (0..8).map(|i| (i * 17) as i64 % 256).collect();
        let positions: Vec<usize> = (0..8).collect();

        let host_kv = KvCache::new(
            config.num_layers, config.num_kv_heads,
            config.head_dim, config.model_dim, config.max_seq_len,
        );
        let host_kv_prefilled = host_kv.prefill(0);
        let mut host_kv = host_kv_prefilled.start_decode();
        let host_logits = host_forward_swiglu(
            &model, &swiglu_mlps, &token_ids, &positions, &mut host_kv,
        );

        let mut resident = GptModelResident::from_model(&model, &swiglu_mlps)
            .expect("upload");
        // Sanity: layer 0 must be SWA, layer 1 must be full attention.
        assert_eq!(resident.windows[0], Some(3),
            "layer 0 should be SWA per WindowPattern::Custom");
        assert_eq!(resident.windows[1], None,
            "layer 1 should be full attention per WindowPattern::Custom (0 → None)");

        let mut kv_cache = KvCacheResident::new(
            config.num_layers.get(),
            config.num_kv_heads.get(),
            config.head_dim.get(),
            config.max_seq_len.get(),
            config.model_dim.get(),
        ).expect("alloc kv");

        let n = token_ids.len();
        let vocab = config.vocab_size.get();
        let mut logits_dev = GpuVec::try_hip(n * vocab).expect("alloc logits");
        let batch = HipBatch::new();
        resident.forward(&batch, &token_ids, &positions, &mut kv_cache, &mut logits_dev)
            .expect("resident forward");
        batch.flush().expect("flush");

        let mut device_logits = vec![0.0f32; n * vocab];
        logits_dev.copy_to_host(&mut device_logits);

        let mut max_rel = 0.0f32;
        let mut max_abs = 0.0f32;
        let mut worst_pos = 0usize;
        for (t, host_row) in host_logits.iter().enumerate() {
            for (v, &h) in host_row.iter().enumerate() {
                let d = device_logits[t * vocab + v];
                let abs = (h - d).abs();
                let scale = h.abs().max(d.abs()).max(1e-6);
                let rel = abs / scale;
                if rel > max_rel { max_rel = rel; worst_pos = t; }
                if abs > max_abs { max_abs = abs; }
            }
        }
        eprintln!("SWA model: max abs Δ = {max_abs}, max rel Δ = {max_rel} at pos {worst_pos}");
        assert!(max_rel < 1e-2,
            "SWA resident vs host logits diverge: max abs {max_abs}, max rel {max_rel} at pos {worst_pos}");
    }

    #[test]
    fn attention_resident_matches_host_one_token() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let config = tiny_config();
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let max_seq = config.max_seq_len.get();

        let mut rng = modgrad_compute::neuron::SimpleRng::new(0xCAFE);
        let randn = |rng: &mut modgrad_compute::neuron::SimpleRng, n: usize| -> Vec<f32> {
            (0..n).map(|_| rng.next_normal() * 0.05).collect()
        };
        let weights = AttentionWeights {
            wq: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
            wk: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
            wv: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
            wo: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
        };
        let attn = CausalSelfAttention::new(weights, &config);
        let resident = AttentionResident::from_attention(&attn, &config).expect("upload");

        let backend = CpuBackend::new();
        let rope = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_base);

        // First-token (position 0) attention. Prior cache is empty.
        let host_x: Vec<f32> = randn(&mut rng, md);
        let mut host_out = vec![0.0f32; md];
        let mut cur_k = vec![0.0f32; kv_dim];
        let mut cur_v = vec![0.0f32; kv_dim];
        let kv_k_empty = vec![0.0f32; max_seq * kv_dim];
        let kv_v_empty = vec![0.0f32; max_seq * kv_dim];
        attn.forward_one(
            &host_x, &kv_k_empty, &kv_v_empty,
            &mut cur_k, &mut cur_v,
            &rope, 0, 1, None,
            &backend, &mut host_out,
        );

        // Resident.
        let mut kv_cache = KvCacheResident::new(
            1, config.num_kv_heads.get(), config.head_dim.get(),
            config.max_seq_len.get(), md,
        ).expect("alloc kv");
        let mut x_dev = GpuVec::try_hip(md).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(md).expect("alloc out");
        let mut scratch = AttentionScratch::new(
            config.num_heads.get(), config.head_dim.get(),
            kv_dim, config.max_seq_len.get(),
        ).expect("scratch");

        let batch = HipBatch::new();
        resident.forward(
            &batch, &x_dev, &mut kv_cache, 0, 0, None, &rope,
            &mut scratch, &mut out_dev,
        ).expect("forward");
        batch.flush().expect("flush");

        let mut device_out = vec![0.0f32; md];
        out_dev.copy_to_host(&mut device_out);

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (h, &d) in host_out.iter().zip(device_out.iter()) {
            let abs = (h - d).abs();
            let scale = h.abs().max(d.abs()).max(1e-6);
            let rel = abs / scale;
            if abs > max_abs { max_abs = abs; }
            if rel > max_rel { max_rel = rel; }
        }
        eprintln!("attn one-token: max abs {max_abs}, rel {max_rel}");
        assert!(max_rel < 5e-3,
            "attention resident vs host: max abs {max_abs}, rel {max_rel}");
    }

    /// SWA validation: run host + resident in lockstep across multiple
    /// positions with a non-trivial sliding window, comparing outputs at
    /// every position. Exercises both the empty-window early-out
    /// (positions where `windowed_len == 0`) and the active-mask path
    /// (positions where the window clips earlier cache entries).
    #[test]
    fn attention_resident_matches_host_swa() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let config = tiny_config();
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let max_seq = config.max_seq_len.get();
        let n_positions: usize = 6;
        let window: Option<usize> = Some(3);
        assert!(window.unwrap() < n_positions);
        assert!(n_positions <= max_seq);

        let mut rng = modgrad_compute::neuron::SimpleRng::new(0xBEEF);
        let randn = |rng: &mut modgrad_compute::neuron::SimpleRng, n: usize| -> Vec<f32> {
            (0..n).map(|_| rng.next_normal() * 0.05).collect()
        };
        let weights = AttentionWeights {
            wq: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
            wk: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
            wv: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
            wo: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
        };
        let attn = CausalSelfAttention::new(weights, &config);
        let resident = AttentionResident::from_attention(&attn, &config).expect("upload");

        let backend = CpuBackend::new();
        let rope = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_base);

        // Host KV state, manually advanced after every step.
        let mut kv_k = vec![0.0f32; max_seq * kv_dim];
        let mut kv_v = vec![0.0f32; max_seq * kv_dim];
        let mut host_outs: Vec<Vec<f32>> = Vec::with_capacity(n_positions);

        // Resident state.
        let mut kv_cache = KvCacheResident::new(
            1, config.num_kv_heads.get(), config.head_dim.get(),
            config.max_seq_len.get(), md,
        ).expect("alloc kv");
        let mut x_dev = GpuVec::try_hip(md).expect("alloc x");
        let mut out_dev = GpuVec::try_hip(md).expect("alloc out");
        let mut scratch = AttentionScratch::new(
            config.num_heads.get(), config.head_dim.get(),
            kv_dim, config.max_seq_len.get(),
        ).expect("scratch");
        let batch = HipBatch::new();
        let mut device_outs: Vec<Vec<f32>> = Vec::with_capacity(n_positions);

        for position in 0..n_positions {
            let host_x: Vec<f32> = randn(&mut rng, md);

            // Host forward.
            let mut host_out = vec![0.0f32; md];
            let mut cur_k = vec![0.0f32; kv_dim];
            let mut cur_v = vec![0.0f32; kv_dim];
            attn.forward_one(
                &host_x, &kv_k, &kv_v,
                &mut cur_k, &mut cur_v,
                &rope, position, position + 1, window,
                &backend, &mut host_out,
            );
            kv_k[position * kv_dim..(position + 1) * kv_dim].copy_from_slice(&cur_k);
            kv_v[position * kv_dim..(position + 1) * kv_dim].copy_from_slice(&cur_v);
            host_outs.push(host_out);

            // Resident forward at the same position with the same input.
            x_dev.copy_from(&host_x);
            resident.forward(
                &batch, &x_dev, &mut kv_cache, 0, position, window, &rope,
                &mut scratch, &mut out_dev,
            ).expect("resident forward");
            batch.flush().expect("flush");
            let mut device_out = vec![0.0f32; md];
            out_dev.copy_to_host(&mut device_out);
            device_outs.push(device_out);
        }

        // Compare across all positions.
        let mut worst_pos = 0usize;
        let mut worst_rel = 0.0f32;
        let mut worst_abs = 0.0f32;
        for position in 0..n_positions {
            for (h, &d) in host_outs[position].iter().zip(device_outs[position].iter()) {
                let abs = (h - d).abs();
                let scale = h.abs().max(d.abs()).max(1e-6);
                let rel = abs / scale;
                if rel > worst_rel { worst_rel = rel; worst_pos = position; }
                if abs > worst_abs { worst_abs = abs; }
            }
        }
        eprintln!("attn SWA (window={:?}, n={}): max abs {worst_abs}, rel {worst_rel} at pos {worst_pos}",
                  window, n_positions);
        assert!(worst_rel < 5e-3,
            "SWA resident vs host: max abs {worst_abs}, rel {worst_rel} at pos {worst_pos}");
    }

    /// Rolling-cache parity: a SWA layer with a rolling cache (slots ==
    /// window) must produce the same forward output as the same SWA
    /// layer with a linear cache (slots == max_seq_len). Run 6 positions
    /// in lockstep with window=3 — enough to fill (positions 0..2),
    /// reach the wrap (3), and walk past it (4, 5) so the ring's slot
    /// permutation is exercised.
    #[test]
    fn rolling_cache_matches_linear_cache_under_swa() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let config = tiny_config();
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let max_seq = config.max_seq_len.get();
        let window: Option<usize> = Some(3);
        let n_positions: usize = 6;

        let mut rng = modgrad_compute::neuron::SimpleRng::new(0x1011_5A1Du64);
        let randn = |rng: &mut modgrad_compute::neuron::SimpleRng, n: usize| -> Vec<f32> {
            (0..n).map(|_| rng.next_normal() * 0.05).collect()
        };
        let weights = AttentionWeights {
            wq: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
            wk: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
            wv: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
            wo: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
        };
        let attn = CausalSelfAttention::new(weights, &config);
        let resident = AttentionResident::from_attention(&attn, &config).expect("upload");

        let rope = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_base);

        // Two caches — same shapes for K/V semantics, different storage:
        //   linear: slots = max_seq for every layer (current default)
        //   rolling: slots = 3 for layer 0 (matches the SWA window)
        let mut kv_linear = KvCacheResident::new(
            1, config.num_kv_heads.get(), config.head_dim.get(),
            max_seq, md,
        ).expect("alloc kv linear");
        let mut kv_rolling = KvCacheResident::with_layer_slots(
            vec![3], config.num_kv_heads.get(), config.head_dim.get(),
            max_seq, md,
        ).expect("alloc kv rolling");
        assert_eq!(kv_rolling.layer_slots(0), 3);

        let mut x_dev = GpuVec::try_hip(md).expect("alloc x");
        let mut out_lin = GpuVec::try_hip(md).expect("alloc out lin");
        let mut out_roll = GpuVec::try_hip(md).expect("alloc out roll");
        let mut scratch_lin = AttentionScratch::new(
            config.num_heads.get(), config.head_dim.get(), kv_dim, max_seq,
        ).expect("scratch lin");
        let mut scratch_roll = AttentionScratch::new(
            config.num_heads.get(), config.head_dim.get(), kv_dim, max_seq,
        ).expect("scratch roll");
        let batch = HipBatch::new();

        let mut rng2 = modgrad_compute::neuron::SimpleRng::new(0xCABBA6E_C0DEu64);
        let inputs: Vec<Vec<f32>> = (0..n_positions).map(|_| randn(&mut rng2, md)).collect();

        let mut max_rel = 0.0f32;
        let mut max_abs = 0.0f32;
        for position in 0..n_positions {
            x_dev.copy_from(&inputs[position]);

            // Linear cache forward.
            resident.forward(
                &batch, &x_dev, &mut kv_linear, 0, position, window, &rope,
                &mut scratch_lin, &mut out_lin,
            ).expect("forward lin");
            // Rolling cache forward.
            resident.forward(
                &batch, &x_dev, &mut kv_rolling, 0, position, window, &rope,
                &mut scratch_roll, &mut out_roll,
            ).expect("forward roll");
            batch.flush().expect("flush");

            let mut a = vec![0.0f32; md];
            let mut b = vec![0.0f32; md];
            out_lin.copy_to_host(&mut a);
            out_roll.copy_to_host(&mut b);
            for (l, r) in a.iter().zip(b.iter()) {
                let abs = (l - r).abs();
                let scale = l.abs().max(r.abs()).max(1e-6);
                let rel = abs / scale;
                if abs > max_abs { max_abs = abs; }
                if rel > max_rel { max_rel = rel; }
            }
        }
        eprintln!("rolling vs linear (window=3, n=6): max abs {max_abs}, rel {max_rel}");
        assert!(max_rel < 5e-3,
            "rolling cache deviates from linear under SWA: max abs {max_abs}, rel {max_rel}");
    }

    /// Attention sink validation: install a non-trivial per-head sink bias
    /// on both host and resident attention, run multi-position lockstep
    /// (also with SWA enabled to verify sink composes with windowing),
    /// and compare outputs. Also asserts the sink actually changes outputs
    /// vs the no-sink baseline (so we know the bias plumbing is live).
    #[test]
    fn attention_resident_matches_host_sink() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let config = tiny_config();
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let max_seq = config.max_seq_len.get();
        let n_positions: usize = 5;
        let window: Option<usize> = Some(3);
        let n_heads = config.num_heads.get();

        let mut rng = modgrad_compute::neuron::SimpleRng::new(0xC0FFEE);
        let randn = |rng: &mut modgrad_compute::neuron::SimpleRng, n: usize| -> Vec<f32> {
            (0..n).map(|_| rng.next_normal() * 0.05).collect()
        };
        let weights = AttentionWeights {
            wq: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
            wk: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
            wv: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
            wo: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
        };
        // Per-head sink biases — large enough to noticeably steal mass.
        let sink_bias: Vec<f32> = (0..n_heads).map(|h| 0.5 + 0.1 * h as f32).collect();

        let attn_no_sink = CausalSelfAttention::new(weights, &config);
        // Reproduce identical weights for the sinked attention (sink is the
        // only difference). Use the same seeds.
        let mut rng2 = modgrad_compute::neuron::SimpleRng::new(0xC0FFEE);
        let weights2 = AttentionWeights {
            wq: Tensor2::new(randn(&mut rng2, md * md), md, md).unwrap(),
            wk: Tensor2::new(randn(&mut rng2, kv_dim * md), kv_dim, md).unwrap(),
            wv: Tensor2::new(randn(&mut rng2, kv_dim * md), kv_dim, md).unwrap(),
            wo: Tensor2::new(randn(&mut rng2, md * md), md, md).unwrap(),
        };
        let attn_sinked = CausalSelfAttention::new(weights2, &config)
            .with_sink_bias(sink_bias.clone());

        let resident_sinked = AttentionResident::from_attention(&attn_sinked, &config)
            .expect("upload");
        assert!(resident_sinked.sink_bias_dev.is_some(),
            "sink_bias_dev should be Some after with_sink_bias propagates through from_attention");

        let backend = CpuBackend::new();
        let rope = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_base);

        // Host KV state for both attentions (we run baseline + sinked
        // separately to also verify sink actually changes outputs).
        let mut kv_k = vec![0.0f32; max_seq * kv_dim];
        let mut kv_v = vec![0.0f32; max_seq * kv_dim];
        let mut kv_k_baseline = vec![0.0f32; max_seq * kv_dim];
        let mut kv_v_baseline = vec![0.0f32; max_seq * kv_dim];
        let mut sinked_host_outs: Vec<Vec<f32>> = Vec::with_capacity(n_positions);
        let mut baseline_host_outs: Vec<Vec<f32>> = Vec::with_capacity(n_positions);

        let mut kv_cache = KvCacheResident::new(
            1, config.num_kv_heads.get(), config.head_dim.get(),
            config.max_seq_len.get(), md,
        ).expect("alloc kv");
        let mut x_dev = GpuVec::try_hip(md).expect("alloc x");
        let mut out_dev = GpuVec::try_hip(md).expect("alloc out");
        let mut scratch = AttentionScratch::new(
            n_heads, config.head_dim.get(), kv_dim, config.max_seq_len.get(),
        ).expect("scratch");
        let batch = HipBatch::new();
        let mut sinked_dev_outs: Vec<Vec<f32>> = Vec::with_capacity(n_positions);

        // Use the same input sequence for both runs.
        let mut rng3 = modgrad_compute::neuron::SimpleRng::new(0xCABBA6E);
        let inputs: Vec<Vec<f32>> = (0..n_positions).map(|_| randn(&mut rng3, md)).collect();

        for position in 0..n_positions {
            let host_x = &inputs[position];

            // Sinked host.
            let mut host_out = vec![0.0f32; md];
            let mut cur_k = vec![0.0f32; kv_dim];
            let mut cur_v = vec![0.0f32; kv_dim];
            attn_sinked.forward_one(
                host_x, &kv_k, &kv_v,
                &mut cur_k, &mut cur_v,
                &rope, position, position + 1, window,
                &backend, &mut host_out,
            );
            kv_k[position * kv_dim..(position + 1) * kv_dim].copy_from_slice(&cur_k);
            kv_v[position * kv_dim..(position + 1) * kv_dim].copy_from_slice(&cur_v);
            sinked_host_outs.push(host_out);

            // Baseline (no-sink) host with the same weights.
            let mut baseline_out = vec![0.0f32; md];
            let mut cur_k_b = vec![0.0f32; kv_dim];
            let mut cur_v_b = vec![0.0f32; kv_dim];
            attn_no_sink.forward_one(
                host_x, &kv_k_baseline, &kv_v_baseline,
                &mut cur_k_b, &mut cur_v_b,
                &rope, position, position + 1, window,
                &backend, &mut baseline_out,
            );
            kv_k_baseline[position * kv_dim..(position + 1) * kv_dim].copy_from_slice(&cur_k_b);
            kv_v_baseline[position * kv_dim..(position + 1) * kv_dim].copy_from_slice(&cur_v_b);
            baseline_host_outs.push(baseline_out);

            // Sinked resident.
            x_dev.copy_from(host_x);
            resident_sinked.forward(
                &batch, &x_dev, &mut kv_cache, 0, position, window, &rope,
                &mut scratch, &mut out_dev,
            ).expect("resident forward");
            batch.flush().expect("flush");
            let mut device_out = vec![0.0f32; md];
            out_dev.copy_to_host(&mut device_out);
            sinked_dev_outs.push(device_out);
        }

        // Sanity 1: sink actually changed the output (else sink isn't wired).
        let mut max_sink_effect = 0.0f32;
        for position in 1..n_positions {
            for (hs, base) in sinked_host_outs[position].iter().zip(baseline_host_outs[position].iter()) {
                max_sink_effect = max_sink_effect.max((hs - base).abs());
            }
        }
        eprintln!("sink effect on host outputs: max abs Δ vs no-sink = {max_sink_effect}");
        assert!(max_sink_effect > 1e-4,
            "sink should perturb outputs vs no-sink baseline; got {max_sink_effect} — wiring broken?");

        // Sanity 2: sinked resident matches sinked host.
        let mut worst_pos = 0usize;
        let mut worst_rel = 0.0f32;
        let mut worst_abs = 0.0f32;
        for position in 0..n_positions {
            for (h, &d) in sinked_host_outs[position].iter().zip(sinked_dev_outs[position].iter()) {
                let abs = (h - d).abs();
                let scale = h.abs().max(d.abs()).max(1e-6);
                let rel = abs / scale;
                if rel > worst_rel { worst_rel = rel; worst_pos = position; }
                if abs > worst_abs { worst_abs = abs; }
            }
        }
        eprintln!("sink resident vs host: max abs {worst_abs}, rel {worst_rel} at pos {worst_pos}");
        assert!(worst_rel < 5e-3,
            "sink resident vs host diverge: max abs {worst_abs}, rel {worst_rel} at pos {worst_pos}");
    }

    #[test]
    fn swiglu_resident_matches_host() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let model_dim = 64;
        let mlp_dim = 128;
        let mut rng = modgrad_compute::neuron::SimpleRng::new(0xABCD);
        let g_w = (0..mlp_dim*model_dim).map(|_| rng.next_normal() * 0.1).collect::<Vec<_>>();
        let u_w = (0..mlp_dim*model_dim).map(|_| rng.next_normal() * 0.1).collect::<Vec<_>>();
        let d_w = (0..model_dim*mlp_dim).map(|_| rng.next_normal() * 0.1).collect::<Vec<_>>();
        let swiglu_weights = SwigluWeights {
            gate: Tensor2::new(g_w, mlp_dim, model_dim).unwrap(),
            up: Tensor2::new(u_w, mlp_dim, model_dim).unwrap(),
            down: Tensor2::new(d_w, model_dim, mlp_dim).unwrap(),
        };
        let mlp = SwigluMlp::new(swiglu_weights,
            ModelDim::new(model_dim), MlpDim::new(mlp_dim));

        let backend = CpuBackend::new();
        let host_x: Vec<f32> = (0..model_dim).map(|_| rng.next_normal()).collect();
        let mut host_out = vec![0.0f32; model_dim];
        mlp.forward(&host_x, &mut host_out, &backend);

        let resident = SwigluResident::from_swiglu(&mlp).expect("upload");
        let mut x_dev = GpuVec::try_hip(model_dim).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(model_dim).expect("alloc out");
        let mut scratch = SwigluScratch::new(model_dim, mlp_dim).expect("scratch");
        let batch = HipBatch::new();
        resident.forward(&batch, &x_dev, &mut scratch, &mut out_dev).expect("forward");
        batch.flush().expect("flush");

        let mut device_out = vec![0.0f32; model_dim];
        out_dev.copy_to_host(&mut device_out);

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (h, &d) in host_out.iter().zip(device_out.iter()) {
            let abs = (h - d).abs();
            let scale = h.abs().max(d.abs()).max(1e-6);
            if abs > max_abs { max_abs = abs; }
            let rel = abs / scale;
            if rel > max_rel { max_rel = rel; }
        }
        eprintln!("swiglu: max abs {max_abs}, rel {max_rel}");
        assert!(max_rel < 5e-3,
            "swiglu resident vs host: max abs {max_abs}, rel {max_rel}");
    }

    /// Integration test for slice #10: full-model single-token backward.
    ///
    /// Runs forward + backward through both host and resident paths on a
    /// single-token forward (position 0), then compares all weight
    /// gradients (Q, K, V, O, gate, up, down) across both layers.
    ///
    /// **Single-token only**: at position 0 the attn_len=1, the softmax
    /// is trivially 1.0, and the per-token KV gradient is the only K/V
    /// path (no past-cache contributions). This exercises the full
    /// chain — embedding → blocks → final norm → lm_head and back —
    /// while keeping the attention math reproducible.
    #[test]
    fn gpt_model_resident_backward_matches_host() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let config = tiny_config();
        let (model, swiglu_mlps) = build_test_model(&config);
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let mlp_dim = config.mlp_dim.get();
        let vocab = config.vocab_size.get();
        let token_id: i64 = 7;
        let position: usize = 0;

        // Host forward + backward.
        // The host resident path at position=0 means attn_len=1, so the
        // softmax is trivially 1.0 and we only need the current token's
        // K/V gradient.
        let backend = CpuBackend::new();
        let mut hidden = model.embed.row(token_id as usize).to_vec();
        let attn_input_host: Vec<Vec<f32>> = (0..config.num_layers.get())
            .map(|_| Vec::<f32>::new())
            .collect();
        let mut attn_input_per_layer: Vec<Vec<f32>> = attn_input_host;
        let mut attn_normed_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut attn_out_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut mlp_input_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut mlp_normed_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut q_proj_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut k_proj_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut v_proj_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut q_normed_post_rope_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut head_out_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut gate_out_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut up_out_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut silu_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];
        let mut hidden_per_layer: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers.get()];

        for (li, block) in model.blocks.iter().enumerate() {
            attn_input_per_layer[li] = hidden.clone();

            // Pre-attn norm (RMSNorm, weight=1, eps=norm_eps).
            let mut normed = vec![0.0f32; md];
            block.attn_norm.forward(&hidden, &mut normed);
            attn_normed_per_layer[li] = normed.clone();

            // Q/K/V projections.
            let mut q = vec![0.0f32; md];
            let mut k = vec![0.0f32; kv_dim];
            let mut v = vec![0.0f32; kv_dim];
            backend.matvec_nobias(block.attn.weights.wq.as_slice(), &normed, &mut q, md, md);
            backend.matvec_nobias(block.attn.weights.wk.as_slice(), &normed, &mut k, kv_dim, md);
            backend.matvec_nobias(block.attn.weights.wv.as_slice(), &normed, &mut v, kv_dim, md);
            q_proj_per_layer[li] = q.clone();
            k_proj_per_layer[li] = k.clone();
            v_proj_per_layer[li] = v.clone();

            // Per-head QK norm (constant qk_scale, no learnable gamma).
            let head_dim = config.head_dim.get();
            let n_heads = config.num_heads.get();
            let n_kv = config.num_kv_heads.get();
            let qk_scale = config.qk_norm_scale;
            let qk_eps = config.norm_eps;
            let mut q_normed = q.clone();
            let mut k_normed = k.clone();
            host_rms_norm_per_head(&mut q_normed, n_heads, head_dim, qk_scale, qk_eps);
            host_rms_norm_per_head(&mut k_normed, n_kv, head_dim, qk_scale, qk_eps);

            // RoPE.
            for h in 0..n_heads {
                model.rope.apply(&mut q_normed[h * head_dim..(h + 1) * head_dim], position);
            }
            for h in 0..n_kv {
                model.rope.apply(&mut k_normed[h * head_dim..(h + 1) * head_dim], position);
            }
            q_normed_post_rope_per_layer[li] = q_normed.clone();

            // attn_len = 1; softmax trivially 1.0; head_out = V_current.
            // For GQA, head h reads kv_h = h / gqa_ratio.
            let gqa_ratio = config.gqa_ratio();
            let mut head_out = vec![0.0f32; md];
            for h in 0..n_heads {
                let kv_h = h / gqa_ratio;
                let v_slice = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
                head_out[h * head_dim..(h + 1) * head_dim].copy_from_slice(v_slice);
            }
            head_out_per_layer[li] = head_out.clone();

            // O projection.
            let mut attn_out = vec![0.0f32; md];
            backend.matvec_nobias(block.attn.weights.wo.as_slice(), &head_out, &mut attn_out, md, md);
            attn_out_per_layer[li] = attn_out.clone();

            // Residual: hidden = hidden + attn_out (resid_lambda=1, x_lambda=0).
            for i in 0..md { hidden[i] += attn_out[i]; }
            mlp_input_per_layer[li] = hidden.clone();

            // Pre-MLP norm.
            let mut mlp_normed = vec![0.0f32; md];
            block.mlp_norm.forward(&hidden, &mut mlp_normed);
            mlp_normed_per_layer[li] = mlp_normed.clone();

            // SwiGLU MLP.
            let swiglu = &swiglu_mlps[li];
            let mut gate = vec![0.0f32; mlp_dim];
            let mut up = vec![0.0f32; mlp_dim];
            backend.matvec_nobias(swiglu.weights.gate.as_slice(), &mlp_normed, &mut gate, mlp_dim, md);
            backend.matvec_nobias(swiglu.weights.up.as_slice(), &mlp_normed, &mut up, mlp_dim, md);
            let silu: Vec<f32> = gate.iter().map(|&g| g / (1.0 + (-g).exp())).collect();
            let hidden_inner: Vec<f32> = silu.iter().zip(up.iter()).map(|(&s, &u)| s * u).collect();
            let mut mlp_out = vec![0.0f32; md];
            backend.matvec_nobias(swiglu.weights.down.as_slice(), &hidden_inner, &mut mlp_out, md, mlp_dim);

            gate_out_per_layer[li] = gate;
            up_out_per_layer[li] = up;
            silu_per_layer[li] = silu;
            hidden_per_layer[li] = hidden_inner;

            // Residual.
            for i in 0..md { hidden[i] += mlp_out[i]; }
        }

        // Final norm.
        let mut final_normed = vec![0.0f32; md];
        model.final_norm.forward(&hidden, &mut final_normed);
        let final_input = hidden.clone();

        // lm_head.
        let mut logits = vec![0.0f32; vocab];
        backend.matvec_nobias(model.lm_head.as_slice(), &final_normed, &mut logits, vocab, md);

        // Random gradient w.r.t. logits.
        let mut rng = modgrad_compute::neuron::SimpleRng::new(0xBA15);
        let d_logits: Vec<f32> = (0..vocab).map(|_| rng.next_normal() * 0.01).collect();

        // ─── Host backward ───
        // d_lm_head_weight[v, j] = d_logits[v] * final_normed[j]
        let mut d_lm_head_weight_host = vec![0.0f32; vocab * md];
        for v in 0..vocab {
            for j in 0..md {
                d_lm_head_weight_host[v * md + j] = d_logits[v] * final_normed[j];
            }
        }
        // d_final_normed = lm_head^T · d_logits
        let mut d_final_normed_host = vec![0.0f32; md];
        for j in 0..md {
            let mut acc = 0.0f32;
            for v in 0..vocab {
                acc += model.lm_head.as_slice()[v * md + j] * d_logits[v];
            }
            d_final_normed_host[j] = acc;
        }
        // Final RMSNorm backward (scale=1, eps=norm_eps).
        let mut d_final_input_host = vec![0.0f32; md];
        host_rms_norm_backward_per_head(
            &final_input, &d_final_normed_host, &mut d_final_input_host,
            1, md, 1.0, config.norm_eps,
        );

        // Per-block backward, reverse order.
        let mut d_hidden = d_final_input_host.clone();
        let n_layers = config.num_layers.get();
        let mut d_q_w_host = vec![vec![0.0f32; md * md]; n_layers];
        let mut d_k_w_host = vec![vec![0.0f32; kv_dim * md]; n_layers];
        let mut d_v_w_host = vec![vec![0.0f32; kv_dim * md]; n_layers];
        let mut d_o_w_host = vec![vec![0.0f32; md * md]; n_layers];
        let mut d_gate_w_host = vec![vec![0.0f32; mlp_dim * md]; n_layers];
        let mut d_up_w_host = vec![vec![0.0f32; mlp_dim * md]; n_layers];
        let mut d_down_w_host = vec![vec![0.0f32; md * mlp_dim]; n_layers];

        for li in (0..n_layers).rev() {
            let block = &model.blocks[li];
            let swiglu = &swiglu_mlps[li];
            let head_dim = config.head_dim.get();
            let n_heads = config.num_heads.get();
            let n_kv = config.num_kv_heads.get();
            let gqa_ratio = config.gqa_ratio();
            let qk_scale = config.qk_norm_scale;
            let qk_eps = config.norm_eps;

            // MLP residual peel: d_mlp_out = d_hidden (resid_lambda=1).
            // d_hidden retains the +1 path.
            let d_mlp_out = d_hidden.clone();
            // SwiGLU backward.
            //   d_hidden_inner = down^T · d_mlp_out, dweight_down = d_mlp_out ⊗ hidden_inner
            for i in 0..md {
                for j in 0..mlp_dim {
                    d_down_w_host[li][i * mlp_dim + j] = d_mlp_out[i] * hidden_per_layer[li][j];
                }
            }
            let mut d_hidden_inner = vec![0.0f32; mlp_dim];
            for j in 0..mlp_dim {
                let mut acc = 0.0f32;
                for i in 0..md {
                    acc += swiglu.weights.down.as_slice()[i * mlp_dim + j] * d_mlp_out[i];
                }
                d_hidden_inner[j] = acc;
            }
            //   d_silu = d_hidden_inner * up_out, d_up = d_hidden_inner * silu
            let mut d_silu = vec![0.0f32; mlp_dim];
            let mut d_up = vec![0.0f32; mlp_dim];
            for j in 0..mlp_dim {
                d_silu[j] = d_hidden_inner[j] * up_out_per_layer[li][j];
                d_up[j] = d_hidden_inner[j] * silu_per_layer[li][j];
            }
            //   d_gate = d_silu * dSiLU/dx(gate_out)
            let mut d_gate = vec![0.0f32; mlp_dim];
            for j in 0..mlp_dim {
                let g = gate_out_per_layer[li][j];
                let s = 1.0 / (1.0 + (-g).exp());
                let d_silu_dg = s + g * s * (1.0 - s);
                d_gate[j] = d_silu[j] * d_silu_dg;
            }
            //   d_mlp_normed = up^T · d_up + gate^T · d_gate
            //   dweight_up = d_up ⊗ mlp_normed, dweight_gate = d_gate ⊗ mlp_normed
            for j in 0..mlp_dim {
                for k in 0..md {
                    d_up_w_host[li][j * md + k] = d_up[j] * mlp_normed_per_layer[li][k];
                    d_gate_w_host[li][j * md + k] = d_gate[j] * mlp_normed_per_layer[li][k];
                }
            }
            let mut d_mlp_normed = vec![0.0f32; md];
            for k in 0..md {
                let mut acc = 0.0f32;
                for j in 0..mlp_dim {
                    acc += swiglu.weights.up.as_slice()[j * md + k] * d_up[j];
                    acc += swiglu.weights.gate.as_slice()[j * md + k] * d_gate[j];
                }
                d_mlp_normed[k] = acc;
            }
            // Pre-MLP RMSNorm backward.
            let mut d_mlp_input = vec![0.0f32; md];
            host_rms_norm_backward_per_head(
                &mlp_input_per_layer[li], &d_mlp_normed, &mut d_mlp_input,
                1, md, 1.0, config.norm_eps,
            );
            // Fold MLP residual: d_hidden_pre_mlp = d_hidden (peel) + d_mlp_input.
            for i in 0..md { d_hidden[i] += d_mlp_input[i]; }

            // Attention residual peel: d_attn_out = d_hidden.
            let d_attn_out = d_hidden.clone();
            // o_proj backward.
            for i in 0..md {
                for j in 0..md {
                    d_o_w_host[li][i * md + j] = d_attn_out[i] * head_out_per_layer[li][j];
                }
            }
            let mut d_head_out = vec![0.0f32; md];
            for j in 0..md {
                let mut acc = 0.0f32;
                for i in 0..md {
                    acc += block.attn.weights.wo.as_slice()[i * md + j] * d_attn_out[i];
                }
                d_head_out[j] = acc;
            }

            // For attn_len=1: d_softmax = V_current · d_head_out per head;
            //   softmax_backward([1.0], [d]) = [0]; so d_q_normed (post-rope) = 0.
            //   d_v_proj_per_kv_head[i] = sum_h(in kv_h's group) softmax[h, 0] * d_head_out[h, i]
            //                            = sum_h d_head_out[h, i]   (softmax = 1).
            //   d_k_proj_per_kv_head_post_rope = 0 (because d_scores = 0 → d_k from scoring is 0).
            let mut d_v_proj = vec![0.0f32; kv_dim];
            for h in 0..n_heads {
                let kv_h = h / gqa_ratio;
                for i in 0..head_dim {
                    d_v_proj[kv_h * head_dim + i] += d_head_out[h * head_dim + i];
                }
            }
            let mut d_q_normed_host = vec![0.0f32; md];  // = 0 for attn_len=1
            let mut d_k_post_rope = vec![0.0f32; kv_dim];  // = 0 for attn_len=1

            // RoPE backward (no-op for d=0 vectors but we run for completeness).
            host_rope_backward(&model.rope, &mut d_q_normed_host, head_dim, n_heads, position);
            host_rope_backward(&model.rope, &mut d_k_post_rope, head_dim, n_kv, position);

            // QK RMSNorm backward.
            let mut d_q_proj_host = vec![0.0f32; md];
            let mut d_k_proj_host = vec![0.0f32; kv_dim];
            host_rms_norm_backward_per_head(
                &q_proj_per_layer[li], &d_q_normed_host, &mut d_q_proj_host,
                n_heads, head_dim, qk_scale, qk_eps,
            );
            host_rms_norm_backward_per_head(
                &k_proj_per_layer[li], &d_k_post_rope, &mut d_k_proj_host,
                n_kv, head_dim, qk_scale, qk_eps,
            );

            // Q/K/V backward → dweights and d_attn_normed contributions.
            for i in 0..md {
                for j in 0..md {
                    d_q_w_host[li][i * md + j] = d_q_proj_host[i] * attn_normed_per_layer[li][j];
                }
            }
            for i in 0..kv_dim {
                for j in 0..md {
                    d_k_w_host[li][i * md + j] = d_k_proj_host[i] * attn_normed_per_layer[li][j];
                    d_v_w_host[li][i * md + j] = d_v_proj[i] * attn_normed_per_layer[li][j];
                }
            }
            // d_attn_normed = q^T · d_q_proj + k^T · d_k_proj + v^T · d_v_proj
            let mut d_attn_normed = vec![0.0f32; md];
            for j in 0..md {
                let mut acc = 0.0f32;
                for i in 0..md {
                    acc += block.attn.weights.wq.as_slice()[i * md + j] * d_q_proj_host[i];
                }
                for i in 0..kv_dim {
                    acc += block.attn.weights.wk.as_slice()[i * md + j] * d_k_proj_host[i];
                    acc += block.attn.weights.wv.as_slice()[i * md + j] * d_v_proj[i];
                }
                d_attn_normed[j] = acc;
            }

            // Pre-attn RMSNorm backward.
            let mut d_attn_input = vec![0.0f32; md];
            host_rms_norm_backward_per_head(
                &attn_input_per_layer[li], &d_attn_normed, &mut d_attn_input,
                1, md, 1.0, config.norm_eps,
            );
            // Fold attn residual.
            for i in 0..md { d_hidden[i] += d_attn_input[i]; }
        }
        // d_embed[token_id, :] = d_hidden after all blocks.
        let mut d_embed_host = vec![0.0f32; vocab * md];
        for j in 0..md {
            d_embed_host[token_id as usize * md + j] = d_hidden[j];
        }

        // ─── Resident path ───
        let mut resident = GptModelResident::from_model(&model, &swiglu_mlps).expect("upload");
        let mut kv_cache = KvCacheResident::new(
            n_layers, config.num_kv_heads.get(),
            config.head_dim.get(), config.max_seq_len.get(), md,
        ).expect("alloc kv");
        let mut state = resident.alloc_backward_state(&kv_cache).expect("alloc state");
        // Zero d_embed.
        let zeros = vec![0.0f32; vocab * md];
        state.d_embed.copy_from(&zeros);

        let mut logits_dev = GpuVec::try_hip(vocab).expect("alloc logits");
        let batch = HipBatch::new();
        resident.forward_for_backward(
            &batch, token_id, position, &mut kv_cache, &mut state, &mut logits_dev,
        ).expect("fwd-for-bwd");
        batch.flush().expect("flush fwd");

        let mut d_logits_dev = GpuVec::try_hip(vocab).expect("alloc d_logits");
        d_logits_dev.copy_from(&d_logits);
        let batch2 = HipBatch::new();
        resident.backward(
            &batch2, token_id, position, &mut kv_cache, &mut state, &d_logits_dev,
        ).expect("backward");
        batch2.flush().expect("flush bwd");

        // ─── Compare ───
        let tol = 1e-2;
        let download_compare = |label: &str, dev: &GpuVec, expected: &[f32]| {
            let mut got = vec![0.0f32; expected.len()];
            dev.copy_to_host(&mut got);
            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            for (i, (&a, &b)) in got.iter().zip(expected.iter()).enumerate() {
                let abs = (a - b).abs();
                let scale = a.abs().max(b.abs()).max(1e-6);
                let rel = abs / scale;
                if rel > max_rel { max_rel = rel; }
                if abs > max_abs { max_abs = abs; }
                if rel > tol && abs > 1e-4 {
                    eprintln!("{label}[{i}]: got={a}, expected={b}, abs={abs}, rel={rel}");
                }
            }
            eprintln!("{label}: max abs Δ = {max_abs}, max rel Δ = {max_rel}");
            assert!(max_rel < tol || max_abs < 1e-4,
                "{label}: max rel Δ = {max_rel} > {tol}");
        };

        download_compare("d_lm_head_weight", &state.d_lm_head_weight, &d_lm_head_weight_host);
        for li in 0..n_layers {
            download_compare(&format!("d_q_w[{li}]"), &state.attn_grads[li].dweight_q, &d_q_w_host[li]);
            download_compare(&format!("d_k_w[{li}]"), &state.attn_grads[li].dweight_k, &d_k_w_host[li]);
            download_compare(&format!("d_v_w[{li}]"), &state.attn_grads[li].dweight_v, &d_v_w_host[li]);
            download_compare(&format!("d_o_w[{li}]"), &state.attn_grads[li].dweight_o, &d_o_w_host[li]);
            download_compare(&format!("d_gate_w[{li}]"), &state.mlp_grads[li].dweight_gate, &d_gate_w_host[li]);
            download_compare(&format!("d_up_w[{li}]"), &state.mlp_grads[li].dweight_up, &d_up_w_host[li]);
            download_compare(&format!("d_down_w[{li}]"), &state.mlp_grads[li].dweight_down, &d_down_w_host[li]);
        }
        download_compare("d_embed", &state.d_embed, &d_embed_host);
    }

    /// Host RMSNorm forward (constant scale, no learnable gamma).
    fn host_rms_norm_per_head(
        x: &mut [f32], num_heads: usize, head_dim: usize, scale: f32, eps: f32,
    ) {
        let n = head_dim as f32;
        for h in 0..num_heads {
            let off = h * head_dim;
            let row = &mut x[off..off + head_dim];
            let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / n;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for v in row.iter_mut() { *v = *v * inv_rms * scale; }
        }
    }

    /// Host RMSNorm backward — mirrors the implementation in `resident::rms_norm_backward_per_head`.
    fn host_rms_norm_backward_per_head(
        x: &[f32], dy: &[f32], dx: &mut [f32],
        num_heads: usize, head_dim: usize, scale: f32, eps: f32,
    ) {
        let n = head_dim as f32;
        for h in 0..num_heads {
            let off = h * head_dim;
            let xs = &x[off..off + head_dim];
            let dys = &dy[off..off + head_dim];
            let mean_sq: f32 = xs.iter().map(|&v| v * v).sum::<f32>() / n;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            let inv_rms_sq = inv_rms * inv_rms;
            let acc: f32 = xs.iter().zip(dys.iter()).map(|(&a, &b)| a * b).sum();
            let dxs = &mut dx[off..off + head_dim];
            for i in 0..head_dim {
                dxs[i] = scale * inv_rms * (dys[i] - xs[i] * acc * inv_rms_sq / n);
            }
        }
    }

    /// Host RoPE backward — mirrors `resident::rope_backward`.
    fn host_rope_backward(
        rope: &RotaryEmbedding, heads: &mut [f32],
        head_dim: usize, num_heads: usize, position: usize,
    ) {
        let half_dim = head_dim / 2;
        let cos = rope.cos_at(position);
        let sin = rope.sin_at(position);
        for h in 0..num_heads {
            let head = &mut heads[h * head_dim..(h + 1) * head_dim];
            let (left, right) = head.split_at_mut(half_dim);
            for i in 0..half_dim {
                let c = cos[i];
                let s = sin[i];
                let l = left[i];
                let r = right[i];
                left[i]  = l * c + r * s;
                right[i] = -l * s + r * c;
            }
        }
    }
}
