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
    activation_resident, matmul_resident_tn, matvec_resident,
    op_tensor_resident, rms_norm_resident, softmax_resident,
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
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_ratio: usize,
    model_dim: usize,
    kv_dim: usize,
    /// QK scale baked into `qk_norm_weight_dev`. Kept as a struct field
    /// for diagnostics (e.g. tests can sanity-check the value); the
    /// resident dispatch reads it through the device buffer.
    #[allow(dead_code)]
    qk_scale: f32,
    norm_eps: f32,
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

        Ok(Self {
            q_proj: LinearResident::from_linear(&q)?,
            k_proj: LinearResident::from_linear(&k)?,
            v_proj: LinearResident::from_linear(&v)?,
            o_proj: LinearResident::from_linear(&o)?,
            qk_norm_weight_dev,
            zero_bias_dev,
            num_heads: config.num_heads.get(),
            num_kv_heads: config.num_kv_heads.get(),
            head_dim,
            gqa_ratio: config.gqa_ratio(),
            model_dim,
            kv_dim,
            qk_scale,
            norm_eps: config.norm_eps,
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
        // `rms_norm_resident` whose `weight` baked in `qk_scale`).
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

        // Stage 3: D2H Q-normed and K-normed, apply RoPE host-side,
        // re-upload. ~0.5-2 µs round-trip per host_dim×n_heads. RoPE
        // residency is a separate slice.
        scratch.q_normed.copy_to_host(&mut scratch.q_normed_host);
        scratch.k_normed.copy_to_host(&mut scratch.k_normed_host);
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
        // K slab for kv_h is `[max_seq_len × head_dim]` row-major;
        // we use the first `attn_len = position + 1` rows.
        // scores[h, t] = K[t, :] · q_normed[h, :]
        // → matvec_resident(weight=K_slab[:attn_len], x=q_normed[h], bias=0,
        //                   out=scores[h], out_dim=attn_len, in_dim=head_dim).
        let attn_len = position + 1;
        let q_normed_base = hip_buf(&scratch.q_normed)?.device_ptr() as *const f32;
        let scores_base = hip_buf(&scratch.scores)?.device_ptr() as *mut f32;
        let max_seq = kv_cache.max_seq_len();

        for h in 0..self.num_heads {
            let kv_h = h / self.gqa_ratio;
            let k_slab = kv_cache.k_slab_ptr(layer, kv_h);
            unsafe {
                matvec_resident(
                    q_normed_base.add(h * self.head_dim),
                    k_slab,                                 // [max_seq × head_dim]
                    self.zero_bias_dev.device_ptr() as *const f32,
                    scores_base.add(h * max_seq),
                    attn_len,
                    self.head_dim,
                )?;
            }
            batch.note_dispatch()?;
        }

        // Stage 6: batched softmax — `n_rows = num_heads`, but each row is
        // `attn_len` wide while the underlying buffer is striped at
        // `max_seq`. To keep the row layout contiguous we softmax in a
        // dense `[num_heads × attn_len]` scratch then copy back. For the
        // test config (max_seq=16, attn_len ≤ 16) the gather is one
        // D2D copy per row — cheap. The performance-tuned version
        // would softmax in place over a tighter scratch by laying out
        // scores `[num_heads × attn_len]` from the matvec stage onward,
        // but that requires per-call recomputation of stride; we keep
        // it readable for slice 9.
        // … Actually we can just softmax across `max_seq` columns and
        // mask attn_len..max_seq. But unwritten K rows past attn_len
        // were zero-init'd so their dot products are 0, which softmax
        // won't ignore — they'd dilute the distribution. Cleanest fix:
        // run softmax over `attn_len` only, reading from the strided
        // buffer. We use a tight `[num_heads × attn_len]` scratch.
        let scores_tight = &scratch.scores_tight;
        let scores_tight_buf = hip_buf(scores_tight)?;
        for h in 0..self.num_heads {
            unsafe {
                use std::os::raw::c_void;
                let src = (scores_base as *const u8)
                    .add(h * max_seq * 4) as *const c_void;
                let dst = (scores_tight_buf.device_ptr() as *mut u8)
                    .add(h * attn_len * 4) as *mut c_void;
                hip_memcpy_d2d(dst, src, attn_len * 4)?;
            }
            batch.note_dispatch()?;
        }
        unsafe {
            softmax_resident(
                scores_tight_buf.device_ptr() as *const f32,
                scores_tight_buf.device_ptr() as *mut f32,
                self.num_heads, attn_len, false,
            )?;
        }
        batch.note_dispatch()?;

        // Stage 7: weighted V sum per head.
        //   head_out[h, :] = softmax[h, :attn_len] @ V_slab_kv_h[:attn_len, :]
        // Using `matmul_resident_tn`: A is `V_slab[:attn_len × head_dim]`
        // (read transposed → `[head_dim × attn_len]`), B is
        // `softmax[h, :attn_len]` viewed as `[attn_len × 1]`, output is
        // `[head_dim × 1]`. m=head_dim, k=attn_len, n=1.
        let head_out_base = hip_buf(&scratch.head_out)?.device_ptr() as *mut f32;
        let scores_tight_base = scores_tight_buf.device_ptr() as *const f32;
        for h in 0..self.num_heads {
            let kv_h = h / self.gqa_ratio;
            let v_slab = kv_cache.v_slab_ptr(layer, kv_h);
            unsafe {
                matmul_resident_tn(
                    v_slab,                                 // [attn_len × head_dim] (transposed read)
                    scores_tight_base.add(h * attn_len),    // [attn_len × 1]
                    head_out_base.add(h * self.head_dim),
                    self.head_dim, attn_len, 1,
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
            scores_tight: GpuVec::try_hip(num_heads * max_seq)?,
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
            self.layer_idx, position, rope,
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
pub struct TransformerBlockScratch {
    /// Pre-norm output: `[model_dim]`.
    pub normed: GpuVec,
    /// Sub-layer output (attention or MLP): `[model_dim]`.
    pub sublayer_out: GpuVec,
}

impl TransformerBlockScratch {
    pub fn new(model_dim: usize) -> Result<Self, ResidencyError> {
        Ok(Self {
            normed: GpuVec::try_hip(model_dim)?,
            sublayer_out: GpuVec::try_hip(model_dim)?,
        })
    }
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

        Ok(Self {
            embed_dev, lm_head, final_norm_weight_dev,
            blocks,
            rope: RotaryEmbedding::new(
                config.head_dim, config.max_seq_len, config.rope_base,
            ),
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
            for block in &self.blocks {
                block.forward(
                    batch,
                    &mut hidden_dev, &x0_dev,
                    kv_cache, pos, &self.rope,
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
    use super::*;
    use modgrad_device::backend::rocm::ffi::runtime_available;
    use std::sync::Mutex;

    /// HIP runtime tests must run serially — see the matching note in
    /// `modgrad-ffn/src/resident.rs` for the rationale (multiple
    /// concurrent matvec_resident dispatches share the default stream,
    /// composite ops like SiLU's three-stage Logistic+Mul+Mul interleave
    /// across tests in unpredictable ways).
    static HIP_TEST_LOCK: Mutex<()> = Mutex::new(());

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
                block.attn.forward_one(
                    &normed, &kv_k, &kv_v, &mut cur_k, &mut cur_v,
                    &model.rope, pos, pos + 1, None,
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
        let _guard = HIP_TEST_LOCK.lock().unwrap();
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

    #[test]
    fn attention_resident_matches_host_one_token() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
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
            &batch, &x_dev, &mut kv_cache, 0, 0, &rope,
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

    #[test]
    fn swiglu_resident_matches_host() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
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
}
