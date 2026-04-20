//! Op enum — the finite set of operations every `Backend` must understand
//! (or explicitly decline via `supports()`).
//!
//! Granularity rule: one variant per *logical* op. Size-tuned kernel
//! variants (`matmul_blocked` vs `matmul_small`) are dispatched *inside*
//! the backend based on tensor shape, not exposed as separate variants.
//! Quantization is an attribute on `Matvec`, not a separate op.
//!
//! Design constraints:
//! - All input slices are `&[f32]`; output/accumulator slices are `&mut [f32]`.
//!   Shape parameters are `usize` — backends cast if their FFI wants `u32/i32`.
//! - No owned buffers inside variants — avoids hidden allocation and
//!   keeps dispatch a pure routing decision.
//! - Forward and backward are separate variants so backends can opt into
//!   just the forward path for inference-only builds.
//!
//! This enum is the stable boundary between modgrad-ctm / modgrad-ffn
//! callers and the `Backend` implementations. Adding a new logical op
//! requires a version bump and coordinated backend updates.

/// Quantization scheme for a matvec weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    /// Dense f32.
    F32,
    /// Q4_K_M quantized (GGUF 4.5 bpw group-quantized).
    /// Weight slice is opaque bytes; only KFD backend decodes today.
    Q4K,
}

/// Arguments to [`Op::AdamW`]. Pulled out of the enum variant so callers
/// get labeled-field ergonomics and future additions (new moment, new
/// hyperparameter) don't break every match arm — struct fields are
/// additive, variant fields are not.
///
/// Semantics match the original `AdamW` variant: `m`/`v` are the
/// optimizer moments (mutated in place); bias-correction terms are
/// pre-computed by the caller as `bc{1,2}_inv = 1/(1-beta^t)` so they
/// can be reused across many parameter groups per step.
#[derive(Debug)]
pub struct AdamWArgs<'a> {
    pub w: &'a mut [f32],
    pub g: &'a [f32],
    pub m: &'a mut [f32],
    pub v: &'a mut [f32],
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub bc1_inv: f32,
    pub bc2_inv: f32,
}

/// Arguments to [`Op::SyncBackwardScatter`]. Pulled out of the enum
/// variant for the same reason as [`AdamWArgs`] — 14 positional fields
/// at every call site was the actual pain point.
///
/// Semantics match the original `SyncBackwardScatter` variant:
///
///   d_act[left[i]]  += d_sync[i] * activated[right[i]] / sqrt(beta[i])
///   d_act[right[i]] += d_sync[i] * activated[left[i]]  / sqrt(beta[i])
#[derive(Debug)]
pub struct SyncBackwardScatterArgs<'a> {
    pub d_sync: &'a [f32],
    pub pairs_left: &'a [u32],
    pub pairs_right: &'a [u32],
    pub activated: &'a [f32],
    pub beta: &'a [f32],
    pub d_act: &'a mut [f32],
    pub n_pairs: usize,
    pub d_model: usize,
}

/// Finite operation set. See module docs for granularity policy.
///
/// Lifetime `'a` borrows every slice from the caller; the op lives no
/// longer than the underlying buffers.
///
/// Matmul variants follow BLAS naming: the two letters describe whether
/// the A and B operands are used as-is (N) or transposed (T). All three
/// produce `out` of shape [m×n]. When `bias` is `Some`, it is
/// broadcast-added to each output row: `out[r, c] = Σ a·b + bias[c]`.
#[derive(Debug)]
pub enum Op<'a> {
    // ─── dense linear algebra ────────────────────────────────

    /// `C = A @ B`. `a` is [m×k], `b` is [k×n], `out` is [m×n].
    MatmulNN {
        a: &'a [f32],
        b: &'a [f32],
        out: &'a mut [f32],
        bias: Option<&'a [f32]>,
        m: usize,
        k: usize,
        n: usize,
    },

    /// `C = A @ B^T` — gradient-of-weights flavor.
    /// `a` is [m×k], `b` is [n×k] (row-major, transposed on the fly),
    /// `out` is [m×n].
    MatmulNT {
        a: &'a [f32],
        b: &'a [f32],
        out: &'a mut [f32],
        bias: Option<&'a [f32]>,
        m: usize,
        k: usize,
        n: usize,
    },

    /// `C = A^T @ B` — used in some backward passes.
    /// `a` is [k×m] (row-major, transposed on the fly), `b` is [k×n],
    /// `out` is [m×n].
    MatmulTN {
        a: &'a [f32],
        b: &'a [f32],
        out: &'a mut [f32],
        bias: Option<&'a [f32]>,
        m: usize,
        k: usize,
        n: usize,
    },

    /// Forward matvec: `out = weight @ x + bias`.
    /// `weight` shape is [out_dim × in_dim] row-major (may be quantized
    /// opaque bytes when `quant != F32`).
    Matvec {
        x: &'a [f32],
        weight: &'a [f32],
        bias: &'a [f32],
        out: &'a mut [f32],
        out_dim: usize,
        in_dim: usize,
        quant: QuantKind,
    },

    /// Transposed matvec (typical gradient-of-input for a Linear layer):
    /// `d_input = weight^T @ d_out`.
    MatvecT {
        d_out: &'a [f32],
        weight: &'a [f32],
        d_input: &'a mut [f32],
        out_dim: usize,
        in_dim: usize,
    },

    /// Accumulating outer product: `accum += a ⊗ b`. Used in gradient
    /// accumulation (d_weight = d_out ⊗ input).
    OuterProductAcc {
        a: &'a [f32],
        b: &'a [f32],
        accum: &'a mut [f32],
        m: usize,
        n: usize,
    },

    // ─── normalization and activations ───────────────────────

    /// LayerNorm forward: normalizes `x` then scales by `gamma` + shifts
    /// by `beta`. Output written into `out`. `cache` stores intermediate
    /// mean/rstd for backward (length = 2 * n_rows).
    LayerNormFwd {
        x: &'a [f32],
        gamma: &'a [f32],
        beta: &'a [f32],
        out: &'a mut [f32],
        cache: &'a mut [f32],
        n_rows: usize,
        n_cols: usize,
    },

    /// LayerNorm backward: returns d_x, d_gamma, d_beta from d_out and
    /// the cached mean/rstd from forward.
    LayerNormBwd {
        d_out: &'a [f32],
        x: &'a [f32],
        gamma: &'a [f32],
        cache: &'a [f32],
        d_x: &'a mut [f32],
        d_gamma: &'a mut [f32],
        d_beta: &'a mut [f32],
        n_rows: usize,
        n_cols: usize,
    },

    /// Fused LayerNorm + SiLU forward. Common CTM path (layer_norm followed
    /// by silu activation). Cache retains mean/rstd for a matched backward.
    LnSiluFwd {
        x: &'a [f32],
        gamma: &'a [f32],
        beta: &'a [f32],
        out: &'a mut [f32],
        cache: &'a mut [f32],
        n_rows: usize,
        n_cols: usize,
    },

    /// SiLU forward: out = x * sigmoid(x).
    SiluFwd {
        x: &'a [f32],
        out: &'a mut [f32],
    },

    /// SiLU backward: d_x = d_out * dSiLU/dx(x).
    SiluBwd {
        d_out: &'a [f32],
        x: &'a [f32],
        d_x: &'a mut [f32],
    },

    /// GLU forward: gated linear unit on a tensor whose last dim splits
    /// into (value, gate). `out.len() == x.len() / 2`.
    GluFwd {
        x: &'a [f32],
        out: &'a mut [f32],
    },

    /// GLU backward.
    GluBwd {
        d_out: &'a [f32],
        x: &'a [f32],
        d_x: &'a mut [f32],
    },

    /// GLU backward, per-neuron variant (CTM-specific, fuses over neuron axis).
    PerNeuronGluBwd {
        d_out: &'a [f32],
        x: &'a [f32],
        d_x: &'a mut [f32],
        n_neurons: usize,
        feat_per_neuron: usize,
    },

    // ─── reductions and optimizer steps ───────────────────────

    /// Reduce sum of squares: scalar `out[0] = sum(x[i]^2)`.
    ReduceL2Sq {
        x: &'a [f32],
        out: &'a mut [f32],
    },

    /// SGD weight update: `w -= lr * g`.
    SgdUpdate {
        w: &'a mut [f32],
        g: &'a [f32],
        lr: f32,
    },

    /// AdamW update. See [`AdamWArgs`] for field semantics.
    AdamW(AdamWArgs<'a>),

    // ─── CTM-specific (no library equivalent) ────────────────

    /// SuperLinear (Neuron-Level MLP) forward: per-neuron matvec over
    /// trace history. `weights` shape [d_model × out_per × memory_length],
    /// `biases` shape [d_model × out_per].
    ///
    /// `cache` is optional: if `Some`, the pre-activation is written
    /// there for use by a subsequent `SuperLinearBwdDw`/`SuperLinearBwdDx`.
    /// Inference paths pass `None`. Backends that can't compute cache
    /// must return `false` from `supports()` when `cache.is_some()`.
    SuperLinearFwd {
        trace: &'a [f32],
        weights: &'a [f32],
        biases: &'a [f32],
        out: &'a mut [f32],
        cache: Option<&'a mut [f32]>,
        d_model: usize,
        memory_length: usize,
        out_per: usize,
    },

    /// SuperLinear backward — gradient w.r.t. weights.
    SuperLinearBwdDw {
        d_out: &'a [f32],
        trace: &'a [f32],
        d_weights: &'a mut [f32],
        d_model: usize,
        memory_length: usize,
        out_per: usize,
    },

    /// SuperLinear backward — gradient w.r.t. input trace.
    SuperLinearBwdDx {
        d_out: &'a [f32],
        weights: &'a [f32],
        d_trace: &'a mut [f32],
        d_model: usize,
        memory_length: usize,
        out_per: usize,
    },

    /// CTM neuron-pair synchronization forward. Accumulates dot-products
    /// between paired neurons into a running sync representation with
    /// learned decay.
    SyncUpdateFwd {
        h: &'a [f32],
        pairs_left: &'a [u32],
        pairs_right: &'a [u32],
        decay: &'a [f32],
        sync_state: &'a mut [f32],
        sync_out: &'a mut [f32],
        n_pairs: usize,
    },

    /// CTM synchronization backward — scatters d_sync onto the
    /// contributing neurons with per-pair decay-based normalisation.
    /// See [`SyncBackwardScatterArgs`] for field semantics.
    SyncBackwardScatter(SyncBackwardScatterArgs<'a>),

    /// Rotate trace memory: shift column-wise by one and write `new_val`
    /// into the most-recent column. In-place on `trace`.
    TraceShiftFwd {
        trace: &'a mut [f32],
        new_val: &'a [f32],
        d_model: usize,
        memory_length: usize,
    },
}

impl<'a> Op<'a> {
    /// Short, backend-agnostic name used for logging and parity tests.
    pub fn name(&self) -> &'static str {
        match self {
            Op::MatmulNN { .. } => "matmul_nn",
            Op::MatmulNT { .. } => "matmul_nt",
            Op::MatmulTN { .. } => "matmul_tn",
            Op::Matvec { .. } => "matvec",
            Op::MatvecT { .. } => "matvec_t",
            Op::OuterProductAcc { .. } => "outer_product_acc",
            Op::LayerNormFwd { .. } => "layer_norm_fwd",
            Op::LayerNormBwd { .. } => "layer_norm_bwd",
            Op::LnSiluFwd { .. } => "ln_silu_fwd",
            Op::SiluFwd { .. } => "silu_fwd",
            Op::SiluBwd { .. } => "silu_bwd",
            Op::GluFwd { .. } => "glu_fwd",
            Op::GluBwd { .. } => "glu_bwd",
            Op::PerNeuronGluBwd { .. } => "per_neuron_glu_bwd",
            Op::ReduceL2Sq { .. } => "reduce_l2_sq",
            Op::SgdUpdate { .. } => "sgd_update",
            Op::AdamW(_) => "adamw",
            Op::SuperLinearFwd { .. } => "superlinear_fwd",
            Op::SuperLinearBwdDw { .. } => "superlinear_bwd_dw",
            Op::SuperLinearBwdDx { .. } => "superlinear_bwd_dx",
            Op::SyncUpdateFwd { .. } => "sync_update_fwd",
            Op::SyncBackwardScatter(_) => "sync_backward_scatter",
            Op::TraceShiftFwd { .. } => "trace_shift_fwd",
        }
    }
}

// ─── Mapping to existing KFD kernel names (for Task 2.1 port) ─────
//
//   MatmulNN               → matmul_blocked | matmul_small (shape-dispatched)
//   MatmulNT / MatmulTN    → (no KFD kernel yet; fall through to CPU)
//   Matvec{F32}            → matvec | matvec_tiled (shape-dispatched)
//   Matvec{Q4K}            → matvec_q4k
//   MatvecT                → matvec_t_tiled
//   OuterProductAcc        → outer_product_acc
//   LayerNormFwd           → layer_norm_fwd
//   LayerNormBwd           → ln_bwd
//   LnSiluFwd              → ln_silu_fwd
//   SiluFwd                → silu_fwd
//   SiluBwd                → silu_bwd
//   GluFwd                 → glu_fwd
//   GluBwd                 → glu_bwd
//   PerNeuronGluBwd        → per_neuron_glu_bwd
//   ReduceL2Sq             → reduce_l2_sq
//   SgdUpdate              → sgd_update
//   AdamW                  → adamw
//   SuperLinearFwd         → superlinear_fwd
//   SuperLinearBwdDw       → superlinear_bwd_dw
//   SuperLinearBwdDx       → superlinear_bwd_dx
//   SyncUpdateFwd          → sync_update_fwd
//   SyncBackwardScatter    → sync_backward_scatter
//   TraceShiftFwd          → trace_shift_fwd
//
// Debug/test kernels (test_store, addr_dump, coop_test, lds_test,
// matmul_dbg) are intentionally NOT exposed — they're dispatch
// internals, not logical ops.
