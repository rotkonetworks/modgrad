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

/// Element-wise activation mode for [`Op::ActivationResident`].
///
/// Each variant maps to a specific MIOpen `miopenActivationMode_t` plus
/// any required compose step. The composite `Silu` variant is
/// implemented as `Logistic` followed by `OpTensor(MUL)` against the
/// original input — MIOpen has no native SiLU/Swish kernel, but the
/// two-call path stays fully device-resident.
///
/// Only the modes we actually consume from the Phase 5b residency chain
/// are listed. Adding a new mode is mechanical: extend the enum, map it
/// in the ROCm dispatch, document any compose semantics here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationMode {
    /// Sigmoid: `y = 1 / (1 + e^-x)`. Maps to `miopenActivationLOGISTIC`.
    Logistic,
    /// Hyperbolic tangent: `y = tanh(x)`. MIOpen's TANH applies
    /// `beta * tanh(alpha * x)`; alpha=beta=1 is the standard form
    /// callers expect. Maps to `miopenActivationTANH`.
    Tanh,
    /// Rectified linear unit: `y = max(0, x)`.
    /// Maps to `miopenActivationRELU`.
    Relu,
    /// SiLU / Swish: `y = x * sigmoid(x)`. Composed as
    /// `Logistic(x) -> y` followed by `OpTensor(MUL, x, y) -> y`.
    /// Two MIOpen calls; both stay device-resident.
    Silu,
}

/// Binary element-wise op for [`Op::OpTensorResident`].
///
/// Maps directly to `miopenTensorOp_t`. The MIOpen call computes
/// `c = op(alpha1 * a, alpha2 * b) + beta * c`. Only the four ops the
/// Phase 5b chain consumes are exposed; extending is mechanical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpKind {
    /// Add: `c = alpha1 * a + alpha2 * b + beta * c`.
    /// Maps to `miopenTensorOpAdd`.
    Add,
    /// Multiply: `c = (alpha1 * a) * (alpha2 * b) + beta * c`.
    /// Used in SiLU compose (and any other gate * value pattern).
    /// Maps to `miopenTensorOpMul`.
    Mul,
    /// Min: `c = min(alpha1 * a, alpha2 * b) + beta * c`.
    /// Maps to `miopenTensorOpMin`.
    Min,
    /// Max: `c = max(alpha1 * a, alpha2 * b) + beta * c`.
    /// Maps to `miopenTensorOpMax`.
    Max,
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

    /// **Device-resident** forward matvec — same math as `Matvec`,
    /// but every operand is a hip-device pointer (from a
    /// `HipBuffer`). The dispatcher hands them straight to
    /// `hipblasSgemv` with no host↔device transfer. Backends that
    /// cannot consume device pointers (CPU, vulkan-without-shared-vram)
    /// must return `BackendError::UnsupportedShape` so callers fall
    /// back to the host-slice `Matvec`.
    ///
    /// Lifetime contract: caller guarantees the device pointers
    /// remain valid for the duration of the dispatch. Typically the
    /// pointers come from owned `HipBuffer`s held by `Linear`'s
    /// weight cache (long-lived) or freshly-allocated activation
    /// `GpuVec::Hip` buffers (lifetime ≥ this dispatch).
    MatvecResident {
        x_dev: *const f32,
        weight_dev: *const f32,
        bias_dev: *const f32,
        out_dev: *mut f32,
        out_dim: usize,
        in_dim: usize,
    },

    /// **Device-resident** matmul: `C = A @ B`. Same math as
    /// [`Op::MatmulNN`] but every operand is a hip-device pointer.
    /// Backends that cannot consume device pointers (CPU, vulkan-
    /// without-shared-vram) return `BackendError::Unsupported`.
    ///
    /// Layout: `a` is `[m × k]` row-major, `b` is `[k × n]` row-major,
    /// `out` is `[m × n]` row-major. No bias — callers that need bias
    /// run a follow-up resident add (e.g. `OpTensorResident{Add}`).
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    MatmulResidentNN {
        a_dev: *const f32,
        b_dev: *const f32,
        out_dev: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    },

    /// **Device-resident** matmul: `C = A @ B^T`. Same math as
    /// [`Op::MatmulNT`] but every operand is a hip-device pointer.
    ///
    /// Layout: `a` is `[m × k]` row-major, `b` is `[n × k]` row-major
    /// (transposed on the fly), `out` is `[m × n]` row-major. No bias.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    MatmulResidentNT {
        a_dev: *const f32,
        b_dev: *const f32,
        out_dev: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    },

    /// **Device-resident** matmul: `C = A^T @ B`. Same math as
    /// [`Op::MatmulTN`] but every operand is a hip-device pointer.
    ///
    /// Layout: `a` is `[k × m]` row-major (transposed on the fly),
    /// `b` is `[k × n]` row-major, `out` is `[m × n]` row-major. No
    /// bias.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    MatmulResidentTN {
        a_dev: *const f32,
        b_dev: *const f32,
        out_dev: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    },

    /// **Device-resident** RMSNorm forward.
    ///
    /// `y[r, c] = x[r, c] / sqrt(mean(x[r, :]^2) + eps) * weight[c]`
    ///
    /// Modern LLM normalisation (Gemma, Llama, Qwen). MIOpen ships
    /// LayerNorm but no RMSNorm, so this op routes to a custom hipcc
    /// kernel compiled by `build.rs` for `gfx1102`.
    ///
    /// Layout: `x` and `y` are `[n, hidden]` row-major; `weight` is
    /// length-`hidden`.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch. CPU and other non-
    /// resident backends return `Unsupported`.
    RmsNormResident {
        x_dev: *const f32,
        weight_dev: *const f32,
        y_dev: *mut f32,
        n: usize,
        hidden: usize,
        eps: f32,
    },

    /// **Device-resident** LayerNorm forward — affine variant matching
    /// PyTorch `nn.LayerNorm(elementwise_affine=True)`.
    ///
    /// `y[r, c] = ((x[r, c] - mean[r]) / sqrt(var[r] + epsilon)) * weight[c] + bias[c]`
    ///
    /// Layout: `x` is `[n, normalized_size]` row-major; `weight` and
    /// `bias` are length-`normalized_size`; `y` matches `x`.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch. Wired only on backends
    /// that consume hip-device pointers (ROCm via MIOpen). Other
    /// backends return `Unsupported`.
    LayerNormResident {
        x_dev: *const f32,
        weight_dev: *const f32,
        bias_dev: *const f32,
        y_dev: *mut f32,
        n: usize,
        normalized_size: usize,
        epsilon: f32,
    },

    /// **Device-resident** softmax forward, row-wise.
    ///
    /// For each of `n_rows` rows of length `row_len`, compute either
    /// `softmax` (`log == false`) or `log_softmax` (`log == true`).
    /// MIOpen's accurate (max-subtracting) algorithm is selected so
    /// numerical behaviour matches the CPU reference.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    SoftmaxResident {
        x_dev: *const f32,
        y_dev: *mut f32,
        n_rows: usize,
        row_len: usize,
        log: bool,
    },

    /// **Device-resident** element-wise activation forward.
    ///
    /// Applies `mode` element-wise across `n` floats. `Silu` decomposes
    /// into `Logistic` followed by `OpTensor(MUL)` against the original
    /// input — see [`ActivationMode::Silu`] for the contract. All other
    /// modes map to a single `miopenActivationForward` call.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    ActivationResident {
        x_dev: *const f32,
        y_dev: *mut f32,
        n: usize,
        mode: ActivationMode,
    },

    /// **Device-resident** GLU forward.
    ///
    /// Input layout — set by MIOpen's GLU kernel: a contiguous
    /// `[2, n_rows, half_size]` block where the first
    /// `n_rows * half_size` floats are the value half and the next
    /// `n_rows * half_size` floats are the gate half. Output is a
    /// contiguous `[n_rows, half_size]` block:
    /// `y[r, i] = value[r, i] * sigmoid(gate[r, i])`.
    ///
    /// **NOT** the per-row interleaved layout
    /// (`[value_0..N, gate_0..N]` repeated per row). MIOpen's GLU
    /// solver only supports `dim = 0` splits today (per
    /// `src/solver/glu/forward_glu.cpp`), so the split MUST be along
    /// the leading axis. Callers using per-row interleaving must
    /// transpose the input before issuing this op (or scatter into
    /// the value/gate planes during the producing matvec).
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    GluResident {
        x_dev: *const f32,
        y_dev: *mut f32,
        n_rows: usize,
        half_size: usize,
    },

    /// **Device-resident** binary element-wise op.
    ///
    /// `c = op(alpha1 * a, alpha2 * b) + beta * c`, applied element-wise
    /// over `n` floats with all three buffers shaped identically.
    ///
    /// Maps to `miopenOpTensor`. The compose step inside the SiLU
    /// activation uses this op variant under the hood, but it's
    /// independently useful for any gate * value pattern that already
    /// has both operands resident.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    OpTensorResident {
        a_dev: *const f32,
        b_dev: *const f32,
        c_dev: *mut f32,
        n: usize,
        alpha1: f32,
        alpha2: f32,
        beta: f32,
        op: BinaryOpKind,
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
    /// by `beta`. Output written into `out`.
    ///
    /// `cache` is optional: if `Some`, intermediate mean/rstd
    /// (length = 2 * n_rows) are written for use by a subsequent
    /// `LayerNormBwd`. Inference paths pass `None` and skip the
    /// scratch allocation. Backends that can't compute cache must
    /// return `false` from `supports()` when `cache.is_some()`.
    LayerNormFwd {
        x: &'a [f32],
        gamma: &'a [f32],
        beta: &'a [f32],
        out: &'a mut [f32],
        cache: Option<&'a mut [f32]>,
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
    /// by silu activation).
    ///
    /// `cache` is optional: if `Some`, mean/rstd (length = 2 * n_rows) are
    /// retained for a matched backward. Inference paths pass `None` and
    /// skip the scratch allocation. Backends that can't compute cache
    /// must return `false` from `supports()` when `cache.is_some()`.
    LnSiluFwd {
        x: &'a [f32],
        gamma: &'a [f32],
        beta: &'a [f32],
        out: &'a mut [f32],
        cache: Option<&'a mut [f32]>,
        n_rows: usize,
        n_cols: usize,
    },

    /// SiLU forward: out = x * sigmoid(x).
    SiluFwd {
        x: &'a [f32],
        out: &'a mut [f32],
    },

    /// SiLU forward, in-place: x = x * sigmoid(x).
    ///
    /// Use when the caller does NOT need the pre-activation preserved
    /// (e.g. inference paths). Backwards training paths must use
    /// `SiluFwd` because `SiluBwd` requires the original `x`. This
    /// variant lets backends whose native kernel is in-place (KFD)
    /// skip the redundant host-side memcpy that `SiluFwd` would
    /// otherwise force.
    SiluFwdInplace {
        x: &'a mut [f32],
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
    TraceRotateInplace {
        trace: &'a mut [f32],
        new_val: &'a [f32],
        d_model: usize,
        memory_length: usize,
    },

    // ─── fused (Stage 1) ──────────────────────────────────────

    /// Fused synapse forward: matvec → GLU(halves) → SiLU(inplace) → LayerNorm(no affine).
    /// `weight` has shape [2*out_dim × in_dim] (GLU halves the channel
    /// dim), `bias` has length `2*out_dim`, `x` has length `in_dim`,
    /// and `out` has length `out_dim`.
    ///
    /// Intentionally exposes no scratch buffer — the matvec/GLU
    /// intermediate is a backend-private implementation detail. CPU
    /// composes and allocates its own Vec; KFD dispatches the fused
    /// kernel against its preallocated VRAM slot.
    ///
    /// **NOT** a fused form of `modgrad_ctm::synapse::SynapseBlock`.
    /// `SynapseBlock::forward` is Linear → LayerNorm(with learnable
    /// affine γ/β) → SiLU, no GLU stage. Op::SynapseForward is a
    /// *different* fused primitive: GLU is present and LayerNorm has
    /// zero-mean-unit-variance only (no γ/β). Fusing `SynapseBlock`
    /// through this op would silently drop γ·x+β and insert a GLU
    /// gate — do not do it. Callers who want the affine-LN sequence
    /// must compose explicitly: `Op::Matvec` + `Op::LayerNormFwd`
    /// (which takes gamma/beta) + `Op::SiluFwd`/`SiluFwdInplace`.
    SynapseForward {
        weight: &'a [f32],
        bias: &'a [f32],
        x: &'a [f32],
        out: &'a mut [f32],
        out_dim: usize,
        in_dim: usize,
    },

    /// LayerNorm forward, in-place, row-wise. Writes normalized values
    /// back into `x`. No gamma/beta — pure mean/rstd normalization (the
    /// affine step is absorbed into the caller, e.g. the synapse-forward
    /// pipeline where weight/bias already handled scaling).
    ///
    /// `n_rows * n_cols` must equal `x.len()`.
    ///
    /// **Pure zero-mean / unit-variance normalization only.** This op
    /// does NOT apply a learnable γ·norm + β affine transform — no
    /// scale, no shift. Callers that need affine LayerNorm (matching
    /// PyTorch `nn.LayerNorm(elementwise_affine=True)`, or anything
    /// carrying trainable `ln_gamma`/`ln_beta` parameters such as
    /// `modgrad_ctm::synapse::SynapseBlock`) must use
    /// [`Op::LayerNormFwd`] instead, which takes `gamma`/`beta` slices.
    LayerNormInplace {
        x: &'a mut [f32],
        n_rows: usize,
        n_cols: usize,
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
            Op::MatvecResident { .. } => "matvec_resident",
            Op::MatmulResidentNN { .. } => "matmul_resident_nn",
            Op::MatmulResidentNT { .. } => "matmul_resident_nt",
            Op::MatmulResidentTN { .. } => "matmul_resident_tn",
            Op::RmsNormResident { .. } => "rms_norm_resident",
            Op::LayerNormResident { .. } => "layer_norm_resident",
            Op::SoftmaxResident { .. } => "softmax_resident",
            Op::ActivationResident { .. } => "activation_resident",
            Op::GluResident { .. } => "glu_resident",
            Op::OpTensorResident { .. } => "op_tensor_resident",
            Op::MatvecT { .. } => "matvec_t",
            Op::OuterProductAcc { .. } => "outer_product_acc",
            Op::LayerNormFwd { .. } => "layer_norm_fwd",
            Op::LayerNormBwd { .. } => "layer_norm_bwd",
            Op::LnSiluFwd { .. } => "ln_silu_fwd",
            Op::SiluFwd { .. } => "silu_fwd",
            Op::SiluFwdInplace { .. } => "silu_fwd_inplace",
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
            Op::TraceRotateInplace { .. } => "trace_rotate_inplace",
            Op::SynapseForward { .. } => "synapse_forward",
            Op::LayerNormInplace { .. } => "layer_norm_inplace",
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
//   SiluFwd                → silu_fwd (with host-side copy_from_slice)
//   SiluFwdInplace         → silu_fwd (native in-place, no copy)
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
//   TraceRotateInplace     → trace_shift_fwd
//   SynapseForward         → synapse_forward (fused: matvec→glu→silu→layer_norm)
//   LayerNormInplace       → layer_norm_fwd (single WG, inplace; same kernel
//                             the KFD synapse pipeline invokes at the end)
//
// Debug/test kernels (test_store, addr_dump, coop_test, lds_test,
// matmul_dbg) are intentionally NOT exposed — they're dispatch
// internals, not logical ops.
