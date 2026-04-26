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

    /// **Device-resident** bf16 matmul: `C = A @ B`. Same math as
    /// [`Op::MatmulResidentNN`] but operands are bf16 (stored as `u16`)
    /// with fp32 accumulate. Dispatched via `hipblasGemmEx` with
    /// `HIP_R_16BF` input/output and `HIPBLAS_COMPUTE_32F` compute.
    ///
    /// `alpha`/`beta` are fp32 — that's the hipblasGemmEx contract for
    /// 32F-compute mode.
    ///
    /// Layout: `a` is `[m × k]` row-major, `b` is `[k × n]` row-major,
    /// `c` is `[m × n]` row-major. No bias.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    MatmulResidentBf16Nn {
        a_dev: *const u16,
        b_dev: *const u16,
        c_dev: *mut u16,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
    },

    /// **Device-resident** bf16 matmul: `C = A @ B^T`. See
    /// [`Op::MatmulResidentBf16Nn`] for the dispatch contract.
    ///
    /// Layout: `a` is `[m × k]` row-major, `b` is `[n × k]` row-major
    /// (transposed on the fly), `c` is `[m × n]` row-major.
    MatmulResidentBf16Nt {
        a_dev: *const u16,
        b_dev: *const u16,
        c_dev: *mut u16,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
    },

    /// **Device-resident** bf16 matmul: `C = A^T @ B`. See
    /// [`Op::MatmulResidentBf16Nn`] for the dispatch contract.
    ///
    /// Layout: `a` is `[k × m]` row-major (transposed on the fly),
    /// `b` is `[k × n]` row-major, `c` is `[m × n]` row-major.
    MatmulResidentBf16Tn {
        a_dev: *const u16,
        b_dev: *const u16,
        c_dev: *mut u16,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
    },

    /// **Device-resident** bf16 forward matvec — same math as
    /// [`Op::MatvecResident`], but every operand is bf16 (stored as
    /// `u16`). Dispatched via `hipblasGemmEx` (a 1-column GEMM since
    /// hipblasGemvEx is not available in our hipblas) with bf16
    /// input/output and fp32 compute.
    ///
    /// `out_dim`/`in_dim` semantics match [`Op::MatvecResident`].
    /// `bias_dev` is bf16; the dispatch seeds `out_dev` from `bias_dev`
    /// with a D2D copy before the GEMM (alpha = 1, beta = 1 means
    /// `out = bias + W·x`).
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    MatvecResidentBf16 {
        x_dev: *const u16,
        weight_dev: *const u16,
        bias_dev: *const u16,
        out_dev: *mut u16,
        out_dim: usize,
        in_dim: usize,
    },

    /// **Device-resident** Q4_K_M dequantize.
    ///
    /// Reads `n_blocks` Q4_K-formatted blocks (144 bytes each) from
    /// `q4k_dev` and writes `n_blocks * 256` fp32 values into
    /// `fp32_dev`. The byte pointer is opaque so callers can hand in
    /// either a hip-device buffer they uploaded with hipMemcpy or a
    /// hip-pinned host mirror.
    ///
    /// Layout of each block (matches `kfd::gguf` / llama.cpp):
    ///   +0x00: d (fp16), dmin (fp16) — super-block scale and min
    ///   +0x04: scales[12]            — packed 6-bit (sc, m) for 8 sub-blocks
    ///   +0x10: qs[128]               — 256 4-bit quants
    ///
    /// Foundation primitive for the streaming-quantised-weight path:
    /// host stores Q4_K bytes (~12.5% of fp32 size), device fp32 buffer
    /// is filled on demand, evicted on reuse pressure.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch. Backends without a
    /// hipcc-built Q4_K kernel (CPU, vulkan, hip without hipcc) return
    /// `Unsupported`.
    DequantQ4KResident {
        q4k_dev: *const u8,
        fp32_dev: *mut f32,
        n_blocks: usize,
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

    /// **Device-resident** RMSNorm backward. Pairs with
    /// [`Op::RmsNormResident`] to keep the final-RMSNorm gradient on
    /// device — eliminates the host-bounce that
    /// `GptModelResident::backward` used to do (D2H input + H2D grad
    /// every step).
    ///
    /// Math (per row, with N = `hidden`):
    ///   rstd  = 1 / sqrt(mean(x^2) + eps)
    ///   S     = Σ_c (dy[c] * x[c] * weight[c])
    ///   dx[c] = (weight[c] * dy[c] - x[c] * S * rstd^2 / N) * rstd
    ///
    /// Per-feature gamma gradient (summed across rows):
    ///   dweight[c] = Σ_r (dy[r, c] * x[r, c] * rstd[r])
    ///
    /// `dweight_dev` may be NULL when the caller doesn't need the
    /// gamma gradient (frozen-norm path); the dispatch elides the
    /// second kernel launch in that case. When non-NULL, the kernel
    /// **accumulates** into `dweight_dev` — caller zeros beforehand if
    /// a fresh gradient is desired (matches MIOpen LayerNorm backward
    /// convention).
    ///
    /// Layout: `x_dev`, `dy_dev`, `dx_dev` are `[n, hidden]` row-major;
    /// `weight_dev` and (optional) `dweight_dev` are length-`hidden`.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch. CPU and other non-
    /// resident backends return `Unsupported`.
    RmsNormBackwardResident {
        x_dev: *const f32,
        dy_dev: *const f32,
        weight_dev: *const f32,
        dx_dev: *mut f32,
        /// Optional gamma gradient accumulator; pass `null_mut()` to
        /// skip the dweight pass entirely.
        dweight_dev: *mut f32,
        n: usize,
        hidden: usize,
        eps: f32,
    },

    /// **Device-resident** RoPE backward (rotary-embedding adjoint).
    ///
    /// Forward: per pair `(x_pre[i], x_pre[i+half])`,
    /// `x_post[i]      = x_pre[i] * cos[i] - x_pre[i+half] * sin[i]`
    /// `x_post[i+half] = x_pre[i] * sin[i] + x_pre[i+half] * cos[i]`.
    ///
    /// Backward: rotate the gradient by the negated angle (rotation
    /// matrices are orthogonal, so the adjoint flips the sign of `sin`):
    /// `dx_pre[i]      =  dx_post[i] * cos[i] + dx_post[i+half] * sin[i]`
    /// `dx_pre[i+half] = -dx_post[i] * sin[i] + dx_post[i+half] * cos[i]`
    ///
    /// Layout: `dx_post` and `dx_pre` are each `[num_heads, head_dim]`
    /// row-major where `head_dim = 2 * half_dim`. `cos` and `sin` are
    /// each length-`half_dim`, indexed at the position used by the
    /// matched forward (caller slices into the precomputed
    /// `RotaryEmbedding` cos/sin tables).
    ///
    /// In-place is supported (`dx_post == dx_pre`): the kernel reads
    /// both halves of each pair before writing either.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch. Routes to the custom
    /// hipcc kernel compiled by `build.rs`; CPU and other non-resident
    /// backends return `Unsupported`.
    RopeBackwardResident {
        dx_post_dev: *const f32,
        cos_dev: *const f32,
        sin_dev: *const f32,
        dx_pre_dev: *mut f32,
        num_heads: usize,
        head_dim: usize,
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

    /// **Device-resident** per-neuron batched GLU forward.
    ///
    /// One launch covering `n_neurons` neurons whose per-neuron layout
    /// is value-then-gate interleaved WITHIN each neuron — NOT MIOpen's
    /// `dim=0` split. For each neuron `n` and channel `i`:
    ///
    /// `y[n, i] = x[n, i] * sigmoid(x[n, half_size + i])`
    ///
    /// where `x` is `[n_neurons, 2 * half_size]` row-major and `y` is
    /// `[n_neurons, half_size]` row-major.
    ///
    /// Replaces the per-neuron `Op::GluResident` loop in the CTM
    /// forward path (`crates/modgrad-ctm/src/forward.rs`); MIOpen's
    /// solver only splits along the leading axis so it cannot batch
    /// the per-neuron interleaved layout in a single call. Routed
    /// through a custom hipcc kernel
    /// (`kernels/per_neuron_glu_batched.hip`).
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch. CPU and other non-
    /// resident backends return `Unsupported`.
    PerNeuronGluBatchedResident {
        x_dev: *const f32,
        y_dev: *mut f32,
        n_neurons: usize,
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

    /// **Device-resident** LayerNorm backward — affine variant matching
    /// PyTorch `nn.LayerNorm.backward`. Returns gradients w.r.t. input,
    /// weight, and bias.
    ///
    /// `mean_dev` and `rstd_dev` are the per-row mean and reciprocal
    /// standard deviation captured by a matching forward pass — see
    /// [`Op::LayerNormResident`]. Each is length `n` (one scalar per
    /// row).
    ///
    /// Layout: `x` and `dy` and `dx` are `[n, normalized_size]`
    /// row-major; `weight` is length-`normalized_size`; `dweight` and
    /// `dbias` are each length-`normalized_size` and **accumulated
    /// into** (caller is responsible for zeroing if a fresh gradient
    /// is desired).
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch. Wired only on backends
    /// that consume hip-device pointers (ROCm via MIOpen). Other
    /// backends return `Unsupported`.
    LayerNormBackwardResident {
        x_dev: *const f32,
        dy_dev: *const f32,
        weight_dev: *const f32,
        mean_dev: *const f32,
        rstd_dev: *const f32,
        dx_dev: *mut f32,
        dweight_dev: *mut f32,
        dbias_dev: *mut f32,
        n: usize,
        normalized_size: usize,
    },

    /// **Device-resident** softmax backward, row-wise.
    ///
    /// For each of `n_rows` rows of length `row_len`, compute the
    /// gradient w.r.t. softmax (or log_softmax when `log == true`)
    /// input given `y` (forward output) and `dy` (incoming gradient).
    ///
    /// Layout: `y`, `dy`, and `dx` are `[n_rows, row_len]` row-major.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    SoftmaxBackwardResident {
        y_dev: *const f32,
        dy_dev: *const f32,
        dx_dev: *mut f32,
        n_rows: usize,
        row_len: usize,
        log: bool,
    },

    /// **Device-resident** element-wise activation backward.
    ///
    /// Computes `dx = dy * dActivation/dx(x, y)` element-wise across
    /// `n` floats. `mode` matches the forward [`ActivationMode`]; the
    /// backward path needs both `x` (pre-activation) and `y`
    /// (post-activation) for some MIOpen modes.
    ///
    /// `Silu`: composed as Logistic backward (`dx_logistic =
    /// dy * y * (1 - y)`) followed by `dx = x * dx_logistic + sigmoid(x) * dy`.
    /// Implemented via `miopenActivationBackward(LOGISTIC) + element-wise
    /// composition` — two MIOpen calls, both resident.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    ActivationBackwardResident {
        x_dev: *const f32,
        y_dev: *const f32,
        dy_dev: *const f32,
        dx_dev: *mut f32,
        n: usize,
        mode: ActivationMode,
    },

    /// **Device-resident** GLU backward.
    ///
    /// Reverses [`Op::GluResident`] — given `x` (the value/gate
    /// concatenation) and `dy` (gradient w.r.t. the GLU output),
    /// compute `dx`.
    ///
    /// Layout matches forward: `x_dev` is `[2, n_rows, half_size]`
    /// contiguous (value half then gate half); `dy_dev` is
    /// `[n_rows, half_size]`; `dx_dev` matches `x_dev` shape.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch.
    GluBackwardResident {
        x_dev: *const f32,
        dy_dev: *const f32,
        dx_dev: *mut f32,
        n_rows: usize,
        half_size: usize,
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

    /// **Device-resident** AdamW update.
    ///
    /// Math (decoupled-weight-decay AdamW), per element:
    ///
    /// ```text
    /// m_t   = beta1 * m_{t-1} + (1 - beta1) * g
    /// v_t   = beta2 * v_{t-1} + (1 - beta2) * g * g
    /// m_hat = m_t / (1 - beta1^t)         (== m_t * bc1_inv)
    /// v_hat = v_t / (1 - beta2^t)         (== v_t * bc2_inv)
    /// w_t   = w_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w_{t-1})
    /// ```
    ///
    /// `w_dev`, `g_dev`, `m_dev`, `v_dev` all point at length-`n` fp32
    /// element buffers. The kernel updates `w_dev`, `m_dev`, `v_dev` in
    /// place; `g_dev` is read-only.
    ///
    /// `bc1_inv` / `bc2_inv` are the bias-correction reciprocals
    /// `1 / (1 - beta1^t)` / `1 / (1 - beta2^t)`, computed by the
    /// caller once per training step (matches the host-side
    /// [`Op::AdamW`] contract — same arithmetic, identical numerics).
    ///
    /// Used by `LmTrainer` (and any future foundation-model trainer) to
    /// eliminate the per-parameter D2H / host-AdamW / H2D round-trip
    /// that the host `AdamW` path forces.
    ///
    /// Lifetime contract: caller guarantees the device pointers remain
    /// valid for the duration of the dispatch. Backends without a
    /// hipcc-built AdamW kernel (CPU, KFD, vulkan, hip without hipcc)
    /// return `Unsupported` and the caller must fall back to the host
    /// `Op::AdamW` path.
    AdamWResident {
        w_dev: *mut f32,
        g_dev: *const f32,
        m_dev: *mut f32,
        v_dev: *mut f32,
        n: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1_inv: f32,
        bc2_inv: f32,
    },

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
            Op::MatmulResidentBf16Nn { .. } => "matmul_resident_bf16_nn",
            Op::MatmulResidentBf16Nt { .. } => "matmul_resident_bf16_nt",
            Op::MatmulResidentBf16Tn { .. } => "matmul_resident_bf16_tn",
            Op::MatvecResidentBf16 { .. } => "matvec_resident_bf16",
            Op::DequantQ4KResident { .. } => "dequant_q4k_resident",
            Op::RmsNormResident { .. } => "rms_norm_resident",
            Op::RmsNormBackwardResident { .. } => "rms_norm_backward_resident",
            Op::RopeBackwardResident { .. } => "rope_backward_resident",
            Op::LayerNormResident { .. } => "layer_norm_resident",
            Op::SoftmaxResident { .. } => "softmax_resident",
            Op::ActivationResident { .. } => "activation_resident",
            Op::GluResident { .. } => "glu_resident",
            Op::PerNeuronGluBatchedResident { .. } => "per_neuron_glu_batched_resident",
            Op::OpTensorResident { .. } => "op_tensor_resident",
            Op::LayerNormBackwardResident { .. } => "layer_norm_backward_resident",
            Op::SoftmaxBackwardResident { .. } => "softmax_backward_resident",
            Op::ActivationBackwardResident { .. } => "activation_backward_resident",
            Op::GluBackwardResident { .. } => "glu_backward_resident",
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
            Op::AdamWResident { .. } => "adamw_resident",
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

// ─── bf16 helpers ────────────────────────────────────────────
//
// bf16 ("brain float 16") is the top 16 bits of an IEEE-754 fp32 bit
// pattern: 1 sign bit + 8 exponent bits + 7 mantissa bits. Same dynamic
// range as fp32, ~3 decimal digits of precision.
//
// Rust has no native `bf16` primitive in stable, so we transport it as
// `u16`. Conversion is bitwise — no library dependency.

/// Convert an `f32` to bf16 (stored as `u16`).
///
/// NaN-safe: any fp32 NaN maps to a quiet bf16 NaN. Round-to-nearest-even
/// per `(bits + 0x7fff + ((bits >> 16) & 1)) >> 16` — the standard
/// bias-by-half-ULP-then-truncate idiom that breaks ties toward even.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        // NaN — collapse to a canonical quiet NaN so we never produce
        // an sNaN that traps on FPUs that honour the signal bit.
        return 0x7fc0;
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// Convert a bf16 (`u16`) back to `f32`.
///
/// Lossless: bf16 is a strict prefix of fp32, so the round-trip is just
/// a shift-and-zero-extend of the original 16 bits into the high half
/// of a fresh fp32 bit pattern.
#[inline]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}
