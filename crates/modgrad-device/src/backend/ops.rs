//! Function façade over the [`Op`] IR + [`registry`] dispatcher.
//!
//! Callers write `ops::matmul_nt(a, w, y, Some(bias), m, k, n)?` instead
//! of the longer `registry().dispatch(&mut Op::MatmulNT { .. })?`
//! boilerplate. Each function builds the corresponding `Op` variant
//! internally and routes through the process-wide backend registry.
//!
//! This is the JAX-style "server as a function" UX surface: each op is a
//! plain function call, not an IR term the caller assembles. The
//! underlying [`Op`] enum, [`registry`], and [`Backend`](super::Backend)
//! trait remain the extension surface for SDK authors — parity tests and
//! new backends still speak to the IR directly.
//!
//! Function signatures mirror the [`Op`] variant fields verbatim
//! (same order, same types, same `'a` lifetime behavior) so migrating
//! between the two forms is mechanical. Variants that already carry an
//! args struct ([`AdamWArgs`], [`SyncBackwardScatterArgs`]) take that
//! struct directly.
//!
//! # Error propagation
//!
//! Every façade fn returns `Result<(), BackendError>` — dispatch failures
//! (hipBLAS non-zero, KFD kernel crash, `Unsupported` slipping past a
//! backend's `supports()` check) surface as an error the caller owns.
//!
//! This matters because a `.expect(...)` *inside* a façade fn would
//! unwind directly through the live hipBLAS / cudarc C FFI frame below
//! it — UB-adjacent on most toolchains, and the exact hazard cudarc's
//! own `OnceLock` panic-propagation workaround was built to avoid.
//! Pushing the panic point up to caller code (training loops, optimiser
//! steps) keeps unwinds on pure-Rust stacks.
//!
//! Callers that truly cannot recover still write `.expect("<op> dispatch")`
//! at the call site — the failure mode is identical, but it now lives
//! one frame above the FFI boundary.
//!
//! [`Op`]: super::Op
//! [`registry`]: super::registry
//! [`AdamWArgs`]: super::AdamWArgs
//! [`SyncBackwardScatterArgs`]: super::SyncBackwardScatterArgs

use super::op::{ActivationMode, AdamWArgs, BinaryOpKind, Op, QuantKind, SyncBackwardScatterArgs};
use super::BackendError;

// ─── dense linear algebra ────────────────────────────────

/// `C = A @ B`. See [`Op::MatmulNN`](super::Op::MatmulNN).
#[inline]
pub fn matmul_nn(
    a: &[f32], b: &[f32], out: &mut [f32], bias: Option<&[f32]>,
    m: usize, k: usize, n: usize,
) -> Result<(), BackendError> {
    let mut op = Op::MatmulNN { a, b, out, bias, m, k, n };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// `C = A @ B^T`. See [`Op::MatmulNT`](super::Op::MatmulNT).
#[inline]
pub fn matmul_nt(
    a: &[f32], b: &[f32], out: &mut [f32], bias: Option<&[f32]>,
    m: usize, k: usize, n: usize,
) -> Result<(), BackendError> {
    let mut op = Op::MatmulNT { a, b, out, bias, m, k, n };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// `C = A^T @ B`. See [`Op::MatmulTN`](super::Op::MatmulTN).
#[inline]
pub fn matmul_tn(
    a: &[f32], b: &[f32], out: &mut [f32], bias: Option<&[f32]>,
    m: usize, k: usize, n: usize,
) -> Result<(), BackendError> {
    let mut op = Op::MatmulTN { a, b, out, bias, m, k, n };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// Forward matvec. See [`Op::Matvec`](super::Op::Matvec).
#[inline]
pub fn matvec(
    x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32],
    out_dim: usize, in_dim: usize, quant: QuantKind,
) -> Result<(), BackendError> {
    let mut op = Op::Matvec { x, weight, bias, out, out_dim, in_dim, quant };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// Transposed matvec. See [`Op::MatvecT`](super::Op::MatvecT).
#[inline]
pub fn matvec_t(
    d_out: &[f32], weight: &[f32], d_input: &mut [f32],
    out_dim: usize, in_dim: usize,
) -> Result<(), BackendError> {
    let mut op = Op::MatvecT { d_out, weight, d_input, out_dim, in_dim };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// Device-resident matvec: `out = weight @ x + bias` where all
/// operands are hip device pointers. Caller must guarantee the
/// pointers stay valid for the duration of this call (typically
/// they come from owned `HipBuffer`s).
///
/// # Safety
/// Caller is responsible for:
/// - All pointers are valid hip-device pointers from the same context.
/// - `out_dev` has at least `out_dim * 4` bytes allocated.
/// - `weight_dev` has `out_dim * in_dim * 4` bytes laid out row-major.
/// - `bias_dev` has `out_dim * 4` bytes.
/// - `x_dev` has at least `in_dim * 4` bytes.
///
/// Backends that don't support resident dispatch (CPU, KFD, others)
/// return `Unsupported` and the caller must fall back to the
/// host-slice `matvec` path.
#[inline]
pub unsafe fn matvec_resident(
    x_dev: *const f32,
    weight_dev: *const f32,
    bias_dev: *const f32,
    out_dev: *mut f32,
    out_dim: usize,
    in_dim: usize,
) -> Result<(), BackendError> {
    let mut op = Op::MatvecResident {
        x_dev, weight_dev, bias_dev, out_dev, out_dim, in_dim,
    };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// Device-resident LayerNorm forward.
/// See [`Op::LayerNormResident`](super::Op::LayerNormResident).
///
/// # Safety
/// Caller is responsible for:
/// - All pointers are valid hip-device pointers from the same context.
/// - `x_dev` and `y_dev` cover at least `n * normalized_size * 4` bytes.
/// - `weight_dev` and `bias_dev` each cover `normalized_size * 4` bytes.
/// - The pointers stay valid for the duration of this call.
///
/// CPU and other non-resident backends return `Unsupported`; callers
/// must fall back to the host-slice [`layer_norm_fwd`] when only CPU
/// dispatch is available.
#[inline]
pub unsafe fn layer_norm_resident(
    x_dev: *const f32,
    weight_dev: *const f32,
    bias_dev: *const f32,
    y_dev: *mut f32,
    n: usize,
    normalized_size: usize,
    epsilon: f32,
) -> Result<(), BackendError> {
    let mut op = Op::LayerNormResident {
        x_dev, weight_dev, bias_dev, y_dev, n, normalized_size, epsilon,
    };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// Device-resident row-wise softmax (or log-softmax).
/// See [`Op::SoftmaxResident`](super::Op::SoftmaxResident).
///
/// # Safety
/// Caller is responsible for:
/// - All pointers are valid hip-device pointers from the same context.
/// - `x_dev` and `y_dev` cover at least `n_rows * row_len * 4` bytes.
/// - `x_dev` and `y_dev` may alias (in-place softmax is permitted).
/// - The pointers stay valid for the duration of this call.
#[inline]
pub unsafe fn softmax_resident(
    x_dev: *const f32,
    y_dev: *mut f32,
    n_rows: usize,
    row_len: usize,
    log: bool,
) -> Result<(), BackendError> {
    let mut op = Op::SoftmaxResident { x_dev, y_dev, n_rows, row_len, log };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// Device-resident element-wise activation forward.
/// See [`Op::ActivationResident`](super::Op::ActivationResident) and
/// [`ActivationMode`].
///
/// # Safety
/// Caller is responsible for:
/// - All pointers are valid hip-device pointers from the same context.
/// - `x_dev` and `y_dev` cover at least `n * 4` bytes.
/// - `x_dev` and `y_dev` may alias for non-`Silu` modes; `Silu` reads
///   `x_dev` after writing `y_dev`, so they MUST be distinct buffers
///   (or the alias is harmless because the read-after-write reads the
///   original input that was just transformed in-place — the caller
///   should not assume that's the case across MIOpen versions).
/// - The pointers stay valid for the duration of this call.
#[inline]
pub unsafe fn activation_resident(
    x_dev: *const f32,
    y_dev: *mut f32,
    n: usize,
    mode: ActivationMode,
) -> Result<(), BackendError> {
    let mut op = Op::ActivationResident { x_dev, y_dev, n, mode };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// Device-resident GLU forward.
/// See [`Op::GluResident`](super::Op::GluResident).
///
/// # Safety
/// Caller is responsible for:
/// - All pointers are valid hip-device pointers from the same context.
/// - `x_dev` covers at least `n_rows * 2 * half_size * 4` bytes.
/// - `y_dev` covers at least `n_rows * half_size * 4` bytes.
/// - `x_dev` and `y_dev` are distinct buffers (GLU is not in-place).
/// - The pointers stay valid for the duration of this call.
#[inline]
pub unsafe fn glu_resident(
    x_dev: *const f32,
    y_dev: *mut f32,
    n_rows: usize,
    half_size: usize,
) -> Result<(), BackendError> {
    let mut op = Op::GluResident { x_dev, y_dev, n_rows, half_size };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// Device-resident binary element-wise op:
/// `c = op(alpha1 * a, alpha2 * b) + beta * c`.
/// See [`Op::OpTensorResident`](super::Op::OpTensorResident) and
/// [`BinaryOpKind`].
///
/// # Safety
/// Caller is responsible for:
/// - All pointers are valid hip-device pointers from the same context.
/// - `a_dev`, `b_dev`, `c_dev` each cover at least `n * 4` bytes.
/// - `c_dev` may alias `a_dev` or `b_dev` (read-modify-write is
///   permitted by MIOpen; the dispatch is read-then-write per element).
/// - The pointers stay valid for the duration of this call.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn op_tensor_resident(
    a_dev: *const f32,
    b_dev: *const f32,
    c_dev: *mut f32,
    n: usize,
    alpha1: f32,
    alpha2: f32,
    beta: f32,
    op: BinaryOpKind,
) -> Result<(), BackendError> {
    let mut variant = Op::OpTensorResident {
        a_dev, b_dev, c_dev, n, alpha1, alpha2, beta, op,
    };
    super::registry().dispatch(&mut variant)?;
    Ok(())
}

/// Accumulating outer product. See [`Op::OuterProductAcc`](super::Op::OuterProductAcc).
#[inline]
pub fn outer_product_acc(
    a: &[f32], b: &[f32], accum: &mut [f32], m: usize, n: usize,
) -> Result<(), BackendError> {
    let mut op = Op::OuterProductAcc { a, b, accum, m, n };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

// ─── normalization and activations ───────────────────────

/// LayerNorm forward. See [`Op::LayerNormFwd`](super::Op::LayerNormFwd).
#[inline]
pub fn layer_norm_fwd(
    x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32],
    cache: Option<&mut [f32]>, n_rows: usize, n_cols: usize,
) -> Result<(), BackendError> {
    let mut op = Op::LayerNormFwd { x, gamma, beta, out, cache, n_rows, n_cols };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// LayerNorm backward. See [`Op::LayerNormBwd`](super::Op::LayerNormBwd).
#[inline]
pub fn layer_norm_bwd(
    d_out: &[f32], x: &[f32], gamma: &[f32], cache: &[f32],
    d_x: &mut [f32], d_gamma: &mut [f32], d_beta: &mut [f32],
    n_rows: usize, n_cols: usize,
) -> Result<(), BackendError> {
    let mut op = Op::LayerNormBwd {
        d_out, x, gamma, cache, d_x, d_gamma, d_beta, n_rows, n_cols,
    };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// Fused LayerNorm + SiLU forward. See [`Op::LnSiluFwd`](super::Op::LnSiluFwd).
#[inline]
pub fn ln_silu_fwd(
    x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32],
    cache: Option<&mut [f32]>, n_rows: usize, n_cols: usize,
) -> Result<(), BackendError> {
    let mut op = Op::LnSiluFwd { x, gamma, beta, out, cache, n_rows, n_cols };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// SiLU forward. See [`Op::SiluFwd`](super::Op::SiluFwd).
#[inline]
pub fn silu_fwd(x: &[f32], out: &mut [f32]) -> Result<(), BackendError> {
    let mut op = Op::SiluFwd { x, out };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// SiLU forward, in-place. See [`Op::SiluFwdInplace`](super::Op::SiluFwdInplace).
#[inline]
pub fn silu_fwd_inplace(x: &mut [f32]) -> Result<(), BackendError> {
    let mut op = Op::SiluFwdInplace { x };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// SiLU backward. See [`Op::SiluBwd`](super::Op::SiluBwd).
#[inline]
pub fn silu_bwd(d_out: &[f32], x: &[f32], d_x: &mut [f32]) -> Result<(), BackendError> {
    let mut op = Op::SiluBwd { d_out, x, d_x };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// GLU forward. See [`Op::GluFwd`](super::Op::GluFwd).
#[inline]
pub fn glu_fwd(x: &[f32], out: &mut [f32]) -> Result<(), BackendError> {
    let mut op = Op::GluFwd { x, out };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// GLU backward. See [`Op::GluBwd`](super::Op::GluBwd).
#[inline]
pub fn glu_bwd(d_out: &[f32], x: &[f32], d_x: &mut [f32]) -> Result<(), BackendError> {
    let mut op = Op::GluBwd { d_out, x, d_x };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// GLU backward, per-neuron. See [`Op::PerNeuronGluBwd`](super::Op::PerNeuronGluBwd).
#[inline]
pub fn per_neuron_glu_bwd(
    d_out: &[f32], x: &[f32], d_x: &mut [f32],
    n_neurons: usize, feat_per_neuron: usize,
) -> Result<(), BackendError> {
    let mut op = Op::PerNeuronGluBwd { d_out, x, d_x, n_neurons, feat_per_neuron };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

// ─── reductions and optimizer steps ───────────────────────

/// Reduce sum of squares. See [`Op::ReduceL2Sq`](super::Op::ReduceL2Sq).
#[inline]
pub fn reduce_l2_sq(x: &[f32], out: &mut [f32]) -> Result<(), BackendError> {
    let mut op = Op::ReduceL2Sq { x, out };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// SGD update: `w -= lr * g`. See [`Op::SgdUpdate`](super::Op::SgdUpdate).
#[inline]
pub fn sgd_update(w: &mut [f32], g: &[f32], lr: f32) -> Result<(), BackendError> {
    let mut op = Op::SgdUpdate { w, g, lr };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// AdamW update. See [`Op::AdamW`](super::Op::AdamW) and [`AdamWArgs`].
#[inline]
pub fn adamw(args: AdamWArgs<'_>) -> Result<(), BackendError> {
    let mut op = Op::AdamW(args);
    super::registry().dispatch(&mut op)?;
    Ok(())
}

// ─── CTM-specific ─────────────────────────────────────────

/// SuperLinear forward. See [`Op::SuperLinearFwd`](super::Op::SuperLinearFwd).
#[inline]
pub fn super_linear_fwd(
    trace: &[f32], weights: &[f32], biases: &[f32], out: &mut [f32],
    cache: Option<&mut [f32]>,
    d_model: usize, memory_length: usize, out_per: usize,
) -> Result<(), BackendError> {
    let mut op = Op::SuperLinearFwd {
        trace, weights, biases, out, cache, d_model, memory_length, out_per,
    };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// SuperLinear backward — gradient w.r.t. weights.
/// See [`Op::SuperLinearBwdDw`](super::Op::SuperLinearBwdDw).
#[inline]
pub fn super_linear_bwd_dw(
    d_out: &[f32], trace: &[f32], d_weights: &mut [f32],
    d_model: usize, memory_length: usize, out_per: usize,
) -> Result<(), BackendError> {
    let mut op = Op::SuperLinearBwdDw {
        d_out, trace, d_weights, d_model, memory_length, out_per,
    };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// SuperLinear backward — gradient w.r.t. input trace.
/// See [`Op::SuperLinearBwdDx`](super::Op::SuperLinearBwdDx).
#[inline]
pub fn super_linear_bwd_dx(
    d_out: &[f32], weights: &[f32], d_trace: &mut [f32],
    d_model: usize, memory_length: usize, out_per: usize,
) -> Result<(), BackendError> {
    let mut op = Op::SuperLinearBwdDx {
        d_out, weights, d_trace, d_model, memory_length, out_per,
    };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// CTM neuron-pair synchronization forward.
/// See [`Op::SyncUpdateFwd`](super::Op::SyncUpdateFwd).
#[inline]
pub fn sync_update_fwd(
    h: &[f32], pairs_left: &[u32], pairs_right: &[u32], decay: &[f32],
    sync_state: &mut [f32], sync_out: &mut [f32], n_pairs: usize,
) -> Result<(), BackendError> {
    let mut op = Op::SyncUpdateFwd {
        h, pairs_left, pairs_right, decay, sync_state, sync_out, n_pairs,
    };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// CTM synchronization backward scatter.
/// See [`Op::SyncBackwardScatter`](super::Op::SyncBackwardScatter) and
/// [`SyncBackwardScatterArgs`].
#[inline]
pub fn sync_backward_scatter(args: SyncBackwardScatterArgs<'_>) -> Result<(), BackendError> {
    let mut op = Op::SyncBackwardScatter(args);
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// Rotate trace memory in-place.
/// See [`Op::TraceRotateInplace`](super::Op::TraceRotateInplace).
#[inline]
pub fn trace_rotate_inplace(
    trace: &mut [f32], new_val: &[f32],
    d_model: usize, memory_length: usize,
) -> Result<(), BackendError> {
    let mut op = Op::TraceRotateInplace { trace, new_val, d_model, memory_length };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

// ─── fused (Stage 1) ──────────────────────────────────────

/// Fused synapse forward (matvec → GLU → SiLU → LayerNorm).
/// See [`Op::SynapseForward`](super::Op::SynapseForward).
#[inline]
pub fn synapse_forward(
    weight: &[f32], bias: &[f32], x: &[f32], out: &mut [f32],
    out_dim: usize, in_dim: usize,
) -> Result<(), BackendError> {
    let mut op = Op::SynapseForward { weight, bias, x, out, out_dim, in_dim };
    super::registry().dispatch(&mut op)?;
    Ok(())
}

/// In-place row-wise LayerNorm (no affine).
/// See [`Op::LayerNormInplace`](super::Op::LayerNormInplace).
#[inline]
pub fn layer_norm_inplace(x: &mut [f32], n_rows: usize, n_cols: usize) -> Result<(), BackendError> {
    let mut op = Op::LayerNormInplace { x, n_rows, n_cols };
    super::registry().dispatch(&mut op)?;
    Ok(())
}
