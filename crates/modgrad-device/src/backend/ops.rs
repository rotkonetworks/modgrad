//! Function façade over the [`Op`] IR + [`registry`] dispatcher.
//!
//! Callers write `ops::matmul_nt(a, w, y, Some(bias), m, k, n)` instead of
//! the longer `registry().dispatch(&mut Op::MatmulNT { .. }).expect(..)`
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
//! Panics on dispatch failure — matching what every caller does today
//! via `.expect("<op_name> dispatch")`. Callers who need fallible
//! dispatch should use the IR layer directly.
//!
//! [`Op`]: super::Op
//! [`registry`]: super::registry
//! [`AdamWArgs`]: super::AdamWArgs
//! [`SyncBackwardScatterArgs`]: super::SyncBackwardScatterArgs

use super::op::{AdamWArgs, Op, QuantKind, SyncBackwardScatterArgs};

// ─── dense linear algebra ────────────────────────────────

/// `C = A @ B`. See [`Op::MatmulNN`](super::Op::MatmulNN).
#[inline]
pub fn matmul_nn(
    a: &[f32], b: &[f32], out: &mut [f32], bias: Option<&[f32]>,
    m: usize, k: usize, n: usize,
) {
    let mut op = Op::MatmulNN { a, b, out, bias, m, k, n };
    super::registry().dispatch(&mut op).expect("matmul_nn dispatch");
}

/// `C = A @ B^T`. See [`Op::MatmulNT`](super::Op::MatmulNT).
#[inline]
pub fn matmul_nt(
    a: &[f32], b: &[f32], out: &mut [f32], bias: Option<&[f32]>,
    m: usize, k: usize, n: usize,
) {
    let mut op = Op::MatmulNT { a, b, out, bias, m, k, n };
    super::registry().dispatch(&mut op).expect("matmul_nt dispatch");
}

/// `C = A^T @ B`. See [`Op::MatmulTN`](super::Op::MatmulTN).
#[inline]
pub fn matmul_tn(
    a: &[f32], b: &[f32], out: &mut [f32], bias: Option<&[f32]>,
    m: usize, k: usize, n: usize,
) {
    let mut op = Op::MatmulTN { a, b, out, bias, m, k, n };
    super::registry().dispatch(&mut op).expect("matmul_tn dispatch");
}

/// Forward matvec. See [`Op::Matvec`](super::Op::Matvec).
#[inline]
pub fn matvec(
    x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32],
    out_dim: usize, in_dim: usize, quant: QuantKind,
) {
    let mut op = Op::Matvec { x, weight, bias, out, out_dim, in_dim, quant };
    super::registry().dispatch(&mut op).expect("matvec dispatch");
}

/// Transposed matvec. See [`Op::MatvecT`](super::Op::MatvecT).
#[inline]
pub fn matvec_t(
    d_out: &[f32], weight: &[f32], d_input: &mut [f32],
    out_dim: usize, in_dim: usize,
) {
    let mut op = Op::MatvecT { d_out, weight, d_input, out_dim, in_dim };
    super::registry().dispatch(&mut op).expect("matvec_t dispatch");
}

/// Accumulating outer product. See [`Op::OuterProductAcc`](super::Op::OuterProductAcc).
#[inline]
pub fn outer_product_acc(
    a: &[f32], b: &[f32], accum: &mut [f32], m: usize, n: usize,
) {
    let mut op = Op::OuterProductAcc { a, b, accum, m, n };
    super::registry().dispatch(&mut op).expect("outer_product_acc dispatch");
}

// ─── normalization and activations ───────────────────────

/// LayerNorm forward. See [`Op::LayerNormFwd`](super::Op::LayerNormFwd).
#[inline]
pub fn layer_norm_fwd(
    x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32],
    cache: Option<&mut [f32]>, n_rows: usize, n_cols: usize,
) {
    let mut op = Op::LayerNormFwd { x, gamma, beta, out, cache, n_rows, n_cols };
    super::registry().dispatch(&mut op).expect("layer_norm_fwd dispatch");
}

/// LayerNorm backward. See [`Op::LayerNormBwd`](super::Op::LayerNormBwd).
#[inline]
pub fn layer_norm_bwd(
    d_out: &[f32], x: &[f32], gamma: &[f32], cache: &[f32],
    d_x: &mut [f32], d_gamma: &mut [f32], d_beta: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    let mut op = Op::LayerNormBwd {
        d_out, x, gamma, cache, d_x, d_gamma, d_beta, n_rows, n_cols,
    };
    super::registry().dispatch(&mut op).expect("layer_norm_bwd dispatch");
}

/// Fused LayerNorm + SiLU forward. See [`Op::LnSiluFwd`](super::Op::LnSiluFwd).
#[inline]
pub fn ln_silu_fwd(
    x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32],
    cache: Option<&mut [f32]>, n_rows: usize, n_cols: usize,
) {
    let mut op = Op::LnSiluFwd { x, gamma, beta, out, cache, n_rows, n_cols };
    super::registry().dispatch(&mut op).expect("ln_silu_fwd dispatch");
}

/// SiLU forward. See [`Op::SiluFwd`](super::Op::SiluFwd).
#[inline]
pub fn silu_fwd(x: &[f32], out: &mut [f32]) {
    let mut op = Op::SiluFwd { x, out };
    super::registry().dispatch(&mut op).expect("silu_fwd dispatch");
}

/// SiLU forward, in-place. See [`Op::SiluFwdInplace`](super::Op::SiluFwdInplace).
#[inline]
pub fn silu_fwd_inplace(x: &mut [f32]) {
    let mut op = Op::SiluFwdInplace { x };
    super::registry().dispatch(&mut op).expect("silu_fwd_inplace dispatch");
}

/// SiLU backward. See [`Op::SiluBwd`](super::Op::SiluBwd).
#[inline]
pub fn silu_bwd(d_out: &[f32], x: &[f32], d_x: &mut [f32]) {
    let mut op = Op::SiluBwd { d_out, x, d_x };
    super::registry().dispatch(&mut op).expect("silu_bwd dispatch");
}

/// GLU forward. See [`Op::GluFwd`](super::Op::GluFwd).
#[inline]
pub fn glu_fwd(x: &[f32], out: &mut [f32]) {
    let mut op = Op::GluFwd { x, out };
    super::registry().dispatch(&mut op).expect("glu_fwd dispatch");
}

/// GLU backward. See [`Op::GluBwd`](super::Op::GluBwd).
#[inline]
pub fn glu_bwd(d_out: &[f32], x: &[f32], d_x: &mut [f32]) {
    let mut op = Op::GluBwd { d_out, x, d_x };
    super::registry().dispatch(&mut op).expect("glu_bwd dispatch");
}

/// GLU backward, per-neuron. See [`Op::PerNeuronGluBwd`](super::Op::PerNeuronGluBwd).
#[inline]
pub fn per_neuron_glu_bwd(
    d_out: &[f32], x: &[f32], d_x: &mut [f32],
    n_neurons: usize, feat_per_neuron: usize,
) {
    let mut op = Op::PerNeuronGluBwd { d_out, x, d_x, n_neurons, feat_per_neuron };
    super::registry().dispatch(&mut op).expect("per_neuron_glu_bwd dispatch");
}

// ─── reductions and optimizer steps ───────────────────────

/// Reduce sum of squares. See [`Op::ReduceL2Sq`](super::Op::ReduceL2Sq).
#[inline]
pub fn reduce_l2_sq(x: &[f32], out: &mut [f32]) {
    let mut op = Op::ReduceL2Sq { x, out };
    super::registry().dispatch(&mut op).expect("reduce_l2_sq dispatch");
}

/// SGD update: `w -= lr * g`. See [`Op::SgdUpdate`](super::Op::SgdUpdate).
#[inline]
pub fn sgd_update(w: &mut [f32], g: &[f32], lr: f32) {
    let mut op = Op::SgdUpdate { w, g, lr };
    super::registry().dispatch(&mut op).expect("sgd_update dispatch");
}

/// AdamW update. See [`Op::AdamW`](super::Op::AdamW) and [`AdamWArgs`].
#[inline]
pub fn adamw(args: AdamWArgs<'_>) {
    let mut op = Op::AdamW(args);
    super::registry().dispatch(&mut op).expect("adamw dispatch");
}

// ─── CTM-specific ─────────────────────────────────────────

/// SuperLinear forward. See [`Op::SuperLinearFwd`](super::Op::SuperLinearFwd).
#[inline]
pub fn super_linear_fwd(
    trace: &[f32], weights: &[f32], biases: &[f32], out: &mut [f32],
    cache: Option<&mut [f32]>,
    d_model: usize, memory_length: usize, out_per: usize,
) {
    let mut op = Op::SuperLinearFwd {
        trace, weights, biases, out, cache, d_model, memory_length, out_per,
    };
    super::registry().dispatch(&mut op).expect("super_linear_fwd dispatch");
}

/// SuperLinear backward — gradient w.r.t. weights.
/// See [`Op::SuperLinearBwdDw`](super::Op::SuperLinearBwdDw).
#[inline]
pub fn super_linear_bwd_dw(
    d_out: &[f32], trace: &[f32], d_weights: &mut [f32],
    d_model: usize, memory_length: usize, out_per: usize,
) {
    let mut op = Op::SuperLinearBwdDw {
        d_out, trace, d_weights, d_model, memory_length, out_per,
    };
    super::registry().dispatch(&mut op).expect("super_linear_bwd_dw dispatch");
}

/// SuperLinear backward — gradient w.r.t. input trace.
/// See [`Op::SuperLinearBwdDx`](super::Op::SuperLinearBwdDx).
#[inline]
pub fn super_linear_bwd_dx(
    d_out: &[f32], weights: &[f32], d_trace: &mut [f32],
    d_model: usize, memory_length: usize, out_per: usize,
) {
    let mut op = Op::SuperLinearBwdDx {
        d_out, weights, d_trace, d_model, memory_length, out_per,
    };
    super::registry().dispatch(&mut op).expect("super_linear_bwd_dx dispatch");
}

/// CTM neuron-pair synchronization forward.
/// See [`Op::SyncUpdateFwd`](super::Op::SyncUpdateFwd).
#[inline]
pub fn sync_update_fwd(
    h: &[f32], pairs_left: &[u32], pairs_right: &[u32], decay: &[f32],
    sync_state: &mut [f32], sync_out: &mut [f32], n_pairs: usize,
) {
    let mut op = Op::SyncUpdateFwd {
        h, pairs_left, pairs_right, decay, sync_state, sync_out, n_pairs,
    };
    super::registry().dispatch(&mut op).expect("sync_update_fwd dispatch");
}

/// CTM synchronization backward scatter.
/// See [`Op::SyncBackwardScatter`](super::Op::SyncBackwardScatter) and
/// [`SyncBackwardScatterArgs`].
#[inline]
pub fn sync_backward_scatter(args: SyncBackwardScatterArgs<'_>) {
    let mut op = Op::SyncBackwardScatter(args);
    super::registry().dispatch(&mut op).expect("sync_backward_scatter dispatch");
}

/// Rotate trace memory in-place.
/// See [`Op::TraceRotateInplace`](super::Op::TraceRotateInplace).
#[inline]
pub fn trace_rotate_inplace(
    trace: &mut [f32], new_val: &[f32],
    d_model: usize, memory_length: usize,
) {
    let mut op = Op::TraceRotateInplace { trace, new_val, d_model, memory_length };
    super::registry().dispatch(&mut op).expect("trace_rotate_inplace dispatch");
}

// ─── fused (Stage 1) ──────────────────────────────────────

/// Fused synapse forward (matvec → GLU → SiLU → LayerNorm).
/// See [`Op::SynapseForward`](super::Op::SynapseForward).
#[inline]
pub fn synapse_forward(
    weight: &[f32], bias: &[f32], x: &[f32], out: &mut [f32],
    out_dim: usize, in_dim: usize,
) {
    let mut op = Op::SynapseForward { weight, bias, x, out, out_dim, in_dim };
    super::registry().dispatch(&mut op).expect("synapse_forward dispatch");
}

/// In-place row-wise LayerNorm (no affine).
/// See [`Op::LayerNormInplace`](super::Op::LayerNormInplace).
#[inline]
pub fn layer_norm_inplace(x: &mut [f32], n_rows: usize, n_cols: usize) {
    let mut op = Op::LayerNormInplace { x, n_rows, n_cols };
    super::registry().dispatch(&mut op).expect("layer_norm_inplace dispatch");
}
