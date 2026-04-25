//! CPU reference backend. Always registered, handles every `Op` variant.
//!
//! Performance is not the primary goal — correctness is. These
//! implementations are the ground truth for the parity harness in
//! Task 1.3; GPU backends are considered correct exactly when their
//! output matches this one to a small tolerance.
//!
//! Parallelism via rayon where it's a clear win (matmul, sync, outer
//! product). Small ops stay serial because rayon overhead dominates.

use rayon::prelude::*;

use super::{AdamWArgs, Backend, BackendError, BufferBackend, ComputeCtx, DeviceInfo, DeviceKind, HostBuffer, Op, QuantKind, SyncBackwardScatterArgs};

/// Always-available fallback. Handles every op via pure Rust / rayon.
pub struct CpuBackend {
    threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self { threads: rayon::current_num_threads() }
    }
}

impl Default for CpuBackend {
    fn default() -> Self { Self::new() }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str { "cpu" }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            kind: DeviceKind::Cpu,
            name: format!("cpu ({} threads)", self.threads),
            total_mem_bytes: 0, // CPU has "infinite" from this layer's POV
            arch: None,
        }
    }

    fn supports(&self, op: &Op) -> bool {
        // CPU declines the Q4_K quantized path — nobody has written a
        // CPU Q4_K decoder in this crate, and the KFD kernel depends on
        // a specific weight layout that isn't trivially unpacked.
        // Future: port the GGUF Q4_K decode from modgrad-io.
        //
        // CPU also declines `MatvecResident` and the other Resident
        // variants — the operands are hip device pointers and CPU can't
        // dereference them. Callers must downgrade to the host-slice
        // variants for CPU dispatch.
        match op {
            Op::Matvec { quant: QuantKind::Q4K, .. } => false,
            Op::MatvecResident { .. } => false,
            Op::MatmulResidentNN { .. } => false,
            Op::MatmulResidentNT { .. } => false,
            Op::MatmulResidentTN { .. } => false,
            Op::MatmulResidentBf16Nn { .. } => false,
            Op::MatmulResidentBf16Nt { .. } => false,
            Op::MatmulResidentBf16Tn { .. } => false,
            Op::MatvecResidentBf16 { .. } => false,
            Op::RmsNormResident { .. } => false,
            Op::DequantQ4KResident { .. } => false,
            Op::LayerNormResident { .. } => false,
            Op::SoftmaxResident { .. } => false,
            Op::ActivationResident { .. } => false,
            Op::GluResident { .. } => false,
            Op::OpTensorResident { .. } => false,
            Op::LayerNormBackwardResident { .. } => false,
            Op::SoftmaxBackwardResident { .. } => false,
            Op::ActivationBackwardResident { .. } => false,
            Op::GluBackwardResident { .. } => false,
            _ => true,
        }
    }

    fn dispatch(&self, op: &mut Op) -> Result<(), BackendError> {
        match op {
            Op::MatmulNN { a, b, out, bias, m, k, n } => {
                matmul(a, b, out, *m, *k, *n, MatmulKind::NN);
                apply_bias(out, *bias, *m, *n);
                Ok(())
            }
            Op::MatmulNT { a, b, out, bias, m, k, n } => {
                matmul(a, b, out, *m, *k, *n, MatmulKind::NT);
                apply_bias(out, *bias, *m, *n);
                Ok(())
            }
            Op::MatmulTN { a, b, out, bias, m, k, n } => {
                matmul(a, b, out, *m, *k, *n, MatmulKind::TN);
                apply_bias(out, *bias, *m, *n);
                Ok(())
            }
            Op::Matvec { x, weight, bias, out, out_dim, in_dim, quant: QuantKind::F32 } => {
                matvec(x, weight, bias, out, *out_dim, *in_dim);
                Ok(())
            }
            Op::Matvec { quant: QuantKind::Q4K, .. } => Err(BackendError::Unsupported {
                op: "matvec",
                backend: "cpu",
            }),
            Op::MatvecResident { .. } => Err(BackendError::Unsupported {
                op: "matvec_resident",
                backend: "cpu",
            }),
            Op::MatmulResidentNN { .. } => Err(BackendError::Unsupported {
                op: "matmul_resident_nn",
                backend: "cpu",
            }),
            Op::MatmulResidentNT { .. } => Err(BackendError::Unsupported {
                op: "matmul_resident_nt",
                backend: "cpu",
            }),
            Op::MatmulResidentTN { .. } => Err(BackendError::Unsupported {
                op: "matmul_resident_tn",
                backend: "cpu",
            }),
            Op::MatmulResidentBf16Nn { .. } => Err(BackendError::Unsupported {
                op: "matmul_resident_bf16_nn",
                backend: "cpu",
            }),
            Op::MatmulResidentBf16Nt { .. } => Err(BackendError::Unsupported {
                op: "matmul_resident_bf16_nt",
                backend: "cpu",
            }),
            Op::MatmulResidentBf16Tn { .. } => Err(BackendError::Unsupported {
                op: "matmul_resident_bf16_tn",
                backend: "cpu",
            }),
            Op::MatvecResidentBf16 { .. } => Err(BackendError::Unsupported {
                op: "matvec_resident_bf16",
                backend: "cpu",
            }),
            Op::RmsNormResident { .. } => Err(BackendError::Unsupported {
                op: "rms_norm_resident",
                backend: "cpu",
            }),
            Op::DequantQ4KResident { .. } => Err(BackendError::Unsupported {
                op: "dequant_q4k_resident",
                backend: "cpu",
            }),
            Op::LayerNormResident { .. } => Err(BackendError::Unsupported {
                op: "layer_norm_resident",
                backend: "cpu",
            }),
            Op::SoftmaxResident { .. } => Err(BackendError::Unsupported {
                op: "softmax_resident",
                backend: "cpu",
            }),
            Op::ActivationResident { .. } => Err(BackendError::Unsupported {
                op: "activation_resident",
                backend: "cpu",
            }),
            Op::GluResident { .. } => Err(BackendError::Unsupported {
                op: "glu_resident",
                backend: "cpu",
            }),
            Op::OpTensorResident { .. } => Err(BackendError::Unsupported {
                op: "op_tensor_resident",
                backend: "cpu",
            }),
            Op::LayerNormBackwardResident { .. } => Err(BackendError::Unsupported {
                op: "layer_norm_backward_resident",
                backend: "cpu",
            }),
            Op::SoftmaxBackwardResident { .. } => Err(BackendError::Unsupported {
                op: "softmax_backward_resident",
                backend: "cpu",
            }),
            Op::ActivationBackwardResident { .. } => Err(BackendError::Unsupported {
                op: "activation_backward_resident",
                backend: "cpu",
            }),
            Op::GluBackwardResident { .. } => Err(BackendError::Unsupported {
                op: "glu_backward_resident",
                backend: "cpu",
            }),
            Op::MatvecT { d_out, weight, d_input, out_dim, in_dim } => {
                matvec_t(d_out, weight, d_input, *out_dim, *in_dim);
                Ok(())
            }
            Op::OuterProductAcc { a, b, accum, m, n } => {
                outer_product_acc(a, b, accum, *m, *n);
                Ok(())
            }

            Op::LayerNormFwd { x, gamma, beta, out, cache, n_rows, n_cols } => {
                layer_norm_fwd(x, gamma, beta, out, cache.as_deref_mut(), *n_rows, *n_cols);
                Ok(())
            }
            Op::LayerNormBwd { d_out, x, gamma, cache, d_x, d_gamma, d_beta, n_rows, n_cols } => {
                layer_norm_bwd(d_out, x, gamma, cache, d_x, d_gamma, d_beta, *n_rows, *n_cols);
                Ok(())
            }
            Op::LnSiluFwd { x, gamma, beta, out, cache, n_rows, n_cols } => {
                ln_silu_fwd(x, gamma, beta, out, cache.as_deref_mut(), *n_rows, *n_cols);
                Ok(())
            }
            Op::SiluFwd { x, out } => { silu_fwd(x, out); Ok(()) }
            Op::SiluFwdInplace { x } => { silu_fwd_inplace(x); Ok(()) }
            Op::SiluBwd { d_out, x, d_x } => { silu_bwd(d_out, x, d_x); Ok(()) }
            Op::GluFwd { x, out } => { glu_fwd(x, out); Ok(()) }
            Op::GluBwd { d_out, x, d_x } => { glu_bwd(d_out, x, d_x); Ok(()) }
            Op::PerNeuronGluBwd { d_out, x, d_x, n_neurons, feat_per_neuron } => {
                per_neuron_glu_bwd(d_out, x, d_x, *n_neurons, *feat_per_neuron);
                Ok(())
            }

            Op::ReduceL2Sq { x, out } => {
                out[0] = x.iter().map(|v| v * v).sum();
                Ok(())
            }
            Op::SgdUpdate { w, g, lr } => {
                let lr = *lr;
                for (w, g) in w.iter_mut().zip(g.iter()) { *w -= lr * *g; }
                Ok(())
            }
            Op::AdamW(args) => {
                adamw(args);
                Ok(())
            }

            Op::SuperLinearFwd { trace, weights, biases, out, cache, d_model, memory_length, out_per } => {
                super_linear_fwd(trace, weights, biases, out, cache.as_deref_mut(), *d_model, *memory_length, *out_per);
                Ok(())
            }
            Op::SuperLinearBwdDw { d_out, trace, d_weights, d_model, memory_length, out_per } => {
                super_linear_bwd_dw(d_out, trace, d_weights, *d_model, *memory_length, *out_per);
                Ok(())
            }
            Op::SuperLinearBwdDx { d_out, weights, d_trace, d_model, memory_length, out_per } => {
                super_linear_bwd_dx(d_out, weights, d_trace, *d_model, *memory_length, *out_per);
                Ok(())
            }

            Op::SyncUpdateFwd { h, pairs_left, pairs_right, decay, sync_state, sync_out, n_pairs } => {
                sync_update_fwd(h, pairs_left, pairs_right, decay, sync_state, sync_out, *n_pairs);
                Ok(())
            }
            Op::SyncBackwardScatter(args) => {
                sync_backward_scatter(args);
                Ok(())
            }
            Op::TraceRotateInplace { trace, new_val, d_model, memory_length } => {
                trace_rotate_inplace(trace, new_val, *d_model, *memory_length);
                Ok(())
            }

            Op::SynapseForward { weight, bias, x, out, out_dim, in_dim } => {
                synapse_forward(weight, bias, x, out, *out_dim, *in_dim);
                Ok(())
            }
            Op::LayerNormInplace { x, n_rows, n_cols } => {
                layer_norm_inplace(x, *n_rows, *n_cols);
                Ok(())
            }
        }
    }
}

/// CPU backend's device-resident "Buffer" is just host memory — a
/// `HostBuffer` (a `Vec<f32>` newtype). Explicit impl is kept simple:
/// zero-allocate and hand back. Not strictly required (the default
/// `HostBuffer` is what every non-GPU backend uses) but making it
/// explicit documents intent at the module level.
impl BufferBackend for CpuBackend {
    type Buffer = HostBuffer;

    fn alloc_buffer(&self, n: usize) -> Result<HostBuffer, BackendError> {
        Ok(HostBuffer::new(n))
    }
}

/// CPU has nothing resembling a device arena or an async submit queue,
/// so both lifecycle hooks are genuine no-ops here. They exist to let
/// generic call sites write `ctx.arena_reset()` / `ctx.flush()` without
/// peeking at the concrete backend.
impl ComputeCtx<CpuBackend> {
    /// No-op on CPU — no arena to reset.
    pub fn arena_reset(&self) {}

    /// No-op on CPU — dispatch is synchronous, nothing to flush.
    pub fn flush(&self) {}
}

// ─── Free-function implementations ───────────────────────────

/// Private transpose selector for the shared matmul kernel. The public
/// `Op` splits NN/NT/TN into distinct variants; we collapse them here
/// so the inner loop stays one implementation instead of three copies.
#[derive(Clone, Copy)]
enum MatmulKind { NN, NT, TN }

#[inline]
fn apply_bias(out: &mut [f32], bias: Option<&[f32]>, m: usize, n: usize) {
    if let Some(bias) = bias {
        // Broadcast bias across rows: out[r, c] += bias[c]
        for r in 0..m {
            let row = &mut out[r * n..(r + 1) * n];
            for c in 0..n { row[c] += bias[c]; }
        }
    }
}

fn matmul(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize, kind: MatmulKind) {
    // Parallelize over output rows.
    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let (a_val, b_val) = match kind {
                    MatmulKind::NN => (a[i * k + p], b[p * n + j]),
                    MatmulKind::TN => (a[p * m + i], b[p * n + j]),
                    MatmulKind::NT => (a[i * k + p], b[j * k + p]),
                };
                acc += a_val * b_val;
            }
            row[j] = acc;
        }
    });
}

/// Rayon scheduling adds ~1-3μs of overhead per `par_iter`. For shapes
/// where the whole compute takes <1μs (tiny matvecs inside a tick loop:
/// d_model=8, d_route=32 etc.) that overhead dwarfs the work. Anything
/// below this total-element threshold stays on the serial path.
const PAR_THRESHOLD: usize = 4096;

fn matvec(x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32], _out_dim: usize, in_dim: usize) {
    // out_dim is implicit in `out.len()` and `bias.len()`; parameter
    // kept for symmetry with GPU backends whose FFI wants it explicit.
    let n_rows = out.len();
    if n_rows * in_dim < PAR_THRESHOLD {
        for i in 0..n_rows {
            let row = &weight[i * in_dim..(i + 1) * in_dim];
            let mut acc = bias[i];
            for j in 0..in_dim { acc += row[j] * x[j]; }
            out[i] = acc;
        }
    } else {
        out.par_iter_mut().enumerate().for_each(|(i, y)| {
            let row = &weight[i * in_dim..(i + 1) * in_dim];
            let mut acc = bias[i];
            for j in 0..in_dim { acc += row[j] * x[j]; }
            *y = acc;
        });
    }
}

fn matvec_t(d_out: &[f32], weight: &[f32], d_input: &mut [f32], out_dim: usize, in_dim: usize) {
    // d_input[j] = sum_i weight[i, j] * d_out[i]
    if out_dim * in_dim < PAR_THRESHOLD {
        for j in 0..in_dim {
            let mut acc = 0.0f32;
            for i in 0..out_dim { acc += weight[i * in_dim + j] * d_out[i]; }
            d_input[j] = acc;
        }
    } else {
        d_input.par_iter_mut().enumerate().for_each(|(j, dx)| {
            let mut acc = 0.0f32;
            for i in 0..out_dim { acc += weight[i * in_dim + j] * d_out[i]; }
            *dx = acc;
        });
    }
}

fn outer_product_acc(a: &[f32], b: &[f32], accum: &mut [f32], m: usize, n: usize) {
    // accum[i, j] += a[i] * b[j]
    if m * n < PAR_THRESHOLD {
        for i in 0..m {
            let ai = a[i];
            let row = &mut accum[i * n..(i + 1) * n];
            for j in 0..n { row[j] += ai * b[j]; }
        }
    } else {
        accum.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let ai = a[i];
            for j in 0..n { row[j] += ai * b[j]; }
        });
    }
    let _ = m;
}

fn layer_norm_fwd(
    x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32],
    mut cache: Option<&mut [f32]>, n_rows: usize, n_cols: usize,
) {
    // `cache` mirrors the `SuperLinearFwd` pattern: when `Some`, we
    // persist mean/rstd for a subsequent backward; when `None`, we
    // still compute them (they feed the output) but drop them.
    for r in 0..n_rows {
        let row = &x[r * n_cols..(r + 1) * n_cols];
        let mean: f32 = row.iter().sum::<f32>() / n_cols as f32;
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n_cols as f32;
        let rstd = 1.0 / (var + 1e-5).sqrt();
        if let Some(ref mut c) = cache {
            c[r * 2] = mean;
            c[r * 2 + 1] = rstd;
        }
        let out_row = &mut out[r * n_cols..(r + 1) * n_cols];
        for c in 0..n_cols {
            out_row[c] = (row[c] - mean) * rstd * gamma[c] + beta[c];
        }
    }
}

fn layer_norm_bwd(
    d_out: &[f32], x: &[f32], gamma: &[f32], cache: &[f32],
    d_x: &mut [f32], d_gamma: &mut [f32], d_beta: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    // Reset gamma/beta grads
    for v in d_gamma.iter_mut() { *v = 0.0; }
    for v in d_beta.iter_mut() { *v = 0.0; }
    for r in 0..n_rows {
        let mean = cache[r * 2];
        let rstd = cache[r * 2 + 1];
        let row = &x[r * n_cols..(r + 1) * n_cols];
        let d_row_in = &d_out[r * n_cols..(r + 1) * n_cols];
        let d_row_out = &mut d_x[r * n_cols..(r + 1) * n_cols];
        // Accumulate d_gamma, d_beta
        for c in 0..n_cols {
            let x_hat = (row[c] - mean) * rstd;
            d_gamma[c] += d_row_in[c] * x_hat;
            d_beta[c] += d_row_in[c];
        }
        // d_x[c] = (1/N) * gamma[c] * rstd * (N*d_out[c] - sum_d - x_hat[c]*sum_dx)
        let n = n_cols as f32;
        let sum_d: f32 = (0..n_cols).map(|c| d_row_in[c] * gamma[c]).sum();
        let sum_dx: f32 = (0..n_cols).map(|c| {
            let x_hat = (row[c] - mean) * rstd;
            d_row_in[c] * gamma[c] * x_hat
        }).sum();
        for c in 0..n_cols {
            let x_hat = (row[c] - mean) * rstd;
            d_row_out[c] = gamma[c] * rstd * (d_row_in[c] - (sum_d + x_hat * sum_dx) / n);
        }
    }
}

fn ln_silu_fwd(
    x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32],
    cache: Option<&mut [f32]>, n_rows: usize, n_cols: usize,
) {
    // Compose: out = silu(layer_norm(x) * gamma + beta)
    layer_norm_fwd(x, gamma, beta, out, cache, n_rows, n_cols);
    for v in out.iter_mut() {
        let s = sigmoid(*v);
        *v *= s;
    }
}

fn silu_fwd(x: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) { *o = v * sigmoid(v); }
}

fn silu_fwd_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        let s = sigmoid(*v);
        *v *= s;
    }
}

fn silu_bwd(d_out: &[f32], x: &[f32], d_x: &mut [f32]) {
    for ((dx, &v), &d) in d_x.iter_mut().zip(x.iter()).zip(d_out.iter()) {
        let s = sigmoid(v);
        *dx = d * (s + v * s * (1.0 - s));
    }
}

fn glu_fwd(x: &[f32], out: &mut [f32]) {
    let half = x.len() / 2;
    for i in 0..half { out[i] = x[i] * sigmoid(x[half + i]); }
}

fn glu_bwd(d_out: &[f32], x: &[f32], d_x: &mut [f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        let v = x[i];
        let g = x[half + i];
        let s = sigmoid(g);
        d_x[i] = d_out[i] * s;
        d_x[half + i] = d_out[i] * v * s * (1.0 - s);
    }
}

fn per_neuron_glu_bwd(d_out: &[f32], x: &[f32], d_x: &mut [f32], n_neurons: usize, feat_per_neuron: usize) {
    // Each neuron's slice is [feat_per_neuron] value + [feat_per_neuron] gate.
    let block = feat_per_neuron * 2;
    for n in 0..n_neurons {
        let xs = &x[n * block..(n + 1) * block];
        let ds = &d_out[n * feat_per_neuron..(n + 1) * feat_per_neuron];
        let dx = &mut d_x[n * block..(n + 1) * block];
        for i in 0..feat_per_neuron {
            let v = xs[i];
            let g = xs[feat_per_neuron + i];
            let s = sigmoid(g);
            dx[i] = ds[i] * s;
            dx[feat_per_neuron + i] = ds[i] * v * s * (1.0 - s);
        }
    }
}

fn adamw(args: &mut AdamWArgs<'_>) {
    let AdamWArgs {
        w, g, m, v,
        lr, beta1, beta2, eps, weight_decay, bc1_inv, bc2_inv,
    } = args;
    let (lr, beta1, beta2, eps, weight_decay, bc1_inv, bc2_inv) =
        (*lr, *beta1, *beta2, *eps, *weight_decay, *bc1_inv, *bc2_inv);
    for i in 0..w.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
        let m_hat = m[i] * bc1_inv;
        let v_hat = v[i] * bc2_inv;
        w[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * w[i]);
    }
}

fn super_linear_fwd(
    trace: &[f32], weights: &[f32], biases: &[f32],
    out: &mut [f32], mut cache: Option<&mut [f32]>,
    d_model: usize, memory_length: usize, out_per: usize,
) {
    // Per-neuron matvec: for each of d_model neurons, weight of shape
    // [out_per × memory_length] applied to the neuron's trace slice.
    // When `cache` is Some, the pre-activation output is mirrored there
    // for a subsequent backward pass.
    for n in 0..d_model {
        let t = &trace[n * memory_length..(n + 1) * memory_length];
        let w_off = n * out_per * memory_length;
        let b_off = n * out_per;
        let o_off = n * out_per;
        for i in 0..out_per {
            let row = &weights[w_off + i * memory_length..w_off + (i + 1) * memory_length];
            let mut acc = biases[b_off + i];
            for k in 0..memory_length { acc += row[k] * t[k]; }
            out[o_off + i] = acc;
            if let Some(ref mut c) = cache {
                c[o_off + i] = acc;
            }
        }
    }
}

fn super_linear_bwd_dw(
    d_out: &[f32], trace: &[f32], d_weights: &mut [f32],
    d_model: usize, memory_length: usize, out_per: usize,
) {
    for n in 0..d_model {
        let t = &trace[n * memory_length..(n + 1) * memory_length];
        let w_off = n * out_per * memory_length;
        let o_off = n * out_per;
        for i in 0..out_per {
            let d = d_out[o_off + i];
            for k in 0..memory_length {
                d_weights[w_off + i * memory_length + k] += d * t[k];
            }
        }
    }
}

fn super_linear_bwd_dx(
    d_out: &[f32], weights: &[f32], d_trace: &mut [f32],
    d_model: usize, memory_length: usize, out_per: usize,
) {
    for n in 0..d_model {
        let w_off = n * out_per * memory_length;
        let o_off = n * out_per;
        let t_off = n * memory_length;
        for k in 0..memory_length {
            let mut acc = 0.0f32;
            for i in 0..out_per {
                acc += weights[w_off + i * memory_length + k] * d_out[o_off + i];
            }
            d_trace[t_off + k] = acc;
        }
    }
}

fn sync_update_fwd(
    h: &[f32], pairs_left: &[u32], pairs_right: &[u32], decay: &[f32],
    sync_state: &mut [f32], sync_out: &mut [f32], n_pairs: usize,
) {
    // sync[p] = decay[p] * state[p] + h[left[p]] * h[right[p]]
    for p in 0..n_pairs {
        let l = pairs_left[p] as usize;
        let r = pairs_right[p] as usize;
        sync_state[p] = decay[p] * sync_state[p] + h[l] * h[r];
        sync_out[p] = sync_state[p];
    }
}

fn sync_backward_scatter(args: &mut SyncBackwardScatterArgs<'_>) {
    let SyncBackwardScatterArgs {
        d_sync, pairs_left, pairs_right,
        activated, beta, d_act, n_pairs, d_model,
    } = args;
    let (n_pairs, d_model) = (*n_pairs, *d_model);
    for p in 0..n_pairs {
        let l = pairs_left[p] as usize;
        let r = pairs_right[p] as usize;
        if l < d_model && r < d_model {
            let inv_sqrt_beta = 1.0 / beta[p].sqrt().max(1e-8);
            d_act[l] += d_sync[p] * activated[r] * inv_sqrt_beta;
            d_act[r] += d_sync[p] * activated[l] * inv_sqrt_beta;
        }
    }
}

/// Fused synapse forward — CPU composition.
///
/// Pipeline: matvec (2*out_dim) → GLU (halves to out_dim) → SiLU inplace →
/// LayerNorm inplace (single row, n_cols = out_dim). Matches the KFD
/// `synapse_forward` kernel chain in `stream.rs`. Scratch is allocated
/// locally (one f32 Vec of size 2*out_dim) and dropped on return — not
/// exposed through the Op boundary by design.
fn synapse_forward(
    weight: &[f32], bias: &[f32], x: &[f32], out: &mut [f32],
    out_dim: usize, in_dim: usize,
) {
    let matvec_out = out_dim * 2;
    let mut scratch = vec![0.0f32; matvec_out];
    // matvec: weight is [2*out_dim × in_dim], bias is [2*out_dim].
    matvec(x, weight, bias, &mut scratch, matvec_out, in_dim);
    // GLU halves: out[i] = scratch[i] * sigmoid(scratch[out_dim + i])
    glu_fwd(&scratch, out);
    // SiLU inplace on the halved buffer.
    silu_fwd_inplace(out);
    // LayerNorm inplace, single row, n_cols = out_dim.
    layer_norm_inplace(out, 1, out_dim);
}

/// Row-wise in-place LayerNorm with no affine.
///
/// For each of `n_rows` rows of length `n_cols`, subtracts the row mean
/// then divides by √(var + 1e-5). No gamma/beta — the affine step is
/// handled by upstream ops (e.g. the synapse pipeline's preceding
/// matvec+GLU already realised the scaling). Matches the KFD
/// `layer_norm_fwd` kernel which the `try_layer_norm_inplace` entry
/// point dispatches.
fn layer_norm_inplace(x: &mut [f32], n_rows: usize, n_cols: usize) {
    debug_assert_eq!(
        n_rows * n_cols, x.len(),
        "layer_norm_inplace: n_rows * n_cols must equal x.len()",
    );
    for r in 0..n_rows {
        let row = &mut x[r * n_cols..(r + 1) * n_cols];
        let mean: f32 = row.iter().sum::<f32>() / n_cols as f32;
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n_cols as f32;
        let rstd = 1.0 / (var + 1e-5).sqrt();
        for v in row.iter_mut() { *v = (*v - mean) * rstd; }
    }
}

fn trace_rotate_inplace(trace: &mut [f32], new_val: &[f32], d_model: usize, memory_length: usize) {
    // Shift each neuron's trace history by one slot; write new_val into slot 0.
    for n in 0..d_model {
        let base = n * memory_length;
        for k in (1..memory_length).rev() { trace[base + k] = trace[base + k - 1]; }
        trace[base] = new_val[n];
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    if x > 12.0 { 1.0 } else if x < -12.0 { 0.0 } else { 1.0 / (1.0 + (-x).exp()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendRegistry, Op};

    #[test]
    fn matmul_nn_2x2() {
        let be = CpuBackend::new();
        let a = [1.0, 2.0, 3.0, 4.0];         // [[1,2],[3,4]]
        let b = [5.0, 6.0, 7.0, 8.0];         // [[5,6],[7,8]]
        let mut out = [0.0f32; 4];
        let mut op = Op::MatmulNN {
            a: &a, b: &b, out: &mut out, bias: None,
            m: 2, k: 2, n: 2,
        };
        be.dispatch(&mut op).unwrap();
        // Expected: [[19,22],[43,50]]
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matvec_identity_plus_bias() {
        let be = CpuBackend::new();
        let x = [1.0, 2.0, 3.0];
        let weight = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // I
        let bias = [10.0, 20.0, 30.0];
        let mut out = [0.0f32; 3];
        let mut op = Op::Matvec {
            x: &x, weight: &weight, bias: &bias, out: &mut out,
            out_dim: 3, in_dim: 3, quant: QuantKind::F32,
        };
        be.dispatch(&mut op).unwrap();
        assert_eq!(out, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn silu_matches_formula() {
        let be = CpuBackend::new();
        let x = [0.0f32, 1.0, -1.0];
        let mut out = [0.0f32; 3];
        let mut op = Op::SiluFwd { x: &x, out: &mut out };
        be.dispatch(&mut op).unwrap();
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0 * sigmoid(1.0)).abs() < 1e-6);
        assert!((out[2] - (-1.0) * sigmoid(-1.0)).abs() < 1e-6);
    }

    #[test]
    fn adamw_step_moves_toward_zero_grad() {
        let be = CpuBackend::new();
        let mut w = [1.0f32, 2.0, 3.0];
        let g = [0.5f32, 0.5, 0.5];
        let mut m = [0.0f32; 3];
        let mut v = [0.0f32; 3];
        let mut op = Op::AdamW(crate::backend::AdamWArgs {
            w: &mut w, g: &g, m: &mut m, v: &mut v,
            lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8,
            weight_decay: 0.0,
            bc1_inv: 1.0 / (1.0 - 0.9_f32.powi(1)),
            bc2_inv: 1.0 / (1.0 - 0.999_f32.powi(1)),
        });
        be.dispatch(&mut op).unwrap();
        // With positive gradients and lr=0.1, weights should decrease by ~lr.
        for wi in &w { assert!(*wi < 3.01); assert!(*wi > 0.0); }
    }

    #[test]
    fn reduce_l2_sq_correct() {
        let be = CpuBackend::new();
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 1];
        let mut op = Op::ReduceL2Sq { x: &x, out: &mut out };
        be.dispatch(&mut op).unwrap();
        assert_eq!(out[0], 30.0); // 1+4+9+16
    }

    #[test]
    fn registry_picks_cpu_when_its_the_only_backend() {
        let mut reg = BackendRegistry::new();
        reg.register(Box::new(CpuBackend::new()));
        let x = [1.0f32, 1.0, 1.0, 1.0];
        let mut out = [0.0f32; 1];
        let mut op = Op::ReduceL2Sq { x: &x, out: &mut out };
        let chose = reg.dispatch(&mut op).unwrap();
        assert_eq!(chose, "cpu");
        assert_eq!(out[0], 4.0);
    }

    #[test]
    fn cpu_declines_q4k() {
        let be = CpuBackend::new();
        let x = [0.0f32; 4];
        // Previously built an empty slice from a mis-aligned `[u8; 0]`
        // via `from_raw_parts` — that's UB (pointer not f32-aligned)
        // and aborted in debug under Rust 2024. Use a real, aligned,
        // zero-length slice of f32.
        let weight: &[f32] = &[];
        let bias = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        let op = Op::Matvec {
            x: &x, weight, bias: &bias, out: &mut out,
            out_dim: 4, in_dim: 4, quant: QuantKind::Q4K,
        };
        assert!(!be.supports(&op));
    }
}
