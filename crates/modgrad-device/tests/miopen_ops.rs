//! MIOpen-backed resident-op tests.
//!
//! For each new `Op::*Resident` variant wired through MIOpen, this file:
//!   1. Builds a small input on host.
//!   2. Allocates `HipBuffer`s, uploads inputs.
//!   3. Dispatches the resident op via the `RocmBackend`.
//!   4. Downloads output back to host.
//!   5. Compares against a tiny CPU reference (`host_*` functions
//!      below) within a 1e-3 absolute / 1e-3 relative tolerance.
//!
//! Skips entire suite when ROCm runtime is absent (returning early
//! from each test), so this file is a no-op on CPU-only CI but still
//! type-checks under `--features rocm`.
//!
//! Tolerances: matches the existing `backend_parity` harness — softmax
//! and layer-norm involve `sqrt` and `exp` which differ slightly
//! between the MIOpen kernel and the host reference; 1e-3 catches
//! real math bugs without flagging the noise.

#![cfg(feature = "rocm")]

use modgrad_device::backend::{
    ActivationMode, Backend, BackendError, BinaryOpKind, HipBatch, HipBuffer, Op, RocmBackend,
};

const ABS_TOL: f32 = 1e-3;
const REL_TOL: f32 = 1e-3;

fn assert_close(a: &[f32], b: &[f32], name: &str) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let scale = av.abs().max(bv.abs()).max(1.0);
        if diff > ABS_TOL && diff / scale > REL_TOL {
            panic!(
                "{name}[{i}] divergence: gpu={av} cpu={bv} (|Δ| = {diff})"
            );
        }
    }
}

/// Allocate + upload a host slice into a fresh HipBuffer.
fn upload(host: &[f32]) -> Result<HipBuffer, BackendError> {
    let buf = HipBuffer::new(host.len() * 4)?;
    buf.copy_from_host(host)?;
    Ok(buf)
}

/// Allocate a zero-initialised device buffer of `n` f32s and seed it
/// from a host slice (the slice may be all-zero scratch).
fn alloc_out(n: usize) -> Result<HipBuffer, BackendError> {
    HipBuffer::new(n * 4)
}

// ─── Host references ─────────────────────────────────────────

fn host_layer_norm(
    x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32],
    n: usize, c: usize, epsilon: f32,
) {
    for r in 0..n {
        let row = &x[r * c..(r + 1) * c];
        let mean: f32 = row.iter().sum::<f32>() / c as f32;
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / c as f32;
        let rstd = 1.0 / (var + epsilon).sqrt();
        let o = &mut out[r * c..(r + 1) * c];
        for i in 0..c {
            o[i] = (row[i] - mean) * rstd * weight[i] + bias[i];
        }
    }
}

fn host_softmax(x: &[f32], out: &mut [f32], n_rows: usize, row_len: usize, log: bool) {
    for r in 0..n_rows {
        let row = &x[r * row_len..(r + 1) * row_len];
        let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let dst = &mut out[r * row_len..(r + 1) * row_len];
        if log {
            for i in 0..row_len {
                dst[i] = (row[i] - max) - sum.ln();
            }
        } else {
            for i in 0..row_len {
                dst[i] = exps[i] / sum;
            }
        }
    }
}

fn host_activation(x: &[f32], out: &mut [f32], mode: ActivationMode) {
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = match mode {
            ActivationMode::Logistic => sigmoid(v),
            ActivationMode::Tanh => v.tanh(),
            ActivationMode::Relu => v.max(0.0),
            ActivationMode::Silu => v * sigmoid(v),
        };
    }
}

/// Host reference for MIOpen's `dim=0` GLU layout.
///
/// Input is `[2, n_rows, half]` contiguous: first `n_rows * half`
/// floats are values, next `n_rows * half` are gates. Output is
/// `[n_rows, half]`.
fn host_glu(x: &[f32], out: &mut [f32], n_rows: usize, half: usize) {
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let total = n_rows * half;
    let values = &x[..total];
    let gates = &x[total..2 * total];
    for i in 0..total {
        out[i] = values[i] * sigmoid(gates[i]);
    }
}

fn host_op_tensor(
    a: &[f32], b: &[f32], out: &mut [f32],
    alpha1: f32, alpha2: f32, beta: f32, op: BinaryOpKind,
) {
    for i in 0..out.len() {
        let av = alpha1 * a[i];
        let bv = alpha2 * b[i];
        let combined = match op {
            BinaryOpKind::Add => av + bv,
            BinaryOpKind::Mul => av * bv,
            BinaryOpKind::Min => av.min(bv),
            BinaryOpKind::Max => av.max(bv),
        };
        out[i] = combined + beta * out[i];
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[test]
fn layer_norm_resident_matches_cpu() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("layer_norm_resident: no ROCm runtime; skipping");
        return;
    };
    let n = 4;
    let c = 64;
    let eps = 1e-5;

    // Deterministic ramp + offset so each row is non-trivial.
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32 * 0.013).sin() * 1.5 + 0.1).collect();
    let weight: Vec<f32> = (0..c).map(|i| 1.0 + 0.01 * i as f32).collect();
    let bias: Vec<f32> = (0..c).map(|i| -0.5 + 0.02 * i as f32).collect();

    let mut expected = vec![0.0f32; n * c];
    host_layer_norm(&x, &weight, &bias, &mut expected, n, c, eps);

    let _batch = HipBatch::new();
    let x_buf = upload(&x).unwrap();
    let w_buf = upload(&weight).unwrap();
    let b_buf = upload(&bias).unwrap();
    let y_buf = alloc_out(n * c).unwrap();

    let mut op = Op::LayerNormResident {
        x_dev: x_buf.device_ptr() as *const f32,
        weight_dev: w_buf.device_ptr() as *const f32,
        bias_dev: b_buf.device_ptr() as *const f32,
        y_dev: y_buf.device_ptr() as *mut f32,
        n,
        normalized_size: c,
        epsilon: eps,
    };
    be.dispatch(&mut op).expect("layer_norm_resident dispatch");

    // Sync before reading output so the GPU has finished writing.
    let mut got = vec![0.0f32; n * c];
    y_buf.copy_to_host(&mut got).unwrap();
    assert_close(&got, &expected, "layer_norm_resident");
}

#[test]
fn softmax_resident_matches_cpu() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("softmax_resident: no ROCm runtime; skipping");
        return;
    };
    let n_rows = 3;
    let row_len = 32;
    let x: Vec<f32> = (0..n_rows * row_len)
        .map(|i| (i as f32 * 0.07).cos() * 2.0 - 0.3)
        .collect();

    for log in [false, true] {
        let mut expected = vec![0.0f32; n_rows * row_len];
        host_softmax(&x, &mut expected, n_rows, row_len, log);

        let _batch = HipBatch::new();
        let x_buf = upload(&x).unwrap();
        let y_buf = alloc_out(n_rows * row_len).unwrap();

        let mut op = Op::SoftmaxResident {
            x_dev: x_buf.device_ptr() as *const f32,
            y_dev: y_buf.device_ptr() as *mut f32,
            n_rows,
            row_len,
            log,
        };
        be.dispatch(&mut op).expect("softmax_resident dispatch");

        let mut got = vec![0.0f32; n_rows * row_len];
        y_buf.copy_to_host(&mut got).unwrap();
        let label = if log { "softmax_resident(log)" } else { "softmax_resident" };
        assert_close(&got, &expected, label);
    }
}

#[test]
fn activation_resident_matches_cpu() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("activation_resident: no ROCm runtime; skipping");
        return;
    };
    let n = 257; // odd-prime length to catch any latent block-size bug.
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();

    for mode in [
        ActivationMode::Logistic,
        ActivationMode::Tanh,
        ActivationMode::Relu,
        ActivationMode::Silu,
    ] {
        let mut expected = vec![0.0f32; n];
        host_activation(&x, &mut expected, mode);

        let _batch = HipBatch::new();
        let x_buf = upload(&x).unwrap();
        let y_buf = alloc_out(n).unwrap();

        let mut op = Op::ActivationResident {
            x_dev: x_buf.device_ptr() as *const f32,
            y_dev: y_buf.device_ptr() as *mut f32,
            n,
            mode,
        };
        be.dispatch(&mut op).expect("activation_resident dispatch");

        let mut got = vec![0.0f32; n];
        y_buf.copy_to_host(&mut got).unwrap();
        assert_close(&got, &expected, &format!("activation_resident({mode:?})"));
    }
}

#[test]
fn glu_resident_matches_cpu() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("glu_resident: no ROCm runtime; skipping");
        return;
    };
    let n_rows = 5;
    let half = 16;
    let x: Vec<f32> = (0..n_rows * 2 * half)
        .map(|i| ((i as f32 * 0.11).sin() + 0.4) * 1.2)
        .collect();
    let mut expected = vec![0.0f32; n_rows * half];
    host_glu(&x, &mut expected, n_rows, half);

    let _batch = HipBatch::new();
    let x_buf = upload(&x).unwrap();
    let y_buf = alloc_out(n_rows * half).unwrap();

    let mut op = Op::GluResident {
        x_dev: x_buf.device_ptr() as *const f32,
        y_dev: y_buf.device_ptr() as *mut f32,
        n_rows,
        half_size: half,
    };
    be.dispatch(&mut op).expect("glu_resident dispatch");

    let mut got = vec![0.0f32; n_rows * half];
    y_buf.copy_to_host(&mut got).unwrap();
    assert_close(&got, &expected, "glu_resident");
}

#[test]
fn op_tensor_resident_matches_cpu() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("op_tensor_resident: no ROCm runtime; skipping");
        return;
    };
    let n = 128;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin() * 2.0).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i as f32 + 1.0) * 0.07).cos() * 1.5).collect();

    for op_kind in [BinaryOpKind::Add, BinaryOpKind::Mul, BinaryOpKind::Min, BinaryOpKind::Max] {
        // c starts as zeros, beta=0 ⇒ c = op(a, b)
        let mut expected = vec![0.0f32; n];
        host_op_tensor(&a, &b, &mut expected, 1.0, 1.0, 0.0, op_kind);

        let _batch = HipBatch::new();
        let a_buf = upload(&a).unwrap();
        let b_buf = upload(&b).unwrap();
        // c starts on device as zeros — alloc_out returns
        // uninitialised memory in the general case, so we explicitly
        // upload zeros to seed it.
        let c_init = vec![0.0f32; n];
        let c_buf = upload(&c_init).unwrap();

        let mut op = Op::OpTensorResident {
            a_dev: a_buf.device_ptr() as *const f32,
            b_dev: b_buf.device_ptr() as *const f32,
            c_dev: c_buf.device_ptr() as *mut f32,
            n,
            alpha1: 1.0,
            alpha2: 1.0,
            beta: 0.0,
            op: op_kind,
        };
        be.dispatch(&mut op).expect("op_tensor_resident dispatch");

        let mut got = vec![0.0f32; n];
        c_buf.copy_to_host(&mut got).unwrap();
        assert_close(&got, &expected, &format!("op_tensor_resident({op_kind:?})"));
    }
}

// ─── Backward tests ──────────────────────────────────────────

/// Host LayerNorm backward reference (matches MIOpen convention).
///
/// Given x, weight, bias, mean, rstd (per-row), and dy:
///   y = (x - mean) * rstd * weight + bias
///   dx[r,c] = (1/N) * weight[c] * rstd[r] * (
///       N * dy[r,c] - sum(dy[r,:] * weight) - x_norm[r,c] * sum(dy[r,:] * weight * x_norm[r,:]))
///   dweight[c] = sum_r dy[r,c] * x_norm[r,c]
///   dbias[c]   = sum_r dy[r,c]
fn host_layer_norm_backward(
    x: &[f32], dy: &[f32], weight: &[f32], mean: &[f32], rstd: &[f32],
    dx: &mut [f32], dweight: &mut [f32], dbias: &mut [f32],
    n: usize, c: usize,
) {
    for v in dweight.iter_mut() { *v = 0.0; }
    for v in dbias.iter_mut() { *v = 0.0; }
    for r in 0..n {
        let row_x = &x[r * c..(r + 1) * c];
        let row_dy = &dy[r * c..(r + 1) * c];
        let row_dx = &mut dx[r * c..(r + 1) * c];
        let m = mean[r];
        let rs = rstd[r];

        // x_norm[c] = (x - m) * rs ; sum_dy_w = Σ dy*weight ; sum_dy_w_xhat = Σ dy*weight*x_norm
        let mut sum_dy_w = 0.0f32;
        let mut sum_dy_w_xhat = 0.0f32;
        for i in 0..c {
            let xhat = (row_x[i] - m) * rs;
            sum_dy_w += row_dy[i] * weight[i];
            sum_dy_w_xhat += row_dy[i] * weight[i] * xhat;
            // dweight, dbias accumulate
            dweight[i] += row_dy[i] * xhat;
            dbias[i] += row_dy[i];
        }
        // Standard backward: dx[r,c] = rs * (dy[r,c] * weight[c] -
        //     (1/N) * sum_dy_w - (1/N) * x_norm[r,c] * sum_dy_w_xhat)
        let nf = c as f32;
        for i in 0..c {
            let xhat = (row_x[i] - m) * rs;
            row_dx[i] = rs *
                (row_dy[i] * weight[i] - sum_dy_w / nf - xhat * sum_dy_w_xhat / nf);
        }
    }
}

fn host_softmax_backward(
    y: &[f32], dy: &[f32], dx: &mut [f32],
    n_rows: usize, row_len: usize, log: bool,
) {
    for r in 0..n_rows {
        let yrow = &y[r * row_len..(r + 1) * row_len];
        let dyrow = &dy[r * row_len..(r + 1) * row_len];
        let dxrow = &mut dx[r * row_len..(r + 1) * row_len];
        if log {
            // log_softmax: dx[i] = dy[i] - exp(y[i]) * sum_dy
            let sum_dy: f32 = dyrow.iter().sum();
            for i in 0..row_len {
                dxrow[i] = dyrow[i] - yrow[i].exp() * sum_dy;
            }
        } else {
            // softmax: dx[i] = y[i] * (dy[i] - sum_j(y[j] * dy[j]))
            let dot: f32 = yrow.iter().zip(dyrow).map(|(&a, &b)| a * b).sum();
            for i in 0..row_len {
                dxrow[i] = yrow[i] * (dyrow[i] - dot);
            }
        }
    }
}

fn host_activation_backward(
    x: &[f32], y: &[f32], dy: &[f32], dx: &mut [f32], mode: ActivationMode,
) {
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    for i in 0..x.len() {
        dx[i] = match mode {
            // d/dx sigmoid(x) = y * (1 - y)
            ActivationMode::Logistic => dy[i] * y[i] * (1.0 - y[i]),
            // d/dx tanh(x) = 1 - y^2
            ActivationMode::Tanh => dy[i] * (1.0 - y[i] * y[i]),
            // d/dx relu(x) = (x > 0) ? 1 : 0
            ActivationMode::Relu => if x[i] > 0.0 { dy[i] } else { 0.0 },
            // d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            ActivationMode::Silu => {
                let s = sigmoid(x[i]);
                dy[i] * (s + x[i] * s * (1.0 - s))
            }
        };
    }
}

fn host_glu_backward(
    x: &[f32], dy: &[f32], dx: &mut [f32], n_rows: usize, half: usize,
) {
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let total = n_rows * half;
    let (values_in, gates_in) = x.split_at(total);
    let (dx_values, dx_gates) = dx.split_at_mut(total);
    for i in 0..total {
        let g = sigmoid(gates_in[i]);
        // d/d_value = dy[i] * sigmoid(gate)
        dx_values[i] = dy[i] * g;
        // d/d_gate = dy[i] * value * sigmoid'(gate)
        //          = dy[i] * value[i] * g * (1 - g)
        dx_gates[i] = dy[i] * values_in[i] * g * (1.0 - g);
    }
}

#[test]
fn layer_norm_backward_resident_matches_cpu() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("layer_norm_backward_resident: no ROCm runtime; skipping");
        return;
    };
    let n = 4;
    let c = 64;
    let eps = 1e-5;

    // Use the same forward inputs as the forward test so the saved
    // mean/rstd are well-defined.
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32 * 0.013).sin() * 1.5 + 0.1).collect();
    let weight: Vec<f32> = (0..c).map(|i| 1.0 + 0.01 * i as f32).collect();
    let bias: Vec<f32> = (0..c).map(|i| -0.5 + 0.02 * i as f32).collect();
    let dy: Vec<f32> = (0..n * c).map(|i| (i as f32 * 0.029).cos() * 0.4 + 0.05).collect();

    // Compute mean / rstd host-side (MIOpen would have produced the
    // same values during forward).
    let mut mean = vec![0.0f32; n];
    let mut rstd = vec![0.0f32; n];
    for r in 0..n {
        let row = &x[r * c..(r + 1) * c];
        let m: f32 = row.iter().sum::<f32>() / c as f32;
        let v: f32 = row.iter().map(|x| (x - m).powi(2)).sum::<f32>() / c as f32;
        mean[r] = m;
        rstd[r] = 1.0 / (v + eps).sqrt();
    }

    // Host reference.
    let mut expected_dx = vec![0.0f32; n * c];
    let mut expected_dweight = vec![0.0f32; c];
    let mut expected_dbias = vec![0.0f32; c];
    host_layer_norm_backward(
        &x, &dy, &weight, &mean, &rstd,
        &mut expected_dx, &mut expected_dweight, &mut expected_dbias,
        n, c,
    );

    let _batch = HipBatch::new();
    let x_buf = upload(&x).unwrap();
    let dy_buf = upload(&dy).unwrap();
    let w_buf = upload(&weight).unwrap();
    let mean_buf = upload(&mean).unwrap();
    let rstd_buf = upload(&rstd).unwrap();
    let dx_buf = alloc_out(n * c).unwrap();
    let dw_buf = alloc_out(c).unwrap();
    let db_buf = alloc_out(c).unwrap();
    // Use bias only to silence unused warning — bias is passed at
    // forward time, not relevant to the backward dispatch under test.
    let _ = bias;

    let mut op = Op::LayerNormBackwardResident {
        x_dev: x_buf.device_ptr() as *const f32,
        dy_dev: dy_buf.device_ptr() as *const f32,
        weight_dev: w_buf.device_ptr() as *const f32,
        mean_dev: mean_buf.device_ptr() as *const f32,
        rstd_dev: rstd_buf.device_ptr() as *const f32,
        dx_dev: dx_buf.device_ptr() as *mut f32,
        dweight_dev: dw_buf.device_ptr() as *mut f32,
        dbias_dev: db_buf.device_ptr() as *mut f32,
        n,
        normalized_size: c,
    };
    be.dispatch(&mut op).expect("layer_norm_backward_resident dispatch");

    let mut got_dx = vec![0.0f32; n * c];
    dx_buf.copy_to_host(&mut got_dx).unwrap();
    let mut got_dw = vec![0.0f32; c];
    dw_buf.copy_to_host(&mut got_dw).unwrap();
    let mut got_db = vec![0.0f32; c];
    db_buf.copy_to_host(&mut got_db).unwrap();

    assert_close(&got_dx, &expected_dx, "layer_norm_backward dx");
    assert_close(&got_dw, &expected_dweight, "layer_norm_backward dweight");
    assert_close(&got_db, &expected_dbias, "layer_norm_backward dbias");
}

#[test]
fn softmax_backward_resident_matches_cpu() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("softmax_backward_resident: no ROCm runtime; skipping");
        return;
    };
    let n_rows = 3;
    let row_len = 32;
    let x: Vec<f32> = (0..n_rows * row_len)
        .map(|i| (i as f32 * 0.07).cos() * 2.0 - 0.3)
        .collect();
    let dy: Vec<f32> = (0..n_rows * row_len)
        .map(|i| (i as f32 * 0.041).sin() * 0.6 + 0.02)
        .collect();

    for log in [false, true] {
        // Compute forward y first, then host backward expected.
        let mut y_host = vec![0.0f32; n_rows * row_len];
        host_softmax(&x, &mut y_host, n_rows, row_len, log);
        let mut expected_dx = vec![0.0f32; n_rows * row_len];
        host_softmax_backward(&y_host, &dy, &mut expected_dx, n_rows, row_len, log);

        let _batch = HipBatch::new();
        let y_buf = upload(&y_host).unwrap();
        let dy_buf = upload(&dy).unwrap();
        let dx_buf = alloc_out(n_rows * row_len).unwrap();

        let mut op = Op::SoftmaxBackwardResident {
            y_dev: y_buf.device_ptr() as *const f32,
            dy_dev: dy_buf.device_ptr() as *const f32,
            dx_dev: dx_buf.device_ptr() as *mut f32,
            n_rows,
            row_len,
            log,
        };
        be.dispatch(&mut op).expect("softmax_backward_resident dispatch");

        let mut got = vec![0.0f32; n_rows * row_len];
        dx_buf.copy_to_host(&mut got).unwrap();
        let label = if log { "softmax_backward(log)" } else { "softmax_backward" };
        assert_close(&got, &expected_dx, label);
    }
}

#[test]
fn activation_backward_resident_matches_cpu() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("activation_backward_resident: no ROCm runtime; skipping");
        return;
    };
    let n = 257;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let dy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).cos() * 0.5 + 0.1).collect();

    for mode in [
        ActivationMode::Logistic,
        ActivationMode::Tanh,
        ActivationMode::Relu,
        ActivationMode::Silu,
    ] {
        // Compute forward y.
        let mut y_host = vec![0.0f32; n];
        host_activation(&x, &mut y_host, mode);
        let mut expected_dx = vec![0.0f32; n];
        host_activation_backward(&x, &y_host, &dy, &mut expected_dx, mode);

        let _batch = HipBatch::new();
        let x_buf = upload(&x).unwrap();
        let y_buf = upload(&y_host).unwrap();
        let dy_buf = upload(&dy).unwrap();
        let dx_buf = alloc_out(n).unwrap();

        let mut op = Op::ActivationBackwardResident {
            x_dev: x_buf.device_ptr() as *const f32,
            y_dev: y_buf.device_ptr() as *const f32,
            dy_dev: dy_buf.device_ptr() as *const f32,
            dx_dev: dx_buf.device_ptr() as *mut f32,
            n,
            mode,
        };
        be.dispatch(&mut op).expect("activation_backward_resident dispatch");

        let mut got = vec![0.0f32; n];
        dx_buf.copy_to_host(&mut got).unwrap();
        assert_close(&got, &expected_dx, &format!("activation_backward({mode:?})"));
    }
}

#[test]
fn glu_backward_resident_matches_cpu() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("glu_backward_resident: no ROCm runtime; skipping");
        return;
    };
    let n_rows = 5;
    let half = 16;
    let x: Vec<f32> = (0..n_rows * 2 * half)
        .map(|i| ((i as f32 * 0.11).sin() + 0.4) * 1.2)
        .collect();
    let dy: Vec<f32> = (0..n_rows * half)
        .map(|i| ((i as f32 * 0.07).cos() + 0.2) * 0.4)
        .collect();
    let mut expected_dx = vec![0.0f32; n_rows * 2 * half];
    host_glu_backward(&x, &dy, &mut expected_dx, n_rows, half);

    let _batch = HipBatch::new();
    let x_buf = upload(&x).unwrap();
    let dy_buf = upload(&dy).unwrap();
    let dx_buf = alloc_out(n_rows * 2 * half).unwrap();

    let mut op = Op::GluBackwardResident {
        x_dev: x_buf.device_ptr() as *const f32,
        dy_dev: dy_buf.device_ptr() as *const f32,
        dx_dev: dx_buf.device_ptr() as *mut f32,
        n_rows,
        half_size: half,
    };
    be.dispatch(&mut op).expect("glu_backward_resident dispatch");

    let mut got = vec![0.0f32; n_rows * 2 * half];
    dx_buf.copy_to_host(&mut got).unwrap();
    assert_close(&got, &expected_dx, "glu_backward");
}

/// Integration smoke-test: SiLU is a Logistic + Mul compose. Verify
/// the full chain produces the same result as the activation_resident
/// path AND a host SiLU reference. This catches a regression where
/// the second-step OpTensor reads stale x or writes the wrong buffer.
#[test]
fn silu_resident_two_step_chain_matches_host() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("silu_resident_two_step_chain: no ROCm runtime; skipping");
        return;
    };
    let n = 64;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let mut expected = vec![0.0f32; n];
    host_activation(&x, &mut expected, ActivationMode::Silu);

    let _batch = HipBatch::new();
    let x_buf = upload(&x).unwrap();
    let y_buf = alloc_out(n).unwrap();

    let mut op = Op::ActivationResident {
        x_dev: x_buf.device_ptr() as *const f32,
        y_dev: y_buf.device_ptr() as *mut f32,
        n,
        mode: ActivationMode::Silu,
    };
    be.dispatch(&mut op).expect("silu chain dispatch");

    let mut got = vec![0.0f32; n];
    y_buf.copy_to_host(&mut got).unwrap();
    assert_close(&got, &expected, "silu_resident_two_step_chain");
}
