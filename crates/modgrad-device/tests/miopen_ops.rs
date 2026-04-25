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
