//! Tests for the device-resident bf16 matmul (NN/NT/TN) and matvec ops.
//!
//! Same pattern as `tests/resident_matmul_rms.rs`:
//!   1. Build small input on host (deterministic ramps/sines).
//!   2. Quantise host operands to bf16 stored as `u16`.
//!   3. Allocate `HipBuffer`s, upload bf16 bytes, dispatch.
//!   4. Download bf16 output, dequantise to fp32, compare to a fp32
//!      CPU reference within ~1% relative tolerance (bf16 carries ~3
//!      decimal digits of precision).
//!
//! Returns early when the ROCm runtime is unavailable so
//! `cargo test --features rocm` is a no-op on hosts without an AMD GPU.

#![cfg(feature = "rocm")]

use modgrad_device::backend::{
    Backend, BackendError, HipBatch, HipBuffer, Op, RocmBackend,
};
use modgrad_device::backend::op::{bf16_to_f32, f32_to_bf16};

/// bf16 carries ~3 decimal digits of precision; absolute tolerance is
/// looser than fp32 by roughly 256× per intermediate accumulation.
const ABS_TOL: f32 = 5e-2;
const REL_TOL: f32 = 1e-2; // 1%

fn assert_close(a: &[f32], b: &[f32], name: &str) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let scale = av.abs().max(bv.abs()).max(1.0);
        if diff > ABS_TOL && diff / scale > REL_TOL {
            panic!("{name}[{i}] divergence: gpu={av} cpu={bv} (|Δ| = {diff})");
        }
    }
}

/// Upload a bf16 host slice (as `u16`) into a fresh `HipBuffer`.
/// Allocates `bytes = n * 2`. Goes through hipMemcpy via the f32
/// view; bf16 buffers are aligned to 2 bytes but the underlying
/// hipMemcpy treats them as opaque bytes.
fn upload_bf16(host: &[u16]) -> Result<HipBuffer, BackendError> {
    let buf = HipBuffer::new(host.len() * 2)?;
    let view = unsafe {
        std::slice::from_raw_parts(host.as_ptr() as *const f32, host.len() / 2)
    };
    buf.copy_from_host(view)?;
    Ok(buf)
}

/// Download a `HipBuffer` known to hold `n` bf16 elements into a
/// `Vec<u16>`.
fn download_bf16(buf: &HipBuffer, n: usize) -> Result<Vec<u16>, BackendError> {
    let mut out_u16 = vec![0u16; n];
    let view = unsafe {
        std::slice::from_raw_parts_mut(out_u16.as_mut_ptr() as *mut f32, n / 2)
    };
    buf.copy_to_host(view)?;
    Ok(out_u16)
}

fn alloc_out_bf16(n: usize) -> Result<HipBuffer, BackendError> {
    HipBuffer::new(n * 2)
}

fn quantise(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&v| f32_to_bf16(v)).collect()
}

fn dequantise(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&v| bf16_to_f32(v)).collect()
}

// ─── Host references ─────────────────────────────────────────

fn host_matmul_nn(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += a[r * k + j] * b[j * n + c];
            }
            out[r * n + c] = acc;
        }
    }
}

fn host_matmul_nt(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += a[r * k + j] * b[c * k + j];
            }
            out[r * n + c] = acc;
        }
    }
}

fn host_matmul_tn(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += a[j * m + r] * b[j * n + c];
            }
            out[r * n + c] = acc;
        }
    }
}

/// Reference for the bf16 matvec: reproduce the bf16-input, fp32-
/// accumulate, bf16-output recipe so the comparison reflects the
/// hardware path rather than a pure-fp32 reference.
fn host_matvec_bf16_reference(
    weight: &[f32], x: &[f32], bias: &[f32], out: &mut [f32],
    out_dim: usize, in_dim: usize,
) {
    for r in 0..out_dim {
        let mut acc = bias[r];
        for c in 0..in_dim {
            acc += weight[r * in_dim + c] * x[c];
        }
        out[r] = acc;
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[test]
fn matmul_resident_bf16_nn_matches_fp32() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("matmul_resident_bf16_nn: no ROCm runtime; skipping");
        return;
    };
    // Even dims for the two-byte-aligned bf16 upload helper.
    let m = 8;
    let k = 16;
    let n = 12;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.07).sin() * 0.8).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32 + 1.0) * 0.05).cos() * 0.6).collect();

    // Reference goes through the round-tripped bf16 inputs so the
    // comparison is fair: bf16 quantisation is part of the path
    // under test, not a separate error source we can tighten away.
    let a_q: Vec<f32> = a.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let b_q: Vec<f32> = b.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let mut expected = vec![0.0f32; m * n];
    host_matmul_nn(&a_q, &b_q, &mut expected, m, k, n);

    let _batch = HipBatch::new();
    let a_bf16 = quantise(&a);
    let b_bf16 = quantise(&b);
    let a_buf = upload_bf16(&a_bf16).unwrap();
    let b_buf = upload_bf16(&b_bf16).unwrap();
    let c_buf = alloc_out_bf16(m * n).unwrap();

    let mut op = Op::MatmulResidentBf16Nn {
        a_dev: a_buf.device_ptr() as *const u16,
        b_dev: b_buf.device_ptr() as *const u16,
        c_dev: c_buf.device_ptr() as *mut u16,
        m, k, n,
        alpha: 1.0, beta: 0.0,
    };
    be.dispatch(&mut op).expect("matmul_resident_bf16_nn dispatch");

    let got_bf16 = download_bf16(&c_buf, m * n).unwrap();
    let got = dequantise(&got_bf16);
    assert_close(&got, &expected, "matmul_resident_bf16_nn");
}

#[test]
fn matmul_resident_bf16_nt_matches_fp32() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("matmul_resident_bf16_nt: no ROCm runtime; skipping");
        return;
    };
    let m = 6;
    let k = 8;
    let n = 10;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.04).cos() * 0.7).collect();
    let b: Vec<f32> = (0..n * k).map(|i| ((i as f32 + 2.0) * 0.06).sin() * 0.5).collect();

    let a_q: Vec<f32> = a.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let b_q: Vec<f32> = b.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let mut expected = vec![0.0f32; m * n];
    host_matmul_nt(&a_q, &b_q, &mut expected, m, k, n);

    let _batch = HipBatch::new();
    let a_buf = upload_bf16(&quantise(&a)).unwrap();
    let b_buf = upload_bf16(&quantise(&b)).unwrap();
    let c_buf = alloc_out_bf16(m * n).unwrap();

    let mut op = Op::MatmulResidentBf16Nt {
        a_dev: a_buf.device_ptr() as *const u16,
        b_dev: b_buf.device_ptr() as *const u16,
        c_dev: c_buf.device_ptr() as *mut u16,
        m, k, n,
        alpha: 1.0, beta: 0.0,
    };
    be.dispatch(&mut op).expect("matmul_resident_bf16_nt dispatch");

    let got = dequantise(&download_bf16(&c_buf, m * n).unwrap());
    assert_close(&got, &expected, "matmul_resident_bf16_nt");
}

#[test]
fn matmul_resident_bf16_tn_matches_fp32() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("matmul_resident_bf16_tn: no ROCm runtime; skipping");
        return;
    };
    let m = 6;
    let k = 8;
    let n = 10;
    let a: Vec<f32> = (0..k * m).map(|i| (i as f32 * 0.03).sin() * 0.9).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32 + 3.0) * 0.05).cos() * 0.4).collect();

    let a_q: Vec<f32> = a.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let b_q: Vec<f32> = b.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let mut expected = vec![0.0f32; m * n];
    host_matmul_tn(&a_q, &b_q, &mut expected, m, k, n);

    let _batch = HipBatch::new();
    let a_buf = upload_bf16(&quantise(&a)).unwrap();
    let b_buf = upload_bf16(&quantise(&b)).unwrap();
    let c_buf = alloc_out_bf16(m * n).unwrap();

    let mut op = Op::MatmulResidentBf16Tn {
        a_dev: a_buf.device_ptr() as *const u16,
        b_dev: b_buf.device_ptr() as *const u16,
        c_dev: c_buf.device_ptr() as *mut u16,
        m, k, n,
        alpha: 1.0, beta: 0.0,
    };
    be.dispatch(&mut op).expect("matmul_resident_bf16_tn dispatch");

    let got = dequantise(&download_bf16(&c_buf, m * n).unwrap());
    assert_close(&got, &expected, "matmul_resident_bf16_tn");
}

#[test]
fn matvec_resident_bf16_matches_fp32() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("matvec_resident_bf16: no ROCm runtime; skipping");
        return;
    };
    let out_dim = 16;
    let in_dim = 24;
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
    let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32 + 1.0) * 0.07).cos() * 0.4).collect();
    let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32 * 0.1).sin() * 0.05).collect();

    // Reference: round-trip operands through bf16 then accumulate in
    // fp32 (matches the device path). Bias is bf16 too, so it
    // contributes its quantisation error to the seed.
    let w_q: Vec<f32> = weight.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let x_q: Vec<f32> = x.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let b_q: Vec<f32> = bias.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let mut expected = vec![0.0f32; out_dim];
    host_matvec_bf16_reference(&w_q, &x_q, &b_q, &mut expected, out_dim, in_dim);

    let _batch = HipBatch::new();
    let w_buf = upload_bf16(&quantise(&weight)).unwrap();
    let x_buf = upload_bf16(&quantise(&x)).unwrap();
    let b_buf = upload_bf16(&quantise(&bias)).unwrap();
    let out_buf = alloc_out_bf16(out_dim).unwrap();

    let mut op = Op::MatvecResidentBf16 {
        x_dev: x_buf.device_ptr() as *const u16,
        weight_dev: w_buf.device_ptr() as *const u16,
        bias_dev: b_buf.device_ptr() as *const u16,
        out_dev: out_buf.device_ptr() as *mut u16,
        out_dim, in_dim,
    };
    be.dispatch(&mut op).expect("matvec_resident_bf16 dispatch");

    let got = dequantise(&download_bf16(&out_buf, out_dim).unwrap());
    assert_close(&got, &expected, "matvec_resident_bf16");
}

#[test]
fn bf16_round_trip_is_idempotent() {
    // Sanity test for the bf16 helpers — round-trip stays within ~1%
    // for a representative range of values. Doesn't require the GPU.
    let cases = [0.0f32, 1.0, -1.0, 3.14159, -2.71828, 1e-3, 1e3, 1.234e-2];
    for &v in &cases {
        let q = bf16_to_f32(f32_to_bf16(v));
        let scale = v.abs().max(1e-6);
        let rel = (q - v).abs() / scale;
        assert!(rel < 1e-2, "round-trip relative error {rel} for {v}");
    }
    // NaN maps to NaN.
    let q = bf16_to_f32(f32_to_bf16(f32::NAN));
    assert!(q.is_nan(), "NaN round-trip lost NaN");
}
