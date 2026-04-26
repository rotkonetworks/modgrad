//! Tests for the device-resident matmul (NN/NT/TN) and RMSNorm ops.
//!
//! Pattern follows `tests/miopen_ops.rs`:
//!   1. Build small input on host (deterministic ramp/sine, no rand
//!      so failures are reproducible across runs).
//!   2. Allocate `HipBuffer`s, upload inputs.
//!   3. Dispatch the resident op via `RocmBackend`.
//!   4. Download output, compare to a tiny CPU reference.
//!
//! The whole suite returns early when the ROCm runtime is missing, so
//! `cargo test --features rocm` on a host without an AMD GPU is a no-op
//! rather than a failure. RmsNormResident additionally returns early
//! when the build script couldn't find hipcc.

#![cfg(feature = "rocm")]

use modgrad_device::backend::{
    Backend, BackendError, HipBatch, HipBuffer, Op, RocmBackend,
};

const ABS_TOL: f32 = 1e-3;
const REL_TOL: f32 = 1e-3;

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

fn upload(host: &[f32]) -> Result<HipBuffer, BackendError> {
    let buf = HipBuffer::new(host.len() * 4)?;
    buf.copy_from_host(host)?;
    Ok(buf)
}

fn alloc_out(n: usize) -> Result<HipBuffer, BackendError> {
    HipBuffer::new(n * 4)
}

// ─── Host references ─────────────────────────────────────────

/// `C[m,n] = A[m,k] @ B[k,n]`, all row-major.
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

/// `C[m,n] = A[m,k] @ B[n,k]^T`, all row-major.
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

/// `C[m,n] = A[k,m]^T @ B[k,n]`, all row-major.
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

/// RMSNorm: `y[r, c] = x[r, c] / sqrt(mean(x[r,:]^2) + eps) * weight[c]`.
fn host_rms_norm(x: &[f32], weight: &[f32], out: &mut [f32], n: usize, hidden: usize, eps: f32) {
    for r in 0..n {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let rstd = 1.0 / (mean_sq + eps).sqrt();
        let dst = &mut out[r * hidden..(r + 1) * hidden];
        for c in 0..hidden {
            dst[c] = row[c] * rstd * weight[c];
        }
    }
}

/// Host RoPE backward — the reference the GPU kernel must match.
/// Per pair `(i, i+half)`:
///   `dx_pre[i]      =  dx_post[i] * cos[i] + dx_post[i+half] * sin[i]`
///   `dx_pre[i+half] = -dx_post[i] * sin[i] + dx_post[i+half] * cos[i]`
fn host_rope_backward(
    dx_post: &[f32],
    cos_tab: &[f32],
    sin_tab: &[f32],
    dx_pre: &mut [f32],
    num_heads: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    assert_eq!(cos_tab.len(), half);
    assert_eq!(sin_tab.len(), half);
    for h in 0..num_heads {
        let off = h * head_dim;
        for i in 0..half {
            let dl = dx_post[off + i];
            let dr = dx_post[off + i + half];
            let c = cos_tab[i];
            let s = sin_tab[i];
            dx_pre[off + i]        = dl * c + dr * s;
            dx_pre[off + i + half] = -dl * s + dr * c;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[test]
fn matmul_resident_nn_matches_host() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("matmul_resident_nn: no ROCm runtime; skipping");
        return;
    };
    let m = 4;
    let k = 8;
    let n = 6;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin() * 1.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32 + 1.0) * 0.07).cos() * 2.0).collect();
    let mut expected = vec![0.0f32; m * n];
    host_matmul_nn(&a, &b, &mut expected, m, k, n);

    let _batch = HipBatch::new();
    let a_buf = upload(&a).unwrap();
    let b_buf = upload(&b).unwrap();
    let c_buf = alloc_out(m * n).unwrap();

    let mut op = Op::MatmulResidentNN {
        a_dev: a_buf.device_ptr() as *const f32,
        b_dev: b_buf.device_ptr() as *const f32,
        out_dev: c_buf.device_ptr() as *mut f32,
        m, k, n,
    };
    be.dispatch(&mut op).expect("matmul_resident_nn dispatch");

    let mut got = vec![0.0f32; m * n];
    c_buf.copy_to_host(&mut got).unwrap();
    assert_close(&got, &expected, "matmul_resident_nn");
}

#[test]
fn matmul_resident_nt_matches_host() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("matmul_resident_nt: no ROCm runtime; skipping");
        return;
    };
    let m = 5;
    let k = 7;
    let n = 9;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.04).cos() * 1.3).collect();
    // B is laid out [n, k] row-major; the op reads it as B^T.
    let b: Vec<f32> = (0..n * k).map(|i| ((i as f32 + 2.0) * 0.06).sin() * 1.8).collect();
    let mut expected = vec![0.0f32; m * n];
    host_matmul_nt(&a, &b, &mut expected, m, k, n);

    let _batch = HipBatch::new();
    let a_buf = upload(&a).unwrap();
    let b_buf = upload(&b).unwrap();
    let c_buf = alloc_out(m * n).unwrap();

    let mut op = Op::MatmulResidentNT {
        a_dev: a_buf.device_ptr() as *const f32,
        b_dev: b_buf.device_ptr() as *const f32,
        out_dev: c_buf.device_ptr() as *mut f32,
        m, k, n,
    };
    be.dispatch(&mut op).expect("matmul_resident_nt dispatch");

    let mut got = vec![0.0f32; m * n];
    c_buf.copy_to_host(&mut got).unwrap();
    assert_close(&got, &expected, "matmul_resident_nt");
}

#[test]
fn matmul_resident_tn_matches_host() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("matmul_resident_tn: no ROCm runtime; skipping");
        return;
    };
    let m = 6;
    let k = 5;
    let n = 8;
    // A is laid out [k, m] row-major; the op reads it as A^T.
    let a: Vec<f32> = (0..k * m).map(|i| (i as f32 * 0.03).sin() * 1.7).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32 + 3.0) * 0.05).cos() * 1.4).collect();
    let mut expected = vec![0.0f32; m * n];
    host_matmul_tn(&a, &b, &mut expected, m, k, n);

    let _batch = HipBatch::new();
    let a_buf = upload(&a).unwrap();
    let b_buf = upload(&b).unwrap();
    let c_buf = alloc_out(m * n).unwrap();

    let mut op = Op::MatmulResidentTN {
        a_dev: a_buf.device_ptr() as *const f32,
        b_dev: b_buf.device_ptr() as *const f32,
        out_dev: c_buf.device_ptr() as *mut f32,
        m, k, n,
    };
    be.dispatch(&mut op).expect("matmul_resident_tn dispatch");

    let mut got = vec![0.0f32; m * n];
    c_buf.copy_to_host(&mut got).unwrap();
    assert_close(&got, &expected, "matmul_resident_tn");
}

#[test]
fn rms_norm_resident_matches_host() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("rms_norm_resident: no ROCm runtime; skipping");
        return;
    };
    // Build-script may have skipped the hipcc compile if hipcc was
    // missing at build time. In that case `supports()` returns false
    // and the registry would fall through to CPU (which is also
    // Unsupported). We probe directly via `supports()` so the test
    // skips cleanly instead of failing on a host without hipcc.
    let probe = Op::RmsNormResident {
        x_dev: std::ptr::null(),
        weight_dev: std::ptr::null(),
        y_dev: std::ptr::null_mut(),
        n: 1,
        hidden: 1,
        eps: 1e-5,
    };
    if !be.supports(&probe) {
        eprintln!("rms_norm_resident: hipcc kernel not built; skipping");
        return;
    }

    let n = 4;
    let hidden = 256;
    let eps = 1e-6;

    let x: Vec<f32> = (0..n * hidden).map(|i| ((i as f32) * 0.013).sin() * 1.5 + 0.2).collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + 0.005 * i as f32).collect();

    let mut expected = vec![0.0f32; n * hidden];
    host_rms_norm(&x, &weight, &mut expected, n, hidden, eps);

    let _batch = HipBatch::new();
    let x_buf = upload(&x).unwrap();
    let w_buf = upload(&weight).unwrap();
    let y_buf = alloc_out(n * hidden).unwrap();

    let mut op = Op::RmsNormResident {
        x_dev: x_buf.device_ptr() as *const f32,
        weight_dev: w_buf.device_ptr() as *const f32,
        y_dev: y_buf.device_ptr() as *mut f32,
        n,
        hidden,
        eps,
    };
    be.dispatch(&mut op).expect("rms_norm_resident dispatch");

    let mut got = vec![0.0f32; n * hidden];
    y_buf.copy_to_host(&mut got).unwrap();
    assert_close(&got, &expected, "rms_norm_resident");
}

#[test]
fn rope_backward_resident_matches_host() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("rope_backward_resident: no ROCm runtime; skipping");
        return;
    };
    // Same hipcc-archive gate as RmsNormResident — the kernel ships in
    // the shared `libmodgrad_kernels.a`.
    let probe = Op::RopeBackwardResident {
        dx_post_dev: std::ptr::null(),
        cos_dev: std::ptr::null(),
        sin_dev: std::ptr::null(),
        dx_pre_dev: std::ptr::null_mut(),
        num_heads: 1,
        head_dim: 2,
    };
    if !be.supports(&probe) {
        eprintln!("rope_backward_resident: hipcc kernel not built; skipping");
        return;
    }

    // Realistic-shape config: 4 heads × head_dim 64 (half_dim = 32),
    // covers the in-warp path. Deterministic ramp-and-sine inputs so
    // any divergence is reproducible.
    let num_heads = 4;
    let head_dim = 64;
    let half = head_dim / 2;

    let dx_post: Vec<f32> = (0..num_heads * head_dim)
        .map(|i| ((i as f32) * 0.017).sin() * 0.9 + 0.1)
        .collect();
    // Treat these like a RoPE cos/sin row — the kernel only cares
    // that they form an orthogonal rotation per pair, but for
    // correctness we just need consistent inputs.
    let angles: Vec<f32> = (0..half).map(|i| 0.1 + i as f32 * 0.07).collect();
    let cos_tab: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    let sin_tab: Vec<f32> = angles.iter().map(|a| a.sin()).collect();

    let mut expected = vec![0.0f32; num_heads * head_dim];
    host_rope_backward(&dx_post, &cos_tab, &sin_tab, &mut expected, num_heads, head_dim);

    let _batch = HipBatch::new();
    let dx_post_buf = upload(&dx_post).unwrap();
    let cos_buf = upload(&cos_tab).unwrap();
    let sin_buf = upload(&sin_tab).unwrap();
    let dx_pre_buf = alloc_out(num_heads * head_dim).unwrap();

    let mut op = Op::RopeBackwardResident {
        dx_post_dev: dx_post_buf.device_ptr() as *const f32,
        cos_dev: cos_buf.device_ptr() as *const f32,
        sin_dev: sin_buf.device_ptr() as *const f32,
        dx_pre_dev: dx_pre_buf.device_ptr() as *mut f32,
        num_heads,
        head_dim,
    };
    be.dispatch(&mut op).expect("rope_backward_resident dispatch");

    let mut got = vec![0.0f32; num_heads * head_dim];
    dx_pre_buf.copy_to_host(&mut got).unwrap();
    assert_close(&got, &expected, "rope_backward_resident");

    // In-place case: source and destination point at the same buffer.
    // Kernel must read both halves of each pair before writing either.
    let inplace_buf = upload(&dx_post).unwrap();
    let mut inplace_op = Op::RopeBackwardResident {
        dx_post_dev: inplace_buf.device_ptr() as *const f32,
        cos_dev: cos_buf.device_ptr() as *const f32,
        sin_dev: sin_buf.device_ptr() as *const f32,
        dx_pre_dev: inplace_buf.device_ptr() as *mut f32,
        num_heads,
        head_dim,
    };
    be.dispatch(&mut inplace_op).expect("rope_backward_resident in-place dispatch");
    let mut got_inplace = vec![0.0f32; num_heads * head_dim];
    inplace_buf.copy_to_host(&mut got_inplace).unwrap();
    assert_close(&got_inplace, &expected, "rope_backward_resident_inplace");
}
