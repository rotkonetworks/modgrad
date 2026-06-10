//! Batched `Linear<D>` — the GEMV→GEMM lever for the CTM tick loop.
//!
//! Two tests:
//!   1. `batched_linear_matches_scalar_cpu` (always runs): proves
//!      `forward_batched`/`backward_batched` are numerically identical to
//!      `batch` independent scalar `forward_into`/`backward` calls. This
//!      is the correctness gate before any tick-loop op is batched.
//!   2. `batched_linear_gemm_vs_gemv_microbench` (rocm only): measures the
//!      actual GEMM-vs-GEMV wall-clock win on the live GPU at CTM sizes —
//!      the number that justifies the whole batching refactor.

use modgrad_device::backend::tensor::{Cpu, Linear, Tensor};

fn approx(a: &[f32], b: &[f32], name: &str) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        let s = x.abs().max(y.abs()).max(1.0);
        assert!(d < 1e-4 || d / s < 1e-4, "{name}[{i}]: {x} vs {y} (|Δ|={d})");
    }
}

#[test]
fn batched_linear_matches_scalar_cpu() {
    let (in_dim, out_dim, batch) = (5usize, 7usize, 4usize);
    let w: Vec<f32> = (0..in_dim * out_dim)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.1).collect();
    let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32 - 3.0) * 0.05).collect();
    let lin = Linear::<Cpu>::from_host(&w, &bias, in_dim, out_dim).unwrap();

    let x: Vec<f32> = (0..batch * in_dim)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.2).collect();
    let dy: Vec<f32> = (0..batch * out_dim)
        .map(|i| ((i * 5 % 9) as f32 - 4.0) * 0.15).collect();

    // ── Batched ──
    let xt = Tensor::<Cpu>::from_slice(&x).unwrap();
    let mut yt = Tensor::<Cpu>::zeros(batch * out_dim).unwrap();
    lin.forward_batched(&xt, &mut yt, batch).unwrap();
    let y_batched = yt.to_vec().unwrap();

    let dyt = Tensor::<Cpu>::from_slice(&dy).unwrap();
    let mut dw_b = Tensor::<Cpu>::zeros(out_dim * in_dim).unwrap();
    let mut db_b = Tensor::<Cpu>::zeros(out_dim).unwrap();
    let mut dx_b = Tensor::<Cpu>::zeros(batch * in_dim).unwrap();
    lin.backward_batched(&dyt, &xt, &mut dw_b, &mut db_b, &mut dx_b, batch).unwrap();

    // ── Scalar reference: `batch` independent fwd/bwd, grads accumulate ──
    let mut y_scalar = vec![0.0f32; batch * out_dim];
    let mut dx_scalar = vec![0.0f32; batch * in_dim];
    let mut dw_s = Tensor::<Cpu>::zeros(out_dim * in_dim).unwrap();
    let mut db_s = Tensor::<Cpu>::zeros(out_dim).unwrap();
    for bi in 0..batch {
        let xr = Tensor::<Cpu>::from_slice(&x[bi * in_dim..(bi + 1) * in_dim]).unwrap();
        let mut yr = Tensor::<Cpu>::zeros(out_dim).unwrap();
        lin.forward_into(&xr, &mut yr).unwrap();
        y_scalar[bi * out_dim..(bi + 1) * out_dim].copy_from_slice(&yr.to_vec().unwrap());

        let dyr = Tensor::<Cpu>::from_slice(&dy[bi * out_dim..(bi + 1) * out_dim]).unwrap();
        let mut dxr = Tensor::<Cpu>::zeros(in_dim).unwrap();
        lin.backward(&dyr, &xr, &mut dw_s, &mut db_s, &mut dxr).unwrap();
        dx_scalar[bi * in_dim..(bi + 1) * in_dim].copy_from_slice(&dxr.to_vec().unwrap());
    }

    approx(&y_batched, &y_scalar, "forward");
    approx(&dx_b.to_vec().unwrap(), &dx_scalar, "d_x");
    approx(&dw_b.to_vec().unwrap(), &dw_s.to_vec().unwrap(), "d_w");
    approx(&db_b.to_vec().unwrap(), &db_s.to_vec().unwrap(), "d_b");
}

#[cfg(feature = "rocm")]
#[test]
fn batched_linear_gemm_vs_gemv_microbench() {
    use modgrad_device::backend::rocm::ffi::runtime_available;
    use modgrad_device::backend::tensor::Rocm;
    use std::time::Instant;

    if !runtime_available() {
        eprintln!("microbench: HIP unavailable, skipping");
        return;
    }

    let (d, batch, iters) = (512usize, 256usize, 50usize);
    let w: Vec<f32> = (0..d * d).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let bias = vec![0.0f32; d];
    let lin = Linear::<Rocm>::from_host(&w, &bias, d, d).unwrap();
    let x: Vec<f32> = (0..batch * d).map(|i| ((i % 53) as f32 - 26.0) * 0.01).collect();

    // ── GEMM: one batched matmul reuses W across all B rows ──
    let xt = Tensor::<Rocm>::from_slice(&x).unwrap();
    let mut yt = Tensor::<Rocm>::zeros(batch * d).unwrap();
    lin.forward_batched(&xt, &mut yt, batch).unwrap();
    let _ = yt.to_vec().unwrap(); // warmup + sync
    let t0 = Instant::now();
    for _ in 0..iters {
        lin.forward_batched(&xt, &mut yt, batch).unwrap();
    }
    let _ = yt.to_vec().unwrap(); // force completion before stopping clock
    let gemm_s = t0.elapsed().as_secs_f64();

    // ── GEMV: B separate matvecs (the current batch-1 reality) ──
    let xrows: Vec<Tensor<Rocm>> = (0..batch)
        .map(|bi| Tensor::<Rocm>::from_slice(&x[bi * d..(bi + 1) * d]).unwrap())
        .collect();
    let mut yr = Tensor::<Rocm>::zeros(d).unwrap();
    lin.forward_into(&xrows[0], &mut yr).unwrap();
    let _ = yr.to_vec().unwrap(); // warmup
    let t1 = Instant::now();
    for _ in 0..iters {
        for xr in &xrows {
            lin.forward_into(xr, &mut yr).unwrap();
        }
    }
    let _ = yr.to_vec().unwrap();
    let gemv_s = t1.elapsed().as_secs_f64();

    let speedup = gemv_s / gemm_s;
    eprintln!(
        "batched-linear microbench  d={d} B={batch} iters={iters}:  \
         GEMM={gemm_s:.4}s  GEMV(×B)={gemv_s:.4}s  speedup={speedup:.1}×"
    );
    assert!(speedup > 1.0, "GEMM should beat B separate GEMVs (got {speedup:.2}×)");
}
