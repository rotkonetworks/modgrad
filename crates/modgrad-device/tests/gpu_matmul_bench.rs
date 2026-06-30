//! Foundational benchmark for the GPU-training infra decision: at what matrix
//! size does the ROCm matmul start beating the CPU? The regional cascade's ops
//! are ~512-dim, so the crossover tells us whether a batched-resident regional
//! path can be GPU-efficient at all. Run with:
//!   HSA_OVERRIDE_GFX_VERSION=11.0.0 cargo test -p modgrad-device --release \
//!     --features rocm --test gpu_matmul_bench -- --nocapture --ignored

#[cfg(feature = "rocm")]
#[test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture"]
fn bench_matmul_rocm_vs_cpu() {
    use modgrad_device::backend::tensor::{matmul_nn, Cpu, Rocm, Tensor};
    use std::time::Instant;

    fn fill(n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.013 % 1.7) - 0.85) * s).collect()
    }

    println!("\n  size |  CPU ms (GFLOP/s) | ROCm ms (GFLOP/s) | speedup");
    println!("  -----+-------------------+-------------------+--------");
    for &d in &[256usize, 512, 1024, 2048] {
        let (m, k, n) = (d, d, d);
        let a = fill(m * k, 0.1);
        let b = fill(k * n, 0.1);
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let iters = if d <= 512 { 50 } else { 20 };

        // ── CPU ──
        let ca = Tensor::<Cpu>::from_slice(&a).unwrap();
        let cb = Tensor::<Cpu>::from_slice(&b).unwrap();
        let mut cc = Tensor::<Cpu>::zeros(m * n).unwrap();
        matmul_nn(&ca, &cb, &mut cc, m, k, n).unwrap(); // warm
        let t = Instant::now();
        for _ in 0..iters {
            matmul_nn(&ca, &cb, &mut cc, m, k, n).unwrap();
        }
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // ── ROCm ── (warm up to JIT-compile the kernel; sync via to_vec)
        let ra = Tensor::<Rocm>::from_slice(&a).unwrap();
        let rb = Tensor::<Rocm>::from_slice(&b).unwrap();
        let mut rc = Tensor::<Rocm>::zeros(m * n).unwrap();
        matmul_nn(&ra, &rb, &mut rc, m, k, n).unwrap();
        let _ = rc.to_vec().unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            matmul_nn(&ra, &rb, &mut rc, m, k, n).unwrap();
        }
        let _ = rc.to_vec().unwrap(); // force completion before measuring
        let rocm_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        println!(
            "  {d:>4} | {cpu_ms:7.3} ({:7.1}) | {rocm_ms:7.3} ({:7.1}) | {:5.2}x",
            flops / cpu_ms / 1e6,
            flops / rocm_ms / 1e6,
            cpu_ms / rocm_ms,
        );
    }
    println!();
}
