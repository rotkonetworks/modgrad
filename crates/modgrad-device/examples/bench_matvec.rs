//! Matvec microbenchmark: KFD vs ROCm vs CPU at a production-like shape.
//!
//! Run: cargo run --release --features rocm --example bench_matvec -- 1024 1024
//! (set RUSTFLAGS='-L /opt/rocm/lib' when rocm feature is on)

use modgrad_device::backend::{Backend, CpuBackend, KfdBackend, Op, QuantKind, RocmBackend};
use std::time::Instant;

fn bench_one(name: &str, backend: &dyn Backend, x: &[f32], w: &[f32], b: &[f32], out_dim: usize, in_dim: usize, iters: usize) {
    let mut out = vec![0.0f32; out_dim];

    // Warmup
    for _ in 0..10 {
        let mut op = Op::Matvec {
            x, weight: w, bias: b, out: &mut out,
            out_dim, in_dim, quant: QuantKind::F32,
        };
        if backend.supports(&op) { let _ = backend.dispatch(&mut op); }
    }

    // Measure
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut op = Op::Matvec {
            x, weight: w, bias: b, out: &mut out,
            out_dim, in_dim, quant: QuantKind::F32,
        };
        if !backend.supports(&op) {
            println!("{name}: declines this shape");
            return;
        }
        backend.dispatch(&mut op).unwrap();
    }
    let dt = t0.elapsed();
    let per_iter_us = dt.as_secs_f64() * 1e6 / iters as f64;
    println!("{name}: {per_iter_us:>8.2} µs/iter  ({iters} iters in {:?})", dt);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dim: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1024);
    let in_dim: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1024);
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1000);

    println!("== matvec benchmark ==");
    println!("   shape: out_dim={out_dim}, in_dim={in_dim}");
    println!("   iters: {iters}");
    println!();

    let w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 1e-4).collect();
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 1e-3).collect();
    let b: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 1e-2).collect();

    // Create ROCm first — HSA runtime gets first claim on the device
    // before KFD opens its own handle.
    let rocm = RocmBackend::try_new();

    let cpu = CpuBackend::new();
    bench_one("cpu      ", &cpu, &x, &w, &b, out_dim, in_dim, iters);

    if let Some(ref r) = rocm {
        bench_one("rocm     ", r, &x, &w, &b, out_dim, in_dim, iters);
    } else {
        println!("rocm     : not available");
    }

    if let Some(kfd) = KfdBackend::try_new() {
        bench_one("kfd      ", &kfd, &x, &w, &b, out_dim, in_dim, iters);
    } else {
        println!("kfd      : not available");
    }
}
