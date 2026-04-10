//! CPU vs GPU crossover benchmark
//! run: cargo test --release --test kfd_matmul_bench_small -- --nocapture --test-threads=1

use modgrad::kfd::{self, HsaDevice};
use modgrad::kfd::dispatch::{CodeObject, KernArgs};
use modgrad_compute::backend::dot;
use std::time::Instant;

static MATMUL_SMALL_CO: &[u8] = include_bytes!("../crates/modgrad-device/src/kfd/kernels/matmul_small.co");
static MATMUL_BLOCKED_CO: &[u8] = include_bytes!("../crates/modgrad-device/src/kfd/kernels/matmul_blocked.co");

/// CPU matmul using AVX-512 dot product: Y[j][i] = dot(W[i], X[j])
fn cpu_matmul(w: &[f32], x: &[f32], y: &mut [f32], m: usize, k: usize, n: usize) {
    for j in 0..n {
        let x_row = &x[j * k..(j + 1) * k];
        for i in 0..m {
            let w_row = &w[i * k..(i + 1) * k];
            y[j * m + i] = dot(w_row, x_row);
        }
    }
}

#[test]
fn matmul_bench_small() {
    if !kfd::is_available() { eprintln!("skip: no kfd"); return; }
    let mut dev = match HsaDevice::open() { Ok(d) => d, Err(e) => { eprintln!("skip: {e}"); return; }};

    for co_bytes in [MATMUL_SMALL_CO, MATMUL_BLOCKED_CO] {
        let co = CodeObject::load(&dev.alloc, co_bytes).unwrap();
        if dev.kernels.is_none() { dev.kernels = Some(std::collections::HashMap::new()); }
        for (name, entry) in &co.kernels { dev.kernels.as_mut().unwrap().insert(name.clone(), entry.clone()); }
        std::mem::forget(co);
    }

    println!();
    println!("  CPU vs GPU crossover (N=32 batch)");
    println!("  {:>10} {:>8} {:>10} {:>10} {:>10} {:>6}",
        "shape", "flops", "cpu_us", "gpu_us", "gflop/s", "pick");
    println!("  {}", "-".repeat(62));

    let shapes: &[(u32, u32)] = &[
        (32, 32), (64, 64), (128, 128), (256, 256), (512, 512),
        (1024, 1024), (2048, 2048), (4096, 4096),
    ];
    let n = 32u32;

    for &(m, k) in shapes {
        let flops = 2.0 * m as f64 * k as f64 * n as f64;

        // --- CPU ---
        let w_data = vec![0.001f32; (m * k) as usize];
        let x_data = vec![0.001f32; (n * k) as usize];
        let mut y_cpu = vec![0.0f32; (n * m) as usize];
        cpu_matmul(&w_data, &x_data, &mut y_cpu, m as usize, k as usize, n as usize);

        let iters_cpu = if m <= 256 { 10000 } else if m <= 1024 { 1000 } else { 100 };
        let t0 = Instant::now();
        for _ in 0..iters_cpu {
            cpu_matmul(&w_data, &x_data, &mut y_cpu, m as usize, k as usize, n as usize);
        }
        let cpu_us = t0.elapsed().as_nanos() as f64 / iters_cpu as f64 / 1000.0;

        // --- GPU ---
        let (kernel_name, nwg, w_buf) = if m >= 1536 {
            ("matmul_blocked", ((m + 127) / 128) * ((n + 31) / 32),
             dev.upload_f32(&w_data).unwrap())
        } else {
            ("matmul_small", ((m + 31) / 32) * ((n + 31) / 32),
             dev.upload_f32_col_major(&w_data, m as usize, k as usize).unwrap())
        };
        let b_buf = dev.upload_f32(&vec![0.0f32; m as usize]).unwrap();
        let x_buf = dev.upload_f32(&x_data).unwrap();
        let y_buf = dev.alloc_output((n as usize * m as usize * 4 + 64) as usize).unwrap();

        let mut args = KernArgs::new();
        args.push_ptr(&w_buf); args.push_ptr(&b_buf);
        args.push_ptr(&x_buf); args.push_ptr(&y_buf);
        args.push_u32(m); args.push_u32(k); args.push_u32(n);
        let args_buf = args.upload(&dev.alloc).unwrap();

        for _ in 0..50 { dev.dispatch_enqueue(kernel_name, &args_buf, [nwg, 1, 1], [256, 1, 1]); }
        assert!(dev.submit_wait(30_000), "warmup timeout");

        let iters_gpu = 500;
        let t0 = Instant::now();
        for _ in 0..iters_gpu { dev.dispatch_enqueue(kernel_name, &args_buf, [nwg, 1, 1], [256, 1, 1]); }
        assert!(dev.submit_wait(60_000), "bench timeout");
        let gpu_us = t0.elapsed().as_nanos() as f64 / iters_gpu as f64 / 1000.0;

        let best_gf = flops / gpu_us.min(cpu_us) / 1e3;
        let pick = if cpu_us < gpu_us { "CPU" } else { "GPU" };

        println!("  {:>4}x{:<4} {:>8.0} {:>8.1} {:>8.1} {:>8.0}   {:>4}",
            m, k, flops, cpu_us, gpu_us, best_gf, pick);
    }
    println!();
}
