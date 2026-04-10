//! Hybrid CPU+GPU pipeline benchmark
//! Simulates: GPU matmul → CPU elementwise → GPU matmul (overlapped)

use modgrad::kfd::{self, HsaDevice};
use modgrad::kfd::dispatch::{CodeObject, KernArgs};
use modgrad_compute::backend::dot;
use std::time::Instant;

static MATMUL_BLOCKED_CO: &[u8] = include_bytes!("../crates/modgrad-device/src/kfd/kernels/matmul_blocked.co");

/// Simulate elementwise: SiLU + residual add + layernorm on N×M floats
fn cpu_elementwise(data: &mut [f32], residual: &[f32], n: usize) {
    for i in 0..n {
        // SiLU: x * sigmoid(x)
        let x = data[i];
        let sig = 1.0 / (1.0 + (-x).exp());
        data[i] = x * sig + residual[i];
    }
    // Fake layernorm (just normalize variance)
    let mut sum = 0.0f32;
    let mut sum2 = 0.0f32;
    for i in 0..n {
        sum += data[i];
        sum2 += data[i] * data[i];
    }
    let mean = sum / n as f32;
    let var = sum2 / n as f32 - mean * mean;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for i in 0..n {
        data[i] = (data[i] - mean) * inv_std;
    }
}

#[test]
fn hybrid_bench() {
    if !kfd::is_available() { eprintln!("skip: no kfd"); return; }
    let mut dev = match HsaDevice::open() { Ok(d) => d, Err(e) => { eprintln!("skip: {e}"); return; }};

    let co = CodeObject::load(&dev.alloc, MATMUL_BLOCKED_CO).unwrap();
    if dev.kernels.is_none() { dev.kernels = Some(std::collections::HashMap::new()); }
    for (name, entry) in &co.kernels { dev.kernels.as_mut().unwrap().insert(name.clone(), entry.clone()); }
    std::mem::forget(co);

    let m = 4096u32; let k = 4096u32; let n = 32u32;

    let w_buf = dev.upload_f32(&vec![0.001f32; (m * k) as usize]).unwrap();
    let b_buf = dev.upload_f32(&vec![0.0f32; m as usize]).unwrap();
    let x_buf = dev.upload_f32(&vec![0.001f32; (n * k) as usize]).unwrap();
    let y_buf = dev.alloc_output((n as usize * m as usize * 4 + 64) as usize).unwrap();

    let mut args = KernArgs::new();
    args.push_ptr(&w_buf); args.push_ptr(&b_buf);
    args.push_ptr(&x_buf); args.push_ptr(&y_buf);
    args.push_u32(m); args.push_u32(k); args.push_u32(n);
    let args_buf = args.upload(&dev.alloc).unwrap();

    let nwg = ((m + 127) / 128) * ((n + 31) / 32);
    let grid = [nwg, 1, 1];
    let block = [256, 1, 1];

    // CPU elementwise data (simulates output of previous matmul)
    let mut cpu_data = vec![0.001f32; (n * m) as usize];
    let residual = vec![0.0f32; (n * m) as usize];
    let elem_size = (n * m) as usize;

    // Warmup
    for _ in 0..50 { dev.dispatch_enqueue("matmul_blocked", &args_buf, grid, block); }
    dev.submit_wait(30_000);
    cpu_elementwise(&mut cpu_data, &residual, elem_size);

    // --- Serial: GPU then CPU then GPU ---
    let iters = 200;
    let t0 = Instant::now();
    for _ in 0..iters {
        dev.dispatch_enqueue("matmul_blocked", &args_buf, grid, block);
        dev.submit_wait(60_000);
        cpu_elementwise(&mut cpu_data, &residual, elem_size);
    }
    let serial_us = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

    // --- Hybrid: CPU elementwise WHILE GPU computes next matmul ---
    let t0 = Instant::now();
    // Prime the pipeline: launch first GPU matmul
    dev.dispatch_enqueue("matmul_blocked", &args_buf, grid, block);
    for _ in 0..iters {
        // GPU is computing matmul N...
        // Meanwhile CPU does elementwise on matmul N-1's output
        cpu_elementwise(&mut cpu_data, &residual, elem_size);
        // Wait for GPU matmul N to finish
        dev.submit_wait(60_000);
        // Launch GPU matmul N+1 immediately
        dev.dispatch_enqueue("matmul_blocked", &args_buf, grid, block);
    }
    dev.submit_wait(60_000); // drain last
    let hybrid_us = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

    // --- CPU-only elementwise timing ---
    let t0 = Instant::now();
    for _ in 0..1000 {
        cpu_elementwise(&mut cpu_data, &residual, elem_size);
    }
    let cpu_elem_us = t0.elapsed().as_nanos() as f64 / 1000.0 / 1000.0;

    // --- GPU-only matmul timing ---
    let t0 = Instant::now();
    for _ in 0..500 {
        dev.dispatch_enqueue("matmul_blocked", &args_buf, grid, block);
    }
    dev.submit_wait(60_000);
    let gpu_mat_us = t0.elapsed().as_nanos() as f64 / 500.0 / 1000.0;

    println!();
    println!("  Hybrid pipeline: 4096x4096 matmul + elementwise (N=32)");
    println!("  -----------------------------------------------");
    println!("  GPU matmul alone:     {:>8.1} us", gpu_mat_us);
    println!("  CPU elementwise alone: {:>8.1} us", cpu_elem_us);
    println!("  Serial (GPU→CPU→GPU): {:>8.1} us/iter", serial_us);
    println!("  Hybrid (overlap):     {:>8.1} us/iter", hybrid_us);
    println!("  Speedup:              {:>8.2}x", serial_us / hybrid_us);
    println!();
}
