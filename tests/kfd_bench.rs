//! kfd gpu performance benchmarks.
//! run: cargo test --release --test kfd_bench -- --nocapture --test-threads=1
//!
//! measures gpu dispatch latency vs cpu at various matrix sizes
//! to find the crossover point where gpu wins.

use modgrad::kfd::{self, HsaDevice};
use modgrad::kfd::dispatch::KernArgs;
use modgrad::compute::{ComputeBackend, CpuBackend};
use std::time::Instant;

fn open_gpu() -> Option<std::sync::MutexGuard<'static, HsaDevice>> {
    use std::sync::{OnceLock, Mutex};
    static GPU: OnceLock<Option<Mutex<HsaDevice>>> = OnceLock::new();
    GPU.get_or_init(|| {
        if !kfd::is_available() { return None; }
        HsaDevice::open().ok().map(Mutex::new)
    }).as_ref().map(|m| m.lock().unwrap())
}

/// cpu reference matvec
fn cpu_matvec_timed(cpu: &CpuBackend, w: &[f32], b: &[f32], x: &[f32],
                    y: &mut [f32], out: usize, inp: usize, iters: usize) -> f64 {
    let t = Instant::now();
    for _ in 0..iters {
        cpu.matvec(w, b, x, y, out, inp);
    }
    t.elapsed().as_micros() as f64 / iters as f64
}

/// gpu matvec — allocate once, dispatch N times
fn gpu_matvec_timed(dev: &mut HsaDevice, w: &[f32], b: &[f32], x: &[f32],
                    out: u32, inp: u32, iters: usize) -> f64 {
    let w_buf = dev.upload_f32(w).unwrap();
    let b_buf = dev.upload_f32(b).unwrap();
    let x_buf = dev.upload_f32(x).unwrap();
    let y_buf = dev.alloc.alloc_userptr_public(((out as u64) * 4 + 4095) & !4095).unwrap();

    let mut args = KernArgs::new();
    args.push_ptr(&w_buf);
    args.push_ptr(&b_buf);
    args.push_ptr(&x_buf);
    args.push_ptr(&y_buf);
    args.push_u32(out);
    args.push_u32(inp);
    let args_buf = args.upload(&dev.alloc).unwrap();
    let block = out.min(256);

    // warmup
    dev.dispatch_kernel("matvec", &args_buf, [out, 1, 1], [block, 1, 1]);

    let t = Instant::now();
    for _ in 0..iters {
        dev.dispatch_kernel("matvec", &args_buf, [out, 1, 1], [block, 1, 1]);
    }
    t.elapsed().as_micros() as f64 / iters as f64
}

#[test]
fn bench_matvec_sizes() {
    let mut dev = match open_gpu() { Some(d) => d, None => {
        eprintln!("  SKIP: no gpu"); return;
    }};
    if dev.kernels.is_none() { eprintln!("  SKIP: no kernels"); return; }
    let cpu = CpuBackend::new();

    println!();
    println!("  {:>6} x {:<6}  {:>10}  {:>10}  {:>8}  winner", "out", "in", "cpu (us)", "gpu (us)", "speedup");
    println!("  {}", "-".repeat(62));

    for &(out, inp) in &[
        (4, 4), (8, 8), (16, 16), (32, 32), (64, 64),
        (128, 128), (256, 256), (512, 512),
        (1024, 1024),
        (64, 1024), (1024, 64),
    ] {
        let n = out * inp;
        let w: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();
        let b: Vec<f32> = (0..out).map(|i| (i % 10) as f32 * 0.1).collect();
        let x: Vec<f32> = (0..inp).map(|i| ((i * 13 + 5) % 100) as f32 / 100.0).collect();
        let mut y_cpu = vec![0.0f32; out];

        let iters = if n < 10000 { 1000 } else if n < 100000 { 100 } else { 10 };

        let cpu_us = cpu_matvec_timed(&cpu, &w, &b, &x, &mut y_cpu, out, inp, iters);
        let gpu_us = gpu_matvec_timed(&mut dev, &w, &b, &x, out as u32, inp as u32, iters);

        let speedup = cpu_us / gpu_us;
        let winner = if gpu_us < cpu_us { "gpu" } else { "cpu" };

        println!("  {:>6} x {:<6}  {:>10.1}  {:>10.1}  {:>7.2}x  {}",
            out, inp, cpu_us, gpu_us, speedup, winner);
    }
    println!();

    // isis-relevant sizes (from CTM config)
    println!("  isis workload sizes:");
    println!("  {:>6} x {:<6}  {:>10}  {:>10}  {:>8}  context", "out", "in", "cpu (us)", "gpu (us)", "speedup");
    println!("  {}", "-".repeat(72));

    let isis_sizes = [
        (128, 20, "input synapse (d_input=20 → 128 neurons)"),
        (256, 128, "attention synapse (128 → 256)"),
        (256, 256, "output synapse (256 → 256)"),
        (128, 256, "motor synapse (256 → 128)"),
        (1024, 20, "large input (d_input=20 → 1024)"),
        (1024, 1024, "large attention (1024 → 1024)"),
        (4096, 1024, "xl attention (1024 → 4096)"),
        (256, 4096, "sync accumulator readout"),
    ];

    for &(out, inp, label) in &isis_sizes {
        let n = out * inp;
        let w: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();
        let b: Vec<f32> = (0..out).map(|i| (i % 10) as f32 * 0.1).collect();
        let x: Vec<f32> = (0..inp).map(|i| ((i * 13 + 5) % 100) as f32 / 100.0).collect();
        let mut y_cpu = vec![0.0f32; out];

        let iters = if n < 10000 { 1000 } else if n < 100000 { 100 } else { 10 };

        let cpu_us = cpu_matvec_timed(&cpu, &w, &b, &x, &mut y_cpu, out, inp, iters);
        let gpu_us = gpu_matvec_timed(&mut dev, &w, &b, &x, out as u32, inp as u32, iters);

        let speedup = cpu_us / gpu_us;
        println!("  {:>6} x {:<6}  {:>10.1}  {:>10.1}  {:>7.2}x  {}",
            out, inp, cpu_us, gpu_us, speedup, label);
    }
    // llm-scale matmul sizes (matvec = batch=1 inference)
    println!("  llm inference sizes (batch=1 matvec):");
    println!("  {:>6} x {:<6}  {:>10}  {:>10}  {:>8}  context", "out", "in", "cpu (us)", "gpu (us)", "speedup");
    println!("  {}", "-".repeat(72));

    let llm_sizes = [
        (896, 896, "qwen2.5-0.5B hidden"),
        (4864, 896, "qwen2.5-0.5B mlp up"),
        (896, 4864, "qwen2.5-0.5B mlp down"),
        (2048, 2048, "llama-1B hidden"),
        (5632, 2048, "llama-1B mlp up"),
    ];

    for &(out, inp, label) in &llm_sizes {
        let n = out * inp;
        let w: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();
        let b = vec![0.0f32; out];
        let x: Vec<f32> = (0..inp).map(|i| ((i * 13 + 5) % 100) as f32 / 100.0).collect();
        let mut y_cpu = vec![0.0f32; out];

        let iters = 10;
        let cpu_us = cpu_matvec_timed(&cpu, &w, &b, &x, &mut y_cpu, out, inp, iters);
        let gpu_us = gpu_matvec_timed(&mut dev, &w, &b, &x, out as u32, inp as u32, iters);

        let speedup = cpu_us / gpu_us;
        let flops = (out * inp * 2) as f64;
        let gpu_gflops = flops / (gpu_us * 1e3); // GFLOP/s
        let cpu_gflops = flops / (cpu_us * 1e3);
        println!("  {:>6} x {:<6}  {:>8.0} ({:>4.1} GF/s)  {:>8.0} ({:>4.1} GF/s)  {:>5.2}x  {}",
            out, inp, cpu_us, cpu_gflops, gpu_us, gpu_gflops, speedup, label);
    }
    println!();
}
