//! matmul kernel benchmark — pure rust dispatch.
//! run: cargo test --release --test kfd_matmul_bench -- --nocapture --test-threads=1

use modgrad::kfd::{self, HsaDevice};
use modgrad::kfd::dispatch::{CodeObject, KernArgs};
use modgrad::kfd::memory::GpuBuffer;
use std::time::Instant;

static MATMUL_BLOCKED_CO: &[u8] = include_bytes!("../crates/modgrad-device/src/kfd/kernels/matmul_blocked.co");
static MATMUL_SMALL_CO: &[u8] = include_bytes!("../crates/modgrad-device/src/kfd/kernels/matmul_small.co");

#[test]
fn matmul_bench() {
    if !kfd::is_available() { eprintln!("skip: no kfd"); return; }
    let mut dev = match HsaDevice::open() { Ok(d) => d, Err(e) => { eprintln!("skip: {e}"); return; }};

    // Load both matmul kernels
    for co_bytes in [MATMUL_BLOCKED_CO, MATMUL_SMALL_CO] {
        let co = CodeObject::load(&dev.alloc, co_bytes).unwrap();
        if dev.kernels.is_none() { dev.kernels = Some(std::collections::HashMap::new()); }
        for (name, entry) in &co.kernels {
            dev.kernels.as_mut().unwrap().insert(name.clone(), entry.clone());
        }
        std::mem::forget(co); // keep code in VRAM
    }

    if let Some(mhz) = dev.current_sclk_mhz() {
        println!("  sclk: {} mhz", mhz);
    }
    println!();

    // ---- Correctness check for matmul_blocked (TM=128) ----
    {
        let m = 128u32; let k = 8u32; let n = 32u32;
        let mut w_data = vec![0.0f32; (m * k) as usize];
        for i in 0..m as usize { w_data[i * k as usize] = (i + 1) as f32; }
        let b_data = vec![0.0f32; m as usize];
        let mut x_data = vec![0.0f32; (n * k) as usize];
        for j in 0..n as usize { x_data[j * k as usize] = 1.0; }

        let w_buf = dev.upload_f32(&w_data).unwrap();
        let b_buf = dev.upload_f32(&b_data).unwrap();
        let x_buf = dev.upload_f32(&x_data).unwrap();
        let y_buf = dev.alloc_output((n as usize * m as usize * 4 + 64) as usize).unwrap();

        let mut args = KernArgs::new();
        args.push_ptr(&w_buf); args.push_ptr(&b_buf);
        args.push_ptr(&x_buf); args.push_ptr(&y_buf);
        args.push_u32(m); args.push_u32(k); args.push_u32(n);
        let args_buf = args.upload(&dev.alloc).unwrap();

        let nwg = ((m + 127) / 128) * ((n + 31) / 32);
        dev.dispatch_kernel("matmul_blocked", &args_buf, [nwg, 1, 1], [256, 1, 1]);

        let y_slice = unsafe { std::slice::from_raw_parts(y_buf.cpu_ptr as *const f32, (n * m) as usize) };
        print!("  matmul_blocked (TM=128): ");
        let mut ok = true;
        for j in 0..n as usize {
            for i in 0..m as usize {
                let expected = (i + 1) as f32;
                if (y_slice[j * m as usize + i] - expected).abs() > 0.1 { ok = false; break; }
            }
            if !ok { break; }
        }
        println!("{}", if ok { "PASS" } else { "FAIL" });
        assert!(ok, "matmul_blocked correctness failed");
    }

    // ---- Correctness check for matmul_small (TM=32) ----
    {
        let m = 32u32; let k = 8u32; let n = 32u32;
        let mut w_data = vec![0.0f32; (m * k) as usize];
        for i in 0..m as usize { w_data[i * k as usize] = (i + 1) as f32; }
        let b_data = vec![0.0f32; m as usize];
        let mut x_data = vec![0.0f32; (n * k) as usize];
        for j in 0..n as usize { x_data[j * k as usize] = 1.0; }

        let w_buf = dev.upload_f32_col_major(&w_data, m as usize, k as usize).unwrap();
        let b_buf = dev.upload_f32(&b_data).unwrap();
        let x_buf = dev.upload_f32(&x_data).unwrap();
        let y_buf = dev.alloc_output((n as usize * m as usize * 4 + 64) as usize).unwrap();

        let mut args = KernArgs::new();
        args.push_ptr(&w_buf); args.push_ptr(&b_buf);
        args.push_ptr(&x_buf); args.push_ptr(&y_buf);
        args.push_u32(m); args.push_u32(k); args.push_u32(n);
        let args_buf = args.upload(&dev.alloc).unwrap();

        let nwg = ((m + 31) / 32) * ((n + 31) / 32);
        dev.dispatch_kernel("matmul_small", &args_buf, [nwg, 1, 1], [256, 1, 1]);

        let y_slice = unsafe { std::slice::from_raw_parts(y_buf.cpu_ptr as *const f32, (n * m) as usize) };
        print!("  matmul_small   (TM=32):  ");
        let mut ok = true;
        for j in 0..n as usize {
            for i in 0..m as usize {
                let expected = (i + 1) as f32;
                if (y_slice[j * m as usize + i] - expected).abs() > 0.1 { ok = false; break; }
            }
            if !ok { break; }
        }
        println!("{}", if ok { "PASS" } else { "FAIL" });
        assert!(ok, "matmul_small correctness failed");
    }
    println!();

    // ---- Performance benchmark: both kernels + dispatch selector ----
    let shapes: &[(u32, u32, u32, &str)] = &[
        (512,  512,  32, ""),
        (1024, 1024, 32, ""),
        (2048, 2048, 32, ""),
        (4096, 4096, 32, "qwen attn"),
        (4096, 11008, 32, "qwen mlp"),
        (4864, 896,  32, "isis layer"),
    ];

    // bench helper
    let bench_kernel = |dev: &mut HsaDevice, kernel: &str, m: u32, k: u32, n: u32,
                        w_buf: &GpuBuffer, b_buf: &GpuBuffer, x_buf: &GpuBuffer, y_buf: &GpuBuffer| -> f64 {
        let mut args = KernArgs::new();
        args.push_ptr(w_buf); args.push_ptr(b_buf);
        args.push_ptr(x_buf); args.push_ptr(y_buf);
        args.push_u32(m); args.push_u32(k); args.push_u32(n);
        let args_buf = args.upload(&dev.alloc).unwrap();

        let (nwg, block) = if kernel == "matmul_blocked" {
            (((m + 127) / 128) * ((n + 31) / 32), [256u32, 1, 1])
        } else {
            (((m + 31) / 32) * ((n + 31) / 32), [256u32, 1, 1])
        };
        let grid = [nwg, 1, 1];

        for _ in 0..50 { dev.dispatch_enqueue(kernel, &args_buf, grid, block); }
        assert!(dev.submit_wait(30_000), "warmup timeout");

        let iters = 500;
        let t0 = Instant::now();
        for _ in 0..iters { dev.dispatch_enqueue(kernel, &args_buf, grid, block); }
        assert!(dev.submit_wait(60_000), "bench timeout");
        t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0
    };

    println!("  {:>12}  {:>10}  {:>10}  {:>10}  {:>6}", "shape", "TM=128", "TM=32", "best", "pick");
    println!("  {}", "-".repeat(58));

    for &(m, k, n, label) in shapes {
        // Allocate buffers (both W formats)
        let w_data = vec![0.001f32; (m * k) as usize];
        let w_row = dev.upload_f32(&w_data).unwrap();
        let w_col = dev.upload_f32_col_major(&w_data, m as usize, k as usize).unwrap();
        let b_buf = dev.upload_f32(&vec![0.0f32; m as usize]).unwrap();
        let x_buf = dev.upload_f32(&vec![0.001f32; (n * k) as usize]).unwrap();
        let y_buf = dev.alloc_output((n as usize * m as usize * 4 + 64) as usize).unwrap();

        let us_128 = bench_kernel(&mut dev, "matmul_blocked", m, k, n, &w_row, &b_buf, &x_buf, &y_buf);
        let us_32  = bench_kernel(&mut dev, "matmul_small",   m, k, n, &w_col, &b_buf, &x_buf, &y_buf);

        let gf_128 = 2.0 * m as f64 * k as f64 * n as f64 / us_128 / 1e3;
        let gf_32  = 2.0 * m as f64 * k as f64 * n as f64 / us_32  / 1e3;
        let (best, pick) = if gf_128 >= gf_32 { (gf_128, "TM128") } else { (gf_32, "TM32") };

        let l = if label.is_empty() { String::new() } else { format!("  {}", label) };
        println!("  {:>5}x{:<5}  {:>8.0}  {:>8.0}  {:>8.0}  {:>6}{}",
            m, k, gf_128, gf_32, best, pick, l);
    }
    println!();
}
