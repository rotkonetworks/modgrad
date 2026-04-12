/// Benchmark: naive matvec vs tiled matvec vs CPU.
/// Also tests fused LN+SiLU and batched dispatch.

use modgrad_device::kfd::{HsaDevice, dispatch_queue::{GpuQueue, VramBuf}};
use modgrad_compute::backend::*;
use std::time::Instant;

fn bench<F: FnMut()>(name: &str, mut f: F, iters: usize) -> f64 {
    for _ in 0..5 { f(); }
    let start = Instant::now();
    for _ in 0..iters { f(); }
    let elapsed = start.elapsed().as_secs_f64();
    let per_iter_us = elapsed / iters as f64 * 1e6;
    println!("  {:<45} {:>8.1} us/iter", name, per_iter_us);
    per_iter_us
}

fn main() {
    println!("=== Tiled Matvec + Fused LN+SiLU Benchmark ===\n");

    let mut dev = match HsaDevice::open() {
        Ok(d) => d,
        Err(e) => { println!("No GPU: {}", e); return; }
    };
    let mut q = GpuQueue::new();
    let cpu = CpuBackend::new();

    // ─── Matvec: CPU vs naive GPU vs tiled GPU ───
    println!("--- Matvec: CPU vs naive vs tiled ---\n");
    for &(m, k) in &[(128, 512), (256, 256), (512, 256), (512, 512), (512, 640)] {
        println!("matvec({}x{}):", m, k);
        let w: Vec<f32> = (0..m*k).map(|i| ((i*7+3)%100) as f32 * 0.001 - 0.05).collect();
        let b: Vec<f32> = (0..m).map(|i| ((i*13)%100) as f32 * 0.001).collect();
        let x_data: Vec<f32> = (0..k).map(|i| ((i*11+5)%100) as f32 * 0.01 - 0.5).collect();
        let iters = 500;

        // CPU
        let mut y_cpu = vec![0.0f32; m];
        let cpu_us = bench("CPU (AVX-512)", || {
            cpu.matvec(&w, &b, &x_data, &mut y_cpu, m, k);
        }, iters);

        // Naive GPU (old kernel, 1 thread/row)
        let x_buf = q.alloc(&dev, k).unwrap();
        let y_buf = q.alloc(&dev, m).unwrap();
        x_buf.upload(&x_data);
        let naive_us = bench("GPU naive (1 thread/row)", || {
            q.enqueue_matvec(&mut dev, &w, &b, &x_buf, &y_buf, m, k);
            q.flush(&mut dev);
        }, iters);

        // Tiled GPU (1 WG/row, 256 threads cooperate)
        let x_buf2 = q.alloc(&dev, k).unwrap();
        let y_buf2 = q.alloc(&dev, m).unwrap();
        x_buf2.upload(&x_data);
        let tiled_us = bench("GPU tiled (256 threads/row, LDS reduce)", || {
            q.enqueue_matvec_tiled(&mut dev, &w, &b, &x_buf2, &y_buf2, m, k);
            q.flush(&mut dev);
        }, iters);

        // Verify tiled
        let y_tiled = y_buf2.download(m);
        let max_err: f32 = (0..m).map(|i| (y_cpu[i] - y_tiled[i]).abs()).fold(0.0f32, f32::max);
        println!("  verify: max_err={:.6}", max_err);
        println!("  naive vs CPU: {:.2}x  |  tiled vs CPU: {:.2}x\n",
            cpu_us / naive_us, cpu_us / tiled_us);
    }

    // ─── Fused LN+SiLU vs separate ───
    println!("--- Fused LN+SiLU vs separate dispatches ---\n");
    for &n in &[128, 256, 512] {
        println!("ln_silu(n={}):", n);
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 25.0).collect();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let iters = 1000;

        // CPU
        let mut x_cpu = x_data.clone();
        let cpu_us = bench("CPU (LN + SiLU separate)", || {
            x_cpu.copy_from_slice(&x_data);
            cpu.layer_norm_inplace(&mut x_cpu);
            cpu.silu_inplace(&mut x_cpu);
        }, iters);

        // GPU fused
        let x_buf = q.alloc(&dev, n).unwrap();
        let g_buf = q.alloc(&dev, n).unwrap();
        let b_buf = q.alloc(&dev, n).unwrap();
        g_buf.upload(&gamma);
        b_buf.upload(&beta);

        let gpu_us = bench("GPU fused (1 dispatch)", || {
            x_buf.upload(&x_data);
            q.enqueue_ln_silu(&mut dev, &x_buf, &g_buf, &b_buf, n);
            q.flush(&mut dev);
        }, iters);

        // Verify
        let y_gpu = x_buf.download(n);
        let mut y_ref = x_data.clone();
        cpu.layer_norm_inplace(&mut y_ref);
        cpu.silu_inplace(&mut y_ref);
        let max_err: f32 = (0..n).map(|i| (y_ref[i] - y_gpu[i]).abs()).fold(0.0f32, f32::max);
        println!("  verify: max_err={:.6}  speedup: {:.2}x\n", max_err, cpu_us / gpu_us);
    }

    // ─── Full synapse block: tiled matvec + fused LN+SiLU ───
    println!("--- Synapse block: matvec_tiled + ln_silu (2 dispatches) ---\n");
    for &(out, inp) in &[(512, 640), (341, 512), (170, 341), (16, 170)] {
        println!("block({}→{}):", inp, out);
        let w: Vec<f32> = (0..out*inp).map(|i| ((i*7+3)%100) as f32 * 0.001 - 0.05).collect();
        let b: Vec<f32> = (0..out).map(|i| ((i*13)%100) as f32 * 0.001).collect();
        let gamma = vec![1.0f32; out];
        let beta = vec![0.0f32; out];
        let x_data: Vec<f32> = (0..inp).map(|i| ((i*11+5)%100) as f32 * 0.01 - 0.5).collect();
        let iters = 300;

        // CPU
        let mut y_cpu = vec![0.0f32; out];
        let cpu_us = bench("CPU (matvec + LN + SiLU)", || {
            cpu.matvec(&w, &b, &x_data, &mut y_cpu, out, inp);
            cpu.layer_norm_inplace(&mut y_cpu);
            cpu.silu_inplace(&mut y_cpu);
        }, iters);

        // GPU: 2 dispatches (tiled matvec + fused LN+SiLU), 1 flush
        let x_buf = q.alloc(&dev, inp).unwrap();
        let y_buf = q.alloc(&dev, out).unwrap();
        let g_buf = q.alloc(&dev, out).unwrap();
        let b_buf = q.alloc(&dev, out).unwrap();
        x_buf.upload(&x_data);
        g_buf.upload(&gamma);
        b_buf.upload(&beta);

        let gpu_us = bench("GPU (matvec_tiled + ln_silu, 1 flush)", || {
            q.enqueue_matvec_tiled(&mut dev, &w, &b, &x_buf, &y_buf, out, inp);
            q.enqueue_ln_silu(&mut dev, &y_buf, &g_buf, &b_buf, out);
            q.flush(&mut dev);
        }, iters);

        println!("  speedup: {:.2}x\n", cpu_us / gpu_us);
    }

    // ─── Full UNet synapse: 7 blocks batched ───
    println!("--- Full UNet synapse: 7 blocks, batched ---\n");
    {
        let dims = [(512,640), (341,512), (170,341), (16,170),
                    (170,16), (341,170), (512,341)];
        let weights: Vec<Vec<f32>> = dims.iter()
            .map(|&(o,i)| (0..o*i).map(|j| ((j*7+3)%100) as f32 * 0.001 - 0.05).collect()).collect();
        let biases: Vec<Vec<f32>> = dims.iter()
            .map(|&(o,_)| (0..o).map(|j| ((j*13)%100) as f32 * 0.001).collect()).collect();
        let gammas: Vec<Vec<f32>> = dims.iter().map(|&(o,_)| vec![1.0f32; o]).collect();
        let betas: Vec<Vec<f32>> = dims.iter().map(|&(o,_)| vec![0.0f32; o]).collect();
        let iters = 200;

        // CPU
        let x_data: Vec<f32> = (0..640).map(|i| ((i*11)%100) as f32 * 0.01 - 0.5).collect();
        let cpu_us = bench("CPU (7 blocks sequential)", || {
            let mut cur = x_data.clone();
            for (idx, &(o, i)) in dims.iter().enumerate() {
                let mut out = vec![0.0f32; o];
                cpu.matvec(&weights[idx], &biases[idx], &cur[..i], &mut out, o, i);
                cpu.layer_norm_inplace(&mut out);
                cpu.silu_inplace(&mut out);
                cur = out;
            }
        }, iters);

        // GPU: all 14 dispatches (7×matvec_tiled + 7×ln_silu) + 1 flush
        let bufs: Vec<VramBuf> = [640, 512, 341, 170, 16, 170, 341, 512].iter()
            .map(|&n| q.alloc(&dev, n).unwrap()).collect();
        let g_bufs: Vec<VramBuf> = dims.iter()
            .map(|&(o,_)| { let b = q.alloc(&dev, o).unwrap(); b.upload(&vec![1.0f32; o]); b }).collect();
        let b_bufs: Vec<VramBuf> = dims.iter()
            .map(|&(o,_)| { let b = q.alloc(&dev, o).unwrap(); b.upload(&vec![0.0f32; o]); b }).collect();

        bufs[0].upload(&x_data);

        let gpu_us = bench("GPU batched (14 dispatches + 1 flush)", || {
            for (idx, &(o, _i)) in dims.iter().enumerate() {
                q.enqueue_matvec_tiled(&mut dev, &weights[idx], &biases[idx],
                    &bufs[idx], &bufs[idx+1], o, _i);
                q.enqueue_ln_silu(&mut dev, &bufs[idx+1], &g_bufs[idx], &b_bufs[idx], o);
            }
            q.flush(&mut dev);
        }, iters);

        println!("  speedup: {:.2}x\n", cpu_us / gpu_us);
    }
}
