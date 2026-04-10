//! Batched synapse forward: CPU rayon vs GPU KFD matmul
//! Tests at current "large" brain size and projected bigger brains.
//! run: cargo test --release --test kfd_qec_batch -- --nocapture --test-threads=1

use modgrad::kfd::{self, HsaDevice};
use modgrad::kfd::dispatch::{CodeObject, KernArgs};
use modgrad_compute::backend::dot;
use std::time::Instant;

static MATMUL_BLOCKED_CO: &[u8] = include_bytes!("../crates/modgrad-device/src/kfd/kernels/matmul_blocked.co");
static MATMUL_SMALL_CO: &[u8] = include_bytes!("../crates/modgrad-device/src/kfd/kernels/matmul_small.co");

/// CPU synapse forward for one sample: y = W*x (+ bias ignored for bench)
/// W is [out_dim × in_dim], x is [in_dim], y is [out_dim]
fn cpu_synapse_forward(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    for r in 0..out_dim {
        y[r] = dot(&w[r * in_dim..(r + 1) * in_dim], x);
    }
}

/// CPU GLU + SiLU + LayerNorm on a batch of synapse outputs
fn cpu_elementwise_batch(data: &mut [f32], out_dim: usize, batch: usize) {
    let half = out_dim / 2;
    for b in 0..batch {
        let base = b * out_dim;
        // GLU: out[i] = data[i] * sigmoid(data[half+i])
        for i in 0..half {
            let gate = 1.0 / (1.0 + (-data[base + half + i]).exp());
            data[base + i] *= gate;
        }
        // SiLU on first half
        for i in 0..half {
            let x = data[base + i];
            data[base + i] = x / (1.0 + (-x).exp());
        }
        // LayerNorm on first half
        let mut sum = 0.0f32;
        let mut sum2 = 0.0f32;
        for i in 0..half {
            sum += data[base + i];
            sum2 += data[base + i] * data[base + i];
        }
        let mean = sum / half as f32;
        let var = sum2 / half as f32 - mean * mean;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for i in 0..half {
            data[base + i] = (data[base + i] - mean) * inv_std;
        }
    }
}

#[test]
fn qec_batch_bench() {
    if !kfd::is_available() { eprintln!("skip: no kfd"); return; }
    let mut dev = match HsaDevice::open() { Ok(d) => d, Err(e) => { eprintln!("skip: {e}"); return; }};

    for co_bytes in [MATMUL_BLOCKED_CO, MATMUL_SMALL_CO] {
        let co = CodeObject::load(&dev.alloc, co_bytes).unwrap();
        if dev.kernels.is_none() { dev.kernels = Some(std::collections::HashMap::new()); }
        for (name, entry) in &co.kernels { dev.kernels.as_mut().unwrap().insert(name.clone(), entry.clone()); }
        std::mem::forget(co);
    }

    println!();
    println!("  Batched synapse forward: CPU (rayon 32 cores) vs GPU (KFD matmul)");
    println!("  Simulates one tick: 8 synapse layers × batch_size samples");
    println!();

    // Synapse shapes for the "large" brain config and scaled versions
    // Each synapse: Linear(in_dim → out_dim*2), then GLU halves it
    let configs: &[(&str, &[(usize, usize)])] = &[
        ("current large (5K neurons)", &[
            (1280, 2560),  // syn_motor_input: 1280 in → 1280 out (×2 for GLU)
            (2560, 2048),  // syn_input_attn
            (2048, 2048),  // syn_attn_output
            (2048, 1536),  // syn_output_motor
            (1280, 2048),  // cerebellum
            (1024, 512),   // basal_ganglia
            (512, 256),    // insula
            (2048, 512),   // hippocampus
        ]),
        ("4x brain (20K neurons)", &[
            (5120, 10240),
            (10240, 8192),
            (8192, 8192),
            (8192, 6144),
            (5120, 8192),
            (4096, 2048),
            (2048, 1024),
            (8192, 2048),
        ]),
        ("isis target (50K neurons)", &[
            (12800, 25600),
            (25600, 20480),
            (20480, 20480),
            (20480, 15360),
            (12800, 20480),
            (10240, 5120),
            (5120, 2560),
            (20480, 5120),
        ]),
    ];

    let batch = 32u32;
    let ticks = 12;

    println!("  {:>25} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "config", "cpu/tick", "gpu/tick", "speedup", "cpu_12", "gpu_12");
    println!("  {}", "-".repeat(80));

    for &(name, synapses) in configs {
        // --- CPU benchmark: rayon across batch, sequential per sample ---
        let cpu_us_per_tick = {
            // Allocate weights and data for all synapses
            let syn_data: Vec<(Vec<f32>, usize, usize)> = synapses.iter()
                .map(|&(in_dim, out_dim)| {
                    (vec![0.001f32; out_dim * in_dim], in_dim, out_dim)
                }).collect();

            let x_data: Vec<Vec<f32>> = synapses.iter()
                .map(|&(in_dim, _)| vec![0.001f32; in_dim])
                .collect();

            let iters = if synapses[0].0 > 5000 { 10 } else { 100 };
            let t0 = Instant::now();
            for _ in 0..iters {
                // Simulate one tick: 8 synapses, batch samples via rayon
                use rayon::prelude::*;
                let _results: Vec<Vec<f32>> = (0..batch).into_par_iter().map(|_| {
                    let mut all_out = Vec::new();
                    for (i, &(in_dim, out_dim)) in synapses.iter().enumerate() {
                        let mut y = vec![0.0f32; out_dim];
                        cpu_synapse_forward(&syn_data[i].0, &x_data[i], &mut y, out_dim, in_dim);
                        // GLU+SiLU+LN
                        let half = out_dim / 2;
                        for j in 0..half {
                            let gate = 1.0 / (1.0 + (-y[half + j]).exp());
                            y[j] = y[j] * gate / (1.0 + (-y[j] * gate).exp());
                        }
                        all_out.extend_from_slice(&y[..half]);
                    }
                    all_out
                }).collect();
            }
            t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0
        };

        // --- GPU benchmark: batched matmuls ---
        let gpu_us_per_tick = {
            let mut total_us = 0.0f64;

            for &(in_dim, out_dim) in synapses {
                let m = out_dim as u32;
                let k = in_dim as u32;
                let n = batch;

                // Select kernel
                let (kernel_name, nwg, w_buf) = if m >= 1536 {
                    let w = dev.upload_f32(&vec![0.001f32; (m * k) as usize]).unwrap();
                    ("matmul_blocked", ((m + 127) / 128) * ((n + 31) / 32), w)
                } else {
                    let w = dev.upload_f32_col_major(
                        &vec![0.001f32; (m * k) as usize], m as usize, k as usize).unwrap();
                    ("matmul_small", ((m + 31) / 32) * ((n + 31) / 32), w)
                };
                let b_buf = dev.upload_f32(&vec![0.0f32; m as usize]).unwrap();
                let x_buf = dev.upload_f32(&vec![0.001f32; (n * k) as usize]).unwrap();
                let y_buf = dev.alloc_output((n as usize * m as usize * 4 + 64) as usize).unwrap();

                let mut args = KernArgs::new();
                args.push_ptr(&w_buf); args.push_ptr(&b_buf);
                args.push_ptr(&x_buf); args.push_ptr(&y_buf);
                args.push_u32(m); args.push_u32(k); args.push_u32(n);
                let args_buf = args.upload(&dev.alloc).unwrap();

                let grid = [nwg, 1, 1];
                let block = [256, 1, 1];

                // warmup
                for _ in 0..20 { dev.dispatch_enqueue(kernel_name, &args_buf, grid, block); }
                dev.submit_wait(30_000);

                // bench
                let iters = if in_dim > 5000 { 50 } else { 200 };
                let t0 = Instant::now();
                for _ in 0..iters { dev.dispatch_enqueue(kernel_name, &args_buf, grid, block); }
                dev.submit_wait(60_000);
                total_us += t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
            }

            // Add elementwise CPU time (GLU+SiLU+LN is cheap)
            let elem_per_tick: usize = synapses.iter().map(|&(_, out)| out * batch as usize).sum();
            let mut elem_buf = vec![0.001f32; elem_per_tick];
            let t0 = Instant::now();
            for _ in 0..100 {
                let mut off = 0;
                for &(_, out_dim) in synapses {
                    cpu_elementwise_batch(&mut elem_buf[off..off + out_dim * batch as usize],
                        out_dim, batch as usize);
                    off += out_dim * batch as usize;
                }
            }
            total_us += t0.elapsed().as_nanos() as f64 / 100.0 / 1000.0;

            total_us
        };

        let speedup = cpu_us_per_tick / gpu_us_per_tick;
        let cpu_12 = cpu_us_per_tick * ticks as f64 / 1000.0;
        let gpu_12 = gpu_us_per_tick * ticks as f64 / 1000.0;

        println!("  {:>25} {:>8.0} us {:>8.0} us {:>8.1}x {:>6.1}ms {:>6.1}ms",
            name, cpu_us_per_tick, gpu_us_per_tick, speedup, cpu_12, gpu_12);
    }
    println!();
    println!("  GPU times include elementwise (GLU+SiLU+LN) on CPU.");
    println!("  CPU times use rayon across {} samples (32 cores).", batch);
    println!();
}
