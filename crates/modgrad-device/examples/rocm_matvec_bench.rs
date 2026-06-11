//! Benchmark the resident ROCm Q4_K matvec vs CPU, and the CPU Q5_K matvec
//! (the path that has no GPU kernel yet), to project per-token throughput.
//!
//! cargo run --release -p modgrad-device --features rocm --example rocm_matvec_bench -- <model.gguf>

#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn main() {
    use std::io::Cursor;
    use std::time::Instant;
    use rayon::prelude::*;
    use modgrad_device::kfd::gguf::{GgufFile, dequantize_row_q4_k, dequantize_row_q5_k, GgmlType};
    use modgrad_device::backend::HipBuffer;

    let path = std::env::args().nth(1).unwrap_or_else(||
        "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf".into());
    let file = std::fs::read(&path).expect("read gguf");
    let g = GgufFile::parse(&mut Cursor::new(&file)).expect("parse");

    let load = |name: &str| {
        let info = g.tensors.get(name).unwrap();
        let in_dim = info.dims[0]; let out_dim = info.dims[1];
        let bpr = in_dim / 256;
        let (bb, _) = info.dtype.block_size();
        let row_bytes = bpr * bb;
        let start = g.data_offset + info.offset;
        let bytes = file[start..start + out_dim * row_bytes].to_vec();
        (bytes, in_dim, out_dim, bpr, info.dtype)
    };

    let cpu_q4 = |bytes: &[u8], x: &[f32], in_dim: usize, out_dim: usize, bpr: usize| -> Vec<f32> {
        let rb = bpr * 144;
        (0..out_dim).into_par_iter().map(|r| {
            let row = &bytes[r*rb..(r+1)*rb];
            let mut w = vec![0.0f32; in_dim];
            dequantize_row_q4_k(row, &mut w, bpr);
            let mut a = 0.0f64; for k in 0..in_dim { a += w[k] as f64 * x[k] as f64; } a as f32
        }).collect()
    };
    let cpu_q5 = |bytes: &[u8], x: &[f32], in_dim: usize, out_dim: usize, bpr: usize| -> Vec<f32> {
        let rb = bpr * 176;
        (0..out_dim).into_par_iter().map(|r| {
            let row = &bytes[r*rb..(r+1)*rb];
            let mut w = vec![0.0f32; in_dim];
            dequantize_row_q5_k(row, &mut w, bpr);
            let mut a = 0.0f64; for k in 0..in_dim { a += w[k] as f64 * x[k] as f64; } a as f32
        }).collect()
    };

    let iters = 30;
    let bench = |label: &str, name: &str| {
        let (bytes, in_dim, out_dim, bpr, dt) = load(name);
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.21).sin() * 30.0).collect();
        // CPU
        let t = Instant::now();
        for _ in 0..iters { std::hint::black_box(match dt {
            GgmlType::Q4_K => cpu_q4(&bytes, &x, in_dim, out_dim, bpr),
            GgmlType::Q5_K => cpu_q5(&bytes, &x, in_dim, out_dim, bpr),
            _ => unreachable!(),
        }); }
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        // GPU steady-state (Q4_K only) — weight + x_dev + y_dev all resident,
        // only the kernel+sync is timed (what the real engine pays per matvec).
        let gpu_ms = if matches!(dt, GgmlType::Q4_K) {
            let w = HipBuffer::new(bytes.len()).unwrap();
            w.copy_from_host_bytes(&bytes).unwrap();
            let x_dev = HipBuffer::new(in_dim * 4).unwrap();
            x_dev.copy_from_host(&x).unwrap();
            let y_dev = HipBuffer::new(out_dim * 4).unwrap();
            w.matvec_q4k_into(&x_dev, &y_dev, out_dim, bpr).unwrap(); // warmup
            let t = Instant::now();
            for _ in 0..iters { w.matvec_q4k_into(&x_dev, &y_dev, out_dim, bpr).unwrap(); }
            t.elapsed().as_secs_f64() * 1000.0 / iters as f64
        } else { f64::NAN };
        println!("{label:<10} {name:<22} {:?} [{}x{}]  CPU {:>7.2}ms  GPU {:>6.3}ms  {}",
            dt, in_dim, out_dim, cpu_ms, gpu_ms,
            if gpu_ms.is_nan() { "(no GPU kernel)".into() } else { format!("{:.1}x", cpu_ms/gpu_ms) });
        (cpu_ms, gpu_ms)
    };

    println!("per-matvec ({iters} iters each):");
    let (q_cpu, q_gpu) = bench("attn_q", "blk.0.attn_q.weight");
    let (_k_cpu, _k_gpu) = bench("attn_k", "blk.0.attn_k.weight");
    let (g_cpu, g_gpu) = bench("ffn_gate", "blk.0.ffn_gate.weight");
    let (_u_cpu, _u_gpu) = bench("ffn_up", "blk.0.ffn_up.weight");
    let (v_cpu, _) = bench("attn_v", "blk.0.attn_v.weight");      // Q5_K, CPU only
    let (d_cpu, _) = bench("ffn_down", "blk.0.ffn_down.weight");  // Q5_K, CPU only

    // crude per-token projection (the 5 Q4_K + 2 Q5_K matvecs per layer × 48)
    let layer_cpu = q_cpu + _k_cpu + q_cpu/* attn_o ~ */ + g_cpu + _u_cpu + v_cpu + d_cpu;
    let layer_gpu_q4 = q_gpu + _k_gpu + q_gpu + g_gpu + _u_gpu; // Q4_K on GPU
    let layer_hybrid = layer_gpu_q4 + v_cpu + d_cpu;            // Q5_K still CPU
    println!("\nper-LAYER matvec sum (rough): all-CPU {:.1}ms | GPU-Q4_K+CPU-Q5_K {:.1}ms",
        layer_cpu, layer_hybrid);
    println!("per-TOKEN (×48, matvecs only): all-CPU {:.2}s | hybrid {:.2}s",
        layer_cpu*48.0/1000.0, layer_hybrid*48.0/1000.0);
    println!("(Q5_K ffn_down/attn_v still CPU — a Q5_K GPU kernel removes that bottleneck.)");
}

#[cfg(not(all(feature = "rocm", modgrad_hipcc_kernels)))]
fn main() { eprintln!("build with --features rocm"); }
