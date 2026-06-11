//! Validate the dp4a Q4_K matvec (quantize_q8_1 + matvec_q4k_dp4a) against the
//! CPU dequant-dot. int8 activation → expect ~1% error, not bit-exact. Tiny data.
//!
//! cargo run --release -p modgrad-device --features rocm --example rocm_dp4a_check

#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn main() {
    use std::io::Cursor;
    use std::time::Instant;
    use modgrad_device::kfd::gguf::{GgufFile, dequantize_row_q4_k, GgmlType};
    use modgrad_device::backend::rocm::{HipBuffer, kern};

    let path = std::env::args().nth(1).unwrap_or_else(||
        "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf".into());
    let file = std::fs::read(&path).expect("read");
    let g = GgufFile::parse(&mut Cursor::new(&file)).expect("parse");

    let bias = std::env::args().nth(2).and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
    let check = |name: &str| {
        let info = g.tensors.get(name).unwrap();
        assert!(matches!(info.dtype, GgmlType::Q4_K));
        let in_dim = info.dims[0]; let out_dim = info.dims[1]; let bpr = in_dim / 256;
        let rb = bpr * 144; let total = out_dim * rb;
        let start = g.data_offset + info.offset;
        let wb = &file[start..start + total];
        let w = HipBuffer::new(total).unwrap(); w.copy_from_host_bytes(wb).unwrap();
        // bias arg lets us test zero-mean (0) vs biased (e.g. 100) activations.
        // If /tmp/cur2.bin exists and matches in_dim, use the REAL engine activation.
        let real: Vec<f32> = std::fs::read("/tmp/cur2.bin").ok()
            .map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())
            .unwrap_or_default();
        let x: Vec<f32> = if real.len() == in_dim && bias == 0.0 {
            eprintln!("    [using REAL cur.bin activation]");
            real
        } else {
            (0..in_dim).map(|i| (i as f32 * 0.21).sin() * 30.0 + bias).collect()
        };
        let xdev = HipBuffer::new(in_dim * 4).unwrap(); xdev.copy_from_host(&x).unwrap();
        let nsub = in_dim / 32;
        let xq = HipBuffer::new(in_dim).unwrap();
        let xd = HipBuffer::new(nsub * 4).unwrap();
        let xs = HipBuffer::new(nsub * 4).unwrap();
        let y = HipBuffer::new(out_dim * 4).unwrap();
        // warmup + timed loop (weight resident; quant+matvec per call)
        let run = || unsafe {
            kern::quantize_q8_1(xdev.f32_ptr(), xq.device_ptr() as *mut i8, xd.f32_at(0), xs.f32_at(0), in_dim).unwrap();
            kern::matvec_q4k_dp4a(w.u8_ptr(), xq.device_ptr() as *const i8, xd.f32_ptr(), xs.f32_ptr(), y.f32_at(0), out_dim, bpr).unwrap();
            kern::hip_sync().unwrap();
        };
        run();
        let it = 30; let t = Instant::now(); for _ in 0..it { run(); }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / it as f64;
        let mut gpu = vec![0f32; out_dim]; y.copy_to_host(&mut gpu).unwrap();
        let mut cpu = vec![0f32; out_dim];
        for r in 0..out_dim {
            let mut wr = vec![0f32; in_dim];
            dequantize_row_q4_k(&wb[r * rb..(r + 1) * rb], &mut wr, bpr);
            let mut a = 0f64; for k in 0..in_dim { a += wr[k] as f64 * x[k] as f64; }
            cpu[r] = a as f32;
        }
        let mut mx = 0f32; let mut se = 0f64; let mut sc = 0f64;
        for i in 0..out_dim { let d = (gpu[i]-cpu[i]).abs(); if d>mx {mx=d;} se += (d*d) as f64; sc += (cpu[i]*cpu[i]) as f64; }
        let rms_rel = (se/sc).sqrt();   // ‖err‖ / ‖cpu‖ — the meaningful int8 metric
        println!("{name:<22} [{in_dim}x{out_dim}] maxabs={mx:.4} RMS_rel={rms_rel:.4} GPU {ms:.3}ms  {}",
            if rms_rel < 0.02 { "OK ✓ (int8 noise)" } else { "TOO HIGH ✗" });
        println!("    gpu[..5]={:?}", &gpu[..5].iter().map(|v|(v*1000.0).round()/1000.0).collect::<Vec<_>>());
        println!("    cpu[..5]={:?}", &cpu[..5].iter().map(|v|(v*1000.0).round()/1000.0).collect::<Vec<_>>());
    };
    check("blk.0.attn_q.weight");      // in=3840 bpr=15
    check("blk.0.attn_k.weight");      // in=3840 out=2048
    check("blk.0.attn_output.weight"); // in=4096 bpr=16
    check("blk.5.attn_output.weight"); // in=8192 bpr=32 (global)
    check("blk.0.ffn_gate.weight");
}

#[cfg(not(all(feature = "rocm", modgrad_hipcc_kernels)))]
fn main() { eprintln!("build with --features rocm"); }
