//! Validate the fused ROCm/HIP Q4_K matvec kernel against the verified
//! CPU dequant-dot, with a resident hipMalloc'd weight. KFD-free — runs
//! entirely on the stable HIP runtime, so it cannot wedge the display.
//!
//! Run: cargo run --release -p modgrad-device --features rocm --example rocm_matvec_check -- <model.gguf>

#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn main() {
    use std::io::Cursor;
    use modgrad_device::kfd::gguf::{GgufFile, dequantize_row_q4_k, dequantize_row_q5_k, GgmlType};
    use modgrad_device::backend::HipBuffer;

    let path = std::env::args().nth(1).unwrap_or_else(||
        "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf".into());
    let file = std::fs::read(&path).expect("read gguf");
    let g = GgufFile::parse(&mut Cursor::new(&file)).expect("parse");
    let info = g.tensors.get("blk.0.attn_q.weight").expect("attn_q");
    assert!(matches!(info.dtype, GgmlType::Q4_K), "expected Q4_K");

    let in_dim = info.dims[0];        // 3840
    let out_dim = info.dims[1];       // 4096
    let blocks_per_row = in_dim / 256;
    let row_bytes = blocks_per_row * 144;
    let total = out_dim * row_bytes;
    let start = g.data_offset + info.offset;
    let wbytes = &file[start..start + total];

    // Resident weight on the GPU (hipMalloc, uploaded once).
    let w = HipBuffer::new(total).expect("hipMalloc");
    w.copy_from_host_bytes(wbytes).expect("upload weight");

    // High-dynamic-range input (like Gemma's `cur`).
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.37).sin() * 180.0
        + ((i * 7) % 13) as f32).collect();

    let gpu = w.matvec_q4k(&x, out_dim, blocks_per_row).expect("rocm matvec");

    // CPU reference: verified dequant + f64 dot.
    let mut cpu = vec![0.0f32; out_dim];
    for r in 0..out_dim {
        let row = &wbytes[r * row_bytes..(r + 1) * row_bytes];
        let mut wr = vec![0.0f32; in_dim];
        dequantize_row_q4_k(row, &mut wr, blocks_per_row);
        let mut acc = 0.0f64;
        for k in 0..in_dim { acc += wr[k] as f64 * x[k] as f64; }
        cpu[r] = acc as f32;
    }

    let mut mx = 0f32; let mut at = 0;
    for i in 0..out_dim { let d = (gpu[i] - cpu[i]).abs(); if d > mx { mx = d; at = i; } }
    println!("out_dim={} maxabs={:.6} at row {} (gpu={:.4} cpu={:.4})", out_dim, mx, at, gpu[at], cpu[at]);
    println!("gpu[..4]={:?}", &gpu[..4]);
    println!("cpu[..4]={:?}", &cpu[..4]);
    println!("{}", if mx < 0.02 { "ROCm Q4_K matvec CORRECT ✓ (stable HIP dispatch)" } else { "DIVERGES ✗" });

    // ── Q5_K (ffn_down) ──
    let info5 = g.tensors.get("blk.0.ffn_down.weight").expect("ffn_down");
    assert!(matches!(info5.dtype, GgmlType::Q5_K), "expected Q5_K");
    let in5 = info5.dims[0]; let out5 = info5.dims[1];
    let bpr5 = in5 / 256; let rb5 = bpr5 * 176;
    let s5 = g.data_offset + info5.offset;
    let wb5 = &file[s5..s5 + out5 * rb5];
    let w5 = HipBuffer::new(wb5.len()).expect("hipMalloc q5");
    w5.copy_from_host_bytes(wb5).expect("upload q5");
    let x5: Vec<f32> = (0..in5).map(|i| (i as f32 * 0.11).sin() * 40.0).collect();
    let gpu5 = w5.matvec_q5k(&x5, out5, bpr5).expect("rocm q5 matvec");
    let mut cpu5 = vec![0.0f32; out5];
    for r in 0..out5 {
        let row = &wb5[r * rb5..(r + 1) * rb5];
        let mut wr = vec![0.0f32; in5];
        dequantize_row_q5_k(row, &mut wr, bpr5);
        let mut acc = 0.0f64; for k in 0..in5 { acc += wr[k] as f64 * x5[k] as f64; }
        cpu5[r] = acc as f32;
    }
    let mut mx5 = 0f32; for i in 0..out5 { let d = (gpu5[i] - cpu5[i]).abs(); if d > mx5 { mx5 = d; } }
    println!("Q5_K [{}x{}] maxabs={:.6}  {}", in5, out5, mx5,
        if mx5 < 0.03 { "ROCm Q5_K matvec CORRECT ✓" } else { "DIVERGES ✗" });
}

#[cfg(not(all(feature = "rocm", modgrad_hipcc_kernels)))]
fn main() {
    eprintln!("build with --features rocm (and hipcc present): \
        cargo run --release -p modgrad-device --features rocm --example rocm_matvec_check");
}
