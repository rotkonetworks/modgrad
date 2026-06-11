//! Validate the resident `matvec_q4k` GPU kernel against the (now-correct)
//! CPU Q4_K matvec, on a real model weight with a non-uniform input.
//!
//! Run: cargo run --release -p modgrad-device --example gpu_matvec_check -- <model.gguf>

use std::io::Cursor;

use modgrad_device::kfd::HsaDevice;
use modgrad_device::kfd::dispatch_queue::GpuQueue;
use modgrad_device::kfd::gguf::GgufFile;
use modgrad_device::kfd::inference::Gemma4Model;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(||
        "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf".into());
    let file = std::fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&mut Cursor::new(&file)).expect("parse");
    let dev = HsaDevice::open().expect("open kfd");
    let mut queue = GpuQueue::new();
    let model = Gemma4Model::load(&gguf, &file, &dev, &mut queue, 64).expect("load");
    let mut dev = dev;

    // Non-uniform deterministic input (uniform input hides ordering bugs).
    let n = 3840usize;
    let smooth: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 2.0).collect();
    // High-dynamic-range input mimicking Gemma's `cur` (values into the hundreds).
    let hidr: Vec<f32> = (0..n).map(|i| {
        let s = if i % 2 == 0 { 1.0 } else { -1.0 };
        s * ((i as f32 * 0.37).sin().abs() * 180.0 + ((i * 7) % 13) as f32)
    }).collect();
    // Real `cur` captured from the forward (if present).
    let real_cur: Vec<f32> = std::fs::read("/tmp/cur.bin").ok()
        .map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())
        .unwrap_or_default();
    let mut cases: Vec<(&str, &Vec<f32>)> = vec![("SMOOTH", &smooth), ("HIGH-DYN-RANGE", &hidr)];
    if real_cur.len() == n { cases.push(("REAL-CUR", &real_cur)); }
    for (lbl, xx) in cases {
        if let Some((g, c)) = model.gpu_matvec_check(&mut dev, &mut queue, "blk.0.attn_q.weight", xx) {
            let mut mx = 0f32; let mut at = 0usize;
            for i in 0..g.len() { let d = (g[i]-c[i]).abs(); if d > mx { mx = d; at = i; } }
            println!("  [{lbl}] maxabs={:.5} at row {} (gpu={:.4} cpu={:.4})", mx, at, g[at], c[at]);
        }
    }
    let x = smooth;

    // Test several weights + repeated dispatch (the forward does ~240/token).
    for (label, name) in [("attn_q#1","blk.0.attn_q.weight"),
                          ("ffn_gate","blk.0.ffn_gate.weight"),
                          ("attn_q#2","blk.0.attn_q.weight"),
                          ("attn_q#3","blk.0.attn_q.weight")] {
        if let Some((g,c)) = model.gpu_matvec_check(&mut dev, &mut queue, name, &x) {
            let mut mx=0f32; for i in 0..g.len(){ let d=(g[i]-c[i]).abs(); if d>mx{mx=d;} }
            println!("  {label}: out={} maxabs={:.5} {}", g.len(), mx,
                if mx<0.02 {"OK"} else {"DIVERGES"});
        } else { println!("  {label}: dispatch returned None"); }
    }

    let (gpu, cpu) = model
        .gpu_matvec_check(&mut dev, &mut queue, "blk.0.attn_q.weight", &x)
        .expect("gpu_matvec_check failed (dispatch/alloc)");

    let mut maxabs = 0f32;
    let mut maxrel = 0f32;
    let mut sumsq = 0f64;
    for i in 0..gpu.len() {
        let d = (gpu[i] - cpu[i]).abs();
        if d > maxabs { maxabs = d; }
        let rel = d / (cpu[i].abs() + 1e-4);
        if rel > maxrel { maxrel = rel; }
        sumsq += (d * d) as f64;
    }
    println!("out_dim={} maxabs={:.5} maxrel={:.4} rms_err={:.6}",
        gpu.len(), maxabs, maxrel, (sumsq / gpu.len() as f64).sqrt());
    println!("gpu[..6]={:?}", &gpu[..6]);
    println!("cpu[..6]={:?}", &cpu[..6]);
    if maxrel < 0.02 {
        println!("VERDICT: kernel MATCHES cpu (≤2%) — GPU matvec is correct ✓");
    } else {
        println!("VERDICT: kernel DIVERGES — needs fixing ✗");
    }
}
