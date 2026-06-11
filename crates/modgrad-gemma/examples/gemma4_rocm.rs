//! Fully GPU-resident Gemma-4 forward on ROCm — validate (argmax 532) + time.
//!
//! cargo run --release -p modgrad-device --features rocm --example gemma4_rocm -- <model.gguf> [tok ...]

#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn main() {
    use std::io::Cursor;
    use std::time::Instant;
    use modgrad_device::kfd::gguf::GgufFile;
    use modgrad_gemma::rocm_gemma::RocmGemma;

    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(||
        "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf".into());
    let tokens: Vec<u32> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse().unwrap()).collect()
    } else { vec![2, 669, 5279, 529, 7001, 563] };

    eprintln!("[1/3] reading + uploading weights to VRAM...");
    let file = std::fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&mut Cursor::new(&file)).expect("parse");
    let t0 = Instant::now();
    let mut model = RocmGemma::load(&gguf, &file, 512).expect("load");
    eprintln!("    loaded in {:.1}s", t0.elapsed().as_secs_f64());

    eprintln!("[2/3] prefill {} tokens...", tokens.len());
    let mut logits = Vec::new();
    let mut times = Vec::new();
    for &tok in &tokens {
        let t = Instant::now();
        logits = model.forward_token(tok).expect("forward");
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    eprintln!("    per-token ms: {:?}", times.iter().map(|t| t.round() as i64).collect::<Vec<_>>());

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    eprintln!("[3/3] top-5:");
    for &t in idx.iter().take(5) { println!("    [{t}] {:.4}", logits[t]); }
    println!("argmax = {} (CPU oracle = 532 '▁and')", idx[0]);

    // steady-state decode timing: generate a few more tokens greedily
    let mut last = idx[0] as u32;
    let mut dt = Vec::new();
    for _ in 0..8 {
        let t = Instant::now();
        let lg = model.forward_token(last).expect("decode");
        dt.push(t.elapsed().as_secs_f64() * 1000.0);
        last = (0..lg.len()).max_by(|&a, &b| lg[a].partial_cmp(&lg[b]).unwrap()).unwrap() as u32;
    }
    let avg = dt.iter().sum::<f64>() / dt.len() as f64;
    println!("decode: {:.1} ms/token  →  {:.1} tok/s", avg, 1000.0 / avg);
}

#[cfg(not(all(feature = "rocm", modgrad_hipcc_kernels)))]
fn main() { eprintln!("build with --features rocm (hipcc required)"); }
