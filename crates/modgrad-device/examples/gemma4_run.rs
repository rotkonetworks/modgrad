//! gemma4_run — drive the resident `Gemma4Model` engine for one forward and
//! compare its logits against the validated `gemma4_infer` reference.
//!
//! Stage 1 of the inference-API work: prove the quantized-block engine
//! (`kfd::inference::Gemma4Model`) produces correct logits on the dense 12B
//! before we build a generate loop + server on it.
//!
//! Default tokens are "The capital of France is" (same ids as gemma4_infer);
//! the reference argmax is 107 ("\n") with healthy ~16-18 logits.
//!
//! Run: cargo run --release -p modgrad-device --example gemma4_run -- <model.gguf> [tok ...]

use std::io::Cursor;

use modgrad_device::kfd::HsaDevice;
use modgrad_device::kfd::dispatch_queue::GpuQueue;
use modgrad_device::kfd::gguf::GgufFile;
use modgrad_device::kfd::inference::Gemma4Model;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf".into()
    });
    let tokens: Vec<u32> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse().unwrap()).collect()
    } else {
        vec![2, 818, 41626, 1340, 31756, 511]
    };

    eprintln!("[1/4] reading {path}");
    let file = std::fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&mut Cursor::new(&file)).expect("parse gguf");

    eprintln!("[2/4] opening GPU + uploading weights to VRAM");
    let dev = HsaDevice::open().expect("open /dev/kfd");
    let mut queue = GpuQueue::new();
    // Stage-1 validation only needs a handful of tokens; keep KV small so it
    // fits alongside the 6.2 GB of weights in 8 GB VRAM. (Real serving needs
    // SWA-windowed + f16 KV — Stage 2/3.)
    let mut model = Gemma4Model::load(&gguf, &file, &dev, &mut queue, 256)
        .expect("load model");

    eprintln!("[3/4] forward over {} prompt tokens", tokens.len());
    let mut dev = dev;
    let mut logits = Vec::new();
    for &tok in &tokens {
        logits = model.forward_token(tok, &mut dev, &mut queue);
    }

    eprintln!("[4/4] top-5 logits (compare to gemma4_infer):");
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    for &t in idx.iter().take(5) {
        println!("    [{t}] logit={:.4}", logits[t]);
    }
    println!("argmax = {}", idx[0]);
}
