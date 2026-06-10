//! gguf-inspect — dump a GGUF file's architecture, dims, and per-tensor
//! quant histogram, and report whether our matvec kernels cover it.
//!
//! Run:
//!   cargo run --release --example gguf_inspect -- /path/to/model.gguf

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;

use modgrad_device::kfd::gguf::{GgmlType, GgufFile};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf".to_string()
    });
    eprintln!("inspecting: {path}\n");

    let f = File::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut r = BufReader::new(f);
    let gguf = GgufFile::parse(&mut r).unwrap_or_else(|e| panic!("parse: {e}"));

    let arch = gguf.architecture().unwrap_or("<none>").to_string();
    println!("architecture : {arch}");
    if let Some(MetaStr(name)) = gguf.meta("general.name").map(meta_str) {
        println!("general.name : {name}");
    }

    // Architecture config (llama.cpp metadata-key convention).
    let keys = [
        "block_count",
        "embedding_length",
        "feed_forward_length",
        "context_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "attention.sliding_window",
        "vocab_size",
    ];
    println!("\n── config ──");
    for k in keys {
        let full = format!("{arch}.{k}");
        match gguf.meta_u32(&full) {
            Some(v) => println!("  {k:28} = {v}"),
            None => println!("  {k:28} = <absent>"),
        }
    }
    for k in ["attention.layer_norm_rms_epsilon", "rope.freq_base", "attention.value_residual_scale"] {
        if let Some(v) = gguf.meta_f32(&format!("{arch}.{k}")) {
            println!("  {k:28} = {v}");
        }
    }

    // Attention dims DERIVED from tensor shapes (ground truth) — mirrors
    // the inference.rs config derivation, validated here without a GPU load.
    let d_model = gguf.meta_u32(&format!("{arch}.embedding_length")).unwrap_or(0) as usize;
    let dim1 = |name: &str| gguf.tensors.get(name).and_then(|t| t.dims.get(1).copied());
    let dim0 = |name: &str| gguf.tensors.get(name).and_then(|t| t.dims.first().copied());
    let q_dim = dim1("blk.0.attn_q.weight");
    let kv_dim = dim1("blk.0.attn_k.weight");
    let head_dim = dim0("blk.0.attn_k_norm.weight");
    println!("\n── attention dims (derived from tensor shapes) ──");
    println!("  d_model   = {d_model}");
    println!("  q_dim     = {q_dim:?}   (= n_heads × head_dim)");
    println!("  kv_dim    = {kv_dim:?}   (= n_kv_heads × head_dim)");
    println!("  head_dim  = {head_dim:?}   (from attn_k_norm)");
    if let (Some(kv), Some(hd)) = (kv_dim, head_dim) {
        println!("  n_kv_heads= {}   (kv_dim / head_dim)", kv / hd.max(1));
    }
    println!("  (metadata key_length = {:?} — the misleading field we now ignore)",
        gguf.meta_u32(&format!("{arch}.attention.key_length")));

    // Per-quant-type histogram.
    let mut by_type: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // type -> (count, bytes)
    let mut total_bytes = 0usize;
    for t in gguf.tensors.values() {
        let e = by_type.entry(format!("{:?}", t.dtype)).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.data_bytes();
        total_bytes += t.data_bytes();
    }
    println!("\n── tensor quant histogram ({} tensors, {:.2} GiB total) ──",
        gguf.tensors.len(), total_bytes as f64 / (1u64 << 30) as f64);
    for (ty, (count, bytes)) in &by_type {
        let supported = matches!(ty.as_str(), "F32" | "F16" | "BF16" | "Q4_K" | "Q5_K");
        let mark = if supported { "OK  " } else { "NEED" };
        println!("  [{mark}] {ty:6} : {count:4} tensors, {:.2} GiB",
            *bytes as f64 / (1u64 << 30) as f64);
    }

    // Verdict: which quant kernels are missing.
    let needs: Vec<&String> = by_type.keys()
        .filter(|ty| !matches!(ty.as_str(), "F32" | "F16" | "BF16" | "Q4_K" | "Q5_K"))
        .collect();
    println!("\n── verdict ──");
    if arch != "gemma4" {
        println!("  ! architecture is '{arch}', engine expects 'gemma4' — needs an arch path.");
    } else {
        println!("  ✓ architecture 'gemma4' — matches the inference engine.");
    }
    if needs.is_empty() {
        println!("  ✓ all tensor quant types covered by existing Q4_K/Q5_K matvec.");
    } else {
        println!("  ! missing matvec kernels for: {needs:?} (likely Q6_K on output/ffn_down).");
    }
    println!("  weights = {:.2} GiB; on 8GB VRAM that leaves ~{:.2} GiB for KV+activations+display.",
        total_bytes as f64 / (1u64 << 30) as f64,
        8.0 - total_bytes as f64 / (1u64 << 30) as f64);

    // Show a few representative tensor names + shapes.
    println!("\n── sample tensors ──");
    let mut names: Vec<&String> = gguf.tensor_list.iter().take(6).collect();
    for n in ["output.weight", "token_embd.weight", "output_norm.weight"] {
        if let Some(t) = gguf.tensors.get(n) {
            let s = &t.name;
            if !names.iter().any(|x| *x == s) {
                names.push(s);
            }
        }
    }
    for n in names {
        if let Some(t) = gguf.tensors.get(n) {
            println!("  {:40} {:?} {:?}", t.name, t.dims, t.dtype);
        }
    }
    let _ = GgmlType::F32; // keep the import if unused above
}

// Tiny helper to pull a string MetaValue without importing the enum variants.
struct MetaStr(String);
fn meta_str(v: &modgrad_device::kfd::gguf::MetaValue) -> MetaStr {
    MetaStr(v.as_str().unwrap_or("<non-str>").to_string())
}
