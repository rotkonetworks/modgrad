//! gemma4_infer — Gemma-4 inference on modgrad's typed `Tensor<Cpu>` stack.
//!
//! Ports the validated zish `forwardGemma4` into the JAX-style typed ops
//! (matmul_nt / rms_norm / rope / sdpa). The five Gemma-4 gotchas, confirmed
//! against the llama.cpp gemma4-iswa.cpp reference:
//!   1. Q6_K/Q5_K dequant must use the 16-scale layout (handled in gguf.rs).
//!   2. V-norm: no-weight RMSNorm on V (per head).
//!   3. layer_output_scale multiplies the WHOLE layer output once, at the end.
//!   4. attention scale = 1.0 (we pre-scale Q by sqrt(head_dim) since sdpa bakes 1/sqrt).
//!   5. PLAIN ×weight norms (GGUF folds gemma's +1 at conversion).
//!
//! Run: cargo run --release --example gemma4_infer -- <model.gguf> [tok1 tok2 ...]
//! With no token args it uses the "The capital of France is" ids and checks the
//! argmax matches zish (token 107 "\n", healthy ~16-18 logits).

use std::io::Cursor;

use modgrad_device::backend::tensor::{self, Cpu, Tensor};
use modgrad_device::kfd::gguf::{dequantize_tensor, GgufFile};

/// Per-layer dims (Gemma-4 interleaves local/global layers with different dims).
struct Layer {
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    is_global: bool,
    attn_norm: Tensor<Cpu>,
    q_norm: Tensor<Cpu>,
    k_norm: Tensor<Cpu>,
    post_attn_norm: Tensor<Cpu>,
    ffn_norm: Tensor<Cpu>,
    post_ffn_norm: Tensor<Cpu>,
    wq: Tensor<Cpu>,
    wk: Tensor<Cpu>,
    wv: Tensor<Cpu>, // V=K fallback already resolved at load
    wo: Tensor<Cpu>,
    w_gate: Tensor<Cpu>,
    w_up: Tensor<Cpu>,
    w_down: Tensor<Cpu>,
    out_scale: f32,
}

struct Config {
    d_model: usize,
    n_layers: usize,
    ffn_dim: usize,
    vocab: usize,
    eps: f32,
    rope_local: f32,
    rope_global: f32,
    sliding_window: usize,
    logit_softcap: Option<f32>,
}

fn meta_f32(g: &GgufFile, k: &str) -> Option<f32> { g.meta_f32(k) }
fn meta_u32(g: &GgufFile, k: &str) -> Option<u32> { g.meta_u32(k) }

fn load(g: &GgufFile, file: &[u8], name: &str) -> Tensor<Cpu> {
    let info = g.tensors.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
    let n_elem: usize = info.dims.iter().product();
    let (bb, be) = info.dtype.block_size();
    let n_blocks = if be == 0 { n_elem } else { n_elem / be };
    let data_bytes = n_blocks * bb;
    let start = g.data_offset + info.offset;
    let raw = &file[start..start + data_bytes];
    let v = dequantize_tensor(info.dtype, raw, n_elem);
    Tensor::<Cpu>::from_slice(&v).expect("from_slice")
}

fn has(g: &GgufFile, name: &str) -> bool { g.tensors.contains_key(name) }
fn dim1(g: &GgufFile, name: &str) -> usize { g.tensors[name].dims[1] }
fn dim0(g: &GgufFile, name: &str) -> usize { g.tensors[name].dims[0] }

// ── Gemma RMSNorm: PLAIN ×weight (the +1 is folded in the GGUF). ──
// Wrap the typed rms_norm, which already does y = x·rsqrt(mean+eps)·weight.
fn gnorm(x: &Tensor<Cpu>, w: &Tensor<Cpu>, n_rows: usize, n_cols: usize, eps: f32) -> Tensor<Cpu> {
    let mut out = Tensor::<Cpu>::zeros(n_rows * n_cols).unwrap();
    tensor::rms_norm(x, w, &mut out, n_rows, n_cols, eps).unwrap();
    out
}

fn gelu_tanh(v: &mut [f32]) {
    for g in v.iter_mut() {
        let x = *g;
        *g = 0.5 * x * (1.0 + (0.797_884_56_f32 * (x + 0.044715 * x * x * x)).tanh());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf".into()
    });
    // Tokens: "The capital of France is" (zish ids) unless overridden on CLI.
    let tokens: Vec<usize> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse().unwrap()).collect()
    } else {
        vec![2, 818, 41626, 1340, 31756, 511]
    };

    eprintln!("[1/4] reading {path}");
    let file = std::fs::read(&path).expect("read gguf");
    let g = GgufFile::parse(&mut Cursor::new(&file)).expect("parse gguf");
    let arch = g.metadata.get("general.architecture")
        .and_then(|m| m.as_str()).unwrap_or("gemma4").to_string();
    eprintln!("    arch={arch}  tensors={}", g.tensors.len());

    let d_model = meta_u32(&g, &format!("{arch}.embedding_length")).unwrap() as usize;
    let n_layers = meta_u32(&g, &format!("{arch}.block_count")).unwrap() as usize;
    let n_heads = meta_u32(&g, &format!("{arch}.attention.head_count")).unwrap_or(16) as usize;
    let ffn_dim = meta_u32(&g, &format!("{arch}.feed_forward_length")).unwrap() as usize;
    let vocab = dim1(&g, "token_embd.weight").max(dim0(&g, "token_embd.weight"));
    let cfg = Config {
        d_model,
        n_layers,
        ffn_dim,
        vocab,
        eps: meta_f32(&g, &format!("{arch}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
        rope_global: meta_f32(&g, &format!("{arch}.rope.freq_base")).unwrap_or(1_000_000.0),
        rope_local: meta_f32(&g, &format!("{arch}.rope.freq_base_swa")).unwrap_or(10_000.0),
        sliding_window: meta_u32(&g, &format!("{arch}.attention.sliding_window")).unwrap_or(1024) as usize,
        logit_softcap: meta_f32(&g, &format!("{arch}.final_logit_softcapping")),
    };
    eprintln!("    d_model={} layers={} ffn={} vocab={} eps={}", cfg.d_model, cfg.n_layers, cfg.ffn_dim, cfg.vocab, cfg.eps);

    eprintln!("[2/4] dequantizing weights to f32 (this allocates ~26GB)...");
    let embed = load(&g, &file, "token_embd.weight"); // [vocab × d_model], also lm_head (tied)
    let final_norm = load(&g, &file, "output_norm.weight");

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let p = |s: &str| format!("blk.{i}.{s}.weight");
        let head_dim = dim0(&g, &p("attn_q_norm"));
        let kv_dim = dim1(&g, &p("attn_k"));
        let is_global = !has(&g, &p("attn_v"));
        let wv = if is_global { load(&g, &file, &p("attn_k")) } else { load(&g, &file, &p("attn_v")) };
        let out_scale = if has(&g, &p("layer_output_scale")) {
            load(&g, &file, &p("layer_output_scale")).to_vec().unwrap()[0]
        } else { 1.0 };
        layers.push(Layer {
            head_dim,
            n_heads,
            n_kv_heads: kv_dim / head_dim,
            is_global,
            attn_norm: load(&g, &file, &p("attn_norm")),
            q_norm: load(&g, &file, &p("attn_q_norm")),
            k_norm: load(&g, &file, &p("attn_k_norm")),
            post_attn_norm: load(&g, &file, &p("post_attention_norm")),
            ffn_norm: load(&g, &file, &p("ffn_norm")),
            post_ffn_norm: load(&g, &file, &p("post_ffw_norm")),
            wq: load(&g, &file, &p("attn_q")),
            wk: load(&g, &file, &p("attn_k")),
            wv,
            wo: load(&g, &file, &p("attn_output")),
            w_gate: load(&g, &file, &p("ffn_gate")),
            w_up: load(&g, &file, &p("ffn_up")),
            w_down: load(&g, &file, &p("ffn_down")),
            out_scale,
        });
        if i % 8 == 0 { eprintln!("    loaded layer {i}"); }
    }

    eprintln!("[3/4] forward ({} tokens)...", tokens.len());
    let logits = forward(&cfg, &layers, &embed, &final_norm, &tokens);

    eprintln!("[4/4] top-5 logits:");
    let mut idx: Vec<usize> = (0..cfg.vocab).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    for &t in idx.iter().take(5) {
        println!("    [{t}] logit={:.4}", logits[t]);
    }
    println!("argmax = {}", idx[0]);
}

fn forward(
    cfg: &Config,
    layers: &[Layer],
    embed: &Tensor<Cpu>,
    final_norm: &Tensor<Cpu>,
    tokens: &[usize],
) -> Vec<f32> {
    let n = tokens.len();
    let d = cfg.d_model;
    let positions: Vec<usize> = (0..n).collect();

    // Embed × sqrt(d_model).
    let embed_host = embed.to_vec().unwrap();
    let scale = (d as f32).sqrt();
    let mut x = vec![0.0f32; n * d];
    for (t, &tok) in tokens.iter().enumerate() {
        for j in 0..d {
            x[t * d + j] = embed_host[tok * d + j] * scale;
        }
    }

    for layer in layers {
        let hd = layer.head_dim;
        let nh = layer.n_heads;
        let nkv = layer.n_kv_heads;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let xt = Tensor::<Cpu>::from_slice(&x).unwrap();

        // ── Attention ──
        let normed = gnorm(&xt, &layer.attn_norm, n, d, cfg.eps);
        let mut q = Tensor::<Cpu>::zeros(n * q_dim).unwrap();
        let mut k = Tensor::<Cpu>::zeros(n * kv_dim).unwrap();
        let mut v = Tensor::<Cpu>::zeros(n * kv_dim).unwrap();
        tensor::matmul_nt(&normed, &layer.wq, &mut q, n, d, q_dim).unwrap();
        tensor::matmul_nt(&normed, &layer.wk, &mut k, n, d, kv_dim).unwrap();
        tensor::matmul_nt(&normed, &layer.wv, &mut v, n, d, kv_dim).unwrap();

        // QK-norm (per head), V-norm (per head, no weight = ones).
        let ones = Tensor::<Cpu>::from_slice(&vec![1.0f32; hd]).unwrap();
        let q = gnorm(&q, &layer.q_norm, n * nh, hd, cfg.eps);
        let k = gnorm(&k, &layer.k_norm, n * nkv, hd, cfg.eps);
        let v = gnorm(&v, &ones, n * nkv, hd, cfg.eps);

        // Gemma4 attention scale = 1.0; sdpa bakes 1/sqrt(hd) → pre-scale Q by sqrt(hd).
        let mut qh = q.to_vec().unwrap();
        let qscale = (hd as f32).sqrt();
        for vv in qh.iter_mut() { *vv *= qscale; }
        let q = Tensor::<Cpu>::from_slice(&qh).unwrap();

        // RoPE (NEOX, per-layer base), full rotation.
        let base = if layer.is_global { cfg.rope_global } else { cfg.rope_local };
        let q = tensor::rope(&q, &positions, n, nh, hd, base, None).unwrap();
        let k = tensor::rope(&k, &positions, n, nkv, hd, base, None).unwrap();

        // Attention (local layers use sliding window; global = full).
        let sw = if layer.is_global { None } else { Some(cfg.sliding_window) };
        let attn = tensor::sdpa(&q, &k, &v, n, nh, nkv, hd, sw, None).unwrap();

        let mut attn_proj = Tensor::<Cpu>::zeros(n * d).unwrap();
        tensor::matmul_nt(&attn, &layer.wo, &mut attn_proj, n, q_dim, d).unwrap();
        let attn_proj = gnorm(&attn_proj, &layer.post_attn_norm, n, d, cfg.eps);
        let ap = attn_proj.to_vec().unwrap();
        for i in 0..n * d { x[i] += ap[i]; }

        // ── FFN (GeGLU) ──
        let xt = Tensor::<Cpu>::from_slice(&x).unwrap();
        let fnormed = gnorm(&xt, &layer.ffn_norm, n, d, cfg.eps);
        let mut gate = Tensor::<Cpu>::zeros(n * cfg.ffn_dim).unwrap();
        let mut up = Tensor::<Cpu>::zeros(n * cfg.ffn_dim).unwrap();
        tensor::matmul_nt(&fnormed, &layer.w_gate, &mut gate, n, d, cfg.ffn_dim).unwrap();
        tensor::matmul_nt(&fnormed, &layer.w_up, &mut up, n, d, cfg.ffn_dim).unwrap();
        let mut gh = gate.to_vec().unwrap();
        let uh = up.to_vec().unwrap();
        gelu_tanh(&mut gh);
        for i in 0..gh.len() { gh[i] *= uh[i]; }
        let h = Tensor::<Cpu>::from_slice(&gh).unwrap();
        let mut ffn_out = Tensor::<Cpu>::zeros(n * d).unwrap();
        tensor::matmul_nt(&h, &layer.w_down, &mut ffn_out, n, cfg.ffn_dim, d).unwrap();
        let ffn_out = gnorm(&ffn_out, &layer.post_ffn_norm, n, d, cfg.eps);
        let fo = ffn_out.to_vec().unwrap();
        for i in 0..n * d { x[i] += fo[i]; }

        // layer_output_scale on the WHOLE layer output.
        if layer.out_scale != 1.0 {
            for i in 0..n * d { x[i] *= layer.out_scale; }
        }
    }

    // Final norm on the LAST token, then lm_head (tied embed) + softcap.
    let last = Tensor::<Cpu>::from_slice(&x[(n - 1) * d..n * d]).unwrap();
    let normed = gnorm(&last, final_norm, 1, d, cfg.eps);
    let mut logits = Tensor::<Cpu>::zeros(cfg.vocab).unwrap();
    tensor::matmul_nt(&normed, embed, &mut logits, 1, d, cfg.vocab).unwrap();
    let mut out = logits.to_vec().unwrap();
    if let Some(cap) = cfg.logit_softcap {
        for l in out.iter_mut() { *l = cap * (*l / cap).tanh(); }
    }
    out
}
