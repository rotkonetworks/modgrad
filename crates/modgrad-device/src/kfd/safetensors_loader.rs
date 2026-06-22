//! Load HuggingFace `safetensors` weights directly (no GGUF / no quantization).
//!
//! Why: the FP16 reference model produces clean speech via PyTorch but our
//! Q4_K_M GGUF path accumulated enough numerical drift through 28 layers
//! that the LM couldn't keep frame-position phase. This loader lets us run
//! the same forward pass against the lossless FP16 weights.
//!
//! Storage rule:
//!   - 1-D tensors (norm scales) → `Weight::CpuF32(Vec<f32>)`
//!   - 2-D tensors → `Weight::CpuF32Mat` (f16 → f32 converted at load,
//!     ~13 GB host RAM for a 3 B model — fits comfortably on 16 GB+ host)
//!
//! Name mapping: HF Llama tensor names → GGUF tensor names (so the rest of
//! `llm.rs` works unchanged):
//!   model.embed_tokens.weight         → token_embd.weight
//!   model.norm.weight                 → output_norm.weight
//!   lm_head.weight                    → output.weight   (only if present)
//!   model.layers.N.input_layernorm.weight       → blk.N.attn_norm.weight
//!   model.layers.N.post_attention_layernorm.weight → blk.N.ffn_norm.weight
//!   model.layers.N.self_attn.q_proj.weight      → blk.N.attn_q.weight
//!   model.layers.N.self_attn.k_proj.weight      → blk.N.attn_k.weight
//!   model.layers.N.self_attn.v_proj.weight      → blk.N.attn_v.weight
//!   model.layers.N.self_attn.o_proj.weight      → blk.N.attn_output.weight
//!   model.layers.N.mlp.gate_proj.weight         → blk.N.ffn_gate.weight
//!   model.layers.N.mlp.up_proj.weight           → blk.N.ffn_up.weight
//!   model.layers.N.mlp.down_proj.weight         → blk.N.ffn_down.weight

use std::collections::HashMap;
use std::path::Path;
use memmap2::Mmap;
use safetensors::{SafeTensors, Dtype};

use super::llm::{LlmWeights, Weight};

/// Map HF parameter name to the GGUF-equivalent used by `llm.rs`.
fn map_name(hf: &str) -> Option<String> {
    Some(match hf {
        "model.embed_tokens.weight" => "token_embd.weight".to_string(),
        "model.norm.weight"         => "output_norm.weight".to_string(),
        "lm_head.weight"            => "output.weight".to_string(),
        other => {
            // model.layers.<N>.<sub> → blk.<N>.<sub>
            if let Some(rest) = other.strip_prefix("model.layers.") {
                let dot = rest.find('.')?;
                let layer: usize = rest[..dot].parse().ok()?;
                let sub = &rest[dot+1..];
                let mapped_sub = match sub {
                    "input_layernorm.weight"         => "attn_norm.weight",
                    "post_attention_layernorm.weight"=> "ffn_norm.weight",
                    "self_attn.q_proj.weight"        => "attn_q.weight",
                    "self_attn.k_proj.weight"        => "attn_k.weight",
                    "self_attn.v_proj.weight"        => "attn_v.weight",
                    "self_attn.o_proj.weight"        => "attn_output.weight",
                    "mlp.gate_proj.weight"           => "ffn_gate.weight",
                    "mlp.up_proj.weight"             => "ffn_up.weight",
                    "mlp.down_proj.weight"           => "ffn_down.weight",
                    _ => return None,
                };
                format!("blk.{layer}.{mapped_sub}")
            } else {
                return None;
            }
        }
    })
}

/// Convert raw FP16 bytes (little-endian) to a Vec<f32>.
fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 2, 0);
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let raw = u16::from_le_bytes([bytes[2*i], bytes[2*i+1]]);
        out.push(super::gguf::f16_to_f32(raw));
    }
    out
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 2, 0);
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let raw = u16::from_le_bytes([bytes[2*i], bytes[2*i+1]]);
        // bf16 is the upper 16 bits of an f32.
        let bits = (raw as u32) << 16;
        out.push(f32::from_bits(bits));
    }
    out
}

fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let raw = [bytes[4*i], bytes[4*i+1], bytes[4*i+2], bytes[4*i+3]];
        out.push(f32::from_le_bytes(raw));
    }
    out
}

/// Load one or more sharded safetensors files into an `LlmWeights` ready for
/// the existing `forward_token` path. Embeddings stored as `CpuF32Mat` rather
/// than mmap'd because we need to convert from F16 to F32 anyway.
///
/// Each path is one shard. For models like `model-00001-of-00002.safetensors`
/// you pass both shards.
pub fn load_from_safetensors(paths: &[impl AsRef<Path>]) -> Result<LlmWeights, String> {
    let mut by_name: HashMap<String, Weight> = HashMap::new();
    let mut total_mb: usize = 0;
    let mut n_loaded = 0usize;
    let mut skipped: Vec<String> = Vec::new();

    // Keep mmaps alive for the duration of the load.
    let mut mmaps: Vec<Mmap> = Vec::with_capacity(paths.len());
    for p in paths {
        let f = std::fs::File::open(p.as_ref())
            .map_err(|e| format!("open {}: {e}", p.as_ref().display()))?;
        let mmap = unsafe { Mmap::map(&f).map_err(|e| format!("mmap: {e}"))? };
        mmaps.push(mmap);
    }

    for (shard_i, mmap) in mmaps.iter().enumerate() {
        let st = SafeTensors::deserialize(mmap)
            .map_err(|e| format!("safetensors parse shard {shard_i}: {e}"))?;

        for (hf_name, view) in st.tensors() {
            let Some(gguf_name) = map_name(&hf_name) else {
                skipped.push(hf_name.to_string());
                continue;
            };

            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            // Convert to f32. We unify on f32 so the matvec path is a plain
            // dot product — no quant dispatch needed.
            let f32_data = match view.dtype() {
                Dtype::F16  => f16_bytes_to_f32(data),
                Dtype::BF16 => bf16_bytes_to_f32(data),
                Dtype::F32  => f32_bytes_to_f32(data),
                other => return Err(format!("unsupported dtype {other:?} for {hf_name}")),
            };

            total_mb += f32_data.len() * 4 / 1_000_000;

            // 1-D = norm scale, 2-D = matrix. forward_token expects matrices
            // as [in_dim, out_dim] but HF stores Linear weights as [out, in].
            // Transpose at load so `matvec()` reads them contiguous-row.
            if shape.len() == 1 {
                by_name.insert(gguf_name, Weight::CpuF32(f32_data));
            } else if shape.len() == 2 {
                let rows = shape[0];  // HF = out_dim
                let cols = shape[1];  // HF = in_dim
                // GGUF convention used elsewhere: dims=[in_dim, out_dim] with
                // storage [row][col] where each row holds in_dim elements.
                // For LM head + embeddings we want row-r = vocab token r's
                // embedding (d_model elems). HF already stores embed_tokens
                // as [vocab, d_model] which IS row-r = embedding. For Linear
                // weights HF stores [out, in] which is also row-r = "r-th
                // output's weights across all inputs" — so y[r] = row[r]·x.
                //
                // Either way the storage is [r][k] = element used in row r at
                // input k. Our matvec interpretation is dims=[in,out] but it
                // also iterates `for r in 0..out_dim` reading `row of in_dim`,
                // so semantics match — just record dims in the order matvec
                // expects: dims[0] = in_dim, dims[1] = out_dim.
                let dims = vec![cols, rows];  // in_dim, out_dim
                by_name.insert(gguf_name, Weight::CpuF32Mat { data: f32_data, dims });
            } else {
                return Err(format!("unexpected rank {} for {hf_name}", shape.len()));
            }
            n_loaded += 1;
        }
    }

    eprintln!("LlmWeights (safetensors): loaded {n_loaded} tensors, {total_mb} MB host RAM");
    if !skipped.is_empty() {
        eprintln!("  skipped {} tensors (no GGUF name mapping): {:?}", skipped.len(),
                  &skipped[..skipped.len().min(3)]);
    }

    // file_data_ptr is unused on this path (no GGUF mmap). Set to null —
    // matvec() never calls qweight() when CpuF32Mat is present.
    Ok(LlmWeights {
        by_name,
        file_data_ptr: std::ptr::null(),
    })
}
