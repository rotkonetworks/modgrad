//! Generalizable transformer architecture spec for GGUF inference.
//!
//! One `TransformerArch` describes the variable parts of a decoder-only
//! transformer; Gemma-4, Llama, Qwen, Mistral are just different values of
//! it. Parsed generically from a GGUF file (our own parser — `kfd::gguf`),
//! deriving dims from metadata where present and from TENSOR SHAPES (ground
//! truth) where the metadata lies (e.g. Gemma's misleading `key_length`).
//!
//! This is the foundation for owning GGUF inference end-to-end on the typed
//! `Tensor<D>` stack — no candle, no architecture hardcoding.

use modgrad_device::kfd::gguf::GgufFile;

/// MLP activation / gating shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// SwiGLU: `silu(gate(x)) * up(x)` (Llama/Qwen/Mistral).
    SwiGlu,
    /// GeGLU: `gelu(gate(x)) * up(x)` (Gemma).
    GeGlu,
}

/// Config-driven decoder-only transformer architecture.
#[derive(Debug, Clone)]
pub struct TransformerArch {
    pub arch: String,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub rope_base: f32,
    /// Separate RoPE base for sliding-window (local) layers (Gemma).
    pub rope_base_swa: Option<f32>,
    /// Sliding-window size for local-attention layers (Gemma); None = global.
    pub sliding_window: Option<usize>,

    // ── architecture feature flags ──
    /// Pre- AND post-norm on each attention/FFN block (Gemma "sandwich").
    pub sandwich_norm: bool,
    /// Per-head RMSNorm on Q and K before RoPE (Gemma).
    pub qk_norm: bool,
    /// Scale token embeddings by `sqrt(d_model)` (Gemma).
    pub embed_scale: bool,
    /// `output.weight` is tied to `token_embd.weight` (no separate lm_head).
    pub tied_embeddings: bool,
    /// Final-logit soft-cap (Gemma): `cap * tanh(logits / cap)`.
    pub logit_softcap: Option<f32>,
    /// Attention-score soft-cap (Gemma2; often absent in 3/4).
    pub attn_logit_softcap: Option<f32>,
    pub activation: Activation,
}

impl TransformerArch {
    /// Parse the architecture generically from a GGUF file.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.architecture().ok_or("GGUF: no general.architecture")?.to_string();
        let mu = |k: &str| gguf.meta_u32(&format!("{arch}.{k}"));
        let mf = |k: &str| gguf.meta_f32(&format!("{arch}.{k}"));
        let has = |n: &str| gguf.tensors.contains_key(n);
        let dim1 = |n: &str| gguf.tensors.get(n).and_then(|t| t.dims.get(1).copied());
        let dim0 = |n: &str| gguf.tensors.get(n).and_then(|t| t.dims.first().copied());

        let d_model = mu("embedding_length").ok_or("GGUF: no embedding_length")? as usize;
        let n_layers = mu("block_count").ok_or("GGUF: no block_count")? as usize;
        let n_heads = mu("attention.head_count").unwrap_or(0) as usize;
        let ffn_dim = mu("feed_forward_length").unwrap_or(0) as usize;
        let norm_eps = mf("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let rope_base = mf("rope.freq_base").unwrap_or(10_000.0);
        let rope_base_swa = mf("rope.freq_base_swa");
        let sliding_window = mu("attention.sliding_window").map(|v| v as usize);

        // Dims from TENSOR SHAPES (ground truth; metadata key_length lies on Gemma).
        let q_dim = dim1("blk.0.attn_q.weight").unwrap_or(n_heads.max(1) * 256);
        let kv_dim = dim1("blk.0.attn_k.weight").unwrap_or(q_dim);
        let head_dim = dim0("blk.0.attn_q_norm.weight")
            .or_else(|| dim0("blk.0.attn_k_norm.weight"))
            .or_else(|| mu("attention.key_length").map(|k| k as usize))
            .unwrap_or(if n_heads > 0 { q_dim / n_heads } else { 256 });
        let n_kv_heads = if head_dim > 0 { (kv_dim / head_dim).max(1) } else { 1 };
        let vocab_size = dim1("token_embd.weight").or_else(|| dim1("output.weight")).unwrap_or(0);

        // Feature flags — detected from tensor presence + arch family.
        let is_gemma = arch.starts_with("gemma");
        let qk_norm = has("blk.0.attn_q_norm.weight") && has("blk.0.attn_k_norm.weight");
        let sandwich_norm = has("blk.0.post_attention_norm.weight")
            || has("blk.0.attn_post_norm.weight")
            || has("blk.0.post_ffw_norm.weight")
            || has("blk.0.post_ffn_norm.weight");
        let logit_softcap = mf("final_logit_softcapping")
            .or(if is_gemma { Some(30.0) } else { None });
        let attn_logit_softcap = mf("attn_logit_softcapping");
        let embed_scale = is_gemma;
        let tied_embeddings = !has("output.weight");
        let activation = if is_gemma { Activation::GeGlu } else { Activation::SwiGlu };

        Ok(Self {
            arch, n_layers, d_model, n_heads, n_kv_heads, head_dim, ffn_dim, vocab_size,
            norm_eps, rope_base, rope_base_swa, sliding_window,
            sandwich_norm, qk_norm, embed_scale, tied_embeddings,
            logit_softcap, attn_logit_softcap, activation,
        })
    }

    pub fn q_dim(&self) -> usize { self.n_heads * self.head_dim }
    pub fn kv_dim(&self) -> usize { self.n_kv_heads * self.head_dim }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;

    /// Parse the real Gemma-4-12B GGUF generically and assert the spec.
    /// (Header-only parse — fast even for a 6.6 GiB file. Skips if absent.)
    #[test]
    fn gemma4_12b_arch_parses_generically() {
        let path = "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf";
        let f = match File::open(path) {
            Ok(f) => f,
            Err(_) => { eprintln!("model absent, skipping"); return; }
        };
        let mut r = BufReader::new(f);
        let gguf = GgufFile::parse(&mut r).expect("parse gguf");
        let a = TransformerArch::from_gguf(&gguf).expect("parse arch");
        eprintln!("{a:#?}");

        assert_eq!(a.arch, "gemma4");
        assert_eq!(a.n_layers, 48);
        assert_eq!(a.d_model, 3840);
        assert_eq!(a.head_dim, 256);
        assert_eq!(a.n_heads, 16);
        assert_eq!(a.n_kv_heads, 8);
        assert_eq!(a.kv_dim(), 2048);
        assert_eq!(a.q_dim(), 4096);
        assert_eq!(a.vocab_size, 262144);
        // Gemma feature flags.
        assert!(a.qk_norm, "Gemma has QK-norm");
        assert!(a.sandwich_norm, "Gemma has pre+post sandwich norms");
        assert!(a.embed_scale, "Gemma scales embeddings by sqrt(d)");
        assert_eq!(a.activation, Activation::GeGlu);
        assert!(a.logit_softcap.is_some());
    }
}
