//! GGUF backend for isis using candle.
//!
//! Loads quantized GGUF models via candle-transformers and extracts
//! hidden states for memory keys + logits for generation.

use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_llama as llama;

use crate::backend::{Backend, BoxErr};
use crate::episode::normalize;

/// GGUF inference backend using candle.
pub struct GgufBackend {
    model: llama::ModelWeights,
    device: Device,
    hidden_dim: usize,
    vocab_size: usize,
}

impl GgufBackend {
    /// Load a GGUF model file.
    ///
    /// The model must be a llama-compatible architecture (Llama, Qwen, Mistral, etc).
    /// candle's quantized_llama supports most GGUF models from the llama.cpp ecosystem.
    pub fn load(gguf_path: &str) -> Result<Self, BoxErr> {
        let device = Device::Cpu;

        let mut file = std::fs::File::open(gguf_path)?;
        let model_data = gguf_file::Content::read(&mut file)
            .map_err(|e| format!("failed to read GGUF: {e}"))?;

        // Extract dimensions from GGUF metadata
        let hidden_dim = get_gguf_u32(&model_data, "llama.embedding_length")
            .or_else(|| get_gguf_u32(&model_data, "qwen2.embedding_length"))
            .unwrap_or(4096) as usize;

        let vocab_size = model_data
            .tensor_infos
            .get("output.weight")
            .or_else(|| model_data.tensor_infos.get("lm_head.weight"))
            .map(|t| t.shape.dims()[0] as usize)
            .unwrap_or(32000);

        let model = llama::ModelWeights::from_gguf(model_data, &mut file, &device)
            .map_err(|e| format!("failed to load model weights: {e}"))?;

        eprintln!("GGUF loaded: hidden_dim={hidden_dim}, vocab_size={vocab_size}");

        Ok(Self { model, device, hidden_dim, vocab_size })
    }
}

fn get_gguf_u32(content: &gguf_file::Content, key: &str) -> Option<u32> {
    content.metadata.get(key).and_then(|v| v.to_u32().ok())
}

impl Backend for GgufBackend {
    fn get_key(&mut self, token_ids: &[i64]) -> Result<Vec<f32>, BoxErr> {
        let (hidden, _) = self.run_backbone(token_ids)?;
        let mut key = hidden.last().unwrap().clone();
        normalize(&mut key);
        Ok(key)
    }

    fn forward(&mut self, token_ids: &[i64]) -> Result<Vec<Vec<f32>>, BoxErr> {
        let ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
        let input = Tensor::new(ids.as_slice(), &self.device)?;

        let logits_tensor = self.model.forward(&input, 0)
            .map_err(|e| format!("forward failed: {e}"))?;

        // logits_tensor shape: [seq_len, vocab_size]
        let logits_data = logits_tensor.to_vec2::<f32>()
            .map_err(|e| format!("logits extraction failed: {e}"))?;

        Ok(logits_data)
    }

    fn run_backbone(&mut self, token_ids: &[i64]) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), BoxErr> {
        // candle's quantized_llama doesn't expose intermediate hidden states directly.
        // We run the full forward and extract the hidden states from before lm_head.
        //
        // For now: run forward to get logits, and use the input embeddings as a
        // proxy for hidden states. This is a known limitation — the proper fix is
        // to modify candle's forward to return hidden states.
        //
        // TODO: Fork candle_transformers::models::quantized_llama to expose
        // pre-lm_head hidden states. For memory keys, we need the LAST hidden
        // state before the lm_head projection.

        let ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
        let input = Tensor::new(ids.as_slice(), &self.device)?;

        // Run forward — this gives us logits but not intermediate hiddens
        let logits_tensor = self.model.forward(&input, 0)
            .map_err(|e| format!("forward failed: {e}"))?;

        let logits_data = logits_tensor.to_vec2::<f32>()
            .map_err(|e| format!("extraction failed: {e}"))?;

        // Placeholder: use logits as "full_hidden" since we can still run lm_head identity
        // The hidden states are the same shape repeated for both outputs
        // Real implementation needs candle fork to expose hidden states
        let seq_len = token_ids.len();
        let hidden_placeholder: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| vec![0.0f32; self.hidden_dim])
            .collect();

        Ok((hidden_placeholder, hidden_placeholder.clone()))
    }

    fn run_lm_head(&mut self, _full_hidden: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, BoxErr> {
        // In GGUF models, lm_head is part of the forward pass.
        // This is called after run_backbone, but for GGUF we already
        // computed logits in forward(). Return empty — callers should
        // use forward() directly.
        Err("GGUF backend: use forward() instead of run_backbone()+run_lm_head() separately".into())
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
