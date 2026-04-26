//! Thin wrapper around the `tokenizers` crate for HuggingFace BPE tokenizers.
//!
//! Loads from a `tokenizer.json` file and exposes encode/decode for inference
//! demos that consume HF-format models (Qwen2, Llama, Mistral, ...). The
//! caller is responsible for adding chat-template wrappers / BOS / EOS
//! around encoded text — this layer is a pure tokenizer face.

use crate::backend::BoxErr;
use tokenizers::Tokenizer;

pub struct HfTokenizer {
    inner: Tokenizer,
}

impl HfTokenizer {
    pub fn from_file(path: &str) -> Result<Self, BoxErr> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| format!("tokenizer load failed: {e}"))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, BoxErr> {
        let enc = self.inner.encode(text, false)
            .map_err(|e| format!("tokenizer encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, BoxErr> {
        self.inner.decode(ids, false)
            .map_err(|e| format!("tokenizer decode failed: {e}").into())
    }

    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}
