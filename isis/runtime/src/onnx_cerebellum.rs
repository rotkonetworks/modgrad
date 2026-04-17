//! ONNX-backed frozen cerebellum for isis runtime.
//!
//! Processes a full context window in ONE forward pass, returning
//! position-aligned hidden states. The cortex reads h[i] when
//! processing token i — 32× faster than per-token inference.
//!
//! Usage:
//!   let mut cereb = OnnxCerebellum::load("backbone.onnx")?;
//!   let cache = cereb.encode_context(&token_ids);  // one ONNX call
//!   // Per-token in training loop:
//!   cerebellum_at_position(&cache, &proj, pos, &mut out);  // zero-alloc read

use modgrad_ctm::cerebellum::{FrozenCerebellum, CerebellumCache};

/// Frozen ONNX model used as cerebellum.
///
/// Wraps an ONNX session that takes a context of token IDs and returns
/// hidden states for each position. One forward pass per context window.
pub struct OnnxCerebellum {
    session: ort::session::Session,
    hidden_dim: usize,
    /// Index of hidden_states in outputs.
    hidden_output_idx: usize,
}

impl OnnxCerebellum {
    /// Load an ONNX model with hidden_states output.
    pub fn load(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = ort::session::Session::builder()?
            .commit_from_file(model_path)?;

        let outputs = session.outputs();
        let n_outputs = outputs.len();
        let n_inputs = session.inputs().len();

        let hidden_output_idx = outputs.iter()
            .position(|o| o.name() == "hidden_states")
            .unwrap_or(1.min(n_outputs - 1));

        let hidden_dim = outputs.get(hidden_output_idx)
            .and_then(|o| {
                o.dtype().tensor_shape()
                    .and_then(|shape| shape.last().copied())
                    .and_then(|d| if d > 0 { Some(d as usize) } else { None })
            })
            .unwrap_or(896);

        eprintln!("ONNX cerebellum: {n_inputs} inputs, {n_outputs} outputs, hidden_dim={hidden_dim}, hidden_idx={hidden_output_idx}");

        Ok(Self { session, hidden_dim, hidden_output_idx })
    }
}

impl FrozenCerebellum for OnnxCerebellum {
    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Encode a full context window in one ONNX call.
    /// Returns hidden states for each position.
    fn encode_context(&mut self, token_ids: &[i64]) -> CerebellumCache {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return CerebellumCache::empty();
        }

        let attention_mask: Vec<i64> = vec![1; seq_len];
        let position_ids: Vec<i64> = (0..seq_len as i64).collect();
        let shape = vec![1usize, seq_len];

        let result = (|| -> Result<CerebellumCache, Box<dyn std::error::Error>> {
            let ids_tensor = ort::value::Tensor::<i64>::from_array(
                (shape.as_slice(), token_ids.to_vec()))?;
            let mask_tensor = ort::value::Tensor::<i64>::from_array(
                (shape.as_slice(), attention_mask))?;
            let pos_tensor = ort::value::Tensor::<i64>::from_array(
                (shape.as_slice(), position_ids))?;

            let outputs = self.session.run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "position_ids" => pos_tensor,
            ])?;

            let (_, data) = outputs[self.hidden_output_idx]
                .try_extract_tensor::<f32>()?;

            // data is [1, seq_len, hidden_dim] flattened
            let expected = seq_len * self.hidden_dim;
            let hidden_states = if data.len() >= expected {
                data[..expected].to_vec()
            } else {
                let mut h = data.to_vec();
                h.resize(expected, 0.0);
                h
            };

            Ok(CerebellumCache {
                hidden_states,
                hidden_dim: self.hidden_dim,
                len: seq_len,
            })
        })();

        match result {
            Ok(cache) => cache,
            Err(e) => {
                eprintln!("ONNX cerebellum encode_context failed: {e}");
                CerebellumCache::empty()
            }
        }
    }

    /// Per-token forward (fallback, not used in normal operation).
    fn forward(&mut self, _input: &[f32]) -> Vec<f32> {
        // In the new architecture, encode_context() is the primary method.
        // This exists for trait completeness.
        vec![0.0; self.hidden_dim]
    }
}
