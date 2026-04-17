//! ONNX-backed frozen cerebellum for isis runtime.
//!
//! Loads a Qwen/Llama-style ONNX model that outputs both logits and
//! hidden_states. Uses only the hidden_states (896-dim for Qwen2.5-0.5B)
//! as the cerebellum's world model output.
//!
//! The model receives token IDs (not float vectors). The cortex projection
//! maps cortex activations → pseudo-token IDs via quantization, then the
//! frozen LLM processes them and returns hidden states.
//!
//! Usage:
//!   let cereb = OnnxCerebellum::load("/steam/llm/qwen2.5-0.5b-onnx/backbone.onnx")?;
//!   let w = RegionalWeights::new(cfg)
//!       .with_frozen_cerebellum(cereb.input_dim(), cereb.output_dim());
//!   // In training loop:
//!   RegionalBrain::forward_cached_frozen(&w, state, &input, &mut cereb);

use modgrad_ctm::cerebellum::FrozenCerebellum;

/// Frozen ONNX model used as cerebellum.
///
/// Wraps an ONNX session that takes token IDs and returns hidden states.
/// The model must have outputs named "hidden_states" (or second output).
pub struct OnnxCerebellum {
    session: ort::session::Session,
    hidden_dim: usize,
    /// Index of hidden_states in outputs.
    hidden_output_idx: usize,
    /// Token position counter (for position_ids).
    position: i64,
}

impl OnnxCerebellum {
    /// Load an ONNX model with hidden_states output.
    ///
    /// The model should have inputs: input_ids, attention_mask, position_ids
    /// and outputs: logits, hidden_states (or just hidden_states).
    pub fn load(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = ort::session::Session::builder()?
            .commit_from_file(model_path)?;

        // Probe hidden_dim by running a dummy forward pass
        let outputs = session.outputs();
        let n_outputs = outputs.len();
        let n_inputs = session.inputs().len();

        // Probe hidden_dim from output shape if available, else default
        let hidden_dim = outputs.iter()
            .find(|o| o.name() == "hidden_states")
            .or_else(|| outputs.get(1))
            .and_then(|o| {
                o.dtype().tensor_shape()
                    .and_then(|shape| shape.last().copied())
                    .and_then(|d| if d > 0 { Some(d as usize) } else { None })
            })
            .unwrap_or(896);

        let hidden_output_idx = outputs.iter()
            .position(|o| o.name() == "hidden_states")
            .unwrap_or(1.min(n_outputs - 1));

        eprintln!("ONNX cerebellum loaded: {n_inputs} inputs, {n_outputs} outputs, hidden_dim={hidden_dim}, hidden_idx={hidden_output_idx}");

        Ok(Self { session, hidden_dim, hidden_output_idx, position: 0 })
    }

    /// Reset position counter (call between sequences).
    pub fn reset_position(&mut self) {
        self.position = 0;
    }
}

impl FrozenCerebellum for OnnxCerebellum {
    fn input_dim(&self) -> usize {
        self.hidden_dim
    }

    fn output_dim(&self) -> usize {
        self.hidden_dim
    }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        // Convert float input → pseudo-token IDs.
        // Each hidden_dim-sized chunk maps to one token via mean quantization.
        let d = self.hidden_dim;
        let n_tokens = ((input.len() + d - 1) / d).max(1);

        let mut token_ids = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let start = t * d;
            let end = (start + d).min(input.len());
            if start >= input.len() {
                token_ids.push(0i64);
                continue;
            }
            let chunk = &input[start..end];
            let mean: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
            // Map activation mean to token range [0, 255]
            let token = ((mean + 2.0) * 63.75).clamp(0.0, 255.0) as i64;
            token_ids.push(token);
        }

        let seq_len = token_ids.len();
        let attention_mask: Vec<i64> = vec![1; seq_len];
        let position_ids: Vec<i64> = (self.position..self.position + seq_len as i64).collect();
        self.position += seq_len as i64;

        // Build ONNX inputs
        let ids_shape = vec![1usize, seq_len];
        let result = (|| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let ids_tensor = ort::value::Tensor::<i64>::from_array(
                (ids_shape.as_slice(), token_ids))?;
            let mask_tensor = ort::value::Tensor::<i64>::from_array(
                (ids_shape.as_slice(), attention_mask))?;
            let pos_tensor = ort::value::Tensor::<i64>::from_array(
                (ids_shape.as_slice(), position_ids))?;

            let outputs = self.session.run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "position_ids" => pos_tensor,
            ])?;

            let hidden_idx = self.hidden_output_idx;
            let (_, data) = outputs[hidden_idx].try_extract_tensor::<f32>()?;

            // Return last token's hidden state
            let last_offset = (seq_len - 1) * self.hidden_dim;
            if last_offset + self.hidden_dim <= data.len() {
                Ok(data[last_offset..last_offset + self.hidden_dim].to_vec())
            } else {
                Ok(data[..self.hidden_dim.min(data.len())].to_vec())
            }
        })();

        match result {
            Ok(hidden) => hidden,
            Err(e) => {
                eprintln!("ONNX cerebellum forward failed: {e}");
                vec![0.0; self.hidden_dim]
            }
        }
    }
}
