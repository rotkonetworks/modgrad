//! ONNX-backed frozen cerebellum for isis runtime.
//!
//! Processes a full context window in ONE forward pass, returning
//! hidden states from all transformer layers. The cerebellum uses
//! a learned weighted combination across layers — the network decides
//! which depth of representation matters most.
//!
//! When the ONNX model only exposes a single hidden_states output,
//! falls back to single-layer mode (backward compatible).

use modgrad_ctm::cerebellum::{FrozenCerebellum, CerebellumCache};

/// Frozen ONNX model used as cerebellum.
pub struct OnnxCerebellum {
    session: ort::session::Session,
    hidden_dim: usize,
    /// Indices of hidden_states outputs (one per layer, or just one).
    hidden_output_indices: Vec<usize>,
    n_layers: usize,
}

impl OnnxCerebellum {
    /// Load an ONNX model.
    ///
    /// If the model has outputs named "hidden_states_0", "hidden_states_1", etc,
    /// all are captured as separate layers. Otherwise falls back to a single
    /// "hidden_states" output (or the second output).
    pub fn load(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = ort::session::Session::builder()?
            .commit_from_file(model_path)?;

        let outputs = session.outputs();
        let n_outputs = outputs.len();
        let n_inputs = session.inputs().len();

        // Try to find multiple hidden_states_N outputs
        let mut layer_indices: Vec<usize> = Vec::new();
        for (i, o) in outputs.iter().enumerate() {
            let name = o.name();
            if name.starts_with("hidden_states_") || name == "hidden_states" {
                layer_indices.push(i);
            }
        }

        // Fallback: second output (index 1) if no named hidden_states found
        if layer_indices.is_empty() && n_outputs > 1 {
            layer_indices.push(1);
        } else if layer_indices.is_empty() {
            layer_indices.push(0);
        }

        let n_layers = layer_indices.len();

        let hidden_dim = outputs.get(*layer_indices.first().unwrap_or(&0))
            .and_then(|o| {
                o.dtype().tensor_shape()
                    .and_then(|shape| shape.last().copied())
                    .and_then(|d| if d > 0 { Some(d as usize) } else { None })
            })
            .unwrap_or(896);

        eprintln!("ONNX cerebellum: {n_inputs} inputs, {n_outputs} outputs, hidden_dim={hidden_dim}, layers={n_layers}");

        Ok(Self { session, hidden_dim, hidden_output_indices: layer_indices, n_layers })
    }
}

impl FrozenCerebellum for OnnxCerebellum {
    fn hidden_dim(&self) -> usize { self.hidden_dim }
    fn n_layers(&self) -> usize { self.n_layers }

    fn encode_context_layers(&mut self, token_ids: &[i64]) -> CerebellumCache {
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

            // Collect hidden states from all layer outputs
            let d = self.hidden_dim;
            let expected_per_layer = seq_len * d;
            let mut all_hidden = Vec::with_capacity(self.n_layers * expected_per_layer);

            for &idx in &self.hidden_output_indices {
                let (_, data) = outputs[idx].try_extract_tensor::<f32>()?;
                if data.len() >= expected_per_layer {
                    all_hidden.extend_from_slice(&data[..expected_per_layer]);
                } else {
                    let mut layer_data = data.to_vec();
                    layer_data.resize(expected_per_layer, 0.0);
                    all_hidden.extend_from_slice(&layer_data);
                }
            }

            Ok(CerebellumCache {
                hidden_states: all_hidden,
                hidden_dim: d,
                n_positions: seq_len,
                n_layers: self.n_layers,
            })
        })();

        match result {
            Ok(cache) => cache,
            Err(e) => {
                eprintln!("ONNX cerebellum encode_context_layers failed: {e}");
                CerebellumCache::empty()
            }
        }
    }

    fn encode_context(&mut self, token_ids: &[i64]) -> CerebellumCache {
        self.encode_context_layers(token_ids)
    }

    fn forward(&mut self, _input: &[f32]) -> Vec<f32> {
        vec![0.0; self.hidden_dim]
    }
}
