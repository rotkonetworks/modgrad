//! ONNX-backed frozen cerebellum for isis runtime.
//!
//! Wraps an ONNX backbone model as a FrozenCerebellum. The model
//! receives projected cortex activations and returns hidden states.
//!
//! Usage:
//!   let cereb = OnnxCerebellum::load("backbone.onnx")?;
//!   let weights = RegionalWeights::new(config).with_frozen_cerebellum(Box::new(cereb));

use modgrad_ctm::cerebellum::FrozenCerebellum;
use modgrad_io::backend::Backend;
use modgrad_io::inference::OnnxBackend;

/// Frozen ONNX backbone model used as cerebellum.
///
/// The ONNX model is split into backbone + lm_head. We only use
/// the backbone — hidden states are the cerebellum's output.
/// The lm_head is loaded but unused (needed for Backend trait).
pub struct OnnxCerebellum {
    backend: OnnxBackend,
    /// Cached input dimension (from backbone input shape).
    input_dim: usize,
}

impl OnnxCerebellum {
    /// Load an ONNX backbone + lm_head pair as a frozen cerebellum.
    pub fn load(backbone_path: &str, lm_head_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let backend = OnnxBackend::load(backbone_path, lm_head_path)?;
        let input_dim = backend.hidden_dim(); // input dim = embedding dim
        Ok(Self { backend, input_dim })
    }
}

impl FrozenCerebellum for OnnxCerebellum {
    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        self.backend.hidden_dim()
    }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        // The ONNX backbone expects token IDs, but we have float activations
        // from the cortex projection. Two modes:
        //
        // 1. Direct embedding injection: requires ONNX model exported with
        //    float input (post-embedding). This is the preferred mode.
        //
        // 2. Quantized proxy: convert float activations to pseudo-token IDs
        //    via nearest-neighbor lookup in the embedding table. Lossy but
        //    works with standard ONNX exports.
        //
        // For now, we implement mode 2 as a working fallback.
        // Mode 1 requires a custom ONNX export script (future work).

        // Convert float input to pseudo-token IDs by treating each
        // chunk of hidden_dim floats as one "token" and hashing to an ID.
        let d = self.input_dim;
        let n_tokens = (input.len() + d - 1) / d;
        let n_tokens = n_tokens.max(1);

        // Simple quantization: map mean activation to token range [0, 255]
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
            // Map [-2, 2] → [0, 255]
            let token = ((mean + 2.0) * 63.75).clamp(0.0, 255.0) as i64;
            token_ids.push(token);
        }

        // Run backbone to get hidden states
        match self.backend.run_backbone(&token_ids) {
            Ok((hidden, _full)) => {
                // Return last hidden state (most complete representation)
                hidden.into_iter().last().unwrap_or_else(|| vec![0.0; self.output_dim()])
            }
            Err(e) => {
                eprintln!("ONNX cerebellum forward failed: {e}");
                vec![0.0; self.output_dim()]
            }
        }
    }
}
