//! Previous-token smear mixing.
//!
//! During training: x[t] += lambda * sigmoid(W @ x[t, :channels]) * x[t-1]  (for t > 0)
//! During decode:   x    += lambda * sigmoid(W @ x[:channels])    * prev_embedding
//!
//! Two modes via typestate: Smear<Training> and Smear<Inference>.

use std::marker::PhantomData;
use super::config::SmearConfig;
use super::tensor::Tensor2;
use super::dims::*;

/// Training mode marker.
pub struct Training;
/// Inference mode marker.
pub struct Inference;

/// Smear weights (shared between modes).
pub struct SmearWeights {
    /// Gate projection: [model_dim, gate_channels].
    pub gate: Tensor2<ModelDim, ModelDim>,
    gate_channels: usize,
    lambda: f32,
    model_dim: usize,
}

impl SmearWeights {
    pub fn new(gate: Vec<f32>, model_dim: ModelDim, config: &SmearConfig) -> Self {
        let md = model_dim.get();
        Self {
            gate: Tensor2::new(gate, md, config.gate_channels)
                .expect("smear gate weight shape mismatch"),
            gate_channels: config.gate_channels,
            lambda: config.lambda,
            model_dim: md,
        }
    }
}

/// Smear module parameterized by mode.
pub struct Smear<Mode> {
    pub weights: SmearWeights,
    _mode: PhantomData<Mode>,
}

impl Smear<Training> {
    pub fn new(weights: SmearWeights) -> Self {
        Self { weights, _mode: PhantomData }
    }

    /// Apply smear to a sequence in-place.
    ///
    /// `xs`: `[seq_len, model_dim]` row-major — modified in-place.
    /// For t > 0: x[t] += lambda * sigmoid(W @ x[t, :ch]) * x[t-1].
    pub fn forward(&self, xs: &mut [f32], seq_len: usize) {
        let md = self.weights.model_dim;
        let ch = self.weights.gate_channels;
        let lambda = self.weights.lambda;
        let gate_w = self.weights.gate.as_slice();

        for t in 1..seq_len {
            let prev_offset = (t - 1) * md;
            let cur_offset = t * md;

            // Compute per-dim gate
            for i in 0..md {
                let w_row = &gate_w[i * ch..(i + 1) * ch];
                let mut dot = 0.0f32;
                for j in 0..ch {
                    dot += w_row[j] * xs[cur_offset + j];
                }
                let gate = sigmoid(dot);
                xs[cur_offset + i] += lambda * gate * xs[prev_offset + i];
            }
        }
    }
}

impl Smear<Inference> {
    pub fn new(weights: SmearWeights) -> Self {
        Self { weights, _mode: PhantomData }
    }

    /// Apply smear to a single token using cached previous embedding.
    ///
    /// `x`: current hidden state `[model_dim]` — modified in-place.
    /// `prev`: previous token's embedding from KV cache.
    pub fn forward_one(&self, x: &mut [f32], prev: &[f32]) {
        let md = self.weights.model_dim;
        let ch = self.weights.gate_channels;
        let lambda = self.weights.lambda;
        let gate_w = self.weights.gate.as_slice();

        for i in 0..md {
            let w_row = &gate_w[i * ch..(i + 1) * ch];
            let mut dot = 0.0f32;
            for j in 0..ch {
                dot += w_row[j] * x[j];
            }
            let gate = sigmoid(dot);
            x[i] += lambda * gate * prev[i];
        }
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
