//! Value embeddings: alternating-layer learned value biases.
//!
//! On layers where has_ve(i) is true, the attention values get blended with
//! a learned embedding looked up from a table, gated by a small linear → sigmoid.
//!
//! gate = range * sigmoid(W @ x[..gate_channels])
//! v += gate * ve_table[token_id]

use super::config::ValueEmbedConfig;
use super::dims::*;
use super::tensor::Tensor2;

/// Value embedding table + gate weights for one layer.
pub struct ValueEmbedding {
    /// Embedding table: [vocab_size, kv_dim].
    pub table: Tensor2<VocabSize, NumKvHeads>,
    /// Gate projection: [kv_dim, gate_channels].
    pub gate_weight: Tensor2<NumKvHeads, ModelDim>,
    /// Config.
    gate_channels: usize,
    gate_range: f32,
    kv_dim: usize,
}

impl ValueEmbedding {
    pub fn new(
        table: Vec<f32>,
        gate_weight: Vec<f32>,
        vocab_size: VocabSize,
        kv_dim: usize,
        config: &ValueEmbedConfig,
    ) -> Self {
        Self {
            table: Tensor2::new(table, vocab_size.get(), kv_dim)
                .expect("VE table shape mismatch"),
            gate_weight: Tensor2::new(gate_weight, kv_dim, config.gate_channels)
                .expect("VE gate weight shape mismatch"),
            gate_channels: config.gate_channels,
            gate_range: config.gate_range,
            kv_dim,
        }
    }

    /// Apply value embedding to a V vector.
    ///
    /// `v`: mutable V projection `[kv_dim]` — modified in-place.
    /// `x`: input hidden state (for gating, only first `gate_channels` elements used).
    /// `token_id`: current token (for VE table lookup).
    pub fn apply(&self, v: &mut [f32], x: &[f32], token_id: usize) {
        debug_assert!(v.len() >= self.kv_dim);
        debug_assert!(x.len() >= self.gate_channels);
        debug_assert!(token_id < self.table.rows);

        // Compute gate: sigmoid(W @ x[..gate_channels]) * range
        let ve_row = self.table.row(token_id);
        let gate_w = self.gate_weight.as_slice();

        for i in 0..self.kv_dim {
            // Dot product for gate element i
            let mut dot = 0.0f32;
            let w_row = &gate_w[i * self.gate_channels..(i + 1) * self.gate_channels];
            for j in 0..self.gate_channels {
                dot += w_row[j] * x[j];
            }
            let gate = self.gate_range * sigmoid(dot);
            v[i] += gate * ve_row[i];
        }
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
