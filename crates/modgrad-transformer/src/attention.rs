//! Causal self-attention with QK norm, sliding window, and GQA.
//!
//! Single-head computation: Q/K/V projection → QK RmsNorm → RoPE → dot-product
//! attention → optional sliding window mask → softmax → V aggregation → output proj.

use super::config::GptConfig;
use super::dims::*;
use super::norm::RmsNorm;
use super::rope::RotaryEmbedding;
use super::tensor::WeightMatrix;
use super::ops::TransformerOps;

/// Per-layer attention weights (bias-free).
pub struct AttentionWeights {
    /// Q projection: [model_dim, model_dim] (all heads packed).
    pub wq: WeightMatrix<ModelDim, ModelDim>,
    /// K projection: [model_dim, kv_dim].
    pub wk: WeightMatrix<ModelDim, NumKvHeads>,
    /// V projection: [model_dim, kv_dim].
    pub wv: WeightMatrix<ModelDim, NumKvHeads>,
    /// Output projection: [model_dim, model_dim].
    pub wo: WeightMatrix<ModelDim, ModelDim>,
}

/// Causal self-attention module.
pub struct CausalSelfAttention {
    pub weights: AttentionWeights,
    qk_norm: RmsNorm,
    qk_scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_ratio: usize,
}

impl CausalSelfAttention {
    pub fn new(weights: AttentionWeights, config: &GptConfig) -> Self {
        Self {
            weights,
            qk_norm: RmsNorm::new(config.norm_eps),
            qk_scale: config.qk_norm_scale,
            num_heads: config.num_heads.get(),
            num_kv_heads: config.num_kv_heads.get(),
            head_dim: config.head_dim.get(),
            gqa_ratio: config.gqa_ratio(),
        }
    }

    /// Forward pass for a single token position (decode mode).
    ///
    /// `x`: input hidden state `[model_dim]`.
    /// `kv_k`, `kv_v`: existing K/V cache slices for this layer `[seq_len, kv_dim]`.
    /// `rope`: rotary embeddings.
    /// `position`: absolute position of this token.
    /// `window`: optional sliding window size.
    /// `backend`: compute backend.
    ///
    /// Returns: output `[model_dim]`.
    pub fn forward_one(
        &self,
        x: &[f32],
        kv_k: &[f32],
        kv_v: &[f32],
        cur_k: &mut [f32],
        cur_v: &mut [f32],
        rope: &RotaryEmbedding,
        position: usize,
        seq_len: usize,
        window: Option<usize>,
        backend: &dyn TransformerOps,
        output: &mut [f32],
    ) {
        let d = self.head_dim;
        let n_heads = self.num_heads;
        let n_kv = self.num_kv_heads;
        let model_dim = n_heads * d;
        let kv_dim = n_kv * d;

        // Allocate projections
        let mut q = vec![0.0f32; model_dim];
        let mut k_proj = vec![0.0f32; kv_dim];
        let mut v_proj = vec![0.0f32; kv_dim];

        // Project Q, K, V (bias-free)
        backend.matvec_nobias(self.weights.wq.as_slice(), x, &mut q, model_dim, model_dim);
        backend.matvec_nobias(self.weights.wk.as_slice(), x, &mut k_proj, kv_dim, model_dim);
        backend.matvec_nobias(self.weights.wv.as_slice(), x, &mut v_proj, kv_dim, model_dim);

        // QK RmsNorm per head, then scale
        let mut q_normed = vec![0.0f32; d];
        let mut k_normed = vec![0.0f32; d];

        // Copy current K/V to cache output
        cur_k.copy_from_slice(&k_proj);
        cur_v.copy_from_slice(&v_proj);

        // Apply RoPE to K
        for h in 0..n_kv {
            rope.apply(&mut cur_k[h * d..(h + 1) * d], position);
        }

        // Per-head attention
        output.fill(0.0);
        let mut head_out = vec![0.0f32; d];

        // Determine attention window
        let start = match window {
            Some(w) => position.saturating_sub(w) + 1,
            None => 0,
        };
        let attn_len = seq_len; // total cached length (including current)

        for h in 0..n_heads {
            let kv_h = h / self.gqa_ratio;

            // Extract and normalize Q head
            let q_head = &q[h * d..(h + 1) * d];
            self.qk_norm.forward(q_head, &mut q_normed);
            for v in &mut q_normed { *v *= self.qk_scale; }

            // Apply RoPE to Q
            rope.apply(&mut q_normed, position);

            // Compute attention scores
            let mut scores = vec![f32::NEG_INFINITY; attn_len];

            for t in start..attn_len {
                // Get cached K for this KV head at position t
                let k_offset = t * kv_dim + kv_h * d;
                let k_t = if t == position {
                    // Current token — use just-projected K
                    &cur_k[kv_h * d..(kv_h + 1) * d]
                } else {
                    &kv_k[k_offset..k_offset + d]
                };

                // Normalize K
                self.qk_norm.forward(k_t, &mut k_normed);
                for v in &mut k_normed { *v *= self.qk_scale; }

                // Dot product
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot += q_normed[i] * k_normed[i];
                }
                scores[t] = dot;
            }

            // Causal mask: positions > current are already NEG_INFINITY
            // Softmax
            backend.softmax_inplace(&mut scores[start..attn_len]);

            // Weighted sum of V
            head_out.fill(0.0);
            for t in start..attn_len {
                let v_offset = t * kv_dim + kv_h * d;
                let v_t = if t == position {
                    &cur_v[kv_h * d..(kv_h + 1) * d]
                } else {
                    &kv_v[v_offset..v_offset + d]
                };

                let w = scores[t];
                for i in 0..d {
                    head_out[i] += w * v_t[i];
                }
            }

            // Write to concatenated output position
            output[h * d..(h + 1) * d].copy_from_slice(&head_out);
        }

        // Output projection: [model_dim] → [model_dim]
        let concat = output.to_vec();
        backend.matvec_nobias(self.weights.wo.as_slice(), &concat, output, model_dim, model_dim);
    }

    /// Prefill forward pass for a full sequence.
    ///
    /// `xs`: `[seq_len, model_dim]` row-major.
    /// `rope`: rotary embeddings.
    /// `start_pos`: starting position (usually 0 for fresh prefill).
    /// `window`: optional sliding window size.
    /// `backend`: compute backend.
    /// `k_out`, `v_out`: KV cache to populate `[seq_len, kv_dim]`.
    /// `output`: `[seq_len, model_dim]` row-major.
    pub fn forward_prefill(
        &self,
        xs: &[f32],
        rope: &RotaryEmbedding,
        start_pos: usize,
        seq_len: usize,
        window: Option<usize>,
        backend: &dyn TransformerOps,
        k_out: &mut [f32],
        v_out: &mut [f32],
        output: &mut [f32],
    ) {
        let d = self.head_dim;
        let n_heads = self.num_heads;
        let n_kv = self.num_kv_heads;
        let model_dim = n_heads * d;
        let kv_dim = n_kv * d;

        // Project all Q, K, V at once
        let mut all_q = vec![0.0f32; seq_len * model_dim];
        let mut all_k = vec![0.0f32; seq_len * kv_dim];
        let mut all_v = vec![0.0f32; seq_len * kv_dim];

        for t in 0..seq_len {
            let x = &xs[t * model_dim..(t + 1) * model_dim];
            backend.matvec_nobias(self.weights.wq.as_slice(), x,
                &mut all_q[t * model_dim..(t + 1) * model_dim], model_dim, model_dim);
            backend.matvec_nobias(self.weights.wk.as_slice(), x,
                &mut all_k[t * kv_dim..(t + 1) * kv_dim], kv_dim, model_dim);
            backend.matvec_nobias(self.weights.wv.as_slice(), x,
                &mut all_v[t * kv_dim..(t + 1) * kv_dim], kv_dim, model_dim);
        }

        // Apply RoPE to K
        for t in 0..seq_len {
            for h in 0..n_kv {
                rope.apply(&mut all_k[t * kv_dim + h * d..t * kv_dim + (h + 1) * d], start_pos + t);
            }
        }

        // Write K/V to cache
        k_out[..seq_len * kv_dim].copy_from_slice(&all_k);
        v_out[..seq_len * kv_dim].copy_from_slice(&all_v);

        // Per-token, per-head attention
        let mut q_normed = vec![0.0f32; d];
        let mut k_normed = vec![0.0f32; d];
        let mut head_out = vec![0.0f32; d];
        let mut scores_buf = vec![0.0f32; seq_len];

        for t in 0..seq_len {
            let pos = start_pos + t;
            let mut concat = vec![0.0f32; model_dim];

            for h in 0..n_heads {
                let kv_h = h / self.gqa_ratio;

                // Normalize Q
                let q_head = &all_q[t * model_dim + h * d..t * model_dim + (h + 1) * d];
                self.qk_norm.forward(q_head, &mut q_normed);
                for v in &mut q_normed { *v *= self.qk_scale; }
                rope.apply(&mut q_normed, pos);

                // Attention scores
                let start = match window {
                    Some(w) => pos.saturating_sub(w) + 1,
                    None => 0,
                };
                // Only attend to positions 0..=t (causal)
                let end = t + 1;

                scores_buf[..end].fill(f32::NEG_INFINITY);
                for s in start.min(end)..end {
                    let k_s = &all_k[s * kv_dim + kv_h * d..s * kv_dim + (kv_h + 1) * d];
                    self.qk_norm.forward(k_s, &mut k_normed);
                    for v in &mut k_normed { *v *= self.qk_scale; }

                    let mut dot = 0.0f32;
                    for i in 0..d { dot += q_normed[i] * k_normed[i]; }
                    scores_buf[s] = dot;
                }

                let score_slice = &mut scores_buf[start.min(end)..end];
                if !score_slice.is_empty() {
                    backend.softmax_inplace(score_slice);
                }

                // Weighted V sum
                head_out.fill(0.0);
                for s in start.min(end)..end {
                    let v_s = &all_v[s * kv_dim + kv_h * d..s * kv_dim + (kv_h + 1) * d];
                    let w = scores_buf[s];
                    for i in 0..d { head_out[i] += w * v_s[i]; }
                }

                concat[h * d..(h + 1) * d].copy_from_slice(&head_out);
            }

            // Output projection
            backend.matvec_nobias(self.weights.wo.as_slice(), &concat,
                &mut output[t * model_dim..(t + 1) * model_dim], model_dim, model_dim);
        }
    }
}
