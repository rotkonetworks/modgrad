//! Rotary Position Embeddings (RoPE).
//!
//! Precomputes cos/sin tables at construction time.
//! Application: split each head into pairs, rotate by position-dependent angle.
//!
//! inv_freq[i] = 1 / (base ^ (2i / d))
//! For position t: angle = t * inv_freq[i]
//! [x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]

use super::dims::{HeadDim, SeqLen};

/// Precomputed RoPE cos/sin tables.
pub struct RotaryEmbedding {
    /// cos[pos * half_dim + i] for pos in 0..max_seq_len, i in 0..half_dim
    cos: Vec<f32>,
    /// sin[pos * half_dim + i]
    sin: Vec<f32>,
    half_dim: usize,
    max_seq_len: usize,
}

impl RotaryEmbedding {
    /// Precompute tables for the given head dimension and max sequence length.
    pub fn new(head_dim: HeadDim, max_seq_len: SeqLen, base: f32) -> Self {
        let d = head_dim.get();
        let half = d / 2;
        let max_len = max_seq_len.get();

        // inv_freq[i] = 1 / (base ^ (2i / d))
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / d as f32))
            .collect();

        let mut cos = vec![0.0f32; max_len * half];
        let mut sin = vec![0.0f32; max_len * half];

        for t in 0..max_len {
            for i in 0..half {
                let angle = t as f32 * inv_freq[i];
                cos[t * half + i] = angle.cos();
                sin[t * half + i] = angle.sin();
            }
        }

        Self { cos, sin, half_dim: half, max_seq_len: max_len }
    }

    /// Access cos table for backward RoPE.
    pub fn cos_at(&self, position: usize) -> &[f32] {
        let base = position * self.half_dim;
        &self.cos[base..base + self.half_dim]
    }

    /// Access sin table for backward RoPE.
    pub fn sin_at(&self, position: usize) -> &[f32] {
        let base = position * self.half_dim;
        &self.sin[base..base + self.half_dim]
    }

    /// Half dimension (head_dim / 2).
    pub fn half_dim(&self) -> usize {
        self.half_dim
    }

    /// Apply RoPE to a single head vector at the given position.
    ///
    /// `head` must have length `2 * half_dim`. Modified in-place.
    #[inline]
    pub fn apply(&self, head: &mut [f32], position: usize) {
        debug_assert_eq!(head.len(), self.half_dim * 2);
        debug_assert!(position < self.max_seq_len, "position {position} >= max {}", self.max_seq_len);

        let base = position * self.half_dim;
        let (left, right) = head.split_at_mut(self.half_dim);
        for i in 0..self.half_dim {
            let cos_i = self.cos[base + i];
            let sin_i = self.sin[base + i];
            let x1 = left[i];
            let x2 = right[i];
            left[i]  = x1 * cos_i - x2 * sin_i;
            right[i] = x1 * sin_i + x2 * cos_i;
        }
    }

    /// Apply RoPE to a batch of head vectors for consecutive positions.
    ///
    /// `heads` is `[seq_len, head_dim]` row-major.
    /// Positions are `start_pos..start_pos+seq_len`.
    pub fn apply_batch(&self, heads: &mut [f32], head_dim: usize, start_pos: usize, seq_len: usize) {
        debug_assert_eq!(head_dim, self.half_dim * 2);
        debug_assert_eq!(heads.len(), seq_len * head_dim);

        for t in 0..seq_len {
            let pos = start_pos + t;
            let offset = t * head_dim;
            self.apply(&mut heads[offset..offset + head_dim], pos);
        }
    }

    /// Apply to positions given by an arbitrary position array (for RePo / dynamic positioning).
    pub fn apply_positions(&self, heads: &mut [f32], head_dim: usize, positions: &[f32]) {
        debug_assert_eq!(head_dim, self.half_dim * 2);
        let seq_len = positions.len();
        debug_assert_eq!(heads.len(), seq_len * head_dim);

        for t in 0..seq_len {
            // Fractional positions: interpolate cos/sin
            let pos_f = positions[t];
            let pos_lo = pos_f.floor() as usize;
            let pos_hi = (pos_lo + 1).min(self.max_seq_len - 1);
            let frac = pos_f - pos_f.floor();

            let offset = t * head_dim;
            let (left, right) = heads[offset..offset + head_dim].split_at_mut(self.half_dim);

            let base_lo = pos_lo * self.half_dim;
            let base_hi = pos_hi * self.half_dim;

            for i in 0..self.half_dim {
                let cos_i = self.cos[base_lo + i] * (1.0 - frac) + self.cos[base_hi + i] * frac;
                let sin_i = self.sin[base_lo + i] * (1.0 - frac) + self.sin[base_hi + i] * frac;
                let x1 = left[i];
                let x2 = right[i];
                left[i]  = x1 * cos_i - x2 * sin_i;
                right[i] = x1 * sin_i + x2 * cos_i;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_position_zero_is_identity() {
        let rope = RotaryEmbedding::new(HeadDim::new(4), SeqLen::new(16), 10000.0);
        // At position 0, angle = 0 for all dims, so cos=1, sin=0 → identity
        let mut head = vec![1.0, 2.0, 3.0, 4.0];
        let original = head.clone();
        rope.apply(&mut head, 0);
        for (a, b) in head.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "pos 0 should be identity");
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        let rope = RotaryEmbedding::new(HeadDim::new(8), SeqLen::new(128), 100000.0);
        let mut head = vec![1.0, 0.5, -0.3, 0.7, 0.2, -0.8, 0.4, 0.1];
        let norm_before: f32 = head.iter().map(|x| x * x).sum::<f32>().sqrt();
        rope.apply(&mut head, 42);
        let norm_after: f32 = head.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-5, "RoPE should preserve vector norm");
    }
}
