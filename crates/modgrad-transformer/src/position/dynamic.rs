//! RePo: dynamic learned positioning.
//!
//! Two modes:
//!   Linear:  pos = W @ hidden
//!   SwiGLU:  pos = final(silu(gate(x)) * content(x))
//!
//! Curriculum annealing: pos = (1 - alpha) * learned + alpha * sequential
//! Alpha anneals from 1 → 0 over training.

use super::PositionEncoding;

/// Mode for RePo position computation.
pub enum RePoMode {
    /// Simple linear projection: W @ hidden → scalar position.
    Linear {
        /// Projection weights [model_dim].
        weight: Vec<f32>,
    },
    /// SwiGLU: final(silu(gate(x)) * content(x)).
    SwiGlu {
        /// Gate projection [hidden_dim, model_dim].
        gate: Vec<f32>,
        /// Content projection [hidden_dim, model_dim].
        content: Vec<f32>,
        /// Final projection [hidden_dim] → scalar.
        final_proj: Vec<f32>,
        hidden_dim: usize,
        model_dim: usize,
    },
}

/// RePo dynamic positioning with curriculum annealing.
pub struct DynamicPositioning {
    mode: RePoMode,
    /// Curriculum alpha: 1.0 = pure sequential, 0.0 = pure learned.
    /// Annealed from 1 → 0 during training.
    pub alpha: f32,
}

impl DynamicPositioning {
    pub fn linear(weight: Vec<f32>) -> Self {
        Self {
            mode: RePoMode::Linear { weight },
            alpha: 1.0,
        }
    }

    pub fn swiglu(
        gate: Vec<f32>,
        content: Vec<f32>,
        final_proj: Vec<f32>,
        hidden_dim: usize,
        model_dim: usize,
    ) -> Self {
        Self {
            mode: RePoMode::SwiGlu { gate, content, final_proj, hidden_dim, model_dim },
            alpha: 1.0,
        }
    }

    /// Set curriculum alpha (for training annealing).
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.0, 1.0);
    }

    /// Compute learned position from a single hidden state.
    fn learned_position(&self, hidden: &[f32]) -> f32 {
        match &self.mode {
            RePoMode::Linear { weight } => {
                let mut pos = 0.0f32;
                for (w, h) in weight.iter().zip(hidden.iter()) {
                    pos += w * h;
                }
                pos
            }
            RePoMode::SwiGlu { gate, content, final_proj, hidden_dim, model_dim } => {
                let hd = *hidden_dim;
                let md = *model_dim;

                // gate(x): [hidden_dim]
                let mut gate_out = vec![0.0f32; hd];
                for i in 0..hd {
                    let mut dot = 0.0f32;
                    for j in 0..md {
                        dot += gate[i * md + j] * hidden[j];
                    }
                    // SiLU
                    gate_out[i] = dot * sigmoid(dot);
                }

                // content(x): [hidden_dim]
                let mut content_out = vec![0.0f32; hd];
                for i in 0..hd {
                    let mut dot = 0.0f32;
                    for j in 0..md {
                        dot += content[i * md + j] * hidden[j];
                    }
                    content_out[i] = dot;
                }

                // Element-wise multiply + final projection
                let mut pos = 0.0f32;
                for i in 0..hd {
                    pos += final_proj[i] * gate_out[i] * content_out[i];
                }
                pos
            }
        }
    }
}

impl PositionEncoding for DynamicPositioning {
    fn positions(&self, hidden: &[f32], start_pos: usize, seq_len: usize) -> Vec<f32> {
        let md = match &self.mode {
            RePoMode::Linear { weight } => weight.len(),
            RePoMode::SwiGlu { model_dim, .. } => *model_dim,
        };

        (0..seq_len).map(|t| {
            let h = &hidden[t * md..(t + 1) * md];
            let learned = self.learned_position(h);
            let sequential = (start_pos + t) as f32;
            // Curriculum: blend learned ↔ sequential
            (1.0 - self.alpha) * learned + self.alpha * sequential
        }).collect()
    }

    fn position_one(&self, hidden: &[f32], seq_pos: usize) -> f32 {
        let learned = self.learned_position(hidden);
        let sequential = seq_pos as f32;
        (1.0 - self.alpha) * learned + self.alpha * sequential
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
