//! Brain → LLM logit modulation seam.
//!
//! The architectural flow this closes:
//!
//! 1. LLM (e.g. `QwenCerebellum`) produces `qwen_logits ∈ ℝ^vocab`.
//! 2. Brain (`RegionalBrain` via `CerebellumService::read_at`) consumes
//!    Qwen's hidden states and produces a brain-output vector
//!    `o ∈ ℝ^brain_dim`.
//! 3. **This module** projects `o` into vocab space and folds it into
//!    the LLM's logits before sampling:
//!
//!    ```text
//!    final_logits = qwen_logits + alpha · (W_proj · o + b_proj)
//!    ```
//!
//! `W_proj ∈ ℝ^{vocab × brain_dim}` is a learnable `Linear`. `alpha` is
//! a scalar gate so the modulator can be initialised neutral
//! (`alpha = 0`) and ramped up during training.
//!
//! The caller picks **which** brain quantity to use — the modulator
//! takes any `&[f32]` of length `brain_dim`. Common choices:
//!
//! - Brain's final-tick prediction (already vocab-sized, but typed as
//!   a generic vector here).
//! - Final hippocampus activated state (d_model-sized).
//! - Aggregated sync-out vector (n_synch_out-sized).
//!
//! This separation keeps the modulator architecture-agnostic.
//!
//! ## Why this exists
//!
//! Without this seam, brain reads from Qwen via `CerebellumService` but
//! never writes back. Qwen's `lm_head` produces logits unmodified by
//! anything the brain computed — so "brain improves Qwen LM
//! performance" is structurally not measurable. With this module wired,
//! Qwen's logits become a function of brain state, the test
//! `Qwen-alone NLL` vs. `Qwen+brain NLL` becomes well-defined, and a
//! trained brain can in principle reduce Qwen's perplexity on
//! held-out text.
//!
//! ## What this does NOT do
//!
//! - Does not train the projection. Training is a separate slice.
//! - Does not auto-pick the brain output vector. Caller chooses.
//! - Does not touch `GptModelResident::forward`. Modulation is host-
//!   side; logits are added on host after a D2H copy. A future
//!   resident-only path can fold this into `op_tensor_resident::Add`.

use modgrad_compute::neuron::Linear;
use serde::{Deserialize, Serialize};
use wincode_derive::{SchemaRead, SchemaWrite};

/// Adds a learnable brain-derived bias to LLM logits before sampling.
/// Field-by-field serialisable so trained projections roundtrip
/// through the standard `modgrad-persist` save/load.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct BrainLogitModulator {
    /// Project brain output → vocab. Weight shape `[vocab × brain_dim]`,
    /// row-major (matches the rest of the codebase's `Linear` layout).
    pub proj: Linear,
    /// Scalar gate. Set to 0.0 at init so the modulator is a no-op
    /// until training warms it up.
    pub alpha: f32,
    /// Brain-output vector dimension expected at `modulate` time.
    pub brain_dim: usize,
    /// Vocab size — must match the LLM's logit length.
    pub vocab: usize,
}

impl BrainLogitModulator {
    /// Random-init projection, `alpha = 0.0` (modulator is neutral).
    /// Identical to `zero` for all `modulate` calls until `alpha` or
    /// `proj.weight` is mutated, but the random projection means the
    /// optimiser can break symmetry the moment training begins.
    pub fn new(brain_dim: usize, vocab: usize) -> Self {
        Self {
            proj: Linear::new(brain_dim, vocab),
            alpha: 0.0,
            brain_dim,
            vocab,
        }
    }

    /// Identity-zero modulator: weights and bias all zero, alpha 0.
    /// Used to sanity-check that "wired but disabled" produces
    /// bit-identical output to "not wired."
    pub fn zero(brain_dim: usize, vocab: usize) -> Self {
        let mut s = Self::new(brain_dim, vocab);
        s.proj.weight.iter_mut().for_each(|w| *w = 0.0);
        s.proj.bias.iter_mut().for_each(|b| *b = 0.0);
        s
    }

    /// Compute modulated logits: `qwen_logits + alpha · (W_proj · o + b)`.
    ///
    /// Shapes (debug-asserted):
    ///   `qwen_logits.len() == vocab`
    ///   `brain_output.len() == brain_dim`
    ///
    /// Returns a fresh `Vec<f32>` of length `vocab`. Allocates — for
    /// hot training loops use `modulate_into` to write into a caller-
    /// provided buffer.
    pub fn modulate(&self, qwen_logits: &[f32], brain_output: &[f32]) -> Vec<f32> {
        let mut out = qwen_logits.to_vec();
        self.modulate_into(qwen_logits, brain_output, &mut out);
        out
    }

    /// Like [`modulate`] but writes into a pre-allocated buffer.
    /// `out` must be length `vocab`. The buffer is overwritten.
    pub fn modulate_into(
        &self,
        qwen_logits: &[f32],
        brain_output: &[f32],
        out: &mut [f32],
    ) {
        debug_assert_eq!(qwen_logits.len(), self.vocab);
        debug_assert_eq!(brain_output.len(), self.brain_dim);
        debug_assert_eq!(out.len(), self.vocab);

        // Fast path: alpha == 0 → just copy qwen_logits.
        if self.alpha == 0.0 {
            out.copy_from_slice(qwen_logits);
            return;
        }

        // proj_out[v] = sum_b W[v, b] · brain_output[b] + bias[v]
        // for v in 0..vocab. Linear is row-major [vocab × brain_dim].
        let w = &self.proj.weight;
        let b = &self.proj.bias;
        let bd = self.brain_dim;
        for v in 0..self.vocab {
            let mut acc = b[v];
            let row = &w[v * bd..(v + 1) * bd];
            for i in 0..bd {
                acc += row[i] * brain_output[i];
            }
            out[v] = qwen_logits[v] + self.alpha * acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `alpha = 0` is the no-op invariant: regardless of brain_output
    /// content, modulated logits must equal qwen_logits bit-exactly.
    /// Without this the modulator can't be safely wired into a hot
    /// path with neutral init.
    #[test]
    fn alpha_zero_is_no_op() {
        let m = BrainLogitModulator::new(8, 16);
        assert_eq!(m.alpha, 0.0);
        let qwen = vec![0.1f32, -0.5, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0,
                        2.1, -1.3, 0.7, 0.4, -0.2, 0.9, 0.0, 0.5];
        let brain_out = vec![0.3f32, -0.7, 1.1, 0.0, 0.5, -0.4, 0.2, 0.8];
        let modulated = m.modulate(&qwen, &brain_out);
        assert_eq!(modulated, qwen,
            "alpha=0 must be bit-identical to qwen_logits");
    }

    /// With non-zero alpha + non-zero projection, different brain
    /// outputs must yield different modulated logits. Asserts the
    /// modulator is content-causal in brain output.
    #[test]
    fn modulation_is_content_causal() {
        let mut m = BrainLogitModulator::new(8, 16);
        m.alpha = 1.0;
        // Random init already non-zero; just enable alpha.

        let qwen = vec![0.0f32; 16];
        let brain_a = vec![0.3, -0.7, 1.1, 0.0, 0.5, -0.4, 0.2, 0.8];
        let brain_b = vec![1.0, 0.5, -0.3, 0.2, -1.1, 0.4, 0.9, -0.6];

        let mod_a = m.modulate(&qwen, &brain_a);
        let mod_b = m.modulate(&qwen, &brain_b);

        let l2: f32 = mod_a.iter().zip(mod_b.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        assert!(l2 > 1e-3,
            "different brain_output must produce different modulated \
             logits (L2 = {l2:.6}); identical would mean projection \
             is degenerate (all-zero, or input-invariant)");
    }

    /// `zero` modulator with non-zero alpha is still a no-op because
    /// the projection itself outputs zero. Locks the contract that
    /// "weights all zero" disables modulation, independent of alpha.
    #[test]
    fn zero_projection_is_no_op_at_any_alpha() {
        let mut m = BrainLogitModulator::zero(8, 16);
        m.alpha = 17.5;
        let qwen = vec![0.1f32, -0.5, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0,
                        2.1, -1.3, 0.7, 0.4, -0.2, 0.9, 0.0, 0.5];
        let brain_out = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let modulated = m.modulate(&qwen, &brain_out);
        // Zero projection: alpha · 0 = 0, so modulated == qwen.
        for (a, b) in modulated.iter().zip(qwen.iter()) {
            assert!((a - b).abs() < 1e-7,
                "zero projection must be no-op regardless of alpha");
        }
    }

    /// Linearity check: the modulator is linear in brain_output, so
    /// modulate(q, a + b) - modulate(q, 0) ==
    ///   modulate(q, a) - modulate(q, 0) + modulate(q, b) - modulate(q, 0)
    /// (alpha factored out). This catches transcription errors in the
    /// matvec inner loop without needing an external reference.
    #[test]
    fn modulation_is_linear_in_brain_output() {
        let mut m = BrainLogitModulator::new(4, 8);
        m.alpha = 0.7;
        let qwen = vec![0.0f32; 8];
        let zero = vec![0.0f32; 4];
        let a = vec![0.5, -0.3, 0.7, 0.1];
        let b = vec![-0.2, 0.4, -0.6, 0.8];
        let ab: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let m_zero = m.modulate(&qwen, &zero);  // bias-only
        let m_a    = m.modulate(&qwen, &a);
        let m_b    = m.modulate(&qwen, &b);
        let m_ab   = m.modulate(&qwen, &ab);

        for i in 0..8 {
            let predicted = m_a[i] + m_b[i] - m_zero[i];
            assert!((m_ab[i] - predicted).abs() < 1e-5,
                "modulator must be linear in brain_output: \
                 m_ab[{i}]={} vs predicted={}", m_ab[i], predicted);
        }
    }
}
