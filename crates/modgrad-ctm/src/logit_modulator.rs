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

/// Mean per-token negative log-likelihood given a sequence of LLM
/// logits, optional per-step brain outputs to fold in, and the target
/// next-token IDs. Returns `f32::INFINITY` on empty input or if any
/// target id is out of range — fail-loud rather than silently
/// reporting a bogus average.
///
/// Shapes:
///   `qwen_logits_seq`:  `&[&[f32; vocab]]` of length `T`
///   `brain_outputs`:    `Some(&[&[f32; brain_dim]])` of length `T` or `None`
///   `targets`:          `&[usize]` of length `T`, each `< vocab`
///
/// When `brain_outputs` is `None`, computes the baseline LLM NLL — no
/// modulation, no projection cost. When `Some`, applies the modulator
/// per step before computing NLL. This is the primitive the Slice C2
/// training loop and the comparison report ("Qwen-alone NLL" vs
/// "Qwen+brain NLL") are both built on.
///
/// NLL definition: `-(1/T) · sum_t log_softmax(modulated_logits[t])[targets[t]]`.
/// Numerically stable: subtracts max-logit before exp.
pub fn nll_per_token(
    modulator: &BrainLogitModulator,
    qwen_logits_seq: &[&[f32]],
    brain_outputs: Option<&[&[f32]]>,
    targets: &[usize],
) -> f32 {
    if qwen_logits_seq.is_empty() || targets.len() != qwen_logits_seq.len() {
        return f32::INFINITY;
    }
    if let Some(bo) = brain_outputs {
        if bo.len() != qwen_logits_seq.len() { return f32::INFINITY; }
    }

    let vocab = modulator.vocab;
    let mut total_nll: f32 = 0.0;
    let mut scratch = vec![0.0f32; vocab];

    for t in 0..qwen_logits_seq.len() {
        let logits_t = qwen_logits_seq[t];
        if logits_t.len() != vocab { return f32::INFINITY; }
        let target_id = targets[t];
        if target_id >= vocab { return f32::INFINITY; }

        let modulated: &[f32] = match brain_outputs {
            Some(bo) => {
                modulator.modulate_into(logits_t, bo[t], &mut scratch);
                &scratch
            }
            None => logits_t,
        };

        // log-softmax(modulated)[target_id], numerically stable.
        let max_l = modulated.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let lse: f32 = modulated.iter()
            .map(|&l| (l - max_l).exp()).sum::<f32>().ln() + max_l;
        total_nll += lse - modulated[target_id];
    }
    total_nll / qwen_logits_seq.len() as f32
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

    /// Sanity for the NLL primitive: with `alpha=0` modulator, the
    /// "Qwen+brain" NLL must equal the "Qwen-alone" NLL bit-exactly.
    /// Without this any future training-loop measurement is
    /// confounded — we'd see a delta even when nothing should differ.
    #[test]
    fn nll_alpha_zero_matches_baseline() {
        let m = BrainLogitModulator::new(8, 16);
        assert_eq!(m.alpha, 0.0);

        // Synthetic logits: 5 timesteps, vocab 16.
        let logits_storage: Vec<Vec<f32>> = (0..5).map(|t| {
            (0..16).map(|v| ((t * 7 + v * 3) as f32 * 0.13).sin()).collect()
        }).collect();
        let logits_refs: Vec<&[f32]> = logits_storage.iter()
            .map(|v| v.as_slice()).collect();

        let brain_storage: Vec<Vec<f32>> = (0..5).map(|t| {
            (0..8).map(|i| ((t * 11 + i * 5) as f32 * 0.17).cos()).collect()
        }).collect();
        let brain_refs: Vec<&[f32]> = brain_storage.iter()
            .map(|v| v.as_slice()).collect();

        let targets: Vec<usize> = vec![3, 7, 1, 12, 5];

        let nll_baseline = nll_per_token(&m, &logits_refs, None, &targets);
        let nll_with_brain = nll_per_token(&m, &logits_refs, Some(&brain_refs), &targets);
        assert!(nll_baseline.is_finite());
        assert_eq!(nll_baseline, nll_with_brain,
            "alpha=0 modulator must produce baseline NLL bit-exactly \
             ({nll_baseline} vs {nll_with_brain})");
    }

    /// With non-zero alpha and random projection, NLL with brain
    /// must differ from baseline. The direction of difference depends
    /// on whether the random projection happens to favor the targets;
    /// here we just assert non-zero delta (content-causal NLL).
    #[test]
    fn nll_with_brain_modulation_differs_from_baseline() {
        let mut m = BrainLogitModulator::new(8, 16);
        m.alpha = 1.0;

        let logits_storage: Vec<Vec<f32>> = (0..5).map(|t| {
            (0..16).map(|v| ((t * 7 + v * 3) as f32 * 0.13).sin()).collect()
        }).collect();
        let logits_refs: Vec<&[f32]> = logits_storage.iter()
            .map(|v| v.as_slice()).collect();

        let brain_storage: Vec<Vec<f32>> = (0..5).map(|t| {
            (0..8).map(|i| ((t * 11 + i * 5) as f32 * 0.17).cos()).collect()
        }).collect();
        let brain_refs: Vec<&[f32]> = brain_storage.iter()
            .map(|v| v.as_slice()).collect();

        let targets: Vec<usize> = vec![3, 7, 1, 12, 5];

        let nll_baseline = nll_per_token(&m, &logits_refs, None, &targets);
        let nll_brain = nll_per_token(&m, &logits_refs, Some(&brain_refs), &targets);
        assert!(nll_baseline.is_finite() && nll_brain.is_finite());
        assert!((nll_brain - nll_baseline).abs() > 1e-4,
            "with alpha=1 and random proj, brain-modulated NLL must \
             differ from baseline (baseline={nll_baseline}, \
             brain={nll_brain})");
    }

    /// NLL primitive must fail-loud on shape mismatches rather than
    /// silently producing a bogus average — the "Qwen+brain NLL"
    /// claim only means something if the inputs are well-formed.
    #[test]
    fn nll_rejects_shape_mismatches() {
        let m = BrainLogitModulator::new(4, 8);

        // Empty.
        assert!(nll_per_token(&m, &[], None, &[]).is_infinite());

        // targets length != logits length.
        let l = vec![0.0f32; 8];
        let lr: Vec<&[f32]> = vec![l.as_slice()];
        assert!(nll_per_token(&m, &lr, None, &[]).is_infinite());

        // target id >= vocab.
        assert!(nll_per_token(&m, &lr, None, &[8]).is_infinite());

        // Wrong vocab.
        let bad = vec![0.0f32; 7];
        let badr: Vec<&[f32]> = vec![bad.as_slice()];
        assert!(nll_per_token(&m, &badr, None, &[0]).is_infinite());
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
