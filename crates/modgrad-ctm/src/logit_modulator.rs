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

/// Common surface for any modulator that maps brain output → vocab-
/// sized logit delta. Lets `nll_per_token` and downstream consumers
/// stay generic across the dense `BrainLogitModulator` and
/// `LowRankBrainLogitModulator` (LoRA-factored). Adding a third
/// modulator (sparse, hashed, hyper-network) only needs a new impl;
/// no per-modulator NLL helper.
pub trait LogitModulator {
    fn vocab(&self) -> usize;
    fn brain_dim(&self) -> usize;
    /// Write `qwen_logits + alpha · proj(brain_output)` into `out`.
    /// Implementations are free to allocate temporary scratch buffers.
    /// Hot training loops should call type-specific
    /// `modulate_into_with_scratch` (or analogous) instead.
    fn modulate_into(&self, qwen_logits: &[f32], brain_output: &[f32], out: &mut [f32]);
}

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

impl BrainLogitModulator {
    /// Backward pass through `modulate`: given the upstream gradient
    /// `d_modulated[v] = ∂L/∂out[v]`, accumulate into `d_proj_weight`
    /// and `d_proj_bias`. Does **not** backprop into `qwen_logits`
    /// (passes straight through — Qwen is frozen) or into
    /// `brain_output` (handled by a separate slice when brain weights
    /// also train).
    ///
    /// Math:
    ///   ∂L/∂W[v,b]  = d_modulated[v] · alpha · brain[b]
    ///   ∂L/∂bias[v] = d_modulated[v] · alpha
    ///
    /// Both gradients are **accumulated** (`+=`) so a caller can sum
    /// across a sequence of timesteps with one allocation.
    pub fn backward(
        &self,
        brain_output: &[f32],
        d_modulated: &[f32],
        d_proj_weight: &mut [f32],
        d_proj_bias: &mut [f32],
    ) {
        debug_assert_eq!(brain_output.len(), self.brain_dim);
        debug_assert_eq!(d_modulated.len(), self.vocab);
        debug_assert_eq!(d_proj_weight.len(), self.vocab * self.brain_dim);
        debug_assert_eq!(d_proj_bias.len(), self.vocab);

        if self.alpha == 0.0 { return; } // gradient is zero w.r.t. proj params

        let bd = self.brain_dim;
        for v in 0..self.vocab {
            let scale = d_modulated[v] * self.alpha;
            d_proj_bias[v] += scale;
            let row = &mut d_proj_weight[v * bd..(v + 1) * bd];
            for b in 0..bd {
                row[b] += scale * brain_output[b];
            }
        }
    }

    /// In-place SGD step. Subtracts `lr * grad` from weight + bias and
    /// zeros the grad buffers — caller can immediately accumulate into
    /// them again. Plain SGD (no momentum / no weight decay) keeps
    /// the slice minimal; swap to AdamW from `modgrad-training` later
    /// if convergence speed becomes the bottleneck.
    pub fn sgd_step(
        &mut self,
        d_proj_weight: &mut [f32],
        d_proj_bias: &mut [f32],
        lr: f32,
    ) {
        debug_assert_eq!(d_proj_weight.len(), self.proj.weight.len());
        debug_assert_eq!(d_proj_bias.len(), self.proj.bias.len());

        for (w, dw) in self.proj.weight.iter_mut().zip(d_proj_weight.iter_mut()) {
            *w -= lr * *dw;
            *dw = 0.0;
        }
        for (b, db) in self.proj.bias.iter_mut().zip(d_proj_bias.iter_mut()) {
            *b -= lr * *db;
            *db = 0.0;
        }
    }
}

/// Low-rank (LoRA-style) variant of `BrainLogitModulator`. Replaces
/// the dense `Linear(brain_dim, vocab)` projection with a factored
/// pair: `Linear(brain_dim, rank) · Linear(rank, vocab)`. Drops
/// parameter count by `(brain_dim + vocab) · rank / (brain_dim · vocab)`
/// — at brain_dim=512, vocab=152K, rank=8 that's a ~57× reduction
/// (78M → 1.37M params).
///
/// This is the structural fix for the brain_qwen_nll overfit observed
/// at commit `c41952c`: 78M dense params on 25 train positions
/// memorised the train split and underperformed Qwen-alone on held-out
/// by ~0.8 nats/token. Low-rank constrains the modulator to a
/// `rank`-dimensional subspace of corrections, which both reduces
/// param count and biases the optimiser toward generalisable
/// structure.
///
/// Forward: `out[v] = qwen[v] + alpha · (up · down · brain + b_up)`
///   `down`: `Linear(brain_dim, rank)` — `[rank × brain_dim]` weight
///   `up`:   `Linear(rank, vocab)`     — `[vocab × rank]` weight,
///                                       `[vocab]` bias
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct LowRankBrainLogitModulator {
    pub down: Linear,
    pub up: Linear,
    pub alpha: f32,
    pub brain_dim: usize,
    pub rank: usize,
    pub vocab: usize,
}

impl LowRankBrainLogitModulator {
    /// Random-init both layers, alpha=0 (neutral). Same convention as
    /// `BrainLogitModulator`: ramp alpha during training so the
    /// modulator can be wired as a no-op.
    pub fn new(brain_dim: usize, rank: usize, vocab: usize) -> Self {
        debug_assert!(rank >= 1, "rank must be ≥ 1");
        debug_assert!(rank < brain_dim.max(vocab),
            "rank ≥ min(brain_dim, vocab) defeats the factorisation");
        Self {
            down: Linear::new(brain_dim, rank),
            up:   Linear::new(rank, vocab),
            alpha: 0.0,
            brain_dim, rank, vocab,
        }
    }

    /// Identity-zero: both layers' weights and biases all zero.
    pub fn zero(brain_dim: usize, rank: usize, vocab: usize) -> Self {
        let mut s = Self::new(brain_dim, rank, vocab);
        s.down.weight.iter_mut().for_each(|w| *w = 0.0);
        s.down.bias.iter_mut().for_each(|b| *b = 0.0);
        s.up.weight.iter_mut().for_each(|w| *w = 0.0);
        s.up.bias.iter_mut().for_each(|b| *b = 0.0);
        s
    }

    /// Compute modulated logits. Allocates the rank-dim scratch.
    pub fn modulate(&self, qwen_logits: &[f32], brain_output: &[f32]) -> Vec<f32> {
        let mut out = qwen_logits.to_vec();
        let mut z = vec![0.0f32; self.rank];
        self.modulate_into_with_scratch(qwen_logits, brain_output, &mut out, &mut z);
        out
    }

    /// Like `modulate` but writes into pre-allocated buffers — the
    /// caller owns `out` (length vocab) and `z_scratch` (length rank).
    /// Hot training loops should call this and reuse buffers.
    pub fn modulate_into_with_scratch(
        &self,
        qwen_logits: &[f32],
        brain_output: &[f32],
        out: &mut [f32],
        z_scratch: &mut [f32],
    ) {
        debug_assert_eq!(qwen_logits.len(), self.vocab);
        debug_assert_eq!(brain_output.len(), self.brain_dim);
        debug_assert_eq!(out.len(), self.vocab);
        debug_assert_eq!(z_scratch.len(), self.rank);

        if self.alpha == 0.0 {
            out.copy_from_slice(qwen_logits);
            return;
        }

        // Step 1: z = down · brain + bias_down
        let dw = &self.down.weight; // [rank × brain_dim]
        let db = &self.down.bias;
        for r in 0..self.rank {
            let mut acc = db[r];
            let row = &dw[r * self.brain_dim..(r + 1) * self.brain_dim];
            for b in 0..self.brain_dim {
                acc += row[b] * brain_output[b];
            }
            z_scratch[r] = acc;
        }

        // Step 2: y = up · z + bias_up; out = qwen + alpha · y
        let uw = &self.up.weight; // [vocab × rank]
        let ub = &self.up.bias;
        for v in 0..self.vocab {
            let mut acc = ub[v];
            let row = &uw[v * self.rank..(v + 1) * self.rank];
            for r in 0..self.rank {
                acc += row[r] * z_scratch[r];
            }
            out[v] = qwen_logits[v] + self.alpha * acc;
        }
    }

    /// Backward pass. Accumulates into `d_down_w`, `d_down_b`,
    /// `d_up_w`, `d_up_b`. Math:
    ///   z = down · brain + b_down                 (forward step 1)
    ///   y = up · z + b_up                         (forward step 2)
    ///   out = qwen + alpha · y
    ///
    /// Given `d_out`:
    ///   d_y       = alpha · d_out
    ///   d_b_up    = d_y
    ///   d_up[v,r] = d_y[v] · z[r]
    ///   d_z[r]    = sum_v d_y[v] · up[v,r]
    ///   d_b_down  = d_z
    ///   d_down[r,b] = d_z[r] · brain[b]
    pub fn backward(
        &self,
        brain_output: &[f32],
        d_modulated: &[f32],
        d_down_w: &mut [f32],
        d_down_b: &mut [f32],
        d_up_w: &mut [f32],
        d_up_b: &mut [f32],
    ) {
        debug_assert_eq!(brain_output.len(), self.brain_dim);
        debug_assert_eq!(d_modulated.len(), self.vocab);
        debug_assert_eq!(d_down_w.len(), self.rank * self.brain_dim);
        debug_assert_eq!(d_down_b.len(), self.rank);
        debug_assert_eq!(d_up_w.len(), self.vocab * self.rank);
        debug_assert_eq!(d_up_b.len(), self.vocab);

        if self.alpha == 0.0 { return; }

        // Recompute z (forward stash would save it; for now, recompute).
        let mut z = vec![0.0f32; self.rank];
        for r in 0..self.rank {
            let mut acc = self.down.bias[r];
            let row = &self.down.weight[r * self.brain_dim..(r + 1) * self.brain_dim];
            for b in 0..self.brain_dim {
                acc += row[b] * brain_output[b];
            }
            z[r] = acc;
        }

        // d_b_up += alpha * d_modulated; d_up[v,r] += alpha * d_modulated[v] * z[r]
        let mut d_z = vec![0.0f32; self.rank];
        for v in 0..self.vocab {
            let dy = self.alpha * d_modulated[v];
            d_up_b[v] += dy;
            let row = &mut d_up_w[v * self.rank..(v + 1) * self.rank];
            let up_row = &self.up.weight[v * self.rank..(v + 1) * self.rank];
            for r in 0..self.rank {
                row[r] += dy * z[r];
                d_z[r] += dy * up_row[r];
            }
        }

        // d_b_down += d_z; d_down[r,b] += d_z[r] * brain[b]
        for r in 0..self.rank {
            d_down_b[r] += d_z[r];
            let row = &mut d_down_w[r * self.brain_dim..(r + 1) * self.brain_dim];
            for b in 0..self.brain_dim {
                row[b] += d_z[r] * brain_output[b];
            }
        }
    }

    /// In-place SGD step on all four parameter buffers; zeros grads.
    pub fn sgd_step(
        &mut self,
        d_down_w: &mut [f32], d_down_b: &mut [f32],
        d_up_w: &mut [f32],   d_up_b: &mut [f32],
        lr: f32,
    ) {
        for (w, dw) in self.down.weight.iter_mut().zip(d_down_w.iter_mut()) {
            *w -= lr * *dw; *dw = 0.0;
        }
        for (b, db) in self.down.bias.iter_mut().zip(d_down_b.iter_mut()) {
            *b -= lr * *db; *db = 0.0;
        }
        for (w, dw) in self.up.weight.iter_mut().zip(d_up_w.iter_mut()) {
            *w -= lr * *dw; *dw = 0.0;
        }
        for (b, db) in self.up.bias.iter_mut().zip(d_up_b.iter_mut()) {
            *b -= lr * *db; *db = 0.0;
        }
    }

    /// Total trainable parameters. Useful for the param/data ratio
    /// budget that drove the introduction of this struct.
    pub fn n_params(&self) -> usize {
        self.down.weight.len() + self.down.bias.len()
            + self.up.weight.len() + self.up.bias.len()
    }
}

/// Cross-entropy gradient w.r.t. logits, given the target token index.
///   ∂L/∂logits[v] = softmax(logits)[v] - I[v == target]
/// (no `1/T` averaging — caller does that at the loop level if
/// summing across timesteps.) Numerically stable softmax.
pub fn cross_entropy_grad(logits: &[f32], target: usize, out: &mut [f32]) {
    debug_assert_eq!(logits.len(), out.len());
    debug_assert!(target < logits.len(),
        "cross_entropy_grad: target {} out of range [0, {})", target, logits.len());

    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum_exp: f32 = 0.0;
    for i in 0..logits.len() {
        let e = (logits[i] - max_l).exp();
        out[i] = e;
        sum_exp += e;
    }
    let inv = 1.0 / sum_exp;
    for v in 0..logits.len() {
        out[v] *= inv;
    }
    out[target] -= 1.0;
}

impl LogitModulator for BrainLogitModulator {
    fn vocab(&self) -> usize { self.vocab }
    fn brain_dim(&self) -> usize { self.brain_dim }
    fn modulate_into(&self, qwen: &[f32], brain: &[f32], out: &mut [f32]) {
        BrainLogitModulator::modulate_into(self, qwen, brain, out);
    }
}

impl LogitModulator for LowRankBrainLogitModulator {
    fn vocab(&self) -> usize { self.vocab }
    fn brain_dim(&self) -> usize { self.brain_dim }
    fn modulate_into(&self, qwen: &[f32], brain: &[f32], out: &mut [f32]) {
        // Allocate the rank-dim scratch per call. Acceptable for
        // measurement helpers (called once per evaluation); training
        // loops should bypass the trait and call
        // `modulate_into_with_scratch` directly.
        let mut z = vec![0.0f32; self.rank];
        self.modulate_into_with_scratch(qwen, brain, out, &mut z);
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
/// Generic over the modulator type via the [`LogitModulator`] trait —
/// works with `BrainLogitModulator`, `LowRankBrainLogitModulator`, or
/// any future variant.
///
/// NLL definition: `-(1/T) · sum_t log_softmax(modulated_logits[t])[targets[t]]`.
/// Numerically stable: subtracts max-logit before exp.
pub fn nll_per_token<M: LogitModulator>(
    modulator: &M,
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

    let vocab = modulator.vocab();
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

    /// The redshiftzero load-bearing claim: the modulator can be
    /// **trained** to reduce NLL. Without this all the prior
    /// content-causality work is just expressivity — the seam can be
    /// non-zero, but does the optimizer find a non-zero gradient
    /// signal that actually moves loss down?
    ///
    /// Setup: deterministic synthetic Qwen logits, 10 steps × vocab 16,
    /// brain output sequence, fixed targets. Modulator alpha=1, random
    /// init. SGD on cross-entropy loss for 200 steps at lr=0.05.
    ///
    /// Assertion: NLL_final < NLL_initial · 0.5 — i.e. training cuts
    /// loss in half. Anything weaker would mean the optimizer is just
    /// drifting around. Anything stronger (e.g. < 0.01) might be
    /// overfitting noise — the bound is honest about what's
    /// expectable for ~3.2K parameters trained on 10 token positions.
    ///
    /// Synthetic Qwen logits are uniform zeros so the brain+modulator
    /// has to do all the lifting. Brain outputs are deterministic
    /// non-zero vectors so there's a well-defined target the
    /// projection can learn.
    #[test]
    fn training_reduces_nll() {
        // Synthetic data.
        let vocab = 16;
        let brain_dim = 8;
        let n_steps = 10;
        let qwen: Vec<Vec<f32>> = (0..n_steps).map(|_| vec![0.0f32; vocab]).collect();
        let qwen_refs: Vec<&[f32]> = qwen.iter().map(|v| v.as_slice()).collect();
        let brain: Vec<Vec<f32>> = (0..n_steps).map(|t| {
            (0..brain_dim).map(|i| ((t * 13 + i * 7) as f32 * 0.21).sin()).collect()
        }).collect();
        let brain_refs: Vec<&[f32]> = brain.iter().map(|v| v.as_slice()).collect();
        // Targets chosen pseudo-randomly but deterministically.
        let targets: Vec<usize> = (0..n_steps).map(|t| (t * 5 + 3) % vocab).collect();

        let mut m = BrainLogitModulator::new(brain_dim, vocab);
        m.alpha = 1.0;

        let nll_initial = nll_per_token(&m, &qwen_refs, Some(&brain_refs), &targets);
        assert!(nll_initial.is_finite() && nll_initial > 0.0);

        // Training loop: SGD on cross-entropy.
        let mut d_w = vec![0.0f32; vocab * brain_dim];
        let mut d_b = vec![0.0f32; vocab];
        let mut modulated = vec![0.0f32; vocab];
        let mut d_modulated = vec![0.0f32; vocab];
        let lr = 0.05;
        let n_train = 200;

        for _step in 0..n_train {
            // Accumulate gradients across the whole sequence.
            for t in 0..n_steps {
                m.modulate_into(qwen_refs[t], brain_refs[t], &mut modulated);
                cross_entropy_grad(&modulated, targets[t], &mut d_modulated);
                // Mean gradient: divide by n_steps so lr scaling is
                // independent of sequence length.
                let scale = 1.0 / n_steps as f32;
                for v in 0..vocab { d_modulated[v] *= scale; }
                m.backward(brain_refs[t], &d_modulated, &mut d_w, &mut d_b);
            }
            m.sgd_step(&mut d_w, &mut d_b, lr);
        }

        let nll_final = nll_per_token(&m, &qwen_refs, Some(&brain_refs), &targets);
        assert!(nll_final.is_finite());
        assert!(nll_final < nll_initial * 0.5,
            "training must reduce NLL by at least 2× \
             (initial = {nll_initial:.4}, final = {nll_final:.4}); \
             smaller delta means the optimizer is not finding the \
             gradient signal end-to-end");
    }

    /// The honest redshiftzero claim: when Qwen is **already
    /// competent** (not uniform-zero), can the modulator still drive
    /// NLL further down using the brain's signal? That's the
    /// real-world question — Qwen alone has a baseline; brain attached
    /// must beat that baseline.
    ///
    /// Setup: synthetic "Qwen" logits peak at `(target + offset[t]) %
    /// vocab` — wrong by a deterministic per-step offset. Qwen alone
    /// has high NLL because the peak is in the wrong place. Brain
    /// output is a distinct deterministic vector per step, so the
    /// modulator has ground to work with: learn a step-conditional
    /// correction that shifts the logit mass back to `target`.
    ///
    /// Assertion: trained NLL < Qwen-alone baseline NLL by at least 30%.
    /// 30% is the empirically-honest bar for ~1.3K parameters
    /// (vocab × brain_dim) trained on 8 token positions; weaker would
    /// be drift, much stronger would risk overfitting noise.
    #[test]
    fn training_improves_over_competent_baseline() {
        let vocab = 16;
        let brain_dim = 8;
        let n_steps = 8;
        let targets: Vec<usize> = (0..n_steps).map(|t| (t * 5 + 3) % vocab).collect();
        // Per-step offset so Qwen's peak is wrong by a *different*
        // amount per step — modulator must use brain output to know
        // which correction to apply.
        let offsets: Vec<i32> = (0..n_steps as i32).map(|t| ((t * 7) % 11) - 5).collect();

        let qwen: Vec<Vec<f32>> = (0..n_steps).map(|t| {
            let wrong_peak = (targets[t] as i32 + offsets[t]).rem_euclid(vocab as i32) as usize;
            let mut logits = vec![0.0f32; vocab];
            logits[wrong_peak] = 4.0;       // strong but not infinite
            logits
        }).collect();
        let qwen_refs: Vec<&[f32]> = qwen.iter().map(|v| v.as_slice()).collect();

        let brain: Vec<Vec<f32>> = (0..n_steps).map(|t| {
            (0..brain_dim).map(|i| ((t * 13 + i * 7) as f32 * 0.21).sin()).collect()
        }).collect();
        let brain_refs: Vec<&[f32]> = brain.iter().map(|v| v.as_slice()).collect();

        let mut m = BrainLogitModulator::new(brain_dim, vocab);
        m.alpha = 1.0;

        let nll_baseline = nll_per_token(&m, &qwen_refs, None, &targets);
        let nll_initial  = nll_per_token(&m, &qwen_refs, Some(&brain_refs), &targets);
        assert!(nll_baseline.is_finite() && nll_baseline > 0.5,
            "synthetic Qwen should be wrong enough to leave headroom \
             (baseline = {nll_baseline:.4})");
        // Sanity: at random init, brain-attached starts close to baseline.
        // (We're not asserting equality — random projection adds noise.)

        // Train.
        let mut d_w = vec![0.0f32; vocab * brain_dim];
        let mut d_b = vec![0.0f32; vocab];
        let mut modulated = vec![0.0f32; vocab];
        let mut d_modulated = vec![0.0f32; vocab];
        let lr = 0.1;
        let n_train = 500;
        for _step in 0..n_train {
            for t in 0..n_steps {
                m.modulate_into(qwen_refs[t], brain_refs[t], &mut modulated);
                cross_entropy_grad(&modulated, targets[t], &mut d_modulated);
                let scale = 1.0 / n_steps as f32;
                for v in 0..vocab { d_modulated[v] *= scale; }
                m.backward(brain_refs[t], &d_modulated, &mut d_w, &mut d_b);
            }
            m.sgd_step(&mut d_w, &mut d_b, lr);
        }

        let nll_final = nll_per_token(&m, &qwen_refs, Some(&brain_refs), &targets);
        assert!(nll_final.is_finite());
        let improvement = (nll_baseline - nll_final) / nll_baseline;
        assert!(improvement > 0.30,
            "trained brain+modulator must reduce NLL by ≥30% over \
             Qwen-alone baseline (baseline = {nll_baseline:.4}, \
             initial-with-random-brain = {nll_initial:.4}, \
             final = {nll_final:.4}, improvement = {:.2}%)",
            improvement * 100.0);
    }

    /// **Generalization** test — the strongest claim available without
    /// real Qwen weights. The prior training tests could be the
    /// modulator memorizing per-step (Qwen, brain, target) triplets.
    /// This test holds out unseen steps and asserts the modulator's
    /// learned correction transfers.
    ///
    /// Setup: 16 steps of synthetic (Qwen, brain, target). Targets
    /// follow a deterministic function of the brain output's argmax
    /// index — so a modulator that learns the brain → target mapping
    /// will generalize to any new brain vector drawn from the same
    /// distribution. Train on first 12 steps; held-out test = last 4.
    ///
    /// Assertion: held-out NLL after training < held-out NLL before
    /// training. The bar is intentionally loose (any improvement) so
    /// the test fails on overfit, not on a tight threshold that
    /// might pass on noise.
    #[test]
    fn training_generalizes_to_held_out_steps() {
        let vocab = 16;
        let brain_dim = 8;
        let n_total = 16;
        let n_train = 12;

        // Brain output: one-hot-ish (sharp peak at deterministic index).
        // Index varies per step so the brain space is well-covered.
        let brain: Vec<Vec<f32>> = (0..n_total).map(|t| {
            let peak = (t * 5 + 1) % brain_dim;
            let mut v = vec![0.1f32; brain_dim];
            v[peak] = 1.0;
            v
        }).collect();
        let brain_refs: Vec<&[f32]> = brain.iter().map(|v| v.as_slice()).collect();

        // Target = function of brain peak (modulo vocab) — same rule
        // for train and test, so the modulator can generalize.
        let targets: Vec<usize> = (0..n_total).map(|t| {
            let peak = (t * 5 + 1) % brain_dim;
            (peak * 2 + 1) % vocab
        }).collect();

        // Qwen baseline: small uniform — modulator does the lifting.
        let qwen: Vec<Vec<f32>> = (0..n_total).map(|_| vec![0.0f32; vocab]).collect();
        let qwen_refs: Vec<&[f32]> = qwen.iter().map(|v| v.as_slice()).collect();

        let train_qwen = &qwen_refs[..n_train];
        let train_brain = &brain_refs[..n_train];
        let train_tgts = &targets[..n_train];
        let test_qwen = &qwen_refs[n_train..];
        let test_brain = &brain_refs[n_train..];
        let test_tgts = &targets[n_train..];

        let mut m = BrainLogitModulator::new(brain_dim, vocab);
        m.alpha = 1.0;
        let test_nll_before = nll_per_token(
            &m, test_qwen, Some(test_brain), test_tgts);
        assert!(test_nll_before.is_finite());

        // Train on first 12 only.
        let mut d_w = vec![0.0f32; vocab * brain_dim];
        let mut d_b = vec![0.0f32; vocab];
        let mut modulated = vec![0.0f32; vocab];
        let mut d_modulated = vec![0.0f32; vocab];
        let lr = 0.1;
        for _step in 0..400 {
            for t in 0..n_train {
                m.modulate_into(train_qwen[t], train_brain[t], &mut modulated);
                cross_entropy_grad(&modulated, train_tgts[t], &mut d_modulated);
                let scale = 1.0 / n_train as f32;
                for v in 0..vocab { d_modulated[v] *= scale; }
                m.backward(train_brain[t], &d_modulated, &mut d_w, &mut d_b);
            }
            m.sgd_step(&mut d_w, &mut d_b, lr);
        }

        let test_nll_after = nll_per_token(
            &m, test_qwen, Some(test_brain), test_tgts);
        assert!(test_nll_after.is_finite());

        assert!(test_nll_after < test_nll_before,
            "trained modulator must improve held-out NLL \
             (before = {test_nll_before:.4}, after = {test_nll_after:.4}); \
             same-or-worse means the optimizer overfit train rather \
             than learning the generalizable brain → target rule");

        // Also report train NLL to surface overfit in test logs if
        // the held-out drop is much smaller than the train drop.
        let train_nll = nll_per_token(
            &m, train_qwen, Some(train_brain), train_tgts);
        eprintln!("train NLL = {train_nll:.4}, held-out NLL = {test_nll_after:.4} \
                   (was {test_nll_before:.4})");
    }

    /// Low-rank modulator: alpha=0 must produce baseline regardless
    /// of brain input. Same invariant as the dense variant.
    #[test]
    fn low_rank_alpha_zero_is_no_op() {
        let m = LowRankBrainLogitModulator::new(8, 2, 16);
        let qwen = vec![0.1f32, -0.5, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0,
                        2.1, -1.3, 0.7, 0.4, -0.2, 0.9, 0.0, 0.5];
        let brain = vec![0.3f32, -0.7, 1.1, 0.0, 0.5, -0.4, 0.2, 0.8];
        let modulated = m.modulate(&qwen, &brain);
        assert_eq!(modulated, qwen);
    }

    /// Low-rank modulator: param-count-vs-rank scaling check.
    /// At brain_dim=512, rank=8, vocab=152K the count must drop ~57×
    /// vs the dense `BrainLogitModulator(512, 152K)`. This is the
    /// structural property the low-rank form was added for; lock it.
    #[test]
    fn low_rank_param_count_drops_vs_dense() {
        let dense = BrainLogitModulator::new(512, 151_936);
        let lowrank = LowRankBrainLogitModulator::new(512, 8, 151_936);
        let dense_params = dense.proj.weight.len() + dense.proj.bias.len();
        let lr_params = lowrank.n_params();
        assert!(dense_params > 50 * lr_params,
            "expected ≥50× parameter reduction at rank=8 \
             (dense = {dense_params}, low-rank = {lr_params})");
    }

    /// Backward must match finite-difference for both the up and down
    /// projections. f(W) = ½‖out‖², so d_modulated = out (chain rule).
    #[test]
    fn low_rank_backward_matches_finite_difference() {
        let mut m = LowRankBrainLogitModulator::new(4, 2, 6);
        m.alpha = 0.7;
        let qwen = vec![0.0f32; 6];
        let brain = vec![0.5, -0.3, 0.7, 0.1];

        let out = m.modulate(&qwen, &brain);
        let d_modulated = out.clone();

        let mut d_dw = vec![0.0f32; 2 * 4];
        let mut d_db = vec![0.0f32; 2];
        let mut d_uw = vec![0.0f32; 6 * 2];
        let mut d_ub = vec![0.0f32; 6];
        m.backward(&brain, &d_modulated, &mut d_dw, &mut d_db, &mut d_uw, &mut d_ub);

        // FD on up.weight[v=3, r=1]
        let v = 3; let r = 1;
        let eps = 1e-3;
        let saved = m.up.weight[v * 2 + r];
        m.up.weight[v * 2 + r] = saved + eps;
        let l_plus: f32 = m.modulate(&qwen, &brain).iter()
            .map(|x| 0.5 * x * x).sum();
        m.up.weight[v * 2 + r] = saved - eps;
        let l_minus: f32 = m.modulate(&qwen, &brain).iter()
            .map(|x| 0.5 * x * x).sum();
        m.up.weight[v * 2 + r] = saved;
        let num = (l_plus - l_minus) / (2.0 * eps);
        let analytic = d_uw[v * 2 + r];
        assert!((num - analytic).abs() < 1e-4,
            "low-rank up.weight[{v},{r}] FD = {num:.6} analytic = {analytic:.6}");

        // FD on down.weight[r=0, b=2]
        let rd = 0; let b = 2;
        let saved = m.down.weight[rd * 4 + b];
        m.down.weight[rd * 4 + b] = saved + eps;
        let l_plus: f32 = m.modulate(&qwen, &brain).iter()
            .map(|x| 0.5 * x * x).sum();
        m.down.weight[rd * 4 + b] = saved - eps;
        let l_minus: f32 = m.modulate(&qwen, &brain).iter()
            .map(|x| 0.5 * x * x).sum();
        m.down.weight[rd * 4 + b] = saved;
        let num = (l_plus - l_minus) / (2.0 * eps);
        let analytic = d_dw[rd * 4 + b];
        assert!((num - analytic).abs() < 1e-4,
            "low-rank down.weight[{rd},{b}] FD = {num:.6} analytic = {analytic:.6}");
    }

    /// Low-rank training reduces NLL on synthetic data. Same setup as
    /// the dense `training_reduces_nll` so the two are comparable.
    #[test]
    fn low_rank_training_reduces_nll() {
        let vocab = 16; let brain_dim = 8; let rank = 4; let n_steps = 10;
        let qwen: Vec<Vec<f32>> = (0..n_steps).map(|_| vec![0.0f32; vocab]).collect();
        let qwen_refs: Vec<&[f32]> = qwen.iter().map(|v| v.as_slice()).collect();
        let brain: Vec<Vec<f32>> = (0..n_steps).map(|t| {
            (0..brain_dim).map(|i| ((t * 13 + i * 7) as f32 * 0.21).sin()).collect()
        }).collect();
        let brain_refs: Vec<&[f32]> = brain.iter().map(|v| v.as_slice()).collect();
        let targets: Vec<usize> = (0..n_steps).map(|t| (t * 5 + 3) % vocab).collect();

        let mut m = LowRankBrainLogitModulator::new(brain_dim, rank, vocab);
        m.alpha = 1.0;

        // Initial NLL via direct compute (no public NLL helper for low-rank yet).
        let initial: f32 = (0..n_steps).map(|t| {
            let modulated = m.modulate(qwen_refs[t], brain_refs[t]);
            let max_l = modulated.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let lse: f32 = modulated.iter()
                .map(|&l| (l - max_l).exp()).sum::<f32>().ln() + max_l;
            lse - modulated[targets[t]]
        }).sum::<f32>() / n_steps as f32;
        assert!(initial.is_finite() && initial > 0.0);

        let mut d_dw = vec![0.0f32; rank * brain_dim];
        let mut d_db = vec![0.0f32; rank];
        let mut d_uw = vec![0.0f32; vocab * rank];
        let mut d_ub = vec![0.0f32; vocab];
        let mut d_modulated = vec![0.0f32; vocab];
        let lr = 0.05;
        let scale = 1.0 / n_steps as f32;
        for _ in 0..200 {
            for t in 0..n_steps {
                let modulated = m.modulate(qwen_refs[t], brain_refs[t]);
                cross_entropy_grad(&modulated, targets[t], &mut d_modulated);
                for v in 0..vocab { d_modulated[v] *= scale; }
                m.backward(brain_refs[t], &d_modulated,
                    &mut d_dw, &mut d_db, &mut d_uw, &mut d_ub);
            }
            m.sgd_step(&mut d_dw, &mut d_db, &mut d_uw, &mut d_ub, lr);
        }

        let final_nll: f32 = (0..n_steps).map(|t| {
            let modulated = m.modulate(qwen_refs[t], brain_refs[t]);
            let max_l = modulated.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let lse: f32 = modulated.iter()
                .map(|&l| (l - max_l).exp()).sum::<f32>().ln() + max_l;
            lse - modulated[targets[t]]
        }).sum::<f32>() / n_steps as f32;
        assert!(final_nll < initial * 0.5,
            "low-rank training must cut NLL by ≥2× \
             (initial = {initial:.4}, final = {final_nll:.4})");
    }

    /// `cross_entropy_grad` must satisfy the gradient identity
    /// `sum_v ∂L/∂logits[v] = 0` (the softmax shifts mass between
    /// classes but doesn't add or remove total probability mass).
    /// Catches sign errors and missing target subtraction.
    #[test]
    fn cross_entropy_grad_sums_to_zero() {
        let logits = vec![0.5f32, -1.2, 0.3, 2.1, -0.7, 1.4, 0.0, -0.4];
        let mut grad = vec![0.0f32; logits.len()];
        cross_entropy_grad(&logits, 3, &mut grad);
        let s: f32 = grad.iter().sum();
        assert!(s.abs() < 1e-5,
            "softmax-target gradient must sum to zero (got {s:.6})");
    }

    /// Modulator backward must agree with a finite-difference reference.
    /// f(W) = sum_v out[v]^2 / 2 (so df/d_modulated[v] = out[v]).
    /// Then backward should produce d_W[v,b] ≈ out[v] · alpha · brain[b].
    /// Compares analytic vs numerical for one (v, b) entry.
    #[test]
    fn backward_matches_finite_difference() {
        let mut m = BrainLogitModulator::new(4, 6);
        m.alpha = 0.7;
        let qwen = vec![0.0f32; 6];
        let brain = vec![0.5, -0.3, 0.7, 0.1];

        let out = m.modulate(&qwen, &brain);
        // d_modulated = out (gradient of L=½‖out‖²)
        let d_modulated = out.clone();

        let mut d_w = vec![0.0f32; 6 * 4];
        let mut d_b = vec![0.0f32; 6];
        m.backward(&brain, &d_modulated, &mut d_w, &mut d_b);

        // Finite-difference d_W[2, 1].
        let v = 2; let bidx = 1;
        let eps = 1e-3;
        let saved = m.proj.weight[v * 4 + bidx];

        m.proj.weight[v * 4 + bidx] = saved + eps;
        let plus_out = m.modulate(&qwen, &brain);
        let l_plus: f32 = plus_out.iter().map(|x| 0.5 * x * x).sum();

        m.proj.weight[v * 4 + bidx] = saved - eps;
        let minus_out = m.modulate(&qwen, &brain);
        let l_minus: f32 = minus_out.iter().map(|x| 0.5 * x * x).sum();

        m.proj.weight[v * 4 + bidx] = saved; // restore
        let num_grad = (l_plus - l_minus) / (2.0 * eps);
        let analytic = d_w[v * 4 + bidx];
        assert!((num_grad - analytic).abs() < 1e-4,
            "analytic backward d_W[{v},{bidx}] = {analytic:.6} vs \
             finite-diff {num_grad:.6}");
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
