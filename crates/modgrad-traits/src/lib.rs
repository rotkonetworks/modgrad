//! Core traits for the modgrad ML SDK.
//!
//! Each component is a pure function (or explicit state transform)
//! that composes with the others.
//!
//! Brain:    (Weights, State, Input) → (Output, State, Cache)
//! Backward: (Weights, Cache, dOutput) → Gradients
//! Loss:     (Output, Target) → (scalar, dOutput)
//! Update:   (Weights, Gradients, lr) → Weights

/// Trait for weight types that can be viewed as a flat f32 slice.
/// Enables Optimizer to operate on Brain weights directly.
pub trait Flattenable {
    /// View all trainable parameters as a flat slice.
    fn as_flat(&self) -> &[f32];
    /// View all trainable parameters as a mutable flat slice.
    fn as_flat_mut(&mut self) -> &mut [f32];
    /// Total number of trainable parameters.
    fn n_params(&self) -> usize { self.as_flat().len() }
}

/// Output of a brain forward pass.
pub struct BrainOutput {
    /// Predictions at each tick. predictions[t] has length out_dims.
    /// Architectures without ticks (e.g. transformer) produce one entry.
    pub predictions: Vec<Vec<f32>>,
    /// Certainty at each tick: [normalized_entropy, 1 - normalized_entropy].
    pub certainties: Vec<[f32; 2]>,
    /// Final representation (for downstream use).
    pub sync: Vec<f32>,
}

// ─── Input types ───────────────────────────────────────────
// Each Brain declares its Input type. These are the standard ones.
// Custom architectures can define their own.

/// Input for brains that take pre-embedded float token vectors.
/// Used by: CTM (external embedding table, Brain receives vectors).
pub struct TokenInput {
    /// Flat token data: [n_tokens × token_dim].
    pub tokens: Vec<f32>,
    /// Number of tokens in the sequence.
    pub n_tokens: usize,
    /// Dimensionality of each token vector.
    pub token_dim: usize,
}

/// Input for brains that take integer token IDs and embed internally.
/// Used by: Transformer (embedding table is part of Weights).
pub struct SeqInput {
    /// Token IDs for the input sequence.
    pub token_ids: Vec<u32>,
}

// ─── Brain trait ────────────────────────────────────────────

/// The core brain interface — a service from Input to BrainOutput.
///
/// A brain is a pure function: (state, input) → (output, new_state).
/// State is explicit — passed in and returned, never mutated in place.
/// Cache captures intermediates for backward pass.
///
/// The Input associated type lets each service declare its own request type. CTM takes TokenInput (pre-embedded
/// float vectors), Transformer takes SeqInput (integer token IDs).
/// Filters can transform one input type into another.
pub trait Brain {
    /// The input representation this brain expects.
    type Input;
    /// Persistent weights (serializable, saveable).
    type Weights;
    /// Per-forward ephemeral state (carried across tokens, not saved).
    type State;
    /// Cached intermediates for backward pass.
    type Cache;
    /// Accumulated gradients.
    type Gradients;

    /// Create initial state for a new sequence.
    fn init_state(weights: &Self::Weights) -> Self::State;

    /// Forward pass. Pure function — state goes in, new state comes out.
    fn forward(
        weights: &Self::Weights,
        state: Self::State,
        input: &Self::Input,
    ) -> (BrainOutput, Self::State);

    /// Forward with caching for BPTT.
    fn forward_cached(
        weights: &Self::Weights,
        state: Self::State,
        input: &Self::Input,
    ) -> (BrainOutput, Self::State, Self::Cache);

    /// Backward pass. Pure function of cache and upstream gradient.
    fn backward(
        weights: &Self::Weights,
        cache: Self::Cache,
        d_predictions: &[Vec<f32>],
    ) -> Self::Gradients;

    /// Create zero-initialized gradients.
    fn zero_gradients(weights: &Self::Weights) -> Self::Gradients;

    /// Apply gradients to weights (SGD step).
    fn apply_gradients(
        weights: &mut Self::Weights,
        grads: &Self::Gradients,
        lr: f32,
        clip_norm: f32,
    );
}

// ─── Loss trait ─────────────────────────────────────────────

/// A loss function: (predictions, certainties, target) → (loss, d_predictions).
/// Pure function, no state.
///
/// The Target type is generic — classification uses `usize`,
/// next-token prediction uses `&[f32]` (distribution),
/// reconstruction uses `Vec<f32>`, RL uses `f32` (reward).
pub trait LossFn {
    /// The target type for this loss.
    type Target: ?Sized;

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        certainties: &[[f32; 2]],
        target: &Self::Target,
    ) -> (f32, Vec<Vec<f32>>);
}

/// Convenience type alias for classification targets.
pub type ClassTarget = usize;

/// CTM-style loss: (min_CE_tick + most_certain_tick) / 2.
pub struct CtmLoss;

impl LossFn for CtmLoss {
    type Target = ClassTarget;

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        certainties: &[[f32; 2]],
        target: &ClassTarget,
    ) -> (f32, Vec<Vec<f32>>) {
        let target = *target;
        let k = predictions.len();
        if k == 0 { return (0.0, Vec::new()); }
        let out_dim = predictions[0].len();

        let ce: Vec<(f32, Vec<f32>)> = predictions.iter()
            .map(|p| cross_entropy_grad(p, target))
            .collect();

        let min_tick = (0..k).min_by(|&a, &b|
            ce[a].0.partial_cmp(&ce[b].0).unwrap()).unwrap_or(0);
        let cert_tick = (0..k).max_by(|&a, &b|
            certainties[a][1].partial_cmp(&certainties[b][1]).unwrap()).unwrap_or(k - 1);

        let loss = (ce[min_tick].0 + ce[cert_tick].0) / 2.0;

        let mut d_preds = vec![vec![0.0f32; out_dim]; k];
        for (j, g) in ce[min_tick].1.iter().enumerate() {
            d_preds[min_tick][j] += 0.5 * g;
        }
        for (j, g) in ce[cert_tick].1.iter().enumerate() {
            d_preds[cert_tick][j] += 0.5 * g;
        }

        (loss, d_preds)
    }
}

/// Simple cross-entropy on the last tick only.
pub struct LastTickCE;

impl LossFn for LastTickCE {
    type Target = ClassTarget;

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        _certainties: &[[f32; 2]],
        target: &ClassTarget,
    ) -> (f32, Vec<Vec<f32>>) {
        let target = *target;
        let k = predictions.len();
        if k == 0 { return (0.0, Vec::new()); }
        let out_dim = predictions[0].len();
        let (loss, grad) = cross_entropy_grad(&predictions[k - 1], target);
        let mut d_preds = vec![vec![0.0f32; out_dim]; k];
        d_preds[k - 1] = grad;
        (loss, d_preds)
    }
}

/// Thinking-aware loss: CTM loss + reward for productive ticks.
pub struct ThinkingLoss {
    pub alpha: f32,
    pub min_improvement: f32,
}

impl Default for ThinkingLoss {
    fn default() -> Self {
        Self { alpha: 0.1, min_improvement: 0.01 }
    }
}

impl LossFn for ThinkingLoss {
    type Target = ClassTarget;

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        certainties: &[[f32; 2]],
        target: &ClassTarget,
    ) -> (f32, Vec<Vec<f32>>) {
        let target = *target;
        let k = predictions.len();
        if k == 0 { return (0.0, Vec::new()); }
        let out_dim = predictions[0].len();

        let ce: Vec<(f32, Vec<f32>)> = predictions.iter()
            .map(|p| cross_entropy_grad(p, target))
            .collect();

        let min_tick = (0..k).min_by(|&a, &b|
            ce[a].0.partial_cmp(&ce[b].0).unwrap()).unwrap_or(0);
        let cert_tick = (0..k).max_by(|&a, &b|
            certainties[a][1].partial_cmp(&certainties[b][1]).unwrap()).unwrap_or(k - 1);
        let base_loss = (ce[min_tick].0 + ce[cert_tick].0) / 2.0;

        let mut thinking_loss = 0.0f32;
        for t in 1..k {
            let improvement = ce[t - 1].0 - ce[t].0;
            if improvement < self.min_improvement {
                thinking_loss += self.min_improvement - improvement;
            }
        }
        thinking_loss /= (k - 1).max(1) as f32;

        let total_loss = base_loss + self.alpha * thinking_loss;

        let mut d_preds = vec![vec![0.0f32; out_dim]; k];
        for (j, g) in ce[min_tick].1.iter().enumerate() {
            d_preds[min_tick][j] += 0.5 * g;
        }
        for (j, g) in ce[cert_tick].1.iter().enumerate() {
            d_preds[cert_tick][j] += 0.5 * g;
        }

        let alpha_per_tick = self.alpha / (k - 1).max(1) as f32;
        for t in 1..k {
            let improvement = ce[t - 1].0 - ce[t].0;
            if improvement < self.min_improvement {
                for (j, g) in ce[t].1.iter().enumerate() {
                    d_preds[t][j] += alpha_per_tick * g;
                }
                for (j, g) in ce[t - 1].1.iter().enumerate() {
                    d_preds[t - 1][j] -= 0.5 * alpha_per_tick * g;
                }
            }
        }

        (total_loss, d_preds)
    }
}

/// Next-token prediction loss: KL divergence against soft labels.
pub struct DistributionLoss;

impl LossFn for DistributionLoss {
    type Target = [f32];

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        _certainties: &[[f32; 2]],
        target: &[f32],
    ) -> (f32, Vec<Vec<f32>>) {
        let k = predictions.len();
        if k == 0 { return (0.0, Vec::new()); }
        let last = &predictions[k - 1];
        let out_dim = last.len();

        let max_l = last.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_s: Vec<f32> = last.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = exp_s.iter().sum();
        let softmax: Vec<f32> = exp_s.iter().map(|&e| e / sum).collect();

        let mut loss = 0.0f32;
        for i in 0..out_dim.min(target.len()) {
            if target[i] > 1e-10 {
                loss += target[i] * (target[i] / softmax[i].max(1e-10)).ln();
            }
        }

        let grad: Vec<f32> = (0..out_dim).map(|i| {
            softmax[i] - target.get(i).copied().unwrap_or(0.0)
        }).collect();

        let mut d_preds = vec![vec![0.0f32; out_dim]; k];
        d_preds[k - 1] = grad;
        (loss, d_preds)
    }
}

/// Reconstruction loss: MSE for autoencoders.
pub struct ReconstructionLoss;

impl LossFn for ReconstructionLoss {
    type Target = [f32];

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        _certainties: &[[f32; 2]],
        target: &[f32],
    ) -> (f32, Vec<Vec<f32>>) {
        let k = predictions.len();
        if k == 0 { return (0.0, Vec::new()); }
        let last = &predictions[k - 1];
        let n = last.len().min(target.len());

        let loss: f32 = (0..n).map(|i| (last[i] - target[i]).powi(2)).sum::<f32>() / n as f32;
        let grad: Vec<f32> = (0..last.len()).map(|i| {
            if i < n { 2.0 * (last[i] - target[i]) / n as f32 } else { 0.0 }
        }).collect();

        let mut d_preds = vec![vec![0.0f32; last.len()]; k];
        d_preds[k - 1] = grad;
        (loss, d_preds)
    }
}

/// Reward-based loss: REINFORCE for RL.
pub struct RewardLoss;

impl LossFn for RewardLoss {
    type Target = f32;

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        _certainties: &[[f32; 2]],
        target: &f32,
    ) -> (f32, Vec<Vec<f32>>) {
        let k = predictions.len();
        if k == 0 { return (0.0, Vec::new()); }
        let reward = *target;
        let last = &predictions[k - 1];
        let out_dim = last.len();

        let max_l = last.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_s: Vec<f32> = last.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = exp_s.iter().sum();
        let softmax: Vec<f32> = exp_s.iter().map(|&e| e / sum).collect();

        let action = last.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);

        let log_prob = softmax[action].max(1e-10).ln();
        let loss = -reward * log_prob;

        let grad: Vec<f32> = (0..out_dim).map(|i| {
            let indicator = if i == action { 1.0 } else { 0.0 };
            -reward * (indicator - softmax[i])
        }).collect();

        let mut d_preds = vec![vec![0.0f32; out_dim]; k];
        d_preds[k - 1] = grad;
        (loss, d_preds)
    }
}

fn cross_entropy_grad(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_s: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp_s.iter().sum();
    let mut softmax: Vec<f32> = exp_s.iter().map(|&e| e / sum).collect();
    let loss = -(softmax.get(target).copied().unwrap_or(1e-8).max(1e-8)).ln();
    if target < softmax.len() { softmax[target] -= 1.0; }
    (loss, softmax)
}
