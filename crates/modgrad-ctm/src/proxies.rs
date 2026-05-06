//! Zero-cost trainability proxies for brain architectures.
//!
//! Cheap signals that score an architecture *at initialisation*,
//! without training. Inspired by NAS Survey §4.4: synflow, NASWOT,
//! grad-norm, etc. The brain's CTM tick loop has no clean "ReLU
//! linear-region" definition that NASWOT assumes, and synflow needs
//! a backward pass that's heavy here, so we use what the brain does
//! naturally: a forward pass + entropy of the prediction head.
//!
//! Usage pattern (from `brain_nas_smoke`):
//!
//! ```text
//! let cfg = arch.to_regional_config(obs_dim, out_dims);
//! let weights = RegionalWeights::new(cfg);
//! let entropy = forward_entropy(&weights, obs_dim, seed);
//! ```
//!
//! A score near `ln(out_dims)` means the random-init brain produces
//! a near-uniform output distribution — healthy signal flow. A score
//! near 0 means one output class dominates — collapsed init,
//! unlikely to train.

use modgrad_compute::neuron::SimpleRng;

use crate::graph::{regional_forward, RegionalState, RegionalWeights};

/// Run one forward pass with a random observation and return the
/// softmax entropy of the *last-tick* prediction. Higher = better
/// signal propagation at init. Range: `[0, ln(out_dims)]`.
///
/// Cost: one full forward through the brain. ~milliseconds for
/// small brains, ~tens of ms for billion-class. No backward.
pub fn forward_entropy(weights: &RegionalWeights, obs_dim: usize, seed: u64) -> f32 {
    let mut rng = SimpleRng::new(seed);
    let observation: Vec<f32> = (0..obs_dim).map(|_| rng.next_normal()).collect();
    let mut state = RegionalState::new(weights);
    let output = regional_forward(weights, &mut state, &observation);

    let logits = match output.predictions.last() {
        Some(p) if !p.is_empty() => p,
        _ => return 0.0,
    };
    softmax_entropy(logits)
}

/// **Recommended proxy**: input-sensitivity at init via the
/// std of forward-entropy over `n_samples` random observations,
/// passed through `tanh(20·x)` for bounded [0,1] output.
///
/// In `brain_nas_correlation` measurements (2026-05-05, N=20),
/// this single-component proxy hits Spearman ρ=+0.614 against
/// training loss drop — beating the combined
/// `entropy_norm × tanh(20·entropy_std)` formula (ρ=+0.581).
/// The `entropy_norm` component turned out to be slightly
/// anti-predictive: high mean entropy with low input-variation
/// is the "dead-uniform output" regime, not the healthy regime.
///
/// Use this as the default ranking signal for NAS.
pub fn forward_sensitivity_score(
    weights: &RegionalWeights,
    obs_dim: usize,
    n_samples: usize,
    seed: u64,
) -> f32 {
    let (_mean, std) = forward_entropy_stats(weights, obs_dim, n_samples, seed);
    (20.0 * std).tanh()
}

/// Average forward entropy across `n_samples` random observations
/// (different seeds). Smooths out per-input variance for a more
/// stable per-arch ranking. Cost: `n_samples` × forward.
pub fn forward_entropy_mean(
    weights: &RegionalWeights,
    obs_dim: usize,
    n_samples: usize,
    seed: u64,
) -> f32 {
    forward_entropy_stats(weights, obs_dim, n_samples, seed).0
}

/// Returns `(mean_entropy, std_entropy)` over `n_samples` random
/// observations. The std measures *input sensitivity*: a brain with
/// high mean entropy but std≈0 is producing the same near-uniform
/// distribution regardless of input — likely insensitive to the obs.
/// A healthy random init has high mean entropy *and* visible std,
/// indicating the prediction head reacts to input variation.
pub fn forward_entropy_stats(
    weights: &RegionalWeights,
    obs_dim: usize,
    n_samples: usize,
    seed: u64,
) -> (f32, f32) {
    if n_samples == 0 {
        return (0.0, 0.0);
    }
    let mut samples = Vec::with_capacity(n_samples);
    for k in 0..n_samples {
        samples.push(forward_entropy(weights, obs_dim, seed.wrapping_add(k as u64)));
    }
    let mean = samples.iter().copied().sum::<f32>() / n_samples as f32;
    if n_samples == 1 {
        return (mean, 0.0);
    }
    let var = samples.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / n_samples as f32;
    (mean, var.sqrt())
}

/// Stable softmax entropy in nats. Subtracts the max logit for
/// numerical stability before the exp pass.
pub fn softmax_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return 0.0;
    }
    let mut sum_exp = 0.0f32;
    for &l in logits {
        sum_exp += (l - max).exp();
    }
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return 0.0;
    }
    let log_sum = sum_exp.ln() + max;
    let mut entropy = 0.0f32;
    for &l in logits {
        let log_p = l - log_sum;
        let p = log_p.exp();
        if p > 0.0 {
            entropy -= p * log_p;
        }
    }
    entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_uniform_logits_equals_log_n() {
        let logits = vec![0.0f32; 16];
        let h = softmax_entropy(&logits);
        let expected = (16f32).ln();
        assert!((h - expected).abs() < 1e-5,
            "uniform entropy {} should be ln(16) ≈ {}", h, expected);
    }

    #[test]
    fn entropy_one_hot_logits_near_zero() {
        let mut logits = vec![0.0f32; 16];
        logits[0] = 100.0;
        let h = softmax_entropy(&logits);
        assert!(h < 1e-3, "near one-hot entropy should be ~0, got {}", h);
    }

    #[test]
    fn entropy_handles_nan_safely() {
        let logits = vec![f32::NAN, 0.0, 0.0];
        let h = softmax_entropy(&logits);
        // Either 0.0 (NaN -> max NaN -> early return) or finite
        assert!(h.is_finite() || h == 0.0, "entropy must not propagate NaN");
    }

    #[test]
    fn forward_entropy_runs_on_small_arch() {
        use crate::search_space::BrainArch;
        let arch = BrainArch::eight_region_small_arch();
        let cfg = arch.to_regional_config(16, 8);
        let weights = RegionalWeights::new(cfg);
        let h = forward_entropy(&weights, 16, 0xBEEF);
        // 8-class output: max entropy = ln(8) ≈ 2.08
        assert!(h.is_finite(), "entropy must be finite, got {}", h);
        assert!(h >= 0.0 && h <= (8f32).ln() + 1e-3,
            "entropy {} out of [0, ln(8)] range", h);
    }
}
