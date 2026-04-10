//! Systems consolidation: host-side weight optimization during sleep.
//!
//! Models the spindle-ripple cycle observed during NREM sleep:
//!   1. Sleep spindle (thalamocortical burst) → perturb weights
//!   2. Sharp-wave ripple (hippocampal fast replay) → evaluate traces
//!   3. Synaptic tag-and-capture → apply if improved, discard if not
//!
//! The organism is gradient-free (Hebbian + LS). The host runs this
//! consolidation during sleep. The organism never sees the computation —
//! it wakes up with better weights ("overnight insight").
//!
//! Implementation: SPSA (Simultaneous Perturbation Stochastic Approximation)
//! maps naturally to the spindle-ripple cycle. Each spindle burst is a
//! random perturbation; each ripple replay evaluates the perturbation.
//!
//! This lives in host/eval/ because it's a HOST capability.
//! The organism module has no dependency on it.

use modgrad_compute::neuron::SimpleRng;

/// Sleep spindle probe: perturbs weights and evaluates the effect.
/// One probe = one spindle-ripple cycle.
pub struct SpindleProbe {
    /// Burst amplitude (perturbation scale).
    pub burst_amplitude: f32,
    /// Learning rate for weight update.
    pub lr: f32,
    /// Momentum coefficient.
    pub momentum: f32,
    /// Accumulated momentum per parameter.
    velocity: Vec<f32>,
    /// RNG for burst patterns.
    rng: SimpleRng,
}

impl SpindleProbe {
    pub fn new(n_params: usize, lr: f32, burst_amplitude: f32, momentum: f32) -> Self {
        Self {
            burst_amplitude,
            lr,
            momentum,
            velocity: vec![0.0; n_params],
            rng: SimpleRng::new(31337),
        }
    }

    /// One spindle-ripple cycle:
    ///   1. Spindle burst: perturb all weights with random Rademacher pattern
    ///   2. Ripple replay+: evaluate loss with positive perturbation
    ///   3. Ripple replay-: evaluate loss with negative perturbation
    ///   4. Capture: update weights in the direction that reduced loss
    ///
    /// Returns (loss_plus, loss_minus) for monitoring.
    pub fn cycle(
        &mut self,
        params: &mut [f32],
        ripple_replay: &mut dyn FnMut(&[f32]) -> f32,
    ) -> (f32, f32) {
        let n = params.len();

        // 1. Spindle burst: Rademacher pattern δ ∈ {-1, +1}^n
        let burst: Vec<f32> = (0..n).map(|_| {
            if (self.rng.next_u64() >> 17) & 1 == 0 { 1.0 } else { -1.0 }
        }).collect();

        // 2. Ripple replay with positive burst
        for i in 0..n { params[i] += self.burst_amplitude * burst[i]; }
        let loss_plus = ripple_replay(params);

        // 3. Ripple replay with negative burst
        for i in 0..n { params[i] -= 2.0 * self.burst_amplitude * burst[i]; }
        let loss_minus = ripple_replay(params);

        // Restore original
        for i in 0..n { params[i] += self.burst_amplitude * burst[i]; }

        // 4. Synaptic capture: update in direction of improvement
        let diff = loss_plus - loss_minus;
        for i in 0..n {
            let grad = diff / (2.0 * self.burst_amplitude) * burst[i];
            self.velocity[i] = self.momentum * self.velocity[i] + (1.0 - self.momentum) * grad;
            params[i] -= self.lr * self.velocity[i];
        }

        (loss_plus, loss_minus)
    }
}

/// Run multiple spindle-ripple cycles.
/// Returns (initial_loss, final_loss).
pub fn spindle_consolidate(
    params: &mut [f32],
    ripple_replay: &mut dyn FnMut(&[f32]) -> f32,
    n_cycles: usize,
    lr: f32,
    burst_amplitude: f32,
) -> (f32, f32) {
    let mut probe = SpindleProbe::new(params.len(), lr, burst_amplitude, 0.5);
    let initial_loss = ripple_replay(params);
    let mut last_loss = initial_loss;

    for _ in 0..n_cycles {
        let (lp, lm) = probe.cycle(params, ripple_replay);
        last_loss = (lp + lm) / 2.0;
    }

    (initial_loss, last_loss)
}

// ─── Systems Consolidation (Pareto-constrained) ────────────

/// Configuration for sleep-time systems consolidation.
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Maximum relative weight change per sleep cycle: |Δw|/|w| ≤ this.
    /// Default 0.01 (1%). The organism wakes up at most 1% different.
    pub max_relative_step: f32,
    /// Allow individual traces to regress up to this fraction.
    /// 0.0 = strict Pareto (nothing gets worse). 0.01 = 1% slack.
    pub pareto_slack: f32,
    /// Spindle-ripple cycles per sleep. More = better estimate, slower.
    pub cycles_per_sleep: usize,
    /// Learning rate for synaptic capture.
    pub lr: f32,
    /// Spindle burst amplitude.
    pub burst_amplitude: f32,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            max_relative_step: 0.01,
            pareto_slack: 0.01,
            cycles_per_sleep: 10,
            lr: 0.001,
            burst_amplitude: 0.1,
        }
    }
}

/// Result of systems consolidation.
#[derive(Debug)]
pub struct ConsolidationResult {
    /// Whether the update was captured (Pareto check passed).
    pub captured: bool,
    /// Mean loss before consolidation.
    pub loss_before: f32,
    /// Mean loss after consolidation (before Pareto check).
    pub loss_after: f32,
    /// Number of traces that regressed beyond slack.
    pub n_regressed: usize,
    /// Total traces evaluated.
    pub n_traces: usize,
}

/// Systems consolidation: optimize weights during sleep with Pareto safety.
///
/// Models the hippocampal-cortical transfer during NREM:
///   1. Snapshot current synaptic weights
///   2. Run spindle-ripple cycles (probe nearby weight configurations)
///   3. Clamp total change (organism wakes up at most 1% different)
///   4. Ripple replay ALL stored traces: check Pareto (nothing gets worse)
///   5. Capture if safe, discard if any trace regressed
///
/// This is "overnight insight": the organism goes to sleep stuck,
/// wakes up with better weights it can't explain.
pub fn systems_consolidation(
    params: &mut [f32],
    eval_per_trace: &mut dyn FnMut(&[f32], usize) -> f32,
    n_traces: usize,
    ripple_replay: &mut dyn FnMut(&[f32]) -> f32,
    config: &ConsolidationConfig,
) -> ConsolidationResult {
    let n = params.len();
    if n == 0 || n_traces == 0 {
        return ConsolidationResult {
            captured: false, loss_before: 0.0, loss_after: 0.0,
            n_regressed: 0, n_traces: 0,
        };
    }

    // 1. Snapshot
    let snapshot: Vec<f32> = params.to_vec();
    let baseline_losses: Vec<f32> = (0..n_traces)
        .map(|i| eval_per_trace(params, i))
        .collect();
    let loss_before = baseline_losses.iter().sum::<f32>() / n_traces as f32;

    // 2. Spindle-ripple cycles
    let mut probe = SpindleProbe::new(n, config.lr, config.burst_amplitude, 0.5);
    for _ in 0..config.cycles_per_sleep {
        probe.cycle(params, ripple_replay);
    }

    // 3. Clamp total change
    let weight_norm: f32 = snapshot.iter().map(|w| w * w).sum::<f32>().sqrt().max(1e-8);
    let delta_norm: f32 = params.iter().zip(&snapshot)
        .map(|(new, old)| (new - old).powi(2)).sum::<f32>().sqrt();
    let max_delta = config.max_relative_step * weight_norm;

    if delta_norm > max_delta {
        let scale = max_delta / delta_norm;
        for i in 0..n {
            params[i] = snapshot[i] + (params[i] - snapshot[i]) * scale;
        }
    }

    // 4. Pareto check via ripple replay
    let new_losses: Vec<f32> = (0..n_traces)
        .map(|i| eval_per_trace(params, i))
        .collect();
    let loss_after = new_losses.iter().sum::<f32>() / n_traces as f32;

    let mut n_regressed = 0;
    for i in 0..n_traces {
        let allowed = baseline_losses[i] * (1.0 + config.pareto_slack);
        if new_losses[i] > allowed {
            n_regressed += 1;
        }
    }

    // 5. Capture or discard
    let mean_improved = loss_after < loss_before;
    let pareto_ok = n_regressed == 0;
    let captured = mean_improved && pareto_ok;

    if !captured {
        params.copy_from_slice(&snapshot);
    }

    ConsolidationResult {
        captured,
        loss_before,
        loss_after,
        n_regressed,
        n_traces,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spindle_consolidation_quadratic() {
        // Minimize f(x) = (x - 3)^2 + (y - 5)^2
        let mut params = vec![0.0f32, 0.0f32];
        let mut ripple = |p: &[f32]| -> f32 {
            (p[0] - 3.0).powi(2) + (p[1] - 5.0).powi(2)
        };

        let (init, final_loss) = spindle_consolidate(&mut params, &mut ripple, 500, 0.01, 0.5);
        eprintln!("Spindle consolidation: {init:.3} → {final_loss:.3}, params=[{:.3}, {:.3}]",
            params[0], params[1]);
        assert!(final_loss < init * 0.5, "loss should decrease: {init} → {final_loss}");
        assert!((params[0] - 3.0).abs() < 2.0, "x near 3: {}", params[0]);
        assert!((params[1] - 5.0).abs() < 2.0, "y near 5: {}", params[1]);
    }
}
