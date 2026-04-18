//! Looped Language Model (LoopLM) — weight-tied recurrence with learned halting.
//!
//! A transformer forward pass is a function `f: hidden → hidden`.
//! A looped transformer applies `f` repeatedly: `f(f(f(x)))`.
//! The exit gate observes the hidden state and produces a halt probability.
//! The halt policy decides when to stop based on accumulated probabilities.
//!
//! Three separate concerns, three separate types:
//!   - `BlockStack` — the N layers, applied once. Pure function.
//!   - `ExitGate` — observes hidden state, produces probability. Pure function.
//!   - `HaltPolicy` — decides when to stop. Caller's responsibility.
//!
//! Design:
//!   - Types encode invariants. `LoopState` carries hidden + cumulative exit prob.
//!   - The loop is a fold with early termination, not special control flow.
//!   - No hidden state mutation — each step produces new state.
//!   - The gate doesn't modify hidden state. It only observes.

// ─── Exit gate ─────────────────────────────────────────────

/// Learned exit gate: Linear(model_dim → 1) → sigmoid.
///
/// Produces the instantaneous exit probability λ_t at each step.
/// Does NOT modify the hidden state.
pub struct ExitGate {
    /// Weight: [1, model_dim] — single row.
    pub weight: Vec<f32>,
    /// Bias: [1].
    pub bias: f32,
}

impl ExitGate {
    pub fn new(model_dim: usize) -> Self {
        Self {
            weight: vec![0.0; model_dim],
            bias: 0.0,
        }
    }

    /// Compute instantaneous exit probability: σ(w·h + b).
    /// Returns λ ∈ (0, 1).
    #[inline]
    pub fn exit_probability(&self, hidden: &[f32]) -> f32 {
        let logit: f32 = self.weight.iter().zip(hidden).map(|(w, h)| w * h).sum::<f32>() + self.bias;
        1.0 / (1.0 + (-logit).exp())
    }
}

// ─── Halt policies ─────────────────────────────────────────

/// Policy that decides when to stop looping.
///
/// Separate from the gate (which produces probabilities) and the blocks
/// (which transform hidden state). The caller picks the policy.
pub trait HaltPolicy {
    /// Given the exit probability for this step and the step index (0-based),
    /// return true if the loop should terminate.
    fn should_halt(&mut self, exit_prob: f32, step: usize) -> bool;

    /// Reset state for a new sequence.
    fn reset(&mut self);
}

/// Always run exactly R steps. Ignores the exit gate entirely.
/// Use for standard (non-adaptive) inference.
pub struct FixedDepth {
    pub max_steps: usize,
}

impl FixedDepth {
    pub fn new(steps: usize) -> Self { Self { max_steps: steps } }
}

impl HaltPolicy for FixedDepth {
    fn should_halt(&mut self, _exit_prob: f32, step: usize) -> bool {
        step + 1 >= self.max_steps
    }
    fn reset(&mut self) {}
}

/// Accumulate exit probabilities as a CDF.
/// Stop when CDF exceeds threshold q (Ouro-style).
///
/// CDF(n) = 1 - ∏(1 - λ_j) for j=1..n.
/// Larger q = more thinking. Smaller q = faster.
pub struct CdfThreshold {
    /// Exit threshold ∈ [0, 1]. 0.5 = majority probability. 0.9 = very confident.
    pub threshold: f32,
    /// Maximum steps (hard cap regardless of CDF).
    pub max_steps: usize,
    /// Running survival probability: ∏(1 - λ_j).
    survival: f32,
}

impl CdfThreshold {
    pub fn new(threshold: f32, max_steps: usize) -> Self {
        Self { threshold, max_steps, survival: 1.0 }
    }

    /// Current CDF value.
    pub fn cdf(&self) -> f32 { 1.0 - self.survival }
}

impl HaltPolicy for CdfThreshold {
    fn should_halt(&mut self, exit_prob: f32, step: usize) -> bool {
        self.survival *= 1.0 - exit_prob;
        let cdf = 1.0 - self.survival;
        cdf >= self.threshold || step + 1 >= self.max_steps
    }

    fn reset(&mut self) { self.survival = 1.0; }
}

/// Stop when hidden state change is small.
/// Measures L2 norm of (h_new - h_old) / dim.
pub struct ConvergenceThreshold {
    pub epsilon: f32,
    pub max_steps: usize,
    pub min_steps: usize,
}

impl ConvergenceThreshold {
    pub fn new(epsilon: f32, min_steps: usize, max_steps: usize) -> Self {
        Self { epsilon, max_steps, min_steps }
    }
}

impl HaltPolicy for ConvergenceThreshold {
    fn should_halt(&mut self, _exit_prob: f32, step: usize) -> bool {
        // Note: this policy doesn't use exit_prob. The caller should
        // compute delta and pass it as exit_prob (overloaded meaning).
        // Better: provide the delta separately. For now, use exit_prob
        // as the normalized delta.
        if step + 1 < self.min_steps { return false; }
        if step + 1 >= self.max_steps { return true; }
        _exit_prob < self.epsilon
    }

    fn reset(&mut self) {}
}

/// Hard compute budget: stop after N steps, no gate needed.
pub struct BudgetCap {
    pub max_steps: usize,
}

impl BudgetCap {
    pub fn new(max_steps: usize) -> Self { Self { max_steps } }
}

impl HaltPolicy for BudgetCap {
    fn should_halt(&mut self, _: f32, step: usize) -> bool {
        step + 1 >= self.max_steps
    }
    fn reset(&mut self) {}
}

// ─── Loop state ────────────────────────────────────────────

/// State carried through the loop. Immutable per step — each step
/// produces a new LoopState.
#[derive(Clone)]
pub struct LoopState {
    /// Current hidden state [model_dim].
    pub hidden: Vec<f32>,
    /// Number of recurrent steps completed.
    pub steps_taken: usize,
    /// Cumulative exit probability (CDF).
    pub exit_cdf: f32,
    /// Whether the loop terminated early.
    pub early_exit: bool,
}

// ─── Loop executor ─────────────────────────────────────────

/// Execute the recurrent loop.
///
/// `apply_blocks` is a closure that runs the full block stack once:
///   `|hidden: &mut [f32], step: usize|`
/// This is a closure, not a method, so the caller controls what "one pass"
/// means — it could be the full GptModel block stack, a subset, or a
/// custom pipeline.
///
/// `gate` is optional. Without a gate, exit_prob is always 0 and the
/// policy receives 0 at each step (FixedDepth and BudgetCap still work).
///
/// Returns the final LoopState.
pub fn execute_loop<F>(
    initial_hidden: Vec<f32>,
    mut apply_blocks: F,
    gate: Option<&ExitGate>,
    policy: &mut dyn HaltPolicy,
) -> LoopState
where
    F: FnMut(&mut [f32], usize),
{
    policy.reset();

    let mut state = LoopState {
        hidden: initial_hidden,
        steps_taken: 0,
        exit_cdf: 0.0,
        early_exit: false,
    };

    let mut survival = 1.0f32;

    loop {
        let step = state.steps_taken;

        // Apply the block stack (modifies hidden in-place for efficiency)
        apply_blocks(&mut state.hidden, step);
        state.steps_taken += 1;

        // Compute exit probability
        let exit_prob = gate.map(|g| g.exit_probability(&state.hidden)).unwrap_or(0.0);
        survival *= 1.0 - exit_prob;
        state.exit_cdf = 1.0 - survival;

        // Check halt condition
        if policy.should_halt(exit_prob, step) {
            state.early_exit = step + 1 < usize::MAX; // true unless we somehow run forever
            break;
        }
    }

    state
}

// ─── Training loss ─────────────────────────────────────────

/// Compute the entropy-regularized LoopLM training loss (Equation 4 from Ouro).
///
/// `step_losses[t]` = cross-entropy loss at recurrent step t.
/// `exit_probs[t]` = λ_t from the exit gate at step t.
/// `beta` = entropy regularization coefficient.
///
/// L = Σ p(t|x) * L_t - β * H(p)
/// where p(t|x) is the exit distribution derived from λ_t.
pub fn loop_training_loss(
    step_losses: &[f32],
    exit_probs: &[f32],
    beta: f32,
) -> f32 {
    let t_max = step_losses.len();
    assert_eq!(exit_probs.len(), t_max);

    // Compute exit distribution p(t|x) from instantaneous probabilities
    let mut p = vec![0.0f32; t_max];
    let mut survival = 1.0f32;

    for t in 0..t_max - 1 {
        p[t] = exit_probs[t] * survival;
        survival *= 1.0 - exit_probs[t];
    }
    // Last step gets remaining mass
    p[t_max - 1] = survival;

    // Expected task loss: Σ p[t] * L[t]
    let expected_loss: f32 = p.iter().zip(step_losses).map(|(pt, lt)| pt * lt).sum();

    // Entropy: H(p) = -Σ p[t] * log(p[t])
    let entropy: f32 = p.iter()
        .filter(|&&pt| pt > 1e-10)
        .map(|&pt| -pt * pt.ln())
        .sum();

    expected_loss - beta * entropy
}

/// Compute the adaptive exit loss for Stage II gate training (Equation 6 from Ouro).
///
/// `step_losses[t]` = L_t (detached, no gradient through model).
/// `exit_probs[t]` = λ_t from the gate.
/// `gamma` = improvement threshold (default 0.005).
/// `k` = sigmoid slope (default 50.0).
///
/// Returns per-step BCE loss averaged over steps.
pub fn adaptive_exit_loss(
    step_losses: &[f32],
    exit_probs: &[f32],
    gamma: f32,
    k: f32,
) -> f32 {
    let t_max = step_losses.len();
    if t_max < 2 { return 0.0; }

    let mut total_loss = 0.0f32;
    let eps = 1e-7;

    for t in 1..t_max {
        // Improvement from step t-1 to t
        let improvement = (step_losses[t - 1] - step_losses[t]).max(0.0);

        // Ideal continuation label: sigmoid(k * (I - γ))
        let w = 1.0 / (1.0 + (-(k * (improvement - gamma))).exp());

        // Binary cross-entropy between (1-λ) and w
        let lambda = exit_probs[t].clamp(eps, 1.0 - eps);
        let cont = (1.0 - lambda).clamp(eps, 1.0 - eps);

        let bce = -(w * cont.ln() + (1.0 - w) * lambda.ln());
        total_loss += bce;
    }

    total_loss / (t_max - 1) as f32
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_depth() {
        let mut policy = FixedDepth::new(4);
        let state = execute_loop(
            vec![1.0; 64],
            |h, _step| { for v in h.iter_mut() { *v += 0.1; } },
            None,
            &mut policy,
        );
        assert_eq!(state.steps_taken, 4);
        assert!((state.hidden[0] - 1.4).abs() < 1e-5);
    }

    #[test]
    fn test_cdf_threshold() {
        let gate = ExitGate {
            weight: vec![0.0; 4],
            bias: 0.0, // sigmoid(0) = 0.5 → each step has 50% exit prob
        };
        let mut policy = CdfThreshold::new(0.9, 10);
        let state = execute_loop(
            vec![0.0; 4],
            |_, _| {},
            Some(&gate),
            &mut policy,
        );
        // Each step: survival *= 0.5. After 4 steps: CDF = 1 - 0.5^4 = 0.9375 > 0.9
        assert_eq!(state.steps_taken, 4);
        assert!(state.exit_cdf > 0.9);
    }

    #[test]
    fn test_training_loss() {
        let step_losses = vec![3.0, 2.5, 2.0, 1.8];
        let exit_probs = vec![0.1, 0.2, 0.3, 0.0]; // last doesn't matter
        let loss = loop_training_loss(&step_losses, &exit_probs, 0.1);
        // Should be a weighted average of losses minus entropy bonus
        assert!(loss > 0.0);
        assert!(loss < 3.0);
    }

    #[test]
    fn test_adaptive_exit_loss() {
        // Step losses decrease → gate should learn to continue
        let step_losses = vec![3.0, 2.0, 1.5, 1.4];
        let exit_probs = vec![0.1, 0.1, 0.1, 0.1];
        let loss = adaptive_exit_loss(&step_losses, &exit_probs, 0.005, 50.0);
        assert!(loss > 0.0);
    }
}
