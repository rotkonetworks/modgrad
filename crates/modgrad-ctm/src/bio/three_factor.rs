//! Three-factor REINFORCE with Titans-style adaptive eligibility traces.
//!
//! S_t = η_t · S_{t-1} + (1-η_t) · post · pre    (eligibility with adaptive decay)
//! ΔW  = θ_t · advantage · S_t                     (salience-scaled update)
//!
//! Titans (Behrouz et al. 2024) dynamics: data-dependent decay and learning rate.
//! Salience gating gave a large accuracy delta on the reference benchmark
//! that first validated this rule (see `examples/mazes`).

use super::salience;

/// Persistent state for three-factor learning on one synapse.
pub struct EligibilityState {
    /// Eligibility trace [out_dim × in_dim].
    pub trace: Vec<f32>,
    /// Running reward baseline (EMA).
    pub reward_baseline: f32,
    /// Per-synapse salience for adaptive forgetting.
    pub salience: f32,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl EligibilityState {
    pub fn new(out_dim: usize, in_dim: usize) -> Self {
        Self {
            trace: vec![0.0; out_dim * in_dim],
            reward_baseline: 1.0,
            salience: 0.0,
            out_dim,
            in_dim,
        }
    }

    /// One step of three-factor REINFORCE.
    ///
    /// `pre`: input activations [in_dim]
    /// `post`: output activations [out_dim]
    /// `reward`: current reward signal (e.g. dopamine)
    /// `motor_output`: for salience computation
    /// `elig_decay_base`: base eligibility decay η₀
    /// `base_lr`: base learning rate
    /// `reward_threshold`: minimum |advantage| to update
    ///
    /// Returns weight deltas [out_dim × in_dim].
    pub fn step(
        &mut self,
        pre: &[f32],
        post: &[f32],
        reward: f32,
        motor_output: &[f32],
        elig_decay_base: f32,
        base_lr: f32,
        reward_threshold: f32,
    ) -> Vec<f32> {
        // Salience gates learning rate
        let sal = salience::compute(reward, self.reward_baseline, motor_output);
        let eta = elig_decay_base + (1.0 - elig_decay_base) * 0.5 * sal.value.min(1.0);
        let theta = base_lr * sal.gate;

        // Update eligibility trace: S_t = η · S_{t-1} + (1-η) · post · pre
        for j in 0..self.out_dim.min(post.len()) {
            if post[j].abs() < 0.01 { continue; }
            for i in 0..self.in_dim.min(pre.len()) {
                let idx = j * self.in_dim + i;
                self.trace[idx] = eta * self.trace[idx] + (1.0 - eta) * post[j] * pre[i];
            }
        }

        // Advantage = reward - baseline
        let advantage = reward - self.reward_baseline;
        self.reward_baseline = 0.99 * self.reward_baseline + 0.01 * reward;

        // Weight deltas: θ · advantage · eligibility
        let mut deltas = vec![0.0f32; self.out_dim * self.in_dim];
        if advantage.abs() > reward_threshold && theta > 0.0 {
            for idx in 0..deltas.len() {
                deltas[idx] = theta * advantage * self.trace[idx];
            }
        }

        // Track salience for adaptive forgetting
        self.salience = 0.95 * self.salience + 0.05 * sal.value;

        deltas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eligibility_accumulates() {
        let mut state = EligibilityState::new(2, 3);
        let pre = vec![1.0, 0.5, 0.0];
        let post = vec![1.0, -0.5];
        let motor = vec![0.5, 0.5];

        // Step with high reward
        let _deltas = state.step(&pre, &post, 2.0, &motor, 0.97, 0.01, 0.1);
        // Trace should be nonzero
        assert!(state.trace.iter().any(|&t| t.abs() > 0.01));
    }

    #[test]
    fn no_update_below_threshold() {
        let mut state = EligibilityState::new(2, 3);
        let pre = vec![1.0, 1.0, 1.0];
        let post = vec![1.0, 1.0];
        let motor = vec![0.9, 0.1]; // clear action = low salience

        // Reward ≈ baseline → advantage ≈ 0 → no update
        let deltas = state.step(&pre, &post, 1.0, &motor, 0.97, 0.01, 0.1);
        assert!(deltas.iter().all(|&d| d.abs() < 1e-6));
    }
}
