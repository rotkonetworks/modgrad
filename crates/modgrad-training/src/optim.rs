//! Optimizers — pure functions from (weights, gradients, state) → (weights, state).
//!
//! Each optimizer is a stateless type (ZST).
//! The mutable state (momentum, second moments) is explicit — passed in,
//! returned out. Learning rate comes from the Scheduler, not the optimizer.
//!
//! The training loop threads state through:
//!   let lr = scheduler.get_lr(step);
//!   opt_state = Adam::step(&mut weights, &grads, opt_state, lr, clip);

use serde::{Deserialize, Serialize};

// ─── Optimizer trait ──────────────────────────────────────────

/// A weight update rule: (weights, gradients, state, lr) → state.
///
/// Weights are mutated in-place (performance — no allocation per step).
/// State is explicit: passed in and returned, never hidden behind &mut self.
/// Learning rate comes from the Scheduler (separation of concerns).
pub trait Optimizer {
    /// Optimizer-specific state (momentum, second moments, step count).
    type State;

    /// Create initial state for n parameters.
    fn init(n_params: usize, config: &OptimizerConfig) -> Self::State;

    /// One optimization step. Mutates weights, returns new state.
    fn step(
        weights: &mut [f32],
        grads: &[f32],
        state: Self::State,
        lr: f32,
        grad_clip: f32,
    ) -> Self::State;
}

/// Optimizer configuration (shared across all optimizer types).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub lr: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub grad_clip: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_clip: 1.0,
        }
    }
}

// ─── Gradient clipping (pure function) ────────────────────────

/// Compute gradient norm and clipping scale factor.
/// Pure function: grads in, scale out.
#[inline]
pub fn clip_scale(grads: &[f32], max_norm: f32) -> f32 {
    if max_norm <= 0.0 { return 1.0; }
    let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max_norm { max_norm / norm } else { 1.0 }
}

// ─── SGD ───────────────────────────────────────────────────

/// SGD with optional weight decay. Zero state.
pub struct Sgd;

impl Optimizer for Sgd {
    type State = ();

    fn init(_n_params: usize, _config: &OptimizerConfig) -> () {}

    fn step(
        weights: &mut [f32], grads: &[f32], _state: (),
        lr: f32, grad_clip: f32,
    ) -> () {
        let scale = clip_scale(grads, grad_clip);
        for (w, g) in weights.iter_mut().zip(grads) {
            *w -= lr * g * scale;
        }
    }
}

// ─── Adam ──────────────────────────────────────────────────

/// Adam optimizer state — explicit, serializable, inspectable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamState {
    pub m: Vec<f32>,  // first moment
    pub v: Vec<f32>,  // second moment
    pub t: usize,     // step count
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

/// Adam optimizer (Kingma & Ba 2014). Zero-sized type.
pub struct Adam;

impl Optimizer for Adam {
    type State = AdamState;

    fn init(n_params: usize, config: &OptimizerConfig) -> AdamState {
        AdamState {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
        }
    }

    fn step(
        weights: &mut [f32], grads: &[f32], mut state: AdamState,
        lr: f32, grad_clip: f32,
    ) -> AdamState {
        state.t += 1;
        let b1 = state.beta1;
        let b2 = state.beta2;
        let eps = state.eps;
        let bc1 = 1.0 - b1.powi(state.t as i32);
        let bc2 = 1.0 - b2.powi(state.t as i32);
        let scale = clip_scale(grads, grad_clip);

        for i in 0..weights.len() {
            let g = grads[i] * scale;
            state.m[i] = b1 * state.m[i] + (1.0 - b1) * g;
            state.v[i] = b2 * state.v[i] + (1.0 - b2) * g * g;
            let m_hat = state.m[i] / bc1;
            let v_hat = state.v[i] / bc2;
            weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        state
    }
}

// ─── AdamW ─────────────────────────────────────────────────

/// AdamW: decoupled weight decay (Loshchilov & Hutter 2017). Zero-sized type.
pub struct AdamW;

/// AdamW state — same structure as Adam, plus weight decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWState {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub t: usize,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Optimizer for AdamW {
    type State = AdamWState;

    fn init(n_params: usize, config: &OptimizerConfig) -> AdamWState {
        AdamWState {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            weight_decay: config.weight_decay,
        }
    }

    fn step(
        weights: &mut [f32], grads: &[f32], mut state: AdamWState,
        lr: f32, grad_clip: f32,
    ) -> AdamWState {
        state.t += 1;
        let b1 = state.beta1;
        let b2 = state.beta2;
        let eps = state.eps;
        let wd = state.weight_decay;
        let bc1 = 1.0 - b1.powi(state.t as i32);
        let bc2 = 1.0 - b2.powi(state.t as i32);
        let scale = clip_scale(grads, grad_clip);

        for i in 0..weights.len() {
            let g = grads[i] * scale;
            state.m[i] = b1 * state.m[i] + (1.0 - b1) * g;
            state.v[i] = b2 * state.v[i] + (1.0 - b2) * g * g;
            let m_hat = state.m[i] / bc1;
            let v_hat = state.v[i] / bc2;
            weights[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * weights[i]);
        }
        state
    }
}

// ─── Scheduler ─────────────────────────────────────────────
// Pure function: step → lr.

/// Learning rate scheduler: step → lr.
pub trait Scheduler {
    fn get_lr(&self, step: usize) -> f32;
}

/// Constant learning rate.
pub struct ConstantLR { pub lr: f32 }
impl Scheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f32 { self.lr }
}

/// Linear warmup then constant.
pub struct LinearWarmup {
    pub base_lr: f32,
    pub warmup_steps: usize,
}
impl Scheduler for LinearWarmup {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            self.base_lr
        }
    }
}

/// Cosine annealing with warmup.
pub struct WarmupCosine {
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}
impl Scheduler for WarmupCosine {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            let cosine = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
            self.min_lr + (self.base_lr - self.min_lr) * cosine
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_step() {
        let mut w = vec![1.0, 2.0, 3.0];
        let g = vec![0.1, 0.2, 0.3];
        Sgd::step(&mut w, &g, (), 0.1, 0.0);
        assert!((w[0] - 0.99).abs() < 1e-6);
        assert!((w[1] - 1.98).abs() < 1e-6);
    }

    #[test]
    fn adam_converges() {
        let cfg = OptimizerConfig { lr: 0.1, grad_clip: 0.0, ..Default::default() };
        let mut state = Adam::init(1, &cfg);
        let mut w = vec![5.0];
        for _ in 0..100 {
            let g = vec![w[0]]; // gradient = w (minimize w²/2)
            state = Adam::step(&mut w, &g, state, 0.1, 0.0);
        }
        assert!(w[0].abs() < 0.1, "Adam should converge near 0, got {}", w[0]);
    }

    #[test]
    fn adam_state_is_explicit() {
        let cfg = OptimizerConfig::default();
        let state = Adam::init(3, &cfg);
        assert_eq!(state.t, 0);
        assert_eq!(state.m.len(), 3);
        // State is inspectable, serializable, checkpointable
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("\"t\":0"));
    }

    #[test]
    fn warmup_cosine_schedule() {
        let sched = WarmupCosine { base_lr: 1e-3, min_lr: 1e-5, warmup_steps: 100, total_steps: 1000 };
        assert!((sched.get_lr(0) - 1e-5).abs() < 1e-5);
        assert!((sched.get_lr(100) - 1e-3).abs() < 1e-5);
        assert!(sched.get_lr(500) < 1e-3);
        assert!(sched.get_lr(999) < sched.get_lr(500));
    }
}
