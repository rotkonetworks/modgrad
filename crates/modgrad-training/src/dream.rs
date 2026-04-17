//! Dream/sleep consolidation — a StepHook for memory replay.
//!
//! Every `interval` steps, replays a random data position through the model
//! in free-running mode (model uses its own predictions as input). The
//! resulting gradient is scaled by `weight` to avoid overwhelming the
//! supervised signal.
//!
//! Analogous to hippocampal replay during sleep. Prevents loss drift when
//! the model encounters new data regions by reinforcing earlier learning.
//!
//! This is generic infrastructure — it takes a replay closure, not a
//! specific model type. The caller provides the dream function; this
//! module handles scheduling and gradient scaling.

use super::trainer::StepHook;

/// Dream/sleep consolidation hook.
///
/// `F`: closure that runs one dream episode given (weights, lr_scaled).
/// The closure is responsible for computing and applying dream gradients.
/// This hook only handles scheduling (when to dream).
pub struct DreamHook<W, F: FnMut(&mut W, f32)> {
    /// Run dream every N training steps.
    pub interval: usize,
    /// Gradient scaling factor (e.g., 0.3 = 30% weight vs supervised).
    pub weight: f32,
    /// The dream function: (weights, scaled_lr) → applies dream gradients.
    pub dream_fn: F,
    _phantom: std::marker::PhantomData<W>,
}

impl<W, F: FnMut(&mut W, f32)> DreamHook<W, F> {
    pub fn new(interval: usize, weight: f32, dream_fn: F) -> Self {
        Self {
            interval,
            weight,
            dream_fn,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<W, F: FnMut(&mut W, f32)> StepHook<W> for DreamHook<W, F> {
    fn after_step(&mut self, weights: &mut W, step: usize, lr: f32) {
        if step > 0 && step % self.interval == 0 {
            (self.dream_fn)(weights, lr * self.weight);
        }
    }
}
