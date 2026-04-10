//! Gradient accumulation: aggregate gradients over N micro-batches
//! before taking an optimizer step. Enables larger effective batch
//! sizes on limited memory.
//!
//! Usage:
//!   let mut accum = GradAccumulator::new(n_params, accumulation_steps);
//!   for micro_batch in batch.chunks(micro_batch_size) {
//!       let grads = compute_gradients(micro_batch);
//!       if accum.accumulate(&grads) {
//!           // Accumulated enough — take optimizer step
//!           optimizer.step(&mut weights, accum.get());
//!           accum.zero();
//!       }
//!   }

/// Accumulates gradients over multiple micro-batches.
pub struct GradAccumulator {
    grads: Vec<f32>,
    steps: usize,
    accumulation_steps: usize,
    current_step: usize,
}

impl GradAccumulator {
    /// Create a new accumulator.
    /// `n_params`: number of trainable parameters.
    /// `accumulation_steps`: how many micro-batches to accumulate before stepping.
    pub fn new(n_params: usize, accumulation_steps: usize) -> Self {
        Self {
            grads: vec![0.0; n_params],
            steps: accumulation_steps.max(1),
            accumulation_steps: accumulation_steps.max(1),
            current_step: 0,
        }
    }

    /// Add gradients from one micro-batch. Returns true when accumulation
    /// is complete and optimizer should step.
    pub fn accumulate(&mut self, micro_grads: &[f32]) -> bool {
        assert_eq!(micro_grads.len(), self.grads.len(),
            "GradAccumulator: grad length mismatch {} vs {}", micro_grads.len(), self.grads.len());

        // Accumulate with averaging
        let scale = 1.0 / self.steps as f32;
        for (acc, &g) in self.grads.iter_mut().zip(micro_grads) {
            *acc += g * scale;
        }

        self.current_step += 1;
        self.current_step >= self.accumulation_steps
    }

    /// Get the accumulated (averaged) gradients.
    pub fn get(&self) -> &[f32] { &self.grads }

    /// Reset for next accumulation cycle.
    pub fn zero(&mut self) {
        self.grads.fill(0.0);
        self.current_step = 0;
    }

    /// How many micro-batches have been accumulated so far.
    pub fn current(&self) -> usize { self.current_step }

    /// Whether accumulation is complete.
    pub fn is_ready(&self) -> bool { self.current_step >= self.accumulation_steps }

    /// Number of parameters.
    pub fn n_params(&self) -> usize { self.grads.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_and_averages() {
        let mut acc = GradAccumulator::new(3, 4);

        // 4 micro-batches, each with gradient [1, 2, 3]
        assert!(!acc.accumulate(&[1.0, 2.0, 3.0]));
        assert!(!acc.accumulate(&[1.0, 2.0, 3.0]));
        assert!(!acc.accumulate(&[1.0, 2.0, 3.0]));
        assert!(acc.accumulate(&[1.0, 2.0, 3.0])); // 4th triggers ready

        // Averaged: each grad = original * (1/4) * 4 = original
        let grads = acc.get();
        assert!((grads[0] - 1.0).abs() < 1e-6);
        assert!((grads[1] - 2.0).abs() < 1e-6);
        assert!((grads[2] - 3.0).abs() < 1e-6);

        acc.zero();
        assert_eq!(acc.current(), 0);
        assert!((acc.get()[0]).abs() < 1e-10);
    }
}
