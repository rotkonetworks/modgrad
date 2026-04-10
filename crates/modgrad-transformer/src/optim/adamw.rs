//! Standard AdamW optimizer for embeddings and scalar parameters.
//!
//! Used for: token embeddings, norm scales, residual lambdas, VE tables.
//! NOT used for weight matrices — those use Muon.

/// AdamW state for a parameter group.
pub struct AdamW {
    /// First moment estimates (per-element).
    m: Vec<f32>,
    /// Second moment estimates (per-element).
    v: Vec<f32>,
    /// Timestep counter.
    t: u64,
    /// Learning rate.
    pub lr: f32,
    /// First moment decay (default: 0.9).
    pub beta1: f32,
    /// Second moment decay (default: 0.999).
    pub beta2: f32,
    /// Numerical stability epsilon (default: 1e-8).
    pub eps: f32,
    /// Weight decay coefficient (default: 0.01).
    pub weight_decay: f32,
}

impl AdamW {
    pub fn new(num_params: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Perform one optimization step.
    ///
    /// `params`: mutable parameter slice.
    /// `grads`: gradient slice (same length as params).
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        debug_assert_eq!(params.len(), grads.len());
        debug_assert_eq!(params.len(), self.m.len());

        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            let g = grads[i];

            // Moment updates
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // Bias correction
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            // Weight decay (decoupled)
            params[i] -= self.lr * self.weight_decay * params[i];

            // Adam update
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    /// Reset state (for learning rate warmup restarts, etc).
    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.t = 0;
    }
}
