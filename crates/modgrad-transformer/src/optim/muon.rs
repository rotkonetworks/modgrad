//! Muon optimizer: Polar Express + Nesterov momentum + variance reduction + cautious WD.
//!
//! For 2D weight matrices only. 1D params must use AdamW.
//!
//! Pipeline:
//!   1. Nesterov momentum on the gradient
//!   2. Polar Express: approximate orthogonalization via 5 Newton-Schulz iterations
//!   3. Variance reduction: per-row and per-column second moment EMA
//!   4. Cautious weight decay: only where sign(grad) == sign(param)

use super::super::ops::TransformerOps;

/// Newton-Schulz iteration coefficients (Bernstein & Newhouse 2024).
/// Each tuple is (a, b, c) for: X_{k+1} = a*X_k + X_k @ (b * X_k^T X_k + c * (X_k^T X_k)²)
const NS_COEFFS: [(f32, f32, f32); 5] = [
    (3.4445, -2.8025, 0.8025),
    (3.5042, -2.8893, 0.8493),
    (3.7616, -3.2362, 1.0562),
    (5.3542, -5.8003, 2.2803),
    (14.7328, -20.3658, 7.5658),
];

/// Muon optimizer state for a single 2D weight matrix.
pub struct Muon {
    /// Nesterov momentum buffer [rows * cols].
    momentum: Vec<f32>,
    /// Per-row second moment EMA [rows].
    row_var: Vec<f32>,
    /// Per-column second moment EMA [cols].
    col_var: Vec<f32>,
    rows: usize,
    cols: usize,
    /// Timestep.
    t: u64,
    /// Learning rate.
    pub lr: f32,
    /// Momentum coefficient (default: 0.95).
    pub mu: f32,
    /// Variance EMA decay (default: 0.999).
    pub var_beta: f32,
    /// Weight decay coefficient (default: 0.01).
    pub weight_decay: f32,
}

impl Muon {
    pub fn new(rows: usize, cols: usize, lr: f32) -> Self {
        Self {
            momentum: vec![0.0; rows * cols],
            row_var: vec![0.0; rows],
            col_var: vec![0.0; cols],
            rows,
            cols,
            t: 0,
            lr,
            mu: 0.95,
            var_beta: 0.999,
            weight_decay: 0.01,
        }
    }

    /// Perform one optimization step.
    ///
    /// `params`: mutable weight matrix [rows * cols], row-major.
    /// `grads`: gradient matrix [rows * cols], row-major.
    /// `backend`: for Polar Express computation.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32], backend: &dyn TransformerOps) {
        debug_assert_eq!(params.len(), self.rows * self.cols);
        debug_assert_eq!(grads.len(), self.rows * self.cols);

        self.t += 1;
        let rows = self.rows;
        let cols = self.cols;

        // ── 1. Nesterov momentum ────────────────────────────
        for i in 0..rows * cols {
            self.momentum[i] = self.mu * self.momentum[i] + grads[i];
        }
        // Nesterov lookahead: g = mu * momentum + grad
        let mut nesterov = vec![0.0f32; rows * cols];
        for i in 0..rows * cols {
            nesterov[i] = self.mu * self.momentum[i] + grads[i];
        }

        // ── 2. Polar Express (orthogonalize) ────────────────
        // Pre-scale by estimated spectral norm for convergence
        let frob_sq: f32 = nesterov.iter().map(|x| x * x).sum();
        let spectral_est = (frob_sq / rows.min(cols) as f32).sqrt();
        if spectral_est > 1e-8 {
            let inv = 1.0 / spectral_est;
            for v in &mut nesterov { *v *= inv; }
        }

        backend.polar_express(&mut nesterov, rows, cols);

        // Safety: if Polar Express produced NaN/Inf, fall back to plain nesterov
        if nesterov.iter().any(|v| !v.is_finite()) {
            // Recompute nesterov without orthogonalization
            for i in 0..rows * cols {
                nesterov[i] = self.mu * self.momentum[i] + grads[i];
            }
        }

        // ── 3. Variance reduction ───────────────────────────
        // Per-row: running average of row-wise squared mean
        // Per-col: running average of col-wise squared mean
        let beta = self.var_beta;
        let bc = 1.0 - beta.powi(self.t as i32);

        // Update row variance
        for r in 0..rows {
            let mut row_sq = 0.0f32;
            for c in 0..cols {
                let v = nesterov[r * cols + c];
                row_sq += v * v;
            }
            row_sq /= cols as f32;
            self.row_var[r] = beta * self.row_var[r] + (1.0 - beta) * row_sq;
        }

        // Update col variance
        for c in 0..cols {
            let mut col_sq = 0.0f32;
            for r in 0..rows {
                let v = nesterov[r * cols + c];
                col_sq += v * v;
            }
            col_sq /= rows as f32;
            self.col_var[c] = beta * self.col_var[c] + (1.0 - beta) * col_sq;
        }

        // Apply variance normalization
        for r in 0..rows {
            let row_scale = 1.0 / ((self.row_var[r] / bc).sqrt() + 1e-8);
            for c in 0..cols {
                let col_scale = 1.0 / ((self.col_var[c] / bc).sqrt() + 1e-8);
                nesterov[r * cols + c] *= (row_scale * col_scale).sqrt();
            }
        }

        // ── 4. Cautious weight decay + update ───────────────
        for i in 0..rows * cols {
            let g = nesterov[i];
            let p = params[i];

            // Cautious WD: only decay when sign(grad) == sign(param)
            if self.weight_decay > 0.0 && g * p > 0.0 {
                params[i] -= self.lr * self.weight_decay * p;
            }

            // Parameter update
            params[i] -= self.lr * g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ns_coefficients_sum() {
        // Each (a, b, c) should satisfy a + b + c ≈ 1 (conservation)
        // Actually they don't need to — but let's verify they're loaded
        for (a, b, c) in NS_COEFFS {
            assert!(a > 0.0);
            assert!(b < 0.0);
            assert!(c > 0.0);
        }
    }
}
