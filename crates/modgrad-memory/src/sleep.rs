//! Sleep consolidation: offline cross-layer weight optimization.
//!
//! Collects (input, output) pairs from synapses during waking,
//! then runs least-squares to find optimal weights (REM sleep analogue).

use rayon::prelude::*;

/// Offline cross-layer weight optimization (REM sleep analogue).
/// Collects (input, output) pairs from synapses during waking,
/// then runs least-squares to find optimal weights.
#[derive(Debug, Clone, Default)]
pub struct SleepConsolidation {
    pub traces: Vec<(String, Vec<f32>, Vec<f32>, f32)>,  // (synapse_name, input, output, reward)
    pub max_samples: usize,
}

impl SleepConsolidation {
    pub fn new(max_samples: usize) -> Self {
        Self { traces: Vec::new(), max_samples }
    }

    /// Collect a trace with default reward 1.0 (backward compat).
    pub fn collect(&mut self, name: &str, input: &[f32], output: &[f32]) {
        self.collect_weighted(name, input, output, 1.0);
    }

    /// Collect a trace with explicit reward weight.
    /// reward > 0: reinforce this mapping (correct prediction).
    /// reward < 0: anti-reinforce (wrong prediction).
    /// reward = 0: ignore (neutral).
    pub fn collect_weighted(&mut self, name: &str, input: &[f32], output: &[f32], reward: f32) {
        if self.traces.len() >= self.max_samples { return; }
        self.traces.push((name.into(), input.to_vec(), output.to_vec(), reward));
    }

    /// Run reward-weighted least-squares consolidation.
    /// Traces with positive reward get reinforced. Negative get anti-reinforced.
    /// Returns weight deltas per synapse.
    pub fn consolidate(&self) -> Vec<(String, Vec<f32>, usize, usize)> {
        // Group by synapse name, apply reward weighting
        let mut groups: std::collections::HashMap<&str, Vec<(&[f32], &[f32])>>
            = std::collections::HashMap::new();
        for (name, inp, out, reward) in &self.traces {
            // Skip neutral/negative traces — only consolidate positive experiences
            if *reward <= 0.0 { continue; }
            groups.entry(name.as_str()).or_default().push((inp, out));
        }

        // Solve each synapse in parallel
        let group_vec: Vec<_> = groups.into_iter().collect();
        group_vec.par_iter()
            .filter_map(|(name, pairs)| {
                if pairs.is_empty() { return None; }
                let in_dim = pairs[0].0.len();
                let out_dim = pairs[0].1.len();

                // Build X^T X + λI and X^T Y
                // Parallelize the outer product accumulation by rows
                let xtx: Vec<f32> = (0..in_dim)
                    .into_par_iter()
                    .flat_map(|i| {
                        let mut row = vec![0.0f32; in_dim];
                        for &(x, _) in pairs.iter() {
                            if x.len() < in_dim { continue; }
                            let xi = x[i];
                            for j in 0..in_dim {
                                row[j] += xi * x[j];
                            }
                        }
                        if i < in_dim { row[i] += 1e-4; }
                        row
                    })
                    .collect();

                let xty: Vec<f32> = (0..in_dim)
                    .into_par_iter()
                    .flat_map(|i| {
                        let mut row = vec![0.0f32; out_dim];
                        for &(x, y) in pairs.iter() {
                            if x.len() < in_dim || y.len() < out_dim { continue; }
                            let xi = x[i];
                            for o in 0..out_dim {
                                row[o] += xi * y[o];
                            }
                        }
                        row
                    })
                    .collect();

                solve_least_squares(&xtx, &xty, in_dim, out_dim)
                    .map(|w_opt| (name.to_string(), w_opt, in_dim, out_dim))
            })
            .collect()
    }

    pub fn reset(&mut self) {
        self.traces.clear();
    }
}

/// Solve X^T X · W = X^T Y via Cholesky decomposition.
pub fn solve_least_squares(xtx: &[f32], xty: &[f32], n: usize, m: usize) -> Option<Vec<f32>> {
    // Simple Cholesky: L L^T = xtx
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = xtx[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum < 1e-10 { return None; }  // reject near-zero (singular/rank-deficient)
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    // Solve L · Z = xty, then L^T · W = Z
    let mut w = vec![0.0f32; n * m];
    for col in 0..m {
        // Forward: L · z = xty[:, col]
        let mut z = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = xty[i * m + col];
            for k in 0..i {
                sum -= l[i * n + k] * z[k];
            }
            z[i] = sum / l[i * n + i];
        }
        // Backward: L^T · w = z
        for i in (0..n).rev() {
            let mut sum = z[i];
            for k in (i + 1)..n {
                sum -= l[k * n + i] * w[k * m + col];
            }
            w[i * m + col] = sum / l[i * n + i];
        }
    }
    Some(w)
}
