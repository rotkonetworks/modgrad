//! Optimized linear algebra primitives for sleep consolidation.
//!
//! These are the hot paths during least-squares solving:
//! - Outer product accumulation (XTX, XTY building)
//! - Cholesky decomposition
//! - Triangular solves
//!
//! All use 16-wide AVX-512 friendly patterns and rayon parallelism.

use rayon::prelude::*;

/// Accumulate X^T X from a batch of vectors.
/// xtx is [n × n] row-major, updated in-place.
/// Each x is [n] — one data point.
///
/// Parallelized: each row of XTX is computed independently.
pub fn accumulate_xtx(xtx: &mut [f32], xs: &[&[f32]], n: usize) {
    if xs.is_empty() { return; }

    // Parallel over rows of XTX
    let row_results: Vec<Vec<f32>> = (0..n).into_par_iter().map(|i| {
        let mut row = vec![0.0f32; n];
        for x in xs {
            if x.len() < n { continue; } // skip mismatched traces
            let xi = x[i];
            for j in 0..n {
                row[j] += xi * x[j];
            }
        }
        row
    }).collect();

    // Copy results into xtx
    for (i, row) in row_results.iter().enumerate() {
        xtx[i * n..(i + 1) * n].copy_from_slice(row);
    }
}

/// Accumulate X^T Y from batches of (x, y) pairs.
/// xty is [n × m] row-major, updated in-place.
/// Each x is [n], each y is [m].
///
/// Parallelized: each row of XTY is computed independently.
pub fn accumulate_xty(xty: &mut [f32], xs: &[&[f32]], ys: &[&[f32]], n: usize, m: usize) {
    if xs.is_empty() { return; }

    let row_results: Vec<Vec<f32>> = (0..n).into_par_iter().map(|i| {
        let mut row = vec![0.0f32; m];
        for (x, y) in xs.iter().zip(ys.iter()) {
            if x.len() < n || y.len() < m { continue; }
            let xi = x[i];
            for o in 0..m {
                row[o] += xi * y[o];
            }
        }
        row
    }).collect();

    for (i, row) in row_results.iter().enumerate() {
        xty[i * m..(i + 1) * m].copy_from_slice(row);
    }
}

/// Cholesky decomposition: A = L L^T where A is [n × n] symmetric positive definite.
/// Returns L as [n × n] lower triangular, or None if not positive definite.
pub fn cholesky(a: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut l = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];

            // Subtract L[i][k] * L[j][k] for k < j
            // This inner loop is the hot path for large n
            let li = &l[i * n..];
            let lj = &l[j * n..];
            for k in 0..j {
                sum -= li[k] * lj[k];
            }

            if i == j {
                if sum < 1e-10 { return None; }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    Some(l)
}

/// Solve L z = b (forward substitution) where L is [n × n] lower triangular.
pub fn forward_solve(l: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut z = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = b[i];
        let li = &l[i * n..];
        for k in 0..i {
            sum -= li[k] * z[k];
        }
        z[i] = sum / li[i];
    }
    z
}

/// Solve L^T x = z (backward substitution) where L is [n × n] lower triangular.
pub fn backward_solve(l: &[f32], z: &[f32], n: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    x
}

/// Full least-squares solve: W_opt = (X^T X + λI)^{-1} X^T Y
/// xs: [num_samples][n], ys: [num_samples][m]
/// Returns W_opt as [n × m] row-major, or None if singular.
///
/// Uses parallel XTX/XTY accumulation + Cholesky solve.
pub fn least_squares(
    xs: &[&[f32]],
    ys: &[&[f32]],
    n: usize,
    m: usize,
    lambda: f32,
) -> Option<Vec<f32>> {
    if xs.is_empty() || n == 0 || m == 0 { return None; }

    // Build X^T X + λI
    let mut xtx = vec![0.0f32; n * n];
    accumulate_xtx(&mut xtx, xs, n);
    for i in 0..n {
        xtx[i * n + i] += lambda;
    }

    // Build X^T Y
    let mut xty = vec![0.0f32; n * m];
    accumulate_xty(&mut xty, xs, ys, n, m);

    // Cholesky
    let l = cholesky(&xtx, n)?;

    // Solve for each column of Y independently (can parallelize for large m)
    // Solve for each column of Y: (X^T X)^{-1} X^T Y[:,col]
    // Result: W[n × m] row-major, W[i * m + col] = weight from input i to output col
    let w: Vec<f32> = if m >= 16 {
        // Parallel: solve each column independently, then transpose into [n × m]
        let col_solutions: Vec<Vec<f32>> = (0..m).into_par_iter().map(|col| {
            let b: Vec<f32> = (0..n).map(|i| xty[i * m + col]).collect();
            let z = forward_solve(&l, &b, n);
            backward_solve(&l, &z, n) // returns [n] for this column
        }).collect();
        // Transpose [m][n] → [n × m] row-major
        let mut w = vec![0.0f32; n * m];
        for col in 0..m {
            for i in 0..n {
                w[i * m + col] = col_solutions[col][i];
            }
        }
        w
    } else {
        let mut w = vec![0.0f32; n * m];
        for col in 0..m {
            let b: Vec<f32> = (0..n).map(|i| xty[i * m + col]).collect();
            let z = forward_solve(&l, &b, n);
            let x = backward_solve(&l, &z, n);
            for i in 0..n {
                w[i * m + col] = x[i];
            }
        }
        w
    };

    Some(w)
}

/// Compute the Angeris bound: what fraction of Y is explained by W_opt · X?
/// Returns (residual_fraction, total_energy).
pub fn angeris_residual(
    xs: &[&[f32]],
    ys: &[&[f32]],
    w: &[f32],
    n: usize,
    m: usize,
) -> (f32, f32) {
    let mut total = 0.0f32;
    let mut residual = 0.0f32;

    for (x, y) in xs.iter().zip(ys.iter()) {
        for o in 0..m {
            total += y[o] * y[o];
            let pred: f32 = (0..n).map(|i| w[i * m + o] * x[i]).sum();
            residual += (y[o] - pred).powi(2);
        }
    }

    let frac = if total > 1e-10 { residual / total } else { 1.0 };
    (frac, total)
}

/// Scalarized Angeris bound (from addendum §3.1, March 2026).
///
/// For each random test direction y, measures the tightest bound on
/// how well the representations X can explain the targets Y.
/// Tighter than LS bound (no matrix inversion, no conditioning issues),
/// cheaper than SDP (~10ms vs minutes).
///
/// Returns (mean_violation, max_violation, n_violated) where violation = 0
/// means the representation is sufficient, violation > 0 means information loss.
///
/// The bound: for all y, |y^T (target - best_linear·input)|² ≤ Σ variance_in_direction_y
/// Sampled over n_directions random y vectors.
pub fn angeris_scalarized(
    xs: &[&[f32]],     // [n_samples][in_dim] — input representations
    ys: &[&[f32]],     // [n_samples][out_dim] — targets
    in_dim: usize,
    out_dim: usize,
    n_directions: usize,
) -> (f32, f32, usize) {
    if xs.is_empty() || ys.is_empty() { return (0.0, 0.0, 0); }
    let n = xs.len();

    // Simple PRNG for random directions (deterministic, reproducible)
    let mut rng = 42u64;
    let mut next_f32 = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };

    let mut total_violation = 0.0f32;
    let mut max_violation = 0.0f32;
    let mut n_violated = 0usize;

    for _ in 0..n_directions {
        // Random direction in output space
        let y: Vec<f32> = (0..out_dim).map(|_| next_f32()).collect();
        let y_norm: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);

        // Project targets and inputs into this direction
        // target_proj[i] = y^T · target[i] (scalar per sample)
        // input_proj[i][j] = x[i][j] (keep all input dims for variance)
        let target_proj: Vec<f32> = ys.iter().map(|yi| {
            yi.iter().zip(&y).map(|(a, b)| a * b).sum::<f32>() / y_norm
        }).collect();

        // Compute: can the inputs explain the target projections?
        // Variance of target in this direction
        let target_mean: f32 = target_proj.iter().sum::<f32>() / n as f32;
        let target_var: f32 = target_proj.iter()
            .map(|t| (t - target_mean).powi(2)).sum::<f32>() / n as f32;

        // Best linear prediction variance: project X onto direction of max correlation with target_proj
        // This is |corr(X, target_proj)|² × var(target_proj)
        let mut explained = 0.0f32;
        for j in 0..in_dim {
            let xj_mean: f32 = xs.iter().map(|x| x[j]).sum::<f32>() / n as f32;
            let xj_var: f32 = xs.iter().map(|x| (x[j] - xj_mean).powi(2)).sum::<f32>() / n as f32;
            if xj_var < 1e-10 { continue; }
            let cov: f32 = xs.iter().zip(&target_proj)
                .map(|(x, t)| (x[j] - xj_mean) * (t - target_mean)).sum::<f32>() / n as f32;
            explained += cov * cov / xj_var;
        }

        // Violation: how much target variance is NOT explained
        let unexplained = (target_var - explained).max(0.0);
        let violation = if target_var > 1e-10 { unexplained / target_var } else { 0.0 };

        if violation > 0.01 { // >1% unexplained
            n_violated += 1;
            total_violation += violation;
            max_violation = max_violation.max(violation);
        }
    }

    let mean_violation = if n_violated > 0 { total_violation / n_violated as f32 } else { 0.0 };
    (mean_violation, max_violation, n_violated)
}

/// Element-wise blend: dst = (1 - alpha) * dst + alpha * src
/// SIMD-friendly: compiler will auto-vectorize this.
#[inline]
pub fn blend(dst: &mut [f32], src: &[f32], alpha: f32) {
    let one_minus = 1.0 - alpha;
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = one_minus * *d + alpha * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_least_squares_simple() {
        // y = 2x + 1
        let xs: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let ys: Vec<Vec<f32>> = vec![vec![3.0], vec![5.0], vec![7.0], vec![9.0]];
        let x_refs: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
        let y_refs: Vec<&[f32]> = ys.iter().map(|v| v.as_slice()).collect();

        let w = least_squares(&x_refs, &y_refs, 1, 1, 1e-4);
        assert!(w.is_some(), "solve failed");
        let w = w.unwrap();
        assert_eq!(w.len(), 1, "wrong dims: {}", w.len());
        // w ≈ 2.0 (slope) — but note: no intercept term, so fit is y = wx
        // Best fit of y = wx through (1,3),(2,5),(3,7),(4,9):
        // w = (1*3 + 2*5 + 3*7 + 4*9) / (1 + 4 + 9 + 16) = (3+10+21+36)/30 = 70/30 ≈ 2.33
        eprintln!("w = {}", w[0]);
        assert!((w[0] - 2.33).abs() < 0.1, "w={}", w[0]);
    }

    #[test]
    fn test_cholesky() {
        // 2×2 positive definite
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2).unwrap();
        // L[0][0] = sqrt(4) = 2
        assert!((l[0] - 2.0).abs() < 1e-6);
        // L[1][0] = 2/2 = 1
        assert!((l[2] - 1.0).abs() < 1e-6);
        // L[1][1] = sqrt(3 - 1) = sqrt(2)
        assert!((l[3] - 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_blend() {
        let mut dst = vec![1.0, 2.0, 3.0, 4.0];
        let src = vec![5.0, 6.0, 7.0, 8.0];
        blend(&mut dst, &src, 0.5);
        assert!((dst[0] - 3.0).abs() < 1e-6);
        assert!((dst[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_angeris_residual() {
        // Perfect fit: y = 2x, W = [2]
        let xs: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0]];
        let ys: Vec<Vec<f32>> = vec![vec![2.0], vec![4.0], vec![6.0]];
        let x_refs: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
        let y_refs: Vec<&[f32]> = ys.iter().map(|v| v.as_slice()).collect();
        let w = vec![2.0];
        let (frac, _) = angeris_residual(&x_refs, &y_refs, &w, 1, 1);
        assert!(frac < 1e-6, "should be perfect fit: {frac}");
    }
}
