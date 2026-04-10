//! TransformerOps trait — transformer-specific compute operations.
//!
//! Separate from ComputeBackend to avoid polluting the base trait with
//! architecture-specific methods. Explicitly implemented per-backend
//! so GPU backends can override individual methods.

/// Transformer-specific compute operations.
///
/// Explicitly implemented for CpuBackend and HsaBackend.
/// GPU backends override methods with dispatch to compute shaders.
pub trait TransformerOps: Send + Sync {
    /// RMS normalization: dst[i] = src[i] / sqrt(mean(src²) + eps)
    fn rms_norm(&self, src: &[f32], dst: &mut [f32], eps: f32);

    /// Bias-free matrix-vector multiply: y = W * x
    /// W is [out_dim × in_dim] row-major.
    fn matvec_nobias(&self, weight: &[f32], x: &[f32],
                     y: &mut [f32], out_dim: usize, in_dim: usize);

    /// Softmax in-place over a slice.
    fn softmax_inplace(&self, x: &mut [f32]);

    /// ReLU² activation: y = max(0, x)²
    fn relu_squared_inplace(&self, x: &mut [f32]);

    /// Polar Express: approximate matrix sign function via Newton-Schulz iteration.
    ///
    /// Input: arbitrary matrix X (flattened row-major, [rows × cols]).
    /// Output: orthogonal approximation in-place.
    ///
    /// 5 iterations with pre-defined coefficient tuples.
    /// Caller must pre-scale X so spectral norm ≈ 1.
    fn polar_express(&self, x: &mut [f32], rows: usize, cols: usize);
}

// ─── CPU default implementations as free functions ──────────

pub(crate) mod cpu {
    use modgrad_compute::backend::dot;

    pub fn rms_norm(src: &[f32], dst: &mut [f32], eps: f32) {
        let n = src.len() as f32;
        let ss: f32 = src.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (ss / n + eps).sqrt();
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = s * inv_rms;
        }
    }

    pub fn matvec_nobias(weight: &[f32], x: &[f32],
                         y: &mut [f32], out_dim: usize, in_dim: usize) {
        debug_assert_eq!(weight.len(), out_dim * in_dim);
        debug_assert!(x.len() >= in_dim);
        debug_assert!(y.len() >= out_dim);
        for r in 0..out_dim {
            let row = &weight[r * in_dim..(r + 1) * in_dim];
            y[r] = dot(row, &x[..in_dim]);
        }
    }

    pub fn softmax_inplace(x: &mut [f32]) {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in x.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }

    pub fn relu_squared_inplace(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = if *v > 0.0 { *v * *v } else { 0.0 };
        }
    }

    pub fn polar_express(x: &mut [f32], rows: usize, cols: usize) {
        debug_assert_eq!(x.len(), rows * cols);

        const COEFFS: [(f32, f32, f32); 5] = [
            (3.4445, -2.8025, 0.8025),
            (3.5042, -2.8893, 0.8493),
            (3.7616, -3.2362, 1.0562),
            (5.3542, -5.8003, 2.2803),
            (14.7328, -20.3658, 7.5658),
        ];

        let use_transpose = rows > cols;
        let (m, n) = if use_transpose { (cols, rows) } else { (rows, cols) };

        let mut xtx = vec![0.0f32; n * n];
        let mut b = vec![0.0f32; n * n];
        let mut xb = vec![0.0f32; m * n];

        for &(a, bcoeff, c) in &COEFFS {
            // X^T X → [n × n]
            for i in 0..n {
                for j in i..n {
                    let mut s = 0.0f32;
                    for k in 0..m {
                        s += x[k * n + i] * x[k * n + j];
                    }
                    xtx[i * n + j] = s;
                    xtx[j * n + i] = s;
                }
            }

            // B = bcoeff * A + c * A²
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for k in 0..n {
                        s += xtx[i * n + k] * xtx[k * n + j];
                    }
                    b[i * n + j] = bcoeff * xtx[i * n + j] + c * s;
                }
            }

            // X = a * X + X @ B
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for k in 0..n {
                        s += x[i * n + k] * b[k * n + j];
                    }
                    xb[i * n + j] = s;
                }
            }
            for i in 0..m * n {
                x[i] = a * x[i] + xb[i];
            }
        }
    }
}

// ─── CpuBackend implementation ──────────────────────────────

use modgrad_compute::backend::CpuBackend;

impl TransformerOps for CpuBackend {
    fn rms_norm(&self, src: &[f32], dst: &mut [f32], eps: f32) {
        cpu::rms_norm(src, dst, eps);
    }

    fn matvec_nobias(&self, weight: &[f32], x: &[f32],
                     y: &mut [f32], out_dim: usize, in_dim: usize) {
        cpu::matvec_nobias(weight, x, y, out_dim, in_dim);
    }

    fn softmax_inplace(&self, x: &mut [f32]) {
        cpu::softmax_inplace(x);
    }

    fn relu_squared_inplace(&self, x: &mut [f32]) {
        cpu::relu_squared_inplace(x);
    }

    fn polar_express(&self, x: &mut [f32], rows: usize, cols: usize) {
        cpu::polar_express(x, rows, cols);
    }
}
