//! AdaLN noise conditioning for DiffusionBlocks (slice 2, pieces 2–3).
//!
//! A conditional RMS norm: `y = (1 + scale) ⊙ rms_norm(x) + shift`, where
//! `(scale, shift)` are linearly projected from a noise-conditioning embedding `c`
//! (e.g. the sinusoidal embedding of `c_noise = ¼·ln σ`). AdaLN-Zero init
//! (projection weights = 0) makes the block start at plain rms_norm, so converting
//! an existing transformer to DiffusionBlocks is non-destructive.
//!
//! `modgrad-transformer` has no autograd, so forward AND backward are by hand —
//! and finite-difference gradchecked (see tests) so the manual gradients are proven.

/// Projection `c (cond_dim) → (scale, shift)` (each `model_dim`) applied around an
/// RMS norm. Weights row-major: `w[r*cond_dim + i]`, rows `0..model_dim` = scale,
/// `model_dim..2*model_dim` = shift.
#[derive(Clone)]
pub struct AdaLn {
    pub cond_dim: usize,
    pub model_dim: usize,
    pub eps: f32,
    pub w: Vec<f32>, // [2*model_dim * cond_dim]
    pub b: Vec<f32>, // [2*model_dim]
}

/// Per-token forward cache consumed by [`AdaLn::backward`].
pub struct AdaLnCache {
    normed: Vec<f32>, // x · inv_rms
    inv_rms: f32,
    scale: Vec<f32>, // 1 + projected scale
    x: Vec<f32>,
}

impl AdaLn {
    /// AdaLN-Zero init: zero projection ⇒ scale=1, shift=0 ⇒ y = rms_norm(x).
    pub fn new(cond_dim: usize, model_dim: usize, eps: f32) -> Self {
        Self { cond_dim, model_dim, eps,
               w: vec![0.0; 2 * model_dim * cond_dim], b: vec![0.0; 2 * model_dim] }
    }

    fn project(&self, cond: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (m, cd) = (self.model_dim, self.cond_dim);
        let mut scale = vec![0.0f32; m];
        let mut shift = vec![0.0f32; m];
        for j in 0..m {
            let mut s = self.b[j];
            let mut h = self.b[m + j];
            for i in 0..cd {
                s += self.w[j * cd + i] * cond[i];
                h += self.w[(m + j) * cd + i] * cond[i];
            }
            scale[j] = 1.0 + s; // AdaLN-Zero offset
            shift[j] = h;
        }
        (scale, shift)
    }

    /// Single-token forward. Returns `y` and the backward cache.
    pub fn forward(&self, cond: &[f32], x: &[f32]) -> (Vec<f32>, AdaLnCache) {
        let m = self.model_dim;
        let ms = x.iter().map(|v| v * v).sum::<f32>() / m as f32;
        let inv_rms = 1.0 / (ms + self.eps).sqrt();
        let normed: Vec<f32> = x.iter().map(|&v| v * inv_rms).collect();
        let (scale, shift) = self.project(cond);
        let y: Vec<f32> = (0..m).map(|i| scale[i] * normed[i] + shift[i]).collect();
        (y, AdaLnCache { normed, inv_rms, scale, x: x.to_vec() })
    }

    /// Backward. Accumulates parameter grads into `d_w`/`d_b`; returns `(d_x, d_cond)`.
    pub fn backward(&self, d_y: &[f32], cond: &[f32], c: &AdaLnCache,
                    d_w: &mut [f32], d_b: &mut [f32]) -> (Vec<f32>, Vec<f32>) {
        let (m, cd) = (self.model_dim, self.cond_dim);
        let n = m as f32;
        let mut d_normed = vec![0.0f32; m];
        let mut d_cond = vec![0.0f32; cd];
        for j in 0..m {
            let d_scale = d_y[j] * c.normed[j]; // y = scale·normed + shift
            let d_shift = d_y[j];
            d_normed[j] = d_y[j] * c.scale[j];
            d_b[j] += d_scale;
            d_b[m + j] += d_shift;
            for i in 0..cd {
                d_w[j * cd + i] += d_scale * cond[i];
                d_w[(m + j) * cd + i] += d_shift * cond[i];
                d_cond[i] += d_scale * self.w[j * cd + i] + d_shift * self.w[(m + j) * cd + i];
            }
        }
        // rms_norm backward: normed_i = x_i·inv_rms, inv_rms = (mean(x²)+eps)^-1/2
        //   d_x_i = inv_rms·d_normed_i − (x_i·inv_rms³/n)·Σ_j d_normed_j·x_j
        let s: f32 = (0..m).map(|j| d_normed[j] * c.x[j]).sum();
        let inv_rms3 = c.inv_rms * c.inv_rms * c.inv_rms;
        let d_x: Vec<f32> = (0..m).map(|i| c.inv_rms * d_normed[i] - (c.x[i] * inv_rms3 / n) * s).collect();
        (d_x, d_cond)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loss(a: &AdaLn, cond: &[f32], x: &[f32]) -> f32 {
        a.forward(cond, x).0.iter().sum()
    }

    #[test]
    fn adaln_zero_init_is_plain_rmsnorm() {
        // Zero projection ⇒ y = x / rms(x): identity-preserving conversion.
        let a = AdaLn::new(4, 6, 1e-5);
        let x = vec![0.5, -1.0, 2.0, 0.0, -0.3, 1.2];
        let (y, _) = a.forward(&[0.1, 0.2, -0.3, 0.4], &x);
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / 6.0 + 1e-5).sqrt();
        for i in 0..6 { assert!((y[i] - x[i] * inv).abs() < 1e-6); }
    }

    #[test]
    fn adaln_backward_matches_finite_difference() {
        let (cd, md) = (4usize, 6usize);
        let mut a = AdaLn::new(cd, md, 1e-5);
        for k in 0..a.w.len() { a.w[k] = 0.05 * (((k * 7 + 3) % 11) as f32 - 5.0); }
        for k in 0..a.b.len() { a.b[k] = 0.03 * (((k * 5 + 1) % 7) as f32 - 3.0); }
        let cond: Vec<f32> = (0..cd).map(|i| 0.3 * i as f32 - 0.5).collect();
        let x: Vec<f32> = (0..md).map(|i| 0.2 * i as f32 - 0.6).collect();

        let (_y, cache) = a.forward(&cond, &x);
        let d_y = vec![1.0f32; md];
        let mut d_w = vec![0.0f32; a.w.len()];
        let mut d_b = vec![0.0f32; a.b.len()];
        let (d_x, d_cond) = a.backward(&d_y, &cond, &cache, &mut d_w, &mut d_b);

        let h = 1e-3f32;
        let ok = |fd: f32, an: f32| (fd - an).abs() <= 1e-2 + 1e-2 * fd.abs();

        for i in 0..md {
            let mut xp = x.clone(); xp[i] += h;
            let mut xm = x.clone(); xm[i] -= h;
            let fd = (loss(&a, &cond, &xp) - loss(&a, &cond, &xm)) / (2.0 * h);
            assert!(ok(fd, d_x[i]), "d_x[{i}] fd={fd} an={}", d_x[i]);
        }
        for i in 0..cd {
            let mut cp = cond.clone(); cp[i] += h;
            let mut cm = cond.clone(); cm[i] -= h;
            let fd = (loss(&a, &cp, &x) - loss(&a, &cm, &x)) / (2.0 * h);
            assert!(ok(fd, d_cond[i]), "d_cond[{i}] fd={fd} an={}", d_cond[i]);
        }
        for &k in &[0usize, 7, 13, a.w.len() - 1] {
            let orig = a.w[k];
            a.w[k] = orig + h; let lp = loss(&a, &cond, &x);
            a.w[k] = orig - h; let lm = loss(&a, &cond, &x);
            a.w[k] = orig;
            let fd = (lp - lm) / (2.0 * h);
            assert!(ok(fd, d_w[k]), "d_w[{k}] fd={fd} an={}", d_w[k]);
        }
        for &k in &[0usize, md, 2 * md - 1] {
            let orig = a.b[k];
            a.b[k] = orig + h; let lp = loss(&a, &cond, &x);
            a.b[k] = orig - h; let lm = loss(&a, &cond, &x);
            a.b[k] = orig;
            let fd = (lp - lm) / (2.0 * h);
            assert!(ok(fd, d_b[k]), "d_b[{k}] fd={fd} an={}", d_b[k]);
        }
    }
}
