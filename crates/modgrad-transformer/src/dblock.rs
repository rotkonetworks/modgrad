//! A self-contained, conditioned residual block for DiffusionBlocks (slice 2, piece 4).
//!
//!   out = x + W2 · gelu(W1 · AdaLN(cond, x) + b1) + b2
//!
//! This is the minimal *denoiser block* — the apparatus for the convergence
//! experiment (does val-gated block-wise denoising training learn?), kept separate
//! from the production `TransformerBlock` so the working 3.88-baseline path is
//! untouched. It reuses the gradchecked [`AdaLn`] for the norm and adds a 2-layer
//! GELU MLP + residual; full forward + manual backward, finite-difference gradchecked.
//! (No attention/gate yet — those are refinements once the method is shown to learn.)

use super::adaln::{AdaLn, AdaLnCache};

#[derive(Clone)]
pub struct DiffusionBlock {
    pub adaln: AdaLn,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub w1: Vec<f32>, // [hidden_dim * model_dim]
    pub b1: Vec<f32>, // [hidden_dim]
    pub w2: Vec<f32>, // [model_dim * hidden_dim]
    pub b2: Vec<f32>, // [model_dim]
}

pub struct DBlockCache {
    acache: AdaLnCache,
    yn: Vec<f32>, // AdaLN output
    h1: Vec<f32>, // pre-GELU
    a: Vec<f32>,  // post-GELU
}

/// Parameter-gradient accumulators, shaped like the block.
pub struct DBlockGrads {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub adaln_w: Vec<f32>,
    pub adaln_b: Vec<f32>,
}

impl DBlockGrads {
    pub fn zeros(b: &DiffusionBlock) -> Self {
        Self { w1: vec![0.0; b.w1.len()], b1: vec![0.0; b.b1.len()],
               w2: vec![0.0; b.w2.len()], b2: vec![0.0; b.b2.len()],
               adaln_w: vec![0.0; b.adaln.w.len()], adaln_b: vec![0.0; b.adaln.b.len()] }
    }
}

#[inline]
fn gelu(x: f32) -> f32 {
    let c = 0.797_884_56_f32; // √(2/π)
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}
#[inline]
fn gelu_grad(x: f32) -> f32 {
    let c = 0.797_884_56_f32;
    let u = c * (x + 0.044715 * x * x * x);
    let t = u.tanh();
    let du = c * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du
}

impl DiffusionBlock {
    /// AdaLN-Zero + zero MLP ⇒ `out = x` at init (identity residual): conversion
    /// of an existing layer is non-destructive.
    pub fn new(cond_dim: usize, model_dim: usize, hidden_dim: usize, eps: f32) -> Self {
        Self {
            adaln: AdaLn::new(cond_dim, model_dim, eps),
            model_dim, hidden_dim,
            w1: vec![0.0; hidden_dim * model_dim], b1: vec![0.0; hidden_dim],
            w2: vec![0.0; model_dim * hidden_dim], b2: vec![0.0; model_dim],
        }
    }

    pub fn forward(&self, cond: &[f32], x: &[f32]) -> (Vec<f32>, DBlockCache) {
        let (m, hd) = (self.model_dim, self.hidden_dim);
        let (yn, acache) = self.adaln.forward(cond, x);
        let mut h1 = self.b1.clone();
        for k in 0..hd {
            for i in 0..m { h1[k] += self.w1[k * m + i] * yn[i]; }
        }
        let a: Vec<f32> = h1.iter().map(|&v| gelu(v)).collect();
        let mut out = x.to_vec(); // residual
        for i in 0..m {
            let mut s = self.b2[i];
            for k in 0..hd { s += self.w2[i * hd + k] * a[k]; }
            out[i] += s;
        }
        (out, DBlockCache { acache, yn, h1, a })
    }

    /// Backward. Accumulates param grads into `g`; returns `(d_x, d_cond)`.
    pub fn backward(&self, d_out: &[f32], cond: &[f32], c: &DBlockCache, g: &mut DBlockGrads)
        -> (Vec<f32>, Vec<f32>) {
        let (m, hd) = (self.model_dim, self.hidden_dim);
        let mut d_x = d_out.to_vec(); // residual branch carries d_out directly
        let mut d_a = vec![0.0f32; hd];
        for i in 0..m {
            g.b2[i] += d_out[i];
            for k in 0..hd {
                g.w2[i * hd + k] += d_out[i] * c.a[k];
                d_a[k] += d_out[i] * self.w2[i * hd + k];
            }
        }
        let mut d_yn = vec![0.0f32; m];
        for k in 0..hd {
            let d_h1 = d_a[k] * gelu_grad(c.h1[k]);
            g.b1[k] += d_h1;
            for i in 0..m {
                g.w1[k * m + i] += d_h1 * c.yn[i];
                d_yn[i] += d_h1 * self.w1[k * m + i];
            }
        }
        let (d_x_adaln, d_cond) = self.adaln.backward(&d_yn, cond, &c.acache, &mut g.adaln_w, &mut g.adaln_b);
        for i in 0..m { d_x[i] += d_x_adaln[i]; }
        (d_x, d_cond)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loss(b: &DiffusionBlock, cond: &[f32], x: &[f32]) -> f32 {
        b.forward(cond, x).0.iter().sum()
    }

    #[test]
    fn dblock_zero_init_is_identity_residual() {
        let b = DiffusionBlock::new(4, 5, 7, 1e-5);
        let x = vec![0.4, -1.1, 0.7, 2.0, -0.5];
        let (out, _) = b.forward(&[0.2, -0.1, 0.5, 0.3], &x);
        for i in 0..5 { assert!((out[i] - x[i]).abs() < 1e-6, "not identity at init"); }
    }

    #[test]
    fn dblock_backward_matches_finite_difference() {
        let (cd, m, hd) = (4usize, 5usize, 7usize);
        let mut b = DiffusionBlock::new(cd, m, hd, 1e-5);
        for k in 0..b.w1.len() { b.w1[k] = 0.04 * (((k * 5 + 1) % 9) as f32 - 4.0); }
        for k in 0..b.b1.len() { b.b1[k] = 0.03 * ((k % 5) as f32 - 2.0); }
        for k in 0..b.w2.len() { b.w2[k] = 0.04 * (((k * 3 + 2) % 11) as f32 - 5.0); }
        for k in 0..b.b2.len() { b.b2[k] = 0.02 * ((k % 3) as f32 - 1.0); }
        for k in 0..b.adaln.w.len() { b.adaln.w[k] = 0.03 * (((k * 7 + 3) % 11) as f32 - 5.0); }
        for k in 0..b.adaln.b.len() { b.adaln.b[k] = 0.02 * ((k % 7) as f32 - 3.0); }
        let cond: Vec<f32> = (0..cd).map(|i| 0.25 * i as f32 - 0.4).collect();
        let x: Vec<f32> = (0..m).map(|i| 0.3 * i as f32 - 0.7).collect();

        let (_o, cache) = b.forward(&cond, &x);
        let d_out = vec![1.0f32; m];
        let mut g = DBlockGrads::zeros(&b);
        let (d_x, d_cond) = b.backward(&d_out, &cond, &cache, &mut g);

        let h = 1e-3f32;
        let ok = |fd: f32, an: f32| (fd - an).abs() <= 1e-2 + 1e-2 * fd.abs();

        for i in 0..m {
            let mut xp = x.clone(); xp[i] += h;
            let mut xm = x.clone(); xm[i] -= h;
            let fd = (loss(&b, &cond, &xp) - loss(&b, &cond, &xm)) / (2.0 * h);
            assert!(ok(fd, d_x[i]), "d_x[{i}] fd={fd} an={}", d_x[i]);
        }
        for i in 0..cd {
            let mut cp = cond.clone(); cp[i] += h;
            let mut cm = cond.clone(); cm[i] -= h;
            let fd = (loss(&b, &cp, &x) - loss(&b, &cm, &x)) / (2.0 * h);
            assert!(ok(fd, d_cond[i]), "d_cond[{i}] fd={fd} an={}", d_cond[i]);
        }
        // sample each parameter tensor
        macro_rules! check { ($field:ident, $grad:ident, $idxs:expr) => {
            for &k in $idxs {
                let orig = b.$field[k];
                b.$field[k] = orig + h; let lp = loss(&b, &cond, &x);
                b.$field[k] = orig - h; let lm = loss(&b, &cond, &x);
                b.$field[k] = orig;
                let fd = (lp - lm) / (2.0 * h);
                assert!(ok(fd, g.$grad[k]), "{}[{k}] fd={fd} an={}", stringify!($field), g.$grad[k]);
            }
        }}
        check!(w1, w1, &[0usize, 11, b.w1.len() - 1]);
        check!(b1, b1, &[0usize, hd - 1]);
        check!(w2, w2, &[0usize, 13, b.w2.len() - 1]);
        check!(b2, b2, &[0usize, m - 1]);
        // AdaLn params live under b.adaln.*, grads under g.adaln_*
        for &k in &[0usize, 9, b.adaln.w.len() - 1] {
            let orig = b.adaln.w[k];
            b.adaln.w[k] = orig + h; let lp = loss(&b, &cond, &x);
            b.adaln.w[k] = orig - h; let lm = loss(&b, &cond, &x);
            b.adaln.w[k] = orig;
            let fd = (lp - lm) / (2.0 * h);
            assert!(ok(fd, g.adaln_w[k]), "adaln.w[{k}] fd={fd} an={}", g.adaln_w[k]);
        }
    }
}
