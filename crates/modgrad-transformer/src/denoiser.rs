//! Minimal DiffusionBlocks denoiser + denoising training step (slice 2, piece 5).
//!
//! A stack of [`DiffusionBlock`]s wrapped in EDM preconditioning:
//!   D(z;σ) = c_skip·z + c_out·F(c_in·z ; c_noise),  F = block-stack.
//! Trained with the EDM-reparameterized objective: the stack F predicts the
//! unit-scaled target t = (y − c_skip·z)/c_out under plain MSE (equivalent to the
//! weighted denoising loss w(σ)‖D−y‖², but numerically stable across all σ).
//!
//! The smoke test is the cheap, decisive experiment the design hinges on: **does
//! block-wise denoising training actually reduce loss?** If this trends down, the
//! block + manual backward + optimizer wiring learn end-to-end — the green light to
//! scale to a real corpus (BPC) and then the async/distributed dynamics.

use super::diffusion::{NoiseSchedule, timestep_embedding};
use super::dblock::{DiffusionBlock, DBlockCache, DBlockGrads};

pub struct Denoiser {
    pub blocks: Vec<DiffusionBlock>,
    pub schedule: NoiseSchedule,
    pub cond_dim: usize,
    pub model_dim: usize,
    /// SGD momentum velocity, shaped per block.
    pub vel: Vec<DBlockGrads>,
    pub momentum: f32,
}

impl Denoiser {
    pub fn new(n_blocks: usize, cond_dim: usize, model_dim: usize, hidden_dim: usize,
               schedule: NoiseSchedule) -> Self {
        let blocks: Vec<DiffusionBlock> = (0..n_blocks)
            .map(|_| DiffusionBlock::new(cond_dim, model_dim, hidden_dim, 1e-5))
            .collect();
        let vel = blocks.iter().map(DBlockGrads::zeros).collect();
        Self { blocks, schedule, cond_dim, model_dim, vel, momentum: 0.9 }
    }

    fn stack_forward(&self, cond: &[f32], zin: &[f32]) -> (Vec<f32>, Vec<DBlockCache>) {
        let mut h = zin.to_vec();
        let mut caches = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let (out, c) = b.forward(cond, &h);
            h = out;
            caches.push(c);
        }
        (h, caches)
    }

    fn stack_backward(&self, d_f: &[f32], cond: &[f32], caches: &[DBlockCache],
                      grads: &mut [DBlockGrads]) {
        let mut d_h = d_f.to_vec();
        for i in (0..self.blocks.len()).rev() {
            let (d_prev, _d_cond) = self.blocks[i].backward(&d_h, cond, &caches[i], &mut grads[i]);
            d_h = d_prev;
        }
    }

    /// Preconditioned denoise (inference/eval): D(z;σ) = c_skip·z + c_out·F.
    pub fn denoise(&self, z: &[f32], sigma: f32) -> Vec<f32> {
        let (cskip, cout, cin) = (self.schedule.c_skip(sigma), self.schedule.c_out(sigma), self.schedule.c_in(sigma));
        let cond = timestep_embedding(self.schedule.c_noise(sigma), self.cond_dim, 10_000.0);
        let zin: Vec<f32> = z.iter().map(|&v| cin * v).collect();
        let (f, _) = self.stack_forward(&cond, &zin);
        (0..self.model_dim).map(|i| cskip * z[i] + cout * f[i]).collect()
    }

    /// One EDM denoising SGD step. `z = y + σ·eps`; the stack F is trained to
    /// predict t = (y − c_skip·z)/c_out (unit-scaled). Returns the F-space MSE.
    pub fn train_step(&mut self, y: &[f32], sigma: f32, eps: &[f32], lr: f32) -> f32 {
        let m = self.model_dim;
        let (cskip, cout, cin) = (self.schedule.c_skip(sigma), self.schedule.c_out(sigma), self.schedule.c_in(sigma));
        let cond = timestep_embedding(self.schedule.c_noise(sigma), self.cond_dim, 10_000.0);
        let z: Vec<f32> = (0..m).map(|i| y[i] + sigma * eps[i]).collect();
        let zin: Vec<f32> = z.iter().map(|&v| cin * v).collect();

        let (f, caches) = self.stack_forward(&cond, &zin);
        let t: Vec<f32> = (0..m).map(|i| (y[i] - cskip * z[i]) / cout).collect();
        let loss: f32 = (0..m).map(|i| (f[i] - t[i]).powi(2)).sum::<f32>() / m as f32;
        let d_f: Vec<f32> = (0..m).map(|i| 2.0 * (f[i] - t[i]) / m as f32).collect();

        let mut grads: Vec<DBlockGrads> = self.blocks.iter().map(DBlockGrads::zeros).collect();
        self.stack_backward(&d_f, &cond, &caches, &mut grads);

        let beta = self.momentum;
        for bi in 0..self.blocks.len() {
            let g = &grads[bi];
            let v = &mut self.vel[bi];
            let b = &mut self.blocks[bi];
            // v ← β·v + g ; θ ← θ − lr·v  (SGD with momentum)
            for k in 0..b.w1.len() { v.w1[k] = beta * v.w1[k] + g.w1[k]; b.w1[k] -= lr * v.w1[k]; }
            for k in 0..b.b1.len() { v.b1[k] = beta * v.b1[k] + g.b1[k]; b.b1[k] -= lr * v.b1[k]; }
            for k in 0..b.w2.len() { v.w2[k] = beta * v.w2[k] + g.w2[k]; b.w2[k] -= lr * v.w2[k]; }
            for k in 0..b.b2.len() { v.b2[k] = beta * v.b2[k] + g.b2[k]; b.b2[k] -= lr * v.b2[k]; }
            for k in 0..b.adaln.w.len() { v.adaln_w[k] = beta * v.adaln_w[k] + g.adaln_w[k]; b.adaln.w[k] -= lr * v.adaln_w[k]; }
            for k in 0..b.adaln.b.len() { v.adaln_b[k] = beta * v.adaln_b[k] + g.adaln_b[k]; b.adaln.b[k] -= lr * v.adaln_b[k]; }
        }
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic PRNG so the smoke is reproducible without a `rand` dep.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn unit(&mut self) -> f32 { (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0 } // [-1,1]
    }

    #[test]
    fn denoiser_smoke_loss_decreases() {
        // The decisive experiment: block-wise denoising training must reduce loss.
        let m = 8usize;
        let mut d = Denoiser::new(2, 16, m, 24, NoiseSchedule::default());
        // Normal (non-zero) init of the MLP matrices so gradients flow from step 0.
        // (Zero-init ⇒ gelu(0)=0 and w2=0 ⇒ dead gradient everywhere but the bias.
        // The production retrofit keeps this small-random init + a zero-init gate.)
        let mut init_rng = Lcg(0xBEEF);
        for b in &mut d.blocks {
            for w in b.w1.iter_mut() { *w = 0.2 * init_rng.unit(); }
            for w in b.w2.iter_mut() { *w = 0.2 * init_rng.unit(); }
        }
        let target: Vec<f32> = (0..m).map(|i| (i as f32 * 0.7).sin()).collect();

        // Fixed σ-set spanning the range, so the tracked loss is comparable
        // window-to-window (one cycle = all σ once); fresh ε every step.
        let k = 8usize;
        let sigmas: Vec<f32> = (0..k)
            .map(|i| d.schedule.sigma_from_quantile(0.06 + 0.88 * (i as f32 + 0.5) / k as f32))
            .collect();
        let mut rng = Lcg(0xC0FFEE);
        // Fix ε per σ ⇒ a small fixed dataset to overfit (memorization smoke, like
        // lm_trainer's): isolates "can the optimizer minimize the denoising objective"
        // from the irreducible-noise capacity floor of fresh-ε denoising.
        let eps_fixed: Vec<Vec<f32>> = (0..k).map(|_| (0..m).map(|_| rng.unit()).collect()).collect();

        let cycles = 150usize;
        let (mut first, mut last) = (0.0f32, 0.0f32);
        for c in 0..cycles {
            let mut cyc = 0.0f32;
            for (i, &sigma) in sigmas.iter().enumerate() {
                cyc += d.train_step(&target, sigma, &eps_fixed[i], 0.02);
            }
            cyc /= k as f32;
            if c == 0 { first = cyc; }
            if c == cycles - 1 { last = cyc; }
        }
        eprintln!("[denoiser smoke] first-cycle loss {first:.4} -> last-cycle loss {last:.4}");
        assert!(last < first * 0.3, "denoising loss did not converge: {first:.4} -> {last:.4}");
        assert!(last.is_finite(), "loss went non-finite");
    }
}
