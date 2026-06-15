//! Slice 3 — the experiment Anneal actually rests on.
//!
//! DiffusionBlocks claims blocks can be trained **independently per noise range**,
//! with zero inter-block communication, and still assemble into a denoiser as good
//! as one trained jointly on the whole range. That independence is the *only* reason
//! disconnected volunteer hardware can contribute. So we test it directly, in-process:
//!
//!   - JOINT: one denoiser trained on the full noise range.
//!   - INDEPENDENT: B denoisers, each trained ONLY on its sub-range (no shared state).
//!   - Eval both on the same per-range partition; independence must be *competitive*
//!     (specialists on a narrow range should match or beat the generalist).
//!
//! If independent ≈ joint, the distributed vision is de-risked. If it's much worse,
//! that's a finding that matters before any browser/JAM/token machinery gets built.

use modgrad_transformer::denoiser::Denoiser;
use modgrad_transformer::diffusion::NoiseSchedule;

/// Deterministic PRNG (reproducible, no `rand` dep).
pub struct Lcg(pub u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// Uniform in [0, 1).
    pub fn unit01(&mut self) -> f32 { self.next_u32() as f32 / (u32::MAX as f32 + 1.0) }
    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 { self.unit01() * 2.0 - 1.0 }
}

/// Small-random MLP init so gradients flow from step 0 (the slice-2 dead-gradient lesson).
pub fn randomize(d: &mut Denoiser, rng: &mut Lcg) {
    for b in &mut d.blocks {
        for w in b.w1.iter_mut() { *w = 0.2 * rng.unit(); }
        for w in b.w2.iter_mut() { *w = 0.2 * rng.unit(); }
    }
}

/// Train `d` on σ drawn from the log-normal restricted to quantile range `[q_lo, q_hi]`,
/// fresh ε every step (real denoising, not memorization).
pub fn train(d: &mut Denoiser, target: &[f32], q_lo: f32, q_hi: f32, steps: usize, lr: f32, rng: &mut Lcg) {
    let m = target.len();
    for _ in 0..steps {
        let q = q_lo + rng.unit01() * (q_hi - q_lo);
        let sigma = d.schedule.sigma_from_quantile(q);
        let eps: Vec<f32> = (0..m).map(|_| rng.unit()).collect();
        d.train_step(target, sigma, &eps, lr);
    }
}

/// Mean denoising loss over `n` fresh (σ, ε) samples in quantile range `[q_lo, q_hi]`.
pub fn eval(d: &Denoiser, target: &[f32], q_lo: f32, q_hi: f32, n: usize, rng: &mut Lcg) -> f32 {
    let m = target.len();
    let mut s = 0.0f32;
    for _ in 0..n {
        let q = q_lo + rng.unit01() * (q_hi - q_lo);
        let sigma = d.schedule.sigma_from_quantile(q);
        let eps: Vec<f32> = (0..m).map(|_| rng.unit()).collect();
        s += d.loss(target, sigma, &eps);
    }
    s / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_blocks_match_joint() {
        let m = 8usize;
        let sched = NoiseSchedule::default();
        let target: Vec<f32> = (0..m).map(|i| (i as f32 * 0.7).sin()).collect();
        let (q_lo, q_hi) = (0.05f32, 0.95f32);
        let b = 3usize;
        let steps = 4000usize; // per denoiser (independent specialists run in parallel in prod)
        let lr = 0.01f32;
        let span = |bi: usize| {
            (q_lo + (q_hi - q_lo) * bi as f32 / b as f32,
             q_lo + (q_hi - q_lo) * (bi + 1) as f32 / b as f32)
        };

        // JOINT — one denoiser, full range.
        let mut rng = Lcg(1);
        let mut joint = Denoiser::new(2, 16, m, 24, sched);
        randomize(&mut joint, &mut rng);
        train(&mut joint, &target, q_lo, q_hi, steps, lr, &mut rng);

        // INDEPENDENT — B denoisers, each only on its sub-range, no shared state.
        let indep: Vec<Denoiser> = (0..b).map(|bi| {
            let mut rb = Lcg(100 + bi as u64);
            let mut d = Denoiser::new(2, 16, m, 24, sched);
            randomize(&mut d, &mut rb);
            let (lo, hi) = span(bi);
            train(&mut d, &target, lo, hi, steps, lr, &mut rb);
            d
        }).collect();

        // Eval both on the SAME per-range partition.
        let mut er = Lcg(999);
        let n = 600usize;
        let (mut joint_avg, mut indep_avg) = (0.0f32, 0.0f32);
        for bi in 0..b {
            let (lo, hi) = span(bi);
            joint_avg += eval(&joint, &target, lo, hi, n, &mut er);
            indep_avg += eval(&indep[bi], &target, lo, hi, n, &mut er);
        }
        joint_avg /= b as f32;
        indep_avg /= b as f32;

        eprintln!("[slice3] joint avg loss {joint_avg:.4} | independent-per-range avg loss {indep_avg:.4}  (ratio {:.3})",
                  indep_avg / joint_avg);
        assert!(joint_avg.is_finite() && indep_avg.is_finite(), "non-finite loss");
        // The claim: independence is competitive — specialists on narrow ranges match
        // or beat the generalist. Margin for SGD noise.
        assert!(indep_avg <= joint_avg * 1.2,
                "independent training materially hurt: indep {indep_avg:.4} vs joint {joint_avg:.4}");
    }
}
