//! DiffusionBlocks primitives — the noise-side machinery shared by every
//! DiffusionBlocks adaptation, kept model-agnostic so the transformer, a
//! masked-diffusion LM, or a recurrent-depth thinker can all reuse it.
//!
//! Two papers define the contract:
//!   - EDM (Karras et al., 2022): the log-normal training distribution over
//!     noise levels σ, the preconditioning (c_skip/c_out/c_in/c_noise), and the
//!     loss weighting λ(σ) that keeps gradient magnitudes balanced across σ.
//!   - DiffusionBlocks (Shing et al., ICLR 2026): equi-probability partitioning
//!     of the noise range into B intervals carrying equal log-normal mass, so
//!     each block handles an equal share of the denoising difficulty.
//!
//! Everything here is pure scalar math (no tensors, no device): boundaries and
//! coefficients a trainer/sampler consumes. The transcendental helpers (erf, the
//! inverse normal CDF) are implemented locally rather than pulling a stats crate.

/// EDM noise schedule + the log-normal distribution used for training and for
/// equi-probability block partitioning.
#[derive(Debug, Clone, Copy)]
pub struct NoiseSchedule {
    /// Lowest noise level used at the clean end of sampling.
    pub sigma_min: f32,
    /// Highest noise level (pure-noise end).
    pub sigma_max: f32,
    /// Data scale σ_data; preconditioning is centered on it (EDM default 0.5).
    pub sigma_data: f32,
    /// Mean of log σ under the training distribution (EDM default −1.2).
    pub p_mean: f32,
    /// Std of log σ under the training distribution (EDM default 1.2).
    pub p_std: f32,
}

impl Default for NoiseSchedule {
    /// EDM defaults (Karras et al., 2022), matching the DiffusionBlocks paper.
    fn default() -> Self {
        Self { sigma_min: 0.002, sigma_max: 80.0, sigma_data: 0.5, p_mean: -1.2, p_std: 1.2 }
    }
}

impl NoiseSchedule {
    // ── EDM preconditioning ──────────────────────────────────────────────────
    // The denoiser is D(z;σ) = c_skip(σ)·z + c_out(σ)·F(c_in(σ)·z; c_noise(σ)),
    // chosen so the network F sees unit-variance inputs/targets at every σ.

    /// Skip scaling: σ_data² / (σ² + σ_data²).
    pub fn c_skip(&self, sigma: f32) -> f32 {
        let (s, sd) = (sigma as f64, self.sigma_data as f64);
        (sd * sd / (s * s + sd * sd)) as f32
    }
    /// Output scaling: σ·σ_data / √(σ² + σ_data²).
    pub fn c_out(&self, sigma: f32) -> f32 {
        let (s, sd) = (sigma as f64, self.sigma_data as f64);
        (s * sd / (s * s + sd * sd).sqrt()) as f32
    }
    /// Input scaling: 1 / √(σ² + σ_data²).
    pub fn c_in(&self, sigma: f32) -> f32 {
        let (s, sd) = (sigma as f64, self.sigma_data as f64);
        (1.0 / (s * s + sd * sd).sqrt()) as f32
    }
    /// Noise conditioning fed to the network (AdaLN input): ¼·ln(σ).
    pub fn c_noise(&self, sigma: f32) -> f32 {
        (0.25 * (sigma as f64).ln()) as f32
    }
    /// Loss weighting λ(σ) = (σ² + σ_data²)/(σ·σ_data)² = 1/c_out(σ)², which makes
    /// the effective training target uniformly weighted across noise levels.
    pub fn weight(&self, sigma: f32) -> f32 {
        let (s, sd) = (sigma as f64, self.sigma_data as f64);
        ((s * s + sd * sd) / (s * sd).powi(2)) as f32
    }

    // ── log-normal noise distribution ────────────────────────────────────────

    /// σ at a log-normal quantile q ∈ (0,1): exp(P_mean + P_std·Φ⁻¹(q)).
    pub fn sigma_from_quantile(&self, q: f32) -> f32 {
        let z = inv_norm_cdf(q as f64);
        ((self.p_mean as f64) + (self.p_std as f64) * z).exp() as f32
    }
    /// Inverse of [`sigma_from_quantile`]: the cumulative log-normal mass at σ,
    /// i.e. Φ((ln σ − P_mean)/P_std).
    pub fn quantile_of_sigma(&self, sigma: f32) -> f32 {
        let z = ((sigma as f64).ln() - self.p_mean as f64) / self.p_std as f64;
        norm_cdf(z) as f32
    }
    /// Draw a σ from the log-normal truncated to [σ_min, σ_max], given a uniform
    /// u ∈ (0,1). Maps u across the truncated CDF window so samples stay in range.
    pub fn sigma_from_uniform(&self, u: f32) -> f32 {
        let (qlo, qhi) = (self.quantile_of_sigma(self.sigma_min), self.quantile_of_sigma(self.sigma_max));
        self.sigma_from_quantile(qlo + u.clamp(0.0, 1.0) * (qhi - qlo))
    }

    // ── DiffusionBlocks equi-probability partitioning ────────────────────────

    /// Partition [σ_min, σ_max] into `b` blocks carrying equal log-normal mass.
    /// Returns `b+1` boundaries ascending from σ_min to σ_max; block k owns
    /// [boundaries[k], boundaries[k+1]]. Equal mass (not equal log-σ spacing)
    /// concentrates capacity on the intermediate noise levels that dominate
    /// learning — the core DiffusionBlocks partitioning strategy.
    pub fn equi_prob_boundaries(&self, b: usize) -> Vec<f32> {
        assert!(b >= 1, "need at least one block");
        let qlo = self.quantile_of_sigma(self.sigma_min) as f64;
        let qhi = self.quantile_of_sigma(self.sigma_max) as f64;
        let mut bounds: Vec<f32> = (0..=b)
            .map(|i| {
                let q = qlo + (i as f64 / b as f64) * (qhi - qlo);
                self.sigma_from_quantile(q as f32)
            })
            .collect();
        // The partition endpoints ARE σ_min/σ_max by definition; pin them exactly
        // so deep-tail roundtrip error in erf/Φ⁻¹ (σ_max sits at ~4.6σ) can't drift
        // the usable noise range. Only the interior boundaries need the mapping.
        bounds[0] = self.sigma_min;
        bounds[b] = self.sigma_max;
        bounds
    }

    /// Draw a σ uniformly-by-mass within block `k` of a `b`-block partition,
    /// given a uniform u ∈ (0,1) — i.e. p_noise renormalized to the block's range.
    pub fn sigma_in_block(&self, k: usize, b: usize, u: f32) -> f32 {
        assert!(k < b, "block index out of range");
        let qlo = self.quantile_of_sigma(self.sigma_min);
        let qhi = self.quantile_of_sigma(self.sigma_max);
        let dq = (qhi - qlo) / b as f32;
        let q = qlo + (k as f32 + u.clamp(0.0, 1.0)) * dq;
        self.sigma_from_quantile(q)
    }
}

// ── special functions (no external stats crate) ─────────────────────────────

/// Standard normal CDF Φ(x) via erf: 0.5·(1 + erf(x/√2)).
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// erf via Abramowitz & Stegun 7.1.26 (|error| ≲ 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t
        - 0.284_496_736) * t + 0.254_829_592) * t * (-x * x).exp();
    sign * y
}

/// Inverse standard normal CDF Φ⁻¹(p) via Acklam's rational approximation
/// (relative error < 1.15e-9 across p ∈ (0,1)).
fn inv_norm_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [-3.969_683_028_665_376e1, 2.209_460_984_245_205e2, -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2, -3.066_479_806_614_716e1, 2.506_628_277_459_239e0];
    const B: [f64; 5] = [-5.447_609_879_822_406e1, 1.615_858_368_580_409e2, -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1, -1.328_068_155_288_572e1];
    const C: [f64; 6] = [-7.784_894_002_430_293e-3, -3.223_964_580_411_365e-1, -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0, 4.374_664_141_464_968e0, 2.938_163_982_698_783e0];
    const D: [f64; 4] = [7.784_695_709_041_462e-3, 3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0, 3.754_408_661_907_416e0];
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    const PLOW: f64 = 0.02425;
    if p < PLOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - PLOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() <= tol }

    #[test]
    fn norm_cdf_inverse_roundtrip() {
        // Φ(0)=0.5, Φ⁻¹(0.5)=0, and Φ⁻¹∘Φ ≈ identity.
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!(inv_norm_cdf(0.5).abs() < 1e-6);
        for &x in &[-2.5, -1.0, -0.3, 0.7, 1.5, 2.8] {
            let back = inv_norm_cdf(norm_cdf(x));
            assert!((back - x).abs() < 1e-4, "roundtrip x={x} back={back}");
        }
    }

    #[test]
    fn preconditioning_identities() {
        // The two EDM identities that pin the coefficients:
        //   weight(σ)·c_out(σ)² = 1   and   c_skip(σ) = σ_data²·c_in(σ)².
        let s = NoiseSchedule::default();
        for &sigma in &[0.01f32, 0.1, 0.5, 2.0, 10.0, 60.0] {
            let w_cout2 = s.weight(sigma) * s.c_out(sigma).powi(2);
            assert!(approx(w_cout2, 1.0, 1e-3), "σ={sigma} w·c_out²={w_cout2}");
            let cskip = s.c_skip(sigma);
            let pred = s.sigma_data.powi(2) * s.c_in(sigma).powi(2);
            assert!(approx(cskip, pred, 1e-4), "σ={sigma} c_skip={cskip} pred={pred}");
        }
    }

    #[test]
    fn sigma_quantile_roundtrip() {
        let s = NoiseSchedule::default();
        for &sigma in &[0.005f32, 0.05, 0.5, 5.0, 50.0] {
            let back = s.sigma_from_quantile(s.quantile_of_sigma(sigma));
            assert!((back - sigma).abs() / sigma < 1e-3, "σ={sigma} back={back}");
        }
    }

    #[test]
    fn equi_prob_boundaries_span_and_equal_mass() {
        let s = NoiseSchedule::default();
        for b in [1usize, 2, 3, 4, 6] {
            let bounds = s.equi_prob_boundaries(b);
            assert_eq!(bounds.len(), b + 1);
            // Endpoints pin to [σ_min, σ_max].
            assert!((bounds[0] - s.sigma_min).abs() / s.sigma_min < 1e-3);
            assert!((bounds[b] - s.sigma_max).abs() / s.sigma_max < 1e-3);
            // Ascending.
            for w in bounds.windows(2) { assert!(w[1] > w[0], "not ascending: {bounds:?}"); }
            // Each block carries equal log-normal mass: ΔΦ constant across blocks.
            let masses: Vec<f32> = bounds.windows(2)
                .map(|w| s.quantile_of_sigma(w[1]) - s.quantile_of_sigma(w[0]))
                .collect();
            let m0 = masses[0];
            for (k, &m) in masses.iter().enumerate() {
                assert!(approx(m, m0, 1e-4), "block {k} mass {m} != {m0} (B={b})");
            }
        }
    }

    #[test]
    fn sigma_in_block_stays_within_its_interval() {
        let s = NoiseSchedule::default();
        let (b, bounds) = (4usize, NoiseSchedule::default().equi_prob_boundaries(4));
        for k in 0..b {
            for &u in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let sigma = s.sigma_in_block(k, b, u);
                // Allow a hair of tolerance at the shared boundaries.
                assert!(sigma >= bounds[k] * 0.999 && sigma <= bounds[k + 1] * 1.001,
                    "block {k} u={u}: σ={sigma} outside [{}, {}]", bounds[k], bounds[k + 1]);
            }
        }
    }
}
