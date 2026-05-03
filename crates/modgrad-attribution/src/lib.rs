//! Gradient-based data attribution. Inspired by Motive (NVIDIA, 2026)
//! (motion-aware video attribution), but the underlying machinery — JL
//! projection of per-sample gradients + cosine similarity ranking — is
//! modality-agnostic and lives here as a reusable Service.
//!
//! # Composition (Eriksen-style)
//!
//! The attribution service is `(query_grad, train_grads) → ranking`.
//! Filters that compose with it:
//!
//!   - [`Projection`]: dimensionality reduction (`R^D → R^D'`) so per-
//!     sample gradients can be stored at scale (`D` is the parameter
//!     count, often `>10^9`; `D'` is typically 256–1024).
//!   - [`Attributor`]: takes already-projected gradients and produces a
//!     cosine ranking (single query) or a majority-vote consensus across
//!     multiple queries.
//!
//! Anything in this SDK that can produce a flat gradient vector for a
//! single training example can plug in: [`crate::Attributable`] is the
//! adapter trait. A backward pass writes its gradients into the model's
//! own gradient struct (e.g. `RegionalGradients`, `AttentionResidentGrads`),
//! and an `Attributable` impl flattens those to `Vec<f32>` for projection.
//!
//! # Win condition
//!
//! The experimental claim — and the binary criterion this crate lets us
//! check — is: given a corpus of `N` training examples and `Q` held-out
//! queries, fine-tuning on the top-`K%` selected by influence beats
//! fine-tuning on a random `K%` on the same held-out metric. This is the
//! exact criterion Motive uses, transposed to whatever modality we wire
//! up first (planned: brain → Qwen NLL, where the held-out metric is
//! per-token NLL on a query corpus).
//!
//! # Scope of this crate
//!
//! Numerics + ranking only. Reading the gradient out of a particular
//! model is the model's job (its [`Attributable`] impl). The gradient
//! extraction loop, frame-length / sequence-length normalisation, and
//! the actual fine-tuning experiment all live in callers — this crate
//! provides the pieces, not the harness.

#![forbid(unsafe_code)]

use modgrad_compute::neuron::SimpleRng;

/// A service or model that can produce a flat gradient vector for one
/// training example. This is the adapter between model-specific gradient
/// structs and the modality-agnostic attribution machinery.
///
/// Implementations are expected to:
///   1. Run a forward + backward over a single example into their own
///      gradient buffers.
///   2. Concatenate the relevant parameter gradients into `Vec<f32>`,
///      preserving a fixed parameter ordering across calls.
///
/// What "relevant" means is up to the caller — for whole-model
/// attribution it would be every parameter gradient, but for restricted
/// attribution (e.g. only the brain's output projection in
/// `brain_qwen_nll`) it might be a subset. The dimensionality of the
/// returned vector is what gets passed to [`Projection::new_dense`].
pub trait Attributable {
    /// Concatenate the relevant gradients for one example into a single
    /// dense vector. Length must be stable across calls so projection
    /// matrices can be reused.
    fn flat_grads(&self) -> Vec<f32>;

    /// Total length of the flat gradient vector. Used to size
    /// [`Projection`] without an actual gradient call.
    fn flat_grads_dim(&self) -> usize;
}

/// Random Johnson-Lindenstrauss projection from `R^D → R^D'`.
///
/// This is the dense-matrix variant — the simplest correct version.
/// Motive uses Fastfood (`O(D' log D')` per project via structured
/// transforms) for `D' = 512` and `D ≈ 1.4 · 10^9`. The Fastfood
/// optimisation matters when `D` is huge; the dense variant suffices
/// for the brain-attribution slice (`D` ≤ 10^7) and is easier to audit.
/// Replacing this with a Fastfood [`Projection`] is a follow-up that
/// preserves the public API: callers see a `project` method either way.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Projection {
    /// Row-major `[out_dim × in_dim]` random Gaussian matrix scaled
    /// by `1 / sqrt(out_dim)` so projected vectors preserve inner-
    /// product geometry up to a JL-bounded distortion.
    matrix: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Projection {
    /// Build a dense JL projection. `seed` is reused across calls so
    /// the same projection can be re-derived (or persisted alongside
    /// the projected vectors).
    pub fn new_dense(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        // Variance 1/out_dim → projected inner products ≈ original
        // inner products in expectation (standard JL scaling).
        let scale = 1.0 / (out_dim as f32).sqrt();
        let matrix: Vec<f32> = (0..in_dim * out_dim)
            .map(|_| rng.next_normal() * scale)
            .collect();
        Self { matrix, in_dim, out_dim }
    }

    pub fn in_dim(&self) -> usize { self.in_dim }
    pub fn out_dim(&self) -> usize { self.out_dim }

    /// Project `g` (length `in_dim`) into `R^out_dim`. No normalisation —
    /// callers that want `g̃ / ‖g̃‖` use [`Self::project_normalized`].
    pub fn project(&self, g: &[f32]) -> Vec<f32> {
        debug_assert_eq!(g.len(), self.in_dim,
            "input gradient length {} != projection in_dim {}", g.len(), self.in_dim);
        let mut out = vec![0.0f32; self.out_dim];
        for i in 0..self.out_dim {
            let row = &self.matrix[i * self.in_dim..(i + 1) * self.in_dim];
            let mut dot = 0.0f32;
            for j in 0..self.in_dim {
                dot += row[j] * g[j];
            }
            out[i] = dot;
        }
        out
    }

    /// Project then `ℓ²`-normalise so subsequent cosine similarity
    /// reduces to a plain dot product. Empty (zero-norm) gradients are
    /// returned as zero vectors rather than NaN.
    pub fn project_normalized(&self, g: &[f32]) -> Vec<f32> {
        let mut p = self.project(g);
        let n = p.iter().map(|v| v * v).sum::<f32>().sqrt();
        if n > 0.0 {
            for v in &mut p { *v /= n; }
        }
        p
    }
}

/// Cosine-similarity ranker over already-projected gradient vectors.
/// Stateless — caller owns the train and query gradient sets.
pub struct Attributor;

impl Attributor {
    /// Cosine similarity between one query gradient and each training
    /// gradient. Inputs are expected to be ℓ²-normalised already (use
    /// [`Projection::project_normalized`]); the cosine reduces to a
    /// plain dot product on normalised inputs.
    ///
    /// Returns one influence score per training sample, in the same
    /// order as `train_grads`. Larger = more influential.
    pub fn rank(query_grad: &[f32], train_grads: &[Vec<f32>]) -> Vec<f32> {
        train_grads.iter().map(|g| dot(query_grad, g)).collect()
    }

    /// Indices of the top-`k` most influential training samples for one
    /// query. Stable wrt ties (preserves earlier index on equal scores).
    pub fn top_k(query_grad: &[f32], train_grads: &[Vec<f32>], k: usize) -> Vec<usize> {
        let scores = Self::rank(query_grad, train_grads);
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
        idx
    }

    /// Majority-vote selection across multiple queries (Motive §3.5
    /// aggregation). For each query, samples scoring above the
    /// `percentile_cutoff` get one vote. The top-`k` samples by total
    /// vote count are returned. This emphasises samples consistently
    /// influential across queries without needing cross-query score
    /// calibration.
    ///
    /// `percentile_cutoff` is in `[0, 1]`; e.g. `0.9` means "top 10%
    /// per query gets a vote".
    pub fn maj_vote_top_k(
        query_grads: &[Vec<f32>],
        train_grads: &[Vec<f32>],
        k: usize,
        percentile_cutoff: f32,
    ) -> Vec<usize> {
        debug_assert!((0.0..=1.0).contains(&percentile_cutoff),
            "percentile_cutoff {percentile_cutoff} not in [0, 1]");
        let n = train_grads.len();
        let mut votes = vec![0usize; n];
        for q in query_grads {
            let scores = Self::rank(q, train_grads);
            // Find the cutoff value at the `percentile_cutoff` rank.
            let mut sorted = scores.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let cutoff_idx = ((n as f32) * percentile_cutoff).floor() as usize;
            let cutoff = if cutoff_idx >= n { f32::INFINITY }
                else { sorted[cutoff_idx] };
            for (i, &s) in scores.iter().enumerate() {
                if s > cutoff { votes[i] += 1; }
            }
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| votes[b].cmp(&votes[a]));
        idx.truncate(k);
        idx
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JL projection preserves inner products in expectation. We can't
    /// assert it pointwise (that's a probabilistic guarantee), but we
    /// can check that the relative ordering of three known cases —
    /// identical, anti-correlated, orthogonal — survives projection.
    #[test]
    fn projection_preserves_pairwise_ordering() {
        let in_dim = 1024;
        let out_dim = 256;
        let proj = Projection::new_dense(in_dim, out_dim, 0xA77816u64);

        // Three test gradients with known cosine relationships.
        let mut rng = SimpleRng::new(0x6740);
        let a: Vec<f32> = (0..in_dim).map(|_| rng.next_normal()).collect();
        let b: Vec<f32> = a.iter().map(|x| -x).collect();        // anti-correlated
        let c: Vec<f32> = (0..in_dim).map(|_| rng.next_normal()).collect();  // ~orthogonal

        let pa = proj.project_normalized(&a);
        let pb = proj.project_normalized(&b);
        let pc = proj.project_normalized(&c);

        let cos_aa = dot(&pa, &pa);   // 1.0 (self)
        let cos_ab = dot(&pa, &pb);   // ≈ -1
        let cos_ac = dot(&pa, &pc);   // ≈ 0

        eprintln!("projected cosines: aa={cos_aa:.3}, ab={cos_ab:.3}, ac={cos_ac:.3}");
        assert!(cos_aa > 0.95, "self-cosine must be ~1, got {cos_aa}");
        assert!(cos_ab < -0.5, "anti-correlated cosine must be << 0, got {cos_ab}");
        assert!(cos_ac.abs() < 0.3, "random-pair cosine must be ~0, got {cos_ac}");
    }

    /// `top_k` should rank a synthetic ground truth correctly: planted
    /// "useful" gradients (close to the query) come above noise.
    #[test]
    fn top_k_ranks_planted_gradients_above_noise() {
        let in_dim = 512;
        let out_dim = 128;
        let proj = Projection::new_dense(in_dim, out_dim, 0xCABBA6Eu64);
        let mut rng = SimpleRng::new(0xC0DEu64);

        // Query: a random direction.
        let query: Vec<f32> = (0..in_dim).map(|_| rng.next_normal()).collect();
        let q_proj = proj.project_normalized(&query);

        // Train set: 50 noise gradients + 5 planted ones (= query + small noise).
        let n_noise = 50;
        let n_planted = 5;
        let mut train: Vec<Vec<f32>> = Vec::with_capacity(n_noise + n_planted);
        for _ in 0..n_noise {
            let g: Vec<f32> = (0..in_dim).map(|_| rng.next_normal()).collect();
            train.push(g);
        }
        for _ in 0..n_planted {
            // query + 0.1·noise → still highly aligned with query.
            let g: Vec<f32> = query.iter()
                .map(|q| q + 0.1 * rng.next_normal())
                .collect();
            train.push(g);
        }
        let train_proj: Vec<Vec<f32>> = train.iter()
            .map(|g| proj.project_normalized(g))
            .collect();

        let top = Attributor::top_k(&q_proj, &train_proj, n_planted);
        // Top-K should be exactly the planted indices (50..55).
        let mut sorted_top = top.clone();
        sorted_top.sort();
        let planted_range: Vec<usize> = (n_noise..n_noise + n_planted).collect();
        eprintln!("top-{n_planted}: {top:?}, planted: {planted_range:?}");
        assert_eq!(sorted_top, planted_range,
            "top-K should recover all planted gradients above noise");
    }

    /// Majority vote across queries should still recover planted samples
    /// when each query independently votes for them.
    #[test]
    fn maj_vote_aggregates_consistent_influencers() {
        let in_dim = 256;
        let out_dim = 64;
        let proj = Projection::new_dense(in_dim, out_dim, 0x4A6Au64);
        let mut rng = SimpleRng::new(0x10E5u64);

        // Three different queries, each with their own planted positives.
        // We make 5 "universal" planted samples that all queries align with
        // (perturbed query average) and 30 noise samples. Top-vote-count
        // should be the universal planted samples.
        let queries: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..in_dim).map(|_| rng.next_normal()).collect())
            .collect();
        let q_avg: Vec<f32> = (0..in_dim)
            .map(|i| queries.iter().map(|q| q[i]).sum::<f32>() / 3.0)
            .collect();

        let n_noise = 30;
        let n_planted = 5;
        let mut train = Vec::with_capacity(n_noise + n_planted);
        for _ in 0..n_noise {
            train.push((0..in_dim).map(|_| rng.next_normal()).collect::<Vec<f32>>());
        }
        for _ in 0..n_planted {
            train.push(q_avg.iter().map(|v| v + 0.05 * rng.next_normal()).collect());
        }

        let q_proj: Vec<Vec<f32>> = queries.iter()
            .map(|q| proj.project_normalized(q))
            .collect();
        let t_proj: Vec<Vec<f32>> = train.iter()
            .map(|g| proj.project_normalized(g))
            .collect();

        // Top 10% per query → planted samples should sweep all three votes.
        let top = Attributor::maj_vote_top_k(&q_proj, &t_proj, n_planted, 0.85);
        let mut sorted_top = top.clone();
        sorted_top.sort();
        let planted: Vec<usize> = (n_noise..n_noise + n_planted).collect();
        eprintln!("maj_vote top-{n_planted}: {top:?}, planted: {planted:?}");
        assert_eq!(sorted_top, planted,
            "majority vote should recover universally-aligned planted samples");
    }
}
