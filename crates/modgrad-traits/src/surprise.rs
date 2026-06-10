//! `SurpriseModel` — the unified next-patch predictor.
//!
//! North-star architecture (CTM-as-BLT-latent): ONE model predicts the next
//! patch, and its prediction error does four jobs at once:
//!   1. patch boundary signal (surprise spike → boundary),
//!   2. CTM thinking-time gate (high surprise → more ticks),
//!   3. cerebellar fast forward model (predict the next event),
//!   4. self-supervised loss (predict-next-patch, no labels).
//! Surprise is the single currency that allocates both spatial chunking
//! (where to cut patches) and temporal compute (how long to think).

/// Predicts the next patch from context and measures surprise (prediction
/// error). The default `surprise` is `½‖predicted − actual‖²` and its
/// gradient w.r.t. `predicted` is `(predicted − actual)` — the seed for the
/// self-supervised backward.
pub trait SurpriseModel {
    /// Per-call cache produced by `predict_next`, consumed by
    /// `predict_backward`.
    type Cache;
    /// Implementor-defined error.
    type Error;

    /// Width of a patch / prediction.
    fn patch_dim(&self) -> usize;

    /// Predict the next patch (`[patch_dim]`) from `ctx` (e.g. the current
    /// patch / thought / CTM state).
    fn predict_next(&mut self, ctx: &[f32]) -> Result<(Vec<f32>, Self::Cache), Self::Error>;

    /// Backward: given `d_pred` (`[patch_dim]`), accumulate weight grads and
    /// return `d_ctx`.
    fn predict_backward(&mut self, d_pred: &[f32], cache: &Self::Cache) -> Result<Vec<f32>, Self::Error>;

    /// Surprise = prediction-error energy `½‖predicted − actual‖²`. Used as
    /// the boundary signal and the thinking-time gate.
    fn surprise(&self, predicted: &[f32], actual: &[f32]) -> f32 {
        0.5 * predicted.iter().zip(actual).map(|(&p, &a)| (p - a) * (p - a)).sum::<f32>()
    }

    /// Loss gradient w.r.t. the prediction: `predicted − actual` (the seed
    /// for `predict_backward` under the `½‖·‖²` surprise loss).
    fn surprise_grad(&self, predicted: &[f32], actual: &[f32]) -> Vec<f32> {
        predicted.iter().zip(actual).map(|(&p, &a)| p - a).collect()
    }
}

/// Surprise-gated thinking time (A6): map a patch's surprise to a CTM
/// tick budget. The cerebellar forward model's miss recruits cortical
/// deliberation — high surprise → more ticks, low surprise → exit early.
/// Monotone in `surprise` (≥0), clamped to `[min_ticks, max_ticks]`;
/// `scale` is the surprise at which the budget reaches ~76% of its range
/// (`tanh(1)`). This is the temporal half of the one surprise currency —
/// the same signal that cuts patches (A5) gates how long to think here.
pub fn surprise_tick_budget(surprise: f32, min_ticks: usize, max_ticks: usize, scale: f32) -> usize {
    debug_assert!(min_ticks <= max_ticks);
    let frac = (surprise.max(0.0) / scale.max(1e-6)).tanh(); // [0,1) for surprise≥0
    min_ticks + (((max_ticks - min_ticks) as f32) * frac).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surprise_tick_budget_monotone_and_clamped() {
        // No surprise → minimum ticks (exit early).
        assert_eq!(surprise_tick_budget(0.0, 2, 8, 1.0), 2);
        // Saturating surprise → max ticks, never above.
        assert_eq!(surprise_tick_budget(100.0, 2, 8, 1.0), 8);
        // Monotone non-decreasing in surprise.
        let mut prev = 0usize;
        for s in [0.0f32, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0] {
            let t = surprise_tick_budget(s, 2, 8, 1.0);
            assert!(t >= prev, "non-monotone at surprise={s}: {t} < {prev}");
            assert!((2..=8).contains(&t));
            prev = t;
        }
        // A mid surprise lands strictly between min and max.
        let mid = surprise_tick_budget(1.0, 2, 8, 1.0);
        assert!(mid > 2 && mid < 8, "mid surprise should be interior: {mid}");
    }
}
