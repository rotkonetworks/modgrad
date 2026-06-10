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
