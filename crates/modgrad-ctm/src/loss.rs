//! CTM-specific loss functions.
//!
//! These use tick semantics (min-tick, most-certain-tick, imagination phases)
//! that are specific to the CTM architecture. Generic losses (DistributionLoss,
//! ReconstructionLoss, RouteLoss, etc.) live in modgrad-traits.

use modgrad_traits::{LossFn, ClassTarget, cross_entropy_grad};

/// CTM-style loss: (min_CE_tick + most_certain_tick) / 2.
pub struct CtmLoss;

impl LossFn for CtmLoss {
    type Target = ClassTarget;

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        certainties: &[[f32; 2]],
        target: &ClassTarget,
    ) -> (f32, Vec<Vec<f32>>) {
        let target = *target;
        let k = predictions.len();
        if k == 0 { return (0.0, Vec::new()); }
        let out_dim = predictions[0].len();

        let ce: Vec<(f32, Vec<f32>)> = predictions.iter()
            .map(|p| cross_entropy_grad(p, target))
            .collect();

        let min_tick = (0..k).min_by(|&a, &b|
            ce[a].0.partial_cmp(&ce[b].0).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
        let cert_tick = (0..k).max_by(|&a, &b|
            certainties[a][1].partial_cmp(&certainties[b][1]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(k - 1);

        let loss = (ce[min_tick].0 + ce[cert_tick].0) / 2.0;

        let mut d_preds = vec![vec![0.0f32; out_dim]; k];
        for (j, g) in ce[min_tick].1.iter().enumerate() {
            d_preds[min_tick][j] += 0.5 * g;
        }
        for (j, g) in ce[cert_tick].1.iter().enumerate() {
            d_preds[cert_tick][j] += 0.5 * g;
        }

        (loss, d_preds)
    }
}

/// Simple cross-entropy on the last tick only.
pub struct LastTickCE;

impl LossFn for LastTickCE {
    type Target = ClassTarget;

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        _certainties: &[[f32; 2]],
        target: &ClassTarget,
    ) -> (f32, Vec<Vec<f32>>) {
        let target = *target;
        let k = predictions.len();
        if k == 0 { return (0.0, Vec::new()); }
        let out_dim = predictions[0].len();
        let (loss, grad) = cross_entropy_grad(&predictions[k - 1], target);
        let mut d_preds = vec![vec![0.0f32; out_dim]; k];
        d_preds[k - 1] = grad;
        (loss, d_preds)
    }
}

/// Thinking-aware loss: CTM loss + reward for productive ticks.
pub struct ThinkingLoss {
    pub alpha: f32,
    pub min_improvement: f32,
}

impl Default for ThinkingLoss {
    fn default() -> Self {
        Self { alpha: 0.1, min_improvement: 0.01 }
    }
}

impl LossFn for ThinkingLoss {
    type Target = ClassTarget;

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        certainties: &[[f32; 2]],
        target: &ClassTarget,
    ) -> (f32, Vec<Vec<f32>>) {
        let target = *target;
        let k = predictions.len();
        if k == 0 { return (0.0, Vec::new()); }
        let out_dim = predictions[0].len();

        let ce: Vec<(f32, Vec<f32>)> = predictions.iter()
            .map(|p| cross_entropy_grad(p, target))
            .collect();

        let min_tick = (0..k).min_by(|&a, &b|
            ce[a].0.partial_cmp(&ce[b].0).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
        let cert_tick = (0..k).max_by(|&a, &b|
            certainties[a][1].partial_cmp(&certainties[b][1]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(k - 1);
        let base_loss = (ce[min_tick].0 + ce[cert_tick].0) / 2.0;

        let mut thinking_loss = 0.0f32;
        for t in 1..k {
            let improvement = ce[t - 1].0 - ce[t].0;
            if improvement < self.min_improvement {
                thinking_loss += self.min_improvement - improvement;
            }
        }
        thinking_loss /= (k - 1).max(1) as f32;

        let total_loss = base_loss + self.alpha * thinking_loss;

        let mut d_preds = vec![vec![0.0f32; out_dim]; k];
        for (j, g) in ce[min_tick].1.iter().enumerate() {
            d_preds[min_tick][j] += 0.5 * g;
        }
        for (j, g) in ce[cert_tick].1.iter().enumerate() {
            d_preds[cert_tick][j] += 0.5 * g;
        }

        let alpha_per_tick = self.alpha / (k - 1).max(1) as f32;
        for t in 1..k {
            let improvement = ce[t - 1].0 - ce[t].0;
            if improvement < self.min_improvement {
                for (j, g) in ce[t].1.iter().enumerate() {
                    d_preds[t][j] += alpha_per_tick * g;
                }
                for (j, g) in ce[t - 1].1.iter().enumerate() {
                    d_preds[t - 1][j] -= 0.5 * alpha_per_tick * g;
                }
            }
        }

        (total_loss, d_preds)
    }
}

/// Imagination loss: two-phase thinking (imagine → commit).
///
/// Early ticks build internal state without loss gradient,
/// later ticks commit. The imagination bonus rewards improvement
/// between phases.
pub struct ImaginationLoss {
    pub imagine_ratio: f32,
    pub imagination_bonus: f32,
}

impl Default for ImaginationLoss {
    fn default() -> Self {
        Self { imagine_ratio: 0.5, imagination_bonus: 0.1 }
    }
}

impl LossFn for ImaginationLoss {
    type Target = ClassTarget;

    fn compute(
        &self,
        predictions: &[Vec<f32>],
        certainties: &[[f32; 2]],
        target: &ClassTarget,
    ) -> (f32, Vec<Vec<f32>>) {
        let target = *target;
        let k = predictions.len();
        if k == 0 { return (0.0, Vec::new()); }
        let out_dim = predictions[0].len();

        let ce: Vec<(f32, Vec<f32>)> = predictions.iter()
            .map(|p| cross_entropy_grad(p, target))
            .collect();

        let commit_start = ((k as f32 * self.imagine_ratio).ceil() as usize).min(k.saturating_sub(1));

        let committed = &ce[commit_start..];
        if committed.is_empty() {
            return (0.0, vec![vec![0.0; out_dim]; k]);
        }

        let min_idx = (0..committed.len()).min_by(|&a, &b|
            committed[a].0.partial_cmp(&committed[b].0).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
        let cert_idx = (commit_start..k).max_by(|&a, &b|
            certainties[a][1].partial_cmp(&certainties[b][1]).unwrap_or(std::cmp::Ordering::Equal))
            .map(|i| i - commit_start).unwrap_or(committed.len() - 1);

        let base_loss = (committed[min_idx].0 + committed[cert_idx].0) / 2.0;

        let imagination_quality = if commit_start > 0 {
            let first_imagine_loss = ce[0].0;
            let first_commit_loss = ce[commit_start].0;
            (first_imagine_loss - first_commit_loss).max(0.0)
        } else {
            0.0
        };

        let total_loss = base_loss - self.imagination_bonus * imagination_quality;

        let mut d_preds = vec![vec![0.0f32; out_dim]; k];
        let abs_min = commit_start + min_idx;
        let abs_cert = commit_start + cert_idx;
        for (j, g) in ce[abs_min].1.iter().enumerate() {
            d_preds[abs_min][j] += 0.5 * g;
        }
        for (j, g) in ce[abs_cert].1.iter().enumerate() {
            d_preds[abs_cert][j] += 0.5 * g;
        }

        (total_loss, d_preds)
    }
}
