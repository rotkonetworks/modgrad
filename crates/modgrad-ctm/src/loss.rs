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

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: simple logits and certainties for tick sequences.
    fn make_predictions(logits: &[&[f32]]) -> Vec<Vec<f32>> {
        logits.iter().map(|l| l.to_vec()).collect()
    }

    fn uniform_certainties(k: usize) -> Vec<[f32; 2]> {
        vec![[0.5, 0.5]; k]
    }

    // ── LastTickCE ──────────────────────────────────────────────

    #[test]
    fn last_tick_ce_uses_final_tick() {
        let loss_fn = LastTickCE;
        // Two ticks, 3 classes, target = 1
        let preds = make_predictions(&[&[1.0, 0.0, 0.0], &[0.0, 5.0, 0.0]]);
        let certs = uniform_certainties(2);
        let (loss, grads) = loss_fn.compute(&preds, &certs, &1);

        // Loss should be low (correct class has high logit in last tick)
        assert!(loss < 0.1, "expected low loss, got {loss}");
        // Gradient on first tick should be zero
        assert!(grads[0].iter().all(|&g| g == 0.0), "first tick grad should be zero");
        // Gradient on last tick should be nonzero
        assert!(grads[1].iter().any(|&g| g != 0.0), "last tick grad should be nonzero");
    }

    #[test]
    fn last_tick_ce_single_tick() {
        let loss_fn = LastTickCE;
        let preds = make_predictions(&[&[0.0, 0.0, 10.0]]);
        let certs = uniform_certainties(1);
        let (loss, grads) = loss_fn.compute(&preds, &certs, &2);
        assert!(loss < 0.01, "correct class should give near-zero loss, got {loss}");
        assert_eq!(grads.len(), 1);
    }

    #[test]
    fn last_tick_ce_empty_predictions() {
        let loss_fn = LastTickCE;
        let (loss, grads) = loss_fn.compute(&[], &[], &0);
        assert_eq!(loss, 0.0);
        assert!(grads.is_empty());
    }

    #[test]
    fn last_tick_ce_wrong_class_high_loss() {
        let loss_fn = LastTickCE;
        // Target is class 0 but logits favor class 2
        let preds = make_predictions(&[&[-5.0, 0.0, 5.0]]);
        let certs = uniform_certainties(1);
        let (loss, _) = loss_fn.compute(&preds, &certs, &0);
        assert!(loss > 5.0, "wrong class should give high loss, got {loss}");
    }

    // ── CtmLoss ────────────────────────────────────────────────

    #[test]
    fn ctm_loss_averages_min_and_cert_ticks() {
        let loss_fn = CtmLoss;
        // Tick 0: bad, Tick 1: good. Certainty higher on tick 0.
        let preds = make_predictions(&[&[0.0, 0.0, 0.0], &[0.0, 10.0, 0.0]]);
        let certs = vec![[0.5, 0.9], [0.5, 0.1]]; // tick 0 more certain
        let (loss, grads) = loss_fn.compute(&preds, &certs, &1);

        // min_tick = 1 (low CE), cert_tick = 0 (high certainty[1])
        // Loss = avg of CE at tick 1 and CE at tick 0
        assert!(loss > 0.0);
        assert_eq!(grads.len(), 2);
        // Both ticks should have nonzero gradients (one from min, one from cert)
        assert!(grads[0].iter().any(|&g| g != 0.0));
        assert!(grads[1].iter().any(|&g| g != 0.0));
    }

    #[test]
    fn ctm_loss_empty() {
        let loss_fn = CtmLoss;
        let (loss, grads) = loss_fn.compute(&[], &[], &0);
        assert_eq!(loss, 0.0);
        assert!(grads.is_empty());
    }

    // ── ThinkingLoss ───────────────────────────────────────────

    #[test]
    fn thinking_loss_penalizes_no_improvement() {
        let loss_fn = ThinkingLoss::default();
        // All ticks have same logits => no improvement between ticks
        let preds = make_predictions(&[&[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0]]);
        let certs = uniform_certainties(3);
        let (loss_no_improve, _) = loss_fn.compute(&preds, &certs, &0);

        // Compare: ticks that DO improve
        let preds_improve = make_predictions(&[&[1.0, 0.0], &[3.0, 0.0], &[5.0, 0.0]]);
        let (loss_improve, _) = loss_fn.compute(&preds_improve, &certs, &0);

        // No-improvement should have higher total loss due to thinking penalty
        assert!(
            loss_no_improve > loss_improve,
            "no-improvement loss ({loss_no_improve}) should exceed improving loss ({loss_improve})"
        );
    }

    #[test]
    fn thinking_loss_single_tick_no_penalty() {
        let loss_fn = ThinkingLoss::default();
        let preds = make_predictions(&[&[0.0, 5.0]]);
        let certs = uniform_certainties(1);
        let (loss, grads) = loss_fn.compute(&preds, &certs, &1);
        // With single tick, thinking_loss loop (1..k) doesn't run
        assert!(loss < 0.1);
        assert_eq!(grads.len(), 1);
    }

    #[test]
    fn thinking_loss_empty() {
        let loss_fn = ThinkingLoss::default();
        let (loss, grads) = loss_fn.compute(&[], &[], &0);
        assert_eq!(loss, 0.0);
        assert!(grads.is_empty());
    }

    #[test]
    fn thinking_loss_alpha_scales_penalty() {
        let preds = make_predictions(&[&[1.0, 0.0], &[1.0, 0.0]]); // no improvement
        let certs = uniform_certainties(2);

        let low_alpha = ThinkingLoss { alpha: 0.01, min_improvement: 0.01 };
        let high_alpha = ThinkingLoss { alpha: 1.0, min_improvement: 0.01 };

        let (loss_low, _) = low_alpha.compute(&preds, &certs, &0);
        let (loss_high, _) = high_alpha.compute(&preds, &certs, &0);

        assert!(loss_high > loss_low, "higher alpha should produce higher loss");
    }

    // ── ImaginationLoss ────────────────────────────────────────

    #[test]
    fn imagination_loss_splits_ticks() {
        // Use zero bonus so imagination doesn't subtract from base loss
        let loss_fn = ImaginationLoss { imagine_ratio: 0.5, imagination_bonus: 0.0 };
        // 4 ticks: first 2 = imagination, last 2 = commit
        let preds = make_predictions(&[
            &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0], &[0.0, 2.0, 0.0],
        ]);
        let certs = uniform_certainties(4);
        let (loss, grads) = loss_fn.compute(&preds, &certs, &1);

        assert!(loss > 0.0, "loss should be positive, got {loss}");
        assert_eq!(grads.len(), 4);
        // Imagination ticks (0, 1) should have zero gradient
        assert!(grads[0].iter().all(|&g| g == 0.0), "imagination tick 0 should have zero grad");
        assert!(grads[1].iter().all(|&g| g == 0.0), "imagination tick 1 should have zero grad");
    }

    #[test]
    fn imagination_loss_bonus_reduces_loss() {
        // When imagination phase improves predictions, bonus should reduce total loss
        let with_bonus = ImaginationLoss { imagine_ratio: 0.5, imagination_bonus: 1.0 };
        let no_bonus = ImaginationLoss { imagine_ratio: 0.5, imagination_bonus: 0.0 };

        // Tick 0 (imagine): bad. Tick 1 (commit): good. => positive improvement
        let preds = make_predictions(&[&[0.0, 0.0], &[0.0, 5.0]]);
        let certs = uniform_certainties(2);

        let (loss_bonus, _) = with_bonus.compute(&preds, &certs, &1);
        let (loss_no_bonus, _) = no_bonus.compute(&preds, &certs, &1);

        assert!(
            loss_bonus < loss_no_bonus,
            "imagination bonus should reduce loss: {loss_bonus} < {loss_no_bonus}"
        );
    }

    #[test]
    fn imagination_loss_empty() {
        let loss_fn = ImaginationLoss::default();
        let (loss, grads) = loss_fn.compute(&[], &[], &0);
        assert_eq!(loss, 0.0);
        assert!(grads.is_empty());
    }

    #[test]
    fn imagination_loss_single_tick() {
        let loss_fn = ImaginationLoss::default();
        // Single tick: commit_start should be 0 (can't imagine without committing)
        let preds = make_predictions(&[&[0.0, 5.0]]);
        let certs = uniform_certainties(1);
        let (loss, grads) = loss_fn.compute(&preds, &certs, &1);
        assert!(loss < 0.1);
        assert_eq!(grads.len(), 1);
    }

    // ── NaN / edge case handling ───────────────────────────────

    #[test]
    fn loss_handles_identical_logits() {
        // All logits equal => uniform softmax => well-defined CE
        let loss_fn = LastTickCE;
        let preds = make_predictions(&[&[0.0, 0.0, 0.0]]);
        let certs = uniform_certainties(1);
        let (loss, grads) = loss_fn.compute(&preds, &certs, &1);
        let expected = (3.0f32).ln(); // -ln(1/3)
        assert!((loss - expected).abs() < 0.01, "expected ~{expected}, got {loss}");
        assert!(grads[0].iter().all(|g| g.is_finite()));
    }

    #[test]
    fn loss_with_extreme_logits() {
        // Very large logits should not produce NaN/Inf due to softmax stability
        let loss_fn = LastTickCE;
        let preds = make_predictions(&[&[1000.0, -1000.0, 0.0]]);
        let certs = uniform_certainties(1);
        let (loss, grads) = loss_fn.compute(&preds, &certs, &0);
        assert!(loss.is_finite(), "loss should be finite, got {loss}");
        assert!(grads[0].iter().all(|g| g.is_finite()));
    }
}
