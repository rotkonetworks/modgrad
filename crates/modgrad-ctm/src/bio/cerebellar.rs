//! Cerebellar delta rule: prediction error → weight deltas.
//!
//! The cerebellum learns to predict the next observation from its own output.
//! Error = observation - prediction. ΔW = lr × error × input (delta rule).
//!
//! Pure functions on slices. No runtime types.

/// Compute prediction error magnitude: RMSE between predicted and observed.
pub fn prediction_error(predicted: &[f32], observed: &[f32]) -> f32 {
    let n = predicted.len().min(observed.len()).max(1);
    predicted.iter().zip(observed)
        .map(|(&p, &o)| (o - p).powi(2))
        .sum::<f32>().sqrt()
        / n as f32
}

/// Compute delta rule weight updates: ΔW[j,i] = lr × error[j] × input[i].
///
/// `predicted`: cerebellum output [out_dim]
/// `observed`: actual observation [out_dim]
/// `input`: input to the synapse [in_dim]
/// `lr`: learning rate (scaled by prediction error magnitude)
///
/// Returns weight deltas [out_dim × in_dim], row-major.
pub fn delta_rule(
    predicted: &[f32],
    observed: &[f32],
    input: &[f32],
    lr: f32,
) -> Vec<f32> {
    let out_dim = predicted.len().min(observed.len());
    let in_dim = input.len();
    let mut deltas = vec![0.0f32; out_dim * in_dim];

    for j in 0..out_dim {
        let error_j = observed[j] - predicted[j];
        if error_j.abs() < 0.01 { continue; }
        let row = j * in_dim;
        for i in 0..in_dim {
            deltas[row + i] = lr * error_j * input[i];
        }
    }
    deltas
}

/// Dopamine update based on prediction error.
///
/// High error → DA burst (surprise signal).
/// Low error → DA decays toward baseline (1.0).
///
/// Returns new dopamine level.
pub fn dopamine_update(
    current_da: f32,
    pred_error: f32,
    da_decay: f32,
    da_burst_alpha: f32,
    da_burst_beta: f32,
    da_min: f32,
    da_max: f32,
) -> f32 {
    if pred_error > 0.1 {
        (current_da * da_burst_alpha + da_burst_beta * (1.0 + pred_error * 2.0))
            .clamp(da_min, da_max)
    } else {
        current_da * da_decay + (1.0 - da_decay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_error_zero_on_match() {
        let v = vec![1.0, 2.0, 3.0];
        assert!(prediction_error(&v, &v) < 1e-6);
    }

    #[test]
    fn delta_rule_produces_updates() {
        let pred = vec![0.0, 0.0];
        let obs = vec![1.0, -1.0];
        let input = vec![1.0, 0.5];
        let deltas = delta_rule(&pred, &obs, &input, 0.1);
        assert_eq!(deltas.len(), 4); // 2×2
        assert!((deltas[0] - 0.1).abs() < 1e-6); // error=1.0 × input=1.0 × lr=0.1
        assert!((deltas[2] - (-0.1)).abs() < 1e-6); // error=-1.0 × input=1.0 × lr=0.1
    }

    #[test]
    fn dopamine_bursts_on_error() {
        let da = dopamine_update(1.0, 0.5, 0.95, 0.7, 0.3, 0.1, 3.0);
        assert!(da > 1.0); // should burst
        let da_calm = dopamine_update(1.5, 0.01, 0.95, 0.7, 0.3, 0.1, 3.0);
        assert!(da_calm < 1.5); // should decay toward 1.0
    }
}
