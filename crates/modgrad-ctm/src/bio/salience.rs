//! Salience: how important is this moment for learning?
//!
//! RPE (reward prediction error) × motor conflict.
//! High salience → learn hard. Low salience → skip.
//!
//! Biological basis: anterior insula + anterior cingulate cortex.

/// Salience signal.
pub struct Salience {
    /// Raw salience [0, ~2].
    pub value: f32,
    /// Gated: 0 if below threshold, else value.
    pub gate: f32,
    pub motor_conflict: f32,
    pub reward_rpe: f32,
}

/// Compute salience from dopamine level and motor output.
///
/// `dopamine`: current DA
/// `reward_baseline`: running average DA
/// `motor_output`: motor region activations (conflict = close top-2)
pub fn compute(dopamine: f32, reward_baseline: f32, motor_output: &[f32]) -> Salience {
    let reward_rpe = (dopamine - reward_baseline).abs();

    let mut sorted: Vec<f32> = motor_output.iter().map(|x| x.abs()).collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let motor_conflict = if sorted.len() >= 2 {
        1.0 / (1.0 + (sorted[0] - sorted[1]) * 5.0)
    } else {
        1.0
    };

    let value = (reward_rpe * (0.3 + 0.7 * motor_conflict)).clamp(0.0, 2.0);
    let gate = if value > 0.05 { value } else { 0.0 };

    Salience { value, gate, motor_conflict, reward_rpe }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_rpe_high_salience() {
        let s = compute(2.0, 1.0, &[0.5, 0.5]);
        assert!(s.value > 0.5);
        assert!(s.gate > 0.0);
    }

    #[test]
    fn low_surprise_clear_action() {
        let s = compute(1.0, 1.0, &[0.9, 0.1]);
        assert!(s.value < 0.1);
    }
}
