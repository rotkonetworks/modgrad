//! Salience computation: RPE × motor conflict.
//!
//! Determines how "important" the current tick is for learning.
//! High salience → learn hard. Low salience → skip update.
//!
//! Biological basis: anterior insula (interoceptive surprise) +
//! anterior cingulate cortex (response conflict monitoring).

use super::TickContext;

/// Salience signal from reward prediction error × motor conflict.
pub struct Salience {
    pub value: f32,
    pub gate: f32,       // 0.0 if below threshold, else value
    pub motor_conflict: f32,
    pub reward_rpe: f32,
}

/// Compute salience for the current tick.
pub fn compute(ctx: &TickContext, da: f32, reward_baseline: f32) -> Salience {
    let reward_rpe = (da - reward_baseline).abs();

    let mut motor_sorted: Vec<f32> = ctx.new_motor.iter().map(|x| x.abs()).collect();
    motor_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let motor_conflict = if motor_sorted.len() >= 2 {
        1.0 / (1.0 + (motor_sorted[0] - motor_sorted[1]) * 5.0)
    } else {
        1.0
    };

    let value = (reward_rpe * (0.3 + 0.7 * motor_conflict)).clamp(0.0, 2.0);
    let gate = if value > 0.05 { value } else { 0.0 };

    Salience { value, gate, motor_conflict, reward_rpe }
}
