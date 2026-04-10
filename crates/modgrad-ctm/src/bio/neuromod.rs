//! Neuromodulation dynamics: dopamine, serotonin, norepinephrine, ACh.
//!
//! Updates the modulation vector based on prediction error, arousal,
//! and active inference (curiosity/anxiety from Friston's free energy).
//!
//! Runs every tick regardless of learning mode.

use super::TickContext;
use super::super::session::CtmSession;
use super::super::tick_state::*;

/// Update neuromodulation: active inference (curiosity/anxiety), serotonin, arousal.
/// Call AFTER cerebellar update (needs pred_error_mag).
pub fn update(ctx: &TickContext, _session: &mut CtmSession, tick_state: &mut CtmTickState) {
    let pred_error_mag = super::cerebellar::prediction_error(ctx);

    let da = tick_state.modulation[MOD_SYNC_SCALE];
    let ne = tick_state.modulation[MOD_AROUSAL];
    let calm = (1.0 - ne / 2.0).max(0.0).min(1.0);
    let stress = (ne / 2.0).min(1.0);

    // Curiosity: high prediction error + low arousal = "I want to understand this"
    let curiosity = pred_error_mag * da * calm;
    // Anxiety: high prediction error + high arousal = "I need to act NOW"
    let anxiety = pred_error_mag * da * stress;

    // Intrinsic motivation: learning progress (Schmidhuber's "fun")
    // Approximated from tick_state stability
    let learning_progress = 0.0f32; // TODO: track across ticks via tick_state

    let curiosity = curiosity + 0.5 * learning_progress;

    // EMA updates
    tick_state.modulation[MOD_CURIOSITY] =
        0.7 * tick_state.modulation[MOD_CURIOSITY] + 0.3 * curiosity;
    tick_state.modulation[MOD_ANXIETY] =
        0.7 * tick_state.modulation[MOD_ANXIETY] + 0.3 * anxiety;

    // Arousal: curiosity → mild engagement, anxiety → fight-or-flight
    tick_state.modulation[MOD_AROUSAL] = (ne
        + 0.05 * curiosity
        + 0.15 * anxiety
    ).clamp(0.1, 2.0);

    // Serotonin: learning feels good, anxiety feels bad
    tick_state.modulation[MOD_GATE] = (tick_state.modulation[MOD_GATE]
        + 0.02 * learning_progress
        - 0.01 * anxiety
    ).clamp(0.1, 2.0);
}
