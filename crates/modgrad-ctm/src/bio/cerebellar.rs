//! Cerebellar delta rule: prediction error → weight update.
//!
//! The cerebellum learns to predict observation from motor + proprioception.
//! Error signal = climbing fiber (observation - prediction).
//! ΔW = lr × error × input (the delta rule, not Hebbian).
//!
//! Also updates dopamine based on prediction error magnitude:
//! high error → DA burst, low error → DA decay toward baseline.

use super::TickContext;
use super::super::session::CtmSession;
use super::super::tick_state::*;

/// Run cerebellar prediction error + delta rule + dopamine update.
pub fn update(ctx: &TickContext, session: &mut CtmSession, tick_state: &mut CtmTickState) {
    let pred_error_mag: f32 = ctx.new_cereb.iter().zip(ctx.observation.iter())
        .map(|(&pred, &obs)| (obs - pred).powi(2))
        .sum::<f32>().sqrt()
        / ctx.observation.len().max(1) as f32;

    if pred_error_mag > 0.1 {
        // DA burst on prediction error
        tick_state.modulation[MOD_SYNC_SCALE] = (
            tick_state.modulation[MOD_SYNC_SCALE] * ctx.cfg.neuromod.da_error_alpha
            + ctx.cfg.neuromod.da_error_beta * (1.0 + pred_error_mag * 2.0)
        ).clamp(ctx.cfg.neuromod.da_min, ctx.cfg.neuromod.da_max);

        // Delta rule: ΔW = lr × (obs - pred) × pre
        if session.hebbian_enabled && !session.syn_deltas.is_empty() {
            let cereb_lr = ctx.cfg.neuromod.hebb_syn_lr * pred_error_mag.min(1.0);
            let cereb_syn = &ctx.weights.syn_cerebellum.linear;
            let in_dim = cereb_syn.in_dim;
            let n_cereb = ctx.cfg.cerebellum_layer.n_neurons.min(ctx.new_cereb.len());
            let syn_idx = 4; // syn_cerebellum is index 4

            for j in 0..n_cereb {
                let error_j = if j < ctx.observation.len() {
                    ctx.observation[j] - ctx.new_cereb[j]
                } else {
                    0.0
                };
                if error_j.abs() < 0.01 { continue; }
                for i in 0..in_dim.min(ctx.syn_in_cereb.len()) {
                    let idx = j * in_dim + i;
                    if idx < session.syn_deltas[syn_idx].len() {
                        session.syn_deltas[syn_idx][idx] += cereb_lr * error_j * ctx.syn_in_cereb[i];
                    }
                }
            }
        }
    } else {
        // No error → DA decays toward baseline
        tick_state.modulation[MOD_SYNC_SCALE] =
            tick_state.modulation[MOD_SYNC_SCALE] * ctx.cfg.neuromod.da_decay
            + (1.0 - ctx.cfg.neuromod.da_decay);
    }
}

/// Return the prediction error magnitude (useful for neuromod and active inference).
pub fn prediction_error(ctx: &TickContext) -> f32 {
    ctx.new_cereb.iter().zip(ctx.observation.iter())
        .map(|(&pred, &obs)| (obs - pred).powi(2))
        .sum::<f32>().sqrt()
        / ctx.observation.len().max(1) as f32
}
