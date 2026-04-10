//! Three-factor REINFORCE with Titans-style dynamics.
//!
//! Eligibility trace × advantage × learning rate, gated by salience.
//! Only updates motor pathway synapses (3=output→motor, 5=BG).
//!
//! Titans (Behrouz et al. 2024) dynamics:
//!   S_t = η_t · S_{t-1} + (1-η_t) · post · pre    (eligibility with adaptive decay)
//!   ΔW  = θ_t · advantage · S_t                     (salience-scaled update)
//!   α_t applied during apply_syn_deltas              (adaptive forgetting)
//!
//! Best result: +26pp on 7×7 maze with salience gating.

use super::{TickContext, salience};
use super::super::session::CtmSession;
use super::super::tick_state::*;

/// Run three-factor REINFORCE update for the current tick.
/// Modifies session.syn_eligibility, session.syn_deltas, session.syn_salience.
pub fn update(ctx: &TickContext, session: &mut CtmSession, tick_state: &mut CtmTickState) {
    if !session.hebbian_enabled || session.syn_eligibility.is_empty() { return; }

    let da = tick_state.modulation[MOD_SYNC_SCALE];
    let elig_decay_base = ctx.cfg.neuromod.bg_elig_decay;

    // Compute salience
    let sal = salience::compute(ctx, da,
        session.syn_reward_baseline.get(3).copied().unwrap_or(1.0));

    // Data-dependent η_t: eligibility decay
    let eta = elig_decay_base + (1.0 - elig_decay_base) * 0.5 * sal.value.min(1.0);

    // Data-dependent θ_t: learning rate
    let theta = ctx.cfg.neuromod.hebb_syn_lr * sal.gate;

    // Only motor pathway synapses
    let motor_synapses: [(usize, &[f32], &[f32]); 2] = [
        (3, ctx.syn_in_motor, ctx.new_motor),
        (5, ctx.bg_input, ctx.new_bg),
    ];

    for &(s, pre, post) in &motor_synapses {
        if s >= session.syn_eligibility.len() { continue; }
        let in_dim = ctx.weights.synapse_refs()[s].linear.in_dim;
        let out_dim = ctx.weights.synapse_refs()[s].linear.out_dim;

        // Eligibility trace: S_t = η · S_{t-1} + (1-η) · post · pre
        for j in 0..out_dim.min(post.len()) {
            if post[j].abs() < 0.01 { continue; }
            for i in 0..in_dim.min(pre.len()) {
                let idx = j * in_dim + i;
                if idx < session.syn_eligibility[s].len() {
                    session.syn_eligibility[s][idx] =
                        eta * session.syn_eligibility[s][idx]
                        + (1.0 - eta) * post[j] * pre[i];
                }
            }
        }

        // Weight delta: θ · advantage · eligibility
        let reward = da;
        let bl = session.syn_reward_baseline[s];
        session.syn_reward_baseline[s] = 0.99 * bl + 0.01 * reward;
        let advantage = reward - bl;

        if advantage.abs() > ctx.cfg.neuromod.bg_reward_threshold && theta > 0.0 {
            for idx in 0..session.syn_deltas[s].len().min(session.syn_eligibility[s].len()) {
                session.syn_deltas[s][idx] += theta * advantage * session.syn_eligibility[s][idx];
            }
        }

        // Track per-synapse salience for adaptive forgetting
        if s < session.syn_salience.len() {
            session.syn_salience[s] = 0.95 * session.syn_salience[s] + 0.05 * sal.value;
        }
    }
}
