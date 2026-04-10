//! Learning rules as composable filters.
//!
//! Each learning rule is a standalone function that takes the current tick state
//! and produces weight deltas. The forward pass calls whichever rules are enabled
//! in the config.
//!
//! To add a new learning rule:
//!   1. Create a new file in learning/
//!   2. Implement `fn apply(ctx: &TickContext, session: &mut CtmSession)`
//!   3. Add it to the `LearningPipeline`
//!
//! To swap rules: change the config. No code changes to forward.rs.

pub mod three_factor;
pub mod salience;
pub mod cerebellar;
pub mod neuromod;
pub mod hebbian;

use super::weights::CtmWeights;
use super::session::CtmSession;
use super::tick_state::*;

/// Everything a learning rule needs to see from the current tick.
/// Passed by reference — learning rules don't own any of this.
pub struct TickContext<'a> {
    pub weights: &'a CtmWeights,
    pub cfg: &'a CtmConfig,
    pub tick_idx: usize,

    // Region activations (post-NLM, this tick)
    pub new_input: &'a [f32],
    pub new_attn: &'a [f32],
    pub new_output: &'a [f32],
    pub new_motor: &'a [f32],
    pub new_cereb: &'a [f32],
    pub new_bg: &'a [f32],
    pub new_insula: &'a [f32],
    pub new_hippo: &'a [f32],

    // Synapse inputs (pre-synapse concatenated vectors)
    pub syn_in_input: &'a [f32],
    pub syn_in_attn: &'a [f32],
    pub syn_in_output: &'a [f32],
    pub syn_in_motor: &'a [f32],
    pub syn_in_cereb: &'a [f32],
    pub bg_input: &'a [f32],
    pub syn_in_insula: &'a [f32],
    pub syn_in_hippo: &'a [f32],

    // Observation (raw input)
    pub observation: &'a [f32],
}

/// Which learning rules to run during the forward pass.
/// Built from config, checked once per forward call.
pub struct LearningPipeline {
    pub three_factor: bool,
    pub cerebellar: bool,
    pub hebbian: bool,
    pub neuromod: bool,
}

impl LearningPipeline {
    /// Build from config: check what's enabled.
    pub fn from_session(session: &CtmSession) -> Self {
        Self {
            three_factor: session.hebbian_enabled && !session.syn_eligibility.is_empty(),
            cerebellar: session.hebbian_enabled && !session.syn_deltas.is_empty(),
            hebbian: false, // disabled by default, never worked
            neuromod: true, // always run neuromodulation dynamics
        }
    }

    /// Run all enabled learning rules for this tick.
    pub fn on_tick(&self, ctx: &TickContext, session: &mut CtmSession, tick_state: &mut CtmTickState) {
        if self.three_factor {
            three_factor::update(ctx, session, tick_state);
        }
        if self.cerebellar {
            cerebellar::update(ctx, session, tick_state);
        }
        if self.neuromod {
            neuromod::update(ctx, session, tick_state);
        }
        // hebbian intentionally omitted — enable explicitly if researching
    }
}
