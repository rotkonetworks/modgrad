//! Top-down attention modulator.
//!
//! Implementation of `modgrad_ctm::modulator::Modulator` that
//! rewrites the observation tokens with a sigmoid-gated gain mask
//! derived from a downstream "attention" region's output. Replaces
//! the inline 50-line block in `examples/mazes/src/main.rs`
//! (around line 1099-1147) with a single struct.
//!
//! Lives in `modgrad-codec` because it depends on `VisualCortex`
//! (gating happens through `encode_gated` / `encode_gated_multistage`).
//! The trait it impls (`Modulator`) lives in `modgrad-ctm`. Dep
//! direction is correct: codec → ctm.
//!
//! Single-scale path: `encode_gated(raw, gain) → TokenInput.tokens`.
//! Multi-scale path: `encode_gated_multistage(raw, gain) → MultiScaleTokens`,
//! flattened by concatenating each scale's tokens (matches mazes'
//! `encode_multiscale_concat` shape).

use modgrad_ctm::modulator::{BrainContext, BrainFlow, Modulator, gain_from_attn};

use crate::retina::VisualCortex;

/// Top-down attention modulator. Reads
/// `state.region_outputs[attn_region_idx]` and rewrites
/// `flow.tokens` via `cortex.encode_gated*`.
///
/// Construction takes `cortex` by reference because the cortex
/// outlives the modulator (held by the training loop). The
/// modulator's `Send` bound (from `Modulator: Send`) requires the
/// reference to be `Send`, which it is when `VisualCortex: Sync`.
///
/// `n_pos` is the size of the gain mask. Caller computes it from
/// the encoder's geometry (typically `cortex.input_h * cortex.input_w`
/// for `preserve_spatial`, or `cortex.spatial_tokens(&raw).1` for the
/// generic case). Passing it explicitly avoids running an extra
/// `spatial_tokens` here just to count positions.
///
/// `alpha = 0.0` short-circuits to a no-op even when wired in,
/// matching `genome::PathwayGates`' zero-init meditation-ring contract.
pub struct TopdownMod<'c> {
    cortex: &'c VisualCortex,
    raw_pixels: Vec<f32>,
    alpha: f32,
    attn_region_idx: usize,
    n_pos: usize,
    multiscale: bool,
}

impl<'c> TopdownMod<'c> {
    pub fn new(
        cortex: &'c VisualCortex,
        raw_pixels: Vec<f32>,
        alpha: f32,
        attn_region_idx: usize,
        n_pos: usize,
        multiscale: bool,
    ) -> Self {
        Self { cortex, raw_pixels, alpha, attn_region_idx, n_pos, multiscale }
    }
}

impl<'c> Modulator for TopdownMod<'c> {
    fn pre_observation(&mut self, flow: &mut BrainFlow, ctx: &BrainContext) {
        if self.alpha == 0.0 { return; }  // explicit no-op short-circuit
        // Caller is responsible for `state` reflecting the post-scratch-
        // forward attention; the original mazes top-down block does this
        // by running `forward_cached` once before applying top-down. If
        // `attn_region_idx` is out of range, skip silently — mirrors the
        // original code's defensive `unwrap_or` style.
        let attn = match ctx.state.region_outputs.get(self.attn_region_idx) {
            Some(v) => v,
            None => return,
        };

        let gain = gain_from_attn(attn, self.n_pos, self.alpha);

        if self.multiscale {
            let multi = self.cortex.encode_gated_multistage(&self.raw_pixels, &gain);
            flow.tokens.clear();
            for s in &multi.scales {
                flow.tokens.extend_from_slice(&s.tokens);
            }
        } else {
            let new = self.cortex.encode_gated(&self.raw_pixels, &gain);
            *flow.tokens = new.tokens;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modgrad_ctm::graph::{RegionalConfig, RegionalState, RegionalWeights};

    /// `n_pos` for a `VisualCortex::random(h, w)` is the V4 spatial
    /// token count, NOT `h*w` — the conv stack reduces dimensions.
    /// Computed once via `spatial_tokens` for test setup.
    fn discover_n_pos(cortex: &VisualCortex) -> usize {
        let raw = vec![0.5f32; cortex.input_h * cortex.input_w * 3];
        cortex.spatial_tokens(&raw).1
    }

    /// `alpha = 0.0` is a hard no-op: gain ≡ 1 should leave tokens
    /// untouched. Matches the `PathwayGates` zero-init contract.
    #[test]
    fn alpha_zero_short_circuits() {
        let cortex = VisualCortex::random(20, 20);
        let n_pos = discover_n_pos(&cortex);

        let cfg = RegionalConfig::eight_region_small(64, 25, 3);
        let w = RegionalWeights::new(cfg);
        let state = RegionalState::new(&w);

        let mut td = TopdownMod::new(
            &cortex, vec![0.5f32; 20 * 20 * 3], 0.0, 1, n_pos, false,
        );
        let mut tokens = vec![1.0f32; 25 * 12];
        let snapshot = tokens.clone();
        let mut region_obs: Vec<Vec<f32>> = vec![];
        let mut flow = BrainFlow {
            tokens: &mut tokens, region_obs: &mut region_obs,
        };
        let ctx = BrainContext {
            cfg: &w.config, weights: &w, state: &state, outer_tick: 0,
        };
        td.pre_observation(&mut flow, &ctx);
        assert_eq!(*flow.tokens, snapshot,
            "alpha=0 must leave tokens unchanged");
    }

    /// `attn_region_idx` out of range silently no-ops. Mirrors the
    /// original mazes code's defensive `unwrap_or(7)` style — top-
    /// down should not crash a brain whose region layout doesn't
    /// expose an attention slot.
    #[test]
    fn out_of_range_attn_idx_no_ops() {
        let cortex = VisualCortex::random(20, 20);
        let n_pos = discover_n_pos(&cortex);

        let cfg = RegionalConfig::eight_region_small(64, 25, 3);
        let w = RegionalWeights::new(cfg);
        let state = RegionalState::new(&w);

        let mut td = TopdownMod::new(
            &cortex, vec![0.5f32; 20 * 20 * 3], 1.0, 999, n_pos, false,
        );
        let mut tokens = vec![1.0f32; 25 * 12];
        let snapshot = tokens.clone();
        let mut region_obs: Vec<Vec<f32>> = vec![];
        let mut flow = BrainFlow {
            tokens: &mut tokens, region_obs: &mut region_obs,
        };
        let ctx = BrainContext {
            cfg: &w.config, weights: &w, state: &state, outer_tick: 0,
        };
        td.pre_observation(&mut flow, &ctx);
        assert_eq!(*flow.tokens, snapshot,
            "out-of-range attn_region_idx must leave tokens unchanged");
    }

    /// With α > 0 and `state.region_outputs` populated (which
    /// `RegionalState::new` does via `start_activated.clone()`), the
    /// modulator runs `encode_gated` and replaces `flow.tokens` with
    /// the gated encoder output. The output length matches what
    /// `encode_gated` itself produces — sanity check that we wired
    /// the correct encoder method.
    #[test]
    fn non_zero_alpha_replaces_tokens() {
        let cortex = VisualCortex::random(20, 20);
        let n_pos = discover_n_pos(&cortex);

        let cfg = RegionalConfig::eight_region_small(64, n_pos, 3);
        let w = RegionalWeights::new(cfg);
        let state = RegionalState::new(&w);

        let raw = vec![0.5f32; 20 * 20 * 3];
        let mut td = TopdownMod::new(
            &cortex, raw.clone(), 1.0, 1, n_pos, false,
        );

        // Reference: encode_gated with the gain we expect TopdownMod
        // to compute internally.
        let attn = &state.region_outputs[1];
        let gain = gain_from_attn(attn, n_pos, 1.0);
        let expected = cortex.encode_gated(&raw, &gain);

        let mut tokens = vec![1.0f32; expected.tokens.len()];
        let mut region_obs: Vec<Vec<f32>> = vec![];
        let mut flow = BrainFlow {
            tokens: &mut tokens, region_obs: &mut region_obs,
        };
        let ctx = BrainContext {
            cfg: &w.config, weights: &w, state: &state, outer_tick: 0,
        };
        td.pre_observation(&mut flow, &ctx);
        assert_eq!(flow.tokens.len(), expected.tokens.len(),
            "TopdownMod output length matches encode_gated");
        // Bit-identical: same encoder, same input, same gain → same output.
        for (a, b) in flow.tokens.iter().zip(&expected.tokens) {
            assert!((a - b).abs() < 1e-6, "values must match encode_gated reference");
        }
    }
}
