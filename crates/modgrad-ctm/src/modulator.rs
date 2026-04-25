//! Modulator filter trait — composable cross-cutting hooks into the
//! brain forward + backward path. The cash-out of
//! `memory/feedback_modularity.md`: every new mechanism (top-down
//! V4→V2/V1, frozen cerebellum, GHL retinal plasticity, future
//! pulvinar/subcortical) gets to live as a `Modulator` impl that
//! composes with `andThen`, instead of editing
//! `Connection`/`RegionalConfig`/`RegionalWeights` core types or
//! cloning whole forward functions.
//!
//! Mirrors the SaaF `Filter` pattern already in `modgrad-io::service`
//! (used for the LLM generation pipeline) — same idiom, applied to
//! the brain forward layer instead of LLM logit modification.
//!
//! ## Trait surface (all hooks default to no-op)
//!
//! Five hooks shipped together so the design is correct for
//! top-down (forward token rewrite), frozen-cerebellum (post-region
//! overwrite), and GHL (post-backward weight write) — the three
//! motivating cases. Concrete impls land per-crate-where-deps-live:
//! `TopdownMod` in `modgrad-codec` (it needs `VisualCortex`);
//! `FrozenCerebMod` in `modgrad-ctm` (it needs `FrozenCerebellum`
//! which already lives here); `GhlMod` in `modgrad-codec`.
//!
//! - `pre_observation` — runs once before the outer-tick loop.
//!   Modify `flow.tokens`. Use case: top-down attention rewriting
//!   the visual input gain.
//! - `pre_region` — per outer tick, after the connection synapse
//!   forward populates `flow.region_obs`, before per-region CTM
//!   dispatches. Use case: pulvinar / thalamic gating.
//! - `post_region` — per outer tick, after the per-region CTM
//!   dispatches, before global sync. Mutates `state.region_outputs`.
//!   Use case: frozen cerebellum override.
//! - `backward_pre_observation` — backward path: project `d_tokens`
//!   if a `pre_observation` modulator participates in backprop.
//! - `post_backward` — at the end of `backward_with_input_grad`,
//!   sees `d_obs`. Use case: GHL retinal plasticity
//!   (`encoder.ghl_step(d_obs, …)`).
//!
//! ## Status of integration
//!
//! Trait surface lands here. Brain forward/backward integration
//! (`forward_cached_with`, `backward_with_input_grad_with`) lands
//! per modulator follow-up:
//! - TopdownMod → `pre_observation` only, can also be invoked
//!   directly as `topdown.pre_observation(&mut flow, &ctx)` without
//!   any brain integration plumbing.
//! - FrozenCerebMod → requires `forward_cached_with` (post_region
//!   hook needs to fire inside the brain's outer-tick loop).
//! - GhlMod → requires `backward_with_input_grad_with`.
//!
//! Each is a separate PR so the diff stays bisectable.

use crate::graph::{RegionalConfig, RegionalState, RegionalWeights};

/// Mutable carrier flowing through filter chain. References to the
/// observation buffer and the per-region projected inputs that
/// modulators may rewrite.
pub struct BrainFlow<'a> {
    /// The flat (possibly multi-scale-concatenated) observation
    /// buffer. `pre_observation` rewrites this before MHA projection.
    pub tokens: &'a mut Vec<f32>,
    /// Per-region projected inputs, one Vec per region. Populated
    /// by the connection-synapse phase; `pre_region` may rewrite
    /// before per-region CTM dispatches consume them. Empty during
    /// `pre_observation` (the connection phase hasn't run yet).
    pub region_obs: &'a mut [Vec<f32>],
}

/// Read-only context handed to every modulator hook.
pub struct BrainContext<'a> {
    pub cfg: &'a RegionalConfig,
    pub weights: &'a RegionalWeights,
    pub state: &'a RegionalState,
    /// Outer tick index. `0` for `pre_observation` (which runs
    /// before the loop); the actual tick index for `pre_region` /
    /// `post_region`.
    pub outer_tick: usize,
}

/// Cross-cutting hook into the brain forward + backward path.
/// Default no-op for every method; override only the hooks the
/// modulator actually needs.
///
/// `Send` because the brain forward dispatches per-region in
/// parallel via rayon; modulators that hold references must respect
/// that.
pub trait Modulator: Send {
    fn pre_observation(
        &mut self, _flow: &mut BrainFlow, _ctx: &BrainContext,
    ) {}

    fn pre_region(
        &mut self, _flow: &mut BrainFlow, _ctx: &BrainContext,
    ) {}

    fn post_region(
        &mut self, _state: &mut RegionalState,
        _flow: &mut BrainFlow, _ctx: &BrainContext,
    ) {}

    fn backward_pre_observation(
        &mut self, _d_tokens: &mut [f32], _ctx: &BrainContext,
    ) {}

    fn post_backward(
        &mut self, _d_obs: &[f32], _ctx: &BrainContext,
    ) {}
}

/// Composition: run `a`'s hook then `b`'s for each callback.
/// Mirrors `modgrad-io`'s `Filtered<F, S>`.
pub struct Chain<A, B> { pub a: A, pub b: B }

impl<A: Modulator, B: Modulator> Modulator for Chain<A, B> {
    fn pre_observation(&mut self, f: &mut BrainFlow, c: &BrainContext) {
        self.a.pre_observation(f, c);
        self.b.pre_observation(f, c);
    }
    fn pre_region(&mut self, f: &mut BrainFlow, c: &BrainContext) {
        self.a.pre_region(f, c);
        self.b.pre_region(f, c);
    }
    fn post_region(
        &mut self, s: &mut RegionalState,
        f: &mut BrainFlow, c: &BrainContext,
    ) {
        self.a.post_region(s, f, c);
        self.b.post_region(s, f, c);
    }
    fn backward_pre_observation(&mut self, d: &mut [f32], c: &BrainContext) {
        self.a.backward_pre_observation(d, c);
        self.b.backward_pre_observation(d, c);
    }
    fn post_backward(&mut self, d: &[f32], c: &BrainContext) {
        self.a.post_backward(d, c);
        self.b.post_backward(d, c);
    }
}

/// Builder for chains. `a.and_then(b)` produces `Chain { a, b }`.
pub trait ModulatorExt: Modulator + Sized {
    fn and_then<B: Modulator>(self, b: B) -> Chain<Self, B> {
        Chain { a: self, b }
    }
}

impl<M: Modulator> ModulatorExt for M {}

/// No-op modulator. Default for callers that don't want any
/// cross-cutting behavior. `forward_cached_with(brain, …, &mut Identity)`
/// is identical to `forward_cached(brain, …)` once the integration
/// helper lands.
pub struct Identity;

impl Modulator for Identity {}

/// Helper math used by `TopdownMod` (in `modgrad-codec`) and any
/// future top-down-style modulator. Placed here so callers don't
/// need a `modgrad-codec` dep just to compute the gain.
///
/// Computes `gain[i] = 1 + alpha · (sigmoid(attn[i % attn.len()]) - 0.5)`.
/// - `alpha = 0.0` → gain ≡ 1 (identity, no-op even when wired in).
/// - `alpha = 1.0` → range `[0.5, 1.5]`, sigmoid centred on no-op.
/// Matches `genome::PathwayGates`' zero-init meditation-ring contract:
/// every new pathway starts at no-op.
pub fn gain_from_attn(attn: &[f32], n_pos: usize, alpha: f32) -> Vec<f32> {
    let al = attn.len().max(1);
    let mut gain = vec![0.0f32; n_pos];
    for i in 0..n_pos {
        let v = attn[i % al];
        let s = 1.0 / (1.0 + (-v).exp());
        gain[i] = 1.0 + alpha * (s - 0.5);
    }
    gain
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity is a no-op that compiles as a Modulator.
    #[test]
    fn identity_compiles() {
        fn _accepts<M: Modulator>(_: M) {}
        _accepts(Identity);
    }

    /// `gain_from_attn` math invariants.
    #[test]
    fn gain_math() {
        // alpha = 0 — identity gain regardless of attn.
        let g = gain_from_attn(&[1.0, -1.0, 5.0, -5.0], 4, 0.0);
        for v in &g { assert!((v - 1.0).abs() < 1e-6); }

        // alpha = 1, attn = 0 → gain = 1.0 exactly.
        let g = gain_from_attn(&[0.0], 1, 1.0);
        assert!((g[0] - 1.0).abs() < 1e-6);

        // alpha = 1, saturating + → 1.5.
        let g = gain_from_attn(&[100.0], 1, 1.0);
        assert!((g[0] - 1.5).abs() < 1e-3);

        // alpha = 1, saturating − → 0.5.
        let g = gain_from_attn(&[-100.0], 1, 1.0);
        assert!((g[0] - 0.5).abs() < 1e-3);

        // Wrap indexing.
        let g = gain_from_attn(&[2.0], 3, 1.0);
        let expected = 1.0 + (1.0 / (1.0 + (-2.0f32).exp()) - 0.5);
        for v in &g { assert!((v - expected).abs() < 1e-6); }
    }

    /// `Chain` composition fires hooks in order.
    #[test]
    fn chain_calls_in_order() {
        use std::sync::{Arc, Mutex};

        struct Recorder { name: char, log: Arc<Mutex<Vec<char>>> }
        impl Modulator for Recorder {
            fn pre_observation(&mut self, _f: &mut BrainFlow, _c: &BrainContext) {
                self.log.lock().unwrap().push(self.name);
            }
            fn pre_region(&mut self, _f: &mut BrainFlow, _c: &BrainContext) {
                self.log.lock().unwrap().push(self.name.to_ascii_uppercase());
            }
        }

        let log = Arc::new(Mutex::new(Vec::new()));
        let mut chain = Recorder { name: 'a', log: log.clone() }
            .and_then(Recorder { name: 'b', log: log.clone() })
            .and_then(Recorder { name: 'c', log: log.clone() });

        // Build minimal context. We don't need a real brain — any
        // RegionalWeights / RegionalState suffices for the type.
        use crate::graph::{RegionalConfig, RegionalWeights, RegionalState};
        let cfg = RegionalConfig::eight_region_small(64, 25, 3);
        let w = RegionalWeights::new(cfg);
        let s = RegionalState::new(&w);
        let mut tokens = vec![0.0f32; 1];
        let mut region_obs: Vec<Vec<f32>> = vec![];
        let mut flow = BrainFlow { tokens: &mut tokens, region_obs: &mut region_obs };
        let ctx = BrainContext { cfg: &w.config, weights: &w, state: &s, outer_tick: 0 };

        // Both hooks fire — verify order.
        chain.pre_observation(&mut flow, &ctx);
        chain.pre_region(&mut flow, &ctx);

        assert_eq!(*log.lock().unwrap(),
            vec!['a', 'b', 'c', 'A', 'B', 'C']);
    }

    /// Identity composed in a chain doesn't disturb other modulators.
    #[test]
    fn chain_with_identity_pass_through() {
        use std::sync::{Arc, Mutex};

        struct Counter(Arc<Mutex<usize>>);
        impl Modulator for Counter {
            fn pre_observation(&mut self, _f: &mut BrainFlow, _c: &BrainContext) {
                *self.0.lock().unwrap() += 1;
            }
        }
        let n = Arc::new(Mutex::new(0));
        let mut chain = Identity.and_then(Counter(n.clone())).and_then(Identity);

        use crate::graph::{RegionalConfig, RegionalWeights, RegionalState};
        let cfg = RegionalConfig::eight_region_small(64, 25, 3);
        let w = RegionalWeights::new(cfg);
        let s = RegionalState::new(&w);
        let mut tokens = vec![0.0f32; 1];
        let mut region_obs: Vec<Vec<f32>> = vec![];
        let mut flow = BrainFlow { tokens: &mut tokens, region_obs: &mut region_obs };
        let ctx = BrainContext { cfg: &w.config, weights: &w, state: &s, outer_tick: 0 };

        chain.pre_observation(&mut flow, &ctx);
        assert_eq!(*n.lock().unwrap(), 1, "Counter should fire exactly once");
    }
}
