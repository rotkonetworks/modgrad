//! Outer-loop optimizer interface — the hook SparseLoCo-family
//! algorithms hang off.
//!
//! Background. Local-update distributed optimizers (DiLoCo, SparseLoCo,
//! MuLoCo, …) run a sequence of *inner* steps on each peer with a
//! standard optimizer (AdamW, Muon) and then, at the end of each outer
//! *round*, communicate a compressed delta — the pseudo-gradient —
//! across peers and update the global model. The inner step function
//! is what modgrad's existing `TrainerLoop` already drives. This trait
//! is the *outer* hook: called with the pre-round parameters and the
//! post-inner-step parameters, it produces the next global parameter
//! state by however it likes (no-op pass-through, SparseLoCo aggregate,
//! gradient clipping, …).
//!
//! Why define the trait now, before anyone uses it. SparseLoCo-style
//! training is the likely shape modgrad grows into for distributed
//! runs. Wiring the interface now — even with only a no-op impl —
//! means callers who want to participate in future distributed
//! experiments can compose against a stable trait, and the day a
//! SparseLoCo implementation lands, no training loop rewires itself.
//! Zero-behavior-change placeholder; zero schema drift risk.
//!
//! The single-peer contract is particularly simple: inner steps
//! happen, outer optimizer is called once per round, the outer
//! optimizer *may* mutate the post-round parameters. [`NoOpOuter`]
//! does nothing, which is exactly right for single-peer: whatever
//! inner AdamW produced *is* the new global state.

/// End-of-round hook for local-update distributed optimizers.
///
/// `P` is the parameter type — e.g. `FfnWeights`, `RegionalWeights`,
/// or a generic `Vec<f32>` shard handle. Real implementations will
/// keep their own per-peer state (error-feedback buffers, moments,
/// communication clients) inside `self`; the trait only specifies
/// the *step*.
pub trait OuterOptimizer<P: Clone> {
    /// Called once per outer round, after the inner optimizer has
    /// run its `H` inner steps. Receives the parameters as they
    /// were at round start (`before`) and as inner steps produced
    /// them (`after`). The implementation may mutate `after` in
    /// place to whatever the next global parameter state should
    /// be — for example, applying a compressed+aggregated
    /// pseudo-gradient on top of `before`.
    ///
    /// Returning normally means the round succeeded; any error
    /// mode (aggregation timeout, validation rejection) is the
    /// implementation's concern and should be signalled through
    /// state the caller inspects, not through a panic.
    fn end_round(&mut self, before: &P, after: &mut P);

    /// Number of outer rounds this optimizer has processed. Default
    /// returns 0 — implementations with internal round counters can
    /// override. Used for logging and by tests.
    fn rounds(&self) -> u64 { 0 }
}

/// Single-peer no-op outer optimizer. The post-inner-step parameters
/// *are* the new global state, so nothing to do.
///
/// Present so that training binaries can take `&mut dyn OuterOptimizer<P>`
/// uniformly in both single-peer and multi-peer deployments — the
/// single-peer call site plugs in `NoOpOuter` with no conditional code.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpOuter {
    rounds: u64,
}

impl NoOpOuter {
    pub fn new() -> Self { Self { rounds: 0 } }
    /// Inherent accessor — avoids the "call resolution needs P"
    /// ambiguity that shows up when asking for rounds off a
    /// concrete `NoOpOuter` value. Trait-object callers go through
    /// `OuterOptimizer::rounds`.
    pub fn rounds(&self) -> u64 { self.rounds }
}

impl<P: Clone> OuterOptimizer<P> for NoOpOuter {
    fn end_round(&mut self, _before: &P, _after: &mut P) {
        self.rounds += 1;
    }
    fn rounds(&self) -> u64 { self.rounds }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct FakeParams(Vec<f32>);

    #[test]
    fn noop_leaves_params_untouched() {
        // The single-peer contract: the inner steps' result stands
        // as the new global state. Any edit to `after` would mean
        // NoOp silently changed training behavior.
        let mut opt = NoOpOuter::new();
        let before = FakeParams(vec![1.0, 2.0, 3.0]);
        let mut after = FakeParams(vec![1.1, 2.1, 3.1]);
        let expected_after = after.clone();

        opt.end_round(&before, &mut after);

        assert_eq!(after, expected_after,
            "NoOpOuter must not mutate `after` — single-peer contract");
        assert_eq!(opt.rounds(), 1);
    }

    #[test]
    fn noop_counts_rounds_across_multiple_calls() {
        let mut opt = NoOpOuter::new();
        let before = FakeParams(vec![0.0]);
        for _ in 0..5 {
            let mut after = before.clone();
            opt.end_round(&before, &mut after);
        }
        assert_eq!(opt.rounds(), 5);
    }

    #[test]
    fn trait_is_object_safe() {
        // Type-check that `dyn OuterOptimizer<P>` compiles — catches
        // a future refactor that adds a method with `Self` as a
        // receiver type or otherwise breaks dyn compatibility.
        // (SparseLoCo wiring will want to hold a boxed optimizer
        // so different impls can be swapped at CLI time.)
        let mut opt: Box<dyn OuterOptimizer<FakeParams>> = Box::new(NoOpOuter::new());
        let before = FakeParams(vec![0.0]);
        let mut after = FakeParams(vec![0.0]);
        opt.end_round(&before, &mut after);
        assert_eq!(opt.rounds(), 1);
    }
}
