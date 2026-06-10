//! `LatentThinker` — the BLT's deliberative "think over patches" stage as
//! a pluggable seam.
//!
//! North-star architecture (CTM-as-BLT-latent): the BLT pipeline is
//! `bytes → encoder → patches → LATENT → decoder → bytes`, and the latent
//! is the thinking stage. Instead of hardcoding a transformer there, make
//! it an interface. The existing transformer latent implements it (depth
//! from stacked layers, parallel/causal attention over patches); a
//! recurrent CTM latent implements it (depth from internal thinking ticks,
//! persistent neuron state). Swapping transformer→CTM is then a
//! constructor change, not a core edit (modularity discipline).

/// Consumes a sequence of patch representations and returns refined
/// ("thought") representations of the same shape. The implementor owns its
/// statefulness (a transformer's KV cache, a recurrent CTM's neuron trace)
/// and its notion of depth (stacked layers vs internal ticks) — the
/// interface is agnostic to both.
///
/// The boundary is substrate-neutral host slices, `[n_patches ×
/// patch_dim]` row-major — the lowest-friction bridge between the resident
/// BLT and the typed CTM during the Path-C migration. A device-resident
/// variant that keeps reps on-GPU is a perf follow-up.
pub trait LatentThinker {
    /// Per-call cache produced by `think_forward`, consumed by
    /// `think_backward` (the activations/state the backward needs).
    type Cache;
    /// Implementor-defined error (e.g. a device/residency error).
    type Error;

    /// Patch representation width.
    fn patch_dim(&self) -> usize;

    /// Forward: refine a patch sequence. Both `patches` and the returned
    /// reps are `[n_patches × patch_dim]` row-major.
    fn think_forward(
        &mut self,
        patches: &[f32],
        n_patches: usize,
    ) -> Result<(Vec<f32>, Self::Cache), Self::Error>;

    /// Backward: given `d_thought` (`[n_patches × patch_dim]`), return
    /// `d_patches` (same shape) and accumulate weight gradients into the
    /// implementor's own grad buffers.
    fn think_backward(
        &mut self,
        d_thought: &[f32],
        cache: &Self::Cache,
        n_patches: usize,
    ) -> Result<Vec<f32>, Self::Error>;
}
