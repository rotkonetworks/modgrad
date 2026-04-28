//! `CerebellumService` — orchestration glue that mounts a
//! [`FrozenCerebellum`] as a **sibling service** of the regional brain
//! (per `docs/BRAIN_ARCHITECTURE.md` §7, option **(b)**).
//!
//! ## Architectural commitment
//!
//! The cerebellum is NOT a region in the regional iteration. The
//! cerebellum-region in [`RegionalConfig::eight_region_v2`] stays a
//! small placeholder; the heavy work (Qwen2.5 / BLT layer-cache
//! computation) happens HERE, outside the per-tick regional loop, and
//! the cortex regions read pre-computed signal from the cache.
//!
//! ## Lifecycle
//!
//! Once per **context window** the orchestrator calls
//! [`CerebellumService::set_context`], which asks the wrapped
//! [`FrozenCerebellum`] for a [`CerebellumCache`] (one
//! `[n_layers × n_positions × hidden_dim]` slab). Per **tick / per
//! token** the cortex calls [`CerebellumService::read_at`] which is a
//! thin wrapper around [`cerebellum_at_position`] with a stored
//! [`CerebProjection`].
//!
//! Cerebellum encode happens at most once per context window; cortex
//! reads are cheap projections (a single matvec). This is what makes
//! the cerebellum-as-LLM affordable: the LLM forward is amortised
//! across every tick of every region that consumes its output.
//!
//! ## What this is NOT
//!
//! - **Not a backward pass.** The cerebellum is frozen. There is no
//!   `backward(grad_out)` method; gradients only flow through
//!   `proj` (which the orchestrator owns separately, via
//!   `RegionalWeights::cereb_projection`).
//! - **Not multimodal-aware (v0).** [`read_at`] is position-based; for
//!   multimodal pooling use [`cerebellum_modality_pool`] directly with
//!   the cache exposed via [`CerebellumService::cache`].
//! - **Not part of `RegionalWeights`.** This deliberately stays
//!   orchestration-side so a single cerebellum can be shared across
//!   training threads / distillation runs without entangling its
//!   state with the trainable cortex weights.

use crate::cerebellum::{
    cerebellum_at_position, cerebellum_modality_pool, CerebProjection,
    CerebellumCache, FrozenCerebellum, Modality,
};

/// Sibling-service wrapper around a [`FrozenCerebellum`].
///
/// Owns the cerebellum implementation, the per-context-window cache,
/// and the cortex-side projection that maps the LLM's hidden dim into
/// the cortex's d_model.
///
/// **Threading:** `Send` so the orchestrator can move it across
/// threads. Reading methods take `&self` (the cache is immutable once
/// `set_context` returns); only `set_context` takes `&mut self`.
pub struct CerebellumService {
    cerebellum: Box<dyn FrozenCerebellum + Send>,
    cache: CerebellumCache,
    projection: CerebProjection,
}

impl CerebellumService {
    /// Construct from an owned cerebellum + projection.
    ///
    /// The projection's `frozen_output_dim` MUST equal the cerebellum's
    /// `hidden_dim()` (otherwise `project_out` would silently zero-pad
    /// or truncate). This is checked in `debug_assert` to keep release
    /// builds zero-overhead — production users should validate at
    /// construction.
    pub fn new(
        cerebellum: Box<dyn FrozenCerebellum + Send>,
        projection: CerebProjection,
    ) -> Self {
        debug_assert_eq!(
            projection.frozen_output_dim,
            cerebellum.hidden_dim(),
            "CerebellumService: projection.frozen_output_dim ({}) must equal cerebellum.hidden_dim() ({})",
            projection.frozen_output_dim,
            cerebellum.hidden_dim(),
        );
        // Empty cache: any read before `set_context` returns zero-fill.
        let hidden_dim = cerebellum.hidden_dim();
        let n_layers = cerebellum.n_layers();
        Self {
            cerebellum,
            cache: CerebellumCache {
                hidden_states: Vec::new(),
                hidden_dim,
                n_positions: 0,
                n_layers,
                modalities: None,
            },
            projection,
        }
    }

    /// Encode a fresh context window. Replaces the previous cache.
    ///
    /// Cost: one `FrozenCerebellum::encode_context_layers` call. For a
    /// `QwenCerebellum` over Qwen2.5-0.5B at 32 tokens that's ~2.7 MB
    /// of host memory and one full transformer forward; for `BltCerebellum`
    /// it's a BLT encoder pass. Either way, amortised across every
    /// per-tick read by cortex regions in the same context window.
    pub fn set_context(&mut self, token_ids: &[i64]) {
        self.cache = self.cerebellum.encode_context_layers(token_ids);
    }

    /// Attach modality tags to the current cache. Lets downstream
    /// `read_modality` calls discriminate text vs image VQ vs audio.
    /// Length must equal `cache.n_positions` or the call panics.
    pub fn set_modalities(&mut self, modalities: Vec<Modality>) {
        assert_eq!(
            modalities.len(),
            self.cache.n_positions,
            "CerebellumService::set_modalities: length {} != n_positions {}",
            modalities.len(),
            self.cache.n_positions,
        );
        self.cache.modalities = Some(modalities);
    }

    /// **Position-based read** (v0 cortex consumption pattern).
    ///
    /// Projects the layer-blended hidden state at `position` into the
    /// cortex d_model. Writes into `out[..proj.cortex_dim]`. Out-of-range
    /// positions zero-fill (matches [`cerebellum_at_position`]).
    ///
    /// **When to call this:** the standard per-tick / per-token cortex
    /// read. If your token sequence is `[t0, t1, …, tn]` and the cortex
    /// is processing token `i`, call `read_at(i, ...)`. The blending
    /// across transformer layers uses the projection's
    /// `layer_weight_logits` (softmaxed at read time).
    pub fn read_at(&self, position: usize, out: &mut [f32]) {
        cerebellum_at_position(&self.cache, &self.projection, position, out);
    }

    /// **Modality-pool read** (multimodal cortex consumption pattern).
    ///
    /// Mean-pools cache rows tagged with `modality`, projects to cortex
    /// d_model, returns true if at least one position matched. Use this
    /// when the cortex region wants a modality-specific summary rather
    /// than a per-position read (e.g. an "image-aware" region that pulls
    /// the pooled image-VQ representation regardless of where in the
    /// token sequence the image appeared).
    ///
    /// Returns false + zero-fills `out` when no position matches the
    /// requested modality.
    pub fn read_modality(&self, modality: Modality, out: &mut [f32]) -> bool {
        cerebellum_modality_pool(&self.cache, &self.projection, modality, out)
    }

    /// Read-only access to the underlying cache. Callers that need
    /// custom blending / pooling beyond `read_at` / `read_modality` can
    /// reach in directly without taking ownership.
    pub fn cache(&self) -> &CerebellumCache { &self.cache }

    /// Read-only access to the projection. Useful when an external
    /// optimizer wants to inspect or train the projection layer.
    pub fn projection(&self) -> &CerebProjection { &self.projection }

    /// Mutable access to the projection. Used by the orchestrator's
    /// optimizer step; keep this scoped (the `&mut` is the API
    /// affordance — callers don't accidentally rotate it).
    pub fn projection_mut(&mut self) -> &mut CerebProjection { &mut self.projection }

    /// Cortex d_model the projection emits into. Cortex regions
    /// reading via `read_at` should size their target buffer to this.
    pub fn cortex_dim(&self) -> usize { self.projection.cortex_dim }

    /// Number of positions currently encoded. Zero before
    /// `set_context` is called.
    pub fn n_positions(&self) -> usize { self.cache.n_positions }

    /// Hidden dim of the wrapped cerebellum. Constant for the lifetime
    /// of the service.
    pub fn hidden_dim(&self) -> usize { self.cache.hidden_dim }

    /// Number of transformer layers in the wrapped cerebellum.
    /// Constant for the lifetime of the service.
    pub fn n_layers(&self) -> usize { self.cache.n_layers }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cerebellum::RandomExpansion;

    /// `CerebellumService::set_context` populates the cache; `read_at`
    /// returns non-zero output proportional to the cerebellum's
    /// hidden state.
    ///
    /// Uses `RandomExpansion` (no GPU) wrapped in a hand-rolled
    /// `FrozenCerebellum` so the test runs on CI without HIP.
    #[test]
    fn service_read_at_returns_nonzero() {
        struct StubCereb {
            hidden_dim: usize,
            n_layers: usize,
        }
        impl FrozenCerebellum for StubCereb {
            fn hidden_dim(&self) -> usize { self.hidden_dim }
            fn n_layers(&self) -> usize { self.n_layers }
            fn encode_context_layers(&mut self, token_ids: &[i64]) -> CerebellumCache {
                let n = token_ids.len();
                let d = self.hidden_dim;
                let l = self.n_layers;
                // Deterministic non-zero fill: each layer/position/dim
                // gets a distinct value so the projection can't
                // accidentally collapse to zero.
                let mut hs = Vec::with_capacity(l * n * d);
                for li in 0..l {
                    for pi in 0..n {
                        for di in 0..d {
                            let v = (li as f32 + 1.0)
                                * (pi as f32 + 1.0)
                                * 0.01
                                * ((di + 1) as f32);
                            hs.push(v);
                        }
                    }
                }
                CerebellumCache {
                    hidden_states: hs,
                    hidden_dim: d,
                    n_positions: n,
                    n_layers: l,
                    modalities: None,
                }
            }
            fn forward(&mut self, _input: &[f32]) -> Vec<f32> {
                vec![0.0; self.hidden_dim]
            }
        }

        let cortex_dim = 16;
        let hidden = 8;
        let n_layers = 2;
        let stub = Box::new(StubCereb { hidden_dim: hidden, n_layers });
        let proj = CerebProjection::with_layers(cortex_dim, hidden, hidden, n_layers);

        let mut svc = CerebellumService::new(stub, proj);
        assert_eq!(svc.cortex_dim(), cortex_dim);
        assert_eq!(svc.hidden_dim(), hidden);
        assert_eq!(svc.n_layers(), n_layers);
        assert_eq!(svc.n_positions(), 0);

        let tokens: Vec<i64> = (0..4).collect();
        svc.set_context(&tokens);
        assert_eq!(svc.n_positions(), 4);

        let mut out = vec![0.0f32; cortex_dim];
        svc.read_at(2, &mut out);
        // Non-degenerate: at least one element non-zero (projection of a
        // non-zero hidden state through a Xavier-init Linear can't be
        // identically zero with overwhelming probability).
        let nonzero = out.iter().any(|&v| v.abs() > 1e-9);
        assert!(nonzero, "service read_at returned all-zero output: {:?}", out);
    }

    /// `read_at` with stub cerebellum that distinguishes positions
    /// produces measurably different cortex outputs at different
    /// positions — the seam carries position-discriminating signal.
    #[test]
    fn service_read_at_distinguishes_positions() {
        // RandomExpansion is position-agnostic (single forward), but
        // we can still build a multi-position cache by hand.
        let exp = RandomExpansion::new(8, 2, 0xBEEF);
        // Wrap in a stub that returns a hand-rolled multi-layer cache.
        struct PosStub { inner: RandomExpansion }
        impl FrozenCerebellum for PosStub {
            fn hidden_dim(&self) -> usize { self.inner.hidden_dim() }
            fn n_layers(&self) -> usize { 1 }
            fn encode_context_layers(&mut self, token_ids: &[i64]) -> CerebellumCache {
                let n = token_ids.len();
                let d = self.inner.hidden_dim();
                let mut hs = vec![0.0f32; n * d];
                for (pi, &t) in token_ids.iter().enumerate() {
                    let f = self.inner.forward(&vec![t as f32; 8]);
                    hs[pi * d..(pi + 1) * d].copy_from_slice(&f);
                }
                CerebellumCache {
                    hidden_states: hs,
                    hidden_dim: d,
                    n_positions: n,
                    n_layers: 1,
                    modalities: None,
                }
            }
            fn forward(&mut self, input: &[f32]) -> Vec<f32> {
                self.inner.forward(input)
            }
        }
        let stub = Box::new(PosStub { inner: exp });
        let proj = CerebProjection::with_layers(16, stub.hidden_dim(), stub.hidden_dim(), 1);
        let mut svc = CerebellumService::new(stub, proj);
        svc.set_context(&[1i64, 50, 100, 200]);

        let mut o0 = vec![0.0f32; 16];
        let mut o3 = vec![0.0f32; 16];
        svc.read_at(0, &mut o0);
        svc.read_at(3, &mut o3);

        let l2_diff: f32 = o0.iter().zip(o3.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        assert!(
            l2_diff > 1e-6,
            "service should produce position-distinct outputs; got identical {:?} vs {:?}",
            o0, o3,
        );
    }
}
