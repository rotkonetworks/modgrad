//! Device-resident companion cache for `RegionalWeights`.
//!
//! Phase 5 of the GPU residency project (see
//! `memory/project_gpu_residency.md`). This module mirrors the
//! top-level Linears in `RegionalWeights` (connection synapses,
//! observation projection, output projection, exit gate, extra
//! prediction heads) as `LinearResident`s. The brain's hot path can
//! then dispatch via `matvec_resident` — zero PCIe transfers per
//! call once the cache is built.
//!
//! **Lifecycle.** Construct from a `RegionalWeights` via
//! `RegionalResidentCache::from_weights(&w)?`. After every optimizer
//! step that mutated host weights, call `sync_from_weights(&w)?` to
//! re-upload. The bench numbers
//! (`memory/feedback_residency_proof.md`) show ~5× sustained per-call
//! speedup on this hardware once the synapse + projection matvecs
//! stop round-tripping through PCIe.
//!
//! **Freshness witness.** The cache snapshots `weights.generation()`
//! at build/refresh time. Every dispatch goes through `cache.fresh(&w)`
//! which compares snapshots — a divergence returns
//! `Err(CacheError::Stale)` instead of silently dispatching against
//! pre-step weights. The in-tree optimizer paths
//! (`RegionalAdamW::step`, `RegionalGradients::apply`,
//! `apply_aux_gradients`) all bump the counter. Custom paths that
//! mutate weight fields directly must call `weights.bump_generation()`
//! — `pub(crate)` visibility on the field is what enforces it.
//!
//! **Sync discipline.** Resident dispatches happen inside a
//! `HipBatch` scope (see `memory/feedback_hip_queue_overflow.md`).
//! `LinearResident::forward` takes `&HipBatch` exactly so callers
//! can't forget — the GPU hang that crashed Xorg is a compile error.
//!
//! **Per-region inner CTM caches.** The `ctm_caches` field holds one
//! `CtmResidentCache` per region (NLM stages, MHA Q/K/V split, kv/q
//! projection, output projection, optional exit gate, synapse U-Net).
//! The outer freshness witness (`weights.generation()`) covers both
//! the top-level Linears and every per-region inner CTM, so a single
//! `cache.fresh(&w)` check protects the entire RegionalBrain
//! resident-forward dispatch chain.

#![cfg(feature = "rocm")]

use modgrad_compute::backend::ResidencyError;
use modgrad_compute::neuron::LinearResident;

use crate::ctm_resident::CtmResidentCache;
use crate::graph::RegionalWeights;

/// Error returned when a resident cache dispatch detects staleness
/// or when the underlying device runtime fails.
///
/// `Stale` carries the snapshot vs. current generation pair so the
/// caller can log the gap; the typical recovery path is
/// `cache.sync_from_weights(&w)` followed by retrying.
#[derive(Debug)]
pub enum CacheError {
    /// `cache.snapshot_gen != weights.generation` — the host weights
    /// have been mutated since this cache was built or last refreshed.
    /// Caller must `sync_from_weights` before re-attempting.
    Stale { snapshot: u64, current: u64 },
    /// Underlying residency error: device OOM, queue overflow, etc.
    Residency(ResidencyError),
}

impl From<ResidencyError> for CacheError {
    fn from(e: ResidencyError) -> Self { Self::Residency(e) }
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stale { snapshot, current } =>
                write!(f, "resident cache is stale: snapshot=gen{snapshot} current=gen{current}"),
            Self::Residency(e) => write!(f, "residency error: {e}"),
        }
    }
}

impl std::error::Error for CacheError {}

/// Device-resident companion to `RegionalWeights`. One
/// `LinearResident` per top-level Linear in the host weights. Built
/// once via `from_weights`; synced (re-uploaded) after every
/// optimizer step via `sync_from_weights`.
///
/// Per-region `CtmWeights` Linears are not in this cache yet — that
/// is the next slice. See module-level docs.
///
/// Fields are `pub(crate)` to force callers through `fresh()`, which
/// bundles the freshness check with the accessor view. Direct field
/// access would defeat the freshness witness.
pub struct RegionalResidentCache {
    pub(crate) connection_synapses: Vec<LinearResident>,
    pub(crate) obs_proj: LinearResident,
    pub(crate) output_proj: LinearResident,
    pub(crate) outer_exit_gate: Option<LinearResident>,
    pub(crate) extra_heads: Vec<LinearResident>,
    /// Per-region inner CTM caches — one entry per region in
    /// `weights.regions`. Each holds the per-region `CtmWeights`
    /// Linears (kv/q proj, MHA Q/K/V row-slices, MHA out, output_proj,
    /// optional exit gate), both NLM SuperLinears, and the synapse
    /// U-Net. Resident dispatch through this slice is what
    /// `RegionalBrain::forward_cached_resident` chains across regions.
    pub(crate) ctm_caches: Vec<CtmResidentCache>,
    /// Generation counter snapshot from `weights.generation()` at
    /// build/refresh time. Compared against `weights.generation()`
    /// on every dispatch to detect staleness.
    pub(crate) snapshot_gen: u64,
}

impl RegionalResidentCache {
    /// Allocate device buffers and upload all top-level Linears.
    /// Snapshots `w.generation()` so subsequent dispatches detect
    /// staleness via `fresh()` / `is_fresh()`.
    pub fn from_weights(w: &RegionalWeights) -> Result<Self, ResidencyError> {
        let connection_synapses = w.connection_synapses.iter()
            .map(LinearResident::from_linear)
            .collect::<Result<Vec<_>, _>>()?;
        let obs_proj = LinearResident::from_linear(&w.obs_proj)?;
        let output_proj = LinearResident::from_linear(&w.output_proj)?;
        let outer_exit_gate = match &w.outer_exit_gate {
            Some(g) => Some(LinearResident::from_linear(g)?),
            None => None,
        };
        let extra_heads = w.extra_heads.iter()
            .map(LinearResident::from_linear)
            .collect::<Result<Vec<_>, _>>()?;
        let ctm_caches = w.regions.iter()
            .map(CtmResidentCache::from_weights)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            connection_synapses, obs_proj, output_proj,
            outer_exit_gate, extra_heads, ctm_caches,
            snapshot_gen: w.generation(),
        })
    }

    /// Re-upload every Linear after an in-place optimizer step and
    /// re-snapshot the generation counter. After this returns,
    /// `is_fresh(&w)` is true.
    pub fn sync_from_weights(&mut self, w: &RegionalWeights) -> Result<(), ResidencyError> {
        debug_assert_eq!(self.connection_synapses.len(), w.connection_synapses.len());
        for (r, lin) in self.connection_synapses.iter_mut().zip(&w.connection_synapses) {
            r.sync_weights_from(lin)?;
        }
        self.obs_proj.sync_weights_from(&w.obs_proj)?;
        self.output_proj.sync_weights_from(&w.output_proj)?;
        if let (Some(r), Some(lin)) = (self.outer_exit_gate.as_mut(), w.outer_exit_gate.as_ref()) {
            r.sync_weights_from(lin)?;
        }
        debug_assert_eq!(self.extra_heads.len(), w.extra_heads.len());
        for (r, lin) in self.extra_heads.iter_mut().zip(&w.extra_heads) {
            r.sync_weights_from(lin)?;
        }
        debug_assert_eq!(self.ctm_caches.len(), w.regions.len());
        for (cache, rw) in self.ctm_caches.iter_mut().zip(&w.regions) {
            cache.sync_from_weights(rw)?;
        }
        self.snapshot_gen = w.generation();
        Ok(())
    }

    /// Quick check: is the snapshot generation in sync with `w`'s
    /// current generation? Cheap; doesn't cross the FFI boundary.
    pub fn is_fresh(&self, w: &RegionalWeights) -> bool {
        self.snapshot_gen == w.generation()
    }

    /// Last-snapshot generation. Useful for logging / telemetry.
    pub fn snapshot_generation(&self) -> u64 { self.snapshot_gen }

    /// Return a `FreshCache` view if and only if the cache is in sync
    /// with `w`. Errors with `CacheError::Stale` otherwise; the
    /// caller's recovery path is `sync_from_weights(&w)`.
    ///
    /// One freshness check, multiple uses: hold the returned
    /// `FreshCache` to call several `LinearResident`s without
    /// re-checking. The lifetime ties the view to both the cache and
    /// `w`, so a `&mut RegionalWeights` borrow is impossible while
    /// the view is alive — staleness can't open up under your feet.
    pub fn fresh<'a>(
        &'a self, w: &'a RegionalWeights,
    ) -> Result<FreshCache<'a>, CacheError> {
        if self.snapshot_gen != w.generation() {
            return Err(CacheError::Stale {
                snapshot: self.snapshot_gen,
                current: w.generation(),
            });
        }
        Ok(FreshCache { cache: self, _bind: std::marker::PhantomData })
    }

    /// Number of top-level Linears mirrored. Useful for debug logs
    /// ("device cache holds N Linears, expected M").
    pub fn n_linears(&self) -> usize {
        self.connection_synapses.len()
            + 2  // obs_proj, output_proj
            + self.outer_exit_gate.as_ref().map_or(0, |_| 1)
            + self.extra_heads.len()
    }
}

/// View on a `RegionalResidentCache` that has been freshness-checked
/// against a specific `RegionalWeights`. Accessing a `LinearResident`
/// through this view cannot dispatch against stale device weights
/// because the freshness check happened on construction.
///
/// The lifetime ties this view to both the cache and the weights;
/// you cannot `&mut weights` while a `FreshCache` is alive, so
/// generation cannot diverge under your feet.
pub struct FreshCache<'a> {
    cache: &'a RegionalResidentCache,
    _bind: std::marker::PhantomData<&'a RegionalWeights>,
}

impl<'a> FreshCache<'a> {
    pub fn obs_proj(&self) -> &'a LinearResident { &self.cache.obs_proj }
    pub fn output_proj(&self) -> &'a LinearResident { &self.cache.output_proj }
    pub fn connection_synapses(&self) -> &'a [LinearResident] {
        &self.cache.connection_synapses
    }
    pub fn outer_exit_gate(&self) -> Option<&'a LinearResident> {
        self.cache.outer_exit_gate.as_ref()
    }
    pub fn extra_heads(&self) -> &'a [LinearResident] { &self.cache.extra_heads }
    /// Per-region inner CTM caches in the same order as
    /// `weights.regions`. Used by `RegionalBrain::forward_cached_resident`
    /// to dispatch each region's `ctm_forward_resident` against its own
    /// resident weight buffers.
    pub fn ctm_caches(&self) -> &'a [CtmResidentCache] { &self.cache.ctm_caches }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{RegionalConfig, RegionalWeights};
    use modgrad_compute::backend::GpuVec;
    use modgrad_device::backend::HipBatch;
    use modgrad_device::backend::rocm::ffi::runtime_available;

    /// Build a small RegionalWeights, mirror it as a cache, and
    /// verify that running output_proj resident produces the same
    /// arithmetic as the host path (within FP error).
    #[test]
    fn cache_output_proj_matches_host() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let cfg = RegionalConfig::eight_region_small(64, 25, 3);
        let weights = RegionalWeights::new(cfg);
        let cache = RegionalResidentCache::from_weights(&weights)
            .expect("cache build");
        assert_eq!(cache.n_linears(),
            weights.connection_synapses.len() + 2
                + weights.outer_exit_gate.as_ref().map_or(0, |_| 1)
                + weights.extra_heads.len());
        assert!(cache.is_fresh(&weights), "build snapshot must match generation");

        // Synthetic input matching output_proj.in_dim.
        let in_dim = weights.output_proj.in_dim;
        let out_dim = weights.output_proj.out_dim;
        let host_x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();

        // Host path
        let mut host_y = vec![0.0f32; out_dim];
        weights.output_proj.forward_into(&host_x, &mut host_y);

        // Resident path — go through the freshness witness.
        let mut x_dev = GpuVec::try_hip(in_dim).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(out_dim).expect("alloc out");
        let batch = HipBatch::new();
        let fresh = cache.fresh(&weights).expect("fresh on first build");
        fresh.output_proj().forward(&batch, &x_dev, &mut out_dev)
            .expect("resident forward");
        batch.flush().expect("flush");
        let mut device_y = vec![0.0f32; out_dim];
        out_dev.copy_to_host(&mut device_y);

        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "output_proj host vs resident: max |Δ| = {max_diff}");
    }

    /// Mutating weights directly (without going through the optimizer
    /// path that bumps generation) leaves the cache stale; `fresh()`
    /// must report `Err(Stale)`. Calling `bump_generation()`
    /// explicitly trips the same detector. After `sync_from_weights`,
    /// `fresh()` returns Ok and dispatch matches the host path.
    #[test]
    fn cache_stale_detection_and_recovery() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let cfg = RegionalConfig::eight_region_small(64, 25, 3);
        let mut weights = RegionalWeights::new(cfg);
        let mut cache = RegionalResidentCache::from_weights(&weights)
            .expect("cache build");

        // Direct field mutation (simulating any path that bypasses the
        // in-tree optimizer): the cache is now stale on-device. The
        // generation counter is `pub(crate)` so external code MUST
        // call `bump_generation()` to invalidate dependents.
        for w in weights.output_proj.weight.iter_mut() { *w *= 1.5; }
        for b in weights.output_proj.bias.iter_mut() { *b += 0.1; }
        weights.bump_generation();

        // Freshness check refuses to hand out a view.
        match cache.fresh(&weights) {
            Err(CacheError::Stale { snapshot, current }) => {
                assert_eq!(snapshot, 0, "snapshot should be the build-time gen");
                assert_eq!(current, 1, "current should reflect bump_generation");
            }
            Err(CacheError::Residency(e)) => panic!("expected Stale, got Residency({e})"),
            Ok(_) => panic!("expected Stale, fresh() returned Ok"),
        }

        // Recovery: sync_from_weights re-uploads + re-snapshots; the
        // next `fresh()` returns Ok and dispatch matches host.
        cache.sync_from_weights(&weights).expect("sync");
        assert!(cache.is_fresh(&weights), "sync must restore freshness");

        let in_dim = weights.output_proj.in_dim;
        let out_dim = weights.output_proj.out_dim;
        let host_x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
        let mut host_y = vec![0.0f32; out_dim];
        weights.output_proj.forward_into(&host_x, &mut host_y);

        let mut x_dev = GpuVec::try_hip(in_dim).expect("x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(out_dim).expect("out");
        let batch = HipBatch::new();
        let fresh = cache.fresh(&weights).expect("fresh after sync");
        fresh.output_proj().forward(&batch, &x_dev, &mut out_dev).expect("forward");
        batch.flush().expect("flush");
        let mut fresh_y = vec![0.0f32; out_dim];
        out_dev.copy_to_host(&mut fresh_y);

        let fresh_diff = host_y.iter().zip(&fresh_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(fresh_diff < 1e-3,
            "after sync, expected match; got max |Δ| = {fresh_diff}");
    }

    /// The optimizer path bumps generation automatically. This is the
    /// "happy path" — caller never has to remember to invalidate.
    #[test]
    fn optimizer_step_bumps_generation() {
        let cfg = RegionalConfig::eight_region_small(64, 25, 3);
        let mut weights = RegionalWeights::new(cfg);
        let gen0 = weights.generation();

        let mut grads = crate::graph::RegionalGradients::zeros(&weights);
        // SGD path — RegionalGradients::apply must bump generation.
        grads.apply(&mut weights, 1e-3, 1.0);
        assert_eq!(weights.generation(), gen0 + 1,
            "RegionalGradients::apply must bump generation");

        // AdamW path — RegionalAdamW::step must bump generation.
        let mut opt = crate::graph::RegionalAdamW::new(&weights);
        let mut grads2 = crate::graph::RegionalGradients::zeros(&weights);
        opt.step(&mut weights, &mut grads2);
        assert_eq!(weights.generation(), gen0 + 2,
            "RegionalAdamW::step must bump generation");
    }
}
