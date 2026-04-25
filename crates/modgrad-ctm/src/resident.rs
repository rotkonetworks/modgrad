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
//! `RegionalResidentCache::from_weights(&w)?`. After every AdamW
//! step that mutated host weights, call `sync_from_weights(&w)?` to
//! re-upload. The bench numbers
//! (`memory/feedback_residency_proof.md`) show ~5× sustained per-call
//! speedup on this hardware once the synapse + projection matvecs
//! stop round-tripping through PCIe.
//!
//! **Sync discipline.** Resident dispatches happen inside a
//! `HipBatch` scope (see `memory/feedback_hip_queue_overflow.md`).
//! The `forward_*_resident` helpers in this module take `&HipBatch`
//! exactly so callers can't forget — the GPU hang that crashed Xorg
//! is a compile error now.
//!
//! **Out of scope (deferred).** Per-region `CtmWeights` Linears
//! (NLM s1, MHA in/out, KV/Q projections) are not yet lifted into
//! the cache. The biggest fixed-cost matvecs in the maze brain are
//! the connection synapses (multi-scale obs concatenated → region
//! d_input), so this slice covers the largest item first. Wiring
//! the per-region internals is a separate iteration.

#![cfg(feature = "rocm")]

use modgrad_compute::backend::ResidencyError;
use modgrad_compute::neuron::LinearResident;

use crate::graph::RegionalWeights;

/// Device-resident companion to `RegionalWeights`. One
/// `LinearResident` per top-level Linear in the host weights. Built
/// once via `from_weights`; synced (re-uploaded) after every AdamW
/// step via `sync_from_weights`.
///
/// Per-region `CtmWeights` Linears are not in this cache yet — that
/// is the next slice. See module-level docs.
pub struct RegionalResidentCache {
    pub connection_synapses: Vec<LinearResident>,
    pub obs_proj: LinearResident,
    pub output_proj: LinearResident,
    pub outer_exit_gate: Option<LinearResident>,
    pub extra_heads: Vec<LinearResident>,
}

impl RegionalResidentCache {
    /// Allocate device buffers and upload all top-level Linears.
    /// Returns `ResidencyError::Backend(_)` on hipMalloc / hipMemcpy
    /// failure; the inner `BackendError` distinguishes
    /// `OutOfMemory` (retry after eviction is sensible) from
    /// `DeviceLost` (fail the run). Build site doesn't know which
    /// Linear failed — that information is gone the moment we go
    /// from String concatenation to typed errors. If diagnostics
    /// matter the caller can `Err`-trace the call.
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
        Ok(Self {
            connection_synapses, obs_proj, output_proj,
            outer_exit_gate, extra_heads,
        })
    }

    /// Re-upload every Linear after an in-place optimizer step.
    /// Caller is responsible for invoking this after AdamW; without
    /// it, the resident path will run forwards against pre-step
    /// weights while the host path uses post-step. Match the host
    /// pattern of "step optimizer → invalidate caches" exactly.
    /// (The PR-AB lift will turn this into a generation-counter
    /// witness; today it is a runtime contract.)
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
        Ok(())
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

        // Synthetic input matching output_proj.in_dim.
        let in_dim = weights.output_proj.in_dim;
        let out_dim = weights.output_proj.out_dim;
        let host_x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();

        // Host path
        let mut host_y = vec![0.0f32; out_dim];
        weights.output_proj.forward_into(&host_x, &mut host_y);

        // Resident path
        let mut x_dev = GpuVec::try_hip(in_dim).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(out_dim).expect("alloc out");
        let batch = HipBatch::new();
        cache.output_proj.forward(&batch, &x_dev, &mut out_dev)
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

    /// AdamW-style mutation invalidates the cache; sync_from_weights
    /// must re-upload so resident output reflects new weights.
    #[test]
    fn cache_sync_after_weight_mutation() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let cfg = RegionalConfig::eight_region_small(64, 25, 3);
        let mut weights = RegionalWeights::new(cfg);
        let mut cache = RegionalResidentCache::from_weights(&weights)
            .expect("cache build");

        // Mutate output_proj weights (simulate AdamW step).
        for w in weights.output_proj.weight.iter_mut() { *w *= 1.5; }
        for b in weights.output_proj.bias.iter_mut() { *b += 0.1; }

        // Without sync, the resident path uses STALE weights — the
        // contract of the cache. Verify the staleness is detectable.
        let in_dim = weights.output_proj.in_dim;
        let out_dim = weights.output_proj.out_dim;
        let host_x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
        let mut host_y = vec![0.0f32; out_dim];
        weights.output_proj.forward_into(&host_x, &mut host_y);

        let mut x_dev = GpuVec::try_hip(in_dim).expect("x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(out_dim).expect("out");
        let batch = HipBatch::new();
        cache.output_proj.forward(&batch, &x_dev, &mut out_dev).expect("forward");
        batch.flush().expect("flush");
        let mut stale_y = vec![0.0f32; out_dim];
        out_dev.copy_to_host(&mut stale_y);

        let stale_diff = host_y.iter().zip(&stale_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(stale_diff > 1e-3,
            "without sync, expected staleness; got max |Δ| = {stale_diff} (looks already-fresh)");

        // After sync, output should match host.
        cache.sync_from_weights(&weights).expect("sync");
        let mut fresh_dev = GpuVec::try_hip(out_dim).expect("out2");
        let batch2 = HipBatch::new();
        cache.output_proj.forward(&batch2, &x_dev, &mut fresh_dev).expect("forward2");
        batch2.flush().expect("flush2");
        let mut fresh_y = vec![0.0f32; out_dim];
        fresh_dev.copy_to_host(&mut fresh_y);

        let fresh_diff = host_y.iter().zip(&fresh_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(fresh_diff < 1e-3,
            "after sync, expected match; got max |Δ| = {fresh_diff}");
    }
}
