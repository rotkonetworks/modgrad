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

// ─── Device-resident gradient accumulators ──────────────────────────
//
// Phase 3b of the brain-on-GPU plan: the per-region NLM weight gradient
// (nlm_s1_w, optional nlm_s2_w) is the hot-path accumulator that the
// new SuperLinearBwdDw resident kernel writes into. Keeping it on
// device avoids the 128 MB-per-call PCIe download at billion config.
//
// One `RegionalGradientsResident` lives for the duration of training.
// At training start: allocate, zero. Per backward: kernels accumulate
// (beta=1.0). At opt.step: download to host RegionalGradients,
// optimizer updates host weights, host-side cache.sync_from_weights
// re-uploads. Then `zero()` resets the device gradient buffers.

use modgrad_device::backend::HipBuffer;

/// Device-resident accumulators for the hot NLM weight gradients.
/// Follows the same allocate-once / sync-per-step pattern as
/// `RegionalResidentCache`. Only carries the buckets that Phase 0
/// profiling identified as dominant (nlm_s1_w/nlm_s2_w); other
/// gradient tensors stay host-resident in `RegionalGradients`
/// because their compute is not on the hot path.
pub struct RegionalGradientsResident {
    /// Per-region NLM stage1 weight gradients on device.
    /// Layout: `[n_neurons × out_per × memory_length]` per region,
    /// flat row-major (matches `SuperLinear.weights`).
    pub(crate) region_nlm_s1_dw: Vec<HipBuffer>,
    /// Per-region NLM stage2 weight gradients on device. `None` for
    /// regions whose `deep_nlms=false`.
    pub(crate) region_nlm_s2_dw: Vec<Option<HipBuffer>>,
}

impl RegionalGradientsResident {
    /// Allocate device buffers sized to match each region's NLM stage
    /// weights, and zero-initialize them. Returns
    /// `ResidencyError::Backend(_)` on hipMalloc / hipMemcpy failure.
    pub fn from_weights(w: &RegionalWeights) -> Result<Self, ResidencyError> {
        let mut s1 = Vec::with_capacity(w.regions.len());
        let mut s2 = Vec::with_capacity(w.regions.len());
        for rw in &w.regions {
            let s1_len = rw.nlm_stage1.weights.len();
            let buf = HipBuffer::new(s1_len * 4)?;
            // Zero-init via host upload. ~ms-scale at typical sizes;
            // happens once at training start so cost is negligible.
            let zeros = vec![0.0f32; s1_len];
            buf.copy_from_host(&zeros)?;
            s1.push(buf);

            let s2_buf = match &rw.nlm_stage2 {
                Some(s2w) => {
                    let len = s2w.weights.len();
                    let buf = HipBuffer::new(len * 4)?;
                    let zeros = vec![0.0f32; len];
                    buf.copy_from_host(&zeros)?;
                    Some(buf)
                }
                None => None,
            };
            s2.push(s2_buf);
        }
        Ok(Self { region_nlm_s1_dw: s1, region_nlm_s2_dw: s2 })
    }

    /// Zero all device buffers. Called between batches when the host
    /// optimizer has consumed accumulated gradients and we want a
    /// fresh accumulator for the next batch.
    ///
    /// Implementation uploads zero vectors. A future optimization is
    /// `hipMemset` (one fast device-side fill) — keeping the simple
    /// path for now since this runs once per opt.step, not per sample.
    pub fn zero(&self) -> Result<(), ResidencyError> {
        for buf in &self.region_nlm_s1_dw {
            let zeros = vec![0.0f32; buf.len_f32()];
            buf.copy_from_host(&zeros)?;
        }
        for opt in &self.region_nlm_s2_dw {
            if let Some(buf) = opt {
                let zeros = vec![0.0f32; buf.len_f32()];
                buf.copy_from_host(&zeros)?;
            }
        }
        Ok(())
    }

    /// Add device-accumulated gradients into host `RegionalGradients`
    /// fields. This is called once per opt.step, after the resident
    /// backward has run for the whole batch. The host-side optimizer
    /// then consumes `host_grads.region_grads[r].nlm_s1_w` etc.
    /// normally.
    pub fn add_to_host(
        &self,
        host_grads: &mut crate::graph::RegionalGradients,
    ) -> Result<(), ResidencyError> {
        for (r, buf) in self.region_nlm_s1_dw.iter().enumerate() {
            let dst = &mut host_grads.region_grads[r].nlm_s1_w;
            // Download into a scratch, then add. Avoids overwriting any
            // pre-existing gradient (e.g. if some region's nlm_s1_w
            // also had host-side contributions during this batch).
            let mut scratch = vec![0.0f32; dst.len()];
            buf.copy_to_host(&mut scratch)?;
            for (d, s) in dst.iter_mut().zip(scratch.iter()) { *d += s; }
        }
        for (r, opt) in self.region_nlm_s2_dw.iter().enumerate() {
            if let (Some(buf), Some(dst)) =
                (opt, host_grads.region_grads[r].nlm_s2_w.as_mut())
            {
                let mut scratch = vec![0.0f32; dst.len()];
                buf.copy_to_host(&mut scratch)?;
                for (d, s) in dst.iter_mut().zip(scratch.iter()) { *d += s; }
            }
        }
        Ok(())
    }

    /// Accumulate one SuperLinear-backward dW into the per-region
    /// device buffer using the new resident kernel.
    /// `d_out` is the gradient w.r.t. the SuperLinear *post-projection*
    /// (the same `d_out` that `super_linear_bwd_dw` host takes as
    /// input); `trace` is the SuperLinear input. Both are host slices
    /// today — they're uploaded into temporary device scratch buffers
    /// by this function, while `dW` stays on device throughout.
    ///
    /// `region` indexes into `region_nlm_s1_dw`; `stage` selects
    /// stage1 vs stage2 (use `false` for stage1, `true` for stage2).
    pub fn accumulate_super_linear_dw(
        &self,
        region: usize,
        stage2: bool,
        d_out: &[f32],
        trace: &[f32],
        n_neurons: usize,
        in_per: usize,
        out_per: usize,
    ) -> Result<(), ResidencyError> {
        // Pick the destination buffer.
        use modgrad_device::backend::BackendError;
        let dst_buf = if stage2 {
            self.region_nlm_s2_dw[region].as_ref()
                .ok_or_else(|| ResidencyError::Backend(
                    BackendError::Runtime(
                        format!("region {region} has no nlm_s2 buffer (deep_nlms=false?)"),
                    )
                ))?
        } else {
            &self.region_nlm_s1_dw[region]
        };
        // Upload host inputs into temporary device scratches.
        let d_out_buf = HipBuffer::new(d_out.len() * 4)?;
        d_out_buf.copy_from_host(d_out)?;
        let trace_buf = HipBuffer::new(trace.len() * 4)?;
        trace_buf.copy_from_host(trace)?;
        // Dispatch the resident bwd_dw kernel — accumulates (beta=1)
        // into `dst_buf`. Backend errors auto-convert via `?`.
        unsafe {
            modgrad_device::backend::ops::super_linear_bwd_dw_resident(
                d_out_buf.device_ptr() as *const f32,
                trace_buf.device_ptr() as *const f32,
                dst_buf.device_ptr() as *mut f32,
                n_neurons, in_per, out_per,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{RegionalConfig, RegionalGradients, RegionalWeights};
    use modgrad_compute::backend::GpuVec;
    use modgrad_device::backend::HipBatch;
    use modgrad_device::backend::rocm::ffi::runtime_available;

    /// Phase 3b: `RegionalGradientsResident` allocates per-region NLM
    /// gradient buffers on device, can be zeroed, and `add_to_host`
    /// correctly accumulates into a host `RegionalGradients`. Exercises
    /// the buffer lifecycle before any consumer wires it.
    #[test]
    fn regional_gradients_resident_lifecycle() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let cfg = RegionalConfig::eight_region_small(64, 25, 3);
        let weights = RegionalWeights::new(cfg);
        let resident = RegionalGradientsResident::from_weights(&weights)
            .expect("alloc resident grads");

        // Buffers should match the per-region nlm_s1 weight sizes.
        assert_eq!(resident.region_nlm_s1_dw.len(), weights.regions.len());
        for (rw, buf) in weights.regions.iter().zip(&resident.region_nlm_s1_dw) {
            assert_eq!(buf.len_f32(), rw.nlm_stage1.weights.len());
        }

        // After allocation, buffers are zero — add_to_host into a fresh
        // host grads must leave the host side at zero.
        let mut host = RegionalGradients::zeros(&weights);
        resident.add_to_host(&mut host).expect("add_to_host (post-alloc)");
        for r in 0..weights.regions.len() {
            for &v in &host.region_grads[r].nlm_s1_w {
                assert_eq!(v, 0.0, "post-alloc add_to_host should be zero");
            }
        }

        // Manually upload non-zero data into the first region's buffer
        // and verify add_to_host folds it into host correctly.
        let r0_len = weights.regions[0].nlm_stage1.weights.len();
        let payload: Vec<f32> = (0..r0_len).map(|i| (i as f32 * 0.001) - 0.5).collect();
        resident.region_nlm_s1_dw[0].copy_from_host(&payload).expect("upload payload");
        let mut host2 = RegionalGradients::zeros(&weights);
        resident.add_to_host(&mut host2).expect("add_to_host (with payload)");
        for (i, &v) in host2.region_grads[0].nlm_s1_w.iter().enumerate() {
            assert_eq!(v, payload[i], "add_to_host should match uploaded payload");
        }

        // After zero(), all buffers are back to zero.
        resident.zero().expect("zero buffers");
        let mut host3 = RegionalGradients::zeros(&weights);
        resident.add_to_host(&mut host3).expect("add_to_host (post-zero)");
        for r in 0..weights.regions.len() {
            for &v in &host3.region_grads[r].nlm_s1_w {
                assert_eq!(v, 0.0, "post-zero add_to_host should be zero");
            }
        }
    }

    /// Phase 3c: `accumulate_super_linear_dw` on the resident path
    /// produces the same dW that the host `Op::SuperLinearBwdDw`
    /// produces, modulo fp32 reorder noise. Validates the full
    /// upload→kernel→download pattern that nlm_backward_resident's
    /// future refactor will use.
    #[test]
    fn accumulate_super_linear_dw_matches_host() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let cfg = RegionalConfig::eight_region_small(64, 25, 3);
        let weights = RegionalWeights::new(cfg);
        let resident = RegionalGradientsResident::from_weights(&weights)
            .expect("alloc resident grads");

        // Operate on region 0 stage1.
        let r = 0usize;
        let s1 = &weights.regions[r].nlm_stage1;
        let n_neurons = s1.n_neurons;
        let in_per = s1.in_per;
        let out_per = s1.out_per;

        // Synthetic d_out and trace (deterministic).
        let d_out: Vec<f32> = (0..n_neurons * out_per)
            .map(|i| ((i as f32) * 0.013 - 0.5).sin())
            .collect();
        let trace: Vec<f32> = (0..n_neurons * in_per)
            .map(|i| ((i as f32) * 0.021 + 0.3).cos())
            .collect();

        // Host reference — populate a fresh dW slice via the existing
        // host op (no accumulation, starts at zero).
        let mut host_dw = vec![0.0f32; n_neurons * out_per * in_per];
        modgrad_device::backend::ops::super_linear_bwd_dw(
            &d_out, &trace, &mut host_dw, n_neurons, in_per, out_per,
        ).expect("host super_linear_bwd_dw");

        // Resident path — accumulate into device buffer (which is
        // already zeroed by from_weights).
        resident.accumulate_super_linear_dw(
            r, false, &d_out, &trace, n_neurons, in_per, out_per,
        ).expect("accumulate_super_linear_dw");

        // Download device dW into a fresh host RegionalGradients.
        let mut host_grads = RegionalGradients::zeros(&weights);
        resident.add_to_host(&mut host_grads).expect("add_to_host");
        let resident_dw = &host_grads.region_grads[r].nlm_s1_w;

        // Compare element-wise.
        let abs_tol = 1e-3f32;
        let rel_tol = 1e-3f32;
        let mut max_abs = 0.0f32;
        for (i, (h, d)) in host_dw.iter().zip(resident_dw.iter()).enumerate() {
            let diff = (h - d).abs();
            let scale = h.abs().max(d.abs()).max(1.0);
            if diff > abs_tol && diff / scale > rel_tol {
                panic!("dw[{i}] mismatch: host={h} resident={d} (|Δ|={diff})");
            }
            if diff > max_abs { max_abs = diff; }
        }
        eprintln!("  accumulate_super_linear_dw |Δ|max = {max_abs:.6}");
    }

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
