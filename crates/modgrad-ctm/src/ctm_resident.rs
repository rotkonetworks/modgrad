//! Per-region device-resident companion cache for `CtmWeights`.
//!
//! Inner-CTM analog of `RegionalResidentCache` (see
//! `crates/modgrad-ctm/src/resident.rs`). One `LinearResident` per
//! top-level `Linear` field on `CtmWeights` (kv/q projection,
//! packed MHA in/out, output projection, optional adaptive exit
//! gate) plus one `SuperLinearResident` per NLM stage. Once
//! built, the CTM tick loop's largest fixed-cost matvecs can
//! dispatch via `matvec_resident` with zero PCIe transfers per
//! call.
//!
//! **Lifecycle.** Construct from a `CtmWeights` via
//! `CtmResidentCache::from_weights(&w)?`. After every optimizer
//! step that mutated host weights, call `sync_from_weights(&w)?`
//! to re-upload. The bench numbers in
//! `memory/feedback_residency_proof.md` show ~5× sustained
//! per-call speedup once the CTM hot path stops round-tripping
//! through PCIe.
//!
//! **Freshness.** This cache does NOT carry its own freshness
//! witness. `CtmWeights` has no `generation()` counter, and the
//! per-region instance is always held inside a parent
//! `RegionalResidentCache` whose `RegionalWeights::generation()`
//! covers the whole region (top-level Linears + the contained
//! `CtmWeights`). Outer freshness is therefore the source of
//! truth: callers go through `RegionalResidentCache::fresh()`,
//! which can refuse stale dispatch before any per-region
//! `CtmResidentCache` field is reached. The resident fields
//! below are `pub(crate)` so direct access is impossible from
//! outside the crate — adding a separate freshness witness
//! here would only duplicate the parent check. (TODO: revisit
//! if a use case ever needs to mutate `CtmWeights` independently
//! of its parent `RegionalWeights`.)
//!
//! **Sync discipline.** Resident dispatches must happen inside a
//! `HipBatch` scope (see `memory/feedback_hip_queue_overflow.md`).
//! `LinearResident::forward` and `SuperLinearResident::forward`
//! both take `&HipBatch` so the requirement is a compile error
//! to forget.
//!
//! **Out of scope (deferred).** The U-Net `synapse: SynapseUNet`
//! field is intentionally NOT mirrored by this slice. hdevalence
//! is implementing `SynapseUNetResident` in a parallel PR; once
//! that lands, a follow-up integrates it here as a
//! `pub(crate) synapse: SynapseUNetResident` field, with matching
//! `from_weights`/`sync_from_weights` plumbing and an updated
//! `n_linears()` accounting.

#![cfg(feature = "rocm")]

use modgrad_compute::backend::ResidencyError;
use modgrad_compute::neuron::{LinearResident, SuperLinearResident};

use crate::weights::CtmWeights;

/// Device-resident companion to `CtmWeights`. One `LinearResident`
/// per top-level `Linear` field (kv/q projection, packed MHA in/out,
/// output projection, optional adaptive exit gate) plus one
/// `SuperLinearResident` per NLM stage.
///
/// Built once via `from_weights`; synced (re-uploaded) after every
/// optimizer step via `sync_from_weights`.
///
/// The `synapse: SynapseUNet` field on `CtmWeights` is deliberately
/// NOT mirrored here — it is the deferred follow-up that integrates
/// `SynapseUNetResident` from `synapse.rs` once that PR lands.
///
/// Fields are `pub(crate)`. Freshness is enforced at the outer level
/// (`RegionalResidentCache::fresh()`); per-region cache lives inside
/// it, so an external caller cannot reach these fields without first
/// passing the outer freshness check.
pub struct CtmResidentCache {
    pub(crate) kv_proj: LinearResident,
    pub(crate) q_proj: LinearResident,
    pub(crate) mha_in_proj: LinearResident,
    pub(crate) mha_out_proj: LinearResident,
    pub(crate) output_proj: LinearResident,
    /// Mirrors `CtmWeights::exit_gate`: `Some` iff the config's
    /// `exit_strategy` is `AdaptiveGate`.
    pub(crate) exit_gate: Option<LinearResident>,
    pub(crate) nlm_stage1: SuperLinearResident,
    /// Mirrors `CtmWeights::nlm_stage2`: `Some` iff `deep_nlms = true`.
    pub(crate) nlm_stage2: Option<SuperLinearResident>,
    // DEFERRED: `pub(crate) synapse: SynapseUNetResident` — follow-up
    // PR after hdevalence's `SynapseUNetResident` lands in
    // `crates/modgrad-ctm/src/synapse.rs`.
}

impl CtmResidentCache {
    /// Allocate device buffers and upload all top-level Linears and
    /// both NLM SuperLinears (stage2 only when present). Returns a
    /// `ResidencyError::Backend(_)` on hipMalloc / hipMemcpy failure;
    /// match the inner `BackendError` variant for typed recovery.
    pub fn from_weights(w: &CtmWeights) -> Result<Self, ResidencyError> {
        let kv_proj = LinearResident::from_linear(&w.kv_proj)?;
        let q_proj = LinearResident::from_linear(&w.q_proj)?;
        let mha_in_proj = LinearResident::from_linear(&w.mha_in_proj)?;
        let mha_out_proj = LinearResident::from_linear(&w.mha_out_proj)?;
        let output_proj = LinearResident::from_linear(&w.output_proj)?;
        let exit_gate = match &w.exit_gate {
            Some(g) => Some(LinearResident::from_linear(g)?),
            None => None,
        };
        let nlm_stage1 = SuperLinearResident::from_super_linear(&w.nlm_stage1)?;
        let nlm_stage2 = match &w.nlm_stage2 {
            Some(s) => Some(SuperLinearResident::from_super_linear(s)?),
            None => None,
        };
        Ok(Self {
            kv_proj, q_proj,
            mha_in_proj, mha_out_proj,
            output_proj, exit_gate,
            nlm_stage1, nlm_stage2,
        })
    }

    /// Re-upload every mirrored Linear and SuperLinear after an
    /// in-place optimizer step. Bias too, in case it was updated.
    /// Optionality must match the build-time shape — calling this on
    /// a `CtmWeights` whose `exit_gate`/`nlm_stage2` flipped from
    /// None ↔ Some since `from_weights` is unsupported (rebuild the
    /// cache). The `debug_assert`s catch this in tests.
    pub fn sync_from_weights(&mut self, w: &CtmWeights) -> Result<(), ResidencyError> {
        self.kv_proj.sync_weights_from(&w.kv_proj)?;
        self.q_proj.sync_weights_from(&w.q_proj)?;
        self.mha_in_proj.sync_weights_from(&w.mha_in_proj)?;
        self.mha_out_proj.sync_weights_from(&w.mha_out_proj)?;
        self.output_proj.sync_weights_from(&w.output_proj)?;
        debug_assert_eq!(self.exit_gate.is_some(), w.exit_gate.is_some(),
            "exit_gate optionality changed since cache build — rebuild required");
        if let (Some(r), Some(lin)) = (self.exit_gate.as_mut(), w.exit_gate.as_ref()) {
            r.sync_weights_from(lin)?;
        }
        self.nlm_stage1.sync_weights_from(&w.nlm_stage1)?;
        debug_assert_eq!(self.nlm_stage2.is_some(), w.nlm_stage2.is_some(),
            "nlm_stage2 optionality changed since cache build — rebuild required");
        if let (Some(r), Some(sl)) = (self.nlm_stage2.as_mut(), w.nlm_stage2.as_ref()) {
            r.sync_weights_from(sl)?;
        }
        Ok(())
    }

    /// Number of mirrored device residents (Linears + SuperLinears).
    /// Useful for debug logs ("ctm cache holds N residents, expected M").
    /// Counts each `Option` field as 0 or 1. The deferred `synapse`
    /// is not in the tally.
    pub fn n_linears(&self) -> usize {
        // 5 always-present Linears (kv, q, mha_in, mha_out, output_proj)
        5
            + self.exit_gate.as_ref().map_or(0, |_| 1)
            // nlm_stage1 always present
            + 1
            + self.nlm_stage2.as_ref().map_or(0, |_| 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CtmConfig, ExitStrategy};
    use modgrad_compute::backend::GpuVec;
    use modgrad_device::backend::HipBatch;
    use modgrad_device::backend::rocm::ffi::runtime_available;

    /// Small CTM config with deep NLMs and an adaptive exit gate.
    /// `nlm_stage2` is allocated (deep_nlms=true), `exit_gate` is
    /// allocated (AdaptiveGate). Used by every test in this module.
    fn cfg_full() -> CtmConfig {
        CtmConfig {
            iterations: 4,
            d_model: 16,
            d_input: 8,
            heads: 2,
            n_synch_out: 16,
            n_synch_action: 8,
            synapse_depth: 1,
            memory_length: 4,
            deep_nlms: true,
            memory_hidden_dims: 2,
            out_dims: 3,
            n_random_pairing_self: 0,
            min_width: 4,
            exit_strategy: ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.5 },
            collect_trajectories: false,
        }
    }

    /// Build a CtmWeights with deep NLMs + AdaptiveGate, mirror as a
    /// cache, and verify the resident count matches the expected
    /// inventory: 5 always-present Linears + exit_gate + nlm_stage1 +
    /// nlm_stage2 = 8.
    #[test]
    fn cache_build_succeeds() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let cfg = cfg_full();
        let raw_input_dim = 10;
        let weights = CtmWeights::new(cfg, raw_input_dim);
        let cache = CtmResidentCache::from_weights(&weights)
            .expect("cache build");

        // 5 always-present Linears (kv, q, mha_in, mha_out, output_proj)
        // + 1 exit_gate (AdaptiveGate)
        // + 1 nlm_stage1 (always present)
        // + 1 nlm_stage2 (deep_nlms = true)
        assert_eq!(cache.n_linears(), 8,
            "expected 5 Linears + exit_gate + 2 NLM stages = 8 residents");
        assert!(cache.exit_gate.is_some(), "AdaptiveGate must allocate gate");
        assert!(cache.nlm_stage2.is_some(), "deep_nlms must allocate stage2");
    }

    /// Run kv_proj resident vs host on random input; outputs must
    /// match within 1e-3 FP tolerance. Mirrors the
    /// `cache_output_proj_matches_host` pattern from `resident.rs`.
    #[test]
    fn cache_kv_proj_matches_host() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let cfg = cfg_full();
        let raw_input_dim = 10;
        let weights = CtmWeights::new(cfg, raw_input_dim);
        let cache = CtmResidentCache::from_weights(&weights)
            .expect("cache build");

        let in_dim = weights.kv_proj.in_dim;
        let out_dim = weights.kv_proj.out_dim;
        assert_eq!(in_dim, raw_input_dim);
        let host_x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.013 - 0.05).collect();

        // Host reference path
        let mut host_y = vec![0.0f32; out_dim];
        weights.kv_proj.forward_into(&host_x, &mut host_y);

        // Resident path
        let mut x_dev = GpuVec::try_hip(in_dim).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(out_dim).expect("alloc out");
        let batch = HipBatch::new();
        cache.kv_proj.forward(&batch, &x_dev, &mut out_dev)
            .expect("resident forward");
        batch.flush().expect("flush");
        let mut device_y = vec![0.0f32; out_dim];
        out_dev.copy_to_host(&mut device_y);

        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "kv_proj host vs resident: max |Δ| = {max_diff}");
    }

    /// Run nlm_stage1 resident vs the CPU `forward_cpu` reference on
    /// random input; outputs must match within 1e-3 FP tolerance.
    /// Tests the per-neuron pointer-offset SuperLinear path.
    #[test]
    fn cache_nlm_stage1_matches_host() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let cfg = cfg_full();
        let raw_input_dim = 10;
        let weights = CtmWeights::new(cfg, raw_input_dim);
        let cache = CtmResidentCache::from_weights(&weights)
            .expect("cache build");

        let n_neurons = weights.nlm_stage1.n_neurons;
        let in_per = weights.nlm_stage1.in_per;
        let out_per = weights.nlm_stage1.out_per;
        let host_x: Vec<f32> = (0..n_neurons * in_per)
            .map(|i| (i as f32) * 0.007 - 0.03)
            .collect();

        // Host reference: `forward_cpu` (deterministic, same arithmetic
        // structure as the device path's per-neuron loop).
        let mut host_y = vec![0.0f32; n_neurons * out_per];
        weights.nlm_stage1.forward_cpu(&host_x, &mut host_y);

        // Resident path
        let mut x_dev = GpuVec::try_hip(n_neurons * in_per).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(n_neurons * out_per).expect("alloc out");
        let batch = HipBatch::new();
        cache.nlm_stage1.forward(&batch, &x_dev, &mut out_dev)
            .expect("resident forward");
        batch.flush().expect("flush");
        let mut device_y = vec![0.0f32; n_neurons * out_per];
        out_dev.copy_to_host(&mut device_y);

        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "nlm_stage1 host vs resident: max |Δ| = {max_diff}");
    }

    /// Mutate kv_proj weight + bias on the host, call
    /// `sync_from_weights`, and verify the resident dispatch now
    /// matches the *new* host weights (not the build-time snapshot).
    /// Proves sync re-uploads correctly.
    #[test]
    fn sync_weights_from_recovers_correctness() {
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let cfg = cfg_full();
        let raw_input_dim = 10;
        let mut weights = CtmWeights::new(cfg, raw_input_dim);
        let mut cache = CtmResidentCache::from_weights(&weights)
            .expect("cache build");

        // Mutate kv_proj on the host: scale weights, shift bias. The
        // resident buffers now diverge from the host until we sync.
        for w in weights.kv_proj.weight.iter_mut() { *w *= 1.5; }
        for b in weights.kv_proj.bias.iter_mut() { *b += 0.1; }

        // Re-upload.
        cache.sync_from_weights(&weights).expect("sync");

        // After sync, dispatch must match the mutated host weights.
        let in_dim = weights.kv_proj.in_dim;
        let out_dim = weights.kv_proj.out_dim;
        let host_x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.011).collect();
        let mut host_y = vec![0.0f32; out_dim];
        weights.kv_proj.forward_into(&host_x, &mut host_y);

        let mut x_dev = GpuVec::try_hip(in_dim).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(out_dim).expect("alloc out");
        let batch = HipBatch::new();
        cache.kv_proj.forward(&batch, &x_dev, &mut out_dev)
            .expect("resident forward");
        batch.flush().expect("flush");
        let mut device_y = vec![0.0f32; out_dim];
        out_dev.copy_to_host(&mut device_y);

        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "after sync, expected match against new host weights; got max |Δ| = {max_diff}");
    }
}
