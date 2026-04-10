//! Per-tick ephemeral state: flat arena, region constants, BPTT cache.
//!
//! Extracted from state.rs. These types are created fresh for each forward pass.

use serde::Serialize;

pub use super::config::*;
pub use super::neuron::*;
pub use super::sync::*;
pub use super::memory::*;

// These types remain in state.rs; import them here for CtmTickState fields.
use super::session::{Neuromodulators, SleepPhase};

// ─── Parent-Child Signal Types ─────────────────────────────

/// Signals emitted by the organism each forward pass.
/// Read-only from the organism's perspective — the parent observes
/// these to decide what to tune, when to sleep, what to consolidate.
#[derive(Debug, Clone, Default)]
pub struct TickSignals {
    /// L2 norm of all region activations (how hard the brain worked).
    pub activation_energy: f32,
    /// Did sync accumulators converge between last two ticks?
    pub sync_converged: bool,
    /// Mean prediction surprise (cross-entropy proxy).
    pub surprise: f32,
    /// Cerebellum prediction error magnitude.
    pub prediction_error: f32,
    /// Insula activation magnitude (body stress level).
    pub insula_magnitude: f32,
    /// Insula activation valence (positive = cool/rested, negative = hot/loaded).
    pub insula_valence: f32,
    /// Whether motor region crossed decision threshold.
    pub motor_decided: bool,
    /// Tick at which motor decided (None = didn't decide).
    pub decision_tick: Option<usize>,
    /// Max cosine similarity from hippocampal retrieval (familiarity).
    pub hippo_max_similarity: f32,
    /// Number of ticks executed.
    pub ticks: usize,
    /// Curiosity level (computed by HOST, not organism).
    pub curiosity: f32,
    /// Anxiety level (computed by HOST, not organism).
    pub anxiety: f32,
    /// Serotonin level (host-injected, organism reads only).
    pub serotonin: f32,
    /// Dopamine level (organism's raw prediction surprise signal).
    pub dopamine: f32,
    /// Norepinephrine level (organism's raw arousal level).
    pub norepinephrine: f32,
}

// ─── Per-tick trace ────────────────────────────────────────

/// Per-tick activation snapshot for visualization/debugging.
#[derive(Debug, Clone, Serialize)]
pub struct TickTrace {
    pub tick: usize,
    pub input_activations: Vec<f32>,
    pub attention_activations: Vec<f32>,
    pub output_activations: Vec<f32>,
    pub motor_activations: Vec<f32>,
    pub sync_out: Vec<f32>,
    pub sync_action: Vec<f32>,
    pub modulation: Vec<f32>,
    pub motor_evidence_max: f32,
    pub motor_decided: bool,
}

impl TickTrace {
    /// Export tick traces to the WebGPU debugger JSON format.
    /// Writes a JSON array compatible with webgpu/assets/ticks.json.
    pub fn export_for_debugger(traces: &[TickTrace], step: u64, loss: f32) -> String {
        let mut ticks = Vec::new();
        for t in traces {
            ticks.push(serde_json::json!({
                "k": t.tick,
                "loss": loss,
                "selected_pct": t.motor_evidence_max,
                "input_activations": t.input_activations,
                "attention_activations": t.attention_activations,
                "output_activations": t.output_activations,
                "motor_activations": t.motor_activations,
                "sync_out": t.sync_out,
                "dopamine": t.modulation.get(MOD_SYNC_SCALE).copied().unwrap_or(1.0),
                "serotonin": t.modulation.get(MOD_GATE).copied().unwrap_or(0.5),
                "norepinephrine": t.modulation.get(MOD_AROUSAL).copied().unwrap_or(0.5),
                "motor_decided": t.motor_decided,
            }));
        }

        serde_json::json!({
            "step": step,
            "loss": loss,
            "ticks": ticks,
        }).to_string()
    }

    /// Write tick traces to a file for the debugger to read.
    /// Appends to JSONL format (one JSON object per line).
    pub fn append_to_file(traces: &[TickTrace], step: u64, loss: f32, path: &str) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true).append(true).open(path)
        {
            let json = Self::export_for_debugger(traces, step, loss);
            writeln!(f, "{}", json).ok();
        }
    }
}

// ─── CtmTickState (flat arena per-tick ephemeral state) ───────

/// Ephemeral per-tick state using flat arena buffers for cache-friendly access.
/// Created fresh for each forward pass, not persisted.
///
/// All 8 regions' traces and activations are packed into flat Vecs with
/// computed offsets. This enables future GPU upload as a single buffer.
///
/// Region order: [input, attention, output, motor, cerebellum, basal_ganglia, insula, hippocampus]
#[derive(Debug, Clone)]
pub struct CtmTickState {
    // Flat arenas with computed offsets
    /// All 8 regions' traces packed contiguously.
    pub traces: Vec<f32>,
    /// All 8 regions' activations packed contiguously.
    pub activations: Vec<f32>,
    /// Byte offset into traces for each region.
    pub trace_offsets: [usize; 8],
    /// Size (in f32s) of each region's trace.
    pub trace_sizes: [usize; 8],
    /// Byte offset into activations for each region.
    pub act_offsets: [usize; 8],
    /// Size (in f32s) of each region's activations.
    pub act_sizes: [usize; 8],

    // Phase (4 cortical regions)
    pub phases: Vec<f32>,
    pub phase_offsets: [usize; 4],
    pub phase_sizes: [usize; 4],

    // Sync state (flat)
    pub sync_alpha: Vec<f32>,
    pub sync_beta: Vec<f32>,
    pub sync_r_shift: Vec<f32>,
    pub sync_initialized: bool,

    // Sync action state (flat)
    pub sync_action_alpha: Vec<f32>,
    pub sync_action_beta: Vec<f32>,
    pub sync_action_r_shift: Vec<f32>,
    pub sync_action_initialized: bool,

    // Rest from CtmState
    pub modulation: Vec<f32>,
    pub neuromod: Neuromodulators,
    pub motor_evidence: Vec<f32>,
    pub motor_decided: bool,
    pub decision_tick: Option<usize>,
    pub last_sync: Vec<f32>,
    pub noise_rng: SimpleRng,
    pub column_noise_buf: Vec<f32>,
    pub tick_traces: Vec<TickTrace>,
    pub noisy: bool,
    pub sleep_phase: SleepPhase,
    pub telemetry_step: u64,
    pub telemetry_loss: f32,
}

impl CtmTickState {
    /// Create a new CtmTickState from config.
    pub fn new(config: &CtmConfig) -> Self {
        let region_neurons = [
            config.input_layer.n_neurons,
            config.attention_layer.n_neurons,
            config.output_layer.n_neurons,
            config.motor_layer.n_neurons,
            config.cerebellum_layer.n_neurons,
            config.basal_ganglia_layer.n_neurons,
            config.insula_layer.n_neurons,
            config.hippocampus_layer.n_neurons,
        ];
        let region_memory = [
            config.input_layer.memory_length,
            config.attention_layer.memory_length,
            config.output_layer.memory_length,
            config.motor_layer.memory_length,
            config.cerebellum_layer.memory_length,
            config.basal_ganglia_layer.memory_length,
            config.insula_layer.memory_length,
            config.hippocampus_layer.memory_length,
        ];

        // Compute trace offsets and sizes
        let mut trace_offsets = [0usize; 8];
        let mut trace_sizes = [0usize; 8];
        let mut offset = 0;
        for i in 0..8 {
            trace_offsets[i] = offset;
            trace_sizes[i] = region_neurons[i] * region_memory[i];
            offset += trace_sizes[i];
        }
        let total_traces = offset;

        // Compute activation offsets and sizes
        let mut act_offsets = [0usize; 8];
        let mut act_sizes = [0usize; 8];
        offset = 0;
        for i in 0..8 {
            act_offsets[i] = offset;
            act_sizes[i] = region_neurons[i];
            offset += act_sizes[i];
        }
        let total_activations = offset;

        // Phase offsets (4 cortical regions only)
        let mut phase_offsets = [0usize; 4];
        let mut phase_sizes = [0usize; 4];
        offset = 0;
        for i in 0..4 {
            phase_offsets[i] = offset;
            phase_sizes[i] = region_neurons[i];
            offset += phase_sizes[i];
        }
        let total_phases = offset;

        let n_out_motor = config.output_layer.n_neurons
            + config.motor_layer.n_neurons + config.d_input;
        let n_in_attn = config.input_layer.n_neurons
            + config.attention_layer.n_neurons;

        let n_sync_out = config.n_sync_out;
        let n_sync_action = config.n_sync_action;

        // Create sync accumulators to get proper indices
        let sync_out_tmp = SyncAccumulator::new(n_sync_out, n_out_motor);
        let sync_action_tmp = SyncAccumulator::new(n_sync_action, n_in_attn);

        Self {
            traces: vec![0.0; total_traces],
            activations: vec![0.0; total_activations],
            trace_offsets,
            trace_sizes,
            act_offsets,
            act_sizes,
            phases: vec![0.5; total_phases],
            phase_offsets,
            phase_sizes,
            sync_alpha: sync_out_tmp.alpha,
            sync_beta: sync_out_tmp.beta,
            sync_r_shift: vec![0.0; n_sync_out],
            sync_initialized: false,
            sync_action_alpha: sync_action_tmp.alpha,
            sync_action_beta: sync_action_tmp.beta,
            sync_action_r_shift: vec![0.0; n_sync_action],
            sync_action_initialized: false,
            modulation: default_modulation(),
            neuromod: Neuromodulators::default(),
            motor_evidence: vec![0.0; config.motor_layer.n_neurons],
            motor_decided: false,
            decision_tick: None,
            last_sync: vec![0.0; n_sync_out],
            noise_rng: SimpleRng::new(n_sync_out as u64 * 31337),
            column_noise_buf: vec![0.0; 512],
            tick_traces: Vec::new(),
            noisy: false,
            sleep_phase: SleepPhase::Awake,
            telemetry_step: 0,
            telemetry_loss: 0.0,
        }
    }

    /// Get a mutable slice for region i's trace data.
    #[inline]
    pub fn trace_mut(&mut self, region: usize) -> &mut [f32] {
        let off = self.trace_offsets[region];
        let sz = self.trace_sizes[region];
        &mut self.traces[off..off + sz]
    }

    /// Get an immutable slice for region i's trace data.
    #[inline]
    pub fn trace(&self, region: usize) -> &[f32] {
        let off = self.trace_offsets[region];
        let sz = self.trace_sizes[region];
        &self.traces[off..off + sz]
    }

    /// Get a mutable slice for region i's activation data.
    #[inline]
    pub fn act_mut(&mut self, region: usize) -> &mut [f32] {
        let off = self.act_offsets[region];
        let sz = self.act_sizes[region];
        &mut self.activations[off..off + sz]
    }

    /// Get an immutable slice for region i's activation data.
    #[inline]
    pub fn act(&self, region: usize) -> &[f32] {
        let off = self.act_offsets[region];
        let sz = self.act_sizes[region];
        &self.activations[off..off + sz]
    }

    /// Get a mutable slice for cortical region i's phase data.
    #[inline]
    pub fn phase_mut(&mut self, region: usize) -> &mut [f32] {
        let off = self.phase_offsets[region];
        let sz = self.phase_sizes[region];
        &mut self.phases[off..off + sz]
    }

    /// Get an immutable slice for cortical region i's phase data.
    #[inline]
    pub fn phase(&self, region: usize) -> &[f32] {
        let off = self.phase_offsets[region];
        let sz = self.phase_sizes[region];
        &self.phases[off..off + sz]
    }
}

// Region index constants for CtmTickState arena accessors
pub const REGION_INPUT: usize = 0;
pub const REGION_ATTENTION: usize = 1;
pub const REGION_OUTPUT: usize = 2;
pub const REGION_MOTOR: usize = 3;
pub const REGION_CEREBELLUM: usize = 4;
pub const REGION_BASAL_GANGLIA: usize = 5;
pub const REGION_INSULA: usize = 6;
pub const REGION_HIPPOCAMPUS: usize = 7;

// ─── BpttTickCache ──────────────────────────────────────────

/// Cached intermediates from one tick of forward_split, for BPTT backward.
/// Stores synapse and NLM caches for the 4 main pipeline regions.
pub struct BpttTickCache {
    /// Synapse caches (input to synapse.backward)
    pub syn_caches: [super::synapse::SynapseForwardCache; 8],
    /// NLM caches (input to region.backward_nlm)
    pub nlm_caches: [crate::neuron::NlmForwardCache; 8],
    /// Synapse inputs (needed for weight gradient: dW = dZ @ X^T)
    pub syn_inputs: [Vec<f32>; 8],
}

// BpttTickCache can't derive Clone/Debug due to the cache types.
// Manual impl not needed — it's consumed, not cloned.
impl std::fmt::Debug for BpttTickCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BpttTickCache").finish()
    }
}

impl Clone for BpttTickCache {
    fn clone(&self) -> Self {
        panic!("BpttTickCache should not be cloned");
    }
}
