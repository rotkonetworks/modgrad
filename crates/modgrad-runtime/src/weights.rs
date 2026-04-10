//! Weight structs: CtmWeights (immutable, Arc-shareable) and Ctm (combined).
//!
//! CtmWeights holds all synapses, region weights, and projection matrices.
//! Ctm is the backward-compatible wrapper that holds both weights and session state.
//! Forward-pass logic lives in ctm.rs, not here.

use serde::{Deserialize, Serialize};

pub use super::config::*;
pub use modgrad_compute::neuron::*;
pub use super::neuron::*;
pub use super::sync::*;
pub use super::session::*;
pub use super::tick_state::*;
pub use super::synapse::*;

// NeuronLayer lives in ctm.rs (the combined region struct).

fn default_bg_da_baseline() -> f32 { 0.7 }

// ─── CtmWeights (immutable, Arc-shareable) ──────────────────

/// All weight matrices and static data. Immutable during forward pass.
/// Can be wrapped in `Arc<CtmWeights>` and shared across threads for
/// batch parallelism without cloning ~550MB of weights.
///
/// Modified ONLY during sleep consolidation (single-threaded).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtmWeights {
    pub config: CtmConfig,

    // Brain region weights
    pub input_region: NeuronLayerWeights,
    pub attention_region: NeuronLayerWeights,
    pub output_region: NeuronLayerWeights,
    pub motor_region: NeuronLayerWeights,

    // Subcortical region weights
    pub cerebellum_region: NeuronLayerWeights,
    pub basal_ganglia_region: NeuronLayerWeights,

    // New regions (v3) weights
    pub insula_region: NeuronLayerWeights,
    pub hippocampus_region: NeuronLayerWeights,

    // Inter-region synapses
    pub syn_motor_input: Synapse,
    pub syn_input_attn: Synapse,
    pub syn_attn_output: Synapse,
    pub syn_output_motor: Synapse,
    pub syn_cerebellum: Synapse,
    pub syn_basal_ganglia: Synapse,
    pub syn_insula: Synapse,
    pub syn_hippocampus: Synapse,

    // Global broadcast
    pub global_projector: Linear,

    // Output projection (sync/hopfield -> prediction)
    pub output_projector: Linear,

    /// Hopfield/attention readout (modern Hopfield network).
    /// When config.use_hopfield_readout is true, this replaces SyncAccumulator.
    /// Maps to hippocampal pattern completion.
    #[serde(default)]
    pub hopfield_readout: Option<HopfieldReadout>,

    // Logit projector (optional, trained during sleep)
    pub logit_projector: Option<Linear>,

    /// Device mesh: maps regions to compute devices.
    #[serde(skip)]
    pub device_mesh: Option<modgrad_device::device::DeviceMesh>,

    /// Persistent position predictor for sync_out accumulator.
    #[serde(skip)]
    pub position_predictor: Option<PositionPredictor>,

    // Sync topology — fixed at construction, shared across threads.
    // These define WHICH neuron pairs contribute to the sync signal.
    /// sync_out pair topology: indices into [output, motor, observation] concatenation.
    #[serde(skip)]
    pub sync_out_indices_l: Vec<usize>,
    #[serde(skip)]
    pub sync_out_indices_r: Vec<usize>,
    #[serde(skip)]
    pub sync_out_decay: Vec<f32>,
    /// sync_action pair topology: indices into [input, attention] concatenation.
    #[serde(skip)]
    pub sync_action_indices_l: Vec<usize>,
    #[serde(skip)]
    pub sync_action_indices_r: Vec<usize>,
    #[serde(skip)]
    pub sync_action_decay: Vec<f32>,
}

// SAFETY: CtmWeights contains no interior mutability (no Cell, RefCell, Mutex, etc).
// All fields are plain data (Vec<f32>, etc) which are Send+Sync by default.
// DeviceMesh contains only HashMap + Vec — all Send+Sync.
unsafe impl Sync for CtmWeights {}

impl CtmWeights {
    pub fn new(config: CtmConfig) -> Self {
        let sizes = [
            config.input_layer.n_neurons,
            config.attention_layer.n_neurons,
            config.output_layer.n_neurons,
            config.motor_layer.n_neurons,
        ];
        let cortical_total = sizes.iter().sum::<usize>();
        let total = cortical_total
            + config.cerebellum_layer.n_neurons
            + config.basal_ganglia_layer.n_neurons
            + config.insula_layer.n_neurons
            + config.hippocampus_layer.n_neurons;
        let g = if config.global_broadcast_dim > 0 {
            config.global_broadcast_dim
        } else {
            total / 4
        };

        let g_in = if config.input_layer.receives_broadcast { g } else { 0 };
        let g_attn = if config.attention_layer.receives_broadcast { g } else { 0 };
        let g_out = if config.output_layer.receives_broadcast { g } else { 0 };
        let g_motor = if config.motor_layer.receives_broadcast { g } else { 0 };

        Self {
            input_region: NeuronLayerWeights::new(&config.input_layer),
            attention_region: NeuronLayerWeights::new(&config.attention_layer),
            output_region: NeuronLayerWeights::new(&config.output_layer),
            motor_region: NeuronLayerWeights::new(&config.motor_layer),

            cerebellum_region: NeuronLayerWeights::new(&config.cerebellum_layer),
            basal_ganglia_region: NeuronLayerWeights::new(&config.basal_ganglia_layer),

            insula_region: NeuronLayerWeights::new(&config.insula_layer),
            hippocampus_region: NeuronLayerWeights::new(&config.hippocampus_layer),

            // Each synapse uses its receiving region's synapse_depth and min_width.
            // depth ≥ 3 → U-Net (Sakana-style), depth 2 → 2-layer MLP, depth 1 → linear.
            syn_motor_input: Synapse::new_with_depth_and_min_width(
                config.d_input + sizes[3] + g_in, sizes[0],
                config.input_layer.synapse_depth, config.input_layer.unet_min_width),
            syn_input_attn: Synapse::new_with_depth_and_min_width(
                sizes[0] + config.d_input + g_attn, sizes[1],
                config.attention_layer.synapse_depth, config.attention_layer.unet_min_width),
            syn_attn_output: Synapse::new_with_depth_and_min_width(
                sizes[1] + config.d_input + g_out, sizes[2],
                config.output_layer.synapse_depth, config.output_layer.unet_min_width),
            syn_output_motor: Synapse::new_with_depth_and_min_width(
                sizes[2] + config.basal_ganglia_layer.n_neurons + g_motor, sizes[3],
                config.motor_layer.synapse_depth, config.motor_layer.unet_min_width),
            syn_cerebellum: Synapse::new_with_depth_and_min_width(
                sizes[3] + config.d_input, config.cerebellum_layer.n_neurons,
                config.cerebellum_layer.synapse_depth, config.cerebellum_layer.unet_min_width),
            syn_basal_ganglia: Synapse::new_with_depth_and_min_width(
                sizes[2] + 1, config.basal_ganglia_layer.n_neurons,
                config.basal_ganglia_layer.synapse_depth, config.basal_ganglia_layer.unet_min_width),
            syn_insula: Synapse::new_with_depth_and_min_width(
                config.d_input + cortical_total, config.insula_layer.n_neurons,
                config.insula_layer.synapse_depth, config.insula_layer.unet_min_width),
            syn_hippocampus: Synapse::new_with_depth_and_min_width(
                sizes[0] + sizes[1] + sizes[2] + sizes[3], config.hippocampus_layer.n_neurons,
                config.hippocampus_layer.synapse_depth, config.hippocampus_layer.unet_min_width),

            global_projector: Linear::new(total, g * 2),
            output_projector: if config.use_hopfield_readout {
                Linear::new(config.n_sync_out, config.out_dims) // d_out from hopfield
            } else {
                Linear::new(config.n_sync_out, config.out_dims)
            },

            hopfield_readout: if config.use_hopfield_readout {
                // d_in = output + motor + observation (same neurons that feed sync)
                let n_out_motor = config.output_layer.n_neurons
                    + config.motor_layer.n_neurons + config.d_input;
                Some(HopfieldReadout::new(n_out_motor, config.n_sync_out, config.hopfield_d_key))
            } else {
                None
            },

            logit_projector: None,
            device_mesh: None,
            position_predictor: {
                let n_out_motor = config.output_layer.n_neurons + config.motor_layer.n_neurons + config.d_input;
                if n_out_motor >= 16 && config.n_sync_out >= 8 {
                    Some(PositionPredictor::new(n_out_motor, config.n_sync_out))
                } else {
                    None
                }
            },
            sync_out_indices_l: Vec::new(),
            sync_out_indices_r: Vec::new(),
            sync_out_decay: Vec::new(),
            sync_action_indices_l: Vec::new(),
            sync_action_indices_r: Vec::new(),
            sync_action_decay: Vec::new(),

            config,
        }
    }

    /// Set the device mesh for multi-GPU region distribution.
    pub fn set_device_mesh(&mut self, mesh: modgrad_device::device::DeviceMesh) {
        self.device_mesh = Some(mesh);
    }

    /// Get sync_out pair left indices (for telemetry manifest).
    pub fn sync_out_indices_left(&self) -> Option<Vec<usize>> {
        let n_out_motor = self.config.output_layer.n_neurons
            + self.config.motor_layer.n_neurons + self.config.d_input;
        let temp = SyncAccumulator::new(self.config.n_sync_out, n_out_motor);
        Some(temp.indices_left)
    }

    /// Get sync_out pair right indices.
    pub fn sync_out_indices_right(&self) -> Option<Vec<usize>> {
        let n_out_motor = self.config.output_layer.n_neurons
            + self.config.motor_layer.n_neurons + self.config.d_input;
        let temp = SyncAccumulator::new(self.config.n_sync_out, n_out_motor);
        Some(temp.indices_right)
    }

    /// Returns references to all 8 synapses in dispatch order.
    pub fn synapse_refs(&self) -> [&Synapse; 8] {
        [
            &self.syn_motor_input, &self.syn_input_attn,
            &self.syn_attn_output, &self.syn_output_motor,
            &self.syn_cerebellum, &self.syn_basal_ganglia,
            &self.syn_insula, &self.syn_hippocampus,
        ]
    }

    /// Returns (in_dim, out_dim) for each synapse's underlying Linear layer.
    pub fn synapse_dims(&self) -> [(usize, usize); 8] {
        self.synapse_refs().map(|s| (s.linear.in_dim, s.linear.out_dim))
    }

    /// Initialize a CtmTickState from these weights, copying start traces/activations.
    pub fn init_tick_state(&self) -> CtmTickState {
        let mut ts = CtmTickState::new(&self.config);

        // Copy start traces into the flat arena
        let regions: [&NeuronLayerWeights; 8] = [
            &self.input_region, &self.attention_region,
            &self.output_region, &self.motor_region,
            &self.cerebellum_region, &self.basal_ganglia_region,
            &self.insula_region, &self.hippocampus_region,
        ];
        for (i, region) in regions.iter().enumerate() {
            let trace_slice = ts.trace_mut(i);
            let src = &region.start_trace;
            trace_slice[..src.len()].copy_from_slice(src);

            let act_slice = ts.act_mut(i);
            let src = &region.start_activated;
            act_slice[..src.len()].copy_from_slice(src);
        }

        ts
    }

    /// Save weights. Format by extension: `.bin` → bincode, `.json` → JSON.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        modgrad_persist::persist::save(self, path).map_err(|e| e.into())
    }

    /// Load weights. Format by extension, with JSON fallback for legacy files.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        modgrad_persist::persist::load(path).map_err(|e| e.into())
    }
}

// ─── Ctm (combined weights + session, backward compat) ─────

/// The full 8-region continuous thought machine.
///
/// This is the backward-compatible wrapper that holds both `CtmWeights`
/// and `CtmSession` together. Existing code that uses `Ctm` directly
/// continues to work unchanged. New parallel code should use
/// `CtmWeights` + `CtmSession` separately.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ctm {
    pub config: CtmConfig,

    // Brain regions
    pub input_region: NeuronLayer,
    pub attention_region: NeuronLayer,
    pub output_region: NeuronLayer,
    pub motor_region: NeuronLayer,

    // Subcortical regions
    pub cerebellum_region: NeuronLayer,
    pub basal_ganglia_region: NeuronLayer,

    // New regions (v3)
    pub insula_region: NeuronLayer,
    pub hippocampus_region: NeuronLayer,

    // Inter-region synapses
    pub syn_motor_input: Synapse,
    pub syn_input_attn: Synapse,
    pub syn_attn_output: Synapse,
    pub syn_output_motor: Synapse,
    pub syn_cerebellum: Synapse,       // motor + observation → cerebellum
    pub syn_basal_ganglia: Synapse,    // output + dopamine → BG
    pub syn_insula: Synapse,           // proprioceptive observation → insula
    pub syn_hippocampus: Synapse,      // all cortical regions concat → hippocampus

    /// Hippocampal content-addressable memory: one-shot (key,value) store.
    /// The NeuronLayer (hippocampus_region) gates what to store/retrieve.
    /// The CAM is the actual memory. Separate from replay buffer (which stores
    /// observations for sleep consolidation). This stores cortical patterns
    /// for within-tick pattern completion.
    #[serde(default)]
    pub hippo_cam: HippocampalCAM,
    /// Last hippocampal retrieval result (used as input to output + insula next tick).
    #[serde(skip)]
    pub hippo_retrieval: Vec<f32>,

    /// Basal ganglia eligibility traces: records recent pre×post activity.
    /// When dopamine arrives (reward/punishment), the trace determines
    /// which synapses to strengthen or weaken.
    /// This is the core of reinforcement learning in the striatum.
    /// Shape: [bg_neurons × (output_neurons + 1)]
    #[serde(default)]
    pub bg_eligibility: Vec<f32>,
    /// Dopamine baseline for BG (running average of DA).
    /// Reward = DA - baseline (positive surprise).
    /// Punishment = DA - baseline (negative surprise).
    #[serde(default = "default_bg_da_baseline")]
    pub bg_da_baseline: f32,

    // Global broadcast
    pub global_projector: Linear,

    // Output projection (sync → prediction)
    pub output_projector: Linear,

    // Local Hebbian per region
    pub hebb_input: LocalHebbian,
    pub hebb_attention: LocalHebbian,
    pub hebb_output: LocalHebbian,
    pub hebb_motor: LocalHebbian,
    pub hebb_cerebellum: LocalHebbian,
    pub hebb_basal_ganglia: LocalHebbian,
    pub hebb_insula: LocalHebbian,
    pub hebb_hippocampus: LocalHebbian,

    // Logit projector: sync → logit bias predictions.
    // This is how the CTM produces answers from its own weights
    // after consolidation (the graduation output path).
    // Maps sync signal to (token_id_prediction, strength_prediction) pairs.
    // Trained during sleep from stored episode logit biases.
    pub logit_projector: Option<Linear>,

    /// Collected (sync, logit_bias) pairs for training logit_projector during sleep.
    #[serde(skip)]
    pub logit_traces: Vec<(Vec<f32>, Vec<f32>)>,

    // Sleep consolidation
    #[serde(skip)]
    pub sleep: SleepConsolidation,

    // Replay buffer for prioritized consolidation
    #[serde(skip)]
    pub replay: ReplayBuffer,

    pub hebbian_enabled: bool,

    /// Device mesh: maps regions to compute devices.
    /// Runtime only — not serialized with the model.
    #[serde(skip)]
    pub device_mesh: Option<modgrad_device::device::DeviceMesh>,

    /// Persistent position predictor for sync_out accumulator.
    /// Lives on Ctm (persistent) rather than CtmState (transient).
    /// The SyncAccumulator in init_state() copies a reference to this.
    #[serde(skip)]
    pub position_predictor: Option<PositionPredictor>,

    /// Optional telemetry recorder. Set by harness before forward pass.
    /// Lives on Ctm (not CtmState) to avoid lifetime complications.
    #[serde(skip)]
    pub telemetry: Option<modgrad_io::telemetry::Telemetry>,
}

impl Ctm {
    pub fn new(config: CtmConfig) -> Self {
        let sizes = [
            config.input_layer.n_neurons,
            config.attention_layer.n_neurons,
            config.output_layer.n_neurons,
            config.motor_layer.n_neurons,
        ];
        let cortical_total = sizes.iter().sum::<usize>();
        let total = cortical_total
            + config.cerebellum_layer.n_neurons
            + config.basal_ganglia_layer.n_neurons
            + config.insula_layer.n_neurons
            + config.hippocampus_layer.n_neurons;
        let g = if config.global_broadcast_dim > 0 {
            config.global_broadcast_dim
        } else {
            total / 4
        };

        let g_in = if config.input_layer.receives_broadcast { g } else { 0 };
        let g_attn = if config.attention_layer.receives_broadcast { g } else { 0 };
        let g_out = if config.output_layer.receives_broadcast { g } else { 0 };
        let g_motor = if config.motor_layer.receives_broadcast { g } else { 0 };

        Self {
            input_region: NeuronLayer::new(&config.input_layer),
            attention_region: NeuronLayer::new(&config.attention_layer),
            output_region: NeuronLayer::new(&config.output_layer),
            motor_region: NeuronLayer::new(&config.motor_layer),

            // Subcortical regions
            cerebellum_region: NeuronLayer::new(&config.cerebellum_layer),
            basal_ganglia_region: NeuronLayer::new(&config.basal_ganglia_layer),

            // New regions (v3)
            insula_region: NeuronLayer::new(&config.insula_layer),
            hippocampus_region: NeuronLayer::new(&config.hippocampus_layer),

            // Inter-region synapses — per-region depth (U-Net when ≥ 3)
            syn_motor_input: Synapse::new_with_depth_and_min_width(
                config.d_input + sizes[3] + g_in, sizes[0],
                config.input_layer.synapse_depth, config.input_layer.unet_min_width),
            syn_input_attn: Synapse::new_with_depth_and_min_width(
                sizes[0] + config.d_input + g_attn, sizes[1],
                config.attention_layer.synapse_depth, config.attention_layer.unet_min_width),
            syn_attn_output: Synapse::new_with_depth_and_min_width(
                sizes[1] + config.d_input + g_out, sizes[2],
                config.output_layer.synapse_depth, config.output_layer.unet_min_width),
            syn_output_motor: Synapse::new_with_depth_and_min_width(
                sizes[2] + config.basal_ganglia_layer.n_neurons + g_motor, sizes[3],
                config.motor_layer.synapse_depth, config.motor_layer.unet_min_width),
            syn_cerebellum: Synapse::new_with_depth_and_min_width(
                sizes[3] + config.d_input, config.cerebellum_layer.n_neurons,
                config.cerebellum_layer.synapse_depth, config.cerebellum_layer.unet_min_width),
            syn_basal_ganglia: Synapse::new_with_depth_and_min_width(
                sizes[2] + 1, config.basal_ganglia_layer.n_neurons,
                config.basal_ganglia_layer.synapse_depth, config.basal_ganglia_layer.unet_min_width),
            syn_insula: Synapse::new_with_depth_and_min_width(
                config.d_input + cortical_total, config.insula_layer.n_neurons,
                config.insula_layer.synapse_depth, config.insula_layer.unet_min_width),
            syn_hippocampus: Synapse::new_with_depth_and_min_width(
                sizes[0] + sizes[1] + sizes[2] + sizes[3], config.hippocampus_layer.n_neurons,
                config.hippocampus_layer.synapse_depth, config.hippocampus_layer.unet_min_width),
            // Hippocampal CAM: key = cortical concat, value = same (auto-associative)
            hippo_cam: HippocampalCAM::new(
                64, // capacity (tunable via Epigenome tier)
                sizes[0] + sizes[1] + sizes[2] + sizes[3], // key_dim = cortical total
                sizes[0] + sizes[1] + sizes[2] + sizes[3], // value_dim = same (pattern completion)
            ),
            hippo_retrieval: vec![0.0; sizes[0] + sizes[1] + sizes[2] + sizes[3]],
            bg_eligibility: vec![0.0; config.basal_ganglia_layer.n_neurons * (sizes[2] + 1)],
            bg_da_baseline: 0.7,

            global_projector: Linear::new(total, g * 2),
            output_projector: Linear::new(config.n_sync_out, config.out_dims),

            hebb_input: LocalHebbian::new(sizes[0], config.input_layer.hebbian_lr),
            hebb_attention: LocalHebbian::new(sizes[1], config.attention_layer.hebbian_lr),
            hebb_output: LocalHebbian::new(sizes[2], config.output_layer.hebbian_lr),
            hebb_motor: LocalHebbian::new(sizes[3], config.motor_layer.hebbian_lr),
            hebb_cerebellum: LocalHebbian::new(config.cerebellum_layer.n_neurons, config.cerebellum_layer.hebbian_lr),
            hebb_basal_ganglia: LocalHebbian::new(config.basal_ganglia_layer.n_neurons, config.basal_ganglia_layer.hebbian_lr),
            hebb_insula: LocalHebbian::new(config.insula_layer.n_neurons, config.insula_layer.hebbian_lr),
            hebb_hippocampus: LocalHebbian::new(config.hippocampus_layer.n_neurons, config.hippocampus_layer.hebbian_lr),

            logit_projector: None,
            logit_traces: Vec::new(),
            sleep: SleepConsolidation::new(50000),
            replay: ReplayBuffer::new(64, 1.5),
            hebbian_enabled: false,
            device_mesh: None,
            position_predictor: {
                let n_out_motor = config.output_layer.n_neurons + config.motor_layer.n_neurons + config.d_input;
                if n_out_motor >= 16 && config.n_sync_out >= 8 {
                    Some(PositionPredictor::new(n_out_motor, config.n_sync_out))
                } else {
                    None
                }
            },
            telemetry: None,

            config,
        }
    }

    /// Set the device mesh for multi-GPU region distribution.
    pub fn set_device_mesh(&mut self, mesh: modgrad_device::device::DeviceMesh) {
        self.device_mesh = Some(mesh);
    }

    /// Attach a telemetry recorder. Each tick in the forward loop will
    /// write one binary record. Pass `None` to detach.
    /// Get sync_out pair left indices (for telemetry manifest).
    pub fn sync_out_indices_left(&self) -> Option<Vec<usize>> {
        // The sync accumulator is created fresh in init_state(), so we
        // create a temporary one to get the indices.
        let n_out_motor = self.config.output_layer.n_neurons
            + self.config.motor_layer.n_neurons + self.config.d_input;
        let temp = SyncAccumulator::new(self.config.n_sync_out, n_out_motor);
        Some(temp.indices_left)
    }

    /// Get sync_out pair right indices.
    pub fn sync_out_indices_right(&self) -> Option<Vec<usize>> {
        let n_out_motor = self.config.output_layer.n_neurons
            + self.config.motor_layer.n_neurons + self.config.d_input;
        let temp = SyncAccumulator::new(self.config.n_sync_out, n_out_motor);
        Some(temp.indices_right)
    }

    pub fn set_telemetry(&mut self, t: Option<modgrad_io::telemetry::Telemetry>) {
        self.telemetry = t;
    }

    /// Initialize state for a new input.
    pub fn init_state(&self) -> CtmState {
        // Sync accumulator sees output + motor + observation (d_input).
        // Including observation prevents sync collapse by injecting input identity.
        let n_out_motor = self.config.output_layer.n_neurons + self.config.motor_layer.n_neurons
            + self.config.d_input;
        let n_in_attn = self.config.input_layer.n_neurons + self.config.attention_layer.n_neurons;

        CtmState {
            trace_input: self.input_region.start_trace.clone(),
            trace_attention: self.attention_region.start_trace.clone(),
            trace_output: self.output_region.start_trace.clone(),
            trace_motor: self.motor_region.start_trace.clone(),
            trace_cerebellum: self.cerebellum_region.start_trace.clone(),
            trace_basal_ganglia: self.basal_ganglia_region.start_trace.clone(),
            act_input: self.input_region.start_activated.clone(),
            act_attention: self.attention_region.start_activated.clone(),
            act_output: self.output_region.start_activated.clone(),
            act_motor: self.motor_region.start_activated.clone(),
            act_cerebellum: self.cerebellum_region.start_activated.clone(),
            act_basal_ganglia: self.basal_ganglia_region.start_activated.clone(),
            trace_insula: self.insula_region.start_trace.clone(),
            trace_hippocampus: self.hippocampus_region.start_trace.clone(),
            act_insula: self.insula_region.start_activated.clone(),
            act_hippocampus: self.hippocampus_region.start_activated.clone(),
            sync_out: SyncAccumulator::new(self.config.n_sync_out, n_out_motor),
            sync_action: SyncAccumulator::new(self.config.n_sync_action, n_in_attn),
            neuromod: Neuromodulators::default(),
            modulation: default_modulation(),
            tick_traces: Vec::new(),
            motor_evidence: vec![0.0; self.config.motor_layer.n_neurons],
            motor_decided: false,
            decision_tick: None,
            last_sync: vec![0.0; self.config.n_sync_out],
            phase_input: vec![0.5; self.config.input_layer.n_neurons],
            phase_attention: vec![0.5; self.config.attention_layer.n_neurons],
            phase_output: vec![0.5; self.config.output_layer.n_neurons],
            phase_motor: vec![0.5; self.config.motor_layer.n_neurons],
            noise_rng: SimpleRng::new(self.config.n_sync_out as u64 * 31337),
            noisy: false,
            sleep_phase: SleepPhase::Awake,
            // Max region size / column size + 1
            column_noise_buf: vec![0.0; 512],
        }
    }

    /// Save CTM. Format by extension: `.bin` → bincode, `.json` → JSON.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        modgrad_persist::persist::save(self, path).map_err(|e| e.into())
    }

    /// Load CTM. Format by extension, with JSON fallback for legacy files.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        modgrad_persist::persist::load(path).map_err(|e| e.into())
    }

    // ─── Split / Merge for parallelism ──────────────────────────

    /// Split this Ctm into immutable weights and mutable session.
    /// Use this to create `Arc<CtmWeights>` shared across threads,
    /// with each thread getting its own `CtmSession`.
    pub fn into_split(self) -> (CtmWeights, CtmSession) {
        let bg_weight_len = self.syn_basal_ganglia.linear.weight.len();
        let weights = CtmWeights {
            config: self.config.clone(),
            input_region: self.input_region.to_weights(),
            attention_region: self.attention_region.to_weights(),
            output_region: self.output_region.to_weights(),
            motor_region: self.motor_region.to_weights(),
            cerebellum_region: self.cerebellum_region.to_weights(),
            basal_ganglia_region: self.basal_ganglia_region.to_weights(),
            insula_region: self.insula_region.to_weights(),
            hippocampus_region: self.hippocampus_region.to_weights(),
            syn_motor_input: self.syn_motor_input,
            syn_input_attn: self.syn_input_attn,
            syn_attn_output: self.syn_attn_output,
            syn_output_motor: self.syn_output_motor,
            syn_cerebellum: self.syn_cerebellum,
            syn_basal_ganglia: self.syn_basal_ganglia,
            syn_insula: self.syn_insula,
            syn_hippocampus: self.syn_hippocampus,
            global_projector: self.global_projector,
            output_projector: self.output_projector,
            logit_projector: self.logit_projector,
            device_mesh: self.device_mesh,
            hopfield_readout: if self.config.use_hopfield_readout {
                let n_out_motor = self.config.output_layer.n_neurons
                    + self.config.motor_layer.n_neurons + self.config.d_input;
                Some(HopfieldReadout::new(n_out_motor, self.config.n_sync_out, self.config.hopfield_d_key))
            } else { None },
            position_predictor: self.position_predictor,
            sync_out_indices_l: Vec::new(),
            sync_out_indices_r: Vec::new(),
            sync_out_decay: Vec::new(),
            sync_action_indices_l: Vec::new(),
            sync_action_indices_r: Vec::new(),
            sync_action_decay: Vec::new(),
        };

        let session = CtmSession {
            hippo_cam: self.hippo_cam,
            hippo_retrieval: self.hippo_retrieval,
            bg_eligibility: self.bg_eligibility,
            bg_da_baseline: self.bg_da_baseline,
            input_state: self.input_region.to_state(),
            attention_state: self.attention_region.to_state(),
            output_state: self.output_region.to_state(),
            motor_state: self.motor_region.to_state(),
            cerebellum_state: self.cerebellum_region.to_state(),
            basal_ganglia_state: self.basal_ganglia_region.to_state(),
            insula_state: self.insula_region.to_state(),
            hippocampus_state: self.hippocampus_region.to_state(),
            hebb_input: self.hebb_input,
            hebb_attention: self.hebb_attention,
            hebb_output: self.hebb_output,
            hebb_motor: self.hebb_motor,
            hebb_cerebellum: self.hebb_cerebellum,
            hebb_basal_ganglia: self.hebb_basal_ganglia,
            hebb_insula: self.hebb_insula,
            hebb_hippocampus: self.hebb_hippocampus,
            sleep: self.sleep,
            logit_traces: self.logit_traces,
            replay: self.replay,
            hebbian_enabled: self.hebbian_enabled,
            telemetry: self.telemetry,
            motor_memory: HippocampalCAM::new(
                256,
                self.config.input_layer.n_neurons + self.config.attention_layer.n_neurons
                    + self.config.output_layer.n_neurons + self.config.motor_layer.n_neurons,
                self.config.motor_layer.n_neurons,
            ),
            motor_bias: vec![0.0; self.config.motor_layer.n_neurons],
            bg_weight_delta: vec![0.0; bg_weight_len],
            syn_deltas: Vec::new(),
            syn_eligibility: Vec::new(),
            syn_reward_baseline: Vec::new(),
            last_reward: 1.0,
            syn_calcium: Vec::new(),
            syn_init_weights: Vec::new(),
            syn_salience: Vec::new(),
            bptt_caches: None,
        };

        (weights, session)
    }

    /// Reassemble a Ctm from weights + session.
    /// Used after parallel batch processing to merge state back.
    pub fn from_split(weights: CtmWeights, session: CtmSession) -> Self {
        Self {
            input_region: NeuronLayer::from_split(&weights.input_region, &session.input_state),
            attention_region: NeuronLayer::from_split(&weights.attention_region, &session.attention_state),
            output_region: NeuronLayer::from_split(&weights.output_region, &session.output_state),
            motor_region: NeuronLayer::from_split(&weights.motor_region, &session.motor_state),
            cerebellum_region: NeuronLayer::from_split(&weights.cerebellum_region, &session.cerebellum_state),
            basal_ganglia_region: NeuronLayer::from_split(&weights.basal_ganglia_region, &session.basal_ganglia_state),
            insula_region: NeuronLayer::from_split(&weights.insula_region, &session.insula_state),
            hippocampus_region: NeuronLayer::from_split(&weights.hippocampus_region, &session.hippocampus_state),
            syn_motor_input: weights.syn_motor_input,
            syn_input_attn: weights.syn_input_attn,
            syn_attn_output: weights.syn_attn_output,
            syn_output_motor: weights.syn_output_motor,
            syn_cerebellum: weights.syn_cerebellum,
            syn_basal_ganglia: weights.syn_basal_ganglia,
            syn_insula: weights.syn_insula,
            syn_hippocampus: weights.syn_hippocampus,
            hippo_cam: session.hippo_cam,
            hippo_retrieval: session.hippo_retrieval,
            bg_eligibility: session.bg_eligibility,
            bg_da_baseline: session.bg_da_baseline,
            global_projector: weights.global_projector,
            output_projector: weights.output_projector,
            hebb_input: session.hebb_input,
            hebb_attention: session.hebb_attention,
            hebb_output: session.hebb_output,
            hebb_motor: session.hebb_motor,
            hebb_cerebellum: session.hebb_cerebellum,
            hebb_basal_ganglia: session.hebb_basal_ganglia,
            hebb_insula: session.hebb_insula,
            hebb_hippocampus: session.hebb_hippocampus,
            logit_projector: weights.logit_projector,
            logit_traces: session.logit_traces,
            sleep: session.sleep,
            replay: session.replay,
            hebbian_enabled: session.hebbian_enabled,
            device_mesh: weights.device_mesh,
            position_predictor: weights.position_predictor,
            telemetry: session.telemetry,
            config: weights.config,
        }
    }

    /// Borrow the immutable weights view without consuming self.
    /// Returns a temporary CtmWeights by cloning. For zero-copy, use `into_split()`.
    pub fn weights(&self) -> CtmWeights {
        CtmWeights {
            config: self.config.clone(),
            input_region: self.input_region.to_weights(),
            attention_region: self.attention_region.to_weights(),
            output_region: self.output_region.to_weights(),
            motor_region: self.motor_region.to_weights(),
            cerebellum_region: self.cerebellum_region.to_weights(),
            basal_ganglia_region: self.basal_ganglia_region.to_weights(),
            insula_region: self.insula_region.to_weights(),
            hippocampus_region: self.hippocampus_region.to_weights(),
            syn_motor_input: self.syn_motor_input.clone(),
            syn_input_attn: self.syn_input_attn.clone(),
            syn_attn_output: self.syn_attn_output.clone(),
            syn_output_motor: self.syn_output_motor.clone(),
            syn_cerebellum: self.syn_cerebellum.clone(),
            syn_basal_ganglia: self.syn_basal_ganglia.clone(),
            syn_insula: self.syn_insula.clone(),
            syn_hippocampus: self.syn_hippocampus.clone(),
            global_projector: self.global_projector.clone(),
            output_projector: self.output_projector.clone(),
            logit_projector: self.logit_projector.clone(),
            hopfield_readout: if self.config.use_hopfield_readout {
                let n_out_motor = self.config.output_layer.n_neurons
                    + self.config.motor_layer.n_neurons + self.config.d_input;
                Some(HopfieldReadout::new(n_out_motor, self.config.n_sync_out, self.config.hopfield_d_key))
            } else { None },
            device_mesh: self.device_mesh.clone(),
            position_predictor: self.position_predictor.clone(),
            sync_out_indices_l: Vec::new(),
            sync_out_indices_r: Vec::new(),
            sync_out_decay: Vec::new(),
            sync_action_indices_l: Vec::new(),
            sync_action_indices_r: Vec::new(),
            sync_action_decay: Vec::new(),
        }
    }

    /// Borrow the mutable session view without consuming self.
    /// Returns a temporary CtmSession by cloning.
    pub fn session(&self) -> CtmSession {
        CtmSession {
            hippo_cam: self.hippo_cam.clone(),
            hippo_retrieval: self.hippo_retrieval.clone(),
            bg_eligibility: self.bg_eligibility.clone(),
            bg_da_baseline: self.bg_da_baseline,
            input_state: self.input_region.to_state(),
            attention_state: self.attention_region.to_state(),
            output_state: self.output_region.to_state(),
            motor_state: self.motor_region.to_state(),
            cerebellum_state: self.cerebellum_region.to_state(),
            basal_ganglia_state: self.basal_ganglia_region.to_state(),
            insula_state: self.insula_region.to_state(),
            hippocampus_state: self.hippocampus_region.to_state(),
            hebb_input: self.hebb_input.clone(),
            hebb_attention: self.hebb_attention.clone(),
            hebb_output: self.hebb_output.clone(),
            hebb_motor: self.hebb_motor.clone(),
            hebb_cerebellum: self.hebb_cerebellum.clone(),
            hebb_basal_ganglia: self.hebb_basal_ganglia.clone(),
            hebb_insula: self.hebb_insula.clone(),
            hebb_hippocampus: self.hebb_hippocampus.clone(),
            sleep: self.sleep.clone(),
            logit_traces: self.logit_traces.clone(),
            replay: self.replay.clone(),
            hebbian_enabled: self.hebbian_enabled,
            telemetry: self.telemetry.clone(),
            motor_memory: HippocampalCAM::new(
                256,
                self.config.input_layer.n_neurons + self.config.attention_layer.n_neurons
                    + self.config.output_layer.n_neurons + self.config.motor_layer.n_neurons,
                self.config.motor_layer.n_neurons,
            ),
            motor_bias: vec![0.0; self.config.motor_layer.n_neurons],
            bg_weight_delta: vec![0.0; self.syn_basal_ganglia.linear.weight.len()],
            syn_deltas: Vec::new(),
            syn_eligibility: Vec::new(),
            syn_reward_baseline: Vec::new(),
            last_reward: 1.0,
            syn_calcium: Vec::new(),
            syn_init_weights: Vec::new(),
            syn_salience: Vec::new(),
            bptt_caches: None,
        }
    }
}
