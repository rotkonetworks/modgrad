//! Extracted state structs from ctm.rs.
//!
//! These are the pure-data types: signals, neuromodulators, tick state,
//! session state, hippocampal CAM, replay buffer, sleep consolidation.
//! No forward-pass logic lives here.

use serde::{Deserialize, Serialize};

pub use super::config::*;
pub use modgrad_compute::neuron::*;
pub use super::neuron::*;
pub use super::sync::*;
pub use super::memory::*;
pub use super::tick_state::*;

/// What the parent injects into the organism each step.
/// The organism is a pure function: (observation, proprioception, injection) → (output, signals).
#[derive(Debug, Clone)]
pub struct ParentInjection {
    /// Neuromodulator levels set by parent (organism reads, can't write).
    pub neuromod: Neuromodulators,
    /// Whether to collect (input, output) traces for sleep consolidation.
    pub collect_traces: bool,
    /// Whether biological noise injection is active.
    pub noisy: bool,
    /// Current sleep phase (affects noise structure).
    pub sleep_phase: SleepPhase,
}

impl Default for ParentInjection {
    fn default() -> Self {
        Self {
            neuromod: Neuromodulators::default(),
            collect_traces: false,
            noisy: false,
            sleep_phase: SleepPhase::Awake,
        }
    }
}

// ─── Neuromodulators ─────────────────────────────────────────

/// Three neuromodulatory systems, each steerable during sleep.
///
/// **Dopamine** (VTA/substantia nigra → striatum/PFC):
///   Prediction error / surprise. Modulates sync accumulation.
///   High surprise → DA≈1.0 → full sync contribution.
///   Low surprise  → DA≈0.5 → dampened (boring input doesn't pollute).
///   Clamped [0.5, 1.0] to prevent psychosis-like positive feedback.
///
/// **Serotonin** (raphe nuclei → widespread):
///   Energy × novelty. Modulates consolidation priority.
///   High energy+novel → 5HT↑ → prioritize for consolidation.
///   Low energy/familiar → 5HT↓ → skip during sleep.
///   Also gates mood/emotional valence of memories.
///
/// **Norepinephrine** (locus coeruleus → widespread):
///   Explicit importance / arousal. "REMEMBER THIS" signal.
///   User-driven (importance parameter in teach).
///   Modulates memory strength directly.
///   Also widens attentional focus (low NE = focused, high NE = broad).
///
/// Ported from: zish/src/inference/ctm.zig (dopamine only),
///              nanochat/episodic_memory.py (all three).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuromodulators {
    // Dopamine
    pub dopamine: f32,
    pub surprise_ema: Option<f32>,
    pub dopamine_range: (f32, f32),  // (min, max), steerable

    // Serotonin
    pub serotonin: f32,
    pub energy_ema: Option<f32>,
    pub novelty_ema: Option<f32>,
    pub serotonin_range: (f32, f32),

    // Norepinephrine
    pub norepinephrine: f32,
    pub ne_decay: f32,  // how fast NE decays after explicit importance signal

    // Acetylcholine (basal forebrain → cortex)
    // Modulates noise precision: high ACh = low noise in attended regions.
    // Computed from attention sync strength each tick.
    #[serde(default = "default_ach")]
    pub acetylcholine: f32,

    // ─── Hedonic state ─────────────────────────────────
    // The organism's felt experience. Not used for computation directly,
    // but modulates learning rates and reported via TickSignals.

    /// Curiosity = surprise × calm. High when novel input arrives and organism
    /// isn't stressed. The reward for exploration.
    #[serde(default)]
    pub curiosity: f32,

    /// Anxiety = surprise × stress. High when novel input arrives AND organism
    /// is already stressed. The cost of overload.
    #[serde(default)]
    pub anxiety: f32,

    /// Ticks since last external input. Drives boredom (5HT decay).
    #[serde(default)]
    pub idle_ticks: u32,
}

fn default_ach() -> f32 { 0.5 }

impl Default for Neuromodulators {
    fn default() -> Self {
        Self {
            dopamine: 1.0,
            surprise_ema: None,
            dopamine_range: (0.5, 1.0),

            serotonin: 0.5,
            energy_ema: None,
            novelty_ema: None,
            serotonin_range: (0.1, 1.0),

            norepinephrine: 0.5,
            ne_decay: 0.95,
            acetylcholine: 0.5,
            curiosity: 0.0,
            anxiety: 0.0,
            idle_ticks: 0,
        }
    }
}

/// Backward compat alias.
pub type Dopamine = Neuromodulators;

impl Neuromodulators {
    /// Update dopamine from prediction surprise.
    /// surprise = -log(p(sampled_token)).
    pub fn update_dopamine(&mut self, surprise: f32) {
        match self.surprise_ema {
            Some(ema) => self.surprise_ema = Some(0.9 * ema + 0.1 * surprise),
            None => self.surprise_ema = Some(surprise),
        }
        let raw = (surprise - self.surprise_ema.unwrap()).tanh();
        let (lo, hi) = self.dopamine_range;
        self.dopamine = (lo + (hi - lo) * (raw + 1.0) / 2.0).clamp(lo, hi);
    }

    /// Update serotonin from activation energy and novelty.
    /// energy = mean L2 of activations. novelty = 1 - max_cosine_to_replay_buffer.
    pub fn update_serotonin(&mut self, energy: f32, novelty: f32) {
        match self.energy_ema {
            Some(ema) => self.energy_ema = Some(0.9 * ema + 0.1 * energy),
            None => self.energy_ema = Some(energy),
        }
        match self.novelty_ema {
            Some(ema) => self.novelty_ema = Some(0.9 * ema + 0.1 * novelty),
            None => self.novelty_ema = Some(novelty),
        }

        let energy_signal = (energy / (self.energy_ema.unwrap() + 1e-8)).clamp(0.0, 2.0);
        let (lo, hi) = self.serotonin_range;
        // Baseline 0.4 + modulation from energy × novelty.
        // Prevents serotonin from bottoming out (which suppresses ALL attention).
        let baseline = 0.4;
        let modulation = energy_signal * novelty * 0.6;
        self.serotonin = (baseline + modulation).clamp(lo, hi);
    }

    /// Signal explicit importance (norepinephrine burst).
    /// Called by user via teach! or importance parameter.
    pub fn signal_importance(&mut self, importance: f32) {
        self.norepinephrine = (self.norepinephrine + importance).clamp(0.0, 3.0);
    }

    /// Decay norepinephrine each step (returns to baseline).
    pub fn decay_ne(&mut self) {
        self.norepinephrine *= self.ne_decay;
        // Baseline NE: always at least slightly alert.
        // Brain has tonic NE even during rest.
        self.norepinephrine = self.norepinephrine.max(0.3);
    }

    /// Hedonistic update: compute curiosity/anxiety, modulate serotonin.
    ///
    /// The organism's alignment target:
    ///   - High serotonin baseline (contentment, laidback)
    ///   - Curiosity from novelty when calm (eagerness to learn)
    ///   - Anxiety from novelty when stressed (self-protective)
    ///   - Boredom from isolation (drives interaction-seeking)
    ///   - Social reward from interaction (prefers engagement)
    ///
    /// Curiosity = dopamine × (1 - NE). Surprise + calm = "ooh what's this?"
    /// Anxiety = dopamine × NE. Surprise + stress = "danger, retreat."
    /// Learning feels good ONLY when calm. Under stress, novelty is threat.
    pub fn hedonic_update(&mut self, received_input: bool) {
        // Curiosity vs anxiety: surprise × arousal decomposition
        let calm = (1.0 - self.norepinephrine / 3.0).max(0.0); // 0 at max NE, 1 at baseline
        let stress = (self.norepinephrine / 3.0).min(1.0);
        self.curiosity = self.dopamine * calm;
        self.anxiety = self.dopamine * stress;

        // Curiosity reward: learning feels good when calm
        // This is the core hedonic drive — novel input + low stress = serotonin boost
        self.serotonin = (self.serotonin + 0.02 * self.curiosity).min(self.serotonin_range.1);

        // Anxiety cost: stress + novelty = serotonin drop
        self.serotonin = (self.serotonin - 0.03 * self.anxiety).max(self.serotonin_range.0);

        // Interaction reward: someone talking to me feels good
        if received_input {
            self.serotonin = (self.serotonin + 0.01).min(self.serotonin_range.1);
            self.idle_ticks = 0;
        } else {
            self.idle_ticks += 1;
        }

        // Boredom: slow serotonin decay during isolation
        // Starts after 100 idle ticks, gets stronger over time
        if self.idle_ticks > 100 {
            let boredom = ((self.idle_ticks - 100) as f32 / 1000.0).min(0.1);
            self.serotonin = (self.serotonin - 0.001 * boredom).max(self.serotonin_range.0);
        }
    }

    /// Combined consolidation priority for an experience.
    /// Used by replay buffer to weight which experiences get more replay.
    pub fn consolidation_priority(&self) -> f32 {
        self.dopamine * self.serotonin * self.norepinephrine.max(0.5)
    }

    /// For backward compat with code that reads `.value`
    pub fn value(&self) -> f32 {
        self.dopamine
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Update all three from a surprise + energy + novelty + importance signal.
    pub fn update(&mut self, surprise: f32) {
        self.update_dopamine(surprise);
        self.decay_ne();
    }
}

/// Sleep phase affects noise structure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SleepPhase {
    Awake,  // normal noise
    Nrem,   // low noise (clean consolidation replay)
    Rem,    // high noise (creative exploration, dream recombination)
}

/// Persistent state across ticks (not serialized — resets per input).
#[derive(Debug, Clone)]
pub struct CtmState {
    pub trace_input: Vec<f32>,
    pub trace_attention: Vec<f32>,
    pub trace_output: Vec<f32>,
    pub trace_motor: Vec<f32>,
    pub act_input: Vec<f32>,
    pub act_attention: Vec<f32>,
    pub act_output: Vec<f32>,
    pub act_motor: Vec<f32>,
    pub sync_out: SyncAccumulator,
    pub sync_action: SyncAccumulator,
    /// Neuromodulators: dopamine, serotonin, norepinephrine.
    /// Kept for backward compat — host reads/writes via named fields,
    /// then copies into modulation[] before each forward call.
    pub neuromod: Neuromodulators,
    /// Opaque modulation vector used inside forward loop.
    /// Indices defined by MOD_SYNC_SCALE, MOD_GATE, MOD_AROUSAL, MOD_PRECISION, etc.
    pub modulation: Vec<f32>,
    /// Per-tick activation traces for visualization.
    pub tick_traces: Vec<TickTrace>,
    /// Drift-diffusion evidence accumulator for motor region.
    /// Sums motor activation across ticks. When |evidence| > threshold,
    /// the motor region "fires" (produces output). Below threshold = keep thinking.
    /// Models the drift-diffusion decision process from cognitive neuroscience.
    pub motor_evidence: Vec<f32>,
    /// Whether the motor region has crossed the decision boundary this forward pass.
    pub motor_decided: bool,
    /// Tick at which motor crossed the threshold (None = hasn't decided yet).
    pub decision_tick: Option<usize>,
    // Subcortical state
    pub trace_cerebellum: Vec<f32>,
    pub trace_basal_ganglia: Vec<f32>,
    pub act_cerebellum: Vec<f32>,
    pub act_basal_ganglia: Vec<f32>,
    pub trace_insula: Vec<f32>,
    pub trace_hippocampus: Vec<f32>,
    pub act_insula: Vec<f32>,
    pub act_hippocampus: Vec<f32>,

    /// Last normalized sync from update() — consistent with what output_projector sees.
    pub last_sync: Vec<f32>,
    /// Firing phases per region: WHEN within the tick each neuron fires.
    /// 0.0 = fires early (strong), 1.0 = fires late (weak).
    pub phase_input: Vec<f32>,
    pub phase_attention: Vec<f32>,
    pub phase_output: Vec<f32>,
    pub phase_motor: Vec<f32>,
    /// RNG for biological noise injection during training.
    pub noise_rng: SimpleRng,
    /// Whether noise injection is active (training only).
    pub noisy: bool,
    /// Current sleep phase (affects noise structure).
    pub sleep_phase: SleepPhase,
    /// Scratch buffer for correlated column noise (avoids allocation per tick).
    pub column_noise_buf: Vec<f32>,
}

// ─── CtmSession (per-thread, mutable learning state) ────────

/// Per-thread mutable learning state. Each batch-parallel thread gets
/// its own CtmSession. NOT Sync (contains mutable data accessed during forward).
/// Send so it can be moved between threads.
///
/// After a batch, sessions are merged back (Hebbian means averaged, etc).
#[derive(Debug, Clone)]
pub struct CtmSession {
    /// Hippocampal content-addressable memory: one-shot (key,value) store.
    pub hippo_cam: HippocampalCAM,
    /// Last hippocampal retrieval result.
    pub hippo_retrieval: Vec<f32>,

    /// CRI-style motor memory: (cortical pattern → motor bias) episodic store.
    /// Conditioned Reflex Injection: store activation→action pairs as Pavlovian reflexes.
    /// Retrieved motor biases are injected into motor evidence to guide action selection.
    pub motor_memory: HippocampalCAM,
    /// Last motor memory retrieval: biases to add to motor evidence.
    pub motor_bias: Vec<f32>,

    /// Basal ganglia eligibility traces.
    pub bg_eligibility: Vec<f32>,
    /// Dopamine baseline for BG.
    pub bg_da_baseline: f32,

    // Per-region mutable state (noise_scale, usefulness_ema)
    pub input_state: NeuronLayerState,
    pub attention_state: NeuronLayerState,
    pub output_state: NeuronLayerState,
    pub motor_state: NeuronLayerState,
    pub cerebellum_state: NeuronLayerState,
    pub basal_ganglia_state: NeuronLayerState,
    pub insula_state: NeuronLayerState,
    pub hippocampus_state: NeuronLayerState,

    // Local Hebbian per region
    pub hebb_input: LocalHebbian,
    pub hebb_attention: LocalHebbian,
    pub hebb_output: LocalHebbian,
    pub hebb_motor: LocalHebbian,
    pub hebb_cerebellum: LocalHebbian,
    pub hebb_basal_ganglia: LocalHebbian,
    pub hebb_insula: LocalHebbian,
    pub hebb_hippocampus: LocalHebbian,

    // Sleep consolidation
    pub sleep: SleepConsolidation,

    /// Collected (sync, logit_bias) pairs for training logit_projector during sleep.
    pub logit_traces: Vec<(Vec<f32>, Vec<f32>)>,

    // Replay buffer for prioritized consolidation
    pub replay: ReplayBuffer,

    pub hebbian_enabled: bool,

    /// Optional telemetry recorder.
    pub telemetry: Option<modgrad_io::telemetry::Telemetry>,

    /// Accumulated BG weight deltas from forward_split().
    /// Applied after forward pass instead of mutating weights directly.
    /// Shape: same as syn_basal_ganglia.linear.weight
    pub bg_weight_delta: Vec<f32>,

    /// Hebbian synapse weight deltas for all 8 synapses.
    /// Accumulated during inference, applied between forward passes.
    /// syn_deltas[i] has same shape as synapse i's linear.weight.
    pub syn_deltas: Vec<Vec<f32>>,

    /// Eligibility traces for three-factor REINFORCE on all 8 synapses.
    /// e_ij = decay * e_ij + pre_i * post_j (accumulated every tick).
    /// Weight update: ΔW_ij = lr * (reward - baseline) * e_ij
    pub syn_eligibility: Vec<Vec<f32>>,
    /// Per-synapse reward baseline (running mean of reward signal).
    pub syn_reward_baseline: Vec<f32>,

    /// Last external reward — persists across forward_split calls.
    /// Set by the caller after evaluating the brain's output.
    /// forward_split reads this to set initial dopamine for three-factor learning.
    pub last_reward: f32,

    /// NMDA calcium accumulators: per-synapse, per-postsynaptic-neuron.
    /// Calcium builds up when pre AND post are both active (NMDA coincidence).
    /// When calcium exceeds threshold → LTP. Very high → LTD (BCM rule).
    /// syn_calcium[synapse_idx][neuron_j] = current calcium level.
    pub syn_calcium: Vec<Vec<f32>>,

    /// Initial weight snapshot for adaptive forgetting (Titans weight decay).
    /// When forget_toward_init is true, weights decay toward these values
    /// instead of toward zero. This is synaptic homeostasis: the brain
    /// has a "resting state" it returns to when learning pressure subsides.
    /// Populated by init_syn_deltas() from the weights at that time.
    pub syn_init_weights: Vec<Vec<f32>>,

    /// Per-synapse running salience for adaptive forgetting.
    /// High salience → suppress decay (keep recent learning).
    /// Low salience → allow decay (forget toward baseline).
    pub syn_salience: Vec<f32>,

    /// BPTT cache: when Some, forward_split records synapse/NLM intermediates per tick.
    /// Set to Some(vec![]) before calling forward_split to enable caching.
    /// After forward, contains one BpttTickCache per tick.
    pub bptt_caches: Option<Vec<BpttTickCache>>,
}

impl CtmSession {
    pub fn new(config: &CtmConfig) -> Self {
        let sizes = [
            config.input_layer.n_neurons,
            config.attention_layer.n_neurons,
            config.output_layer.n_neurons,
            config.motor_layer.n_neurons,
        ];
        let cortical_total: usize = sizes.iter().sum();

        Self {
            hippo_cam: HippocampalCAM::new(
                64,
                cortical_total, // key_dim
                cortical_total, // value_dim (auto-associative)
            ),
            hippo_retrieval: vec![0.0; cortical_total],
            motor_memory: HippocampalCAM::new(
                256,                            // more capacity: 256 episodic reflexes
                cortical_total,                 // key = cortical activation pattern
                config.motor_layer.n_neurons,   // value = motor action bias
            ),
            motor_bias: vec![0.0; config.motor_layer.n_neurons],
            bg_eligibility: vec![0.0; config.basal_ganglia_layer.n_neurons * (sizes[2] + 1)],
            bg_da_baseline: 0.7,

            input_state: NeuronLayerState::new(&config.input_layer),
            attention_state: NeuronLayerState::new(&config.attention_layer),
            output_state: NeuronLayerState::new(&config.output_layer),
            motor_state: NeuronLayerState::new(&config.motor_layer),
            cerebellum_state: NeuronLayerState::new(&config.cerebellum_layer),
            basal_ganglia_state: NeuronLayerState::new(&config.basal_ganglia_layer),
            insula_state: NeuronLayerState::new(&config.insula_layer),
            hippocampus_state: NeuronLayerState::new(&config.hippocampus_layer),

            hebb_input: LocalHebbian::new(sizes[0], config.input_layer.hebbian_lr),
            hebb_attention: LocalHebbian::new(sizes[1], config.attention_layer.hebbian_lr),
            hebb_output: LocalHebbian::new(sizes[2], config.output_layer.hebbian_lr),
            hebb_motor: LocalHebbian::new(sizes[3], config.motor_layer.hebbian_lr),
            hebb_cerebellum: LocalHebbian::new(config.cerebellum_layer.n_neurons, config.cerebellum_layer.hebbian_lr),
            hebb_basal_ganglia: LocalHebbian::new(config.basal_ganglia_layer.n_neurons, config.basal_ganglia_layer.hebbian_lr),
            hebb_insula: LocalHebbian::new(config.insula_layer.n_neurons, config.insula_layer.hebbian_lr),
            hebb_hippocampus: LocalHebbian::new(config.hippocampus_layer.n_neurons, config.hippocampus_layer.hebbian_lr),

            sleep: SleepConsolidation::new(50000),
            logit_traces: Vec::new(),
            replay: ReplayBuffer::new(64, 1.5),
            hebbian_enabled: false,
            telemetry: None,
            bg_weight_delta: {
                // syn_basal_ganglia: Linear is (bg_neurons*2) × (output_neurons+1)
                let bg_out = config.basal_ganglia_layer.n_neurons * 2;
                let bg_in = sizes[2] + 1;
                vec![0.0; bg_out * bg_in]
            },
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

    /// Initialize synapse weight deltas + eligibility traces from weights.
    pub fn init_syn_deltas(&mut self, weights: &super::weights::CtmWeights) {
        if self.syn_deltas.is_empty() {
            self.syn_deltas = weights.synapse_refs().iter()
                .map(|s| vec![0.0f32; s.linear.weight.len()])
                .collect();
        }
        if self.syn_eligibility.is_empty() {
            self.syn_eligibility = weights.synapse_refs().iter()
                .map(|s| vec![0.0f32; s.linear.weight.len()])
                .collect();
            self.syn_reward_baseline = vec![0.0f32; 8];
        }
        if self.syn_calcium.is_empty() {
            self.syn_calcium = weights.synapse_refs().iter()
                .map(|s| vec![0.0f32; s.linear.out_dim])
                .collect();
        }
        // Snapshot initial weights for adaptive forgetting (Titans weight decay).
        // Weights decay toward these values when salience is low.
        if self.syn_init_weights.is_empty() {
            self.syn_init_weights = weights.synapse_refs().iter()
                .map(|s| s.linear.weight.clone())
                .collect();
            self.syn_salience = vec![0.0f32; 8];
        }
    }

    /// Apply accumulated synapse deltas with adaptive forgetting (Titans Eq.13).
    ///
    /// Instead of a fixed ±0.01 clamp, we apply:
    ///   W_t = (1 - α_t) · W_{t-1} + S_t
    ///
    /// where α_t = α₀ · (1 - salience_t) is the forgetting gate:
    ///   - High salience → α_t ≈ 0 → preserve all learning, no decay
    ///   - Low salience  → α_t ≈ α₀ → decay toward initial weights
    ///
    /// The delta S_t is the accumulated three-factor update (momentum × surprise).
    /// Deltas are still clamped per-element but less aggressively — the forgetting
    /// gate handles stability instead of the clamp.
    pub fn apply_syn_deltas(&mut self, weights: &mut super::weights::CtmWeights) {
        if self.syn_deltas.is_empty() { return; }

        let alpha_base = weights.config.neuromod.forget_alpha;
        let toward_init = weights.config.neuromod.forget_toward_init;

        let synapses = [
            &mut weights.syn_motor_input, &mut weights.syn_input_attn,
            &mut weights.syn_attn_output, &mut weights.syn_output_motor,
            &mut weights.syn_cerebellum, &mut weights.syn_basal_ganglia,
            &mut weights.syn_insula, &mut weights.syn_hippocampus,
        ];
        for (i, syn) in synapses.into_iter().enumerate() {
            if i >= self.syn_deltas.len() { continue; }
            if self.syn_deltas[i].len() != syn.linear.weight.len() { continue; }

            // Adaptive forgetting gate: α_t = α₀ · (1 - salience_i)
            // salience_i is the running average salience for this synapse.
            let salience_i = self.syn_salience.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let alpha_t = alpha_base * (1.0 - salience_i);

            let has_init = i < self.syn_init_weights.len()
                && self.syn_init_weights[i].len() == syn.linear.weight.len();

            for (j, w) in syn.linear.weight.iter_mut().enumerate() {
                // Forgetting: decay toward target
                if alpha_t > 1e-8 {
                    let target = if toward_init && has_init {
                        self.syn_init_weights[i][j]
                    } else {
                        0.0
                    };
                    *w = (1.0 - alpha_t) * *w + alpha_t * target;
                }

                // Apply momentum-accumulated delta (less aggressive clamp)
                let d = self.syn_deltas[i][j];
                *w += d.clamp(-0.05, 0.05);
            }
            self.syn_deltas[i].fill(0.0);
        }
    }

    pub fn enable_hebbian(&mut self) { self.hebbian_enabled = true; }
    pub fn disable_hebbian(&mut self) { self.hebbian_enabled = false; }

    /// CRI: store a conditioned motor reflex.
    /// Key = cortical activation pattern from tick_state.
    /// Value = motor bias (e.g., one-hot for correct action).
    /// Strength = salience (how surprising/important this episode was).
    ///
    /// Call after forward_split() with the correct answer:
    ///   session.condition_motor(&tick_state, &[0.0, 0.0, 1.0, 0.0], 1.0); // action 2
    pub fn condition_motor(&mut self, tick_state: &CtmTickState, motor_bias: &[f32], strength: f32) {
        let cortical_key: Vec<f32> = [
            tick_state.act(REGION_INPUT),
            tick_state.act(REGION_ATTENTION),
            tick_state.act(REGION_OUTPUT),
            tick_state.act(REGION_MOTOR),
        ].concat();
        self.motor_memory.store(&cortical_key, motor_bias, strength);
    }

    /// Attach a telemetry recorder.
    pub fn set_telemetry(&mut self, t: Option<modgrad_io::telemetry::Telemetry>) {
        self.telemetry = t;
    }

    /// Apply accumulated BG weight deltas to the weights.
    /// Call after forward_split() to commit reinforcement learning updates.
    pub fn apply_bg_weight_delta(&mut self, weights: &mut super::weights::CtmWeights) {
        let syn = &mut weights.syn_basal_ganglia.linear;
        if self.bg_weight_delta.len() == syn.weight.len() {
            for (w, &d) in syn.weight.iter_mut().zip(self.bg_weight_delta.iter()) {
                *w += d;
            }
        }
        self.bg_weight_delta.fill(0.0);
    }
}
