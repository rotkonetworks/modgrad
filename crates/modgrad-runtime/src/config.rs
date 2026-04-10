//! Configuration structs for the Continuous Thought Machine.
//!
//! Extracted from ctm.rs — pure data, no logic.

use serde::{Deserialize, Serialize};

pub fn default_par_threshold() -> usize { 32 }
fn default_hopfield_d_key() -> usize { 32 }
fn default_nmda_pre_threshold() -> f32 { 0.15 }
fn default_nmda_post_threshold() -> f32 { 0.2 }
fn default_calcium_decay() -> f32 { 0.9 }  // ~10 tick half-life
fn default_calcium_ltp_threshold() -> f32 { 0.3 }
fn default_calcium_ltd_threshold() -> f32 { 0.8 }
fn default_forget_alpha() -> f32 { 0.001 }
fn default_forget_toward_init() -> bool { true }
fn default_region_synapse_depth() -> usize { 1 }  // legacy: linear synapse
fn default_sub_ticks() -> usize { 1 }              // 1:1 with outer tick
fn default_unet_min_width() -> usize { 16 }

// ─── Modulation Vector ─────────────────────────────────────
// The organism receives neuromodulation as an opaque Vec<f32>.
// It uses mod[i] to modulate dynamics but doesn't know what
// each dimension "means." The layout is defined by the HOST.
// These indices are public so host and organism agree on positions,
// but the organism only sees them as numeric indices, not as named
// neurotransmitters. Pre-puberty: no labels. Post-Aware: host
// provides a name map.

/// Modulation vector dimension count.
pub const MOD_DIM: usize = 6;
/// Index: sync accumulator scaling (host calls it "dopamine").
pub const MOD_SYNC_SCALE: usize = 0;
/// Index: gating/consolidation priority (host calls it "serotonin").
pub const MOD_GATE: usize = 1;
/// Index: noise amplitude / arousal (host calls it "norepinephrine").
pub const MOD_AROUSAL: usize = 2;
/// Index: attention precision (host calls it "acetylcholine").
pub const MOD_PRECISION: usize = 3;
/// Index: hedonic curiosity signal (host-computed, read-only).
pub const MOD_CURIOSITY: usize = 4;
/// Index: hedonic anxiety signal (host-computed, read-only).
pub const MOD_ANXIETY: usize = 5;

/// Create a default modulation vector (neutral state).
pub fn default_modulation() -> Vec<f32> {
    let mut m = vec![0.5; MOD_DIM];
    m[MOD_SYNC_SCALE] = 1.0;  // full sync contribution
    m[MOD_GATE] = 0.5;        // moderate gating
    m[MOD_AROUSAL] = 0.5;     // moderate arousal
    m[MOD_PRECISION] = 0.5;   // moderate precision
    m[MOD_CURIOSITY] = 0.0;   // no curiosity
    m[MOD_ANXIETY] = 0.0;     // no anxiety
    m
}

// ─── Configuration ──────────────────────────────────────────

/// Configuration for a single neuron layer (brain region).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub n_neurons: usize,
    pub memory_length: usize,
    pub memory_hidden_dims: usize,
    pub nlm_depth: usize,        // 1=shallow (linear+GLU), 2=deep (2×SuperLinear+GLU)
    pub hebbian: bool,
    pub hebbian_lr: f32,
    pub receives_broadcast: bool,
    pub sparsity_target: f32,
    /// Fraction of neurons that are inhibitory (GABA). Default 0.2 (20%).
    /// Inhibitory neurons negate their output, creating competition
    /// and winner-take-all dynamics within the region.
    pub inhibitory_fraction: f32,
    /// U-Net synapse depth for this region. 1 = MLP (legacy), >1 = U-Net.
    /// Cortex: depth=4 (deep processing). Subcortex: depth=2 (fast reflex).
    #[serde(default = "default_region_synapse_depth")]
    pub synapse_depth: usize,
    /// Sub-ticks per outer tick. Subcortical regions fire faster than cortex.
    /// Cerebellum: 3-5× (fast error correction, ~200Hz vs cortex ~40Hz).
    /// Cortex: 1 (one update per outer tick).
    #[serde(default = "default_sub_ticks")]
    pub sub_ticks: usize,
    /// U-Net minimum bottleneck width.
    #[serde(default = "default_unet_min_width")]
    pub unet_min_width: usize,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            n_neurons: 64,
            memory_length: 8,
            memory_hidden_dims: 64,
            nlm_depth: 2,
            hebbian: true,
            hebbian_lr: 0.01,
            receives_broadcast: true,
            sparsity_target: 0.0,
            inhibitory_fraction: 0.3,
            synapse_depth: 1,   // legacy default (linear). Set >1 for U-Net.
            sub_ticks: 1,       // 1:1 with outer tick
            unet_min_width: 16,
        }
    }
}

/// Gamma cycle: the async synchronization rhythm.
///
/// Regions run async within a gamma window. At cycle boundaries:
///   1. Sync accumulator measures coherence
///   2. Hippocampus binds current pattern
///   3. Neuromod updates from prediction error
///   4. Motor checks decision threshold
///
/// Like JAM's slot model: async execution within window, commit at boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaCycleConfig {
    /// Minimum region firings before a commit point.
    /// Each region must fire at least this many times within a window.
    /// Lower = faster cycling, less computation per window.
    /// Higher = more intra-window deliberation.
    pub min_firings: usize,
    /// Maximum region firings before forced commit.
    /// Safety bound: prevent runaway async loops.
    pub max_firings: usize,
    /// Sync coherence threshold to trigger early commit.
    /// If sync coherence exceeds this, commit early (consensus reached).
    /// Range: 0.0 (never early) to 1.0 (commit immediately).
    pub coherence_threshold: f32,
    /// Whether arousal modulates cycle speed.
    /// High NE → shorter windows (reactive). Low NE → longer windows (contemplative).
    pub arousal_modulated: bool,
}

impl Default for GammaCycleConfig {
    fn default() -> Self {
        Self {
            min_firings: 1,
            max_firings: 4,
            coherence_threshold: 0.8,
            arousal_modulated: true,
        }
    }
}

/// Neuromodulation and plasticity parameters.
/// All values are hot-reloadable during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodConfig {
    // ── Synapse plasticity ──
    /// Oja's rule learning rate. 0.0 = frozen projections.
    pub hebb_syn_lr: f32,
    /// Dopamine gate: plasticity only fires when DA > this.
    pub da_gate: f32,

    // ── Dopamine dynamics ──
    /// Dopamine clamp range.
    pub da_min: f32,
    pub da_max: f32,
    /// DA update blend on prediction error (alpha * old + beta * new).
    pub da_error_alpha: f32,
    pub da_error_beta: f32,
    /// DA decay toward baseline when no error.
    pub da_decay: f32,
    /// Intra-tick DA update blend.
    pub da_intra_alpha: f32,
    pub da_intra_beta: f32,

    // ── Basal ganglia ──
    /// Eligibility trace decay per tick. Brain: ~0.97.
    /// This is the momentum η from Titans: controls how past surprise
    /// carries forward into the current update.
    pub bg_elig_decay: f32,
    /// Reward significance threshold for BG weight update.
    pub bg_reward_threshold: f32,
    /// DA baseline EMA for BG reward signal.
    pub bg_da_baseline_alpha: f32,

    // ── Adaptive forgetting (Titans weight decay) ──
    /// Base weight decay rate α₀. Actual α_t = α₀ × (1 - salience).
    /// High salience → low decay → preserve new learning.
    /// Low salience → high decay → forget toward initial weights.
    /// Titans Eq.13: M_t = (1 - α_t)·M_{t-1} + S_t
    #[serde(default = "default_forget_alpha")]
    pub forget_alpha: f32,
    /// Weight decay target: decay toward initial weights (true) or toward zero (false).
    /// Brain: synaptic homeostasis pulls weights toward baseline, not zero.
    #[serde(default = "default_forget_toward_init")]
    pub forget_toward_init: bool,

    // ── Attention / ACh ──
    /// Serotonin gate: suppress activations below this.
    pub serotonin_gate_factor: f32,
    /// ACh precision EMA alpha.
    pub ach_ema_alpha: f32,

    // ── Cerebellar learning ──
    /// Error threshold for cerebellar weight updates.
    pub cereb_error_threshold: f32,

    // ── Phase synchrony ──
    /// Gaussian kernel width for phase-aware sync binding.
    pub phase_sync_sigma: f32,

    // ── NMDA / calcium plasticity ──
    /// Pre-synaptic activation threshold for NMDA gate.
    /// Below this, the presynaptic signal is too weak to release glutamate.
    #[serde(default = "default_nmda_pre_threshold")]
    pub nmda_pre_threshold: f32,
    /// Post-synaptic depolarization threshold for NMDA unblock.
    /// NMDA channels are Mg²⁺-blocked at rest; post must be depolarized.
    #[serde(default = "default_nmda_post_threshold")]
    pub nmda_post_threshold: f32,
    /// Calcium decay per tick (τ). Higher = longer memory of coincidence.
    /// Brain: τ ≈ 100ms ≈ 10 gamma cycles.
    #[serde(default = "default_calcium_decay")]
    pub calcium_decay: f32,
    /// Calcium threshold for LTP (long-term potentiation).
    /// Below this, no plasticity. Above: strengthen synapse.
    #[serde(default = "default_calcium_ltp_threshold")]
    pub calcium_ltp_threshold: f32,
    /// Calcium threshold for LTD (long-term depression).
    /// Above LTP but below LTD: strengthen. Above LTD: weaken.
    /// This is the BCM rule: moderate activity → LTP, excessive → LTD.
    #[serde(default = "default_calcium_ltd_threshold")]
    pub calcium_ltd_threshold: f32,
}

impl Default for NeuromodConfig {
    fn default() -> Self {
        Self {
            hebb_syn_lr: 0.00005,
            da_gate: 1.2,
            da_min: 0.1,
            da_max: 3.0,
            da_error_alpha: 0.7,
            da_error_beta: 0.3,
            da_decay: 0.95,
            da_intra_alpha: 0.8,
            da_intra_beta: 0.2,
            bg_elig_decay: 0.97,
            bg_reward_threshold: 0.1,
            bg_da_baseline_alpha: 0.99,
            serotonin_gate_factor: 0.1,
            ach_ema_alpha: 0.8,
            cereb_error_threshold: 0.01,
            phase_sync_sigma: 0.3,
            forget_alpha: default_forget_alpha(),
            forget_toward_init: default_forget_toward_init(),
            nmda_pre_threshold: default_nmda_pre_threshold(),
            nmda_post_threshold: default_nmda_post_threshold(),
            calcium_decay: default_calcium_decay(),
            calcium_ltp_threshold: default_calcium_ltp_threshold(),
            calcium_ltd_threshold: default_calcium_ltd_threshold(),
        }
    }
}

/// Full CTM v2 configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtmConfig {
    /// Maximum ticks before forced output.
    /// In the async pipeline model, this is the max number of gamma cycles.
    /// The actual number may be lower if motor decides early.
    pub iterations: usize,
    pub d_model: usize,
    pub d_input: usize,
    pub heads: usize,
    pub n_sync_out: usize,
    pub n_sync_action: usize,
    pub synapse_depth: usize,
    pub out_dims: usize,
    pub global_broadcast_dim: usize,
    /// Motor decision threshold. Motor region only produces output when
    /// accumulated evidence exceeds this value. Higher = more deliberation.
    pub motor_threshold: f32,

    /// Neuromodulation and plasticity parameters.
    /// All hot-reloadable for tuning during training.
    #[serde(default)]
    pub neuromod: NeuromodConfig,

    /// Minimum neurons per region before rayon parallelism kicks in.
    /// Below this, sequential is faster (rayon scheduling overhead).
    #[serde(default = "default_par_threshold")]
    pub par_threshold: usize,
    // Gamma cycle config moved to TuningConfig (harness controls timing, not DNA).

    pub input_layer: LayerConfig,
    pub attention_layer: LayerConfig,
    pub output_layer: LayerConfig,
    pub motor_layer: LayerConfig,

    /// Cerebellum: forward model predicting next observation from motor + sensory.
    /// "What will I see if I do this?" Drives prediction error → dopamine.
    #[serde(default)]
    pub cerebellum_layer: LayerConfig,
    /// Basal ganglia: habitual action selection from output + reward.
    /// "Which action worked before?" Biases motor region.
    #[serde(default)]
    pub basal_ganglia_layer: LayerConfig,
    /// Insula: interoceptive processing (body sensors, proprioception).
    /// "How does my body feel?" Feeds into global broadcast for self-regulation.
    #[serde(default)]
    pub insula_layer: LayerConfig,
    /// Hippocampus: fast one-shot episodic binding.
    /// "I've seen this before." Pattern completion from single exposure.
    #[serde(default)]
    pub hippocampus_layer: LayerConfig,

    /// Use Hopfield/attention readout instead of SyncAccumulator.
    /// Maps to hippocampal pattern completion in the brain.
    #[serde(default)]
    pub use_hopfield_readout: bool,
    /// Key dimension for Hopfield readout (query/key comparison space).
    /// Analogous to entorhinal cortex grid cell dimensionality.
    #[serde(default = "default_hopfield_d_key")]
    pub hopfield_d_key: usize,
}

impl Default for CtmConfig {
    fn default() -> Self {
        Self {
            iterations: 16,
            d_model: 256,
            d_input: 64,
            heads: 4,
            n_sync_out: 64,
            n_sync_action: 32,
            synapse_depth: 1,
            out_dims: 4,
            global_broadcast_dim: 0,
            // Higher threshold = more deliberation before deciding.
            // Autotuned from 5.0 → 10.0 on QEC benchmark.
            motor_threshold: 10.0,
            neuromod: NeuromodConfig::default(),
            par_threshold: 32,

            input_layer: LayerConfig {
                n_neurons: 64, memory_length: 4, nlm_depth: 1,
                hebbian_lr: 0.02, receives_broadcast: true,
                ..Default::default()
            },
            attention_layer: LayerConfig {
                n_neurons: 64, memory_length: 8, nlm_depth: 2,
                hebbian_lr: 0.01, receives_broadcast: false,
                sparsity_target: 0.1,
                ..Default::default()
            },
            output_layer: LayerConfig {
                n_neurons: 64, memory_length: 16, nlm_depth: 2,
                hebbian_lr: 0.005, receives_broadcast: true,
                ..Default::default()
            },
            motor_layer: LayerConfig {
                n_neurons: 64, memory_length: 4, nlm_depth: 1,
                hebbian_lr: 0.005, receives_broadcast: true,
                ..Default::default()
            },
            // Subcortical: small by default (12% of cortical)
            cerebellum_layer: LayerConfig {
                n_neurons: 8, memory_length: 4, nlm_depth: 1,
                hebbian_lr: 0.01, receives_broadcast: false,
                ..Default::default()
            },
            basal_ganglia_layer: LayerConfig {
                n_neurons: 8, memory_length: 8, nlm_depth: 1,
                hebbian_lr: 0.01, receives_broadcast: false,
                ..Default::default()
            },
            insula_layer: LayerConfig {
                n_neurons: 8, memory_length: 4, nlm_depth: 1,
                hebbian_lr: 0.01, receives_broadcast: false,
                ..Default::default()
            },
            hippocampus_layer: LayerConfig {
                n_neurons: 8, memory_length: 16, nlm_depth: 2,
                hebbian_lr: 0.02, receives_broadcast: true,
                ..Default::default()
            },
            use_hopfield_readout: false, // default: old sync for backward compat
            hopfield_d_key: default_hopfield_d_key(),
        }
    }
}
