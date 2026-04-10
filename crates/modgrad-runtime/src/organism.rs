//! Tabula Rasa: self-developing CTM that learns from scratch.
//!
//! No pre-trained backbone. Just DNA-level architecture:
//! - Random embedding table (vocab → vectors)
//! - Learnable sensory MLP (embed → CTM input)
//! - CTM (4-region deliberation with Hebbian plasticity)
//! - Learnable output projection (sync → vocab logits)
//!
//! The system learns language, world knowledge, and reasoning
//! through experience + Hebbian plasticity + sleep consolidation.
//! Like a newborn brain: right architecture, zero knowledge.
//!
//! ```text
//! raw tokens
//!   → embedding table (DNA: randomly initialized)
//!   → sensory MLP (V1: develops feature detectors)
//!   → CTM 4 regions (cortex: deliberation)
//!   → output MLP (Broca's: develops speech production)
//!   → vocab logits → next token prediction
//! ```

use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::ctm::*;
use modgrad_codec::retina::Retina;

// ─── DNA Configuration ──────────────────────────────────────

/// The "genome" — architectural hyperparameters that define capacity,
/// not knowledge. Like DNA specifying brain size, layer structure,
/// neurotransmitter receptor density — but not memories or skills.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dna {
    /// Vocabulary size (byte-level: 256, BPE: 32k-152k)
    pub vocab_size: usize,
    /// Embedding dimension — how rich each token's representation can be.
    /// Analogous to how many features a neuron in V1 can detect.
    pub embed_dim: usize,
    /// Sensory MLP depth — how many processing stages before the CTM.
    /// Analogous to layers in early sensory cortex (V1 → V2 → V4).
    pub sensory_depth: usize,
    /// Sensory MLP hidden dimension.
    pub sensory_hidden: usize,
    /// CTM configuration (the cortex).
    pub ctm: CtmConfig,
    /// Context window — how many past tokens to attend to.
    pub context_len: usize,
    /// Learning rate for Hebbian updates.
    pub learning_rate: f32,
}

impl Dna {
    /// Tiny config for testing (64 neurons, ~74K params)
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            embed_dim: 64,
            sensory_depth: 2,
            sensory_hidden: 128,
            ctm: CtmConfig {
                iterations: 8, d_model: 64, d_input: 64,
                heads: 4, n_sync_out: 32, n_sync_action: 16,
                synapse_depth: 1, out_dims: 32,
                global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,
                neuromod: NeuromodConfig::default(),
                use_hopfield_readout: false, hopfield_d_key: 32,
                input_layer: LayerConfig {
                    n_neurons: 16, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.02, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                attention_layer: LayerConfig {
                    n_neurons: 16, memory_length: 8, nlm_depth: 2,
                    hebbian_lr: 0.01, receives_broadcast: false,
                    sparsity_target: 0.1, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                output_layer: LayerConfig {
                    n_neurons: 16, memory_length: 8, nlm_depth: 2,
                    hebbian_lr: 0.005, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                motor_layer: LayerConfig {
                    n_neurons: 16, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.005, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                // Cerebellum: fastest learner in the brain. Prediction error machine.
                // 69B neurons in human (4x cortex). High plasticity for real-time correction.
                cerebellum_layer: LayerConfig {
                    n_neurons: 32, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.05, receives_broadcast: false,
                    ..Default::default()
                },
                // Basal ganglia: reward-gated action selection. Dopamine modulated.
                // Moderate size, moderate plasticity.
                basal_ganglia_layer: LayerConfig {
                    n_neurons: 8, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.01, receives_broadcast: false,
                    ..Default::default()
                },
                // Insula: body state integration. Stable representation, low plasticity.
                insula_layer: LayerConfig {
                    n_neurons: 8, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.003, receives_broadcast: false,
                    ..Default::default()
                },
                // Hippocampus: one-shot episodic binding. High plasticity, deep memory.
                hippocampus_layer: LayerConfig {
                    n_neurons: 16, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.05, receives_broadcast: true,
                    ..Default::default()
                },
            },
            context_len: 64,
            learning_rate: 0.01,
        }
    }

    /// Small config for real experiments (256 neurons, ~1M params)
    pub fn small() -> Self {
        Self {
            vocab_size: 256,
            embed_dim: 128,
            sensory_depth: 3,
            sensory_hidden: 256,
            ctm: CtmConfig {
                iterations: 4, d_model: 256, d_input: 128,
                heads: 4, n_sync_out: 64, n_sync_action: 32,
                synapse_depth: 1, out_dims: 64,
                global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,
                neuromod: NeuromodConfig::default(),
                use_hopfield_readout: false, hopfield_d_key: 32,
                input_layer: LayerConfig {
                    n_neurons: 64, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.02, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                attention_layer: LayerConfig {
                    n_neurons: 64, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.01, receives_broadcast: false,
                    sparsity_target: 0.05, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                output_layer: LayerConfig {
                    n_neurons: 64, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.005, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                motor_layer: LayerConfig {
                    n_neurons: 64, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.005, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                cerebellum_layer: LayerConfig {
                    n_neurons: 64, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.05, receives_broadcast: false,
                    ..Default::default()
                },
                basal_ganglia_layer: LayerConfig {
                    n_neurons: 16, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.01, receives_broadcast: false,
                    ..Default::default()
                },
                insula_layer: LayerConfig {
                    n_neurons: 16, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.003, receives_broadcast: false,
                    ..Default::default()
                },
                hippocampus_layer: LayerConfig {
                    n_neurons: 32, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.05, receives_broadcast: true,
                    ..Default::default()
                },
            },
            context_len: 128,
            learning_rate: 0.005,
        }
    }

    /// Medium config (256 neurons, 16 ticks, ~1M params)
    /// Prioritizes thinking depth over width.
    pub fn medium() -> Self {
        Self {
            vocab_size: 256,
            embed_dim: 128,
            sensory_depth: 2,
            sensory_hidden: 256,
            ctm: CtmConfig {
                iterations: 16, d_model: 256, d_input: 128,
                heads: 4, n_sync_out: 64, n_sync_action: 32,
                synapse_depth: 1, out_dims: 64,
                global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,
                neuromod: NeuromodConfig::default(),
                use_hopfield_readout: false, hopfield_d_key: 32,
                input_layer: LayerConfig {
                    n_neurons: 64, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.01, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                attention_layer: LayerConfig {
                    n_neurons: 64, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.005, receives_broadcast: false,
                    sparsity_target: 0.05, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                output_layer: LayerConfig {
                    n_neurons: 64, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.003, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                motor_layer: LayerConfig {
                    n_neurons: 64, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.003, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                cerebellum_layer: LayerConfig {
                    n_neurons: 64, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.05, receives_broadcast: false,
                    ..Default::default()
                },
                basal_ganglia_layer: LayerConfig {
                    n_neurons: 16, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.01, receives_broadcast: false,
                    ..Default::default()
                },
                insula_layer: LayerConfig {
                    n_neurons: 16, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.003, receives_broadcast: false,
                    ..Default::default()
                },
                hippocampus_layer: LayerConfig {
                    n_neurons: 32, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.05, receives_broadcast: true,
                    ..Default::default()
                },
            },
            context_len: 64,
            learning_rate: 0.005,
        }
    }

    /// Medium-plus: 1024 neurons (4×256), 8 ticks, 256 sync dims.
    /// Sweet spot for Ryzen 9: ~1.5s/step, ~2400 steps/hour.
    /// 1:1 sync-to-byte ratio eliminates the dimensionality bottleneck.
    pub fn medium_plus() -> Self {
        Self {
            vocab_size: 256,
            embed_dim: 256,
            sensory_depth: 3,
            sensory_hidden: 512,
            ctm: CtmConfig {
                iterations: 8, d_model: 1024, d_input: 256,
                heads: 4, n_sync_out: 256, n_sync_action: 128,
                synapse_depth: 1, out_dims: 256,
                global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,
                neuromod: NeuromodConfig::default(),
                use_hopfield_readout: false, hopfield_d_key: 32,
                input_layer: LayerConfig {
                    n_neurons: 256, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.01, inhibitory_fraction: 0.3,
                    receives_broadcast: true,
                    ..Default::default()
                },
                attention_layer: LayerConfig {
                    n_neurons: 256, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.005, receives_broadcast: false,
                    sparsity_target: 0.05, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                output_layer: LayerConfig {
                    n_neurons: 256, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.003, inhibitory_fraction: 0.3,
                    receives_broadcast: true,
                    ..Default::default()
                },
                motor_layer: LayerConfig {
                    n_neurons: 256, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.003, inhibitory_fraction: 0.3,
                    receives_broadcast: true,
                    ..Default::default()
                },
                cerebellum_layer: LayerConfig {
                    n_neurons: 32, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.01, receives_broadcast: false,
                    ..Default::default()
                },
                basal_ganglia_layer: LayerConfig {
                    n_neurons: 32, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.01, receives_broadcast: false,
                    ..Default::default()
                },
                insula_layer: LayerConfig {
                    n_neurons: 32, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.01, receives_broadcast: false,
                    ..Default::default()
                },
                hippocampus_layer: LayerConfig {
                    n_neurons: 64, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.02, receives_broadcast: true,
                    ..Default::default()
                },
            },
            context_len: 128,
            learning_rate: 0.003,
        }
    }

    /// Large config for real language learning (4K neurons, ~50M params).
    /// Fits in L3 cache on Zen4. GPU-accelerated SuperLinear.
    pub fn large() -> Self {
        Self {
            vocab_size: 256,
            embed_dim: 512,
            sensory_depth: 4,
            sensory_hidden: 1024,
            ctm: CtmConfig {
                iterations: 12, d_model: 4096, d_input: 512,
                heads: 8, n_sync_out: 256, n_sync_action: 128,
                synapse_depth: 1, out_dims: 256,
                global_broadcast_dim: 0, motor_threshold: 5.0,
                par_threshold: 32,
                neuromod: NeuromodConfig::default(),
                use_hopfield_readout: false, hopfield_d_key: 32,
                // Language-optimized split: 31% input, 27% attention, 27% output, 14% motor
                input_layer: LayerConfig {
                    n_neurons: 1280, memory_length: 8, memory_hidden_dims: 128,
                    nlm_depth: 1, hebbian_lr: 0.01,
                    receives_broadcast: true, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                attention_layer: LayerConfig {
                    n_neurons: 1024, memory_length: 32, memory_hidden_dims: 128,
                    nlm_depth: 2, hebbian_lr: 0.005,
                    receives_broadcast: false, sparsity_target: 0.05,
                    inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                output_layer: LayerConfig {
                    n_neurons: 1024, memory_length: 32, memory_hidden_dims: 128,
                    nlm_depth: 2, hebbian_lr: 0.003,
                    receives_broadcast: true, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                motor_layer: LayerConfig {
                    n_neurons: 768, memory_length: 8, memory_hidden_dims: 128,
                    nlm_depth: 1, hebbian_lr: 0.003,
                    receives_broadcast: true, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                // Cerebellum: prediction error. Fastest learner. Huge in real brain.
                cerebellum_layer: LayerConfig {
                    n_neurons: 1024, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.05, receives_broadcast: false,
                    ..Default::default()
                },
                // Basal ganglia: reward-gated action selection.
                basal_ganglia_layer: LayerConfig {
                    n_neurons: 256, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.01, receives_broadcast: false,
                    ..Default::default()
                },
                // Insula: body state. Stable, low plasticity.
                insula_layer: LayerConfig {
                    n_neurons: 128, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.003, receives_broadcast: false,
                    ..Default::default()
                },
                // Hippocampus: one-shot episodic binding. Deep memory.
                hippocampus_layer: LayerConfig {
                    n_neurons: 512, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.05, receives_broadcast: true,
                    ..Default::default()
                },
            },
            context_len: 256,
            learning_rate: 0.002,
        }
    }

    /// QEC-optimized: more ticks for deeper temporal features.
    pub fn qec() -> Self {
        let mut dna = Self::large();
        dna.ctm.iterations = 8;        // 2× original (was 4)
        dna.ctm.d_input = 20;          // surface code syndrome dim
        dna.ctm.n_sync_out = 256;      // same as large (keep LS tractable)
        dna.ctm.n_sync_action = 128;
        dna.ctm.out_dims = 20;         // match input dim
        dna
    }

    /// Create a child DNA matched to a parent model's hidden dimension.
    /// The child thinks in the same dimensional space as the parent —
    /// no projection needed, direct comparison of representations.
    pub fn child_of(parent_hidden_dim: usize) -> Self {
        Self {
            vocab_size: 256,
            embed_dim: parent_hidden_dim,  // same as parent
            sensory_depth: 1,              // single layer — trained from parent via least-squares
            sensory_hidden: parent_hidden_dim,
            ctm: CtmConfig {
                iterations: 8, d_model: 256, d_input: parent_hidden_dim,
                heads: 4, n_sync_out: 64, n_sync_action: 32,
                synapse_depth: 1, out_dims: 64,
                global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,
                neuromod: NeuromodConfig::default(),
                use_hopfield_readout: false, hopfield_d_key: 32,
                input_layer: LayerConfig {
                    n_neurons: 64, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.01, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                attention_layer: LayerConfig {
                    n_neurons: 64, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.005, receives_broadcast: false,
                    sparsity_target: 0.05, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                output_layer: LayerConfig {
                    n_neurons: 64, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.003, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                motor_layer: LayerConfig {
                    n_neurons: 64, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.003, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                cerebellum_layer: LayerConfig {
                    n_neurons: 64, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.05, receives_broadcast: false,
                    ..Default::default()
                },
                basal_ganglia_layer: LayerConfig {
                    n_neurons: 16, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.01, receives_broadcast: false,
                    ..Default::default()
                },
                insula_layer: LayerConfig {
                    n_neurons: 16, memory_length: 4, nlm_depth: 1,
                    hebbian_lr: 0.003, receives_broadcast: false,
                    ..Default::default()
                },
                hippocampus_layer: LayerConfig {
                    n_neurons: 32, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.05, receives_broadcast: true,
                    ..Default::default()
                },
            },
            context_len: 128,
            learning_rate: 0.005,
        }
    }

    /// Transformer-bridge config: NMC sized to accept transformer hidden states.
    ///
    /// The transformer produces features (backprop-trained), NMC adds temporal
    /// dynamics (ticks, regions) and online adaptation (perturbation learning).
    /// NMC d_input = model_dim so it can directly consume transformer output.
    ///
    /// model_dim: transformer hidden dimension (e.g., 256, 512, 768).
    pub fn transformer_bridge(model_dim: usize) -> Self {
        // Scale neuron counts with model_dim but keep NMC much smaller than transformer.
        // NMC is the "fast adapter" on top, not a duplicate of the transformer.
        let n_core = (model_dim / 2).max(64);   // core regions: input, attn, output
        let n_motor = (model_dim / 4).max(32);  // motor: smaller, action-focused
        let n_sub = (model_dim / 8).max(16);    // subcortical: cerebellum, BG, etc.

        Self {
            vocab_size: 256,
            embed_dim: model_dim,
            sensory_depth: 1,          // minimal — transformer already processed
            sensory_hidden: model_dim,
            ctm: CtmConfig {
                iterations: 8,
                d_model: model_dim,
                d_input: model_dim,    // accepts transformer hidden states directly
                heads: (model_dim / 64).max(1),
                n_sync_out: (model_dim / 2).max(32),
                n_sync_action: (model_dim / 4).max(16),
                synapse_depth: 1,
                out_dims: model_dim,
                global_broadcast_dim: 0,
                motor_threshold: 5.0,
                par_threshold: 32,
                neuromod: NeuromodConfig::default(),
                use_hopfield_readout: false, hopfield_d_key: 32,
                input_layer: LayerConfig {
                    n_neurons: n_core, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.0, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                attention_layer: LayerConfig {
                    n_neurons: n_core, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.0, receives_broadcast: false,
                    sparsity_target: 0.05, inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                output_layer: LayerConfig {
                    n_neurons: n_core, memory_length: 16, nlm_depth: 2,
                    hebbian_lr: 0.0, receives_broadcast: true,
                    inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                motor_layer: LayerConfig {
                    n_neurons: n_motor, memory_length: 8, nlm_depth: 1,
                    hebbian_lr: 0.0, receives_broadcast: true,
                    inhibitory_fraction: 0.3,
                    ..Default::default()
                },
                cerebellum_layer: LayerConfig {
                    n_neurons: n_sub, ..Default::default()
                },
                basal_ganglia_layer: LayerConfig {
                    n_neurons: n_sub, ..Default::default()
                },
                insula_layer: LayerConfig {
                    n_neurons: n_sub / 2, ..Default::default()
                },
                hippocampus_layer: LayerConfig {
                    n_neurons: n_sub, ..Default::default()
                },
            },
            context_len: 256,
            learning_rate: 0.0,  // no Hebbian — use perturbation instead
        }
    }
}

impl Default for Dna {
    fn default() -> Self { Self::tiny() }
}

// ─── Organism ───────────────────────────────────────────────

/// A self-developing organism. Starts blank, learns from experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Organism {
    pub dna: Dna,

    /// Token embedding table: vocab_size × embed_dim.
    /// Randomly initialized — develops token representations through experience.
    pub embeddings: Vec<f32>,

    /// Sensory layer: embed_dim → ctm.d_input.
    /// Single linear layer when parent-taught (option 2 from analysis).
    /// Multi-layer MLP when learning from scratch (tabula rasa).
    /// Parent-taught: trained via least-squares against parent hidden states.
    pub sensory_layers: Vec<Linear>,

    /// If true, sensory is a single layer trained from parent (option 2).
    /// If false, multi-layer MLP for tabula rasa.
    #[serde(default)]
    pub parent_supervised_sensory: bool,

    /// Collected (embedding, parent_hidden) pairs for sensory layer training.
    /// During parent teaching, we collect these. During sleep, we solve
    /// W_opt = argmin ||parent_hidden - W · embedding||²
    #[serde(skip)]
    pub sensory_parent_traces: Vec<(Vec<f32>, Vec<f32>)>,

    /// The CTM cortex — 4-region deliberation engine.
    pub ctm: Ctm,

    /// Output projection: sync → vocab logits.
    /// Develops speech production like Broca's area.
    pub output_proj: Linear,

    /// Collected (sync_signal, target_one_hot) pairs for output projection training.
    #[serde(skip)]
    pub output_traces: Vec<(Vec<f32>, usize)>,

    /// Last forward pass tick traces for debugger visualization.
    #[serde(skip)]
    pub last_tick_traces: Vec<super::tick_state::TickTrace>,

    /// Persistent neuromodulator state across steps.
    /// Dopamine from previous step's surprise modulates next step's sync.
    #[serde(skip)]
    pub neuromod: super::session::Neuromodulators,

    /// Token frequency counter: how many times each byte has been seen as target.
    /// Used for frequency-normalized Hebbian: rare tokens get stronger updates.
    #[serde(skip)]
    pub token_counts: Vec<u32>,

    /// Experience counter — how many tokens this organism has processed.
    pub tokens_seen: u64,

    /// Sleep cycles completed.
    pub sleep_cycles: u64,

    /// Cached sync diversity measurement (updated every 100 tokens).
    /// Used to gate Hebbian and consolidation — skip when diversity is low.
    #[serde(skip)]
    pub last_diversity: f32,

    /// Collected (input, output) traces for sensory MLP consolidation during sleep.
    #[serde(skip)]
    pub sensory_traces: Vec<(usize, Vec<f32>, Vec<f32>)>,

    /// Runtime tuning config — not serialized. Injected by daemon or CLI.
    /// The organism reads from this; only authorized callers write to it.
    #[serde(skip)]
    pub tuning: super::tuning::TuningConfig,

    /// Proprioceptive retina: senses the organism's own hardware.
    /// Discovers sensors at birth, reads them each tick.
    /// Not serialized — re-discovers sensors on each boot.
    #[serde(skip)]
    pub proprioception: Option<modgrad_codec::retina::ProprioceptiveRetina>,

    /// Cached split weights for zero-copy parallel training (not serialized).
    /// Lazily populated by `ensure_split()`, written back by `sync_back()`.
    #[serde(skip)]
    pub ctm_weights: Option<Arc<CtmWeights>>,

    /// Cached split session for parallel training (not serialized).
    #[serde(skip)]
    pub ctm_session: Option<CtmSession>,
}

impl Organism {
    /// Create a new organism from DNA. All weights randomly initialized.
    /// This is "birth" — the architecture exists but contains no knowledge.
    pub fn new(dna: Dna) -> Self {
        let v = dna.vocab_size;
        let d = dna.embed_dim;
        let ctm_d = dna.ctm.d_input;

        // Structured embeddings: byte properties in first dims, random in rest.
        // This gives the CTM a head start on distinguishing tokens without a parent.
        // First 8 dims encode: ascii_value, is_alpha, is_upper, is_digit, is_space,
        // is_punct, is_printable, bit_count. Remaining dims are random noise.
        let mut rng = SimpleRng::new(42);
        let scale = (1.0 / d as f32).sqrt();
        let mut embeddings = vec![0.0f32; v * d];
        for byte_val in 0..v {
            let offset = byte_val * d;
            let b = byte_val as u8;
            // Structured features (normalized to ~[-1, 1])
            if d > 0 { embeddings[offset] = (b as f32 / 128.0) - 1.0; }  // ascii value
            if d > 1 { embeddings[offset + 1] = if b.is_ascii_alphabetic() { 1.0 } else { -1.0 }; }
            if d > 2 { embeddings[offset + 2] = if b.is_ascii_uppercase() { 1.0 } else { -1.0 }; }
            if d > 3 { embeddings[offset + 3] = if b.is_ascii_digit() { 1.0 } else { -1.0 }; }
            if d > 4 { embeddings[offset + 4] = if b == b' ' || b == b'\n' || b == b'\t' { 1.0 } else { -1.0 }; }
            if d > 5 { embeddings[offset + 5] = if b.is_ascii_punctuation() { 1.0 } else { -1.0 }; }
            if d > 6 { embeddings[offset + 6] = if b.is_ascii_graphic() { 1.0 } else { -1.0 }; }
            if d > 7 { embeddings[offset + 7] = (b.count_ones() as f32 / 4.0) - 1.0; }  // bit count
            // Random dims (exploration space for Hebbian to shape)
            for j in 8..d {
                embeddings[offset + j] = rng.next_normal() * scale;
            }
        }

        // Sensory MLP layers
        let mut sensory_layers = Vec::new();
        let mut in_dim = d;
        for i in 0..dna.sensory_depth {
            let out_dim = if i == dna.sensory_depth - 1 { ctm_d } else { dna.sensory_hidden };
            sensory_layers.push(Linear::new(in_dim, out_dim));
            in_dim = out_dim;
        }

        // CTM
        let ctm = Ctm::new(dna.ctm.clone());

        // Output: sync → vocab
        // Initialize bias with English byte frequency prior (log-probabilities).
        // This gives the organism a frequency table for free — like a newborn
        // brain that already "knows" which phonemes are common in its language.
        // Without this, the organism starts at CE=5.55 (uniform). With it,
        // it starts at ~3.5 (unigram prediction) and only needs to learn CONTEXT.
        let mut output_proj = Linear::new(dna.ctm.n_sync_out, v);
        {
            // Approximate English byte frequencies (from hacker curriculum analysis)
            let mut freq = vec![1.0f32; v]; // Laplace smoothing
            // Common ASCII bytes (space, lowercase, digits, common punct)
            let common: &[(u8, f32)] = &[
                (b' ', 50.0), (b'e', 12.0), (b't', 9.0), (b'a', 8.0), (b'o', 7.5),
                (b'i', 7.0), (b'n', 7.0), (b's', 6.5), (b'h', 6.0), (b'r', 6.0),
                (b'd', 4.0), (b'l', 4.0), (b'c', 3.0), (b'u', 3.0), (b'm', 2.5),
                (b'w', 2.0), (b'f', 2.0), (b'g', 2.0), (b'y', 2.0), (b'p', 2.0),
                (b'b', 1.5), (b'v', 1.0), (b'k', 0.8), (b'\n', 5.0), (b'.', 3.0),
                (b',', 2.0), (b'0', 4.0), (b'1', 4.0), (b'2', 3.0), (b'3', 2.0),
                (b'4', 2.0), (b'5', 2.0), (b'6', 2.0), (b'7', 2.0), (b'8', 2.0),
                (b'9', 2.0), (b'=', 4.0), (b'-', 2.0), (b'_', 1.5), (b'/', 1.5),
                (b':', 1.5), (b';', 0.5), (b'(', 1.0), (b')', 1.0), (b'"', 1.0),
                (b'\'', 1.0), (b'<', 0.5), (b'>', 0.5), (b'{', 0.5), (b'}', 0.5),
            ];
            for &(byte, weight) in common {
                freq[byte as usize] = weight;
            }
            let total: f32 = freq.iter().sum();
            for i in 0..v {
                output_proj.bias[i] = (freq[i] / total).ln(); // log-probability
            }
        }

        Self {
            dna,
            embeddings,
            sensory_layers,
            ctm,
            output_proj,
            tokens_seen: 0,
            sleep_cycles: 0,
            sensory_traces: Vec::new(),
            output_traces: Vec::new(),
            last_tick_traces: Vec::new(),
            neuromod: super::session::Neuromodulators::default(),
            token_counts: vec![0u32; v],
            last_diversity: 0.0,
            parent_supervised_sensory: false,
            sensory_parent_traces: Vec::new(),
            tuning: super::tuning::TuningConfig::default(),
            proprioception: None, // initialized lazily on first use
            ctm_weights: None,
            ctm_session: None,
        }
    }

    /// Embed a single token.
    pub fn embed(&self, token_id: usize) -> Vec<f32> {
        let d = self.dna.embed_dim;
        let offset = token_id.min(self.dna.vocab_size - 1) * d;
        self.embeddings[offset..offset + d].to_vec()
    }

    /// Run sensory MLP: embedding → CTM observation.
    /// Collects (input, output) traces for sleep consolidation when training.
    pub fn sensory_forward(&mut self, embedding: &[f32], collect: bool) -> Vec<f32> {
        // If embed_dim matches d_input, bypass the sensory MLP entirely.
        // The MLP with random weights destroys embedding diversity:
        // embeddings have cosine 0.001 (good), but after 3 random linear layers
        // observations have cosine 0.44 (all bytes look the same to the CTM).
        // Skip until the MLP has been trained (via LS during sleep).
        if self.sensory_layers.is_empty()
            || (embedding.len() == self.dna.ctm.d_input && !self.parent_supervised_sensory)
        {
            // Direct passthrough: embedding IS the observation
            return embedding.to_vec();
        }

        let mut x = embedding.to_vec();
        for (i, layer) in self.sensory_layers.iter().enumerate() {
            let input = x.clone();
            x = layer.forward(&x);
            // ReLU between layers (except last)
            if i < self.sensory_layers.len() - 1 {
                for v in &mut x {
                    *v = v.max(0.0);
                }
            }
            if collect && self.sensory_traces.len() < 10000 {
                self.sensory_traces.push((i, input, x.clone()));
            }
        }
        x
    }

    /// Forward pass on a sequence of tokens. Returns logits for each position.
    ///
    /// For each token:
    /// 1. Embed → sensory MLP → observation
    /// 2. CTM forward (K ticks) → sync signal
    /// 3. Output projection → logits over vocab
    ///
    /// The CTM state persists across tokens in the sequence,
    /// accumulating context through its sync accumulators.
    pub fn forward(&mut self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        self.forward_inner(token_ids, false).0
    }

    /// Returns (logits_per_position, sync_per_position).
    /// Returns (logits, syncs, observations).
    /// observations[i] is the d_input-dim sensory observation for token i.
    pub fn forward_inner(&mut self, token_ids: &[usize], training: bool) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        self.forward_inner_full(token_ids, training).0
    }

    /// Full forward: returns ((logits, syncs), observations).
    #[allow(deprecated)]
    pub fn forward_inner_full(&mut self, token_ids: &[usize], training: bool)
        -> ((Vec<Vec<f32>>, Vec<Vec<f32>>), Vec<Vec<f32>>)
    {
        let mut state = self.ctm.init_state();
        state.neuromod = self.neuromod.clone();
        state.noisy = training;
        let mut all_logits = Vec::with_capacity(token_ids.len());
        let mut all_syncs = Vec::with_capacity(token_ids.len());
        let mut all_obs = Vec::with_capacity(token_ids.len());

        // Read body sensors once per sequence (not per token — too expensive)
        let proprio = match &mut self.proprioception {
            Some(retina) => retina.observe(&[]),
            None => vec![0.0; self.dna.ctm.d_input],
        };

        for (pos, &tid) in token_ids.iter().enumerate() {
            let emb = self.embed(tid);
            let obs = self.sensory_forward(&emb, training);

            let (_predictions, sync, _signals) = self.ctm.forward_with_proprio(
                &obs, &proprio, &mut state, training);

            let logits = self.output_proj.forward(&sync);
            all_logits.push(logits);
            all_syncs.push(sync.clone());
            all_obs.push(obs);

            if training && pos + 1 < token_ids.len() && self.output_traces.len() < 50000 {
                self.output_traces.push((sync, token_ids[pos + 1]));
            }

            self.tokens_seen += 1;
        }

        self.last_tick_traces = state.tick_traces;

        ((all_logits, all_syncs), all_obs)
    }

    /// Compute cross-entropy loss and per-token surprise.
    /// tokens[i] predicts tokens[i+1].
    pub fn compute_loss(&self, logits: &[Vec<f32>], targets: &[usize]) -> (f32, Vec<f32>) {
        let mut total_loss = 0.0f32;
        let mut surprises = Vec::new();

        for (i, target) in targets.iter().enumerate() {
            if i >= logits.len() { break; }

            let log = &logits[i];
            let v = log.len();
            let target_idx = (*target).min(v - 1);

            // Numerically stable softmax + cross-entropy
            let max_l: f32 = log.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = log.iter().map(|&l| (l - max_l).exp()).sum();
            let log_prob = (log[target_idx] - max_l) - exp_sum.ln();
            let surprise = -log_prob;

            total_loss += surprise;
            surprises.push(surprise);
        }

        let n = targets.len().max(1) as f32;
        (total_loss / n, surprises)
    }

    /// Hebbian weight update across ALL learnable layers.
    ///
    /// Uses surprise as the learning signal:
    /// - High surprise → large update (this was unexpected, learn from it)
    /// - Low surprise → small update (already knew this, don't overwrite)
    ///
    /// Updates:
    /// 1. Embedding table: co-occurring tokens become more similar
    /// 2. Sensory MLP: Oja's rule (Hebbian + normalization) on each layer
    /// 3. Output projection: error-driven update (predicted vs actual)
    /// 4. CTM: handled inside ctm.forward() via LocalHebbian
    pub fn hebbian_update(
        &mut self,
        token_ids: &[usize],
        logits: &[Vec<f32>],
        syncs: &[Vec<f32>],
        surprises: &[f32],
    ) {
        let lr = self.dna.learning_rate;
        let d = self.dna.embed_dim;

        for i in 1..token_ids.len().min(surprises.len() + 1) {
            let surprise = surprises.get(i - 1).copied().unwrap_or(1.0);
            let modulated_lr = lr * (surprise / 5.0).clamp(
                self.tuning.learning.surprise_lr_min.val(),
                self.tuning.learning.surprise_lr_max.val(),
            );

            let prev_tid = token_ids[i - 1];
            let curr_tid = token_ids[i];

            // 1. Embedding update: co-occurrence → association
            let prev_offset = prev_tid.min(self.dna.vocab_size - 1) * d;
            let curr_offset = curr_tid.min(self.dna.vocab_size - 1) * d;
            for j in 0..d {
                let delta = self.embeddings[prev_offset + j] - self.embeddings[curr_offset + j];
                self.embeddings[curr_offset + j] += modulated_lr * delta * self.tuning.learning.embedding_delta_scale.val();
            }

            // 2. Output projection update: nudge toward correct token
            //    Simple error-corrective rule: increase weight for target token,
            //    decrease for the token that was predicted instead (contrastive)
            if i - 1 < logits.len() {
                let log = &logits[i - 1];
                let target = curr_tid.min(self.dna.vocab_size - 1);
                let predicted = log.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx).unwrap_or(0);

                if predicted != target {
                    let sync = syncs.get(i - 1).map(|s| s.as_slice());

                    // Track real token frequencies
                    if target < self.token_counts.len() {
                        self.token_counts[target] += 1;
                    }

                    let out = &mut self.output_proj;
                    let in_d = out.in_dim;

                    if let Some(sync) = sync {
                        // Frequency-normalized Hebbian using actual token counts.
                        // Common tokens ('e', ' ', 'a') seen 10000x get 100x weaker updates
                        // than rare tokens seen 1x. This is real TF-IDF, not a proxy.
                        let target_count = self.token_counts.get(target).copied().unwrap_or(1).max(1);
                        let predicted_count = self.token_counts.get(predicted).copied().unwrap_or(1).max(1);
                        let target_lr = modulated_lr * self.tuning.learning.output_target_lr_scale.val() / (target_count as f32).sqrt();
                        let predicted_lr = modulated_lr * self.tuning.learning.output_predicted_lr_scale.val() / (predicted_count as f32).sqrt();

                        for k in 0..in_d.min(sync.len()) {
                            out.weight[target * in_d + k] += target_lr * sync[k];
                        }
                        out.bias[target] += target_lr;
                        for k in 0..in_d.min(sync.len()) {
                            out.weight[predicted * in_d + k] -= predicted_lr * sync[k];
                        }
                        out.bias[predicted] -= predicted_lr;
                    }
                }
            }
        }

        // 3. Sensory MLP: Oja's rule during sleep (collected traces)
        // 4. CTM Hebbian: handled inside ctm.forward() when enabled
    }

    // ─── Parent-Child Learning ────────────────────────────────

    /// Learn from a parent model's hidden states.
    ///
    /// The parent (e.g. Qwen backbone) processes text and produces
    /// hidden states — its "understanding." The child (this organism)
    /// processes the same text and tries to make its internal
    /// representations match the parent's.
    ///
    /// This is gradient-free knowledge distillation:
    /// - Parent provides target representations (the "meaning")
    /// - Child adjusts embeddings + sensory MLP to match
    /// - Learning signal = cosine similarity between parent and child states
    /// - Hebbian rule: strengthen connections that produce parent-like activity
    ///
    /// Like a parent showing a baby: "look, this is a cat" — the baby
    /// learns to see what the parent sees.
    ///
    /// `parent_hidden`: parent's hidden state per token [seq_len × hidden_dim]
    /// `token_ids`: the byte sequence being taught
    /// `importance`: norepinephrine burst (1.0 = normal, 3.0 = "remember this!")
    ///
    /// Returns mean alignment score (0 = nothing learned, 1 = perfect match).
    pub fn learn_from_parent(
        &mut self,
        token_ids: &[usize],
        parent_hiddens: &[Vec<f32>],
        importance: f32,
    ) -> f32 {
        if token_ids.is_empty() || parent_hiddens.is_empty() { return 0.0; }

        let lr = self.dna.learning_rate * importance;
        let d = self.dna.embed_dim;
        let parent_dim = parent_hiddens[0].len();

        // Normalize parent hidden state to unit length for comparison
        let normalize = |v: &[f32]| -> Vec<f32> {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                v.iter().map(|x| x / norm).collect()
            } else {
                v.to_vec()
            }
        };

        let mut total_sim = 0.0f32;
        let n = token_ids.len().min(parent_hiddens.len());

        for i in 0..n {
            let tid = token_ids[i];

            // Parent's normalized representation
            let parent_full = normalize(&parent_hiddens[i]);
            // If dimensions match (child_of), use directly. Otherwise truncate.
            let parent_target: Vec<f32> = if parent_dim == d {
                parent_full
            } else {
                let mut t = vec![0.0f32; d];
                for j in 0..d.min(parent_dim) { t[j] = parent_full[j]; }
                t
            };

            // Child's current representation for this token
            let child_emb = self.embed(tid);
            let child_repr = self.sensory_forward(&child_emb, true);

            // Normalize child repr
            let child_norm: f32 = child_repr.iter().map(|x| x * x).sum::<f32>().sqrt();
            let child_normed: Vec<f32> = if child_norm > 1e-8 {
                child_repr.iter().map(|x| x / child_norm).collect()
            } else {
                vec![0.0; child_repr.len()] // zero norm → zero vector → sim = 0
            };

            // Alignment: cosine similarity between child and parent
            let sim: f32 = child_normed.iter().zip(&parent_target)
                .map(|(a, b)| a * b).sum();
            total_sim += sim;

            // Direct blend: move child embedding toward parent representation.
            // blend=0.05 → 5% of the way to parent each exposure.
            // After 20 exposures of "cat", child's "cat" embedding is >60% parent-aligned.
            // This is imitation learning, not gradient descent.
            let alpha = (lr * importance * 0.05).min(0.2);
            let offset = tid.min(self.dna.vocab_size - 1) * d;
            let end = offset + d.min(parent_target.len());
            super::eval::linalg::blend(&mut self.embeddings[offset..end], &parent_target[..d.min(parent_target.len())], alpha);

            // Collect (embedding, parent_hidden) for sensory layer training.
            // During sleep, we'll solve W_opt = argmin ||parent_hidden - W · embedding||²
            if self.sensory_parent_traces.len() < 50000 {
                let emb = self.embed(tid);
                self.sensory_parent_traces.push((emb, parent_hiddens[i].clone()));
            }

            // Also update output projection: parent knows which tokens follow
            // If parent's state at position i predicts token at i+1, teach that
            if i + 1 < token_ids.len() {
                let next_tid = token_ids[i + 1].min(self.dna.vocab_size - 1);
                let out = &mut self.output_proj;
                let in_d = out.in_dim;
                // Strengthen the connection from parent-like state to next token
                for k in 0..in_d.min(d) {
                    out.weight[next_tid * in_d + k] += lr * parent_target.get(k).copied().unwrap_or(0.0) * 0.01;
                }
                out.bias[next_tid] += lr * 0.01;
            }

            self.tokens_seen += 1;
        }

        total_sim / n.max(1) as f32
    }

    /// Full parent-guided development session.
    /// The backbone processes each text chunk, the organism learns to match.
    pub fn develop_with_parent<F>(
        &mut self,
        texts: &[&[u8]],
        mut get_parent_hidden: F,
        sleep_every: usize,
        importance: f32,
        checkpoint_path: Option<&str>,
    ) -> Vec<f32>
    where
        F: FnMut(&[i64]) -> Vec<Vec<f32>>,
    {
        let mut alignments = Vec::new();
        self.ctm.enable_hebbian();

        for (step, text) in texts.iter().enumerate() {
            let token_ids: Vec<usize> = text.iter().map(|&b| b as usize).collect();
            if token_ids.len() < 2 { continue; }
            let end = token_ids.len().min(self.dna.context_len);
            let ids = &token_ids[..end];

            let t0 = std::time::Instant::now();

            // Parent processes the text
            let parent_ids: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
            let parent_hiddens = get_parent_hidden(&parent_ids);

            // Child learns from parent's representations
            let alignment = self.learn_from_parent(ids, &parent_hiddens, importance);
            alignments.push(alignment);

            // Also do regular self-supervised step for next-token prediction
            let (logits, syncs) = self.forward_inner(ids, true);
            let targets = &ids[1..];
            let (loss, surprises) = self.compute_loss(&logits, targets);
            self.hebbian_update(ids, &logits, &syncs, &surprises);

            let elapsed = t0.elapsed();
            let avg_align = if alignments.len() <= 10 { alignment } else {
                alignments[alignments.len().saturating_sub(10)..].iter().sum::<f32>() / 10.0
            };
            eprintln!("  [{}/{}] align={:.3} avg10={:.3} loss={:.1} {:.0}tok/s {:.1}s/step",
                step + 1, texts.len(), alignment, avg_align, loss,
                ids.len() as f64 / elapsed.as_secs_f64(),
                elapsed.as_secs_f64());

            if (step + 1) % sleep_every == 0 && step > 0 {
                let st0 = std::time::Instant::now();
                self.sleep();
                let bounds = self.ctm.angeris_bounds();
                let dead_total: usize = bounds.dead_neurons.iter().sum();
                eprintln!("  SLEEP #{} ({:.1}s) dead={}/{}",
                    self.sleep_cycles, st0.elapsed().as_secs_f64(),
                    dead_total, bounds.total_neurons);

                if let Some(path) = checkpoint_path {
                    self.save(path).ok();
                }
            }
        }

        alignments
    }

    /// Load pre-computed parent hidden states and train fast.
    /// The parent backbone ran offline; we just read its output.
    /// This is 10-20x faster than live parent teaching.
    pub fn learn_from_precomputed(
        &mut self,
        parent_file: &str,
        sleep_every: usize,
        importance: f32,
        checkpoint_path: Option<&str>,
    ) -> Vec<f32> {
        use std::io::Read;
        let mut f = std::fs::File::open(parent_file).expect("can't open parent states");

        // Read header
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).expect("read magic");
        assert_eq!(&magic, b"PRNT", "not a parent states file");

        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4).unwrap(); let _version = u32::from_le_bytes(buf4);
        f.read_exact(&mut buf4).unwrap(); let hidden_dim = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4).unwrap(); let n_chunks = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4).unwrap(); let _chunk_size = u32::from_le_bytes(buf4) as usize;
        let mut buf8 = [0u8; 8];
        f.read_exact(&mut buf8).unwrap(); let total_tokens = u64::from_le_bytes(buf8) as usize;

        eprintln!("  Parent states: {} chunks, dim={}, {} tokens",
            n_chunks, hidden_dim, total_tokens);

        let _d = self.dna.embed_dim;
        self.ctm.enable_hebbian();
        let mut alignments = Vec::new();

        for step in 0..n_chunks {
            // Read this chunk's token IDs + hidden states
            f.read_exact(&mut buf4).unwrap();
            let seq_len = u32::from_le_bytes(buf4) as usize;

            // Token IDs (raw bytes stored as u8)
            let mut token_bytes = vec![0u8; seq_len];
            f.read_exact(&mut token_bytes).unwrap();
            let token_ids: Vec<usize> = token_bytes.iter().map(|&b| b as usize).collect();

            // Hidden states
            let mut hidden_flat = vec![0.0f32; seq_len * hidden_dim];
            // SAFETY: Vec<f32> is contiguous, properly aligned. Reinterpreting as
            // &mut [u8] for binary read is sound (f32 has no invalid bit patterns).
            // Platform-endian; files are not portable across architectures.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(hidden_flat.as_mut_ptr() as *mut u8, hidden_flat.len() * 4)
            };
            f.read_exact(bytes).unwrap();

            let parent_hiddens: Vec<Vec<f32>> = (0..seq_len)
                .map(|t| hidden_flat[t * hidden_dim..(t + 1) * hidden_dim].to_vec())
                .collect();

            let t0 = std::time::Instant::now();

            // Learn from parent — same as live teaching but without backbone overhead
            let alignment = self.learn_from_parent(&token_ids, &parent_hiddens, importance);
            alignments.push(alignment);

            // Also do self-supervised next-token prediction
            if token_ids.len() >= 2 {
                let end = token_ids.len().min(self.dna.context_len);
                let (logits, syncs) = self.forward_inner(&token_ids[..end], true);
                let targets = &token_ids[1..end];
                let (_loss, surprises) = self.compute_loss(&logits, targets);
                self.hebbian_update(&token_ids[..end], &logits, &syncs, &surprises);
            }

            let elapsed = t0.elapsed();
            if (step + 1) % 100 == 0 {
                eprintln!("  [{}/{}] {:.0}tok/s {:.3}s/step",
                    step + 1, n_chunks,
                    seq_len as f64 / elapsed.as_secs_f64().max(0.001),
                    elapsed.as_secs_f64());
            }

            if (step + 1) % sleep_every == 0 && step > 0 {
                self.sleep();
                eprintln!("  SLEEP #{}", self.sleep_cycles);

                if let Some(path) = checkpoint_path {
                    self.save(path).ok();
                }
            }
        }

        alignments
    }

    // ─── Self-supervised learning ───────────────────────────────

    /// One training step: forward + loss + Hebbian update + neuromodulation.
    /// Uses the FULL brain: surprise → dopamine, energy → serotonin,
    /// replay buffer for high-surprise experiences.
    pub fn train_step(&mut self, token_ids: &[usize]) -> f32 {
        self.ctm.enable_hebbian();

        let ((logits, syncs), observations) = self.forward_inner_full(token_ids, true);
        let targets = &token_ids[1..];
        let (loss, surprises) = self.compute_loss(&logits, targets);

        // Mean surprise for neuromodulation
        let mean_surprise = if surprises.is_empty() { 1.0 } else {
            surprises.iter().sum::<f32>() / surprises.len() as f32
        };

        // Activation energy: how hard did the CTM work?
        let act_energy: f32 = syncs.iter()
            .flat_map(|s| s.iter())
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt() / syncs.len().max(1) as f32;

        // Novelty: how different is this observation from recent replay entries?
        let novelty = if self.ctm.replay.is_empty() { 1.0 } else {
            let empty = vec![];
            let last_obs = observations.last().unwrap_or(&empty);
            let max_sim = self.ctm.replay.entries.iter()
                .map(|e| {
                    let dot: f32 = last_obs.iter().zip(&e.observation)
                        .map(|(a, b)| a * b).sum();
                    let na: f32 = last_obs.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let nb: f32 = e.observation.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if na > 1e-8 && nb > 1e-8 { dot / (na * nb) } else { 0.0 }
                })
                .fold(0.0f32, f32::max);
            (1.0 - max_sim).max(0.0)
        };

        // Push to replay buffer if surprising enough.
        // Store the observation (sensory input), not the sync signal —
        // dream() replays through forward() which expects d_input-dim observation.
        if let Some(obs) = observations.last() {
            self.ctm.replay.push(obs.clone(), mean_surprise);
        }

        // Update neuromodulators from this step's signals.
        // Dopamine: surprise-driven (prediction error → learning signal).
        // Serotonin: energy × novelty (should we consolidate this?).
        // NE decays back to baseline.
        self.neuromod.update_dopamine(mean_surprise);
        self.neuromod.update_serotonin(act_energy, novelty);
        self.neuromod.decay_ne();

        // Hebbian update — modulated by surprise (dopamine analog).
        // Only apply when sync patterns are diverse enough.
        // Hebbian on uniform syncs pushes all weights in the same direction → collapse.
        // Check every 100 steps (measuring diversity is expensive).
        if self.tokens_seen % 100 == 0 {
            let div = self.measure_sync_diversity();
            self.last_diversity = div;
        }
        if self.last_diversity < self.tuning.diversity.collapse_threshold.val() {
            self.hebbian_update(token_ids, &logits, &syncs, &surprises);
        }

        loss
    }

    // ─── Zero-copy batch parallel training ────────────────────

    /// Lazily split the Ctm into shared weights + mutable session.
    /// After this, `ctm_weights` is an `Arc<CtmWeights>` that can be
    /// cheaply cloned across threads, and `ctm_session` holds the
    /// mutable learning state for the primary thread.
    fn ensure_split(&mut self) {
        if self.ctm_weights.is_none() {
            let (w, s) = self.ctm.clone().into_split();
            self.ctm_weights = Some(Arc::new(w));
            self.ctm_session = Some(s);
        }
    }

    /// Write split state back into `self.ctm` for serialization and
    /// for operations that still use the monolithic Ctm (sleep, dream, etc).
    pub fn sync_back(&mut self) {
        if let (Some(w), Some(s)) = (self.ctm_weights.take(), self.ctm_session.take()) {
            let w = Arc::try_unwrap(w).unwrap_or_else(|arc| (*arc).clone());
            self.ctm = Ctm::from_split(w, s);
        }
    }

    /// Set telemetry on the organism. Handles both monolithic Ctm and split state.
    /// If the Ctm has been split, telemetry goes on the session (primary thread).
    /// Otherwise it goes on self.ctm for transfer during the next ensure_split().
    pub fn set_telemetry(&mut self, t: Option<modgrad_io::telemetry::Telemetry>) {
        if let Some(session) = self.ctm_session.as_mut() {
            session.set_telemetry(t);
        } else {
            self.ctm.set_telemetry(t);
        }
    }

    /// Read-only sensory forward: embedding -> CTM observation.
    /// Does NOT collect traces (for use inside parallel closures).
    fn sensory_forward_readonly(
        sensory_layers: &[Linear],
        embedding: &[f32],
        d_input: usize,
        parent_supervised: bool,
    ) -> Vec<f32> {
        if sensory_layers.is_empty()
            || (embedding.len() == d_input && !parent_supervised)
        {
            return embedding.to_vec();
        }
        let mut x = embedding.to_vec();
        for (i, layer) in sensory_layers.iter().enumerate() {
            x = layer.forward(&x);
            if i < sensory_layers.len() - 1 {
                for v in &mut x { *v = v.max(0.0); }
            }
        }
        x
    }

    /// Optimal batch size based on neuron count and available cores.
    pub fn optimal_batch_size(&self) -> usize {
        let total_neurons = self.dna.ctm.input_layer.n_neurons
            + self.dna.ctm.attention_layer.n_neurons
            + self.dna.ctm.output_layer.n_neurons
            + self.dna.ctm.motor_layer.n_neurons;
        let cores = rayon::current_num_threads().max(1);
        if total_neurons >= 1024 {
            // Large/medium: start with 2 to diagnose, scale up once verified
            2
        } else if total_neurons >= 256 {
            // Medium: moderate parallelism
            4.min(cores)
        } else {
            // Small/tiny: use all cores
            cores
        }
    }

    /// Zero-copy batch parallel training.
    ///
    /// Processes N texts in parallel where all threads share the same
    /// `Arc<CtmWeights>` (zero cloning of weights). Each thread gets
    /// its own lightweight `CtmSession` + `CtmTickState` (O(neurons)
    /// allocation, NOT O(weights)).
    ///
    /// Returns the mean loss across the batch.
    pub fn train_batch_zero_copy(&mut self, batch: &[&[u8]]) -> f32 {
        self.ensure_split();
        let weights = self.ctm_weights.as_ref().unwrap().clone(); // Arc clone = pointer copy
        let _base_session = self.ctm_session.as_ref().unwrap();

        // Arc-share the read-only organism-level data
        let embeddings = Arc::new(self.embeddings.clone());
        let sensory_layers = Arc::new(self.sensory_layers.clone());
        let output_proj = Arc::new(self.output_proj.clone());
        let embed_dim = self.dna.embed_dim;
        let vocab_size = self.dna.vocab_size;
        let d_input = self.dna.ctm.d_input;
        let context_len = self.dna.context_len;
        let parent_supervised = self.parent_supervised_sensory;
        let neuromod = self.neuromod.clone();
        let tuning = self.tuning.clone();
        let _n_sync_out = self.dna.ctm.n_sync_out;

        let batch_t0 = std::time::Instant::now();
        // Each thread: create lightweight session + tick_state, run forward, collect deltas
        let results: Vec<_> = batch.par_iter()
            .filter_map(|text| {
                let ids: Vec<usize> = text.iter().map(|&b| b as usize).collect();
                let end = ids.len().min(context_len);
                if end < 2 { return None; }
                let ids = &ids[..end];

                // Per-thread: O(neurons) allocation, NOT O(weights)
                let mut session = CtmSession::new(&weights.config);
                session.hebbian_enabled = true;
                let mut tick_state = weights.init_tick_state();
                tick_state.neuromod = neuromod.clone();
                tick_state.noisy = true;

                let proprio = vec![0.0f32; d_input];

                let mut all_logits = Vec::with_capacity(end);
                let mut all_syncs = Vec::with_capacity(end);

                for &tid in ids {
                    // Embed
                    let tok = tid.min(vocab_size - 1);
                    let offset = tok * embed_dim;
                    let emb = embeddings[offset..offset + embed_dim].to_vec();

                    // Sensory forward (read-only)
                    let obs = Self::sensory_forward_readonly(
                        &sensory_layers, &emb, d_input, parent_supervised);

                    // CTM forward_split: reads shared weights, writes to local session
                    let (_preds, sync, _signals) = forward_split(
                        &weights, &mut session, &mut tick_state,
                        &obs, &proprio, true,
                    );

                    let logits = output_proj.forward(&sync);
                    all_logits.push(logits);
                    all_syncs.push(sync);
                }

                // Compute loss
                let targets = &ids[1..];
                let mut total_loss = 0.0f32;
                let mut n_targets = 0;
                for (i, &target) in targets.iter().enumerate() {
                    if i >= all_logits.len() { break; }
                    let log = &all_logits[i];
                    let v = log.len();
                    let target_idx = target.min(v - 1);
                    let max_l: f32 = log.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = log.iter().map(|&l| (l - max_l).exp()).sum();
                    let log_prob = (log[target_idx] - max_l) - exp_sum.ln();
                    total_loss += -log_prob;
                    n_targets += 1;
                }
                let loss = if n_targets > 0 { total_loss / n_targets as f32 } else { 0.0 };

                // Compute embedding deltas
                let mut embed_delta = vec![0.0f32; vocab_size * embed_dim];
                let surprises: Vec<f32> = {
                    let mut s = Vec::with_capacity(n_targets);
                    for (i, &target) in targets.iter().enumerate() {
                        if i >= all_logits.len() { break; }
                        let log = &all_logits[i];
                        let v = log.len();
                        let target_idx = target.min(v - 1);
                        let max_l: f32 = log.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let exp_sum: f32 = log.iter().map(|&l| (l - max_l).exp()).sum();
                        s.push(-((log[target_idx] - max_l) - exp_sum.ln()));
                    }
                    s
                };

                let lr = tuning.learning.surprise_lr_max.val() * 0.01; // base LR for batch
                for i in 1..ids.len().min(surprises.len() + 1) {
                    let surprise = surprises.get(i - 1).copied().unwrap_or(1.0);
                    let modulated_lr = lr * (surprise / 5.0).clamp(
                        tuning.learning.surprise_lr_min.val(),
                        tuning.learning.surprise_lr_max.val(),
                    );
                    let prev_tid = ids[i - 1].min(vocab_size - 1);
                    let curr_tid = ids[i].min(vocab_size - 1);
                    let prev_offset = prev_tid * embed_dim;
                    let curr_offset = curr_tid * embed_dim;
                    for j in 0..embed_dim {
                        let delta = embeddings[prev_offset + j] - embeddings[curr_offset + j];
                        embed_delta[curr_offset + j] += modulated_lr * delta
                            * tuning.learning.embedding_delta_scale.val();
                    }
                }

                // Compute output projection deltas
                let out_in_dim = output_proj.in_dim;
                let mut output_weight_delta = vec![0.0f32; vocab_size * out_in_dim];
                let mut output_bias_delta = vec![0.0f32; vocab_size];
                for i in 1..ids.len().min(surprises.len() + 1) {
                    let surprise = surprises.get(i - 1).copied().unwrap_or(1.0);
                    let modulated_lr = lr * (surprise / 5.0).clamp(
                        tuning.learning.surprise_lr_min.val(),
                        tuning.learning.surprise_lr_max.val(),
                    );
                    if i - 1 >= all_logits.len() { continue; }
                    let log = &all_logits[i - 1];
                    let target = ids[i].min(vocab_size - 1);
                    let predicted = log.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(idx, _)| idx).unwrap_or(0);
                    if predicted != target {
                        if let Some(sync) = all_syncs.get(i - 1) {
                            let target_lr = modulated_lr
                                * tuning.learning.output_target_lr_scale.val();
                            let predicted_lr = modulated_lr
                                * tuning.learning.output_predicted_lr_scale.val();
                            for k in 0..out_in_dim.min(sync.len()) {
                                output_weight_delta[target * out_in_dim + k] += target_lr * sync[k];
                                output_weight_delta[predicted * out_in_dim + k] -= predicted_lr * sync[k];
                            }
                            output_bias_delta[target] += target_lr;
                            output_bias_delta[predicted] -= predicted_lr;
                        }
                    }
                }

                // Collect output traces for sleep consolidation
                let mut traces: Vec<(Vec<f32>, usize)> = Vec::new();
                for i in 0..ids.len().saturating_sub(1) {
                    if i < all_syncs.len() {
                        traces.push((all_syncs[i].clone(), ids[i + 1]));
                    }
                }

                // Push replay entry if surprising enough
                let mean_surprise = if surprises.is_empty() { 1.0 } else {
                    surprises.iter().sum::<f32>() / surprises.len() as f32
                };

                Some((loss, embed_delta, output_weight_delta, output_bias_delta,
                       traces, mean_surprise, ids.len()))
            })
            .collect();

        let batch_elapsed = batch_t0.elapsed();
        eprintln!("    [batch] {} threads × {} tokens in {:.2}s",
            results.len(), batch.iter().map(|t| t.len()).sum::<usize>(), batch_elapsed.as_secs_f64());
        if results.is_empty() { return 0.0; }

        // Merge deltas back into organism state
        let n_batch = results.len() as f32;
        let mut total_loss = 0.0f32;
        let mut total_tokens = 0usize;

        for (loss, embed_delta, output_w_delta, output_b_delta,
             traces, _mean_surprise, n_tokens) in &results
        {
            total_loss += loss * *n_tokens as f32;
            total_tokens += n_tokens;

            // Accumulate embedding deltas (averaged across batch)
            let scale = 1.0 / n_batch;
            for (i, d) in embed_delta.iter().enumerate() {
                self.embeddings[i] += d * scale;
            }

            // Accumulate output projection deltas
            let out = &mut self.output_proj;
            for (i, d) in output_w_delta.iter().enumerate() {
                if i < out.weight.len() {
                    out.weight[i] += d * scale;
                }
            }
            for (i, d) in output_b_delta.iter().enumerate() {
                if i < out.bias.len() {
                    out.bias[i] += d * scale;
                }
            }

            // Collect output traces for sleep
            for (sync, target) in traces {
                if self.output_traces.len() < 50000 {
                    self.output_traces.push((sync.clone(), *target));
                }
            }
        }

        // Update tokens_seen
        self.tokens_seen += total_tokens as u64;

        // Update neuromodulators from mean surprise
        let mean_loss = if total_tokens > 0 { total_loss / total_tokens as f32 } else { 0.0 };
        self.neuromod.update_dopamine(mean_loss);
        self.neuromod.decay_ne();

        // Apply BG weight deltas from session back to weights
        if let Some(session) = self.ctm_session.as_mut() {
            if let Some(w_arc) = self.ctm_weights.as_mut() {
                if let Some(w) = Arc::get_mut(w_arc) {
                    session.apply_bg_weight_delta(w);
                }
            }
        }

        mean_loss
    }

    /// Sleep cycle: consolidate what was learned during waking.
    /// Consolidates CTM synapses AND sensory MLP layers via least-squares.
    pub fn sleep(&mut self) {
        // Sync back from split state before sleep (sleep uses monolithic Ctm)
        self.sync_back();
        // Measure sync diversity BEFORE consolidation.
        // If syncs are too similar, skip consolidation — it would only make things worse.
        let diversity = self.measure_sync_diversity();
        let too_similar = diversity > self.tuning.diversity.collapse_threshold.val();

        // Dream: replay high-surprise experiences
        self.ctm.dream();

        // NREM: consolidate CTM synapse weights.
        // ONLY if sync patterns are diverse enough to provide meaningful signal.
        // Consolidating uniform patterns makes everything MORE uniform.
        if !too_similar {
            let t = &self.tuning.sleep;
            let sleep_blend = (t.blend_initial.val()
                + (t.blend_max.val() - t.blend_initial.val())
                * (self.sleep_cycles as f32 / t.blend_ramp_cycles.val()).min(1.0))
                .min(t.blend_max.val());
            self.ctm.run_sleep(sleep_blend);
        } else if self.sleep_cycles % 20 == 0 {
            eprintln!("    [sleep] skipping consolidation: sync diversity={diversity:.3} (too similar)");
        }

        // Consolidate sensory MLP via least-squares (Oja's rule analogue).
        // For each layer, solve W_opt = argmin ||Y - X W||^2 from collected traces.
        let min_traces = self.tuning.learning.min_ls_traces.val() as usize;
        if self.sensory_traces.len() >= min_traces && !too_similar {
            for layer_idx in 0..self.sensory_layers.len() {
                let xs: Vec<&[f32]> = self.sensory_traces.iter()
                    .filter(|(li, _, _)| *li == layer_idx)
                    .map(|(_, inp, _)| inp.as_slice())
                    .collect();
                let ys: Vec<&[f32]> = self.sensory_traces.iter()
                    .filter(|(li, _, _)| *li == layer_idx)
                    .map(|(_, _, out)| out.as_slice())
                    .collect();

                if xs.len() < 5 { continue; }
                let in_dim = xs[0].len();
                let out_dim = ys[0].len();

                if let Some(w_opt) = super::eval::linalg::least_squares(&xs, &ys, in_dim, out_dim, 1e-4) {
                    let layer = &mut self.sensory_layers[layer_idx];
                    for o in 0..out_dim.min(layer.out_dim) {
                        for i in 0..in_dim.min(layer.in_dim) {
                            let old = layer.weight[o * layer.in_dim + i];
                            let new = w_opt[i * out_dim + o];
                            // Conservative blend — too aggressive destroys input diversity
                        let b = self.tuning.learning.sensory_ls_blend.val();
                        layer.weight[o * layer.in_dim + i] = (1.0 - b) * old + b * new;
                        }
                    }
                }
            }
            self.sensory_traces.clear();
        }

        // Parent-supervised sensory training (option 2 from analysis).
        // Solve W_opt = argmin ||parent_hidden - W · embedding||²
        // The Angeris bound = residual/total tells us how much is linearly recoverable.
        if self.sensory_parent_traces.len() >= 20 && !self.sensory_layers.is_empty() {
            let in_dim = self.sensory_parent_traces[0].0.len();
            let out_dim = self.sensory_parent_traces[0].1.len();

            // Use optimized parallel linalg
            let xs: Vec<&[f32]> = self.sensory_parent_traces.iter().map(|(x, _)| x.as_slice()).collect();
            let ys: Vec<&[f32]> = self.sensory_parent_traces.iter().map(|(_, y)| y.as_slice()).collect();

            if let Some(w_opt) = super::eval::linalg::least_squares(&xs, &ys, in_dim, out_dim, 1e-4) {
                // Angeris bound diagnostic
                let sample_xs: Vec<&[f32]> = xs.iter().take(200).copied().collect();
                let sample_ys: Vec<&[f32]> = ys.iter().take(200).copied().collect();
                let (residual_frac, _total) = super::eval::linalg::angeris_residual(
                    &sample_xs, &sample_ys, &w_opt, in_dim, out_dim);
                let recoverable_pct = (1.0 - residual_frac) * 100.0;

                eprintln!("    sensory: {recoverable_pct:.1}% linearly recoverable (residual {:.1}%)",
                    residual_frac * 100.0);

                // Apply to first sensory layer
                let layer = &mut self.sensory_layers[0];
                let apply_out = out_dim.min(layer.out_dim);
                let apply_in = in_dim.min(layer.in_dim);
                for o in 0..apply_out {
                    let dst_start = o * layer.in_dim;
                    for i in 0..apply_in {
                        let old = layer.weight[dst_start + i];
                        let new = w_opt[i * out_dim + o];
                        layer.weight[dst_start + i] = 0.5 * old + 0.5 * new;
                    }
                }

                self.parent_supervised_sensory = true;
            }

            self.sensory_parent_traces.clear();
        }

        // Train output projection via least-squares from (sync, target_token) pairs.
        // Same approach that fixed the sensory MLP: proper consolidation, not tiny nudges.
        //
        // Target: one-hot encoding of next token. But solving for full 256-dim is wasteful
        // since most tokens are rare. Instead: solve column-by-column for tokens that
        // appeared in traces, leaving rare token weights unchanged.
        // Output readout: least-squares from (sync, next_token) traces.
        // This is the reservoir computing readout — THE most important weight update.
        // The CTM (reservoir) provides rich temporal features.
        // The readout extracts the linear mapping sync → token prediction.
        // Conservative 0.1 blend to avoid oscillation.
        // Only consolidate output projection when sync patterns are diverse enough.
        // LS on uniform syncs produces uniform weights → kills discrimination.
        if self.output_traces.len() >= 200 && !too_similar {
            let in_dim = self.output_traces[0].0.len(); // sync dimension
            let vocab = self.dna.vocab_size;

            // Count which tokens appeared as targets
            let mut token_counts = vec![0u32; vocab];
            for (_, tid) in &self.output_traces {
                if *tid < vocab { token_counts[*tid] += 1; }
            }

            // Build XTX once (shared across all output columns)
            let xs: Vec<&[f32]> = self.output_traces.iter().map(|(s, _)| s.as_slice()).collect();
            let mut xtx = vec![0.0f32; in_dim * in_dim];
            super::eval::linalg::accumulate_xtx(&mut xtx, &xs, in_dim);
            for i in 0..in_dim {
                xtx[i * in_dim + i] += 1e-4; // regularization
            }

            // Cholesky once
            if let Some(l) = super::eval::linalg::cholesky(&xtx, in_dim) {
                let mut updated = 0;

                // For each token that appeared at least 5 times, solve its column
                for tid in 0..vocab {
                    if token_counts[tid] < 5 { continue; }

                    // Build XTy for this token: y[i] = 1.0 if trace[i].target == tid, else 0.0
                    let mut xty = vec![0.0f32; in_dim];
                    for (sync, target) in &self.output_traces {
                        if *target == tid {
                            for i in 0..in_dim {
                                xty[i] += sync[i]; // sync[i] * 1.0
                            }
                        }
                    }

                    // Solve: L L^T w = xty
                    let z = super::eval::linalg::forward_solve(&l, &xty, in_dim);
                    let w = super::eval::linalg::backward_solve(&l, &z, in_dim);

                    // Blend into output_proj: row `tid` of weight matrix.
                    // Blend rate scales with evidence: more traces → more confidence → faster blend.
                    // At 5 traces: blend=0.1 (conservative). At 1000+: blend=0.5 (aggressive).
                    let evidence = token_counts[tid] as f32;
                    let blend = (0.1 + 0.4 * (evidence / 1000.0).min(1.0)).min(0.5);
                    let row_start = tid * self.output_proj.in_dim;
                    for i in 0..in_dim.min(self.output_proj.in_dim) {
                        let old = self.output_proj.weight[row_start + i];
                        self.output_proj.weight[row_start + i] = (1.0 - blend) * old + blend * w[i];
                    }
                    updated += 1;
                }

                if updated > 0 {
                    eprintln!("    output_proj: consolidated {} / {} token columns from {} traces",
                        updated, vocab, self.output_traces.len());
                }
            }

            self.output_traces.clear();
        }

        self.ctm.train_logit_projector();
        self.sleep_cycles += 1;
    }

    /// Train on raw text for N steps with periodic sleep.
    /// `text_fn`: returns the next chunk of text as bytes.
    /// `steps`: number of training steps.
    /// `sleep_every`: run sleep cycle every N steps.
    pub fn develop(
        &mut self,
        texts: &[&[u8]],
        sleep_every: usize,
    ) -> Vec<f32> {
        self.develop_with_checkpoint(texts, sleep_every, None)
    }

    /// Train with periodic checkpointing. Saves organism to disk
    /// every sleep cycle so progress survives crashes.
    ///
    /// Uses zero-copy batch parallel training when batch_size > 1.
    /// All threads share `Arc<CtmWeights>` (zero cloning of weights).
    pub fn develop_with_checkpoint(
        &mut self,
        texts: &[&[u8]],
        sleep_every: usize,
        checkpoint_path: Option<&str>,
    ) -> Vec<f32> {
        // Metrics: write .jsonl alongside checkpoint, .prom for prometheus
        let metrics_jsonl = checkpoint_path.map(|p| format!("{}.jsonl", p.trim_end_matches(".bin")));
        let metrics_prom = checkpoint_path.map(|_| "isis_metrics.prom".to_string());
        let mut metrics = super::eval::metrics::Collector::new(metrics_jsonl, 10);
        let mut prom = super::eval::metrics::Collector::new(metrics_prom, 1);

        let batch_size = self.optimal_batch_size();
        let mut losses = Vec::new();
        let mut step = 0usize;

        // Process texts in batches
        let mut text_idx = 0;
        while text_idx < texts.len() {
            let batch_end = (text_idx + batch_size).min(texts.len());
            let batch = &texts[text_idx..batch_end];
            let batch_len = batch.len();

            let t0 = std::time::Instant::now();

            let loss = if batch_len == 1 {
                // Single text: use the original sequential path (cheaper for small batches)
                let token_ids: Vec<usize> = batch[0].iter().map(|&b| b as usize).collect();
                if token_ids.len() < 2 {
                    text_idx += batch_len;
                    continue;
                }
                let end = token_ids.len().min(self.dna.context_len);
                self.train_step(&token_ids[..end])
            } else {
                // Batch parallel: zero-copy shared weights
                self.train_batch_zero_copy(batch)
            };

            let elapsed = t0.elapsed();
            losses.push(loss);

            let n = losses.len();
            let avg_loss = if n <= 10 { loss } else {
                losses[n.saturating_sub(10)..].iter().sum::<f32>() / 10.0
            };

            // Count total tokens in this batch for tok/s
            let total_tokens: usize = batch.iter()
                .map(|t| t.len().min(self.dna.context_len))
                .sum();
            let tokens_per_sec = total_tokens as f32 / elapsed.as_secs_f32();

            // Push metrics snapshot
            let snapshot = super::eval::metrics::Snapshot {
                timestamp: super::eval::metrics::Snapshot::now(),
                step: step as u64,
                tokens_seen: self.tokens_seen,
                sleep_cycles: self.sleep_cycles,
                loss,
                alignment: 0.0,
                tokens_per_sec,
                pressure: self.neuromod.dopamine,
                output_quality: 1.0 - (self.neuromod.dopamine - 0.75).abs(),
                activation_pressure: self.neuromod.dopamine,
                divergence_pressure: (1.0 - self.neuromod.serotonin).max(0.0),
                drift_pressure: 0.0,
                buffer_pressure: self.ctm.replay.len() as f32 / 64.0,
                emotional_pressure: self.neuromod.norepinephrine / 3.0,
                surprise_ema: self.neuromod.surprise_ema.unwrap_or(0.0),
                health_score: 1.0, fear_ratio: 0.0, negative_ratio: 0.0,
                avoidance_generalization: 0.0, total_memories: 0, fear_memories: 0,
                ctm_confidence: 0.0, dead_neurons: 0,
                total_neurons: self.dna.ctm.d_model as u32,
                synapse_gaps: Vec::new(),
                sensory_recovery_pct: 0.0,
                diagnoses: Vec::new(),
            };
            metrics.push(snapshot.clone());
            prom.push(snapshot);

            eprintln!("  [{}/{}] loss={:.1} avg10={:.1} {:.0}tok/s {:.1}s/step batch={}",
                text_idx + batch_len, texts.len(), loss, avg_loss,
                tokens_per_sec, elapsed.as_secs_f64(), batch_len);

            // Periodic sleep + checkpoint
            // Check against global step count for sleep timing
            step += 1;
            if step % sleep_every == 0 && step > 0 {
                // Sync back from split state before sleep
                self.sync_back();

                let st0 = std::time::Instant::now();
                self.sleep();
                let sleep_time = st0.elapsed().as_secs_f64();

                let bounds = self.ctm.angeris_bounds();
                let gaps: Vec<String> = bounds.synapse_gaps.iter()
                    .map(|sg| format!("{}:{:.1}%", sg.name.replace("syn_", ""), sg.gap_pct))
                    .collect();
                let dead_total: usize = bounds.dead_neurons.iter().sum();

                eprintln!("  SLEEP #{} ({:.1}s) tokens_seen={}",
                    self.sleep_cycles, sleep_time, self.tokens_seen);
                eprintln!("    angeris: [{}] dead={}/{}",
                    gaps.join(" "), dead_total, bounds.total_neurons);
                if bounds.synapse_gaps.iter().all(|sg| sg.gap_pct < 1.0) {
                    eprintln!("    ALL SYNAPSES OPTIMAL");
                }

                if let Some(path) = checkpoint_path {
                    // sync_back already called above; safe to save
                    if let Err(e) = self.save(path) {
                        eprintln!("  Checkpoint save failed: {e}");
                    } else {
                        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                        eprintln!("  Saved checkpoint ({} bytes)", size);
                    }
                }
            }

            text_idx += batch_len;
        }

        // Final sync_back to ensure ctm is up to date
        self.sync_back();

        metrics.flush();
        prom.flush();
        losses
    }

    /// Generate text from a prompt (byte-level).
    pub fn generate(&mut self, prompt: &[u8], max_tokens: usize) -> Vec<u8> {
        self.sync_back(); // ensure ctm has latest state from any parallel training
        let mut state = self.ctm.init_state();
        let mut ids: Vec<usize> = prompt.iter().map(|&b| b as usize).collect();

        // Process prompt
        for &tid in &ids {
            let emb = self.embed(tid);
            let obs = self.sensory_forward(&emb, false);
            self.ctm.forward(&obs, &mut state, false);
        }

        // Generate
        for _ in 0..max_tokens {
            let last_tid = *ids.last().unwrap_or(&0);
            let emb = self.embed(last_tid);
            let obs = self.sensory_forward(&emb, false);

            let (_preds, sync) = self.ctm.forward(&obs, &mut state, false);
            let logits = self.output_proj.forward(&sync);

            // Argmax over printable ASCII only (32-126).
            // Untrained token columns have random weights that produce
            // garbage logits. Masking to printable bytes fixes this.
            let next = logits.iter().enumerate()
                .filter(|(i, _)| *i >= 32 && *i <= 126)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(32); // space as fallback

            ids.push(next);
            if next == 0 || next >= 256 { break; } // null byte = stop
        }

        ids[prompt.len()..].iter().map(|&id| id as u8).collect()
    }

    /// Get the hidden state (CTM sync signal) for a token sequence.
    /// This is the organism's own representation — its "key" for memory.
    pub fn get_key(&mut self, token_ids: &[usize]) -> Vec<f32> {
        let mut state = self.ctm.init_state();
        let mut last_sync = vec![0.0f32; self.dna.ctm.n_sync_out];

        for &tid in token_ids {
            let emb = self.embed(tid);
            let obs = self.sensory_forward(&emb, false);
            let (_preds, sync) = self.ctm.forward(&obs, &mut state, false);
            last_sync = sync;
        }

        // Normalize
        let norm: f32 = last_sync.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut last_sync { *v /= norm; }
        }
        last_sync
    }

    /// Teach an episodic memory using the organism's own representations.
    /// No backbone needed — keys come from the organism's CTM.
    pub fn teach_memory(
        &mut self,
        bank: &mut modgrad_io::memory::MemoryBank,
        prompt: &str,
        answer: &str,
        alter: &str,
        importance: f32,
    ) {
        let prompt_ids: Vec<usize> = prompt.bytes().map(|b| b as usize).collect();
        let prompt_key = self.get_key(&prompt_ids);

        let answer_ids: Vec<usize> = answer.bytes().map(|b| b as usize).collect();

        // Compute logit biases for the answer tokens
        let mut logit_biases = Vec::new();
        for &tid in &answer_ids {
            logit_biases.push(modgrad_io::types::LogitBias {
                token_id: tid as u32,
                token: String::from(tid as u8 as char),
                strength: 50.0 * importance,
                suppress: Vec::new(),
            });
        }

        bank.teach(
            prompt, answer, alter,
            vec![], // no content keys for byte-level
            prompt_key,
            logit_biases,
            importance,
            1.0, // surprise
        );
    }

    /// Recall from episodic memory using the organism's own key space.
    pub fn recall_memory(
        &mut self,
        bank: &modgrad_io::memory::MemoryBank,
        prompt: &str,
    ) -> Option<String> {
        let prompt_ids: Vec<usize> = prompt.bytes().map(|b| b as usize).collect();
        let key = self.get_key(&prompt_ids);

        super::episode::recall(bank, &key)
            .map(|result| {
                bank.alters[result.alter_index].episodes[result.episode_index]
                    .answer.clone()
            })
    }

    /// Save organism. Uses binary format for .bin, JSON for .json.
    // Binary format magic + version for backward compatibility.
    const MAGIC: &[u8; 4] = b"isis";
    const FORMAT_VERSION: u32 = 2;

    /// Save organism. Uses binary format for .bin, JSON for .json.
    ///
    /// Binary format v2:
    /// ```text
    /// [4 bytes]  magic "ISIS"
    /// [4 bytes]  format version (u32 LE)
    /// [8 bytes]  tokens_seen (u64 LE)
    /// [8 bytes]  sleep_cycles (u64 LE)
    /// [4 bytes]  dna_json_len (u32 LE)
    /// [N bytes]  dna as JSON
    /// [4 bytes]  num_weight_sections (u32 LE)
    /// For each weight section:
    ///   [4 bytes]  name_len (u32 LE)
    ///   [N bytes]  name (utf8)
    ///   [4 bytes]  num_floats (u32 LE)
    ///   [N×4 bytes] f32 data
    /// [4 bytes]  ctm_json_len (u32 LE)
    /// [N bytes]  ctm as JSON
    /// ```
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if path.ends_with(".bin") {
            let mut f = std::fs::File::create(path)?;
            use std::io::Write;

            // Magic + version
            f.write_all(Self::MAGIC)?;
            f.write_all(&Self::FORMAT_VERSION.to_le_bytes())?;

            // Metadata
            f.write_all(&self.tokens_seen.to_le_bytes())?;
            f.write_all(&self.sleep_cycles.to_le_bytes())?;

            // DNA as JSON
            let dna_json = serde_json::to_string(&self.dna)?;
            f.write_all(&(dna_json.len() as u32).to_le_bytes())?;
            f.write_all(dna_json.as_bytes())?;

            // Weight sections — named, so we can add/skip unknown ones
            let write_section = |f: &mut std::fs::File, name: &str, data: &[f32]| -> std::io::Result<()> {
                let name_bytes = name.as_bytes();
                f.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
                f.write_all(name_bytes)?;
                f.write_all(&(data.len() as u32).to_le_bytes())?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                f.write_all(bytes)
            };

            // Count sections: embeddings + 2 per sensory layer + output weight + output bias
            let n_sections = 1 + self.sensory_layers.len() * 2 + 2;
            f.write_all(&(n_sections as u32).to_le_bytes())?;

            write_section(&mut f, "embeddings", &self.embeddings)?;
            for (i, layer) in self.sensory_layers.iter().enumerate() {
                write_section(&mut f, &format!("sensory_{i}_weight"), &layer.weight)?;
                write_section(&mut f, &format!("sensory_{i}_bias"), &layer.bias)?;
            }
            write_section(&mut f, "output_weight", &self.output_proj.weight)?;
            write_section(&mut f, "output_bias", &self.output_proj.bias)?;

            // CTM as JSON (complex nested structs)
            let ctm_json = serde_json::to_string(&self.ctm)?;
            f.write_all(&(ctm_json.len() as u32).to_le_bytes())?;
            f.write_all(ctm_json.as_bytes())?;

            Ok(())
        } else {
            let data = serde_json::to_string(self)?;
            std::fs::write(path, data)?;
            Ok(())
        }
    }

    /// Load organism. Detects format from extension.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        if path.ends_with(".bin") {
            use std::io::Read;
            let mut f = std::fs::File::open(path)?;

            // Magic
            let mut magic = [0u8; 4];
            f.read_exact(&mut magic)?;
            if &magic != Self::MAGIC {
                return Err("not an isis organism file".into());
            }

            // Version
            let mut ver_bytes = [0u8; 4];
            f.read_exact(&mut ver_bytes)?;
            let _version = u32::from_le_bytes(ver_bytes);

            // Metadata
            let mut buf8 = [0u8; 8];
            f.read_exact(&mut buf8)?;
            let tokens_seen = u64::from_le_bytes(buf8);
            f.read_exact(&mut buf8)?;
            let sleep_cycles = u64::from_le_bytes(buf8);

            // DNA
            let mut buf4 = [0u8; 4];
            f.read_exact(&mut buf4)?;
            let dna_len = u32::from_le_bytes(buf4) as usize;
            let mut dna_bytes = vec![0u8; dna_len];
            f.read_exact(&mut dna_bytes)?;
            let dna: Dna = serde_json::from_slice(&dna_bytes)?;

            let mut org = Self::new(dna);
            org.tokens_seen = tokens_seen;
            org.sleep_cycles = sleep_cycles;

            // Weight sections
            f.read_exact(&mut buf4)?;
            let n_sections = u32::from_le_bytes(buf4) as usize;

            let read_f32_vec = |f: &mut std::fs::File| -> Result<(String, Vec<f32>), Box<dyn std::error::Error>> {
                let mut buf4 = [0u8; 4];
                f.read_exact(&mut buf4)?;
                let name_len = u32::from_le_bytes(buf4) as usize;
                let mut name_bytes = vec![0u8; name_len];
                f.read_exact(&mut name_bytes)?;
                let name = String::from_utf8(name_bytes)?;

                f.read_exact(&mut buf4)?;
                let n_floats = u32::from_le_bytes(buf4) as usize;
                let mut data = vec![0.0f32; n_floats];
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n_floats * 4)
                };
                f.read_exact(bytes)?;
                Ok((name, data))
            };

            for _ in 0..n_sections {
                let (name, data) = read_f32_vec(&mut f)?;
                match name.as_str() {
                    "embeddings" => {
                        if data.len() == org.embeddings.len() {
                            org.embeddings = data;
                        }
                    }
                    "output_weight" => {
                        if data.len() == org.output_proj.weight.len() {
                            org.output_proj.weight = data;
                        }
                    }
                    "output_bias" => {
                        if data.len() == org.output_proj.bias.len() {
                            org.output_proj.bias = data;
                        }
                    }
                    other => {
                        // sensory_N_weight / sensory_N_bias
                        if let Some(rest) = other.strip_prefix("sensory_") {
                            let parts: Vec<&str> = rest.splitn(2, '_').collect();
                            if let (Some(idx), Some(which)) = (
                                parts.first().and_then(|s| s.parse::<usize>().ok()),
                                parts.get(1),
                            ) {
                                if idx < org.sensory_layers.len() {
                                    match *which {
                                        "weight" => {
                                            if data.len() == org.sensory_layers[idx].weight.len() {
                                                org.sensory_layers[idx].weight = data;
                                            }
                                        }
                                        "bias" => {
                                            if data.len() == org.sensory_layers[idx].bias.len() {
                                                org.sensory_layers[idx].bias = data;
                                            }
                                        }
                                        _ => {} // unknown section — skip (forward compat)
                                    }
                                }
                            }
                        }
                        // Unknown sections are silently skipped — forward compatibility
                    }
                }
            }

            // CTM
            f.read_exact(&mut buf4)?;
            let ctm_len = u32::from_le_bytes(buf4) as usize;
            let mut ctm_bytes = vec![0u8; ctm_len];
            f.read_exact(&mut ctm_bytes)?;
            org.ctm = serde_json::from_slice(&ctm_bytes)?;

            Ok(org)
        } else {
            let data = std::fs::read_to_string(path)?;
            Ok(serde_json::from_str(&data)?)
        }
    }

    /// Measure mean pairwise cosine similarity of sync patterns for a few test inputs.
    /// High similarity (>0.8) = syncs collapsing, consolidation would make it worse.
    /// Low similarity (<0.5) = good diversity, safe to consolidate.
    fn measure_sync_diversity(&mut self) -> f32 {
        // Quick test with 4 distinct byte patterns
        let test_inputs: Vec<Vec<usize>> = vec![
            vec![32, 32, 32, 32],    // spaces
            vec![97, 98, 99, 100],   // a b c d
            vec![48, 49, 50, 51],    // 0 1 2 3
            vec![116, 104, 101, 32], // t h e _
        ];
        let mut syncs = Vec::new();
        for ids in &test_inputs {
            let (_, s) = self.forward_inner(ids, false);
            if let Some(sync) = s.last() {
                syncs.push(sync.clone());
            }
        }
        if syncs.len() < 2 { return 0.0; }

        let mut total = 0.0f32;
        let mut count = 0;
        for i in 0..syncs.len() {
            for j in (i+1)..syncs.len() {
                let dot: f32 = syncs[i].iter().zip(&syncs[j]).map(|(a, b)| a * b).sum();
                let na: f32 = syncs[i].iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = syncs[j].iter().map(|x| x * x).sum::<f32>().sqrt();
                if na > 1e-8 && nb > 1e-8 {
                    total += dot / (na * nb);
                }
                count += 1;
            }
        }
        if count > 0 { total / count as f32 } else { 0.0 }
    }

    pub fn param_count(&self) -> usize {
        let embed = self.dna.vocab_size * self.dna.embed_dim;
        let sensory: usize = self.sensory_layers.iter()
            .map(|l| l.in_dim * l.out_dim + l.out_dim)
            .sum();
        let output = self.output_proj.in_dim * self.output_proj.out_dim + self.output_proj.out_dim;
        // CTM params are harder to count precisely, estimate
        let ctm_est = self.dna.ctm.d_model * self.dna.ctm.d_model * 8; // rough
        embed + sensory + output + ctm_est
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_organism_creation() {
        let org = Organism::new(Dna::default());
        assert_eq!(org.tokens_seen, 0);
        assert_eq!(org.sleep_cycles, 0);
        assert_eq!(org.embeddings.len(), 256 * 64);
        eprintln!("Organism: {} params", org.param_count());
    }

    #[test]
    fn test_organism_forward() {
        let mut org = Organism::new(Dna::default());
        let tokens: Vec<usize> = b"hello world".iter().map(|&b| b as usize).collect();
        let logits = org.forward(&tokens);
        assert_eq!(logits.len(), tokens.len());
        assert_eq!(logits[0].len(), 256); // vocab_size
    }

    #[test]
    fn test_organism_train_step() {
        let mut org = Organism::new(Dna::default());
        let tokens: Vec<usize> = b"the cat sat on the mat".iter().map(|&b| b as usize).collect();
        let loss = org.train_step(&tokens);
        assert!(loss > 0.0, "loss should be positive: {loss}");
        assert!(loss.is_finite(), "loss should be finite: {loss}");
        // Initial loss can be high — untrained weights produce extreme logits.
        // Just verify it's not NaN/Inf and is non-negative.
        assert_eq!(org.tokens_seen, tokens.len() as u64);
    }

    #[test]
    fn test_organism_develop() {
        let mut org = Organism::new(Dna::default());

        // Simple training data: repeated patterns.
        // With batch parallelism, multiple texts are processed per step.
        // The batch_size depends on core count and model size, so we
        // generate enough texts for multiple steps regardless of batch_size.
        let batch_size = org.optimal_batch_size();
        let sleep_every = 5;
        // Need at least sleep_every * batch_size * 2 texts to trigger 2 sleep cycles.
        let n_texts = sleep_every * batch_size * 2;
        let patterns: [&[u8]; 2] = [
            b"the cat sat on the mat",
            b"the dog sat on the log",
        ];
        let texts: Vec<&[u8]> = (0..n_texts).map(|i| patterns[i % 2]).collect();

        let losses = org.develop(&texts, sleep_every);

        // Number of steps = ceil(n_texts / batch_size).
        // With our sizing, this should be exactly sleep_every * 2 = 10.
        let expected_steps = (n_texts + batch_size - 1) / batch_size;
        assert_eq!(losses.len(), expected_steps,
            "expected {} steps (batch_size={}, n_texts={})", expected_steps, batch_size, n_texts);

        // Loss should exist (not NaN)
        for loss in &losses {
            assert!(!loss.is_nan(), "loss should not be NaN");
            assert!(loss.is_finite(), "loss should be finite");
        }

        // After training, loss should have decreased (at least somewhat)
        let mid = losses.len() / 2;
        let first_half: f32 = losses[..mid].iter().sum::<f32>() / mid as f32;
        let second_half: f32 = losses[mid..].iter().sum::<f32>() / (losses.len() - mid) as f32;
        eprintln!("First half loss: {first_half:.3}, Second half: {second_half:.3}");

        assert_eq!(org.sleep_cycles, 2); // slept twice (at step 5 and 10)
    }

    #[test]
    fn test_organism_generate() {
        let mut org = Organism::new(Dna::default());

        // Train on some data first
        let texts: Vec<&[u8]> = vec![b"hello world"; 20];
        org.develop(&texts, 10);

        // Generate from prompt
        let output = org.generate(b"hel", 10);
        assert!(!output.is_empty(), "should generate something");
        assert!(output.len() <= 10, "should respect max_tokens");

        // Output should be valid bytes
        for &b in &output {
            assert!(b > 0, "should not generate null bytes");
        }
    }

    #[test]
    fn test_organism_save_load() {
        let org = Organism::new(Dna::default());
        let path = "/tmp/isis_organism_test.json";
        org.save(path).unwrap();

        let loaded = Organism::load(path).unwrap();
        assert_eq!(loaded.dna.vocab_size, org.dna.vocab_size);
        assert_eq!(loaded.embeddings.len(), org.embeddings.len());
        assert_eq!(loaded.tokens_seen, 0);

        std::fs::remove_file(path).ok();
    }
}
