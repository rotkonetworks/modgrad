//! Weights and state for the faithful Ctm CTM.

use serde::{Deserialize, Serialize};
use modgrad_compute::neuron::{Linear, SuperLinear, SimpleRng};
use super::config::CtmConfig;
use super::synapse::SynapseUNet;

/// All trainable weights for the Ctm CTM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtmWeights {
    pub config: CtmConfig,

    // ── Synapse: concat(attn_out, activated_state) → pre-activations ──
    pub synapse: SynapseUNet,

    // ── NLM (trace_processor) ──
    // Deep:    SuperLinear(M → 2H) → GLU → SuperLinear(H → 2) → GLU → [D]
    // Shallow: SuperLinear(M → 2)  → GLU → [D]
    pub nlm_stage1: SuperLinear,
    pub nlm_stage2: Option<SuperLinear>,

    // ── Learnable start states ──
    pub start_activated: Vec<f32>,   // [d_model]
    pub start_trace: Vec<f32>,       // [d_model × memory_length]

    // ── Input feature projection ──
    pub kv_proj: Linear,             // raw_input_dim → d_input
    pub kv_ln_gamma: Vec<f32>,       // [d_input]
    pub kv_ln_beta: Vec<f32>,        // [d_input]

    // ── Attention ──
    pub q_proj: Linear,              // synch_action_size → d_input
    // Multihead attention: packed in_proj and out_proj
    pub mha_in_proj: Linear,         // d_input → 3×d_input (Q,K,V packed)
    pub mha_out_proj: Linear,        // d_input → d_input

    // ── Sync topology (random-pairing indices) ──
    pub sync_out_left: Vec<usize>,
    pub sync_out_right: Vec<usize>,
    pub sync_action_left: Vec<usize>,
    pub sync_action_right: Vec<usize>,

    // ── Learnable decay parameters ──
    pub decay_params_out: Vec<f32>,    // [n_synch_out]
    pub decay_params_action: Vec<f32>, // [n_synch_action]

    // ── Output projector ──
    pub output_proj: Linear,           // synch_out_size → out_dims

    // ── Adaptive exit gate (LoopLM-style) ──
    // Present iff exit_strategy is AdaptiveGate.
    // sync_out → 1 → sigmoid → per-tick halting probability.
    #[serde(default)]
    pub exit_gate: Option<Linear>,     // synch_out_size → 1
}

impl CtmWeights {
    pub fn new(config: CtmConfig, raw_input_dim: usize) -> Self {
        let d = config.d_model;
        let m = config.memory_length;
        let h = config.memory_hidden_dims;

        // NLM stages
        let (nlm1_out, nlm_stage2) = if config.deep_nlms {
            (2 * h, Some(SuperLinear::new(d, h, 2)))
        } else {
            (2, None)
        };
        let nlm_stage1 = SuperLinear::new(d, m, nlm1_out);

        // Start states: uniform init matching PyTorch default
        let mut rng = SimpleRng::new(42);
        let bound1 = 1.0 / (d as f32).sqrt();
        let start_activated: Vec<f32> = (0..d)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound1).collect();
        let bound2 = 1.0 / (m as f32).sqrt();
        let start_trace: Vec<f32> = (0..d * m)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound2).collect();

        // Sync indices: random-pairing
        let sync_out_left = random_indices(d, config.n_synch_out, &mut rng);
        let mut sync_out_right = random_indices(d, config.n_synch_out, &mut rng);
        // Self-pairing for first n_random_pairing_self entries
        for i in 0..config.n_random_pairing_self.min(config.n_synch_out) {
            sync_out_right[i] = sync_out_left[i];
        }

        let sync_action_left = random_indices(d, config.n_synch_action, &mut rng);
        let mut sync_action_right = random_indices(d, config.n_synch_action, &mut rng);
        for i in 0..config.n_random_pairing_self.min(config.n_synch_action) {
            sync_action_right[i] = sync_action_left[i];
        }

        // Decay params: init to 1.0 (will be clamped to [0,15], exp(-1) ≈ 0.37 decay)
        let decay_params_out = vec![1.0; config.n_synch_out];
        let decay_params_action = vec![1.0; config.n_synch_action];

        Self {
            synapse: SynapseUNet::new(
                config.synapse_in_dim(), config.d_model,
                config.synapse_depth, config.min_width,
            ),
            nlm_stage1,
            nlm_stage2,
            start_activated,
            start_trace,
            kv_proj: Linear::new(raw_input_dim, config.d_input),
            kv_ln_gamma: vec![1.0; config.d_input],
            kv_ln_beta: vec![0.0; config.d_input],
            q_proj: Linear::new(config.synch_size_action(), config.d_input),
            mha_in_proj: Linear::new(config.d_input, 3 * config.d_input),
            mha_out_proj: Linear::new(config.d_input, config.d_input),
            sync_out_left,
            sync_out_right,
            sync_action_left,
            sync_action_right,
            decay_params_out,
            decay_params_action,
            output_proj: Linear::new(config.synch_size_out(), config.out_dims),
            exit_gate: if config.exit_strategy.has_gate() {
                Some(Linear::new(config.synch_size_out(), 1))
            } else {
                None
            },
            config,
        }
    }

    /// Save weights. Format by extension: `.bin` → bincode, `.json` → JSON.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        modgrad_persist::persist::save(self, path).map_err(|e| e.into())
    }

    /// Load weights. Format by extension, with JSON fallback for legacy files.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        modgrad_persist::persist::load(path).map_err(|e| e.into())
    }

    /// Total trainable parameters.
    pub fn n_params(&self) -> usize {
        let mut n = 0usize;
        n += self.synapse.n_params();
        n += self.nlm_stage1.weights.len() + self.nlm_stage1.biases.len();
        if let Some(ref s2) = self.nlm_stage2 {
            n += s2.weights.len() + s2.biases.len();
        }
        n += self.start_activated.len() + self.start_trace.len();
        n += self.kv_proj.weight.len() + self.kv_proj.bias.len();
        n += self.kv_ln_gamma.len() * 2;
        n += self.q_proj.weight.len() + self.q_proj.bias.len();
        n += self.mha_in_proj.weight.len() + self.mha_in_proj.bias.len();
        n += self.mha_out_proj.weight.len() + self.mha_out_proj.bias.len();
        n += self.decay_params_out.len() + self.decay_params_action.len();
        n += self.output_proj.weight.len() + self.output_proj.bias.len();
        if let Some(ref g) = self.exit_gate { n += g.weight.len() + g.bias.len(); }
        n
    }
}

/// Per-forward mutable state (not persisted).
pub struct CtmState {
    /// Pre-activation history: [d_model × memory_length], row-major by neuron.
    pub trace: Vec<f32>,
    /// Post-activation state: [d_model].
    pub activated: Vec<f32>,

    /// Sync accumulators.
    pub alpha_action: Option<Vec<f32>>,
    pub beta_action: Option<Vec<f32>>,
    pub alpha_out: Vec<f32>,
    pub beta_out: Vec<f32>,
}

impl CtmState {
    pub fn new(w: &CtmWeights) -> Self {
        Self {
            trace: w.start_trace.clone(),
            activated: w.start_activated.clone(),
            alpha_action: None,
            beta_action: None,
            alpha_out: vec![0.0; w.config.n_synch_out],
            beta_out: vec![0.0; w.config.n_synch_out],
        }
    }
}

fn random_indices(max: usize, count: usize, rng: &mut SimpleRng) -> Vec<usize> {
    (0..count).map(|_| (rng.next_u64() as usize) % max).collect()
}
