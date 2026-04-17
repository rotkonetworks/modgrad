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

    /// Episodic memory: persistent KV buffer for long-range context.
    /// When present, MHA attends over episodic entries in addition to
    /// the current observation. Entries accumulate across forward calls
    /// with bounded memory via hierarchical compression.
    pub episodic: Option<modgrad_compute::kv_buffer::EpisodicMemory>,
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
            episodic: None,
        }
    }

    /// Create state with episodic memory enabled.
    /// `short`, `mid`, `long`: capacity in entries per tier.
    pub fn with_episodic(w: &CtmWeights, short: usize, mid: usize, long: usize) -> Self {
        let mut s = Self::new(w);
        s.episodic = Some(modgrad_compute::kv_buffer::EpisodicMemory::new(
            w.config.d_input, short, mid, long,
        ));
        s
    }
}

fn random_indices(max: usize, count: usize, rng: &mut SimpleRng) -> Vec<usize> {
    (0..count).map(|_| (rng.next_u64() as usize) % max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CtmConfig, ExitStrategy};

    fn small_config() -> CtmConfig {
        CtmConfig {
            iterations: 4,
            d_model: 16,
            d_input: 8,
            heads: 2,
            n_synch_out: 16,
            n_synch_action: 8,
            synapse_depth: 1,
            memory_length: 4,
            deep_nlms: false,
            memory_hidden_dims: 0,
            out_dims: 3,
            n_random_pairing_self: 0,
            min_width: 4,
            exit_strategy: ExitStrategy::None,
        }
    }

    fn small_config_deep() -> CtmConfig {
        CtmConfig {
            deep_nlms: true,
            memory_hidden_dims: 2,
            ..small_config()
        }
    }

    // ── CtmWeights::new ────────────────────────────────────────

    #[test]
    fn weights_new_creates_correct_start_state_sizes() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        assert_eq!(w.start_activated.len(), cfg.d_model);
        assert_eq!(w.start_trace.len(), cfg.d_model * cfg.memory_length);
    }

    #[test]
    fn weights_new_kv_proj_dimensions() {
        let raw_input = 10;
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), raw_input);
        assert_eq!(w.kv_proj.in_dim, raw_input);
        assert_eq!(w.kv_proj.out_dim, cfg.d_input);
        assert_eq!(w.kv_ln_gamma.len(), cfg.d_input);
        assert_eq!(w.kv_ln_beta.len(), cfg.d_input);
    }

    #[test]
    fn weights_new_mha_dimensions() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        assert_eq!(w.mha_in_proj.in_dim, cfg.d_input);
        assert_eq!(w.mha_in_proj.out_dim, 3 * cfg.d_input);
        assert_eq!(w.mha_out_proj.in_dim, cfg.d_input);
        assert_eq!(w.mha_out_proj.out_dim, cfg.d_input);
    }

    #[test]
    fn weights_new_output_proj_dimensions() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        assert_eq!(w.output_proj.in_dim, cfg.n_synch_out);
        assert_eq!(w.output_proj.out_dim, cfg.out_dims);
    }

    #[test]
    fn weights_new_sync_indices_correct_length() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        assert_eq!(w.sync_out_left.len(), cfg.n_synch_out);
        assert_eq!(w.sync_out_right.len(), cfg.n_synch_out);
        assert_eq!(w.sync_action_left.len(), cfg.n_synch_action);
        assert_eq!(w.sync_action_right.len(), cfg.n_synch_action);
    }

    #[test]
    fn weights_new_sync_indices_in_range() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        for &idx in &w.sync_out_left {
            assert!(idx < cfg.d_model, "sync_out_left index {idx} >= d_model {}", cfg.d_model);
        }
        for &idx in &w.sync_action_left {
            assert!(idx < cfg.d_model);
        }
    }

    #[test]
    fn weights_new_decay_params() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        assert_eq!(w.decay_params_out.len(), cfg.n_synch_out);
        assert_eq!(w.decay_params_action.len(), cfg.n_synch_action);
        assert!(w.decay_params_out.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn weights_no_exit_gate_when_none_strategy() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);
        assert!(w.exit_gate.is_none());
    }

    #[test]
    fn weights_has_exit_gate_when_adaptive() {
        let mut cfg = small_config();
        cfg.exit_strategy = ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.5 };
        let w = CtmWeights::new(cfg.clone(), 10);
        assert!(w.exit_gate.is_some());
        let gate = w.exit_gate.as_ref().unwrap();
        assert_eq!(gate.in_dim, cfg.n_synch_out);
        assert_eq!(gate.out_dim, 1);
    }

    // ── Deep NLM stages ────────────────────────────────────────

    #[test]
    fn weights_shallow_nlm_no_stage2() {
        let cfg = small_config(); // deep_nlms = false
        let w = CtmWeights::new(cfg, 10);
        assert!(w.nlm_stage2.is_none());
    }

    #[test]
    fn weights_deep_nlm_has_stage2() {
        let cfg = small_config_deep();
        let w = CtmWeights::new(cfg, 10);
        assert!(w.nlm_stage2.is_some());
    }

    // ── n_params ───────────────────────────────────────────────

    #[test]
    fn n_params_positive_and_consistent() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);
        let n = w.n_params();
        assert!(n > 0, "n_params should be positive");

        // Create the same config again — deterministic, so should match
        let cfg2 = small_config();
        let w2 = CtmWeights::new(cfg2, 10);
        assert_eq!(w2.n_params(), n, "n_params should be deterministic");
    }

    #[test]
    fn n_params_deep_greater_than_shallow() {
        let w_shallow = CtmWeights::new(small_config(), 10);
        let w_deep = CtmWeights::new(small_config_deep(), 10);
        assert!(
            w_deep.n_params() > w_shallow.n_params(),
            "deep NLM should have more params: {} vs {}",
            w_deep.n_params(), w_shallow.n_params()
        );
    }

    #[test]
    fn n_params_includes_gate_when_present() {
        let mut cfg_no_gate = small_config();
        cfg_no_gate.exit_strategy = ExitStrategy::None;
        let mut cfg_gate = small_config();
        cfg_gate.exit_strategy = ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.5 };

        let w_no = CtmWeights::new(cfg_no_gate, 10);
        let w_yes = CtmWeights::new(cfg_gate, 10);

        assert!(
            w_yes.n_params() > w_no.n_params(),
            "gate should add params: {} vs {}",
            w_yes.n_params(), w_no.n_params()
        );
    }

    // ── CtmState::new ──────────────────────────────────────────

    #[test]
    fn state_new_dimensions() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        let s = CtmState::new(&w);

        assert_eq!(s.activated.len(), cfg.d_model);
        assert_eq!(s.trace.len(), cfg.d_model * cfg.memory_length);
        assert_eq!(s.alpha_out.len(), cfg.n_synch_out);
        assert_eq!(s.beta_out.len(), cfg.n_synch_out);
        assert!(s.alpha_action.is_none());
        assert!(s.beta_action.is_none());
        assert!(s.episodic.is_none());
    }

    #[test]
    fn state_new_copies_start_state() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);
        let s = CtmState::new(&w);

        assert_eq!(s.activated, w.start_activated);
        assert_eq!(s.trace, w.start_trace);
    }

    #[test]
    fn state_new_sync_accumulators_zeroed() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);
        let s = CtmState::new(&w);

        assert!(s.alpha_out.iter().all(|&v| v == 0.0));
        assert!(s.beta_out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn state_with_episodic_creates_memory() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);
        let s = CtmState::with_episodic(&w, 8, 4, 2);

        assert!(s.episodic.is_some());
    }

    // ── Save/load round-trip ───────────────────────────────────

    #[test]
    fn weights_save_load_roundtrip() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);

        let dir = std::env::temp_dir().join("modgrad_ctm_test_weights");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_weights.json");
        let path_str = path.to_str().unwrap();

        w.save(path_str).unwrap();
        let w2 = CtmWeights::load(path_str).unwrap();

        assert_eq!(w2.n_params(), w.n_params());
        assert_eq!(w2.start_activated, w.start_activated);
        assert_eq!(w2.start_trace, w.start_trace);
        assert_eq!(w2.config.d_model, w.config.d_model);

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Self-pairing ───────────────────────────────────────────

    #[test]
    fn self_pairing_matches_left_right() {
        let mut cfg = small_config();
        cfg.n_random_pairing_self = 4;
        let w = CtmWeights::new(cfg, 10);

        for i in 0..4 {
            assert_eq!(
                w.sync_out_left[i], w.sync_out_right[i],
                "self-pairing mismatch at index {i}"
            );
        }
    }
}
