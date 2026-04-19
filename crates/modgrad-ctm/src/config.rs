//! Configuration for the faithful Ctm CTM (single neuron pool).

use serde::{Deserialize, Serialize};
use wincode_derive::{SchemaRead, SchemaWrite};

/// Configuration matching the Ctm AI Continuous Thought Machine.
///
/// One neuron pool of `d_model` neurons. One U-Net synapse. One NLM.
/// Two sync readouts (action for attention, output for predictions).
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct CtmConfig {
    /// Number of internal ticks (T in paper).
    pub iterations: usize,
    /// Neuron pool size (D in paper).
    pub d_model: usize,
    /// Projected input feature dimension.
    pub d_input: usize,
    /// Number of attention heads.
    pub heads: usize,
    /// Neurons for output synchronization (D_out).
    pub n_synch_out: usize,
    /// Neurons for action/attention synchronization (D_action).
    pub n_synch_action: usize,
    /// U-Net synapse depth. 1 = MLP, >1 = U-Net with depth-1 down/up blocks.
    pub synapse_depth: usize,
    /// NLM history length (M in paper).
    pub memory_length: usize,
    /// Deep NLMs: true = 2-stage with hidden dim, false = single linear.
    pub deep_nlms: bool,
    /// NLM hidden dimension (H). Only used when deep_nlms = true.
    pub memory_hidden_dims: usize,
    /// Final output dimension (number of classes, vocab size, etc.).
    pub out_dims: usize,
    /// Self-to-self sync pairs for random-pairing.
    pub n_random_pairing_self: usize,
    /// U-Net minimum bottleneck width.
    pub min_width: usize,
    /// How the tick loop decides when to stop thinking.
    #[serde(default)]
    pub exit_strategy: ExitStrategy,
}

/// Tick-loop exit strategy. Exactly one mechanism — no overlap, no ambiguity.
///
/// Serde uses the *default externally-tagged* representation — bincode-
/// compatible. An earlier version had `#[serde(tag = "kind")]` for
/// human-readable JSON, but that requires `deserialize_any` which
/// bincode doesn't support, breaking `.bin` round-trips of any type
/// transitively containing an ExitStrategy (RegionalWeights, CtmWeights,
/// CheckpointBundle<_, _>). The tagged representation was a
/// human-readable convenience that cost us binary persistence.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub enum ExitStrategy {
    /// Run all ticks unconditionally.
    None,
    /// Stop when entropy-derived certainty exceeds a fixed threshold.
    Certainty {
        /// 0.0–1.0. Higher = more thinking. Default 0.95.
        threshold: f32,
    },
    /// Learned exit gate (Ouro/LoopLM-style).
    /// A Linear(sync_out → 1 → sigmoid) produces per-tick halting probability.
    /// Training uses entropy-regularized weighted loss across all ticks.
    /// Inference exits when cumulative exit probability exceeds `threshold`.
    AdaptiveGate {
        /// KL coefficient (β). Larger = more exploration across tick depths.
        /// Ouro uses 0.1 early, 0.05 later.
        beta: f32,
        /// CDF threshold for inference early-exit. 0.5 = exit when majority
        /// probability reached. 0.9 = keep going unless very confident.
        threshold: f32,
    },
}

impl Default for ExitStrategy {
    fn default() -> Self { ExitStrategy::None }
}

impl ExitStrategy {
    /// Whether a learned exit gate should be allocated in weights.
    pub fn has_gate(&self) -> bool {
        matches!(self, ExitStrategy::AdaptiveGate { .. })
    }

    /// Beta coefficient, if adaptive.
    pub fn beta(&self) -> f32 {
        match self {
            ExitStrategy::AdaptiveGate { beta, .. } => *beta,
            _ => 0.0,
        }
    }

    /// Inference exit threshold, if any exit strategy is active.
    pub fn threshold(&self) -> Option<f32> {
        match self {
            ExitStrategy::None => Option::None,
            ExitStrategy::Certainty { threshold } => Some(*threshold),
            ExitStrategy::AdaptiveGate { threshold, .. } => Some(*threshold),
        }
    }
}

impl Default for CtmConfig {
    fn default() -> Self {
        Self {
            iterations: 8,
            d_model: 512,
            d_input: 128,
            heads: 4,
            n_synch_out: 512,
            n_synch_action: 512,
            synapse_depth: 4,
            memory_length: 25,
            deep_nlms: true,
            memory_hidden_dims: 4,
            out_dims: 10,
            n_random_pairing_self: 0,
            min_width: 16,
            exit_strategy: ExitStrategy::None,
        }
    }
}

impl CtmConfig {
    /// Sync representation size for random-pairing (= n_synch).
    pub fn synch_size_out(&self) -> usize { self.n_synch_out }
    pub fn synch_size_action(&self) -> usize { self.n_synch_action }

    /// Synapse input dim: concat(attn_out, activated_state) = d_input + d_model.
    pub fn synapse_in_dim(&self) -> usize { self.d_input + self.d_model }

    // ─── Region constructors for hierarchical CTM ──────────────
    //
    // Each region in a RegionalCtm is a full CTM instance.
    // These constructors create configs sized for each brain region.
    // The `d_input` param is the observation dim for this region,
    // determined by the inter-region connection synapse output.

    /// Create a region-sized CTM config.
    ///
    /// `d_input`: observation dimension (from inter-region synapse).
    /// `out_dims` is set to d_model — region output = its activated state.
    /// The global output projection is separate (not per-region).
    pub fn region(
        name: &str,
        d_model: usize,
        d_input: usize,
        memory_length: usize,
        deep_nlms: bool,
        iterations: usize,
        exit_strategy: ExitStrategy,
    ) -> Self {
        let synapse_depth = if d_model >= 32 { 3 } else { 1 };
        let min_width = (d_model / 4).max(4);
        let heads = if d_input >= 8 { (d_input / 8).max(1).min(8) } else { 1 };
        let memory_hidden = if deep_nlms { (memory_length / 2).max(2) } else { 0 };
        let _ = name; // used by caller for labeling

        Self {
            iterations,
            d_model,
            d_input,
            heads,
            n_synch_out: d_model,
            n_synch_action: d_model.min(d_input),
            synapse_depth,
            memory_length,
            deep_nlms,
            memory_hidden_dims: memory_hidden,
            out_dims: d_model, // region output = activated state
            n_random_pairing_self: 0,
            min_width,
            exit_strategy,
        }
    }

    // ─── Preset region configs ─────────────────────────────────
    //
    // Per-region beta values (from Jones × Darlow analysis):
    //   Peripheral (input, cerebellum, insula, motor): beta=0.05 — fast, exit early
    //   Deliberative (attention, output, basal ganglia): beta=0.1 — needs iteration
    //   Memory (hippocampus): beta=0.15 — longest deliberation, memory-dependent
    //
    // All thresholds set to 0.99 — never exit early during inference until validated.

    /// Input region: wide, shallow NLM, short memory, fast. (β=0.05)
    pub fn input_region(d_input: usize, ticks: usize) -> Self {
        Self::region("input", 64, d_input, 4, false, ticks,
            ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 })
    }

    /// Attention region: medium, deep NLM, longer memory. (β=0.1)
    pub fn attention_region(d_input: usize, ticks: usize) -> Self {
        Self::region("attention", 64, d_input, 8, true, ticks,
            ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 })
    }

    /// Output region: medium, deep NLM, longest memory (evidence accumulation). (β=0.1)
    pub fn output_region(d_input: usize, ticks: usize) -> Self {
        Self::region("output", 64, d_input, 16, true, ticks,
            ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 })
    }

    /// Motor region: small, shallow, short memory, decisive. (β=0.05)
    pub fn motor_region(d_input: usize, ticks: usize) -> Self {
        Self::region("motor", 64, d_input, 4, false, ticks,
            ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 })
    }

    /// Cerebellum: tiny, fast forward model. (β=0.05)
    pub fn cerebellum_region(d_input: usize, ticks: usize) -> Self {
        Self::region("cerebellum", 8, d_input, 4, false, ticks,
            ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 })
    }

    /// Basal ganglia: tiny, action selection. (β=0.1)
    pub fn basal_ganglia_region(d_input: usize, ticks: usize) -> Self {
        Self::region("basal_ganglia", 8, d_input, 8, false, ticks,
            ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 })
    }

    /// Insula: tiny, interoceptive state. (β=0.05)
    pub fn insula_region(d_input: usize, ticks: usize) -> Self {
        Self::region("insula", 8, d_input, 4, false, ticks,
            ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 })
    }

    /// Hippocampus: tiny but deep NLM, long memory for pattern completion. (β=0.15)
    pub fn hippocampus_region(d_input: usize, ticks: usize) -> Self {
        Self::region("hippocampus", 8, d_input, 16, true, ticks,
            ExitStrategy::AdaptiveGate { beta: 0.15, threshold: 0.99 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CtmConfig::default ─────────────────────────────────────

    #[test]
    fn default_config_has_sane_values() {
        let c = CtmConfig::default();
        assert_eq!(c.iterations, 8);
        assert_eq!(c.d_model, 512);
        assert_eq!(c.d_input, 128);
        assert_eq!(c.heads, 4);
        assert_eq!(c.memory_length, 25);
        assert_eq!(c.out_dims, 10);
        assert!(c.deep_nlms);
        assert!(matches!(c.exit_strategy, ExitStrategy::None));
    }

    #[test]
    fn default_config_synapse_in_dim() {
        let c = CtmConfig::default();
        assert_eq!(c.synapse_in_dim(), c.d_input + c.d_model);
    }

    // ── CtmConfig::region ──────────────────────────────────────

    #[test]
    fn region_sets_d_model_and_d_input() {
        let c = CtmConfig::region("test", 64, 32, 8, true, 4, ExitStrategy::None);
        assert_eq!(c.d_model, 64);
        assert_eq!(c.d_input, 32);
        assert_eq!(c.memory_length, 8);
        assert_eq!(c.iterations, 4);
        assert!(c.deep_nlms);
    }

    #[test]
    fn region_out_dims_equals_d_model() {
        let c = CtmConfig::region("test", 128, 64, 8, false, 4, ExitStrategy::None);
        assert_eq!(c.out_dims, c.d_model);
    }

    #[test]
    fn region_synapse_depth_scales_with_d_model() {
        // d_model >= 32 => depth 3
        let big = CtmConfig::region("big", 64, 16, 4, false, 4, ExitStrategy::None);
        assert_eq!(big.synapse_depth, 3);

        // d_model < 32 => depth 1
        let small = CtmConfig::region("small", 16, 8, 4, false, 4, ExitStrategy::None);
        assert_eq!(small.synapse_depth, 1);
    }

    #[test]
    fn region_heads_clamp() {
        // d_input=64 => heads = 64/8 = 8 (clamped to max 8)
        let c = CtmConfig::region("test", 64, 64, 4, false, 4, ExitStrategy::None);
        assert_eq!(c.heads, 8);

        // d_input=4 => heads = 1 (below 8 threshold)
        let c2 = CtmConfig::region("test", 64, 4, 4, false, 4, ExitStrategy::None);
        assert_eq!(c2.heads, 1);
    }

    #[test]
    fn region_deep_nlm_sets_memory_hidden() {
        let c = CtmConfig::region("test", 64, 32, 8, true, 4, ExitStrategy::None);
        assert_eq!(c.memory_hidden_dims, 4); // memory_length/2

        let c2 = CtmConfig::region("test", 64, 32, 8, false, 4, ExitStrategy::None);
        assert_eq!(c2.memory_hidden_dims, 0); // no deep NLMs
    }

    #[test]
    fn region_synch_action_capped_at_d_input() {
        let c = CtmConfig::region("test", 64, 16, 4, false, 4, ExitStrategy::None);
        assert_eq!(c.n_synch_action, 16); // min(d_model=64, d_input=16)
    }

    // ── Preset region constructors ─────────────────────────────

    #[test]
    fn cerebellum_region_is_small() {
        let c = CtmConfig::cerebellum_region(16, 4);
        assert_eq!(c.d_model, 8);
        assert!(!c.deep_nlms);
        assert_eq!(c.memory_length, 4);
        assert!(c.exit_strategy.has_gate());
    }

    #[test]
    fn hippocampus_region_has_long_memory() {
        let c = CtmConfig::hippocampus_region(16, 4);
        assert_eq!(c.memory_length, 16);
        assert!(c.deep_nlms);
        assert!(c.exit_strategy.has_gate());
    }

    #[test]
    fn input_region_is_fast() {
        let c = CtmConfig::input_region(32, 4);
        assert!(!c.deep_nlms);
        assert_eq!(c.memory_length, 4);
    }

    // ── ExitStrategy ───────────────────────────────────────────

    #[test]
    fn exit_strategy_default_is_none() {
        let e = ExitStrategy::default();
        assert!(matches!(e, ExitStrategy::None));
        assert!(!e.has_gate());
        assert_eq!(e.threshold(), None);
        assert_eq!(e.beta(), 0.0);
    }

    #[test]
    fn certainty_strategy_has_threshold() {
        let e = ExitStrategy::Certainty { threshold: 0.95 };
        assert!(!e.has_gate());
        assert_eq!(e.threshold(), Some(0.95));
        assert_eq!(e.beta(), 0.0);
    }

    #[test]
    fn adaptive_gate_has_gate_and_params() {
        let e = ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.5 };
        assert!(e.has_gate());
        assert_eq!(e.beta(), 0.1);
        assert_eq!(e.threshold(), Some(0.5));
    }

    // ── Serde round-trip ───────────────────────────────────────

    #[test]
    fn config_serde_roundtrip() {
        let c = CtmConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let c2: CtmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.d_model, c.d_model);
        assert_eq!(c2.iterations, c.iterations);
        assert_eq!(c2.memory_length, c.memory_length);
    }

    #[test]
    fn exit_strategy_serde_roundtrip() {
        let e = ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 };
        let json = serde_json::to_string(&e).unwrap();
        let e2: ExitStrategy = serde_json::from_str(&json).unwrap();
        assert!(e2.has_gate());
        assert_eq!(e2.beta(), 0.1);
    }
}
