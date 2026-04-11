//! Configuration for the faithful Ctm CTM (single neuron pool).

use serde::{Deserialize, Serialize};

/// Configuration matching the Ctm AI Continuous Thought Machine.
///
/// One neuron pool of `d_model` neurons. One U-Net synapse. One NLM.
/// Two sync readouts (action for attention, output for predictions).
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
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
