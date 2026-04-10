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
    /// Enable certainty-based early exit (variable thinking length).
    /// When true, the tick loop exits early if certainty exceeds threshold.
    /// The brain decides when it's confident enough to stop thinking.
    #[serde(default)]
    pub early_exit: bool,
    /// Certainty threshold for early exit (0.0 to 1.0).
    /// Higher = more thinking before stopping. 0.95 = stop when 95% certain.
    #[serde(default = "default_certainty_threshold")]
    pub certainty_threshold: f32,
}

fn default_certainty_threshold() -> f32 { 0.95 }

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
            early_exit: false,
            certainty_threshold: 0.95,
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
            early_exit: false,
            certainty_threshold: 0.95,
        }
    }

    // ─── Preset region configs ─────────────────────────────────

    /// Input region: wide, shallow NLM, short memory, fast.
    pub fn input_region(d_input: usize, ticks: usize) -> Self {
        Self::region("input", 64, d_input, 4, false, ticks)
    }

    /// Attention region: medium, deep NLM, longer memory.
    pub fn attention_region(d_input: usize, ticks: usize) -> Self {
        Self::region("attention", 64, d_input, 8, true, ticks)
    }

    /// Output region: medium, deep NLM, longest memory (evidence accumulation).
    pub fn output_region(d_input: usize, ticks: usize) -> Self {
        Self::region("output", 64, d_input, 16, true, ticks)
    }

    /// Motor region: small, shallow, short memory, decisive.
    pub fn motor_region(d_input: usize, ticks: usize) -> Self {
        Self::region("motor", 64, d_input, 4, false, ticks)
    }

    /// Cerebellum: tiny, fast forward model.
    pub fn cerebellum_region(d_input: usize, ticks: usize) -> Self {
        Self::region("cerebellum", 8, d_input, 4, false, ticks)
    }

    /// Basal ganglia: tiny, action selection.
    pub fn basal_ganglia_region(d_input: usize, ticks: usize) -> Self {
        Self::region("basal_ganglia", 8, d_input, 8, false, ticks)
    }

    /// Insula: tiny, interoceptive state.
    pub fn insula_region(d_input: usize, ticks: usize) -> Self {
        Self::region("insula", 8, d_input, 4, false, ticks)
    }

    /// Hippocampus: tiny but deep NLM, long memory for pattern completion.
    pub fn hippocampus_region(d_input: usize, ticks: usize) -> Self {
        Self::region("hippocampus", 8, d_input, 16, true, ticks)
    }
}
