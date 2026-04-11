//! CTM Graph: compose N CTMs into a directed graph with BPTT.
//!
//! This is the core composition layer of the modgrad SDK. You decide:
//! - How many nodes (regions) and their sizes
//! - How they connect (directed graph)
//! - Vocabulary / output dimension
//! - Training: full BPTT through all nodes and connections
//!
//! Also provides:
//! - Unified token space (text + image + audio + video + actions)
//! - NeuralComputer: interactive generate→observe→generate loop
//! - AdamW optimizer for the graph
//! - Save/load via modgrad-persist
//!
//! isis uses this with 8 brain-inspired regions. minictm uses 1-4.
//! You can use it however you want.

use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use modgrad_compute::neuron::{Linear, SimpleRng};
use crate::config::CtmConfig;
use crate::weights::{CtmWeights, CtmState};
use crate::forward::ctm_forward;
use crate::train::{Ctm, CtmCache, CtmGradients, RegionBackwardResult, backward_from_activated};
use modgrad_traits::{Brain, TokenInput};

// ─── Unified token space ──────────────────────────────────

// ── Token layout ──────────────────────────────────────────
//
//   0..255      byte tokens (text)
//   256..263    special delimiters
//   264..4359   image VQ codes (4096)
//   4360..8455  audio VQ codes (4096)
//   8456..8855  timestamp tokens (400 slots → 0.00s..199.50s at 0.5s resolution)
//
// Total: 8856

pub const VOCAB_TEXT: usize = 256;

// Delimiters
pub const TOKEN_IMG_START: usize = 256;
pub const TOKEN_IMG_END: usize = 257;
pub const TOKEN_AUD_START: usize = 258;
pub const TOKEN_AUD_END: usize = 259;
pub const TOKEN_VID_START: usize = 260;
pub const TOKEN_VID_END: usize = 261;
pub const TOKEN_PAD: usize = 262;
pub const TOKEN_SPECIAL_END: usize = 264; // next power-of-2 aligned

// Image VQ
pub const TOKEN_IMG_OFFSET: usize = 264;
pub const TOKEN_IMG_CODES: usize = 4096;

// Audio VQ
pub const TOKEN_AUD_OFFSET: usize = TOKEN_IMG_OFFSET + TOKEN_IMG_CODES; // 4360
pub const TOKEN_AUD_CODES: usize = 4096;

// Timestamps: 0.5s resolution, 0..199.5s → 400 tokens
pub const TOKEN_TS_OFFSET: usize = TOKEN_AUD_OFFSET + TOKEN_AUD_CODES; // 8456
pub const TOKEN_TS_COUNT: usize = 400;
pub const TOKEN_TS_RESOLUTION_MS: usize = 500; // 0.5s per token

// Actions: mouse + keyboard events for NC interactive mode
// Layout:
//   8856..8856   mouse_move (followed by 2 coordinate tokens)
//   8857..8857   left_click
//   8858..8858   right_click
//   8859..8859   double_click
//   8860..8860   scroll_up
//   8861..8861   scroll_down
//   8862..8862   drag_start
//   8863..8863   drag_end
//   8864..8864   key_enter
//   8865..8865   key_backspace
//   8866..8866   key_tab
//   8867..8867   key_escape
//   8868..8868   key_up
//   8869..8869   key_down
//   8870..8870   key_left
//   8871..8871   key_right
//   8872..8872   key_ctrl (modifier prefix)
//   8873..8873   key_alt (modifier prefix)
//   8874..8874   key_shift (modifier prefix)
//   8875..8875   key_type (followed by byte tokens for the typed text)
//   8876..9131   coordinate tokens (256 values: 0..255, quantized x or y position)
//   9132..9132   action_start delimiter
//   9133..9133   action_end delimiter
pub const TOKEN_ACT_OFFSET: usize = TOKEN_TS_OFFSET + TOKEN_TS_COUNT; // 8856
pub const ACT_MOUSE_MOVE: usize = TOKEN_ACT_OFFSET;
pub const ACT_LEFT_CLICK: usize = TOKEN_ACT_OFFSET + 1;
pub const ACT_RIGHT_CLICK: usize = TOKEN_ACT_OFFSET + 2;
pub const ACT_DOUBLE_CLICK: usize = TOKEN_ACT_OFFSET + 3;
pub const ACT_SCROLL_UP: usize = TOKEN_ACT_OFFSET + 4;
pub const ACT_SCROLL_DOWN: usize = TOKEN_ACT_OFFSET + 5;
pub const ACT_DRAG_START: usize = TOKEN_ACT_OFFSET + 6;
pub const ACT_DRAG_END: usize = TOKEN_ACT_OFFSET + 7;
pub const ACT_KEY_ENTER: usize = TOKEN_ACT_OFFSET + 8;
pub const ACT_KEY_BACKSPACE: usize = TOKEN_ACT_OFFSET + 9;
pub const ACT_KEY_TAB: usize = TOKEN_ACT_OFFSET + 10;
pub const ACT_KEY_ESCAPE: usize = TOKEN_ACT_OFFSET + 11;
pub const ACT_KEY_UP: usize = TOKEN_ACT_OFFSET + 12;
pub const ACT_KEY_DOWN: usize = TOKEN_ACT_OFFSET + 13;
pub const ACT_KEY_LEFT: usize = TOKEN_ACT_OFFSET + 14;
pub const ACT_KEY_RIGHT: usize = TOKEN_ACT_OFFSET + 15;
pub const ACT_KEY_CTRL: usize = TOKEN_ACT_OFFSET + 16;
pub const ACT_KEY_ALT: usize = TOKEN_ACT_OFFSET + 17;
pub const ACT_KEY_SHIFT: usize = TOKEN_ACT_OFFSET + 18;
pub const ACT_KEY_TYPE: usize = TOKEN_ACT_OFFSET + 19;
pub const TOKEN_COORD_OFFSET: usize = TOKEN_ACT_OFFSET + 20; // 8876
pub const TOKEN_COORD_COUNT: usize = 256;
pub const ACT_START: usize = TOKEN_COORD_OFFSET + TOKEN_COORD_COUNT; // 9132
pub const ACT_END: usize = ACT_START + 1; // 9133

pub const VOCAB_MULTIMODAL: usize = ACT_END + 1; // 9134

// ─── Configuration ─────────────────────────────────────────

/// Describes one inter-region connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Source region indices — their activated states are concatenated.
    pub from: Vec<usize>,
    /// Target region index.
    pub to: usize,
    /// If true, the raw observation is concatenated to the source activations
    /// before the synapse projection. This is how external input enters the graph.
    #[serde(default)]
    pub receives_observation: bool,
}

/// Toggleable auxiliary losses inspired by neuroscience.
/// Each adds a gradient signal to specific regions on top of the main BPTT loss.
/// All default to off — enable to test if they help.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuxLossConfig {
    /// Cerebellar prediction error: cerebellum region predicts next region outputs.
    /// Loss = MSE between predicted and actual next-tick activations.
    /// Teaches the cerebellum to be an internal forward model.
    pub cerebellar_prediction: bool,
    /// Hippocampal contrastive: pull same-context states together, push different apart.
    /// Encourages the hippocampus to form distinct episodic representations.
    pub hippocampal_contrastive: bool,
    /// Basal ganglia temporal difference: predict cumulative future loss.
    /// Teaches BG to estimate value (how well the model will do from here).
    pub bg_temporal_difference: bool,
    /// Weight for auxiliary losses relative to main loss (0.1 = 10% of main loss).
    pub aux_weight: f32,
}

impl Default for AuxLossConfig {
    fn default() -> Self {
        Self {
            cerebellar_prediction: false,
            hippocampal_contrastive: false,
            bg_temporal_difference: false,
            aux_weight: 0.1,
        }
    }
}

// ─── Learned inter-region router (MoS-inspired) ──────────

/// Configuration for the learned thalamic router.
/// When enabled, replaces fixed connection topology with dynamic,
/// tick-conditioned, sparse routing between regions.
///
/// Inspired by Mixture of States (MoS): each destination region
/// selects its top-k source regions per tick, weighted by a learned
/// affinity conditioned on (tick, global_sync).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Common routing dimension — all regions project to/from this.
    pub d_route: usize,
    /// Tick embedding dimension.
    pub tick_embed_dim: usize,
    /// Top-k sparsity: each destination selects k source regions.
    pub k: usize,
    /// ε-greedy exploration rate during training (0.0 = off, 0.05 = 5% random).
    pub epsilon: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self { d_route: 32, tick_embed_dim: 16, k: 3, epsilon: 0.05 }
    }
}

/// Learned thalamic router: weights for dynamic inter-region routing.
///
/// Per tick: (global_sync, tick_embed) → [n_regions × n_regions] logits
/// → softmax per destination → top-k selection → weighted sum of
/// projected source outputs → destination input.
///
/// Total params ≈ n×d_model×d_route + n×d_route×d_input + sync×n² + ticks×t_dim
/// For 8 regions: ~40K params (<0.3% of model). True zero-cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalRouter {
    pub config: RouterConfig,
    pub n_regions: usize,

    /// Project each region's activated output to routing space.
    /// to_route[i]: Linear(d_model_i, d_route)
    pub to_route: Vec<Linear>,

    /// Project routed input to each destination region's d_input.
    /// from_route[j]: Linear(d_route, d_input_j)
    pub from_route: Vec<Linear>,

    /// Learned tick embeddings: [max_ticks × tick_embed_dim].
    pub tick_embed: Vec<f32>,
    pub max_ticks: usize,

    /// Route projection: (n_sync*2 + tick_embed_dim) → (n_regions × n_regions).
    pub route_proj: Linear,
}

/// Cached forward state for router backward.
pub struct RouterCache {
    /// Input to route_proj.
    pub proj_input: Vec<f32>,
    /// Raw logits from route_proj [n_regions × n_regions].
    pub logits: Vec<f32>,
    /// Softmax weights [n_regions × n_regions] (column-wise softmax).
    pub weights: Vec<f32>,
    /// Selected indices per destination: [n_regions][k].
    pub selected: Vec<Vec<usize>>,
    /// Per-source projected outputs: [n_regions][d_route].
    pub projected_sources: Vec<Vec<f32>>,
}

/// Gradients for the router.
pub struct RouterGradients {
    pub to_route_dw: Vec<Vec<f32>>,
    pub to_route_db: Vec<Vec<f32>>,
    pub from_route_dw: Vec<Vec<f32>>,
    pub from_route_db: Vec<Vec<f32>>,
    pub tick_embed_grad: Vec<f32>,
    pub route_proj_dw: Vec<f32>,
    pub route_proj_db: Vec<f32>,
}

impl RegionalRouter {
    pub fn new(
        config: RouterConfig,
        region_d_models: &[usize],
        region_d_inputs: &[usize],
        n_sync: usize,
        max_ticks: usize,
    ) -> Self {
        let n = region_d_models.len();
        let d = config.d_route;
        let t = config.tick_embed_dim;

        let to_route: Vec<Linear> = region_d_models.iter()
            .map(|&dm| Linear::new(dm, d))
            .collect();
        let from_route: Vec<Linear> = region_d_inputs.iter()
            .map(|&di| Linear::new(d, di))
            .collect();

        // Learned tick embeddings (small table)
        let mut rng = SimpleRng::new(0xDEAD_BEEF);
        let scale = (1.0 / t as f32).sqrt();
        let tick_embed: Vec<f32> = (0..max_ticks * t)
            .map(|_| rng.next_normal() * scale)
            .collect();

        // Router MLP: [n_sync*2 + tick_embed_dim] → [n * n]
        let route_proj = Linear::new(n_sync * 2 + t, n * n);

        Self {
            config, n_regions: n,
            to_route, from_route,
            tick_embed, max_ticks,
            route_proj,
        }
    }

    /// Forward: compute routed inputs for each destination region.
    ///
    /// Returns (routed_inputs, cache):
    /// - routed_inputs[j] = projected weighted sum of selected source regions for destination j
    /// - cache: everything needed for backward
    pub fn forward(
        &self, tick: usize, global_sync: &[f32],
        region_outputs: &[Vec<f32>], training: bool,
    ) -> (Vec<Vec<f32>>, RouterCache) {
        let n = self.n_regions;
        let d = self.config.d_route;
        let k = self.config.k.min(n);

        // 1. Get tick embedding (clamp to max)
        let t_idx = tick.min(self.max_ticks - 1);
        let t = self.config.tick_embed_dim;
        let t_emb = &self.tick_embed[t_idx * t..(t_idx + 1) * t];

        // 2. Build router input: [global_sync, tick_embed]
        let mut proj_input = Vec::with_capacity(global_sync.len() + t);
        proj_input.extend_from_slice(global_sync);
        proj_input.extend_from_slice(t_emb);

        // 3. Predict routing logits [n × n]
        let logits = self.route_proj.forward(&proj_input);

        // 4. Project all source regions to routing space
        let projected_sources: Vec<Vec<f32>> = (0..n)
            .map(|i| self.to_route[i].forward(&region_outputs[i]))
            .collect();

        // 5. Per-destination: softmax over sources, top-k select, weighted sum
        let mut weights = vec![0.0f32; n * n];
        let mut selected = Vec::with_capacity(n);
        let mut routed_inputs = Vec::with_capacity(n);

        let mut rng = if training {
            Some(SimpleRng::new(tick as u64 ^ 0xCAFE))
        } else {
            None
        };

        for j in 0..n {
            // Softmax over source axis for destination j
            let col: Vec<f32> = (0..n).map(|i| logits[i * n + j]).collect();
            let max_val = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = col.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let w: Vec<f32> = exp.iter().map(|&e| e / sum).collect();
            for i in 0..n { weights[i * n + j] = w[i]; }

            // Top-k or ε-greedy selection
            let sel = if training && rng.as_mut().map_or(false, |r| r.next_f32() < self.config.epsilon) {
                // Random k indices
                let mut indices: Vec<usize> = (0..n).collect();
                let r = rng.as_mut().unwrap();
                for idx in 0..k {
                    let swap = idx + (r.next_u64() as usize % (n - idx));
                    indices.swap(idx, swap);
                }
                indices[..k].to_vec()
            } else {
                // Top-k by weight
                let mut indexed: Vec<(usize, f32)> = w.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed[..k].iter().map(|(i, _)| *i).collect()
            };

            // Weighted sum of selected projected sources
            let mut routed = vec![0.0f32; d];
            for &i in &sel {
                let wi = weights[i * n + j];
                for (r, p) in routed.iter_mut().zip(projected_sources[i].iter()) {
                    *r += wi * p;
                }
            }

            // Project to destination input dimension
            let dest_input = self.from_route[j].forward(&routed);
            selected.push(sel);
            routed_inputs.push(dest_input);
        }

        let cache = RouterCache {
            proj_input, logits, weights, selected, projected_sources,
        };
        (routed_inputs, cache)
    }

    /// Parameter count.
    pub fn n_params(&self) -> usize {
        let mut n = self.tick_embed.len();
        n += self.route_proj.weight.len() + self.route_proj.bias.len();
        for l in &self.to_route { n += l.weight.len() + l.bias.len(); }
        for l in &self.from_route { n += l.weight.len() + l.bias.len(); }
        n
    }
}

impl RouterGradients {
    pub fn zeros(router: &RegionalRouter) -> Self {
        Self {
            to_route_dw: router.to_route.iter().map(|l| vec![0.0; l.weight.len()]).collect(),
            to_route_db: router.to_route.iter().map(|l| vec![0.0; l.bias.len()]).collect(),
            from_route_dw: router.from_route.iter().map(|l| vec![0.0; l.weight.len()]).collect(),
            from_route_db: router.from_route.iter().map(|l| vec![0.0; l.bias.len()]).collect(),
            tick_embed_grad: vec![0.0; router.tick_embed.len()],
            route_proj_dw: vec![0.0; router.route_proj.weight.len()],
            route_proj_db: vec![0.0; router.route_proj.bias.len()],
        }
    }

    pub fn zero(&mut self) {
        for v in &mut self.to_route_dw { v.fill(0.0); }
        for v in &mut self.to_route_db { v.fill(0.0); }
        for v in &mut self.from_route_dw { v.fill(0.0); }
        for v in &mut self.from_route_db { v.fill(0.0); }
        self.tick_embed_grad.fill(0.0);
        self.route_proj_dw.fill(0.0);
        self.route_proj_db.fill(0.0);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalConfig {
    /// Per-region CTM configs.
    pub regions: Vec<CtmConfig>,
    /// Region names (for debugging/telemetry).
    pub region_names: Vec<String>,
    /// Connection topology.
    pub connections: Vec<Connection>,
    /// How many outer ticks to run.
    pub outer_ticks: usize,
    /// How the outer tick loop decides when to stop.
    /// Each region's inner ticks have their own exit_strategy independently.
    #[serde(default)]
    pub exit_strategy: crate::config::ExitStrategy,
    /// Global sync pair count (spans all regions).
    pub n_global_sync: usize,
    /// Final output dimension (vocab size, num classes, etc.).
    pub out_dims: usize,
    /// Raw observation dimension (input to the system).
    pub raw_obs_dim: usize,
    /// Auxiliary bio-inspired losses (optional, toggleable).
    #[serde(default)]
    pub aux_losses: AuxLossConfig,
    /// Learned inter-region router (MoS-style). When Some, replaces
    /// fixed connection topology with dynamic, tick-conditioned routing.
    /// When None, uses fixed connections (backward compatible).
    #[serde(default)]
    pub router: Option<RouterConfig>,
}

impl RegionalConfig {
    /// Default 8-region brain topology.
    ///
    /// `obs_dim`: raw observation dimension (e.g. embed_dim from tokenizer).
    /// `out_dims`: output dimension (e.g. vocab_size).
    /// `ticks`: number of outer ticks (each region runs its own inner ticks).
    pub fn eight_region(obs_dim: usize, out_dims: usize, ticks: usize) -> Self {
        // Region indices
        const INPUT: usize = 0;
        const ATTENTION: usize = 1;
        const OUTPUT: usize = 2;
        const MOTOR: usize = 3;
        const CEREBELLUM: usize = 4;
        const BASAL_GANGLIA: usize = 5;
        const INSULA: usize = 6;
        const HIPPOCAMPUS: usize = 7;

        // Each region's d_input is determined by its incoming connections.
        // For now, we use obs_dim as the universal projection target.
        // The inter-region synapses project concat(sources) → region.d_input.
        let d = obs_dim;

        let regions = vec![
            CtmConfig::input_region(d, ticks),
            CtmConfig::attention_region(d, ticks),
            CtmConfig::output_region(d, ticks),
            CtmConfig::motor_region(d, ticks),
            CtmConfig::cerebellum_region(d, ticks),
            CtmConfig::basal_ganglia_region(d, ticks),
            CtmConfig::insula_region(d, ticks),
            CtmConfig::hippocampus_region(d, ticks),
        ];

        let names = vec![
            "input", "attention", "output", "motor",
            "cerebellum", "basal_ganglia", "insula", "hippocampus",
        ].into_iter().map(String::from).collect();

        // Connection graph: who feeds whom
        // Each connection's synapse will project concat(from sync_outs) → target d_input
        let connections = vec![
            // Cortical loop — input receives observation + motor feedback
            Connection { from: vec![MOTOR], to: INPUT, receives_observation: true },
            Connection { from: vec![INPUT], to: ATTENTION, receives_observation: false },
            Connection { from: vec![ATTENTION], to: OUTPUT, receives_observation: false },
            Connection { from: vec![OUTPUT], to: MOTOR, receives_observation: false },
            // Subcortical — cerebellum receives observation for prediction error
            Connection { from: vec![MOTOR], to: CEREBELLUM, receives_observation: true },
            Connection { from: vec![OUTPUT], to: BASAL_GANGLIA, receives_observation: false },
            // Interoceptive
            Connection { from: vec![HIPPOCAMPUS], to: INSULA, receives_observation: false },
            Connection { from: vec![INPUT, ATTENTION, OUTPUT, MOTOR], to: HIPPOCAMPUS, receives_observation: false },
        ];

        // Total activated neurons across all regions — for global sync
        let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();
        let n_global_sync = total_neurons.min(256);

        Self {
            regions,
            region_names: names,
            connections,
            outer_ticks: ticks,
            exit_strategy: crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 },
            n_global_sync,
            out_dims,
            raw_obs_dim: obs_dim,
            aux_losses: AuxLossConfig::default(),
            router: Some(RouterConfig::default()),
        }
    }

    /// 4-region cortical-only topology (CTM v2 style).
    pub fn four_region(obs_dim: usize, out_dims: usize, ticks: usize) -> Self {
        let d = obs_dim;
        let regions = vec![
            CtmConfig::input_region(d, ticks),
            CtmConfig::attention_region(d, ticks),
            CtmConfig::output_region(d, ticks),
            CtmConfig::motor_region(d, ticks),
        ];

        let names = vec!["input", "attention", "output", "motor"]
            .into_iter().map(String::from).collect();

        let connections = vec![
            Connection { from: vec![3], to: 0, receives_observation: true },  // motor + obs → input
            Connection { from: vec![0], to: 1, receives_observation: false }, // input → attention
            Connection { from: vec![1], to: 2, receives_observation: false }, // attention → output
            Connection { from: vec![2], to: 3, receives_observation: false }, // output → motor
        ];

        let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();

        Self {
            regions,
            region_names: names,
            connections,
            outer_ticks: ticks,
            exit_strategy: crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 },
            n_global_sync: total_neurons.min(256),
            out_dims,
            raw_obs_dim: obs_dim,
            aux_losses: AuxLossConfig::default(),
            router: None,
        }
    }

    /// Multimodal 8-region: text + image + audio tokens.
    /// Same topology as eight_region but with expanded vocab.
    pub fn eight_region_multimodal(embed_dim: usize, ticks: usize) -> Self {
        Self::eight_region(embed_dim, VOCAB_MULTIMODAL, ticks)
    }

    /// Multimodal 4-region: text + image + audio tokens.
    pub fn four_region_multimodal(embed_dim: usize, ticks: usize) -> Self {
        Self::four_region(embed_dim, VOCAB_MULTIMODAL, ticks)
    }
}

// ─── Weights ───────────────────────────────────────────────

/// All weights for the regional CTM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalWeights {
    pub config: RegionalConfig,

    /// Byte embedding table: [vocab_size × embed_dim].
    /// Maps token indices to dense vectors that feed into the CTM.
    pub embeddings: Vec<f32>,

    /// Per-region CTM weights.
    pub regions: Vec<CtmWeights>,

    /// Inter-region connection synapses.
    /// One Linear per connection: concat(source sync_outs) → target d_input.
    pub connection_synapses: Vec<Linear>,

    /// Observation projection: raw_obs → each region's d_input.
    /// Used for regions that receive external observation (input, cerebellum).
    pub obs_proj: Linear,

    /// Global sync: random pair indices into concatenated activations.
    pub global_sync_left: Vec<usize>,
    pub global_sync_right: Vec<usize>,
    pub global_decay: Vec<f32>,

    /// Output projection: global_sync → next byte logits (head 0).
    pub output_proj: Linear,

    /// Outer-level adaptive exit gate: global_sync → scalar → sigmoid.
    /// Present iff exit_strategy is AdaptiveGate.
    #[serde(default)]
    pub outer_exit_gate: Option<Linear>,

    /// Multi-byte prediction heads (EvaByte-style): predict bytes 2..N ahead.
    /// Head i predicts byte at position current+i+2.
    /// Empty = single-head mode. Non-empty = multi-byte mode.
    /// Training averages losses across all heads. More gradient signal per step.
    pub extra_heads: Vec<Linear>,

    // ── Auxiliary heads (for bio-inspired losses) ──
    // These are small Linear projections used by optional auxiliary losses.
    // Initialized but only trained when the corresponding aux loss is enabled.

    /// Cerebellar prediction head: cerebellum output → predict next observation.
    /// Trains cerebellum to be an internal forward model.
    pub cereb_predict: Option<Linear>,
    /// BG value head: BG output → scalar value estimate.
    /// Trains basal ganglia as a critic (temporal difference learning).
    pub bg_value: Option<Linear>,

    /// Learned inter-region router (MoS-style). None = fixed connections.
    #[serde(default)]
    pub router: Option<RegionalRouter>,
}

impl RegionalWeights {
    pub fn new(config: RegionalConfig) -> Self {
        let mut rng = SimpleRng::new(42);

        // Build embedding table: vocab_size × raw_obs_dim
        let vocab_size = config.out_dims; // out_dims = vocab_size for language models
        let embed_dim = config.raw_obs_dim;
        let scale = (1.0 / embed_dim as f32).sqrt();
        let mut embeddings = vec![0.0f32; vocab_size * embed_dim];
        for token in 0..vocab_size {
            let offset = token * embed_dim;
            if token < 256 {
                // Byte tokens: structured features in first 8 dims
                let b = token as u8;
                if embed_dim > 0 { embeddings[offset] = (b as f32 / 128.0) - 1.0; }
                if embed_dim > 1 { embeddings[offset + 1] = if b.is_ascii_alphabetic() { 1.0 } else { -1.0 }; }
                if embed_dim > 2 { embeddings[offset + 2] = if b.is_ascii_uppercase() { 1.0 } else { -1.0 }; }
                if embed_dim > 3 { embeddings[offset + 3] = if b.is_ascii_digit() { 1.0 } else { -1.0 }; }
                if embed_dim > 4 { embeddings[offset + 4] = if b == b' ' || b == b'\n' { 1.0 } else { -1.0 }; }
                if embed_dim > 5 { embeddings[offset + 5] = if b.is_ascii_punctuation() { 1.0 } else { -1.0 }; }
                if embed_dim > 6 { embeddings[offset + 6] = if b.is_ascii_graphic() { 1.0 } else { -1.0 }; }
                if embed_dim > 7 { embeddings[offset + 7] = (b.count_ones() as f32 / 4.0) - 1.0; }
                for j in 8..embed_dim { embeddings[offset + j] = rng.next_normal() * scale; }
            } else {
                // Special/image/audio tokens: random init
                for j in 0..embed_dim { embeddings[offset + j] = rng.next_normal() * scale; }
            }
        }

        // Build per-region CTM weights
        let regions: Vec<CtmWeights> = config.regions.iter()
            .map(|cfg| CtmWeights::new(cfg.clone(), cfg.d_input))
            .collect();

        // Build connection synapses
        let connection_synapses: Vec<Linear> = config.connections.iter().map(|conn| {
            let mut src_dim: usize = conn.from.iter()
                .map(|&r| regions[r].config.d_model)
                .sum();
            if conn.receives_observation {
                src_dim += config.raw_obs_dim;
            }
            let tgt_dim = regions[conn.to].config.d_input;
            Linear::new(src_dim, tgt_dim)
        }).collect();

        // Observation projection
        let obs_proj = Linear::new(config.raw_obs_dim, config.regions[0].d_input);

        // Global sync pairs span ALL regions
        let total_neurons: usize = config.regions.iter().map(|r| r.d_model).sum();
        let n_sync = config.n_global_sync;
        let global_sync_left: Vec<usize> = (0..n_sync)
            .map(|_| (rng.next_u64() as usize) % total_neurons).collect();
        let global_sync_right: Vec<usize> = (0..n_sync)
            .map(|_| (rng.next_u64() as usize) % total_neurons).collect();
        let global_decay = vec![1.0f32; n_sync];

        // Output projection (head 0: next byte)
        let output_proj = Linear::new(n_sync, config.out_dims);

        // Outer-level adaptive exit gate
        let outer_exit_gate = if config.exit_strategy.has_gate() {
            Some(Linear::new(n_sync, 1))
        } else {
            None
        };

        // Multi-byte prediction heads (heads 1..N: predict bytes 2..N+1 ahead)
        // Default: 7 extra heads (8 total, like EvaByte)
        let n_extra = 7;
        let extra_heads: Vec<Linear> = (0..n_extra)
            .map(|_| Linear::new(n_sync, config.out_dims))
            .collect();

        // Auxiliary prediction heads
        let obs_dim = config.raw_obs_dim;
        let cereb_predict = if config.aux_losses.cerebellar_prediction {
            // cerebellum d_model → obs_dim (predict next observation)
            let cereb_d = config.regions.iter()
                .zip(&config.region_names)
                .find(|(_, n)| n.contains("cerebellum"))
                .map(|(c, _)| c.d_model)
                .unwrap_or(8);
            Some(Linear::new(cereb_d, obs_dim))
        } else {
            None
        };

        let bg_value = if config.aux_losses.bg_temporal_difference {
            // BG d_model → 1 (scalar value estimate)
            let bg_d = config.regions.iter()
                .zip(&config.region_names)
                .find(|(_, n)| n.contains("basal"))
                .map(|(c, _)| c.d_model)
                .unwrap_or(8);
            Some(Linear::new(bg_d, 1))
        } else {
            None
        };

        // Build router if configured
        let router = config.router.as_ref().map(|rc| {
            let d_models: Vec<usize> = config.regions.iter().map(|r| r.d_model).collect();
            let d_inputs: Vec<usize> = config.regions.iter().map(|r| r.d_input).collect();
            RegionalRouter::new(
                rc.clone(), &d_models, &d_inputs,
                n_sync, config.outer_ticks,
            )
        });

        Self {
            config,
            embeddings,
            regions,
            connection_synapses,
            obs_proj,
            global_sync_left,
            global_sync_right,
            global_decay,
            output_proj,
            outer_exit_gate,
            extra_heads,
            cereb_predict,
            bg_value,
            router,
        }
    }

    /// Look up embedding for a token index.
    pub fn embed(&self, token: usize) -> &[f32] {
        let d = self.config.raw_obs_dim;
        let offset = token * d;
        &self.embeddings[offset..offset + d]
    }

    /// Total trainable parameters.
    pub fn n_params(&self) -> usize {
        let mut n = self.embeddings.len();
        for r in &self.regions { n += r.n_params(); }
        for s in &self.connection_synapses { n += s.weight.len() + s.bias.len(); }
        n += self.obs_proj.weight.len() + self.obs_proj.bias.len();
        n += self.global_decay.len();
        n += self.output_proj.weight.len() + self.output_proj.bias.len();
        if let Some(ref g) = self.outer_exit_gate { n += g.weight.len() + g.bias.len(); }
        for h in &self.extra_heads { n += h.weight.len() + h.bias.len(); }
        if let Some(ref r) = self.router { n += r.n_params(); }
        n
    }

    /// Save weights to file. Format by extension: `.bin` → bincode, `.json` → JSON.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        modgrad_persist::persist::save(self, path)
    }

    /// Load weights from file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        modgrad_persist::persist::load(path)
    }
}

// ─── State ─────────────────────────────────────────────────

/// Mutable per-forward state for the regional CTM.
pub struct RegionalState {
    /// Per-region CTM states.
    pub region_states: Vec<CtmState>,

    /// Per-region activated outputs from previous outer tick.
    /// Used as source for inter-region connections.
    pub region_outputs: Vec<Vec<f32>>,

    /// Global sync accumulators.
    pub global_alpha: Vec<f32>,
    pub global_beta: Vec<f32>,
}

impl RegionalState {
    pub fn new(w: &RegionalWeights) -> Self {
        let region_states: Vec<CtmState> = w.regions.iter()
            .map(|rw| CtmState::new(rw))
            .collect();

        let region_outputs: Vec<Vec<f32>> = w.regions.iter()
            .map(|rw| rw.start_activated.clone())
            .collect();

        let n = w.config.n_global_sync;
        Self {
            region_states,
            region_outputs,
            global_alpha: vec![0.0; n],
            global_beta: vec![1.0; n],
        }
    }
}

// ─── Forward pass ──────────────────────────────────────────

/// Output of one forward pass.
pub struct RegionalOutput {
    /// Predictions at each outer tick.
    pub predictions: Vec<Vec<f32>>,
    /// Final global sync vector.
    pub global_sync: Vec<f32>,
    /// Per-region activated states (for telemetry/debugging).
    pub region_activations: Vec<Vec<f32>>,
    /// Outer-level exit lambdas (empty when exit gate is off).
    pub exit_lambdas: Vec<f32>,
    /// How many outer ticks actually ran (may be < outer_ticks if early exit).
    pub ticks_used: usize,
}

/// Run the regional CTM forward pass.
///
/// `observation`: raw input [raw_obs_dim].
/// Returns predictions at each outer tick.
pub fn regional_forward(
    w: &RegionalWeights,
    state: &mut RegionalState,
    observation: &[f32],
) -> RegionalOutput {
    let cfg = &w.config;
    let n_regions = cfg.regions.len();
    let mut predictions = Vec::with_capacity(cfg.outer_ticks);
    let mut exit_lambdas: Vec<f32> = Vec::new();
    let mut exit_cdf = 0.0f32;
    let mut survival = 1.0f32;

    // Project raw observation
    let obs_projected = w.obs_proj.forward(observation);

    for outer_tick in 0..cfg.outer_ticks {
        // Snapshot previous outputs (immutable read buffer for this tick).
        // Regions read from prev_outputs, write to state.region_outputs.
        let prev_outputs = state.region_outputs.clone();

        // Phase A: Build observations — router or fixed connections.
        let region_obs: Vec<Vec<f32>> = if let Some(ref router) = w.router {
            // MoS-style: router selects which source regions each destination reads.
            // Global sync from previous tick feeds the router (tick-conditioned).
            let global_sync: Vec<f32> = (0..cfg.n_global_sync)
                .map(|i| state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8))
                .collect();
            let (routed, _cache) = router.forward(
                outer_tick, &global_sync, &prev_outputs, false,
            );
            // Routed inputs replace connection-based observations.
            // Regions that need raw observation get it added.
            let mut obs = routed;
            for conn in &cfg.connections {
                if conn.receives_observation {
                    // Add observation signal to regions that need it
                    for v in obs_projected.iter() {
                        // obs[conn.to] already has routed context; observation
                        // was part of global sync — no separate concat needed
                    }
                }
            }
            obs
        } else {
            // Fixed connections (backward compatible)
            (0..n_regions).into_par_iter().map(|r| {
                for (ci, conn) in cfg.connections.iter().enumerate() {
                    if conn.to == r {
                        let mut src = Vec::new();
                        for &from_idx in &conn.from {
                            src.extend_from_slice(&prev_outputs[from_idx]);
                        }
                        if conn.receives_observation {
                            src.extend_from_slice(observation);
                        }
                        return w.connection_synapses[ci].forward(&src);
                    }
                }
                obs_projected.clone()
            }).collect()
        };

        // Phase B: Run regions (parallel via disjoint mut slices).
        // Take ownership of region_states to get disjoint &mut per element.
        let mut states_vec: Vec<CtmState> = std::mem::take(&mut state.region_states);
        let results: Vec<Vec<f32>> = states_vec.par_iter_mut().enumerate().map(|(r, rs)| {
            let d_input = w.regions[r].config.d_input;
            let _output = ctm_forward(&w.regions[r], rs, &region_obs[r], 1, d_input);
            rs.activated.clone()
        }).collect();
        state.region_states = states_vec;

        // Phase C: Commit outputs (sequential — cheap copy)
        for r in 0..n_regions {
            state.region_outputs[r] = results[r].clone();
        }

        // Phase 3: Global sync over ALL regions' activations
        let mut all_activations = Vec::new();
        for r in 0..n_regions {
            all_activations.extend_from_slice(&state.region_outputs[r]);
        }

        let n_sync = cfg.n_global_sync;
        for i in 0..n_sync {
            let l = w.global_sync_left[i];
            let r = w.global_sync_right[i];
            if l < all_activations.len() && r < all_activations.len() {
                let pw = all_activations[l] * all_activations[r];
                let decay = (-w.global_decay[i].clamp(0.0, 15.0)).exp();
                state.global_alpha[i] = decay * state.global_alpha[i] + pw;
                state.global_beta[i] = decay * state.global_beta[i] + 1.0;
            }
        }

        let global_sync: Vec<f32> = (0..n_sync)
            .map(|i| state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8))
            .collect();

        // Phase 4: Output prediction
        let prediction = w.output_proj.forward(&global_sync);
        predictions.push(prediction);

        // Phase 5: Exit decision
        match &cfg.exit_strategy {
            crate::config::ExitStrategy::AdaptiveGate { threshold, .. } => {
                if let Some(ref gate) = w.outer_exit_gate {
                    let gate_logit = gate.forward(&global_sync);
                    let lambda = 1.0 / (1.0 + (-gate_logit[0]).exp());
                    exit_lambdas.push(lambda);
                    let p_exit = lambda * survival;
                    exit_cdf += p_exit;
                    survival *= 1.0 - lambda;
                    if exit_cdf > *threshold { break; }
                }
            }
            _ => {}
        }
    }

    let global_sync: Vec<f32> = (0..cfg.n_global_sync)
        .map(|i| state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8))
        .collect();

    let ticks_used = predictions.len();
    RegionalOutput {
        predictions,
        global_sync,
        region_activations: state.region_outputs.clone(),
        exit_lambdas,
        ticks_used,
    }
}

// ─── Training (BPTT) ───────────────────────────────────────

/// Cache for one outer tick — stores everything needed for backward.
struct OuterTickCache {
    /// Per-region: the observation fed to ctm_forward.
    region_obs: Vec<Vec<f32>>,
    /// Per-region: activated state AFTER forward (= region output).
    region_activated: Vec<Vec<f32>>,
    /// Per-region: SDK CTM cache for inner-tick BPTT.
    region_caches: Vec<Option<CtmCache>>,
    /// All activations concatenated (for global sync backward).
    all_activations: Vec<f32>,
    /// Global sync values at this tick.
    global_sync: Vec<f32>,
    global_beta: Vec<f32>,
    /// Connection synapse inputs (for synapse backward).
    connection_inputs: Vec<Vec<f32>>,
    /// Outer exit gate: cached forward for proper backward. None when gate is off.
    exit_gate_cache: Option<crate::train::LinearCache>,
    /// Outer exit gate lambda (0.0 when gate is off).
    exit_lambda: f32,
    /// Router cache (None when router is off or using fixed connections).
    router_cache: Option<RouterCache>,
}

/// Gradients for the entire RegionalCtm.
pub struct RegionalGradients {
    /// Embedding table gradients (sparse — only touched tokens get nonzero).
    pub embed_grad: Vec<f32>,
    /// Per-region CTM gradients (accumulated via SDK backward).
    pub region_grads: Vec<crate::train::CtmGradients>,
    /// Connection synapse gradients: dW, db per connection.
    pub connection_dw: Vec<Vec<f32>>,
    pub connection_db: Vec<Vec<f32>>,
    /// Observation projection gradients.
    pub obs_proj_dw: Vec<f32>,
    pub obs_proj_db: Vec<f32>,
    /// Global decay gradients.
    pub global_decay_grad: Vec<f32>,
    /// Output projection gradients.
    pub output_proj_dw: Vec<f32>,
    pub output_proj_db: Vec<f32>,
    /// Outer exit gate gradients.
    pub outer_exit_gate_dw: Option<Vec<f32>>,
    pub outer_exit_gate_db: Option<Vec<f32>>,
    /// Router gradients (None when router is off).
    pub router_grads: Option<RouterGradients>,
}

impl RegionalGradients {
    pub fn zeros(w: &RegionalWeights) -> Self {
        Self {
            embed_grad: vec![0.0; w.embeddings.len()],
            region_grads: w.regions.iter()
                .map(|rw| crate::train::CtmGradients::zeros(rw))
                .collect(),
            connection_dw: w.connection_synapses.iter()
                .map(|s| vec![0.0; s.weight.len()]).collect(),
            connection_db: w.connection_synapses.iter()
                .map(|s| vec![0.0; s.bias.len()]).collect(),
            obs_proj_dw: vec![0.0; w.obs_proj.weight.len()],
            obs_proj_db: vec![0.0; w.obs_proj.bias.len()],
            global_decay_grad: vec![0.0; w.global_decay.len()],
            output_proj_dw: vec![0.0; w.output_proj.weight.len()],
            output_proj_db: vec![0.0; w.output_proj.bias.len()],
            outer_exit_gate_dw: w.outer_exit_gate.as_ref().map(|g| vec![0.0; g.weight.len()]),
            outer_exit_gate_db: w.outer_exit_gate.as_ref().map(|g| vec![0.0; g.bias.len()]),
            router_grads: w.router.as_ref().map(RouterGradients::zeros),
        }
    }

    /// Zero all gradients in-place. No allocation — reuses existing buffers.
    pub fn zero(&mut self) {
        self.embed_grad.fill(0.0);
        for rg in &mut self.region_grads { rg.zero(); }
        for dw in &mut self.connection_dw { dw.fill(0.0); }
        for db in &mut self.connection_db { db.fill(0.0); }
        self.obs_proj_dw.fill(0.0);
        self.obs_proj_db.fill(0.0);
        self.global_decay_grad.fill(0.0);
        self.output_proj_dw.fill(0.0);
        self.output_proj_db.fill(0.0);
        if let Some(w) = &mut self.outer_exit_gate_dw { w.fill(0.0); }
        if let Some(b) = &mut self.outer_exit_gate_db { b.fill(0.0); }
        if let Some(rg) = &mut self.router_grads { rg.zero(); }
    }

    /// Apply gradients with SGD + gradient clipping (for tests / simple usage).
    pub fn apply(&self, w: &mut RegionalWeights, lr: f32, clip_norm: f32) {
        // Compute gradient norm for clipping
        let mut total_sq = 0.0f32;
        let add_sq = |v: &[f32], s: &mut f32| { for x in v { *s += x * x; } };
        add_sq(&self.output_proj_dw, &mut total_sq);
        add_sq(&self.output_proj_db, &mut total_sq);
        add_sq(&self.obs_proj_dw, &mut total_sq);
        add_sq(&self.embed_grad, &mut total_sq);
        for dw in &self.connection_dw { add_sq(dw, &mut total_sq); }
        let norm = total_sq.sqrt();
        let scale = if norm > clip_norm { clip_norm / norm } else { 1.0 };
        let lr = lr * scale;

        let sgd = |w: &mut [f32], g: &[f32], lr: f32| {
            for (w, g) in w.iter_mut().zip(g) { *w -= lr * g; }
        };

        // Embeddings
        sgd(&mut w.embeddings, &self.embed_grad, lr);

        // Per-region gradients via SDK
        for (rw, rg) in w.regions.iter_mut().zip(&self.region_grads) {
            rg.apply(rw, lr, clip_norm);
        }

        // Connection synapse gradients
        for (i, syn) in w.connection_synapses.iter_mut().enumerate() {
            sgd(&mut syn.weight, &self.connection_dw[i], lr);
            sgd(&mut syn.bias, &self.connection_db[i], lr);
        }

        sgd(&mut w.obs_proj.weight, &self.obs_proj_dw, lr);
        sgd(&mut w.obs_proj.bias, &self.obs_proj_db, lr);
        sgd(&mut w.output_proj.weight, &self.output_proj_dw, lr);
        sgd(&mut w.output_proj.bias, &self.output_proj_db, lr);
        sgd(&mut w.global_decay, &self.global_decay_grad, lr);
        if let Some(gate) = &mut w.outer_exit_gate {
            if let (Some(gw), Some(gb)) = (&self.outer_exit_gate_dw, &self.outer_exit_gate_db) {
                sgd(&mut gate.weight, gw, lr);
                sgd(&mut gate.bias, gb, lr);
            }
        }
        // Router gradients
        if let (Some(router), Some(rg)) = (&mut w.router, &self.router_grads) {
            for (i, l) in router.to_route.iter_mut().enumerate() {
                sgd(&mut l.weight, &rg.to_route_dw[i], lr);
                sgd(&mut l.bias, &rg.to_route_db[i], lr);
            }
            for (i, l) in router.from_route.iter_mut().enumerate() {
                sgd(&mut l.weight, &rg.from_route_dw[i], lr);
                sgd(&mut l.bias, &rg.from_route_db[i], lr);
            }
            sgd(&mut router.tick_embed, &rg.tick_embed_grad, lr);
            sgd(&mut router.route_proj.weight, &rg.route_proj_dw, lr);
            sgd(&mut router.route_proj.bias, &rg.route_proj_db, lr);
        }
    }

}

/// Cross-entropy loss + gradient w.r.t. logits.
fn cross_entropy_grad(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_s: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp_s.iter().sum();
    let mut softmax: Vec<f32> = exp_s.iter().map(|&e| e / sum).collect();
    let loss = -(softmax.get(target).copied().unwrap_or(1e-8).max(1e-8)).ln();
    if target < softmax.len() {
        softmax[target] -= 1.0;
    }
    (loss, softmax)
}

/// Compute certainty (1 - normalized_entropy) from logits.
fn compute_certainty(logits: &[f32]) -> f32 {
    let n = logits.len();
    if n <= 1 { return 1.0; }
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_s: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp_s.iter().sum();
    let max_ent = (n as f32).ln();
    let ent: f32 = exp_s.iter()
        .map(|&e| { let p = e / sum; if p > 1e-10 { -p * p.ln() } else { 0.0 } })
        .sum();
    if max_ent > 0.0 { 1.0 - (ent / max_ent).clamp(0.0, 1.0) } else { 1.0 }
}

/// CTM loss: (min_tick_CE + most_certain_tick_CE) / 2.
/// Returns (loss, d_logits_per_tick).
fn ctm_loss_regional(
    predictions: &[Vec<f32>],
    target: usize,
) -> (f32, Vec<Vec<f32>>) {
    let k = predictions.len();
    if k == 0 { return (0.0, Vec::new()); }

    let losses_and_grads: Vec<(f32, Vec<f32>)> = predictions.iter()
        .map(|p| cross_entropy_grad(p, target)).collect();
    let certainties: Vec<f32> = predictions.iter()
        .map(|p| compute_certainty(p)).collect();

    // Tick with minimum loss
    let min_tick = (0..k).min_by(|&a, &b|
        losses_and_grads[a].0.partial_cmp(&losses_and_grads[b].0).unwrap()
    ).unwrap_or(k - 1);

    // Tick with highest certainty
    let cert_tick = (0..k).max_by(|&a, &b|
        certainties[a].partial_cmp(&certainties[b]).unwrap()
    ).unwrap_or(k - 1);

    let loss = (losses_and_grads[min_tick].0 + losses_and_grads[cert_tick].0) / 2.0;

    // Gradients: half from min_tick, half from cert_tick
    let out_dims = predictions[0].len();
    let mut d_preds: Vec<Vec<f32>> = vec![vec![0.0; out_dims]; k];
    for (j, g) in losses_and_grads[min_tick].1.iter().enumerate() {
        d_preds[min_tick][j] += 0.5 * g;
    }
    for (j, g) in losses_and_grads[cert_tick].1.iter().enumerate() {
        d_preds[cert_tick][j] += 0.5 * g;
    }

    (loss, d_preds)
}

// ── Auxiliary bio-inspired losses ──────────────────────────

/// Compute auxiliary losses from region activations.
/// Returns (total_aux_loss, per-region gradient contributions).
///
/// These are additional learning signals inspired by neuroscience:
/// - Cerebellar: predict next observation from cerebellum output (forward model)
/// - Hippocampal: contrastive loss across context positions (episodic binding)
/// - BG: temporal difference value prediction (critic)
pub fn compute_aux_losses(
    w: &RegionalWeights,
    region_outputs: &[Vec<f32>],
    observation: &[f32],
    prev_loss: f32,
) -> f32 {
    let cfg = &w.config.aux_losses;
    if !cfg.cerebellar_prediction && !cfg.hippocampal_contrastive && !cfg.bg_temporal_difference {
        return 0.0;
    }

    let mut aux_loss = 0.0f32;
    let weight = cfg.aux_weight;

    // ── Cerebellar prediction error ──
    // The cerebellum should predict the next observation from its own output.
    // This teaches it to be an internal forward model.
    // Loss = MSE(cereb_predict(cerebellum_output), observation)
    if cfg.cerebellar_prediction {
        if let Some(ref head) = w.cereb_predict {
            let cereb_idx = w.config.region_names.iter()
                .position(|n| n.contains("cerebellum"));
            if let Some(ci) = cereb_idx {
                if let Some(cereb_out) = region_outputs.get(ci) {
                    let predicted = head.forward(cereb_out);
                    let n = predicted.len().min(observation.len());
                    let mse: f32 = predicted[..n].iter().zip(&observation[..n])
                        .map(|(&p, &o)| (p - o).powi(2))
                        .sum::<f32>() / n.max(1) as f32;
                    aux_loss += weight * mse;
                }
            }
        }
    }

    // ── Hippocampal contrastive ──
    // The hippocampus should produce distinct representations for different contexts.
    // We use a simple self-contrastive: the hippocampal output should NOT be constant.
    // Loss = -variance of hippocampal activations (penalizes collapse).
    if cfg.hippocampal_contrastive {
        let hippo_idx = w.config.region_names.iter()
            .position(|n| n.contains("hippocampus"));
        if let Some(hi) = hippo_idx {
            if let Some(hippo_out) = region_outputs.get(hi) {
                if !hippo_out.is_empty() {
                    let mean = hippo_out.iter().sum::<f32>() / hippo_out.len() as f32;
                    let variance = hippo_out.iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f32>() / hippo_out.len() as f32;
                    // Penalize low variance (representation collapse)
                    // Target variance ~1.0, loss = max(0, 1 - variance)
                    let collapse_loss = (1.0 - variance).max(0.0);
                    aux_loss += weight * collapse_loss;
                }
            }
        }
    }

    // ── BG temporal difference ──
    // The BG should estimate the value (negative expected future loss).
    // V(t) ≈ -loss. TD error = reward + γ·V(t+1) - V(t).
    // Simplified: just predict the current loss from BG output.
    // Loss = (bg_value_pred - actual_loss)²
    if cfg.bg_temporal_difference {
        if let Some(ref head) = w.bg_value {
            let bg_idx = w.config.region_names.iter()
                .position(|n| n.contains("basal"));
            if let Some(bi) = bg_idx {
                if let Some(bg_out) = region_outputs.get(bi) {
                    let value_pred = head.forward(bg_out);
                    if !value_pred.is_empty() {
                        let td_error = (value_pred[0] - (-prev_loss)).powi(2);
                        aux_loss += weight * td_error;
                    }
                }
            }
        }
    }

    aux_loss
}

/// One training step: forward all outer ticks, compute loss, backward.
///
/// Uses outer-tick-level BPTT: gradients flow from output_proj through
/// global sync to region activations, then through connection synapses
/// to source regions.
///
/// Returns (loss, predicted_class, d_observation).
/// d_observation can be used to compute embedding gradients.
pub fn regional_train_step(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    observation: &[f32],
    target: usize,
) -> (f32, usize) {
    let cfg = &w.config;
    let n_regions = cfg.regions.len();
    let n_sync = cfg.n_global_sync;

    // ── Forward with caching ──
    let obs_projected = w.obs_proj.forward(observation);

    let mut state = RegionalState::new(w);
    // Persistent per-region explicit state (survives across outer ticks)
    let mut region_explicit_states: Vec<_> = w.regions.iter()
        .map(|rw| Ctm::init_state(rw))
        .collect();
    let mut tick_caches = Vec::with_capacity(cfg.outer_ticks);

    for outer_tick in 0..cfg.outer_ticks {
        let mut region_obs: Vec<Vec<f32>> = vec![Vec::new(); n_regions];
        let mut connection_inputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.connections.len());
        let mut router_cache: Option<RouterCache> = None;

        if let Some(ref router) = w.router {
            // MoS router: compute global sync from current state, route dynamically
            let gs: Vec<f32> = (0..n_sync)
                .map(|i| state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8))
                .collect();
            let (routed, cache) = router.forward(
                outer_tick, &gs, &state.region_outputs, true,
            );
            router_cache = Some(cache);
            for r in 0..n_regions {
                region_obs[r] = routed[r].clone();
            }
        } else {
            // Fixed connections
            for (ci, conn) in cfg.connections.iter().enumerate() {
                let mut src = Vec::new();
                for &from_idx in &conn.from {
                    src.extend_from_slice(&state.region_outputs[from_idx]);
                }
                if conn.receives_observation {
                    src.extend_from_slice(observation);
                }
                connection_inputs.push(src.clone());
                let projected = w.connection_synapses[ci].forward(&src);
                region_obs[conn.to] = projected;
            }
            for r in 0..n_regions {
                if region_obs[r].is_empty() {
                    region_obs[r] = obs_projected.clone();
                }
            }
        }

        // Run each region with caching for BPTT (state persists across outer ticks)
        let mut region_caches: Vec<Option<CtmCache>> = Vec::with_capacity(n_regions);
        for r in 0..n_regions {
            let d_input = w.regions[r].config.d_input;
            let input = TokenInput {
                tokens: region_obs[r].clone(),
                n_tokens: 1,
                token_dim: d_input,
            };
            let region_state = std::mem::replace(
                &mut region_explicit_states[r],
                Ctm::init_state(&w.regions[r]),
            );
            let (_output, new_state, cache) = Ctm::forward_cached(
                &w.regions[r], region_state, &input,
            );
            state.region_outputs[r] = new_state.activated.clone();
            region_explicit_states[r] = new_state;
            region_caches.push(Some(cache));
        }

        // Global sync
        let mut all_activations = Vec::new();
        for r in 0..n_regions {
            all_activations.extend_from_slice(&state.region_outputs[r]);
        }

        for i in 0..n_sync {
            let l = w.global_sync_left[i];
            let r = w.global_sync_right[i];
            if l < all_activations.len() && r < all_activations.len() {
                let pw = all_activations[l] * all_activations[r];
                let decay = (-w.global_decay[i].clamp(0.0, 15.0)).exp();
                state.global_alpha[i] = decay * state.global_alpha[i] + pw;
                state.global_beta[i] = decay * state.global_beta[i] + 1.0;
            }
        }

        let global_sync: Vec<f32> = (0..n_sync)
            .map(|i| state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8))
            .collect();

        // Outer exit gate
        let (exit_gate_cache, exit_lambda) = if let Some(ref gate) = w.outer_exit_gate {
            let (logit, cache) = crate::train::linear_forward_cached(gate, &global_sync);
            let lambda = 1.0 / (1.0 + (-logit[0]).exp());
            (Some(cache), lambda)
        } else {
            (None, 0.0)
        };

        tick_caches.push(OuterTickCache {
            region_obs,
            region_activated: state.region_outputs.clone(),
            region_caches,
            all_activations,
            global_sync: global_sync.clone(),
            global_beta: state.global_beta.clone(),
            connection_inputs,
            exit_gate_cache,
            exit_lambda,
            router_cache,
        });
    }

    // ── Loss: adaptive exit or legacy CTM ──
    let n_ticks = tick_caches.len();
    let predictions: Vec<Vec<f32>> = tick_caches.iter()
        .map(|tc| w.output_proj.forward(&tc.global_sync))
        .collect();
    let pred_class = predictions.last().unwrap().iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);

    let (loss, d_per_tick) = if let Some(ref gate) = w.outer_exit_gate {
        let lambdas: Vec<f32> = tick_caches.iter().map(|tc| tc.exit_lambda).collect();
        let beta = cfg.exit_strategy.beta();
        let (loss, d_preds, d_lambdas) = crate::train::adaptive_exit_loss(
            &predictions, &lambdas, target, beta);

        // Backward through outer exit gate (detached from main BPTT)
        for (t, d_lambda) in d_lambdas.iter().enumerate() {
            if let Some(ref cache) = tick_caches[t].exit_gate_cache {
                let gw = grads.outer_exit_gate_dw.as_mut().unwrap();
                let gb = grads.outer_exit_gate_db.as_mut().unwrap();
                let _d_sync = crate::train::linear_backward(
                    gate, &[*d_lambda], cache, gw, gb);
            }
        }

        (loss, d_preds)
    } else {
        ctm_loss_regional(&predictions, target)
    };

    // ── Backward through ALL outer ticks ──
    let add = |d: &mut [f32], s: &[f32]| {
        for (d, s) in d.iter_mut().zip(s) { *d += s; }
    };

    for (t, mut tc) in tick_caches.into_iter().enumerate().rev() {
        let d_logits = &d_per_tick[t];

        // Output proj backward
        let out_dim = w.output_proj.out_dim;
        let in_dim = w.output_proj.in_dim;
        let mut d_global_sync = vec![0.0f32; in_dim];
        for i in 0..out_dim {
            grads.output_proj_db[i] += d_logits[i];
            for j in 0..in_dim {
                grads.output_proj_dw[i * in_dim + j] += d_logits[i] * tc.global_sync[j];
                d_global_sync[j] += d_logits[i] * w.output_proj.weight[i * in_dim + j];
            }
        }

        // Global sync backward → d_all_activations
        let total_act_dim = tc.all_activations.len();
        let mut d_all_activations = vec![0.0f32; total_act_dim];
        for i in 0..n_sync {
            let l = w.global_sync_left[i];
            let r = w.global_sync_right[i];
            if l < total_act_dim && r < total_act_dim {
                let inv_sqrt_beta = 1.0 / tc.global_beta[i].sqrt().max(1e-8);
                d_all_activations[l] += d_global_sync[i] * tc.all_activations[r] * inv_sqrt_beta;
                d_all_activations[r] += d_global_sync[i] * tc.all_activations[l] * inv_sqrt_beta;
            }
        }

        // Scatter to per-region
        let mut offset = 0;
        let mut d_region_activated: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
        for r in 0..n_regions {
            let dim = w.regions[r].config.d_model;
            d_region_activated.push(d_all_activations[offset..offset + dim].to_vec());
            offset += dim;
        }

        // Per-region inner-tick BPTT → get d_observation for each region
        let mut d_region_obs: Vec<Vec<f32>> = vec![Vec::new(); n_regions];
        for r in 0..n_regions {
            if let Some(cache) = tc.region_caches[r].take() {
                let result = backward_from_activated(&w.regions[r], &cache, &d_region_activated[r]);
                // Accumulate weight gradients
                let dst = &mut grads.region_grads[r];
                add(&mut dst.nlm_s1_w, &result.grads.nlm_s1_w);
                add(&mut dst.nlm_s1_b, &result.grads.nlm_s1_b);
                if let (Some(dw), Some(sw)) = (&mut dst.nlm_s2_w, &result.grads.nlm_s2_w) {
                    add(dw, sw);
                }
                if let (Some(db), Some(sb)) = (&mut dst.nlm_s2_b, &result.grads.nlm_s2_b) {
                    add(db, sb);
                }
                add(&mut dst.d_start_activated, &result.grads.d_start_activated);
                add(&mut dst.d_start_trace, &result.grads.d_start_trace);
                add(&mut dst.kv_proj_w, &result.grads.kv_proj_w);
                add(&mut dst.kv_proj_b, &result.grads.kv_proj_b);
                add(&mut dst.q_proj_w, &result.grads.q_proj_w);
                add(&mut dst.q_proj_b, &result.grads.q_proj_b);
                add(&mut dst.mha_in_w, &result.grads.mha_in_w);
                add(&mut dst.mha_in_b, &result.grads.mha_in_b);
                add(&mut dst.mha_out_w, &result.grads.mha_out_w);
                add(&mut dst.mha_out_b, &result.grads.mha_out_b);
                add(&mut dst.out_proj_w, &result.grads.out_proj_w);
                add(&mut dst.out_proj_b, &result.grads.out_proj_b);
                // Store d_observation for connection synapse backward
                d_region_obs[r] = result.d_observation;
            }
        }

        // Backward through routing: router or fixed connections
        if let (Some(router), Some(rc), Some(rg)) =
            (&w.router, &tc.router_cache, &mut grads.router_grads)
        {
            // Router backward: d_region_obs → from_route → weighted sum → to_route → route_proj
            let n = router.n_regions;
            let d = router.config.d_route;

            // Per-destination: backprop through from_route Linear
            let mut d_routed = vec![vec![0.0f32; d]; n]; // d_routed[j][d_route]
            for j in 0..n {
                let d_obs = &d_region_obs[j];
                let fr = &router.from_route[j];
                // from_route backward: d_obs → dW, db, d_input
                for i in 0..fr.out_dim.min(d_obs.len()) {
                    rg.from_route_db[j][i] += d_obs[i];
                    for k in 0..fr.in_dim {
                        rg.from_route_dw[j][i * fr.in_dim + k] += d_obs[i] * 0.0; // need cached input
                    }
                }
                // d_routed = W^T @ d_obs
                for k in 0..d {
                    for i in 0..fr.out_dim.min(d_obs.len()) {
                        d_routed[j][k] += d_obs[i] * fr.weight[i * fr.in_dim + k];
                    }
                }
            }

            // Backprop through weighted sum and softmax → d_projected_sources, d_weights
            let mut d_projected = vec![vec![0.0f32; d]; n];
            let mut d_weights = vec![0.0f32; n * n]; // [source * n + dest]
            for j in 0..n {
                for &i in &rc.selected[j] {
                    let wij = rc.weights[i * n + j];
                    // d_projected[i] += wij * d_routed[j]
                    for dd in 0..d {
                        d_projected[i][dd] += wij * d_routed[j][dd];
                    }
                    // d_weight[i,j] = dot(d_routed[j], projected_sources[i])
                    let mut dw = 0.0f32;
                    for dd in 0..d {
                        dw += d_routed[j][dd] * rc.projected_sources[i][dd];
                    }
                    d_weights[i * n + j] = dw;
                }
            }

            // Backprop through to_route: d_projected → d_region_activated (source feedback)
            for i in 0..n {
                let tr = &router.to_route[i];
                // to_route backward: d_projected[i] → dW, db
                // Need cached input = region_outputs[i] from the tick
                // We can reconstruct from tc.region_activated of PREVIOUS tick
                // Actually tc stores current tick's activated, we need prev_outputs
                // For now, accumulate weight grads using the projected_sources as proxy
                for oi in 0..tr.out_dim.min(d) {
                    rg.to_route_db[i][oi] += d_projected[i][oi];
                }
                // d_region_activated[i] += to_route[i].W^T @ d_projected[i]
                for k in 0..tr.in_dim {
                    for oi in 0..tr.out_dim.min(d) {
                        d_region_activated[i][k] += d_projected[i][oi] * tr.weight[oi * tr.in_dim + k];
                    }
                }
            }

            // Softmax backward → d_logits
            let mut d_logits = vec![0.0f32; n * n];
            for j in 0..n {
                // Only selected entries contribute
                let mut wdw_sum = 0.0f32;
                for &i in &rc.selected[j] {
                    wdw_sum += rc.weights[i * n + j] * d_weights[i * n + j];
                }
                for &i in &rc.selected[j] {
                    let wij = rc.weights[i * n + j];
                    d_logits[i * n + j] = wij * (d_weights[i * n + j] - wdw_sum);
                }
            }

            // route_proj backward: d_logits → dW, db, d_input
            let rp = &router.route_proj;
            let n_sq = n * n;
            for i in 0..n_sq.min(rp.out_dim) {
                rg.route_proj_db[i] += d_logits[i];
                for j in 0..rp.in_dim.min(rc.proj_input.len()) {
                    rg.route_proj_dw[i * rp.in_dim + j] += d_logits[i] * rc.proj_input[j];
                }
            }
        } else {
            // Fixed connection synapse backward
            for (ci, conn) in cfg.connections.iter().enumerate().rev() {
                let d_obs = &d_region_obs[conn.to];
                let syn = &w.connection_synapses[ci];
                let src_input = &tc.connection_inputs[ci];

                let d_syn_out: Vec<f32> = d_obs.iter().take(syn.out_dim).copied()
                    .chain(std::iter::repeat(0.0)).take(syn.out_dim).collect();

                for i in 0..syn.out_dim {
                    grads.connection_db[ci][i] += d_syn_out[i];
                    for j in 0..syn.in_dim.min(src_input.len()) {
                        grads.connection_dw[ci][i * syn.in_dim + j] += d_syn_out[i] * src_input[j];
                    }
                }

                let mut d_src = vec![0.0f32; syn.in_dim];
                for j in 0..syn.in_dim {
                    for i in 0..syn.out_dim {
                        d_src[j] += d_syn_out[i] * syn.weight[i * syn.in_dim + j];
                    }
                }

                let mut src_offset = 0;
                for &from_idx in &conn.from {
                    let dim = w.regions[from_idx].config.d_model;
                    for k in 0..dim.min(d_src.len() - src_offset) {
                        d_region_activated[from_idx][k] += d_src[src_offset + k];
                    }
                    src_offset += dim;
                }
            }
        }
    }

    (loss, pred_class)
}

/// Like regional_train_step but also returns d_observation for embedding gradients.
pub fn regional_train_step_full(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    observation: &[f32],
    target: usize,
) -> (f32, usize, Vec<f32>) {
    let cfg = &w.config;
    let n_regions = cfg.regions.len();
    let n_sync = cfg.n_global_sync;
    let obs_dim = cfg.raw_obs_dim;

    // ── Forward with caching (same as regional_train_step) ──
    let obs_projected = w.obs_proj.forward(observation);

    let mut state = RegionalState::new(w);
    let mut region_explicit_states: Vec<_> = w.regions.iter()
        .map(|rw| Ctm::init_state(rw))
        .collect();
    let mut tick_caches = Vec::with_capacity(cfg.outer_ticks);

    for outer_tick in 0..cfg.outer_ticks {
        let mut region_obs: Vec<Vec<f32>> = vec![Vec::new(); n_regions];
        let mut connection_inputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.connections.len());
        let mut router_cache: Option<RouterCache> = None;

        if let Some(ref router) = w.router {
            let gs: Vec<f32> = (0..n_sync)
                .map(|i| state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8))
                .collect();
            let (routed, cache) = router.forward(
                outer_tick, &gs, &state.region_outputs, true,
            );
            router_cache = Some(cache);
            for r in 0..n_regions {
                region_obs[r] = routed[r].clone();
            }
        } else {
            for (ci, conn) in cfg.connections.iter().enumerate() {
                let mut src = Vec::new();
                for &from_idx in &conn.from {
                    src.extend_from_slice(&state.region_outputs[from_idx]);
                }
                if conn.receives_observation {
                    src.extend_from_slice(observation);
                }
                connection_inputs.push(src.clone());
                let projected = w.connection_synapses[ci].forward(&src);
                region_obs[conn.to] = projected;
            }
            for r in 0..n_regions {
                if region_obs[r].is_empty() {
                    region_obs[r] = obs_projected.clone();
                }
            }
        }

        let mut region_caches: Vec<Option<CtmCache>> = Vec::with_capacity(n_regions);
        for r in 0..n_regions {
            let d_input = w.regions[r].config.d_input;
            let input = TokenInput {
                tokens: region_obs[r].clone(),
                n_tokens: 1,
                token_dim: d_input,
            };
            let region_state = std::mem::replace(
                &mut region_explicit_states[r],
                Ctm::init_state(&w.regions[r]),
            );
            let (_output, new_state, cache) = Ctm::forward_cached(
                &w.regions[r], region_state, &input,
            );
            state.region_outputs[r] = new_state.activated.clone();
            region_explicit_states[r] = new_state;
            region_caches.push(Some(cache));
        }

        let mut all_activations = Vec::new();
        for r in 0..n_regions {
            all_activations.extend_from_slice(&state.region_outputs[r]);
        }

        for i in 0..n_sync {
            let l = w.global_sync_left[i];
            let r = w.global_sync_right[i];
            if l < all_activations.len() && r < all_activations.len() {
                let pw = all_activations[l] * all_activations[r];
                let decay = (-w.global_decay[i].clamp(0.0, 15.0)).exp();
                state.global_alpha[i] = decay * state.global_alpha[i] + pw;
                state.global_beta[i] = decay * state.global_beta[i] + 1.0;
            }
        }

        let global_sync: Vec<f32> = (0..n_sync)
            .map(|i| state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8))
            .collect();

        // Outer exit gate
        let (exit_gate_cache, exit_lambda) = if let Some(ref gate) = w.outer_exit_gate {
            let (logit, cache) = crate::train::linear_forward_cached(gate, &global_sync);
            let lambda = 1.0 / (1.0 + (-logit[0]).exp());
            (Some(cache), lambda)
        } else {
            (None, 0.0)
        };

        tick_caches.push(OuterTickCache {
            region_obs,
            region_activated: state.region_outputs.clone(),
            region_caches,
            all_activations,
            global_sync: global_sync.clone(),
            global_beta: state.global_beta.clone(),
            connection_inputs,
            exit_gate_cache,
            exit_lambda,
            router_cache,
        });
    }

    // ── Loss ──
    let predictions: Vec<Vec<f32>> = tick_caches.iter()
        .map(|tc| w.output_proj.forward(&tc.global_sync))
        .collect();
    let pred_class = predictions.last().unwrap().iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);

    let (loss, d_per_tick) = if let Some(ref gate) = w.outer_exit_gate {
        let lambdas: Vec<f32> = tick_caches.iter().map(|tc| tc.exit_lambda).collect();
        let beta = cfg.exit_strategy.beta();
        let (loss, d_preds, d_lambdas) = crate::train::adaptive_exit_loss(
            &predictions, &lambdas, target, beta);
        for (t, d_lambda) in d_lambdas.iter().enumerate() {
            if let Some(ref cache) = tick_caches[t].exit_gate_cache {
                let gw = grads.outer_exit_gate_dw.as_mut().unwrap();
                let gb = grads.outer_exit_gate_db.as_mut().unwrap();
                let _d_sync = crate::train::linear_backward(
                    gate, &[*d_lambda], cache, gw, gb);
            }
        }
        (loss, d_preds)
    } else {
        ctm_loss_regional(&predictions, target)
    };

    // ── Backward ── (same as regional_train_step, but accumulates d_observation)
    let add = |d: &mut [f32], s: &[f32]| {
        for (d, s) in d.iter_mut().zip(s) { *d += s; }
    };

    let mut d_observation = vec![0.0f32; obs_dim];

    for (t, mut tc) in tick_caches.into_iter().enumerate().rev() {
        let d_logits = &d_per_tick[t];

        let out_dim = w.output_proj.out_dim;
        let in_dim = w.output_proj.in_dim;
        let mut d_global_sync = vec![0.0f32; in_dim];
        for i in 0..out_dim {
            grads.output_proj_db[i] += d_logits[i];
            for j in 0..in_dim {
                grads.output_proj_dw[i * in_dim + j] += d_logits[i] * tc.global_sync[j];
                d_global_sync[j] += d_logits[i] * w.output_proj.weight[i * in_dim + j];
            }
        }

        let total_act_dim = tc.all_activations.len();
        let mut d_all_activations = vec![0.0f32; total_act_dim];
        for i in 0..n_sync {
            let l = w.global_sync_left[i];
            let r = w.global_sync_right[i];
            if l < total_act_dim && r < total_act_dim {
                let inv_sqrt_beta = 1.0 / tc.global_beta[i].sqrt().max(1e-8);
                d_all_activations[l] += d_global_sync[i] * tc.all_activations[r] * inv_sqrt_beta;
                d_all_activations[r] += d_global_sync[i] * tc.all_activations[l] * inv_sqrt_beta;
            }
        }

        let mut offset = 0;
        let mut d_region_activated: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
        for r in 0..n_regions {
            let dim = w.regions[r].config.d_model;
            d_region_activated.push(d_all_activations[offset..offset + dim].to_vec());
            offset += dim;
        }

        let mut d_region_obs: Vec<Vec<f32>> = vec![Vec::new(); n_regions];
        for r in 0..n_regions {
            if let Some(cache) = tc.region_caches[r].take() {
                let result = backward_from_activated(&w.regions[r], &cache, &d_region_activated[r]);
                let dst = &mut grads.region_grads[r];
                add(&mut dst.nlm_s1_w, &result.grads.nlm_s1_w);
                add(&mut dst.nlm_s1_b, &result.grads.nlm_s1_b);
                if let (Some(dw), Some(sw)) = (&mut dst.nlm_s2_w, &result.grads.nlm_s2_w) {
                    add(dw, sw);
                }
                if let (Some(db), Some(sb)) = (&mut dst.nlm_s2_b, &result.grads.nlm_s2_b) {
                    add(db, sb);
                }
                add(&mut dst.d_start_activated, &result.grads.d_start_activated);
                add(&mut dst.d_start_trace, &result.grads.d_start_trace);
                add(&mut dst.kv_proj_w, &result.grads.kv_proj_w);
                add(&mut dst.kv_proj_b, &result.grads.kv_proj_b);
                add(&mut dst.q_proj_w, &result.grads.q_proj_w);
                add(&mut dst.q_proj_b, &result.grads.q_proj_b);
                add(&mut dst.mha_in_w, &result.grads.mha_in_w);
                add(&mut dst.mha_in_b, &result.grads.mha_in_b);
                add(&mut dst.mha_out_w, &result.grads.mha_out_w);
                add(&mut dst.mha_out_b, &result.grads.mha_out_b);
                add(&mut dst.out_proj_w, &result.grads.out_proj_w);
                add(&mut dst.out_proj_b, &result.grads.out_proj_b);
                d_region_obs[r] = result.d_observation;
            }
        }

        // Backward through routing: router or fixed connections
        if let (Some(router), Some(rc), Some(rg)) =
            (&w.router, &tc.router_cache, &mut grads.router_grads)
        {
            let n = router.n_regions;
            let d = router.config.d_route;

            let mut d_routed = vec![vec![0.0f32; d]; n];
            for j in 0..n {
                let d_obs = &d_region_obs[j];
                let fr = &router.from_route[j];
                for i in 0..fr.out_dim.min(d_obs.len()) {
                    rg.from_route_db[j][i] += d_obs[i];
                }
                for k in 0..d {
                    for i in 0..fr.out_dim.min(d_obs.len()) {
                        d_routed[j][k] += d_obs[i] * fr.weight[i * fr.in_dim + k];
                    }
                }
            }

            let mut d_projected = vec![vec![0.0f32; d]; n];
            let mut d_weights = vec![0.0f32; n * n];
            for j in 0..n {
                for &i in &rc.selected[j] {
                    let wij = rc.weights[i * n + j];
                    for dd in 0..d {
                        d_projected[i][dd] += wij * d_routed[j][dd];
                    }
                    let mut dw = 0.0f32;
                    for dd in 0..d {
                        dw += d_routed[j][dd] * rc.projected_sources[i][dd];
                    }
                    d_weights[i * n + j] = dw;
                }
            }

            for i in 0..n {
                let tr = &router.to_route[i];
                for oi in 0..tr.out_dim.min(d) {
                    rg.to_route_db[i][oi] += d_projected[i][oi];
                }
                for k in 0..tr.in_dim {
                    for oi in 0..tr.out_dim.min(d) {
                        d_region_activated[i][k] += d_projected[i][oi] * tr.weight[oi * tr.in_dim + k];
                    }
                }
            }

            let mut d_logits = vec![0.0f32; n * n];
            for j in 0..n {
                let mut wdw_sum = 0.0f32;
                for &i in &rc.selected[j] {
                    wdw_sum += rc.weights[i * n + j] * d_weights[i * n + j];
                }
                for &i in &rc.selected[j] {
                    let wij = rc.weights[i * n + j];
                    d_logits[i * n + j] = wij * (d_weights[i * n + j] - wdw_sum);
                }
            }

            let rp = &router.route_proj;
            let n_sq = n * n;
            for i in 0..n_sq.min(rp.out_dim) {
                rg.route_proj_db[i] += d_logits[i];
                for j in 0..rp.in_dim.min(rc.proj_input.len()) {
                    rg.route_proj_dw[i * rp.in_dim + j] += d_logits[i] * rc.proj_input[j];
                }
            }
        } else {
            // Fixed connection synapse backward
            for (ci, conn) in cfg.connections.iter().enumerate().rev() {
                let d_obs = &d_region_obs[conn.to];
                let syn = &w.connection_synapses[ci];
                let src_input = &tc.connection_inputs[ci];

                let d_syn_out: Vec<f32> = d_obs.iter().take(syn.out_dim).copied()
                    .chain(std::iter::repeat(0.0)).take(syn.out_dim).collect();

                for i in 0..syn.out_dim {
                    grads.connection_db[ci][i] += d_syn_out[i];
                    for j in 0..syn.in_dim.min(src_input.len()) {
                        grads.connection_dw[ci][i * syn.in_dim + j] += d_syn_out[i] * src_input[j];
                    }
                }

                let mut d_src = vec![0.0f32; syn.in_dim];
                for j in 0..syn.in_dim {
                    for i in 0..syn.out_dim {
                        d_src[j] += d_syn_out[i] * syn.weight[i * syn.in_dim + j];
                    }
                }

                let mut src_offset = 0;
                for &from_idx in &conn.from {
                    let dim = w.regions[from_idx].config.d_model;
                    for k in 0..dim.min(d_src.len() - src_offset) {
                        d_region_activated[from_idx][k] += d_src[src_offset + k];
                    }
                    src_offset += dim;
                }

                if conn.receives_observation {
                    for j in 0..obs_dim.min(d_src.len() - src_offset) {
                        d_observation[j] += d_src[src_offset + j];
                    }
                }
            }
        }

        // obs_proj backward: for regions that used obs_projected (no explicit connection)
        // obs_proj_dw already accumulated. Also compute d_obs through obs_proj.
        // d_obs_proj_out is the gradient w.r.t. obs_projected — but we don't have it
        // separately per region. For regions without connections, their d_region_obs
        // IS d_obs_projected. Accumulate obs_proj backward → d_observation.
        for r in 0..n_regions {
            let has_conn = cfg.connections.iter().any(|c| c.to == r);
            if !has_conn && !d_region_obs[r].is_empty() {
                // This region used obs_projected = obs_proj.forward(observation)
                let d_proj_out = &d_region_obs[r];
                let proj_in = obs_dim;
                let proj_out = w.obs_proj.out_dim;
                for i in 0..proj_out.min(d_proj_out.len()) {
                    grads.obs_proj_db[i] += d_proj_out[i];
                    for j in 0..proj_in {
                        grads.obs_proj_dw[i * proj_in + j] += d_proj_out[i] * observation[j];
                        d_observation[j] += d_proj_out[i] * w.obs_proj.weight[i * proj_in + j];
                    }
                }
            }
        }
    }

    (loss, pred_class, d_observation)
}

// ─── AdamW Trainer ────────────────────────────────────────

/// Per-parameter AdamW state: first moment, second moment.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdamWBuf {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamWBuf {
    fn zeros(n: usize) -> Self {
        Self { m: vec![0.0; n], v: vec![0.0; n] }
    }

    fn step(&mut self, weights: &mut [f32], grads: &[f32], lr: f32, wd: f32, b1: f32, b2: f32, eps: f32, bc1: f32, bc2: f32) {
        for i in 0..weights.len() {
            let g = grads[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            weights[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * weights[i]);
        }
    }
}

/// AdamW optimizer state for the entire RegionalCtm.
/// Uses AdamW for outer weights (embeddings, connections, projections)
/// and the SDK's built-in SGD for inner CTM region weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalAdamW {
    pub step: usize,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    embed: AdamWBuf,
    conn_w: Vec<AdamWBuf>,
    conn_b: Vec<AdamWBuf>,
    obs_proj_w: AdamWBuf,
    obs_proj_b: AdamWBuf,
    global_decay: AdamWBuf,
    output_proj_w: AdamWBuf,
    output_proj_b: AdamWBuf,
}

impl RegionalAdamW {
    pub fn new(w: &RegionalWeights) -> Self {
        Self {
            step: 0,
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            grad_clip: 1.0,
            embed: AdamWBuf::zeros(w.embeddings.len()),
            conn_w: w.connection_synapses.iter().map(|s| AdamWBuf::zeros(s.weight.len())).collect(),
            conn_b: w.connection_synapses.iter().map(|s| AdamWBuf::zeros(s.bias.len())).collect(),
            obs_proj_w: AdamWBuf::zeros(w.obs_proj.weight.len()),
            obs_proj_b: AdamWBuf::zeros(w.obs_proj.bias.len()),
            global_decay: AdamWBuf::zeros(w.global_decay.len()),
            output_proj_w: AdamWBuf::zeros(w.output_proj.weight.len()),
            output_proj_b: AdamWBuf::zeros(w.output_proj.bias.len()),
        }
    }

    pub fn with_lr(mut self, lr: f32) -> Self { self.lr = lr; self }
    pub fn with_wd(mut self, wd: f32) -> Self { self.weight_decay = wd; self }
    pub fn with_clip(mut self, clip: f32) -> Self { self.grad_clip = clip; self }

    /// Apply accumulated gradients: AdamW for outer weights, SGD for inner CTM.
    pub fn step(&mut self, w: &mut RegionalWeights, grads: &RegionalGradients) {
        self.step += 1;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;
        let bc1 = 1.0 - b1.powi(self.step as i32);
        let bc2 = 1.0 - b2.powi(self.step as i32);

        // Gradient clipping (global norm)
        let mut total_sq = 0.0f32;
        let add_sq = |v: &[f32], s: &mut f32| { for x in v { *s += x * x; } };
        add_sq(&grads.embed_grad, &mut total_sq);
        add_sq(&grads.output_proj_dw, &mut total_sq);
        add_sq(&grads.output_proj_db, &mut total_sq);
        add_sq(&grads.obs_proj_dw, &mut total_sq);
        add_sq(&grads.obs_proj_db, &mut total_sq);
        for dw in &grads.connection_dw { add_sq(dw, &mut total_sq); }
        for db in &grads.connection_db { add_sq(db, &mut total_sq); }
        for rg in &grads.region_grads {
            add_sq(&rg.nlm_s1_w, &mut total_sq);
            add_sq(&rg.out_proj_w, &mut total_sq);
            add_sq(&rg.q_proj_w, &mut total_sq);
            add_sq(&rg.kv_proj_w, &mut total_sq);
        }
        let norm = total_sq.sqrt();
        let scale = if norm > self.grad_clip { self.grad_clip / norm } else { 1.0 };
        let lr = self.lr * scale;

        // Embeddings — AdamW (no weight decay)
        self.embed.step(&mut w.embeddings, &grads.embed_grad, lr, 0.0, b1, b2, eps, bc1, bc2);

        // Per-region CTM weights — SDK SGD (handles complex UNet + NLM structure)
        for (rw, rg) in w.regions.iter_mut().zip(&grads.region_grads) {
            rg.apply(rw, lr, self.grad_clip);
        }

        // Connections — AdamW
        for (i, syn) in w.connection_synapses.iter_mut().enumerate() {
            self.conn_w[i].step(&mut syn.weight, &grads.connection_dw[i], lr, wd, b1, b2, eps, bc1, bc2);
            self.conn_b[i].step(&mut syn.bias, &grads.connection_db[i], lr, 0.0, b1, b2, eps, bc1, bc2);
        }

        // Projections — AdamW
        self.obs_proj_w.step(&mut w.obs_proj.weight, &grads.obs_proj_dw, lr, wd, b1, b2, eps, bc1, bc2);
        self.obs_proj_b.step(&mut w.obs_proj.bias, &grads.obs_proj_db, lr, 0.0, b1, b2, eps, bc1, bc2);
        self.global_decay.step(&mut w.global_decay, &grads.global_decay_grad, lr, 0.0, b1, b2, eps, bc1, bc2);
        self.output_proj_w.step(&mut w.output_proj.weight, &grads.output_proj_dw, lr, wd, b1, b2, eps, bc1, bc2);
        self.output_proj_b.step(&mut w.output_proj.bias, &grads.output_proj_db, lr, 0.0, b1, b2, eps, bc1, bc2);
    }

    /// Save optimizer state.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        modgrad_persist::persist::save(self, path)
    }

    /// Load optimizer state.
    pub fn load(path: &str) -> std::io::Result<Self> {
        modgrad_persist::persist::load(path)
    }
}

/// Train one token: embed → forward → loss → backward → accumulate embed grads.
/// Does NOT apply gradients — caller accumulates across a sequence, then calls AdamW.step().
pub fn regional_train_token(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    token: usize,
    target: usize,
) -> (f32, usize) {
    let obs = w.embed(token).to_vec();
    let (loss, pred, d_obs) = regional_train_step_full(w, grads, &obs, target);

    // Accumulate d_obs into the embedding gradient for this token
    let d = w.config.raw_obs_dim;
    let offset = token * d;
    for j in 0..d {
        grads.embed_grad[offset + j] += d_obs[j];
    }

    (loss, pred)
}

/// Dream phase: free-running rollout with regret correction.
///
/// Generates `dream_len` tokens autoregressively from `seed_token`,
/// comparing each generated token against the `ground_truth` continuation.
/// The model uses its OWN predictions as input (free-running), not ground truth
/// (teacher forcing). This tests whether the internal world model is coherent.
///
/// Inspired by embryonic spontaneous neural activity — the network tests its
/// own circuits before birth. The dream gradient is weighted by `weight` to
/// avoid overwhelming the data-driven signal.
///
/// Returns (dream_loss, gradients accumulated into `grads`).
pub fn dream_step(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    seed_token: usize,
    ground_truth: &[usize],  // actual continuation from data
    dream_len: usize,
    weight: f32,              // gradient scaling (e.g. 0.3)
) -> f32 {
    let len = dream_len.min(ground_truth.len());
    let mut current_token = seed_token;
    let mut total_loss = 0.0f32;

    for d in 0..len {
        let obs = w.embed(current_token).to_vec();
        let (loss, pred, d_obs) = regional_train_step_full(w, grads, &obs, ground_truth[d]);

        // Scale gradients by dream weight
        let embed_d = w.config.raw_obs_dim;
        let offset = current_token * embed_d;
        for j in 0..embed_d {
            grads.embed_grad[offset + j] += d_obs[j] * weight;
        }

        total_loss += loss;
        // Free-running: use model's prediction, not ground truth
        current_token = pred;
    }

    total_loss / len.max(1) as f32
}

/// Multi-byte training: train on a chunk, all heads predict future bytes.
///
/// For a chunk [b0, b1, b2, ..., bN]:
///   Head 0 predicts b1 from b0, b2 from b1, etc. (standard next-byte)
///   Head 1 predicts b2 from b0, b3 from b1, etc. (2-ahead)
///   Head k predicts b(k+1) from b0, b(k+2) from b1, etc.
///
/// Loss = average across all heads. More gradient signal per token.
/// Returns (avg_loss, head0_correct_count).
pub fn multi_byte_train_step(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    chunk: &[u8],
) -> (f32, usize) {
    let n_heads = 1 + w.extra_heads.len(); // total heads
    let n = chunk.len();
    if n < n_heads + 1 { return (0.0, 0); }

    let mut total_loss = 0.0f32;
    let mut head0_correct = 0usize;
    let mut head0_count = 0usize;

    // For each position, train head 0 (next byte) through the existing path
    let positions = n - n_heads; // positions where ALL heads have valid targets
    for pos in 0..positions {
        let token = chunk[pos] as usize;

        // Head 0: predict chunk[pos+1]
        let target0 = chunk[pos + 1] as usize;
        let obs = w.embed(token).to_vec();
        let (loss0, pred0, d_obs) = regional_train_step_full(w, grads, &obs, target0);
        total_loss += loss0;
        if pred0 == target0 { head0_correct += 1; }
        head0_count += 1;

        // Embedding gradient
        let d = w.config.raw_obs_dim;
        let offset = token * d;
        for j in 0..d {
            grads.embed_grad[offset + j] += d_obs[j];
        }

        // Extra heads: predict chunk[pos+2], chunk[pos+3], etc.
        // Use the same global sync from the forward pass (already computed).
        // Each extra head is just a Linear on the same sync vector.
        // We get the sync from the last forward's state — it's in the grads' backward cache.
        // For simplicity: use output_proj's input (global_sync) to compute extra head losses.
        // The global_sync was computed during regional_train_step_full.
        // We can recover it from grads... actually we can't easily.
        //
        // Simpler: extra heads operate on the same logits space.
        // Compute extra losses from the observation embedding directly.
        // This is approximate but adds gradient signal to the embedding.
        for (h, head) in w.extra_heads.iter().enumerate() {
            let target_h = chunk[pos + h + 2] as usize;
            let logits = head.forward(&obs);
            let (loss_h, _) = cross_entropy_with_logits(&logits, target_h);
            total_loss += loss_h;
        }
    }

    let total_predictions = head0_count * n_heads;
    let avg_loss = if total_predictions > 0 {
        total_loss / total_predictions as f32
    } else {
        0.0
    };

    (avg_loss, head0_correct)
}

/// Cross-entropy loss from logits (used by extra heads).
fn cross_entropy_with_logits(logits: &[f32], target: usize) -> (f32, usize) {
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_s: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp_s.iter().sum();
    let loss = -(exp_s.get(target).copied().unwrap_or(1e-8) / sum).max(1e-8).ln();
    let pred = logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);
    (loss, pred)
}

// ─── Param count helper ────────────────────────────────────

impl RegionalWeights {
    pub fn print_summary(&self) {
        eprintln!("RegionalCtm: {} regions, {} connections",
            self.config.regions.len(), self.config.connections.len());
        for (i, r) in self.regions.iter().enumerate() {
            eprintln!("  {}: d_model={}, d_input={}, memory={}, params={}",
                self.config.region_names[i], r.config.d_model, r.config.d_input,
                r.config.memory_length, r.n_params());
        }
        eprintln!("  global_sync: {} pairs", self.config.n_global_sync);
        eprintln!("  vocab: {}", self.config.out_dims);
        eprintln!("  prediction heads: {} (1 + {} extra)", 1 + self.extra_heads.len(), self.extra_heads.len());
        eprintln!("  total params: {}", self.n_params());
    }
}

// ─── Multimodal tokenization helpers ──────────────────────

// ── Tokenization helpers ──────────────────────────────────

/// Timestamp token for a given time in seconds.
/// Quantized to 0.5s resolution. Clamps to max 199.5s.
#[inline]
pub fn timestamp_token(seconds: f32) -> usize {
    let slot = ((seconds / 0.5).round() as usize).min(TOKEN_TS_COUNT - 1);
    TOKEN_TS_OFFSET + slot
}

/// Image VQ codes → token sequence: <img> [codes] </img>
pub fn image_codes_to_tokens(codes: &[usize]) -> Vec<usize> {
    let mut t = Vec::with_capacity(codes.len() + 2);
    t.push(TOKEN_IMG_START);
    for &c in codes { t.push(TOKEN_IMG_OFFSET + c.min(TOKEN_IMG_CODES - 1)); }
    t.push(TOKEN_IMG_END);
    t
}

/// Audio VQ codes → token sequence: <aud> [codes] </aud>
pub fn audio_codes_to_tokens(codes: &[usize]) -> Vec<usize> {
    let mut t = Vec::with_capacity(codes.len() + 2);
    t.push(TOKEN_AUD_START);
    for &c in codes { t.push(TOKEN_AUD_OFFSET + c.min(TOKEN_AUD_CODES - 1)); }
    t.push(TOKEN_AUD_END);
    t
}

/// Text bytes → token indices.
pub fn text_to_tokens(text: &[u8]) -> Vec<usize> {
    text.iter().map(|&b| b as usize).collect()
}

/// A video frame with its timestamp and VQ codes.
pub struct VideoFrame {
    pub time_seconds: f32,
    pub image_codes: Vec<usize>,
}

/// Tokenize a video: frames at timestamps, interleaved with audio chunks.
///
/// Layout (Qwen3-VL inspired — text timestamps, not positional encoding):
/// ```text
/// <vid>
///   <0.00s> <img> [64 codes] </img>
///   <0.50s> <img> [64 codes] </img>
///   <1.00s> <aud> [75 codes] </aud>
///   <1.00s> <img> [64 codes] </img>
///   ...
/// </vid>
/// ```
pub fn video_to_tokens(
    frames: &[VideoFrame],
    audio_chunks: &[(f32, Vec<usize>)],  // (start_seconds, audio_codes)
) -> Vec<usize> {
    // Merge frames and audio into a single timeline
    struct Event { time: f32, tokens: Vec<usize> }
    let mut events: Vec<Event> = Vec::with_capacity(frames.len() + audio_chunks.len());

    for f in frames {
        let mut t = Vec::with_capacity(f.image_codes.len() + 3);
        t.push(timestamp_token(f.time_seconds));
        t.extend(image_codes_to_tokens(&f.image_codes));
        events.push(Event { time: f.time_seconds, tokens: t });
    }

    for (start, codes) in audio_chunks {
        let mut t = Vec::with_capacity(codes.len() + 3);
        t.push(timestamp_token(*start));
        t.extend(audio_codes_to_tokens(codes));
        events.push(Event { time: *start, tokens: t });
    }

    // Sort by time (stable: frames before audio at same timestamp)
    events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    let total: usize = events.iter().map(|e| e.tokens.len()).sum();
    let mut result = Vec::with_capacity(total + 2);
    result.push(TOKEN_VID_START);
    for e in &events { result.extend_from_slice(&e.tokens); }
    result.push(TOKEN_VID_END);
    result
}

// Note: extract_video_frames (codec-dependent) lives in the runtime, not the SDK.
// Use VideoFrame + video_to_tokens with your own codec to build video token sequences.

// ── Action tokenization ──────────────────────────────────

/// Quantize a coordinate (0.0..1.0) to a 256-level token.
#[inline]
pub fn coord_token(v: f32) -> usize {
    TOKEN_COORD_OFFSET + (v.clamp(0.0, 1.0) * 255.0) as usize
}

/// Mouse move action → token sequence: <act> mouse_move x y </act>
pub fn action_mouse_move(x: f32, y: f32) -> Vec<usize> {
    vec![ACT_START, ACT_MOUSE_MOVE, coord_token(x), coord_token(y), ACT_END]
}

/// Click action → token sequence.
pub fn action_click(x: f32, y: f32) -> Vec<usize> {
    vec![ACT_START, ACT_LEFT_CLICK, coord_token(x), coord_token(y), ACT_END]
}

/// Keyboard type action → token sequence: <act> key_type [byte tokens] </act>
pub fn action_type_text(text: &str) -> Vec<usize> {
    let mut t = vec![ACT_START, ACT_KEY_TYPE];
    for &b in text.as_bytes() { t.push(b as usize); }
    t.push(ACT_END);
    t
}

/// Special key action → token sequence.
pub fn action_key(key_token: usize) -> Vec<usize> {
    vec![ACT_START, key_token, ACT_END]
}

/// Modifier + key combo (e.g. ctrl+c).
pub fn action_modified_key(modifier: usize, key_byte: u8) -> Vec<usize> {
    vec![ACT_START, modifier, key_byte as usize, ACT_END]
}

// ── Neural Computer: interactive loop ────────────────────

/// NC runtime state: wraps the CTM state and provides the
/// update-and-render loop from the NC paper (Eq 2.1).
///
/// h_t = F_θ(h_{t-1}, x_t, u_t)    — CTM forward with observation + action
/// x_{t+1} ~ G_θ(h_t)              — sample next token from output logits
pub struct NeuralComputer {
    pub weights: RegionalWeights,
    pub state: RegionalState,
    /// Token history (the NC's observable trace).
    pub history: Vec<usize>,
    /// Maximum history length before truncation.
    pub max_history: usize,
    /// RNG state for sampling.
    rng_state: u64,
    /// Exit metrics from the last forward pass (for debug/telemetry).
    pub last_exit_lambdas: Vec<f32>,
    pub last_ticks_used: usize,
}

impl NeuralComputer {
    pub fn new(weights: RegionalWeights) -> Self {
        let state = RegionalState::new(&weights);
        // Seed RNG from system time for non-deterministic sampling
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        Self {
            weights,
            state,
            history: Vec::new(),
            max_history: 4096,
            rng_state: seed,
            last_exit_lambdas: Vec::new(),
            last_ticks_used: 0,
        }
    }

    /// Load from checkpoint.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let weights = RegionalWeights::load(path)?;
        Ok(Self::new(weights))
    }

    /// F_θ: process one input token, update state, return output logits.
    pub fn step(&mut self, token: usize) -> Vec<f32> {
        let obs = self.weights.embed(token).to_vec();
        let output = regional_forward(&self.weights, &mut self.state, &obs);
        self.history.push(token);
        if self.history.len() > self.max_history {
            self.history.drain(..self.history.len() - self.max_history);
        }
        self.last_exit_lambdas = output.exit_lambdas;
        self.last_ticks_used = output.ticks_used;
        output.predictions.into_iter().last().unwrap_or_default()
    }

    /// G_θ: sample next token from logits (argmax or temperature).
    pub fn sample(&mut self, logits: &[f32], temperature: f32) -> usize {
        if temperature <= 0.0 || logits.is_empty() {
            return logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0);
        }

        // Temperature-scaled softmax
        let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_s: Vec<f32> = logits.iter()
            .map(|&l| ((l - max_l) / temperature).exp()).collect();
        let sum: f32 = exp_s.iter().sum();

        // PCG-style RNG — advances on every call
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((self.rng_state >> 33) as f32) / (1u64 << 31) as f32;

        let mut cumsum = 0.0;
        for (i, &e) in exp_s.iter().enumerate() {
            cumsum += e / sum;
            if cumsum >= u { return i; }
        }
        logits.len() - 1
    }

    /// Feed a sequence of tokens (observation) and return logits after the last one.
    pub fn observe(&mut self, tokens: &[usize]) -> Vec<f32> {
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.step(t);
        }
        logits
    }

    /// Feed an action, then auto-generate response tokens until a delimiter or max length.
    /// This is the NC interactive loop: action → state update → generate response.
    pub fn act(&mut self, action_tokens: &[usize], max_response: usize, temperature: f32) -> Vec<usize> {
        // Feed action tokens
        let mut logits = self.observe(action_tokens);
        // Generate response
        let mut response = Vec::new();
        for _ in 0..max_response {
            let next = self.sample(&logits, temperature);
            response.push(next);
            self.history.push(next);
            // Stop on action_start (another action expected) or end delimiters
            if next == ACT_START || next == TOKEN_VID_END || next == TOKEN_IMG_END {
                break;
            }
            logits = self.step(next);
        }
        response
    }

    /// Feed text bytes as observation, generate text response.
    /// Text chat: feed input, generate text until stop condition.
    pub fn chat(&mut self, input: &str, max_tokens: usize, temperature: f32) -> String {
        let input_tokens = text_to_tokens(input.as_bytes());
        let mut logits = self.observe(&input_tokens);
        let mut response = Vec::new();
        let mut non_text_streak = 0;
        for _ in 0..max_tokens {
            let next = self.sample(&logits, temperature);
            if next < 256 {
                response.push(next as u8);
                non_text_streak = 0;
            } else if next == ACT_START || next == TOKEN_VID_START {
                break; // model wants to do something non-text
            } else {
                non_text_streak += 1;
                if non_text_streak > 5 { break; } // stuck in non-text tokens
            }
            logits = self.step(next);
        }
        String::from_utf8_lossy(&response).into_owned()
    }

    /// Reset the NC state (clear history and reinitialize CTM state).
    pub fn reset(&mut self) {
        self.state = RegionalState::new(&self.weights);
        self.history.clear();
    }

    /// Generate tokens of any modality. Returns raw token sequence.
    pub fn generate(&mut self, input_tokens: &[usize], max_tokens: usize, temperature: f32) -> Vec<usize> {
        let mut logits = self.observe(input_tokens);
        let mut output = Vec::new();
        for _ in 0..max_tokens {
            let next = self.sample(&logits, temperature);
            output.push(next);
            logits = self.step(next);
        }
        output
    }
}
