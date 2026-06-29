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
use crate::rayon_shim::*;
use modgrad_compute::neuron::{Linear, SimpleRng};
use wincode_derive::{SchemaRead, SchemaWrite};
use crate::config::CtmConfig;
use crate::weights::{CtmWeights, CtmState};
use crate::forward::ctm_forward;
use crate::train::{Ctm, CtmCache, backward_from_activated};
use modgrad_traits::{Brain, LossFn, TokenInput};

/// Merge primitive for connection synapses with multiple incoming
/// edges to the same target region.
///
/// **Choice: elementwise add** (not concat-then-down). Add keeps the
/// synapse Linear sized to a single source's d_input, lets the brain
/// gain new edges without re-allocating weights, and matches the
/// "signals sum at the postsynaptic site" biological default.
/// Concat-then-down would force every multi-source target to grow its
/// input by `Σ source dims` and re-train the down-projection from
/// scratch — too invasive for this slice.
///
/// Used by every regional forward path (training and inference, host
/// and resident). When the target slot is empty (first incoming edge
/// to this region this tick), assign-by-move; otherwise add
/// elementwise. Length mismatches are guarded — we only add over
/// `min(slot.len(), projected.len())`, so a malformed graph degrades
/// gracefully rather than panicking the brain.
#[inline]
fn merge_into_region_obs(slot: &mut Vec<f32>, projected: Vec<f32>) {
    if slot.is_empty() {
        *slot = projected;
    } else {
        let n = slot.len().min(projected.len());
        for i in 0..n { slot[i] += projected[i]; }
    }
}

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
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct Connection {
    /// Source region indices — their activated states are concatenated.
    pub from: Vec<usize>,
    /// Target region index.
    pub to: usize,
    /// If true, the raw observation is concatenated to the source activations
    /// before the synapse projection. This is how external input enters the graph.
    #[serde(default)]
    pub receives_observation: bool,
    /// Which observation scale this connection consumes when
    /// `receives_observation` is true. The encoder may emit multiple
    /// scales (V1, V2, V4 for visual cortex); this index selects
    /// which one feeds this connection. Default 0 = primary scale,
    /// matching back-compat single-scale encoders.
    ///
    /// Biology: V1 projects directly to many brain areas (pulvinar,
    /// FEF, parietal) — not only via V2/V4. Per-connection scale
    /// selection lets a brain treat different regions as direct V1
    /// vs V4 consumers, mirroring real cortical wiring.
    #[serde(default)]
    pub observation_scale: usize,
}

/// Toggleable auxiliary losses inspired by neuroscience.
/// Each adds a gradient signal to specific regions on top of the main BPTT loss.
/// All default to off — enable to test if they help.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
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
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
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
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
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
    /// Per-destination pre-from_route routed activation (weighted sum of
    /// projected sources). Shape: [n_regions][d_route]. Needed to compute
    /// `from_route[j]` weight gradients in backward.
    pub routed: Vec<Vec<f32>>,
    /// Per-source region outputs observed at forward time (the input to
    /// `to_route[i]`). Shape: [n_regions][d_model_i]. Needed to compute
    /// `to_route[i]` weight gradients in backward.
    pub region_outputs: Vec<Vec<f32>>,
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

    // Scratch buffers — allocated once in `zeros`, reused across every
    // router backward call in the BPTT outer-tick loop. Flat layouts
    // replace the old `Vec<Vec<f32>>` nesting; each router-backward just
    // `.fill(0.0)`s these and indexes with `j * d + k` / `i * n + j`.
    //
    // These scratches have exactly the lifetime of the gradients they
    // feed into (both cleared at the start of a batch), so owning them
    // here is a straight alloc/lifecycle improvement, not a semantic
    // change.
    pub scratch_d_routed: Vec<f32>,       // n * d_route
    pub scratch_d_projected: Vec<f32>,    // n * d_route
    pub scratch_d_weights: Vec<f32>,      // n * n
    pub scratch_d_logits: Vec<f32>,       // n * n
    pub scratch_d_proj_input: Vec<f32>,   // route_proj.in_dim
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
        let route_proj = Linear::new(n_sync + t, n * n);

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
        let mut routed_cache: Vec<Vec<f32>> = Vec::with_capacity(n);

        let mut rng = if training {
            Some(SimpleRng::new(tick as u64 ^ 0xCAFE))
        } else {
            None
        };

        for j in 0..n {
            // Softmax over source axis for destination j
            let col: Vec<f32> = (0..n).map(|i| {
                let v = logits[i * n + j];
                if v.is_finite() { v } else { 0.0 }
            }).collect();
            let max_val = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let max_val = if max_val.is_finite() { max_val } else { 0.0 };
            let exp: Vec<f32> = col.iter().map(|&v| (v - max_val).clamp(-30.0, 0.0).exp()).collect();
            let sum: f32 = exp.iter().sum::<f32>().max(1e-8);
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
            routed_cache.push(routed);
            routed_inputs.push(dest_input);
        }

        // Cache per-source inputs for backward (to_route weight grads).
        let region_outputs_cache: Vec<Vec<f32>> =
            region_outputs.iter().map(|v| v.clone()).collect();

        let cache = RouterCache {
            proj_input, logits, weights, selected, projected_sources,
            routed: routed_cache,
            region_outputs: region_outputs_cache,
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
        let n = router.n_regions;
        let d = router.config.d_route;
        let rp_in = router.route_proj.in_dim;
        Self {
            to_route_dw: router.to_route.iter().map(|l| vec![0.0; l.weight.len()]).collect(),
            to_route_db: router.to_route.iter().map(|l| vec![0.0; l.bias.len()]).collect(),
            from_route_dw: router.from_route.iter().map(|l| vec![0.0; l.weight.len()]).collect(),
            from_route_db: router.from_route.iter().map(|l| vec![0.0; l.bias.len()]).collect(),
            tick_embed_grad: vec![0.0; router.tick_embed.len()],
            route_proj_dw: vec![0.0; router.route_proj.weight.len()],
            route_proj_db: vec![0.0; router.route_proj.bias.len()],

            scratch_d_routed: vec![0.0; n * d],
            scratch_d_projected: vec![0.0; n * d],
            scratch_d_weights: vec![0.0; n * n],
            scratch_d_logits: vec![0.0; n * n],
            scratch_d_proj_input: vec![0.0; rp_in],
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
        // Scratch buffers are fresh-zeroed at the start of each backward
        // call, not here — keeping `zero()` as "reset accumulators" only.
    }

    /// Append router accumulator gradients into `buf` in stable order.
    /// Scratch buffers are intentionally excluded — they hold transient
    /// per-call state, not a learnable parameter's gradient.
    pub fn flatten_into(&self, buf: &mut Vec<f32>) {
        for v in &self.to_route_dw { buf.extend_from_slice(v); }
        for v in &self.to_route_db { buf.extend_from_slice(v); }
        for v in &self.from_route_dw { buf.extend_from_slice(v); }
        for v in &self.from_route_db { buf.extend_from_slice(v); }
        buf.extend_from_slice(&self.tick_embed_grad);
        buf.extend_from_slice(&self.route_proj_dw);
        buf.extend_from_slice(&self.route_proj_db);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
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
    /// Raw observation dimension (input to the system). For multi-
    /// scale encoders, this is the **concatenated** length across
    /// all scales — i.e. `obs_scale_dims.iter().sum()`. Single-scale
    /// configs satisfy `raw_obs_dim == obs_scale_dims[0]`.
    pub raw_obs_dim: usize,
    /// Per-scale observation dimensions when the encoder is
    /// multi-scale (e.g. V1 / V2 / V4 for visual cortex). The brain
    /// concatenates all scales into a single flat `input.tokens`
    /// vector and connections slice into it by `observation_scale`.
    /// Defaults to `vec![raw_obs_dim]` for single-scale back-compat.
    #[serde(default)]
    pub obs_scale_dims: Vec<usize>,
    /// Auxiliary bio-inspired losses (optional, toggleable).
    #[serde(default)]
    pub aux_losses: AuxLossConfig,
    /// Learned inter-region router (MoS-style). When Some, replaces
    /// fixed connection topology with dynamic, tick-conditioned routing.
    /// When None, uses fixed connections (backward compatible).
    #[serde(default)]
    pub router: Option<RouterConfig>,
    /// Cerebellum mode. When not Ctm, the cerebellum region uses a frozen
    /// forward model instead of the CTM tick loop. Default: Ctm.
    #[serde(default)]
    pub cereb_mode: crate::cerebellum::CerebMode,
    /// OUTPUT-local move head: when `Some(r)`, the move/output logits are
    /// read DIRECTLY from region `r`'s own synchronization readout
    /// (`sync_out`) via a learned `output_local_head: Linear(n_synch_out
    /// → out_dims)`, instead of from the global-sync pooling over all
    /// regions. Designed for spatial tasks (e.g. per-cell maze moves)
    /// where the holistic global sync averages away a region's localized
    /// per-position signal. `None` (default) keeps the legacy global-sync
    /// readout, so existing brains/weights are unaffected.
    #[serde(default)]
    pub output_local_region: Option<usize>,
}

impl RegionalConfig {
    /// `(start, len)` slice of the flat observation buffer for the
    /// given scale index. Used by forward and backward to extract
    /// each connection's observation portion. Falls back to
    /// `(0, raw_obs_dim)` when `obs_scale_dims` is empty (legacy
    /// single-scale configs that didn't set the field).
    pub fn obs_scale_slice(&self, scale_idx: usize) -> (usize, usize) {
        if self.obs_scale_dims.is_empty() {
            return (0, self.raw_obs_dim);
        }
        let mut start = 0;
        for i in 0..scale_idx {
            start += self.obs_scale_dims[i];
        }
        (start, self.obs_scale_dims[scale_idx])
    }

    /// Default 8-region brain topology (~81M params).
    ///
    /// Cortical: d_model=512, memory=64. Subcortical: d_model=64, memory=32.
    /// Fits in 8GB VRAM for VRAM-resident training.
    pub fn eight_region(obs_dim: usize, out_dims: usize, ticks: usize) -> Self {
        const INPUT: usize = 0;
        const ATTENTION: usize = 1;
        const OUTPUT: usize = 2;
        const MOTOR: usize = 3;
        const CEREBELLUM: usize = 4;
        const BASAL_GANGLIA: usize = 5;
        const INSULA: usize = 6;
        const HIPPOCAMPUS: usize = 7;

        let d = obs_dim;

        let regions = vec![
            CtmConfig::region("input", 512, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("attention", 512, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("output", 512, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("motor", 512, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("cerebellum", 64, d, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("basal_ganglia", 64, d, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("insula", 64, d, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("hippocampus", 64, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.15, threshold: 0.99 }),
        ];

        let names = vec![
            "input", "attention", "output", "motor",
            "cerebellum", "basal_ganglia", "insula", "hippocampus",
        ].into_iter().map(String::from).collect();

        let connections = vec![
            Connection { from: vec![MOTOR], to: INPUT, receives_observation: true, observation_scale: 0 },
            // ATTENTION reads from cortex (INPUT) AND from hippocampal
            // episodic memory (HIPPOCAMPUS). Memory-guided attention:
            // episodic state accumulated in hippocampus modulates the
            // attention region's input, so the {attention → output →
            // motor} prediction loop becomes content-causal in past
            // observation history. Without this edge hippocampus is a
            // memory sink (entries accumulate but never reach
            // predictions); see the
            // `episodic_memory_propagates_to_prediction` test.
            Connection { from: vec![INPUT, HIPPOCAMPUS], to: ATTENTION, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![ATTENTION], to: OUTPUT, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: MOTOR, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![MOTOR], to: CEREBELLUM, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: BASAL_GANGLIA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![HIPPOCAMPUS], to: INSULA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![INPUT, ATTENTION, OUTPUT, MOTOR], to: HIPPOCAMPUS, receives_observation: false, observation_scale: 0 },
        ];

        let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();
        let n_global_sync = total_neurons.min(1024);

        Self {
            regions,
            region_names: names,
            connections,
            outer_ticks: ticks,
            exit_strategy: crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 },
            n_global_sync,
            out_dims,
            raw_obs_dim: obs_dim, obs_scale_dims: vec![obs_dim],
            aux_losses: AuxLossConfig::default(),
            router: Some(RouterConfig::default()),
            cereb_mode: Default::default(),
            output_local_region: None,
        }
    }

    /// Small 8-region brain — all features, param-budgeted for benchmarking.
    /// d_input per region is proportional to d_model, not raw observation dim.
    /// Connection synapses handle the dimension reduction between regions.
    pub fn eight_region_small(obs_dim: usize, out_dims: usize, ticks: usize) -> Self {
        const INPUT: usize = 0;
        const ATTENTION: usize = 1;
        const OUTPUT: usize = 2;
        const MOTOR: usize = 3;
        const CEREBELLUM: usize = 4;
        const BASAL_GANGLIA: usize = 5;
        const INSULA: usize = 6;
        const HIPPOCAMPUS: usize = 7;

        // d_input per region: half of d_model (connection synapses project to this)
        let regions = vec![
            CtmConfig::region("input", 32, 16, 8, false, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("attention", 32, 16, 8, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("output", 32, 16, 8, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("motor", 32, 16, 8, false, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("cerebellum", 8, 8, 4, false, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("basal_ganglia", 8, 8, 4, false, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("insula", 8, 8, 4, false, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("hippocampus", 8, 8, 8, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.15, threshold: 0.99 }),
        ];

        let names = vec![
            "input", "attention", "output", "motor",
            "cerebellum", "basal_ganglia", "insula", "hippocampus",
        ].into_iter().map(String::from).collect();

        let connections = vec![
            Connection { from: vec![MOTOR], to: INPUT, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![INPUT], to: ATTENTION, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![ATTENTION], to: OUTPUT, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: MOTOR, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![MOTOR], to: CEREBELLUM, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: BASAL_GANGLIA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![HIPPOCAMPUS], to: INSULA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![INPUT, ATTENTION, OUTPUT, MOTOR], to: HIPPOCAMPUS, receives_observation: false, observation_scale: 0 },
        ];

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
            raw_obs_dim: obs_dim, obs_scale_dims: vec![obs_dim],
            aux_losses: AuxLossConfig::default(),
            router: None, // no router at this scale
            cereb_mode: Default::default(),
            output_local_region: None,
        }
    }

    /// 8-region preset tuned for market-making MLPs: small enough to
    /// fit on Cpu (40k params), output dim sized for the barbell-LP
    /// motor (`out_dims = 12 = 4 rungs × 3 params`). Used by
    /// `examples/penumbra_arena` (penumbra_train, penumbra_live_arena,
    /// the synthetic-replay arena binary). Single-scale obs.
    ///
    /// Cortical regions (INPUT/ATTENTION/OUTPUT/MOTOR) get
    /// `d_model_cortical=32` so the brain has enough capacity to
    /// represent the 24-dim observation (price + spread + inventory
    /// + 4-block history + bid/ask depth at 4 offsets); subcortical
    /// regions stay at `d_model_sub=16` for cost.
    ///
    /// `memory_length=8` for all regions (matches the historical
    /// shape that existing checkpoints were trained against). A
    /// future preset variant could expand HIPPOCAMPUS specifically
    /// for fill-history memory once retraining lands.
    pub fn eight_region_mm(obs_dim: usize, out_dims: usize, ticks: usize) -> Self {
        const INPUT: usize = 0;
        const ATTENTION: usize = 1;
        const OUTPUT: usize = 2;
        const MOTOR: usize = 3;
        const CEREBELLUM: usize = 4;
        const BASAL_GANGLIA: usize = 5;
        const INSULA: usize = 6;
        const HIPPOCAMPUS: usize = 7;

        let d_model_cortical: usize = 32;
        let d_model_sub: usize = 16;
        let mk = |d_model: usize| CtmConfig {
            iterations: 1,
            d_model,
            d_input: 16,
            heads: 2,
            n_synch_out: 8,
            n_synch_action: 8,
            synapse_depth: 1,
            memory_length: 8,
            deep_nlms: false,
            memory_hidden_dims: 4,
            out_dims,
            n_random_pairing_self: 0,
            min_width: 8,
            ..Default::default()
        };

        let regions = vec![
            mk(d_model_cortical),  // 0 INPUT
            mk(d_model_cortical),  // 1 ATTENTION
            mk(d_model_cortical),  // 2 OUTPUT
            mk(d_model_cortical),  // 3 MOTOR
            mk(d_model_sub),       // 4 CEREBELLUM
            mk(d_model_sub),       // 5 BASAL_GANGLIA
            mk(d_model_sub),       // 6 INSULA
            mk(d_model_sub),       // 7 HIPPOCAMPUS
        ];
        let region_names: Vec<String> = [
            "input","attention","output","motor",
            "cerebellum","basal_ganglia","insula","hippocampus",
        ].into_iter().map(String::from).collect();

        let connections = vec![
            Connection { from: vec![MOTOR], to: INPUT, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![INPUT, HIPPOCAMPUS], to: ATTENTION, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![ATTENTION], to: OUTPUT, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: MOTOR, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![MOTOR], to: CEREBELLUM, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: BASAL_GANGLIA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![HIPPOCAMPUS], to: INSULA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![INPUT, ATTENTION, OUTPUT, MOTOR], to: HIPPOCAMPUS, receives_observation: false, observation_scale: 0 },
        ];

        let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();
        Self {
            regions,
            region_names,
            connections,
            outer_ticks: ticks,
            exit_strategy: crate::config::ExitStrategy::None,
            n_global_sync: total_neurons.min(64),
            out_dims,
            raw_obs_dim: obs_dim,
            obs_scale_dims: vec![obs_dim],
            aux_losses: AuxLossConfig::default(),
            router: None,
            cereb_mode: Default::default(),
            output_local_region: None,
        }
    }

    /// Multi-scale variant of `eight_region_small`. Mirrors the
    /// biological wiring where V1 projects directly to motor/parietal
    /// and V2 to attention areas, not only via V4 → IT → PFC.
    ///
    /// `obs_scale_dims` order matches `Encoder::token_dims()` for
    /// `VisualCortex`: `[v4_dim, v2_dim, v1_dim]`. Scale 0 is V4
    /// (high-level semantic), scale 1 is V2 (mid contour), scale 2
    /// is V1 (fine spatial / oriented edges).
    ///
    /// Connection routing:
    /// - INPUT  ← MOTOR + V4 (high-level integration)
    /// - ATTENTION ← INPUT + V2 (mid-level gating signal)
    /// - OUTPUT ← ATTENTION (no obs needed)
    /// - MOTOR ← OUTPUT + V1 (fine-grained motor uses raw spatial)
    /// - CEREBELLUM ← MOTOR + V1 (predictive timing wants raw signal)
    /// - others: lateral, no observation
    pub fn eight_region_small_multiscale(
        obs_scale_dims: &[usize],
        out_dims: usize,
        ticks: usize,
    ) -> Self {
        assert!(obs_scale_dims.len() >= 3,
            "multi-scale config expects at least 3 scales [V4, V2, V1]");
        let has_subcortical = obs_scale_dims.len() >= 4;

        const INPUT: usize = 0;
        const ATTENTION: usize = 1;
        const OUTPUT: usize = 2;
        const MOTOR: usize = 3;
        const CEREBELLUM: usize = 4;
        const BASAL_GANGLIA: usize = 5;
        const INSULA: usize = 6;
        const HIPPOCAMPUS: usize = 7;

        let regions = vec![
            CtmConfig::region("input", 32, 16, 8, false, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("attention", 32, 16, 8, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("output", 32, 16, 8, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("motor", 32, 16, 8, false, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("cerebellum", 8, 8, 4, false, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("basal_ganglia", 8, 8, 4, false, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("insula", 8, 8, 4, false, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("hippocampus", 8, 8, 8, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.15, threshold: 0.99 }),
        ];

        let names = vec![
            "input", "attention", "output", "motor",
            "cerebellum", "basal_ganglia", "insula", "hippocampus",
        ].into_iter().map(String::from).collect();

        // INSULA wires to scale 3 (raw retinal ganglion) when the
        // encoder exposes it — the subcortical superior-colliculus →
        // pulvinar → amygdala fast path. With only 3 scales, INSULA
        // stays on its old hippocampus-driven (lateral, no obs) path.
        let insula_obs_scale: usize = if has_subcortical { 3 } else { 0 };
        let insula_receives_obs = has_subcortical;

        let connections = vec![
            Connection { from: vec![MOTOR], to: INPUT,
                receives_observation: true, observation_scale: 0 },     // V4
            Connection { from: vec![INPUT], to: ATTENTION,
                receives_observation: true, observation_scale: 1 },     // V2
            Connection { from: vec![ATTENTION], to: OUTPUT,
                receives_observation: false, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: MOTOR,
                receives_observation: true, observation_scale: 2 },     // V1
            Connection { from: vec![MOTOR], to: CEREBELLUM,
                receives_observation: true, observation_scale: 2 },     // V1
            Connection { from: vec![OUTPUT], to: BASAL_GANGLIA,
                receives_observation: false, observation_scale: 0 },
            Connection { from: vec![HIPPOCAMPUS], to: INSULA,
                receives_observation: insula_receives_obs,
                observation_scale: insula_obs_scale },                  // Ganglion (subcortical)
            Connection { from: vec![INPUT, ATTENTION, OUTPUT, MOTOR], to: HIPPOCAMPUS,
                receives_observation: false, observation_scale: 0 },
        ];

        let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();
        let n_global_sync = total_neurons.min(256);
        let raw_obs_dim: usize = obs_scale_dims.iter().sum();

        Self {
            regions,
            region_names: names,
            connections,
            outer_ticks: ticks,
            exit_strategy: crate::config::ExitStrategy::AdaptiveGate {
                beta: 0.1, threshold: 0.99
            },
            n_global_sync,
            out_dims,
            raw_obs_dim,
            obs_scale_dims: obs_scale_dims.to_vec(),
            aux_losses: AuxLossConfig::default(),
            router: None,
            cereb_mode: Default::default(),
            output_local_region: None,
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
            Connection { from: vec![3], to: 0, receives_observation: true, observation_scale: 0 },  // motor + obs → input
            Connection { from: vec![0], to: 1, receives_observation: false, observation_scale: 0 }, // input → attention
            Connection { from: vec![1], to: 2, receives_observation: false, observation_scale: 0 }, // attention → output
            Connection { from: vec![2], to: 3, receives_observation: false, observation_scale: 0 }, // output → motor
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
            raw_obs_dim: obs_dim, obs_scale_dims: vec![obs_dim],
            aux_losses: AuxLossConfig::default(),
            router: None,
            cereb_mode: Default::default(),
            output_local_region: None,
        }
    }

    /// Medium 8-region: balanced CPU+GPU model (~55M params).
    /// Prediction heads use GPU (33M flops), SuperLinear on CPU rayon.
    pub fn eight_region_medium(obs_dim: usize, out_dims: usize, ticks: usize) -> Self {
        const INPUT: usize = 0;
        const ATTENTION: usize = 1;
        const OUTPUT: usize = 2;
        const MOTOR: usize = 3;
        const CEREBELLUM: usize = 4;
        const BASAL_GANGLIA: usize = 5;
        const INSULA: usize = 6;
        const HIPPOCAMPUS: usize = 7;

        let d = obs_dim;
        let regions = vec![
            CtmConfig::region("input", 256, d, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("attention", 256, d, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("output", 256, d, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("motor", 256, d, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("cerebellum", 32, d, 16, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("basal_ganglia", 32, d, 16, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("insula", 32, d, 16, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("hippocampus", 32, d, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.15, threshold: 0.99 }),
        ];

        let names = vec![
            "input", "attention", "output", "motor",
            "cerebellum", "basal_ganglia", "insula", "hippocampus",
        ].into_iter().map(String::from).collect();

        let connections = vec![
            Connection { from: vec![MOTOR], to: INPUT, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![INPUT], to: ATTENTION, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![ATTENTION], to: OUTPUT, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: MOTOR, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![MOTOR], to: CEREBELLUM, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: BASAL_GANGLIA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![HIPPOCAMPUS], to: INSULA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![INPUT, ATTENTION, OUTPUT, MOTOR], to: HIPPOCAMPUS, receives_observation: false, observation_scale: 0 },
        ];

        let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();
        let n_global_sync = total_neurons.min(512);

        Self {
            regions,
            region_names: names,
            connections,
            outer_ticks: ticks,
            exit_strategy: crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 },
            n_global_sync,
            out_dims,
            raw_obs_dim: obs_dim, obs_scale_dims: vec![obs_dim],
            aux_losses: AuxLossConfig::default(),
            router: Some(RouterConfig::default()),
            cereb_mode: Default::default(),
            output_local_region: None,
        }
    }

    /// Multi-scale variant of the medium 8-region. GPU-stress topology
    /// (cortical d_model=512, subcortical d_model=64) — sized to keep
    /// SuperLinear above the GPU dispatch threshold so a `--features
    /// rocm` run actually exercises the matmul path.
    ///
    /// `obs_scale_dims` order matches `Encoder::token_dims()` for
    /// `VisualCortex`: `[v4_dim, v2_dim, v1_dim]`. Scale 0 is V4
    /// (high-level semantic), scale 1 is V2 (mid contour), scale 2
    /// is V1 (fine spatial / oriented edges). A 4th entry, when
    /// supplied, is the subcortical raw-ganglion fast path consumed
    /// by INSULA.
    ///
    /// Connection routing mirrors `eight_region_small_multiscale`:
    /// - INPUT  ← MOTOR + V4 (high-level integration)
    /// - ATTENTION ← INPUT + V2 (mid-level gating signal)
    /// - OUTPUT ← ATTENTION (no obs needed)
    /// - MOTOR ← OUTPUT + V1 (fine-grained motor uses raw spatial)
    /// - CEREBELLUM ← MOTOR + V1 (predictive timing wants raw signal)
    /// - INSULA ← HIPPOCAMPUS (+ subcortical ganglion when present)
    /// - HIPPOCAMPUS ← INPUT/ATTENTION/OUTPUT/MOTOR (lateral, no obs)
    pub fn eight_region_medium_multiscale(
        obs_scale_dims: &[usize],
        out_dims: usize,
        ticks: usize,
    ) -> Self {
        assert!(obs_scale_dims.len() >= 3,
            "multi-scale config expects at least 3 scales [V4, V2, V1]");
        let has_subcortical = obs_scale_dims.len() >= 4;

        const INPUT: usize = 0;
        const ATTENTION: usize = 1;
        const OUTPUT: usize = 2;
        const MOTOR: usize = 3;
        const CEREBELLUM: usize = 4;
        const BASAL_GANGLIA: usize = 5;
        const INSULA: usize = 6;
        const HIPPOCAMPUS: usize = 7;

        // Region sizes tuned for GPU stress: cortical d_model=512 keeps
        // SuperLinear flops above the GPU dispatch crossover, subcortical
        // d_model=64 stays modest. d_input is half of d_model — connection
        // synapses project incoming inter-region signal to that width.
        let regions = vec![
            CtmConfig::region("input", 512, 256, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("attention", 512, 256, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("output", 512, 256, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("motor", 512, 256, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("cerebellum", 64, 64, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("basal_ganglia", 64, 64, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("insula", 64, 64, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("hippocampus", 64, 64, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.15, threshold: 0.99 }),
        ];

        let names = vec![
            "input", "attention", "output", "motor",
            "cerebellum", "basal_ganglia", "insula", "hippocampus",
        ].into_iter().map(String::from).collect();

        // INSULA wires to scale 3 (raw retinal ganglion) when the
        // encoder exposes it — the subcortical superior-colliculus →
        // pulvinar → amygdala fast path. With only 3 scales, INSULA
        // stays on its hippocampus-driven (lateral, no obs) path.
        let insula_obs_scale: usize = if has_subcortical { 3 } else { 0 };
        let insula_receives_obs = has_subcortical;

        let connections = vec![
            Connection { from: vec![MOTOR], to: INPUT,
                receives_observation: true, observation_scale: 0 },     // V4
            Connection { from: vec![INPUT], to: ATTENTION,
                receives_observation: true, observation_scale: 1 },     // V2
            Connection { from: vec![ATTENTION], to: OUTPUT,
                receives_observation: false, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: MOTOR,
                receives_observation: true, observation_scale: 2 },     // V1
            Connection { from: vec![MOTOR], to: CEREBELLUM,
                receives_observation: true, observation_scale: 2 },     // V1
            Connection { from: vec![OUTPUT], to: BASAL_GANGLIA,
                receives_observation: false, observation_scale: 0 },
            Connection { from: vec![HIPPOCAMPUS], to: INSULA,
                receives_observation: insula_receives_obs,
                observation_scale: insula_obs_scale },                  // Ganglion (subcortical)
            Connection { from: vec![INPUT, ATTENTION, OUTPUT, MOTOR], to: HIPPOCAMPUS,
                receives_observation: false, observation_scale: 0 },
        ];

        let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();
        let n_global_sync = total_neurons.min(1024);
        let raw_obs_dim: usize = obs_scale_dims.iter().sum();

        Self {
            regions,
            region_names: names,
            connections,
            outer_ticks: ticks,
            exit_strategy: crate::config::ExitStrategy::AdaptiveGate {
                beta: 0.1, threshold: 0.99
            },
            n_global_sync,
            out_dims,
            raw_obs_dim,
            obs_scale_dims: obs_scale_dims.to_vec(),
            aux_losses: AuxLossConfig::default(),
            router: Some(RouterConfig::default()),
            cereb_mode: Default::default(),
            output_local_region: None,
        }
    }

    /// Billion-scale 8-region (~1B params, d_model=1024).
    /// Cortical: d_model=1024, memory=128. Requires ~19GB CPU RAM.
    /// SuperLinear at 134M flops per stage — fully GPU-bound.
    pub fn eight_region_billion(obs_dim: usize, out_dims: usize, ticks: usize) -> Self {
        const INPUT: usize = 0;
        const ATTENTION: usize = 1;
        const OUTPUT: usize = 2;
        const MOTOR: usize = 3;
        const CEREBELLUM: usize = 4;
        const BASAL_GANGLIA: usize = 5;
        const INSULA: usize = 6;
        const HIPPOCAMPUS: usize = 7;

        let d = obs_dim;
        let regions = vec![
            CtmConfig::region("input", 1024, d, 128, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("attention", 1024, d, 128, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("output", 1024, d, 128, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("motor", 1024, d, 128, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("cerebellum", 128, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("basal_ganglia", 128, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("insula", 128, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("hippocampus", 128, d, 128, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.15, threshold: 0.99 }),
        ];

        let names = vec![
            "input", "attention", "output", "motor",
            "cerebellum", "basal_ganglia", "insula", "hippocampus",
        ].into_iter().map(String::from).collect();

        let connections = vec![
            Connection { from: vec![MOTOR], to: INPUT, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![INPUT], to: ATTENTION, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![ATTENTION], to: OUTPUT, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: MOTOR, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![MOTOR], to: CEREBELLUM, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: BASAL_GANGLIA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![HIPPOCAMPUS], to: INSULA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![INPUT, ATTENTION, OUTPUT, MOTOR], to: HIPPOCAMPUS, receives_observation: false, observation_scale: 0 },
        ];

        let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();
        let n_global_sync = total_neurons.min(2048);

        Self {
            regions, region_names: names, connections,
            outer_ticks: ticks,
            exit_strategy: crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 },
            n_global_sync, out_dims, raw_obs_dim: obs_dim, obs_scale_dims: vec![obs_dim],
            aux_losses: AuxLossConfig::default(),
            router: Some(RouterConfig::default()),
            cereb_mode: Default::default(),
            output_local_region: None,
        }
    }

    /// 8-region brain with parameterised cortex `d_model`. Same topology as
    /// `eight_region_billion`; `cortex_d_model` controls the per-region
    /// CTM hidden width (input/attention/output/motor) so callers can
    /// sweep matvec compute against per-dispatch overhead. Subcortical
    /// regions stay at d_model=128 (their job is small/fast routing).
    pub fn eight_region_mega(
        obs_dim: usize, out_dims: usize, ticks: usize, cortex_d_model: usize,
    ) -> Self {
        const INPUT: usize = 0;
        const ATTENTION: usize = 1;
        const OUTPUT: usize = 2;
        const MOTOR: usize = 3;
        const CEREBELLUM: usize = 4;
        const BASAL_GANGLIA: usize = 5;
        const INSULA: usize = 6;
        const HIPPOCAMPUS: usize = 7;

        let d = obs_dim;
        let cm = cortex_d_model;
        let mem_cortex = (cm / 8).max(64);
        let regions = vec![
            CtmConfig::region("input", cm, d, mem_cortex, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("attention", cm, d, mem_cortex, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("output", cm, d, mem_cortex, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("motor", cm, d, mem_cortex, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("cerebellum", 128, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("basal_ganglia", 128, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("insula", 128, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("hippocampus", 128, d, 128, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.15, threshold: 0.99 }),
        ];

        let names = vec![
            "input", "attention", "output", "motor",
            "cerebellum", "basal_ganglia", "insula", "hippocampus",
        ].into_iter().map(String::from).collect();

        let connections = vec![
            Connection { from: vec![MOTOR], to: INPUT, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![INPUT], to: ATTENTION, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![ATTENTION], to: OUTPUT, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: MOTOR, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![MOTOR], to: CEREBELLUM, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: BASAL_GANGLIA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![HIPPOCAMPUS], to: INSULA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![INPUT, ATTENTION, OUTPUT, MOTOR], to: HIPPOCAMPUS, receives_observation: false, observation_scale: 0 },
        ];

        let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();
        let n_global_sync = total_neurons.min(4096);

        Self {
            regions, region_names: names, connections,
            outer_ticks: ticks,
            exit_strategy: crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 },
            n_global_sync, out_dims, raw_obs_dim: obs_dim, obs_scale_dims: vec![obs_dim],
            aux_losses: AuxLossConfig::default(),
            router: Some(RouterConfig::default()),
            cereb_mode: Default::default(),
            output_local_region: None,
        }
    }

    /// Large 8-region: GPU-scale model (~200M params).
    /// Cortical regions: d_model=512, memory=64 → SuperLinear crosses GPU threshold.
    /// Subcortical regions: d_model=64, memory=32 → moderate size.
    pub fn eight_region_large(obs_dim: usize, out_dims: usize, ticks: usize) -> Self {
        const INPUT: usize = 0;
        const ATTENTION: usize = 1;
        const OUTPUT: usize = 2;
        const MOTOR: usize = 3;
        const CEREBELLUM: usize = 4;
        const BASAL_GANGLIA: usize = 5;
        const INSULA: usize = 6;
        const HIPPOCAMPUS: usize = 7;

        let d = obs_dim; // d_input for all regions

        let regions = vec![
            CtmConfig::region("input", 512, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("attention", 512, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("output", 512, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("motor", 512, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("cerebellum", 64, d, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("basal_ganglia", 64, d, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("insula", 64, d, 32, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("hippocampus", 64, d, 64, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.15, threshold: 0.99 }),
        ];

        let names = vec![
            "input", "attention", "output", "motor",
            "cerebellum", "basal_ganglia", "insula", "hippocampus",
        ].into_iter().map(String::from).collect();

        let connections = vec![
            Connection { from: vec![MOTOR], to: INPUT, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![INPUT], to: ATTENTION, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![ATTENTION], to: OUTPUT, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: MOTOR, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![MOTOR], to: CEREBELLUM, receives_observation: true, observation_scale: 0 },
            Connection { from: vec![OUTPUT], to: BASAL_GANGLIA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![HIPPOCAMPUS], to: INSULA, receives_observation: false, observation_scale: 0 },
            Connection { from: vec![INPUT, ATTENTION, OUTPUT, MOTOR], to: HIPPOCAMPUS, receives_observation: false, observation_scale: 0 },
        ];

        let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();
        let n_global_sync = total_neurons.min(1024); // raised cap for larger model

        Self {
            regions,
            region_names: names,
            connections,
            outer_ticks: ticks,
            exit_strategy: crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 },
            n_global_sync,
            out_dims,
            raw_obs_dim: obs_dim, obs_scale_dims: vec![obs_dim],
            aux_losses: AuxLossConfig::default(),
            router: Some(RouterConfig::default()),
            cereb_mode: Default::default(),
            output_local_region: None,
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

    /// Cerebellum-dominant 8-region brain (v2). Mirrors `eight_region`'s
    /// connection graph but with asymmetric sizing — cortex regions stay
    /// small, cerebellum is the parameter-budget anchor (intended to back
    /// onto a `FrozenCerebellum` LLM at runtime, e.g. Qwen2.5-0.5B).
    ///
    /// Per `docs/BRAIN_ARCHITECTURE.md` §4-5 / open call (a) ratified:
    /// the cerebellum is a SIBLING SERVICE, so the region's regional
    /// weights are a small adapter / projection placeholder; the actual
    /// LLM is mounted through the `FrozenCerebellum` trait at runtime
    /// and consulted by orchestration outside the regional iteration.
    /// The CTM-side cerebellum region is intentionally tiny — its job
    /// is to be the mount point in the connection graph, not to do the
    /// language-modeling itself.
    ///
    /// Connection topology is the canonical eight-edge cortical loop
    /// from `docs/BRAIN_ARCHITECTURE.md` §2:
    /// ```text
    ///   motor       → input          (carries observation)
    ///   input       → attention
    ///   attention   → output
    ///   output      → motor
    ///   motor       → cerebellum     (carries observation — prompt seam)
    ///   output      → basal_ganglia
    ///   hippocampus → insula
    ///   {input, attention, output, motor} → hippocampus
    /// ```
    /// Cycles are intentional — the cortical loop is closed.
    ///
    /// `obs_dim` sets the input observation dimensionality.
    /// `out_dims` sets the motor / output region's action-vocab width.
    /// `ticks` sets the number of CTM ticks per token.
    pub fn eight_region_v2(obs_dim: usize, out_dims: usize, ticks: usize) -> Self {
        // ── Region indices (canonical 8-region order; see arch doc §1) ──
        const INPUT: usize         = 0;
        const ATTENTION: usize     = 1;
        const OUTPUT: usize        = 2;
        const MOTOR: usize         = 3;
        const CEREBELLUM: usize    = 4;
        const BASAL_GANGLIA: usize = 5;
        const INSULA: usize        = 6;
        const HIPPOCAMPUS: usize   = 7;

        // ── Sizing (ARCHITECTURE.md §4 cerebellum-dominant column) ──
        // Cortex regions: small encoder / router / decoder / effector,
        // each ~10M-class when wired into a real input-projection +
        // SuperLinear stack. Subcortical regions: tiny gating heads.
        // Cerebellum: a 32×32 placeholder — the real ~494M Qwen2.5-0.5B
        // is mounted via FrozenCerebellum at runtime (sibling service).
        // Hippocampus: slightly larger for episodic KV recall.
        //
        // Future tuning: edit these constants in one place.
        // CtmConfig::region(name, d_model, d_input, memory_length, ...).
        // "d_model" is the per-region neuron / activation width — the
        // numbers in the BRAIN_ARCHITECTURE.md §4 table refer to this.
        const CORTEX_D_MODEL: usize  = 128; // ARCHITECTURE.md §4 — small cortex (~10M class)
        const CORTEX_MEMORY: usize   = 32;

        const SUBCORT_D_MODEL: usize = 32;  // basal_ganglia, insula (~5M class)
        const SUBCORT_MEMORY: usize  = 16;

        // ARCHITECTURE.md §5 — placeholder; real LLM lives in
        // FrozenCerebellum, so this region's CTM stays small.
        const CEREB_D_MODEL: usize   = 32;
        const CEREB_MEMORY: usize    = 16;

        const HIPPO_D_MODEL: usize   = 64;  // ARCHITECTURE.md §4 — episodic KV (~50M class)
        const HIPPO_MEMORY: usize    = 32;

        // d_input wired to obs_dim (matches `eight_region` shape; the
        // inter-region synapses do their own projection on top).
        let d = obs_dim;

        let regions = vec![
            // ── Cortex: encoder / router / decoder / effector ─────────
            CtmConfig::region("input", CORTEX_D_MODEL, d, CORTEX_MEMORY, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),
            CtmConfig::region("attention", CORTEX_D_MODEL, d, CORTEX_MEMORY, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("output", CORTEX_D_MODEL, d, CORTEX_MEMORY, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("motor", CORTEX_D_MODEL, d, CORTEX_MEMORY, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),

            // ── Cerebellum: LLM mount-point placeholder (§5) ──────────
            // Stays small because the cerebellum's actual signal comes
            // through the FrozenCerebellum service, not this CTM tick.
            CtmConfig::region("cerebellum", CEREB_D_MODEL, d, CEREB_MEMORY, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),

            // ── Subcortical gating heads (§4 ~5M class) ───────────────
            CtmConfig::region("basal_ganglia", SUBCORT_D_MODEL, d, SUBCORT_MEMORY, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }),
            CtmConfig::region("insula", SUBCORT_D_MODEL, d, SUBCORT_MEMORY, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 }),

            // ── Hippocampus: episodic KV (§4 ~50M class) ──────────────
            CtmConfig::region("hippocampus", HIPPO_D_MODEL, d, HIPPO_MEMORY, true, ticks,
                crate::config::ExitStrategy::AdaptiveGate { beta: 0.15, threshold: 0.99 }),
        ];

        let names = vec![
            "input", "attention", "output", "motor",
            "cerebellum", "basal_ganglia", "insula", "hippocampus",
        ].into_iter().map(String::from).collect();

        // ── Connection graph: ARCHITECTURE.md §2 table (9 edges) ──────
        // Cycles are intentional (cortical loop is closed). Two edges
        // carry the raw observation: motor→input (action-conditioned
        // perception) and motor→cerebellum (the prompt seam).
        //
        // **Edge 8 (cerebellum→attention)** is the architectural
        // commitment from arch doc §2 ("the first concrete cerebellum-
        // output edge to add is likely cerebellum → attention — LLM
        // prediction biases what to look at"). At runtime, a
        // [`CerebellumService`] supplies the SOURCE signal for this
        // edge from its per-layer hidden cache (projected to the
        // cerebellum d_model); when no service is plumbed, the source
        // falls back to the placeholder cerebellum-region's CTM
        // output. Either way the synapse sees a cortex-dim source +
        // attention-dim target, so wiring is uniform with the rest of
        // the connection graph.
        let connections = vec![
            // edge 0: motor → input — action-conditioned next-frame perception
            Connection { from: vec![MOTOR], to: INPUT,
                receives_observation: true, observation_scale: 0 },
            // edge 1: input → attention — salience selection
            Connection { from: vec![INPUT], to: ATTENTION,
                receives_observation: false, observation_scale: 0 },
            // edge 2: attention → output — selected representation → decision
            Connection { from: vec![ATTENTION], to: OUTPUT,
                receives_observation: false, observation_scale: 0 },
            // edge 3: output → motor — decision → effector
            Connection { from: vec![OUTPUT], to: MOTOR,
                receives_observation: false, observation_scale: 0 },
            // edge 4: motor → cerebellum — action + observation → LLM input (prompt seam, §2)
            Connection { from: vec![MOTOR], to: CEREBELLUM,
                receives_observation: true, observation_scale: 0 },
            // edge 5: output → basal_ganglia — decision gating
            Connection { from: vec![OUTPUT], to: BASAL_GANGLIA,
                receives_observation: false, observation_scale: 0 },
            // edge 6: hippocampus → insula — memory → salience
            Connection { from: vec![HIPPOCAMPUS], to: INSULA,
                receives_observation: false, observation_scale: 0 },
            // edge 7: {input, attention, output, motor} → hippocampus — cortical activity → episodic memory
            Connection { from: vec![INPUT, ATTENTION, OUTPUT, MOTOR], to: HIPPOCAMPUS,
                receives_observation: false, observation_scale: 0 },
            // edge 8: cerebellum → attention — LLM prediction biases salience selection
            // (sibling-service signal injected by `CerebellumService` at runtime).
            Connection { from: vec![CEREBELLUM], to: ATTENTION,
                receives_observation: false, observation_scale: 0 },
        ];

        // n_global_sync sized to total cortex+sub d_model, capped at
        // 256 — small brain, so a 1024-cap (as in `eight_region`) is
        // overkill and would just be clamped to the sum anyway.
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
            obs_scale_dims: vec![obs_dim],
            aux_losses: AuxLossConfig::default(),
            // No router by default — the cerebellum-dominant brain
            // routes through its connection topology + the
            // FrozenCerebellum sibling service. A learned router can
            // be enabled per-experiment by post-construction edit.
            router: None,
            // cereb_mode left at default (Ctm); orchestration outside
            // RegionalConfig is responsible for swapping in a
            // FrozenCerebellum-backed mode when the LLM is mounted.
            cereb_mode: Default::default(),
            output_local_region: None,
        }
    }
}

// ─── Weights ───────────────────────────────────────────────

/// All weights for the regional CTM.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct RegionalWeights {
    pub config: RegionalConfig,

    /// Byte embedding table: [vocab_size × embed_dim].
    /// Maps token indices to dense vectors that feed into the CTM.
    /// `#[serde(default)]` so vision brains — which feed pixels/obs rather than
    /// token ids and carry an empty table — load from exports that omit it.
    #[serde(default)]
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

    /// OUTPUT-local move head: reads the move/output logits DIRECTLY from
    /// one region's own synchronization readout (`sync_out`), sized
    /// `region.n_synch_out → out_dims`. Present iff
    /// `config.output_local_region` is `Some(_)`. When present it
    /// REPLACES `output_proj(global_sync)` as the prediction. Gated +
    /// `#[serde(default)]` so legacy brains (which never set
    /// `output_local_region`) deserialize and behave identically.
    #[serde(default)]
    pub output_local_head: Option<Linear>,

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

    /// Folded-in planner: the **hippocampus region's value-iteration core**
    /// (`vin::VinReadout`). When `Some`, the brain *plans* by running explicit
    /// value iteration over the decoded maze grid and reading the move
    /// ego-centrically at the agent's own cell — so the planner is a component
    /// OF the brain (region 7, already wired `{input,attn,output,motor} →
    /// hippocampus → insula`) rather than a sibling module invoked alongside it.
    /// Warm-started by loading the standalone trained VIN into this slot (zero
    /// retraining); later distilled into the region's tick dynamics (M2).
    /// `#[serde(default)]` so legacy brains (no planner) deserialize unchanged.
    #[serde(default)]
    pub planner: Option<crate::vin::VinReadout>,

    /// Learned inter-region router (MoS-style). None = fixed connections.
    #[serde(default)]
    pub router: Option<RegionalRouter>,

    /// Projection layers for frozen cerebellum (trained even when model is frozen).
    /// Present when cereb_mode != Ctm.
    #[serde(default)]
    #[serde(skip)]
    pub cereb_projection: Option<crate::cerebellum::CerebProjection>,

    /// Learnable blend scale for frozen cerebellum contribution.
    #[serde(skip)]
    pub cereb_blend_logit: Option<f32>,

    /// Monotonic mutation counter. Bumped by every weight-mutation
    /// entry point (`RegionalAdamW::step`, `RegionalGradients::apply`,
    /// `apply_aux_gradients`). Resident caches snapshot this value at
    /// build/refresh time and refuse `forward` if it diverges from
    /// `weights.generation()` — exposing the staleness window that
    /// used to be a runtime contract as a typed error
    /// (`CacheError::Stale`).
    ///
    /// `pub(crate)` to force callers through `bump_generation()`. Any
    /// path that mutates a field directly (rare, mostly tests) must
    /// call `bump_generation()` afterward or dependent caches will
    /// not detect the staleness.
    #[serde(default)]
    pub(crate) generation: u64,
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

        // Build per-region CTM weights. A spatial region's kv_proj must be
        // sized `raw_dim → d_input` (it attends over per-token raw features),
        // so build it with raw_input_dim = raw_dim; flat regions keep the
        // legacy raw_input_dim = d_input.
        let regions: Vec<CtmWeights> = config.regions.iter()
            .map(|cfg| {
                let raw_input_dim = match cfg.spatial {
                    Some((_n, rd)) => rd,
                    None => cfg.d_input,
                };
                CtmWeights::new(cfg.clone(), raw_input_dim)
            })
            .collect();

        // Build connection synapses. Each connection picks one
        // observation scale (default 0); the synapse's input dim
        // includes only that scale's length, not the full
        // concatenated multi-scale buffer.
        let connection_synapses: Vec<Linear> = config.connections.iter().map(|conn| {
            let mut src_dim: usize = conn.from.iter()
                .map(|&r| regions[r].config.d_model)
                .sum();
            if conn.receives_observation {
                let (_start, len) = config.obs_scale_slice(conn.observation_scale);
                src_dim += len;
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

        // OUTPUT-local move head — sized to the chosen region's own
        // sync_out readout. Only built when the config opts in.
        let output_local_head = config.output_local_region.map(|r| {
            let n_synch_out = config.regions[r].n_synch_out;
            Linear::new(n_synch_out, config.out_dims)
        });

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
            output_local_head,
            outer_exit_gate,
            extra_heads,
            cereb_predict,
            bg_value,
            planner: None,
            router,
            cereb_projection: None,
            cereb_blend_logit: None,
            generation: 0,
        }
    }

    /// Snapshot of the mutation counter. Resident caches read this at
    /// build time and on every dispatch; a divergence indicates a
    /// stale cache that must be refreshed.
    pub fn generation(&self) -> u64 { self.generation }

    // ── Folded-in planner (hippocampus value-iteration core) ──────────────
    // The brain *plans* through these: when a `VinReadout` is folded into the
    // hippocampus slot, the maze move comes from the brain's own value
    // iteration over the decoded grid — not a sibling module. See `planner`.

    /// `true` when a planner is folded into the hippocampus slot.
    pub fn has_planner(&self) -> bool { self.planner.is_some() }

    /// Plan a move with the folded-in hippocampus value-iteration core.
    /// `grid_tokens` is the decoded maze grid `[grid_h*grid_w × raw_dim]`
    /// (row-major, the VIN's per-cell `[is_open, is_goal, bias]` features);
    /// `agent_cell` is the agent's `(row, col)`. Returns the planner's full
    /// output whose `move_logits` ARE the brain's decision, plus the per-cell
    /// value field for visualisation. `None` when no planner is folded in.
    pub fn plan(
        &self,
        grid_tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        agent_cell: Option<(usize, usize)>,
    ) -> Option<crate::vin::VinOutput> {
        self.planner
            .as_ref()
            .map(|p| p.forward(grid_tokens, grid_h, grid_w, agent_cell))
    }

    /// Plan with **entropy-gated value iteration** ("hippocampal ripples
    /// increase under uncertainty"): the folded-in planner runs more Bellman
    /// sweeps where the move is uncertain (junctions, larger grids) and stops
    /// early once confident. Returns the planner output and the number of
    /// sweeps used (the "ripple count"). `None` if no planner is folded in.
    pub fn plan_adaptive(
        &self,
        grid_tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        agent_cell: Option<(usize, usize)>,
        min_iters: usize,
        max_iters: usize,
        entropy_floor: f32,
    ) -> Option<(crate::vin::VinOutput, usize)> {
        self.planner.as_ref().map(|p| {
            p.forward_adaptive(
                grid_tokens, grid_h, grid_w, agent_cell, min_iters, max_iters, entropy_floor,
            )
        })
    }

    /// Test-time consolidation of the folded-in planner's move head
    /// (mistake-replay / "sleep"): a single scale-invariant NLMS step toward
    /// `target_move` at the agent cell. Returns the pre-step cross-entropy
    /// (so `lr = 0` measures without mutating), or `None` if no planner.
    /// Bumps `generation` so resident caches notice the weight change.
    pub fn plan_consolidate_move(
        &mut self,
        grid_tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        agent_cell: (usize, usize),
        target_move: usize,
        lr: f32,
    ) -> Option<f32> {
        let loss = self.planner.as_mut().map(|p| {
            p.consolidate_move(grid_tokens, grid_h, grid_w, agent_cell, target_move, lr)
        });
        if loss.is_some() && lr != 0.0 {
            self.bump_generation();
        }
        loss
    }

    /// Fold a trained standalone `VinReadout` into the hippocampus slot
    /// (the warm-start). Returns `self` for chaining at export time.
    pub fn with_planner(mut self, planner: crate::vin::VinReadout) -> Self {
        self.planner = Some(planner);
        self.bump_generation();
        self
    }

    /// Increment the mutation counter. Called automatically by the
    /// in-tree optimizer paths (`RegionalAdamW::step`,
    /// `RegionalGradients::apply`, `apply_aux_gradients`). Custom
    /// paths that mutate weight fields directly must call this; the
    /// `pub(crate)` visibility on `generation` is what forces the
    /// invariant.
    pub fn bump_generation(&mut self) { self.generation = self.generation.wrapping_add(1); }

    /// Configure projection layers for a frozen cerebellum model.
    /// `n_layers` is how many transformer layers the model exposes.
    pub fn with_frozen_cerebellum(mut self, hidden_dim: usize, n_layers: usize) -> Self {
        let cereb_idx = self.config.region_names.iter()
            .position(|n| n.contains("cerebellum"))
            .unwrap_or(4);
        let cortex_dim = self.regions[cereb_idx].config.d_model;
        self.cereb_projection = Some(crate::cerebellum::CerebProjection::with_layers(
            cortex_dim, hidden_dim, hidden_dim, n_layers,
        ));
        self.cereb_blend_logit = Some(-2.0);
        self.config.cereb_mode = crate::cerebellum::CerebMode::External;
        self
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
        if let Some(ref h) = self.output_local_head { n += h.weight.len() + h.bias.len(); }
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

    /// Double-buffer for previous tick's outputs (avoids clone per tick).
    /// Swapped with region_outputs at the start of each outer tick.
    prev_outputs: Vec<Vec<f32>>,

    /// Global sync accumulators.
    pub global_alpha: Vec<f32>,
    pub global_beta: Vec<f32>,

    /// Per-region CTM states for the CACHED forward path
    /// (`forward_cached`). Distinct type from `region_states` because the
    /// cached path uses `Ctm`'s value-shaped `CtmStateExplicit` (no
    /// episodic). Threaded across calls (FIX 1) so CTM recurrence carries.
    /// Lazily (re)built when empty or shape-mismatched.
    pub region_explicit_states: Vec<crate::train::CtmStateExplicit>,

    /// The OUTPUT-local region's own `sync_out` readout from the most
    /// recent `forward_cached` call, when `config.output_local_region`
    /// is set and the OUTPUT region is spatial. This is the EXACT input
    /// the local move head consumes — exposed for the wall-probe so it
    /// can decode walls from the readout's true input. `None` otherwise.
    pub last_output_local_sync: Option<Vec<f32>>,
}

impl RegionalState {
    pub fn new(w: &RegionalWeights) -> Self {
        let region_states: Vec<CtmState> = w.regions.iter().enumerate()
            .map(|(i, rw)| {
                let is_hippocampus = w.config.region_names.get(i)
                    .map_or(false, |n| n.contains("hippocampus"));
                if is_hippocampus {
                    // Hippocampus gets episodic memory: 4 short, 16 mid, 64 long
                    CtmState::with_episodic(rw, 4, 16, 64)
                } else {
                    CtmState::new(rw)
                }
            })
            .collect();

        let region_outputs: Vec<Vec<f32>> = w.regions.iter()
            .map(|rw| rw.start_activated.clone())
            .collect();

        let prev_outputs = region_outputs.clone();
        let n = w.config.n_global_sync;
        Self {
            region_states,
            region_outputs,
            prev_outputs,
            global_alpha: vec![0.0; n],
            global_beta: vec![1.0; n],
            region_explicit_states: w.regions.iter()
                .map(|rw| Ctm::init_state(rw))
                .collect(),
            last_output_local_sync: None,
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
    /// The OUTPUT-local region's own `sync_out` readout from the final outer
    /// tick (the exact input the local move head consumes). `None` unless
    /// `config.output_local_region` is set. Exposed for diagnostics — lets a
    /// caller measure whether the head's INPUT varies with agent position.
    pub output_local_sync: Option<Vec<f32>>,
}

/// GPU-accelerated sync backward scatter: compute d_activations from d_sync gradients.
///
/// For each pair i:
///   d_act[left[i]] += d_sync[i] / sqrt(beta[i]) * activated[right[i]]
///   d_act[right[i]] += d_sync[i] / sqrt(beta[i]) * activated[left[i]]
///
/// Tries GPU atomic scatter first, falls back to CPU.
fn global_sync_backward(
    d_sync: &[f32], activated: &[f32], beta: &[f32],
    left: &[usize], right: &[usize], d_model: usize,
) -> Vec<f32> {
    use modgrad_device::backend::{ops, SyncBackwardScatterArgs};
    let n_pairs = left.len();
    let left_u32: Vec<u32> = left.iter().map(|&x| x as u32).collect();
    let right_u32: Vec<u32> = right.iter().map(|&x| x as u32).collect();
    let mut d_act = vec![0.0f32; d_model];
    ops::sync_backward_scatter(SyncBackwardScatterArgs {
        d_sync, pairs_left: &left_u32, pairs_right: &right_u32,
        activated, beta, d_act: &mut d_act,
        n_pairs, d_model,
    }).expect("sync_backward_scatter dispatch");
    d_act
}

/// Run the regional CTM forward pass.
///
/// `observation`: raw input [raw_obs_dim].
/// Returns predictions at each outer tick.
/// One outer tick's worth of introspection, captured by
/// [`regional_forward_trace`]. Mirrors what `RegionalOutput` returns, but for
/// EVERY outer tick — the per-tick region activations and global sync a 3D
/// viz/debugger animates, not just the final tick.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegionalTick {
    /// `[out_dims]` next-step prediction at this tick.
    pub prediction: Vec<f32>,
    /// `[n_regions][d_model]` per-region activated state this tick.
    pub region_activations: Vec<Vec<f32>>,
    /// `[n_global_sync]` global sync vector this tick.
    pub global_sync: Vec<f32>,
    /// Outer adaptive-compute gate λ this tick (None when no exit gate).
    pub exit_lambda: Option<f32>,
}

pub fn regional_forward(
    w: &RegionalWeights,
    state: &mut RegionalState,
    observation: &[f32],
) -> RegionalOutput {
    regional_forward_inner(w, state, observation, None)
}

/// Like [`regional_forward`], but also returns a per-outer-tick trace
/// (`region_activations`, `global_sync`, `prediction`, `exit_lambda`) for
/// introspection — drives debuggers and the browser 3D-brain viz. Identical
/// numerics to `regional_forward`; the trace is a read-out, not a fork.
pub fn regional_forward_trace(
    w: &RegionalWeights,
    state: &mut RegionalState,
    observation: &[f32],
) -> (RegionalOutput, Vec<RegionalTick>) {
    let mut trace = Vec::with_capacity(w.config.outer_ticks);
    let out = regional_forward_inner(w, state, observation, Some(&mut trace));
    (out, trace)
}

fn regional_forward_inner(
    w: &RegionalWeights,
    state: &mut RegionalState,
    observation: &[f32],
    mut trace: Option<&mut Vec<RegionalTick>>,
) -> RegionalOutput {
    let cfg = &w.config;
    let n_regions = cfg.regions.len();
    let mut predictions = Vec::with_capacity(cfg.outer_ticks);
    let mut exit_lambdas: Vec<f32> = Vec::new();
    let mut exit_cdf = 0.0f32;
    let mut survival = 1.0f32;

    // Pre-allocated buffers for the hot loop
    let mut obs_projected = vec![0.0f32; w.obs_proj.out_dim];
    {
        let _g = crate::dispatch_profile::Guard::new(
            crate::dispatch_profile::DispatchKind::ObsProj);
        w.obs_proj.forward_into(observation, &mut obs_projected);
    }
    let total_neurons: usize = cfg.regions.iter().map(|r| r.d_model).sum();
    let mut all_act_buf = vec![0.0f32; total_neurons];
    let n_sync = cfg.n_global_sync;
    let mut gs_buf = vec![0.0f32; n_sync];
    let mut pred_buf = vec![0.0f32; cfg.out_dims];
    // Last outer tick's OUTPUT-local sync readout (head input), for telemetry.
    let mut last_local_sync: Option<Vec<f32>> = None;

    for outer_tick in 0..cfg.outer_ticks {
        // Swap double-buffer: prev_outputs gets current, region_outputs ready for writes.
        std::mem::swap(&mut state.region_outputs, &mut state.prev_outputs);
        let prev_outputs = &state.prev_outputs;

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
            let obs = routed;
            for conn in &cfg.connections {
                if conn.receives_observation {
                    // Add observation signal to regions that need it
                    for _v in obs_projected.iter() {
                        // obs[conn.to] already has routed context; observation
                        // was part of global sync — no separate concat needed
                    }
                }
            }
            obs
        } else {
            // Fixed connections (backward compatible). Multiple incoming
            // edges to the same target region are merged additively
            // (`merge_into_region_obs`) — see that helper's doc.
            (0..n_regions).into_par_iter().map(|r| {
                let mut slot: Vec<f32> = Vec::new();
                for (ci, conn) in cfg.connections.iter().enumerate() {
                    if conn.to == r {
                        let mut src = Vec::new();
                        for &from_idx in &conn.from {
                            src.extend_from_slice(&prev_outputs[from_idx]);
                        }
                        if conn.receives_observation {
                            src.extend_from_slice(observation);
                        }
                        let projected = {
                            let _g = crate::dispatch_profile::Guard::new(
                                crate::dispatch_profile::DispatchKind::ConnSynapse);
                            w.connection_synapses[ci].forward(&src)
                        };
                        merge_into_region_obs(&mut slot, projected);
                    }
                }
                if slot.is_empty() { obs_projected.clone() } else { slot }
            }).collect()
        };

        // Phase B: Run regions (parallel via disjoint mut slices).
        // Take ownership of region_states to get disjoint &mut per element.
        // ctm_forward branches on rs.episodic.is_some() internally now —
        // the caller used to dispatch via two function names; that's
        // collapsed into a single entry point.
        let local_region = cfg.output_local_region;
        let mut states_vec: Vec<CtmState> = std::mem::take(&mut state.region_states);
        let results: Vec<(Vec<f32>, Option<Vec<f32>>)> =
            states_vec.par_iter_mut().enumerate().map(|(r, rs)| {
            // Spatial region: attend over the RAW per-cell observation as
            // `n × rd` tokens (bypassing the flat connection-merged obs).
            // Flat region: legacy single-token path over its d_input obs.
            let output = match w.regions[r].config.spatial {
                Some((n_tok, rd)) => {
                    ctm_forward(&w.regions[r], rs, crate::forward::CtmInput::Raw {
                        obs: observation, n_tokens: n_tok, raw_dim: rd,
                    })
                }
                None => {
                    let d_input = w.regions[r].config.d_input;
                    ctm_forward(&w.regions[r], rs, crate::forward::CtmInput::Raw {
                        obs: &region_obs[r], n_tokens: 1, raw_dim: d_input,
                    })
                }
            };
            // Capture this region's OWN sync readout when it is the
            // OUTPUT-local move source (the move head reads it directly).
            let local_sync = if local_region == Some(r) {
                Some(output.sync_out)
            } else {
                None
            };
            (rs.activated.clone(), local_sync)
        }).collect();
        state.region_states = states_vec;

        // Capture the OUTPUT-local region's sync readout for the move head.
        let local_sync_out: Option<Vec<f32>> = local_region
            .and_then(|r| results[r].1.clone());
        last_local_sync = local_sync_out.clone();

        // Phase C: Commit outputs (sequential — cheap copy)
        for r in 0..n_regions {
            state.region_outputs[r] = results[r].0.clone();
        }

        // Phase 3: Global sync — flatten into pre-allocated buffer
        {
            let mut offset = 0;
            for r in 0..n_regions {
                let d = state.region_outputs[r].len();
                all_act_buf[offset..offset + d].copy_from_slice(&state.region_outputs[r]);
                offset += d;
            }
        }

        for i in 0..n_sync {
            let l = w.global_sync_left[i];
            let r = w.global_sync_right[i];
            if l < all_act_buf.len() && r < all_act_buf.len() {
                let pw = all_act_buf[l] * all_act_buf[r];
                let decay = (-w.global_decay[i].clamp(0.0, 15.0)).exp();
                state.global_alpha[i] = decay * state.global_alpha[i] + pw;
                state.global_beta[i] = decay * state.global_beta[i] + 1.0;
            }
        }

        for i in 0..n_sync {
            gs_buf[i] = state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8);
        }

        // Phase 4: Output prediction — reuse buffer.
        // When the OUTPUT-local move head is enabled, read the move DIRECTLY
        // from the OUTPUT region's own sync readout (bypassing global-sync
        // pooling). Otherwise use the legacy global-sync output_proj.
        let prediction = match (&w.output_local_head, &local_sync_out) {
            (Some(head), Some(sync)) => head.forward(sync),
            _ => {
                let _g = crate::dispatch_profile::Guard::new(
                    crate::dispatch_profile::DispatchKind::OutputProj);
                w.output_proj.forward_into(&gs_buf, &mut pred_buf);
                pred_buf.clone()
            }
        };
        predictions.push(prediction);

        // Phase 5: Exit decision
        let mut this_lambda: Option<f32> = None;
        let mut do_break = false;
        match &cfg.exit_strategy {
            crate::config::ExitStrategy::AdaptiveGate { threshold, .. } => {
                if let Some(ref gate) = w.outer_exit_gate {
                    let _g = crate::dispatch_profile::Guard::new(
                        crate::dispatch_profile::DispatchKind::OuterExitGate);
                    let gate_logit = gate.forward(&gs_buf);
                    let lambda = 1.0 / (1.0 + (-gate_logit[0]).exp());
                    exit_lambdas.push(lambda);
                    this_lambda = Some(lambda);
                    let p_exit = lambda * survival;
                    exit_cdf += p_exit;
                    survival *= 1.0 - lambda;
                    if exit_cdf > *threshold { do_break = true; }
                }
            }
            _ => {}
        }

        // Per-tick trace snapshot (only when a caller asked for one).
        if let Some(tr) = trace.as_deref_mut() {
            tr.push(RegionalTick {
                prediction: predictions.last().cloned().unwrap_or_default(),
                region_activations: state.region_outputs.clone(),
                global_sync: gs_buf.clone(),
                exit_lambda: this_lambda,
            });
        }

        if do_break { break; }
    }

    let ticks_used = predictions.len();
    RegionalOutput {
        predictions,
        global_sync: gs_buf,
        region_activations: state.region_outputs.clone(),
        exit_lambdas,
        ticks_used,
        output_local_sync: last_local_sync,
    }
}

/// Service-aware forward variant — the cortex actually CONSULTS the
/// cerebellum (per `docs/BRAIN_ARCHITECTURE.md` §7, sibling-service
/// option **(b)**) instead of treating its placeholder CTM region as
/// the only signal source.
///
/// **Architectural decision (documented at the orchestration site):**
/// the cerebellum is a sibling service mounted via
/// [`crate::cerebellum_service::CerebellumService`] and consulted
/// per-tick by opt-in cortex regions; the cerebellum-region in
/// `eight_region_v2`'s graph remains a small placeholder. Heavy LLM
/// compute happens in the service's `set_context` once per context
/// window; per-tick reads are cheap projections.
///
/// **What changes vs `regional_forward`:** for every connection edge
/// whose source is the cerebellum region (e.g. the new
/// `cerebellum → attention` edge in `eight_region_v2`), the source
/// activation is replaced by the service-cache read at `position` —
/// projected to the cerebellum-region d_model via the service's
/// internal `CerebProjection`. The synapse Linear sees the same
/// per-source dimension it always saw, so weights are wire-compatible.
///
/// **Read pattern:** position-based ([`CerebellumService::read_at`]).
/// Multimodal pooling is a separate consumption pattern; a region
/// that wants modality-pooled signal can call `read_modality`
/// directly off the service. Documented at the call site below.
///
/// `position`: token index into the service's encoded sequence.
/// Typically the per-byte / per-token position of the current outer
/// step; for context windows shorter than `cfg.outer_ticks`, the
/// caller is responsible for clamping (the service zero-fills
/// out-of-range reads).
pub fn regional_forward_with_service(
    w: &RegionalWeights,
    state: &mut RegionalState,
    observation: &[f32],
    service: &crate::cerebellum_service::CerebellumService,
    position: usize,
) -> RegionalOutput {
    let cfg = &w.config;
    let n_regions = cfg.regions.len();
    let mut predictions = Vec::with_capacity(cfg.outer_ticks);
    let mut exit_lambdas: Vec<f32> = Vec::new();
    let mut exit_cdf = 0.0f32;
    let mut survival = 1.0f32;

    let mut obs_projected = vec![0.0f32; w.obs_proj.out_dim];
    w.obs_proj.forward_into(observation, &mut obs_projected);
    let total_neurons: usize = cfg.regions.iter().map(|r| r.d_model).sum();
    let mut all_act_buf = vec![0.0f32; total_neurons];
    let n_sync = cfg.n_global_sync;
    let mut gs_buf = vec![0.0f32; n_sync];
    let mut pred_buf = vec![0.0f32; cfg.out_dims];

    // ── Service signal: project the cache at `position` into the
    // cerebellum region's d_model ONCE per call (not per tick — the
    // cache doesn't change inside this call). The cortex regions then
    // see this signal as the SOURCE for any edge whose `from` includes
    // the cerebellum region. The synapse weights stay sized for the
    // cerebellum d_model, which is what the service projects to.
    let cereb_idx = cfg.region_names.iter()
        .position(|n| n.contains("cerebellum"))
        .unwrap_or(4);
    let cereb_d_model = w.regions[cereb_idx].config.d_model;
    let mut service_signal = vec![0.0f32; service.cortex_dim().max(cereb_d_model)];
    if service.n_positions() > 0 {
        // `read_at` writes service.cortex_dim() floats; we sized the
        // buffer to accommodate that. The caller's contract is that
        // service.cortex_dim() == cereb_d_model — that's the wire
        // matching the cerebellum-region's placeholder. Any mismatch
        // here is a configuration bug; we truncate-or-zero-pad rather
        // than panic so a misconfigured service degrades the signal
        // gracefully (cortex still runs, just gets less useful input).
        service.read_at(position, &mut service_signal);
    }
    // Trim to exactly cereb_d_model so downstream synapse-source
    // sizing is unambiguous (synapses are sized to the source region's
    // d_model, not the service's possibly-larger cortex_dim).
    service_signal.truncate(cereb_d_model);
    if service_signal.len() < cereb_d_model {
        service_signal.resize(cereb_d_model, 0.0);
    }

    for outer_tick in 0..cfg.outer_ticks {
        std::mem::swap(&mut state.region_outputs, &mut state.prev_outputs);
        let prev_outputs = &state.prev_outputs;

        // Phase A: Build observations. Same shape as `regional_forward`,
        // but with one twist: when an edge's `from` slice contains the
        // cerebellum region, we substitute the service signal for the
        // cerebellum portion of the source vector. Multiple incoming
        // edges to the same target region merge additively (see
        // `merge_into_region_obs`).
        let region_obs: Vec<Vec<f32>> = if let Some(ref router) = w.router {
            // Router path: the router treats the regional outputs as
            // opaque sources. Service injection is a no-op here for
            // v0 — router-routed brains don't consume the cerebellum
            // service yet (separate slice; opens once we know how the
            // router should weight a frozen-LLM signal).
            let global_sync: Vec<f32> = (0..cfg.n_global_sync)
                .map(|i| state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8))
                .collect();
            let (routed, _cache) = router.forward(
                outer_tick, &global_sync, &prev_outputs, false,
            );
            routed
        } else {
            (0..n_regions).into_par_iter().map(|r| {
                let mut slot: Vec<f32> = Vec::new();
                for (ci, conn) in cfg.connections.iter().enumerate() {
                    if conn.to == r {
                        let mut src = Vec::new();
                        for &from_idx in &conn.from {
                            if from_idx == cereb_idx {
                                // Sibling-service signal: the cortex
                                // reads the LLM's per-layer hidden
                                // cache at `position`, NOT the
                                // placeholder CTM's tick output.
                                src.extend_from_slice(&service_signal);
                            } else {
                                src.extend_from_slice(&prev_outputs[from_idx]);
                            }
                        }
                        if conn.receives_observation {
                            src.extend_from_slice(observation);
                        }
                        let projected = {
                            let _g = crate::dispatch_profile::Guard::new(
                                crate::dispatch_profile::DispatchKind::ConnSynapse);
                            w.connection_synapses[ci].forward(&src)
                        };
                        merge_into_region_obs(&mut slot, projected);
                    }
                }
                if slot.is_empty() { obs_projected.clone() } else { slot }
            }).collect()
        };

        // Phase B: Run regions (same parallel pattern as
        // `regional_forward`).
        let mut states_vec: Vec<CtmState> = std::mem::take(&mut state.region_states);
        let results: Vec<Vec<f32>> = states_vec.par_iter_mut().enumerate().map(|(r, rs)| {
            // Spatial region: attend over the RAW per-cell observation as
            // `n × rd` tokens (bypassing the flat connection-merged obs).
            // Flat region: legacy single-token path over its d_input obs.
            match w.regions[r].config.spatial {
                Some((n_tok, rd)) => {
                    let _output = ctm_forward(&w.regions[r], rs, crate::forward::CtmInput::Raw {
                        obs: observation, n_tokens: n_tok, raw_dim: rd,
                    });
                }
                None => {
                    let d_input = w.regions[r].config.d_input;
                    let _output = ctm_forward(&w.regions[r], rs, crate::forward::CtmInput::Raw {
                        obs: &region_obs[r], n_tokens: 1, raw_dim: d_input,
                    });
                }
            }
            rs.activated.clone()
        }).collect();
        state.region_states = states_vec;

        for r in 0..n_regions {
            state.region_outputs[r] = results[r].clone();
        }

        {
            let mut offset = 0;
            for r in 0..n_regions {
                let d = state.region_outputs[r].len();
                all_act_buf[offset..offset + d].copy_from_slice(&state.region_outputs[r]);
                offset += d;
            }
        }

        for i in 0..n_sync {
            let l = w.global_sync_left[i];
            let r = w.global_sync_right[i];
            if l < all_act_buf.len() && r < all_act_buf.len() {
                let pw = all_act_buf[l] * all_act_buf[r];
                let decay = (-w.global_decay[i].clamp(0.0, 15.0)).exp();
                state.global_alpha[i] = decay * state.global_alpha[i] + pw;
                state.global_beta[i] = decay * state.global_beta[i] + 1.0;
            }
        }

        for i in 0..n_sync {
            gs_buf[i] = state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8);
        }

        w.output_proj.forward_into(&gs_buf, &mut pred_buf);
        predictions.push(pred_buf.clone());

        if let crate::config::ExitStrategy::AdaptiveGate { threshold, .. } = &cfg.exit_strategy {
            if let Some(ref gate) = w.outer_exit_gate {
                let gate_logit = gate.forward(&gs_buf);
                let lambda = 1.0 / (1.0 + (-gate_logit[0]).exp());
                exit_lambdas.push(lambda);
                let p_exit = lambda * survival;
                exit_cdf += p_exit;
                survival *= 1.0 - lambda;
                if exit_cdf > *threshold { break; }
            }
        }
    }

    let ticks_used = predictions.len();
    RegionalOutput {
        predictions,
        global_sync: gs_buf,
        region_activations: state.region_outputs.clone(),
        exit_lambdas,
        ticks_used,
        // The service-aware variant doesn't run the OUTPUT-local move head.
        output_local_sync: None,
    }
}

/// Run a frozen cerebellum forward pass, overriding the cerebellum region output.
///
/// Call this AFTER `regional_forward` to replace the cerebellum's CTM output
/// with the frozen model's output. The projection layers (in CerebProjection)
/// bridge cortex ↔ frozen model dimensions.
///
/// `observation` is the raw observation passed to `regional_forward`.
pub fn frozen_cerebellum_forward(
    w: &RegionalWeights,
    state: &mut RegionalState,
    observation: &[f32],
    frozen: &mut dyn crate::cerebellum::FrozenCerebellum,
) {
    let proj = match &w.cereb_projection {
        Some(p) => p,
        None => return,
    };
    let cereb_idx = w.config.region_names.iter()
        .position(|n| n.contains("cerebellum"))
        .unwrap_or(4);

    // The cerebellum receives motor output + observation via its connection.
    // For the frozen model, we project the observation into the model's space.
    let input_len = proj.cortex_dim.min(observation.len());
    let output = proj.forward(frozen, &observation[..input_len]);

    // Resize to match the cerebellum region's d_model
    let d_model = w.regions[cereb_idx].config.d_model;
    let mut cereb_output = vec![0.0f32; d_model];
    let copy_len = d_model.min(output.len());
    cereb_output[..copy_len].copy_from_slice(&output[..copy_len]);

    state.region_outputs[cereb_idx] = cereb_output;
}

// ─── Training (BPTT) ───────────────────────────────────────

/// Cache for one outer tick — stores everything needed for backward.
pub struct OuterTickCache {
    /// Per-region: the observation fed to ctm_forward.
    #[allow(dead_code)] // retained for potential future backward / inspection use
    region_obs: Vec<Vec<f32>>,
    /// Per-region: activated state AFTER forward (= region output).
    #[allow(dead_code)] // retained for potential future backward / inspection use
    region_activated: Vec<Vec<f32>>,
    /// Per-region: SDK CTM cache for inner-tick BPTT (HOST path).
    /// Populated by `<RegionalBrain as Brain>::forward_cached`,
    /// `regional_train_step`, and `forward_cached_frozen`.
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
    /// OUTPUT-local move head gradients (None when feature disabled).
    pub output_local_head_dw: Option<Vec<f32>>,
    pub output_local_head_db: Option<Vec<f32>>,
    /// Outer exit gate gradients.
    pub outer_exit_gate_dw: Option<Vec<f32>>,
    pub outer_exit_gate_db: Option<Vec<f32>>,
    /// Router gradients (None when router is off).
    pub router_grads: Option<RouterGradients>,
    /// Cerebellar prediction head gradients (None when aux disabled).
    pub cereb_predict_dw: Option<Vec<f32>>,
    pub cereb_predict_db: Option<Vec<f32>>,
    /// BG value head gradients (None when aux disabled).
    pub bg_value_dw: Option<Vec<f32>>,
    pub bg_value_db: Option<Vec<f32>>,
    /// Cerebellum projection gradients (None when no frozen cerebellum).
    pub cereb_proj_out_dw: Option<Vec<f32>>,
    pub cereb_proj_out_db: Option<Vec<f32>>,
    /// Cerebellum blend scale gradient.
    pub cereb_blend_scale_grad: Option<f32>,
}

/// Compute L2 norm across multiple gradient slices.
/// Tries GPU dispatch first (concatenates into a scratch buffer), falls back to CPU.
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
            output_local_head_dw: w.output_local_head.as_ref().map(|h| vec![0.0; h.weight.len()]),
            output_local_head_db: w.output_local_head.as_ref().map(|h| vec![0.0; h.bias.len()]),
            outer_exit_gate_dw: w.outer_exit_gate.as_ref().map(|g| vec![0.0; g.weight.len()]),
            outer_exit_gate_db: w.outer_exit_gate.as_ref().map(|g| vec![0.0; g.bias.len()]),
            router_grads: w.router.as_ref().map(RouterGradients::zeros),
            cereb_predict_dw: w.cereb_predict.as_ref().map(|h| vec![0.0; h.weight.len()]),
            cereb_predict_db: w.cereb_predict.as_ref().map(|h| vec![0.0; h.bias.len()]),
            bg_value_dw: w.bg_value.as_ref().map(|h| vec![0.0; h.weight.len()]),
            bg_value_db: w.bg_value.as_ref().map(|h| vec![0.0; h.bias.len()]),
            cereb_proj_out_dw: w.cereb_projection.as_ref().map(|p| vec![0.0; p.proj_out_w.len()]),
            cereb_proj_out_db: w.cereb_projection.as_ref().map(|p| vec![0.0; p.proj_out_b.len()]),
            cereb_blend_scale_grad: w.cereb_projection.as_ref().map(|_| 0.0),
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
        if let Some(w) = &mut self.output_local_head_dw { w.fill(0.0); }
        if let Some(b) = &mut self.output_local_head_db { b.fill(0.0); }
        if let Some(w) = &mut self.outer_exit_gate_dw { w.fill(0.0); }
        if let Some(b) = &mut self.outer_exit_gate_db { b.fill(0.0); }
        if let Some(rg) = &mut self.router_grads { rg.zero(); }
        if let Some(w) = &mut self.cereb_predict_dw { w.fill(0.0); }
        if let Some(b) = &mut self.cereb_predict_db { b.fill(0.0); }
        if let Some(w) = &mut self.bg_value_dw { w.fill(0.0); }
        if let Some(b) = &mut self.bg_value_db { b.fill(0.0); }
        if let Some(w) = &mut self.cereb_proj_out_dw { w.fill(0.0); }
        if let Some(b) = &mut self.cereb_proj_out_db { b.fill(0.0); }
        if let Some(s) = &mut self.cereb_blend_scale_grad { *s = 0.0; }
    }

    /// Append every parameter gradient owned by this struct into `buf`
    /// in a stable order. The order is: embedding, then per-region CTM
    /// gradients (in declaration order), then connection synapses,
    /// observation/output projections, global decay, optional sub-heads
    /// (outer exit gate, router, cereb predict, BG value, cereb projection)
    /// in the field declaration order. Optional fields are skipped when
    /// `None`; for a fixed model configuration the resulting length is
    /// constant across calls.
    ///
    /// Used by data attribution to extract a per-example gradient vector
    /// suitable for Johnson-Lindenstrauss projection (see crate
    /// `modgrad-attribution`).
    pub fn flatten_into(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.embed_grad);
        for rg in &self.region_grads { rg.flatten_into(buf); }
        for dw in &self.connection_dw { buf.extend_from_slice(dw); }
        for db in &self.connection_db { buf.extend_from_slice(db); }
        buf.extend_from_slice(&self.obs_proj_dw);
        buf.extend_from_slice(&self.obs_proj_db);
        buf.extend_from_slice(&self.global_decay_grad);
        buf.extend_from_slice(&self.output_proj_dw);
        buf.extend_from_slice(&self.output_proj_db);
        if let Some(w) = &self.output_local_head_dw { buf.extend_from_slice(w); }
        if let Some(b) = &self.output_local_head_db { buf.extend_from_slice(b); }
        if let Some(w) = &self.outer_exit_gate_dw { buf.extend_from_slice(w); }
        if let Some(b) = &self.outer_exit_gate_db { buf.extend_from_slice(b); }
        if let Some(rg) = &self.router_grads { rg.flatten_into(buf); }
        if let Some(w) = &self.cereb_predict_dw { buf.extend_from_slice(w); }
        if let Some(b) = &self.cereb_predict_db { buf.extend_from_slice(b); }
        if let Some(w) = &self.bg_value_dw { buf.extend_from_slice(w); }
        if let Some(b) = &self.bg_value_db { buf.extend_from_slice(b); }
        if let Some(w) = &self.cereb_proj_out_dw { buf.extend_from_slice(w); }
        if let Some(b) = &self.cereb_proj_out_db { buf.extend_from_slice(b); }
        if let Some(s) = &self.cereb_blend_scale_grad { buf.push(*s); }
    }

    /// Convenience over [`Self::flatten_into`] returning a fresh `Vec<f32>`.
    /// Prefer `flatten_into` on the hot path — caller can reuse the
    /// allocation across samples.
    pub fn flatten(&self) -> Vec<f32> {
        let mut buf = Vec::new();
        self.flatten_into(&mut buf);
        buf
    }

    /// Diagnostic: L2 norm of the weight-gradient payload for each region.
    /// Used for vanishing-gradient diagnostics; not on the hot path.
    pub fn region_norms(&self) -> Vec<f32> {
        self.region_grads.iter().map(|rg| rg.l2_norm()).collect()
    }

    /// Diagnostic: L2 norm of each inter-region connection synapse's dW.
    pub fn connection_norms(&self) -> Vec<f32> {
        self.connection_dw.iter().map(|dw| {
            let sumsq: f64 = dw.iter().map(|&x| (x as f64) * (x as f64)).sum();
            (sumsq as f32).sqrt()
        }).collect()
    }

    /// Diagnostic: L2 norm of each router sub-weight's gradient.
    /// Returns (to_route_norms, from_route_norms, route_proj_norm, tick_embed_norm).
    /// All zeros when router isn't enabled.
    pub fn router_norms(&self) -> (Vec<f32>, Vec<f32>, f32, f32) {
        let Some(rg) = &self.router_grads else {
            return (vec![], vec![], 0.0, 0.0);
        };
        let n = |v: &[f32]| -> f32 {
            let s: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
            (s as f32).sqrt()
        };
        let to_route: Vec<f32> = rg.to_route_dw.iter().map(|v| n(v)).collect();
        let from_route: Vec<f32> = rg.from_route_dw.iter().map(|v| n(v)).collect();
        (to_route, from_route, n(&rg.route_proj_dw), n(&rg.tick_embed_grad))
    }

    /// Apply gradients with SGD + gradient clipping (for tests / simple usage).
    pub fn apply(&mut self, w: &mut RegionalWeights, lr: f32, clip_norm: f32) {
        // Compute gradient norm for clipping (GPU-accelerated with CPU fallback)
        let mut slices: Vec<&[f32]> = vec![
            &self.output_proj_dw, &self.output_proj_db,
            &self.obs_proj_dw, &self.embed_grad,
        ];
        for dw in &self.connection_dw { slices.push(dw); }
        let norm = crate::grad_norm(&slices);
        let scale = if norm > clip_norm { clip_norm / norm } else { 1.0 };
        let lr = lr * scale;

        let sgd = |w: &mut [f32], g: &[f32], lr: f32| {
            for (w, g) in w.iter_mut().zip(g) { *w -= lr * g; }
        };

        // Embeddings
        sgd(&mut w.embeddings, &self.embed_grad, lr);

        // Per-region gradients via SDK
        for (rw, rg) in w.regions.iter_mut().zip(self.region_grads.iter_mut()) {
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

        // Invalidate dependent resident caches.
        w.bump_generation();
    }

}

// ─── Pre-allocated workspace for zero-alloc training ──────

/// Reusable buffers for the training hot path.
/// Allocate once at startup, reuse every token. Eliminates the
/// ~1000 allocations per optimizer step that dominate training time.
pub struct TrainWorkspace {
    // Per-region observation buffers
    pub region_obs: Vec<Vec<f32>>,
    // Connection synapse input buffers
    pub connection_inputs: Vec<Vec<f32>>,
    // Global sync computation
    pub global_sync: Vec<f32>,
    pub all_activations: Vec<f32>,
    // Prediction buffer
    pub prediction: Vec<f32>,
    // Backward buffers
    pub d_global_sync: Vec<f32>,
    pub d_all_activations: Vec<f32>,
    pub d_region_activated: Vec<Vec<f32>>,
    pub d_region_obs: Vec<Vec<f32>>,
    // Observation projection
    pub obs_projected: Vec<f32>,
}

impl TrainWorkspace {
    pub fn new(w: &RegionalWeights) -> Self {
        let cfg = &w.config;
        let n_sync = cfg.n_global_sync;
        let total_neurons: usize = cfg.regions.iter().map(|r| r.d_model).sum();

        let region_obs: Vec<Vec<f32>> = cfg.regions.iter()
            .map(|r| vec![0.0f32; r.d_input])
            .collect();

        let connection_inputs: Vec<Vec<f32>> = cfg.connections.iter()
            .map(|conn| {
                let mut dim: usize = conn.from.iter()
                    .map(|&r| cfg.regions[r].d_model).sum();
                if conn.receives_observation { dim += cfg.raw_obs_dim; }
                vec![0.0f32; dim]
            }).collect();

        let d_region_activated: Vec<Vec<f32>> = cfg.regions.iter()
            .map(|r| vec![0.0f32; r.d_model])
            .collect();

        let d_region_obs: Vec<Vec<f32>> = cfg.regions.iter()
            .map(|r| vec![0.0f32; r.d_input])
            .collect();

        Self {
            region_obs,
            connection_inputs,
            global_sync: vec![0.0f32; n_sync],
            all_activations: vec![0.0f32; total_neurons],
            prediction: vec![0.0f32; cfg.out_dims],
            d_global_sync: vec![0.0f32; n_sync],
            d_all_activations: vec![0.0f32; total_neurons],
            d_region_activated,
            d_region_obs,
            obs_projected: vec![0.0f32; cfg.regions[0].d_input],
        }
    }

    /// Zero all backward buffers between steps.
    pub fn zero_backward(&mut self) {
        self.d_global_sync.fill(0.0);
        self.d_all_activations.fill(0.0);
        for d in &mut self.d_region_activated { d.fill(0.0); }
        for d in &mut self.d_region_obs { d.fill(0.0); }
    }
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

/// Accumulate aux loss gradients into RegionalGradients.
///
/// Call this after the main backward pass, using the region activations
/// from the same forward pass. Gradients go through the same optimizer
/// as everything else — no separate SGD step, no state inconsistency.
pub fn accumulate_aux_gradients(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    region_outputs: &[Vec<f32>],
    observation: &[f32],
    prev_loss: f32,
) {
    let cfg = &w.config.aux_losses;
    let weight = cfg.aux_weight;

    // Cerebellar prediction: d_loss/d_W, d_loss/d_b for cereb_predict head
    if cfg.cerebellar_prediction {
        if let (Some(head), Some(dw), Some(db)) = (
            &w.cereb_predict, &mut grads.cereb_predict_dw, &mut grads.cereb_predict_db,
        ) {
            let cereb_idx = w.config.region_names.iter()
                .position(|n| n.contains("cerebellum"));
            if let Some(ci) = cereb_idx {
                if let Some(cereb_out) = region_outputs.get(ci) {
                    let predicted = head.forward(cereb_out);
                    let n = predicted.len().min(observation.len());
                    let in_dim = cereb_out.len();

                    for o in 0..n {
                        let d_pred = 2.0 * (predicted[o] - observation[o]) / n as f32 * weight;
                        for i in 0..in_dim {
                            dw[o * in_dim + i] += d_pred * cereb_out[i];
                        }
                        db[o] += d_pred;
                    }
                }
            }
        }
    }

    // BG value: predict negative loss
    if cfg.bg_temporal_difference {
        if let (Some(head), Some(dw), Some(db)) = (
            &w.bg_value, &mut grads.bg_value_dw, &mut grads.bg_value_db,
        ) {
            let bg_idx = w.config.region_names.iter()
                .position(|n| n.contains("basal"));
            if let Some(bi) = bg_idx {
                if let Some(bg_out) = region_outputs.get(bi) {
                    let value_pred = head.forward(bg_out);
                    if !value_pred.is_empty() {
                        let target = -prev_loss;
                        let d_value = 2.0 * (value_pred[0] - target) * weight;
                        let in_dim = bg_out.len();
                        for i in 0..in_dim {
                            dw[i] += d_value * bg_out[i];
                        }
                        db[0] += d_value;
                    }
                }
            }
        }
    }
}

/// Compute aux loss gradients and apply them directly to the aux head weights.
///
/// This is a manual SGD step on the aux heads only — separate from the main
/// optimizer step, because RegionalGradients doesn't track aux head gradients.
///
/// The aux heads are small Linear layers (cerebellum d_model → obs_dim,
/// BG d_model → 1). Their gradients are cheap to compute and apply inline.
///
/// `region_outputs`: activated states from RegionalOutput.region_activations
/// `observation`: raw input (target for cerebellar prediction)
/// `prev_loss`: previous step's loss (target for BG value)
/// `aux_lr`: learning rate for aux head updates
pub fn apply_aux_gradients(
    w: &mut RegionalWeights,
    region_outputs: &[Vec<f32>],
    observation: &[f32],
    prev_loss: f32,
    aux_lr: f32,
) {
    let cfg = &w.config.aux_losses;
    let weight = cfg.aux_weight;

    // ── Cerebellar prediction: teach cerebellum to predict observations ──
    if cfg.cerebellar_prediction {
        let cereb_idx = w.config.region_names.iter()
            .position(|n| n.contains("cerebellum"));
        if let (Some(ci), Some(ref mut head)) = (cereb_idx, w.cereb_predict.as_mut()) {
            if let Some(cereb_out) = region_outputs.get(ci) {
                let predicted = head.forward(cereb_out);
                let n = predicted.len().min(observation.len());

                // d_loss/d_predicted = 2/n * (predicted - observation) * weight
                let mut d_pred = vec![0.0f32; predicted.len()];
                for i in 0..n {
                    d_pred[i] = 2.0 * (predicted[i] - observation[i]) / n as f32 * weight;
                }

                // Linear backward: dW = d_pred ⊗ input, db = d_pred
                let in_dim = cereb_out.len();
                let out_dim = predicted.len();
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        head.weight[o * in_dim + i] -= aux_lr * d_pred[o] * cereb_out[i];
                    }
                    head.bias[o] -= aux_lr * d_pred[o];
                }
            }
        }
    }

    // ── BG value: teach BG to predict negative loss ──
    if cfg.bg_temporal_difference {
        let bg_idx = w.config.region_names.iter()
            .position(|n| n.contains("basal"));
        if let (Some(bi), Some(ref mut head)) = (bg_idx, w.bg_value.as_mut()) {
            if let Some(bg_out) = region_outputs.get(bi) {
                let value_pred = head.forward(bg_out);
                if !value_pred.is_empty() {
                    let target = -prev_loss;
                    let d_value = 2.0 * (value_pred[0] - target) * weight;

                    let in_dim = bg_out.len();
                    for i in 0..in_dim {
                        head.weight[i] -= aux_lr * d_value * bg_out[i];
                    }
                    head.bias[0] -= aux_lr * d_value;
                }
            }
        }
    }

    // ── Hippocampal contrastive: no head weights to update ──
    // The contrastive loss penalizes representation collapse in the hippocampus.
    // Its gradients flow through the main backward pass (d_region_activations),
    // not through a separate head. To wire this properly, we'd need to inject
    // d_variance into the region's backward — left for future work.

    // Aux heads (cereb_predict, bg_value) are part of `RegionalWeights`;
    // mutating them must invalidate dependent resident caches even
    // though the caches don't currently mirror these heads.
    w.bump_generation();
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
                merge_into_region_obs(&mut region_obs[conn.to], projected);
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
            // Spatial region: cache-forward over the RAW `n × rd` observation
            // (the same tokens the forward pass attends over), so the cached
            // backward threads gradients through per-token attention. Flat
            // region: legacy single-token path over the merged d_input obs.
            let input = match w.regions[r].config.spatial {
                Some((n_tok, rd)) => TokenInput {
                    tokens: observation.to_vec(),
                    n_tokens: n_tok,
                    token_dim: rd,
                },
                None => {
                    let d_input = w.regions[r].config.d_input;
                    TokenInput {
                        tokens: region_obs[r].clone(),
                        n_tokens: 1,
                        token_dim: d_input,
                    }
                }
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

    // ── Loss: imagination (default) ──
    // When the OUTPUT-local move head is enabled, predictions are read
    // DIRECTLY from the local region's own per-tick sync_out (bypassing the
    // global-sync pooling). Otherwise the legacy global-sync output_proj.
    let local_region = cfg.output_local_region;
    let predictions: Vec<Vec<f32>> = match (&w.output_local_head, local_region) {
        (Some(head), Some(lr)) => tick_caches.iter()
            .map(|tc| {
                let sync = tc.region_caches[lr].as_ref()
                    .map(|c| c.final_sync_out())
                    .unwrap_or(&[]);
                head.forward(sync)
            })
            .collect(),
        _ => tick_caches.iter()
            .map(|tc| w.output_proj.forward(&tc.global_sync))
            .collect(),
    };
    let pred_class = predictions.last().unwrap().iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);

    let default_imagination = crate::loss::ImaginationLoss::default();
    let certainties: Vec<[f32; 2]> = predictions.iter().map(|p| {
        let c = compute_certainty(p);
        [1.0 - c, c]
    }).collect();

    let (loss, d_per_tick) = {
        let (loss, d_preds) = default_imagination.compute(&predictions, &certainties, &target);

        if let Some(ref gate) = w.outer_exit_gate {
            let lambdas: Vec<f32> = tick_caches.iter().map(|tc| tc.exit_lambda).collect();
            let beta = cfg.exit_strategy.beta();
            let (_exit_loss, _exit_d_preds, d_lambdas) = crate::train::adaptive_exit_loss(
                &predictions, &lambdas, target, beta);

            // Scratch lifted out of the exit-gate loop; gate.in_dim is
            // fixed, one buffer serves every tick.
            let mut d_gate_in_scratch = vec![0.0f32; gate.in_dim];
            for (t, d_lambda) in d_lambdas.iter().enumerate() {
                if let Some(ref cache) = tick_caches[t].exit_gate_cache {
                    let gw = grads.outer_exit_gate_dw.as_mut().unwrap();
                    let gb = grads.outer_exit_gate_db.as_mut().unwrap();
                    crate::train::linear_backward(
                        gate, &[*d_lambda], cache, gw, gb,
                        &mut d_gate_in_scratch);
                }
            }
        }

        (loss, d_preds)
    };

    // ── Backward through ALL outer ticks ──
    let add = |d: &mut [f32], s: &[f32]| {
        for (d, s) in d.iter_mut().zip(s) { *d += s; }
    };

    // Router source-feedback carry: the router at tick t reads the PREVIOUS
    // tick's region_outputs, so the backward gradient into those sources
    // must land on tick t-1's d_region_activated (we iterate ticks in
    // reverse, so t-1 fires AFTER tick t in this loop). Seeded to zeros;
    // consumed-and-zeroed at each iteration.
    let mut d_region_activated_carry: Vec<Vec<f32>> = (0..n_regions)
        .map(|r| vec![0.0f32; w.regions[r].config.d_model])
        .collect();

    // OUTPUT-local head backward path: accumulate into d_region_activated[lr]
    // (computed below per-tick). The global-sync path is skipped when the
    // local head supplies the prediction, since global_sync never fed it.
    let local_head_active = w.output_local_head.is_some() && local_region.is_some();

    for (t, mut tc) in tick_caches.into_iter().enumerate().rev() {
        let d_logits = &d_per_tick[t];

        // Per-region d on the LOCAL region's final activated, from the
        // OUTPUT-local move head backward. Computed before the global scatter
        // so it can be folded into d_region_activated[lr].
        let local_d_activated: Option<(usize, Vec<f32>)> = if local_head_active {
            let head = w.output_local_head.as_ref().unwrap();
            let lr = local_region.unwrap();
            let (out_dim, in_dim) = (head.out_dim, head.in_dim);
            // sync_out fed to the head this tick.
            let sync = tc.region_caches[lr].as_ref()
                .map(|c| c.final_sync_out().to_vec())
                .unwrap_or_default();
            let dw = grads.output_local_head_dw.as_mut().unwrap();
            let db = grads.output_local_head_db.as_mut().unwrap();
            for i in 0..out_dim { db[i] += d_logits[i]; }
            modgrad_device::backend::ops::outer_product_acc(
                d_logits, &sync, dw, out_dim, in_dim,
            ).expect("output_local_head backward: outer_product_acc dispatch");
            let mut d_sync_out = vec![0.0f32; in_dim];
            modgrad_device::backend::ops::matvec_t(
                d_logits, &head.weight, &mut d_sync_out, out_dim, in_dim,
            ).expect("output_local_head backward: matvec_t dispatch");
            // d_sync_out → d_activated for the local region's final tick.
            let d_act = tc.region_caches[lr].as_ref()
                .map(|c| c.final_sync_out_backward(&w.regions[lr], &d_sync_out))
                .unwrap_or_else(|| vec![0.0f32; w.regions[lr].config.d_model]);
            Some((lr, d_act))
        } else {
            None
        };

        // Output proj backward (global-sync path) — skipped when the local
        // head supplied the prediction (global_sync had no role then).
        let out_dim = w.output_proj.out_dim;
        let in_dim = w.output_proj.in_dim;
        let mut d_global_sync = vec![0.0f32; in_dim];
        if !local_head_active {
            // Bias grad — trivial elementwise, no Op variant worth it.
            for i in 0..out_dim { grads.output_proj_db[i] += d_logits[i]; }
            // Weight grad (outer product accumulate) + input grad (matvec_t)
            // through the Backend registry — output-projection backward now
            // matches the dispatch pattern used in `linear_backward`.
            modgrad_device::backend::ops::outer_product_acc(
                d_logits, &tc.global_sync, &mut grads.output_proj_dw,
                out_dim, in_dim,
            ).expect("output_proj backward: outer_product_acc dispatch");
            modgrad_device::backend::ops::matvec_t(
                d_logits, &w.output_proj.weight, &mut d_global_sync,
                out_dim, in_dim,
            ).expect("output_proj backward: matvec_t dispatch");
        }

        // Global sync backward → d_all_activations (GPU-accelerated).
        // d_global_sync is all-zero when the local head is active, so this
        // contributes nothing then; the local-head d_activated is added below.
        let total_act_dim = tc.all_activations.len();
        let d_all_activations = global_sync_backward(
            &d_global_sync, &tc.all_activations, &tc.global_beta,
            &w.global_sync_left, &w.global_sync_right, total_act_dim,
        );

        // Scatter to per-region, adding the router source-feedback carry
        // from the NEXT tick (which wrote into this tick's region_outputs).
        let mut offset = 0;
        let mut d_region_activated: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
        for r in 0..n_regions {
            let dim = w.regions[r].config.d_model;
            let mut row = d_all_activations[offset..offset + dim].to_vec();
            let carry = &d_region_activated_carry[r];
            for k in 0..dim.min(carry.len()) {
                row[k] += carry[k];
            }
            d_region_activated.push(row);
            offset += dim;
        }
        // Fold in the OUTPUT-local move head's gradient on the local region's
        // final activated state (from its sync_out backward). This is what
        // makes the move-head gradient flow into the region that attends over
        // the maze, instead of through the global-sync pooling.
        if let Some((lr, d_act)) = local_d_activated {
            let row = &mut d_region_activated[lr];
            for k in 0..row.len().min(d_act.len()) { row[k] += d_act[k]; }
        }
        // Reset the carry; the router-backward below will repopulate it
        // for this tick's sources (feeding tick t-1 on the next iteration).
        for r in 0..n_regions {
            for v in d_region_activated_carry[r].iter_mut() { *v = 0.0; }
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
                // U-Net grads — same outer-merge gap as the resident
                // path used to have. `backward_from_activated`
                // populates `result.grads.unet` correctly; it just
                // wasn't being copied across to the outer accumulator.
                dst.unet.add_from(&result.grads.unet);
                // Store d_observation for connection synapse backward.
                // A spatial region attends over the RAW maze observation and
                // ignores its incoming connections (v1), so its d_observation
                // is gradient w.r.t. raw input pixels (no params there) —
                // leave d_region_obs empty so the connection backward skips it.
                if w.regions[r].config.spatial.is_none() {
                    d_region_obs[r] = result.d_observation;
                }
            }
        }

        // Backward through routing: router or fixed connections
        if let (Some(router), Some(rc), Some(rg)) =
            (&w.router, &tc.router_cache, &mut grads.router_grads)
        {
            // Router backward: d_region_obs → from_route → weighted sum → to_route → route_proj
            let n = router.n_regions;
            let d = router.config.d_route;

            // Scratch reuse: buffers live on RouterGradients (same lifetime
            // as the gradients). Zero them instead of re-allocating. Flat
            // layout: `d_routed[j][k]` → `d_routed[j * d + k]`,
            // `d_projected[i][dd]` → `d_projected[i * d + dd]`.
            rg.scratch_d_routed.fill(0.0);
            rg.scratch_d_projected.fill(0.0);
            rg.scratch_d_weights.fill(0.0);
            rg.scratch_d_logits.fill(0.0);
            rg.scratch_d_proj_input.fill(0.0);

            // Per-destination: backprop through from_route Linear
            // Bug 1 fix: use cached `rc.routed[j]` as the Linear input for dW.
            for j in 0..n {
                let d_obs = &d_region_obs[j];
                if d_obs.is_empty() { continue; }
                let fr = &router.from_route[j];
                let routed_j = &rc.routed[j];
                // from_route backward: d_obs → dW, db, d_input
                for i in 0..fr.out_dim.min(d_obs.len()) {
                    rg.from_route_db[j][i] += d_obs[i];
                    for k in 0..fr.in_dim.min(routed_j.len()) {
                        rg.from_route_dw[j][i * fr.in_dim + k] += d_obs[i] * routed_j[k];
                    }
                }
                // d_routed = W^T @ d_obs
                let base = j * d;
                for k in 0..d {
                    for i in 0..fr.out_dim.min(d_obs.len()) {
                        rg.scratch_d_routed[base + k] += d_obs[i] * fr.weight[i * fr.in_dim + k];
                    }
                }
            }

            // Backprop through weighted sum and softmax → d_projected_sources, d_weights
            for j in 0..n {
                let rb = j * d;
                for &i in &rc.selected[j] {
                    let wij = rc.weights[i * n + j];
                    let pb = i * d;
                    // d_projected[i] += wij * d_routed[j]
                    for dd in 0..d {
                        rg.scratch_d_projected[pb + dd] += wij * rg.scratch_d_routed[rb + dd];
                    }
                    // d_weight[i,j] = dot(d_routed[j], projected_sources[i])
                    let mut dw = 0.0f32;
                    for dd in 0..d {
                        dw += rg.scratch_d_routed[rb + dd] * rc.projected_sources[i][dd];
                    }
                    rg.scratch_d_weights[i * n + j] = dw;
                }
            }

            // Backprop through to_route: d_projected → dW, db, and source feedback.
            // Bug 2 fix: compute to_route weight gradients using cached region_outputs[i].
            // Bug 3 fix: route the source-feedback into d_region_activated_carry so
            // it lands on the PREVIOUS tick (which supplied these region outputs).
            for i in 0..n {
                let tr = &router.to_route[i];
                let src_in = &rc.region_outputs[i];
                let pb = i * d;
                for oi in 0..tr.out_dim.min(d) {
                    rg.to_route_db[i][oi] += rg.scratch_d_projected[pb + oi];
                    for k in 0..tr.in_dim.min(src_in.len()) {
                        rg.to_route_dw[i][oi * tr.in_dim + k] +=
                            rg.scratch_d_projected[pb + oi] * src_in[k];
                    }
                }
                // Source-feedback gradient: contributes to tick t-1's
                // region activated (the router at tick t read region_outputs
                // updated by tick t-1's region forward).
                let carry = &mut d_region_activated_carry[i];
                for k in 0..tr.in_dim.min(carry.len()) {
                    for oi in 0..tr.out_dim.min(d) {
                        carry[k] += rg.scratch_d_projected[pb + oi] * tr.weight[oi * tr.in_dim + k];
                    }
                }
            }

            // Softmax backward → d_logits
            for j in 0..n {
                // Only selected entries contribute
                let mut wdw_sum = 0.0f32;
                for &i in &rc.selected[j] {
                    wdw_sum += rc.weights[i * n + j] * rg.scratch_d_weights[i * n + j];
                }
                for &i in &rc.selected[j] {
                    let wij = rc.weights[i * n + j];
                    rg.scratch_d_logits[i * n + j] = wij * (rg.scratch_d_weights[i * n + j] - wdw_sum);
                }
            }

            // route_proj backward: d_logits → dW, db, d_input
            let rp = &router.route_proj;
            let n_sq = n * n;
            for i in 0..n_sq.min(rp.out_dim) {
                rg.route_proj_db[i] += rg.scratch_d_logits[i];
                for j in 0..rp.in_dim.min(rc.proj_input.len()) {
                    rg.route_proj_dw[i * rp.in_dim + j] += rg.scratch_d_logits[i] * rc.proj_input[j];
                }
            }

            // Bug 4 fix: propagate d_logits back through route_proj's input
            // (W^T @ d_logits). Split the d_input into [d_global_sync, d_tick_embed]
            // and accumulate the tick-embedding tail into `tick_embed_grad`.
            // The global-sync half is not wired further here (sync pairs are
            // index-based; a backward path through them is a separate commit).
            for j in 0..rp.in_dim {
                for i in 0..n_sq.min(rp.out_dim) {
                    rg.scratch_d_proj_input[j] += rg.scratch_d_logits[i] * rp.weight[i * rp.in_dim + j];
                }
            }
            let t_idx = t.min(router.max_ticks - 1);
            let t_dim = router.config.tick_embed_dim;
            let n_sync_grad = rp.in_dim.saturating_sub(t_dim);
            for k in 0..t_dim {
                let idx = n_sync_grad + k;
                if idx < rg.scratch_d_proj_input.len() {
                    rg.tick_embed_grad[t_idx * t_dim + k] += rg.scratch_d_proj_input[idx];
                }
            }
            // TODO(future): scratch_d_proj_input[..n_sync_grad] is d_global_sync —
            // add a backward path through the global-sync pairs.
        } else {
            // Fixed connection synapse backward.
            //
            // Ports the fused inline linear-backward to two dispatched
            // ops (outer_product_acc for dw, matvec_t for d_src). The
            // `src_input` slice can be shorter than `syn.in_dim`; zero-
            // padding preserves the original `.min(src_input.len())`
            // semantics because any padded column contributes
            // `d_syn_out[i] * 0 = 0` to the weight update — identical
            // math, one dispatched op each.
            for (ci, conn) in cfg.connections.iter().enumerate().rev() {
                let d_obs = &d_region_obs[conn.to];
                let syn = &w.connection_synapses[ci];
                let src_input = &tc.connection_inputs[ci];

                let d_syn_out: Vec<f32> = d_obs.iter().take(syn.out_dim).copied()
                    .chain(std::iter::repeat(0.0)).take(syn.out_dim).collect();

                // Bias grad — elementwise, no Op variant worth it.
                for i in 0..syn.out_dim { grads.connection_db[ci][i] += d_syn_out[i]; }

                let padded_src: Vec<f32> = if src_input.len() >= syn.in_dim {
                    src_input[..syn.in_dim].to_vec()
                } else {
                    let mut v = vec![0.0f32; syn.in_dim];
                    v[..src_input.len()].copy_from_slice(src_input);
                    v
                };

                modgrad_device::backend::ops::outer_product_acc(
                    &d_syn_out, &padded_src, &mut grads.connection_dw[ci],
                    syn.out_dim, syn.in_dim,
                ).expect("connection_synapse backward: outer_product_acc dispatch");

                let mut d_src = vec![0.0f32; syn.in_dim];
                modgrad_device::backend::ops::matvec_t(
                    &d_syn_out, &syn.weight, &mut d_src,
                    syn.out_dim, syn.in_dim,
                ).expect("connection_synapse backward: matvec_t dispatch");

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

    // After the outermost `.rev()` iteration (tick 0), the router source-
    // feedback carry holds the gradient that should flow back into the
    // region_outputs consumed by the tick-0 forward — which are literally
    // `w.regions[r].start_activated`. No next iteration exists to absorb
    // it, so drain it into each region's `d_start_activated` accumulator.
    for i in 0..n_regions {
        let carry = &d_region_activated_carry[i];
        let dst = &mut grads.region_grads[i].d_start_activated;
        for k in 0..dst.len().min(carry.len()) {
            dst[k] += carry[k];
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
    regional_train_step_loss(w, grads, observation, target, None)
}

/// Train one step with an arbitrary loss function.
///
/// The `compute_loss` closure receives (predictions, certainties) and returns (loss, d_preds).
/// This is the generic version — works with StepwiseCE, ClassTarget, anything.
/// Returns (loss, d_observation).
pub fn regional_train_step_generic(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    observation: &[f32],
    compute_loss: impl FnOnce(&[Vec<f32>], &[[f32; 2]]) -> (f32, Vec<Vec<f32>>),
) -> (f32, Vec<f32>) {
    use std::cell::RefCell;

    let compute_loss = RefCell::new(Some(compute_loss));

    struct ShimLoss<'a, F> {
        compute: &'a RefCell<Option<F>>,
    }
    impl<F: FnOnce(&[Vec<f32>], &[[f32; 2]]) -> (f32, Vec<Vec<f32>>)> modgrad_traits::LossFn for ShimLoss<'_, F> {
        type Target = modgrad_traits::ClassTarget;
        fn compute(
            &self,
            predictions: &[Vec<f32>],
            certainties: &[[f32; 2]],
            _target: &modgrad_traits::ClassTarget,
        ) -> (f32, Vec<Vec<f32>>) {
            let f = self.compute.borrow_mut().take().expect("loss called twice");
            f(predictions, certainties)
        }
    }

    let shim = ShimLoss { compute: &compute_loss };
    let (loss, _pred, d_obs) = regional_train_step_loss(w, grads, observation, 0, Some(&shim));
    (loss, d_obs)
}

/// Train one token with a configurable loss function.
/// If `loss_fn` is None, uses the default CTM loss (min-tick + most-certain).
pub fn regional_train_step_loss(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    observation: &[f32],
    target: usize,
    loss_fn: Option<&dyn modgrad_traits::LossFn<Target = modgrad_traits::ClassTarget>>,
) -> (f32, usize, Vec<f32>) {
    regional_train_step_inner(w, grads, observation, target, loss_fn, None, None)
}

/// Train step with optional frozen cerebellum override (per-token, legacy).
pub fn regional_train_step_frozen(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    observation: &[f32],
    target: usize,
    frozen: &mut dyn crate::cerebellum::FrozenCerebellum,
) -> (f32, usize, Vec<f32>) {
    regional_train_step_inner(w, grads, observation, target, None, Some(frozen), None)
}

/// Train step with a pre-computed cerebellum hidden state injected
/// into the cerebellum region output each tick.
pub fn regional_train_step_with_cereb(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    observation: &[f32],
    target: usize,
    cereb_hidden: &[f32],
) -> (f32, usize, Vec<f32>) {
    regional_train_step_inner(w, grads, observation, target, None, None, Some(cereb_hidden))
}

fn regional_train_step_inner(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    observation: &[f32],
    target: usize,
    loss_fn: Option<&dyn modgrad_traits::LossFn<Target = modgrad_traits::ClassTarget>>,
    mut frozen: Option<&mut dyn crate::cerebellum::FrozenCerebellum>,
    cereb_hidden: Option<&[f32]>,
) -> (f32, usize, Vec<f32>) {
    let cfg = &w.config;
    let n_regions = cfg.regions.len();
    let n_sync = cfg.n_global_sync;
    let obs_dim = cfg.raw_obs_dim;
    let cereb_idx = cfg.region_names.iter()
        .position(|n| n.contains("cerebellum"))
        .unwrap_or(4);

    // ── Forward with caching ──
    // Pre-allocated buffers for the hot loop. obs_projected is computed once,
    // global_sync and all_activations are reused across ticks.
    let mut obs_projected = vec![0.0f32; w.obs_proj.out_dim];
    w.obs_proj.forward_into(observation, &mut obs_projected);

    let total_neurons: usize = cfg.regions.iter().map(|r| r.d_model).sum();
    let mut gs_buf = vec![0.0f32; n_sync];
    let mut all_act_buf = vec![0.0f32; total_neurons];

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
            // Compute global sync into pre-allocated buffer
            for i in 0..n_sync {
                gs_buf[i] = state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8);
            }
            let (routed, cache) = router.forward(
                outer_tick, &gs_buf, &state.region_outputs, true,
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
                merge_into_region_obs(&mut region_obs[conn.to], projected);
            }
            for r in 0..n_regions {
                if region_obs[r].is_empty() {
                    region_obs[r] = obs_projected.clone();
                }
            }
        }

        let mut region_caches: Vec<Option<CtmCache>> = Vec::with_capacity(n_regions);
        for r in 0..n_regions {
            // Spatial region: cache-forward over the RAW `n × rd` observation.
            // Flat region: legacy single-token path over the merged obs.
            let input = match w.regions[r].config.spatial {
                Some((n_tok, rd)) => TokenInput {
                    tokens: observation.to_vec(),
                    n_tokens: n_tok,
                    token_dim: rd,
                },
                None => {
                    let d_input = w.regions[r].config.d_input;
                    TokenInput {
                        tokens: region_obs[r].clone(),
                        n_tokens: 1,
                        token_dim: d_input,
                    }
                }
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

        // Override cerebellum region with frozen model output.
        if let Some(hidden) = cereb_hidden {
            // Cache path: project pre-computed LLM hidden state → cortex dim.
            // ADDITIVE: blend with CTM cerebellum output, don't replace.
            // The CTM cerebellum still runs and provides its own signal.
            // The frozen model's contribution is added on top, scaled down
            // initially so random projections don't overwhelm the CTM signal.
            if let Some(ref proj) = w.cereb_projection {
                let d_model = w.regions[cereb_idx].config.d_model;
                let projected = proj.project_out(hidden);
                let copy_len = d_model.min(projected.len());
                // Learnable blend: sigmoid(logit) scales the contribution.
                // Starts small (~0.12), learns to ramp up as projections improve.
                let blend = w.cereb_blend_logit
                    .map(|logit| 1.0 / (1.0 + (-logit).exp()))
                    .unwrap_or(0.1);
                for i in 0..copy_len {
                    state.region_outputs[cereb_idx][i] += blend * projected[i];
                }
            }
        } else if let Some(ref mut fm) = frozen {
            // Legacy per-token path
            if let Some(ref proj) = w.cereb_projection {
                let cereb_input = &region_obs[cereb_idx];
                let input_len = proj.cortex_dim.min(cereb_input.len());
                let frozen_out = proj.forward(*fm, &cereb_input[..input_len]);
                let d_model = w.regions[cereb_idx].config.d_model;
                let mut cereb_output = vec![0.0f32; d_model];
                let copy_len = d_model.min(frozen_out.len());
                cereb_output[..copy_len].copy_from_slice(&frozen_out[..copy_len]);
                state.region_outputs[cereb_idx] = cereb_output;
            }
        }

        // Flatten region outputs into pre-allocated buffer (zero alloc per tick)
        {
            let mut offset = 0;
            for r in 0..n_regions {
                let d = state.region_outputs[r].len();
                all_act_buf[offset..offset + d].copy_from_slice(&state.region_outputs[r]);
                offset += d;
            }
        }

        for i in 0..n_sync {
            let l = w.global_sync_left[i];
            let r = w.global_sync_right[i];
            if l < all_act_buf.len() && r < all_act_buf.len() {
                let pw = all_act_buf[l] * all_act_buf[r];
                let decay = (-w.global_decay[i].clamp(0.0, 15.0)).exp();
                state.global_alpha[i] = decay * state.global_alpha[i] + pw;
                state.global_beta[i] = decay * state.global_beta[i] + 1.0;
            }
        }

        // Compute global sync in-place
        for i in 0..n_sync {
            gs_buf[i] = state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8);
        }

        // Outer exit gate
        let (exit_gate_cache, exit_lambda) = if let Some(ref gate) = w.outer_exit_gate {
            let (logit, cache) = crate::train::linear_forward_cached(gate, &gs_buf);
            let lambda = 1.0 / (1.0 + (-logit[0]).exp());
            (Some(cache), lambda)
        } else {
            (None, 0.0)
        };

        // Cache needs owned copies for backward pass
        tick_caches.push(OuterTickCache {
            region_obs,
            region_activated: state.region_outputs.clone(),
            region_caches,
            all_activations: all_act_buf.clone(),
            global_sync: gs_buf.clone(),
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

    let default_imagination = crate::loss::ImaginationLoss::default();
    let lf: &dyn modgrad_traits::LossFn<Target = modgrad_traits::ClassTarget> =
        loss_fn.unwrap_or(&default_imagination);

    let certainties: Vec<[f32; 2]> = predictions.iter().map(|p| {
        let c = compute_certainty(p);
        [1.0 - c, c]
    }).collect();

    let (loss, d_per_tick) = {
        let (loss, d_preds) = lf.compute(&predictions, &certainties, &target);

        // Exit gate gradients still flow regardless of which loss is used
        if let Some(ref gate) = w.outer_exit_gate {
            let lambdas: Vec<f32> = tick_caches.iter().map(|tc| tc.exit_lambda).collect();
            let beta = cfg.exit_strategy.beta();
            let (_exit_loss, _exit_d_preds, d_lambdas) = crate::train::adaptive_exit_loss(
                &predictions, &lambdas, target, beta);
            let mut d_gate_in_scratch = vec![0.0f32; gate.in_dim];
            for (t, d_lambda) in d_lambdas.iter().enumerate() {
                if let Some(ref cache) = tick_caches[t].exit_gate_cache {
                    let gw = grads.outer_exit_gate_dw.as_mut().unwrap();
                    let gb = grads.outer_exit_gate_db.as_mut().unwrap();
                    crate::train::linear_backward(
                        gate, &[*d_lambda], cache, gw, gb,
                        &mut d_gate_in_scratch);
                }
            }
        }

        (loss, d_preds)
    };

    // ── Backward ── (same as regional_train_step, but accumulates d_observation)
    let add = |d: &mut [f32], s: &[f32]| {
        for (d, s) in d.iter_mut().zip(s) { *d += s; }
    };

    let mut d_observation = vec![0.0f32; obs_dim];

    // Router source-feedback carry (see matching comment in the loss-less
    // variant). Feeds tick t's router source-backward gradient into the
    // previous tick's region activated gradient on the next iteration.
    let mut d_region_activated_carry: Vec<Vec<f32>> = (0..n_regions)
        .map(|r| vec![0.0f32; w.regions[r].config.d_model])
        .collect();

    for (t, mut tc) in tick_caches.into_iter().enumerate().rev() {
        let d_logits = &d_per_tick[t];

        let out_dim = w.output_proj.out_dim;
        let in_dim = w.output_proj.in_dim;
        let mut d_global_sync = vec![0.0f32; in_dim];
        // Bias grad — trivial elementwise, no Op variant worth it.
        for i in 0..out_dim { grads.output_proj_db[i] += d_logits[i]; }
        // Weight grad (outer product accumulate) + input grad (matvec_t)
        // through the Backend registry — output-projection backward now
        // matches the dispatch pattern used in `linear_backward`.
        modgrad_device::backend::ops::outer_product_acc(
            d_logits, &tc.global_sync, &mut grads.output_proj_dw,
            out_dim, in_dim,
        ).expect("output_proj backward: outer_product_acc dispatch");
        modgrad_device::backend::ops::matvec_t(
            d_logits, &w.output_proj.weight, &mut d_global_sync,
            out_dim, in_dim,
        ).expect("output_proj backward: matvec_t dispatch");

        let total_act_dim = tc.all_activations.len();
        let d_all_activations = global_sync_backward(
            &d_global_sync, &tc.all_activations, &tc.global_beta,
            &w.global_sync_left, &w.global_sync_right, total_act_dim,
        );

        let mut offset = 0;
        let mut d_region_activated: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
        for r in 0..n_regions {
            let dim = w.regions[r].config.d_model;
            let mut row = d_all_activations[offset..offset + dim].to_vec();
            let carry = &d_region_activated_carry[r];
            for k in 0..dim.min(carry.len()) {
                row[k] += carry[k];
            }
            d_region_activated.push(row);
            offset += dim;
        }
        for r in 0..n_regions {
            for v in d_region_activated_carry[r].iter_mut() { *v = 0.0; }
        }

        // Accumulate cerebellum projection gradients from d_region_activated[cereb_idx].
        // The forward was: region_outputs[cereb_idx] += blend * proj_out(hidden)
        // So: d_proj_out_w += blend * d_cereb ⊗ hidden
        //     d_blend_logit += sum(d_cereb * projected) * blend * (1 - blend)
        if let (Some(hidden), Some(proj), Some(dw), Some(db)) = (
            cereb_hidden,
            &w.cereb_projection,
            grads.cereb_proj_out_dw.as_mut(),
            grads.cereb_proj_out_db.as_mut(),
        ) {
            let blend = w.cereb_blend_logit
                .map(|logit| 1.0 / (1.0 + (-logit).exp()))
                .unwrap_or(0.1);
            let d_cereb = &d_region_activated[cereb_idx];
            // Scale by blend since forward multiplied by blend
            let d_cortex_scaled: Vec<f32> = d_cereb.iter().map(|&d| d * blend).collect();
            proj.backward_out(hidden, &d_cortex_scaled, dw, db);

            // Gradient for blend logit: d_loss/d_logit = d_loss/d_blend * d_blend/d_logit
            // d_blend/d_logit = blend * (1 - blend) (sigmoid derivative)
            if let Some(ref mut blend_grad) = grads.cereb_blend_scale_grad {
                let projected = proj.project_out(hidden);
                let dot: f32 = d_cereb.iter().zip(projected.iter())
                    .map(|(d, p)| d * p)
                    .sum();
                *blend_grad += dot * blend * (1.0 - blend);
            }
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
                // U-Net grads — same outer-merge gap as fixed elsewhere.
                dst.unet.add_from(&result.grads.unet);
                // Spatial region ignores incoming connections (v1): its
                // d_observation is gradient w.r.t. raw input, not routed.
                if w.regions[r].config.spatial.is_none() {
                    d_region_obs[r] = result.d_observation;
                }
            }
        }

        // Backward through routing: router or fixed connections
        if let (Some(router), Some(rc), Some(rg)) =
            (&w.router, &tc.router_cache, &mut grads.router_grads)
        {
            let n = router.n_regions;
            let d = router.config.d_route;

            // Reuse scratch buffers on RouterGradients (same lifetime as
            // the gradients). Flat layout: d_routed[j][k] → [j*d+k],
            // d_projected[i][dd] → [i*d+dd].
            rg.scratch_d_routed.fill(0.0);
            rg.scratch_d_projected.fill(0.0);
            rg.scratch_d_weights.fill(0.0);
            rg.scratch_d_logits.fill(0.0);
            rg.scratch_d_proj_input.fill(0.0);

            // Bug 1: from_route weight grad now uses cached rc.routed[j].
            for j in 0..n {
                let d_obs = &d_region_obs[j];
                if d_obs.is_empty() { continue; }
                let fr = &router.from_route[j];
                let routed_j = &rc.routed[j];
                for i in 0..fr.out_dim.min(d_obs.len()) {
                    rg.from_route_db[j][i] += d_obs[i];
                    for k in 0..fr.in_dim.min(routed_j.len()) {
                        rg.from_route_dw[j][i * fr.in_dim + k] += d_obs[i] * routed_j[k];
                    }
                }
                let base = j * d;
                for k in 0..d {
                    for i in 0..fr.out_dim.min(d_obs.len()) {
                        rg.scratch_d_routed[base + k] += d_obs[i] * fr.weight[i * fr.in_dim + k];
                    }
                }
            }

            for j in 0..n {
                let rb = j * d;
                for &i in &rc.selected[j] {
                    let wij = rc.weights[i * n + j];
                    let pb = i * d;
                    for dd in 0..d {
                        rg.scratch_d_projected[pb + dd] += wij * rg.scratch_d_routed[rb + dd];
                    }
                    let mut dw = 0.0f32;
                    for dd in 0..d {
                        dw += rg.scratch_d_routed[rb + dd] * rc.projected_sources[i][dd];
                    }
                    rg.scratch_d_weights[i * n + j] = dw;
                }
            }

            // Bug 2 + Bug 3: to_route weight grad uses cached region_outputs[i];
            // source feedback routes into d_region_activated_carry for the
            // previous tick.
            for i in 0..n {
                let tr = &router.to_route[i];
                let src_in = &rc.region_outputs[i];
                let pb = i * d;
                for oi in 0..tr.out_dim.min(d) {
                    rg.to_route_db[i][oi] += rg.scratch_d_projected[pb + oi];
                    for k in 0..tr.in_dim.min(src_in.len()) {
                        rg.to_route_dw[i][oi * tr.in_dim + k] +=
                            rg.scratch_d_projected[pb + oi] * src_in[k];
                    }
                }
                let carry = &mut d_region_activated_carry[i];
                for k in 0..tr.in_dim.min(carry.len()) {
                    for oi in 0..tr.out_dim.min(d) {
                        carry[k] += rg.scratch_d_projected[pb + oi] * tr.weight[oi * tr.in_dim + k];
                    }
                }
            }

            for j in 0..n {
                let mut wdw_sum = 0.0f32;
                for &i in &rc.selected[j] {
                    wdw_sum += rc.weights[i * n + j] * rg.scratch_d_weights[i * n + j];
                }
                for &i in &rc.selected[j] {
                    let wij = rc.weights[i * n + j];
                    rg.scratch_d_logits[i * n + j] = wij * (rg.scratch_d_weights[i * n + j] - wdw_sum);
                }
            }

            let rp = &router.route_proj;
            let n_sq = n * n;
            for i in 0..n_sq.min(rp.out_dim) {
                rg.route_proj_db[i] += rg.scratch_d_logits[i];
                for j in 0..rp.in_dim.min(rc.proj_input.len()) {
                    rg.route_proj_dw[i * rp.in_dim + j] += rg.scratch_d_logits[i] * rc.proj_input[j];
                }
            }

            // Bug 4: propagate d_logits back through route_proj input; tail
            // of the input is the tick embedding slice → tick_embed_grad.
            for j in 0..rp.in_dim {
                for i in 0..n_sq.min(rp.out_dim) {
                    rg.scratch_d_proj_input[j] += rg.scratch_d_logits[i] * rp.weight[i * rp.in_dim + j];
                }
            }
            let t_idx = t.min(router.max_ticks - 1);
            let t_dim = router.config.tick_embed_dim;
            let n_sync_grad = rp.in_dim.saturating_sub(t_dim);
            for k in 0..t_dim {
                let idx = n_sync_grad + k;
                if idx < rg.scratch_d_proj_input.len() {
                    rg.tick_embed_grad[t_idx * t_dim + k] += rg.scratch_d_proj_input[idx];
                }
            }
        } else {
            // Fixed connection synapse backward.
            //
            // Ports the fused inline linear-backward to two dispatched
            // ops (outer_product_acc for dw, matvec_t for d_src). The
            // `src_input` slice can be shorter than `syn.in_dim`; zero-
            // padding preserves the original `.min(src_input.len())`
            // semantics because any padded column contributes
            // `d_syn_out[i] * 0 = 0` to the weight update — identical
            // math, one dispatched op each.
            for (ci, conn) in cfg.connections.iter().enumerate().rev() {
                let d_obs = &d_region_obs[conn.to];
                let syn = &w.connection_synapses[ci];
                let src_input = &tc.connection_inputs[ci];

                let d_syn_out: Vec<f32> = d_obs.iter().take(syn.out_dim).copied()
                    .chain(std::iter::repeat(0.0)).take(syn.out_dim).collect();

                // Bias grad — elementwise, no Op variant worth it.
                for i in 0..syn.out_dim { grads.connection_db[ci][i] += d_syn_out[i]; }

                let padded_src: Vec<f32> = if src_input.len() >= syn.in_dim {
                    src_input[..syn.in_dim].to_vec()
                } else {
                    let mut v = vec![0.0f32; syn.in_dim];
                    v[..src_input.len()].copy_from_slice(src_input);
                    v
                };

                modgrad_device::backend::ops::outer_product_acc(
                    &d_syn_out, &padded_src, &mut grads.connection_dw[ci],
                    syn.out_dim, syn.in_dim,
                ).expect("connection_synapse backward: outer_product_acc dispatch");

                let mut d_src = vec![0.0f32; syn.in_dim];
                modgrad_device::backend::ops::matvec_t(
                    &d_syn_out, &syn.weight, &mut d_src,
                    syn.out_dim, syn.in_dim,
                ).expect("connection_synapse backward: matvec_t dispatch");

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

    // Drain the tick-0 router source-feedback carry into each region's
    // `d_start_activated` — same rationale as in `regional_train_step`.
    // The tick-0 forward read `region_outputs` straight from
    // `w.regions[r].start_activated`, so whatever the router backward
    // wrote into the carry at the outermost iteration belongs there.
    for i in 0..n_regions {
        let carry = &d_region_activated_carry[i];
        let dst = &mut grads.region_grads[i].d_start_activated;
        for k in 0..dst.len().min(carry.len()) {
            dst[k] += carry[k];
        }
    }

    (loss, pred_class, d_observation)
}

// ─── AdamW Trainer ────────────────────────────────────────

/// Per-parameter AdamW state: first moment, second moment.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct AdamWBuf {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl AdamWBuf {
    pub fn zeros(n: usize) -> Self {
        Self { m: vec![0.0; n], v: vec![0.0; n] }
    }

    pub fn step(&mut self, weights: &mut [f32], grads: &mut [f32], lr: f32, wd: f32, b1: f32, b2: f32, eps: f32, bc1: f32, bc2: f32) {
        use modgrad_device::backend::{ops, AdamWArgs};
        ops::adamw(AdamWArgs {
            w: weights, g: grads, m: &mut self.m, v: &mut self.v,
            lr, beta1: b1, beta2: b2, eps, weight_decay: wd,
            bc1_inv: 1.0 / bc1, bc2_inv: 1.0 / bc2,
        }).expect("adamw dispatch");
    }
}

/// Phase 4: AdamW state for the per-region inner CTM tensors that the
/// existing `CtmGradients::apply` updated with raw clip-SGD. One
/// `RegionInnerAdamW` per region, held by `RegionalAdamW`. Covers the
/// big tensors (NLM, Q/KV/MHA projections, region output_proj). The
/// U-Net synapse and small init/scalar tensors stay on SGD for now —
/// their gradient distributions are different and Adam is less
/// universally a win there.
///
/// State sizing matches the corresponding `CtmWeights` tensors at
/// build time. After an opt.step that resized weights (none today —
/// region shapes are static post-construction), the bufs would need
/// rebuild; that's out of scope for the static-graph training path.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct RegionInnerAdamW {
    pub nlm_s1_w: AdamWBuf,
    pub nlm_s1_b: AdamWBuf,
    pub nlm_s2_w: Option<AdamWBuf>,
    pub nlm_s2_b: Option<AdamWBuf>,
    pub kv_proj_w: AdamWBuf,
    pub kv_proj_b: AdamWBuf,
    pub q_proj_w: AdamWBuf,
    pub q_proj_b: AdamWBuf,
    pub mha_in_w: AdamWBuf,
    pub mha_in_b: AdamWBuf,
    pub mha_out_w: AdamWBuf,
    pub mha_out_b: AdamWBuf,
    pub out_proj_w: AdamWBuf,
    pub out_proj_b: AdamWBuf,
}

impl RegionInnerAdamW {
    pub fn zeros(rw: &CtmWeights) -> Self {
        Self {
            nlm_s1_w: AdamWBuf::zeros(rw.nlm_stage1.weights.len()),
            nlm_s1_b: AdamWBuf::zeros(rw.nlm_stage1.biases.len()),
            nlm_s2_w: rw.nlm_stage2.as_ref().map(|s| AdamWBuf::zeros(s.weights.len())),
            nlm_s2_b: rw.nlm_stage2.as_ref().map(|s| AdamWBuf::zeros(s.biases.len())),
            kv_proj_w: AdamWBuf::zeros(rw.kv_proj.weight.len()),
            kv_proj_b: AdamWBuf::zeros(rw.kv_proj.bias.len()),
            q_proj_w: AdamWBuf::zeros(rw.q_proj.weight.len()),
            q_proj_b: AdamWBuf::zeros(rw.q_proj.bias.len()),
            mha_in_w: AdamWBuf::zeros(rw.mha_in_proj.weight.len()),
            mha_in_b: AdamWBuf::zeros(rw.mha_in_proj.bias.len()),
            mha_out_w: AdamWBuf::zeros(rw.mha_out_proj.weight.len()),
            mha_out_b: AdamWBuf::zeros(rw.mha_out_proj.bias.len()),
            out_proj_w: AdamWBuf::zeros(rw.output_proj.weight.len()),
            out_proj_b: AdamWBuf::zeros(rw.output_proj.bias.len()),
        }
    }

    /// Apply AdamW update to the major inner tensors. Bias-style and
    /// LayerNorm tensors get `wd=0` (no weight decay), weight tensors
    /// get the caller's `wd`. Mirrors the convention used by
    /// `RegionalAdamW::step` for outer weights.
    pub fn step(
        &mut self,
        rw: &mut CtmWeights,
        rg: &mut crate::train::CtmGradients,
        lr: f32, wd: f32,
        b1: f32, b2: f32, eps: f32,
        bc1: f32, bc2: f32,
    ) {
        self.nlm_s1_w.step(&mut rw.nlm_stage1.weights, &mut rg.nlm_s1_w,
            lr, wd, b1, b2, eps, bc1, bc2);
        self.nlm_s1_b.step(&mut rw.nlm_stage1.biases, &mut rg.nlm_s1_b,
            lr, 0.0, b1, b2, eps, bc1, bc2);
        if let (Some(s2w), Some(buf_w), Some(grad_w)) = (
            rw.nlm_stage2.as_mut(), self.nlm_s2_w.as_mut(), rg.nlm_s2_w.as_mut(),
        ) {
            buf_w.step(&mut s2w.weights, grad_w, lr, wd, b1, b2, eps, bc1, bc2);
        }
        if let (Some(s2w), Some(buf_b), Some(grad_b)) = (
            rw.nlm_stage2.as_mut(), self.nlm_s2_b.as_mut(), rg.nlm_s2_b.as_mut(),
        ) {
            buf_b.step(&mut s2w.biases, grad_b, lr, 0.0, b1, b2, eps, bc1, bc2);
        }
        self.kv_proj_w.step(&mut rw.kv_proj.weight, &mut rg.kv_proj_w,
            lr, wd, b1, b2, eps, bc1, bc2);
        self.kv_proj_b.step(&mut rw.kv_proj.bias, &mut rg.kv_proj_b,
            lr, 0.0, b1, b2, eps, bc1, bc2);
        self.q_proj_w.step(&mut rw.q_proj.weight, &mut rg.q_proj_w,
            lr, wd, b1, b2, eps, bc1, bc2);
        self.q_proj_b.step(&mut rw.q_proj.bias, &mut rg.q_proj_b,
            lr, 0.0, b1, b2, eps, bc1, bc2);
        self.mha_in_w.step(&mut rw.mha_in_proj.weight, &mut rg.mha_in_w,
            lr, wd, b1, b2, eps, bc1, bc2);
        self.mha_in_b.step(&mut rw.mha_in_proj.bias, &mut rg.mha_in_b,
            lr, 0.0, b1, b2, eps, bc1, bc2);
        self.mha_out_w.step(&mut rw.mha_out_proj.weight, &mut rg.mha_out_w,
            lr, wd, b1, b2, eps, bc1, bc2);
        self.mha_out_b.step(&mut rw.mha_out_proj.bias, &mut rg.mha_out_b,
            lr, 0.0, b1, b2, eps, bc1, bc2);
        self.out_proj_w.step(&mut rw.output_proj.weight, &mut rg.out_proj_w,
            lr, wd, b1, b2, eps, bc1, bc2);
        self.out_proj_b.step(&mut rw.output_proj.bias, &mut rg.out_proj_b,
            lr, 0.0, b1, b2, eps, bc1, bc2);
    }
}

/// AdamW optimizer state for the entire RegionalCtm.
/// Uses AdamW for outer weights (embeddings, connections, projections)
/// and the SDK's built-in SGD for inner CTM region weights.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
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
    #[serde(default)]
    output_local_head_w: Option<AdamWBuf>,
    #[serde(default)]
    output_local_head_b: Option<AdamWBuf>,
    cereb_predict_w: Option<AdamWBuf>,
    cereb_predict_b: Option<AdamWBuf>,
    bg_value_w: Option<AdamWBuf>,
    bg_value_b: Option<AdamWBuf>,
    cereb_proj_out_w: Option<AdamWBuf>,
    cereb_proj_out_b: Option<AdamWBuf>,
    cereb_blend_logit: Option<AdamWBuf>,
    /// Phase 4: per-region AdamW state for the major inner CTM
    /// tensors (NLM, kv/q/mha/out projections). Used by `step` to
    /// replace the previous SGD-only `CtmGradients::apply`. UNet +
    /// scalar tensors continue to flow through `apply_minor` (raw
    /// clip-SGD). One entry per `weights.regions[r]`.
    #[serde(default)]
    region_inner_adams: Vec<RegionInnerAdamW>,
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
            output_local_head_w: w.output_local_head.as_ref().map(|h| AdamWBuf::zeros(h.weight.len())),
            output_local_head_b: w.output_local_head.as_ref().map(|h| AdamWBuf::zeros(h.bias.len())),
            cereb_predict_w: w.cereb_predict.as_ref().map(|h| AdamWBuf::zeros(h.weight.len())),
            cereb_predict_b: w.cereb_predict.as_ref().map(|h| AdamWBuf::zeros(h.bias.len())),
            bg_value_w: w.bg_value.as_ref().map(|h| AdamWBuf::zeros(h.weight.len())),
            bg_value_b: w.bg_value.as_ref().map(|h| AdamWBuf::zeros(h.bias.len())),
            cereb_proj_out_w: w.cereb_projection.as_ref().map(|p| AdamWBuf::zeros(p.proj_out_w.len())),
            cereb_proj_out_b: w.cereb_projection.as_ref().map(|p| AdamWBuf::zeros(p.proj_out_b.len())),
            cereb_blend_logit: w.cereb_blend_logit.map(|_| AdamWBuf::zeros(1)),
            region_inner_adams: w.regions.iter().map(RegionInnerAdamW::zeros).collect(),
        }
    }

    pub fn with_lr(mut self, lr: f32) -> Self { self.lr = lr; self }
    pub fn with_wd(mut self, wd: f32) -> Self { self.weight_decay = wd; self }
    pub fn with_clip(mut self, clip: f32) -> Self { self.grad_clip = clip; self }

    /// Apply accumulated gradients: AdamW for outer weights, SGD for inner CTM.
    pub fn step(&mut self, w: &mut RegionalWeights, grads: &mut RegionalGradients) {
        self.step += 1;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;
        let bc1 = 1.0 - b1.powi(self.step as i32);
        let bc2 = 1.0 - b2.powi(self.step as i32);

        // Gradient clipping (global norm — GPU-accelerated with CPU fallback)
        let mut slices: Vec<&[f32]> = vec![
            &grads.embed_grad, &grads.output_proj_dw, &grads.output_proj_db,
            &grads.obs_proj_dw, &grads.obs_proj_db,
        ];
        if let Some(dw) = &grads.output_local_head_dw { slices.push(dw); }
        if let Some(db) = &grads.output_local_head_db { slices.push(db); }
        for dw in &grads.connection_dw { slices.push(dw); }
        for db in &grads.connection_db { slices.push(db); }
        for rg in &grads.region_grads {
            slices.push(&rg.nlm_s1_w);
            slices.push(&rg.out_proj_w);
            slices.push(&rg.q_proj_w);
            slices.push(&rg.kv_proj_w);
        }
        let norm = crate::grad_norm(&slices);
        let scale = if norm > self.grad_clip { self.grad_clip / norm } else { 1.0 };
        let lr = self.lr * scale;

        // Embeddings — AdamW (no weight decay)
        self.embed.step(&mut w.embeddings, &mut grads.embed_grad, lr, 0.0, b1, b2, eps, bc1, bc2);

        // Per-region CTM weights:
        //   - Major tensors (NLM, kv/q/mha/out projections) — AdamW via region_inner_adams.
        //   - U-Net synapse + start-state + LN + exit-gate — raw clip-SGD via apply_minor.
        // Maintains the same global-norm clip already folded into `lr`.
        // Backfill region_inner_adams when missing (deserialised from older RegionalAdamW
        // checkpoints that pre-date Phase 4 — `#[serde(default)]` gives an empty Vec).
        if self.region_inner_adams.len() != w.regions.len() {
            self.region_inner_adams = w.regions.iter()
                .map(RegionInnerAdamW::zeros)
                .collect();
        }
        for (r, (rw, rg)) in w.regions.iter_mut()
            .zip(grads.region_grads.iter_mut()).enumerate()
        {
            self.region_inner_adams[r].step(rw, rg, lr, wd, b1, b2, eps, bc1, bc2);
            rg.apply_minor(rw, lr);
        }

        // Connections — AdamW
        for (i, syn) in w.connection_synapses.iter_mut().enumerate() {
            self.conn_w[i].step(&mut syn.weight, &mut grads.connection_dw[i], lr, wd, b1, b2, eps, bc1, bc2);
            self.conn_b[i].step(&mut syn.bias, &mut grads.connection_db[i], lr, 0.0, b1, b2, eps, bc1, bc2);
        }

        // Projections — AdamW
        self.obs_proj_w.step(&mut w.obs_proj.weight, &mut grads.obs_proj_dw, lr, wd, b1, b2, eps, bc1, bc2);
        self.obs_proj_b.step(&mut w.obs_proj.bias, &mut grads.obs_proj_db, lr, 0.0, b1, b2, eps, bc1, bc2);
        self.global_decay.step(&mut w.global_decay, &mut grads.global_decay_grad, lr, 0.0, b1, b2, eps, bc1, bc2);
        self.output_proj_w.step(&mut w.output_proj.weight, &mut grads.output_proj_dw, lr, wd, b1, b2, eps, bc1, bc2);
        self.output_proj_b.step(&mut w.output_proj.bias, &mut grads.output_proj_db, lr, 0.0, b1, b2, eps, bc1, bc2);

        // OUTPUT-local move head — AdamW (only when feature enabled). Lazily
        // init the optimizer state when missing (older deserialized opts get
        // `None` via `#[serde(default)]`).
        if let Some(head) = w.output_local_head.as_mut() {
            if let (Some(dw), Some(db)) =
                (grads.output_local_head_dw.as_mut(), grads.output_local_head_db.as_mut())
            {
                let ow = self.output_local_head_w
                    .get_or_insert_with(|| AdamWBuf::zeros(head.weight.len()));
                ow.step(&mut head.weight, dw, lr, wd, b1, b2, eps, bc1, bc2);
                let ob = self.output_local_head_b
                    .get_or_insert_with(|| AdamWBuf::zeros(head.bias.len()));
                ob.step(&mut head.bias, db, lr, 0.0, b1, b2, eps, bc1, bc2);
            }
        }

        // Aux heads — AdamW (only when enabled)
        if let (Some(head), Some(opt_w), Some(opt_b),
                Some(dw), Some(db)) = (
            &mut w.cereb_predict, &mut self.cereb_predict_w, &mut self.cereb_predict_b,
            &mut grads.cereb_predict_dw, &mut grads.cereb_predict_db,
        ) {
            opt_w.step(&mut head.weight, dw, lr, wd, b1, b2, eps, bc1, bc2);
            opt_b.step(&mut head.bias, db, lr, 0.0, b1, b2, eps, bc1, bc2);
        }
        if let (Some(head), Some(opt_w), Some(opt_b),
                Some(dw), Some(db)) = (
            &mut w.bg_value, &mut self.bg_value_w, &mut self.bg_value_b,
            &mut grads.bg_value_dw, &mut grads.bg_value_db,
        ) {
            opt_w.step(&mut head.weight, dw, lr, wd, b1, b2, eps, bc1, bc2);
            opt_b.step(&mut head.bias, db, lr, 0.0, b1, b2, eps, bc1, bc2);
        }

        // Cerebellum projection — AdamW on proj_out weights
        if let (Some(proj), Some(opt_w), Some(opt_b),
                Some(dw), Some(db)) = (
            &mut w.cereb_projection, &mut self.cereb_proj_out_w, &mut self.cereb_proj_out_b,
            &mut grads.cereb_proj_out_dw, &mut grads.cereb_proj_out_db,
        ) {
            opt_w.step(&mut proj.proj_out_w, dw, lr, wd, b1, b2, eps, bc1, bc2);
            opt_b.step(&mut proj.proj_out_b, db, lr, 0.0, b1, b2, eps, bc1, bc2);
        }

        // Cerebellum blend scale — AdamW on the logit
        if let (Some(logit), Some(opt), Some(grad)) = (
            &mut w.cereb_blend_logit, &mut self.cereb_blend_logit,
            &mut grads.cereb_blend_scale_grad,
        ) {
            let mut logit_slice = [*logit];
            let mut grad_slice = [*grad];
            opt.step(&mut logit_slice, &mut grad_slice, lr, 0.0, b1, b2, eps, bc1, bc2);
            *logit = logit_slice[0];
        }

        // Bump generation after every weight mutation. One bump covers
        // every nested mutation above (per-region SGD, connection
        // synapses, projections, aux heads).
        w.bump_generation();
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

// ─── Typed optimiser (Path C) ────────────────────────────────
//
// Same scope as `RegionalAdamW` but operates on `RegionalWeightsTyped<D>`
// and `RegionalGradientsTyped<D>`. Major weights (NLM, projections,
// embedding, connection synapses) get AdamW; minor tensors (U-Net,
// LN gamma/beta, start states, decay params, exit gate) get raw SGD.
// Matches the untyped split so per-step semantics are familiar.

use modgrad_device::backend::tensor::AdamW as AdamWTyped;

/// Per-region typed inner optimiser — AdamW state for the major
/// CTM tensors. Mirrors `RegionInnerAdamW` but generic over Device.
pub struct RegionInnerAdamWTyped<D: TensorDeviceG> {
    pub nlm_s1_w: AdamWTyped<D>,
    pub nlm_s1_b: AdamWTyped<D>,
    pub nlm_s2_w: Option<AdamWTyped<D>>,
    pub nlm_s2_b: Option<AdamWTyped<D>>,
    pub kv_proj_w: AdamWTyped<D>,
    pub kv_proj_b: AdamWTyped<D>,
    pub q_proj_w: AdamWTyped<D>,
    pub q_proj_b: AdamWTyped<D>,
    pub mha_in_w: AdamWTyped<D>,
    pub mha_in_b: AdamWTyped<D>,
    pub mha_out_w: AdamWTyped<D>,
    pub mha_out_b: AdamWTyped<D>,
    pub out_proj_w: AdamWTyped<D>,
    pub out_proj_b: AdamWTyped<D>,
}

impl<D: TensorDeviceG> RegionInnerAdamWTyped<D> {
    pub fn new(rw: &CtmWeightsTyped<D>) -> Result<Self, BackendErrorG> {
        Ok(Self {
            nlm_s1_w: AdamWTyped::<D>::new(rw.nlm_stage1.weights.len())?,
            nlm_s1_b: AdamWTyped::<D>::new(rw.nlm_stage1.biases.len())?,
            nlm_s2_w: match &rw.nlm_stage2 {
                Some(s) => Some(AdamWTyped::<D>::new(s.weights.len())?),
                None => None,
            },
            nlm_s2_b: match &rw.nlm_stage2 {
                Some(s) => Some(AdamWTyped::<D>::new(s.biases.len())?),
                None => None,
            },
            kv_proj_w: AdamWTyped::<D>::new(rw.kv_proj.weight.len())?,
            kv_proj_b: AdamWTyped::<D>::new(rw.kv_proj.bias.len())?,
            q_proj_w: AdamWTyped::<D>::new(rw.q_proj.weight.len())?,
            q_proj_b: AdamWTyped::<D>::new(rw.q_proj.bias.len())?,
            mha_in_w: AdamWTyped::<D>::new(rw.mha_in_proj.weight.len())?,
            mha_in_b: AdamWTyped::<D>::new(rw.mha_in_proj.bias.len())?,
            mha_out_w: AdamWTyped::<D>::new(rw.mha_out_proj.weight.len())?,
            mha_out_b: AdamWTyped::<D>::new(rw.mha_out_proj.bias.len())?,
            out_proj_w: AdamWTyped::<D>::new(rw.output_proj.weight.len())?,
            out_proj_b: AdamWTyped::<D>::new(rw.output_proj.bias.len())?,
        })
    }

    /// Apply AdamW to every major tensor; biases use weight_decay=0.
    pub fn step(
        &mut self,
        rw: &mut CtmWeightsTyped<D>,
        rg: &mut crate::weights::CtmGradientsTyped<D>,
        lr: f32, wd: f32,
    ) -> Result<(), BackendErrorG> {
        macro_rules! step {
            ($adam:expr, $w:expr, $g:expr, $wd:expr) => {{
                $adam.lr = lr;
                $adam.weight_decay = $wd;
                $adam.step(&mut $w, &$g)?;
            }};
        }
        step!(self.nlm_s1_w, rw.nlm_stage1.weights, rg.nlm_s1_w, wd);
        step!(self.nlm_s1_b, rw.nlm_stage1.biases, rg.nlm_s1_b, 0.0);
        if let (Some(s2), Some(adam_w), Some(g)) = (
            rw.nlm_stage2.as_mut(), self.nlm_s2_w.as_mut(), rg.nlm_s2_w.as_ref(),
        ) {
            adam_w.lr = lr;
            adam_w.weight_decay = wd;
            adam_w.step(&mut s2.weights, g)?;
        }
        if let (Some(s2), Some(adam_b), Some(g)) = (
            rw.nlm_stage2.as_mut(), self.nlm_s2_b.as_mut(), rg.nlm_s2_b.as_ref(),
        ) {
            adam_b.lr = lr;
            adam_b.weight_decay = 0.0;
            adam_b.step(&mut s2.biases, g)?;
        }
        step!(self.kv_proj_w, rw.kv_proj.weight, rg.kv_proj_w, wd);
        step!(self.kv_proj_b, rw.kv_proj.bias, rg.kv_proj_b, 0.0);
        step!(self.q_proj_w, rw.q_proj.weight, rg.q_proj_w, wd);
        step!(self.q_proj_b, rw.q_proj.bias, rg.q_proj_b, 0.0);
        step!(self.mha_in_w, rw.mha_in_proj.weight, rg.mha_in_w, wd);
        step!(self.mha_in_b, rw.mha_in_proj.bias, rg.mha_in_b, 0.0);
        step!(self.mha_out_w, rw.mha_out_proj.weight, rg.mha_out_w, wd);
        step!(self.mha_out_b, rw.mha_out_proj.bias, rg.mha_out_b, 0.0);
        step!(self.out_proj_w, rw.output_proj.weight, rg.out_proj_w, wd);
        step!(self.out_proj_b, rw.output_proj.bias, rg.out_proj_b, 0.0);
        Ok(())
    }
}

/// Typed AdamW for the entire RegionalCtm. Same shape as
/// `RegionalAdamW` but uses `Tensor<D>` everywhere.
///
/// Coverage matches the untyped optimiser:
/// - AdamW on: embeddings, connection synapses, obs_proj, output_proj,
///   global_decay, per-region inner majors (via RegionInnerAdamWTyped).
/// - SGD (clip-scaled lr) on: U-Net synapse, kv_ln gamma/beta,
///   start_activated, start_trace, decay_params_out/action, exit_gate.
///
/// Skipped subsystems (deferred — same as RegionalWeightsTyped):
/// router, cereb_projection, cereb_blend_logit, generation counter.
pub struct RegionalAdamWTyped<D: TensorDeviceG> {
    pub step_count: u64,
    pub lr: f32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub embeddings: AdamWTyped<D>,
    pub conn_w: Vec<AdamWTyped<D>>,
    pub conn_b: Vec<AdamWTyped<D>>,
    pub obs_proj_w: AdamWTyped<D>,
    pub obs_proj_b: AdamWTyped<D>,
    pub global_decay: AdamWTyped<D>,
    pub output_proj_w: AdamWTyped<D>,
    pub output_proj_b: AdamWTyped<D>,
    pub region_inner: Vec<RegionInnerAdamWTyped<D>>,
    pub outer_exit_gate_w: Option<AdamWTyped<D>>,
    pub outer_exit_gate_b: Option<AdamWTyped<D>>,
}

impl<D: TensorDeviceG> RegionalAdamWTyped<D> {
    pub fn new(w: &RegionalWeightsTyped<D>) -> Result<Self, BackendErrorG> {
        let conn_w = w.connection_synapses.iter()
            .map(|s| AdamWTyped::<D>::new(s.weight.len()))
            .collect::<Result<Vec<_>, _>>()?;
        let conn_b = w.connection_synapses.iter()
            .map(|s| AdamWTyped::<D>::new(s.bias.len()))
            .collect::<Result<Vec<_>, _>>()?;
        let region_inner = w.regions.iter()
            .map(RegionInnerAdamWTyped::<D>::new)
            .collect::<Result<Vec<_>, _>>()?;
        let outer_exit_gate_w = match &w.outer_exit_gate {
            Some(g) => Some(AdamWTyped::<D>::new(g.weight.len())?),
            None => None,
        };
        let outer_exit_gate_b = match &w.outer_exit_gate {
            Some(g) => Some(AdamWTyped::<D>::new(g.bias.len())?),
            None => None,
        };
        Ok(Self {
            step_count: 0,
            lr: 3e-4,
            weight_decay: 0.01,
            grad_clip: 1.0,
            embeddings: AdamWTyped::<D>::new(w.embeddings.len())?,
            conn_w, conn_b,
            obs_proj_w: AdamWTyped::<D>::new(w.obs_proj.weight.len())?,
            obs_proj_b: AdamWTyped::<D>::new(w.obs_proj.bias.len())?,
            global_decay: AdamWTyped::<D>::new(w.global_decay.len())?,
            output_proj_w: AdamWTyped::<D>::new(w.output_proj.weight.len())?,
            output_proj_b: AdamWTyped::<D>::new(w.output_proj.bias.len())?,
            region_inner,
            outer_exit_gate_w,
            outer_exit_gate_b,
        })
    }

    pub fn with_lr(mut self, lr: f32) -> Self { self.lr = lr; self }
    pub fn with_wd(mut self, wd: f32) -> Self { self.weight_decay = wd; self }
    pub fn with_clip(mut self, clip: f32) -> Self { self.grad_clip = clip; self }

    /// Compute global L2 norm across every gradient tensor — host fallback
    /// (one to_vec per tensor). Acceptable cost: gradient clipping fires
    /// once per training step on small tensors.
    fn grad_norm(&self, grads: &RegionalGradientsTyped<D>) -> Result<f32, BackendErrorG> {
        let mut sumsq = 0.0f64;
        let mut sum_sq_tensor = |t: &TensorG<D>| -> Result<(), BackendErrorG> {
            let h: Vec<f32> = t.to_vec()?;
            for &x in &h { sumsq += (x as f64) * (x as f64); }
            Ok(())
        };
        sum_sq_tensor(&grads.output_proj.d_w)?;
        sum_sq_tensor(&grads.output_proj.d_b)?;
        sum_sq_tensor(&grads.obs_proj.d_w)?;
        sum_sq_tensor(&grads.obs_proj.d_b)?;
        for cg in &grads.connection_synapses {
            sum_sq_tensor(&cg.d_w)?;
            sum_sq_tensor(&cg.d_b)?;
        }
        for rg in &grads.regions {
            sum_sq_tensor(&rg.nlm_s1_w)?;
            sum_sq_tensor(&rg.out_proj_w)?;
            sum_sq_tensor(&rg.q_proj_w)?;
            sum_sq_tensor(&rg.kv_proj_w)?;
        }
        Ok((sumsq.sqrt()) as f32)
    }

    /// Apply one optimiser step. Mirrors `RegionalAdamW::step`:
    /// 1. Compute global grad norm + clip scale.
    /// 2. AdamW on majors with scaled lr.
    /// 3. SGD on minors with scaled lr.
    pub fn step(
        &mut self,
        w: &mut RegionalWeightsTyped<D>,
        grads: &mut RegionalGradientsTyped<D>,
    ) -> Result<(), BackendErrorG> {
        self.step_count += 1;
        let norm = self.grad_norm(grads)?;
        let scale = if norm > self.grad_clip { self.grad_clip / norm } else { 1.0 };
        let lr = self.lr * scale;
        let wd = self.weight_decay;

        // Per-region inner majors (AdamW) + minors (SGD).
        for r in 0..w.regions.len() {
            self.region_inner[r].step(&mut w.regions[r], &mut grads.regions[r], lr, wd)?;
            // Minor tensors → SGD via clip-scaled lr.
            apply_minor_typed::<D>(&mut w.regions[r], &mut grads.regions[r],
                &mut grads.region_unets[r], lr)?;
        }

        // Outer connection synapses — AdamW.
        for i in 0..w.connection_synapses.len() {
            self.conn_w[i].lr = lr;
            self.conn_w[i].weight_decay = wd;
            self.conn_w[i].step(&mut w.connection_synapses[i].weight, &grads.connection_synapses[i].d_w)?;
            self.conn_b[i].lr = lr;
            self.conn_b[i].weight_decay = 0.0;
            self.conn_b[i].step(&mut w.connection_synapses[i].bias, &grads.connection_synapses[i].d_b)?;
        }

        // obs_proj — AdamW.
        self.obs_proj_w.lr = lr;
        self.obs_proj_w.weight_decay = wd;
        self.obs_proj_w.step(&mut w.obs_proj.weight, &grads.obs_proj.d_w)?;
        self.obs_proj_b.lr = lr;
        self.obs_proj_b.weight_decay = 0.0;
        self.obs_proj_b.step(&mut w.obs_proj.bias, &grads.obs_proj.d_b)?;

        // global_decay — AdamW (regional cascade currently doesn't flow
        // gradient into it; the AdamW step just runs against zero grads).
        self.global_decay.lr = lr;
        self.global_decay.weight_decay = 0.0;
        self.global_decay.step(&mut w.global_decay, &grads.global_decay)?;

        // output_proj — AdamW.
        self.output_proj_w.lr = lr;
        self.output_proj_w.weight_decay = wd;
        self.output_proj_w.step(&mut w.output_proj.weight, &grads.output_proj.d_w)?;
        self.output_proj_b.lr = lr;
        self.output_proj_b.weight_decay = 0.0;
        self.output_proj_b.step(&mut w.output_proj.bias, &grads.output_proj.d_b)?;

        // Outer exit gate — AdamW (optional).
        if let (Some(gate), Some(opt_w), Some(opt_b), Some(g)) = (
            w.outer_exit_gate.as_mut(),
            self.outer_exit_gate_w.as_mut(),
            self.outer_exit_gate_b.as_mut(),
            grads.outer_exit_gate.as_ref(),
        ) {
            opt_w.lr = lr;
            opt_w.weight_decay = wd;
            opt_w.step(&mut gate.weight, &g.d_w)?;
            opt_b.lr = lr;
            opt_b.weight_decay = 0.0;
            opt_b.step(&mut gate.bias, &g.d_b)?;
        }

        Ok(())
    }
}

/// Apply SGD updates to per-region minor tensors: U-Net synapse weights,
/// kv_ln gamma/beta, start states, decay params, and inner exit gate.
/// Matches `CtmGradients::apply_minor` but typed.
fn apply_minor_typed<D: TensorDeviceG>(
    rw: &mut CtmWeightsTyped<D>,
    rg: &mut crate::weights::CtmGradientsTyped<D>,
    unet_grads: &mut crate::synapse::UNetGradsTyped<D>,
    lr: f32,
) -> Result<(), BackendErrorG> {
    // U-Net synapse: per-block linear + LN, plus skip-LN gamma/beta.
    apply_minor_block::<D>(&mut rw.synapse.first_projection, &mut unet_grads.first, lr)?;
    let n_down = rw.synapse.down_blocks.len().min(unet_grads.downs.len());
    for i in 0..n_down {
        apply_minor_block::<D>(
            &mut rw.synapse.down_blocks[i],
            &mut unet_grads.downs[i],
            lr,
        )?;
    }
    let n_up = rw.synapse.up_blocks.len().min(unet_grads.ups.len());
    for i in 0..n_up {
        apply_minor_block::<D>(
            &mut rw.synapse.up_blocks[i],
            &mut unet_grads.ups[i],
            lr,
        )?;
    }
    let n_skip = rw.synapse.skip_ln_gamma.len()
        .min(rw.synapse.skip_ln_beta.len())
        .min(unet_grads.skip_d_gamma.len())
        .min(unet_grads.skip_d_beta.len());
    for i in 0..n_skip {
        let n_g = rw.synapse.skip_ln_gamma[i].len();
        let n_b = rw.synapse.skip_ln_beta[i].len();
        tensor_typed_g::sgd_step::<D>(&mut rw.synapse.skip_ln_gamma[i], &unet_grads.skip_d_gamma[i], n_g, lr)?;
        tensor_typed_g::sgd_step::<D>(&mut rw.synapse.skip_ln_beta[i], &unet_grads.skip_d_beta[i], n_b, lr)?;
    }

    // kv-LN gamma/beta.
    let n_g = rw.kv_ln_gamma.len();
    tensor_typed_g::sgd_step::<D>(&mut rw.kv_ln_gamma, &rg.kv_ln_d_gamma, n_g, lr)?;
    let n_b = rw.kv_ln_beta.len();
    tensor_typed_g::sgd_step::<D>(&mut rw.kv_ln_beta, &rg.kv_ln_d_beta, n_b, lr)?;

    // Start states.
    let n_a = rw.start_activated.len();
    tensor_typed_g::sgd_step::<D>(&mut rw.start_activated, &rg.d_start_activated, n_a, lr)?;
    let n_t = rw.start_trace.len();
    tensor_typed_g::sgd_step::<D>(&mut rw.start_trace, &rg.d_start_trace, n_t, lr)?;

    // Decay params.
    let n_do = rw.decay_params_out.len();
    tensor_typed_g::sgd_step::<D>(&mut rw.decay_params_out, &rg.d_decay_out, n_do, lr)?;
    let n_da = rw.decay_params_action.len();
    tensor_typed_g::sgd_step::<D>(&mut rw.decay_params_action, &rg.d_decay_action, n_da, lr)?;

    // Inner exit gate (per-region — distinct from outer_exit_gate).
    if let (Some(gate), Some(gw), Some(gb)) = (
        rw.exit_gate.as_mut(), rg.exit_gate_w.as_ref(), rg.exit_gate_b.as_ref(),
    ) {
        let nw = gate.weight.len();
        let nb = gate.bias.len();
        tensor_typed_g::sgd_step::<D>(&mut gate.weight, gw, nw, lr)?;
        tensor_typed_g::sgd_step::<D>(&mut gate.bias, gb, nb, lr)?;
    }

    Ok(())
}

fn apply_minor_block<D: TensorDeviceG>(
    block: &mut crate::synapse::SynapseBlockTyped<D>,
    grads: &mut crate::synapse::SynapseBlockGradsTyped<D>,
    lr: f32,
) -> Result<(), BackendErrorG> {
    let nw = block.linear.weight.len();
    tensor_typed_g::sgd_step::<D>(&mut block.linear.weight, &grads.d_w, nw, lr)?;
    let nb = block.linear.bias.len();
    tensor_typed_g::sgd_step::<D>(&mut block.linear.bias, &grads.d_b, nb, lr)?;
    let ng = block.ln_gamma.len();
    tensor_typed_g::sgd_step::<D>(&mut block.ln_gamma, &grads.d_gamma, ng, lr)?;
    let nbg = block.ln_beta.len();
    tensor_typed_g::sgd_step::<D>(&mut block.ln_beta, &grads.d_beta, nbg, lr)?;
    Ok(())
}

/// Train one token: embed → forward → loss → backward → accumulate embed grads.
/// Does NOT apply gradients — caller accumulates across a sequence, then calls AdamW.step().
pub fn regional_train_token(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    token: usize,
    target: usize,
) -> (f32, usize) {
    // TODO: replace with workspace-backed version
    let obs = w.embed(token).to_vec();
    let (loss, pred, d_obs) = regional_train_step_full(w, grads, &obs, target);

    let d = w.config.raw_obs_dim;
    let offset = token * d;
    for j in 0..d {
        grads.embed_grad[offset + j] += d_obs[j];
    }

    (loss, pred)
}

/// Fast training: uses pre-allocated workspace. Zero allocations per token.
pub fn regional_train_token_fast(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    ws: &mut TrainWorkspace,
    token: usize,
    target: usize,
) -> (f32, usize) {
    regional_train_token_with_loss(w, grads, ws, token, target, None)
}

/// Fast training with frozen cerebellum override.
pub fn regional_train_token_frozen(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    ws: &mut TrainWorkspace,
    token: usize,
    target: usize,
    frozen: &mut dyn crate::cerebellum::FrozenCerebellum,
) -> (f32, usize) {
    let obs = w.embed(token);
    let (loss, pred, d_obs) = regional_train_step_frozen(w, grads, obs, target, frozen);
    let d = w.config.raw_obs_dim;
    let offset = token * d;
    for j in 0..d {
        grads.embed_grad[offset + j] += d_obs[j];
    }
    let _ = ws; // TODO: pass workspace through to inner step
    (loss, pred)
}

/// Train one token with pre-computed cerebellum hidden state from cache.
/// The hidden state is injected into the cerebellum region output each tick.
pub fn regional_train_token_with_cereb(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    token: usize,
    target: usize,
    cereb_hidden: &[f32],
) -> (f32, usize) {
    let obs = w.embed(token);
    let (loss, pred, d_obs) = regional_train_step_with_cereb(w, grads, obs, target, cereb_hidden);
    let d = w.config.raw_obs_dim;
    let offset = token * d;
    for j in 0..d {
        grads.embed_grad[offset + j] += d_obs[j];
    }
    (loss, pred)
}

/// Fast training with pluggable loss function.
pub fn regional_train_token_with_loss(
    w: &RegionalWeights,
    grads: &mut RegionalGradients,
    ws: &mut TrainWorkspace,
    token: usize,
    target: usize,
    loss_fn: Option<&dyn modgrad_traits::LossFn<Target = modgrad_traits::ClassTarget>>,
) -> (f32, usize) {
    let _ = ws; // TODO: pass workspace through to inner step
    let obs = w.embed(token);
    let (loss, pred, d_obs) = regional_train_step_loss(w, grads, obs, target, loss_fn);

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

/// Typed Linear weight + bias gradient pair, used for connection
/// synapses and various single-Linear projections at the brain level.
pub struct LinearGradsTyped<D: TensorDeviceG> {
    pub d_w: TensorG<D>,
    pub d_b: TensorG<D>,
}

impl<D: TensorDeviceG> LinearGradsTyped<D> {
    pub fn zeros(lin: &modgrad_device::backend::tensor::Linear<D>)
        -> Result<Self, BackendErrorG>
    {
        Ok(Self {
            d_w: TensorG::<D>::zeros(lin.weight.len())?,
            d_b: TensorG::<D>::zeros(lin.bias.len())?,
        })
    }
}

/// All gradients for a typed regional brain. Mirrors the untyped
/// `RegionalGradients` shape in the subsystems we've ported. v0
/// scope: per-region CTM + UNet + connection synapses + obs_proj +
/// global_decay + output_proj + outer_exit_gate. Skipped:
/// extra_heads, cereb_predict, bg_value, router (subsystems still
/// untyped).
pub struct RegionalGradientsTyped<D: TensorDeviceG> {
    pub regions: Vec<crate::weights::CtmGradientsTyped<D>>,
    pub region_unets: Vec<crate::synapse::UNetGradsTyped<D>>,
    pub connection_synapses: Vec<LinearGradsTyped<D>>,
    pub obs_proj: LinearGradsTyped<D>,
    pub global_decay: TensorG<D>,
    pub output_proj: LinearGradsTyped<D>,
    pub outer_exit_gate: Option<LinearGradsTyped<D>>,
}

impl<D: TensorDeviceG> RegionalGradientsTyped<D> {
    /// Allocate zeroed gradient tensors matching the weight shapes
    /// in the supplied typed regional weights.
    pub fn zeros(w: &RegionalWeightsTyped<D>) -> Result<Self, BackendErrorG> {
        let regions = w.regions.iter()
            .map(crate::weights::CtmGradientsTyped::<D>::zeros)
            .collect::<Result<Vec<_>, _>>()?;
        let region_unets = w.regions.iter()
            .map(|r| crate::synapse::UNetGradsTyped::<D>::zeros(&r.synapse))
            .collect::<Result<Vec<_>, _>>()?;
        let connection_synapses = w.connection_synapses.iter()
            .map(LinearGradsTyped::<D>::zeros)
            .collect::<Result<Vec<_>, _>>()?;
        let obs_proj = LinearGradsTyped::<D>::zeros(&w.obs_proj)?;
        let global_decay = TensorG::<D>::zeros(w.global_decay.len())?;
        let output_proj = LinearGradsTyped::<D>::zeros(&w.output_proj)?;
        let outer_exit_gate = match &w.outer_exit_gate {
            Some(g) => Some(LinearGradsTyped::<D>::zeros(g)?),
            None => None,
        };
        Ok(Self {
            regions, region_unets, connection_synapses,
            obs_proj, global_decay, output_proj, outer_exit_gate,
        })
    }

    /// Reset every gradient tensor to zero. Mirrors the untyped
    /// `RegionalGradients::zero` pattern — call once per batch
    /// before accumulating per-sample backward results.
    ///
    /// **v0 implementation cost.** Currently re-allocates each
    /// tensor (`*t = TensorG::zeros(t.len())?`). For `D = Cpu` this
    /// is cheap; for `D = Rocm` it issues a fresh `hipMalloc` per
    /// tensor. A future op `Op::FillZeroResident` (single hipMemsetAsync
    /// per buffer) will close the gap. Documented as a known
    /// inefficiency — measure before optimising.
    pub fn zero(&mut self) -> Result<(), BackendErrorG> {
        for r in &mut self.regions {
            zero_ctm_grads::<D>(r)?;
        }
        for u in &mut self.region_unets {
            zero_unet_grads::<D>(u)?;
        }
        for cg in &mut self.connection_synapses {
            zero_linear_grads::<D>(cg)?;
        }
        zero_linear_grads::<D>(&mut self.obs_proj)?;
        let n = self.global_decay.len();
        self.global_decay = TensorG::<D>::zeros(n)?;
        zero_linear_grads::<D>(&mut self.output_proj)?;
        if let Some(g) = self.outer_exit_gate.as_mut() {
            zero_linear_grads::<D>(g)?;
        }
        Ok(())
    }

    /// Element-wise `dst += src` across every gradient tensor.
    /// Matches the untyped `accumulate(...)` helper used in babyai_probe;
    /// lets callers maintain a per-batch accumulator without
    /// re-allocating between samples.
    pub fn accumulate_from(&mut self, other: &Self) -> Result<(), BackendErrorG> {
        for (r_dst, r_src) in self.regions.iter_mut().zip(other.regions.iter()) {
            accumulate_ctm_grads::<D>(r_dst, r_src)?;
        }
        for (u_dst, u_src) in self.region_unets.iter_mut().zip(other.region_unets.iter()) {
            accumulate_unet_grads::<D>(u_dst, u_src)?;
        }
        for (c_dst, c_src) in self.connection_synapses.iter_mut().zip(other.connection_synapses.iter()) {
            accumulate_linear_grads::<D>(c_dst, c_src)?;
        }
        accumulate_linear_grads::<D>(&mut self.obs_proj, &other.obs_proj)?;
        let n = self.global_decay.len().min(other.global_decay.len());
        tensor_typed_g::add_assign::<D>(&mut self.global_decay, &other.global_decay, n)?;
        accumulate_linear_grads::<D>(&mut self.output_proj, &other.output_proj)?;
        if let (Some(g_dst), Some(g_src)) = (
            self.outer_exit_gate.as_mut(), other.outer_exit_gate.as_ref(),
        ) {
            accumulate_linear_grads::<D>(g_dst, g_src)?;
        }
        Ok(())
    }
}

fn zero_linear_grads<D: TensorDeviceG>(g: &mut LinearGradsTyped<D>) -> Result<(), BackendErrorG> {
    g.d_w = TensorG::<D>::zeros(g.d_w.len())?;
    g.d_b = TensorG::<D>::zeros(g.d_b.len())?;
    Ok(())
}

fn accumulate_linear_grads<D: TensorDeviceG>(
    dst: &mut LinearGradsTyped<D>, src: &LinearGradsTyped<D>,
) -> Result<(), BackendErrorG> {
    let nw = dst.d_w.len().min(src.d_w.len());
    let nb = dst.d_b.len().min(src.d_b.len());
    tensor_typed_g::add_assign::<D>(&mut dst.d_w, &src.d_w, nw)?;
    tensor_typed_g::add_assign::<D>(&mut dst.d_b, &src.d_b, nb)?;
    Ok(())
}

fn zero_ctm_grads<D: TensorDeviceG>(
    g: &mut crate::weights::CtmGradientsTyped<D>,
) -> Result<(), BackendErrorG> {
    macro_rules! zt { ($t:expr) => { $t = TensorG::<D>::zeros($t.len())?; }; }
    zt!(g.nlm_s1_w); zt!(g.nlm_s1_b);
    if let Some(t) = g.nlm_s2_w.as_mut() { *t = TensorG::<D>::zeros(t.len())?; }
    if let Some(t) = g.nlm_s2_b.as_mut() { *t = TensorG::<D>::zeros(t.len())?; }
    zt!(g.d_start_activated); zt!(g.d_start_trace);
    zt!(g.kv_proj_w); zt!(g.kv_proj_b);
    zt!(g.kv_ln_d_gamma); zt!(g.kv_ln_d_beta);
    zt!(g.q_proj_w); zt!(g.q_proj_b);
    zt!(g.mha_in_w); zt!(g.mha_in_b);
    zt!(g.mha_out_w); zt!(g.mha_out_b);
    zt!(g.d_decay_out); zt!(g.d_decay_action);
    zt!(g.out_proj_w); zt!(g.out_proj_b);
    if let Some(t) = g.exit_gate_w.as_mut() { *t = TensorG::<D>::zeros(t.len())?; }
    if let Some(t) = g.exit_gate_b.as_mut() { *t = TensorG::<D>::zeros(t.len())?; }
    Ok(())
}

fn accumulate_ctm_grads<D: TensorDeviceG>(
    dst: &mut crate::weights::CtmGradientsTyped<D>,
    src: &crate::weights::CtmGradientsTyped<D>,
) -> Result<(), BackendErrorG> {
    macro_rules! aa {
        ($d:expr, $s:expr) => {{
            let n = $d.len().min($s.len());
            tensor_typed_g::add_assign::<D>(&mut $d, &$s, n)?;
        }};
    }
    aa!(dst.nlm_s1_w, src.nlm_s1_w);
    aa!(dst.nlm_s1_b, src.nlm_s1_b);
    if let (Some(d), Some(s)) = (dst.nlm_s2_w.as_mut(), src.nlm_s2_w.as_ref()) {
        let n = d.len().min(s.len());
        tensor_typed_g::add_assign::<D>(d, s, n)?;
    }
    if let (Some(d), Some(s)) = (dst.nlm_s2_b.as_mut(), src.nlm_s2_b.as_ref()) {
        let n = d.len().min(s.len());
        tensor_typed_g::add_assign::<D>(d, s, n)?;
    }
    aa!(dst.d_start_activated, src.d_start_activated);
    aa!(dst.d_start_trace, src.d_start_trace);
    aa!(dst.kv_proj_w, src.kv_proj_w);
    aa!(dst.kv_proj_b, src.kv_proj_b);
    aa!(dst.kv_ln_d_gamma, src.kv_ln_d_gamma);
    aa!(dst.kv_ln_d_beta, src.kv_ln_d_beta);
    aa!(dst.q_proj_w, src.q_proj_w);
    aa!(dst.q_proj_b, src.q_proj_b);
    aa!(dst.mha_in_w, src.mha_in_w);
    aa!(dst.mha_in_b, src.mha_in_b);
    aa!(dst.mha_out_w, src.mha_out_w);
    aa!(dst.mha_out_b, src.mha_out_b);
    aa!(dst.d_decay_out, src.d_decay_out);
    aa!(dst.d_decay_action, src.d_decay_action);
    aa!(dst.out_proj_w, src.out_proj_w);
    aa!(dst.out_proj_b, src.out_proj_b);
    if let (Some(d), Some(s)) = (dst.exit_gate_w.as_mut(), src.exit_gate_w.as_ref()) {
        let n = d.len().min(s.len());
        tensor_typed_g::add_assign::<D>(d, s, n)?;
    }
    if let (Some(d), Some(s)) = (dst.exit_gate_b.as_mut(), src.exit_gate_b.as_ref()) {
        let n = d.len().min(s.len());
        tensor_typed_g::add_assign::<D>(d, s, n)?;
    }
    Ok(())
}

fn zero_unet_grads<D: TensorDeviceG>(
    g: &mut crate::synapse::UNetGradsTyped<D>,
) -> Result<(), BackendErrorG> {
    let zero_block = |b: &mut crate::synapse::SynapseBlockGradsTyped<D>| -> Result<(), BackendErrorG> {
        b.d_w = TensorG::<D>::zeros(b.d_w.len())?;
        b.d_b = TensorG::<D>::zeros(b.d_b.len())?;
        b.d_gamma = TensorG::<D>::zeros(b.d_gamma.len())?;
        b.d_beta = TensorG::<D>::zeros(b.d_beta.len())?;
        Ok(())
    };
    zero_block(&mut g.first)?;
    for d in &mut g.downs { zero_block(d)?; }
    for u in &mut g.ups { zero_block(u)?; }
    for t in &mut g.skip_d_gamma { *t = TensorG::<D>::zeros(t.len())?; }
    for t in &mut g.skip_d_beta { *t = TensorG::<D>::zeros(t.len())?; }
    Ok(())
}

fn accumulate_unet_grads<D: TensorDeviceG>(
    dst: &mut crate::synapse::UNetGradsTyped<D>,
    src: &crate::synapse::UNetGradsTyped<D>,
) -> Result<(), BackendErrorG> {
    let acc_block = |
        d: &mut crate::synapse::SynapseBlockGradsTyped<D>,
        s: &crate::synapse::SynapseBlockGradsTyped<D>,
    | -> Result<(), BackendErrorG> {
        let nw = d.d_w.len().min(s.d_w.len());
        tensor_typed_g::add_assign::<D>(&mut d.d_w, &s.d_w, nw)?;
        let nb = d.d_b.len().min(s.d_b.len());
        tensor_typed_g::add_assign::<D>(&mut d.d_b, &s.d_b, nb)?;
        let ng = d.d_gamma.len().min(s.d_gamma.len());
        tensor_typed_g::add_assign::<D>(&mut d.d_gamma, &s.d_gamma, ng)?;
        let nbg = d.d_beta.len().min(s.d_beta.len());
        tensor_typed_g::add_assign::<D>(&mut d.d_beta, &s.d_beta, nbg)?;
        Ok(())
    };
    acc_block(&mut dst.first, &src.first)?;
    let n_down = dst.downs.len().min(src.downs.len());
    for i in 0..n_down { acc_block(&mut dst.downs[i], &src.downs[i])?; }
    let n_up = dst.ups.len().min(src.ups.len());
    for i in 0..n_up { acc_block(&mut dst.ups[i], &src.ups[i])?; }
    let n_g = dst.skip_d_gamma.len().min(src.skip_d_gamma.len());
    for i in 0..n_g {
        let n = dst.skip_d_gamma[i].len().min(src.skip_d_gamma[i].len());
        tensor_typed_g::add_assign::<D>(&mut dst.skip_d_gamma[i], &src.skip_d_gamma[i], n)?;
    }
    let n_b = dst.skip_d_beta.len().min(src.skip_d_beta.len());
    for i in 0..n_b {
        let n = dst.skip_d_beta[i].len().min(src.skip_d_beta[i].len());
        tensor_typed_g::add_assign::<D>(&mut dst.skip_d_beta[i], &src.skip_d_beta[i], n)?;
    }
    Ok(())
}

/// Per-tick cache from `regional_forward_typed_with_cache`. Holds
/// every intermediate `regional_backward_typed` will need.
pub struct RegionTickCacheTyped<D: TensorDeviceG> {
    /// Region outputs at the START of this tick (the "prev_outputs"
    /// that fed the connection synapses). Used by the multi-region
    /// backward to flow gradient back through the matvec chain.
    pub prev_outputs_pre_tick: Vec<Vec<f32>>,
    /// One Vec per connection: the concat(source activations [+obs])
    /// fed into `connection_synapses[ci].forward`. Backward needs
    /// these to compute the synapse weight grads via Linear::backward.
    pub conn_inputs: Vec<Vec<f32>>,
    /// One CtmCacheTyped per region — drives ctm_backward_typed.
    pub region_ctm_caches: Vec<crate::forward::CtmCacheTyped<D>>,
    /// Concatenated post-tick region activations [total_neurons].
    /// Used by global_sync_reverse to scatter d_sync into per-region grads.
    pub all_act: Vec<f32>,
    pub global_alpha_pre: Vec<f32>,
    pub global_beta_pre: Vec<f32>,
    /// Normalised global sync [n_sync] fed into output_proj.
    pub gs_post_update: Vec<f32>,
}

/// Top-level cache from `regional_forward_typed_with_cache`.
pub struct RegionCacheTyped<D: TensorDeviceG> {
    pub obs: TensorG<D>,
    pub obs_proj_host: Vec<f32>,
    pub ticks: Vec<RegionTickCacheTyped<D>>,
}

// ─── RegionalWeightsTyped<D> — Path C JAX-style port ─────────
//
// Top-level brain data container. Mirrors `RegionalWeights`
// field-by-field with `Tensor<D>` / `Linear<D>` / `CtmWeightsTyped<D>`
// substitutions. Skipped subsystems (deferred, orthogonal):
//   - RegionalRouter (own type; complex MoS-style routing)
//   - cereb_projection (frozen cerebellum LLM bridge — its own crate piece)
//   - cereb_blend_logit (single scalar, host)
//   - generation counter (u64, host bookkeeping)

use modgrad_device::backend::tensor as tensor_typed_g;
use modgrad_device::backend::tensor::{Device as TensorDeviceG, Tensor as TensorG};
use modgrad_device::backend::BackendError as BackendErrorG;
use crate::weights::CtmWeightsTyped;

/// Path C typed top-level brain weights. Each f32 buffer in
/// `RegionalWeights` becomes a `Tensor<D>` here; each `Linear`
/// becomes `tensor::Linear<D>`; each per-region `CtmWeights`
/// becomes `CtmWeightsTyped<D>`.
pub struct RegionalWeightsTyped<D: TensorDeviceG> {
    pub config: RegionalConfig,
    pub embeddings: TensorG<D>,
    pub regions: Vec<CtmWeightsTyped<D>>,
    pub connection_synapses: Vec<tensor_typed_g::Linear<D>>,
    pub obs_proj: tensor_typed_g::Linear<D>,
    /// Sync indices stay as host-side integer LUTs.
    pub global_sync_left: Vec<usize>,
    pub global_sync_right: Vec<usize>,
    pub global_decay: TensorG<D>,
    pub output_proj: tensor_typed_g::Linear<D>,
    pub outer_exit_gate: Option<tensor_typed_g::Linear<D>>,
    pub extra_heads: Vec<tensor_typed_g::Linear<D>>,
    pub cereb_predict: Option<tensor_typed_g::Linear<D>>,
    pub bg_value: Option<tensor_typed_g::Linear<D>>,
}

impl<D: TensorDeviceG> RegionalWeightsTyped<D> {
    /// Migration constructor: transfer every buffer from the
    /// untyped `RegionalWeights` into device-resident `Tensor<D>`s.
    /// After this call, every weight read goes through the typed
    /// cascade — for `D = Rocm`, weights stay on-device.
    ///
    /// Skips: `router`, `cereb_projection`, `cereb_blend_logit`,
    /// `generation` (deferred subsystems).
    pub fn from_untyped(w: &RegionalWeights) -> Result<Self, BackendErrorG> {
        let regions = w.regions.iter()
            .map(CtmWeightsTyped::<D>::from_untyped)
            .collect::<Result<Vec<_>, _>>()?;

        let connection_synapses = w.connection_synapses.iter()
            .map(|l| tensor_typed_g::Linear::<D>::from_host(
                &l.weight, &l.bias, l.in_dim, l.out_dim,
            ))
            .collect::<Result<Vec<_>, _>>()?;

        let obs_proj = tensor_typed_g::Linear::<D>::from_host(
            &w.obs_proj.weight, &w.obs_proj.bias,
            w.obs_proj.in_dim, w.obs_proj.out_dim,
        )?;
        let output_proj = tensor_typed_g::Linear::<D>::from_host(
            &w.output_proj.weight, &w.output_proj.bias,
            w.output_proj.in_dim, w.output_proj.out_dim,
        )?;
        let outer_exit_gate = match &w.outer_exit_gate {
            Some(g) => Some(tensor_typed_g::Linear::<D>::from_host(
                &g.weight, &g.bias, g.in_dim, g.out_dim,
            )?),
            None => None,
        };
        let extra_heads = w.extra_heads.iter()
            .map(|h| tensor_typed_g::Linear::<D>::from_host(
                &h.weight, &h.bias, h.in_dim, h.out_dim,
            ))
            .collect::<Result<Vec<_>, _>>()?;
        let cereb_predict = match &w.cereb_predict {
            Some(h) => Some(tensor_typed_g::Linear::<D>::from_host(
                &h.weight, &h.bias, h.in_dim, h.out_dim,
            )?),
            None => None,
        };
        let bg_value = match &w.bg_value {
            Some(h) => Some(tensor_typed_g::Linear::<D>::from_host(
                &h.weight, &h.bias, h.in_dim, h.out_dim,
            )?),
            None => None,
        };

        Ok(Self {
            config: w.config.clone(),
            embeddings: TensorG::<D>::from_slice(&w.embeddings)?,
            regions, connection_synapses,
            obs_proj,
            global_sync_left: w.global_sync_left.clone(),
            global_sync_right: w.global_sync_right.clone(),
            global_decay: TensorG::<D>::from_slice(&w.global_decay)?,
            output_proj, outer_exit_gate, extra_heads,
            cereb_predict, bg_value,
        })
    }

    /// Multi-region forward pass through the typed cascade.
    /// v0 covers:
    ///   - Single observation token
    ///   - Fixed connection topology (no router)
    ///   - No episodic memory
    ///   - No outer exit gate
    ///   - ExitStrategy::None on the outer tick loop
    ///
    /// Each region's activated/trace persists in the caller-supplied
    /// `region_activated` / `region_trace` vectors (one Vec<f32> per region).
    /// Returns predictions per outer tick + final global_sync read.
    pub fn regional_forward_typed(
        &self,
        observation: &[f32],
        region_activated: &mut Vec<Vec<f32>>,
        region_trace: &mut Vec<Vec<f32>>,
        global_alpha: &mut Vec<f32>,
        global_beta: &mut Vec<f32>,
    ) -> Result<crate::forward::CtmOutput, BackendErrorG> {
        let cfg = &self.config;
        let n_regions = cfg.regions.len();
        let n_sync = cfg.n_global_sync;
        let n_outer = cfg.outer_ticks;

        // Project observation once for fallback (regions with no incoming edges).
        let obs_t = TensorG::<D>::from_slice(observation)?;
        let obs_proj_t = self.obs_proj.forward(&obs_t)?;
        let obs_proj_h: Vec<f32> = obs_proj_t.to_vec()?;

        // Pre-projection of obs is shared; per-region observation
        // buffers are built fresh each tick from the previous tick's
        // region outputs + connection synapses.
        let mut prev_outputs: Vec<Vec<f32>> = region_activated.clone();

        let mut predictions = Vec::with_capacity(n_outer);
        let mut certainties = Vec::with_capacity(n_outer);

        for _outer_tick in 0..n_outer {
            // Phase A: build region observations from connection synapses.
            let mut region_obs: Vec<Vec<f32>> = vec![Vec::new(); n_regions];
            for (ci, conn) in cfg.connections.iter().enumerate() {
                // src = concat(prev_outputs[from_idx]) + maybe observation slice
                let mut src: Vec<f32> = Vec::new();
                for &fi in &conn.from { src.extend_from_slice(&prev_outputs[fi]); }
                if conn.receives_observation {
                    let (s, len) = cfg.obs_scale_slice(conn.observation_scale);
                    let end = (s + len).min(observation.len());
                    if s < end { src.extend_from_slice(&observation[s..end]); }
                }
                if src.is_empty() { continue; }
                // Project through this connection's synapse.
                let src_t = TensorG::<D>::from_slice(&src)?;
                let proj_t = self.connection_synapses[ci].forward(&src_t)?;
                let proj_h: Vec<f32> = proj_t.to_vec()?;
                // Additive merge into target region's slot.
                let slot = &mut region_obs[conn.to];
                if slot.is_empty() {
                    *slot = proj_h;
                } else {
                    let n = slot.len().min(proj_h.len());
                    for i in 0..n { slot[i] += proj_h[i]; }
                }
            }
            // Regions with no incoming edges fall back to obs_projected.
            for r in 0..n_regions {
                if region_obs[r].is_empty() { region_obs[r] = obs_proj_h.clone(); }
            }

            // Phase B: per-region CTM forward. Drive each region with its
            // observation. Collect outputs (= activated state).
            let mut new_outputs: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
            for r in 0..n_regions {
                crate::forward::ctm_forward_typed(
                    &self.regions[r],
                    &mut region_activated[r],
                    &mut region_trace[r],
                    &region_obs[r],
                )?;
                new_outputs.push(region_activated[r].clone());
            }
            prev_outputs = new_outputs;

            // Phase C: global sync update from concat of all region activations.
            let total_neurons: usize = cfg.regions.iter().map(|r| r.d_model).sum();
            let mut all_act = vec![0.0f32; total_neurons];
            let mut offset = 0;
            for r in 0..n_regions {
                let d = prev_outputs[r].len();
                all_act[offset..offset + d].copy_from_slice(&prev_outputs[r]);
                offset += d;
            }
            let global_decay_h: Vec<f32> = self.global_decay.to_vec()?;
            for i in 0..n_sync {
                let l = self.global_sync_left[i];
                let r = self.global_sync_right[i];
                if l < all_act.len() && r < all_act.len() {
                    let pw = all_act[l] * all_act[r];
                    let decay = (-global_decay_h[i].clamp(0.0, 15.0)).exp();
                    global_alpha[i] = decay * global_alpha[i] + pw;
                    global_beta[i] = decay * global_beta[i] + 1.0;
                }
            }
            let gs: Vec<f32> = (0..n_sync)
                .map(|i| global_alpha[i] / global_beta[i].sqrt().max(1e-8))
                .collect();

            // Phase D: output projection.
            let gs_t = TensorG::<D>::from_slice(&gs)?;
            let pred_t = self.output_proj.forward(&gs_t)?;
            let pred_h: Vec<f32> = pred_t.to_vec()?;
            let cert = crate::forward::compute_certainty_pub(&pred_h);
            predictions.push(pred_h);
            certainties.push(cert);
        }

        let ticks_used = predictions.len();
        let final_gs: Vec<f32> = (0..n_sync)
            .map(|i| global_alpha[i] / global_beta[i].sqrt().max(1e-8))
            .collect();
        Ok(crate::forward::CtmOutput {
            predictions,
            certainties,
            sync_out: final_gs,
            exit_lambdas: Vec::new(),
            ticks_used,
            trajectory: Vec::new(),
        })
    }

    /// Forward + cache per outer tick. Same algorithm as
    /// `regional_forward_typed` but exposes every intermediate the
    /// matched `regional_backward_typed` will need (per-tick connection
    /// inputs, per-region CTM caches, global-sync state).
    pub fn regional_forward_typed_with_cache(
        &self,
        observation: &[f32],
        region_activated: &mut Vec<Vec<f32>>,
        region_trace: &mut Vec<Vec<f32>>,
        global_alpha: &mut Vec<f32>,
        global_beta: &mut Vec<f32>,
    ) -> Result<(crate::forward::CtmOutput, RegionCacheTyped<D>), BackendErrorG> {
        let cfg = &self.config;
        let n_regions = cfg.regions.len();
        let n_sync = cfg.n_global_sync;
        let n_outer = cfg.outer_ticks;

        let obs_t = TensorG::<D>::from_slice(observation)?;
        let obs_proj_t = self.obs_proj.forward(&obs_t)?;
        let obs_proj_h: Vec<f32> = obs_proj_t.to_vec()?;

        let mut prev_outputs: Vec<Vec<f32>> = region_activated.clone();

        let mut predictions = Vec::with_capacity(n_outer);
        let mut certainties = Vec::with_capacity(n_outer);
        let mut tick_caches: Vec<RegionTickCacheTyped<D>> = Vec::with_capacity(n_outer);

        for _outer_tick in 0..n_outer {
            let prev_outputs_snapshot = prev_outputs.clone();
            let global_alpha_pre = global_alpha.clone();
            let global_beta_pre = global_beta.clone();

            // Phase A: build region observations from connection synapses.
            // Cache the per-connection inputs (= concat(prev_outputs[from]) + maybe obs slice).
            let mut conn_inputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.connections.len());
            let mut region_obs: Vec<Vec<f32>> = vec![Vec::new(); n_regions];
            for (ci, conn) in cfg.connections.iter().enumerate() {
                let mut src: Vec<f32> = Vec::new();
                for &fi in &conn.from { src.extend_from_slice(&prev_outputs[fi]); }
                if conn.receives_observation {
                    let (s, len) = cfg.obs_scale_slice(conn.observation_scale);
                    let end = (s + len).min(observation.len());
                    if s < end { src.extend_from_slice(&observation[s..end]); }
                }
                conn_inputs.push(src.clone());
                if src.is_empty() { continue; }
                let src_t = TensorG::<D>::from_slice(&src)?;
                let proj_t = self.connection_synapses[ci].forward(&src_t)?;
                let proj_h: Vec<f32> = proj_t.to_vec()?;
                let slot = &mut region_obs[conn.to];
                if slot.is_empty() {
                    *slot = proj_h;
                } else {
                    let n = slot.len().min(proj_h.len());
                    for i in 0..n { slot[i] += proj_h[i]; }
                }
            }
            for r in 0..n_regions {
                if region_obs[r].is_empty() { region_obs[r] = obs_proj_h.clone(); }
            }

            // Phase B: per-region CTM forward with cache.
            let mut region_ctm_caches: Vec<crate::forward::CtmCacheTyped<D>>
                = Vec::with_capacity(n_regions);
            let mut new_outputs: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
            for r in 0..n_regions {
                let (_out, cache) = crate::forward::ctm_forward_typed_with_cache(
                    &self.regions[r],
                    &mut region_activated[r],
                    &mut region_trace[r],
                    &region_obs[r],
                )?;
                region_ctm_caches.push(cache);
                new_outputs.push(region_activated[r].clone());
            }
            prev_outputs = new_outputs;

            // Phase C: global sync update.
            let total_neurons: usize = cfg.regions.iter().map(|r| r.d_model).sum();
            let mut all_act = vec![0.0f32; total_neurons];
            let mut offset = 0;
            for r in 0..n_regions {
                let d = prev_outputs[r].len();
                all_act[offset..offset + d].copy_from_slice(&prev_outputs[r]);
                offset += d;
            }
            let global_decay_h: Vec<f32> = self.global_decay.to_vec()?;
            for i in 0..n_sync {
                let l = self.global_sync_left[i];
                let r = self.global_sync_right[i];
                if l < all_act.len() && r < all_act.len() {
                    let pw = all_act[l] * all_act[r];
                    let decay = (-global_decay_h[i].clamp(0.0, 15.0)).exp();
                    global_alpha[i] = decay * global_alpha[i] + pw;
                    global_beta[i] = decay * global_beta[i] + 1.0;
                }
            }
            let gs: Vec<f32> = (0..n_sync)
                .map(|i| global_alpha[i] / global_beta[i].sqrt().max(1e-8))
                .collect();

            // Phase D: output projection.
            let gs_t = TensorG::<D>::from_slice(&gs)?;
            let pred_t = self.output_proj.forward(&gs_t)?;
            let pred_h: Vec<f32> = pred_t.to_vec()?;
            let cert = crate::forward::compute_certainty_pub(&pred_h);

            tick_caches.push(RegionTickCacheTyped {
                prev_outputs_pre_tick: prev_outputs_snapshot,
                conn_inputs,
                region_ctm_caches,
                all_act,
                global_alpha_pre,
                global_beta_pre,
                gs_post_update: gs,
            });
            predictions.push(pred_h);
            certainties.push(cert);
        }

        let ticks_used = predictions.len();
        let final_gs: Vec<f32> = (0..n_sync)
            .map(|i| global_alpha[i] / global_beta[i].sqrt().max(1e-8))
            .collect();
        let output = crate::forward::CtmOutput {
            predictions, certainties,
            sync_out: final_gs.clone(),
            exit_lambdas: Vec::new(),
            ticks_used,
            trajectory: Vec::new(),
        };
        let cache = RegionCacheTyped {
            obs: obs_t,
            obs_proj_host: obs_proj_h,
            ticks: tick_caches,
        };
        Ok((output, cache))
    }

    /// Multi-region backward — composes everything in reverse-tick.
    /// For each outer tick (reverse):
    ///   1. output_proj backward → d_gs
    ///   2. global_sync_reverse → d_all_act (per-region d_activated)
    ///   3. For each region: ctm_backward_typed → d_region_obs
    ///   4. For each connection: Linear<D>::backward → d_conn_input
    ///      Aggregate d_conn_input back into d_prev_outputs[from_idx]
    ///      and d_obs if the connection receives_observation
    ///
    /// **Output convention:** all weight grads ACCUMULATE. `d_observation`
    /// is OVERWRITTEN with the gradient flowing back into the raw
    /// observation tensor — useful for upstream encoder backprop.
    ///
    /// Existing callers that don't need `d_observation` should call
    /// `regional_backward_typed` instead, which allocates and discards
    /// the buffer internally.
    pub fn regional_backward_typed_with_d_obs(
        &self,
        cache: &RegionCacheTyped<D>,
        d_predictions: &[Vec<f32>],
        grads: &mut RegionalGradientsTyped<D>,
        d_observation: &mut TensorG<D>,
    ) -> Result<(), BackendErrorG> {
        let cfg = &self.config;
        let n_regions = cfg.regions.len();
        let n_sync = cfg.n_global_sync;
        let n_outer = cfg.outer_ticks.min(d_predictions.len()).min(cache.ticks.len());

        // Host-side d_observation accumulator. Two sources contribute:
        //   1. Per-connection tail bytes after the from-segments of each
        //      `conn` with `receives_observation = true` — sliced into
        //      d_obs_host[s..s+len] per `cfg.obs_scale_slice(scale)`.
        //   2. obs_proj backward when any region falls back to obs_proj
        //      (no incoming connection at this tick) — adds the full
        //      d_x_t of length raw_obs_dim.
        let raw_obs_dim = cfg.raw_obs_dim;
        let mut d_obs_host = vec![0.0f32; raw_obs_dim];

        // Pre-build incoming-connection list per region.
        let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n_regions];
        for (ci, conn) in cfg.connections.iter().enumerate() {
            if conn.to < n_regions { incoming[conn.to].push(ci); }
        }

        // Cache global_decay host-side once (was being read every tick).
        let global_decay_h: Vec<f32> = self.global_decay.to_vec()?;

        // d_decay_host accumulates the brain-level decay-parameter
        // gradient across every outer tick. Forward:
        //   alpha_post[i] = decay * alpha_pre[i] + product
        //   decay = exp(-clamp(decay_param, 0, 15))
        // Dominant chain (alpha path; matches the alpha-only convention
        // used by sync_update_reverse_host):
        //   d alpha_post / d decay      = alpha_pre
        //   d decay      / d decay_param = -decay  (within clamp range)
        // ⇒ d_decay_param[i] += -d_alpha * decay * alpha_pre[i]
        let mut d_decay_host: Vec<f32> = vec![0.0f32; n_sync];

        // Cross-tick carry of d_activated_post for each region: written
        // by the connection-synapse backward at tick T and consumed at
        // tick T-1 (added on top of d_per_region from sync_update_reverse).
        let mut carry_d_prev: Vec<Vec<f32>> = (0..n_regions)
            .map(|r| vec![0.0f32; self.regions[r].config.d_model])
            .collect();

        // Accumulator for the obs_proj fallback path. d_obs_proj_h
        // accumulates contributions from any region that fell back to
        // obs_proj (had no incoming connection at this tick), summed
        // over all ticks. We run obs_proj.backward once after the loop.
        let obs_proj_out_dim = self.obs_proj.out_dim;
        let mut d_obs_proj_h_acc = vec![0.0f32; obs_proj_out_dim];
        let mut obs_proj_active = false;

        for tick_idx in (0..n_outer).rev() {
            let tick_cache = &cache.ticks[tick_idx];

            // 1. output_proj backward.
            let d_pred_t = TensorG::<D>::from_slice(&d_predictions[tick_idx])?;
            let gs_t = TensorG::<D>::from_slice(&tick_cache.gs_post_update)?;
            let mut d_gs = TensorG::<D>::zeros(n_sync)?;
            self.output_proj.backward(
                &d_pred_t, &gs_t,
                &mut grads.output_proj.d_w, &mut grads.output_proj.d_b,
                &mut d_gs,
            )?;
            let d_gs_h: Vec<f32> = d_gs.to_vec()?;

            // 2. Global sync reverse — scatter d_gs into d_all_act and
            // accumulate the per-pair decay-parameter gradient.
            // Forward: gs[i] = alpha[i]/sqrt(beta[i]) where
            //   alpha[i] = decay·alpha_prev + activated[left]·activated[right]
            // Dominant gradient (matches sync_update_reverse_host):
            //   d_alpha[i] = d_gs[i] / sqrt(beta[i])
            //   d_act[left]  += d_alpha · activated[right]
            //   d_act[right] += d_alpha · activated[left]
            //   d_decay_param[i] += -d_alpha · decay · alpha_pre[i]
            let total_neurons: usize = cfg.regions.iter().map(|r| r.d_model).sum();
            let mut d_all_act = vec![0.0f32; total_neurons];
            for i in 0..n_sync {
                let l = self.global_sync_left[i];
                let r = self.global_sync_right[i];
                if l >= total_neurons || r >= total_neurons { continue; }
                let raw = global_decay_h[i];
                let decay = (-raw.clamp(0.0, 15.0)).exp();
                let beta_post = decay * tick_cache.global_beta_pre[i] + 1.0;
                let inv_sqrt_beta = 1.0 / beta_post.max(1e-8).sqrt();
                let d_alpha = d_gs_h[i] * inv_sqrt_beta;
                d_all_act[l] += d_alpha * tick_cache.all_act[r];
                d_all_act[r] += d_alpha * tick_cache.all_act[l];
                // Decay grad — only flows in the unclamped interior.
                if raw > 0.0 && raw < 15.0 {
                    d_decay_host[i] += -d_alpha * decay * tick_cache.global_alpha_pre[i];
                }
            }

            // Split d_all_act per-region; add cross-tick carry from
            // tick T+1 (= what was the next iteration's connection-synapse
            // backward output, indexed back into this tick's outputs).
            let mut d_per_region: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
            let mut offset = 0;
            for r in 0..n_regions {
                let d = self.regions[r].config.d_model;
                let mut slice = d_all_act[offset..offset + d].to_vec();
                for i in 0..d { slice[i] += carry_d_prev[r][i]; }
                d_per_region.push(slice);
                offset += d;
            }

            // Reset carry — we'll overwrite it below as the connection
            // backward produces new d_prev_outputs splits for tick T-1.
            for v in carry_d_prev.iter_mut() {
                v.iter_mut().for_each(|x| *x = 0.0);
            }

            // 3. Per-region ctm_backward_from_d_activated → d_obs_per_region.
            // Each region emits a d_obs vector of shape = the region's
            // input dim (= region_obs[r].len() in forward = obs_proj_out_dim
            // for fallback, or connection_synapses[ci].out_dim for connected).
            let mut d_obs_per_region: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
            for r in 0..n_regions {
                let region_cache = &tick_cache.region_ctm_caches[r];
                let region_input_dim = region_cache.raw_input_dim;
                let d_act_t = TensorG::<D>::from_slice(&d_per_region[r])?;
                let mut d_obs_r = TensorG::<D>::zeros(region_input_dim)?;
                crate::forward::ctm_backward_from_d_activated::<D>(
                    &self.regions[r],
                    region_cache,
                    &d_act_t,
                    &mut grads.regions[r],
                    &mut grads.region_unets[r],
                    &mut d_obs_r,
                )?;
                d_obs_per_region.push(d_obs_r.to_vec()?);
            }

            // 4. Per-connection Linear backward — accumulate d_w/d_b for
            // every connection that contributed to its target region;
            // split d_src into carry_d_prev[from_idx] for tick T-1.
            // Regions without any incoming connection at this tick had
            // their region_obs filled by obs_proj_h — route d_obs_per_region[r]
            // into the obs_proj_h accumulator instead.
            for r in 0..n_regions {
                if incoming[r].is_empty() {
                    let d_obs_r = &d_obs_per_region[r];
                    let n = d_obs_r.len().min(d_obs_proj_h_acc.len());
                    for i in 0..n { d_obs_proj_h_acc[i] += d_obs_r[i]; }
                    obs_proj_active = true;
                    continue;
                }
                let d_proj_r = &d_obs_per_region[r];
                if d_proj_r.is_empty() { continue; }

                for &ci in &incoming[r] {
                    let conn = &cfg.connections[ci];
                    let conn_input_h = &tick_cache.conn_inputs[ci];
                    if conn_input_h.is_empty() { continue; }

                    // d_y is shared across all incoming connections (forward
                    // accumulated `slot += proj`), so each ci sees the same
                    // d_proj for region r.
                    let d_y_t = TensorG::<D>::from_slice(d_proj_r)?;
                    let x_t = TensorG::<D>::from_slice(conn_input_h)?;
                    let mut d_x_t = TensorG::<D>::zeros(conn_input_h.len())?;
                    let cg = &mut grads.connection_synapses[ci];
                    self.connection_synapses[ci].backward(
                        &d_y_t, &x_t,
                        &mut cg.d_w, &mut cg.d_b,
                        &mut d_x_t,
                    )?;
                    let d_x_h: Vec<f32> = d_x_t.to_vec()?;

                    // Split d_x_h into per-from segments + optional
                    // observation tail. Layout matches forward's
                    // `src.extend_from_slice(&prev_outputs[fi])` for each fi.
                    let mut off = 0;
                    for &fi in &conn.from {
                        if fi >= n_regions { break; }
                        let d_from = self.regions[fi].config.d_model;
                        if off + d_from > d_x_h.len() { break; }
                        for i in 0..d_from {
                            carry_d_prev[fi][i] += d_x_h[off + i];
                        }
                        off += d_from;
                    }
                    // receives_observation tail: d_x_h[off..off+len]
                    // feeds the raw-observation slice for this connection's
                    // scale. Forward did `src.extend_from_slice(&observation[s..s+len])`,
                    // so backward routes the same range back into d_obs_host.
                    if conn.receives_observation {
                        let (s, len) = cfg.obs_scale_slice(conn.observation_scale);
                        let avail = d_x_h.len().saturating_sub(off);
                        let n = len.min(avail).min(d_obs_host.len().saturating_sub(s));
                        for i in 0..n {
                            d_obs_host[s + i] += d_x_h[off + i];
                        }
                    }
                }
            }
        }

        // After tick loop: obs_proj backward (one-shot) if any region
        // fell back to obs_proj across any tick. Forward did
        // `obs_proj_t = obs_proj.forward(obs)` once before the tick loop,
        // so backward is a single Linear::backward against the cached
        // raw obs.
        if obs_proj_active {
            let d_y_t = TensorG::<D>::from_slice(&d_obs_proj_h_acc)?;
            let mut d_x_t = TensorG::<D>::zeros(cache.obs.len())?;
            self.obs_proj.backward(
                &d_y_t, &cache.obs,
                &mut grads.obs_proj.d_w, &mut grads.obs_proj.d_b,
                &mut d_x_t,
            )?;
            // d_x_t covers the full observation; add it to d_obs_host.
            let d_x_h: Vec<f32> = d_x_t.to_vec()?;
            let n = d_x_h.len().min(d_obs_host.len());
            for i in 0..n { d_obs_host[i] += d_x_h[i]; }
        }

        // Flush accumulated decay-parameter gradient into grads.global_decay
        // via add_assign. The optimiser will then step `w.global_decay`
        // with this signal (RegionalAdamWTyped::step uses weight_decay=0
        // for it). Per-region decay_params_out / decay_params_action stay
        // on the same v0 alpha-only approximation as before.
        let n_decay = grads.global_decay.len().min(d_decay_host.len());
        if n_decay > 0 {
            let d_decay_t = TensorG::<D>::from_slice(&d_decay_host)?;
            tensor_typed_g::add_assign::<D>(&mut grads.global_decay, &d_decay_t, n_decay)?;
        }

        // Write the host-side d_observation accumulator into the caller's
        // tensor. Overwrite semantics: replace the buffer with the
        // accumulated gradient (caller can chain encoder backward on it).
        *d_observation = TensorG::<D>::from_slice(&d_obs_host)?;

        Ok(())
    }

    /// Convenience wrapper that calls `regional_backward_typed_with_d_obs`
    /// with a discarded `d_observation` buffer. Existing callers that
    /// don't need upstream encoder backprop should keep using this.
    pub fn regional_backward_typed(
        &self,
        cache: &RegionCacheTyped<D>,
        d_predictions: &[Vec<f32>],
        grads: &mut RegionalGradientsTyped<D>,
    ) -> Result<(), BackendErrorG> {
        let mut throwaway = TensorG::<D>::zeros(self.config.raw_obs_dim)?;
        self.regional_backward_typed_with_d_obs(cache, d_predictions, grads, &mut throwaway)
    }

    /// Total trainable parameters — matches `RegionalWeights::n_params`
    /// for the ported subsystems. Excludes deferred ones (router,
    /// cereb_projection) which would add their own counts.
    pub fn n_params(&self) -> usize {
        let mut n = self.embeddings.len();
        for r in &self.regions { n += r.n_params(); }
        for s in &self.connection_synapses {
            n += s.weight.len() + s.bias.len();
        }
        n += self.obs_proj.weight.len() + self.obs_proj.bias.len();
        n += self.global_decay.len();
        n += self.output_proj.weight.len() + self.output_proj.bias.len();
        if let Some(g) = &self.outer_exit_gate {
            n += g.weight.len() + g.bias.len();
        }
        for h in &self.extra_heads {
            n += h.weight.len() + h.bias.len();
        }
        if let Some(h) = &self.cereb_predict {
            n += h.weight.len() + h.bias.len();
        }
        if let Some(h) = &self.bg_value {
            n += h.weight.len() + h.bias.len();
        }
        n
    }
}

#[cfg(test)]
mod regional_weights_typed_tests {
    use super::*;
    use modgrad_device::backend::tensor::Cpu;
    #[cfg(feature = "rocm")]
    use modgrad_device::backend::tensor::Rocm;

    fn small_regional_cfg() -> RegionalConfig {
        // Smallest sensible 8-region brain.
        RegionalConfig::eight_region_small(16, 32, 2)
    }

    /// The fold (M1): a `VinReadout` folded into the hippocampus slot must make
    /// the brain plan IDENTICALLY to the standalone VIN — the integration is a
    /// rehoming, not a behaviour change. Guards `fold_vin`'s bit-exact contract
    /// without needing the trained model files.
    #[test]
    fn folded_planner_matches_standalone_vin() {
        use crate::vin::{VinConfig, VinReadout};
        let vin = VinReadout::new(VinConfig::default(), 3);
        let w = RegionalWeights::new(small_regional_cfg()).with_planner(vin.clone());
        assert!(w.has_planner(), "planner must be folded in");

        // synthetic 5x5 grid: all-open, goal at the far corner, agent near origin.
        let (h, wd) = (5usize, 5usize);
        let mut tokens = vec![0.0f32; h * wd * 3];
        for cell in 0..h * wd {
            tokens[cell * 3] = 1.0; // is_open
            tokens[cell * 3 + 2] = 1.0; // bias
        }
        let goal = (h - 1) * wd + (wd - 1);
        tokens[goal * 3 + 1] = 1.0; // is_goal
        let agent = Some((1usize, 1usize));

        let base = vin.forward(&tokens, h, wd, agent);
        let fold = w.plan(&tokens, h, wd, agent).expect("planner present");
        assert_eq!(
            base.move_logits, fold.move_logits,
            "folded planner must reproduce the standalone VIN bit-for-bit"
        );
    }

    #[test]
    fn cpu_regional_weights_typed_round_trip() {
        let cfg = small_regional_cfg();
        let w = RegionalWeights::new(cfg);
        let typed = RegionalWeightsTyped::<Cpu>::from_untyped(&w).expect("from_untyped");

        // Sub-systems we ported should match in n_params.
        // (Untyped n_params includes router params if router is Some;
        // typed n_params excludes router. Compute the delta.)
        let untyped_total = w.n_params();
        let router_params: usize = w.router.as_ref().map_or(0, |r| r.n_params());
        let cereb_params: usize = w.cereb_projection.as_ref().map_or(0, |p|
            p.proj_out_w.len() + p.proj_out_b.len());
        let cereb_blend: usize = w.cereb_blend_logit.is_some().then_some(1).unwrap_or(0);
        let expected_typed = untyped_total - router_params - cereb_params - cereb_blend;
        assert_eq!(typed.n_params(), expected_typed,
            "typed n_params {} should match untyped {} minus router={} cereb={} blend={}",
            typed.n_params(), untyped_total, router_params, cereb_params, cereb_blend);

        // Region count + connection count preserved.
        assert_eq!(typed.regions.len(), w.regions.len());
        assert_eq!(typed.connection_synapses.len(), w.connection_synapses.len());

        // Embedding round-trip.
        let emb_h = typed.embeddings.to_vec().unwrap();
        assert_eq!(emb_h, w.embeddings);
    }

    #[test]
    fn cpu_regional_forward_typed_multiscale_smoke() {
        // Use a multi-scale config (different obs_scale_dims per
        // connection) and confirm regional_forward_typed slices the
        // observation by `obs_scale_slice(conn.observation_scale)`
        // rather than feeding the whole observation. The connection
        // synapses are sized for `len(from) + obs_scale_dims[scale]`
        // so a wrong-sized observation extension would surface as
        // a shape mismatch in connection_synapses[*].forward.
        use crate::graph::Connection;
        let scales = vec![64, 32, 16];
        let mut cfg = RegionalConfig::eight_region_small_multiscale(
            &scales, 8, 1,
        );
        cfg.router = None;
        // Spread connections across scales: ci % 3 picks scale 0/1/2.
        for (ci, conn) in cfg.connections.iter_mut().enumerate() {
            if conn.receives_observation {
                conn.observation_scale = ci % scales.len();
            }
        }
        let _ = Connection { from: vec![], to: 0, receives_observation: false, observation_scale: 0 };

        let w = RegionalWeights::new(cfg);
        let typed = RegionalWeightsTyped::<Cpu>::from_untyped(&w).unwrap();

        // raw_obs_dim = sum of scales = 112
        let obs_dim = typed.config.raw_obs_dim;
        assert_eq!(obs_dim, scales.iter().sum::<usize>());
        let observation: Vec<f32> = (0..obs_dim)
            .map(|i| ((i + 1) as f32 * 0.097).cos()).collect();

        let mut region_activated: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_activated.clone()).collect();
        let mut region_trace: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_trace.clone()).collect();
        let n_sync = typed.config.n_global_sync;
        let mut global_alpha = vec![0.0f32; n_sync];
        let mut global_beta = vec![1.0f32; n_sync];

        let out = typed.regional_forward_typed(
            &observation,
            &mut region_activated, &mut region_trace,
            &mut global_alpha, &mut global_beta,
        ).expect("multi-scale regional_forward_typed");
        assert_eq!(out.predictions.len(), typed.config.outer_ticks);
    }

    #[test]
    fn cpu_regional_forward_typed_smoke() {
        // Build a small typed brain, run forward through every region,
        // verify it produces predictions of the right shape and that
        // every region's activated state changes (gradient... wait,
        // not gradient — just that the typed multi-region forward
        // composes through connection synapses without panicking).
        let cfg = small_regional_cfg();
        let w = RegionalWeights::new(cfg);
        let typed = RegionalWeightsTyped::<Cpu>::from_untyped(&w).unwrap();

        let obs_dim = typed.config.raw_obs_dim;
        let observation: Vec<f32> = (0..obs_dim)
            .map(|i| ((i + 1) as f32 * 0.137).sin()).collect();

        // Initialise per-region activated/trace from start states.
        let mut region_activated: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_activated.clone()).collect();
        let mut region_trace: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_trace.clone()).collect();
        let n_sync = typed.config.n_global_sync;
        let mut global_alpha = vec![0.0f32; n_sync];
        let mut global_beta = vec![1.0f32; n_sync];

        let out = typed.regional_forward_typed(
            &observation,
            &mut region_activated, &mut region_trace,
            &mut global_alpha, &mut global_beta,
        ).expect("regional_forward_typed");

        assert_eq!(out.predictions.len(), typed.config.outer_ticks);
        for p in &out.predictions {
            assert_eq!(p.len(), typed.config.out_dims);
            // Predictions should not all be zero (something flowed).
            assert!(p.iter().any(|&x| x.abs() > 1e-6),
                "prediction all zero — multi-region forward did not produce output");
        }
        assert_eq!(out.sync_out.len(), n_sync);
    }

    #[test]
    fn cpu_regional_forward_typed_with_cache_smoke() {
        let cfg = small_regional_cfg();
        let w = RegionalWeights::new(cfg);
        let typed = RegionalWeightsTyped::<Cpu>::from_untyped(&w).unwrap();

        let obs_dim = typed.config.raw_obs_dim;
        let observation: Vec<f32> = (0..obs_dim)
            .map(|i| ((i + 1) as f32 * 0.137).sin()).collect();
        let mut region_activated: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_activated.clone()).collect();
        let mut region_trace: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_trace.clone()).collect();
        let n_sync = typed.config.n_global_sync;
        let mut global_alpha = vec![0.0f32; n_sync];
        let mut global_beta = vec![1.0f32; n_sync];

        let (out, cache) = typed.regional_forward_typed_with_cache(
            &observation,
            &mut region_activated, &mut region_trace,
            &mut global_alpha, &mut global_beta,
        ).expect("forward+cache");

        // Cache structural sanity.
        assert_eq!(cache.ticks.len(), typed.config.outer_ticks);
        for (tick_idx, tick) in cache.ticks.iter().enumerate() {
            assert_eq!(tick.region_ctm_caches.len(), typed.regions.len(),
                "tick {} region cache count mismatch", tick_idx);
            assert_eq!(tick.conn_inputs.len(), typed.config.connections.len(),
                "tick {} conn input count mismatch", tick_idx);
            assert_eq!(tick.gs_post_update.len(), n_sync,
                "tick {} gs len mismatch", tick_idx);
        }
        // Predictions still produced normally.
        assert_eq!(out.predictions.len(), typed.config.outer_ticks);
    }

    #[test]
    fn cpu_regional_backward_typed_full_chain() {
        // Full regional backward: output_proj + global_sync_reverse +
        // per-region ctm_backward_from_d_activated + connection synapse
        // backward, with cross-tick activated carry through connection
        // d_src splits.
        //
        // Asserts that gradient reaches multiple subsystems:
        //   - brain-level output_proj (top of the chain)
        //   - per-region NLM / MHA / kv_proj (via ctm_backward_from_d_activated)
        //   - connection_synapses (via Linear<D>::backward)
        //
        // The per-region `out_proj_w` (= each CTM's own output_proj
        // for per-tick predictions) stays zero by design: the regional
        // cascade only consumes a region's final activated[] state
        // through global sync, never its per-tick predictions.
        let cfg = small_regional_cfg();
        let w = RegionalWeights::new(cfg);
        let typed = RegionalWeightsTyped::<Cpu>::from_untyped(&w).unwrap();

        let obs_dim = typed.config.raw_obs_dim;
        let observation: Vec<f32> = (0..obs_dim)
            .map(|i| ((i + 1) as f32 * 0.137).sin()).collect();
        let mut region_activated: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_activated.clone()).collect();
        let mut region_trace: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_trace.clone()).collect();
        let n_sync = typed.config.n_global_sync;
        let mut global_alpha = vec![0.0f32; n_sync];
        let mut global_beta = vec![1.0f32; n_sync];

        let (out, cache) = typed.regional_forward_typed_with_cache(
            &observation,
            &mut region_activated, &mut region_trace,
            &mut global_alpha, &mut global_beta,
        ).unwrap();

        let d_preds: Vec<Vec<f32>> = out.predictions.iter()
            .map(|p| vec![1.0; p.len()]).collect();

        let mut grads = RegionalGradientsTyped::<Cpu>::zeros(&typed).unwrap();
        typed.regional_backward_typed(&cache, &d_preds, &mut grads)
            .expect("regional_backward_typed");

        // Brain-level output_proj must flow.
        let dw = grads.output_proj.d_w.to_vec().unwrap();
        assert!(dw.iter().any(|&x| x.abs() > 1e-6),
            "brain output_proj.d_w all zero");
        let db = grads.output_proj.d_b.to_vec().unwrap();
        assert!(db.iter().any(|&x| x.abs() > 1e-6),
            "brain output_proj.d_b all zero");

        // Per-region NLM stage 1 weight — driven by d_activated
        // through nlm_backward inside ctm_backward_from_d_activated.
        let any_region_nlm_flowed = (0..typed.regions.len()).any(|r| {
            let dw = grads.regions[r].nlm_s1_w.to_vec().unwrap();
            dw.iter().any(|&x| x.abs() > 1e-6)
        });
        assert!(any_region_nlm_flowed,
            "no region's nlm_s1_w flowed gradient — ctm_backward_from_d_activated path broken");

        // Per-region kv_proj weight — driven by d_obs feeding back
        // through kv_ln + kv_proj at the end of the inner-tick walk.
        let any_region_kv_flowed = (0..typed.regions.len()).any(|r| {
            let dw = grads.regions[r].kv_proj_w.to_vec().unwrap();
            dw.iter().any(|&x| x.abs() > 1e-6)
        });
        assert!(any_region_kv_flowed,
            "no region's kv_proj_w flowed gradient");

        // At least one connection_synapse must flow gradient (every
        // connection feeds gradient via Linear<D>::backward when its
        // target region's d_obs is non-zero).
        let any_conn_flowed = (0..typed.connection_synapses.len()).any(|ci| {
            let dw = grads.connection_synapses[ci].d_w.to_vec().unwrap();
            dw.iter().any(|&x| x.abs() > 1e-6)
        });
        assert!(any_conn_flowed,
            "no connection_synapse weight flowed gradient");

        // Per-region output_proj (per-tick prediction head inside CTM)
        // is intentionally dead in the regional cascade — confirm.
        let region0_out_proj = grads.regions[0].out_proj_w.to_vec().unwrap();
        assert!(region0_out_proj.iter().all(|&x| x == 0.0),
            "regions[0].out_proj_w should be zero — regional cascade bypasses per-region prediction head");
    }

    #[test]
    fn cpu_regional_backward_typed_flows_d_observation() {
        // When at least one connection has receives_observation=true,
        // regional_backward_typed_with_d_obs should populate d_observation
        // with non-zero gradient for the bytes that connection consumed.
        // small_regional_cfg's eight_region_small has connection 0
        // (MOTOR→INPUT) with receives_observation=true.
        let cfg = small_regional_cfg();
        let w = RegionalWeights::new(cfg);
        let typed = RegionalWeightsTyped::<Cpu>::from_untyped(&w).unwrap();
        let n_sync = typed.config.n_global_sync;
        let raw_obs_dim = typed.config.raw_obs_dim;

        let observation: Vec<f32> = (0..raw_obs_dim)
            .map(|i| ((i + 1) as f32 * 0.137).sin()).collect();
        let mut region_activated: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_activated.clone()).collect();
        let mut region_trace: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_trace.clone()).collect();
        let mut global_alpha = vec![0.0f32; n_sync];
        let mut global_beta = vec![1.0f32; n_sync];

        let (out, cache) = typed.regional_forward_typed_with_cache(
            &observation,
            &mut region_activated, &mut region_trace,
            &mut global_alpha, &mut global_beta,
        ).unwrap();
        let d_preds: Vec<Vec<f32>> = out.predictions.iter()
            .map(|p| vec![1.0; p.len()]).collect();

        let mut grads = RegionalGradientsTyped::<Cpu>::zeros(&typed).unwrap();
        let mut d_obs = TensorG::<Cpu>::zeros(raw_obs_dim).unwrap();
        typed.regional_backward_typed_with_d_obs(
            &cache, &d_preds, &mut grads, &mut d_obs,
        ).expect("regional_backward_typed_with_d_obs");

        let dh = d_obs.to_vec().unwrap();
        assert_eq!(dh.len(), raw_obs_dim);
        assert!(dh.iter().any(|&x| x.abs() > 1e-9),
            "d_observation all zero — connection-tail backward did not populate it");
    }

    #[test]
    fn output_local_head_gradcheck_and_flows_into_output_region() {
        // The OUTPUT-local move head reads the move from one region's own
        // sync_out. This verifies (a) the analytic gradient of the local
        // head weight matches central finite differences, and (b) the
        // gradient actually flows back INTO the OUTPUT region (its kv_proj
        // weight gets a non-zero grad) — i.e. the move-head signal reaches
        // the region that attends over the input, which is the whole point.
        const OUTPUT_R: usize = 2;
        let mut cfg = RegionalConfig::eight_region_small(16, 4, 2);
        cfg.router = None; // exercise the fixed-connection backward path
        // Make OUTPUT spatial (attends over raw tokens) and route the move
        // through its own sync_out.
        let raw_dim = 4usize;
        let n_tok = cfg.raw_obs_dim / raw_dim;
        cfg.regions[OUTPUT_R].spatial = Some((n_tok, raw_dim));
        cfg.output_local_region = Some(OUTPUT_R);

        let mut w = RegionalWeights::new(cfg);
        assert!(w.output_local_head.is_some(), "local head not built");

        let obs: Vec<f32> = (0..w.config.raw_obs_dim)
            .map(|i| ((i + 1) as f32 * 0.21).sin()).collect();
        let target = 2usize;

        // Analytic gradients.
        let mut grads = RegionalGradients::zeros(&w);
        let (_loss, _) = regional_train_step(&w, &mut grads, &obs, target);

        // (b) The OUTPUT region received gradient (move-head → sync_out →
        //     activated → region BPTT → kv_proj).
        let out_kv_grad_norm: f32 = grads.region_grads[OUTPUT_R].kv_proj_w
            .iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(out_kv_grad_norm > 1e-8,
            "OUTPUT region kv_proj got no gradient from the local head ({out_kv_grad_norm})");

        // The global output_proj should be UNTOUCHED (local head replaces it).
        let global_grad_norm: f32 = grads.output_proj_dw.iter()
            .map(|g| g * g).sum::<f32>().sqrt();
        assert!(global_grad_norm < 1e-12,
            "global output_proj got gradient but local head is active ({global_grad_norm})");

        // (a) Analytic-vs-finite-difference gradcheck of the local-head
        //     readout, against the EXACT scalar the analytic backward
        //     differentiates: the imagination loss restricted to the single
        //     committed tick whose CE the loss actually weights. With
        //     outer_ticks=2 and the default imagine_ratio, both selected
        //     ticks collapse to the committed tick, so for that tick the loss
        //     IS smooth (plain 0.5·CE + 0.5·CE on the same logits = CE) and
        //     FD matches the analytic head gradient up to f32 noise. We check
        //     the weights carrying the largest analytic gradient.
        let head_w = grads.output_local_head_dw.as_ref().unwrap().clone();
        let mut by_mag: Vec<usize> = (0..head_w.len()).collect();
        by_mag.sort_by(|&a, &b| head_w[b].abs().partial_cmp(&head_w[a].abs()).unwrap());
        let eps = 5e-4f32;
        let mut checked = 0;
        for &idx in by_mag.iter().take(3) {
            let an = head_w[idx];
            if an.abs() < 1e-4 { continue; }
            let orig = w.output_local_head.as_ref().unwrap().weight[idx];
            w.output_local_head.as_mut().unwrap().weight[idx] = orig + eps;
            let mut gp = RegionalGradients::zeros(&w);
            let (lp, _) = regional_train_step(&w, &mut gp, &obs, target);
            w.output_local_head.as_mut().unwrap().weight[idx] = orig - eps;
            let mut gm = RegionalGradients::zeros(&w);
            let (lm, _) = regional_train_step(&w, &mut gm, &obs, target);
            w.output_local_head.as_mut().unwrap().weight[idx] = orig;
            let fd = (lp - lm) / (2.0 * eps);
            let denom = fd.abs().max(an.abs()).max(1e-3);
            let rel = (fd - an).abs() / denom;
            // Same sign, and within tolerance that accommodates the loss's
            // discrete tick-selection surrogate (the analytic omits the
            // imagination-bonus / argmax-cert terms FD captures).
            assert!(fd * an > 0.0 && rel < 0.2,
                "local-head gradcheck idx={idx}: analytic={an:.6} fd={fd:.6} rel={rel:.4}");
            checked += 1;
        }
        assert!(checked > 0, "no local-head weight had a checkable gradient");
    }

    #[test]
    fn cpu_regional_backward_typed_flows_global_decay_grad() {
        // Brain-level decay-parameter gradient must flow during
        // regional_backward_typed. With weights initialised to a
        // mid-range (decay_param > 0), the unclamped interior gives
        // a non-zero grad through `-d_alpha · decay · alpha_pre`.
        let cfg = small_regional_cfg();
        let w = RegionalWeights::new(cfg);
        let mut typed = RegionalWeightsTyped::<Cpu>::from_untyped(&w).unwrap();

        // Bump global_decay to a known non-zero, non-clamped value so
        // the d_decay branch is active.
        let n_sync = typed.config.n_global_sync;
        let raw: Vec<f32> = (0..n_sync).map(|_| 0.5_f32).collect();
        typed.global_decay = TensorG::<Cpu>::from_slice(&raw).unwrap();

        let obs_dim = typed.config.raw_obs_dim;
        let observation: Vec<f32> = (0..obs_dim)
            .map(|i| ((i + 1) as f32 * 0.137).sin()).collect();
        let mut region_activated: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_activated.clone()).collect();
        let mut region_trace: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_trace.clone()).collect();
        let mut global_alpha = vec![0.0f32; n_sync];
        let mut global_beta = vec![1.0f32; n_sync];

        // Two-step forward so the second tick has alpha_pre != 0
        // (decay grad is multiplied by alpha_pre, so a fresh first tick
        // alone sees alpha_pre = 0 and contributes nothing).
        let (out, cache) = typed.regional_forward_typed_with_cache(
            &observation,
            &mut region_activated, &mut region_trace,
            &mut global_alpha, &mut global_beta,
        ).unwrap();
        let d_preds: Vec<Vec<f32>> = out.predictions.iter()
            .map(|p| vec![1.0; p.len()]).collect();

        let mut grads = RegionalGradientsTyped::<Cpu>::zeros(&typed).unwrap();
        typed.regional_backward_typed(&cache, &d_preds, &mut grads)
            .expect("regional_backward_typed");

        let dd = grads.global_decay.to_vec().unwrap();
        assert!(dd.iter().any(|&x| x.abs() > 1e-9),
            "global_decay grad all zero — backward did not propagate decay-parameter signal");
    }

    #[test]
    fn cpu_regional_gradients_typed_zero_and_accumulate() {
        // Build typed brain + grads, run a real backward to populate
        // grads, snapshot a probe value, then:
        //   1. zero() resets every tracked tensor to zeros.
        //   2. accumulate_from(other) adds another set of grads in.
        let cfg = small_regional_cfg();
        let w = RegionalWeights::new(cfg);
        let typed = RegionalWeightsTyped::<Cpu>::from_untyped(&w).unwrap();

        let obs_dim = typed.config.raw_obs_dim;
        let observation: Vec<f32> = (0..obs_dim)
            .map(|i| ((i + 1) as f32 * 0.137).sin()).collect();
        let mut region_activated: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_activated.clone()).collect();
        let mut region_trace: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_trace.clone()).collect();
        let n_sync = typed.config.n_global_sync;
        let mut global_alpha = vec![0.0f32; n_sync];
        let mut global_beta = vec![1.0f32; n_sync];
        let (out, cache) = typed.regional_forward_typed_with_cache(
            &observation,
            &mut region_activated, &mut region_trace,
            &mut global_alpha, &mut global_beta,
        ).unwrap();
        let d_preds: Vec<Vec<f32>> = out.predictions.iter()
            .map(|p| vec![1.0; p.len()]).collect();

        let mut g_a = RegionalGradientsTyped::<Cpu>::zeros(&typed).unwrap();
        typed.regional_backward_typed(&cache, &d_preds, &mut g_a).unwrap();

        // Probe: at least one connection synapse should have non-zero
        // d_w after a real backward.
        let probe_before_zero = g_a.connection_synapses[0].d_w.to_vec().unwrap();
        assert!(probe_before_zero.iter().any(|&x| x.abs() > 1e-6),
            "test setup: expected non-zero connection grad after backward");

        // Step 1: zero().
        g_a.zero().unwrap();
        let probe_after_zero = g_a.connection_synapses[0].d_w.to_vec().unwrap();
        assert!(probe_after_zero.iter().all(|&x| x == 0.0),
            "zero() did not reset connection_synapses[0].d_w");
        let region_probe = g_a.regions[0].nlm_s1_w.to_vec().unwrap();
        assert!(region_probe.iter().all(|&x| x == 0.0),
            "zero() did not reset regions[0].nlm_s1_w");
        let unet_probe = g_a.region_unets[0].first.d_w.to_vec().unwrap();
        assert!(unet_probe.iter().all(|&x| x == 0.0),
            "zero() did not reset region_unets[0].first.d_w");

        // Step 2: accumulate_from with a second backward pass.
        let mut g_b = RegionalGradientsTyped::<Cpu>::zeros(&typed).unwrap();
        typed.regional_backward_typed(&cache, &d_preds, &mut g_b).unwrap();
        g_a.accumulate_from(&g_b).unwrap();
        let probe_after_acc = g_a.connection_synapses[0].d_w.to_vec().unwrap();
        let probe_b = g_b.connection_synapses[0].d_w.to_vec().unwrap();
        // After accumulate, g_a should match g_b (since g_a was zero'd then accumulated once).
        for (a, b) in probe_after_acc.iter().zip(&probe_b) {
            assert!((a - b).abs() < 1e-9,
                "accumulate_from after zero produced wrong sum: {} vs {}", a, b);
        }
    }

    #[test]
    fn cpu_regional_adamw_typed_step_changes_weights() {
        // End-to-end: typed forward+cache → typed backward → typed AdamW
        // step. Asserts the optimiser actually mutates weight tensors
        // and bumps step_count.
        let cfg = small_regional_cfg();
        let w = RegionalWeights::new(cfg);
        let mut typed = RegionalWeightsTyped::<Cpu>::from_untyped(&w).unwrap();
        let mut opt = RegionalAdamWTyped::<Cpu>::new(&typed).unwrap()
            .with_lr(1e-2).with_clip(10.0);

        // Snapshot a few weight tensors.
        let output_proj_w_before = typed.output_proj.weight.to_vec().unwrap();
        let region0_nlm_s1_w_before = typed.regions[0].nlm_stage1.weights.to_vec().unwrap();
        let conn0_w_before = typed.connection_synapses[0].weight.to_vec().unwrap();

        let obs_dim = typed.config.raw_obs_dim;
        let observation: Vec<f32> = (0..obs_dim)
            .map(|i| ((i + 1) as f32 * 0.137).sin()).collect();
        let mut region_activated: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_activated.clone()).collect();
        let mut region_trace: Vec<Vec<f32>> = w.regions.iter()
            .map(|r| r.start_trace.clone()).collect();
        let n_sync = typed.config.n_global_sync;
        let mut global_alpha = vec![0.0f32; n_sync];
        let mut global_beta = vec![1.0f32; n_sync];

        let (out, cache) = typed.regional_forward_typed_with_cache(
            &observation,
            &mut region_activated, &mut region_trace,
            &mut global_alpha, &mut global_beta,
        ).unwrap();
        let d_preds: Vec<Vec<f32>> = out.predictions.iter()
            .map(|p| vec![1.0; p.len()]).collect();

        let mut grads = RegionalGradientsTyped::<Cpu>::zeros(&typed).unwrap();
        typed.regional_backward_typed(&cache, &d_preds, &mut grads).unwrap();
        opt.step(&mut typed, &mut grads).unwrap();
        assert_eq!(opt.step_count, 1, "step_count must bump");

        // Each previously-flowing weight must now differ from snapshot.
        let output_proj_w_after = typed.output_proj.weight.to_vec().unwrap();
        let any_changed = output_proj_w_before.iter().zip(&output_proj_w_after)
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(any_changed, "output_proj.weight unchanged after AdamW step");

        let nlm_after = typed.regions[0].nlm_stage1.weights.to_vec().unwrap();
        let any_nlm_changed = region0_nlm_s1_w_before.iter().zip(&nlm_after)
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(any_nlm_changed,
            "regions[0].nlm_stage1.weights unchanged — RegionInnerAdamWTyped did not step");

        let conn_after = typed.connection_synapses[0].weight.to_vec().unwrap();
        let any_conn_changed = conn0_w_before.iter().zip(&conn_after)
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(any_conn_changed,
            "connection_synapses[0].weight unchanged — connection AdamW did not step");
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_regional_weights_typed_constructible() {
        let cfg = small_regional_cfg();
        let w = RegionalWeights::new(cfg);
        let typed = match RegionalWeightsTyped::<Rocm>::from_untyped(&w) {
            Ok(t) => t,
            Err(BackendErrorG::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(e) => panic!("rocm RegionalWeightsTyped build failed: {:?}", e),
        };
        assert_eq!(typed.regions.len(), w.regions.len());
        // Embedding round-trip on Rocm.
        let emb = typed.embeddings.to_vec().unwrap();
        for (a, b) in emb.iter().zip(&w.embeddings) {
            assert!((a - b).abs() < 1e-6,
                "embedding diverged on Rocm: {} vs {}", a, b);
        }
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
    events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));

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

// ═══════════════════════════════════════════════════════════════
// BRAIN TRAIT IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════

/// Cache for regional brain forward pass (consumed by backward).
pub struct RegionalCache {
    pub tick_caches: Vec<OuterTickCache>,
    pub observation: Vec<f32>,
    /// Region activations from the last outer tick (for aux gradient computation).
    pub last_region_activations: Vec<Vec<f32>>,
}

/// The 8-region brain as a Brain.
pub struct RegionalBrain;

impl RegionalBrain {
    /// Returns `true` when the resident forward+backward path is
    /// likely faster than the host path for this brain's shape on
    /// AMD ROCm hardware.
    ///
    /// The heuristic encodes the Phase 5 crossover measurement on
    /// gfx1102 (RX 7600M XT): the resident path is profitable only
    /// when the *cortical* per-region `d_model ≥ 1024`. Below that,
    /// MIOpen + hipBLAS launch overhead exceeds the kernel work and
    /// the host path is faster — sometimes dramatically (5× regression
    /// at maze-scale d_model=32).
    ///
    /// Heuristic preserved from the now-removed device-resident path.
    /// `bench_brain_crossover` (typed Cpu vs Rocm) is the current
    /// authoritative comparison; this helper still answers "is the
    /// brain large enough that ROCm would beat host CPU" using the
    /// same d_model threshold the resident path used. See memory
    /// `feedback_residency_dispatch_overhead.md` for the measured
    /// speedup table at small/medium/large/billion configs
    /// (0.18×, 0.50×, 0.85×, 1.83× respectively).
    pub fn is_resident_likely_faster(weights: &RegionalWeights) -> bool {
        // Take the maximum cortical d_model across regions. The big
        // regions (input, attention, output, motor) are what drive
        // the GPU dispatch ratio; small subcortical regions
        // (cerebellum, basal_ganglia, etc.) are fast either way.
        const RESIDENT_DMODEL_THRESHOLD: usize = 1024;
        weights.regions.iter()
            .map(|r| r.config.d_model)
            .max()
            .map(|d| d >= RESIDENT_DMODEL_THRESHOLD)
            .unwrap_or(false)
    }

    /// Forward with caching + frozen cerebellum override.
    ///
    /// Same as `forward_cached` from the Brain trait, but after each outer tick,
    /// the cerebellum region's output is replaced with the frozen model's output.
    /// This lets the frozen world model contribute to global sync each tick.
    pub fn forward_cached_frozen(
        weights: &RegionalWeights,
        state: RegionalState,
        input: &modgrad_traits::TokenInput,
        frozen: &mut dyn crate::cerebellum::FrozenCerebellum,
    ) -> (modgrad_traits::BrainOutput, RegionalState, RegionalCache) {
        // If no projection layers configured, fall back to normal forward
        if weights.cereb_projection.is_none() {
            return <Self as modgrad_traits::Brain>::forward_cached(weights, state, input);
        }

        let proj = weights.cereb_projection.as_ref().unwrap();
        let cfg = &weights.config;
        let n_regions = cfg.regions.len();
        let n_sync = cfg.n_global_sync;
        let cereb_idx = cfg.region_names.iter()
            .position(|n| n.contains("cerebellum"))
            .unwrap_or(4);

        let obs_projected = weights.obs_proj.forward(&input.tokens);
        // FIX 1: thread the passed-in state (mirror forward_cached). The
        // spatial readout (FIX 2) is DEFERRED for the frozen path — it is
        // not exercised by the wall-probe (run without --frozen-cereb).
        let mut state = state;
        let mut region_explicit_states: Vec<_> =
            std::mem::take(&mut state.region_explicit_states);
        if region_explicit_states.len() != n_regions {
            region_explicit_states = weights.regions.iter()
                .map(|rw| Ctm::init_state(rw))
                .collect();
        }
        let mut tick_caches = Vec::with_capacity(cfg.outer_ticks);

        for outer_tick in 0..cfg.outer_ticks {
            let mut region_obs: Vec<Vec<f32>> = vec![Vec::new(); n_regions];
            let mut connection_inputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.connections.len());
            let mut router_cache: Option<RouterCache> = None;

            if let Some(ref router) = weights.router {
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
                        let (s, len) = cfg.obs_scale_slice(conn.observation_scale);
                        src.extend_from_slice(&input.tokens[s..s + len]);
                    }
                    connection_inputs.push(src.clone());
                    let projected = weights.connection_synapses[ci].forward(&src);
                    merge_into_region_obs(&mut region_obs[conn.to], projected);
                }
                for r in 0..n_regions {
                    if region_obs[r].is_empty() {
                        region_obs[r] = obs_projected.clone();
                    }
                }
            }

            // Run regions in parallel
            let states_taken: Vec<_> = region_explicit_states.drain(..).collect();
            let region_results: Vec<_> = states_taken.into_par_iter().enumerate().map(|(r, rstate)| {
                let d_input = weights.regions[r].config.d_input;
                let rinput = TokenInput {
                    tokens: region_obs[r].clone(),
                    n_tokens: 1,
                    token_dim: d_input,
                };
                let (_output, new_state, cache) = Ctm::forward_cached(
                    &weights.regions[r], rstate, &rinput,
                );
                (new_state.activated.clone(), new_state, cache)
            }).collect();

            state.region_outputs.clear();
            let mut region_caches: Vec<Option<CtmCache>> = Vec::with_capacity(n_regions);
            for (activated, new_state, cache) in region_results {
                state.region_outputs.push(activated);
                region_explicit_states.push(new_state);
                region_caches.push(Some(cache));
            }

            // Override cerebellum with frozen model output
            let cereb_input = &region_obs[cereb_idx];
            let input_len = proj.cortex_dim.min(cereb_input.len());
            let frozen_out = proj.forward(frozen, &cereb_input[..input_len]);
            let d_model = weights.regions[cereb_idx].config.d_model;
            let mut cereb_output = vec![0.0f32; d_model];
            let copy_len = d_model.min(frozen_out.len());
            cereb_output[..copy_len].copy_from_slice(&frozen_out[..copy_len]);
            state.region_outputs[cereb_idx] = cereb_output;

            // Global sync
            let mut all_activations = Vec::new();
            for r in 0..n_regions {
                all_activations.extend_from_slice(&state.region_outputs[r]);
            }

            for i in 0..n_sync {
                let l = weights.global_sync_left[i];
                let ri = weights.global_sync_right[i];
                if l < all_activations.len() && ri < all_activations.len() {
                    let pw = all_activations[l] * all_activations[ri];
                    let decay = (-weights.global_decay[i].clamp(0.0, 15.0)).exp();
                    state.global_alpha[i] = decay * state.global_alpha[i] + pw;
                    state.global_beta[i] = decay * state.global_beta[i] + 1.0;
                }
            }

            let global_sync: Vec<f32> = (0..n_sync)
                .map(|i| state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8))
                .collect();

            let (exit_gate_cache, exit_lambda) = if let Some(ref gate) = weights.outer_exit_gate {
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

        // FIX 1: thread region CTM states back into the returned state.
        state.region_explicit_states = region_explicit_states;

        let predictions: Vec<Vec<f32>> = tick_caches.iter()
            .map(|tc| weights.output_proj.forward(&tc.global_sync))
            .collect();
        let certainties: Vec<[f32; 2]> = predictions.iter().map(|p| {
            let c = compute_certainty(p);
            [1.0 - c, c]
        }).collect();

        let brain_output = modgrad_traits::BrainOutput {
            predictions,
            certainties,
            sync: tick_caches.last()
                .map(|tc| tc.global_sync.clone())
                .unwrap_or_default(),
        };

        let cache = RegionalCache {
            tick_caches,
            observation: input.tokens.clone(),
            last_region_activations: state.region_outputs.clone(),
        };

        (brain_output, state, cache)
    }

}

impl modgrad_traits::Brain for RegionalBrain {
    type Input = modgrad_traits::TokenInput;
    type Weights = RegionalWeights;
    type State = RegionalState;
    type Cache = RegionalCache;
    type Gradients = RegionalGradients;

    fn init_state(weights: &RegionalWeights) -> RegionalState {
        RegionalState::new(weights)
    }

    fn forward(
        weights: &RegionalWeights,
        mut state: RegionalState,
        input: &modgrad_traits::TokenInput,
    ) -> (modgrad_traits::BrainOutput, RegionalState) {
        let output = regional_forward(weights, &mut state, &input.tokens);
        let brain_output = modgrad_traits::BrainOutput {
            predictions: output.predictions,
            certainties: output.exit_lambdas.iter()
                .map(|&l| [1.0 - l, l])
                .collect::<Vec<_>>(),
            sync: output.global_sync,
        };
        // Pad certainties if exit_lambdas was empty (no gate)
        let brain_output = if brain_output.certainties.is_empty() {
            modgrad_traits::BrainOutput {
                certainties: brain_output.predictions.iter()
                    .map(|p| {
                        let c = crate::forward::compute_certainty_pub(p);
                        c
                    }).collect(),
                ..brain_output
            }
        } else {
            brain_output
        };
        (brain_output, state)
    }

    fn forward_cached(
        weights: &RegionalWeights,
        state: RegionalState,
        input: &modgrad_traits::TokenInput,
    ) -> (modgrad_traits::BrainOutput, RegionalState, RegionalCache) {
        // FIX 1 (was: KNOWN REGRESSION, tracked by PR-AB): this impl now
        // THREADS the passed-in `state` (mirroring `forward`) instead of
        // rebuilding a fresh `RegionalState::new(weights)`. CTM recurrence
        // (per-region activated/trace, global sync accumulators) carries
        // across calls so continuous-thinking-across-calls is expressed.
        // The inner region CTM states (`state.region_states`) are threaded
        // through `Ctm::forward_cached` per outer tick. NOTE: the
        // BACKWARD/training rewiring of the spatial readout below is
        // DEFERRED — the cache populated here still records the legacy
        // single-token region inputs/global-sync output_proj path for
        // backprop. The spatial forward additions (pos-encoded raw tokens
        // + output_local_head readout) affect only the returned
        // BrainOutput + the stashed local sync, which is all the
        // wall-probe consumes.
        let cfg = &weights.config;
        let n_regions = cfg.regions.len();
        let n_sync = cfg.n_global_sync;

        let obs_projected = weights.obs_proj.forward(&input.tokens);
        let mut state = state;
        // Thread the per-region CTM states from the incoming state.
        let mut region_explicit_states: Vec<_> =
            std::mem::take(&mut state.region_explicit_states);
        if region_explicit_states.len() != n_regions {
            // Defensive: a mismatched incoming state (shouldn't happen for
            // a state built from these weights) — rebuild region states.
            region_explicit_states = weights.regions.iter()
                .map(|rw| Ctm::init_state(rw))
                .collect();
        }
        let mut tick_caches = Vec::with_capacity(cfg.outer_ticks);

        // FIX 2: positional encoding can be disabled (ablation) via
        // env var MODGRAD_DISABLE_POS=1.
        let pos_enabled = std::env::var("MODGRAD_DISABLE_POS")
            .map(|v| v != "1" && v != "true")
            .unwrap_or(true);
        // Pre-compute the pos-encoded raw token buffer once per call, used
        // by any spatial region. Grid width = sqrt(n_tokens) (square grid).
        let local_region = cfg.output_local_region;
        // The OUTPUT region's local sync from the final outer tick.
        let mut last_local_sync: Option<Vec<f32>> = None;

        for outer_tick in 0..cfg.outer_ticks {
            let mut region_obs: Vec<Vec<f32>> = vec![Vec::new(); n_regions];
            let mut connection_inputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.connections.len());
            let mut router_cache: Option<RouterCache> = None;

            if let Some(ref router) = weights.router {
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
                        let (s, len) = cfg.obs_scale_slice(conn.observation_scale);
                        src.extend_from_slice(&input.tokens[s..s + len]);
                    }
                    connection_inputs.push(src.clone());
                    let projected = weights.connection_synapses[ci].forward(&src);
                    merge_into_region_obs(&mut region_obs[conn.to], projected);
                }
                for r in 0..n_regions {
                    if region_obs[r].is_empty() {
                        region_obs[r] = obs_projected.clone();
                    }
                }
            }

            // Run regions in parallel — take ownership for disjoint mut.
            // FIX 2: spatial regions are fed the RAW pos-encoded tokens as
            // `n_tok × raw_dim` (so attention sees per-cell position),
            // bypassing the flat connection-merged single-token obs. The
            // OUTPUT-local region additionally reports its own sync_out
            // (the readout the move head consumes).
            let states_taken: Vec<_> = region_explicit_states.drain(..).collect();
            let region_results: Vec<_> = states_taken.into_par_iter().enumerate().map(|(r, rstate)| {
                let rinput = match weights.regions[r].config.spatial {
                    Some((n_tok, raw_dim)) => {
                        let mut toks = input.tokens.clone();
                        if pos_enabled {
                            let grid_w = (n_tok as f64).sqrt().round() as usize;
                            crate::pos::add_sinusoidal_pos_2d(
                                &mut toks, n_tok, raw_dim, grid_w.max(1));
                        }
                        TokenInput { tokens: toks, n_tokens: n_tok, token_dim: raw_dim }
                    }
                    None => {
                        let d_input = weights.regions[r].config.d_input;
                        TokenInput {
                            tokens: region_obs[r].clone(),
                            n_tokens: 1,
                            token_dim: d_input,
                        }
                    }
                };
                let (output, new_state, cache) = Ctm::forward_cached(
                    &weights.regions[r], rstate, &rinput,
                );
                // The OUTPUT-local region's sync readout (BrainOutput.sync
                // == final-tick sync_out) is the exact move-head input.
                let local_sync = if local_region == Some(r) {
                    Some(output.sync.clone())
                } else {
                    None
                };
                (new_state.activated.clone(), new_state, cache, local_sync)
            }).collect();

            state.region_outputs.clear();
            let mut region_caches: Vec<Option<CtmCache>> = Vec::with_capacity(n_regions);
            let mut tick_local_sync: Option<Vec<f32>> = None;
            for (activated, new_state, cache, local_sync) in region_results {
                state.region_outputs.push(activated);
                region_explicit_states.push(new_state);
                region_caches.push(Some(cache));
                if local_sync.is_some() { tick_local_sync = local_sync; }
            }
            // Carry the OUTPUT region's readout from THIS outer tick;
            // overwritten each tick so the final value reflects the last.
            if tick_local_sync.is_some() { last_local_sync = tick_local_sync; }

            let mut all_activations = Vec::new();
            for r in 0..n_regions {
                all_activations.extend_from_slice(&state.region_outputs[r]);
            }

            for i in 0..n_sync {
                let l = weights.global_sync_left[i];
                let ri = weights.global_sync_right[i];
                if l < all_activations.len() && ri < all_activations.len() {
                    let pw = all_activations[l] * all_activations[ri];
                    let decay = (-weights.global_decay[i].clamp(0.0, 15.0)).exp();
                    state.global_alpha[i] = decay * state.global_alpha[i] + pw;
                    state.global_beta[i] = decay * state.global_beta[i] + 1.0;
                }
            }

            let global_sync: Vec<f32> = (0..n_sync)
                .map(|i| state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8))
                .collect();

            let (exit_gate_cache, exit_lambda) = if let Some(ref gate) = weights.outer_exit_gate {
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

        // Thread the per-region CTM states back into the returned state so
        // the next call carries recurrence (FIX 1).
        state.region_explicit_states = region_explicit_states;
        // Stash the OUTPUT region's local sync for the wall-probe (FIX 2).
        state.last_output_local_sync = last_local_sync.clone();

        // FIX 2: when the OUTPUT-local head is active, predictions come
        // from `output_local_head.forward(local_sync)` (the spatial
        // readout) instead of `output_proj(global_sync)`. We only have the
        // FINAL-tick local sync here (the per-tick spatial syncs are not
        // cached), so the local-head path emits a single prediction for
        // the committed (final) tick; the legacy path emits one per tick.
        let local_head_active = weights.output_local_head.is_some()
            && local_region.is_some()
            && last_local_sync.is_some();
        let predictions: Vec<Vec<f32>> = if local_head_active {
            let head = weights.output_local_head.as_ref().unwrap();
            let sync = last_local_sync.as_ref().unwrap();
            vec![head.forward(sync)]
        } else {
            tick_caches.iter()
                .map(|tc| weights.output_proj.forward(&tc.global_sync))
                .collect()
        };
        let certainties: Vec<[f32; 2]> = predictions.iter().map(|p| {
            let c = compute_certainty(p);
            [1.0 - c, c]
        }).collect();

        // BrainOutput.sync stays the GLOBAL sync (the probe's baseline
        // representation A). The spatial readout (representation C) is read
        // from `state.last_output_local_sync`.
        let brain_output = modgrad_traits::BrainOutput {
            predictions,
            certainties,
            sync: tick_caches.last()
                .map(|tc| tc.global_sync.clone())
                .unwrap_or_default(),
        };

        let cache = RegionalCache {
            tick_caches,
            observation: input.tokens.clone(),
            last_region_activations: state.region_outputs.clone(),
        };

        (brain_output, state, cache)
    }

    fn backward(
        weights: &RegionalWeights,
        cache: RegionalCache,
        d_predictions: &[Vec<f32>],
    ) -> RegionalGradients {
        let cfg = &weights.config;
        let n_regions = cfg.regions.len();
        let mut grads = RegionalGradients::zeros(weights);

        let add = |d: &mut [f32], s: &[f32]| {
            for (d, s) in d.iter_mut().zip(s) { *d += s; }
        };

        // Exit gate gradients
        if let Some(ref gate) = weights.outer_exit_gate {
            let predictions: Vec<Vec<f32>> = cache.tick_caches.iter()
                .map(|tc| weights.output_proj.forward(&tc.global_sync))
                .collect();
            let lambdas: Vec<f32> = cache.tick_caches.iter().map(|tc| tc.exit_lambda).collect();
            let beta = cfg.exit_strategy.beta();
            let (_exit_loss, _exit_d_preds, d_lambdas) = crate::train::adaptive_exit_loss(
                &predictions, &lambdas, 0, beta);
            let mut d_gate_in_scratch = vec![0.0f32; gate.in_dim];
            for (t, d_lambda) in d_lambdas.iter().enumerate() {
                if let Some(ref ec) = cache.tick_caches[t].exit_gate_cache {
                    let gw = grads.outer_exit_gate_dw.as_mut().unwrap();
                    let gb = grads.outer_exit_gate_db.as_mut().unwrap();
                    crate::train::linear_backward(
                        gate, &[*d_lambda], ec, gw, gb,
                        &mut d_gate_in_scratch);
                }
            }
        }

        // Save for aux gradients before cache is consumed
        let aux_region_activations = cache.last_region_activations.clone();
        let aux_observation = cache.observation.clone();

        for (t, tc) in cache.tick_caches.into_iter().enumerate().rev() {
            let d_logits = &d_predictions[t];

            // output_proj backward
            let out_dim = weights.output_proj.out_dim;
            let in_dim = weights.output_proj.in_dim;
            let mut d_global_sync = vec![0.0f32; in_dim];
            for i in 0..out_dim { grads.output_proj_db[i] += d_logits[i]; }
            modgrad_device::backend::ops::outer_product_acc(
                d_logits, &tc.global_sync, &mut grads.output_proj_dw,
                out_dim, in_dim,
            ).expect("output_proj backward: outer_product_acc dispatch");
            modgrad_device::backend::ops::matvec_t(
                d_logits, &weights.output_proj.weight, &mut d_global_sync,
                out_dim, in_dim,
            ).expect("output_proj backward: matvec_t dispatch");

            // global sync backward
            let total_act_dim = tc.all_activations.len();
            let d_all_activations = global_sync_backward(
                &d_global_sync, &tc.all_activations, &tc.global_beta,
                &weights.global_sync_left, &weights.global_sync_right, total_act_dim,
            );

            // Split d_all_activations per region
            let mut offset = 0;
            let mut d_region_activated: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
            for r in 0..n_regions {
                let dim = weights.regions[r].config.d_model;
                d_region_activated.push(d_all_activations[offset..offset + dim].to_vec());
                offset += dim;
            }

            let mut d_region_obs: Vec<Vec<f32>> = vec![Vec::new(); n_regions];
            let mut region_caches = tc.region_caches;
            for r in 0..n_regions {
                if let Some(rcache) = region_caches[r].take() {
                    let result = backward_from_activated(&weights.regions[r], &rcache, &d_region_activated[r]);
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
                    // U-Net grads — same outer-merge gap as fixed elsewhere.
                    dst.unet.add_from(&result.grads.unet);
                    d_region_obs[r] = result.d_observation;
                }
            }

            // Connection synapse backward
            for (ci, conn) in cfg.connections.iter().enumerate() {
                let r = conn.to;
                if d_region_obs[r].is_empty() { continue; }
                let d_proj_out = &d_region_obs[r];
                let syn = &weights.connection_synapses[ci];
                let syn_input = &tc.connection_inputs.get(ci);
                if let Some(input) = syn_input {
                    for i in 0..syn.out_dim.min(d_proj_out.len()) {
                        grads.connection_db[ci][i] += d_proj_out[i];
                        for j in 0..syn.in_dim.min(input.len()) {
                            grads.connection_dw[ci][i * syn.in_dim + j] += d_proj_out[i] * input[j];
                        }
                    }
                }
            }
        }

        // Aux head gradients — computed from the last tick's region activations.
        if cfg.aux_losses.cerebellar_prediction
            || cfg.aux_losses.bg_temporal_difference
        {
            accumulate_aux_gradients(
                weights, &mut grads,
                &aux_region_activations,
                &aux_observation,
                0.0,
            );
        }

        grads
    }

    fn zero_gradients(weights: &RegionalWeights) -> RegionalGradients {
        RegionalGradients::zeros(weights)
    }

    fn apply_gradients(
        weights: &mut RegionalWeights,
        grads: &mut RegionalGradients,
        lr: f32,
        clip_norm: f32,
    ) {
        grads.apply(weights, lr, clip_norm);
    }
}

impl RegionalBrain {
    /// Like `<Self as Brain>::backward` but additionally returns the
    /// gradient of the task loss with respect to the input
    /// `observation` (the concatenated token buffer passed into
    /// `forward_cached`). This unblocks end-to-end training of any
    /// encoder that feeds RegionalBrain — e.g., GHL-adaptive retina,
    /// V4-CTM trained jointly with the brain, or a future learnable
    /// top-down projection seeded from the observation-side gradient.
    ///
    /// Limitations:
    /// - Router-based configs (`config.router.is_some()`) currently
    ///   propagate d_observation only through the optional `obs_proj`
    ///   fallback, which `eight_region_small` doesn't use. For the
    ///   non-router connection-synapse path (the default in
    ///   `eight_region_small`), d_observation is computed exactly
    ///   via the transpose of each connection synapse whose
    ///   `receives_observation == true`.
    ///
    /// Returns `(grads, d_observation)`. `d_observation.len() ==
    /// cache.observation.len()`.
    pub fn backward_with_input_grad(
        weights: &RegionalWeights,
        cache: RegionalCache,
        d_predictions: &[Vec<f32>],
    ) -> (RegionalGradients, Vec<f32>) {
        let cfg = &weights.config;
        let n_regions = cfg.regions.len();
        let mut grads = RegionalGradients::zeros(weights);
        let obs_len = cache.observation.len();
        let mut d_observation = vec![0.0f32; obs_len];

        let add = |d: &mut [f32], s: &[f32]| {
            for (d, s) in d.iter_mut().zip(s) { *d += s; }
        };

        // Exit gate gradients (same as trait backward).
        if let Some(ref gate) = weights.outer_exit_gate {
            let predictions: Vec<Vec<f32>> = cache.tick_caches.iter()
                .map(|tc| weights.output_proj.forward(&tc.global_sync))
                .collect();
            let lambdas: Vec<f32> = cache.tick_caches.iter()
                .map(|tc| tc.exit_lambda).collect();
            let beta = cfg.exit_strategy.beta();
            let (_exit_loss, _exit_d_preds, d_lambdas) =
                crate::train::adaptive_exit_loss(&predictions, &lambdas, 0, beta);
            let mut d_gate_in_scratch = vec![0.0f32; gate.in_dim];
            for (t, d_lambda) in d_lambdas.iter().enumerate() {
                if let Some(ref ec) = cache.tick_caches[t].exit_gate_cache {
                    let gw = grads.outer_exit_gate_dw.as_mut().unwrap();
                    let gb = grads.outer_exit_gate_db.as_mut().unwrap();
                    crate::train::linear_backward(
                        gate, &[*d_lambda], ec, gw, gb,
                        &mut d_gate_in_scratch);
                }
            }
        }

        let aux_region_activations = cache.last_region_activations.clone();
        let aux_observation = cache.observation.clone();

        for (t, tc) in cache.tick_caches.into_iter().enumerate().rev() {
            let d_logits = &d_predictions[t];

            // output_proj backward
            let out_dim = weights.output_proj.out_dim;
            let in_dim = weights.output_proj.in_dim;
            let mut d_global_sync = vec![0.0f32; in_dim];
            for i in 0..out_dim { grads.output_proj_db[i] += d_logits[i]; }
            modgrad_device::backend::ops::outer_product_acc(
                d_logits, &tc.global_sync, &mut grads.output_proj_dw,
                out_dim, in_dim,
            ).expect("output_proj backward: outer_product_acc dispatch");
            modgrad_device::backend::ops::matvec_t(
                d_logits, &weights.output_proj.weight, &mut d_global_sync,
                out_dim, in_dim,
            ).expect("output_proj backward: matvec_t dispatch");

            // global sync backward
            let total_act_dim = tc.all_activations.len();
            let d_all_activations = global_sync_backward(
                &d_global_sync, &tc.all_activations, &tc.global_beta,
                &weights.global_sync_left, &weights.global_sync_right, total_act_dim,
            );

            // Split per region
            let mut offset = 0;
            let mut d_region_activated: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
            for r in 0..n_regions {
                let dim = weights.regions[r].config.d_model;
                d_region_activated.push(d_all_activations[offset..offset + dim].to_vec());
                offset += dim;
            }

            // Per-region BPTT → d_region_obs[r] (gradient w.r.t. what went into region r)
            let mut d_region_obs: Vec<Vec<f32>> = vec![Vec::new(); n_regions];
            let mut region_caches = tc.region_caches;
            for r in 0..n_regions {
                if let Some(rcache) = region_caches[r].take() {
                    let result = backward_from_activated(
                        &weights.regions[r], &rcache, &d_region_activated[r]);
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
                    // U-Net grads — same outer-merge gap as fixed elsewhere.
                    dst.unet.add_from(&result.grads.unet);
                    d_region_obs[r] = result.d_observation;
                }
            }

            // Connection synapse backward — weight grads AND d_observation.
            // Each connection synapse projects `concat(from_region_outputs[, tokens])`
            // to region_obs[to]. To get d_observation, back through the
            // synapse's weight and slice out the observation portion.
            for (ci, conn) in cfg.connections.iter().enumerate() {
                let r = conn.to;
                if d_region_obs[r].is_empty() { continue; }
                let d_proj_out = &d_region_obs[r];
                let syn = &weights.connection_synapses[ci];
                let Some(input) = tc.connection_inputs.get(ci) else { continue };

                // Weight grads (same as trait backward).
                let o_max = syn.out_dim.min(d_proj_out.len());
                let j_max = syn.in_dim.min(input.len());
                for i in 0..o_max {
                    grads.connection_db[ci][i] += d_proj_out[i];
                    for j in 0..j_max {
                        grads.connection_dw[ci][i * syn.in_dim + j] += d_proj_out[i] * input[j];
                    }
                }

                // d_input[j] = Σᵢ d_proj_out[i] · W[i, j]. The synapse's
                // observation portion is the slice of length
                // `cfg.obs_scale_slice(conn.observation_scale).1`,
                // appended after the from-region outputs. Backprop
                // sends that gradient back to the matching slice of
                // the flat multi-scale `d_observation` buffer.
                if conn.receives_observation {
                    let mut src_offset = 0usize;
                    for &from_idx in &conn.from {
                        src_offset += weights.regions[from_idx].config.d_model;
                    }
                    let (obs_start_in_buf, scale_len) =
                        cfg.obs_scale_slice(conn.observation_scale);
                    let obs_end = (src_offset + scale_len).min(syn.in_dim);
                    for j in src_offset..obs_end {
                        let mut acc = 0.0f32;
                        for i in 0..o_max {
                            acc += d_proj_out[i] * syn.weight[i * syn.in_dim + j];
                        }
                        let dobs_idx = obs_start_in_buf + (j - src_offset);
                        if dobs_idx < d_observation.len() {
                            d_observation[dobs_idx] += acc;
                        }
                    }
                }
            }
        }

        // Aux head gradients — same as trait backward.
        if cfg.aux_losses.cerebellar_prediction
            || cfg.aux_losses.bg_temporal_difference
        {
            accumulate_aux_gradients(
                weights, &mut grads,
                &aux_region_activations,
                &aux_observation,
                0.0,
            );
        }

        (grads, d_observation)
    }
}

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

/// Continuity payload persisted alongside the NC checkpoint. The
/// fields here are the minimum needed to replay the CTM trajectory
/// back into its pre-save state: `history` (tokens) + `rng_state`
/// (sampling seed) + `max_history` (truncation horizon).
///
/// Bumping this struct means bumping the sidecar file format —
/// keep additions backwards-compatible via `#[serde(default)]`.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct NcContinuity {
    history: Vec<usize>,
    rng_state: u64,
    #[serde(default = "default_max_history")]
    max_history: usize,
}
fn default_max_history() -> usize { 4096 }

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

    /// Save weights + a continuity sidecar. Weights go to `path` via
    /// `RegionalWeights::save` (persist module, `ISIS` magic). The
    /// continuity blob (`history + rng_state + max_history` — the
    /// trajectory that reconstructs state on replay) goes to a sibling
    /// `<path>.ncstate.bin`.
    ///
    /// Known limitation — `isis nc` loading *training-produced*
    /// bundle checkpoints (`MGCK` magic) is still broken: bincode
    /// cannot deserialize `RegionalWeights` (hits
    /// `DeserializeAnyNotSupported`). This sidecar scheme works for
    /// the legacy pure-weights ISIS-magic path and is a no-op-safe
    /// baseline for when the bundle-load bug is fixed.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        self.weights.save(path)?;
        let bytes = bincode::serialize(&NcContinuity {
            history: self.history.clone(),
            rng_state: self.rng_state,
            max_history: self.max_history,
        }).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(Self::sidecar_path(path), bytes)?;
        Ok(())
    }

    /// Path for the continuity sidecar alongside the weights file.
    pub fn sidecar_path(path: &str) -> String {
        if let Some(stripped) = path.strip_suffix(".bin") {
            format!("{stripped}.ncstate.bin")
        } else if let Some(stripped) = path.strip_suffix(".json") {
            format!("{stripped}.ncstate.bin")
        } else {
            format!("{path}.ncstate")
        }
    }

    /// Load from checkpoint. If a continuity sidecar exists, the NC
    /// wakes up with its previous token history and the CTM's state
    /// is reconstructed by replaying that history through `step()`.
    /// No sidecar = fresh state (legacy behavior).
    pub fn load(path: &str) -> std::io::Result<Self> {
        let weights = RegionalWeights::load(path)?;
        let mut nc = Self::new(weights);
        Self::replay_sidecar(&mut nc, path);
        Ok(nc)
    }

    /// Replay history from the sidecar to reconstruct state. No-op if
    /// sidecar is missing or corrupt.
    fn replay_sidecar(nc: &mut Self, path: &str) {
        let sidecar = Self::sidecar_path(path);
        let Ok(bytes) = std::fs::read(&sidecar) else { return };
        let Ok(cont) = bincode::deserialize::<NcContinuity>(&bytes) else { return };
        nc.max_history = cont.max_history;
        nc.rng_state = cont.rng_state;
        eprintln!("  NC continuity: replaying {} tokens of history to \
            reconstruct state...", cont.history.len());
        for &tok in &cont.history {
            let _ = nc.step(tok);
        }
        nc.history = cont.history;
        if nc.history.len() > nc.max_history {
            let drop = nc.history.len() - nc.max_history;
            nc.history.drain(..drop);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `eight_region_v2` mounts the canonical 8-region topology with
    /// cerebellum-dominant intent. This test pins the regional shape
    /// and the connection-graph table from `docs/BRAIN_ARCHITECTURE.md`
    /// §2 so the architectural commitment can't silently drift.
    #[test]
    fn eight_region_v2_shape() {
        let cfg = RegionalConfig::eight_region_v2(128, 64, 16);

        // Eight regions, in canonical order.
        assert_eq!(cfg.regions.len(), 8);
        assert_eq!(cfg.region_names.len(), 8);
        let expected_names = [
            "input", "attention", "output", "motor",
            "cerebellum", "basal_ganglia", "insula", "hippocampus",
        ];
        for (i, want) in expected_names.iter().enumerate() {
            assert_eq!(cfg.region_names[i], *want,
                "region {i} name mismatch: got {} want {want}",
                cfg.region_names[i]);
        }

        // Cerebellum region exists as a small-but-nonzero placeholder
        // (sibling-service architecture: real LLM lives in
        // FrozenCerebellum, not here).
        let cereb_idx = cfg.region_names.iter()
            .position(|n| n == "cerebellum")
            .expect("eight_region_v2 must have a cerebellum region");
        let cereb = &cfg.regions[cereb_idx];
        assert!(cereb.d_model > 0,
            "cerebellum region must be non-empty (it's the LLM mount point)");
        assert!(cereb.d_model <= 64,
            "cerebellum CTM placeholder should stay small (≤64 d_model); \
             actual LLM is mounted via FrozenCerebellum. got {}",
            cereb.d_model);

        // Cortex regions should be small (per arch doc §4).
        for name in ["input", "attention", "output", "motor"] {
            let idx = cfg.region_names.iter().position(|n| n == name).unwrap();
            assert!(cfg.regions[idx].d_model <= 128,
                "cortex region {name} should be small in v2; got d_model={}",
                cfg.regions[idx].d_model);
        }

        // Connection graph: ARCHITECTURE.md §2 — 8 base edges + 1
        // cerebellum→attention sibling-service edge (the first concrete
        // cerebellum-output edge per arch doc §2 closing paragraph).
        assert_eq!(cfg.connections.len(), 9,
            "ARCHITECTURE.md §2 specifies 8 base edges + 1 cerebellum→attention edge");

        // Resolve indices by name (don't rely on hard-coded constants).
        let idx_of = |name: &str| cfg.region_names.iter()
            .position(|n| n == name).unwrap();
        let i_input         = idx_of("input");
        let i_attention     = idx_of("attention");
        let i_output        = idx_of("output");
        let i_motor         = idx_of("motor");
        let i_cerebellum    = idx_of("cerebellum");
        let i_basal_ganglia = idx_of("basal_ganglia");
        let i_insula        = idx_of("insula");
        let i_hippocampus   = idx_of("hippocampus");

        // Helper: does the edge `from = froms, to = to` exist with the
        // expected observation flag?
        let has_edge = |froms: &[usize], to: usize, obs: bool| -> bool {
            cfg.connections.iter().any(|c|
                c.to == to
                && c.from.len() == froms.len()
                && froms.iter().all(|f| c.from.contains(f))
                && c.receives_observation == obs
            )
        };

        // Each row of ARCHITECTURE.md §2:
        assert!(has_edge(&[i_motor], i_input, true),
            "edge 0: motor → input (carries observation)");
        assert!(has_edge(&[i_input], i_attention, false),
            "edge 1: input → attention");
        assert!(has_edge(&[i_attention], i_output, false),
            "edge 2: attention → output");
        assert!(has_edge(&[i_output], i_motor, false),
            "edge 3: output → motor");
        assert!(has_edge(&[i_motor], i_cerebellum, true),
            "edge 4: motor → cerebellum (carries observation — prompt seam)");
        assert!(has_edge(&[i_output], i_basal_ganglia, false),
            "edge 5: output → basal_ganglia");
        assert!(has_edge(&[i_hippocampus], i_insula, false),
            "edge 6: hippocampus → insula");
        assert!(has_edge(
            &[i_input, i_attention, i_output, i_motor],
            i_hippocampus,
            false,
        ), "edge 7: {{input, attention, output, motor}} → hippocampus");
        assert!(has_edge(&[i_cerebellum], i_attention, false),
            "edge 8: cerebellum → attention (sibling-service signal)");

        // Public arg semantics.
        assert_eq!(cfg.raw_obs_dim, 128);
        assert_eq!(cfg.out_dims, 64);
        assert_eq!(cfg.outer_ticks, 16);
    }

    /// `backward_with_input_grad` returns the same region/weight grads
    /// as the trait `backward`, plus a non-zero `d_observation` of
    /// length matching the input token buffer. Needed for GHL retina,
    /// V4-CTM joint training, and any learnable top-down projection.
    #[test]
    fn backward_with_input_grad_returns_d_observation() {
        let token_dim = 16usize;
        let n_tokens = 4usize;
        let out_dims = 64usize;
        let ticks = 2usize;
        let cfg = RegionalConfig::eight_region_small(token_dim, out_dims, ticks);
        let w = RegionalWeights::new(cfg);

        let tokens = modgrad_traits::TokenInput {
            tokens: (0..n_tokens * token_dim).map(|i| (i as f32 * 0.01).sin()).collect(),
            n_tokens,
            token_dim,
        };
        let state = <RegionalBrain as modgrad_traits::Brain>::init_state(&w);
        let (output, _state, cache) =
            <RegionalBrain as modgrad_traits::Brain>::forward_cached(&w, state, &tokens);

        let d_preds: Vec<Vec<f32>> = output.predictions.iter()
            .map(|p| p.iter().map(|&v| v * 1e-3).collect()).collect();

        let (_grads, d_obs) = RegionalBrain::backward_with_input_grad(&w, cache, &d_preds);

        assert_eq!(d_obs.len(), n_tokens * token_dim,
            "d_observation length must match input.tokens length");
        // At least one element should be non-zero — the observation
        // entered a connection synapse on some tick, so gradient flows.
        let max_abs: f32 = d_obs.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_abs > 0.0,
            "d_observation should have at least one non-zero component");
        for (i, &v) in d_obs.iter().enumerate() {
            assert!(v.is_finite(), "d_obs[{i}] = {v} must be finite");
        }
    }

    #[test]
    fn episodic_memory_accumulates() {
        let cfg = RegionalConfig::eight_region(16, 256, 2);
        let w = RegionalWeights::new(cfg);
        let mut nc = NeuralComputer::new(w);

        // Find hippocampus
        let idx = nc.weights.config.region_names.iter()
            .position(|n| n.contains("hippocampus"))
            .expect("eight_region should have hippocampus");

        let epi = nc.state.region_states[idx].episodic.as_ref()
            .expect("hippocampus should have episodic memory");
        assert_eq!(epi.n_tokens(), 0);

        // Run 20 tokens
        for i in 0..20 {
            let _logits = nc.step(i % 256);
        }

        let epi = nc.state.region_states[idx].episodic.as_ref().unwrap();
        assert!(epi.n_tokens() > 0,
            "episodic memory should have entries after inference, got {}", epi.n_tokens());

        // Memory should be bounded
        let max = 4 + 16 + 64; // short + mid + long capacity
        assert!(epi.n_tokens() <= max,
            "episodic memory should be bounded: {} > {}", epi.n_tokens(), max);
    }

    /// Hippocampus episodic memory must be **content-dependent**:
    /// different input sequences from a fresh weight init must yield
    /// different episodic KV buffers. The existing
    /// `episodic_memory_accumulates` only asserts entries exist; this
    /// asserts that what they hold actually depends on what was seen
    /// (rules out degenerate constant / zero-fill memory).
    #[test]
    fn episodic_memory_is_content_dependent() {
        let cfg = RegionalConfig::eight_region(16, 256, 2);
        let w = RegionalWeights::new(cfg);

        let hippo_idx = w.config.region_names.iter()
            .position(|n| n.contains("hippocampus"))
            .expect("eight_region should have hippocampus");

        let mut nc_a = NeuralComputer::new(w.clone());
        for i in 1..=19usize { let _ = nc_a.step(i); }
        let kv_a: Vec<f32> = nc_a.state.region_states[hippo_idx]
            .episodic.as_ref().unwrap().as_kv();

        let mut nc_b = NeuralComputer::new(w);
        for i in 200..=218usize { let _ = nc_b.step(i); }
        let kv_b: Vec<f32> = nc_b.state.region_states[hippo_idx]
            .episodic.as_ref().unwrap().as_kv();

        assert!(!kv_a.is_empty() && !kv_b.is_empty(),
            "both paths should produce non-empty episodic KV");
        let m = kv_a.len().min(kv_b.len());
        let l2: f32 = kv_a[..m].iter().zip(kv_b[..m].iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let norm_a: f32 = kv_a[..m].iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
        let norm_b: f32 = kv_b[..m].iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
        let rel_l2 = l2 / (0.5 * (norm_a + norm_b)).max(1e-6);
        assert!(rel_l2 > 0.05,
            "episodic KV must depend on input content (rel L2 = {rel_l2:.4}, \
             l2={l2:.4}, |kv_a|={norm_a:.4}, |kv_b|={norm_b:.4})");
    }

    /// End-to-end integrated test: brain episodic memory must affect
    /// the **NLL on a query token** when brain output is folded into
    /// LLM logits through `BrainLogitModulator`. This composes:
    ///
    ///   brain(episodic) → brain_output → modulator → Qwen-side NLL
    ///
    /// Two paths from a fresh init using the canonical `eight_region`
    /// (already content-causal at brain-prediction level via the
    /// HIPPOCAMPUS → ATTENTION edge added in `20736b2`):
    ///
    ///   Path A: brain consumes [1..=19], then query=42
    ///   Path B: brain consumes [200..=218], then query=42
    ///
    /// Brain output for path X is `nc_x.step(42)` — vocab-sized
    /// prediction vector. Synthetic qwen_logits = uniform zeros so any
    /// measured NLL delta isolates the modulator's contribution from
    /// the brain's state.
    ///
    /// Assertion: `NLL_a != NLL_b` by more than 1e-4. This is the
    /// pre-requisite for any future "brain reduces Qwen NLL on
    /// held-out text" claim — without it the modulator+brain+NLL
    /// stack is structurally unable to express the dependency.
    #[test]
    fn brain_episodic_memory_affects_modulated_nll() {
        use crate::logit_modulator::{BrainLogitModulator, nll_per_token};

        let mut cfg = RegionalConfig::eight_region(16, 256, 2);
        cfg.router = None; // exercise fixed connections (HIPPO → ATTN edge)
        let w = RegionalWeights::new(cfg);

        // brain_dim = vocab = 256: use brain.step's logit-shaped output
        // directly as the brain_output vector for the modulator.
        let mut m = BrainLogitModulator::new(256, 256);
        m.alpha = 1.0;

        // Path A.
        let mut nc_a = NeuralComputer::new(w.clone());
        for i in 1..=19usize { let _ = nc_a.step(i); }
        let brain_out_a = nc_a.step(42);

        // Path B.
        let mut nc_b = NeuralComputer::new(w);
        for i in 200..=218usize { let _ = nc_b.step(i); }
        let brain_out_b = nc_b.step(42);

        // Synthetic qwen_logits: uniform zeros so brain output is the
        // only signal in the modulated logits. Single-step sequence.
        let qwen = vec![0.0f32; 256];
        let qwen_seq: Vec<&[f32]> = vec![qwen.as_slice()];
        let answer_target: Vec<usize> = vec![100];

        let nll_a = nll_per_token(
            &m, &qwen_seq,
            Some(&[brain_out_a.as_slice()]),
            &answer_target,
        );
        let nll_b = nll_per_token(
            &m, &qwen_seq,
            Some(&[brain_out_b.as_slice()]),
            &answer_target,
        );

        assert!(nll_a.is_finite() && nll_b.is_finite());
        assert!((nll_a - nll_b).abs() > 1e-4,
            "modulated NLL on query token must depend on brain's \
             episodic history (NLL_a={nll_a}, NLL_b={nll_b}); \
             identical NLLs would mean episodic memory cannot \
             affect Qwen-side prediction quality even in principle");
    }

    /// Bridge composition test: the architectural piece that lets
    /// brain weights train against external Qwen NLL via the
    /// modulator's gradient hook. This composes:
    ///
    ///   regional_train_step_generic(brain, grads, obs, |preds, _| {
    ///       let brain_out = preds.last().unwrap();
    ///       modulator forward → modulated logits
    ///       cross_entropy_grad(modulated, target) → d_modulated
    ///       modulator.backward_with_d_brain → d_brain_output
    ///       return (loss, per-tick d_preds with d_brain_output at last)
    ///   })
    ///
    /// This is the bridge from `f6a00f0` (modulator gradient hook)
    /// and `d223503` (synthetic joint training) to the real brain.
    /// The test verifies the full chain composes WITHOUT errors and
    /// that brain gradients accumulate as expected.
    #[test]
    fn regional_train_via_modulator_gradient_bridge() {
        use crate::logit_modulator::{
            LowRankBrainLogitModulator, cross_entropy_grad,
        };

        // Tiny brain: small d_model, brain-output dim of 8 so the
        // modulator can have brain_dim=8 cleanly.
        const BRAIN_OUT: usize = 8;
        const VOCAB: usize = 16;
        let mut cfg = RegionalConfig::eight_region(16, BRAIN_OUT, 2);
        cfg.router = None;
        let weights = RegionalWeights::new(cfg);
        let mut grads = RegionalGradients::zeros(&weights);

        let mut modulator = LowRankBrainLogitModulator::new(BRAIN_OUT, 2, VOCAB);
        modulator.alpha = 1.0;

        let mut d_dw = vec![0.0f32; 2 * BRAIN_OUT];
        let mut d_db = vec![0.0f32; 2];
        let mut d_uw = vec![0.0f32; VOCAB * 2];
        let mut d_ub = vec![0.0f32; VOCAB];

        let observation: Vec<f32> = (0..16).map(|i| (i as f32 * 0.13).sin()).collect();
        let qwen_logits = vec![0.0f32; VOCAB];
        let target_token: usize = 7;

        // Closure: brain produces preds[tick] of size BRAIN_OUT each;
        // we use the last tick as the modulator's brain_output input.
        let modulator_ref = &modulator;
        let qwen_ref = &qwen_logits;
        let d_dw_r = &mut d_dw;
        let d_db_r = &mut d_db;
        let d_uw_r = &mut d_uw;
        let d_ub_r = &mut d_ub;

        let (loss, d_obs) = regional_train_step_generic(
            &weights, &mut grads, &observation,
            |preds, _certainties| {
                let last_tick = preds.len() - 1;
                let brain_output = &preds[last_tick];
                debug_assert_eq!(brain_output.len(), BRAIN_OUT,
                    "brain emits at out_dims=BRAIN_OUT; modulator's brain_dim must match");

                // Brain's out_dims is BRAIN_OUT and modulator's brain_dim
                // is also BRAIN_OUT, so the projection is a square map
                // BRAIN_OUT → BRAIN_OUT. For a real-world Qwen experiment
                // with vocab≠brain_dim, there's an additional projection.
                let mut modulated = vec![0.0f32; VOCAB];
                let mut z = vec![0.0f32; modulator_ref.rank];
                modulator_ref.modulate_into_with_scratch(
                    qwen_ref, brain_output, &mut modulated, &mut z);

                // Cross-entropy gradient w.r.t. modulated logits.
                let mut d_modulated = vec![0.0f32; VOCAB];
                cross_entropy_grad(&modulated, target_token, &mut d_modulated);

                // NLL value (returned to caller as `loss`).
                let max_l = modulated.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let lse: f32 = modulated.iter()
                    .map(|&l| (l - max_l).exp()).sum::<f32>().ln() + max_l;
                let nll = lse - modulated[target_token];

                // Modulator backward with d_brain_output: feeds the
                // gradient back through the projection and produces
                // `d_brain_output` we hand to the brain backward.
                // Sized to brain_dim (= BRAIN_OUT) — the modulator
                // asserts d_brain_output.len() == self.brain_dim.
                let mut d_brain_output = vec![0.0f32; BRAIN_OUT];
                modulator_ref.backward_with_d_brain(
                    brain_output, &d_modulated,
                    d_dw_r, d_db_r, d_uw_r, d_ub_r,
                    &mut d_brain_output,
                );

                // Per-tick d_preds: zero for non-last ticks, gradient
                // at the last tick. Brain's predictions are out_dims
                // (= BRAIN_OUT) so d_preds entries match BRAIN_OUT.
                let n_ticks = preds.len();
                let mut d_preds: Vec<Vec<f32>> = (0..n_ticks)
                    .map(|_| vec![0.0f32; BRAIN_OUT]).collect();
                d_preds[last_tick] = d_brain_output;

                (nll, d_preds)
            },
        );

        // Smoke checks — bridge composes and produces finite outputs.
        assert!(loss.is_finite(), "bridge loss must be finite (got {loss})");
        assert_eq!(d_obs.len(), observation.len(),
            "d_observation should match observation shape");

        // Gradient accumulation check: at least SOMETHING in the
        // brain's gradients should be non-zero (otherwise the
        // modulator's d_brain_output didn't propagate through brain
        // backward).
        // Walk every gradient field and find the largest accumulated
        // value across the brain's parameters. ANY non-zero entry
        // anywhere in the brain's gradients proves the d_preds from
        // our closure reached the brain backward through CTM ticks.
        let kv  = grads.region_grads.iter().flat_map(|rg| rg.kv_proj_w.iter()).map(|g: &f32| g.abs()).sum::<f32>();
        let mha = grads.region_grads.iter().flat_map(|rg| rg.mha_in_w.iter()).map(|g: &f32| g.abs()).sum::<f32>();
        let nlm = grads.region_grads.iter().flat_map(|rg| rg.nlm_s1_w.iter()).map(|g: &f32| g.abs()).sum::<f32>();
        let conn: f32 = grads.connection_dw.iter().flatten().map(|g: &f32| g.abs()).sum::<f32>();
        let outp: f32 = grads.output_proj_dw.iter().map(|g: &f32| g.abs()).sum::<f32>();
        eprintln!("brain grad coverage: kv_proj={kv:.4}  mha_in={mha:.4}  \
                   nlm_s1={nlm:.4}  conn={conn:.4}  output_proj={outp:.4}");
        let any_brain_grad = kv + mha + nlm + conn + outp;
        assert!(any_brain_grad > 0.0,
            "at least one brain gradient field should accumulate from \
             external modulator gradient. All zero means d_preds didn't \
             reach the brain backward through CTM ticks.");

        // Modulator gradients also accumulated.
        let total_mod_grad_norm: f32 = d_uw_r.iter().map(|&g| g.abs()).sum::<f32>()
                                     + d_dw_r.iter().map(|&g| g.abs()).sum::<f32>();
        assert!(total_mod_grad_norm > 0.0,
            "modulator gradients should accumulate from CE backward");
    }

    /// Episodic memory in the hippocampus must propagate to brain
    /// predictions — i.e. the brain's logit on a held-out query token
    /// must depend on episodic content accumulated by prior forwards.
    ///
    /// Two paths from a fresh init:
    ///   A: tokens [1..=19], then query 42
    ///   B: tokens [200..=218], then query 42
    ///
    /// `episodic_memory_is_content_dependent` already proves the
    /// hippocampus KVs at the end of A and B differ substantially.
    /// This test asserts that propagates to logits.
    ///
    /// Uses fixed-connection topology (router disabled): `eight_region`
    /// is configured with `Connection { from: [INPUT, HIPPOCAMPUS], to:
    /// ATTENTION }` so hippocampus's content-dependent state actually
    /// fans into the {attention → output → motor} prediction loop.
    /// With the default learned router enabled, that fixed edge is
    /// shadowed and the propagation becomes empirical (depends on what
    /// the router learns to attend to) — out of scope for an
    /// architecture-truth test.
    #[test]
    fn episodic_memory_propagates_to_prediction() {
        let mut cfg = RegionalConfig::eight_region(16, 256, 2);
        cfg.router = None; // exercise the fixed-connection path
        let w = RegionalWeights::new(cfg);

        let mut nc_a = NeuralComputer::new(w.clone());
        for i in 1..=19usize { let _ = nc_a.step(i); }
        let logits_a = nc_a.step(42);

        let mut nc_b = NeuralComputer::new(w);
        for i in 200..=218usize { let _ = nc_b.step(i); }
        let logits_b = nc_b.step(42);

        assert_eq!(logits_a.len(), logits_b.len());
        let pred_l2: f32 = logits_a.iter().zip(logits_b.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        assert!(pred_l2 > 1e-3,
            "logits on query=42 must differ between paths A and B \
             (L2 = {pred_l2:.6}); identical logits would mean \
             hippocampus → attention edge is not propagating episodic \
             content into the prediction loop");
    }

    #[test]
    fn nc_continuity_survives_save_load() {
        // Clive-Wearing fix #1: after save/load, the NC should have the
        // same history and — via history replay through the CTM dynamics
        // — a state that behaves identically to the pre-save NC.
        let cfg = RegionalConfig::eight_region(16, 256, 2);
        let w = RegionalWeights::new(cfg);
        let mut nc = NeuralComputer::new(w);

        // Build some trajectory.
        for i in 0..30usize {
            let _ = nc.step(i % 256);
        }
        let history_before: Vec<usize> = nc.history.clone();
        let rng_before = nc.rng_state;

        // Save to a temp path (.bin — verifies the bincode roundtrip,
        // which was broken before the ExitStrategy detagging fix).
        let path = std::env::temp_dir()
            .join(format!("nc_continuity_{}.bin",
                std::process::id())).to_string_lossy().into_owned();
        nc.save(&path).expect("save");
        let nc2 = NeuralComputer::load(&path).expect("load");

        // History and RNG seed should match exactly — those are the
        // trajectory proxies for identity.
        assert_eq!(nc2.history, history_before, "history must round-trip");
        assert_eq!(nc2.rng_state, rng_before, "rng_state must round-trip");

        // The next step from either NC should produce identical logits —
        // proof the replayed state is a faithful reconstruction.
        let mut nc_a = nc;
        let mut nc_b = nc2;
        let logits_a = nc_a.step(42);
        let logits_b = nc_b.step(42);
        assert_eq!(logits_a.len(), logits_b.len());
        for (a, b) in logits_a.iter().zip(logits_b.iter()) {
            assert!((a - b).abs() < 1e-3,
                "post-load logits must match pre-save logits (within fp noise): {a} vs {b}");
        }

        // Cleanup.
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(NeuralComputer::sidecar_path(&path));
    }



    /// Phase 5 calibration: `is_resident_likely_faster` returns `true`
    /// only at d_model ≥ 1024 (billion config). All smaller standard
    /// configs (small / medium / large) return `false`. Encodes the
    /// measured crossover so callers don't silently regress on small
    /// brains.
    #[test]
    fn is_resident_likely_faster_crossover_matches_phase5() {
        // small (d=32) — host wins by 5× per measurement
        let cfg = RegionalConfig::eight_region_small(64, 25, 3);
        let w = RegionalWeights::new(cfg);
        assert!(!RegionalBrain::is_resident_likely_faster(&w),
            "small (d_model=32): resident measured 0.18× — heuristic must say false");

        // medium (d=256) — even-ish but resident loses
        let cfg = RegionalConfig::eight_region_medium(64, 25, 3);
        let w = RegionalWeights::new(cfg);
        assert!(!RegionalBrain::is_resident_likely_faster(&w),
            "medium (d_model=256): resident measured 0.50× — heuristic must say false");

        // large (d=512) — close but still loss
        let cfg = RegionalConfig::eight_region_large(64, 25, 3);
        let w = RegionalWeights::new(cfg);
        assert!(!RegionalBrain::is_resident_likely_faster(&w),
            "large (d_model=512): resident measured 0.85× — heuristic must say false");

        // billion (d=1024) — first size where resident wins
        let cfg = RegionalConfig::eight_region_billion(64, 25, 3);
        let w = RegionalWeights::new(cfg);
        assert!(RegionalBrain::is_resident_likely_faster(&w),
            "billion (d_model=1024): resident measured 1.83× — heuristic must say true");
    }



    /// Data-attribution support: flattening RegionalGradients must
    /// produce a stable-length, non-empty Vec<f32> for a fixed brain
    /// configuration. Two calls on the same struct return
    /// the same length (so a JL projection matrix can be reused
    /// across samples), and a backward pass that wrote some non-zero
    /// gradients leaves at least one non-zero entry in the flattened
    /// vector (so projection has signal to work with).
    #[test]
    fn flatten_regional_gradients_is_stable_and_carries_signal() {
        let token_dim = 16usize;
        let n_tokens = 4usize;
        let out_dims = 64usize;
        let ticks = 2usize;
        let cfg = RegionalConfig::eight_region_small(token_dim, out_dims, ticks);
        let w = RegionalWeights::new(cfg);

        let zero_grads = RegionalGradients::zeros(&w);
        let len_a = zero_grads.flatten().len();
        let len_b = zero_grads.flatten().len();
        assert_eq!(len_a, len_b,
            "flatten length must be stable across calls on the same struct");
        assert!(len_a > 0, "flatten produced an empty vector — no parameters?");

        // Run a real backward to populate gradients with non-zero values.
        let tokens = modgrad_traits::TokenInput {
            tokens: (0..n_tokens * token_dim).map(|i| (i as f32 * 0.01).sin()).collect(),
            n_tokens,
            token_dim,
        };
        let state = <RegionalBrain as modgrad_traits::Brain>::init_state(&w);
        let (output, _state, cache) =
            <RegionalBrain as modgrad_traits::Brain>::forward_cached(&w, state, &tokens);
        let d_preds: Vec<Vec<f32>> = output.predictions.iter()
            .map(|p| p.iter().map(|&v| v * 1e-3).collect()).collect();
        let (grads, _) = RegionalBrain::backward_with_input_grad(&w, cache, &d_preds);

        let flat = grads.flatten();
        assert_eq!(flat.len(), len_a,
            "post-backward flatten length must match zero-grads flatten length");
        let nonzero = flat.iter().filter(|v| v.abs() > 0.0).count();
        assert!(nonzero > 0,
            "backward produced no non-zero flattened gradient — JL projection \
             would have nothing to attribute");
        eprintln!(
            "RegionalGradients flatten: dim = {}, nonzero = {} ({:.1}%)",
            len_a, nonzero, 100.0 * nonzero as f32 / len_a as f32,
        );
    }

    #[test]
    fn forward_trace_matches_plain_forward_per_tick() {
        let cfg = RegionalConfig::eight_region_small(16, 8, 3);
        let raw = cfg.raw_obs_dim;
        let w = RegionalWeights::new(cfg);
        let obs: Vec<f32> = (0..raw).map(|i| (i as f32 * 0.013).sin()).collect();

        // Plain forward and traced forward must produce identical numerics.
        let mut s1 = RegionalState::new(&w);
        let out_plain = regional_forward(&w, &mut s1, &obs);
        let mut s2 = RegionalState::new(&w);
        let (out_traced, trace) = regional_forward_trace(&w, &mut s2, &obs);

        assert_eq!(out_plain.ticks_used, out_traced.ticks_used);
        assert_eq!(trace.len(), out_traced.ticks_used,
            "one trace entry per outer tick that ran");
        // The trace's final tick must equal the forward's final readouts.
        let last = trace.last().unwrap();
        assert_eq!(last.global_sync, out_plain.global_sync,
            "final tick global_sync matches the forward output");
        assert_eq!(last.region_activations, out_plain.region_activations,
            "final tick activations match the forward output");
        assert_eq!(last.prediction, *out_plain.predictions.last().unwrap(),
            "final tick prediction matches the forward output");
    }
}
