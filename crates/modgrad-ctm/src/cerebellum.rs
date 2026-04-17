//! Frozen cerebellum: a world model that processes the same sensory stream
//! as the cortex, but in one fast forward pass.
//!
//! Architecture:
//!   token stream → LLM backbone (all layers, one forward pass)
//!                → learned weighted combination across layers
//!                → projection into cerebellum region
//!                → existing brain topology distributes to cortex
//!
//! The cerebellum is the GATEWAY. Cortex regions never touch the LLM
//! directly — they receive the cerebellum's output through the existing
//! connection topology. If we swap Qwen for Llama, only the cerebellum
//! projection changes. The rest of the brain never knows.
//!
//! Multi-layer blending: a learned softmax over all transformer layers
//! determines which depth of representation the cerebellum uses. Early
//! training may favor syntax (shallow layers); later training learns to
//! trust world knowledge (deep layers).

use serde::{Deserialize, Serialize};

// ─── Cache ──────────────────────────────────────────────────

/// Cached hidden states from ALL layers of a frozen cerebellum.
/// Computed once per context window, read per-token during training.
pub struct CerebellumCache {
    /// All layer hidden states, contiguous.
    /// Layout: [layer][position][hidden_dim]
    /// Flat: layer * (n_positions * hidden_dim) + position * hidden_dim
    pub hidden_states: Vec<f32>,
    /// Dimension of each hidden state vector.
    pub hidden_dim: usize,
    /// Number of valid positions.
    pub n_positions: usize,
    /// Number of layers.
    pub n_layers: usize,
}

impl CerebellumCache {
    /// Get hidden state at a specific layer and position.
    pub fn at(&self, layer: usize, position: usize) -> &[f32] {
        if layer < self.n_layers && position < self.n_positions {
            let start = layer * (self.n_positions * self.hidden_dim)
                + position * self.hidden_dim;
            &self.hidden_states[start..start + self.hidden_dim]
        } else {
            &[]
        }
    }

    /// Get the weighted combination across all layers at a position.
    /// `layer_weights` should be softmaxed (sum to 1.0).
    pub fn blend_layers(&self, position: usize, layer_weights: &[f32]) -> Vec<f32> {
        let d = self.hidden_dim;
        let mut out = vec![0.0f32; d];
        let n = self.n_layers.min(layer_weights.len());
        for l in 0..n {
            let h = self.at(l, position);
            if h.is_empty() { continue; }
            let w = layer_weights[l];
            for i in 0..d {
                out[i] += w * h[i];
            }
        }
        out
    }

    /// Same as blend_layers but writes into pre-allocated buffer.
    pub fn blend_layers_into(&self, position: usize, layer_weights: &[f32], out: &mut [f32]) {
        let d = self.hidden_dim;
        out[..d].fill(0.0);
        let n = self.n_layers.min(layer_weights.len());
        for l in 0..n {
            let h = self.at(l, position);
            if h.is_empty() { continue; }
            let w = layer_weights[l];
            for i in 0..d {
                out[i] += w * h[i];
            }
        }
    }

    /// Empty cache (no cerebellum active).
    pub fn empty() -> Self {
        Self { hidden_states: Vec::new(), hidden_dim: 0, n_positions: 0, n_layers: 0 }
    }

    pub fn is_empty(&self) -> bool { self.n_positions == 0 }

    /// Single-layer cache (backward compat with old encode_context).
    pub fn single_layer(hidden_states: Vec<f32>, hidden_dim: usize, n_positions: usize) -> Self {
        Self { hidden_states, hidden_dim, n_positions, n_layers: 1 }
    }
}

// ─── Trait ──────────────────────────────────────────────────

/// A frozen forward model used as cerebellum region.
///
/// LLM implementations should override `encode_context_layers()` to return
/// hidden states from ALL layers. The learned layer weights (in CerebProjection)
/// blend these into a single representation per position.
pub trait FrozenCerebellum: Send {
    /// Dimension of the model's output (hidden states per position).
    fn hidden_dim(&self) -> usize;

    /// Number of layers in the model.
    fn n_layers(&self) -> usize { 1 }

    /// Encode a full context window, returning hidden states from ALL layers.
    /// Default: calls encode_context() for single-layer models.
    fn encode_context_layers(&mut self, token_ids: &[i64]) -> CerebellumCache {
        // Default: single layer (backward compat)
        let cache = self.encode_context(token_ids);
        cache
    }

    /// Encode a full context window (single layer output).
    /// Override this for simple models; override encode_context_layers for LLMs.
    fn encode_context(&mut self, token_ids: &[i64]) -> CerebellumCache {
        let d = self.hidden_dim();
        let n = token_ids.len();
        let mut hidden_states = vec![0.0f32; n * d];
        for (i, &tid) in token_ids.iter().enumerate() {
            let input = vec![tid as f32 / 128.0; d];
            let h = self.forward(&input);
            let copy_len = d.min(h.len());
            hidden_states[i * d..i * d + copy_len].copy_from_slice(&h[..copy_len]);
        }
        CerebellumCache::single_layer(hidden_states, d, n)
    }

    /// Per-token forward pass (for non-LLM models like RandomExpansion).
    fn forward(&mut self, input: &[f32]) -> Vec<f32>;
}

// ─── Projection layers ─────────────────────────────────────

/// Trained projection: frozen cerebellum → cortex.
///
/// Now includes learned layer weights (softmax over N transformer layers)
/// that determine which depth of representation the cerebellum uses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebProjection {
    /// cortex d_model → frozen input dim (used by RandomExpansion)
    pub proj_in_w: Vec<f32>,
    pub proj_in_b: Vec<f32>,
    /// frozen output dim → cortex d_model
    pub proj_out_w: Vec<f32>,
    pub proj_out_b: Vec<f32>,
    /// Dimensions
    pub cortex_dim: usize,
    pub frozen_input_dim: usize,
    pub frozen_output_dim: usize,
    /// Learned layer weight logits (pre-softmax). Length = n_layers.
    /// Softmax determines how much each transformer layer contributes.
    /// Initialized uniform so all layers contribute equally at start.
    #[serde(default)]
    pub layer_weight_logits: Vec<f32>,
}

impl CerebProjection {
    /// Create with Xavier-initialized weights.
    pub fn new(cortex_dim: usize, frozen_input_dim: usize, frozen_output_dim: usize) -> Self {
        Self::with_layers(cortex_dim, frozen_input_dim, frozen_output_dim, 1)
    }

    /// Create with multi-layer support.
    pub fn with_layers(cortex_dim: usize, frozen_input_dim: usize, frozen_output_dim: usize, n_layers: usize) -> Self {
        let scale_in = (6.0 / (cortex_dim + frozen_input_dim) as f32).sqrt();
        let scale_out = (6.0 / (frozen_output_dim + cortex_dim) as f32).sqrt();

        let mut rng = SimpleRng(0xDEAD_BEEF);
        let proj_in_w: Vec<f32> = (0..frozen_input_dim * cortex_dim)
            .map(|_| rng.uniform(-scale_in, scale_in))
            .collect();
        let proj_in_b = vec![0.0; frozen_input_dim];

        let proj_out_w: Vec<f32> = (0..cortex_dim * frozen_output_dim)
            .map(|_| rng.uniform(-scale_out, scale_out))
            .collect();
        let proj_out_b = vec![0.0; cortex_dim];

        // Initialize layer weights uniform (all layers contribute equally)
        let layer_weight_logits = vec![0.0f32; n_layers];

        Self {
            proj_in_w, proj_in_b,
            proj_out_w, proj_out_b,
            cortex_dim, frozen_input_dim, frozen_output_dim,
            layer_weight_logits,
        }
    }

    /// Compute softmaxed layer weights from logits.
    pub fn layer_weights(&self) -> Vec<f32> {
        softmax(&self.layer_weight_logits)
    }

    /// Project cortex activations → frozen model input space.
    pub fn project_in(&self, cortex: &[f32]) -> Vec<f32> {
        debug_assert!(cortex.len() <= self.cortex_dim,
            "project_in: input {} > cortex_dim {}", cortex.len(), self.cortex_dim);
        let mut out = self.proj_in_b.clone();
        for i in 0..self.frozen_input_dim {
            let row_start = i * self.cortex_dim;
            for j in 0..self.cortex_dim.min(cortex.len()) {
                out[i] += self.proj_in_w[row_start + j] * cortex[j];
            }
        }
        out
    }

    /// Project frozen model output → cortex activation space.
    pub fn project_out(&self, hidden: &[f32]) -> Vec<f32> {
        let mut out = self.proj_out_b.clone();
        for i in 0..self.cortex_dim {
            let row_start = i * self.frozen_output_dim;
            for j in 0..self.frozen_output_dim.min(hidden.len()) {
                out[i] += self.proj_out_w[row_start + j] * hidden[j];
            }
        }
        out
    }

    /// Project frozen hidden state into a pre-allocated cortex buffer.
    pub fn project_out_into(&self, hidden: &[f32], out: &mut [f32]) {
        out[..self.cortex_dim].copy_from_slice(&self.proj_out_b);
        for i in 0..self.cortex_dim {
            let row_start = i * self.frozen_output_dim;
            for j in 0..self.frozen_output_dim.min(hidden.len()) {
                out[i] += self.proj_out_w[row_start + j] * hidden[j];
            }
        }
    }

    /// Backward pass for project_out.
    pub fn backward_out(
        &self,
        hidden: &[f32],
        d_cortex: &[f32],
        d_w: &mut [f32],
        d_b: &mut [f32],
    ) {
        for i in 0..self.cortex_dim {
            d_b[i] += d_cortex[i];
        }
        for i in 0..self.cortex_dim {
            let row_start = i * self.frozen_output_dim;
            for j in 0..self.frozen_output_dim.min(hidden.len()) {
                d_w[row_start + j] += d_cortex[i] * hidden[j];
            }
        }
    }

    /// Full forward: cortex → project_in → frozen → project_out → cortex.
    pub fn forward(&self, frozen: &mut dyn FrozenCerebellum, cortex_input: &[f32]) -> Vec<f32> {
        let projected_in = self.project_in(cortex_input);
        let hidden = frozen.forward(&projected_in);
        self.project_out(&hidden)
    }

    /// Total trainable parameters.
    pub fn n_params(&self) -> usize {
        self.proj_in_w.len() + self.proj_in_b.len()
            + self.proj_out_w.len() + self.proj_out_b.len()
            + self.layer_weight_logits.len()
    }

    /// Mutably view parameters for gradient update.
    pub fn params_mut(&mut self) -> [&mut [f32]; 4] {
        [
            &mut self.proj_in_w,
            &mut self.proj_in_b,
            &mut self.proj_out_w,
            &mut self.proj_out_b,
        ]
    }
}

/// Read a position from the cerebellum cache with multi-layer blending,
/// project to cortex dim. Uses pre-allocated buffer.
pub fn cerebellum_at_position(
    cache: &CerebellumCache,
    proj: &CerebProjection,
    position: usize,
    out: &mut [f32],
) {
    if cache.is_empty() || position >= cache.n_positions {
        out[..proj.cortex_dim].fill(0.0);
        return;
    }

    if cache.n_layers == 1 {
        // Single layer: direct project (backward compat)
        let hidden = cache.at(0, position);
        if hidden.is_empty() {
            out[..proj.cortex_dim].fill(0.0);
        } else {
            proj.project_out_into(hidden, out);
        }
    } else {
        // Multi-layer: blend across layers with learned weights, then project
        let weights = proj.layer_weights();
        let blended = cache.blend_layers(position, &weights);
        proj.project_out_into(&blended, out);
    }
}

/// Get the blended hidden state at a position (for gradient computation).
/// Returns the blended hidden vector before projection.
pub fn blended_hidden_at(
    cache: &CerebellumCache,
    proj: &CerebProjection,
    position: usize,
) -> Vec<f32> {
    if cache.n_layers <= 1 {
        cache.at(0, position).to_vec()
    } else {
        let weights = proj.layer_weights();
        cache.blend_layers(position, &weights)
    }
}

// ─── Random expansion cerebellum ───────────────────────────

/// Biological cerebellum: frozen random sparse projection (granule cells).
pub struct RandomExpansion {
    weights: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
}

impl RandomExpansion {
    pub fn new(input_dim: usize, expansion_factor: usize, seed: u64) -> Self {
        let output_dim = input_dim * expansion_factor;
        let scale = 1.0 / (input_dim as f32).sqrt();
        let mut rng = SimpleRng(seed);

        let weights: Vec<f32> = (0..output_dim * input_dim)
            .map(|_| {
                if rng.uniform(0.0, 1.0) > 0.3 { 0.0 }
                else if rng.uniform(0.0, 1.0) > 0.5 { scale }
                else { -scale }
            })
            .collect();

        Self { weights, input_dim, output_dim }
    }

    pub fn input_dim(&self) -> usize { self.input_dim }
    pub fn output_dim(&self) -> usize { self.output_dim }
}

impl FrozenCerebellum for RandomExpansion {
    fn hidden_dim(&self) -> usize { self.output_dim }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.output_dim];
        for i in 0..self.output_dim {
            let row = i * self.input_dim;
            let mut sum = 0.0f32;
            for j in 0..self.input_dim.min(input.len()) {
                sum += self.weights[row + j] * input[j];
            }
            out[i] = sum.max(0.0);
        }
        out
    }
}

// ─── Utilities ─────────────────────────────────────────────

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() { return vec![]; }
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum::<f32>().max(1e-8);
    exp.iter().map(|&e| e / sum).collect()
}

struct SimpleRng(u64);
impl SimpleRng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let t = (self.next() >> 33) as f32 / (1u64 << 31) as f32;
        lo + t * (hi - lo)
    }
}

// ─── Config ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CerebMode {
    Ctm,
    Expansion { expansion_factor: usize, seed: u64 },
    External,
}

impl Default for CerebMode {
    fn default() -> Self { Self::Ctm }
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_expansion_dims() {
        let mut exp = RandomExpansion::new(32, 4, 42);
        assert_eq!(exp.hidden_dim(), 128);
        let output = exp.forward(&vec![1.0f32; 32]);
        assert_eq!(output.len(), 128);
        assert!(output.iter().all(|&x| x >= 0.0));
        assert!(output.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn projection_roundtrip() {
        let proj = CerebProjection::new(32, 64, 128);
        let cortex = vec![1.0f32; 32];
        assert_eq!(proj.project_in(&cortex).len(), 64);
        assert_eq!(proj.project_out(&vec![0.5f32; 128]).len(), 32);
    }

    #[test]
    fn full_forward_with_expansion() {
        let mut exp = RandomExpansion::new(64, 4, 42);
        let proj = CerebProjection::new(32, 64, 256);
        let output = proj.forward(&mut exp, &vec![1.0f32; 32]);
        assert_eq!(output.len(), 32);
    }

    #[test]
    fn multi_layer_cache() {
        let d = 4;
        let n_pos = 3;
        let n_layers = 3;
        // Layer 0: all 1s, Layer 1: all 2s, Layer 2: all 3s
        let mut data = Vec::new();
        for l in 0..n_layers {
            for _p in 0..n_pos {
                data.extend(vec![(l + 1) as f32; d]);
            }
        }
        let cache = CerebellumCache {
            hidden_states: data,
            hidden_dim: d,
            n_positions: n_pos,
            n_layers,
        };

        // Layer 0, pos 0 should be [1,1,1,1]
        assert_eq!(cache.at(0, 0), &[1.0, 1.0, 1.0, 1.0]);
        // Layer 2, pos 1 should be [3,3,3,3]
        assert_eq!(cache.at(2, 1), &[3.0, 3.0, 3.0, 3.0]);

        // Uniform blend: (1+2+3)/3 = 2.0
        let weights = vec![1.0 / 3.0; 3];
        let blended = cache.blend_layers(0, &weights);
        assert!((blended[0] - 2.0).abs() < 1e-6);

        // Weighted toward layer 2: 0.1*1 + 0.1*2 + 0.8*3 = 2.7
        let weights = vec![0.1, 0.1, 0.8];
        let blended = cache.blend_layers(0, &weights);
        assert!((blended[0] - 2.7).abs() < 1e-6);
    }

    #[test]
    fn layer_weights_softmax() {
        let proj = CerebProjection::with_layers(8, 16, 64, 3);
        let w = proj.layer_weights();
        assert_eq!(w.len(), 3);
        // Uniform logits → uniform weights
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((w[0] - w[1]).abs() < 1e-6);
    }

    #[test]
    fn cerebellum_at_position_multilayer() {
        let d = 4;
        let n_pos = 2;
        let n_layers = 2;
        let mut data = Vec::new();
        // Layer 0: [1,1,1,1], Layer 1: [3,3,3,3]
        for l in 0..n_layers {
            for _p in 0..n_pos {
                data.extend(vec![if l == 0 { 1.0 } else { 3.0 }; d]);
            }
        }
        let cache = CerebellumCache {
            hidden_states: data, hidden_dim: d, n_positions: n_pos, n_layers,
        };
        let proj = CerebProjection::with_layers(4, 4, 4, 2);
        let mut out = vec![0.0f32; 4];
        cerebellum_at_position(&cache, &proj, 0, &mut out);
        // Should be non-zero (projection of blended [2,2,2,2])
        assert!(out.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn backward_out_accumulates() {
        let proj = CerebProjection::new(4, 8, 16);
        let hidden = vec![1.0f32; 16];
        let d_cortex = vec![1.0f32; 4];
        let mut d_w = vec![0.0f32; 4 * 16];
        let mut d_b = vec![0.0f32; 4];
        proj.backward_out(&hidden, &d_cortex, &mut d_w, &mut d_b);
        assert_eq!(d_b, vec![1.0; 4]);
        for i in 0..4 {
            for j in 0..16 {
                assert_eq!(d_w[i * 16 + j], d_cortex[i] * hidden[j]);
            }
        }
    }
}
