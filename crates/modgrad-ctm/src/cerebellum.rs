//! Frozen cerebellum: a world model that processes the same sensory stream
//! as the cortex, but in one fast forward pass.
//!
//! Architecture (from the Henry de Valence meditation):
//!
//!   token stream: [t0, t1, t2, ..., tn]
//!                   ↓
//!   LLM backbone:  [h0, h1, h2, ..., hn]  ← one forward pass, cached
//!                   ↓
//!   cortex at token ti reads h[i] through trained projection
//!
//! The cerebellum processes the full context window ONCE. The cortex reads
//! position-aligned hidden states during its per-token tick loop. This is
//! the biological architecture: cerebellum is fast feedforward, cortex is
//! slow recurrent. The cortex learns WHEN to trust the cerebellum.
//!
//! Three modes:
//!   1. CTM (default) — cerebellum is a normal CTM region with tick loop
//!   2. Frozen LLM — pre-trained language model, processes full context
//!   3. Random expansion — biological granule cell model (per-token)

use serde::{Deserialize, Serialize};

// ─── Cache ──────────────────────────────────────────────────

/// Cached hidden states from a frozen cerebellum's context encoding.
/// Computed once per context window, read per-token during training.
pub struct CerebellumCache {
    /// hidden_states[position] has length hidden_dim.
    /// Contiguous: position i starts at offset i * hidden_dim.
    pub hidden_states: Vec<f32>,
    /// Dimension of each hidden state vector.
    pub hidden_dim: usize,
    /// Number of valid positions.
    pub len: usize,
}

impl CerebellumCache {
    /// Get hidden state at a position. Returns zeros if out of bounds.
    pub fn at(&self, position: usize) -> &[f32] {
        if position < self.len {
            let start = position * self.hidden_dim;
            &self.hidden_states[start..start + self.hidden_dim]
        } else {
            // Return a static zero slice would require unsafe; instead
            // callers should check bounds. This path shouldn't be hit
            // in correct usage.
            &[]
        }
    }

    /// Empty cache (no cerebellum active).
    pub fn empty() -> Self {
        Self { hidden_states: Vec::new(), hidden_dim: 0, len: 0 }
    }

    /// Number of cached positions.
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

// ─── Trait ──────────────────────────────────────────────────

/// A frozen forward model used as cerebellum region.
///
/// Two usage patterns:
///   1. Context-based (LLMs): call `encode_context()` once per window,
///      then read positions from the cache during per-token training.
///   2. Per-token (RandomExpansion): call `forward()` per token.
///
/// LLM implementations should override `encode_context()`.
/// `forward()` is the fallback for per-token models.
pub trait FrozenCerebellum: Send {
    /// Dimension of the model's output (hidden states per position).
    fn hidden_dim(&self) -> usize;

    /// Encode a full context window of token IDs.
    /// Returns cached hidden states for each position.
    /// Default: calls forward() per token (for non-LLM models).
    fn encode_context(&mut self, token_ids: &[i64]) -> CerebellumCache {
        let d = self.hidden_dim();
        let n = token_ids.len();
        let mut hidden_states = vec![0.0f32; n * d];
        for (i, &tid) in token_ids.iter().enumerate() {
            // Convert token ID to a simple float input for per-token models
            let input = vec![tid as f32 / 128.0; d];
            let h = self.forward(&input);
            let copy_len = d.min(h.len());
            hidden_states[i * d..i * d + copy_len].copy_from_slice(&h[..copy_len]);
        }
        CerebellumCache { hidden_states, hidden_dim: d, len: n }
    }

    /// Per-token forward pass (for non-LLM models like RandomExpansion).
    fn forward(&mut self, input: &[f32]) -> Vec<f32>;
}

// ─── Projection layers ─────────────────────────────────────

/// Trained projection: frozen hidden_dim → cortex d_model.
///
/// Only proj_out is needed in the new architecture — the cerebellum
/// sees raw tokens (not cortex activations), so proj_in is vestigial.
/// Kept for backward compatibility with RandomExpansion mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebProjection {
    /// cortex d_model → frozen input dim (used by RandomExpansion)
    pub proj_in_w: Vec<f32>,  // [frozen_input_dim × cortex_d_model]
    pub proj_in_b: Vec<f32>,  // [frozen_input_dim]
    /// frozen output dim → cortex d_model (the important one)
    pub proj_out_w: Vec<f32>, // [cortex_d_model × frozen_output_dim]
    pub proj_out_b: Vec<f32>, // [cortex_d_model]
    /// Dimensions
    pub cortex_dim: usize,
    pub frozen_input_dim: usize,
    pub frozen_output_dim: usize,
}

impl CerebProjection {
    /// Create with Xavier-initialized weights.
    pub fn new(cortex_dim: usize, frozen_input_dim: usize, frozen_output_dim: usize) -> Self {
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

        Self {
            proj_in_w, proj_in_b,
            proj_out_w, proj_out_b,
            cortex_dim, frozen_input_dim, frozen_output_dim,
        }
    }

    /// Project cortex activations → frozen model input space.
    /// Used by RandomExpansion mode (cortex → granule cells).
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
    /// This is the main projection for LLM cerebellum mode.
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

    /// Project frozen hidden state into a pre-allocated cortex buffer (zero alloc).
    pub fn project_out_into(&self, hidden: &[f32], out: &mut [f32]) {
        out[..self.cortex_dim].copy_from_slice(&self.proj_out_b);
        for i in 0..self.cortex_dim {
            let row_start = i * self.frozen_output_dim;
            for j in 0..self.frozen_output_dim.min(hidden.len()) {
                out[i] += self.proj_out_w[row_start + j] * hidden[j];
            }
        }
    }

    /// Backward pass for project_out: given d_cortex, compute d_hidden and
    /// accumulate gradients for proj_out_w and proj_out_b.
    ///
    /// Returns d_hidden (gradient w.r.t. frozen hidden states — discarded
    /// since the LLM is frozen, but useful for debugging).
    pub fn backward_out(
        &self,
        hidden: &[f32],      // input to project_out (frozen, constant)
        d_cortex: &[f32],    // upstream gradient
        d_w: &mut [f32],     // accumulate into proj_out_w gradient
        d_b: &mut [f32],     // accumulate into proj_out_b gradient
    ) {
        // y = W @ hidden + b
        // d_b += d_cortex
        for i in 0..self.cortex_dim {
            d_b[i] += d_cortex[i];
        }
        // d_W[i, j] += d_cortex[i] * hidden[j]
        for i in 0..self.cortex_dim {
            let row_start = i * self.frozen_output_dim;
            for j in 0..self.frozen_output_dim.min(hidden.len()) {
                d_w[row_start + j] += d_cortex[i] * hidden[j];
            }
        }
    }

    /// Full forward: cortex → project_in → frozen → project_out → cortex.
    /// Used by RandomExpansion mode.
    pub fn forward(&self, frozen: &mut dyn FrozenCerebellum, cortex_input: &[f32]) -> Vec<f32> {
        let projected_in = self.project_in(cortex_input);
        let hidden = frozen.forward(&projected_in);
        self.project_out(&hidden)
    }

    /// Total trainable parameters.
    pub fn n_params(&self) -> usize {
        self.proj_in_w.len() + self.proj_in_b.len()
            + self.proj_out_w.len() + self.proj_out_b.len()
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

/// Read a position from the cerebellum cache, project to cortex dim.
/// Zero allocations when using project_out_into.
pub fn cerebellum_at_position(
    cache: &CerebellumCache,
    proj: &CerebProjection,
    position: usize,
    out: &mut [f32],
) {
    let hidden = cache.at(position);
    if hidden.is_empty() {
        out[..proj.cortex_dim].fill(0.0);
    } else {
        proj.project_out_into(hidden, out);
    }
}

// ─── Random expansion cerebellum ───────────────────────────

/// Biological cerebellum: frozen random sparse projection (granule cells)
/// + trained readout (Purkinje cells via CerebProjection).
///
/// The expansion matrix is generated once from a seed and never updated.
/// expansion_factor=4 means output_dim = input_dim × 4 (biological 4:1).
pub struct RandomExpansion {
    weights: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
}

impl RandomExpansion {
    /// Create with sparse random weights (±1/sqrt(input_dim), ~30% density).
    pub fn new(input_dim: usize, expansion_factor: usize, seed: u64) -> Self {
        let output_dim = input_dim * expansion_factor;
        let scale = 1.0 / (input_dim as f32).sqrt();
        let mut rng = SimpleRng(seed);

        let weights: Vec<f32> = (0..output_dim * input_dim)
            .map(|_| {
                if rng.uniform(0.0, 1.0) > 0.3 {
                    0.0
                } else if rng.uniform(0.0, 1.0) > 0.5 {
                    scale
                } else {
                    -scale
                }
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
            out[i] = sum.max(0.0); // ReLU
        }
        out
    }
}

// ─── Simple RNG ────────────────────────────────────────────

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

/// Configuration for the cerebellum region mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CerebMode {
    /// Standard CTM region (current behavior, backward compatible).
    Ctm,
    /// Frozen random expansion (biological granule cell model).
    Expansion { expansion_factor: usize, seed: u64 },
    /// Frozen external model (ONNX, GGUF, etc).
    /// The FrozenCerebellum is injected at runtime, not serialized.
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

        let input = vec![1.0f32; 32];
        let output = exp.forward(&input);
        assert_eq!(output.len(), 128);
        assert!(output.iter().all(|&x| x >= 0.0));
        assert!(output.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn projection_roundtrip() {
        let proj = CerebProjection::new(32, 64, 128);
        assert_eq!(proj.n_params(), 32 * 64 + 64 + 32 * 128 + 32);

        let cortex = vec![1.0f32; 32];
        let projected = proj.project_in(&cortex);
        assert_eq!(projected.len(), 64);

        let hidden = vec![0.5f32; 128];
        let back = proj.project_out(&hidden);
        assert_eq!(back.len(), 32);
    }

    #[test]
    fn full_forward_with_expansion() {
        let mut exp = RandomExpansion::new(64, 4, 42);
        let proj = CerebProjection::new(32, 64, 256);

        let cortex_input = vec![1.0f32; 32];
        let output = proj.forward(&mut exp, &cortex_input);
        assert_eq!(output.len(), 32);
    }

    #[test]
    fn cache_encode_and_read() {
        let mut exp = RandomExpansion::new(16, 4, 42);
        let cache = exp.encode_context(&[1, 2, 3, 4]);
        assert_eq!(cache.len, 4);
        assert_eq!(cache.hidden_dim, 64);

        let h0 = cache.at(0);
        assert_eq!(h0.len(), 64);
        let h3 = cache.at(3);
        assert_eq!(h3.len(), 64);
        // Out of bounds returns empty
        assert!(cache.at(4).is_empty());
    }

    #[test]
    fn cerebellum_at_position_works() {
        let mut exp = RandomExpansion::new(16, 4, 42);
        let proj = CerebProjection::new(8, 16, 64);
        let cache = exp.encode_context(&[10, 20, 30]);

        let mut out = vec![0.0f32; 8];
        cerebellum_at_position(&cache, &proj, 0, &mut out);
        assert_eq!(out.len(), 8);
        // Should be non-zero (projection of non-zero hidden states)
        assert!(out.iter().any(|&x| x != 0.0));

        // Out of bounds gives zeros
        let mut out2 = vec![0.0f32; 8];
        cerebellum_at_position(&cache, &proj, 99, &mut out2);
        assert!(out2.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn backward_out_accumulates() {
        let proj = CerebProjection::new(4, 8, 16);
        let hidden = vec![1.0f32; 16];
        let d_cortex = vec![1.0f32; 4];
        let mut d_w = vec![0.0f32; 4 * 16];
        let mut d_b = vec![0.0f32; 4];

        proj.backward_out(&hidden, &d_cortex, &mut d_w, &mut d_b);

        // d_b should equal d_cortex
        assert_eq!(d_b, vec![1.0; 4]);
        // d_w should be outer product d_cortex ⊗ hidden
        for i in 0..4 {
            for j in 0..16 {
                assert_eq!(d_w[i * 16 + j], d_cortex[i] * hidden[j]);
            }
        }
    }
}
