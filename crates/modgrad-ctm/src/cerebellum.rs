//! Frozen cerebellum: a forward model that replaces the CTM tick loop
//! for the cerebellum region with a single feedforward pass.
//!
//! Biological basis: the cerebellar cortex has uniform cytoarchitecture —
//! granule cell expansion (frozen random projection) + Purkinje readout
//! (trained). Function is determined by connectivity, not local structure.
//!
//! Three modes:
//!   1. CTM (default) — cerebellum is a normal CTM region with tick loop
//!   2. Frozen LLM — pre-trained language model as world model, frozen weights
//!   3. Random expansion — biological: sparse random projection + trained readout
//!
//! In modes 2 and 3, the cerebellum runs in a single forward pass (no ticks),
//! matching the biological cerebellum's fast feedforward computation.

use serde::{Deserialize, Serialize};

// ─── Trait ──────────────────────────────────────────────────

/// A frozen forward model used as cerebellum region.
///
/// Implementations: ONNX backbone (modgrad-io), GGUF (future),
/// RandomExpansion (this module).
///
/// The frozen model receives projected cortex activations and returns
/// hidden states. Projection layers (in/out) are trained; the model
/// itself never updates.
pub trait FrozenCerebellum: Send {
    /// Dimension of the model's input.
    fn input_dim(&self) -> usize;

    /// Dimension of the model's output (hidden states).
    fn output_dim(&self) -> usize;

    /// Forward pass: projected input → hidden states.
    /// Input length = input_dim(), output length = output_dim().
    fn forward(&mut self, input: &[f32]) -> Vec<f32>;
}

// ─── Projection layers ─────────────────────────────────────

/// Trained projection layers that bridge cortex ↔ frozen cerebellum.
///
/// proj_in:  cortex_d_model → frozen_input_dim
/// proj_out: frozen_output_dim → cortex_d_model
///
/// These are the only trainable parameters in the cerebellar pathway
/// when using a frozen model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebProjection {
    /// cortex d_model → frozen input dim
    pub proj_in_w: Vec<f32>,  // [frozen_input_dim × cortex_d_model]
    pub proj_in_b: Vec<f32>,  // [frozen_input_dim]
    /// frozen output dim → cortex d_model
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
    pub fn project_in(&self, cortex: &[f32]) -> Vec<f32> {
        // y = W @ x + b, W is [frozen_input_dim × cortex_dim]
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
        // y = W @ x + b, W is [cortex_dim × frozen_output_dim]
        let mut out = self.proj_out_b.clone();
        for i in 0..self.cortex_dim {
            let row_start = i * self.frozen_output_dim;
            for j in 0..self.frozen_output_dim.min(hidden.len()) {
                out[i] += self.proj_out_w[row_start + j] * hidden[j];
            }
        }
        out
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
    }

    /// View all trainable parameters as flat slice.
    pub fn as_flat(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.n_params());
        v.extend_from_slice(&self.proj_in_w);
        v.extend_from_slice(&self.proj_in_b);
        v.extend_from_slice(&self.proj_out_w);
        v.extend_from_slice(&self.proj_out_b);
        v
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

// ─── Random expansion cerebellum ───────────────────────────

/// Biological cerebellum: frozen random sparse projection (granule cells)
/// + trained readout (Purkinje cells via CerebProjection).
///
/// The expansion matrix is generated once from a seed and never updated.
/// This mimics the ~100 billion granule cells that provide a massive
/// random expansion of the input space.
///
/// expansion_factor=4 means output_dim = input_dim × 4 (matching
/// the biological 4:1 granule:mossy fiber ratio).
pub struct RandomExpansion {
    /// Frozen random weights [output_dim × input_dim].
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
                // ~30% sparse: most weights are zero
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
}

impl FrozenCerebellum for RandomExpansion {
    fn input_dim(&self) -> usize { self.input_dim }
    fn output_dim(&self) -> usize { self.output_dim }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        // y = ReLU(W @ x) — sparse random projection with ReLU
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
    /// expansion_factor = output_dim / input_dim (biological: 4).
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
        assert_eq!(exp.input_dim(), 32);
        assert_eq!(exp.output_dim(), 128);

        let input = vec![1.0f32; 32];
        let output = exp.forward(&input);
        assert_eq!(output.len(), 128);
        // ReLU: all outputs >= 0
        assert!(output.iter().all(|&x| x >= 0.0));
        // Not all zeros (with 30% density and uniform input, most outputs should be nonzero)
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
}
