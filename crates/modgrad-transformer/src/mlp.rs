//! MLP modules: ReLU² and SwiGLU (no bias).
//!
//! ReLU²:  y = proj(relu(fc(x))²)
//! SwiGLU: y = down(silu(gate(x)) * up(x))
//!
//! SwiGLU is used by Llama, Mistral, Gemma, Qwen — the standard for
//! modern LLMs. ReLU² is the original modgrad choice (sparser activation).

use super::dims::*;
use super::tensor::WeightMatrix;
use super::ops::TransformerOps;

/// Per-layer ReLU² MLP weights (bias-free).
pub struct MlpWeights {
    /// Up-projection: [mlp_dim, model_dim].
    pub fc: WeightMatrix<MlpDim, ModelDim>,
    /// Down-projection: [model_dim, mlp_dim].
    pub proj: WeightMatrix<ModelDim, MlpDim>,
}

/// Per-layer SwiGLU MLP weights (bias-free).
/// Three matrices: gate, up (both model→mlp), down (mlp→model).
pub struct SwigluWeights {
    /// Gate projection: [mlp_dim, model_dim].
    pub gate: WeightMatrix<MlpDim, ModelDim>,
    /// Up projection: [mlp_dim, model_dim].
    pub up: WeightMatrix<MlpDim, ModelDim>,
    /// Down projection: [model_dim, mlp_dim].
    pub down: WeightMatrix<ModelDim, MlpDim>,
}

/// ReLU² MLP module.
pub struct Mlp {
    pub weights: MlpWeights,
    model_dim: usize,
    mlp_dim: usize,
}

impl Mlp {
    pub fn new(weights: MlpWeights, model_dim: ModelDim, mlp_dim: MlpDim) -> Self {
        Self {
            weights,
            model_dim: model_dim.get(),
            mlp_dim: mlp_dim.get(),
        }
    }

    /// Forward: x → fc → relu² → proj → output.
    pub fn forward(&self, x: &[f32], output: &mut [f32], backend: &dyn TransformerOps) {
        let mut hidden = vec![0.0f32; self.mlp_dim];

        backend.matvec_nobias(
            self.weights.fc.as_slice(), x, &mut hidden,
            self.mlp_dim, self.model_dim,
        );
        backend.relu_squared_inplace(&mut hidden);
        backend.matvec_nobias(
            self.weights.proj.as_slice(), &hidden, output,
            self.model_dim, self.mlp_dim,
        );
    }
}

/// SwiGLU MLP module.
/// y = down(silu(gate(x)) * up(x))
/// Standard for Llama 2/3, Mistral, Gemma 2/3/4, Qwen 2/3.
pub struct SwigluMlp {
    pub weights: SwigluWeights,
    model_dim: usize,
    mlp_dim: usize,
}

impl SwigluMlp {
    pub fn new(weights: SwigluWeights, model_dim: ModelDim, mlp_dim: MlpDim) -> Self {
        Self {
            weights,
            model_dim: model_dim.get(),
            mlp_dim: mlp_dim.get(),
        }
    }

    /// Forward: x → gate+up → silu_gated → down → output.
    pub fn forward(&self, x: &[f32], output: &mut [f32], backend: &dyn TransformerOps) {
        let mut gate = vec![0.0f32; self.mlp_dim];
        let mut up = vec![0.0f32; self.mlp_dim];
        let mut hidden = vec![0.0f32; self.mlp_dim];

        // Gate and up projections
        backend.matvec_nobias(
            self.weights.gate.as_slice(), x, &mut gate,
            self.mlp_dim, self.model_dim,
        );
        backend.matvec_nobias(
            self.weights.up.as_slice(), x, &mut up,
            self.mlp_dim, self.model_dim,
        );

        // SiLU gating: hidden = silu(gate) * up
        backend.silu_gated(&gate, &up, &mut hidden);

        // Down projection
        backend.matvec_nobias(
            self.weights.down.as_slice(), &hidden, output,
            self.model_dim, self.mlp_dim,
        );
    }
}
