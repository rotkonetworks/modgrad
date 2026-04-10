//! ReLU² MLP (no bias).
//!
//! y = proj(relu(fc(x))²)
//!
//! Two weight matrices: fc (up-project) and proj (down-project).
//! Activation: max(0, x)² — sparser than GELU, cheaper than SwiGLU.

use super::dims::*;
use super::tensor::WeightMatrix;
use super::ops::TransformerOps;

/// Per-layer MLP weights (bias-free).
pub struct MlpWeights {
    /// Up-projection: [mlp_dim, model_dim].
    pub fc: WeightMatrix<MlpDim, ModelDim>,
    /// Down-projection: [model_dim, mlp_dim].
    pub proj: WeightMatrix<ModelDim, MlpDim>,
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
    ///
    /// `x`: input `[model_dim]`.
    /// `output`: result `[model_dim]`.
    pub fn forward(&self, x: &[f32], output: &mut [f32], backend: &dyn TransformerOps) {
        let mut hidden = vec![0.0f32; self.mlp_dim];

        // Up-project
        backend.matvec_nobias(
            self.weights.fc.as_slice(), x, &mut hidden,
            self.mlp_dim, self.model_dim,
        );

        // ReLU²
        backend.relu_squared_inplace(&mut hidden);

        // Down-project
        backend.matvec_nobias(
            self.weights.proj.as_slice(), &hidden, output,
            self.model_dim, self.mlp_dim,
        );
    }
}
