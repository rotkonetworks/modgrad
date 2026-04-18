//! TransformerBuilder: builder for constructing a GptModel.
//!
//! Usage:
//! ```ignore
//! let service = TransformerBuilder::new(config)
//!     .with_weights(weights)
//!     .with_backend(backend)
//!     .with_position(DynamicPositioning::linear(w))
//!     .build()?;
//! ```

use super::attention::{AttentionWeights, CausalSelfAttention};
use super::block::TransformerBlock;
use super::config::GptConfig;
use super::dims::*;
use super::error::{TransformerError, Result};
use super::mlp::{Mlp, MlpWeights};
use super::model::GptModel;
use super::norm::ScaledRmsNorm;
use super::ops::TransformerOps;
use super::position::PositionEncoding;
use super::position::fixed::FixedPositioning;
use super::residual::ResidualLambdas;
use super::rope::RotaryEmbedding;
use super::smear::{Inference, Training, Smear, SmearWeights};
use super::tensor::Tensor2;
use super::value_embed::ValueEmbedding;
use super::weights::GptWeights;

pub struct TransformerBuilder {
    config: GptConfig,
    weights: Option<GptWeights>,
    backend: Option<Box<dyn TransformerOps>>,
    position: Option<Box<dyn PositionEncoding>>,
}

impl TransformerBuilder {
    pub fn new(config: GptConfig) -> Self {
        Self {
            config,
            weights: None,
            backend: None,
            position: None,
        }
    }

    pub fn with_weights(mut self, weights: GptWeights) -> Self {
        self.weights = Some(weights);
        self
    }

    pub fn with_backend(mut self, backend: impl TransformerOps + 'static) -> Self {
        self.backend = Some(Box::new(backend));
        self
    }

    /// Set a custom position encoding strategy. Defaults to FixedPositioning (standard RoPE).
    pub fn with_position(mut self, pos: impl PositionEncoding + 'static) -> Self {
        self.position = Some(Box::new(pos));
        self
    }

    /// Build the GptModel.
    ///
    /// Validates config + weights at build time.
    pub fn build(self) -> Result<GptModel> {
        let config = self.config;
        config.validate()?;

        let weights = self.weights
            .ok_or(TransformerError::BuilderMissing("weights"))?;
        weights.validate(&config)?;

        let _backend = self.backend
            .ok_or(TransformerError::BuilderMissing("backend"))?;

        let position: Box<dyn PositionEncoding> = self.position
            .unwrap_or_else(|| Box::new(FixedPositioning));

        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let mlp_dim = config.mlp_dim.get();
        let vocab = config.vocab_size.get();

        // Build embed + lm_head
        let embed = Tensor2::new(weights.token_embed, vocab, md)
            .ok_or(TransformerError::ShapeMismatch {
                context: "embed", expected: vocab * md, actual: 0,
            })?;

        let lm_head = Tensor2::new(weights.lm_head, vocab, md)
            .ok_or(TransformerError::ShapeMismatch {
                context: "lm_head", expected: vocab * md, actual: 0,
            })?;

        // Final norm
        let final_norm = ScaledRmsNorm::new(weights.final_norm_scale, config.norm_eps);

        // Smear — both modes share the same weights (clone the data)
        let smear_weights_inf = SmearWeights::new(
            weights.smear_gate.clone(), config.model_dim, &config.smear);
        let smear_weights_train = SmearWeights::new(
            weights.smear_gate, config.model_dim, &config.smear);
        let smear_inference = Smear::<Inference>::new(smear_weights_inf);
        let smear_training = Smear::<Training>::new(smear_weights_train);

        // RoPE
        let rope = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_base);

        // Residual lambdas
        let lambdas = ResidualLambdas::from_config(&config.residual, config.num_layers);

        // Build blocks
        let mut blocks = Vec::with_capacity(config.num_layers.get());
        for (i, bw) in weights.blocks.into_iter().enumerate() {
            let layer_idx = LayerIdx::new_unchecked(i, config.num_layers);

            let attn_weights = AttentionWeights {
                wq: Tensor2::new(bw.wq, md, md).unwrap(),
                wk: Tensor2::new(bw.wk, kv_dim, md).unwrap(),
                wv: Tensor2::new(bw.wv, kv_dim, md).unwrap(),
                wo: Tensor2::new(bw.wo, md, md).unwrap(),
            };
            let attn = CausalSelfAttention::new(attn_weights, &config);

            let mlp_weights = MlpWeights {
                fc: Tensor2::new(bw.mlp_fc, mlp_dim, md).unwrap(),
                proj: Tensor2::new(bw.mlp_proj, md, mlp_dim).unwrap(),
            };
            let mlp = Mlp::new(mlp_weights, config.model_dim, config.mlp_dim);

            let ve = if config.has_value_embed(layer_idx) {
                Some(ValueEmbedding::new(
                    bw.ve_table.unwrap(),
                    bw.ve_gate.unwrap(),
                    config.vocab_size,
                    kv_dim,
                    &config.value_embed,
                ))
            } else {
                None
            };

            blocks.push(TransformerBlock::new(attn, mlp, ve, layer_idx, &config));
        }

        let model = GptModel {
            embed,
            lm_head,
            final_norm,
            smear_inference,
            smear_training,
            blocks,
            lambdas,
            rope,
            position,
            config,
        };

        Ok(model)
    }
}
