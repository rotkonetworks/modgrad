//! Transformer configuration.
//!
//! All hyperparameters in one place. Validated at construction time.

use serde::{Deserialize, Serialize};
use super::dims::*;
use super::error::{TransformerError, Result};

/// Sliding window pattern for attention layers.
/// "SSSL" = Short, Short, Short, Long (repeating).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowPattern {
    /// All layers use full attention.
    Full,
    /// Repeating SSSL: short window = ceil(seq_len / 4 / 128) * 128.
    Sssl,
    /// Custom per-layer window sizes.
    Custom(Vec<usize>),
}

/// Precision policy for compute.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Precision {
    /// f32 everywhere.
    F32,
    /// bf16 weights, f32 accumulation.
    Bf16Accum,
}

/// Value embedding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueEmbedConfig {
    /// Number of input channels for the VE gate (default: 12).
    pub gate_channels: usize,
    /// Output range for the sigmoid gate: gate = range * sigmoid(W @ x[..channels]).
    pub gate_range: f32,
}

impl Default for ValueEmbedConfig {
    fn default() -> Self {
        Self { gate_channels: 12, gate_range: 3.0 }
    }
}

/// Residual stream configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualConfig {
    /// Initial resid_lambda for layer 0 (linearly interpolated to `resid_end`).
    pub resid_start: f32,
    /// Final resid_lambda for last layer.
    pub resid_end: f32,
    /// Initial x0_lambda for layer 0.
    pub x0_start: f32,
    /// Final x0_lambda for last layer.
    pub x0_end: f32,
    /// Backout lambda (applied to cached midpoint hidden state).
    pub backout_lambda: f32,
}

impl Default for ResidualConfig {
    fn default() -> Self {
        Self {
            resid_start: 1.15,
            resid_end: 0.69,
            x0_start: 0.20,
            x0_end: -0.10,
            backout_lambda: 0.5,
        }
    }
}

/// Smear (previous-token mixing) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmearConfig {
    /// Number of input channels for the smear gate.
    pub gate_channels: usize,
    /// Mixing strength lambda.
    pub lambda: f32,
}

impl Default for SmearConfig {
    fn default() -> Self {
        Self { gate_channels: 24, lambda: 1.0 }
    }
}

/// MLP activation type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MlpActivation {
    /// y = max(0, x)² — sparser than GELU.
    ReluSquared,
    /// SiLU-gated (SwiGLU): y = silu(gate(x)) * up(x)
    /// Used by Llama, Mistral, Gemma, Qwen.
    SwiGlu,
}

/// Per-layer attention configuration override.
/// When None, uses the global config values.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LayerOverride {
    /// Override head_dim for this layer (e.g. SWA vs full attention).
    pub head_dim: Option<usize>,
    /// Override num_heads for this layer.
    pub num_heads: Option<usize>,
    /// Override num_kv_heads for this layer.
    pub num_kv_heads: Option<usize>,
    /// Override RoPE base frequency for this layer.
    pub rope_base: Option<f32>,
    /// KV sharing: instead of computing K/V, reuse from this layer index.
    pub share_kv_from: Option<usize>,
    /// Whether this layer has a per-layer input gate (Gemma 4).
    pub has_input_gate: bool,
}

/// Full transformer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptConfig {
    pub model_dim: ModelDim,
    pub num_heads: NumHeads,
    pub num_kv_heads: NumKvHeads,
    pub head_dim: HeadDim,
    pub num_layers: NumLayers,
    pub vocab_size: VocabSize,
    pub mlp_dim: MlpDim,
    pub max_seq_len: SeqLen,

    /// RoPE base frequency (default: 100_000).
    pub rope_base: f32,
    /// QK norm scale factor (default: 1.2).
    pub qk_norm_scale: f32,
    /// Attention window pattern.
    pub window_pattern: WindowPattern,
    /// MLP activation type.
    pub mlp_activation: MlpActivation,
    /// Per-layer overrides (indexed by layer). Empty = all layers use global config.
    pub layer_overrides: Vec<LayerOverride>,
    /// Whether to tie embedding and output weights.
    pub tie_embeddings: bool,
    /// Logit soft-capping value (0.0 = disabled). Gemma uses 30.0.
    pub logit_cap: f32,
    /// Recurrent depth: apply the block stack this many times per token.
    /// 1 = standard transformer. >1 = LoopLM (weight-tied recurrence).
    /// Ouro uses 4 steps for both 1.4B and 2.6B models.
    pub recurrent_steps: usize,
    /// Whether this model has a learned exit gate for adaptive halting.
    pub has_exit_gate: bool,

    pub value_embed: ValueEmbedConfig,
    pub residual: ResidualConfig,
    pub smear: SmearConfig,
    pub precision: Precision,

    /// RMS norm epsilon.
    pub norm_eps: f32,
}

impl GptConfig {
    /// Validate the configuration. Called by the builder before constructing the model.
    pub fn validate(&self) -> Result<()> {
        // model_dim == num_heads * head_dim
        if self.model_dim.get() != self.num_heads.get() * self.head_dim.get() {
            return Err(TransformerError::DimInconsistency {
                model_dim: self.model_dim,
                num_heads: self.num_heads,
                head_dim: self.head_dim,
            });
        }

        // GQA: num_heads divisible by num_kv_heads
        if self.num_heads.get() % self.num_kv_heads.get() != 0 {
            return Err(TransformerError::GqaIndivisible {
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
            });
        }

        Ok(())
    }

    /// Number of query heads per KV head (GQA ratio).
    #[inline]
    pub fn gqa_ratio(&self) -> usize {
        self.num_heads.get() / self.num_kv_heads.get()
    }

    /// Compute sliding window size for a given layer.
    pub fn window_size(&self, layer: LayerIdx) -> Option<usize> {
        match &self.window_pattern {
            WindowPattern::Full => None,
            WindowPattern::Sssl => {
                // Every 4th layer (0-indexed: 3, 7, 11, ...) gets full attention
                if (layer.get() + 1) % 4 == 0 {
                    None
                } else {
                    let short = ((self.max_seq_len.get() / 4 + 127) / 128) * 128;
                    Some(short.max(128))
                }
            }
            WindowPattern::Custom(sizes) => {
                sizes.get(layer.get()).copied().and_then(|s| if s == 0 { None } else { Some(s) })
            }
        }
    }

    /// Whether layer `i` has value embeddings.
    /// Pattern: alternating, starting from the same parity as the last layer.
    pub fn has_value_embed(&self, layer: LayerIdx) -> bool {
        layer.get() % 2 == (self.num_layers.get() - 1) % 2
    }

    /// Get per-layer override (or default).
    pub fn layer_override(&self, layer: usize) -> &LayerOverride {
        static DEFAULT: LayerOverride = LayerOverride {
            head_dim: None, num_heads: None, num_kv_heads: None,
            rope_base: None, share_kv_from: None, has_input_gate: false,
        };
        self.layer_overrides.get(layer).unwrap_or(&DEFAULT)
    }

    /// Effective head_dim for a layer.
    pub fn layer_head_dim(&self, layer: usize) -> usize {
        self.layer_override(layer).head_dim.unwrap_or(self.head_dim.get())
    }

    /// Effective num_heads for a layer.
    pub fn layer_num_heads(&self, layer: usize) -> usize {
        self.layer_override(layer).num_heads.unwrap_or(self.num_heads.get())
    }

    /// Effective num_kv_heads for a layer.
    pub fn layer_num_kv_heads(&self, layer: usize) -> usize {
        self.layer_override(layer).num_kv_heads.unwrap_or(self.num_kv_heads.get())
    }

    /// Effective RoPE base for a layer.
    pub fn layer_rope_base(&self, layer: usize) -> f32 {
        self.layer_override(layer).rope_base.unwrap_or(self.rope_base)
    }

    /// KV dim for a layer (num_kv_heads * head_dim).
    pub fn layer_kv_dim(&self, layer: usize) -> usize {
        self.layer_num_kv_heads(layer) * self.layer_head_dim(layer)
    }

    /// Q dim for a layer (num_heads * head_dim).
    pub fn layer_q_dim(&self, layer: usize) -> usize {
        self.layer_num_heads(layer) * self.layer_head_dim(layer)
    }

    /// Which layer to source KV from (for KV sharing). Returns self if no sharing.
    pub fn kv_source_layer(&self, layer: usize) -> usize {
        self.layer_override(layer).share_kv_from.unwrap_or(layer)
    }

    /// Whether this layer uses SwiGLU activation.
    pub fn is_swiglu(&self) -> bool {
        matches!(self.mlp_activation, MlpActivation::SwiGlu)
    }
}

impl Default for GptConfig {
    fn default() -> Self {
        Self {
            model_dim: ModelDim::new(768),
            num_heads: NumHeads::new(12),
            num_kv_heads: NumKvHeads::new(12),
            head_dim: HeadDim::new(64),
            num_layers: NumLayers::new(12),
            vocab_size: VocabSize::new(50257),
            mlp_dim: MlpDim::new(3072),
            max_seq_len: SeqLen::new(1024),
            rope_base: 100_000.0,
            qk_norm_scale: 1.2,
            window_pattern: WindowPattern::Full,
            mlp_activation: MlpActivation::ReluSquared,
            layer_overrides: Vec::new(),
            tie_embeddings: false,
            logit_cap: 0.0,
            recurrent_steps: 1,
            has_exit_gate: false,
            value_embed: ValueEmbedConfig::default(),
            residual: ResidualConfig::default(),
            smear: SmearConfig::default(),
            precision: Precision::F32,
            norm_eps: 1e-5,
        }
    }
}
