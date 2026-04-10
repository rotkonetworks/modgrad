//! Transformer error types.
//!
//! Caught at build() time where possible, not at inference time.

use super::dims::*;
use std::fmt;

#[derive(Debug)]
pub enum TransformerError {
    /// Weight tensor shape doesn't match config.
    ShapeMismatch {
        context: &'static str,
        expected: usize,
        actual: usize,
    },
    /// model_dim must equal num_heads * head_dim.
    DimInconsistency {
        model_dim: ModelDim,
        num_heads: NumHeads,
        head_dim: HeadDim,
    },
    /// num_heads must be divisible by num_kv_heads (GQA).
    GqaIndivisible {
        num_heads: NumHeads,
        num_kv_heads: NumKvHeads,
    },
    /// Sliding window size exceeds sequence length.
    WindowExceedsSeqLen {
        window: usize,
        seq_len: SeqLen,
    },
    /// Builder is missing a required field.
    BuilderMissing(&'static str),
    /// NaN or Inf detected in forward pass (debug builds only).
    NumericalInstability {
        context: &'static str,
    },
    /// Weight file I/O error.
    Io(std::io::Error),
}

impl fmt::Display for TransformerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { context, expected, actual } =>
                write!(f, "shape mismatch in {context}: expected {expected}, got {actual}"),
            Self::DimInconsistency { model_dim, num_heads, head_dim } =>
                write!(f, "model_dim ({model_dim}) != num_heads ({num_heads}) * head_dim ({head_dim})"),
            Self::GqaIndivisible { num_heads, num_kv_heads } =>
                write!(f, "num_heads ({num_heads}) not divisible by num_kv_heads ({num_kv_heads})"),
            Self::WindowExceedsSeqLen { window, seq_len } =>
                write!(f, "sliding window {window} exceeds seq_len {seq_len}"),
            Self::BuilderMissing(field) =>
                write!(f, "builder missing required field: {field}"),
            Self::NumericalInstability { context } =>
                write!(f, "NaN/Inf detected in {context}"),
            Self::Io(e) =>
                write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for TransformerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TransformerError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

pub type Result<T> = std::result::Result<T, TransformerError>;
