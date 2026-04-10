//! Position encoding strategies.
//!
//! Trait-based: PositionEncoding produces position values for each token,
//! which are then used by RoPE to rotate Q/K vectors.

pub mod fixed;
pub mod dynamic;

/// Position encoding trait.
///
/// Given a sequence of hidden states, produces a position value per token.
/// For standard RoPE: positions are 0, 1, 2, ...
/// For RePo: positions are learned from the hidden states.
pub trait PositionEncoding {
    /// Compute positions for a sequence.
    ///
    /// `hidden`: `[seq_len, model_dim]` row-major.
    /// `start_pos`: starting position offset.
    /// `seq_len`: number of tokens.
    ///
    /// Returns: position per token `[seq_len]`.
    fn positions(&self, hidden: &[f32], start_pos: usize, seq_len: usize) -> Vec<f32>;

    /// Compute position for a single token (decode mode).
    fn position_one(&self, hidden: &[f32], seq_pos: usize) -> f32;
}
