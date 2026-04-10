//! Fixed (standard) RoPE positioning.
//!
//! Positions are simply 0, 1, 2, ... Sequential.

use super::PositionEncoding;

/// Standard sequential positioning for RoPE.
pub struct FixedPositioning;

impl PositionEncoding for FixedPositioning {
    fn positions(&self, _hidden: &[f32], start_pos: usize, seq_len: usize) -> Vec<f32> {
        (0..seq_len).map(|t| (start_pos + t) as f32).collect()
    }

    fn position_one(&self, _hidden: &[f32], seq_pos: usize) -> f32 {
        seq_pos as f32
    }
}
