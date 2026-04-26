//! Patch-aware cross-attention — the bridge between byte-level and
//! patch-level representations.
//!
//! Two flavours, mirroring BLT paper §3.2.2 (encoder) and §3.3.1
//! (decoder):
//!
//! - **Encoder cross-attn (`bytes → patches`):** queries are patch
//!   reps, keys/values are byte reps. Each patch query attends only
//!   to the bytes within that patch (per-patch mask).
//! - **Decoder cross-attn (`patches → bytes`):** queries are byte
//!   reps, keys/values are patch reps. Each byte attends to its
//!   containing patch (and patches before it under causal masking).
//!
//! No positional embeddings inside cross-attn (paper §3.2.2);
//! pre-LayerNorm on Q/K/V; multi-headed; residual around the block.
//!
//! Owner: agent noah. TODO: implement.

#![allow(dead_code)]

/// Placeholder so the crate compiles while the impl is in flight.
/// Replace with real types when the slice lands.
pub struct CrossAttention;
