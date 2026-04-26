//! Local encoder: bytes → patch representations.
//!
//! Per BLT paper §3.2: a lightweight transformer with `lE` layers
//! (typically 1) where each layer is a transformer block followed by
//! a cross-attention block. The cross-attention pools the byte
//! representations within each patch into a patch query.
//!
//! Initial patch query: `pool(byte_embeds_in_patch)` → linear
//! projection (max-pooling per BLT §4.8).
//!
//! Owner: agent sasha (with [`crate::decoder`] and [`crate::model`]).

#![allow(dead_code)]

/// Placeholder.
pub struct LocalEncoder;
