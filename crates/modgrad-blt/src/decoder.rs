//! Local decoder: patches → bytes.
//!
//! Per BLT paper §3.3: a lightweight transformer with `lD` layers
//! (typically 9 for the 8B-class config) where each layer is a
//! cross-attention block followed by a transformer block. The
//! cross-attention reads byte queries and patch keys/values, with
//! roles inverted from the encoder.
//!
//! Final byte transformer layer feeds an LM head (vocab = 256).
//!
//! Owner: agent sasha (with [`crate::encoder`] and [`crate::model`]).

#![allow(dead_code)]

/// Placeholder.
pub struct LocalDecoder;
