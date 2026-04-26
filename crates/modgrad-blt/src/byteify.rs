//! Byte-ification recipe — initialise a `BltModel`'s Latent Transformer
//! from a pretrained `GptModelResident` (Qwen2.5 / Llama).
//!
//! Per BLT paper §6.2: copy the global transformer weights from the
//! pretrained checkpoint, train the local encoder + decoder + cross-
//! attention from scratch, with 1/10 the LR applied to the global
//! parameters.
//!
//! For us, the pretrained checkpoint comes from
//! `modgrad_io::qwen2::load_qwen2_5_0_5b` — already loaded into a
//! `GptModelResident`. The byte-ification routine constructs a
//! `BltModel` whose Latent is *that exact resident model*, plumbing
//! the LR multiplier through the [`crate::trainer`] into the AdamW
//! step.
//!
//! Owner: agent ronan (with [`crate::trainer`]).

#![allow(dead_code)]

/// Placeholder.
pub struct ByteifyRecipe;
