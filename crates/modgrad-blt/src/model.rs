//! `BltModel` — top-level assembly of LocalEncoder + LatentTransformer +
//! LocalDecoder, with the `LanguageModel` impl (`isis_runtime`'s trait)
//! so existing `LmTrainer<M>` and `Sampler` compose unchanged.
//!
//! The Latent Transformer is `GptModelResident` (canonically — for
//! byte-ification, initialised from a pretrained Qwen2.5 backbone via
//! [`crate::byteify`]).
//!
//! Owner: agent sasha (with [`crate::encoder`] and [`crate::decoder`]).

#![allow(dead_code)]

/// Placeholder.
pub struct BltModel;
