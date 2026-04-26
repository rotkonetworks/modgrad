//! Byte Latent Transformer — hierarchical byte-level model with
//! entropy-based dynamic patching.
//!
//! Reference: Pagnoni et al., "Byte Latent Transformer: Patches Scale
//! Better Than Tokens", 2024 (`https://arxiv.org/abs/2412.09871`).
//!
//! ## Architecture
//!
//! ```text
//!   bytes
//!     ↓
//!   Local Encoder       (small transformer + per-layer cross-attn)
//!     ↓ patches
//!   Latent Transformer  (the heavy "global" model — for our
//!                        byte-ification recipe, initialised from a
//!                        pretrained Qwen2.5 / Llama backbone)
//!     ↓
//!   Local Decoder       (cross-attn before each transformer layer)
//!     ↓
//!   bytes
//! ```
//!
//! ## Module map
//!
//! - [`cross_attn`] — patch-aware cross-attention (byte↔patch).
//! - Hash n-gram embeddings — re-exported from
//!   [`modgrad_codec::ngram_hash::HashNgramEmbeddings`]; the implementation
//!   already lives there with the rolling polynomial hash and is shared
//!   with other byte-level paths in the SDK.
//! - [`entropy`]    — small byte-LM that estimates next-byte entropy.
//! - [`patcher`]    — offline byte-stream → patch-boundary segmenter.
//! - [`encoder`]    — local encoder (bytes → patches).
//! - [`decoder`]    — local decoder (patches → bytes).
//! - [`model`]      — `BltModel` assembling encoder + latent + decoder.
//! - [`trainer`]    — BLT-specific training loop wrapper.
//! - [`byteify`]    — recipe for byte-ifying a pretrained model:
//!                    init Latent from `GptModelResident`, low-LR on
//!                    global, full-LR on local components.

pub mod cross_attn;
pub mod entropy;
pub mod patcher;
pub mod encoder;
pub mod decoder;
pub mod model;
pub mod trainer;
pub mod byteify;

/// Re-export of the BLT hash n-gram embedding (see paper §3.2.1). The
/// implementation lives in `modgrad-codec` because it is byte-level
/// utility shared with non-BLT paths; we re-export here so BLT-specific
/// callers don't reach across the SDK.
pub use modgrad_codec::ngram_hash::NgramHashEmbeddings;
