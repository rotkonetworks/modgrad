//! Small byte-level autoregressive language model used to estimate
//! next-byte entropy `H(x_i)` for entropy-based patch boundaries.
//!
//! Per BLT paper §2.3 / §4.2: a transformer with ~100M params,
//! 14 layers, hidden 512, sliding-window attention 512 bytes. (We can
//! train smaller variants — paper §7 shows diminishing returns past
//! 50M params.)
//!
//! Architecture-wise this is just a regular small transformer over
//! byte vocab (256). We can reuse `GptModelResident` with a Qwen-class
//! `GptConfig` keyed for byte-level (`vocab=256`, small hidden,
//! `tie_embeddings=true`).
//!
//! Owner: agent leah (tightly coupled with `patcher`).

#![allow(dead_code)]

/// Placeholder.
pub struct EntropyModel;
