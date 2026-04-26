//! Offline byte-stream → patch-boundary segmenter.
//!
//! Per BLT paper §2.3, identifies patch boundaries from per-byte
//! entropies under a small byte-LM:
//!
//! - **Global threshold:** `H(x_t) > θ_g` starts a new patch.
//! - **Approx. monotonic:** `H(x_t) − H(x_{t-1}) > θ_r` starts a new
//!   patch (handles "entropy drift" in long contexts; reset entropy
//!   model context on newlines per §4.4).
//!
//! Boundaries are computed once per dataloading step, then the
//! tokenizer-equivalent step in BLT training is "look up which patch
//! this byte belongs to."
//!
//! Owner: agent leah (tightly coupled with `entropy`).

#![allow(dead_code)]

/// Placeholder.
pub struct EntropyPatcher;
