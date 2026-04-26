//! BLT-specific training loop wrapper.
//!
//! BLT training is conceptually identical to standard LM training —
//! same forward/CE/backward/AdamW — except:
//!   - inputs are byte sequences plus their precomputed patch
//!     boundaries (from [`crate::patcher`])
//!   - the optimizer can apply different LRs to local vs global
//!     parameters (paper §6.2: 1/10 LR on global when byte-ifying a
//!     pretrained backbone)
//!
//! The cleanest framing is "wrap `isis_runtime::LmTrainer<BltModel>`
//! with an LR-multiplier filter." This lives here rather than in
//! `isis-runtime` to keep the SDK / orchestration layering clean.
//!
//! Owner: agent ronan (with [`crate::byteify`]).

#![allow(dead_code)]

/// Placeholder.
pub struct BltTrainer;
