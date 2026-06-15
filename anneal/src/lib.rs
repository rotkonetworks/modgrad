//! # Anneal
//!
//! A crowd slowly annealing a shared model out of noise.
//!
//! Anneal is a decentralized, verifiable training network. Contributors train
//! individual DiffusionBlocks blocks in-browser (WebGPU), submit them, and earn
//! when their block verifiably improves the shared model. The whole thing is
//! possible because of one structural fact: DiffusionBlocks blocks are
//! **independent** — each owns a noise range, trained by score matching, with
//! *zero inter-block communication*. That is the only known training scheme where
//! disconnected volunteer hardware can actually contribute, and it hands you a
//! second gift: each block's quality is independently, cheaply verifiable (its
//! denoising loss on a held-out shard), which is exactly what an honest reward
//! market needs.
//!
//! See `README.md` for the full architecture and — importantly — the build order.
//! The load-bearing assumption under the *entire* system is unproven and is NOT a
//! crypto question: **does val-gated async block training converge to a good
//! model?** Everything else is plumbing around that. So the code grows core-first.
//!
//! This crate is the distributed/verification layer. The ML engine it stands on
//! (the EDM noise schedule + equi-prob partitioning + the block-wise denoising
//! trainer) lives in [`modgrad_transformer`].

// Re-export the noise-side primitives so the network layer speaks the same units.
pub use modgrad_transformer::diffusion;

/// Slice 3 — the distributed-convergence experiment: does independent, per-noise-range
/// block training match joint training? (The claim Anneal rests on.)
pub mod convergence;
