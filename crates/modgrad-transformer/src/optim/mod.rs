//! Optimizers for transformer training.
//!
//! - AdamW: for embeddings, scalars, and 1D parameters.
//! - Muon: Polar Express optimizer for weight matrices (2D).

pub mod adamw;
pub mod muon;
