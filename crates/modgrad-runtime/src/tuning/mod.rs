//! Parameter governance: capability-based tuning with hot-reload.
//! Includes developmental capability escalation (puberty).

pub mod tuning;
pub mod puberty;

// Re-export main types at this level
pub use tuning::*;
