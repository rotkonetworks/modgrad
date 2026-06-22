//! Memory primitives for CTM systems.
//!
//! - `hippocampus`: Content-addressable episodic memory (cosine retrieval, ring buffer)
//! - `replay`: Prioritized experience buffer (surprise-gated)
//! - `sleep`: Offline consolidation via least-squares (REM sleep analogue)

// Threading prelude shim: rayon on native, serial fallback on wasm32.
pub mod rayon_shim;

pub mod hippocampus;
pub mod episodic;
pub mod replay;
pub mod sleep;
