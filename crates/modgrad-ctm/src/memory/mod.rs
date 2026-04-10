//! Memory primitives for CTM systems.
//!
//! - `hippocampus`: Content-addressable episodic memory (cosine retrieval, ring buffer)
//! - `replay`: Prioritized experience buffer (surprise-gated)
//! - `sleep`: Offline consolidation via least-squares (REM sleep analogue)

pub mod hippocampus;
pub mod replay;
pub mod sleep;
