//! Memory primitives for CTM systems.
//!
//! - `hippocampus`: Content-addressable episodic memory (cosine retrieval, ring buffer)
//! - `replay`: Prioritized experience buffer (surprise-gated)
//! - `sleep`: Offline consolidation via least-squares (REM sleep analogue)
//!
//! These are independent of the CTM graph — usable in any runtime.

// TODO: update imports to use SDK types (currently references old runtime types)
// pub mod hippocampus;
// pub mod replay;
// pub mod sleep;
