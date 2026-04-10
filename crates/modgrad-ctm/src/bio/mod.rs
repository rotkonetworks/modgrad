//! Bio-inspired primitives: learning, monitoring, and consolidation.
//!
//! Active (compiling, tested):
//!   - `cerebellar`: Delta rule forward model + dopamine dynamics
//!   - `salience`: RPE × motor conflict → learning rate gate
//!   - `neuromod`: Dopamine/serotonin/norepinephrine state machine
//!   - `three_factor`: REINFORCE with Titans-style eligibility traces
//!
//! Preserved (need import updates to compile):
//!   - `autonomic`: Emotional health monitoring (PTSD/depression/hate risk)
//!   - `homeostasis`: Self-monitoring, sleep pressure signals
//!   - `consolidation`: SPSA spindle-ripple weight optimization

pub mod cerebellar;
pub mod salience;
pub mod neuromod;
pub mod three_factor;

// Preserved — uncomment when imports are updated
// pub mod autonomic;
// pub mod homeostasis;
// pub mod consolidation;
