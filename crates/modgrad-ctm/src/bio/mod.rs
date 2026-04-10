//! Bio-inspired learning primitives.
//!
//! Standalone pure functions and structs — no runtime coupling.
//! Use these as auxiliary signals alongside BPTT, or on their own.
//!
//! - `cerebellar`: Delta rule forward model + dopamine dynamics
//! - `salience`: RPE × motor conflict → learning rate gate
//! - `neuromod`: Dopamine/serotonin/norepinephrine state machine
//! - `three_factor`: REINFORCE with Titans-style eligibility traces

pub mod cerebellar;
pub mod salience;
pub mod neuromod;
pub mod three_factor;
