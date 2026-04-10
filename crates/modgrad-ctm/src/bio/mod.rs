//! Bio-inspired learning rules — reference implementations.
//!
//! These implement neuroscience-inspired learning as auxiliary signals:
//! - `cerebellar`: Delta rule forward model (predict observation from motor)
//! - `three_factor`: REINFORCE with eligibility traces + salience gating
//! - `neuromod`: Dopamine/serotonin/norepinephrine dynamics
//! - `salience`: RPE × motor conflict → learning rate gate
//!
//! Originally designed for the Hebbian system. Being adapted for use as
//! toggleable auxiliary losses alongside BPTT (see graph::AuxLossConfig).

// TODO: update imports to use SDK types
// pub mod cerebellar;
// pub mod three_factor;
// pub mod neuromod;
// pub mod salience;
