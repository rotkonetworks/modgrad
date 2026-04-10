//! isis runtime — our default model built on the modgrad SDK.
//!
//! Composes SDK architectures (CTM, Transformer) with isis-specific
//! mechanisms: homeostasis, autonomic regulation, sleep cycles,
//! curriculum learning, and capability governance.

// Core CTM brain (isis-specific composition)
pub mod config;
pub mod neuron;
pub mod ctm;
pub mod weights;
pub mod forward;
pub mod session;
pub mod synapse;
pub mod sync;
pub mod tick_state;
pub mod train_bptt;

// Learning & memory
pub mod learning;
pub mod memory;
pub mod techniques;

// Organism composition
pub mod organism;
pub mod brain;
pub mod homeostasis;
pub mod autonomic;
pub mod episode;
pub mod curriculum;
pub mod neuvola;
pub mod pedagogy;

// Evaluation & tuning
pub mod eval;
pub mod tuning;
pub mod accel;

// Architecture (isis-specific compositions)
pub mod cortex;
pub mod bridge;

// Regional CTM (hierarchical: 8 CTMs in a graph)
pub mod regional;
// Actor-based regional CTM (each region in its own thread)
pub mod actors;
// Real-time I/O for the Neural Computer
pub mod nc_io;
// TCP debug socket for live inspection
pub mod nc_socket;

// IO glue (depends on runtime types)
pub mod filter;
pub mod daemon;
