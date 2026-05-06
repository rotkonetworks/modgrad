//! Penumbra arena — multi-agent CTM market maker for UM/USDC.
//!
//! The brain (8-region `RegionalWeightsTyped<D>`) emits a 12-dim
//! MOTOR vector that decodes into a barbell LP layout: tight inner
//! rung at low fee for throughput + wider outer rung at higher fee
//! for adverse-selection profit. Inventory imbalance and recent
//! fills enter through HIPPOCAMPUS / INPUT regions.
//!
//! Layered design:
//!   - [`chain`]   — `PenumbraChain` trait. v0 ships [`chain::PcliChain`]
//!                   (subprocess pcli, parses stdout). v1 will add
//!                   `DirectChain` once the penumbra Rust workspace is
//!                   wired in.
//!   - [`features`] — block state + own state → observation `Vec<f32>`.
//!   - [`motor`]   — 12-dim brain output → list of (price, size, fee)
//!                   position requests.
//!   - [`agent`]   — single MM agent driven by the typed brain.
//!   - [`reward`]  — fee_capture − adverse_selection − rebalance_gas.
//!   - [`arena`]   — population + PBT (multi-agent orchestrator).

pub mod chain;
pub mod features;
pub mod motor;
pub mod agent;
pub mod reward;
pub mod arena;
pub mod replay;
pub mod embedded;
pub mod hybrid;
pub mod checkpoint;
pub mod viz;

use modgrad_ctm::graph::RegionalConfig;

/// Default 8-region MM brain config — a single source of truth used by
/// every binary in this crate. Wired into `eight_region_mm` with the
/// encoder's [`features::RAW_OBS_DIM`] and the motor's
/// [`motor::MOTOR_DIM`].
pub fn default_mm_cfg() -> RegionalConfig {
    RegionalConfig::eight_region_mm(features::RAW_OBS_DIM, motor::MOTOR_DIM, 1)
}
