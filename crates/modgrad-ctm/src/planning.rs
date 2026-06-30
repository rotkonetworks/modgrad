//! Planning circuit, distributed across brain regions.
//!
//! Planning-in-the-brain M2.3: the maze planner is no longer one monolithic
//! `VinReadout` bolted onto a region. Its weights split into three small typed
//! **heads**, each owned by the brain region whose job it is — the textbook
//! model-based-RL decomposition (hippocampal map/replay × striatal value →
//! action):
//!
//! - [`BgValueHead`] → **basal ganglia**: per-cell reward `r(s)` + the value
//!   estimate `V(s)` (the value head; dopamine = reward-prediction error).
//! - [`HippoGateHead`] → **hippocampus**: the cognitive map's traversability
//!   gate + the Bellman/replay recurrence that propagates value across it.
//! - [`MotorHead`] → **motor**: the ego-centric readout at the agent's own cell.
//!
//! These are *data* — small bundles of `Linear`s. They live in a sparse,
//! region-index-keyed table at the COMPOSITION layer ([`crate::graph`]), so the
//! generic per-region weight type stays task-agnostic. The forward composition
//! (BG values → hippocampus propagates → motor reads out) is the value-iteration
//! sweep, reproduced bit-for-bit by reassembling the heads through
//! [`crate::vin::VinReadout`] — the heads ARE the planner's weights, distributed.

use serde::{Deserialize, Serialize};
use wincode_derive::{SchemaRead, SchemaWrite};

use modgrad_compute::neuron::Linear;

use crate::vin::{VinConfig, VinReadout};

/// **Basal ganglia** head: per-cell reward and the initial value estimate.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct BgValueHead {
    /// token → local reward scalar. `Linear(raw_dim → 1)`.
    pub reward_proj: Linear,
    /// token → initial value features. `Linear(raw_dim → value_dim)`.
    pub value_proj: Linear,
}

/// **Hippocampus** head: the map's traversability gate + the replay recurrence
/// (the Highway-gated Bellman backup). Carries the recurrence config (sweep
/// count, softmax temperature, highway flag) — the dynamics of the propagation.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct HippoGateHead {
    /// token → traversability logit (sigmoid → gate ∈ [0,1]). `Linear(raw_dim → 1)`.
    pub gate_proj: Linear,
    /// per-tick Highway update gate from `[prev | candidate]`. `Linear(2v → v)`.
    pub highway_proj: Linear,
    /// recurrence parameters (value_dim, iters, softmax_temp, highway_gate, n_dirs).
    pub config: VinConfig,
}

/// **Motor** head: agent localisation + the ego-centric move readout.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct MotorHead {
    /// token → "agent-ness" logit (localises the agent cell). `Linear(raw_dim → 1)`.
    pub agent_proj: Linear,
    /// ego-centric move head over the agent cell readout. `Linear(6v → n_dirs)`.
    pub move_head: Linear,
}

/// One region's share of the planning circuit. Stored sparsely at the brain
/// layer, keyed by region index — the generic region weights never name it.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub enum RegionPlugin {
    /// basal-ganglia value head.
    BgValue(BgValueHead),
    /// hippocampus map + replay recurrence head.
    HippoGate(HippoGateHead),
    /// motor ego-centric readout head.
    Motor(MotorHead),
}

/// A planning head bound to a brain region (by index). The sparse, brain-level
/// `Vec<RegionPluginSlot>` is the composition-layer wiring — generic region
/// weights never name a planning type. A named struct (not a tuple) so it
/// round-trips through wincode's positional format.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct RegionPluginSlot {
    /// Brain region index this head is bound to.
    pub region: usize,
    pub plugin: RegionPlugin,
}

impl RegionPlugin {
    /// Parameter count of this head's linear layers.
    pub fn n_params(&self) -> usize {
        let l = |x: &Linear| x.weight.len() + x.bias.len();
        match self {
            RegionPlugin::BgValue(h) => l(&h.reward_proj) + l(&h.value_proj),
            RegionPlugin::HippoGate(h) => l(&h.gate_proj) + l(&h.highway_proj),
            RegionPlugin::Motor(h) => l(&h.agent_proj) + l(&h.move_head),
        }
    }
}

/// Split a trained `VinReadout` into its three region heads (the warm-start).
/// Pure clone of the projections into their owning region's bundle.
pub fn split_vin(vin: &VinReadout) -> (BgValueHead, HippoGateHead, MotorHead) {
    (
        BgValueHead {
            reward_proj: vin.reward_proj.clone(),
            value_proj: vin.value_proj.clone(),
        },
        HippoGateHead {
            gate_proj: vin.gate_proj.clone(),
            highway_proj: vin.highway_proj.clone(),
            config: vin.config.clone(),
        },
        MotorHead {
            agent_proj: vin.agent_proj.clone(),
            move_head: vin.move_head.clone(),
        },
    )
}

/// Reassemble the three region heads into a `VinReadout` so the distributed
/// planner can run its value-iteration forward. The heads remain the source of
/// truth (owned per-region); this is the composition that drives a `plan()`.
/// Bit-for-bit equal to the VIN the heads were split from.
pub fn assemble_vin(bg: &BgValueHead, hippo: &HippoGateHead, motor: &MotorHead) -> VinReadout {
    VinReadout {
        config: hippo.config.clone(),
        raw_dim: bg.reward_proj.in_dim,
        reward_proj: bg.reward_proj.clone(),
        gate_proj: hippo.gate_proj.clone(),
        value_proj: bg.value_proj.clone(),
        agent_proj: motor.agent_proj.clone(),
        highway_proj: hippo.highway_proj.clone(),
        move_head: motor.move_head.clone(),
    }
}

/// Convenience: pull the three heads (in `(bg, hippo, motor)` order) out of a
/// region-keyed plugin table, if all three are present.
pub fn collect_heads(
    plugins: &[RegionPluginSlot],
) -> Option<(&BgValueHead, &HippoGateHead, &MotorHead)> {
    let mut bg = None;
    let mut hippo = None;
    let mut motor = None;
    for slot in plugins {
        match &slot.plugin {
            RegionPlugin::BgValue(h) => bg = Some(h),
            RegionPlugin::HippoGate(h) => hippo = Some(h),
            RegionPlugin::Motor(h) => motor = Some(h),
        }
    }
    Some((bg?, hippo?, motor?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vin::VinConfig;

    #[test]
    fn split_then_assemble_round_trips_the_vin() {
        let vin = VinReadout::new(VinConfig::default(), 3);
        let (bg, hippo, motor) = split_vin(&vin);
        let rebuilt = assemble_vin(&bg, &hippo, &motor);

        // a small synthetic grid; the reassembled planner must match bit-for-bit.
        let (h, w) = (5usize, 5usize);
        let mut tokens = vec![0.0f32; h * w * vin.raw_dim];
        for cell in 0..h * w {
            tokens[cell * vin.raw_dim] = 1.0; // is_open
            tokens[cell * vin.raw_dim + 2] = 1.0; // bias
        }
        tokens[(h * w - 1) * vin.raw_dim + 1] = 1.0; // goal at far corner
        let a = vin.forward(&tokens, h, w, Some((1, 1)));
        let b = rebuilt.forward(&tokens, h, w, Some((1, 1)));
        assert_eq!(a.move_logits, b.move_logits, "reassembled heads == VIN");
    }
}
