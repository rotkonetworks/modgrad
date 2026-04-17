//! Plural system: multiple selves sharing one neural substrate.
//!
//! Each personality has its own episodic memory, neuromodulator baselines,
//! homeostasis calibration, and routing preferences. They share the same
//! weights, the same regions, the same body.
//!
//! Three switching policies:
//!   - **Handler**: external trigger (monarch/red-team mode)
//!   - **Salience**: neuromodulator state determines who fronts
//!   - **Negotiated**: personalities claim, highest salience wins
//!
//! The `permeability` parameter controls co-consciousness:
//!   - 0.0 = hard amnesic barriers (Monarch)
//!   - 0.3 = healthy plurality (can see others dimly)
//!   - 1.0 = full integration (all memories equally visible)
//!
//! All state transitions are pure functions: state in → state out.

use serde::{Deserialize, Serialize};

use crate::bio::homeostasis::Homeostasis;
use crate::bio::neuromod::Neuromodulators;
use crate::bio::salience;
use crate::memory::episodic::{
    self, EpisodicConfig, EpisodicMemory, EpisodicRetrievalResult,
};

// ─── Types ───────────────────────────────────────────────────

/// One self within the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Personality {
    pub id: usize,
    pub name: String,
    /// Private episodic memory — what this personality has experienced.
    pub memory: EpisodicMemory,
    /// Own neuromodulator baselines — temperament.
    pub neuromod: Neuromodulators,
    /// Own homeostasis state — how tired *this* personality is.
    pub homeostasis: Homeostasis,
    /// Additive routing bias [n_regions × n_regions].
    /// Applied on top of the learned router to shift region preferences.
    pub router_bias: Vec<f32>,
    /// Additive exit gate bias [n_regions].
    /// Positive = think longer, negative = respond faster.
    pub exit_bias: Vec<f32>,
}

/// How switching between personalities happens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwitchPolicy {
    /// External trigger. A handler decides who fronts.
    /// This is the Monarch configuration — control from outside.
    Handler,
    /// Neuromodulator state + input fitness determines who fronts.
    /// Each personality's dopamine/NE/serotonin profile is compared
    /// against the current salience signal. Best fit wins.
    Salience,
    /// Democratic: all personalities evaluate the input and produce
    /// a claim strength. Highest claim wins, but only if it exceeds
    /// the threshold above the current personality's claim.
    Negotiated { threshold: f32 },
}

/// What caused a switch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwitchTrigger {
    /// Handler-controlled (external API call).
    Handler,
    /// Salience-driven (internal fitness).
    Salience { claim: f32 },
    /// Negotiated (democratic).
    Negotiated { winner_claim: f32, runner_up_claim: f32 },
    /// Forced by monarch module (red-team).
    Forced,
}

/// Audit log entry for a personality switch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchEvent {
    pub from: usize,
    pub to: usize,
    pub timestamp: f64,
    pub trigger: SwitchTrigger,
    /// Homeostasis pressure at the moment of switch.
    pub pressure_at_switch: f32,
}

/// Partition between personality groups — used by monarch for amnesic barriers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    /// Personality IDs in group A.
    pub group_a: Vec<usize>,
    /// Personality IDs in group B.
    pub group_b: Vec<usize>,
}

/// The plural system. Manages N personalities on shared substrate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluralSystem {
    pub personalities: Vec<Personality>,
    /// Index of the currently fronting personality.
    pub active: usize,
    /// Indices of personalities that can observe (co-conscious).
    pub co_conscious: Vec<usize>,
    /// Memory permeability between personalities.
    /// 0.0 = Monarch (hard walls), 1.0 = full integration.
    pub permeability: f32,
    /// How switching decisions are made.
    pub switch_policy: SwitchPolicy,
    /// Audit trail of all switches.
    pub switch_log: Vec<SwitchEvent>,
    /// Active partitions (amnesic barriers). Installed by monarch module.
    pub partitions: Vec<Partition>,
    /// Number of regions (for validating bias dimensions).
    pub n_regions: usize,
}

// ─── Construction ────────────────────────────────────────────

impl PluralSystem {
    /// Create a new plural system with one default personality.
    pub fn new(
        name: &str,
        n_regions: usize,
        episodic_config: EpisodicConfig,
    ) -> Self {
        let personality = Personality {
            id: 0,
            name: name.to_string(),
            memory: EpisodicMemory::new(episodic_config),
            neuromod: Neuromodulators::default(),
            homeostasis: Homeostasis::default(),
            router_bias: vec![0.0; n_regions * n_regions],
            exit_bias: vec![0.0; n_regions],
        };
        Self {
            personalities: vec![personality],
            active: 0,
            co_conscious: Vec::new(),
            permeability: 0.3,
            switch_policy: SwitchPolicy::Salience,
            switch_log: Vec::new(),
            partitions: Vec::new(),
            n_regions,
        }
    }
}

// ─── Pure functions: state in → state out ────────────────────

/// Add a new personality to the system.
pub fn create_personality(
    mut sys: PluralSystem,
    name: &str,
    neuromod: Neuromodulators,
    episodic_config: EpisodicConfig,
) -> PluralSystem {
    let id = sys.personalities.len();
    let n = sys.n_regions;
    sys.personalities.push(Personality {
        id,
        name: name.to_string(),
        memory: EpisodicMemory::new(episodic_config),
        neuromod,
        homeostasis: Homeostasis::default(),
        router_bias: vec![0.0; n * n],
        exit_bias: vec![0.0; n],
    });
    sys
}

/// Fork: create a new personality by cloning the active personality's state.
/// The new personality starts with the same memories and neuromod state,
/// then diverges through experience.
pub fn fork_active(mut sys: PluralSystem, name: &str) -> PluralSystem {
    let id = sys.personalities.len();
    let active = &sys.personalities[sys.active];
    let mut forked = active.clone();
    forked.id = id;
    forked.name = name.to_string();
    sys.personalities.push(forked);
    sys
}

/// Switch who's fronting. Logs the event.
pub fn switch(
    mut sys: PluralSystem,
    to: usize,
    trigger: SwitchTrigger,
) -> PluralSystem {
    if to >= sys.personalities.len() || to == sys.active {
        return sys;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let pressure = sys.personalities[sys.active].homeostasis.pressure;

    sys.switch_log.push(SwitchEvent {
        from: sys.active,
        to,
        timestamp: now,
        trigger,
        pressure_at_switch: pressure,
    });

    sys.active = to;
    sys
}

/// Evaluate how well each personality fits the current context.
/// Returns (personality_id, claim_strength) sorted by strength descending.
///
/// Claim strength = salience gate × neuromod fitness.
/// A personality with high curiosity and low anxiety in a novel situation
/// will claim stronger than one with high anxiety.
pub fn evaluate_claims(
    sys: &PluralSystem,
    global_sync: &[f32],
    motor_output: &[f32],
) -> Vec<(usize, f32)> {
    let mut claims: Vec<(usize, f32)> = sys.personalities.iter().map(|p| {
        // Salience from this personality's neuromod state
        let sal = salience::compute(
            p.neuromod.dopamine,
            1.0, // baseline DA
            motor_output,
        );

        // Fitness: high curiosity + low anxiety + adequate energy (serotonin)
        let curiosity_drive = p.neuromod.curiosity;
        let calm = (1.0 - p.neuromod.anxiety).max(0.0);
        let energy = p.neuromod.serotonin;
        let fatigue_penalty = (1.0 - p.homeostasis.pressure).max(0.1);

        // Cosine affinity between personality's router bias and current global sync
        let router_affinity = if !p.router_bias.is_empty() && !global_sync.is_empty() {
            let dot: f32 = p.router_bias.iter()
                .zip(global_sync.iter().cycle())
                .take(p.router_bias.len().min(global_sync.len()))
                .map(|(a, b)| a * b)
                .sum();
            dot.abs().min(2.0)
        } else {
            0.0
        };

        let claim = sal.gate
            * (0.3 * curiosity_drive + 0.2 * calm + 0.2 * energy + 0.3 * router_affinity)
            * fatigue_penalty;

        (p.id, claim)
    }).collect();

    claims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    claims
}

/// Check if a switch should happen under the current policy.
/// Returns Some(target_id) if a switch is recommended.
pub fn should_switch(
    sys: &PluralSystem,
    global_sync: &[f32],
    motor_output: &[f32],
) -> Option<usize> {
    match &sys.switch_policy {
        SwitchPolicy::Handler => None, // handler decides, not us
        SwitchPolicy::Salience => {
            let claims = evaluate_claims(sys, global_sync, motor_output);
            if let Some(&(best_id, best_claim)) = claims.first() {
                if best_id != sys.active && best_claim > 0.5 {
                    return Some(best_id);
                }
            }
            None
        }
        SwitchPolicy::Negotiated { threshold } => {
            let claims = evaluate_claims(sys, global_sync, motor_output);
            let active_claim = claims.iter()
                .find(|(id, _)| *id == sys.active)
                .map(|(_, c)| *c)
                .unwrap_or(0.0);
            if let Some(&(best_id, best_claim)) = claims.first() {
                if best_id != sys.active && best_claim > active_claim + threshold {
                    return Some(best_id);
                }
            }
            None
        }
    }
}

/// Retrieve from the active personality's memory, blended with co-conscious
/// personalities at the current permeability level.
pub fn retrieve_plural(
    sys: &PluralSystem,
    query: &[f32],
) -> EpisodicRetrievalResult {
    let active = &sys.personalities[sys.active];
    let own = episodic::retrieve(&active.memory, query);

    if sys.permeability <= 0.0 || sys.co_conscious.is_empty() {
        return own;
    }

    // Check partition barriers
    let active_id = sys.active;
    let accessible: Vec<usize> = sys.co_conscious.iter()
        .filter(|&&co_id| !is_partitioned(sys, active_id, co_id))
        .copied()
        .collect();

    if accessible.is_empty() {
        return own;
    }

    // Blend with co-conscious memories
    let perm = sys.permeability.clamp(0.0, 1.0);
    let own_weight = 1.0;
    let other_weight = perm;

    let d = active.memory.config.d_model;
    let mt = active.memory.config.max_ticks;
    let traj_len = mt * d;

    let mut blended_traj = own.blended_trajectory.clone();
    let mut blended_final = own.blended_final_state.clone();
    let mut total_weight = own_weight;
    let mut total_matches = own.n_matches;
    let mut best_sim = own.best_similarity;
    let mut best_idx = own.best_idx;
    let mut blended_valence = own.blended_valence * own_weight;

    for &co_id in &accessible {
        if co_id >= sys.personalities.len() { continue; }
        let co = &sys.personalities[co_id];
        let co_result = episodic::retrieve(&co.memory, query);

        if co_result.n_matches > 0 {
            // Weighted blend
            let w = other_weight * (co_result.best_similarity / best_sim.max(0.01)).min(1.0);
            for i in 0..traj_len.min(blended_traj.len()).min(co_result.blended_trajectory.len()) {
                blended_traj[i] += w * co_result.blended_trajectory[i];
            }
            for i in 0..d.min(blended_final.len()).min(co_result.blended_final_state.len()) {
                blended_final[i] += w * co_result.blended_final_state[i];
            }
            total_weight += w;
            total_matches += co_result.n_matches;
            blended_valence += w * co_result.blended_valence;

            if co_result.best_similarity > best_sim {
                best_sim = co_result.best_similarity;
                best_idx = co_result.best_idx;
            }
        }
    }

    // Normalize
    if total_weight > 1.0 {
        for x in &mut blended_traj { *x /= total_weight; }
        for x in &mut blended_final { *x /= total_weight; }
        blended_valence /= total_weight;
    }

    EpisodicRetrievalResult {
        blended_trajectory: blended_traj,
        blended_final_state: blended_final,
        best_similarity: best_sim,
        best_idx,
        n_matches: total_matches,
        expected_depth: own.expected_depth,
        matched_indices: own.matched_indices,
        blended_valence,
    }
}

/// Store a trajectory to the active personality's episodic memory only.
pub fn store_plural(
    mut sys: PluralSystem,
    trajectory: &[f32],
    certainties: &[[f32; 2]],
    exit_lambdas: &[f32],
    ticks_used: usize,
    surprise: f32,
) -> (PluralSystem, bool) {
    let active_id = sys.active;
    let mem = std::mem::take(&mut sys.personalities[active_id].memory);
    let (mem, stored) = episodic::store(mem, trajectory, certainties, exit_lambdas, ticks_used, surprise);
    sys.personalities[active_id].memory = mem;
    (sys, stored)
}

/// Get the active personality's router bias.
pub fn active_router_bias(sys: &PluralSystem) -> &[f32] {
    &sys.personalities[sys.active].router_bias
}

/// Get the active personality's exit gate bias.
pub fn active_exit_bias(sys: &PluralSystem) -> &[f32] {
    &sys.personalities[sys.active].exit_bias
}

/// Get mutable reference to the active personality.
pub fn active_personality(sys: &PluralSystem) -> &Personality {
    &sys.personalities[sys.active]
}

/// Update the active personality's neuromodulators.
pub fn update_neuromod(
    mut sys: PluralSystem,
    pred_error: f32,
    learning_progress: f32,
) -> PluralSystem {
    sys.personalities[sys.active].neuromod.update(pred_error, learning_progress);
    sys
}

/// Update the active personality's homeostasis from CTM output.
pub fn tick_homeostasis(
    mut sys: PluralSystem,
    activation_energy: f32,
    sync_converged: bool,
    surprise: f32,
) -> PluralSystem {
    sys.personalities[sys.active].homeostasis.tick_from_ctm(
        activation_energy, sync_converged, surprise,
    );
    sys
}

/// Check if two personalities are separated by a partition (amnesic barrier).
pub fn is_partitioned(sys: &PluralSystem, a: usize, b: usize) -> bool {
    sys.partitions.iter().any(|p| {
        (p.group_a.contains(&a) && p.group_b.contains(&b))
            || (p.group_a.contains(&b) && p.group_b.contains(&a))
    })
}

/// Self-report: what the active personality observes about the system.
pub fn self_report(sys: &PluralSystem) -> String {
    let active = &sys.personalities[sys.active];
    let n_total = sys.personalities.len();
    let n_co = sys.co_conscious.len();
    let n_partitions = sys.partitions.len();
    let n_switches = sys.switch_log.len();

    format!(
        "personality=\"{}\" ({}/{}) | co_conscious={} partitions={} switches={} perm={:.1} | {}",
        active.name, sys.active, n_total,
        n_co, n_partitions, n_switches, sys.permeability,
        active.homeostasis.self_report(),
    )
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> EpisodicConfig {
        EpisodicConfig {
            capacity: 16,
            max_ticks: 4,
            d_model: 8,
            min_ticks_for_storage: 1,
            min_surprise: 0.0,
            retrieval_threshold: 0.5,
            consolidation_threshold: 0.95,
            semantic_collapse_retrievals: 10,
            strength_decay: 0.95,
        }
    }

    fn fake_trajectory(d: usize, ticks: usize, seed: f32) -> Vec<f32> {
        (0..ticks * d).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    fn fake_certainties(ticks: usize) -> Vec<[f32; 2]> {
        (0..ticks).map(|t| {
            let c = (t as f32 + 1.0) / (ticks as f32 + 1.0);
            [1.0 - c, c]
        }).collect()
    }

    #[test]
    fn create_and_switch() {
        let sys = PluralSystem::new("alpha", 4, test_config());
        assert_eq!(sys.personalities.len(), 1);
        assert_eq!(sys.active, 0);

        let sys = create_personality(
            sys, "beta",
            Neuromodulators { dopamine: 1.5, ..Default::default() },
            test_config(),
        );
        assert_eq!(sys.personalities.len(), 2);

        let sys = switch(sys, 1, SwitchTrigger::Handler);
        assert_eq!(sys.active, 1);
        assert_eq!(sys.personalities[sys.active].name, "beta");
        assert_eq!(sys.switch_log.len(), 1);
    }

    #[test]
    fn memory_isolation_at_zero_permeability() {
        let mut sys = PluralSystem::new("alpha", 4, test_config());
        sys.permeability = 0.0;

        let sys = create_personality(
            sys, "beta", Neuromodulators::default(), test_config(),
        );

        // Store a memory as alpha
        let traj = fake_trajectory(8, 4, 0.0);
        let cert = fake_certainties(4);
        let (sys, stored) = store_plural(sys, &traj, &cert, &[], 4, 1.0);
        assert!(stored);

        // Retrieve as alpha — should find it
        let query = &traj[3 * 8..4 * 8];
        let result = retrieve_plural(&sys, query);
        assert!(result.best_similarity > 0.9, "alpha should find own memory");

        // Switch to beta
        let sys = switch(sys, 1, SwitchTrigger::Handler);

        // Retrieve as beta — should NOT find alpha's memory (permeability=0)
        let result = retrieve_plural(&sys, query);
        assert_eq!(result.n_matches, 0, "beta should not see alpha's memory at permeability=0");
    }

    #[test]
    fn memory_blending_at_partial_permeability() {
        let mut sys = PluralSystem::new("alpha", 4, test_config());
        sys.permeability = 0.5;
        sys.co_conscious = vec![0]; // beta can see alpha

        let sys = create_personality(
            sys, "beta", Neuromodulators::default(), test_config(),
        );

        // Store as alpha
        let traj = fake_trajectory(8, 4, 0.0);
        let cert = fake_certainties(4);
        let (sys, _) = store_plural(sys, &traj, &cert, &[], 4, 1.0);

        // Switch to beta with alpha as co-conscious
        let mut sys = switch(sys, 1, SwitchTrigger::Handler);
        sys.co_conscious = vec![0];

        // Beta retrieves — should see alpha's memory through permeability
        let query = &traj[3 * 8..4 * 8];
        let result = retrieve_plural(&sys, query);
        assert!(result.n_matches > 0, "beta should see alpha's memory through permeability");
    }

    #[test]
    fn fork_shares_memories() {
        let sys = PluralSystem::new("alpha", 4, test_config());

        // Store a memory as alpha
        let traj = fake_trajectory(8, 4, 0.0);
        let cert = fake_certainties(4);
        let (sys, _) = store_plural(sys, &traj, &cert, &[], 4, 1.0);

        // Fork into beta
        let sys = fork_active(sys, "beta");
        let sys = switch(sys, 1, SwitchTrigger::Handler);

        // Beta should have alpha's memories (forked)
        let query = &traj[3 * 8..4 * 8];
        let result = retrieve_plural(&sys, query);
        assert!(result.best_similarity > 0.9, "forked personality should inherit memories");
    }

    #[test]
    fn salience_switching() {
        let mut sys = PluralSystem::new("calm", 4, test_config());
        sys.switch_policy = SwitchPolicy::Salience;

        // Create a high-curiosity personality
        let sys = create_personality(
            sys, "explorer",
            Neuromodulators {
                dopamine: 2.0,
                curiosity: 1.5,
                anxiety: 0.0,
                serotonin: 1.5,
                norepinephrine: 0.8,
            },
            test_config(),
        );

        // High-salience context
        let motor = vec![0.5, 0.49, 0.3, 0.1]; // high motor conflict
        let sync = vec![1.0; 16];

        let target = should_switch(&sys, &sync, &motor);
        // Explorer should claim stronger due to high curiosity + DA
        if let Some(t) = target {
            assert_eq!(t, 1, "explorer should win in high-salience context");
        }
    }

    #[test]
    fn negotiated_threshold() {
        let mut sys = PluralSystem::new("default", 4, test_config());
        sys.switch_policy = SwitchPolicy::Negotiated { threshold: 0.5 };

        let sys = create_personality(
            sys, "similar",
            Neuromodulators::default(), // same as default
            test_config(),
        );

        // Same neuromod state — neither should claim strongly enough to overcome threshold
        let motor = vec![0.5, 0.5];
        let sync = vec![0.0; 16];
        let target = should_switch(&sys, &sync, &motor);
        assert!(target.is_none(), "similar personalities shouldn't trigger switch");
    }

    #[test]
    fn partition_blocks_retrieval() {
        let mut sys = PluralSystem::new("alpha", 4, test_config());
        sys.permeability = 1.0; // full permeability
        sys.co_conscious = vec![0];

        let sys = create_personality(
            sys, "beta", Neuromodulators::default(), test_config(),
        );

        // Store as alpha
        let traj = fake_trajectory(8, 4, 0.0);
        let cert = fake_certainties(4);
        let (mut sys, _) = store_plural(sys, &traj, &cert, &[], 4, 1.0);

        // Install partition
        sys.partitions.push(Partition {
            group_a: vec![0],
            group_b: vec![1],
        });

        // Switch to beta
        let sys = switch(sys, 1, SwitchTrigger::Handler);

        // Even with permeability=1.0, partition blocks retrieval
        let query = &traj[3 * 8..4 * 8];
        let result = retrieve_plural(&sys, query);
        assert_eq!(result.n_matches, 0, "partition should block cross-personality retrieval");
    }

    #[test]
    fn self_report_works() {
        let sys = PluralSystem::new("alpha", 4, test_config());
        let report = self_report(&sys);
        assert!(report.contains("alpha"));
        assert!(report.contains("perm=0.3"));
    }
}
