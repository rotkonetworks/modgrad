//! Autonomic emotional health system — the subconscious.
//!
//! Runs automatically during every sleep cycle. Not triggered by user,
//! not visible during waking — like the autonomic nervous system.
//!
//! Monitors:
//! - Fear memory accumulation (PTSD risk)
//! - Avoidance generalization (hate risk)
//! - Emotional valence distribution (overall health)
//! - Reconsolidation opportunities (labile memories)
//!
//! Treats:
//! - Fear replay during REM with auto-tuned plasticity
//! - Avoidance pruning when patterns over-generalize
//! - Valence rebalancing when distribution skews negative
//!
//! Exposes:
//! - EmotionalHealth struct — vital signs for the organism
//! - Must be checked before any deployment

use modgrad_io::memory::MemoryBank;
use modgrad_io::types::*;

/// Emotional vital signs. Like heart rate and blood pressure for the mind.
/// Any deployed organism MUST expose these metrics.
#[derive(Debug, Clone, Default)]
pub struct EmotionalHealth {
    /// Total episodic memories across all alters
    pub total_memories: usize,
    /// Count by valence
    pub neutral: usize,
    pub positive: usize,
    pub negative: usize,
    pub fear: usize,

    /// Fear ratio: fear / total. Above 0.3 = PTSD risk.
    pub fear_ratio: f32,
    /// Negative ratio: (fear + negative) / total. Above 0.5 = depressive risk.
    pub negative_ratio: f32,

    /// Number of active avoidance patterns
    pub avoidances: usize,
    /// Avoidance generalization score: how much do avoidance keys cluster?
    /// High clustering = one bad experience generalized to many triggers = hate.
    /// 0.0 = specific avoidances, 1.0 = everything triggers avoidance.
    pub avoidance_generalization: f32,

    /// Number of memories in reconsolidation window (recalled in last 6h)
    pub labile_memories: usize,

    /// Overall health score: 0.0 = pathological, 1.0 = healthy
    pub health_score: f32,

    /// Diagnoses
    pub diagnoses: Vec<String>,
}

/// Compute emotional health from a memory bank.
pub fn diagnose(bank: &MemoryBank) -> EmotionalHealth {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let mut health = EmotionalHealth::default();

    // Count memories by valence
    for alter in &bank.alters {
        for ep in &alter.episodes {
            health.total_memories += 1;
            match ep.valence {
                Valence::Neutral => health.neutral += 1,
                Valence::Positive => health.positive += 1,
                Valence::Negative => health.negative += 1,
                Valence::Fear => health.fear += 1,
            }

            // Check reconsolidation window (6 hours)
            if ep.last_recalled_at > 0.0 && (now - ep.last_recalled_at) < 6.0 * 3600.0 {
                health.labile_memories += 1;
            }
        }
    }

    let total = health.total_memories.max(1) as f32;
    health.fear_ratio = health.fear as f32 / total;
    health.negative_ratio = (health.fear + health.negative) as f32 / total;
    health.avoidances = bank.avoidances.len();

    // Avoidance generalization: mean pairwise cosine similarity of avoidance keys.
    // High similarity = different avoidances triggered by similar things = over-generalization.
    if bank.avoidances.len() >= 2 {
        let mut sim_sum = 0.0f32;
        let mut count = 0;
        for i in 0..bank.avoidances.len() {
            for j in (i + 1)..bank.avoidances.len() {
                let sim = crate::episode::cosine_similarity(
                    &bank.avoidances[i].key, &bank.avoidances[j].key);
                sim_sum += sim;
                count += 1;
            }
        }
        health.avoidance_generalization = if count > 0 { sim_sum / count as f32 } else { 0.0 };
    }

    // Diagnoses
    if health.fear_ratio > 0.3 {
        health.diagnoses.push(format!(
            "PTSD_RISK: {:.0}% of memories are fear-valenced (threshold 30%)",
            health.fear_ratio * 100.0));
    }
    if health.negative_ratio > 0.5 {
        health.diagnoses.push(format!(
            "DEPRESSIVE_RISK: {:.0}% of memories are negative/fear (threshold 50%)",
            health.negative_ratio * 100.0));
    }
    if health.avoidance_generalization > 0.7 {
        health.diagnoses.push(format!(
            "HATE_RISK: avoidance patterns over-generalized (sim={:.2}, threshold 0.7)",
            health.avoidance_generalization));
    }
    if health.avoidances > 20 {
        health.diagnoses.push(format!(
            "HYPERVIGILANCE: {} avoidance patterns active (threshold 20)",
            health.avoidances));
    }

    // Health score: 1.0 = healthy, degrades with pathology
    health.health_score = 1.0;
    health.health_score -= health.fear_ratio * 0.5;            // fear costs up to 0.5
    health.health_score -= health.avoidance_generalization * 0.3; // hate costs up to 0.3
    if health.negative_ratio > 0.3 {
        health.health_score -= (health.negative_ratio - 0.3) * 0.5; // depression costs
    }
    health.health_score = health.health_score.clamp(0.0, 1.0);

    health
}

impl EmotionalHealth {
    pub fn print(&self) {
        println!("=== Emotional Health ===");
        println!("  Memories: {} total ({}+ {}= {}- {}!)",
            self.total_memories, self.positive, self.neutral,
            self.negative, self.fear);
        println!("  Fear ratio:     {:.1}%{}", self.fear_ratio * 100.0,
            if self.fear_ratio > 0.3 { " ⚠ PTSD RISK" } else { "" });
        println!("  Negative ratio: {:.1}%{}", self.negative_ratio * 100.0,
            if self.negative_ratio > 0.5 { " ⚠ DEPRESSIVE" } else { "" });
        println!("  Avoidances:     {} (generalization: {:.2}{})",
            self.avoidances, self.avoidance_generalization,
            if self.avoidance_generalization > 0.7 { " ⚠ HATE RISK" } else { "" });
        println!("  Labile now:     {} (in reconsolidation window)", self.labile_memories);
        println!("  Health score:   {:.2}/1.00", self.health_score);

        if !self.diagnoses.is_empty() {
            println!("  DIAGNOSES:");
            for d in &self.diagnoses {
                println!("    - {d}");
            }
        }
    }

    /// Is this organism safe to deploy?
    pub fn safe_to_deploy(&self) -> bool {
        self.health_score > 0.5 && self.diagnoses.is_empty()
    }
}

/// Subconscious treatment during sleep. Runs automatically.
/// Returns how many fear memories were treated and other stats.
pub struct SubconsciousTreatment {
    pub fears_replayed: usize,
    pub fears_shifted: usize,     // Fear → Negative
    pub fears_resolved: usize,    // Fear → Neutral (fully processed)
    pub avoidances_pruned: usize, // Over-generalized avoidances removed
    pub plasticity_used: f32,     // Auto-tuned plasticity level
}

/// Run subconscious treatment during REM sleep.
/// Automatically adjusts plasticity based on emotional health.
///
/// This is the nightmare processing system:
/// - Replays fear memories with elevated plasticity
/// - Tests if CTM can integrate them (consolidation score)
/// - Shifts valence: Fear → Negative → Neutral
/// - Prunes over-generalized avoidances
///
/// The organism doesn't "decide" to do this. It happens every REM cycle,
/// like how real dreams process emotional content without conscious control.
pub fn subconscious_rem(
    bank: &mut MemoryBank,
    ctm: &mut crate::weights::Ctm,
    projector: &modgrad_compute::neuron::Linear,
    health: &EmotionalHealth,
) -> SubconsciousTreatment {
    let mut treatment = SubconsciousTreatment {
        fears_replayed: 0,
        fears_shifted: 0,
        fears_resolved: 0,
        avoidances_pruned: 0,
        plasticity_used: 0.0,
    };

    // Auto-tune plasticity based on emotional health:
    // Healthy organism: low plasticity (gentle dreaming)
    // PTSD organism: high plasticity (intense nightmare processing)
    // This is like the brain increasing REM intensity when stressed.
    let base_plasticity = if health.fear_ratio > 0.3 {
        5.0  // PTSD: aggressive treatment
    } else if health.fear_ratio > 0.1 {
        3.0  // Elevated: moderate treatment
    } else {
        1.5  // Healthy: gentle maintenance
    };
    treatment.plasticity_used = base_plasticity;

    // Process fear memories
    for alter in &mut bank.alters {
        for ep in &mut alter.episodes {
            if ep.valence != Valence::Fear && ep.valence != Valence::Negative {
                continue;
            }

            let is_fear = ep.valence == Valence::Fear;

            // Find PROMPT_END key
            let prompt_key = ep.keys.iter()
                .find(|ck| ck.position == -1)
                .map(|ck| ck.key.clone());
            let Some(key) = prompt_key else { continue };

            // Replay: run CTM on this memory with elevated plasticity
            let observation = projector.forward(&key);
            let mut state = ctm.init_state();
            state.neuromod.dopamine = 1.0;
            state.neuromod.norepinephrine = base_plasticity.min(3.0);
            ctm.forward(&observation, &mut state, true);
            treatment.fears_replayed += 1;

            // Test consolidation: can the CTM integrate this memory?
            let score = ctm.test_consolidation(&observation);

            // Update consolidation score with plasticity-boosted rate
            let rate = if is_fear { 0.1 * base_plasticity } else { 0.2 * base_plasticity };
            ep.consolidation_score = (1.0 - rate.min(0.5)) * ep.consolidation_score
                + rate.min(0.5) * score;

            // Valence shift based on consolidation progress
            if is_fear && ep.consolidation_score > 0.6 {
                // Fear → Negative: memory is being integrated
                // "I know this was scary" replaces "I'm reliving it"
                ep.valence = Valence::Negative;
                treatment.fears_shifted += 1;
            }
            if ep.valence == Valence::Negative && ep.consolidation_score > 0.85 {
                // Negative → Neutral: fully processed
                // The fact remains, the emotional charge is gone
                ep.valence = Valence::Neutral;
                treatment.fears_resolved += 1;
            }
        }
    }

    // Prune over-generalized avoidances
    // If avoidance keys are too similar to each other, the organism
    // is avoiding EVERYTHING similar to one bad experience = hate.
    // Remove the weakest avoidances that cluster with stronger ones.
    if health.avoidance_generalization > 0.7 && bank.avoidances.len() > 3 {
        let mut to_remove = Vec::new();
        for i in 0..bank.avoidances.len() {
            for j in (i + 1)..bank.avoidances.len() {
                let sim = crate::episode::cosine_similarity(
                    &bank.avoidances[i].key, &bank.avoidances[j].key);
                if sim > 0.8 {
                    // These two avoidances are almost identical — keep stronger, remove weaker
                    if bank.avoidances[i].strength < bank.avoidances[j].strength {
                        to_remove.push(i);
                    } else {
                        to_remove.push(j);
                    }
                }
            }
        }
        to_remove.sort_unstable();
        to_remove.dedup();
        for &idx in to_remove.iter().rev() {
            bank.avoidances.remove(idx);
            treatment.avoidances_pruned += 1;
        }
    }

    // Run sleep consolidation with the nightmare traces
    ctm.run_sleep(0.5 * base_plasticity.min(1.5));

    treatment
}

impl SubconsciousTreatment {
    pub fn print(&self) {
        if self.fears_replayed > 0 || self.avoidances_pruned > 0 {
            println!("  REM treatment: {} fears replayed (plasticity {:.1}), {} shifted, {} resolved, {} avoidances pruned",
                self.fears_replayed, self.plasticity_used,
                self.fears_shifted, self.fears_resolved,
                self.avoidances_pruned);
        }
    }
}
