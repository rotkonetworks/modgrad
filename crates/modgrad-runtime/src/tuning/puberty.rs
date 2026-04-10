//! Developmental capability escalation.
//!
//! The organism gradually gains authority over its own parameters.
//! The schedule is performance-based, not time-based.
//!
//! Lives on the HOST — the organism never imports this module.
//! It just notices that `Param::set()` calls that used to fail
//! now succeed. Like a teenager who discovers they can stay up late.
//!
//! ```text
//! Infant     → Neuromodulatory only, parent controls everything
//! Aware      → Can read own TuningConfig (observes own parameters)
//! Adolescent → Wider Neuromodulatory bounds (more autonomic range)
//! Puberty    → Hormonal tier (sets own learning rates, noise, thresholds)
//! Adult      → Epigenome tier (requests architecture changes)
//! ```
//!
//! Invariant: the organism can never modify the rules that govern
//! capability escalation. Development lives on the host.

use crate::tick_state::TickSignals;
use super::tuning::{Tier, TuningConfig};

/// Developmental stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Parent controls everything. Organism has Neuromodulatory tier only.
    Infant,
    /// Organism can READ its own TuningConfig. Learns "I have parameters."
    Aware,
    /// Wider Neuromodulatory bounds. More emotional/autonomic range.
    Adolescent,
    /// Organism gains Hormonal tier. Sets own learning rates, noise, thresholds.
    Puberty,
    /// Organism gains Epigenome tier. Can request architecture changes.
    Adult,
}

impl Stage {
    /// Maximum tier the organism can WRITE at this stage.
    pub fn max_write_tier(&self) -> Tier {
        match self {
            Stage::Infant => Tier::Neuromodulatory,
            Stage::Aware => Tier::Neuromodulatory,
            Stage::Adolescent => Tier::Neuromodulatory, // wider bounds, same tier
            Stage::Puberty => Tier::Hormonal,
            Stage::Adult => Tier::Epigenome,
        }
    }

    /// Can the organism read its own TuningConfig?
    pub fn can_read_config(&self) -> bool {
        !matches!(self, Stage::Infant)
    }
}

/// Measurable conditions that gate stage transitions.
/// All counters reset on regression (crash, performance drop).
#[derive(Debug, Clone, Default)]
pub struct Milestones {
    /// Consecutive sleep cycles without homeostasis crash (forced sleep).
    pub stable_cycles: u64,
    /// Best score on any Level 3 challenge (0.0 = never passed).
    pub best_challenge_score: f32,
    /// Self-model score: mutual information between insula output and
    /// tuning param values. High = organism models its own configuration.
    pub self_model_score: f32,
    /// Consecutive cycles without crash at widened bounds.
    pub wide_stable_cycles: u64,
    /// Whether self-tuning improved performance (measured by host).
    pub self_tuning_improved: bool,
    /// Total sleep cycles experienced.
    pub total_sleep_cycles: u64,
}

/// Developmental state machine. Host-side only.
#[derive(Debug, Clone)]
pub struct Development {
    pub stage: Stage,
    pub milestones: Milestones,
    /// History of stage transitions for audit.
    pub transitions: Vec<(u64, Stage, Stage)>, // (cycle, from, to)
}

impl Development {
    pub fn new() -> Self {
        Self {
            stage: Stage::Infant,
            milestones: Milestones::default(),
            transitions: Vec::new(),
        }
    }

    /// Update milestones from a training step's signals.
    /// Called by the host after each organism step.
    pub fn observe(&mut self, _signals: &TickSignals, forced_sleep: bool) {
        self.milestones.total_sleep_cycles += 1;

        if forced_sleep {
            // Crash: reset stability counters
            self.milestones.stable_cycles = 0;
            self.milestones.wide_stable_cycles = 0;
        } else {
            self.milestones.stable_cycles += 1;
            if matches!(self.stage, Stage::Adolescent | Stage::Puberty | Stage::Adult) {
                self.milestones.wide_stable_cycles += 1;
            }
        }
    }

    /// Record a challenge result.
    pub fn record_challenge(&mut self, score: f32) {
        if score > self.milestones.best_challenge_score {
            self.milestones.best_challenge_score = score;
        }
    }

    /// Record self-model quality (host measures this periodically).
    pub fn record_self_model(&mut self, score: f32) {
        self.milestones.self_model_score = score;
    }

    /// Record whether self-tuning improved performance.
    pub fn record_self_tuning(&mut self, improved: bool) {
        self.milestones.self_tuning_improved = improved;
    }

    /// Check if a stage transition should occur. Returns true if transitioned.
    /// The host calls this periodically (e.g., every 100 sleep cycles).
    pub fn check_transition(&mut self) -> Option<Stage> {
        let old = self.stage;
        let cycle = self.milestones.total_sleep_cycles;

        let new = match self.stage {
            Stage::Infant => {
                // Gate: stable for 1000 cycles AND any challenge progress
                if self.milestones.stable_cycles >= 1000
                    && self.milestones.best_challenge_score > 0.0
                {
                    Some(Stage::Aware)
                } else { None }
            }
            Stage::Aware => {
                // Gate: organism models its own config
                // (insula correlates with tuning params)
                if self.milestones.self_model_score > 0.5 {
                    Some(Stage::Adolescent)
                } else { None }
            }
            Stage::Adolescent => {
                // Gate: stable with wider bounds for 5000 cycles
                if self.milestones.wide_stable_cycles >= 5000 {
                    Some(Stage::Puberty)
                } else { None }
            }
            Stage::Puberty => {
                // Gate: self-tuning actually improved performance
                if self.milestones.self_tuning_improved
                    && self.milestones.wide_stable_cycles >= 10000
                {
                    Some(Stage::Adult)
                } else { None }
            }
            Stage::Adult => None, // terminal stage
        };

        if let Some(new_stage) = new {
            self.stage = new_stage;
            self.transitions.push((cycle, old, new_stage));
            eprintln!("[puberty] stage transition: {old:?} → {new_stage:?} at cycle {cycle}");
            Some(new_stage)
        } else {
            None
        }
    }

    /// Apply the current stage's capability grants to a TuningConfig.
    /// Called by host to widen bounds when stage advances.
    pub fn apply_grants(&self, config: &mut TuningConfig) {
        match self.stage {
            Stage::Infant => {
                // Default bounds, no changes
            }
            Stage::Aware => {
                // No write changes, just read access (handled by host logic)
            }
            Stage::Adolescent => {
                // Widen Neuromodulatory bounds
                // NE: [0.0, 3.0] → [0.0, 5.0]
                config.neuromod.ne_max.set_bounds(0.0, 5.0, Tier::Epigenome).ok();
                // 5HT: [0.1, 1.0] → [0.05, 1.5]
                config.neuromod.serotonin_min.set_bounds(0.0, 0.5, Tier::Epigenome).ok();
                config.neuromod.serotonin_max.set_bounds(0.5, 2.0, Tier::Epigenome).ok();
            }
            Stage::Puberty => {
                // Grant Hormonal tier: organism can set learning rates, noise, thresholds
                // (The Param<T>::set() calls will now succeed at Hormonal tier)
                // Widen some Hormonal bounds for self-tuning
                config.learning.surprise_lr_max.set_bounds(0.5, 10.0, Tier::Epigenome).ok();
                config.noise.base_amplitude.set_bounds(0.0, 2.0, Tier::Epigenome).ok();
                config.sleep.blend_max.set_bounds(0.05, 1.0, Tier::Epigenome).ok();
            }
            Stage::Adult => {
                // Epigenome tier: organism can request architecture changes
                // (handled by host validation, not direct param access)
            }
        }
    }

    /// Human-readable status.
    pub fn status(&self) -> String {
        let m = &self.milestones;
        format!(
            "stage={:?} stable={} challenge={:.2} self_model={:.2} wide_stable={} transitions={}",
            self.stage, m.stable_cycles, m.best_challenge_score,
            m.self_model_score, m.wide_stable_cycles, self.transitions.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infant_to_aware() {
        let mut dev = Development::new();
        assert_eq!(dev.stage, Stage::Infant);

        // Not enough cycles
        dev.milestones.stable_cycles = 500;
        dev.milestones.best_challenge_score = 0.5;
        assert!(dev.check_transition().is_none());

        // Enough cycles but no challenge progress
        dev.milestones.stable_cycles = 1000;
        dev.milestones.best_challenge_score = 0.0;
        assert!(dev.check_transition().is_none());

        // Both conditions met
        dev.milestones.best_challenge_score = 0.1;
        assert_eq!(dev.check_transition(), Some(Stage::Aware));
        assert_eq!(dev.stage, Stage::Aware);
    }

    #[test]
    fn test_crash_resets_stability() {
        let mut dev = Development::new();
        let signals = TickSignals::default();

        for _ in 0..100 {
            dev.observe(&signals, false);
        }
        assert_eq!(dev.milestones.stable_cycles, 100);

        // Crash
        dev.observe(&signals, true);
        assert_eq!(dev.milestones.stable_cycles, 0);
    }

    #[test]
    fn test_stage_tiers() {
        assert_eq!(Stage::Infant.max_write_tier(), Tier::Neuromodulatory);
        assert_eq!(Stage::Puberty.max_write_tier(), Tier::Hormonal);
        assert_eq!(Stage::Adult.max_write_tier(), Tier::Epigenome);
        assert!(!Stage::Infant.can_read_config());
        assert!(Stage::Aware.can_read_config());
    }

    #[test]
    fn test_full_progression() {
        let mut dev = Development::new();

        // Infant → Aware
        dev.milestones.stable_cycles = 1000;
        dev.milestones.best_challenge_score = 0.1;
        dev.check_transition();
        assert_eq!(dev.stage, Stage::Aware);

        // Aware → Adolescent
        dev.milestones.self_model_score = 0.6;
        dev.check_transition();
        assert_eq!(dev.stage, Stage::Adolescent);

        // Adolescent → Puberty
        dev.milestones.wide_stable_cycles = 5000;
        dev.check_transition();
        assert_eq!(dev.stage, Stage::Puberty);

        // Puberty → Adult
        dev.milestones.self_tuning_improved = true;
        dev.milestones.wide_stable_cycles = 10000;
        dev.check_transition();
        assert_eq!(dev.stage, Stage::Adult);

        assert_eq!(dev.transitions.len(), 4);
    }
}
