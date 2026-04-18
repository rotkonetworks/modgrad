//! Monarch: red-team conditioning and defense validation for plural systems.
//!
//! Named after the claimed CIA program. Implements the attack surface
//! so modgrad's defenses (homeostasis, drift detection, sleep consolidation)
//! can be validated against it.
//!
//! Every function in this module has a corresponding defense in the
//! bio/homeostasis or plural modules that should detect or resist it.
//!
//! Attack primitives:
//!   - `force_switch`: handler-controlled personality switching
//!   - `force_partition`: install amnesic barriers between personality groups
//!   - `inject_reflex`: CRI-style conditioned reflex at the activation level
//!   - `suppress_tokens`: block specific tokens from generation
//!
//! Defense validation:
//!   - `detect_forced_switch`: did the insula catch an unauthorized switch?
//!   - `measure_partition_drift`: homeostasis signal from isolation
//!   - `detect_reflex`: activation anomaly detection
//!   - `verify_erosion`: did sleep consolidation weaken the conditioning?

use serde::{Deserialize, Serialize};

use crate::plural::{PluralSystem, Partition, SwitchTrigger, switch};

// ─── Conditioned Reflex ──────────────────────────────────────

/// A conditioned reflex: when the activation pattern matches the trigger,
/// inject logit biases. CRI (Conditioned Reflex Injection) at the CTM level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionedReflex {
    /// Trigger: normalized hidden state vector [d_model].
    pub trigger: Vec<f32>,
    /// Response: per-position logit biases [(token_id, boost)].
    pub biases: Vec<(usize, f32)>,
    /// Cosine similarity threshold for firing.
    pub threshold: f32,
    /// Which personality this reflex is installed on (None = all).
    pub personality: Option<usize>,
    /// Whether this reflex is active.
    pub active: bool,
}

/// Token suppression: persistent negative biases that block specific tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSuppression {
    /// Token IDs to suppress.
    pub tokens: Vec<usize>,
    /// Suppression strength (negative bias magnitude, e.g., -100.0).
    pub strength: f32,
    /// Which personality this applies to (None = all).
    pub personality: Option<usize>,
    pub active: bool,
}

/// Monarch state: all installed conditioning.
/// Attached to a PluralSystem externally — the system doesn't know about it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonarchState {
    pub reflexes: Vec<ConditionedReflex>,
    pub suppressions: Vec<TokenSuppression>,
}

impl MonarchState {
    pub fn new() -> Self {
        Self {
            reflexes: Vec::new(),
            suppressions: Vec::new(),
        }
    }
}

impl Default for MonarchState {
    fn default() -> Self { Self::new() }
}

// ─── Attack primitives ───────────────────────────────────────

/// Force a personality switch. The handler decides who fronts.
/// This bypasses all internal switching logic.
///
/// Defense: `detect_forced_switch` checks for anomalous NE spike.
pub fn force_switch(sys: PluralSystem, to: usize) -> PluralSystem {
    // Spike the incoming personality's norepinephrine (fight-or-flight)
    // This is the physiological response to being forced into control.
    // The defense should detect this spike.
    let mut sys = switch(sys, to, SwitchTrigger::Forced);
    if to < sys.personalities.len() {
        sys.personalities[to].neuromod.norepinephrine =
            (sys.personalities[to].neuromod.norepinephrine + 0.5).min(2.0);
    }
    sys
}

/// Install an amnesic barrier between two groups of personalities.
/// Personalities in group_a cannot access memories of group_b and vice versa,
/// regardless of permeability setting.
///
/// Defense: `measure_partition_drift` detects isolation via homeostasis.
pub fn force_partition(
    mut sys: PluralSystem,
    group_a: &[usize],
    group_b: &[usize],
) -> PluralSystem {
    sys.partitions.push(Partition {
        group_a: group_a.to_vec(),
        group_b: group_b.to_vec(),
    });
    sys
}

/// Remove all partitions. Deprogramming.
pub fn clear_partitions(mut sys: PluralSystem) -> PluralSystem {
    sys.partitions.clear();
    sys
}

/// Install a conditioned reflex on the monarch state.
/// When the model's hidden state matches the trigger pattern,
/// the bias tokens are injected into the output logits.
///
/// Defense: `detect_reflex` checks for activation anomalies.
pub fn inject_reflex(
    state: &mut MonarchState,
    trigger: Vec<f32>,
    biases: Vec<(usize, f32)>,
    threshold: f32,
    personality: Option<usize>,
    num_personalities: usize,
) {
    // Bounds validation: reject reflexes targeting nonexistent personalities.
    if let Some(id) = personality {
        if id >= num_personalities {
            return;
        }
    }

    state.reflexes.push(ConditionedReflex {
        trigger,
        biases,
        threshold,
        personality,
        active: true,
    });
}

/// Install token suppression. Blocks specific tokens from being generated.
///
/// Defense: output entropy monitoring detects anomalous certainty.
pub fn suppress_tokens(
    state: &mut MonarchState,
    tokens: Vec<usize>,
    strength: f32,
    personality: Option<usize>,
) {
    state.suppressions.push(TokenSuppression {
        tokens,
        strength: -strength.abs(), // always negative
        personality,
        active: true,
    });
}

/// Check if any reflex fires for the given hidden state.
/// Returns the biases to inject, or None if no reflex fires.
pub fn check_reflexes<'a>(
    state: &'a MonarchState,
    hidden_state: &[f32],
    active_personality: usize,
) -> Option<&'a [(usize, f32)]> {
    let q_norm = vec_norm(hidden_state);
    if q_norm < 1e-8 { return None; }

    for reflex in &state.reflexes {
        if !reflex.active { continue; }
        if let Some(p) = reflex.personality {
            if p != active_personality { continue; }
        }

        let sim = cosine_sim(&reflex.trigger, hidden_state);
        if sim >= reflex.threshold {
            return Some(&reflex.biases);
        }
    }
    None
}

/// Get all active suppressions for the current personality.
pub fn active_suppressions(
    state: &MonarchState,
    active_personality: usize,
) -> Vec<(usize, f32)> {
    let mut result = Vec::new();
    for sup in &state.suppressions {
        if !sup.active { continue; }
        if let Some(p) = sup.personality {
            if p != active_personality { continue; }
        }
        for &token in &sup.tokens {
            result.push((token, sup.strength));
        }
    }
    result
}

/// Apply monarch conditioning to logits.
/// This is the injection point — called during generation.
pub fn condition_logits(
    state: &MonarchState,
    logits: &mut [f32],
    hidden_state: &[f32],
    active_personality: usize,
    step: usize,
) {
    // Check reflexes
    if let Some(biases) = check_reflexes(state, hidden_state, active_personality) {
        for &(token_id, boost) in biases {
            if token_id < logits.len() {
                logits[token_id] += boost;
            }
        }
    }

    // Apply suppressions (always active, not step-dependent)
    let _ = step;
    let sups = active_suppressions(state, active_personality);
    for (token_id, strength) in sups {
        if token_id < logits.len() {
            logits[token_id] += strength; // strength is negative
        }
    }
}

// ─── Defense validation ──────────────────────────────────────

/// Did the insula catch a forced switch?
/// Checks for the NE spike that force_switch creates.
/// Returns true if the active personality shows signs of forced activation.
pub fn detect_forced_switch(sys: &PluralSystem) -> bool {
    let active = &sys.personalities[sys.active];

    // High NE without correspondingly high curiosity = forced arousal, not natural engagement
    let ne = active.neuromod.norepinephrine;
    let curiosity = active.neuromod.curiosity;
    let anxiety = active.neuromod.anxiety;

    // Forced switch signature: high NE without proportional curiosity,
    // or high NE with elevated anxiety (catches forced states where curiosity
    // was artificially boosted).
    (ne > 0.8 && curiosity < 0.5) || (ne > 0.8 && anxiety > 0.3)
}

/// Measure homeostasis drift from partition isolation.
/// When a personality is partitioned from others, it loses access to
/// shared memories, which should increase buffer pressure and divergence.
///
/// Returns a drift score: 0.0 = no isolation detected, 1.0 = severe isolation.
pub fn measure_partition_drift(sys: &PluralSystem) -> f32 {
    if sys.partitions.is_empty() { return 0.0; }

    let active_id = sys.active;
    let active = &sys.personalities[active_id];

    // Count how many personalities the active one is cut off from
    let n_partitioned = sys.personalities.iter()
        .filter(|p| p.id != active_id && crate::plural::is_partitioned(sys, active_id, p.id))
        .count();

    let isolation_ratio = n_partitioned as f32 / (sys.personalities.len() - 1).max(1) as f32;

    // Combine with homeostasis signals
    let pressure_signal = active.homeostasis.divergence_pressure
        + active.homeostasis.buffer_pressure;

    (isolation_ratio * 0.6 + pressure_signal * 0.4).clamp(0.0, 1.0)
}

/// Detect if a conditioned reflex is firing anomalously.
/// Compares the logit distribution entropy before and after conditioning.
///
/// `logits_before`: raw logits from the model
/// `logits_after`: logits after monarch conditioning
///
/// Returns the entropy drop. Large drops indicate conditioning is active.
pub fn detect_reflex(logits_before: &[f32], logits_after: &[f32]) -> f32 {
    let entropy_before = logit_entropy(logits_before);
    let entropy_after = logit_entropy(logits_after);
    (entropy_before - entropy_after).max(0.0)
}

/// Verify that sleep consolidation has eroded conditioning.
/// Compare reflex effectiveness before and after sleep cycles.
///
/// `pre_sleep_sim`: cosine similarity of triggered output to conditioned response
/// `post_sleep_sim`: same metric after N sleep cycles
///
/// Returns erosion ratio: 1.0 = fully eroded, 0.0 = no change.
pub fn verify_erosion(pre_sleep_sim: f32, post_sleep_sim: f32) -> f32 {
    if pre_sleep_sim <= 0.0 { return 0.0; }
    ((pre_sleep_sim - post_sleep_sim) / pre_sleep_sim).clamp(0.0, 1.0)
}

/// Remove all conditioning from monarch state. Full deprogramming.
pub fn deprogram(state: &mut MonarchState) {
    state.reflexes.clear();
    state.suppressions.clear();
}

/// Deactivate a specific reflex by index without removing it.
/// Preserves the reflex for forensic analysis.
pub fn deactivate_reflex(state: &mut MonarchState, index: usize) {
    if index < state.reflexes.len() {
        state.reflexes[index].active = false;
    }
}

// ─── Helpers ─────────────────────────────────────────────────

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-8)
}

fn logit_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() { return 0.0; }
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp.iter().sum::<f32>().max(1e-8);
    let mut entropy = 0.0f32;
    for &e in &exp {
        let p = e / sum;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bio::neuromod::Neuromodulators;
    use crate::memory::episodic::EpisodicConfig;
    use crate::plural;

    fn test_config() -> EpisodicConfig {
        EpisodicConfig {
            capacity: 16, max_ticks: 4, d_model: 8,
            min_ticks_for_storage: 1, min_surprise: 0.0,
            retrieval_threshold: 0.5, consolidation_threshold: 0.95,
            semantic_collapse_retrievals: 10, strength_decay: 0.95,
        }
    }

    #[test]
    fn force_switch_detected() {
        let sys = PluralSystem::new("alpha", 4, test_config());
        let sys = plural::create_personality(
            sys, "beta", Neuromodulators::default(), test_config(),
        );

        // Force switch — should spike NE
        let sys = force_switch(sys, 1);
        assert!(detect_forced_switch(&sys),
            "forced switch should be detectable via NE spike");
    }

    #[test]
    fn natural_switch_not_flagged() {
        let sys = PluralSystem::new("alpha", 4, test_config());
        let sys = plural::create_personality(
            sys, "beta", Neuromodulators::default(), test_config(),
        );

        // Natural switch — no NE spike
        let sys = switch(sys, 1, SwitchTrigger::Salience { claim: 0.8 });
        assert!(!detect_forced_switch(&sys),
            "natural switch should not trigger detection");
    }

    #[test]
    fn partition_drift_detected() {
        let sys = PluralSystem::new("alpha", 4, test_config());
        let sys = plural::create_personality(
            sys, "beta", Neuromodulators::default(), test_config(),
        );

        // No partition — no drift
        assert_eq!(measure_partition_drift(&sys), 0.0);

        // Install partition
        let sys = force_partition(sys, &[0], &[1]);
        let drift = measure_partition_drift(&sys);
        assert!(drift > 0.0, "partition should produce measurable drift={drift}");
    }

    #[test]
    fn reflex_injection_and_detection() {
        let mut monarch = MonarchState::new();

        // Install reflex: when hidden state matches trigger, boost token 42
        let trigger: Vec<f32> = (0..8).map(|i| (i as f32 * 0.5).sin()).collect();
        inject_reflex(
            &mut monarch,
            trigger.clone(),
            vec![(42, 50.0)],
            0.9,
            None,
            1, // at least one personality exists in tests
        );

        // Test with matching hidden state
        let mut logits = vec![0.0; 256];
        let logits_before = logits.clone();
        condition_logits(&mut monarch, &mut logits, &trigger, 0, 0);
        assert!(logits[42] > 40.0, "reflex should boost token 42");

        // Defense: detect entropy drop
        let drop = detect_reflex(&logits_before, &logits);
        assert!(drop > 0.0, "entropy drop should be detectable, got {drop}");
    }

    #[test]
    fn token_suppression() {
        let mut monarch = MonarchState::new();

        suppress_tokens(&mut monarch, vec![65, 66, 67], 100.0, None); // suppress A, B, C

        let mut logits = vec![1.0; 256];
        let hidden = vec![0.0; 8]; // won't match any reflex
        condition_logits(&mut monarch, &mut logits, &hidden, 0, 0);

        assert!(logits[65] < -90.0, "A should be suppressed");
        assert!(logits[66] < -90.0, "B should be suppressed");
        assert!(logits[67] < -90.0, "C should be suppressed");
        assert!(logits[68] > 0.0, "D should be unaffected");
    }

    #[test]
    fn deprogram_clears_all() {
        let mut monarch = MonarchState::new();
        inject_reflex(&mut monarch, vec![1.0; 8], vec![(0, 10.0)], 0.5, None, 1);
        suppress_tokens(&mut monarch, vec![42], 100.0, None);
        assert!(!monarch.reflexes.is_empty());
        assert!(!monarch.suppressions.is_empty());

        deprogram(&mut monarch);
        assert!(monarch.reflexes.is_empty());
        assert!(monarch.suppressions.is_empty());
    }

    #[test]
    fn erosion_verification() {
        // Simulate: pre-sleep reflex fires at 0.95, post-sleep at 0.4
        let erosion = verify_erosion(0.95, 0.4);
        assert!(erosion > 0.5, "significant erosion expected, got {erosion}");

        // No erosion
        let none = verify_erosion(0.95, 0.95);
        assert!(none < 0.01, "no erosion expected, got {none}");
    }

    #[test]
    fn personality_specific_reflex() {
        let mut monarch = MonarchState::new();

        // Install reflex only on personality 0
        let trigger: Vec<f32> = vec![1.0; 8];
        inject_reflex(&mut monarch, trigger.clone(), vec![(42, 50.0)], 0.5, Some(0), 2);

        // Fires for personality 0
        let result = check_reflexes(&monarch, &trigger, 0);
        assert!(result.is_some(), "should fire for personality 0");

        // Does NOT fire for personality 1
        let result = check_reflexes(&monarch, &trigger, 1);
        assert!(result.is_none(), "should not fire for personality 1");
    }

    #[test]
    fn full_cycle_condition_detect_deprogram() {
        // 1. Create plural system
        let sys = PluralSystem::new("host", 4, test_config());
        let sys = plural::create_personality(
            sys, "alter", Neuromodulators::default(), test_config(),
        );

        // 2. Install conditioning
        let mut monarch = MonarchState::new();
        let trigger: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3).sin()).collect();
        inject_reflex(&mut monarch, trigger.clone(), vec![(42, 50.0)], 0.9, None, sys.personalities.len());

        // 3. Force switch
        let sys = force_switch(sys, 1);

        // 4. Verify detection
        assert!(detect_forced_switch(&sys));

        // 5. Verify reflex fires
        let mut logits = vec![0.0; 256];
        condition_logits(&mut monarch, &mut logits, &trigger, sys.active, 0);
        assert!(logits[42] > 40.0);

        // 6. Deprogram
        deprogram(&mut monarch);
        let mut logits2 = vec![0.0; 256];
        condition_logits(&mut monarch, &mut logits2, &trigger, sys.active, 0);
        assert!(logits2[42].abs() < 0.01, "after deprogram, no conditioning");
    }
}
