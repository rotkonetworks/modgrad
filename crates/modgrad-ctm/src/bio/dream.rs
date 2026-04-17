//! Dream replay: offline practice during sleep.
//!
//! During sleep, the organism re-runs stored episodic memories through
//! a caller-provided forward function. If current performance exceeds
//! the stored valence, the organism experiences relief and the memory
//! is updated. If performance is worse, the pain deepens.
//!
//! This is REM sleep — the organism practices with eyes closed.
//!
//! Also provides retrieval priming: blending retrieved trajectories
//! into the organism's initial state before a forward pass, so past
//! experience directly influences computation, not just feelings.

use crate::memory::episodic::{self, EpisodicMemory, EpisodicRetrievalResult};
use super::pain::{self, PainConfig};
use super::homeostasis::Homeostasis;
use super::neuromod::Neuromodulators;

/// Result of one dream replay cycle.
#[derive(Debug, Clone)]
pub struct DreamResult {
    /// Episodes that improved (organism now handles them better).
    pub overcame: usize,
    /// Episodes that stayed the same.
    pub unchanged: usize,
    /// Episodes that got worse (regression).
    pub regressed: usize,
    /// Total dopamine earned from overcoming.
    pub total_relief: f32,
}

/// Run a dream replay cycle over the most painful stored episodes.
///
/// `evaluate`: a closure that takes an episode index and returns
/// (current_loss, was_correct) for that episode's input.
/// The caller is responsible for running the forward pass.
///
/// `max_replays`: how many episodes to replay (most painful first).
pub fn dream_replay(
    mut memory: EpisodicMemory,
    homeostasis: &mut Homeostasis,
    neuromod: &mut Neuromodulators,
    evaluate: &dyn Fn(usize, &[f32]) -> (f32, bool),
    max_replays: usize,
    pain_cfg: &PainConfig,
) -> (EpisodicMemory, DreamResult) {
    let mut result = DreamResult {
        overcame: 0, unchanged: 0, regressed: 0, total_relief: 0.0,
    };

    if memory.count == 0 { return (memory, result); }

    // Find most painful episodes
    let mut painful: Vec<(usize, f32)> = (0..memory.count)
        .filter_map(|i| {
            let meta = memory.meta(i)?;
            if !meta.alive || meta.valence >= 0.0 { return None; }
            Some((i, meta.valence))
        })
        .collect();

    painful.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)); // most negative first
    painful.truncate(max_replays);

    for &(idx, old_valence) in &painful {
        let d = memory.config.d_model;
        let key = memory.key(idx).to_vec();

        // Caller evaluates this episode with current weights
        let (current_loss, was_correct) = evaluate(idx, &key);

        let meta = memory.meta(idx).unwrap();
        let old_loss = meta.loss_at_storage;

        if old_loss <= 0.0 { result.unchanged += 1; continue; }

        let improvement = (old_loss - current_loss) / old_loss;

        if improvement > 0.1 && was_correct {
            // Overcame: organism now handles this better
            result.overcame += 1;

            let resp = pain::on_overcoming(
                homeostasis, neuromod, old_valence, true, pain_cfg,
            );
            result.total_relief += resp.valence_for_storage;
        } else if improvement < -0.1 {
            // Regression: organism got worse at this
            result.regressed += 1;
        } else {
            result.unchanged += 1;
        }
    }

    // Reappraise all painful memories based on general improvement
    let avg_current_loss = if !painful.is_empty() {
        painful.iter().map(|&(idx, _)| {
            let key = memory.key(idx).to_vec();
            let (loss, _) = evaluate(idx, &key);
            loss
        }).sum::<f32>() / painful.len() as f32
    } else {
        0.0
    };

    let (mem, _outcomes) = episodic::reappraise(
        memory,
        &|_| Some(avg_current_loss),
        0.5,
    );
    memory = mem;

    (memory, result)
}

// ─── Retrieval priming ───────────────────────────────────────

/// Prime a region's initial state using a retrieved episodic trajectory.
///
/// Blends the retrieved trajectory into `region_outputs` before the
/// forward pass. This means the organism's computation starts from
/// a state informed by past experience, not from zero.
///
/// `region_outputs`: mutable ref to RegionalState.region_outputs
/// `retrieval`: result from episodic::retrieve()
/// `blend`: how much to mix (0.0 = no priming, 1.0 = full override)
/// `target_region`: which region to prime (typically hippocampus)
pub fn prime_state(
    region_outputs: &mut [Vec<f32>],
    retrieval: &EpisodicRetrievalResult,
    blend: f32,
    target_region: usize,
) {
    if blend <= 0.0 || retrieval.n_matches == 0 { return; }
    if target_region >= region_outputs.len() { return; }

    let target = &mut region_outputs[target_region];
    let source = &retrieval.blended_final_state;
    let n = target.len().min(source.len());
    let b = blend.clamp(0.0, 1.0);

    for i in 0..n {
        target[i] = (1.0 - b) * target[i] + b * source[i];
    }
}

/// Prime multiple regions from a retrieved trajectory.
///
/// The trajectory is sliced per-region based on d_model sizes.
/// `region_dims`: d_model for each region (to know how to slice the trajectory)
pub fn prime_state_multi(
    region_outputs: &mut [Vec<f32>],
    retrieval: &EpisodicRetrievalResult,
    blend: f32,
    region_dims: &[usize],
) {
    if blend <= 0.0 || retrieval.n_matches == 0 { return; }

    let traj = &retrieval.blended_final_state;
    let mut offset = 0;
    for (r, &dim) in region_dims.iter().enumerate() {
        if r >= region_outputs.len() { break; }
        let end = (offset + dim).min(traj.len());
        if offset >= traj.len() { break; }

        let target = &mut region_outputs[r];
        let b = blend.clamp(0.0, 1.0);
        for i in 0..dim.min(target.len()).min(end - offset) {
            target[i] = (1.0 - b) * target[i] + b * traj[offset + i];
        }
        offset = end;
    }
}

// ─── Adaptive pain focus ─────────────────────────────────────

/// Tracks per-position failure rates and adjusts pain weights.
///
/// Instead of a fixed decay (position 0 = 3x, position 1 = 2x, ...),
/// the weight for each position is proportional to how often the
/// organism fails there. Positions the organism has mastered get
/// low weight. Positions it struggles with get high weight.
#[derive(Debug, Clone)]
pub struct AdaptivePainFocus {
    /// Per-position failure rate EMA.
    failure_rates: Vec<f32>,
    /// Smoothing factor.
    alpha: f32,
    /// Minimum weight (no position goes to zero).
    min_weight: f32,
}

impl AdaptivePainFocus {
    pub fn new(n_positions: usize, alpha: f32) -> Self {
        Self {
            failure_rates: vec![0.5; n_positions], // start neutral
            alpha,
            min_weight: 0.5,
        }
    }

    /// Update failure rate for a position.
    pub fn update(&mut self, position: usize, was_correct: bool) {
        if position >= self.failure_rates.len() { return; }
        let signal = if was_correct { 0.0 } else { 1.0 };
        self.failure_rates[position] =
            self.alpha * self.failure_rates[position] + (1.0 - self.alpha) * signal;
    }

    /// Get pain weight for a position.
    /// High failure rate → high weight. Low failure rate → low weight.
    pub fn weight(&self, position: usize) -> f32 {
        if position >= self.failure_rates.len() {
            return self.min_weight;
        }
        // Scale: failure_rate 0.0 → min_weight, failure_rate 1.0 → 3.0
        self.min_weight + (3.0 - self.min_weight) * self.failure_rates[position]
    }

    /// Get all weights as a slice (for diagnostics).
    pub fn weights(&self) -> &[f32] {
        &self.failure_rates
    }

    /// Which position has the highest failure rate?
    pub fn weakest_position(&self) -> usize {
        self.failure_rates.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_focus_tracks_failures() {
        let mut focus = AdaptivePainFocus::new(5, 0.9);

        // Position 0: always correct → low weight
        for _ in 0..20 { focus.update(0, true); }

        // Position 3: always wrong → high weight
        for _ in 0..20 { focus.update(3, false); }

        assert!(focus.weight(3) > focus.weight(0),
            "failing position should have higher weight: {} vs {}",
            focus.weight(3), focus.weight(0));
        assert_eq!(focus.weakest_position(), 3);
    }

    #[test]
    fn prime_state_blends() {
        let mut outputs = vec![vec![1.0; 8], vec![2.0; 8]];
        let retrieval = EpisodicRetrievalResult {
            blended_trajectory: vec![0.0; 32],
            blended_final_state: vec![0.5; 8],
            best_similarity: 0.9,
            best_idx: Some(0),
            n_matches: 1,
            expected_depth: 4.0,
            matched_indices: vec![0],
            blended_valence: 0.0,
        };

        prime_state(&mut outputs, &retrieval, 0.5, 0);

        // Region 0 should be blended: (1.0 * 0.5 + 0.5 * 0.5) = 0.75
        assert!((outputs[0][0] - 0.75).abs() < 0.01,
            "blended value should be 0.75, got {}", outputs[0][0]);
        // Region 1 should be unchanged
        assert!((outputs[1][0] - 2.0).abs() < 0.01,
            "unprimed region should be unchanged");
    }

    #[test]
    fn no_priming_when_no_matches() {
        let mut outputs = vec![vec![1.0; 8]];
        let retrieval = EpisodicRetrievalResult {
            blended_trajectory: vec![],
            blended_final_state: vec![],
            best_similarity: 0.0,
            best_idx: None,
            n_matches: 0,
            expected_depth: 0.0,
            matched_indices: vec![],
            blended_valence: 0.0,
        };

        prime_state(&mut outputs, &retrieval, 0.5, 0);
        assert!((outputs[0][0] - 1.0).abs() < 0.01, "should be unchanged with no matches");
    }
}
