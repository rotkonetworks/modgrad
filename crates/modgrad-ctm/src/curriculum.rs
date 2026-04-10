//! Capability-gated curriculum: the model earns advancement by passing tests.
//!
//! No loss thresholds. No arbitrary step counts. The model must demonstrate
//! actual capability before advancing to the next stage.
//!
//! Each stage has:
//!   - Training data generator
//!   - A challenge (test game) that gates advancement
//!   - The challenge returns a score; pass threshold is configurable
//!
//! The stage order is configurable. Default is byte→text (computer-native).
//! Human-inspired order would be: voice→image→video→byte→text.
//! The pipeline doesn't care — it runs stages in whatever order you give it.
//!
//! Design: each challenge is a pure function. No traits, no dynamic dispatch.
//! Compose them however you want.

use crate::graph::*;

// ── Challenge results ─────────────────────────────────────

/// Result of running a challenge.
#[derive(Debug, Clone)]
pub struct ChallengeResult {
    /// Score from 0.0 (total failure) to 1.0 (perfect).
    pub score: f32,
    /// Number of test cases attempted.
    pub total: usize,
    /// Number correct.
    pub correct: usize,
    /// Human-readable summary.
    pub summary: String,
}

impl ChallengeResult {
    pub fn passed(&self, threshold: f32) -> bool {
        self.score >= threshold
    }
}

// ── Built-in challenges ───────────────────────────────────

pub struct Stage {
    pub name: &'static str,
    /// Challenge function that gates advancement.
    pub challenge: fn(&mut NeuralComputer) -> ChallengeResult,
    /// Score needed to pass (0.0 - 1.0).
    pub pass_threshold: f32,
    /// Maximum training steps before giving up on this stage.
    pub max_steps: usize,
    /// How often to run the challenge (every N steps).
    pub test_every: usize,
}

/// Graduation result for one test cycle.
#[derive(Debug)]
pub struct GraduationReport {
    /// Per-stage scores (index = stage index).
    pub scores: Vec<ChallengeResult>,
    /// Did all stages up to and including `target_stage` pass?
    pub graduated: bool,
    /// Which stage (if any) regressed below threshold?
    pub regression: Option<usize>,
}

/// Run the full curriculum with graduation gates.
///
/// To graduate from stage N, the model must:
///   1. Pass stage N's challenge above threshold
///   2. ALSO pass ALL stages 0..N-1 (no regression)
///   3. If any previous stage regresses, retrain on mixed data until recovered
///
/// Data accumulates: stage N trains on data[0] + data[1] + ... + data[N].
/// LR decays 0.7× per stage to protect earlier learning.
///
/// `stage_data`: per-stage training data. Stage i trains on concat(stage_data[0..=i]).
/// Returns the highest stage graduated.
pub fn run_curriculum(
    w: &mut RegionalWeights,
    opt: &mut RegionalAdamW,
    stage_data: &[Vec<u8>],
    stages: &[Stage],
    context_len: usize,
    log: &mut dyn FnMut(&str),
) -> usize {
    assert_eq!(stage_data.len(), stages.len(), "need one data vec per stage");
    let mut grads = RegionalGradients::zeros(w);
    let mut global_step = 0usize;
    let base_lr = opt.lr;

    for si in 0..stages.len() {
        // Accumulate data: train on ALL stages up to current
        let mut data = Vec::new();
        for d in &stage_data[..=si] {
            data.extend_from_slice(d);
        }

        // Decay LR per stage
        opt.lr = base_lr * 0.7f32.powi(si as i32);

        let stage = &stages[si];
        log(&format!("\n=== Stage {si}: {} (pass: {:.0}%, max: {} steps, lr: {:.5}) ===",
            stage.name, stage.pass_threshold * 100.0, stage.max_steps, opt.lr));
        log(&format!("  Training on {:.1}KB accumulated data", data.len() as f64 / 1024.0));

        let mut stage_step = 0;
        let mut graduated = false;

        while stage_step < stage.max_steps {
            // Train one step on accumulated data
            let offset = (global_step * context_len) % data.len().saturating_sub(context_len + 1).max(1);
            let end = (offset + context_len + 1).min(data.len());
            if end - offset < context_len + 1 { global_step += 1; stage_step += 1; continue; }
            let chunk = &data[offset..end];

            grads.zero();
            let mut loss = 0.0f32;
            let mut correct = 0;
            for pos in 0..context_len {
                let token = chunk[pos] as usize;
                let target = chunk[pos + 1] as usize;
                let (l, pred) = regional_train_token(w, &mut grads, token, target);
                loss += l;
                if pred == target { correct += 1; }
            }
            opt.step(w, &grads);

            loss /= context_len as f32;
            global_step += 1;
            stage_step += 1;

            if stage_step % 500 == 0 {
                log(&format!("  step {stage_step}: loss={loss:.3} acc={correct}/{context_len}"));
            }

            // Graduation exam
            if stage_step % stage.test_every == 0 {
                let report = graduation_exam(w, stages, si);

                // Log all scores
                for (i, r) in report.scores.iter().enumerate() {
                    let marker = if r.passed(stages[i].pass_threshold) { "✓" } else { "✗" };
                    log(&format!("  {marker} {}: {}", stages[i].name, r.summary));
                }

                if report.graduated {
                    log(&format!("  ★ GRADUATED {} at step {stage_step}!", stage.name));
                    graduated = true;
                    break;
                } else if let Some(reg) = report.regression {
                    log(&format!("  ↓ regression in {} — continuing training", stages[reg].name));
                }
            }
        }

        if !graduated {
            log(&format!("  Stage {} not graduated after {} steps", stage.name, stage.max_steps));
            // Restore LR
            opt.lr = base_lr;
            return si;
        }
    }

    opt.lr = base_lr;
    stages.len()
}

/// Run graduation exam: test current stage AND all previous stages.
fn graduation_exam(
    w: &RegionalWeights,
    stages: &[Stage],
    target_stage: usize,
) -> GraduationReport {
    let mut scores = Vec::new();
    let mut all_passed = true;
    let mut regression = None;

    for i in 0..=target_stage {
        let mut nc = NeuralComputer::new(w.clone());
        let result = (stages[i].challenge)(&mut nc);
        let passed = result.passed(stages[i].pass_threshold);
        if !passed {
            all_passed = false;
            if i < target_stage && regression.is_none() {
                regression = Some(i);
            }
        }
        scores.push(result);
    }

    GraduationReport {
        scores,
        graduated: all_passed,
        regression,
    }
}

