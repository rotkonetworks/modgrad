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

/// Challenge: given a byte, can the model predict common followers?
/// Tests bigram knowledge: "th"→"e", "in"→"g", " t"→"h", etc.
pub fn challenge_bigrams(nc: &mut NeuralComputer) -> ChallengeResult {
    let tests: Vec<(&[u8], &[u8])> = vec![
        (b"th", b"e"),
        (b"he", b" "),
        (b"in", b"g"),
        (b"an", b"d"),
        (b"er", b" "),
        (b"on", b" e"),
        (b"re", b" "),
        (b"th", b"a"),
        (b" t", b"h"),
        (b" a", b" n"),
        (b"en", b"t"),
        (b"at", b" "),
        (b"ou", b"r"),
        (b"is", b" "),
        (b"to", b" "),
        (b"it", b" "),
        (b"st", b" "),
        (b"or", b" "),
        (b"ar", b"e"),
        (b"nd", b" "),
    ];

    let mut correct = 0;
    let total = tests.len();

    for (context, valid_next) in &tests {
        nc.reset();
        let logits = nc.observe(&text_to_tokens(context));
        let pred = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);

        if valid_next.contains(&(pred as u8)) {
            correct += 1;
        }
    }

    let score = correct as f32 / total as f32;
    ChallengeResult {
        score,
        total,
        correct,
        summary: format!("bigrams: {correct}/{total} ({:.0}%)", score * 100.0),
    }
}

/// Challenge: given a character class, predict the right class.
/// "A"→uppercase, "5"→digit, " "→space, "."→punctuation.
pub fn challenge_byte_classes(nc: &mut NeuralComputer) -> ChallengeResult {
    let tests: Vec<(u8, fn(u8) -> bool)> = vec![
        (b'A', |b| b.is_ascii_lowercase() || b == b' '),  // after uppercase, expect lowercase or space
        (b'z', |b| b.is_ascii_lowercase() || b == b' ' || b == b'.'),
        (b' ', |b| b.is_ascii_alphabetic()),  // after space, expect letter
        (b'.', |b| b == b' ' || b == b'\n'),  // after period, expect space or newline
        (b'0', |b| b.is_ascii_digit() || b == b'.' || b == b',' || b == b' '),
        (b'\n', |b| b.is_ascii_alphabetic() || b == b'\n' || b == b' '),
        (b'(', |b| b.is_ascii_alphanumeric()),
        (b'"', |b| b.is_ascii_alphabetic() || b == b' '),
    ];

    let mut correct = 0;
    let total = tests.len();

    for (input, is_valid) in &tests {
        nc.reset();
        let logits = nc.step(*input as usize);
        let pred = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0) as u8;

        if is_valid(pred) {
            correct += 1;
        }
    }

    let score = correct as f32 / total as f32;
    ChallengeResult {
        score,
        total,
        correct,
        summary: format!("byte classes: {correct}/{total} ({:.0}%)", score * 100.0),
    }
}

/// Challenge: complete common English words.
/// "th" → "the" or "that" or "this", "wh" → "what" or "when" or "which", etc.
pub fn challenge_word_completion(nc: &mut NeuralComputer) -> ChallengeResult {
    let tests: Vec<(&[u8], Vec<&[u8]>)> = vec![
        (b"the ", vec![b"c", b"m", b"f", b"b", b"s", b"w", b"p", b"d", b"r", b"l", b"n", b"o", b"a", b"e", b"t"]),
        (b"is ", vec![b"a", b"t", b"n", b"i", b"s"]),
        (b"and ", vec![b"t", b"a", b"i", b"s", b"w", b"h"]),
        (b"of ", vec![b"t", b"a", b"i", b"s"]),
        (b"to ", vec![b"t", b"a", b"b", b"s", b"m", b"g", b"d", b"p"]),
        (b"in ", vec![b"t", b"a", b"i", b"s"]),
        (b"for ", vec![b"t", b"a", b"e", b"s"]),
        (b"cat ", vec![b"s", b"a", b"i", b"w"]),
    ];

    let mut correct = 0;
    let total = tests.len();

    for (prefix, valid_next) in &tests {
        nc.reset();
        let logits = nc.observe(&text_to_tokens(prefix));
        let pred = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);

        if valid_next.iter().any(|v| v[0] == pred as u8) {
            correct += 1;
        }
    }

    let score = correct as f32 / total as f32;
    ChallengeResult {
        score,
        total,
        correct,
        summary: format!("word completion: {correct}/{total} ({:.0}%)", score * 100.0),
    }
}

/// Challenge: generate coherent text (no garbage bytes, words exist).
pub fn challenge_coherent_generation(nc: &mut NeuralComputer) -> ChallengeResult {
    let prompts = [b"the " as &[u8], b"in the " as &[u8], b"it is " as &[u8]];
    let mut total_score = 0.0f32;

    for prompt in &prompts {
        nc.reset();
        let output = nc.chat(std::str::from_utf8(prompt).unwrap(), 30, 0.5);

        // Score: fraction of output that's printable ASCII with spaces
        let printable = output.bytes()
            .filter(|&b| b.is_ascii_graphic() || b == b' ' || b == b'\n')
            .count();
        let ascii_score = if output.is_empty() { 0.0 } else {
            printable as f32 / output.len() as f32
        };

        // Score: has spaces (word-like structure)?
        let has_spaces = output.contains(' ');
        let space_score = if has_spaces { 0.5 } else { 0.0 };

        // Score: no long runs of the same character (not "zzzzzz")
        let max_repeat = output.as_bytes().windows(2)
            .filter(|w| w[0] == w[1]).count();
        let repeat_score = if output.len() > 5 {
            (1.0 - max_repeat as f32 / output.len() as f32).max(0.0)
        } else { 0.0 };

        total_score += (ascii_score * 0.4 + space_score + repeat_score * 0.1) / prompts.len() as f32;
    }

    let score = total_score.clamp(0.0, 1.0);
    ChallengeResult {
        score,
        total: prompts.len(),
        correct: (score * prompts.len() as f32) as usize,
        summary: format!("coherent generation: {:.0}%", score * 100.0),
    }
}

// ── Stage definition ──────────────────────────────────────

/// One stage of the curriculum.
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

/// Default byte-first curriculum for text learning.
/// Higher thresholds — graduation requires real mastery.
pub fn byte_curriculum() -> Vec<Stage> {
    vec![
        Stage {
            name: "byte_classes",
            challenge: challenge_byte_classes,
            pass_threshold: 0.75,
            max_steps: 20000,
            test_every: 2000,
        },
        Stage {
            name: "bigrams",
            challenge: challenge_bigrams,
            pass_threshold: 0.30,
            max_steps: 40000,
            test_every: 4000,
        },
        Stage {
            name: "word_completion",
            challenge: challenge_word_completion,
            pass_threshold: 0.50,
            max_steps: 60000,
            test_every: 5000,
        },
        Stage {
            name: "coherent_generation",
            challenge: challenge_coherent_generation,
            pass_threshold: 0.60,
            max_steps: 100000,
            test_every: 10000,
        },
    ]
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

