//! Capability-gated training primitives.
//!
//! The SDK provides types and evaluation. Runtimes provide the loop.
//!
//! ```rust
//! // Runtime code — NOT in the SDK:
//! let tests = load_tests("bigrams.json");
//! loop {
//!     train_step(&mut w, &mut grads, token, target);
//!     opt.step(&mut w, &grads);
//!     if step % 1000 == 0 {
//!         let result = evaluate(&mut nc, &tests);
//!         if result.passed(0.75) { break; }
//!     }
//! }
//! ```

use crate::graph::NeuralComputer;

// ── Types ─────────────────────────────────────────────────

/// Result of evaluating a model against a set of tests.
#[derive(Debug, Clone)]
pub struct ChallengeResult {
    pub score: f32,
    pub total: usize,
    pub correct: usize,
    pub summary: String,
}

impl ChallengeResult {
    pub fn passed(&self, threshold: f32) -> bool {
        self.score >= threshold
    }
}

/// One test case: feed context tokens, check if predicted token is in the accept set.
pub struct TestCase {
    /// Tokens to feed as context.
    pub context: Vec<usize>,
    /// Any of these tokens counts as correct.
    pub accept: Vec<usize>,
}

// ── Evaluation ────────────────────────────────────────────

/// Evaluate: feed each test's context, check if argmax prediction is accepted.
///
/// This is the general-purpose evaluator. Works for any modality, any language,
/// any task. The runtime provides the test cases.
///
/// ```rust
/// let tests = vec![
///     TestCase { context: vec![116, 104], accept: vec![101] },  // "th" → "e"
///     TestCase { context: vec![32, 116], accept: vec![104] },   // " t" → "h"
/// ];
/// let result = evaluate(&mut nc, &tests);
/// ```
pub fn evaluate(nc: &mut NeuralComputer, tests: &[TestCase]) -> ChallengeResult {
    let mut correct = 0;
    let total = tests.len();

    for test in tests {
        nc.reset();
        let logits = nc.observe(&test.context);
        let pred = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if test.accept.contains(&pred) {
            correct += 1;
        }
    }

    let score = if total > 0 { correct as f32 / total as f32 } else { 0.0 };
    ChallengeResult {
        score,
        total,
        correct,
        summary: format!("{correct}/{total} ({:.0}%)", score * 100.0),
    }
}

/// Evaluate text generation: feed prompt, check what fraction of
/// generated words appear in a provided vocabulary.
///
/// `vocab`: set of acceptable words (lowercase).
/// `prompts`: text prompts to complete.
/// `max_tokens`: how many tokens to generate per prompt.
pub fn evaluate_generation(
    nc: &mut NeuralComputer,
    prompts: &[&str],
    vocab: &[&str],
    max_tokens: usize,
) -> ChallengeResult {
    let mut total_words = 0usize;
    let mut real_words = 0usize;

    for prompt in prompts {
        nc.reset();
        let output = nc.chat(prompt, max_tokens, 0.5);

        for word in output.split_whitespace() {
            let clean: String = word.chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
                .to_lowercase();
            if clean.is_empty() { continue; }
            total_words += 1;
            if vocab.contains(&clean.as_str()) {
                real_words += 1;
            }
        }
    }

    let score = if total_words > 0 { real_words as f32 / total_words as f32 } else { 0.0 };
    ChallengeResult {
        score,
        total: total_words,
        correct: real_words,
        summary: format!("{real_words}/{total_words} valid words ({:.0}%)", score * 100.0),
    }
}
