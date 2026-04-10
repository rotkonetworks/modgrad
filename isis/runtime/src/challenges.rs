//! isis challenges: English-specific test games and training loop.
//!
//! The SDK provides `evaluate()` and `TestCase` — generic primitives.
//! This module provides isis-specific challenges, stages, and the training loop.

use modgrad_ctm::graph::*;
use modgrad_ctm::curriculum::{self, ChallengeResult, TestCase};

/// One stage of isis curriculum.
pub struct Stage {
    pub name: &'static str,
    pub challenge: fn(&mut NeuralComputer) -> ChallengeResult,
    pub pass_threshold: f32,
    pub max_steps: usize,
    pub test_every: usize,
}

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

/// Challenge: generate text containing real English words.
///
/// Not "is it printable ASCII" — that's trivial.
/// Actually checks: do the generated words exist in a basic English vocabulary?
pub fn challenge_coherent_generation(nc: &mut NeuralComputer) -> ChallengeResult {
    // 100 most common English words — the model must produce THESE, not random ASCII
    const COMMON_WORDS: &[&str] = &[
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "come", "could", "than", "look", "day", "had", "has", "was", "is", "are",
        "were", "been", "did", "its", "into", "our", "then", "them", "very",
        "cat", "sat", "on", "mat", "dog", "ran", "old", "new", "big", "red",
        "man", "see", "now", "way", "may", "how", "two", "did", "got", "let",
    ];

    let prompts = ["the ", "in the ", "it is ", "he was ", "she said "];
    let mut total_words = 0usize;
    let mut real_words = 0usize;

    for prompt in &prompts {
        nc.reset();
        let output = nc.chat(prompt, 40, 0.5);

        // Split on spaces and check each token against vocabulary
        let words: Vec<&str> = output.split_whitespace()
            .filter(|w| w.len() >= 1 && w.len() <= 10)
            .collect();

        for word in &words {
            total_words += 1;
            // Strip trailing punctuation for matching
            let clean: String = word.chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
                .to_lowercase();
            if clean.len() >= 1 && COMMON_WORDS.contains(&clean.as_str()) {
                real_words += 1;
            }
        }
    }

    let score = if total_words > 0 {
        real_words as f32 / total_words as f32
    } else {
        0.0
    };

    ChallengeResult {
        score,
        total: total_words,
        correct: real_words,
        summary: format!("coherent generation: {real_words}/{total_words} real words ({:.0}%)", score * 100.0),
    }
}

// ── Stage definition ──────────────────────────────────────

/// One stage of the curriculum.

/// Default byte-first curriculum for English text learning.
/// Graduation requires real mastery — not low loss, demonstrated capability.
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

/// Run staged training with capability gates.
/// Returns highest stage graduated.
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
        let mut data = Vec::new();
        for d in &stage_data[..=si] { data.extend_from_slice(d); }

        opt.lr = base_lr * 0.7f32.powi(si as i32);
        let stage = &stages[si];
        log(&format!("\n=== Stage {si}: {} (pass: {:.0}%, lr: {:.5}) ===",
            stage.name, stage.pass_threshold * 100.0, opt.lr));

        let mut stage_step = 0;
        let mut graduated = false;

        while stage_step < stage.max_steps {
            let offset = (global_step * context_len) % data.len().saturating_sub(context_len + 1).max(1);
            let end = (offset + context_len + 1).min(data.len());
            if end - offset < context_len + 1 { global_step += 1; stage_step += 1; continue; }
            let chunk = &data[offset..end];

            grads.zero();
            let mut loss = 0.0f32;
            let mut correct = 0;
            for pos in 0..context_len {
                let (l, pred) = regional_train_token(w, &mut grads, chunk[pos] as usize, chunk[pos+1] as usize);
                loss += l;
                if pred == chunk[pos+1] as usize { correct += 1; }
            }
            opt.step(w, &grads);
            loss /= context_len as f32;
            global_step += 1;
            stage_step += 1;

            if stage_step % 500 == 0 {
                log(&format!("  step {stage_step}: loss={loss:.3} acc={correct}/{context_len}"));
            }

            if stage_step % stage.test_every == 0 {
                let mut nc = NeuralComputer::new(w.clone());
                let result = (stage.challenge)(&mut nc);
                let marker = if result.passed(stage.pass_threshold) { "PASS" } else { "FAIL" };
                log(&format!("  TEST [{marker}]: {}", result.summary));

                if result.passed(stage.pass_threshold) {
                    log(&format!("  GRADUATED {} at step {stage_step}!", stage.name));
                    graduated = true;
                    break;
                }
            }
        }

        if !graduated {
            log(&format!("  {} not graduated after {} steps", stage.name, stage.max_steps));
            opt.lr = base_lr;
            return si;
        }
    }

    opt.lr = base_lr;
    stages.len()
}
