//! Level 3: Honest organism intelligence challenges.
//!
//! Unlike the architecture tests (Level 1-2) which use hand-crafted float
//! encodings, challenges feed RAW BYTES through the organism's full pipeline:
//!   embeddings → sensory MLP → 8-region CTM → output projection → byte prediction
//!
//! The organism must discover representations AND computations from scratch.
//! No pre-solved encodings, no feature engineering.
//!
//! Each challenge:
//!   1. Generates training sequences as raw byte strings
//!   2. Trains an organism from scratch using train_step()
//!   3. Tests on HELD-OUT examples (different concrete values, same structure)
//!   4. Measures next-byte prediction accuracy at critical positions
//!
//! This is the honest test of gradient-free learning.

use crate::organism::{Organism, Dna};

/// A byte-level challenge.
pub struct ByteChallenge {
    pub name: &'static str,
    pub description: &'static str,
    /// Training sequences (raw bytes, e.g., b"2+3=5\n4+1=5\n")
    pub train: Vec<Vec<u8>>,
    /// Test cases: (prompt_bytes, expected_next_byte)
    /// The organism sees the prompt and must predict the next byte.
    pub test: Vec<(Vec<u8>, u8)>,
}

/// Result of running a challenge.
pub struct ChallengeResult {
    pub name: &'static str,
    pub train_loss_start: f32,
    pub train_loss_end: f32,
    pub test_accuracy: f32,
    pub test_correct: usize,
    pub test_total: usize,
}

impl ChallengeResult {
    pub fn passed(&self) -> bool { self.test_accuracy >= 0.8 }
}

// ─── Challenge generators ──────────────────────────────────

/// Echo: input → same output. "abc\xff" → predict 'a','b','c'.
/// The simplest possible challenge. Can the organism copy bytes?
pub fn echo_challenge() -> ByteChallenge {
    let mut train = Vec::new();
    let mut test = Vec::new();

    // Training: various byte sequences with \xff delimiter
    // Pattern: bytes, \xff, same bytes repeated
    for seed in 0..50u8 {
        let a = b'a' + (seed % 26);
        let b = b'a' + ((seed + 7) % 26);
        let c = b'a' + ((seed + 13) % 26);
        let seq = vec![a, b, c, 0xFF, a, b, c];
        train.push(seq);
    }

    // Test: held-out combinations
    for seed in 50..70u8 {
        let a = b'a' + (seed % 26);
        let b = b'a' + ((seed + 3) % 26);
        // After seeing "xy\xff", should predict 'x'
        test.push((vec![a, b, 0xFF], a));
    }

    ByteChallenge {
        name: "Echo",
        description: "Copy bytes after delimiter. Tests: does the organism preserve input identity?",
        train, test,
    }
}

/// Single digit addition: "3+5=8", "2+7=9".
/// Tests: can the organism learn arithmetic from byte sequences?
pub fn addition_challenge() -> ByteChallenge {
    let mut train = Vec::new();
    let mut test = Vec::new();

    // Training: all single-digit additions where a+b < 10 (no carry)
    // Use most pairs for training, hold out some for test
    let mut pairs: Vec<(u8, u8)> = Vec::new();
    for a in 0..10u8 {
        for b in 0..10u8 {
            if a + b < 10 {
                pairs.push((a, b));
            }
        }
    }

    // 80% train, 20% test
    let split = pairs.len() * 4 / 5;
    for &(a, b) in &pairs[..split] {
        let sum = a + b;
        // "3+5=8\n" repeated for learning
        let seq = vec![b'0' + a, b'+', b'0' + b, b'=', b'0' + sum, b'\n'];
        for _ in 0..10 { // repetition for Hebbian
            train.push(seq.clone());
        }
    }
    for &(a, b) in &pairs[split..] {
        let sum = a + b;
        // Prompt: "3+5=" → should predict '8'
        test.push((vec![b'0' + a, b'+', b'0' + b, b'='], b'0' + sum));
    }

    ByteChallenge {
        name: "Addition (single digit, no carry)",
        description: "3+5=8. Tests: can the organism learn arithmetic from raw bytes?",
        train, test,
    }
}

/// Hex digit recognition: "0x4" → predict '1' for 0x41='A'.
/// Actually simpler: "A=41\n" style — learn ASCII↔hex mapping.
pub fn hex_value_challenge() -> ByteChallenge {
    let mut train = Vec::new();
    let mut test = Vec::new();

    // Training: character=hex pairs for printable ASCII
    // "A=41\n", "B=42\n", etc.
    let train_chars: Vec<u8> = (b'A'..=b'Z').chain(b'0'..=b'9').collect();
    let test_chars: Vec<u8> = (b'a'..=b'f').collect(); // held out lowercase hex

    let hex_digit = |n: u8| -> u8 {
        if n < 10 { b'0' + n } else { b'a' + n - 10 }
    };

    for &c in &train_chars {
        let hi = hex_digit(c >> 4);
        let lo = hex_digit(c & 0x0F);
        let seq = vec![c, b'=', hi, lo, b'\n'];
        for _ in 0..10 {
            train.push(seq.clone());
        }
    }

    // Test: given "a=", predict '6' (0x61, hi nibble = 6)
    for &c in &test_chars {
        let hi = hex_digit(c >> 4);
        test.push((vec![c, b'='], hi));
    }

    ByteChallenge {
        name: "Hex value (char→hex high nibble)",
        description: "A=41. Tests: can the organism learn ASCII↔hex mapping from examples?",
        train, test,
    }
}

/// Case flip: "a→A\n", "b→B\n". Test on held-out letters.
pub fn case_flip_challenge() -> ByteChallenge {
    let mut train = Vec::new();
    let mut test = Vec::new();

    // Train on first 20 lowercase letters
    for i in 0..20u8 {
        let lo = b'a' + i;
        let up = b'A' + i;
        let seq = vec![lo, b'>', up, b'\n'];
        for _ in 0..20 {
            train.push(seq.clone());
        }
    }

    // Test on last 6 lowercase letters
    for i in 20..26u8 {
        let lo = b'a' + i;
        let up = b'A' + i;
        test.push((vec![lo, b'>'], up));
    }

    ByteChallenge {
        name: "Case flip (a→A)",
        description: "a>A. Tests: can the organism learn a byte transformation and generalize?",
        train, test,
    }
}

/// Successor: "3→4\n", "7→8\n". Test on held-out digits.
pub fn successor_challenge() -> ByteChallenge {
    let mut train = Vec::new();
    let mut test = Vec::new();

    // Train on 0-6
    for d in 0..7u8 {
        let seq = vec![b'0' + d, b'>', b'0' + d + 1, b'\n'];
        for _ in 0..30 {
            train.push(seq.clone());
        }
    }

    // Test on 7,8 (held out)
    for d in 7..9u8 {
        test.push((vec![b'0' + d, b'>'], b'0' + d + 1));
    }

    ByteChallenge {
        name: "Successor (3→4)",
        description: "3>4. Tests: can the organism learn +1 and generalize to unseen digits?",
        train, test,
    }
}

/// Variable binding: "x=5\ny=3\nx?" → should predict '5'.
pub fn binding_challenge() -> ByteChallenge {
    let mut train = Vec::new();
    let mut test = Vec::new();

    // Train: single variable binding + query
    let train_names = b"abcd";
    let test_names = b"ef";

    for &name in train_names {
        for val in b'0'..=b'9' {
            let seq = vec![name, b'=', val, b'\n', name, b'?', val, b'\n'];
            for _ in 0..5 {
                train.push(seq.clone());
            }
        }
    }

    // Test: held-out variable names with known values
    for &name in test_names {
        for val in b'0'..=b'5' {
            // Prompt includes the binding + query
            test.push((vec![name, b'=', val, b'\n', name, b'?'], val));
        }
    }

    ByteChallenge {
        name: "Variable binding (x=5, x?→5)",
        description: "x=5\\nx?5. Tests: can the organism bind and retrieve values from raw bytes?",
        train, test,
    }
}

// ─── Challenge runner ──────────────────────────────────────

/// Train an organism on a challenge and evaluate.
pub fn run_challenge(challenge: &ByteChallenge, dna: Dna, train_epochs: usize) -> ChallengeResult {
    let mut org = Organism::new(dna);

    // Shuffle-free training: just iterate epochs over train data
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    let sleep_every = 30;

    for epoch in 0..train_epochs {
        let mut epoch_loss = 0.0f32;
        let mut n = 0;

        for seq in &challenge.train {
            let token_ids: Vec<usize> = seq.iter().map(|&b| b as usize).collect();
            if token_ids.len() < 2 { continue; }
            let loss = org.train_step(&token_ids);
            epoch_loss += loss;
            n += 1;

            if n % sleep_every == 0 && n > 0 {
                org.sleep();
            }
        }

        let avg = if n > 0 { epoch_loss / n as f32 } else { 0.0 };
        if epoch == 0 { first_loss = avg; }
        last_loss = avg;

        if epoch % 5 == 0 || epoch == train_epochs - 1 {
            eprintln!("    epoch {epoch:3}: loss={avg:.3}");
        }
    }

    // Final sleep to consolidate
    org.sleep();

    // Evaluate on held-out test cases
    let mut correct = 0;
    for (prompt, expected) in &challenge.test {
        let token_ids: Vec<usize> = prompt.iter().map(|&b| b as usize).collect();
        let (logits, _syncs) = org.forward_inner(&token_ids, false);

        // The last logit predicts the next byte
        if let Some(last_logits) = logits.last() {
            let predicted = last_logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            if predicted == *expected as usize {
                correct += 1;
            }
        }
    }

    let total = challenge.test.len();
    let accuracy = if total > 0 { correct as f32 / total as f32 } else { 0.0 };

    ChallengeResult {
        name: challenge.name,
        train_loss_start: first_loss,
        train_loss_end: last_loss,
        test_accuracy: accuracy,
        test_correct: correct,
        test_total: total,
    }
}

/// Run all challenges.
pub fn run_all_challenges(dna_name: &str) {
    let dna = match dna_name {
        "tiny" => Dna::tiny(),
        "medium" => Dna::medium(),
        "medium_plus" | "medplus" => Dna::medium_plus(),
        "large" => Dna::large(),
        _ => Dna::small(),
    };

    eprintln!("\n=== LEVEL 3: ORGANISM INTELLIGENCE CHALLENGES ===");
    eprintln!("DNA: {dna_name}, raw bytes, no hand-crafted features\n");

    let challenges: Vec<(ByteChallenge, usize)> = vec![
        (echo_challenge(), 5),
        (successor_challenge(), 10),
        (case_flip_challenge(), 10),
        (addition_challenge(), 15),
        (hex_value_challenge(), 15),
        (binding_challenge(), 15),
    ];

    let mut passed = 0;
    let total = challenges.len();

    for (challenge, epochs) in &challenges {
        eprintln!("── {} ──", challenge.name);
        eprintln!("  {}", challenge.description);
        eprintln!("  train: {} sequences, test: {} cases, {} epochs",
            challenge.train.len(), challenge.test.len(), epochs);

        let result = run_challenge(challenge, dna.clone(), *epochs);

        let status = if result.passed() { passed += 1; "PASS" } else { "FAIL" };
        eprintln!("  [{status}] {:.1}% ({}/{}) loss: {:.3}→{:.3}\n",
            result.test_accuracy * 100.0,
            result.test_correct, result.test_total,
            result.train_loss_start, result.train_loss_end);
    }

    eprintln!("=== RESULT: {passed}/{total} challenges passed ===\n");
}
