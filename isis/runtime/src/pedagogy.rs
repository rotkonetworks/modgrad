//! Pedagogy: interactive teaching with scaffolding, testing, and correction.
//!
//! Models how humans learn language:
//! 1. Child TRIES to produce output
//! 2. Parent CORRECTS: "no, not X, say Y"
//! 3. Child adjusts weights toward correct answer
//! 4. Repetition with variation until mastery
//! 5. Test → advance or repeat
//!
//! This is NOT passive absorption. The organism actively generates,
//! fails, gets corrected, and tries again. Like a baby learning to speak.

use super::organism::Organism;

/// One teaching interaction: prompt → child tries → correction.
#[derive(Debug)]
pub struct Correction {
    pub prompt: Vec<u8>,
    pub child_said: u8,
    pub correct: u8,
    pub was_right: bool,
}

/// Mastery level for a curriculum phase.
#[derive(Debug)]
pub struct MasteryTest {
    pub total: usize,
    pub correct: usize,
    pub accuracy: f32,
    pub passed: bool,
}

/// Teach the organism interactively with a backbone parent.
/// Returns corrections made and final mastery test.
pub fn teach_interactive(
    org: &mut Organism,
    prompts: &[&[u8]],        // teaching prompts
    correct_nexts: &[u8],      // what should follow each prompt
    repetitions: usize,        // how many times to drill each
    mastery_threshold: f32,    // accuracy needed to pass (0.8 = 80%)
) -> (Vec<Correction>, MasteryTest) {
    let mut corrections = Vec::new();

    for _rep in 0..repetitions {
        for (prompt, &correct) in prompts.iter().zip(correct_nexts) {
            // Child tries to generate
            let output = org.generate(prompt, 1);
            let child_said = output.first().copied().unwrap_or(0);
            let was_right = child_said == correct;

            corrections.push(Correction {
                prompt: prompt.to_vec(),
                child_said,
                correct,
                was_right,
            });

            // Correction: adjust output weights directly
            // This is the "no, not X, say Y" signal
            if !was_right {
                correct_output(org, prompt, correct, child_said);
            } else {
                // Reward: strengthen the correct mapping
                reward_output(org, prompt, correct);
            }
        }

        // Sleep after each repetition — consolidate corrections
        org.sleep();
    }

    // Mastery test: can the child get them right now?
    let test = test_mastery(org, prompts, correct_nexts, mastery_threshold);

    (corrections, test)
}

/// Correct the organism: weaken wrong answer, strengthen right answer.
/// Uses the SYNC SIGNAL from the prompt — this is what the CTM computed.
fn correct_output(org: &mut Organism, prompt: &[u8], correct: u8, wrong: u8) {
    let ids: Vec<usize> = prompt.iter().map(|&b| b as usize).collect();
    let (_, syncs) = org.forward_inner(&ids, false);

    if let Some(sync) = syncs.last() {
        let out = &mut org.output_proj;
        let in_d = out.in_dim;
        let correct_idx = correct as usize;
        let wrong_idx = wrong as usize;

        // DIRECT SET: don't nudge, IMPRINT the sync pattern.
        // Like a parent physically moving the child's mouth.
        // The correct token's weight row becomes the sync pattern itself.
        // Blend 50% old + 50% sync to avoid total overwrite.
        for k in 0..in_d.min(sync.len()) {
            out.weight[correct_idx * in_d + k] =
                0.5 * out.weight[correct_idx * in_d + k] + 0.5 * sync[k];
            // Actively anti-correlate the wrong token
            out.weight[wrong_idx * in_d + k] =
                0.5 * out.weight[wrong_idx * in_d + k] - 0.3 * sync[k];
        }
        out.bias[correct_idx] += 0.5;
        out.bias[wrong_idx] -= 0.3;
    }
}

/// Reward the organism for correct output: strengthen the mapping.
fn reward_output(org: &mut Organism, prompt: &[u8], correct: u8) {
    let ids: Vec<usize> = prompt.iter().map(|&b| b as usize).collect();
    let (_, syncs) = org.forward_inner(&ids, false);

    if let Some(sync) = syncs.last() {
        let out = &mut org.output_proj;
        let in_d = out.in_dim;
        let correct_idx = correct as usize;

        // Gentle reinforcement: "yes, good!"
        let lr = 0.05;
        for k in 0..in_d.min(sync.len()) {
            out.weight[correct_idx * in_d + k] += lr * sync[k];
        }
        out.bias[correct_idx] += lr * 0.5;
    }
}

/// Test mastery: how many prompts can the child complete correctly?
pub fn test_mastery(
    org: &mut Organism,
    prompts: &[&[u8]],
    correct_nexts: &[u8],
    threshold: f32,
) -> MasteryTest {
    let mut correct = 0;
    let total = prompts.len();

    for (prompt, &expected) in prompts.iter().zip(correct_nexts) {
        let output = org.generate(prompt, 1);
        let got = output.first().copied().unwrap_or(0);
        if got == expected { correct += 1; }
    }

    let accuracy = correct as f32 / total.max(1) as f32;
    MasteryTest {
        total,
        correct,
        accuracy,
        passed: accuracy >= threshold,
    }
}

/// Full staged pedagogy: babbling → words → sentences.
/// Advances only when mastery is demonstrated.
pub fn full_pedagogy(
    org: &mut Organism,
    max_attempts_per_phase: usize,
) {
    // Phase 1: Single character after "the "
    eprintln!("\n=== Phase 1: What comes after 'the '? ===");
    let p1_prompts: Vec<&[u8]> = vec![b"the ", b"the ", b"the ", b"the ", b"the "];
    let p1_correct = vec![b'c', b'd', b'b', b'f', b'm']; // cat, dog, bird, fish, man

    for attempt in 0..max_attempts_per_phase {
        let (corrections, test) = teach_interactive(
            org, &p1_prompts, &p1_correct, 5, 0.6);

        let n_right = corrections.iter().filter(|c| c.was_right).count();
        eprintln!("  Attempt {}: {}/{} correct during training, test: {}/{}={:.0}%{}",
            attempt + 1, n_right, corrections.len(),
            test.correct, test.total, test.accuracy * 100.0,
            if test.passed { " PASSED" } else { "" });

        if test.passed {
            eprintln!("  Phase 1 MASTERED");
            break;
        }
    }

    // Phase 2: Complete common words
    eprintln!("\n=== Phase 2: Complete the word ===");
    let p2_prompts: Vec<&[u8]> = vec![
        b"ca", b"do", b"th", b"bi", b"sa", b"ra",
    ];
    let p2_correct = vec![b't', b'g', b'e', b'r', b't', b'n']; // cat, dog, the, bird, sat, ran

    for attempt in 0..max_attempts_per_phase {
        let (corrections, test) = teach_interactive(
            org, &p2_prompts, &p2_correct, 5, 0.6);

        let n_right = corrections.iter().filter(|c| c.was_right).count();
        eprintln!("  Attempt {}: {}/{} correct, test: {}/{}={:.0}%{}",
            attempt + 1, n_right, corrections.len(),
            test.correct, test.total, test.accuracy * 100.0,
            if test.passed { " PASSED" } else { "" });

        if test.passed {
            eprintln!("  Phase 2 MASTERED");
            break;
        }
    }

    // Phase 3: Next word after article
    eprintln!("\n=== Phase 3: What word follows 'the '? ===");
    let p3_prompts: Vec<&[u8]> = vec![
        b"the c", b"the d", b"the b", b"the m",
    ];
    let p3_correct = vec![b'a', b'o', b'i', b'a']; // cat, dog, bird, man

    for attempt in 0..max_attempts_per_phase {
        let (corrections, test) = teach_interactive(
            org, &p3_prompts, &p3_correct, 10, 0.5);

        let n_right = corrections.iter().filter(|c| c.was_right).count();
        eprintln!("  Attempt {}: {}/{} correct, test: {}/{}={:.0}%{}",
            attempt + 1, n_right, corrections.len(),
            test.correct, test.total, test.accuracy * 100.0,
            if test.passed { " PASSED" } else { "" });

        if test.passed {
            eprintln!("  Phase 3 MASTERED");
            break;
        }
    }

    // Final test
    eprintln!("\n=== Final Generation Test ===");
    for prompt in &[b"the " as &[u8], b"the cat ", b"once ", b"he "] {
        let output = org.generate(prompt, 20);
        let text = String::from_utf8_lossy(&output);
        let prompt_str = std::str::from_utf8(prompt).unwrap_or("?");
        eprintln!("  \"{prompt_str}\" → \"{text}\"");
    }
}
