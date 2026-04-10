//! Neuvola: word-level interactive pedagogy.
//!
//! Named after the Finnish maternity clinic system that follows
//! child development with staged milestones and targeted support.
//!
//! Key differences from the old pedagogy module:
//! 1. WORD-level, not byte-level. "the" is one token, not three bytes.
//! 2. STRONG corrections: directly SET output weights, don't nudge.
//! 3. STAGED milestones with mastery testing between phases.
//! 4. REPETITION: same material drilled hundreds of times.

use super::organism::Organism;
use modgrad_persist::vocab::Vocab;
use serde::Deserialize;

/// Teach the organism at word level with strong corrections.
/// Returns accuracy at each phase.
pub fn teach_words(
    org: &mut Organism,
    vocab: &Vocab,
    phases: &[Phase],
    reps_per_phase: usize,
    _mastery: f32,  // per-phase mastery overrides this
) -> Vec<f32> {
    let mut accuracies = Vec::new();

    for (pi, phase) in phases.iter().enumerate() {
        eprintln!("\n=== Neuvola Phase {}: {} ===", pi, phase.name);
        if phase.drills.is_empty() {
            eprintln!("  (no drills — skipping)");
            accuracies.push(0.0);
            continue;
        }

        for attempt in 0..phase.max_attempts {
            let mut correct = 0;
            let mut total = 0;

            for _ in 0..reps_per_phase {
                for (prompt_ids, target_id) in &phase.drills {
                    // Child tries
                    let (_, syncs) = org.forward_inner(prompt_ids, false);
                    let sync = syncs.last().cloned().unwrap_or_default();
                    let logits = org.output_proj.forward(&sync);

                    // Only consider vocab tokens (not byte IDs)
                    let predicted = logits.iter().enumerate()
                        .take(vocab.size())
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    if predicted == *target_id {
                        correct += 1;
                        // Reward: gentle reinforcement
                        reinforce_output(org, &sync, *target_id, 0.3);
                    } else {
                        // STRONG correction: directly imprint the sync→target mapping
                        imprint_output(org, &sync, *target_id, predicted);
                    }
                    total += 1;
                }

                // Sleep after each rep
                org.sleep();
            }

            let acc = correct as f32 / total.max(1) as f32;
            eprintln!("  Attempt {}: {}/{} = {:.0}%{}",
                attempt + 1, correct, total, acc * 100.0,
                if acc >= phase.mastery { " PASSED" } else { "" });

            if acc >= phase.mastery {
                eprintln!("  Phase {} MASTERED", pi);
                break;
            }
        }

        // Test after phase
        let acc = test_phase(org, vocab, phase);
        accuracies.push(acc);
        eprintln!("  Final test: {:.0}%", acc * 100.0);
    }

    accuracies
}

/// STRONG correction: directly SET the output weight row for the correct token
/// to match the sync pattern. Not blend, not nudge — IMPRINT.
/// Like a parent physically forming the child's mouth to say the word.
fn imprint_output(org: &mut Organism, sync: &[f32], correct: usize, wrong: usize) {
    let out = &mut org.output_proj;
    let in_d = out.in_dim;

    if correct >= out.out_dim || wrong >= out.out_dim { return; }

    // DIRECT SET: correct token's weight row = sync pattern (normalized)
    let sync_norm: f32 = sync.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for k in 0..in_d.min(sync.len()) {
        let normalized = sync[k] / sync_norm;
        // 70% new (sync pattern), 30% old (don't totally erase)
        out.weight[correct * in_d + k] = 0.3 * out.weight[correct * in_d + k] + 0.7 * normalized;
    }
    out.bias[correct] = 1.0; // strong positive bias

    // ANTI-CORRELATE: wrong token's weights pushed AWAY from this sync
    for k in 0..in_d.min(sync.len()) {
        let normalized = sync[k] / sync_norm;
        out.weight[wrong * in_d + k] -= 0.3 * normalized;
    }
    out.bias[wrong] -= 0.3;
}

/// Gentle reinforcement for correct output.
fn reinforce_output(org: &mut Organism, sync: &[f32], correct: usize, strength: f32) {
    let out = &mut org.output_proj;
    let in_d = out.in_dim;

    if correct >= out.out_dim { return; }

    let sync_norm: f32 = sync.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for k in 0..in_d.min(sync.len()) {
        out.weight[correct * in_d + k] += strength * sync[k] / sync_norm;
    }
    out.bias[correct] += strength * 0.5;
}

/// Test phase accuracy without corrections.
fn test_phase(org: &mut Organism, vocab: &Vocab, phase: &Phase) -> f32 {
    let mut correct = 0;
    let total = phase.drills.len();

    for (prompt_ids, target_id) in &phase.drills {
        let (_, syncs) = org.forward_inner(prompt_ids, false);
        let sync = syncs.last().cloned().unwrap_or_default();
        let logits = org.output_proj.forward(&sync);

        let predicted = logits.iter().enumerate()
            .take(vocab.size())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        if predicted == *target_id {
            correct += 1;
        }
    }

    correct as f32 / total.max(1) as f32
}

/// A teaching phase with drills.
pub struct Phase {
    pub name: String,
    /// Each drill: (prompt as word IDs, correct next word ID)
    pub drills: Vec<(Vec<usize>, usize)>,
    pub mastery: f32,
    pub max_attempts: usize,
}

/// JSON curriculum format — loadable from file.
#[derive(Deserialize)]
pub struct CurriculumJson {
    pub vocab: Vec<String>,
    pub phases: Vec<PhaseJson>,
}

#[derive(Deserialize)]
pub struct PhaseJson {
    pub name: String,
    pub mastery: f32,
    pub max_attempts: usize,
    pub drills: Vec<DrillJson>,
}

#[derive(Deserialize)]
pub struct DrillJson {
    pub prompt: Vec<String>,
    pub target: String,
    #[serde(default)]
    pub frequency: u32,
}

/// Load curriculum from JSON file and convert to phases with word IDs.
pub fn load_curriculum(path: &str, vocab: &Vocab) -> Vec<Phase> {
    let data = std::fs::read_to_string(path).expect("can't read curriculum");
    let cj: CurriculumJson = serde_json::from_str(&data).expect("invalid curriculum JSON");

    cj.phases.iter().map(|pj| {
        let drills: Vec<(Vec<usize>, usize)> = pj.drills.iter()
            .filter_map(|d| {
                let prompt_ids: Vec<usize> = d.prompt.iter()
                    .map(|w| *vocab.word_to_id.get(w).unwrap_or(&0))
                    .collect();
                let target_id = *vocab.word_to_id.get(&d.target).unwrap_or(&0);
                // Skip drills with unknown words
                if prompt_ids.contains(&0) || target_id == 0 { return None; }
                Some((prompt_ids, target_id))
            })
            .collect();

        Phase {
            name: format!("{} ({} drills)", pj.name, drills.len()),
            drills,
            mastery: pj.mastery,
            max_attempts: pj.max_attempts,
        }
    }).collect()
}

/// Build comprehensive neuvola curriculum.
/// Hundreds of patterns across graded phases.
pub fn standard_curriculum(vocab: &Vocab) -> Vec<Phase> {
    let w = |s: &str| -> usize { *vocab.word_to_id.get(s).unwrap_or(&0) };

    // Only include drills where all words are in vocab
    let has = |s: &str| -> bool { vocab.word_to_id.contains_key(s) };
    let drill = |prompt: &[&str], target: &str| -> Option<(Vec<usize>, usize)> {
        if prompt.iter().all(|p| has(p)) && has(target) {
            Some((prompt.iter().map(|p| w(p)).collect(), w(target)))
        } else {
            None
        }
    };

    let nouns = ["cat", "dog", "man", "world", "time", "way", "day", "people",
                 "water", "part", "year", "place", "case", "work", "point",
                 "number", "home", "hand", "life", "child"];
    let _verbs = ["is", "are", "was", "have", "has", "can", "will", "may",
                 "be", "do", "not", "also", "should"];
    let adjs = ["new", "first", "other", "more", "many", "most", "some",
                "all", "these", "such"];

    let mut phase0 = Vec::new();
    let mut phase1 = Vec::new();
    let mut phase2 = Vec::new();
    let mut phase3 = Vec::new();

    // Phase 0: "the" + noun (many nouns!)
    for noun in &nouns {
        if let Some(d) = drill(&["the"], noun) { phase0.push(d); }
    }
    // "a" + noun
    for noun in &nouns {
        if let Some(d) = drill(&["a"], noun) { phase0.push(d); }
    }

    // Phase 1: subject + verb
    for noun in &nouns {
        if let Some(d) = drill(&["the", noun], "is") { phase1.push(d); }
        if let Some(d) = drill(&["the", noun], "was") { phase1.push(d); }
        if let Some(d) = drill(&["the", noun], "has") { phase1.push(d); }
        if let Some(d) = drill(&["the", noun], "can") { phase1.push(d); }
    }
    if let Some(d) = drill(&["i"], "am") { phase1.push(d); }
    if let Some(d) = drill(&["i"], "have") { phase1.push(d); }
    if let Some(d) = drill(&["i"], "can") { phase1.push(d); }
    if let Some(d) = drill(&["i"], "will") { phase1.push(d); }
    if let Some(d) = drill(&["you"], "are") { phase1.push(d); }
    if let Some(d) = drill(&["you"], "have") { phase1.push(d); }
    if let Some(d) = drill(&["you"], "can") { phase1.push(d); }
    if let Some(d) = drill(&["they"], "are") { phase1.push(d); }
    if let Some(d) = drill(&["they"], "have") { phase1.push(d); }
    if let Some(d) = drill(&["we"], "are") { phase1.push(d); }
    if let Some(d) = drill(&["we"], "have") { phase1.push(d); }
    if let Some(d) = drill(&["it"], "is") { phase1.push(d); }
    if let Some(d) = drill(&["it"], "was") { phase1.push(d); }
    if let Some(d) = drill(&["it"], "has") { phase1.push(d); }

    // Phase 2: adjective + noun
    for adj in &adjs {
        for noun in &nouns[..8] {
            if let Some(d) = drill(&["the", adj], noun) { phase2.push(d); }
        }
    }
    // "is a" + noun
    for noun in &nouns {
        if let Some(d) = drill(&["is", "a"], noun) { phase2.push(d); }
    }

    // Phase 3: common multiword patterns
    if let Some(d) = drill(&["in", "the"], "world") { phase3.push(d); }
    if let Some(d) = drill(&["in", "the"], "case") { phase3.push(d); }
    if let Some(d) = drill(&["of", "the"], "most") { phase3.push(d); }
    if let Some(d) = drill(&["of", "the"], "world") { phase3.push(d); }
    if let Some(d) = drill(&["it", "is"], "a") { phase3.push(d); }
    if let Some(d) = drill(&["it", "is"], "not") { phase3.push(d); }
    if let Some(d) = drill(&["it", "is"], "the") { phase3.push(d); }
    if let Some(d) = drill(&["there", "are"], "many") { phase3.push(d); }
    if let Some(d) = drill(&["this", "is"], "the") { phase3.push(d); }
    if let Some(d) = drill(&["this", "is"], "a") { phase3.push(d); }
    if let Some(d) = drill(&["what", "is"], "the") { phase3.push(d); }
    if let Some(d) = drill(&["one", "of"], "the") { phase3.push(d); }
    if let Some(d) = drill(&["as", "a"], "part") { phase3.push(d); }
    for noun in &nouns[..5] {
        if let Some(d) = drill(&["the", noun, "is"], "a") { phase3.push(d); }
        if let Some(d) = drill(&["the", noun, "is"], "not") { phase3.push(d); }
        if let Some(d) = drill(&["the", noun, "was"], "a") { phase3.push(d); }
    }

    vec![
        Phase { name: format!("Article → noun ({} drills)", phase0.len()), drills: phase0, mastery: 0.6, max_attempts: 20 },
        Phase { name: format!("Subject → verb ({} drills)", phase1.len()), drills: phase1, mastery: 0.4, max_attempts: 20 },
        Phase { name: format!("Adjective → noun ({} drills)", phase2.len()), drills: phase2, mastery: 0.3, max_attempts: 20 },
        Phase { name: format!("Multi-word patterns ({} drills)", phase3.len()), drills: phase3, mastery: 0.3, max_attempts: 20 },
    ]
}
