//! Probes: test what an organism actually understands.
//!
//! Goes beyond loss to test:
//! 1. Do similar words have similar representations? (semantic clustering)
//! 2. Can it complete common patterns? (sequence prediction)
//! 3. Does it distinguish categories? (nouns vs verbs vs prepositions)

use crate::organism::Organism;
use modgrad_compute::ops::dot;

/// Run all probes and print a report.
pub fn probe_organism(org: &mut Organism) {
    eprintln!("\n=== PROBE: Semantic Clustering ===");
    probe_clustering(org);

    eprintln!("\n=== PROBE: Next-Byte Prediction ===");
    probe_prediction(org);

    eprintln!("\n=== PROBE: Category Discrimination ===");
    probe_categories(org);
}

/// Test if similar words produce similar internal representations.
/// "cat" and "dog" should be more similar to each other than to "the".
fn probe_clustering(org: &mut Organism) {
    let words = &[
        // Animals (should cluster)
        "cat", "dog", "bird", "fish",
        // Places
        "house", "door", "tree", "road",
        // Function words
        "the", "a", "on", "in",
        // Verbs
        "sat", "ran", "fell", "walked",
    ];

    let keys: Vec<(&str, Vec<f32>)> = words.iter()
        .map(|w| {
            let ids: Vec<usize> = w.bytes().map(|b| b as usize).collect();
            let key = org.get_key(&ids);
            (*w, key)
        })
        .collect();

    // Print similarity matrix for interesting pairs
    let pairs = &[
        ("cat", "dog"),      // should be high (both animals)
        ("cat", "bird"),     // should be high
        ("cat", "the"),      // should be low (animal vs function word)
        ("cat", "sat"),      // should be medium (co-occur but different type)
        ("the", "a"),        // should be high (both articles)
        ("on", "in"),        // should be high (both prepositions)
        ("sat", "ran"),      // should be high (both verbs)
        ("house", "tree"),   // should be high (both nouns/places)
        ("house", "ran"),    // should be low (place vs verb)
    ];

    for (a, b) in pairs {
        let ka = keys.iter().find(|(w, _)| w == a).map(|(_, k)| k);
        let kb = keys.iter().find(|(w, _)| w == b).map(|(_, k)| k);
        if let (Some(ka), Some(kb)) = (ka, kb) {
            let sim = cosine(ka, kb);
            let label = if sim > 0.8 { "HIGH" }
                else if sim > 0.5 { "MED" }
                else if sim > 0.2 { "LOW" }
                else { "NONE" };
            eprintln!("  {a:>6} ~ {b:<6} sim={sim:.3} [{label}]");
        }
    }
}

/// Test if the organism can predict common next bytes.
fn probe_prediction(org: &mut Organism) {
    let tests = &[
        ("the ", 'c'),         // "the c..." (cat?)
        ("the cat ", 's'),     // "the cat s..." (sat?)
        ("once upon a ", 't'), // "once upon a t..." (time?)
        ("he said ", '"'),     // dialogue marker
        (". ", 't'),           // start of new sentence, "the"?
    ];

    for (prompt, expected) in tests {
        let output = org.generate(prompt.as_bytes(), 1);
        let got = output.first().map(|&b| b as char).unwrap_or('?');
        let correct = got == *expected;
        let mark = if correct { "OK" } else { "  " };
        eprintln!("  [{mark}] \"{prompt}\" → '{got}' (expected '{expected}')");
    }

    // Also test longer generation
    eprintln!("\n  Generation samples:");
    for prompt in &["the ", "once ", "he ", "she ", "it was "] {
        let output = org.generate(prompt.as_bytes(), 30);
        let text = String::from_utf8_lossy(&output);
        eprintln!("    \"{prompt}\" → \"{text}\"");
    }
}

/// Test if the organism discriminates between word categories.
/// Compute mean similarity within categories vs between categories.
fn probe_categories(org: &mut Organism) {
    let categories: &[(&str, &[&str])] = &[
        ("animals", &["cat", "dog", "bird", "fish"]),
        ("verbs", &["sat", "ran", "fell", "walked"]),
        ("preps", &["on", "in", "by", "near"]),
        ("articles", &["the", "a"]),
    ];

    let mut cat_keys: Vec<(&str, Vec<Vec<f32>>)> = Vec::new();
    for (name, words) in categories {
        let keys: Vec<Vec<f32>> = words.iter()
            .map(|w| {
                let ids: Vec<usize> = w.bytes().map(|b| b as usize).collect();
                org.get_key(&ids)
            })
            .collect();
        cat_keys.push((name, keys));
    }

    // Within-category similarity
    eprintln!("  Within-category similarity (higher = learned categories):");
    for (name, keys) in &cat_keys {
        let mut sum = 0.0f32;
        let mut count = 0;
        for i in 0..keys.len() {
            for j in (i+1)..keys.len() {
                sum += cosine(&keys[i], &keys[j]);
                count += 1;
            }
        }
        let avg = if count > 0 { sum / count as f32 } else { 0.0 };
        eprintln!("    {name:>10}: {avg:.3}");
    }

    // Between-category similarity
    eprintln!("  Between-category similarity (lower = discriminating):");
    for i in 0..cat_keys.len() {
        for j in (i+1)..cat_keys.len() {
            let mut sum = 0.0f32;
            let mut count = 0;
            for ka in &cat_keys[i].1 {
                for kb in &cat_keys[j].1 {
                    sum += cosine(ka, kb);
                    count += 1;
                }
            }
            let avg = if count > 0 { sum / count as f32 } else { 0.0 };
            eprintln!("    {:>10} × {:<10}: {avg:.3}",
                cat_keys[i].0, cat_keys[j].0);
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 1e-8 && nb > 1e-8 { d / (na * nb) } else { 0.0 }
}
