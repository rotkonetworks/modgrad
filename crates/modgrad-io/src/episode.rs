use crate::memory::MemoryBank;
use crate::types::*;

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-8 {
        return 0.0;
    }
    dot / denom
}

/// Normalize a vector to unit length in-place.
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Effective strength with valence-dependent time decay and reconsolidation.
///
/// Decay half-life varies by emotional valence:
///   Positive: 720h (30 days) — fades normally
///   Neutral:  720h (30 days) — fades normally
///   Negative: 4320h (180 days) — slow fade
///   Fear:     43200h (5 years) — extremely slow, but NOT zero
///
/// The fear half-life is steerable via `fear_decay_multiplier`:
///   1.0 = default (5-year half-life, mimics untreated PTSD)
///   10.0 = accelerated extinction (psilocybin-assisted therapy)
///   0.1 = reinforced (re-traumatization)
///
/// Reconsolidation: each recall adds 10%, BUT during the labile
/// window (6h after last recall) the memory can be weakened by
/// conflicting input. This is the mechanism behind EMDR and
/// reconsolidation-based therapies.
pub fn effective_strength(ep: &Episode, now: f64) -> f32 {
    effective_strength_with(ep, now, 1.0)
}

pub fn effective_strength_with(ep: &Episode, now: f64, fear_decay_multiplier: f32) -> f32 {
    let age_hours = (now - ep.created_at) / 3600.0;

    // Valence-dependent half-life
    let base_half_life = match ep.valence {
        Valence::Positive => 720.0,    // 30 days
        Valence::Neutral => 720.0,     // 30 days
        Valence::Negative => 4320.0,   // 180 days
        Valence::Fear => 43200.0,      // 5 years — steerable
    };

    // Apply fear decay multiplier (psilocybin = higher multiplier = faster decay)
    let half_life = match ep.valence {
        Valence::Fear => base_half_life / fear_decay_multiplier.max(0.01),
        _ => base_half_life,
    };

    let decay = 0.5f64.powf(age_hours / half_life as f64) as f32;

    // Reconsolidation: each recall adds 10%
    let reconsol = 1.0 + 0.1 * ep.recall_count as f32;

    // Labile window: if recalled within last 6h, memory is slightly weakened
    // (reconsolidation interference — basis for therapeutic intervention)
    let labile = if ep.last_recalled_at > 0.0 {
        let hours_since_recall = (now - ep.last_recalled_at) / 3600.0;
        if hours_since_recall < 6.0 {
            0.95 // 5% weaker during labile window
        } else {
            1.0
        }
    } else {
        1.0
    };

    ep.strength * decay * reconsol * labile
}

/// Find the best matching episode across all alters.
pub struct RecallResult {
    pub alter_index: usize,
    pub episode_index: usize,
    pub similarity: f32,
    pub gate: f32,
}

/// Search all episodes for the best key match against a query vector.
/// Searches ALL alters and ALL episodes (no amnesia barriers).
pub fn recall(bank: &MemoryBank, query: &[f32]) -> Option<RecallResult> {
    recall_as(bank, query, None)
}

/// Search episodes visible to a specific alter.
///
/// Amnesia barriers:
/// - If `active_alter` is None, searches everything (backward compat).
/// - If set, only searches:
///   1. The active alter's own episodes
///   2. Other alters listed in active_alter's `can_see` ("*" = all)
///   3. Episodes whose `visible_to` includes the active alter (or is empty = public)
///
/// Reference: Reinders et al. (2006) — DID alters have asymmetric memory access.
pub fn recall_as(bank: &MemoryBank, query: &[f32], active_alter: Option<&str>) -> Option<RecallResult> {
    let mut best_sim = -1.0f32;
    let mut best_alter = 0;
    let mut best_ep = 0;

    for (ai, alter) in bank.alters.iter().enumerate() {
        // Check if the active alter can see this alter's memories
        if let Some(active) = active_alter {
            if alter.name != active {
                // Check can_see list on the active alter
                let active_alter_obj = bank.alters.iter().find(|a| a.name == active);
                let can_see = active_alter_obj
                    .map(|a| {
                        // empty can_see = own memories only (amnesic barrier)
                        // must explicitly list alters or use "*" for co-consciousness
                        if a.can_see.is_empty() {
                            false
                        } else {
                            a.can_see.iter().any(|s| s == "*" || s == &alter.name)
                        }
                    })
                    .unwrap_or(false);

                if !can_see {
                    continue; // amnesia barrier — can't see this alter's memories
                }
            }
        }

        for (ei, ep) in alter.episodes.iter().enumerate() {
            // Check episode-level visibility
            if let Some(active) = active_alter {
                if !ep.visible_to.is_empty()
                    && !ep.visible_to.iter().any(|v| v == active || v == "*")
                {
                    continue; // this specific episode is hidden from this alter
                }
            }

            for ck in &ep.keys {
                let sim = cosine_similarity(query, &ck.key);
                if sim > best_sim {
                    best_sim = sim;
                    best_alter = ai;
                    best_ep = ei;
                }
            }
        }
    }

    if best_sim < bank.threshold {
        return None;
    }

    let gate = ((best_sim - bank.threshold) / (1.0 - bank.threshold)).min(1.0);

    Some(RecallResult {
        alter_index: best_alter,
        episode_index: best_ep,
        similarity: best_sim,
        gate,
    })
}

/// Multi-query recall: try multiple query positions, return best match.
pub fn recall_multi(bank: &MemoryBank, queries: &[Vec<f32>]) -> Option<RecallResult> {
    recall_multi_as(bank, queries, None)
}

/// Multi-query recall with amnesia barriers.
pub fn recall_multi_as(bank: &MemoryBank, queries: &[Vec<f32>], active_alter: Option<&str>) -> Option<RecallResult> {
    let mut best: Option<RecallResult> = None;

    for query in queries {
        if let Some(result) = recall_as(bank, query, active_alter) {
            if best.as_ref().map_or(true, |b| result.similarity > b.similarity) {
                best = Some(result);
            }
        }
    }

    best
}

/// Engram competition: check if a new episode conflicts with existing ones.
pub fn find_competitors(
    bank: &MemoryBank,
    prompt_key: &[f32],
    similarity_threshold: f32,
) -> Vec<(usize, usize, f32)> {
    // Returns (alter_index, episode_index, similarity)
    let mut competitors = Vec::new();

    for (ai, alter) in bank.alters.iter().enumerate() {
        for (ei, ep) in alter.episodes.iter().enumerate() {
            // Check PROMPT_END key specifically
            for ck in &ep.keys {
                if ck.position == -1 {
                    let sim = cosine_similarity(prompt_key, &ck.key);
                    if sim > similarity_threshold {
                        competitors.push((ai, ei, sim));
                    }
                    break;
                }
            }
        }
    }

    competitors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_recall_empty() {
        let bank = MemoryBank::default();
        let query = vec![0.0; 896];
        assert!(recall(&bank, &query).is_none());
    }

    #[test]
    fn test_recall_exact_match() {
        let key = vec![1.0; 896];
        let mut normalized_key = key.clone();
        normalize(&mut normalized_key);

        let bank = MemoryBank {
            alters: vec![Alter {
                name: "test".into(),
                attention_bias: Vec::new(),
                can_see: Vec::new(),
                episodes: vec![Episode {
                    prompt: "test prompt".into(),
                    answer: "test answer".into(),
                    alter: "test".into(),
                    keys: vec![ContentKey {
                        key: normalized_key.clone(),
                        token: "test".into(),
                        position: -1,
                    }],
                    logit_biases: vec![],
                    strength: 1.0,
                    recall_count: 0,
                    created_at: 0.0,
                    consolidated: false, consolidation_score: 0.0, sleep_cycles: 0, valence: Valence::Neutral, last_recalled_at: 0.0, visible_to: Vec::new(),
                }],
            }],
            ..Default::default()
        };

        let result = recall(&bank, &normalized_key).unwrap();
        assert!((result.similarity - 1.0).abs() < 1e-5);
        assert_eq!(result.alter_index, 0);
        assert_eq!(result.episode_index, 0);
    }

    #[test]
    fn test_effective_strength_decay() {
        let ep = Episode {
            prompt: "".into(),
            answer: "".into(),
            alter: "".into(),
            keys: vec![],
            logit_biases: vec![],
            strength: 2.0,
            recall_count: 0,
            created_at: 0.0,
            consolidated: false, consolidation_score: 0.0, sleep_cycles: 0, valence: Valence::Neutral, last_recalled_at: 0.0, visible_to: Vec::new(),
        };
        // 720 hours later = half-life
        let s = effective_strength(&ep, 720.0 * 3600.0);
        assert!((s - 1.0).abs() < 0.01); // 2.0 * 0.5 = 1.0
    }
}
