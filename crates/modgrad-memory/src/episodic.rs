//! Episodic memory: tick-trajectory storage, retrieval, and consolidation.
//!
//! Follows "Your Server as a Function" (Eriksen): all operations are pure
//! functions that take state in and return state out. No hidden mutation.
//!
//!   `store(mem, episode) -> mem'`
//!   `retrieve(&mem, query) -> result`       (pure read, no side effects)
//!   `mark_retrieved(mem, indices) -> mem'`   (explicit reconsolidation)
//!   `consolidate(mem) -> (mem', stats)`
//!
//! Each episode captures the full path h1→h2→...→hK through the CTM tick loop,
//! plus metadata (certainty progression, exit depth, surprise/salience).
//! Retrieval is triggered by cosine similarity on the final tick's activated state
//! (CRI-style conditioning). Stored value is the full trajectory.
//!
//! Three operational modes (caller-orchestrated, not hardwired):
//!   - **Theta**: all ticks run, every trajectory stored (developmental absorption)
//!   - **Wake**: adaptive exit, selective storage, retrieval biases initial state
//!   - **REM**: replay stored trajectories, measure fidelity, consolidate

use serde::{Deserialize, Serialize};

// ─── Configuration ────────────────────────────────────────────

/// Configuration for the episodic memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicConfig {
    /// Maximum number of episodes in the ring buffer.
    pub capacity: usize,
    /// Maximum ticks per trajectory.
    pub max_ticks: usize,
    /// Dimension of the activated state vector (CtmConfig::d_model).
    pub d_model: usize,
    /// Minimum ticks_used to be worth storing (salience gate).
    pub min_ticks_for_storage: usize,
    /// Minimum surprise for storage.
    pub min_surprise: f32,
    /// Cosine similarity threshold for retrieval.
    pub retrieval_threshold: f32,
    /// Cosine similarity threshold for consolidation merging.
    pub consolidation_threshold: f32,
    /// Retrieval count before eligible for semantic collapse.
    pub semantic_collapse_retrievals: u32,
    /// Strength decay factor per consolidation cycle.
    pub strength_decay: f32,
}

impl Default for EpisodicConfig {
    fn default() -> Self {
        Self {
            capacity: 256,
            max_ticks: 8,
            d_model: 64,
            min_ticks_for_storage: 2,
            min_surprise: 0.5,
            retrieval_threshold: 0.7,
            consolidation_threshold: 0.9,
            semantic_collapse_retrievals: 10,
            strength_decay: 0.95,
        }
    }
}

// ─── Episode metadata ─────────────────────────────────────────

/// Per-episode metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeMeta {
    pub ticks_used: usize,
    pub surprise: f32,
    pub certainties_flat: Vec<f32>,
    pub exit_lambdas: Vec<f32>,
    pub store_id: u64,
    pub timestamp: f64,
    pub strength: f32,
    pub retrieval_count: u32,
    pub is_semantic: bool,
    pub merge_count: u32,
    pub alive: bool,
    /// Emotional valence: -1.0 = painful, 0.0 = neutral, 1.0 = joyful.
    /// Derived from loss at storage time, updated during reappraisal.
    pub valence: f32,
    /// The receipts — what actually happened, for reappraisal.
    /// Loss at the moment this episode was stored.
    pub loss_at_storage: f32,
    /// How confident was the prediction (0.0 = uncertain, 1.0 = certain).
    pub prediction_confidence: f32,
    /// Was the prediction correct?
    pub was_correct: bool,
}

// ─── Return types ─────────────────────────────────────────────

/// Result of an episodic retrieval. Pure data, no side effects.
pub struct EpisodicRetrievalResult {
    /// Soft-attention weighted average of matching trajectories.
    pub blended_trajectory: Vec<f32>,
    /// Blended final-tick activated state [d_model].
    pub blended_final_state: Vec<f32>,
    /// Highest cosine similarity among all episodes.
    pub best_similarity: f32,
    /// Index of the best-matching episode.
    pub best_idx: Option<usize>,
    /// Number of episodes above retrieval threshold.
    pub n_matches: usize,
    /// Weighted-average ticks_used of matching episodes.
    pub expected_depth: f32,
    /// Indices of episodes that participated (for mark_retrieved).
    pub matched_indices: Vec<usize>,
    /// Weighted-average valence of matched episodes.
    /// Negative = this context was painful before. Positive = joyful.
    pub blended_valence: f32,
}

/// Result of comparing a new trajectory against a stored episode.
pub struct TrajectoryComparison {
    pub per_tick_cosine: Vec<f32>,
    pub mean_similarity: f32,
    pub similarity_std: f32,
    pub convergent: bool,
    pub slope: f32,
}

/// Result of a consolidation cycle.
pub struct ConsolidationResult {
    pub merges: usize,
    pub semantic_collapses: usize,
    pub evictions: usize,
}

// ─── State ────────────────────────────────────────────────────

/// Episodic memory state. Passed into and returned from all operations.
///
/// All operations are free functions: `store(mem, ...) -> mem`,
/// `retrieve(&mem, ...) -> result`, `consolidate(mem) -> (mem, stats)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub config: EpisodicConfig,
    keys: Vec<f32>,
    trajectories: Vec<f32>,
    meta: Vec<EpisodeMeta>,
    write_ptr: usize,
    pub count: usize,
    next_store_id: u64,
}

impl EpisodicMemory {
    pub fn new(config: EpisodicConfig) -> Self {
        let cap = config.capacity;
        let d = config.d_model;
        let mt = config.max_ticks;
        Self {
            keys: vec![0.0; cap * d],
            trajectories: vec![0.0; cap * mt * d],
            meta: (0..cap).map(|_| EpisodeMeta {
                ticks_used: 0, surprise: 0.0,
                certainties_flat: Vec::new(), exit_lambdas: Vec::new(),
                store_id: 0, timestamp: 0.0, strength: 0.0,
                retrieval_count: 0, is_semantic: false, merge_count: 0,
                alive: false,
                valence: 0.0, loss_at_storage: 0.0,
                prediction_confidence: 0.0, was_correct: false,
            }).collect(),
            write_ptr: 0,
            count: 0,
            next_store_id: 0,
            config,
        }
    }

    // ─── Accessors (pure reads) ───────────────────────────────

    pub fn trajectory(&self, idx: usize) -> &[f32] {
        let mt = self.config.max_ticks;
        let d = self.config.d_model;
        &self.trajectories[idx * mt * d..(idx + 1) * mt * d]
    }

    pub fn key(&self, idx: usize) -> &[f32] {
        let d = self.config.d_model;
        &self.keys[idx * d..(idx + 1) * d]
    }

    pub fn meta(&self, idx: usize) -> Option<&EpisodeMeta> {
        if idx < self.count { Some(&self.meta[idx]) } else { None }
    }

    pub fn prioritized(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.count)
            .filter(|&i| self.meta[i].alive)
            .collect();
        indices.sort_by(|&a, &b| {
            self.meta[b].strength.partial_cmp(&self.meta[a].strength).unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn capacity(&self) -> usize { self.config.capacity }
}

impl Default for EpisodicMemory {
    fn default() -> Self { Self::new(EpisodicConfig::default()) }
}

// ─── Pure functions: state in → state out ─────────────────────

/// Valence receipt: the evidence for how this episode felt.
/// Stored alongside the valence tag so the organism can reappraise later.
#[derive(Debug, Clone, Default)]
pub struct ValenceReceipt {
    /// Emotional valence: -1.0 = painful, 0.0 = neutral, 1.0 = joyful.
    pub valence: f32,
    /// Loss at the moment of storage.
    pub loss: f32,
    /// Prediction confidence (0.0 = uncertain, 1.0 = certain).
    pub confidence: f32,
    /// Was the prediction correct?
    pub correct: bool,
}

/// Store a complete episode. Returns updated memory (or unchanged if rejected).
///
/// `trajectory`: flat per-tick activated states [ticks_used * d_model].
/// `certainties`: per-tick [entropy, 1-entropy].
/// `exit_lambdas`: per-tick halting probs (may be empty).
/// `surprise`: scalar salience signal.
/// `receipt`: optional valence receipt (pain/joy + evidence for reappraisal).
pub fn store(
    mem: EpisodicMemory,
    trajectory: &[f32],
    certainties: &[[f32; 2]],
    exit_lambdas: &[f32],
    ticks_used: usize,
    surprise: f32,
) -> (EpisodicMemory, bool) {
    store_with_valence(mem, trajectory, certainties, exit_lambdas, ticks_used, surprise, None)
}

/// Store with explicit valence receipt.
pub fn store_with_valence(
    mut mem: EpisodicMemory,
    trajectory: &[f32],
    certainties: &[[f32; 2]],
    exit_lambdas: &[f32],
    ticks_used: usize,
    surprise: f32,
    receipt: Option<ValenceReceipt>,
) -> (EpisodicMemory, bool) {
    if mem.config.capacity == 0 { return (mem, false); }
    let d = mem.config.d_model;
    let mt = mem.config.max_ticks;

    if ticks_used < mem.config.min_ticks_for_storage
        && surprise < mem.config.min_surprise
    {
        return (mem, false);
    }

    let ticks = ticks_used.min(mt);
    let expected_len = ticks * d;
    if trajectory.len() < expected_len || d == 0 { return (mem, false); }

    let slot = mem.write_ptr;

    // Normalized key from final-tick activated state
    let final_tick_start = (ticks - 1) * d;
    let final_tick = &trajectory[final_tick_start..final_tick_start + d];
    let norm = vec_norm(final_tick);
    let k_start = slot * d;
    for i in 0..d {
        mem.keys[k_start + i] = final_tick[i] / norm;
    }

    // Trajectory (zero-pad unused ticks)
    let t_start = slot * mt * d;
    let valid = ticks * d;
    mem.trajectories[t_start..t_start + valid]
        .copy_from_slice(&trajectory[..valid]);
    for i in valid..mt * d {
        mem.trajectories[t_start + i] = 0.0;
    }

    // Metadata
    let mut cert_flat = Vec::with_capacity(ticks * 2);
    for c in &certainties[..ticks.min(certainties.len())] {
        cert_flat.push(c[0]);
        cert_flat.push(c[1]);
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let r = receipt.unwrap_or_default();
    mem.meta[slot] = EpisodeMeta {
        ticks_used: ticks,
        surprise,
        certainties_flat: cert_flat,
        exit_lambdas: exit_lambdas.to_vec(),
        store_id: mem.next_store_id,
        timestamp: now,
        strength: 1.0,
        retrieval_count: 0,
        is_semantic: false,
        merge_count: 0,
        alive: true,
        valence: r.valence,
        loss_at_storage: r.loss,
        prediction_confidence: r.confidence,
        was_correct: r.correct,
    };

    mem.next_store_id += 1;
    mem.write_ptr = (mem.write_ptr + 1) % mem.config.capacity;
    if mem.count < mem.config.capacity {
        mem.count += 1;
    }

    (mem, true)
}

// TODO: re-enable when CtmOutput gains a trajectory field.
// pub fn store_from_output(
//     mem: EpisodicMemory,
//     output: &crate::forward::CtmOutput,
//     surprise: f32,
// ) -> (EpisodicMemory, bool) {
//     store(mem, &output.trajectory, &output.certainties,
//           &output.exit_lambdas, output.ticks_used, surprise)
// }

/// Pure retrieval — no side effects on memory state.
/// Returns result with matched_indices for optional mark_retrieved.
pub fn retrieve(mem: &EpisodicMemory, query: &[f32]) -> EpisodicRetrievalResult {
    let d = mem.config.d_model;
    let mt = mem.config.max_ticks;
    let threshold = mem.config.retrieval_threshold;

    let empty = || EpisodicRetrievalResult {
        blended_trajectory: vec![0.0; mt * d],
        blended_final_state: vec![0.0; d],
        best_similarity: 0.0,
        best_idx: None,
        n_matches: 0,
        expected_depth: 0.0,
        matched_indices: Vec::new(),
        blended_valence: 0.0,
    };

    if mem.count == 0 || query.len() < d { return empty(); }

    let q_norm = vec_norm(query);
    if q_norm < 1e-8 { return empty(); }

    // Compute similarities
    let mut best_sim = 0.0f32;
    let mut best_idx = 0usize;
    let mut sims = vec![0.0f32; mem.count];

    for i in 0..mem.count {
        if !mem.meta[i].alive { continue; }
        let k_start = i * d;
        let mut dot = 0.0f32;
        for j in 0..d {
            dot += mem.keys[k_start + j] * query[j];
        }
        let sim = dot / q_norm;
        sims[i] = sim;
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }

    if best_sim < threshold {
        return EpisodicRetrievalResult {
            blended_trajectory: vec![0.0; mt * d],
            blended_final_state: vec![0.0; d],
            best_similarity: best_sim,
            best_idx: Some(best_idx),
            n_matches: 0,
            expected_depth: 0.0,
            matched_indices: Vec::new(),
            blended_valence: 0.0,
        };
    }

    // Soft-attention over matching episodes
    let temperature = 0.1f32;
    let mut weights = vec![0.0f32; mem.count];
    let mut weight_sum = 0.0f32;
    let mut n_matches = 0usize;
    let mut matched_indices = Vec::new();

    for i in 0..mem.count {
        if sims[i] > threshold && mem.meta[i].alive {
            let w = ((sims[i] - threshold) / temperature).exp() * mem.meta[i].strength;
            weights[i] = w;
            weight_sum += w;
            n_matches += 1;
            matched_indices.push(i);
        }
    }

    let mut blended = vec![0.0f32; mt * d];
    let mut expected_depth = 0.0f32;
    let mut blended_valence = 0.0f32;

    if weight_sum > 1e-8 {
        for i in 0..mem.count {
            if weights[i] > 0.0 {
                let w = weights[i] / weight_sum;
                let t_start = i * mt * d;
                for j in 0..mt * d {
                    blended[j] += w * mem.trajectories[t_start + j];
                }
                expected_depth += w * mem.meta[i].ticks_used as f32;
                blended_valence += w * mem.meta[i].valence;
            }
        }
    }

    let final_tick = ((expected_depth.round() as usize).max(1) - 1).min(mt - 1);
    let fs_start = final_tick * d;
    let blended_final_state = blended[fs_start..fs_start + d].to_vec();

    EpisodicRetrievalResult {
        blended_trajectory: blended,
        blended_final_state,
        best_similarity: best_sim,
        best_idx: Some(best_idx),
        n_matches,
        expected_depth,
        matched_indices,
        blended_valence,
    }
}

/// Pure best-match retrieval. No side effects.
pub fn retrieve_best(mem: &EpisodicMemory, query: &[f32]) -> Option<(usize, f32)> {
    let d = mem.config.d_model;
    if mem.count == 0 || query.len() < d { return None; }

    let q_norm = vec_norm(query);
    if q_norm < 1e-8 { return None; }

    let mut best_sim = 0.0f32;
    let mut best_idx = 0usize;

    for i in 0..mem.count {
        if !mem.meta[i].alive { continue; }
        let k_start = i * d;
        let mut dot = 0.0f32;
        for j in 0..d {
            dot += mem.keys[k_start + j] * query[j];
        }
        let sim = dot / q_norm;
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }

    if best_sim >= mem.config.retrieval_threshold {
        Some((best_idx, best_sim))
    } else {
        None
    }
}

/// Explicit reconsolidation: boost strength of retrieved episodes.
/// Separates the side effect from the pure retrieval.
pub fn mark_retrieved(mut mem: EpisodicMemory, indices: &[usize]) -> EpisodicMemory {
    for &i in indices {
        if i < mem.count && mem.meta[i].alive {
            mem.meta[i].strength = (mem.meta[i].strength * 1.05).min(2.0);
            mem.meta[i].retrieval_count += 1;
        }
    }
    mem
}

/// Compare a new trajectory against a stored episode. Pure read.
pub fn compare_trajectory(
    mem: &EpisodicMemory,
    episode_idx: usize,
    new_trajectory: &[f32],
    new_ticks_used: usize,
) -> Option<TrajectoryComparison> {
    if episode_idx >= mem.count || !mem.meta[episode_idx].alive {
        return None;
    }

    let d = mem.config.d_model;
    let mt = mem.config.max_ticks;
    let stored_ticks = mem.meta[episode_idx].ticks_used;
    let overlap = new_ticks_used.min(stored_ticks);

    if overlap == 0 || new_trajectory.len() < overlap * d { return None; }

    let t_start = episode_idx * mt * d;
    let mut cosines = Vec::with_capacity(overlap);

    for t in 0..overlap {
        let stored_slice = &mem.trajectories[t_start + t * d..t_start + (t + 1) * d];
        let new_slice = &new_trajectory[t * d..(t + 1) * d];
        cosines.push(cosine_sim(stored_slice, new_slice));
    }

    let n = cosines.len() as f32;
    let mean = cosines.iter().sum::<f32>() / n;
    let variance = cosines.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();

    let slope = if overlap >= 2 {
        let sum_t: f32 = (0..overlap).map(|t| t as f32).sum();
        let sum_t2: f32 = (0..overlap).map(|t| (t as f32).powi(2)).sum();
        let sum_tc: f32 = (0..overlap).map(|t| t as f32 * cosines[t]).sum();
        let sum_c: f32 = cosines.iter().sum();
        (n * sum_tc - sum_t * sum_c) / (n * sum_t2 - sum_t * sum_t).max(1e-8)
    } else {
        0.0
    };

    Some(TrajectoryComparison {
        per_tick_cosine: cosines,
        mean_similarity: mean,
        similarity_std: std_dev,
        convergent: slope > 0.0,
        slope,
    })
}

/// Consolidation: decay, evict, merge, semantic collapse.
/// Returns updated memory and stats.
pub fn consolidate(mut mem: EpisodicMemory) -> (EpisodicMemory, ConsolidationResult) {
    let d = mem.config.d_model;
    let mt = mem.config.max_ticks;
    let mut merges = 0usize;
    let mut semantic_collapses = 0usize;
    let mut evictions = 0usize;

    // 1. Decay
    for i in 0..mem.count {
        if mem.meta[i].alive {
            mem.meta[i].strength *= mem.config.strength_decay;
        }
    }

    // 2. Evict
    for i in 0..mem.count {
        if mem.meta[i].alive && mem.meta[i].strength < 0.01 {
            mem.meta[i].alive = false;
            evictions += 1;
        }
    }

    // 3. Merge similar
    for i in 0..mem.count {
        if !mem.meta[i].alive { continue; }
        for j in (i + 1)..mem.count {
            if !mem.meta[j].alive { continue; }

            let ki = i * d;
            let kj = j * d;
            let mut dot = 0.0f32;
            for n in 0..d {
                dot += mem.keys[ki + n] * mem.keys[kj + n];
            }

            if dot > mem.config.consolidation_threshold {
                let (keep, kill) = if mem.meta[i].strength >= mem.meta[j].strength {
                    (i, j)
                } else {
                    (j, i)
                };

                let total_s = mem.meta[keep].strength + mem.meta[kill].strength;
                let w_keep = mem.meta[keep].strength / total_s;
                let w_kill = 1.0 - w_keep;

                let t_keep = keep * mt * d;
                let t_kill = kill * mt * d;
                for e in 0..mt * d {
                    mem.trajectories[t_keep + e] =
                        w_keep * mem.trajectories[t_keep + e]
                        + w_kill * mem.trajectories[t_kill + e];
                }

                let ticks = mem.meta[keep].ticks_used.max(mem.meta[kill].ticks_used);
                let final_tick = (ticks.min(mt) - 1).max(0);
                let ft_start = t_keep + final_tick * d;
                let norm = vec_norm(&mem.trajectories[ft_start..ft_start + d]);
                let k_keep = keep * d;
                for n in 0..d {
                    mem.keys[k_keep + n] = mem.trajectories[ft_start + n] / norm;
                }

                mem.meta[keep].merge_count += mem.meta[kill].merge_count + 1;
                mem.meta[keep].strength += mem.meta[kill].strength * 0.5;
                mem.meta[keep].retrieval_count += mem.meta[kill].retrieval_count;
                mem.meta[keep].ticks_used = ticks;
                mem.meta[kill].alive = false;
                merges += 1;
            }
        }
    }

    // 4. Semantic collapse
    for i in 0..mem.count {
        if !mem.meta[i].alive || mem.meta[i].is_semantic { continue; }
        if mem.meta[i].retrieval_count >= mem.config.semantic_collapse_retrievals {
            let ticks = mem.meta[i].ticks_used;
            if ticks > 1 {
                let t_start = i * mt * d;
                for t in 0..(ticks - 1) {
                    let base = t_start + t * d;
                    for n in 0..d {
                        mem.trajectories[base + n] = 0.0;
                    }
                }
            }
            mem.meta[i].is_semantic = true;
            semantic_collapses += 1;
        }
    }

    (mem, ConsolidationResult { merges, semantic_collapses, evictions })
}

/// Reappraisal result for one episode.
pub struct ReappraisalOutcome {
    pub episode_idx: usize,
    pub old_valence: f32,
    pub new_valence: f32,
    pub reappraised: bool,
}

/// Reappraise painful memories during sleep consolidation.
///
/// For each painful episode (valence < 0), the caller provides a
/// `recompute_loss` function that evaluates the episode's input
/// with current weights. If the organism has learned since storage,
/// the recomputed loss will be lower → the pain fades.
///
/// This is therapy: "was the pain justified given what I know now?"
pub fn reappraise(
    mut mem: EpisodicMemory,
    recompute_loss: &dyn Fn(usize) -> Option<f32>,
    decay_rate: f32,
) -> (EpisodicMemory, Vec<ReappraisalOutcome>) {
    let mut outcomes = Vec::new();

    for i in 0..mem.count {
        if !mem.meta[i].alive { continue; }
        if mem.meta[i].valence >= 0.0 { continue; } // only reappraise pain

        let old_valence = mem.meta[i].valence;
        let original_loss = mem.meta[i].loss_at_storage;

        if original_loss <= 0.0 { continue; }

        if let Some(current_loss) = recompute_loss(i) {
            // The organism has improved → pain no longer justified
            let improvement = ((original_loss - current_loss) / original_loss).clamp(0.0, 1.0);

            if improvement > 0.1 {
                // Fade the scar proportionally to improvement
                let fade = 1.0 - improvement * decay_rate;
                mem.meta[i].valence *= fade;

                outcomes.push(ReappraisalOutcome {
                    episode_idx: i,
                    old_valence,
                    new_valence: mem.meta[i].valence,
                    reappraised: true,
                });
            } else {
                outcomes.push(ReappraisalOutcome {
                    episode_idx: i,
                    old_valence,
                    new_valence: old_valence,
                    reappraised: false,
                });
            }
        }
    }

    (mem, outcomes)
}

// ─── Helpers ──────────────────────────────────────────────────

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-8)
}

// ─── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(d: usize, ticks: usize) -> EpisodicConfig {
        EpisodicConfig {
            capacity: 16, max_ticks: ticks, d_model: d,
            min_ticks_for_storage: 2, min_surprise: 0.5,
            retrieval_threshold: 0.5, consolidation_threshold: 0.95,
            semantic_collapse_retrievals: 3, strength_decay: 0.9,
        }
    }

    fn fake_trajectory(d: usize, ticks: usize, seed: f32) -> Vec<f32> {
        (0..ticks * d).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    fn fake_certainties(ticks: usize) -> Vec<[f32; 2]> {
        (0..ticks).map(|t| {
            let c = (t as f32 + 1.0) / (ticks as f32 + 1.0);
            [1.0 - c, c]
        }).collect()
    }

    #[test]
    fn store_and_retrieve_round_trip() {
        let cfg = make_config(8, 4);
        let mem = EpisodicMemory::new(cfg);

        let traj = fake_trajectory(8, 4, 0.0);
        let cert = fake_certainties(4);
        let (mem, stored) = store(mem, &traj, &cert, &[], 4, 1.0);
        assert!(stored);
        assert_eq!(mem.len(), 1);

        let final_state = &traj[3 * 8..4 * 8];
        let result = retrieve(&mem, final_state);
        assert_eq!(result.n_matches, 1);
        assert!(result.best_similarity > 0.99, "sim={}", result.best_similarity);
    }

    #[test]
    fn salience_gate_rejects_trivial() {
        let cfg = make_config(8, 4);
        let mem = EpisodicMemory::new(cfg);

        let traj = fake_trajectory(8, 1, 0.0);
        let cert = fake_certainties(1);
        let (mem, stored) = store(mem, &traj, &cert, &[], 1, 0.1);
        assert!(!stored);
        assert_eq!(mem.len(), 0);

        let (mem, stored) = store(mem, &traj, &cert, &[], 1, 2.0);
        assert!(stored);
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn trajectory_comparison_convergent() {
        let cfg = make_config(8, 4);
        let mem = EpisodicMemory::new(cfg);

        let traj1 = fake_trajectory(8, 4, 0.0);
        let cert = fake_certainties(4);
        let (mem, _) = store(mem, &traj1, &cert, &[], 4, 1.0);

        let traj2 = fake_trajectory(8, 4, 0.01);
        let cmp = compare_trajectory(&mem, 0, &traj2, 4).unwrap();
        assert!(cmp.mean_similarity > 0.9, "mean_sim={}", cmp.mean_similarity);
    }

    #[test]
    fn consolidation_merges_similar() {
        let cfg = make_config(8, 4);
        let mem = EpisodicMemory::new(cfg);

        let traj1 = fake_trajectory(8, 4, 0.0);
        let traj2 = fake_trajectory(8, 4, 0.001);
        let cert = fake_certainties(4);
        let (mem, _) = store(mem, &traj1, &cert, &[], 4, 1.0);
        let (mem, _) = store(mem, &traj2, &cert, &[], 4, 1.0);
        assert_eq!(mem.len(), 2);

        let (mem, result) = consolidate(mem);
        let alive = (0..mem.count).filter(|&i| mem.meta[i].alive).count();
        eprintln!("merges={}, alive={}", result.merges, alive);
    }

    #[test]
    fn semantic_collapse_after_retrievals() {
        let mut cfg = make_config(8, 4);
        cfg.semantic_collapse_retrievals = 2;
        let mem = EpisodicMemory::new(cfg);

        let traj = fake_trajectory(8, 4, 0.0);
        let cert = fake_certainties(4);
        let (mem, _) = store(mem, &traj, &cert, &[], 4, 1.0);

        let final_state = &traj[3 * 8..4 * 8];

        // Retrieve twice (pure) then mark (explicit mutation)
        let r1 = retrieve(&mem, final_state);
        let mem = mark_retrieved(mem, &r1.matched_indices);
        let r2 = retrieve(&mem, final_state);
        let mem = mark_retrieved(mem, &r2.matched_indices);

        let (mem, result) = consolidate(mem);
        assert_eq!(result.semantic_collapses, 1);
        assert!(mem.meta[0].is_semantic);

        let t = mem.trajectory(0);
        let tick0_norm: f32 = t[..8].iter().map(|x| x.abs()).sum();
        assert!(tick0_norm < 1e-6, "tick 0 should be zeroed, norm={}", tick0_norm);
        let final_norm: f32 = t[3 * 8..4 * 8].iter().map(|x| x.abs()).sum();
        assert!(final_norm > 0.1, "final tick preserved, norm={}", final_norm);
    }

    #[test]
    fn ring_buffer_wraps() {
        let cfg = make_config(4, 2);
        let mut mem = EpisodicMemory::new(cfg);

        for i in 0..20 {
            let traj = fake_trajectory(4, 2, i as f32);
            let cert = fake_certainties(2);
            let (m, _) = store(mem, &traj, &cert, &[], 2, 1.0);
            mem = m;
        }

        assert_eq!(mem.len(), 16);
        assert_eq!(mem.next_store_id, 20);
    }

    #[test]
    fn retrieve_best_respects_threshold() {
        let cfg = make_config(8, 4);
        let mem = EpisodicMemory::new(cfg);

        let traj = fake_trajectory(8, 4, 0.0);
        let cert = fake_certainties(4);
        let (mem, _) = store(mem, &traj, &cert, &[], 4, 1.0);

        let unrelated: Vec<f32> = (0..8).map(|i| (i as f32 * 7.7).cos()).collect();
        assert!(retrieve_best(&mem, &unrelated).is_none() ||
            retrieve_best(&mem, &unrelated).unwrap().1 < 0.5);
    }

    #[test]
    fn prioritized_ordering() {
        let mut cfg = make_config(4, 2);
        cfg.retrieval_threshold = 0.99;
        let mem = EpisodicMemory::new(cfg);

        let mut mem = mem;
        for i in 0..3 {
            let traj = fake_trajectory(4, 2, i as f32 * 100.0);
            let cert = fake_certainties(2);
            let (m, _) = store(mem, &traj, &cert, &[], 2, 1.0);
            mem = m;
        }

        // Explicit strength boost (no hidden mutation)
        mem = mark_retrieved(mem, &[1]);
        mem = mark_retrieved(mem, &[1]);
        mem = mark_retrieved(mem, &[1]);
        // 1.0 * 1.05^3 ≈ 1.158 vs 1.0 for others

        let order = mem.prioritized();
        assert_eq!(order[0], 1, "episode 1 should be first (most retrieved)");
    }

    // TODO: re-enable when CtmOutput gains a trajectory field.
    // Full integration: CTM forward → store → retrieve → compare.
    // #[test]
    // fn ctm_forward_episodic_round_trip() { ... }
}
