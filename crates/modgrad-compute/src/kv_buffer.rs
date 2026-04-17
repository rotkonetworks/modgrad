//! Hierarchical bounded KV buffer with adaptive compression.
//!
//! Three tiers, each with a fixed capacity:
//!   Short-term:  recent entries at full resolution.
//!   Mid-term:    temporally deduplicated (novel entries only).
//!   Long-term:   spatially consolidated (similar entries merged into prototypes).
//!
//! Overflow cascades: short → mid (via temporal selection) → long (via merging).
//! Compression thresholds are data-driven via Otsu's method — no manual tuning.
//!
//! The memory stores flat f32 vectors of a fixed dimension. It knows nothing
//! about attention, tokens, or models. The consumer (e.g., CTM's MHA) reads
//! the entries as a KV buffer. This separation is deliberate.

/// Adaptive hierarchical memory with bounded capacity.
pub struct EpisodicMemory {
    dim: usize,
    short: Vec<f32>,   // [n_short × dim]
    mid: Vec<f32>,     // [n_mid × dim]
    long: Vec<f32>,    // [n_long × dim]
    cap_short: usize,  // max entries in short-term
    cap_mid: usize,    // max entries in mid-term
    cap_long: usize,   // max entries in long-term
    recent_distances: Vec<f32>, // rolling window for Otsu threshold
}

impl EpisodicMemory {
    /// Create a new episodic memory.
    ///
    /// `dim`: vector dimension per entry.
    /// `short`, `mid`, `long`: capacity in number of entries per tier.
    pub fn new(dim: usize, short: usize, mid: usize, long: usize) -> Self {
        Self {
            dim,
            short: Vec::with_capacity(short * dim),
            mid: Vec::with_capacity(mid * dim),
            long: Vec::with_capacity(long * dim),
            cap_short: short,
            cap_mid: mid,
            cap_long: long,
            recent_distances: Vec::with_capacity(64),
        }
    }

    /// Insert a new entry. Triggers cascade compression if tiers overflow.
    pub fn push(&mut self, entry: &[f32]) {
        debug_assert_eq!(entry.len(), self.dim);
        self.short.extend_from_slice(entry);

        if self.n_short() > self.cap_short {
            let d = self.dim;
            let evicted: Vec<f32> = self.short.drain(..d).collect();
            self.push_mid(evicted);
        }
    }

    /// All entries as a contiguous KV buffer: [n_total × dim].
    /// Order: long (oldest) → mid → short (newest).
    pub fn as_kv(&self) -> Vec<f32> {
        let total = self.long.len() + self.mid.len() + self.short.len();
        let mut kv = Vec::with_capacity(total);
        kv.extend_from_slice(&self.long);
        kv.extend_from_slice(&self.mid);
        kv.extend_from_slice(&self.short);
        kv
    }

    /// Total entries across all tiers.
    pub fn n_tokens(&self) -> usize {
        self.n_short() + self.n_mid() + self.n_long()
    }

    pub fn is_empty(&self) -> bool { self.short.is_empty() && self.mid.is_empty() && self.long.is_empty() }
    pub fn n_short(&self) -> usize { self.short.len() / self.dim.max(1) }
    pub fn n_mid(&self) -> usize { self.mid.len() / self.dim.max(1) }
    pub fn n_long(&self) -> usize { self.long.len() / self.dim.max(1) }
    pub fn dim(&self) -> usize { self.dim }

    /// Reset all tiers.
    pub fn clear(&mut self) {
        self.short.clear();
        self.mid.clear();
        self.long.clear();
        self.recent_distances.clear();
    }

    // ── Temporal adjacency selection (TAS) ─────────────────────

    /// Push into mid-term with temporal adjacency selection.
    /// Keeps a rolling window of recent cosine distances and uses Otsu
    /// to adaptively threshold novelty — no hardcoded constants.
    fn push_mid(&mut self, entry: Vec<f32>) {
        if self.mid.len() >= self.dim {
            let recent = &self.mid[self.mid.len() - self.dim..];
            let dist = cosine_distance(&entry, recent);
            self.recent_distances.push(dist);
            if self.recent_distances.len() > 64 {
                self.recent_distances.remove(0);
            }

            // Otsu over recent distance distribution to find novelty threshold
            let threshold = if self.recent_distances.len() >= 4 {
                otsu_threshold(&self.recent_distances)
            } else {
                0.05 // bootstrap: accept most entries until we have enough data
            };

            if dist <= threshold {
                // Redundant — don't store
                if self.n_mid() > self.cap_mid {
                    self.compress_mid_to_long();
                }
                return;
            }
        }

        self.mid.extend_from_slice(&entry);

        if self.n_mid() > self.cap_mid {
            self.compress_mid_to_long();
        }
    }

    // ── Spatial domain consolidation (SDC) ─────────────────────

    /// Evict oldest half of mid-term, merge similar entries, push to long-term.
    fn compress_mid_to_long(&mut self) {
        let n_evict = self.cap_mid / 2;
        let evict_bytes = n_evict * self.dim;
        if evict_bytes > self.mid.len() { return; }

        let evicted: Vec<f32> = self.mid.drain(..evict_bytes).collect();
        let entries: Vec<&[f32]> = evicted.chunks_exact(self.dim).collect();
        let merged = merge_similar(&entries, self.dim);

        for proto in &merged {
            self.long.extend_from_slice(proto);
        }

        // Evict oldest long-term entries if over capacity
        let overflow = self.n_long().saturating_sub(self.cap_long);
        if overflow > 0 {
            self.long.drain(..overflow * self.dim);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// PURE FUNCTIONS
// ═══════════════════════════════════════════════════════════════

/// Cosine distance: 1 - cos(a, b). Range [0, 2].
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { return 1.0; }
    1.0 - dot / denom
}

/// Otsu's method: find threshold that maximally separates a distribution.
fn otsu_threshold(values: &[f32]) -> f32 {
    if values.is_empty() { return 0.0; }

    let min_v = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_v = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_v - min_v;
    if range < 1e-10 { return min_v; }

    const BINS: usize = 64;
    let mut hist = [0u32; BINS];
    for &v in values {
        let bin = (((v - min_v) / range) * (BINS - 1) as f32) as usize;
        hist[bin.min(BINS - 1)] += 1;
    }

    let total = values.len() as f32;
    let sum_total: f32 = hist.iter().enumerate()
        .map(|(i, &c)| i as f32 * c as f32).sum();

    let (mut best_bin, mut best_var) = (0usize, 0.0f32);
    let (mut w0, mut sum0) = (0.0f32, 0.0f32);

    for (i, &count) in hist.iter().enumerate() {
        w0 += count as f32;
        if w0 == 0.0 { continue; }
        let w1 = total - w0;
        if w1 == 0.0 { break; }
        sum0 += i as f32 * count as f32;
        let diff = sum0 / w0 - (sum_total - sum0) / w1;
        let var = w0 * w1 * diff * diff;
        if var > best_var { best_var = var; best_bin = i; }
    }

    min_v + (best_bin as f32 / (BINS - 1) as f32) * range
}

/// Merge similar entries into prototypes using Otsu-thresholded union-find.
fn merge_similar(entries: &[&[f32]], dim: usize) -> Vec<Vec<f32>> {
    let n = entries.len();
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![entries[0].to_vec()]; }

    // Pairwise distances (sparse: only nearby entries)
    let mut distances = Vec::with_capacity(n * 3);
    for i in 0..n {
        for j in (i + 1)..n.min(i + 4) {
            distances.push((i, j, cosine_distance(entries[i], entries[j])));
        }
    }

    let dist_values: Vec<f32> = distances.iter().map(|d| d.2).collect();
    let threshold = otsu_threshold(&dist_values);

    // Union-find
    let mut parent: Vec<usize> = (0..n).collect();
    for &(i, j, d) in &distances {
        if d <= threshold {
            union(&mut parent, i, j);
        }
    }

    // Group by root
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = Default::default();
    for i in 0..n {
        groups.entry(find(&mut parent, i)).or_default().push(i);
    }

    // Mean prototype per group
    groups.values().map(|members| {
        let mut proto = vec![0.0f32; dim];
        let w = 1.0 / members.len() as f32;
        for &idx in members {
            for d in 0..dim {
                proto[d] += w * entries[idx][d];
            }
        }
        proto
    }).collect()
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb { parent[ra] = rb; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_and_read() {
        let mut mem = EpisodicMemory::new(4, 2, 4, 8);
        mem.push(&[1.0, 0.0, 0.0, 0.0]);
        mem.push(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(mem.n_short(), 2);
        assert_eq!(mem.n_tokens(), 2);
        assert_eq!(mem.as_kv().len(), 8);
    }

    #[test]
    fn overflow_short_to_mid() {
        let mut mem = EpisodicMemory::new(4, 2, 4, 8);
        mem.push(&[1.0, 0.0, 0.0, 0.0]);
        mem.push(&[0.0, 1.0, 0.0, 0.0]);
        mem.push(&[0.0, 0.0, 1.0, 0.0]); // triggers overflow
        assert_eq!(mem.n_short(), 2);
        assert!(mem.n_mid() >= 1);
    }

    #[test]
    fn similar_entries_deduplicated() {
        let mut mem = EpisodicMemory::new(4, 2, 10, 10);
        for _ in 0..10 {
            mem.push(&[1.0, 0.0, 0.0, 0.0]);
        }
        assert!(mem.n_mid() < 8, "expected dedup, got {} mid entries", mem.n_mid());
    }

    #[test]
    fn bounded_memory() {
        let mut mem = EpisodicMemory::new(4, 2, 4, 8);
        for i in 0..100u32 {
            let v = [
                (i as f32 * 0.1).sin(),
                (i as f32 * 0.2).cos(),
                (i as f32 * 0.3).sin(),
                (i as f32 * 0.4).cos(),
            ];
            mem.push(&v);
        }
        let max = 2 + 4 + 8;
        assert!(mem.n_tokens() <= max, "memory exceeded bounds: {} > {}", mem.n_tokens(), max);
    }

    #[test]
    fn otsu_separates_bimodal() {
        let values: Vec<f32> = (0..50).map(|i| i as f32 * 0.01)
            .chain((0..50).map(|i| 0.8 + i as f32 * 0.004))
            .collect();
        let t = otsu_threshold(&values);
        assert!(t > 0.3 && t < 0.9, "otsu should split bimodal: got {}", t);
    }

    #[test]
    fn merge_groups_similar() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.99, 0.01, 0.0, 0.0];
        let c = [0.0, 0.0, 1.0, 0.0];
        let entries: Vec<&[f32]> = vec![&a, &b, &c];
        let merged = merge_similar(&entries, 4);
        assert!(merged.len() <= 2, "expected ≤2 prototypes, got {}", merged.len());
    }
}
