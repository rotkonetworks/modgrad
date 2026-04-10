//! Hippocampal Content-Addressable Memory.
//!
//! Fast one-shot episodic binding. Stores (key, value) pairs from single
//! exposures, retrieves by cosine similarity. Fixed capacity, ring buffer
//! eviction, no allocation in hot path.

use serde::{Deserialize, Serialize};

/// Fast one-shot episodic binding. Stores (key, value) pairs from single
/// exposures, retrieves by cosine similarity. Fixed capacity, ring buffer
/// eviction, no allocation in hot path.
///
/// The hippocampus proper (NeuronLayer) acts as a gating controller:
/// its activation magnitude decides whether to store, and its output
/// gates the retrieval (novelty detection). The CAM is the memory itself.
///
/// Biologically: CA3 auto-associative network + dentate gyrus pattern
/// separation + CA1 output to cortex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HippocampalCAM {
    /// Stored keys: flat [capacity × key_dim]. Each "row" is a normalized key.
    keys: Vec<f32>,
    /// Stored values: flat [capacity × value_dim]. Pattern to reinstate.
    values: Vec<f32>,
    /// Per-entry strength. Decays over time, boosted by retrieval.
    strengths: Vec<f32>,
    /// Write pointer (ring buffer, wraps at capacity).
    write_ptr: usize,
    /// Number of valid entries (≤ capacity).
    pub count: usize,
    pub capacity: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl HippocampalCAM {
    pub fn new(capacity: usize, key_dim: usize, value_dim: usize) -> Self {
        Self {
            keys: vec![0.0; capacity * key_dim],
            values: vec![0.0; capacity * value_dim],
            strengths: vec![0.0; capacity],
            write_ptr: 0,
            count: 0,
            capacity,
            key_dim,
            value_dim,
        }
    }

    /// Store a new (key, value) binding. O(1).
    /// Key is normalized before storage (cosine similarity = dot product on unit vectors).
    pub fn store(&mut self, key: &[f32], value: &[f32], strength: f32) {
        if self.capacity == 0 { return; }
        let kd = self.key_dim;
        let vd = self.value_dim;

        // Normalize key
        let k_start = self.write_ptr * kd;
        let norm: f32 = key.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for i in 0..kd.min(key.len()) {
            self.keys[k_start + i] = key[i] / norm;
        }

        // Store value as-is
        let v_start = self.write_ptr * vd;
        for i in 0..vd.min(value.len()) {
            self.values[v_start + i] = value[i];
        }

        self.strengths[self.write_ptr] = strength;
        self.write_ptr = (self.write_ptr + 1) % self.capacity;
        if self.count < self.capacity { self.count += 1; }
    }

    /// Retrieve by cosine similarity. O(capacity).
    /// Returns strength-weighted soft-attention over stored values.
    /// Only entries with similarity > threshold participate.
    /// Returns (retrieved_pattern, max_similarity).
    pub fn retrieve(&mut self, query: &[f32], threshold: f32) -> (Vec<f32>, f32) {
        let kd = self.key_dim;
        let vd = self.value_dim;
        let mut result = vec![0.0f32; vd];

        if self.count == 0 {
            return (result, 0.0);
        }

        // Normalize query
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);

        // Compute similarities and find max
        let mut sims = Vec::with_capacity(self.count);
        let mut max_sim = 0.0f32;
        for i in 0..self.count {
            let k_start = i * kd;
            let mut dot = 0.0f32;
            let n = kd.min(query.len());
            for j in 0..n {
                dot += self.keys[k_start + j] * query[j];
            }
            let sim = dot / q_norm; // keys are already normalized
            max_sim = max_sim.max(sim);
            sims.push(sim);
        }

        // Soft attention: exp(sim / temperature) for entries above threshold
        let temperature = 0.1f32;
        let mut weights = Vec::with_capacity(self.count);
        let mut weight_sum = 0.0f32;
        for i in 0..self.count {
            if sims[i] > threshold {
                let w = ((sims[i] - threshold) / temperature).exp() * self.strengths[i];
                weights.push(w);
                weight_sum += w;
                // Boost strength on retrieval (reconsolidation)
                self.strengths[i] = (self.strengths[i] * 1.05).min(2.0);
            } else {
                weights.push(0.0);
            }
        }

        // Weighted sum of values
        if weight_sum > 1e-8 {
            for i in 0..self.count {
                if weights[i] > 0.0 {
                    let w = weights[i] / weight_sum;
                    let v_start = i * vd;
                    for j in 0..vd {
                        result[j] += w * self.values[v_start + j];
                    }
                }
            }
        }

        (result, max_sim)
    }

    /// Decay all strengths. Call once per sleep cycle.
    pub fn decay_strengths(&mut self, factor: f32) {
        for s in &mut self.strengths[..self.count] {
            *s *= factor;
        }
    }

    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
}

impl Default for HippocampalCAM {
    fn default() -> Self { Self::new(0, 0, 0) }
}
