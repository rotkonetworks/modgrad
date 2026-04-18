//! Replay buffer for prioritized experience consolidation.
//!
//! Ring buffer of high-surprise experiences. During sleep,
//! high-surprise replays get priority — the brain consolidates
//! surprising events more aggressively than boring ones.

use serde::{Deserialize, Serialize};

/// Ring buffer of high-surprise experiences for prioritized consolidation.
///
/// Only stores sequences where surprise > threshold. During sleep,
/// high-surprise replays get priority — the brain consolidates
/// surprising events more aggressively than boring ones.
///
/// Ported from zish/src/inference/ctm.zig line 1341.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    pub entries: Vec<ReplayEntry>,
    pub capacity: usize,
    pub write_pos: usize,
    pub threshold: f32,  // min surprise to store
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEntry {
    pub observation: Vec<f32>,  // d_input-dim sensory observation (what the organism saw)
    pub surprise: f32,          // mean surprise over the sequence
    pub timestamp: f64,         // unix timestamp
    pub reward: f32,            // reward received for this episode
    pub action: usize,          // action taken (motor neuron index)
    pub correct: bool,          // was the action correct?
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self::new(64, 1.5)
    }
}

impl ReplayBuffer {
    pub fn new(capacity: usize, threshold: f32) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
            threshold,
        }
    }

    /// Store an experience if surprising enough.
    pub fn push(&mut self, observation: Vec<f32>, surprise: f32) {
        self.push_episode(observation, surprise, 0.0, 0, false);
    }

    /// Store a full episode: observation + action + reward + salience.
    /// Only stores if surprise exceeds threshold. Prioritized: high-surprise
    /// episodes evict low-surprise ones when buffer is full.
    pub fn push_episode(&mut self, observation: Vec<f32>, surprise: f32,
                        reward: f32, action: usize, correct: bool) {
        if surprise < self.threshold { return; }

        let entry = ReplayEntry {
            observation, surprise, reward, action, correct,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        };

        if self.entries.len() < self.capacity {
            self.entries.push(entry);
        } else {
            // Evict lowest-surprise entry instead of oldest (prioritized replay)
            let min_idx = self.entries.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.surprise.partial_cmp(&b.surprise).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if surprise > self.entries[min_idx].surprise {
                self.entries[min_idx] = entry;
            }
        }
        self.write_pos += 1;
    }

    /// Get entries sorted by surprise (highest first) for prioritized replay.
    pub fn prioritized(&self) -> Vec<&ReplayEntry> {
        let mut sorted: Vec<&ReplayEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.surprise.partial_cmp(&a.surprise).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Prune old entries with low surprise.
    pub fn prune(&mut self, max_age_secs: f64, decay_threshold: f32) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        self.entries.retain(|e| {
            let age = now - e.timestamp;
            !(age > max_age_secs && e.surprise < decay_threshold)
        });
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}
