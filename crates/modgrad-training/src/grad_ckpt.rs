//! Gradient checkpointing: trade compute for memory.
//!
//! Instead of caching all T ticks' intermediates for backward,
//! only cache every Kth tick. During backward, recompute the
//! missing ticks by re-running forward from the nearest checkpoint.
//!
//! Memory: O(T/K) instead of O(T). Compute: ~1.5× instead of 1×.
//! Essential for training with many ticks or large models.
//!
//! This is architecture-agnostic — it wraps any Brain impl.

/// Policy for which ticks to checkpoint.
#[derive(Debug, Clone)]
pub enum CheckpointPolicy {
    /// Cache every tick (default, no memory saving).
    All,
    /// Cache every Kth tick. Others are recomputed during backward.
    EveryK(usize),
    /// Cache only first and last tick (maximum memory saving).
    Endpoints,
    /// Cache sqrt(T) evenly spaced ticks (optimal tradeoff).
    Sqrt,
}

impl CheckpointPolicy {
    /// Given T total ticks, which ones should be cached?
    pub fn checkpointed_ticks(&self, total_ticks: usize) -> Vec<usize> {
        match self {
            CheckpointPolicy::All => (0..total_ticks).collect(),
            CheckpointPolicy::EveryK(k) => {
                let k = (*k).max(1);
                (0..total_ticks).filter(|t| t % k == 0 || *t == total_ticks - 1).collect()
            }
            CheckpointPolicy::Endpoints => {
                if total_ticks <= 2 {
                    (0..total_ticks).collect()
                } else {
                    vec![0, total_ticks - 1]
                }
            }
            CheckpointPolicy::Sqrt => {
                let k = (total_ticks as f32).sqrt().ceil() as usize;
                CheckpointPolicy::EveryK(k).checkpointed_ticks(total_ticks)
            }
        }
    }

    /// Is this tick checkpointed?
    pub fn is_checkpointed(&self, tick: usize, total_ticks: usize) -> bool {
        match self {
            CheckpointPolicy::All => true,
            CheckpointPolicy::EveryK(k) => tick % k == 0 || tick == total_ticks - 1,
            CheckpointPolicy::Endpoints => tick == 0 || tick == total_ticks - 1,
            CheckpointPolicy::Sqrt => {
                let k = (total_ticks as f32).sqrt().ceil() as usize;
                tick % k == 0 || tick == total_ticks - 1
            }
        }
    }

    /// Number of cached ticks for T total.
    pub fn n_cached(&self, total_ticks: usize) -> usize {
        self.checkpointed_ticks(total_ticks).len()
    }

    /// Memory saving ratio vs caching all ticks.
    pub fn memory_ratio(&self, total_ticks: usize) -> f32 {
        self.n_cached(total_ticks) as f32 / total_ticks.max(1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_k_policy() {
        let p = CheckpointPolicy::EveryK(3);
        let cached = p.checkpointed_ticks(10);
        // 0, 3, 6, 9 (last always included)
        assert_eq!(cached, vec![0, 3, 6, 9]);
        assert!(p.is_checkpointed(0, 10));
        assert!(!p.is_checkpointed(1, 10));
        assert!(p.is_checkpointed(9, 10));
    }

    #[test]
    fn sqrt_policy() {
        let p = CheckpointPolicy::Sqrt;
        // T=16 → K=4 → cache 0,4,8,12,15
        let cached = p.checkpointed_ticks(16);
        assert!(cached.contains(&0));
        assert!(cached.contains(&15));
        assert_eq!(p.memory_ratio(16), cached.len() as f32 / 16.0);
        eprintln!("  sqrt(16): {:?}, memory ratio: {:.1}%",
            cached, p.memory_ratio(16) * 100.0);
    }

    #[test]
    fn endpoints_policy() {
        let p = CheckpointPolicy::Endpoints;
        let cached = p.checkpointed_ticks(100);
        assert_eq!(cached, vec![0, 99]);
        assert_eq!(p.memory_ratio(100), 0.02); // 2% memory
    }

    #[test]
    fn all_policy() {
        let p = CheckpointPolicy::All;
        assert_eq!(p.n_cached(10), 10);
        assert_eq!(p.memory_ratio(10), 1.0);
    }
}
