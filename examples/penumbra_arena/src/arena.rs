//! Arena — population of `MmAgent`s + leaderboard + PBT loop.
//!
//! v0 stub. Holds the agent population and a leaderboard keyed by
//! cumulative reward. Mutation / replacement / capital reallocation
//! land in follow-up slices.

use crate::reward::AgentSnapshot;

#[derive(Debug, Clone, Default)]
pub struct LeaderboardEntry {
    pub agent_id: u32,
    pub cumulative_reward: f64,
    pub last_snapshot: Option<AgentSnapshot>,
}

#[derive(Debug, Clone, Default)]
pub struct Leaderboard {
    pub entries: Vec<LeaderboardEntry>,
}

impl Leaderboard {
    pub fn new(n_agents: u32) -> Self {
        Self {
            entries: (0..n_agents).map(|i| LeaderboardEntry {
                agent_id: i,
                cumulative_reward: 0.0,
                last_snapshot: None,
            }).collect(),
        }
    }

    /// Sort entries by cumulative_reward descending.
    pub fn ranked(&self) -> Vec<&LeaderboardEntry> {
        let mut v: Vec<&LeaderboardEntry> = self.entries.iter().collect();
        v.sort_by(|a, b| b.cumulative_reward.partial_cmp(&a.cumulative_reward)
            .unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PbtPolicy {
    /// Quartile of the population to replace each PBT generation.
    pub bottom_replace_frac: f32,
    /// Mutation noise scale on top model's weights.
    pub mutation_noise: f32,
    /// Generation interval — how many `step` calls between PBT events.
    pub interval_steps: u32,
}

impl Default for PbtPolicy {
    fn default() -> Self {
        Self {
            bottom_replace_frac: 0.25,
            mutation_noise: 0.02,
            interval_steps: 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaderboard_ranks_descending() {
        let mut lb = Leaderboard::new(3);
        lb.entries[0].cumulative_reward = 1.0;
        lb.entries[1].cumulative_reward = 3.0;
        lb.entries[2].cumulative_reward = 2.0;
        let ranked = lb.ranked();
        assert_eq!(ranked[0].agent_id, 1);
        assert_eq!(ranked[1].agent_id, 2);
        assert_eq!(ranked[2].agent_id, 0);
    }
}
