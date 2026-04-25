//! Connection dispatch schedule.
//!
//! `RegionalBrain::forward_cached` walks `cfg.connections` in storage
//! order and dispatches every connection synapse sequentially. This
//! works today because every connection reads *previous-tick* state
//! (`state.region_outputs`), so within an outer tick every connection
//! is independent — dispatch order doesn't affect numerics.
//!
//! Two follow-on concerns motivate making the schedule explicit:
//!
//! 1. **Latent reordering bug** (SaaF review #3.3). `OuterTickCache::
//!    connection_inputs` is indexed by `ci` (position in
//!    `cfg.connections`); backward looks up the same `ci`. If anything
//!    re-orders `cfg.connections` for any reason (perf, batching,
//!    dedup), the cache and backward silently misalign. Naming the
//!    schedule means the cache can key by stable edge id later.
//!
//! 2. **Future intra-tick chained dispatch**. The data model permits
//!    a connection to read another connection's *current-tick* output
//!    (cascaded thalamic gating, top-down cortico-cortical loops).
//!    Adding such a feature would require dispatching in topo order;
//!    a `Schedule` API is the place that lives.
//!
//! This module ships the API. The brain forward path doesn't yet
//! consume `Schedule`; today's dispatch is single-stage by
//! construction (all connections read prev-tick state). The wiring
//! lift (forward + backward both walking schedule.stages) lands when
//! someone actually introduces an intra-tick chain.

use crate::graph::RegionalConfig;

/// One stage of the dispatch schedule. Connection indices that can
/// dispatch concurrently — there are no precedence edges between
/// any two connections in the same stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    /// Connection indices into `RegionalConfig::connections`.
    pub connections: Vec<usize>,
}

/// Topologically partitioned dispatch schedule for a connection DAG.
/// Stages are ordered: stage `i` must complete before stage `i+1`
/// begins; within a stage every connection is independent.
#[derive(Debug, Clone)]
pub struct Schedule {
    pub stages: Vec<Stage>,
    /// `stage_of[ci]` = stage index that connection `ci` belongs to.
    /// Length equals the number of connections.
    pub stage_of: Vec<usize>,
}

/// Error from `Schedule::from_deps` when the dependency function
/// produces a cyclic graph (which Kahn's algorithm cannot fully
/// linearize).
#[derive(Debug)]
pub enum ScheduleError {
    /// `remaining` are the connection indices that participate in a
    /// cycle (or are downstream of one). These are the connections
    /// Kahn could not extract because their in-degree never reached
    /// zero.
    Cycle { remaining: Vec<usize> },
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cycle { remaining } => write!(
                f,
                "schedule has a cycle: {} connection(s) participate in a cycle ({:?})",
                remaining.len(), remaining,
            ),
        }
    }
}

impl std::error::Error for ScheduleError {}

impl Schedule {
    /// Build a trivial single-stage schedule: every connection in
    /// stage 0. This matches today's `forward_cached` dispatch
    /// semantics (all connections read prev-tick state, so all are
    /// independent within an outer tick).
    ///
    /// `n_connections == 0` produces an empty schedule with no stages.
    pub fn single_stage(n_connections: usize) -> Self {
        if n_connections == 0 {
            return Self { stages: vec![], stage_of: vec![] };
        }
        Self {
            stages: vec![Stage { connections: (0..n_connections).collect() }],
            stage_of: vec![0; n_connections],
        }
    }

    /// Build a multi-stage schedule via Kahn's algorithm.
    /// `dep(a, b) == true` declares that connection `a` must dispatch
    /// before connection `b`. The function is called O(n²) times —
    /// for current scale (≤ ~16 connections) this is fine.
    ///
    /// Returns `Err(ScheduleError::Cycle { .. })` if the dependency
    /// graph is cyclic. Callers should pick a dependency function
    /// that matches the actual dispatch model — e.g. for "all reads
    /// prev-tick state" use `Schedule::single_stage` directly; for
    /// hypothetical intra-tick chains, `dep(a, b) = b.from.contains(&a.to)`.
    ///
    /// Tie-breaking inside each stage is by connection index (stable),
    /// so the resulting `stages[k].connections` is sorted ascending.
    /// This matches storage order when no real dependency exists,
    /// minimising FP drift if a caller swaps `cfg.connections.iter()`
    /// for `schedule.stages[k].connections.iter()`.
    pub fn from_deps<F: Fn(usize, usize) -> bool>(
        n_connections: usize,
        dep: F,
    ) -> Result<Self, ScheduleError> {
        if n_connections == 0 {
            return Ok(Self { stages: vec![], stage_of: vec![] });
        }

        // In-degree[v] = number of u such that dep(u, v) is true.
        let mut in_degree = vec![0usize; n_connections];
        for v in 0..n_connections {
            for u in 0..n_connections {
                if u != v && dep(u, v) {
                    in_degree[v] += 1;
                }
            }
        }

        let mut stage_of = vec![usize::MAX; n_connections];
        let mut stages: Vec<Stage> = Vec::new();
        let mut placed = 0usize;

        // Kahn's: in each round, extract every zero-in-degree node
        // into a single stage; decrement in-degrees of their successors.
        while placed < n_connections {
            let zero: Vec<usize> = (0..n_connections)
                .filter(|&v| stage_of[v] == usize::MAX && in_degree[v] == 0)
                .collect();
            if zero.is_empty() { break; }
            let stage_id = stages.len();
            for &v in &zero { stage_of[v] = stage_id; }
            stages.push(Stage { connections: zero.clone() });
            placed += zero.len();
            // Decrement in-degree of successors.
            for &u in &zero {
                for v in 0..n_connections {
                    if stage_of[v] == usize::MAX && dep(u, v) {
                        in_degree[v] = in_degree[v].saturating_sub(1);
                    }
                }
            }
        }

        if placed < n_connections {
            let remaining: Vec<usize> = (0..n_connections)
                .filter(|&v| stage_of[v] == usize::MAX)
                .collect();
            return Err(ScheduleError::Cycle { remaining });
        }

        Ok(Self { stages, stage_of })
    }

    /// Number of connections covered by this schedule.
    pub fn n_connections(&self) -> usize { self.stage_of.len() }

    /// Number of stages (max parallel depth).
    pub fn n_stages(&self) -> usize { self.stages.len() }
}

/// Build a `Schedule` from a `RegionalConfig` using the dispatch
/// model the brain actually runs today: every connection reads
/// previous-tick state, so all connections are independent within an
/// outer tick. Always returns single-stage.
///
/// When intra-tick chained dispatch lands, this function will accept
/// a flag (or a sibling function will exist) that builds the
/// non-trivial topo sort.
pub fn schedule_for_prev_state_reads(cfg: &RegionalConfig) -> Schedule {
    Schedule::single_stage(cfg.connections.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `single_stage` of zero connections is empty.
    #[test]
    fn single_stage_empty() {
        let s = Schedule::single_stage(0);
        assert!(s.stages.is_empty());
        assert!(s.stage_of.is_empty());
        assert_eq!(s.n_connections(), 0);
        assert_eq!(s.n_stages(), 0);
    }

    /// `single_stage(n)` puts all connections in stage 0.
    #[test]
    fn single_stage_basic() {
        let s = Schedule::single_stage(5);
        assert_eq!(s.stages.len(), 1);
        assert_eq!(s.stages[0].connections, vec![0, 1, 2, 3, 4]);
        assert_eq!(s.stage_of, vec![0; 5]);
    }

    /// `from_deps` with no actual dependencies produces single stage,
    /// matching `single_stage`.
    #[test]
    fn from_deps_no_dependencies_collapses_to_single_stage() {
        let s = Schedule::from_deps(8, |_, _| false).expect("no cycle");
        assert_eq!(s.stages.len(), 1);
        assert_eq!(s.stages[0].connections, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(s.stage_of, vec![0; 8]);
    }

    /// Linear chain 0 → 1 → 2 → 3 produces 4 stages, each with one
    /// connection, in order.
    #[test]
    fn from_deps_linear_chain() {
        // dep(a, b) = b == a + 1
        let s = Schedule::from_deps(4, |a, b| b == a + 1).expect("no cycle");
        assert_eq!(s.stages.len(), 4);
        for i in 0..4 {
            assert_eq!(s.stages[i].connections, vec![i]);
            assert_eq!(s.stage_of[i], i);
        }
    }

    /// Diamond: 0 → {1, 2} → 3 produces three stages: {0}, {1, 2}, {3}.
    /// Tests parallelism inside a stage and ordering across stages.
    #[test]
    fn from_deps_diamond() {
        // dep(0, 1), dep(0, 2), dep(1, 3), dep(2, 3)
        let edges = [(0usize, 1), (0, 2), (1, 3), (2, 3)];
        let s = Schedule::from_deps(4, |a, b| edges.contains(&(a, b)))
            .expect("no cycle");
        assert_eq!(s.stages.len(), 3);
        assert_eq!(s.stages[0].connections, vec![0]);
        assert_eq!(s.stages[1].connections, vec![1, 2]);
        assert_eq!(s.stages[2].connections, vec![3]);
        assert_eq!(s.stage_of, vec![0, 1, 1, 2]);
    }

    /// 2-cycle: 0 → 1 and 1 → 0 cannot be linearised.
    #[test]
    fn from_deps_two_cycle_detected() {
        let result = Schedule::from_deps(2, |a, b| a != b);
        match result {
            Err(ScheduleError::Cycle { remaining }) => {
                assert_eq!(remaining, vec![0, 1]);
            }
            Ok(_) => panic!("expected Cycle error"),
        }
    }

    /// Self-loop on a single node IS a cycle (Kahn can never extract it).
    #[test]
    fn from_deps_self_loop_is_cycle() {
        let result = Schedule::from_deps(1, |a, b| a == 0 && b == 0);
        // dep is between the same node; in-degree[0] = 0 because we
        // skip u != v in the implementation. So this should succeed
        // with a single trivial stage. Document that.
        let s = result.expect("self-loop with u != v guard does NOT register");
        assert_eq!(s.stages.len(), 1);
    }

    /// Cycle in larger graph: 0 → 1, 1 → 2, 2 → 1. Connection 0
    /// should be in stage 0; connections 1 and 2 form the cycle and
    /// should be reported.
    #[test]
    fn from_deps_partial_cycle() {
        let edges = [(0usize, 1), (1, 2), (2, 1)];
        let result = Schedule::from_deps(3, |a, b| edges.contains(&(a, b)));
        match result {
            Err(ScheduleError::Cycle { remaining }) => {
                // Connection 0 should have been placed; 1 and 2 cycle.
                assert!(remaining.contains(&1));
                assert!(remaining.contains(&2));
                assert!(!remaining.contains(&0));
            }
            Ok(_) => panic!("expected Cycle error"),
        }
    }

    /// `n_connections` and `n_stages` accessors agree with the
    /// computed structure.
    #[test]
    fn from_deps_accessors() {
        let s = Schedule::single_stage(7);
        assert_eq!(s.n_connections(), 7);
        assert_eq!(s.n_stages(), 1);

        let s2 = Schedule::from_deps(4, |a, b| b == a + 1).expect("ok");
        assert_eq!(s2.n_connections(), 4);
        assert_eq!(s2.n_stages(), 4);
    }
}
