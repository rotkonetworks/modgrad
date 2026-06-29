//! Reverse-replay credit assignment — prioritized DAgger over a policy's own
//! mistakes.
//!
//! Roll a policy out in some environment; wherever its action disagrees with an
//! expert, those states are the mistakes to learn from. On a FAILED rollout
//! (looped / ran out of budget without reaching the goal) the mistakes are
//! ordered BACKWARD from the failure — assigning credit/blame from the point of
//! failure outward.
//!
//! This is the **awake, planning-time** counterpart to [`crate::dream`]'s
//! sleep-time consolidation replay. Together they mirror the two roles of
//! hippocampal replay: reverse replay for credit assignment during behaviour
//! (Foster & Wilson, 2006) and consolidation replay during rest (Wilson &
//! McNaughton, 1994; Ólafsdóttir, Bush & Barry, 2018).
//!
//! Generic infrastructure: these functions operate on an already-collected
//! `(state, action)` trajectory plus an expert-oracle closure. No model,
//! environment, or task type is baked in — the same primitive drives a grid
//! maze planner, a language policy, or any sequential decision policy that can
//! be rolled out and scored against an expert.

/// One step of a rolled-out trajectory: the state visited and the action the
/// policy took there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayStep<S> {
    pub state: S,
    pub action: usize,
}

impl<S> ReplayStep<S> {
    pub fn new(state: S, action: usize) -> Self {
        Self { state, action }
    }
}

/// A policy's rolled-out trajectory plus whether it ended in failure.
#[derive(Clone, Debug)]
pub struct Trajectory<S> {
    pub steps: Vec<ReplayStep<S>>,
    /// `true` if the rollout looped or hit its step budget without success.
    pub failed: bool,
}

impl<S> Trajectory<S> {
    pub fn new(steps: Vec<ReplayStep<S>>, failed: bool) -> Self {
        Self { steps, failed }
    }
}

/// The states where the policy's action disagreed with `expert` — the
/// mistakes to replay. On a failed rollout they are reversed so the states
/// nearest the failure come first (reverse replay = credit assignment from the
/// failure backward). Empty when the policy made no mistakes.
///
/// `expert` returns `Some(expert_action)` for states it has an opinion about,
/// or `None` to skip (e.g. terminal / undecidable states).
pub fn mistake_states<S, E>(traj: &Trajectory<S>, expert: E) -> Vec<S>
where
    S: Clone,
    E: Fn(&S) -> Option<usize>,
{
    let mut mistakes: Vec<S> = traj
        .steps
        .iter()
        .filter(|st| expert(&st.state).is_some_and(|e| e != st.action))
        .map(|st| st.state.clone())
        .collect();
    if traj.failed {
        mistakes.reverse();
    }
    mistakes
}

/// Build the states to train on this rollout: prefer the policy's mistakes
/// (reverse-replay ordered), falling back to *all* visited states when it made
/// none. This is the state-selection that focuses learning on the policy's own
/// failure distribution — the fix for compounding behaviour-cloning error.
pub fn prioritized_replay<S, E>(traj: &Trajectory<S>, expert: E) -> Vec<S>
where
    S: Clone,
    E: Fn(&S) -> Option<usize>,
{
    let mistakes = mistake_states(traj, &expert);
    if mistakes.is_empty() {
        traj.steps.iter().map(|s| s.state.clone()).collect()
    } else {
        mistakes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traj(steps: &[(i32, usize)], failed: bool) -> Trajectory<i32> {
        Trajectory::new(
            steps.iter().map(|&(s, a)| ReplayStep::new(s, a)).collect(),
            failed,
        )
    }

    // expert: state s wants action (s as usize % 4).
    fn expert(s: &i32) -> Option<usize> {
        Some(*s as usize % 4)
    }

    #[test]
    fn picks_only_the_mistakes() {
        // states 0..4 with the policy taking the expert action at 0 and 2,
        // wrong at 1 and 3.
        let t = traj(&[(0, 0), (1, 3), (2, 2), (3, 0)], false);
        assert_eq!(mistake_states(&t, expert), vec![1, 3]);
    }

    #[test]
    fn failure_reverses_order_credit_from_failure_backward() {
        let t = traj(&[(1, 3), (3, 0), (5, 0)], true); // all wrong, failed
        // reversed: nearest the failure (last) first.
        assert_eq!(mistake_states(&t, expert), vec![5, 3, 1]);
    }

    #[test]
    fn falls_back_to_all_states_when_no_mistakes() {
        let t = traj(&[(0, 0), (4, 0), (8, 0)], false); // all correct
        assert!(mistake_states(&t, expert).is_empty());
        assert_eq!(prioritized_replay(&t, expert), vec![0, 4, 8]);
    }
}
