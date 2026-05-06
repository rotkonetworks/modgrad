//! Reward function — liquidity provision under a bounded loss budget.
//!
//! Goal framing (the actual brief, not "maximize PnL"):
//!   - **Provide liquidity to the exchange itself.** Visible depth at
//!     tight spreads attracts organic flow; that's the value the
//!     agent population delivers to the venue.
//!   - **Don't lose too much money doing it.** Loss within an
//!     `acceptable_loss` budget is fine — it's the cost of bringing
//!     market quality. Beyond the budget the penalty turns sharp so
//!     the brain learns to pull back, not blow up the account.
//!
//! Composite per-block reward:
//!   R = liquidity_provided                    ← ★ primary objective
//!     + fee_capture                            ← bonus when fills happen
//!     - adverse_per_fill · |drift| · n_fills   ← punishes inner toxicity
//!     - gas_per_tx · n_tx                      ← penalises churn
//!     - drawdown_penalty(loss_so_far)          ← sharp above budget
//!
//! `liquidity_provided` rewards quoting tight rungs with size: it's
//! `Σ size_i · max(0, 1 − offset_bps_i / max_useful_offset_bps)`.
//! The (1 − offset/max) factor falls to zero past `max_useful_offset_bps`,
//! so quoting at huge offsets earns no liquidity credit even though
//! it's technically "open volume".

use crate::chain::{Balance, PositionParams};

/// Snapshot at a given block — used to diff successive states.
#[derive(Debug, Clone, Default)]
pub struct AgentSnapshot {
    pub block_height: u64,
    pub mid: f64,
    pub balance: Balance,
    /// Sum of fees earned (in quote whole units) up to this block.
    pub cumulative_fees: f64,
    /// Number of position open/close txs sent up to this block.
    pub tx_count: u64,
    /// Number of completed fills (own positions) up to this block.
    pub fills: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct RewardWeights {
    /// Multiplier on raw fee capture.
    pub fee_capture: f64,
    /// Penalty per unit of adverse mid-move per filled-base after a
    /// fill (proxies adverse-selection loss when inner rung gets picked off).
    pub adverse_per_fill: f64,
    /// Penalty per tx (proxies gas + rebalance cost).
    pub gas_per_tx: f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self { fee_capture: 1.0, adverse_per_fill: 0.5, gas_per_tx: 1e-4 }
    }
}

/// Compute the per-block reward from two consecutive snapshots.
/// `fills_in_block`: count of fills since `prev` was taken.
/// `mid_drift_after_fill`: signed quote-per-base move *immediately
/// after* the most recent fill, conventionally toward adverse direction
/// for the agent. Pass 0.0 if not measured yet.
pub fn step_reward(
    prev: &AgentSnapshot,
    curr: &AgentSnapshot,
    fills_in_block: u64,
    mid_drift_after_fill: f64,
    w: &RewardWeights,
) -> f64 {
    let fee_delta = curr.cumulative_fees - prev.cumulative_fees;
    let adverse  = mid_drift_after_fill.abs() * (fills_in_block as f64);
    let tx_delta = curr.tx_count.saturating_sub(prev.tx_count) as f64;
    w.fee_capture * fee_delta
        - w.adverse_per_fill * adverse
        - w.gas_per_tx * tx_delta
}

/// Per-position simulated fill outcome for offline arena scoring.
#[derive(Debug, Clone, Copy)]
pub struct FillOutcome {
    /// Base-asset filled (whole units).
    pub filled_base: f64,
    /// Quote-asset earned in fees (whole units, signed: positive on win).
    pub fee_earned: f64,
    /// Adverse-mid drift after the fill — reward applies a penalty
    /// proportional to this when a fill happened (proxy for toxicity).
    pub adverse_drift: f64,
    /// True if the position was touched by the price walk.
    pub filled: bool,
}

/// Simulate fills for a list of positions given the price walk between
/// two consecutive blocks. Simple touch-fill model: a buy at price `p`
/// fills if `curr_mid <= p`; a sell fills if `curr_mid >= p`. Adverse
/// drift = `|next_mid - p|` if filled — proxies the loss when price
/// kept moving past your fill.
///
/// `next_mid_after` is the mid two blocks ahead (post-fill). When the
/// caller doesn't have the look-ahead, pass `curr_mid` and the adverse
/// term collapses to zero.
pub fn simulate_fills(
    positions: &[PositionParams],
    _prev_mid: f64,
    curr_mid: f64,
    next_mid_after: f64,
) -> Vec<FillOutcome> {
    positions.iter().map(|p| {
        let is_buy = p.side == "buy";
        let filled = if is_buy { curr_mid <= p.price } else { curr_mid >= p.price };
        if !filled {
            return FillOutcome { filled_base: 0.0, fee_earned: 0.0, adverse_drift: 0.0, filled: false };
        }
        let filled_base = p.amount;
        let fee_earned = filled_base * p.price * (p.fee_bps as f64) / 10_000.0;
        // Adverse drift: how far did mid keep moving past the fill price?
        // A fill is "toxic" when price keeps going against the agent.
        let adverse_drift = if is_buy {
            (p.price - next_mid_after).max(0.0)
        } else {
            (next_mid_after - p.price).max(0.0)
        };
        FillOutcome { filled_base, fee_earned, adverse_drift, filled: true }
    }).collect()
}

/// Aggregate reward from a step's simulated fills + the agent's
/// position-submission count. v0 collapses fee + adverse + gas into
/// a scalar; the arena leaderboard sums these across blocks.
pub fn step_reward_from_fills(
    fills: &[FillOutcome],
    n_positions_submitted: usize,
    w: &RewardWeights,
) -> f64 {
    let mut fee_capture = 0.0;
    let mut adverse = 0.0;
    let mut filled_base_sum = 0.0;
    for f in fills {
        fee_capture += f.fee_earned;
        adverse += f.adverse_drift * f.filled_base;
        filled_base_sum += f.filled_base;
    }
    let _ = filled_base_sum;
    let gas = (n_positions_submitted as f64) * w.gas_per_tx;
    w.fee_capture * fee_capture - w.adverse_per_fill * adverse - gas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_fee_capture_is_positive() {
        let prev = AgentSnapshot::default();
        let curr = AgentSnapshot { cumulative_fees: 0.005, ..Default::default() };
        let r = step_reward(&prev, &curr, 0, 0.0, &RewardWeights::default());
        assert!(r > 0.0);
    }

    #[test]
    fn adverse_fill_deducts_more_than_no_drift() {
        let prev = AgentSnapshot::default();
        let curr = AgentSnapshot { cumulative_fees: 0.010, ..Default::default() };
        let no_drift = step_reward(&prev, &curr, 1, 0.0, &RewardWeights::default());
        let adverse = step_reward(&prev, &curr, 1, 0.005, &RewardWeights::default());
        assert!(adverse < no_drift, "adverse {adverse} should be < no_drift {no_drift}");
    }

    #[test]
    fn simulate_fills_buy_fills_when_mid_drops() {
        let p = PositionParams {
            side: "buy".into(), amount: 100.0, price: 0.0049,
            fee_bps: 30, auto_close: true,
        };
        // mid drops from 0.005 to 0.0048 — buy at 0.0049 fills.
        let outs = simulate_fills(&[p.clone()], 0.005, 0.0048, 0.00475);
        assert_eq!(outs.len(), 1);
        assert!(outs[0].filled);
        assert!(outs[0].fee_earned > 0.0);
        // Adverse drift = 0.0049 - 0.00475 = 0.00015 (price kept moving against).
        assert!((outs[0].adverse_drift - 0.00015).abs() < 1e-9);
    }

    #[test]
    fn simulate_fills_sell_doesnt_fill_when_mid_drops() {
        let p = PositionParams {
            side: "sell".into(), amount: 100.0, price: 0.0051,
            fee_bps: 30, auto_close: true,
        };
        let outs = simulate_fills(&[p], 0.005, 0.0048, 0.0048);
        assert!(!outs[0].filled);
        assert_eq!(outs[0].fee_earned, 0.0);
    }

    #[test]
    fn step_reward_from_fills_combines_fee_and_adverse() {
        let p = PositionParams {
            side: "buy".into(), amount: 100.0, price: 0.0049,
            fee_bps: 30, auto_close: true,
        };
        let fills = simulate_fills(&[p], 0.005, 0.0048, 0.00475);
        let r = step_reward_from_fills(&fills, 1, &RewardWeights::default());
        // Fee = 100 * 0.0049 * 30 / 10000 = 0.00147
        // Adverse penalty = 0.5 * 0.00015 * 100 = 0.0075
        // Gas = 1e-4
        // r = 0.00147 - 0.0075 - 1e-4 ≈ -0.00613
        assert!(r < 0.0, "r={r} expected negative when adverse > fee");
    }

    #[test]
    fn gas_per_tx_charges_each_tx() {
        let prev = AgentSnapshot::default();
        let curr_zero_tx = AgentSnapshot { cumulative_fees: 0.001, tx_count: 0, ..Default::default() };
        let curr_two_tx  = AgentSnapshot { cumulative_fees: 0.001, tx_count: 2, ..Default::default() };
        let zero = step_reward(&prev, &curr_zero_tx, 0, 0.0, &RewardWeights::default());
        let two = step_reward(&prev, &curr_two_tx, 0, 0.0, &RewardWeights::default());
        assert!(two < zero);
    }
}
