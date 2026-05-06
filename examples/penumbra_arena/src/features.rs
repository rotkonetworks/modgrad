//! Feature encoder — `BlockFeatures` + agent state → observation `Vec<f32>`.
//!
//! Layout (raw_obs_dim = 24):
//!   [0]   log(mid_price)
//!   [1]   current spread_bps / 1000
//!   [2]   inventory_balance_signed = (base * mid - quote) / portfolio_value
//!   [3]   open_position_count_norm = open_positions / 8
//!   [4..8]    recent log-returns (last 4 blocks)
//!   [8..12]   recent spread_bps / 1000 (last 4 blocks)
//!   [12..16]  recent own-fill rate per block (last 4 blocks)
//!   [16..20]  bid-side depth at offsets [10, 50, 100, 200] bps
//!             (log1p-normalised so big depth doesn't saturate)
//!   [20..24]  ask-side depth at the same offsets
//!
//! The depth slots are what the brain uses to decide where to
//! position rungs — if the inner band already has heavy depth at
//! 50 bps, quoting tighter (lower offset) is the value-add. If the
//! curve is empty inside 100 bps the venue desperately needs liquidity
//! and the brain should quote close to mid. Either way: data-driven.

use crate::chain::{BlockFeatures, STD_DEPTH_OFFSETS_BPS};

pub const RAW_OBS_DIM: usize = 24;
pub const RECENT_BLOCKS: usize = 4;
pub const N_DEPTH_LEVELS: usize = 4;

/// Per-block snapshot stored in the encoder's ring buffer.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockSnapshot {
    pub mid: f64,
    pub spread_bps: f64,
    pub own_fills_in_block: f64,
}

#[derive(Debug, Clone)]
pub struct FeatureEncoder {
    history: Vec<BlockSnapshot>,
}

impl FeatureEncoder {
    pub fn new() -> Self {
        Self { history: Vec::with_capacity(RECENT_BLOCKS) }
    }

    pub fn push(&mut self, snap: BlockSnapshot) {
        if self.history.len() >= RECENT_BLOCKS {
            self.history.remove(0);
        }
        self.history.push(snap);
    }

    /// Build the observation vector. Pads with zeros when history
    /// is shorter than `RECENT_BLOCKS`.
    pub fn encode(&self, bf: &BlockFeatures) -> Vec<f32> {
        let mut obs = vec![0.0f32; RAW_OBS_DIM];
        obs[0] = (bf.quote.mid.max(1e-9)).ln() as f32;
        obs[1] = (bf.quote.spread_bps().min(10_000.0).max(-10_000.0) / 1000.0) as f32;
        let portfolio_value = bf.balance.base * bf.quote.mid + bf.balance.quote;
        if portfolio_value > 1e-9 {
            obs[2] = ((bf.balance.base * bf.quote.mid - bf.balance.quote) / portfolio_value) as f32;
        }
        obs[3] = (bf.open_positions as f32) / 8.0;

        // Recent windows. Walk newest→oldest, fill from index 4/8/12.
        for (idx, s) in self.history.iter().rev().enumerate().take(RECENT_BLOCKS) {
            let off_ret = 4 + idx;
            let off_spr = 8 + idx;
            let off_fil = 12 + idx;
            // log-return between current snapshot and this past one.
            let prev = s.mid.max(1e-9);
            let ret = (bf.quote.mid.max(1e-9) / prev).ln();
            obs[off_ret] = ret as f32;
            obs[off_spr] = (s.spread_bps / 1000.0) as f32;
            obs[off_fil] = s.own_fills_in_block as f32;
        }

        // Depth: walk the standard probe offsets, log1p-normalise the
        // total_base on each side so a few orders of magnitude don't
        // saturate the input. Indices 16..20 = bids, 20..24 = asks,
        // matching the offset order in STD_DEPTH_OFFSETS_BPS.
        for (i, &off) in STD_DEPTH_OFFSETS_BPS.iter().enumerate().take(N_DEPTH_LEVELS) {
            let bid = bf.book.bids.iter().find(|d| (d.offset_bps - off).abs() < 1e-6)
                .map(|d| d.total_base).unwrap_or(0.0);
            let ask = bf.book.asks.iter().find(|d| (d.offset_bps - off).abs() < 1e-6)
                .map(|d| d.total_base).unwrap_or(0.0);
            obs[16 + i] = (bid.max(0.0).ln_1p()) as f32;
            obs[20 + i] = (ask.max(0.0).ln_1p()) as f32;
        }
        obs
    }
}

impl Default for FeatureEncoder {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{Balance, DepthLevel, OrderBook, Quote};

    fn synth_book() -> OrderBook {
        let mut b = OrderBook::default();
        for &off in STD_DEPTH_OFFSETS_BPS {
            b.bids.push(DepthLevel { offset_bps: off, total_base: 100.0 * off, min_competitor_fee_bps: 30 });
            b.asks.push(DepthLevel { offset_bps: off, total_base: 100.0 * off, min_competitor_fee_bps: 30 });
        }
        b
    }

    #[test]
    fn encode_shape_matches_dim() {
        let enc = FeatureEncoder::new();
        let bf = BlockFeatures {
            block_height: 1,
            quote: Quote { sell_price: 0.013, buy_price: 0.0132, mid: 0.0131 },
            balance: Balance { base: 100.0, quote: 1.0 },
            open_positions: 2,
            book: synth_book(),
        };
        let obs = enc.encode(&bf);
        assert_eq!(obs.len(), RAW_OBS_DIM);
        // log mid is finite, spread is non-zero
        assert!(obs[0].is_finite());
        assert!(obs[1] != 0.0);
        // Depth slots populated and non-zero with synthetic book.
        for i in 16..24 { assert!(obs[i] > 0.0, "depth slot {i} zero"); }
    }

    #[test]
    fn history_ring_caps_at_recent_blocks() {
        let mut enc = FeatureEncoder::new();
        for i in 0..(RECENT_BLOCKS + 3) {
            enc.push(BlockSnapshot {
                mid: 0.013 + (i as f64) * 1e-4,
                spread_bps: 100.0,
                own_fills_in_block: i as f64,
            });
        }
        assert_eq!(enc.history.len(), RECENT_BLOCKS);
    }
}
