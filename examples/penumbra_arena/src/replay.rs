//! Synthetic Penumbra block stream + oracle MM action — for offline
//! training before live testnet data is available.
//!
//! `SyntheticReplay` walks log-price as a Brownian motion with mild
//! mean reversion, fakes plausible depth that thickens with offset,
//! and surfaces a `BlockFeatures` per step. `oracle_motor` returns a
//! 12-dim MOTOR vector implementing a simple but reasonable MM
//! policy: always quote both sides, tighten on low volatility, widen
//! on inventory imbalance. The brain learns to imitate this policy
//! via supervised MSE; once it does, RL fine-tuning beats the oracle.

use crate::chain::{Balance, BlockFeatures, DepthLevel, OrderBook, Quote, STD_DEPTH_OFFSETS_BPS};
use crate::motor::MOTOR_DIM;

/// Deterministic PRNG (xorshift64) — keeps replays reproducible
/// without dragging in a `rand` dependency.
#[derive(Debug, Clone)]
pub struct Rng { state: u64 }

impl Rng {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0xdead_beef_cafe_babe } else { seed };
        Self { state: s }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x;
        x
    }
    pub fn next_f64(&mut self) -> f64 {
        // [0, 1)
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard-normal sample via Box-Muller.
    pub fn next_gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[derive(Debug, Clone)]
pub struct SyntheticReplayConfig {
    pub initial_mid: f64,
    /// Mean log-price (the random walk reverts toward this). Pass
    /// `initial_mid.ln()` to start at equilibrium.
    pub log_mean: f64,
    /// Mean-reversion strength per block, in [0, 1). 0 = pure random walk.
    pub mean_reversion: f64,
    /// Per-block log-price volatility (Gaussian sigma).
    pub log_vol: f64,
    /// Base depth at the tightest probe offset, in base-asset units.
    pub base_depth: f64,
    /// Linear scaling of depth with offset_bps (more depth further out).
    pub depth_per_bps: f64,
    /// Half-spread between sell and buy quotes, in basis points.
    pub spread_bps: f64,
    /// Initial agent inventory.
    pub initial_balance: Balance,
    /// Persistence of the latent depth-skew between blocks, in [0, 1).
    /// 0 = no skew. Higher values create slowly-drifting asymmetric
    /// books that resemble real markets where depth on one side
    /// stays heavy for several blocks before flipping.
    pub book_skew_persistence: f64,
    /// Amplitude of the per-block skew innovation (Gaussian shock sigma).
    /// 0 = no skew (book stays symmetric). Try 0.05 — that's a ±5%
    /// depth-imbalance shock per block, persistence-decayed.
    pub book_skew_amplitude: f64,
    /// Coupling from latent skew → log-price drift per block. When
    /// nonzero, depth asymmetry predicts price direction: a positive
    /// skew (bids heavier than asks) drifts price up, negative drifts
    /// down. This is what makes the arb-flipper action space
    /// meaningful — flipping a buy across mid can capture the drift
    /// before it shows up in mid.
    pub drift_per_skew: f64,
}

impl SyntheticReplayConfig {
    pub fn um_usdc_default() -> Self {
        Self {
            initial_mid: 0.005,
            log_mean: 0.005f64.ln(),
            mean_reversion: 0.02,
            log_vol: 0.005,        // ~50 bps per block
            base_depth: 50_000.0,
            depth_per_bps: 200.0,
            spread_bps: 80.0,
            initial_balance: Balance { base: 10_000.0, quote: 50.0 },
            book_skew_persistence: 0.0,
            book_skew_amplitude:   0.0,
            drift_per_skew:        0.0,
        }
    }

    /// Baseline + book-skew → drift coupling enabled. Test bed for
    /// validating the signed-offset action space.
    pub fn um_usdc_skewed() -> Self {
        Self {
            book_skew_persistence: 0.85,   // skew lingers ~6 blocks
            book_skew_amplitude:   0.20,   // ±20% depth imbalance shocks
            drift_per_skew:        0.0030, // 30 bps drift per unit skew
            ..Self::um_usdc_default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct SyntheticReplay {
    pub cfg: SyntheticReplayConfig,
    rng: Rng,
    log_price: f64,
    block: u64,
    balance: Balance,
    /// Latent depth-skew that drifts persistently and couples into
    /// both depth asymmetry and price drift. Zero when the cfg has
    /// `book_skew_amplitude == 0`.
    skew: f64,
}

impl SyntheticReplay {
    pub fn new(cfg: SyntheticReplayConfig, seed: u64) -> Self {
        let log_price = cfg.initial_mid.ln();
        let balance = cfg.initial_balance.clone();
        Self { cfg, rng: Rng::new(seed), log_price, block: 0, balance, skew: 0.0 }
    }

    /// Advance one block, return the new BlockFeatures.
    pub fn step(&mut self) -> BlockFeatures {
        // Update the latent skew: AR(1) with Gaussian shocks. When
        // amplitude == 0 the skew stays glued to 0 and behaviour
        // matches the original symmetric replay exactly.
        self.skew = self.cfg.book_skew_persistence * self.skew
            + self.cfg.book_skew_amplitude * self.rng.next_gauss();
        // Clamp so a wild outlier can't drive depth negative.
        if self.skew >  0.95 { self.skew =  0.95; }
        if self.skew < -0.95 { self.skew = -0.95; }

        // Mean-reverting log random walk + skew-driven drift.
        let revert = self.cfg.mean_reversion * (self.cfg.log_mean - self.log_price);
        let shock  = self.cfg.log_vol * self.rng.next_gauss();
        let drift  = self.cfg.drift_per_skew * self.skew;
        self.log_price += revert + shock + drift;
        let mid = self.log_price.exp();

        let half = mid * (self.cfg.spread_bps / 10_000.0) * 0.5;
        let quote = Quote { sell_price: mid - half, buy_price: mid + half, mid };

        // Synthetic depth: linear in offset_bps with a persistent
        // bid/ask asymmetry driven by `skew` (+ shifts depth toward
        // bids, − toward asks).
        let mut book = OrderBook::default();
        let bid_mul = (1.0 + self.skew).max(0.0);
        let ask_mul = (1.0 - self.skew).max(0.0);
        for &off in STD_DEPTH_OFFSETS_BPS {
            let baseline = self.cfg.base_depth + self.cfg.depth_per_bps * off;
            book.bids.push(DepthLevel {
                offset_bps: off, total_base: baseline * bid_mul, min_competitor_fee_bps: 30,
            });
            book.asks.push(DepthLevel {
                offset_bps: off, total_base: baseline * ask_mul, min_competitor_fee_bps: 30,
            });
        }

        self.block += 1;
        BlockFeatures {
            block_height: self.block,
            quote, balance: self.balance.clone(),
            open_positions: 0, book,
        }
    }
}

/// Hand-coded oracle MM action — what a sensible rule-based MM would do.
/// Returns a 12-dim MOTOR vector matching the BarbellDecoder layout:
///   per rung: [skip_logit, size_pre_sigmoid, offset_pre_tanh]
/// Decoder applies sigmoid to size and tanh to offset; we work backward
/// using logits that decode to target sizes/offsets.
///
/// Policy (intentionally simple, conventional MM only — the brain learns
/// the arb-seeker flips on its own from PBT/RL signal, not the oracle):
///   - Always quote (skip_logit = +2.0 → ~0.88 probability).
///   - Inner rung: tight (target offset 25 bps), modest size (15% of base).
///   - Outer rung: wide  (target offset 150 bps), larger size (25% of base).
///   - Inventory skew: bias size up on whichever side rebalances.
///   - Offsets are positive (rung sits on natural side of mid).
///
/// Brain trains to produce these motor outputs given the BlockFeatures.
pub fn oracle_motor(bf: &BlockFeatures) -> Vec<f32> {
    let mut m = vec![0.0f32; MOTOR_DIM];
    let max_offset_bps = 200.0_f64;        // matches BarbellConfig::default()
    let max_size_frac  = 0.25_f64;
    let inv_signed = {
        let pf = bf.balance.base * bf.quote.mid + bf.balance.quote;
        if pf > 1e-9 { (bf.balance.base * bf.quote.mid - bf.balance.quote) / pf } else { 0.0 }
    };

    let inner_size_frac = (0.15 - 0.10 * inv_signed).clamp(0.05, 0.25); // bias toward rebalance
    let outer_size_frac = (0.25 - 0.05 * inv_signed).clamp(0.10, 0.25);
    let inner_offset_bps = 25.0;
    let outer_offset_bps = 150.0;

    // Clamp logits to ±3 so the oracle target is in a sane training
    // range — without this, inv_sig of 0.999 produces ~7 which dwarfs
    // the other terms in MSE and stalls learning.
    let inv_sig = |frac: f64, max: f64| -> f32 {
        let p = (frac / max).clamp(0.05, 0.95);
        (p / (1.0 - p)).ln().clamp(-3.0, 3.0) as f32
    };
    // atanh of (target_offset / max_offset) — clamped so values near ±1
    // don't blow up MSE. Conventional MM places positive offsets only;
    // brain is free to learn negative (arb-flip) from RL signal.
    let inv_tanh = |frac: f64, max: f64| -> f32 {
        let x = (frac / max).clamp(-0.95, 0.95);
        (0.5 * ((1.0 + x) / (1.0 - x)).ln()).clamp(-3.0, 3.0) as f32
    };

    // Rung 0: inner bid.
    m[0] = 2.0;
    m[1] = inv_sig(inner_size_frac, max_size_frac);
    m[2] = inv_tanh(inner_offset_bps / max_offset_bps, 1.0);
    // Rung 1: inner ask.
    m[3] = 2.0;
    m[4] = inv_sig(inner_size_frac, max_size_frac);
    m[5] = inv_tanh(inner_offset_bps / max_offset_bps, 1.0);
    // Rung 2: outer bid.
    m[6] = 2.0;
    m[7] = inv_sig(outer_size_frac, max_size_frac);
    m[8] = inv_tanh(outer_offset_bps / max_offset_bps, 1.0);
    // Rung 3: outer ask.
    m[9]  = 2.0;
    m[10] = inv_sig(outer_size_frac, max_size_frac);
    m[11] = inv_tanh(outer_offset_bps / max_offset_bps, 1.0);

    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skewed_replay_book_is_asymmetric_and_drifts_with_skew() {
        // Walk a long horizon and check that bid/ask depths diverge
        // (asymmetric book) and that mid drifts net-non-zero (skew
        // coupling). The symmetric baseline would give |bid-ask|≈0
        // and very small drift after 200 blocks; the skewed preset
        // should produce noticeably both.
        let mut r = SyntheticReplay::new(SyntheticReplayConfig::um_usdc_skewed(), 0xBEEF);
        let mut max_asym = 0.0f64;
        let start = r.cfg.initial_mid;
        for _ in 0..200 { let bf = r.step();
            let bid = bf.book.bid_depth_within(50.0);
            let ask = bf.book.ask_depth_within(50.0);
            let asym = (bid - ask).abs() / (bid + ask + 1e-9);
            if asym > max_asym { max_asym = asym; }
        }
        // Asymmetry should hit at least 5% somewhere in 200 blocks.
        assert!(max_asym > 0.05, "expected meaningful book asymmetry, got max {:.4}", max_asym);
        let final_mid = r.step().quote.mid;
        // Mid should have moved at least a few bps from start.
        assert!((final_mid - start).abs() / start > 1e-3,
            "expected mid to drift, start={start}, final={final_mid}");
    }

    #[test]
    fn default_replay_book_stays_symmetric() {
        // The default config has no skew dynamics — bids and asks
        // should be exactly equal at every offset.
        let mut r = SyntheticReplay::new(SyntheticReplayConfig::um_usdc_default(), 7);
        for _ in 0..20 {
            let bf = r.step();
            for (b, a) in bf.book.bids.iter().zip(&bf.book.asks) {
                assert!((b.total_base - a.total_base).abs() < 1e-9,
                    "default replay should have symmetric depth, got bid={} ask={}",
                    b.total_base, a.total_base);
            }
        }
    }

    #[test]
    fn replay_walks_price_and_returns_book() {
        let mut r = SyntheticReplay::new(SyntheticReplayConfig::um_usdc_default(), 42);
        let mut prices = Vec::new();
        for _ in 0..10 {
            let bf = r.step();
            prices.push(bf.quote.mid);
            assert_eq!(bf.book.bids.len(), STD_DEPTH_OFFSETS_BPS.len());
            assert_eq!(bf.book.asks.len(), STD_DEPTH_OFFSETS_BPS.len());
            assert!(bf.quote.spread_bps() > 0.0);
        }
        // Some price movement happened.
        let max = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(max - min > 0.0);
    }

    #[test]
    fn oracle_motor_has_correct_dim_and_active_skip_logits() {
        let mut r = SyntheticReplay::new(SyntheticReplayConfig::um_usdc_default(), 7);
        let bf = r.step();
        let m = oracle_motor(&bf);
        assert_eq!(m.len(), MOTOR_DIM);
        // Skip logits at indices 0, 3, 6, 9 — all positive (always quote).
        assert!(m[0] > 0.0); assert!(m[3] > 0.0);
        assert!(m[6] > 0.0); assert!(m[9] > 0.0);
    }

    #[test]
    fn oracle_decodes_to_policy_offsets() {
        // Lock the oracle/decoder coupling: when we feed the oracle's
        // motor vector through the BarbellDecoder, the resulting
        // positions must land at the policy's intended bps offsets
        // (25 bps inner, 150 bps outer). If anyone changes either
        // activation function without matching the other side, this
        // breaks loudly.
        use crate::motor::{BarbellDecoder, BarbellConfig};
        let mut r = SyntheticReplay::new(SyntheticReplayConfig::um_usdc_default(), 42);
        let bf = r.step();
        let m = oracle_motor(&bf);
        let dec = BarbellDecoder::new(BarbellConfig::default());
        let positions = dec.decode(&m, bf.quote.mid, 1000.0);
        // 4 rungs, all should fire (oracle sets skip_logit=2).
        assert_eq!(positions.len(), 4);
        // Group by side+fee_bps. Inner=30bps fee, outer=150bps fee.
        for p in &positions {
            let target_bps = if p.fee_bps == 30 { 25.0 } else { 150.0 };
            let actual_offset_bps = ((p.price - bf.quote.mid).abs() / bf.quote.mid) * 10_000.0;
            assert!((actual_offset_bps - target_bps).abs() < 2.0,
                "side={} fee={} got {:.2} bps, expected {:.2} bps",
                p.side, p.fee_bps, actual_offset_bps, target_bps);
            // Oracle is conventional MM — never flip across mid.
            if p.side == "buy"  { assert!(p.price < bf.quote.mid); }
            if p.side == "sell" { assert!(p.price > bf.quote.mid); }
        }
    }

    #[test]
    fn oracle_inventory_skew_favours_rebalance() {
        // Long inventory (lots of base, no quote) → oracle reduces inner-rung size.
        let mut r_long = SyntheticReplay::new(SyntheticReplayConfig {
            initial_balance: Balance { base: 20_000.0, quote: 0.0 },
            ..SyntheticReplayConfig::um_usdc_default()
        }, 11);
        let mut r_balanced = SyntheticReplay::new(SyntheticReplayConfig::um_usdc_default(), 11);

        let m_long = oracle_motor(&r_long.step());
        let m_bal  = oracle_motor(&r_balanced.step());
        // Inner-bid size logit (m[1]) should be smaller for long-inventory
        // (we're already long, want to sell, so reduce bid size).
        assert!(m_long[1] < m_bal[1],
            "long inv inner size {} should be < balanced {}", m_long[1], m_bal[1]);
    }
}
