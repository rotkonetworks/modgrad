//! `BarbellDecoder` — 12-dim brain output → list of LP position requests.
//!
//! Layout (out_dims = 12, four rungs × three params each):
//!   inner_bid:  [skip_logit, size_norm, offset_bps_signed]
//!   inner_ask:  [skip_logit, size_norm, offset_bps_signed]
//!   outer_bid:  [skip_logit, size_norm, offset_bps_signed]
//!   outer_ask:  [skip_logit, size_norm, offset_bps_signed]
//!
//! Fee tier per rung is fixed at construction (typically inner=30
//! bps, outer=150 bps). Sizes are scaled (sigmoid) by [0, MAX_SIZE_FRAC];
//! offsets are signed (tanh) over [−MAX_OFFSET_BPS, +MAX_OFFSET_BPS].
//! A positive offset places a buy rung *below* mid / sell rung *above*
//! mid (conventional MM); a negative offset flips the rung *across* mid
//! (aggressive arb-seeker — Penumbra's batch auction routes swap flow
//! through any position whose price is crossable, capturing free arb
//! when the brain detects skewed depth or stale mid).
//! Negative skip_logit → don't quote this rung this block.

use crate::chain::PositionParams;

pub const MOTOR_DIM: usize = 12;

#[derive(Debug, Clone)]
pub struct BarbellConfig {
    pub inner_fee_bps: u32,
    pub outer_fee_bps: u32,
    /// Maximum per-rung size as a fraction of available base balance.
    pub max_size_frac: f64,
    /// Maximum per-rung offset from mid, in basis points.
    pub max_offset_bps: f64,
    /// Drop the outer (high-fee, wide) rungs entirely. Useful for
    /// minimal-capital live runs where each open+close costs gas and
    /// the outer rungs fill rarely.
    pub skip_outer: bool,
    /// When `Some(N)`, the decoder ignores the brain's motor vector
    /// and emits a deterministic N-rung-per-side ladder at log-spaced
    /// offsets from `inner_offset_bps` (5) to `max_offset_bps`. Total
    /// positions = 2·N. Each rung sized at `max_size_frac` of the base
    /// balance. Use for volume-generation runs where the brain isn't
    /// the source of edge — we just want a thick visible quote book.
    pub ladder_rungs_per_side: Option<usize>,
    /// Tightest offset in the ladder (only used when
    /// `ladder_rungs_per_side` is set). Outermost is `max_offset_bps`.
    pub ladder_inner_bps: f64,
}

impl Default for BarbellConfig {
    fn default() -> Self {
        Self {
            inner_fee_bps: 30,
            outer_fee_bps: 150,
            max_size_frac: 0.25,
            max_offset_bps: 200.0,
            skip_outer: false,
            ladder_rungs_per_side: None,
            ladder_inner_bps: 5.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BarbellDecoder {
    cfg: BarbellConfig,
}

impl BarbellDecoder {
    pub fn new(cfg: BarbellConfig) -> Self { Self { cfg } }

    /// Decode a 12-dim motor vector into 0..4 `PositionParams`.
    /// `mid_price`: current quote-per-base midpoint.
    /// `available_base`: base-asset balance to allocate across rungs.
    /// When `cfg.ladder_rungs_per_side` is set, the motor vector is
    /// ignored and a deterministic ladder is emitted instead.
    pub fn decode(&self, motor: &[f32], mid_price: f64, available_base: f64) -> Vec<PositionParams> {
        if let Some(n_per_side) = self.cfg.ladder_rungs_per_side {
            return self.decode_ladder(n_per_side, mid_price, available_base);
        }
        assert!(motor.len() >= MOTOR_DIM,
            "motor.len()={} expected ≥ {}", motor.len(), MOTOR_DIM);

        let mut out = Vec::with_capacity(4);
        let inner = [
            ("buy",  &motor[0..3],   self.cfg.inner_fee_bps,  -1.0),  // inner bid below mid
            ("sell", &motor[3..6],   self.cfg.inner_fee_bps,   1.0),  // inner ask above mid
        ];
        let outer = [
            ("buy",  &motor[6..9],   self.cfg.outer_fee_bps,  -1.0),
            ("sell", &motor[9..12],  self.cfg.outer_fee_bps,   1.0),
        ];
        let rungs: Box<dyn Iterator<Item = _>> = if self.cfg.skip_outer {
            Box::new(inner.into_iter())
        } else {
            Box::new(inner.into_iter().chain(outer.into_iter()))
        };

        for (side, slice, fee_bps, sign) in rungs {
            let skip_logit = slice[0] as f64;
            if skip_logit <= 0.0 { continue; }      // brain says don't quote
            let size_frac  = sigmoid(slice[1] as f64) * self.cfg.max_size_frac;
            let offset_bps = (slice[2] as f64).tanh() * self.cfg.max_offset_bps;

            let amount = (size_frac * available_base).max(0.0);
            if amount <= 0.0 { continue; }
            let price = mid_price * (1.0 + sign * offset_bps / 10_000.0);

            out.push(PositionParams {
                side: side.to_string(),
                amount,
                price,
                fee_bps,
                auto_close: true,
            });
        }
        out
    }

    /// Deterministic ladder decoder — emits `2·n_per_side` positions
    /// at log-spaced offsets from `cfg.ladder_inner_bps` to
    /// `cfg.max_offset_bps`. Each side (buy/sell) gets `n_per_side`
    /// rungs sized at `cfg.max_size_frac` of `available_base`. Brain
    /// motor is ignored — for volume-generation runs.
    fn decode_ladder(&self, n_per_side: usize, mid_price: f64, available_base: f64) -> Vec<PositionParams> {
        let mut out = Vec::with_capacity(2 * n_per_side);
        if n_per_side == 0 { return out; }
        let inner = self.cfg.ladder_inner_bps.max(1.0);
        let outer = self.cfg.max_offset_bps.max(inner * 2.0);
        let amount = (self.cfg.max_size_frac * available_base).max(0.0);
        if amount <= 0.0 { return out; }
        for k in 0..n_per_side {
            let t = if n_per_side == 1 { 0.0 } else { k as f64 / (n_per_side - 1) as f64 };
            let offset_bps = inner * (outer / inner).powf(t);
            out.push(PositionParams {
                side: "buy".into(),
                amount,
                price: mid_price * (1.0 - offset_bps / 10_000.0),
                fee_bps: self.cfg.inner_fee_bps,
                auto_close: true,
            });
            out.push(PositionParams {
                side: "sell".into(),
                amount,
                price: mid_price * (1.0 + offset_bps / 10_000.0),
                fee_bps: self.cfg.inner_fee_bps,
                auto_close: true,
            });
        }
        out
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_logit_negative_yields_no_position() {
        let dec = BarbellDecoder::new(BarbellConfig::default());
        let motor = vec![-1.0f32; MOTOR_DIM];
        let positions = dec.decode(&motor, 0.013, 1000.0);
        assert!(positions.is_empty());
    }

    #[test]
    fn all_logits_positive_yields_four_positions() {
        let dec = BarbellDecoder::new(BarbellConfig::default());
        let mut motor = vec![1.0f32; MOTOR_DIM];
        // Make sizes meaningful (not vanishing through sigmoid).
        for i in 0..4 { motor[i*3 + 1] = 2.0; motor[i*3 + 2] = 0.0; }
        let positions = dec.decode(&motor, 0.013, 1000.0);
        assert_eq!(positions.len(), 4);
        // Two buys + two sells.
        let buys = positions.iter().filter(|p| p.side == "buy").count();
        let sells = positions.iter().filter(|p| p.side == "sell").count();
        assert_eq!((buys, sells), (2, 2));
    }

    #[test]
    fn inner_outer_fee_tiers_distinct() {
        let dec = BarbellDecoder::new(BarbellConfig::default());
        let mut motor = vec![1.0f32; MOTOR_DIM];
        for i in 0..4 { motor[i*3 + 1] = 2.0; }
        let positions = dec.decode(&motor, 0.013, 1000.0);
        // Two inner (lower fee) + two outer (higher fee).
        let inner = positions.iter().filter(|p| p.fee_bps == 30).count();
        let outer = positions.iter().filter(|p| p.fee_bps == 150).count();
        assert_eq!((inner, outer), (2, 2));
    }

    #[test]
    fn positive_offset_keeps_rung_on_natural_side() {
        let dec = BarbellDecoder::new(BarbellConfig::default());
        let mid = 0.005;
        // skip=+1, size=+2, offset=+2 (atanh-ish positive → rung sits below
        // mid for buys, above mid for sells — conventional MM placement).
        let mut motor = vec![0.0f32; MOTOR_DIM];
        for i in 0..4 { motor[i*3] = 1.0; motor[i*3+1] = 2.0; motor[i*3+2] = 2.0; }
        let positions = dec.decode(&motor, mid, 1000.0);
        for p in &positions {
            if p.side == "buy"  { assert!(p.price < mid, "buy at {} should be below mid {}", p.price, mid); }
            if p.side == "sell" { assert!(p.price > mid, "sell at {} should be above mid {}", p.price, mid); }
        }
    }

    #[test]
    fn negative_offset_flips_rung_across_mid_for_arb() {
        let dec = BarbellDecoder::new(BarbellConfig::default());
        let mid = 0.005;
        // Same shape but with negative offset logits — brain has decided to
        // place aggressive crossing orders (arb-seeker).
        let mut motor = vec![0.0f32; MOTOR_DIM];
        for i in 0..4 { motor[i*3] = 1.0; motor[i*3+1] = 2.0; motor[i*3+2] = -2.0; }
        let positions = dec.decode(&motor, mid, 1000.0);
        for p in &positions {
            if p.side == "buy"  { assert!(p.price > mid, "arb-buy at {} should cross above mid {}", p.price, mid); }
            if p.side == "sell" { assert!(p.price < mid, "arb-sell at {} should cross below mid {}", p.price, mid); }
        }
    }

    #[test]
    fn zero_offset_lands_at_mid() {
        let dec = BarbellDecoder::new(BarbellConfig::default());
        let mid = 0.005;
        let mut motor = vec![0.0f32; MOTOR_DIM];
        for i in 0..4 { motor[i*3] = 1.0; motor[i*3+1] = 2.0; motor[i*3+2] = 0.0; }
        let positions = dec.decode(&motor, mid, 1000.0);
        for p in &positions {
            assert!((p.price - mid).abs() < 1e-12, "tanh(0) → offset 0 → price = mid; got {}", p.price);
        }
    }
}
