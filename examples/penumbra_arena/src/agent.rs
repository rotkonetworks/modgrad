//! `MmAgent<D>` — single market maker driven by a typed `RegionalWeightsTyped<D>`.
//!
//! v0 carries: weights, optimizer, feature encoder, motor decoder, chain
//! handle, and the per-region region-state (activated/trace/sync) which
//! evolves across blocks. One `step` call = one block's worth of work:
//! read state → encode → forward → decode → diff against current
//! positions → submit. Training (REINFORCE-style on realized PnL) is
//! a follow-up slice.

use anyhow::{Context, Result};

use modgrad_ctm::graph::{
    RegionalAdamWTyped, RegionalGradientsTyped, RegionalWeights, RegionalWeightsTyped,
};
use modgrad_device::backend::tensor::Device;

use crate::chain::{Pair, PenumbraChain, PositionId, PositionParams};
use crate::features::{BlockSnapshot, FeatureEncoder, RAW_OBS_DIM};
use crate::motor::{BarbellConfig, BarbellDecoder};
use crate::reward::AgentSnapshot;

/// Per-region state carried across blocks (one outer-tick per block).
#[derive(Debug, Clone)]
pub struct RegionState {
    pub region_activated: Vec<Vec<f32>>,
    pub region_trace:     Vec<Vec<f32>>,
    pub global_alpha:     Vec<f32>,
    pub global_beta:      Vec<f32>,
}

impl RegionState {
    pub fn from_weights(w: &RegionalWeights) -> Self {
        let n_sync = w.config.n_global_sync;
        Self {
            region_activated: w.regions.iter().map(|r| r.start_activated.clone()).collect(),
            region_trace:     w.regions.iter().map(|r| r.start_trace.clone()).collect(),
            global_alpha:     vec![0.0f32; n_sync],
            global_beta:      vec![1.0f32; n_sync],
        }
    }

    /// Reset to the brain's start states without reallocating — the
    /// outer Vecs and inner buffers are reused. Hot-path replacement
    /// for `*self = Self::from_weights(w)` when called many times
    /// (every epoch / every eval seed).
    pub fn reset_in_place(&mut self, w: &RegionalWeights) {
        for (dst, r) in self.region_activated.iter_mut().zip(&w.regions) {
            dst.clone_from_slice(&r.start_activated);
        }
        for (dst, r) in self.region_trace.iter_mut().zip(&w.regions) {
            dst.clone_from_slice(&r.start_trace);
        }
        self.global_alpha.fill(0.0);
        self.global_beta.fill(1.0);
    }
}

/// Single market-maker agent.
pub struct MmAgent<D: Device, C: PenumbraChain> {
    pub weights:   RegionalWeightsTyped<D>,
    pub optimizer: RegionalAdamWTyped<D>,
    pub encoder:   FeatureEncoder,
    pub motor:     BarbellDecoder,
    pub chain:     C,
    pub state:     RegionState,
    pub pair:      Pair,
    pub last_snapshot: Option<AgentSnapshot>,
    /// Currently-open positions known to this agent (best-effort —
    /// reconciled against `chain.pending_fills` each step). Stored
    /// as (id, params_at_open) so the smart diff can match desired
    /// positions against existing ones without re-querying the chain.
    pub open_positions: Vec<(PositionId, PositionParams)>,
    /// Match tolerance for the smart-diff: tighter = fresher quotes,
    /// more gas; looser = less churn.
    pub match_price_tol_bps: f64,
    pub match_size_tol_frac: f64,
    /// Arb-flip count from the most recent `step()` — callers
    /// accumulate if they care about totals.
    pub last_step_arb_flips: u64,
}

impl<D: Device, C: PenumbraChain> MmAgent<D, C> {
    pub fn new(
        untyped: &RegionalWeights,
        weights: RegionalWeightsTyped<D>,
        chain: C,
        pair: Pair,
        motor_cfg: BarbellConfig,
    ) -> Result<Self> {
        // Sanity: brain's raw_obs_dim must match the encoder's RAW_OBS_DIM
        // (brain config is set up to match by the eight_region_mm preset).
        if weights.config.raw_obs_dim != RAW_OBS_DIM {
            anyhow::bail!(
                "brain raw_obs_dim {} ≠ encoder RAW_OBS_DIM {} — \
                 use eight_region_mm preset",
                weights.config.raw_obs_dim, RAW_OBS_DIM,
            );
        }
        let optimizer = RegionalAdamWTyped::<D>::new(&weights)
            .context("RegionalAdamWTyped::new")?
            .with_lr(1e-4).with_clip(1.0);
        let state = RegionState::from_weights(untyped);
        Ok(Self {
            weights, optimizer,
            encoder: FeatureEncoder::new(),
            motor: BarbellDecoder::new(motor_cfg),
            chain, pair, state,
            last_snapshot: None,
            open_positions: Vec::new(),
            match_price_tol_bps: 30.0,
            match_size_tol_frac: 0.10,
            last_step_arb_flips: 0,
        })
    }

    /// Single block step: read chain → encode → forward → decode → diff
    /// positions → submit. Returns the list of `PositionParams` it
    /// decided to open this block (mainly useful for dry-run logging).
    pub fn step(&mut self) -> Result<Vec<PositionParams>> {
        // 1. Read chain state.
        let bf = self.chain.block_features(&self.pair)
            .context("block_features")?;

        // 2. Encode observation.
        let obs = self.encoder.encode(&bf);

        // 3. Forward through the brain.
        let out = self.weights.regional_forward_typed(
            &obs,
            &mut self.state.region_activated,
            &mut self.state.region_trace,
            &mut self.state.global_alpha,
            &mut self.state.global_beta,
        ).map_err(|e| anyhow::anyhow!("regional_forward_typed: {e}"))?;

        // 4. Decode the last-tick MOTOR vector into barbell positions.
        let last_tick = out.predictions.len().saturating_sub(1);
        let motor_vec = &out.predictions[last_tick];
        let positions = self.motor.decode(motor_vec, bf.quote.mid, bf.balance.base);

        let mid = bf.quote.mid;
        self.last_step_arb_flips = positions.iter()
            .filter(|p| p.is_arb_flip(mid)).count() as u64;

        // 5. Smart diff against currently-open positions.
        //
        // Match each desired position against the existing set: same
        // side + same fee_bps, plus price within `match_price_tol_bps`
        // of mid and size within `match_size_tol_frac`. Matched
        // positions are kept (no churn); unmatched-open positions are
        // closed; unmatched-desired positions are opened.
        //
        // Saves gas + slippage: with a trained brain whose output is
        // stable across consecutive blocks, only inventory drift (size
        // changes from rebalancing) drives any churn.
        let mid = bf.quote.mid.max(1e-12);
        let mut kept: Vec<(PositionId, PositionParams)> = Vec::new();
        let mut still_unmatched: Vec<bool> = vec![true; self.open_positions.len()];
        let mut desired_satisfied: Vec<bool> = vec![false; positions.len()];

        for (di, d) in positions.iter().enumerate() {
            for (oi, (oid, o)) in self.open_positions.iter().enumerate() {
                if !still_unmatched[oi] { continue; }
                if o.side != d.side || o.fee_bps != d.fee_bps { continue; }
                let price_drift_bps = ((o.price - d.price).abs() / mid) * 10_000.0;
                if price_drift_bps > self.match_price_tol_bps { continue; }
                let size_ref = d.amount.max(o.amount).max(1e-12);
                let size_drift = (o.amount - d.amount).abs() / size_ref;
                if size_drift > self.match_size_tol_frac { continue; }
                still_unmatched[oi] = false;
                desired_satisfied[di] = true;
                kept.push((oid.clone(), o.clone()));
                break;
            }
        }

        // Close unmatched-open.
        for (oi, unmatched) in still_unmatched.iter().enumerate() {
            if *unmatched {
                if let Err(e) = self.chain.close_position(&self.open_positions[oi].0) {
                    eprintln!("close_position {:?}: {e}", self.open_positions[oi].0);
                }
            }
        }

        // Open unmatched-desired.
        for (di, satisfied) in desired_satisfied.iter().enumerate() {
            if !*satisfied {
                let p = positions[di].clone();
                let id = self.chain.open_position(&p)
                    .with_context(|| format!("open_position {:?}", p))?;
                kept.push((id, p));
            }
        }

        self.open_positions = kept;

        // 6. Update encoder history with this block's snapshot.
        self.encoder.push(BlockSnapshot {
            mid: bf.quote.mid,
            spread_bps: bf.quote.spread_bps(),
            own_fills_in_block: 0.0,    // wired once pending_fills is real
        });

        // 7. Snapshot for reward computation on the NEXT step.
        self.last_snapshot = Some(AgentSnapshot {
            block_height: bf.block_height,
            mid: bf.quote.mid,
            balance: bf.balance,
            cumulative_fees: 0.0,
            tx_count: positions.len() as u64,
            fills: 0,
        });

        Ok(positions)
    }

    /// REINFORCE-style update — stub pending the replay-buffer slice.
    pub fn train_step(&mut self) -> Result<f32> {
        let _grads = RegionalGradientsTyped::<D>::zeros(&self.weights)
            .map_err(|e| anyhow::anyhow!("RegionalGradientsTyped::zeros: {e}"))?;
        Ok(0.0)
    }
}
