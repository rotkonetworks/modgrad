//! Imitation training loop — brain learns to reproduce the oracle MM
//! action on a synthetic Penumbra replay.
//!
//! Per training step:
//!   1. Generate one replay block → BlockFeatures
//!   2. Encode → forward through the typed brain
//!   3. Compute MSE between brain's last-tick MOTOR vector and the
//!      oracle's target MOTOR vector
//!   4. Per-tick d_predictions: zero everywhere except the last tick,
//!      where d = 2·(brain_pred − oracle) / motor_dim (MSE gradient)
//!   5. regional_backward_typed → RegionalAdamWTyped::step
//!
//! Verification: epoch loss should drop monotonically (small noise OK)
//! and brain MOTOR output should converge toward the oracle's logit
//! shape (skip_logits ~ +2, sizes/offsets in the right range).

use anyhow::Result;

use modgrad_ctm::graph::{
    RegionalAdamWTyped, RegionalGradientsTyped,
    RegionalWeights, RegionalWeightsTyped,
};
use modgrad_device::backend::tensor::Cpu;

use penumbra_arena::agent::RegionState;
use penumbra_arena::default_mm_cfg;
use penumbra_arena::features::FeatureEncoder;
use penumbra_arena::motor::{BarbellConfig, BarbellDecoder};
use penumbra_arena::replay::{SyntheticReplay, SyntheticReplayConfig, oracle_motor};

fn main() -> Result<()> {
    let n_blocks_per_epoch: usize = std::env::var("BLOCKS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let n_epochs: usize = std::env::var("EPOCHS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let lr: f32 = std::env::var("LR")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(3e-3);
    let bs: usize = std::env::var("BS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let skewed = std::env::var("SKEWED").is_ok();

    eprintln!("== penumbra imitation trainer ==");
    eprintln!("  blocks/epoch={n_blocks_per_epoch}  epochs={n_epochs}  bs={bs}  lr={lr}");
    eprintln!("  replay={}",
        if skewed { "SKEWED (depth-asymmetry → drift)" } else { "default (symmetric)" });
    let untyped = RegionalWeights::new(default_mm_cfg());
    eprintln!("  brain params = {}", untyped.n_params());
    let mut typed = RegionalWeightsTyped::<Cpu>::from_untyped(&untyped)
        .map_err(|e| anyhow::anyhow!("from_untyped: {e}"))?;
    let mut opt = RegionalAdamWTyped::<Cpu>::new(&typed)
        .map_err(|e| anyhow::anyhow!("RegionalAdamWTyped::new: {e}"))?
        .with_lr(lr).with_clip(1.0);

    let cfg = if skewed {
        SyntheticReplayConfig::um_usdc_skewed()
    } else {
        SyntheticReplayConfig::um_usdc_default()
    };
    let n_sync = typed.config.n_global_sync;

    for epoch in 0..n_epochs {
        let mut replay = SyntheticReplay::new(cfg.clone(), 0xFEED + epoch as u64);
        let mut encoder = FeatureEncoder::new();
        let mut total_mse = 0.0f64;
        let mut n_seen = 0usize;

        let mut batch_grads = RegionalGradientsTyped::<Cpu>::zeros(&typed)
            .map_err(|e| anyhow::anyhow!("RegionalGradientsTyped::zeros: {e}"))?;
        let t0 = std::time::Instant::now();

        for step_i in 0..n_blocks_per_epoch {
            // Per-sample state init from the brain's start states.
            let mut region_activated: Vec<Vec<f32>> = untyped.regions.iter()
                .map(|r| r.start_activated.clone()).collect();
            let mut region_trace: Vec<Vec<f32>> = untyped.regions.iter()
                .map(|r| r.start_trace.clone()).collect();
            let mut global_alpha = vec![0.0f32; n_sync];
            let mut global_beta = vec![1.0f32; n_sync];

            // 1. New replay block.
            let bf = replay.step();
            // 2. Encode.
            let obs = encoder.encode(&bf);
            // 3. Forward.
            let (out, cache) = typed.regional_forward_typed_with_cache(
                &obs,
                &mut region_activated, &mut region_trace,
                &mut global_alpha, &mut global_beta,
            ).map_err(|e| anyhow::anyhow!("regional_forward_typed_with_cache: {e}"))?;

            // 4. Oracle target + MSE loss.
            let target = oracle_motor(&bf);
            let last_tick = out.predictions.len() - 1;
            let pred = &out.predictions[last_tick];
            let mse: f64 = pred.iter().zip(&target)
                .map(|(p, t)| ((p - t) as f64).powi(2)).sum::<f64>() / pred.len() as f64;
            total_mse += mse;
            n_seen += 1;

            // 5. d_predictions: 2/N · (pred - target) at last tick, zero otherwise.
            let mut d_preds: Vec<Vec<f32>> = (0..out.predictions.len())
                .map(|_| vec![0.0f32; pred.len()]).collect();
            for i in 0..pred.len() {
                d_preds[last_tick][i] = 2.0 * (pred[i] - target[i]) / pred.len() as f32;
            }

            // 6. Backward → batch_grads.
            typed.regional_backward_typed(&cache, &d_preds, &mut batch_grads)
                .map_err(|e| anyhow::anyhow!("regional_backward_typed: {e}"))?;

            // 7. Step every `bs` blocks.
            if (step_i + 1) % bs == 0 {
                opt.step(&mut typed, &mut batch_grads)
                    .map_err(|e| anyhow::anyhow!("opt.step: {e}"))?;
                batch_grads.zero()
                    .map_err(|e| anyhow::anyhow!("batch_grads.zero: {e}"))?;
            }

            // Push a snapshot for the encoder's history.
            encoder.push(penumbra_arena::features::BlockSnapshot {
                mid: bf.quote.mid,
                spread_bps: bf.quote.spread_bps(),
                own_fills_in_block: 0.0,
            });
        }

        let avg_mse = total_mse / n_seen.max(1) as f64;
        eprintln!("  epoch {:>2}/{}  mse={:.4}  {:.2}s",
            epoch + 1, n_epochs, avg_mse, t0.elapsed().as_secs_f32());
    }

    // Post-train sanity: a converged brain imitating the conventional
    // MM oracle should flip < 10% of rungs across mid. Higher = not
    // yet converged. State is persistent across the 64 blocks so this
    // mirrors what `MmAgent` does in production.
    {
        let mut eval = SyntheticReplay::new(cfg.clone(), 0xEEFF);
        let mut enc  = FeatureEncoder::new();
        let dec = BarbellDecoder::new(BarbellConfig::default());
        let mut state = RegionState::from_weights(&untyped);
        let mut total_rungs = 0u64;
        let mut total_flips = 0u64;
        for _ in 0..64 {
            let bf = eval.step();
            let obs = enc.encode(&bf);
            let out = typed.regional_forward_typed(
                &obs, &mut state.region_activated, &mut state.region_trace,
                &mut state.global_alpha,           &mut state.global_beta,
            ).map_err(|e| anyhow::anyhow!("eval forward: {e}"))?;
            let last = out.predictions.len() - 1;
            let positions = dec.decode(&out.predictions[last], bf.quote.mid, bf.balance.base);
            total_rungs += positions.len() as u64;
            for p in &positions {
                if p.is_arb_flip(bf.quote.mid) { total_flips += 1; }
            }
            enc.push(penumbra_arena::features::BlockSnapshot {
                mid: bf.quote.mid, spread_bps: bf.quote.spread_bps(), own_fills_in_block: 0.0,
            });
        }
        let flip_rate = if total_rungs > 0 {
            (total_flips as f64) / (total_rungs as f64)
        } else { 0.0 };
        eprintln!("\n[post-train eval] 64 blocks  rungs={}  flips={}  flip_rate={:.1}%",
            total_rungs, total_flips, flip_rate * 100.0);
        if flip_rate > 0.10 {
            eprintln!("  ⚠ flip rate >10% — brain hasn't fully converged on oracle policy");
        }
    }

    // Save the trained brain so live_arena can pick it up.
    if let Some(out_path) = std::env::var("CHECKPOINT").ok() {
        eprintln!("\n[checkpoint] saving trained brain to {out_path}");
        penumbra_arena::checkpoint::save(&typed, &out_path)?;
    } else {
        let default = "/tmp/penumbra_arena_brain.bin";
        eprintln!("\n[checkpoint] saving trained brain to {default}");
        eprintln!("  (set CHECKPOINT=<path> to override)");
        penumbra_arena::checkpoint::save(&typed, default)?;
    }

    eprintln!("Done.");
    Ok(())
}
