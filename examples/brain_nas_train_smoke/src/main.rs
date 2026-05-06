//! Brain NAS train smoke — closes the loop on the search space.
//!
//! Loads top-K architectures from `brain_nas_top10.json`, builds a
//! synthetic linear-classification task (random obs → label via a
//! fixed linear projection), and trains each arch for `N_STEPS`
//! optimiser steps via `regional_train_step` + `RegionalAdamW`.
//!
//! Reports initial loss, final loss, and percent drop. The
//! comparison answers the open question from the proxy work:
//! does our forward-entropy proxy correlate with actual training
//! loss decrease? If the top-3 by proxy beat the
//! `eight_region_small` baseline by visible margin on the same
//! task, the proxy is doing its job.
//!
//! CPU-only by design. Tuned to fit in ~60-90s total wall-clock so
//! it doesn't lag the host. Bigger archs and more steps via env
//! vars (see `MODGRAD_NAS_TRAIN_*` below).
//!
//! ```text
//! cargo run --release -p brain_nas_train_smoke
//! cargo run --release -p brain_nas_train_smoke -- brain_nas_top10.json 2
//! MODGRAD_NAS_TRAIN_STEPS=30 cargo run --release -p brain_nas_train_smoke
//! ```

use std::time::{Duration, Instant};

use modgrad_compute::neuron::SimpleRng;
use modgrad_ctm::graph::{
    regional_train_step, RegionalAdamW, RegionalGradients, RegionalWeights,
};
use modgrad_ctm::search_space::BrainArch;

const OBS_DIM: usize = 32;
const OUT_DIMS: usize = 8;
const N_DATA: usize = 32;

fn main() {
    let json_path = std::env::args().nth(1)
        .unwrap_or_else(|| "brain_nas_top10.json".to_string());
    let n_archs: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let n_steps: usize = std::env::var("MODGRAD_NAS_TRAIN_STEPS")
        .ok().and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let lr: f32 = std::env::var("MODGRAD_NAS_TRAIN_LR")
        .ok().and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);

    println!("brain_nas_train_smoke");
    println!("  task: synthetic linear classification");
    println!("  obs_dim={} out_dims={} n_data={} n_steps={} lr={:.0e}",
        OBS_DIM, OUT_DIMS, N_DATA, n_steps, lr);
    println!();

    let dataset = synthetic_dataset(0xCAFE_F00D, N_DATA);

    // ─── Baseline: hand-designed eight_region_small ─────────────
    let small = BrainArch::eight_region_small_arch();
    println!("=== Baseline: eight_region_small (hand-designed) ===");
    let r = train_arch(&small, &dataset, n_steps, lr);
    print_result(&r);
    let baseline_drop_pct = pct_drop(r.initial_loss, r.final_loss);
    println!();

    // ─── Top-K candidates from NAS ──────────────────────────────
    let json = match std::fs::read_to_string(&json_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read {}: {}", json_path, e);
            eprintln!("Run brain_nas_smoke first to generate the file.");
            std::process::exit(1);
        }
    };
    let records: Vec<serde_json::Value> = match serde_json::from_str(&json) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to parse {}: {}", json_path, e);
            std::process::exit(1);
        }
    };

    let mut nas_drops = Vec::new();
    for (i, rec) in records.iter().take(n_archs).enumerate() {
        let arch_value = match rec.get("arch") {
            Some(v) => v.clone(),
            None => continue,
        };
        let arch: BrainArch = match serde_json::from_value(arch_value) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("warning: failed to parse arch for rank #{}: {}", i + 1, e);
                continue;
            }
        };
        let entropy = rec.get("entropy_mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let entropy_std = rec.get("entropy_std").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let n_params = rec.get("n_params").and_then(|v| v.as_u64()).unwrap_or(0);

        println!("=== NAS rank #{}  params={} proxy: entH={:.2} entσ={:.3} ===",
            i + 1, fmt_n(n_params as usize), entropy, entropy_std);
        let r = train_arch(&arch, &dataset, n_steps, lr);
        print_result(&r);
        nas_drops.push(pct_drop(r.initial_loss, r.final_loss));
        println!();
    }

    // ─── Verdict ─────────────────────────────────────────────────
    if nas_drops.is_empty() {
        println!("No NAS architectures evaluated.");
        return;
    }
    let best_nas = nas_drops.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_nas = nas_drops.iter().sum::<f32>() / nas_drops.len() as f32;
    println!("─── Verdict ─────────────────────────────────────");
    println!("  baseline (eight_region_small) loss drop:  {:+.1}%", baseline_drop_pct);
    println!("  best NAS rank by training loss drop:       {:+.1}%", best_nas);
    println!("  mean NAS loss drop ({} archs):              {:+.1}%", nas_drops.len(), mean_nas);
    if best_nas > baseline_drop_pct + 2.0 {
        println!("  → NAS top-{} BEATS baseline by ≥2pp on this task.", n_archs);
    } else if best_nas + 2.0 < baseline_drop_pct {
        println!("  → baseline beats NAS top-{} on this task.", n_archs);
    } else {
        println!("  → roughly tied (within 2pp).");
    }
    println!();
    println!("Caveats: synthetic linear task; {} steps; CPU-only;", n_steps);
    println!("  proxy was forward-entropy at init, not training-loss prediction.");
    println!("  This validates the *pipeline*, not the proxy's predictive power yet.");
}

struct TrainResult {
    initial_loss: f32,
    final_loss: f32,
    n_params: usize,
    elapsed: Duration,
}

fn train_arch(
    arch: &BrainArch,
    data: &[(Vec<f32>, usize)],
    n_steps: usize,
    lr: f32,
) -> TrainResult {
    let cfg = arch.to_regional_config(OBS_DIM, OUT_DIMS);
    let mut weights = RegionalWeights::new(cfg);
    let n_params = weights.n_params();
    let mut opt = RegionalAdamW::new(&weights).with_lr(lr).with_clip(5.0);

    let t_start = Instant::now();

    let initial_loss = eval_loss(&weights, data);

    for step in 0..n_steps {
        let (obs, label) = &data[step % data.len()];
        let mut grads = RegionalGradients::zeros(&weights);
        regional_train_step(&weights, &mut grads, obs, *label);
        opt.step(&mut weights, &mut grads);
    }

    let final_loss = eval_loss(&weights, data);

    TrainResult {
        initial_loss,
        final_loss,
        n_params,
        elapsed: t_start.elapsed(),
    }
}

/// Mean cross-entropy loss across the first 8 dataset entries.
/// Uses `regional_train_step` for the forward+loss, with throwaway
/// gradients (doesn't apply optimiser).
fn eval_loss(weights: &RegionalWeights, data: &[(Vec<f32>, usize)]) -> f32 {
    let n = data.len().min(8);
    if n == 0 { return 0.0; }
    let mut sum = 0.0f32;
    for (obs, label) in &data[..n] {
        let mut grads = RegionalGradients::zeros(weights);
        let (loss, _pred) = regional_train_step(weights, &mut grads, obs, *label);
        sum += loss;
    }
    sum / n as f32
}

fn print_result(r: &TrainResult) {
    let drop = pct_drop(r.initial_loss, r.final_loss);
    println!("  params={}  initial_loss={:.4}  final_loss={:.4}  Δ={:+.1}%  t={:.1}s",
        fmt_n(r.n_params),
        r.initial_loss, r.final_loss, drop,
        r.elapsed.as_secs_f64());
}

fn pct_drop(initial: f32, final_: f32) -> f32 {
    if initial.abs() < 1e-8 { return 0.0; }
    (initial - final_) / initial * 100.0
}

/// Generate a synthetic classification dataset. Labels come from
/// `arg_max(W·obs)` for a random fixed projection W — so a linear
/// model could in principle solve it; the brain's job is to learn
/// the same mapping through its tick loop.
fn synthetic_dataset(seed: u64, n: usize) -> Vec<(Vec<f32>, usize)> {
    let mut rng = SimpleRng::new(seed);
    let proj: Vec<f32> = (0..OBS_DIM * OUT_DIMS).map(|_| rng.next_normal()).collect();
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let obs: Vec<f32> = (0..OBS_DIM).map(|_| rng.next_normal()).collect();
        let mut logits = vec![0.0f32; OUT_DIMS];
        for c in 0..OUT_DIMS {
            for d in 0..OBS_DIM {
                logits[c] += proj[c * OBS_DIM + d] * obs[d];
            }
        }
        let label = logits.iter().enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                if v > best.1 { (i, v) } else { best }
            }).0;
        data.push((obs, label));
    }
    data
}

fn fmt_n(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f32 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.1}K", n as f32 / 1_000.0) }
    else { n.to_string() }
}
