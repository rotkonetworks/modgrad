//! Brain NAS correlation — Survey §9 follow-on to the smoke pipeline.
//!
//! The previous `brain_nas_train_smoke` proved the search space
//! *contains* good architectures (top-1 beat baseline +17pp). What
//! it didn't answer: is the forward-entropy proxy actually
//! *predictive* of training loss, or did we get lucky on rank-#1?
//!
//! This example measures that directly:
//!   1. Sample N valid architectures (uniform bias — small enough
//!      for CPU training in budget).
//!   2. Score each by the proxy (`entropy * tanh(20·entropy_std)`).
//!   3. Train each for K steps on the synthetic task.
//!   4. Report Spearman rank correlation ρ between proxy score
//!      and final training loss.
//!
//! Interpretation:
//!   |ρ| > 0.5  — proxy meaningfully predicts trainability
//!   |ρ| < 0.2  — proxy is mostly noise; need a better one
//!   ρ < 0      — proxy is *anti-predictive*; using it harms NAS
//!
//! The Survey expects |ρ| in 0.3-0.6 range for forward-only proxies
//! on a per-task basis, varying widely across tasks. Anything in
//! that band would be a real signal.
//!
//! Default budget: N=20 samples, K=15 train steps. CPU-only, ~3 min
//! wall-clock. Bigger via `MODGRAD_NAS_CORR_N` and
//! `MODGRAD_NAS_CORR_STEPS`.
//!
//! ```text
//! cargo run --release -p brain_nas_correlation
//! MODGRAD_NAS_CORR_N=40 cargo run --release -p brain_nas_correlation
//! ```

use std::time::Instant;

use modgrad_compute::neuron::SimpleRng;
use modgrad_ctm::graph::{
    regional_train_step, RegionalAdamW, RegionalGradients, RegionalWeights,
};
use modgrad_ctm::proxies::forward_entropy_stats;
use modgrad_ctm::search_space::{BrainArch, SamplerBias};

const OBS_DIM: usize = 32;
const OUT_DIMS: usize = 8;
const N_DATA: usize = 32;

fn main() {
    let n_samples: usize = std::env::var("MODGRAD_NAS_CORR_N")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let n_steps: usize = std::env::var("MODGRAD_NAS_CORR_STEPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    let lr: f32 = std::env::var("MODGRAD_NAS_CORR_LR")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    let bias_name = std::env::var("MODGRAD_NAS_CORR_BIAS")
        .unwrap_or_else(|_| "uniform".to_string());

    let bias = match bias_name.as_str() {
        "uniform" => SamplerBias::uniform(),
        "resident" | "resident_friendly" => SamplerBias::resident_friendly(),
        other => {
            eprintln!("unknown bias '{}'. expected uniform | resident.", other);
            std::process::exit(2);
        }
    };

    println!("brain_nas_correlation");
    println!("  N samples = {}, train steps = {}, lr = {:.0e}, bias = {}",
        n_samples, n_steps, lr, bias_name);
    println!("  task: synthetic linear classification (obs_dim={} out_dims={})",
        OBS_DIM, OUT_DIMS);
    println!();

    let mut rng = SimpleRng::new(0x4E_4153_C0_AAA1u64);
    let dataset = synthetic_dataset(0xCAFE_F00D, N_DATA);

    let t_start = Instant::now();
    let mut records: Vec<Record> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Sample
        let (arch, _attempts) = match BrainArch::sample_valid_biased(&mut rng, &bias, 200) {
            Some(t) => t,
            None => {
                eprintln!("  sample {}: failed valid in 200 attempts", i + 1);
                continue;
            }
        };
        let cfg = arch.to_regional_config(OBS_DIM, OUT_DIMS);
        let weights = RegionalWeights::new(cfg);
        let n_params = weights.n_params();

        // Proxy: forward-entropy mean+std over 4 random inputs.
        let (entropy, entropy_std) = forward_entropy_stats(
            &weights, OBS_DIM, 4,
            0xBEEFu64.wrapping_add(i as u64),
        );
        let max_entropy = (OUT_DIMS as f32).ln();
        let entropy_norm = (entropy / max_entropy).clamp(0.0, 1.0);

        // proxy_hard: original `entropy_norm × tanh(20·entropy_std)`.
        // Saturates below entropy_std ≈ 0.005 → many arches tie at 0.
        let sensitivity_hard = (20.0 * entropy_std).tanh();
        let proxy_hard = entropy_norm * sensitivity_hard;

        // proxy_smooth: additive mix. Sigmoid(10·entropy_std) is
        // monotone but never 0; plus baseline weight on entropy_norm.
        // No two arches collapse to identical scores unless their
        // (entropy, entropy_std) pairs are identical.
        let sensitivity_smooth = sigmoid(10.0 * entropy_std);
        let proxy_smooth = 0.6 * entropy_norm + 0.4 * sensitivity_smooth;

        // proxy_score is what the user sees; keep `hard` as default
        // for back-compat. Both stored for correlation comparison.
        let proxy_score = proxy_hard;

        // Decomposed components — to learn which axis carries the
        // signal. If `proxy_entropy_only` correlates ≈ as well as
        // the full formula, the sensitivity multiplier is redundant.
        // If `proxy_params_only` wins, capacity dominates and the
        // entropy proxy is decorative.
        let proxy_entropy_only = entropy_norm;
        let proxy_sensitivity_only = sensitivity_hard;
        let proxy_params_only = (n_params as f32).ln()
            / (modgrad_ctm::search_space::MAX_PARAMS as f32).ln();
        // Combined v2: sensitivity is the strong signal (+0.614),
        // params is a weaker independent signal (+0.325). Weighted
        // mix should beat either if their predictive content is
        // partially independent.
        let proxy_sens_plus_params = 0.7 * proxy_sensitivity_only
            + 0.3 * proxy_params_only;

        // Train and record initial / final loss.
        let mut weights = weights;
        let initial_loss = eval_loss(&weights, &dataset);
        let mut opt = RegionalAdamW::new(&weights).with_lr(lr).with_clip(5.0);
        let t_train = Instant::now();
        for step in 0..n_steps {
            let (obs, label) = &dataset[step % dataset.len()];
            let mut grads = RegionalGradients::zeros(&weights);
            regional_train_step(&weights, &mut grads, obs, *label);
            opt.step(&mut weights, &mut grads);
        }
        let final_loss = eval_loss(&weights, &dataset);
        let train_secs = t_train.elapsed().as_secs_f64();

        let drop_pct = if initial_loss.abs() > 1e-8 {
            (initial_loss - final_loss) / initial_loss * 100.0
        } else { 0.0 };

        eprintln!(
            "  [{:>3}/{}] params={:>6} entH={:.2} entσ={:.3} proxy={:.3} \
             init={:.3} final={:.3} Δ={:+.1}% t={:.1}s",
            i + 1, n_samples, fmt_n(n_params),
            entropy, entropy_std, proxy_score,
            initial_loss, final_loss, drop_pct, train_secs,
        );

        records.push(Record {
            n_params,
            entropy,
            entropy_std,
            proxy_score,
            proxy_smooth,
            proxy_entropy_only,
            proxy_sensitivity_only,
            proxy_params_only,
            proxy_sens_plus_params,
            initial_loss,
            final_loss,
            drop_pct,
        });
    }

    let total_secs = t_start.elapsed().as_secs_f64();
    println!();
    println!("Trained {} architectures in {:.1}s", records.len(), total_secs);
    println!();

    if records.len() < 5 {
        eprintln!("Need ≥5 samples for a meaningful correlation; got {}", records.len());
        return;
    }

    // ─── Correlations: hard vs smooth proxies ──────────────────
    //
    // We rank archs and compute Spearman ρ between proxy and
    // (a) -final_loss (lower loss = better trained)
    // (b) loss_drop_pct (relative improvement from each arch's start)
    let proxy_h: Vec<f32> = records.iter().map(|r| r.proxy_score).collect();
    let proxy_s: Vec<f32> = records.iter().map(|r| r.proxy_smooth).collect();
    let neg_loss: Vec<f32> = records.iter().map(|r| -r.final_loss).collect();
    let drop_v: Vec<f32> = records.iter().map(|r| r.drop_pct).collect();

    let proxy_e: Vec<f32> = records.iter().map(|r| r.proxy_entropy_only).collect();
    let proxy_sens: Vec<f32> = records.iter().map(|r| r.proxy_sensitivity_only).collect();
    let proxy_p: Vec<f32> = records.iter().map(|r| r.proxy_params_only).collect();

    println!("─── Correlations: which proxy component carries signal? ──");
    println!("                       {:>7}  {:>7}  {:>7}",
        "ρ_negL", "ρ_drop", "r_drop");
    let row = |label: &str, p: &[f32]| {
        println!("  {:>20}  {:+7.3}  {:+7.3}  {:+7.3}",
            label,
            spearman(p, &neg_loss),
            spearman(p, &drop_v),
            pearson(p, &drop_v));
    };
    let proxy_sens_p: Vec<f32> = records.iter().map(|r| r.proxy_sens_plus_params).collect();

    row("hard (entH × σ)", &proxy_h);
    row("smooth (additive)", &proxy_s);
    row("entropy_only", &proxy_e);
    row("sensitivity_only", &proxy_sens);
    row("params_only", &proxy_p);
    row("0.7·sens + 0.3·params", &proxy_sens_p);
    println!();
    let rho_h = spearman(&proxy_h, &drop_v);
    let rho_s = spearman(&proxy_s, &drop_v);
    let winner = if rho_s.abs() > rho_h.abs() { "smooth" } else { "hard" };
    let delta = (rho_s - rho_h).abs();
    println!("  Winner: {} (|Δρ|={:.2}). hard proxy = entropy_norm × tanh(20·entropy_std);",
        winner, delta);
    println!("          smooth proxy = 0.6·entropy_norm + 0.4·sigmoid(10·entropy_std).");
    println!();

    let rho_proxy_vs_drop = if winner == "smooth" { rho_s } else { rho_h };
    let strength = match rho_proxy_vs_drop.abs() {
        x if x > 0.5 => "strong",
        x if x > 0.3 => "moderate",
        x if x > 0.15 => "weak",
        _ => "negligible",
    };
    let direction = if rho_proxy_vs_drop > 0.0 { "predictive" } else { "anti-predictive" };
    println!("  → {} {} signal (|ρ|={:.2}, winner)",
        strength, direction, rho_proxy_vs_drop.abs());
    println!();

    // ─── Distribution of training outcomes ─────────────────────
    let drops: Vec<f32> = records.iter().map(|r| r.drop_pct).collect();
    let mut sorted_drops = drops.clone();
    sorted_drops.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted_drops[sorted_drops.len() / 2];
    let p25 = sorted_drops[sorted_drops.len() / 4];
    let p75 = sorted_drops[3 * sorted_drops.len() / 4];
    println!("─── Training outcome distribution (loss drop %) ──");
    println!("  min={:+.1}  p25={:+.1}  median={:+.1}  p75={:+.1}  max={:+.1}",
        sorted_drops[0], p25, med, p75, sorted_drops[sorted_drops.len() - 1]);
    let n_diverged = drops.iter().filter(|&&d| d < 0.0).count();
    println!("  diverged (Δ<0): {} / {}", n_diverged, records.len());
    println!();

    // ─── Top-3 vs bottom-3 by proxy ────────────────────────────
    let mut sorted = records.clone();
    sorted.sort_by(|a, b| b.proxy_score.partial_cmp(&a.proxy_score).unwrap());
    println!("─── Top-3 by proxy ──");
    for (i, r) in sorted.iter().take(3).enumerate() {
        println!("  {} proxy={:.3} → loss_drop={:+.1}%", i + 1, r.proxy_score, r.drop_pct);
    }
    println!("─── Bottom-3 by proxy ──");
    for (i, r) in sorted.iter().rev().take(3).enumerate() {
        println!("  {} proxy={:.3} → loss_drop={:+.1}%", i + 1, r.proxy_score, r.drop_pct);
    }
}

#[derive(Clone)]
struct Record {
    n_params: usize,
    entropy: f32,
    entropy_std: f32,
    proxy_score: f32,
    proxy_smooth: f32,
    proxy_entropy_only: f32,
    proxy_sensitivity_only: f32,
    proxy_params_only: f32,
    proxy_sens_plus_params: f32,
    initial_loss: f32,
    final_loss: f32,
    drop_pct: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn eval_loss(weights: &RegionalWeights, data: &[(Vec<f32>, usize)]) -> f32 {
    let n = data.len().min(8);
    if n == 0 { return 0.0; }
    let mut sum = 0.0f32;
    for (obs, label) in &data[..n] {
        let mut grads = RegionalGradients::zeros(weights);
        let (loss, _) = regional_train_step(weights, &mut grads, obs, *label);
        sum += loss;
    }
    sum / n as f32
}

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

/// Spearman rank correlation. Converts both vectors to ranks
/// (with average-rank tie-breaking), then computes Pearson on ranks.
/// Returns 0.0 for fewer than 2 points or zero variance.
fn spearman(xs: &[f32], ys: &[f32]) -> f32 {
    if xs.len() != ys.len() || xs.len() < 2 { return 0.0; }
    pearson(&ranks(xs), &ranks(ys))
}

fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f32;
    if xs.len() < 2 || xs.len() != ys.len() { return 0.0; }
    let mx = xs.iter().sum::<f32>() / n;
    let my = ys.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut dx = 0.0f32;
    let mut dy = 0.0f32;
    for (x, y) in xs.iter().zip(ys) {
        let ex = x - mx;
        let ey = y - my;
        num += ex * ey;
        dx += ex * ex;
        dy += ey * ey;
    }
    let denom = (dx * dy).sqrt();
    if denom < 1e-12 { 0.0 } else { num / denom }
}

/// Convert values to ranks (1-indexed). Ties get the average rank.
fn ranks(xs: &[f32]) -> Vec<f32> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && xs[idx[j + 1]] == xs[idx[i]] { j += 1; }
        let avg_rank = ((i + 1 + j + 1) as f32) / 2.0;
        for k in i..=j { r[idx[k]] = avg_rank; }
        i = j + 1;
    }
    r
}

fn fmt_n(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f32 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.1}K", n as f32 / 1_000.0) }
    else { n.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spearman_perfect_positive() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert!((spearman(&xs, &ys) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn spearman_perfect_negative() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![50.0, 40.0, 30.0, 20.0, 10.0];
        assert!((spearman(&xs, &ys) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn spearman_independent_near_zero() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![3.0, 1.0, 4.0, 2.0];
        // No correlation in this small permutation; expect ~0.
        let rho = spearman(&xs, &ys);
        assert!(rho.abs() < 0.5, "expected weak corr, got {}", rho);
    }

    #[test]
    fn ranks_handle_ties() {
        let xs = vec![1.0, 2.0, 2.0, 3.0];
        let r = ranks(&xs);
        assert_eq!(r[0], 1.0);
        // Two tied values at positions 1,2 should both get rank 2.5 = (2+3)/2.
        assert!((r[1] - 2.5).abs() < 1e-5);
        assert!((r[2] - 2.5).abs() < 1e-5);
        assert_eq!(r[3], 4.0);
    }
}
