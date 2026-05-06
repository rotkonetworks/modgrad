//! Brain NAS smoke — random-search the BrainArch space using free
//! structural proxies (param_count, total_d_model, edge density,
//! resident-likely flag). No training; this is the encoding-pipeline
//! sanity check that proves we can sample, validate, decode to a
//! `RegionalConfig`, and instantiate weights without exploding.
//!
//! Run with: `cargo run --release -p brain_nas_smoke -- [n_samples]`
//! Default n=64. Reports valid-fraction, param distribution and
//! the top-10 architectures by a simple composite score.

use modgrad_compute::neuron::SimpleRng;
use modgrad_ctm::graph::{RegionalBrain, RegionalWeights};
use modgrad_ctm::proxies::{forward_entropy_stats, forward_sensitivity_score};
use modgrad_ctm::search_space::{
    BrainArch, SamplerBias, CORTICAL, REGION_NAMES, MAX_PARAMS, N_REGIONS,
};

const OBS_DIM: usize = 64;
const OUT_DIMS: usize = 16;

fn main() {
    let n_samples: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let mode = std::env::args().nth(2).unwrap_or_else(|| "uniform".to_string());

    let bias = match mode.as_str() {
        "uniform" => SamplerBias::uniform(),
        "resident" | "resident_friendly" => SamplerBias::resident_friendly(),
        other => {
            eprintln!("Unknown sampler mode '{}'. Use 'uniform' or 'resident'.", other);
            std::process::exit(2);
        }
    };

    let max_attempts: usize = n_samples * 20;

    println!("brain_nas_smoke — random sampling brain architectures");
    println!("obs_dim={} out_dims={} n_samples={} bias={}",
        OBS_DIM, OUT_DIMS, n_samples, mode);
    println!("  cortex_large_prob={:.2} recv_obs_prob={:.2} edge_prob={:.2}",
        bias.cortex_large_prob, bias.recv_obs_prob, bias.edge_prob);
    println!();

    let mut rng = SimpleRng::new(0x4E_4153_5EED); // "NAS\0SEED"

    let mut valid_archs: Vec<BrainArch> = Vec::with_capacity(n_samples);
    let mut total_attempts = 0usize;

    while valid_archs.len() < n_samples && total_attempts < max_attempts {
        if let Some((arch, attempts)) =
            BrainArch::sample_valid_biased(&mut rng, &bias, 64)
        {
            valid_archs.push(arch);
            total_attempts += attempts;
        } else {
            total_attempts += 64;
        }
    }

    let collected = valid_archs.len();
    let valid_fraction = collected as f32 / total_attempts.max(1) as f32;
    println!(
        "Sampler: {} valid architectures over {} attempts ({:.1}% valid)",
        collected, total_attempts, valid_fraction * 100.0,
    );
    println!();

    // ─── Score each valid arch (free proxies only) ─────────────
    let mut scored: Vec<Scored> = Vec::with_capacity(collected);
    let mut skipped_oversize = 0usize;
    let max_entropy = (OUT_DIMS as f32).ln();
    for (idx, arch) in valid_archs.iter().enumerate() {
        let cfg = arch.to_regional_config(OBS_DIM, OUT_DIMS);
        let weights = RegionalWeights::new(cfg);
        let n_params = weights.n_params();
        if n_params > MAX_PARAMS {
            skipped_oversize += 1;
            continue;
        }
        let total_d: usize = arch.regions.iter().map(|r| r.d_model as usize).sum();
        let edge_count: u32 = arch.mask.sources.iter().map(|m| m.count_ones()).sum();
        let recv_obs_count = arch.mask.recv_obs.iter().filter(|&&b| b).count();
        // Use the SDK's own crossover heuristic so this rank ties
        // directly to the residency decision in production code.
        let resident_likely = RegionalBrain::is_resident_likely_faster(&weights);
        // Forward-entropy stats — mean for telemetry/display, std
        // converted to input-sensitivity score that drives ranking.
        // Per the 2026-05-05 ablation (memory: feedback_nas_proxy_correlation):
        // sensitivity_only is the winning proxy (ρ=+0.614 vs +0.581 combined).
        let (entropy, entropy_std) = forward_entropy_stats(
            &weights, OBS_DIM, 4,
            0xBEEFu64.wrapping_add(idx as u64),
        );
        let sensitivity = forward_sensitivity_score(
            &weights, OBS_DIM, 4,
            0xBEEFu64.wrapping_add(idx as u64),
        );
        scored.push(Scored {
            arch: arch.clone(),
            n_params,
            total_d_model: total_d,
            edge_count: edge_count as usize,
            recv_obs_count,
            resident_likely,
            entropy,
            entropy_std,
            entropy_norm: (entropy / max_entropy).clamp(0.0, 1.0),
            sensitivity,
            score: 0.0,
        });
    }
    println!("Param-budget filter: kept {} / {} (rejected {} > {} params)",
        scored.len(), collected, skipped_oversize, MAX_PARAMS);
    println!();

    if scored.is_empty() {
        eprintln!("All sampled architectures exceed the param budget — try a larger sample size or relax MAX_PARAMS");
        std::process::exit(1);
    }

    // ─── Distributions ──────────────────────────────────────────
    let mut params_sorted: Vec<usize> = scored.iter().map(|s| s.n_params).collect();
    params_sorted.sort_unstable();
    let p_min = params_sorted[0];
    let p_med = params_sorted[params_sorted.len() / 2];
    let p_max = *params_sorted.last().unwrap();
    println!("Param-count distribution (post-filter):");
    println!("  min={} median={} max={}", fmt_n(p_min), fmt_n(p_med), fmt_n(p_max));
    let resident_count = scored.iter().filter(|s| s.resident_likely).count();
    println!("  resident-likely (any cortex d ≥ 1024): {} / {}", resident_count, scored.len());
    println!();

    // ─── Entropy distribution ──────────────────────────────────
    let mut entropies: Vec<f32> = scored.iter().map(|s| s.entropy_norm).collect();
    entropies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let e_min = entropies.first().copied().unwrap_or(0.0);
    let e_med = entropies.get(entropies.len() / 2).copied().unwrap_or(0.0);
    let e_max = entropies.last().copied().unwrap_or(0.0);
    println!("Forward-entropy distribution (normalised, 1.0 = uniform output):");
    println!("  min={:.3} median={:.3} max={:.3}", e_min, e_med, e_max);
    println!();

    // ─── Composite score: sensitivity-led ──────────────────────
    //
    // Per the 2026-05-05 ablation (N=20 brain_nas_correlation):
    //   sensitivity_only:           ρ=+0.614 ← winner
    //   entropy × sensitivity:      ρ=+0.581
    //   entropy_only:               ρ=−0.177 (anti-predictive!)
    //   params_only:                ρ=+0.325
    //   0.7·sens + 0.3·params:      ρ=+0.571 (overlap, dilutes)
    //
    // Sensitivity is the dominant signal. Mean entropy is dropped
    // entirely from the composite. Params/edges/resident remain as
    // secondary structural tiebreakers but with reduced weights.
    //
    //   0.65 sensitivity      — main trainability proxy
    //   0.15 param capacity   — gentle bias toward larger nets
    //   0.05 edge density     — peaks at ~40%
    //   0.15 resident_bonus   — hardware fitness on this gpu
    let mut scored = scored;
    for s in &mut scored {
        let p_norm = s.n_params as f32 / MAX_PARAMS as f32;
        let edges_norm = (s.edge_count as f32 / 28.0).min(1.0);
        let edge_score = 1.0 - (edges_norm - 0.4).abs() * 2.0;
        let resident_bonus = if s.resident_likely { 1.0 } else { 0.5 };
        s.score = s.sensitivity   * 0.65
                + p_norm          * 0.15
                + edge_score      * 0.05
                + resident_bonus  * 0.15;
    }
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let top_n = 10.min(scored.len());
    println!("Top-{} by composite score (sensitivity-led):", top_n);
    println!("{:>3} {:>10} {:>6} {:>5} {:>5} {:>4} {:>5} {:>5} cortical d_model:",
        "#", "params", "total_d", "edges", "recv", "tick", "resnt", "sens");
    for (i, s) in scored.iter().take(top_n).enumerate() {
        let cortex_dms: Vec<String> = CORTICAL.iter()
            .map(|&j| s.arch.regions[j].d_model.to_string())
            .collect();
        println!(
            "{:>3} {:>10} {:>6} {:>5} {:>5} {:>4} {:>5} {:>5.3} [{}] beta_outer={:.2}",
            i + 1,
            fmt_n(s.n_params),
            s.total_d_model,
            s.edge_count,
            s.recv_obs_count,
            s.arch.globals.outer_ticks,
            if s.resident_likely { "yes" } else { "no" },
            s.sensitivity,
            cortex_dms.join(","),
            s.arch.globals.outer_beta,
        );
    }
    println!();

    // ─── Persist top-K to JSON for downstream training ─────────
    let out_path = std::env::var("MODGRAD_NAS_OUT")
        .unwrap_or_else(|_| "brain_nas_top10.json".to_string());
    let top_records: Vec<TopRecord> = scored.iter().take(top_n).map(|s| TopRecord {
        score: s.score,
        n_params: s.n_params,
        entropy_mean: s.entropy,
        entropy_std: s.entropy_std,
        sensitivity: s.sensitivity,
        resident_likely: s.resident_likely,
        edge_count: s.edge_count,
        outer_ticks: s.arch.globals.outer_ticks as usize,
        arch: s.arch.clone(),
    }).collect();
    let json = serde_json::to_string_pretty(&top_records)
        .expect("top records serialise");
    if let Err(e) = std::fs::write(&out_path, json) {
        eprintln!("warning: failed to write {}: {}", out_path, e);
    } else {
        println!("Top-{} architectures written to {}", top_n, out_path);
        println!("  Reload via: BrainArch::load_json(\"<path>\")");
        println!();
    }

    // ─── Round-trip the billion preset for sanity ──────────────
    let billion = BrainArch::eight_region_billion_arch();
    let billion_cfg = billion.to_regional_config(OBS_DIM, OUT_DIMS);
    let billion_w = RegionalWeights::new(billion_cfg);
    println!("Reference: eight_region_billion encoded → {} params",
        fmt_n(billion_w.n_params()));
    for &i in &CORTICAL {
        assert_eq!(billion.regions[i].d_model, 1024);
    }
    for i in 0..N_REGIONS {
        println!("  {:14} d_model={:5} mem={:3} deep={} beta={:.2} ticks={}",
            REGION_NAMES[i],
            billion.regions[i].d_model,
            billion.regions[i].memory_length,
            billion.regions[i].deep_nlms,
            billion.regions[i].beta,
            billion.regions[i].iterations);
    }
}

#[derive(serde::Serialize)]
struct TopRecord {
    score: f32,
    n_params: usize,
    entropy_mean: f32,
    entropy_std: f32,
    sensitivity: f32,
    resident_likely: bool,
    edge_count: usize,
    outer_ticks: usize,
    arch: BrainArch,
}

#[derive(Clone)]
struct Scored {
    arch: BrainArch,
    n_params: usize,
    total_d_model: usize,
    edge_count: usize,
    recv_obs_count: usize,
    resident_likely: bool,
    /// Mean forward-entropy in nats (display only).
    entropy: f32,
    /// Std of forward-entropy across random inputs (display only).
    entropy_std: f32,
    /// Entropy normalised to [0, 1] (display only).
    entropy_norm: f32,
    /// `tanh(20·entropy_std)` — the *recommended* trainability proxy
    /// per the 2026-05-05 ablation (ρ=+0.614 vs synthetic linear).
    sensitivity: f32,
    score: f32,
}

fn fmt_n(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f32 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.1}K", n as f32 / 1_000.0) }
    else { n.to_string() }
}
