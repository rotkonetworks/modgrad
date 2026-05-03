//! Controlled win/loss test: does attribution-selected fine-tuning
//! actually beat random selection on held-out loss?
//!
//! # Why a controlled task and not Qwen
//!
//! The paper's claim is: "fine-tuning on the top-K%% by influence
//! beats fine-tuning on a random K%% on the same held-out metric." On
//! Qwen-0.5B with a real corpus this needs hours of GPU time per
//! data point. To get a binary signal NOW, we run the same comparison
//! on a controlled synthetic task with a small brain. If attribution
//! beats random here, we have evidence the technique works in our SDK;
//! Qwen scaling is a separate downstream test.
//!
//! # Task setup
//!
//! - Brain: `eight_region_small` (token_dim=16, out_dims=8, 2 ticks).
//! - Two clusters of training data:
//!     - Cluster A ("concept"): inputs share a structured pattern
//!       (sin wave at phase 0), targets are a fixed direction.
//!     - Cluster B ("noise"): inputs are random, targets are random.
//! - Queries: 3 cluster-A samples (held-out from training).
//! - Held-out test: 5 cluster-A samples (held-out from queries AND
//!   training).
//!
//! # Procedure
//!
//! 1. For every training sample, run forward + MSE backward, flatten
//!    `RegionalGradients`, project to `R^256` and ℓ²-normalise.
//! 2. Same for the queries; aggregate query gradients via mean.
//! 3. Rank training samples by cosine vs aggregated query.
//! 4. **Run A**: clone the initial weights, train 5 epochs on
//!    `top_k = 10` selected samples, measure held-out MSE.
//! 5. **Run B**: clone the same initial weights, train 5 epochs on a
//!    random 10 samples, measure held-out MSE.
//! 6. **Win condition**: `held_out_mse(top_k) < held_out_mse(random)`.
//!
//! Two run pairs (different random seeds) reduce variance — the win
//! must hold in both, not just on average. If both pairs show
//! attribution beating random, the test passes; otherwise it fails
//! and we report that honestly.

use modgrad_attribution::{Attributor, Projection};
use modgrad_compute::neuron::SimpleRng;
use modgrad_ctm::graph::{
    RegionalBrain, RegionalConfig, RegionalGradients, RegionalWeights,
};
use modgrad_traits::{Brain, TokenInput};

const TOKEN_DIM: usize = 16;
const N_TOKENS: usize = 4;
const OUT_DIMS: usize = 8;
const TICKS: usize = 2;
const N_CONCEPT: usize = 15;
const N_NOISE: usize = 15;
const N_QUERIES: usize = 3;
const N_HELD_OUT: usize = 5;
const K: usize = 10;
const EPOCHS: usize = 5;
const LR: f32 = 0.01;
const PROJ_DIM: usize = 256;

/// Generate a "concept-A" sample: structured input + fixed-direction
/// target. The phase parameter adds within-cluster variation so the
/// queries and training samples aren't identical.
fn concept_sample(phase: f32) -> (TokenInput, Vec<f32>) {
    let tokens: Vec<f32> = (0..N_TOKENS * TOKEN_DIM)
        .map(|i| ((i as f32 * 0.3) + phase).sin() * 0.5)
        .collect();
    // Fixed target direction for cluster A.
    let target: Vec<f32> = (0..OUT_DIMS).map(|i| (i as f32 * 0.4).cos()).collect();
    (
        TokenInput { tokens, n_tokens: N_TOKENS, token_dim: TOKEN_DIM },
        target,
    )
}

/// Noise sample: random input, random target.
fn noise_sample(rng: &mut SimpleRng) -> (TokenInput, Vec<f32>) {
    let tokens: Vec<f32> = (0..N_TOKENS * TOKEN_DIM)
        .map(|_| rng.next_normal() * 0.5).collect();
    let target: Vec<f32> = (0..OUT_DIMS).map(|_| rng.next_normal() * 0.5).collect();
    (
        TokenInput { tokens, n_tokens: N_TOKENS, token_dim: TOKEN_DIM },
        target,
    )
}

/// Forward + MSE backward → RegionalGradients (host buffer reused
/// across calls). The d_pred for the last tick is `pred - target`
/// (MSE gradient); other ticks are zero so the loss only weighs the
/// final-tick prediction.
fn backward_mse(
    weights: &RegionalWeights,
    input: &TokenInput,
    target: &[f32],
) -> RegionalGradients {
    let state = <RegionalBrain as Brain>::init_state(weights);
    let (output, _state, cache) =
        <RegionalBrain as Brain>::forward_cached(weights, state, input);
    let last_tick = output.predictions.len() - 1;
    let d_preds: Vec<Vec<f32>> = output.predictions.iter().enumerate().map(|(t, p)| {
        if t == last_tick {
            p.iter().zip(target.iter()).map(|(pv, tv)| pv - tv).collect()
        } else {
            vec![0.0f32; p.len()]
        }
    }).collect();
    let (grads, _) = RegionalBrain::backward_with_input_grad(weights, cache, &d_preds);
    grads
}

/// Forward only; return MSE on the last-tick prediction.
fn forward_mse(weights: &RegionalWeights, input: &TokenInput, target: &[f32]) -> f32 {
    let state = <RegionalBrain as Brain>::init_state(weights);
    let (output, _state, _cache) =
        <RegionalBrain as Brain>::forward_cached(weights, state, input);
    let pred = output.predictions.last().expect("at least one tick");
    let mut sumsq = 0.0f32;
    for (p, t) in pred.iter().zip(target.iter()) {
        let d = p - t;
        sumsq += d * d;
    }
    sumsq / (pred.len() as f32)
}

/// Run the full pipeline once with `seed` and return
/// `(top_k_held_out_mse, random_k_held_out_mse)`.
fn run_one_trial(seed: u64) -> (f32, f32) {
    let cfg = RegionalConfig::eight_region_small(TOKEN_DIM, OUT_DIMS, TICKS);
    let initial_weights = RegionalWeights::new(cfg);

    let mut rng = SimpleRng::new(seed);

    // ── Training corpus: N_CONCEPT cluster-A + N_NOISE noise ──
    let mut train_samples: Vec<(TokenInput, Vec<f32>)> = Vec::new();
    let mut concept_indices: Vec<usize> = Vec::new();
    for i in 0..(N_CONCEPT + N_NOISE) {
        let from_concept_cluster = i % 2 == 0; // interleave so order doesn't bias
        if from_concept_cluster && train_samples.iter().filter(|_| true).count() < N_CONCEPT * 2 {
            // Use a unique phase per concept sample so they aren't identical.
            train_samples.push(concept_sample(0.05 * i as f32));
            concept_indices.push(train_samples.len() - 1);
        } else {
            train_samples.push(noise_sample(&mut rng));
        }
    }
    // Above interleave logic can be off by one; just rebuild cleanly.
    train_samples.clear();
    concept_indices.clear();
    for i in 0..N_CONCEPT {
        train_samples.push(concept_sample(0.07 * i as f32 + 0.1));
        concept_indices.push(train_samples.len() - 1);
    }
    for _ in 0..N_NOISE {
        train_samples.push(noise_sample(&mut rng));
    }

    // ── Queries (held-out from training) ──
    let queries: Vec<(TokenInput, Vec<f32>)> = (0..N_QUERIES)
        .map(|q| concept_sample(2.0 + 0.3 * q as f32))
        .collect();

    // ── Held-out evaluation set (held-out from training AND queries) ──
    let held_out: Vec<(TokenInput, Vec<f32>)> = (0..N_HELD_OUT)
        .map(|h| concept_sample(5.0 + 0.4 * h as f32))
        .collect();

    // ── Probe gradient dim, build projection ──
    let probe_grads = RegionalGradients::zeros(&initial_weights);
    let in_dim = probe_grads.flatten().len();
    let proj = Projection::new_dense(in_dim, PROJ_DIM, seed ^ 0xA77216u64);

    // ── Per-sample projected gradients for training corpus ──
    let train_proj: Vec<Vec<f32>> = train_samples.iter().map(|(input, target)| {
        let grads = backward_mse(&initial_weights, input, target);
        proj.project_normalized(&grads.flatten())
    }).collect();

    // ── Aggregated query gradient (mean of normalised per-query) ──
    let query_proj_each: Vec<Vec<f32>> = queries.iter().map(|(input, target)| {
        let grads = backward_mse(&initial_weights, input, target);
        proj.project_normalized(&grads.flatten())
    }).collect();
    let mut q_avg = vec![0.0f32; PROJ_DIM];
    for q in &query_proj_each {
        for (a, v) in q_avg.iter_mut().zip(q) { *a += v; }
    }
    let n_q = query_proj_each.len() as f32;
    for v in &mut q_avg { *v /= n_q; }
    // Re-normalise the aggregated direction.
    let n: f32 = q_avg.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n > 0.0 { for v in &mut q_avg { *v /= n; } }

    // ── Rank ──
    let top_k = Attributor::top_k(&q_avg, &train_proj, K);
    let random_k: Vec<usize> = {
        let mut idx: Vec<usize> = (0..train_samples.len()).collect();
        // Deterministic shuffle via Fisher-Yates with our SimpleRng.
        let mut rng2 = SimpleRng::new(seed ^ 0xCAFE);
        for i in (1..idx.len()).rev() {
            // Use next_normal as a poor-man's random uniform: take its
            // bit pattern modulo (i+1).
            let r = rng2.next_normal().to_bits() as usize % (i + 1);
            idx.swap(i, r);
        }
        idx.truncate(K);
        idx
    };

    // Diagnostic: how many of top_k are actually concept-cluster?
    let top_k_concept = top_k.iter().filter(|i| concept_indices.contains(i)).count();
    let random_k_concept = random_k.iter().filter(|i| concept_indices.contains(i)).count();
    eprintln!(
        "trial seed={seed:#x}: top_k cluster-A hits = {top_k_concept}/{K}, \
         random_k cluster-A hits = {random_k_concept}/{K}",
    );

    // ── Train brain A on top_k ──
    let mut weights_a = initial_weights.clone();
    for _epoch in 0..EPOCHS {
        for &i in &top_k {
            let (input, target) = &train_samples[i];
            let mut grads = backward_mse(&weights_a, input, target);
            grads.apply(&mut weights_a, LR, 5.0);
        }
    }
    let mse_a: f32 = held_out.iter()
        .map(|(input, target)| forward_mse(&weights_a, input, target))
        .sum::<f32>() / (held_out.len() as f32);

    // ── Train brain B on random_k ──
    let mut weights_b = initial_weights.clone();
    for _epoch in 0..EPOCHS {
        for &i in &random_k {
            let (input, target) = &train_samples[i];
            let mut grads = backward_mse(&weights_b, input, target);
            grads.apply(&mut weights_b, LR, 5.0);
        }
    }
    let mse_b: f32 = held_out.iter()
        .map(|(input, target)| forward_mse(&weights_b, input, target))
        .sum::<f32>() / (held_out.len() as f32);

    eprintln!(
        "trial seed={seed:#x}: held-out MSE: top_k = {mse_a:.6}, random = {mse_b:.6}",
    );

    (mse_a, mse_b)
}

/// The actual win-condition test. Run two trials with different seeds
/// (variance check); both must show attribution-selected K beating
/// random K on held-out MSE. If either fails, the test fails — we
/// report exactly which trial regressed and by how much.
#[test]
fn attribution_selection_beats_random_on_controlled_task() {
    let trials = [0xA77216u64, 0xBEEF42u64];
    let mut wins = 0usize;
    let mut details = Vec::new();
    for &seed in &trials {
        let (top_k_mse, random_mse) = run_one_trial(seed);
        let won = top_k_mse < random_mse;
        if won { wins += 1; }
        details.push((seed, top_k_mse, random_mse, won));
    }

    eprintln!("\n──── win/loss summary ────");
    for (seed, a, b, won) in &details {
        let marker = if *won { "WIN" } else { "LOSS" };
        eprintln!(
            "  seed={seed:#x}: top_k = {a:.6}, random = {b:.6}, delta = {:+.6}  [{marker}]",
            a - b,
        );
    }
    eprintln!("  total: {wins}/{} wins", trials.len());

    // Both trials must win for the test to pass. Either single trial
    // could be lucky; requiring both gives modest variance robustness
    // without claiming significance we can't actually establish at
    // n=2 trials.
    assert_eq!(wins, trials.len(),
        "attribution-selected K did not beat random K on held-out MSE in all trials — \
         the technique didn't work on this synthetic task. details: {details:?}");
}
