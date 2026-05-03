//! End-to-end attribution wiring test on a real (tiny) brain.
//!
//! This test isn't measuring whether attribution improves anything —
//! that's the experiment in `brain_qwen_nll` after the harness lands.
//! Here we just prove the pipeline composes: a real backward through
//! the SDK's regional brain produces a flattenable RegionalGradients,
//! the JL projection accepts it, and the resulting cosine ranking
//! puts a synthetic-target sample above noise.
//!
//! Pipeline:
//!   1. Build a small `eight_region_small` brain.
//!   2. Generate `n_samples` token inputs.
//!   3. For each sample: forward → backward → flatten → project →
//!      ℓ²-normalise → store.
//!   4. Pick one sample as the "query" (planted positive); the
//!      remaining samples are "distractors".
//!   5. Run [`Attributor::top_k`] with `k=1` and assert the planted
//!      sample comes back as the top match — a sanity check that the
//!      whole pipeline preserves enough signal end-to-end.

use modgrad_attribution::{Attributor, Projection};
use modgrad_ctm::graph::{RegionalBrain, RegionalConfig, RegionalGradients, RegionalWeights};
use modgrad_traits::{Brain, TokenInput};

#[test]
fn brain_attribution_pipeline_recovers_planted_query() {
    let token_dim = 16usize;
    let n_tokens = 4usize;
    let out_dims = 32usize;
    let ticks = 2usize;
    let cfg = RegionalConfig::eight_region_small(token_dim, out_dims, ticks);
    let weights = RegionalWeights::new(cfg);

    // Probe gradient dimensionality up front so we can size the
    // projection matrix without an extra backward call.
    let zero_grads = RegionalGradients::zeros(&weights);
    let in_dim = zero_grads.flatten().len();
    let out_dim = 256usize;
    let proj = Projection::new_dense(in_dim, out_dim, 0xA77216u64);
    eprintln!(
        "brain_attribution: gradient flat dim = {in_dim}, projecting to {out_dim}"
    );

    // Helper: run a real backward for a sample, return the ℓ²-normalised
    // projected gradient.
    let project_sample = |sample: &TokenInput| -> Vec<f32> {
        let state = <RegionalBrain as Brain>::init_state(&weights);
        let (output, _state, cache) =
            <RegionalBrain as Brain>::forward_cached(&weights, state, sample);
        // Synthetic upstream gradient: tiny scaled prediction. The exact
        // value doesn't matter — we only need a non-zero d_pred so the
        // backward writes something into RegionalGradients.
        let d_preds: Vec<Vec<f32>> = output.predictions.iter()
            .map(|p| p.iter().map(|&v| v * 1e-3).collect()).collect();
        let (grads, _) = RegionalBrain::backward_with_input_grad(&weights, cache, &d_preds);
        let flat = grads.flatten();
        assert_eq!(flat.len(), in_dim, "gradient flat dim must be stable");
        proj.project_normalized(&flat)
    };

    // Build n_samples deterministic inputs with simple parametric
    // diversity (sin-ish patterns of different phases).
    let n_samples = 12usize;
    let inputs: Vec<TokenInput> = (0..n_samples)
        .map(|s| {
            let phase = s as f32 * 0.7;
            TokenInput {
                tokens: (0..n_tokens * token_dim)
                    .map(|i| ((i as f32 * 0.05) + phase).sin())
                    .collect(),
                n_tokens,
                token_dim,
            }
        })
        .collect();

    let train_proj: Vec<Vec<f32>> = inputs.iter().map(&project_sample).collect();

    // Pick the middle sample as the query (its own gradient acts as
    // the "what we want to attribute against").
    let query_idx = n_samples / 2;
    let query_proj = project_sample(&inputs[query_idx]);

    // The query's own projected gradient must rank itself top — same
    // input → same backward → same flat gradient → cosine = 1.
    let top1 = Attributor::top_k(&query_proj, &train_proj, 1);
    assert_eq!(top1, vec![query_idx],
        "self-query top-1 must be the query sample's own index — \
         pipeline lost determinism: top1 = {top1:?}, expected {query_idx}");

    // Sanity: top-3 should include the query plus its temporal
    // neighbours (similar phase). At minimum, the query is in there.
    let top3 = Attributor::top_k(&query_proj, &train_proj, 3);
    assert!(top3.contains(&query_idx),
        "self-query must be in top-3: got {top3:?}");
    eprintln!("brain_attribution: top-3 for query idx {query_idx}: {top3:?}");
}
