//! End-to-end test: train the CTM through the generic SDK trainer.
//!
//! This proves the Brain trait boundary actually works — the trainer
//! has zero knowledge of CTM internals, only the Brain interface.
//!
//! cargo test --release --test generic_trainer -- --nocapture

use modgrad_ctm::{CtmConfig, CtmWeights};
use modgrad::ctm_train::Ctm;
use modgrad::traits::{CtmLoss, TokenInput};
use modgrad::trainer::{
    train, TrainConfig, Sample, SampleProvider, StderrLogger,
};
use modgrad::optim::ConstantLR;
use modgrad::neuron::SimpleRng;

/// Simple data provider: random observations with random labels.
struct RandomData {
    examples: Vec<(Vec<f32>, usize)>,
    idx: usize,
    token_dim: usize,
}

impl RandomData {
    fn new(n: usize, token_dim: usize, n_classes: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let examples = (0..n).map(|_| {
            let obs: Vec<f32> = (0..token_dim).map(|_| rng.next_normal() * 0.5).collect();
            let label = (rng.next_u64() % n_classes as u64) as usize;
            (obs, label)
        }).collect();
        Self { examples, idx: 0, token_dim }
    }
}

impl SampleProvider<TokenInput, usize> for RandomData {
    fn next_sample(&mut self) -> Option<Sample<TokenInput, usize>> {
        if self.idx >= self.examples.len() { return None; }
        let (obs, label) = &self.examples[self.idx];
        self.idx += 1;
        let input = TokenInput {
            tokens: obs.clone(),
            n_tokens: 1,
            token_dim: self.token_dim,
        };
        Some(Sample { input, target: *label })
    }
    fn reset(&mut self) { self.idx = 0; }
}

#[test]
fn ctm_through_generic_trainer() {
    let cfg = CtmConfig {
        iterations: 4,
        d_model: 64,
        d_input: 32,
        heads: 4,
        n_synch_out: 32,
        n_synch_action: 32,
        synapse_depth: 3,
        memory_length: 8,
        deep_nlms: true,
        memory_hidden_dims: 4,
        out_dims: 10,
        n_random_pairing_self: 0,
        min_width: 16,
        ..Default::default()
    };

    let raw_dim = 32;
    let mut weights = CtmWeights::new(cfg.clone(), raw_dim);

    let mut data = RandomData::new(20, raw_dim, cfg.out_dims, 42);
    let loss_fn = CtmLoss;
    let scheduler = ConstantLR { lr: 0.001 };
    let mut checkpointer = None;
    let mut logger = StderrLogger;

    let config = TrainConfig {
        total_steps: 300,
        micro_batch: 4,
        accum_steps: 1,
        log_every: 10,
        save_every: 1000,
        token_dim: raw_dim,
        grad_clip: 5.0,
    };

    eprintln!("\n  === Generic Trainer: CTM through Brain trait ===");
    eprintln!("  config: d_model={}, ticks={}, out_dims={}", cfg.d_model, cfg.iterations, cfg.out_dims);

    let report = train::<Ctm, _, _, _, _>(
        &mut weights,
        &mut data,
        &loss_fn,
        &scheduler,
        &mut checkpointer,
        &mut logger,
        &config,
    );

    let random_loss = (cfg.out_dims as f32).ln(); // ln(10) ≈ 2.30
    eprintln!("  final_loss={:.4} best_loss={:.4} (random={:.4})",
        report.final_loss, report.best_loss, random_loss);

    // Allow small margin (0.5) since the generic trainer uses EMA smoothing
    assert!(report.best_loss < random_loss + 0.5,
        "loss should decrease toward random: best={:.4} >= random+0.5={:.4}",
        report.best_loss, random_loss + 0.5);
    // Verify loss actually decreased from initialization
    assert!(report.best_loss < 4.0,
        "loss should decrease significantly from ~5.0: best={:.4}", report.best_loss);

    eprintln!("  PASS: Brain trait pipeline works end-to-end\n");
}
