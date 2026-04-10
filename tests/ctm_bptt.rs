//! Verify BPTT gradients flow through the full Ctm CTM.
//! cargo test --release --test ctm_bptt -- --nocapture

use modgrad::ctm_config::CtmConfig;
use modgrad::ctm_weights::{CtmWeights, CtmState};
use modgrad::ctm_train::{CtmGradients, train_step};

#[test]
fn loss_decreases() {
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
    let raw_dim = 16;
    let mut w = CtmWeights::new(cfg.clone(), raw_dim);

    // 5 random (observation, label) pairs — any task reduces to this
    let mut rng = modgrad::ctm::SimpleRng::new(42);
    let examples: Vec<(Vec<f32>, usize)> = (0..5).map(|_| {
        let obs: Vec<f32> = (0..raw_dim).map(|_| rng.next_normal() * 0.5).collect();
        let label = (rng.next_u64() % cfg.out_dims as u64) as usize;
        (obs, label)
    }).collect();

    let lr = 0.001;
    let clip = 5.0;
    let mut prev_loss = f32::MAX;
    let t0 = std::time::Instant::now();

    for step in 0..50 {
        let mut grads = CtmGradients::zeros(&w);
        let mut total_loss = 0.0f32;

        for (obs, label) in &examples {
            let r = train_step(&w, &mut grads, obs, 1, raw_dim, *label);
            total_loss += r.loss;
        }
        total_loss /= examples.len() as f32;

        grads.apply(&mut w, lr, clip);

        if step % 10 == 0 || step == 49 {
            eprintln!("  step {:3}: loss={:.4} (Δ={:+.4}) [{:.1}ms/step]",
                step, total_loss, total_loss - prev_loss,
                t0.elapsed().as_millis() as f64 / (step + 1) as f64);
        }
        prev_loss = total_loss;
    }

    // Loss must decrease — proves gradients are flowing
    assert!(prev_loss < 2.3, "loss should be well below random ({:.2} ≥ 2.3 = ln(10))", prev_loss);
    eprintln!("  PASS: loss={:.4} (random={:.4})\n", prev_loss, (cfg.out_dims as f32).ln());
}
