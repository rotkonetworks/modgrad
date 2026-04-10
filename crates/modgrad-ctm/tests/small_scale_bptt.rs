//! Verify SDK CTM BPTT works at small scale (8 neurons).
//! This is the cerebellum/BG size — if BPTT can't train 8 neurons, the
//! regional architecture won't work.

use modgrad_ctm::config::CtmConfig;
use modgrad_ctm::weights::CtmWeights;
use modgrad_ctm::train::{CtmGradients, train_step};

#[test]
fn small_ctm_bptt_converges() {
    let cfg = CtmConfig {
        iterations: 4,
        d_model: 8,
        d_input: 8,
        heads: 2,
        n_synch_out: 8,
        n_synch_action: 8,
        synapse_depth: 1, // simple Linear+SiLU, no U-Net
        memory_length: 4,
        deep_nlms: false,
        memory_hidden_dims: 2,
        out_dims: 4, // 4-class classification
        n_random_pairing_self: 0,
        min_width: 4,
        ..Default::default()
    };

    let raw_dim = 4;
    let mut w = CtmWeights::new(cfg.clone(), raw_dim);
    eprintln!("  small CTM params: {}", w.n_params());

    // 4 training examples: each class has a distinct input pattern
    let examples: Vec<(Vec<f32>, usize)> = vec![
        (vec![1.0, 0.0, 0.0, 0.0], 0),
        (vec![0.0, 1.0, 0.0, 0.0], 1),
        (vec![0.0, 0.0, 1.0, 0.0], 2),
        (vec![0.0, 0.0, 0.0, 1.0], 3),
    ];

    let lr = 0.01;
    let clip = 5.0;
    let mut losses = Vec::new();

    for epoch in 0..200 {
        let mut epoch_loss = 0.0;
        let mut correct = 0;

        for (obs, target) in &examples {
            let mut grads = CtmGradients::zeros(&w);
            let result = train_step(&w, &mut grads, obs, 1, raw_dim, *target);
            grads.apply(&mut w, lr, clip);
            epoch_loss += result.loss;
            if result.prediction == *target { correct += 1; }
        }

        epoch_loss /= examples.len() as f32;
        losses.push(epoch_loss);

        if epoch % 50 == 0 || epoch == 199 {
            eprintln!("  epoch {epoch:3}: loss={epoch_loss:.4} acc={correct}/{}", examples.len());
        }
    }

    // Loss should decrease
    let first_10: f32 = losses[..10].iter().sum::<f32>() / 10.0;
    let last_10: f32 = losses[190..].iter().sum::<f32>() / 10.0;
    eprintln!("  first 10 avg: {first_10:.4}, last 10 avg: {last_10:.4}");
    assert!(last_10 < first_10, "loss should decrease: first={first_10:.4} last={last_10:.4}");

    // Should get at least 3/4 correct on this trivial task
    let mut final_correct = 0;
    for (obs, target) in &examples {
        let mut grads = CtmGradients::zeros(&w);
        let result = train_step(&w, &mut grads, obs, 1, raw_dim, *target);
        if result.prediction == *target { final_correct += 1; }
    }
    eprintln!("  final accuracy: {final_correct}/{}", examples.len());
    assert!(final_correct >= 3, "should get >=3/4 correct, got {final_correct}");
}

#[test]
fn medium_ctm_bptt_converges() {
    // 64 neurons — typical cortical region size
    let cfg = CtmConfig {
        iterations: 8,
        d_model: 64,
        d_input: 32,
        heads: 4,
        n_synch_out: 64,
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
    eprintln!("  medium CTM params: {}", w.n_params());

    // 10-class XOR-like patterns
    let mut examples = Vec::new();
    for class in 0..10 {
        let mut obs = vec![0.0f32; raw_dim];
        // Each class gets a unique pair of features active
        obs[class % raw_dim] = 1.0;
        obs[(class * 3 + 7) % raw_dim] = 0.5;
        examples.push((obs, class));
    }

    let lr = 0.005;
    let clip = 5.0;
    let mut losses = Vec::new();

    for epoch in 0..100 {
        let mut epoch_loss = 0.0;
        for (obs, target) in &examples {
            let mut grads = CtmGradients::zeros(&w);
            let result = train_step(&w, &mut grads, obs, 1, raw_dim, *target);
            grads.apply(&mut w, lr, clip);
            epoch_loss += result.loss;
        }
        epoch_loss /= examples.len() as f32;
        losses.push(epoch_loss);

        if epoch % 25 == 0 || epoch == 99 {
            eprintln!("  epoch {epoch:3}: loss={epoch_loss:.4}");
        }
    }

    let first_10: f32 = losses[..10].iter().sum::<f32>() / 10.0;
    let last_10: f32 = losses[90..].iter().sum::<f32>() / 10.0;
    eprintln!("  first 10 avg: {first_10:.4}, last 10 avg: {last_10:.4}");
    assert!(last_10 < first_10, "loss should decrease: first={first_10:.4} last={last_10:.4}");
}
