//! Test regional CTM training with BPTT.

use modgrad_runtime::regional::*;

#[test]
fn four_region_bptt_loss_decreases() {
    let cfg = RegionalConfig::four_region(16, 4, 4); // 4 ticks, 4 classes
    let mut w = RegionalWeights::new(cfg);
    eprintln!("  params: {}", w.n_params());

    let examples: Vec<(Vec<f32>, usize)> = vec![
        (vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
        (vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1),
        (vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2),
        (vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3),
    ];

    let lr = 0.01;
    let clip = 5.0;
    let mut losses = Vec::new();

    for epoch in 0..100 {
        let mut epoch_loss = 0.0;
        let mut correct = 0;

        for (obs, target) in &examples {
            let mut grads = RegionalGradients::zeros(&w);
            let (loss, pred) = regional_train_step(&w, &mut grads, obs, *target);
            grads.apply(&mut w, lr, clip);
            epoch_loss += loss;
            if pred == *target { correct += 1; }
        }

        epoch_loss /= examples.len() as f32;
        losses.push(epoch_loss);

        if epoch % 25 == 0 || epoch == 99 {
            eprintln!("  epoch {epoch:3}: loss={epoch_loss:.4} acc={correct}/{}",
                examples.len());
        }
    }

    let first_10: f32 = losses[..10].iter().sum::<f32>() / 10.0;
    let last_10: f32 = losses[90..].iter().sum::<f32>() / 10.0;
    eprintln!("  first 10 avg: {first_10:.4}, last 10 avg: {last_10:.4}");
    assert!(last_10 < first_10, "loss should decrease: first={first_10:.4} last={last_10:.4}");
}

#[test]
fn eight_region_bptt_loss_decreases() {
    // 8-region has more params + longer gradient chains → needs lower lr
    let cfg = RegionalConfig::eight_region(16, 4, 4);
    let mut w = RegionalWeights::new(cfg);
    eprintln!("  8-region params: {}", w.n_params());

    let examples: Vec<(Vec<f32>, usize)> = vec![
        (vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
        (vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1),
        (vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2),
        (vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3),
    ];

    let lr = 0.001; // lower lr for deeper network
    let clip = 2.0;  // tighter clipping
    let mut losses = Vec::new();

    for epoch in 0..200 {
        let mut epoch_loss = 0.0;
        for (obs, target) in &examples {
            let mut grads = RegionalGradients::zeros(&w);
            let (loss, _) = regional_train_step(&w, &mut grads, obs, *target);
            grads.apply(&mut w, lr, clip);
            epoch_loss += loss;
        }
        epoch_loss /= examples.len() as f32;
        losses.push(epoch_loss);

        if epoch % 50 == 0 || epoch == 199 {
            eprintln!("  epoch {epoch:3}: loss={epoch_loss:.4}");
        }
    }

    let first_10: f32 = losses[..10].iter().sum::<f32>() / 10.0;
    let last_10: f32 = losses[190..].iter().sum::<f32>() / 10.0;
    eprintln!("  first 10 avg: {first_10:.4}, last 10 avg: {last_10:.4}");
    assert!(last_10 < first_10, "loss should decrease: first={first_10:.4} last={last_10:.4}");
}
