//! Stage 2: Train Ctm CTM on CIFAR-10 via patch tokens.
//!
//! 32×32×3 image → 64 patches of 4×4×3=48 → patch_proj(48→128) → 64 tokens of 128 dims
//! Same brain architecture as Stage 1 (text). Only the input encoder and output head change.
//!
//! cargo test --release --test ctm_cifar -- --nocapture

use modgrad::ctm_config::CtmConfig;
use modgrad::ctm_weights::{CtmWeights, CtmState};
use modgrad::ctm_train::{CtmGradients, train_step};
use modgrad::cifar::{load_feat, extract_patches_4x4};
use modgrad::ctm::{Linear, SimpleRng};

/// Project patches through a Linear(48→128) to produce tokens.
fn patches_to_tokens(patches: &[f32], proj: &Linear) -> Vec<f32> {
    let n_patches = patches.len() / 48;
    let mut tokens = Vec::with_capacity(n_patches * 128);
    for i in 0..n_patches {
        let patch = &patches[i * 48..(i + 1) * 48];
        tokens.extend_from_slice(&proj.forward(patch));
    }
    tokens
}

#[test]
fn stage2_cifar10() {
    let d_input = 128;
    let cfg = CtmConfig {
        iterations: 4,
        d_model: 128,
        d_input,
        heads: 4,
        n_synch_out: 64,
        n_synch_action: 64,
        synapse_depth: 3,
        memory_length: 8,
        deep_nlms: true,
        memory_hidden_dims: 4,
        out_dims: 10,          // CIFAR-10 classes
        n_random_pairing_self: 0,
        min_width: 16,
        ..Default::default()
    };

    // Load CIFAR-10
    let train_path = "cifar10_train_pixels.feat";
    if !std::path::Path::new(train_path).exists() {
        eprintln!("  SKIP: {} not found", train_path);
        return;
    }
    let train_data = load_feat(train_path).expect("failed to load CIFAR-10 train");
    let test_data = load_feat("cifar10_test_pixels.feat").expect("failed to load CIFAR-10 test");
    eprintln!("\n  === Stage 2: CIFAR-10 classification ===");
    eprintln!("  train: {} images, test: {} images", train_data.len(), test_data.len());

    // Try to load Stage 1 brain, fall back to fresh
    let mut w = if std::path::Path::new("brain_stage1.json").exists() {
        eprintln!("  loading brain_stage1.json (Stage 1 weights)...");
        let mut w1 = CtmWeights::load("brain_stage1.json").expect("load failed");
        // Stage 1 has out_dims=256 (bytes), Stage 2 needs out_dims=10 (classes).
        // Swap output_proj, keep everything else.
        w1.output_proj = Linear::new(cfg.n_synch_out, 10);
        w1.config.out_dims = 10;
        eprintln!("  swapped output head: sync → 10 classes");
        w1
    } else {
        eprintln!("  no Stage 1 brain found, starting fresh");
        CtmWeights::new(cfg.clone(), d_input)
    };

    // Patch projection: 48 → 128 (trainable, but not through BPTT yet — treated as encoder)
    let patch_proj = Linear::new(48, d_input);

    eprintln!("  brain params: {}", w.n_params());
    eprintln!("  patch_proj params: {}", 48 * d_input + d_input);
    eprintln!();

    let lr = 0.001;
    let clip = 5.0;
    let batch_size = 8;
    let n_steps = 500;
    let t0 = std::time::Instant::now();
    let mut ema_loss = (10.0f32).ln();
    let mut correct = 0usize;
    let mut count = 0usize;

    for step in 0..n_steps {
        let mut grads = CtmGradients::zeros(&w);

        for b in 0..batch_size {
            let idx = (step * batch_size + b) % train_data.len();
            let img = &train_data[idx];

            // Image → patches → tokens
            let patches = extract_patches_4x4(&img.pixels);
            let tokens = patches_to_tokens(&patches, &patch_proj);

            let r = train_step(&w, &mut grads, &tokens, 64, d_input, img.label);
            ema_loss = 0.99 * ema_loss + 0.01 * r.loss;
            if r.prediction == img.label { correct += 1; }
            count += 1;
        }

        grads.apply(&mut w, lr / batch_size as f32, clip);

        if step % 100 == 0 || step == n_steps - 1 {
            // Quick eval on 100 test images
            let mut test_correct = 0usize;
            let eval_n = 100;
            for i in 0..eval_n {
                let img = &test_data[i];
                let patches = extract_patches_4x4(&img.pixels);
                let tokens = patches_to_tokens(&patches, &patch_proj);
                let mut state = CtmState::new(&w);
                let out = modgrad::ctm_forward::ctm_forward(
                    &w, &mut state, &tokens, 64, d_input);
                let pred = out.predictions.last().unwrap().iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);
                if pred == img.label { test_correct += 1; }
            }

            eprintln!("  step {:4}: loss={:.3} train_acc={:.1}% test_acc={:.0}% [{:.0}ms/step]",
                step, ema_loss,
                correct as f32 / count.max(1) as f32 * 100.0,
                test_correct as f32 / eval_n as f32 * 100.0,
                t0.elapsed().as_millis() as f64 / (step + 1) as f64);
        }
    }

    eprintln!("\n  random = 10%, final train_acc = {:.1}%",
        correct as f32 / count.max(1) as f32 * 100.0);

    w.save("brain_stage2.json").ok();
    eprintln!("  saved brain_stage2.json\n");
    std::fs::remove_file("brain_stage2.json").ok();
}
