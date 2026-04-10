//! Stage 3: Multimodal binding — image patches + label text in one token sequence.
//!
//! Train:  [patch_0, ..., patch_63, 'c', 'a', 't'] → predict class
//! Test:   [patch_0, ..., patch_63]                 → predict class (no label)
//!
//! The brain sees image and text together during training. Attention binds them.
//! At test time, only vision — the brain must recall the association.
//!
//! cargo test --release --test ctm_multimodal -- --nocapture

use modgrad::ctm_config::CtmConfig;
use modgrad::ctm_weights::{CtmWeights, CtmState};
use modgrad::ctm_train::{CtmGradients, train_step};
use modgrad::ctm_forward::ctm_forward;
use modgrad::cifar::{load_feat, extract_patches_4x4, CLASSES};
use modgrad::ctm::{Linear, SimpleRng};

/// Project patches through Linear(48→128).
fn patches_to_tokens(patches: &[f32], proj: &Linear) -> Vec<f32> {
    let n = patches.len() / 48;
    let mut tokens = Vec::with_capacity(n * 128);
    for i in 0..n {
        tokens.extend_from_slice(&proj.forward(&patches[i * 48..(i + 1) * 48]));
    }
    tokens
}

/// Encode label string as byte tokens via embed_table.
fn label_to_tokens(label: usize, embed_table: &[f32], d: usize) -> Vec<f32> {
    let name = CLASSES[label].as_bytes();
    let mut tokens = Vec::with_capacity(name.len() * d);
    for &b in name {
        let off = b as usize * d;
        tokens.extend_from_slice(&embed_table[off..off + d]);
    }
    tokens
}

/// Evaluate on test set (vision only, no label tokens).
fn eval_vision_only(
    w: &CtmWeights, data: &[modgrad::cifar::CifarImage],
    patch_proj: &Linear, n: usize,
) -> f32 {
    let d = w.config.d_input;
    let mut correct = 0usize;
    for img in data.iter().take(n) {
        let patches = extract_patches_4x4(&img.pixels);
        let tokens = patches_to_tokens(&patches, patch_proj);
        let mut state = CtmState::new(w);
        let out = ctm_forward(w, &mut state, &tokens, 64, d);
        let pred = out.predictions.last().unwrap().iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if pred == img.label { correct += 1; }
    }
    correct as f32 / n as f32
}

#[test]
fn stage3_multimodal() {
    let d = 128;
    let cfg = CtmConfig {
        iterations: 4, d_model: 128, d_input: d, heads: 4,
        n_synch_out: 64, n_synch_action: 64, synapse_depth: 3,
        memory_length: 8, deep_nlms: true, memory_hidden_dims: 4,
        out_dims: 10, n_random_pairing_self: 0, min_width: 16,
        ..Default::default()
    };

    // Load CIFAR
    let train_path = "cifar10_train_pixels.feat";
    if !std::path::Path::new(train_path).exists() {
        eprintln!("  SKIP: {} not found", train_path);
        return;
    }
    let train_data = load_feat(train_path).unwrap();
    let test_data = load_feat("cifar10_test_pixels.feat").unwrap();

    // Load Stage 2 brain or Stage 1 brain or fresh
    let mut w = if std::path::Path::new("brain_stage2.json").exists() {
        eprintln!("  loading brain_stage2.json (Stage 2 weights)");
        CtmWeights::load("brain_stage2.json").unwrap()
    } else if std::path::Path::new("brain_stage1.json").exists() {
        eprintln!("  loading brain_stage1.json (Stage 1 weights)");
        let mut w1 = CtmWeights::load("brain_stage1.json").unwrap();
        w1.output_proj = Linear::new(cfg.n_synch_out, 10);
        w1.config.out_dims = 10;
        w1
    } else {
        eprintln!("  no prior brain found, starting fresh");
        CtmWeights::new(cfg.clone(), d)
    };

    // Shared embed table for label bytes (same as Stage 1 text encoding)
    let mut rng = SimpleRng::new(42);
    let embed_table: Vec<f32> = (0..256 * d)
        .map(|_| rng.next_normal() * (1.0 / (d as f32).sqrt()))
        .collect();

    // Patch projection
    let patch_proj = Linear::new(48, d);

    eprintln!("\n  === Stage 3: Multimodal binding (image + text) ===");
    eprintln!("  brain params: {}", w.n_params());

    // Baseline: vision-only accuracy before multimodal training
    let baseline = eval_vision_only(&w, &test_data, &patch_proj, 200);
    eprintln!("  vision-only baseline: {:.1}% (random=10%)", baseline * 100.0);
    eprintln!();

    let lr = 0.001;
    let clip = 5.0;
    let batch = 8;
    let steps = 500;
    let t0 = std::time::Instant::now();
    let mut ema_loss = (10.0f32).ln();
    let mut correct = 0usize;
    let mut count = 0usize;

    for step in 0..steps {
        let mut grads = CtmGradients::zeros(&w);

        for b in 0..batch {
            let idx = (step * batch + b) % train_data.len();
            let img = &train_data[idx];

            // Image patches → 64 tokens of 128 dims
            let patches = extract_patches_4x4(&img.pixels);
            let mut tokens = patches_to_tokens(&patches, &patch_proj);
            let n_patch_tokens = 64;

            // Label text → N tokens of 128 dims (e.g. "cat" = 3 tokens)
            let label_tokens = label_to_tokens(img.label, &embed_table, d);
            let n_label_tokens = label_tokens.len() / d;
            tokens.extend_from_slice(&label_tokens);

            let n_total = n_patch_tokens + n_label_tokens;

            let r = train_step(&w, &mut grads, &tokens, n_total, d, img.label);
            ema_loss = 0.99 * ema_loss + 0.01 * r.loss;
            if r.prediction == img.label { correct += 1; }
            count += 1;
        }

        grads.apply(&mut w, lr / batch as f32, clip);

        if step % 100 == 0 || step == steps - 1 {
            // Test: vision-only (the real test — no label at test time)
            let test_acc = eval_vision_only(&w, &test_data, &patch_proj, 200);

            eprintln!("  step {:4}: loss={:.3} train_acc={:.1}% test_vision_only={:.1}% [{:.0}ms/step]",
                step, ema_loss,
                correct as f32 / count.max(1) as f32 * 100.0,
                test_acc * 100.0,
                t0.elapsed().as_millis() as f64 / (step + 1) as f64);
        }
    }

    let final_vision = eval_vision_only(&w, &test_data, &patch_proj, 500);
    eprintln!("\n  baseline (vision-only before training): {:.1}%", baseline * 100.0);
    eprintln!("  final (vision-only after multimodal):   {:.1}%", final_vision * 100.0);
    eprintln!("  delta: {:+.1}pp", (final_vision - baseline) * 100.0);

    w.save("brain_stage3.json").ok();
    eprintln!("  saved brain_stage3.json\n");
    std::fs::remove_file("brain_stage3.json").ok();
}
