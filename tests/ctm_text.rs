//! Stage 1: Train the Ctm CTM on next-byte prediction.
//!
//! Each byte → embed_table → 128-dim token. Context window of previous bytes
//! as KV tokens. CTM attends, thinks, predicts next byte.
//!
//! cargo test --release --test ctm_text -- --nocapture

use modgrad::ctm_config::CtmConfig;
use modgrad::ctm_weights::{CtmWeights, CtmState};
use modgrad::ctm_train::{CtmGradients, train_step};
use modgrad::ctm::{Linear, SimpleRng};

/// Embed a byte sequence into tokens: [context_len × embed_dim].
fn embed_bytes(
    bytes: &[u8], embed_table: &[f32], embed_dim: usize,
) -> Vec<f32> {
    let mut tokens = Vec::with_capacity(bytes.len() * embed_dim);
    for &b in bytes {
        let offset = b as usize * embed_dim;
        tokens.extend_from_slice(&embed_table[offset..offset + embed_dim]);
    }
    tokens
}

#[test]
fn stage1_next_byte() {
    // Config: small brain for fast iteration
    let d_input = 128;
    let cfg = CtmConfig {
        iterations: 4,
        d_model: 128,
        d_input,
        heads: 4,
        n_synch_out: 64,
        n_synch_action: 64,
        synapse_depth: 3,      // U-Net with skip connections
        memory_length: 8,
        deep_nlms: true,
        memory_hidden_dims: 4,
        out_dims: 256,         // byte-level prediction
        n_random_pairing_self: 0,
        min_width: 16,
        ..Default::default()
    };

    // Load text
    let text = std::fs::read("train_climbmix.txt")
        .expect("train_climbmix.txt not found");
    let text = &text[..text.len().min(500_000)]; // use first 500KB

    // Embedding table: 256 bytes × 128 dims
    let vocab = 256;
    let mut rng = SimpleRng::new(42);
    let mut embed_table: Vec<f32> = (0..vocab * d_input)
        .map(|_| rng.next_normal() * (1.0 / (d_input as f32).sqrt()))
        .collect();

    // Brain: kv_proj expects raw_dim = d_input (embeddings are already d_input)
    let mut w = CtmWeights::new(cfg.clone(), d_input);
    eprintln!("\n  === Stage 1: Next-byte prediction ===");
    eprintln!("  brain params: {}", w.n_params());
    eprintln!("  embed params: {}", embed_table.len());
    eprintln!("  text: {} bytes", text.len());
    eprintln!("  d_model={}, d_input={}, ticks={}, synapse_depth={}",
        cfg.d_model, cfg.d_input, cfg.iterations, cfg.synapse_depth);
    eprintln!();

    let context_len = 32; // how many previous bytes the brain sees
    let lr = 0.001;
    let clip = 5.0;
    let batch_size = 8;
    let n_steps = 200;

    let t0 = std::time::Instant::now();
    let mut total_loss = 0.0f32;
    let mut total_correct = 0usize;
    let mut total_count = 0usize;

    for step in 0..n_steps {
        let mut grads = CtmGradients::zeros(&w);
        let mut step_loss = 0.0f32;

        for b in 0..batch_size {
            // Pick a random position in the text
            let pos = ((step * batch_size + b) * 137 + 7) % (text.len() - context_len - 1);
            let context = &text[pos..pos + context_len];
            let target = text[pos + context_len] as usize;

            // Embed context bytes into tokens
            let obs = embed_bytes(context, &embed_table, d_input);

            let r = train_step(&w, &mut grads, &obs, context_len, d_input, target);
            step_loss += r.loss;

            if r.prediction == target { total_correct += 1; }
            total_count += 1;
        }

        grads.apply(&mut w, lr / batch_size as f32, clip);
        step_loss /= batch_size as f32;
        total_loss = 0.95 * total_loss + 0.05 * step_loss;

        if step % 50 == 0 || step == n_steps - 1 {
            let ms_per_step = t0.elapsed().as_millis() as f64 / (step + 1) as f64;
            let acc = total_correct as f32 / total_count.max(1) as f32;
            eprintln!("  step {:4}: loss={:.3} acc={:.1}% [{:.0}ms/step]",
                step, total_loss, acc * 100.0, ms_per_step);
        }
    }

    let final_loss = total_loss;
    let random_loss = (256.0f32).ln(); // 5.55
    eprintln!("\n  final loss: {:.3} (random={:.3})", final_loss, random_loss);
    assert!(final_loss < random_loss, "loss should be below random");

    // Save brain
    w.save("brain_stage1.json").expect("save failed");
    eprintln!("  saved to brain_stage1.json ({:.1} MB)",
        std::fs::metadata("brain_stage1.json").map(|m| m.len() as f64 / 1e6).unwrap_or(0.0));
    std::fs::remove_file("brain_stage1.json").ok(); // cleanup
    eprintln!();
}
