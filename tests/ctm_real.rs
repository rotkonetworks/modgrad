//! Real training run: Ctm CTM with n-gram hash embeddings on climbmix.
//! 10K+ steps, proper batch size, n-gram augmented byte embeddings.
//!
//! cargo test --release --test ctm_real -- --nocapture

use modgrad::ctm_config::CtmConfig;
use modgrad::ctm_weights::CtmWeights;
use modgrad::ctm_train::{CtmGradients, train_step};
use modgrad::ngram_hash::NgramHashEmbeddings;
use modgrad::ctm::SimpleRng;

#[test]
fn real_text_training() {
    let d = 128;
    let cfg = CtmConfig {
        iterations: 4,
        d_model: 128,
        d_input: d,
        heads: 4,
        n_synch_out: 64,
        n_synch_action: 64,
        synapse_depth: 3,
        memory_length: 8,
        deep_nlms: true,
        memory_hidden_dims: 4,
        out_dims: 256,
        n_random_pairing_self: 0,
        min_width: 16,
        ..Default::default()
    };

    let text = std::fs::read("train_climbmix.txt").expect("train_climbmix.txt not found");

    // Byte embedding table
    let mut rng = SimpleRng::new(42);
    let embed_table: Vec<f32> = (0..256 * d)
        .map(|_| rng.next_normal() * (1.0 / (d as f32).sqrt()))
        .collect();

    // N-gram hash embeddings (BLT-style, scaled for our hardware)
    // 50K vocab per n-gram size, n=3..6 (4 tables × 50K × 128 = 25.6M floats = ~100MB)
    let ngram = NgramHashEmbeddings::new(d, 50_000, 3, 6);

    // Brain
    let mut w = CtmWeights::new(cfg.clone(), d);

    eprintln!("\n  === Real Training: Ctm CTM + n-gram hash ===");
    eprintln!("  brain params:  {}", w.n_params());
    eprintln!("  ngram params:  {} ({:.1}MB)", ngram.n_params(),
        ngram.n_params() as f64 * 4.0 / 1e6);
    eprintln!("  text:          {:.1}MB", text.len() as f64 / 1e6);
    eprintln!("  config:        d_model={} ticks={} synapse_depth={} ctx={}",
        cfg.d_model, cfg.iterations, cfg.synapse_depth, 64);
    eprintln!();

    let ctx = 64;       // context window (bytes)
    let lr = 0.001;
    let clip = 5.0;
    let batch = 16;
    let steps = 10_000;

    let t0 = std::time::Instant::now();
    let mut ema_loss = 5.5f32;
    let mut correct = 0usize;
    let mut count = 0usize;
    let mut best_loss = f32::MAX;

    for step in 0..steps {
        let mut grads = CtmGradients::zeros(&w);

        for b in 0..batch {
            // Random position in text
            let pos = ((step as u64 * batch as u64 + b as u64).wrapping_mul(2654435761) as usize + 7)
                % (text.len() - ctx - 1);
            let context = &text[pos..pos + ctx];
            let target = text[pos + ctx] as usize;

            // N-gram augmented byte embeddings
            let obs = ngram.embed_bytes(context, &embed_table, d);

            let r = train_step(&w, &mut grads, &obs, ctx, d, target);
            ema_loss = 0.995 * ema_loss + 0.005 * r.loss;
            if r.prediction == target { correct += 1; }
            count += 1;
        }

        grads.apply(&mut w, lr / batch as f32, clip);

        if ema_loss < best_loss { best_loss = ema_loss; }

        if step % 500 == 0 || step == steps - 1 {
            let elapsed = t0.elapsed().as_secs_f64();
            let ms_per = elapsed * 1000.0 / (step + 1) as f64;
            let acc = correct as f32 / count.max(1) as f32 * 100.0;
            let eta_min = (steps - step) as f64 * ms_per / 60_000.0;
            eprintln!("  step {:5}/{}: loss={:.3} best={:.3} acc={:.1}% [{:.0}ms/step, ETA {:.0}min]",
                step, steps, ema_loss, best_loss, acc, ms_per, eta_min);
        }
    }

    let random_loss = (256.0f32).ln(); // 5.545
    eprintln!("\n  FINAL: loss={:.3} (random={:.3}, best={:.3})", ema_loss, random_loss, best_loss);
    eprintln!("  accuracy: {:.1}%", correct as f32 / count.max(1) as f32 * 100.0);

    w.save("brain_ngram.json").ok();
    let size = std::fs::metadata("brain_ngram.json").map(|m| m.len()).unwrap_or(0);
    eprintln!("  saved brain_ngram.json ({:.1}MB)", size as f64 / 1e6);
    eprintln!();

    assert!(ema_loss < random_loss, "loss should be below random");
}
