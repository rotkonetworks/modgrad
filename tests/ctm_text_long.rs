//! Long Stage 1 training run. cargo test --release --test ctm_text_long -- --nocapture
use modgrad::ctm_config::CtmConfig;
use modgrad::ctm_weights::CtmWeights;
use modgrad::ctm_train::{CtmGradients, train_step};
use modgrad::ctm::SimpleRng;

fn embed_bytes(bytes: &[u8], table: &[f32], d: usize) -> Vec<f32> {
    let mut t = Vec::with_capacity(bytes.len() * d);
    for &b in bytes { t.extend_from_slice(&table[b as usize * d..(b as usize + 1) * d]); }
    t
}

#[test]
fn stage1_long() {
    let d = 128;
    let cfg = CtmConfig {
        iterations: 4, d_model: 128, d_input: d, heads: 4,
        n_synch_out: 64, n_synch_action: 64, synapse_depth: 3,
        memory_length: 8, deep_nlms: true, memory_hidden_dims: 4,
        out_dims: 256, n_random_pairing_self: 0, min_width: 16,
        ..Default::default()
    };
    let text = std::fs::read("train_climbmix.txt").unwrap();
    let mut rng = SimpleRng::new(42);
    let embed: Vec<f32> = (0..256 * d).map(|_| rng.next_normal() * (1.0 / (d as f32).sqrt())).collect();
    let mut w = CtmWeights::new(cfg.clone(), d);
    eprintln!("\n  Stage 1 long: {} params, {} bytes text", w.n_params(), text.len());

    let ctx = 32;
    let lr = 0.001;
    let batch = 16;
    let steps = 2000;
    let t0 = std::time::Instant::now();
    let mut ema_loss = 5.5f32;
    let mut correct = 0usize;
    let mut count = 0usize;

    for step in 0..steps {
        let mut grads = CtmGradients::zeros(&w);
        for b in 0..batch {
            let pos = ((step * batch + b) * 137 + 7) % (text.len() - ctx - 1);
            let obs = embed_bytes(&text[pos..pos + ctx], &embed, d);
            let r = train_step(&w, &mut grads, &obs, ctx, d, text[pos + ctx] as usize);
            ema_loss = 0.99 * ema_loss + 0.01 * r.loss;
            if r.prediction == text[pos + ctx] as usize { correct += 1; }
            count += 1;
        }
        grads.apply(&mut w, lr / batch as f32, 5.0);
        if step % 200 == 0 || step == steps - 1 {
            eprintln!("  step {:5}: loss={:.3} acc={:.1}% [{:.0}ms/step]",
                step, ema_loss, correct as f32 / count.max(1) as f32 * 100.0,
                t0.elapsed().as_millis() as f64 / (step + 1) as f64);
        }
    }
    w.save("brain_stage1.json").unwrap();
    eprintln!("  saved brain_stage1.json\n");
}
