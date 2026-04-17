//! End-to-end test: 4-region hierarchical CTM learns byte prediction.
//!
//! No organism, no CLI, no device abstraction. Just:
//!   byte → embedding → regional_train_step → does loss decrease?
//!
//! This is the proof that the architecture can learn language.

use modgrad_runtime::regional::*;
use modgrad_compute::neuron::SimpleRng;

/// Simple byte embedding table: 256 × embed_dim.
struct ByteEmbeddings {
    table: Vec<f32>,
    embed_dim: usize,
}

impl ByteEmbeddings {
    fn new(embed_dim: usize) -> Self {
        let mut rng = SimpleRng::new(42);
        let scale = (1.0 / embed_dim as f32).sqrt();
        let mut table = vec![0.0f32; 256 * embed_dim];

        for byte_val in 0..256u16 {
            let offset = byte_val as usize * embed_dim;
            let b = byte_val as u8;
            // Structured features in first 8 dims (same as organism)
            if embed_dim > 0 { table[offset] = (b as f32 / 128.0) - 1.0; }
            if embed_dim > 1 { table[offset + 1] = if b.is_ascii_alphabetic() { 1.0 } else { -1.0 }; }
            if embed_dim > 2 { table[offset + 2] = if b.is_ascii_uppercase() { 1.0 } else { -1.0 }; }
            if embed_dim > 3 { table[offset + 3] = if b.is_ascii_digit() { 1.0 } else { -1.0 }; }
            if embed_dim > 4 { table[offset + 4] = if b == b' ' || b == b'\n' { 1.0 } else { -1.0 }; }
            if embed_dim > 5 { table[offset + 5] = if b.is_ascii_punctuation() { 1.0 } else { -1.0 }; }
            if embed_dim > 6 { table[offset + 6] = if b.is_ascii_graphic() { 1.0 } else { -1.0 }; }
            if embed_dim > 7 { table[offset + 7] = (b.count_ones() as f32 / 4.0) - 1.0; }
            for j in 8..embed_dim {
                table[offset + j] = rng.next_normal() * scale;
            }
        }
        Self { table, embed_dim }
    }

    fn embed(&self, byte: u8) -> &[f32] {
        let offset = byte as usize * self.embed_dim;
        &self.table[offset..offset + self.embed_dim]
    }
}

#[test]
fn four_region_byte_prediction() {
    let embed_dim = 16;  // small for fast test
    let vocab_size = 256;
    let ticks = 2;       // fewer ticks for speed
    let lr = 0.005;
    let clip = 3.0;

    let cfg = RegionalConfig::four_region(embed_dim, vocab_size, ticks);
    let mut w = RegionalWeights::new(cfg);
    let embeddings = ByteEmbeddings::new(embed_dim);

    eprintln!("  4-region byte CTM: {} params", w.n_params());

    // Training data: simple repeating pattern
    let training_text = b"the cat sat. the cat sat. the cat sat. ";
    let text_bytes: Vec<u8> = training_text.iter().copied()
        .cycle().take(training_text.len() * 10).collect();

    let mut losses = Vec::new();
    let context_len = 8;
    let max_steps = 100;

    for step in 0..max_steps {
        let offset = (step * context_len) % (text_bytes.len() - context_len - 1);
        let chunk = &text_bytes[offset..offset + context_len + 1];

        let mut step_loss = 0.0f32;
        let mut step_correct = 0usize;

        for pos in 0..context_len {
            let obs = embeddings.embed(chunk[pos]);
            let target = chunk[pos + 1] as usize;

            let mut grads = RegionalGradients::zeros(&w);
            let (loss, pred) = regional_train_step(&w, &mut grads, obs, target);
            grads.apply(&mut w, lr, clip);

            step_loss += loss;
            if pred == target { step_correct += 1; }
        }

        step_loss /= context_len as f32;
        losses.push(step_loss);

        if step % 25 == 0 || step == max_steps - 1 {
            eprintln!("  step {step:3}: loss={step_loss:.3} acc={step_correct}/{context_len}");
        }
    }

    let first_20: f32 = losses[..20].iter().sum::<f32>() / 20.0;
    let last_20: f32 = losses[80..].iter().sum::<f32>() / 20.0;
    eprintln!("  first 20 avg: {first_20:.3}, last 20 avg: {last_20:.3}");

    assert!(last_20 < first_20,
        "loss should decrease: first={first_20:.3} last={last_20:.3}");
}

#[test]
fn eight_region_byte_prediction() {
    let embed_dim = 16;
    let vocab_size = 256;
    let ticks = 2;
    let lr = 0.002;
    let clip = 2.0;

    let cfg = RegionalConfig::eight_region(embed_dim, vocab_size, ticks);
    let mut w = RegionalWeights::new(cfg);
    let embeddings = ByteEmbeddings::new(embed_dim);

    eprintln!("  8-region byte CTM: {} params", w.n_params());

    let training_text = b"the cat sat. the cat sat. ";
    let text_bytes: Vec<u8> = training_text.iter().copied()
        .cycle().take(training_text.len() * 10).collect();

    let mut losses = Vec::new();
    let context_len = 8;

    for step in 0..80 {
        let offset = (step * context_len) % (text_bytes.len() - context_len - 1);
        let chunk = &text_bytes[offset..offset + context_len + 1];

        let mut step_loss = 0.0f32;
        for pos in 0..context_len {
            let obs = embeddings.embed(chunk[pos]);
            let target = chunk[pos + 1] as usize;
            let mut grads = RegionalGradients::zeros(&w);
            let (loss, _) = regional_train_step(&w, &mut grads, obs, target);
            grads.apply(&mut w, lr, clip);
            step_loss += loss;
        }
        step_loss /= context_len as f32;
        losses.push(step_loss);

        if step % 20 == 0 || step == 79 {
            eprintln!("  step {step:2}: loss={step_loss:.3}");
        }
    }

    let first_15: f32 = losses[..15].iter().sum::<f32>() / 15.0;
    let last_15: f32 = losses[65..].iter().sum::<f32>() / 15.0;
    eprintln!("  first 15 avg: {first_15:.3}, last 15 avg: {last_15:.3}");
    assert!(last_15 < first_15,
        "loss should decrease: first={first_15:.3} last={last_15:.3}");
}

#[test]
fn adamw_with_builtin_embeddings() {
    // Test the new self-contained training path: embed table + AdamW + regional_train_token
    let embed_dim = 16;
    let vocab_size = 256;
    let ticks = 2;

    let cfg = RegionalConfig::four_region(embed_dim, vocab_size, ticks);
    let mut w = RegionalWeights::new(cfg);
    let mut opt = RegionalAdamW::new(&w).with_lr(1e-3).with_clip(2.0);

    eprintln!("  AdamW test: {} params", w.n_params());

    let training_text = b"the cat sat. the cat sat. the cat sat. ";
    let text_bytes: Vec<u8> = training_text.iter().copied()
        .cycle().take(training_text.len() * 10).collect();

    let mut losses = Vec::new();
    let context_len = 8;
    let max_steps = 100;

    for step in 0..max_steps {
        let offset = (step * context_len) % (text_bytes.len() - context_len - 1);
        let chunk = &text_bytes[offset..offset + context_len + 1];

        let mut grads = RegionalGradients::zeros(&w);
        let mut step_loss = 0.0f32;
        let mut step_correct = 0usize;

        for pos in 0..context_len {
            let token = chunk[pos] as usize;
            let target = chunk[pos + 1] as usize;
            let (loss, pred) = regional_train_token(&w, &mut grads, token, target);
            step_loss += loss;
            if pred == target { step_correct += 1; }
        }

        opt.step(&mut w, &mut grads);

        step_loss /= context_len as f32;
        losses.push(step_loss);

        if step % 25 == 0 || step == max_steps - 1 {
            eprintln!("  step {step:3}: loss={step_loss:.3} acc={step_correct}/{context_len}");
        }
    }

    let first_20: f32 = losses[..20].iter().sum::<f32>() / 20.0;
    let last_20: f32 = losses[80..].iter().sum::<f32>() / 20.0;
    eprintln!("  first 20 avg: {first_20:.3}, last 20 avg: {last_20:.3}");
    assert!(last_20 < first_20,
        "loss should decrease with AdamW: first={first_20:.3} last={last_20:.3}");

    // Test save/load roundtrip
    let tmp = "/tmp/test_regional_adamw.json";
    let opt_tmp = "/tmp/test_regional_adamw.opt.json";
    w.save(tmp).expect("save failed");
    opt.save(opt_tmp).expect("opt save failed");
    let w2 = RegionalWeights::load(tmp).expect("load failed");
    let opt2 = RegionalAdamW::load(opt_tmp).expect("opt load failed");
    assert_eq!(w2.n_params(), w.n_params());
    assert_eq!(opt2.step, opt.step);
    std::fs::remove_file(tmp).ok();
    std::fs::remove_file(opt_tmp).ok();
}
