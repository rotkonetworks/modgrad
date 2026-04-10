//! Phase 0: BPTT training on next-byte prediction.
//! cargo test --release --test train_bptt -- --nocapture

use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, Linear, LayerConfig, SimpleRng};
use modgrad_runtime::train_bptt::{BpttTrainConfig, train_step};

#[test]
fn train_on_text() {
    let text = std::fs::read("train_climbmix.txt")
        .expect("train_climbmix.txt not found — run from project root");
    let tokens: Vec<usize> = text.iter().map(|&b| b as usize).collect();

    let vocab_size = 256;
    let embed_dim = 128;
    let d_input = 128;

    // CTM config: medium brain
    let cfg = CtmConfig {
        iterations: 4, d_input, n_sync_out: 64,
        input_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        ..CtmConfig::default()
    };

    let ctm = Ctm::new(cfg.clone());
    let (mut weights, _) = ctm.into_split();

    // Embedding table: random byte embeddings
    let mut rng = SimpleRng::new(42);
    let embed_table: Vec<f32> = (0..vocab_size * embed_dim)
        .map(|_| rng.next_normal() * 0.1)
        .collect();

    // Sensory projection: embed_dim → d_input
    let sensory = Linear::new(embed_dim, d_input);

    // Output projection: sync_dim → vocab_size
    let sync_dim = cfg.n_sync_out;
    let mut output_proj = Linear::new(sync_dim, vocab_size);

    let train_cfg = BpttTrainConfig {
        lr: 0.01,
        seq_len: 256,
        vocab_size,
        embed_dim,
        log_every: 50,
        ..Default::default()
    };

    eprintln!("\n  === Phase 0: BPTT on next-byte prediction ===");
    eprintln!("  vocab: {} (bytes), embed: {}, d_input: {}, sync: {}",
        vocab_size, embed_dim, d_input, sync_dim);
    eprintln!("  brain: {} ticks, {} total neurons",
        cfg.iterations,
        cfg.input_layer.n_neurons + cfg.attention_layer.n_neurons +
        cfg.output_layer.n_neurons + cfg.motor_layer.n_neurons);
    eprintln!("  text: {} bytes\n", tokens.len());

    let t0 = std::time::Instant::now();

    for step in 0..500 {
        // Random window into the 48MB text
        let mut rng2 = SimpleRng::new(step as u64 ^ 0xbeef);
        let start = (rng2.next_u64() as usize) % tokens.len().saturating_sub(train_cfg.seq_len + 1);
        let end = (start + train_cfg.seq_len).min(tokens.len());
        let seq = &tokens[start..end];

        let result = train_step(
            &mut weights, &embed_table, &sensory, &mut output_proj,
            seq, &train_cfg,
        );

        if (step + 1) % 50 == 0 {
            let acc = result.correct as f32 / result.total.max(1) as f32;
            eprintln!("  Step {:4}: loss={:.3} acc={:.1}% ({}/{}) ({:.1}s)",
                step + 1, result.loss, acc * 100.0,
                result.correct, result.total,
                t0.elapsed().as_secs_f64());
        }
    }

    // Final eval on full text
    let final_result = train_step(
        &mut weights, &embed_table, &sensory, &mut output_proj,
        &tokens, &train_cfg,
    );
    let final_acc = final_result.correct as f32 / final_result.total.max(1) as f32;
    eprintln!("\n  Final: loss={:.3} acc={:.1}% ({}/{})",
        final_result.loss, final_acc * 100.0,
        final_result.correct, final_result.total);
    eprintln!("  Random baseline: {:.1}% (1/256)", 100.0 / 256.0);
    eprintln!("  Unigram baseline: ~15% (common bytes)\n");
}
