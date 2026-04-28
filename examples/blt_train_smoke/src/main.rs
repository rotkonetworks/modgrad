//! Proof-of-life: a tiny BLT trains end-to-end on a fixed low-entropy
//! corpus and the loss measurably decreases. Asserts mean of the last
//! 5 step losses is at least 5% lower than the mean of the first 5.
//!
//! HIP is required at runtime — the binary exits cleanly with status 0
//! when no HIP device is reachable, so it stays harmless on CI hosts
//! without a GPU.
//!
//! Run:
//!   cargo run --release -p blt_train_smoke

use modgrad_blt::byteify::ByteifyRecipe;
use modgrad_blt::decoder::LocalDecoderConfig;
use modgrad_blt::encoder::LocalEncoderConfig;
use modgrad_blt::model::{BltConfig, BltLatentConfig, BltModel};
use modgrad_blt::trainer::{BltModelTrainer, BltTrainerConfig};
use modgrad_device::backend::rocm::ffi::runtime_available;

fn main() {
    if !runtime_available() {
        eprintln!("BLT smoke training: HIP runtime unavailable, skipping");
        std::process::exit(0);
    }

    // ── Tiny BLT — same shape as `model::tests::tiny_config` ──────
    //
    // Dimensional cross-checks (must hold for `BltConfig::validate`):
    //   encoder.patch_dim == latent.patch_dim == decoder.patch_dim = 64
    //   encoder.byte_dim  == decoder.byte_dim                       = 32
    //   encoder.n_heads * encoder.head_dim == byte_dim   (4 *  8 = 32)
    //   latent.n_heads  * latent.head_dim  == patch_dim  (4 * 16 = 64)
    //   decoder.n_heads * decoder.head_dim == byte_dim   (4 *  8 = 32)
    //   max_patches = n_patches  (LocalDecoder backward asserts equality)
    let byte_dim = 32usize;
    let n_byte_heads = 4usize;
    let byte_head_dim = byte_dim / n_byte_heads; // 8
    let patch_dim = 64usize;
    let n_patch_heads = 4usize;
    let patch_head_dim = patch_dim / n_patch_heads; // 16
    let max_seq = 32usize;
    let max_patches = 8usize;

    let blt_cfg = BltConfig {
        encoder: LocalEncoderConfig {
            n_layers: 1,
            byte_dim,
            patch_dim,
            n_heads: n_byte_heads,
            head_dim: byte_head_dim,
            mlp_dim: byte_dim * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_seq_len: max_seq,
            ngram_min_n: 3,
            ngram_max_n: 5,
            ngram_vocab_per_n: 256,
        },
        latent: BltLatentConfig {
            n_layers: 2,
            patch_dim,
            n_heads: n_patch_heads,
            head_dim: patch_head_dim,
            mlp_dim: patch_dim * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_patches,
        },
        decoder: LocalDecoderConfig {
            n_layers: 1,
            byte_dim,
            patch_dim,
            n_heads: n_byte_heads,
            head_dim: byte_head_dim,
            mlp_dim: byte_dim * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_seq_len: max_seq,
        },
    };
    blt_cfg.validate().expect("tiny BLT config validates");

    let model = BltModel::new(blt_cfg).expect("BltModel::new");

    // ── Trainer ──────────────────────────────────────────────────
    //
    // `BltModelTrainer::train_step` requires `bytes.len() == micro_batch_size`
    // (per train_step's contract — last position has no target, first n-1
    // contribute to CE). Override the default 16 to 32 so each step
    // covers a full window.
    let mbs = max_seq;
    let trainer_cfg = BltTrainerConfig {
        micro_batch_size: mbs,
        seq_len: mbs,
        ..BltTrainerConfig::default()
    };
    let mut trainer = BltModelTrainer::new(
        model,
        trainer_cfg,
        ByteifyRecipe::global_predicate(),
    )
    .expect("BltModelTrainer::new");

    // ── Fixed low-entropy corpus ─────────────────────────────────
    //
    // 45-byte seed; an 8-window slide at stride 1 picks distinct
    // training spans inside it. Tiny model + tight corpus is the only
    // way to see meaningful learning signal in 50 steps.
    let pattern: &[u8] = b"the quick brown fox jumps over the lazy dog. ";
    debug_assert!(pattern.len() >= mbs + 7);
    let n_windows = 8usize;
    let windows: Vec<Vec<u8>> = (0..n_windows)
        .map(|w| pattern[w..w + mbs].to_vec())
        .collect();
    for w in &windows {
        assert_eq!(w.len(), mbs, "every window must be {mbs} bytes");
    }

    // 8 patches of 4 bytes — fits max_patches=8 exactly.
    let boundaries: Vec<usize> = (0..=8).map(|p| p * 4).collect();
    assert_eq!(boundaries[0], 0);
    assert_eq!(*boundaries.last().unwrap(), mbs);

    eprintln!(
        "blt_train_smoke: model dims byte_dim={byte_dim} patch_dim={patch_dim} \
         lE={} lL={} lD={}, mbs={mbs} n_windows={n_windows}",
        1, 2, 1,
    );

    // ── Loop ─────────────────────────────────────────────────────
    let n_steps = 50usize;
    let mut losses: Vec<f32> = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let bytes = &windows[step % n_windows];
        let loss = trainer
            .train_step(bytes, &boundaries)
            .unwrap_or_else(|e| panic!("train_step failed at step {step}: {e:?}"));
        assert!(loss.is_finite(), "loss not finite at step {step}: {loss}");
        losses.push(loss);
        if step % 10 == 0 || step == n_steps - 1 {
            eprintln!("step {step:>3}: loss = {loss:.4}");
        }
    }

    // ── Assertion ────────────────────────────────────────────────
    //
    // Mean of last 5 must be at least 5% below mean of first 5.
    // Loose enough for stochastic noise; tight enough that a backward
    // chain that doesn't produce useful gradients fails it.
    let mean = |xs: &[f32]| xs.iter().copied().sum::<f32>() / xs.len() as f32;
    let mean_first = mean(&losses[..5]);
    let mean_last = mean(&losses[losses.len() - 5..]);
    let threshold = mean_first * 0.95;

    eprintln!(
        "blt_train_smoke: mean(first 5) = {mean_first:.4}, \
         mean(last 5) = {mean_last:.4}, threshold = {threshold:.4}"
    );

    if mean_last < threshold {
        println!(
            "PASS: BLT trains end-to-end (mean loss {mean_first:.4} → {mean_last:.4})"
        );
    } else {
        eprintln!("FAIL: full loss history:");
        for (i, l) in losses.iter().enumerate() {
            eprintln!("  step {i:>3}: {l:.4}");
        }
        panic!("BLT smoke training: loss did not decrease — backward chain may be broken");
    }
}
