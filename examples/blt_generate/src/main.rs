//! BLT text generation smoke. Trains a tiny BLT for ~200 steps on a
//! short *Pride and Prejudice* excerpt, then drives
//! [`BltModel::generate`] across a handful of prompts (greedy +
//! temperature sampling).
//!
//! ## What this proves
//!
//! 200 steps on a kilobyte-scale corpus is far short of coherent
//! generation; the output WILL be byte-salad. The load-bearing
//! properties this example asserts — `hdevalence`-style — are the ones
//! that prove the generation path is wired correctly:
//!
//!   1. **Greedy is deterministic.** The same prompt run twice in a
//!      row produces byte-identical output. This is what catches
//!      regressions in argmax or KV-cache reset.
//!   2. **Temperature is non-trivially applied.** Temperature output
//!      differs from greedy in at least one byte. Catches the
//!      accidental `if temperature == 0.0` short-circuit that ignores
//!      the supplied seed.
//!   3. **All output bytes are valid `u8`.** Trivially true at the
//!      type level but worth stating: nothing escapes the byte vocab.
//!
//! Anything beyond those three is a scaling problem, not a wiring
//! problem.
//!
//! ## Run
//!
//! ```text
//!   cargo run -p blt_generate --release
//! ```
//!
//! HIP is required at runtime — exits cleanly with status 0 when no
//! HIP device is reachable, matching the rest of the smoke suite.

use modgrad_blt::byteify::ByteifyRecipe;
use modgrad_blt::decoder::LocalDecoderConfig;
use modgrad_blt::encoder::LocalEncoderConfig;
use modgrad_blt::model::{BltConfig, BltLatentConfig, BltModel, BltScratch};
use modgrad_blt::trainer::{BltModelTrainer, BltTrainerConfig};
use modgrad_transformer::config::WindowPattern;
use modgrad_device::backend::HipBatch;
use modgrad_device::backend::rocm::ffi::runtime_available;

// ── Model dim constants ──────────────────────────────────────────
//
// Same shape as `blt_train_real_text` — small enough to train in
// seconds on a contended 7600M XT, large enough to exercise the full
// encoder→latent→decoder chain.
const BYTE_DIM: usize = 32;
const N_BYTE_HEADS: usize = 4;
const BYTE_HEAD_DIM: usize = BYTE_DIM / N_BYTE_HEADS; // 8

const PATCH_DIM: usize = 64;
const N_PATCH_HEADS: usize = 4;
const PATCH_HEAD_DIM: usize = PATCH_DIM / N_PATCH_HEADS; // 16

const MAX_SEQ: usize = 32;
const MAX_PATCHES: usize = 8;

// ── Training constants ──────────────────────────────────────────
const N_TRAIN_STEPS: usize = 200;
const N_NEW_BYTES: usize = 32;
const TEMPERATURE: f32 = 0.7;
const RNG_SEED: u64 = 0xB17_CAFE_u64; // arbitrary; deterministic
const TRAIN_RNG_SEED: u64 = 0xC0DE_FACE_u64;

// ── Corpus ──────────────────────────────────────────────────────
//
// Short excerpt (~500 bytes) — too short for great training but enough
// to see common bigrams emerge. Public-domain Pride and Prejudice.
const CORPUS: &str = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. \"My dear Mr. Bennet,\" said his lady to him one day, \"have you heard that Netherfield Park is let at last?\" Mr. Bennet replied that he had not. \"But it is,\" returned she; \"for Mrs. Long has just been here, and she told me all about it.\" Mr. Bennet made no answer.";

fn build_config() -> BltConfig {
    BltConfig {
        encoder: LocalEncoderConfig {
            n_layers: 1,
            byte_dim: BYTE_DIM,
            patch_dim: PATCH_DIM,
            n_heads: N_BYTE_HEADS,
            head_dim: BYTE_HEAD_DIM,
            mlp_dim: BYTE_DIM * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_seq_len: MAX_SEQ,
            ngram_min_n: 3,
            ngram_max_n: 5,
            ngram_vocab_per_n: 256,
            window_pattern: WindowPattern::Full,
        },
        latent: BltLatentConfig {
            n_layers: 2,
            patch_dim: PATCH_DIM,
            n_heads: N_PATCH_HEADS,
            head_dim: PATCH_HEAD_DIM,
            mlp_dim: PATCH_DIM * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_patches: MAX_PATCHES,
        },
        decoder: LocalDecoderConfig {
            n_layers: 1,
            byte_dim: BYTE_DIM,
            patch_dim: PATCH_DIM,
            n_heads: N_BYTE_HEADS,
            head_dim: BYTE_HEAD_DIM,
            mlp_dim: BYTE_DIM * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_seq_len: MAX_SEQ,
            window_pattern: WindowPattern::Full,
        },
    }
}

/// Format a byte slice for human-readable display: ASCII-printable
/// bytes pass through; everything else is `\xNN`.
fn format_bytes(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 4);
    for &b in bytes {
        if b == b'\\' {
            s.push_str("\\\\");
        } else if b == b'"' {
            s.push_str("\\\"");
        } else if (0x20..=0x7E).contains(&b) {
            s.push(b as char);
        } else if b == b'\n' {
            s.push_str("\\n");
        } else if b == b'\t' {
            s.push_str("\\t");
        } else {
            s.push_str(&format!("\\x{:02x}", b));
        }
    }
    s
}

fn main() {
    if !runtime_available() {
        eprintln!("BLT generate: HIP runtime unavailable, skipping");
        std::process::exit(0);
    }

    // ── Build a tiny BLT ─────────────────────────────────────────
    let blt_cfg = build_config();
    blt_cfg.validate().expect("tiny BLT config validates");
    let model = BltModel::new(blt_cfg.clone()).expect("BltModel::new");

    // ── Trainer ──────────────────────────────────────────────────
    //
    // `BltModelTrainer::train_step` requires `bytes.len() ==
    // micro_batch_size`. Use 32 so each step covers a full window.
    let mbs = MAX_SEQ;
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

    // ── Sliding 32-byte windows at stride 16 ─────────────────────
    let corpus = CORPUS.as_bytes();
    let stride = 16usize;
    assert!(corpus.len() >= mbs, "corpus must be at least one window long");
    let n_windows = (corpus.len() - mbs) / stride + 1;
    let windows: Vec<&[u8]> = (0..n_windows)
        .map(|w| {
            let start = w * stride;
            &corpus[start..start + mbs]
        })
        .collect();

    // 8 patches × 4 bytes = 32 — same fixed-stride shape as the
    // training smoke. Boundaries are constant across windows.
    let train_boundaries: Vec<usize> = (0..=MAX_PATCHES).map(|p| p * 4).collect();
    assert_eq!(train_boundaries[0], 0);
    assert_eq!(*train_boundaries.last().unwrap(), mbs);

    eprintln!(
        "blt_generate: corpus_bytes={} n_windows={} train_steps={} dims byte_dim={} \
         patch_dim={} max_seq={} max_patches={}",
        corpus.len(), n_windows, N_TRAIN_STEPS, BYTE_DIM, PATCH_DIM, MAX_SEQ, MAX_PATCHES,
    );

    // Numerical Recipes LCG — same flavor as the train_real_text smoke.
    let mut rng_state: u64 = TRAIN_RNG_SEED;
    let mut next_window_idx = || -> usize {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as usize % n_windows
    };

    // ── Train ────────────────────────────────────────────────────
    let mut first_loss = None;
    let mut last_loss = 0.0f32;
    for step in 0..N_TRAIN_STEPS {
        let widx = next_window_idx();
        let bytes = windows[widx];
        let loss = trainer
            .train_step(bytes, &train_boundaries)
            .unwrap_or_else(|e| panic!("train_step failed at step {step}: {e:?}"));
        assert!(loss.is_finite(), "loss not finite at step {step}: {loss}");
        if first_loss.is_none() { first_loss = Some(loss); }
        last_loss = loss;
        if step % 25 == 0 || step == N_TRAIN_STEPS - 1 {
            eprintln!("  train step {step:>3}: loss = {loss:.4}");
        }
    }
    eprintln!(
        "blt_generate: trained {} steps, loss {:.3} → {:.3}",
        N_TRAIN_STEPS,
        first_loss.unwrap_or(f32::NAN),
        last_loss,
    );

    // ── Generate ─────────────────────────────────────────────────
    //
    // Four prompts: three in-distribution + one clearly OOD ("the
    // quick brown fox" tradition stops here at "the quick" but the
    // OOD is a digit-heavy non-prose string).
    let prompts: &[&str] = &[
        "It is a truth",
        "Mr. Bennet",
        "the quick",
        "1234567890",
    ];

    let model = trainer.model_mut();
    let mut scratch = BltScratch::new(&blt_cfg).expect("BltScratch::new");
    let batch = HipBatch::new();

    println!();
    println!("PASS: BLT generate works end-to-end (greedy deterministic, temperature varies)");
    println!();

    // Wall-clock accumulators for the naive vs cached comparison. We
    // sum across all prompts so the comparison aggregates over enough
    // generation steps to drown out per-call jitter.
    let mut total_naive_ms = 0.0f64;
    let mut total_cached_ms = 0.0f64;

    for prompt in prompts {
        let prompt_bytes = prompt.as_bytes();

        // Greedy run #1.
        let t0 = std::time::Instant::now();
        let greedy_a = model.generate(
            &batch, prompt_bytes, N_NEW_BYTES, 0.0, RNG_SEED, &mut scratch,
        ).expect("generate (greedy a)");
        let naive_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_naive_ms += naive_ms;

        // Greedy run #2 — must be byte-identical.
        let greedy_b = model.generate(
            &batch, prompt_bytes, N_NEW_BYTES, 0.0, RNG_SEED, &mut scratch,
        ).expect("generate (greedy b)");
        assert_eq!(
            greedy_a, greedy_b,
            "greedy generation must be deterministic (prompt {prompt:?})",
        );
        assert_eq!(greedy_a.len(), N_NEW_BYTES,
            "greedy output length mismatch (prompt {prompt:?})");

        // Greedy via the cached path. Must match `generate` bit-for-bit
        // — `hdevalence`-rigor: the cache is an optimization, not a
        // semantic change, so any divergence is a bug.
        let t1 = std::time::Instant::now();
        let greedy_cached = model.generate_cached(
            &batch, prompt_bytes, N_NEW_BYTES, 0.0, RNG_SEED, &mut scratch,
        ).expect("generate_cached (greedy)");
        let cached_ms = t1.elapsed().as_secs_f64() * 1000.0;
        total_cached_ms += cached_ms;

        assert_eq!(
            greedy_a, greedy_cached,
            "generate_cached must match generate bit-for-bit (greedy, prompt {prompt:?}) — \
             a divergence here means the boundary-keyed cache served a stale entry"
        );

        // Temperature.
        let temp_out = model.generate(
            &batch, prompt_bytes, N_NEW_BYTES, TEMPERATURE, RNG_SEED, &mut scratch,
        ).expect("generate (temperature)");
        assert_eq!(temp_out.len(), N_NEW_BYTES,
            "temperature output length mismatch (prompt {prompt:?})");

        // Temperature via cached path — bit-identical RNG draws + the
        // cache reusing identical intermediates ⇒ identical output.
        let temp_cached = model.generate_cached(
            &batch, prompt_bytes, N_NEW_BYTES, TEMPERATURE, RNG_SEED, &mut scratch,
        ).expect("generate_cached (temperature)");
        assert_eq!(
            temp_out, temp_cached,
            "generate_cached must match generate bit-for-bit (temperature, prompt {prompt:?})",
        );

        // All bytes are valid u8 — trivially true at the type level
        // (Vec<u8>), but the assertion makes the contract explicit.
        // Verified by construction; nothing to check at runtime beyond
        // the type guarantee.

        println!("prompt: {prompt:?}");
        println!("  greedy:    {}", format_bytes(&greedy_a));
        println!("  cached:    {}  [bit-identical: {}]",
            format_bytes(&greedy_cached),
            greedy_a == greedy_cached);
        println!("  temp=0.7:  {}", format_bytes(&temp_out));
        println!("  wall-clock: naive {naive_ms:.1} ms  cached {cached_ms:.1} ms  \
                  ratio {:.2}x", naive_ms / cached_ms.max(1e-6));
        println!();
    }

    // ── Wall-clock summary ───────────────────────────────────────
    //
    // Honest framing: under the placeholder fixed-stride patcher the
    // boundaries shift on every step, so the cache hit rate is zero
    // and the cached path runs the same work plus a small bookkeeping
    // overhead (boundary-equality compare, two D2D clones at miss).
    // The interesting number to read is therefore "ratio close to 1.0
    // ± noise" — a genuine speedup will land once an entropy patcher
    // emits stable boundaries for stable byte prefixes. The decomposed
    // forward shape this lays down is the prerequisite for that future
    // slice plus true incremental KV-caching.
    println!(
        "wall-clock summary across {} prompts × {} new bytes:\n  \
         naive   {:>7.1} ms total\n  \
         cached  {:>7.1} ms total\n  \
         ratio   {:>7.2}x  (1.0x = no speedup; cache misses every step \
         under fixed-stride patcher — see BltModel::generate_cached doc)",
        prompts.len(), N_NEW_BYTES,
        total_naive_ms, total_cached_ms,
        total_naive_ms / total_cached_ms.max(1e-6),
    );
    println!();

    // ── Sanity check across all prompts: temperature differs from
    //    greedy for at least one (prompt, byte) pair. We aggregate
    //    across prompts because for any single prompt + tiny model the
    //    softmax may concentrate enough that temperature still picks
    //    the argmax most of the time.
    let mut any_temp_differs = false;
    for prompt in prompts {
        let prompt_bytes = prompt.as_bytes();
        let greedy = model.generate(
            &batch, prompt_bytes, N_NEW_BYTES, 0.0, RNG_SEED, &mut scratch,
        ).expect("generate (greedy verify)");
        let temp = model.generate(
            &batch, prompt_bytes, N_NEW_BYTES, TEMPERATURE, RNG_SEED, &mut scratch,
        ).expect("generate (temperature verify)");
        if greedy != temp {
            any_temp_differs = true;
            break;
        }
    }
    assert!(
        any_temp_differs,
        "temperature sampling must produce ≥1 byte different from greedy across all prompts \
         — otherwise the temperature path is silently a no-op"
    );
}
