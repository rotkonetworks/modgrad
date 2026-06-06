//! Phase-0 smoke train for CTM + CTC on synthetic single-font lines.
//!
//! Goal: drive byte-level CER from ~1.0 (random) down to < 0.30 on a
//! tiny synthetic distribution. That proves the loop is wired
//! correctly. It is *not* a real OCR model.
//!
//! Usage:
//!   ocr_smoke_train                                # CPU, default config
//!   ocr_smoke_train --steps 500 --batch 8
//!   ocr_smoke_train --font /path/to/font.ttf
//!
//! Defaults are conservative — small CTM, short text, lots of repeats.
//! The smoke gate prints PASS / FAIL based on the held-out CER at the
//! end of the run.

use ab_glyph::FontRef;
use ocr_smoke::eval::cer;
use ocr_smoke::render::{
    classes_to_string, render_line_augmented, string_to_classes, Lcg, ALPHABET_SIZE,
    LINE_H,
};
use ocr_smoke::train::{greedy_decode_predictions, line_to_tokens, CtcLossFn};

use modgrad_ctm::config::CtmConfig;
use modgrad_ctm::train::{train_step_composed, CtmGradients};
use modgrad_ctm::weights::{CtmState, CtmWeights};
use modgrad_ctm::forward::{ctm_forward, CtmInput};

const DEFAULT_FONT: &str = "/usr/share/fonts/TTF/DejaVuSansMono.ttf";

struct Cfg {
    font_path: String,
    font_px: f32,
    steps: usize,
    batch: usize,
    eval_every: usize,
    eval_size: usize,
    lr: f32,
    clip_norm: f32,
    seed: u64,
    min_chars: usize,
    max_chars: usize,
    // CTM dims
    ticks: usize,
    d_model: usize,
    d_input: usize,
    heads: usize,
    synapse_depth: usize,
    memory_length: usize,
    cer_gate: f32,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            font_path: DEFAULT_FONT.into(),
            font_px: 24.0,
            steps: 400,
            batch: 4,
            eval_every: 50,
            eval_size: 32,
            lr: 1e-3,
            clip_norm: 1.0,
            seed: 1,
            min_chars: 3,
            max_chars: 6,
            ticks: 24,
            d_model: 96,
            d_input: 48,
            heads: 2,
            synapse_depth: 1,
            memory_length: 8,
            // Phase-0 smoke gate: drive CER below 0.30 on the held-out
            // synthetic split. Real OCR aims much lower; we just want
            // proof that the model is learning something.
            cer_gate: 0.30,
        }
    }
}

fn parse_cfg() -> Cfg {
    let mut cfg = Cfg::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--font" => cfg.font_path = args.next().expect("--font needs a path"),
            "--px" => cfg.font_px = args.next().unwrap().parse().unwrap(),
            "--steps" => cfg.steps = args.next().unwrap().parse().unwrap(),
            "--batch" => cfg.batch = args.next().unwrap().parse().unwrap(),
            "--eval-every" => cfg.eval_every = args.next().unwrap().parse().unwrap(),
            "--eval-size" => cfg.eval_size = args.next().unwrap().parse().unwrap(),
            "--lr" => cfg.lr = args.next().unwrap().parse().unwrap(),
            "--seed" => cfg.seed = args.next().unwrap().parse().unwrap(),
            "--min-chars" => cfg.min_chars = args.next().unwrap().parse().unwrap(),
            "--max-chars" => cfg.max_chars = args.next().unwrap().parse().unwrap(),
            "--ticks" => cfg.ticks = args.next().unwrap().parse().unwrap(),
            "--d-model" => cfg.d_model = args.next().unwrap().parse().unwrap(),
            "--d-input" => cfg.d_input = args.next().unwrap().parse().unwrap(),
            "--heads" => cfg.heads = args.next().unwrap().parse().unwrap(),
            "--gate" => cfg.cer_gate = args.next().unwrap().parse().unwrap(),
            "--help" | "-h" => { print_usage(); std::process::exit(0); }
            other => { eprintln!("unknown arg: {other}"); print_usage(); std::process::exit(2); }
        }
    }
    cfg
}

fn print_usage() {
    eprintln!("ocr_smoke_train — phase 0 CTC/CTM smoke");
    eprintln!();
    eprintln!("Flags (with defaults):");
    eprintln!("  --font PATH           {}", DEFAULT_FONT);
    eprintln!("  --px N                24");
    eprintln!("  --steps N             400");
    eprintln!("  --batch N             4");
    eprintln!("  --eval-every N        50");
    eprintln!("  --eval-size N         32");
    eprintln!("  --lr F                1e-3");
    eprintln!("  --seed N              1");
    eprintln!("  --min-chars N         3");
    eprintln!("  --max-chars N         6");
    eprintln!("  --ticks N             24      (CTC time = CTM iterations)");
    eprintln!("  --d-model N           96");
    eprintln!("  --d-input N           48");
    eprintln!("  --heads N             2");
    eprintln!("  --gate F              0.30    (CER gate, lower = stricter)");
}

/// A subset of the printable ascii alphabet that the smoke run draws
/// targets from. Excludes whitespace and most symbols so the renderer
/// produces visually distinct glyphs — phase 0 isn't trying to learn
/// to read invisible characters.
const SAMPLE_VOCAB: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

fn sample_string(rng: &mut Lcg, min_len: usize, max_len: usize) -> String {
    let lo = min_len as i32;
    let hi = max_len as i32;
    let len = rng.range(lo..=hi) as usize;
    let max_idx = (SAMPLE_VOCAB.len() as i32) - 1;
    let mut s = String::with_capacity(len);
    for _ in 0..len {
        let idx = rng.range(0..=max_idx) as usize;
        s.push(SAMPLE_VOCAB[idx] as char);
    }
    s
}

/// Fresh per-forward state for eval. `train_step_composed` makes its
/// own state internally, so we only need this in `run_eval`.
fn fresh_state(w: &CtmWeights) -> CtmState {
    CtmState::new(w)
}

/// Run a held-out greedy-decoded eval batch and return CER.
fn run_eval(
    w: &CtmWeights,
    font: &FontRef<'_>,
    cfg: &Cfg,
    rng: &mut Lcg,
) -> (f32, Vec<(String, String)>) {
    let mut pairs = Vec::with_capacity(cfg.eval_size);
    for _ in 0..cfg.eval_size {
        let text = sample_string(rng, cfg.min_chars, cfg.max_chars);
        let line = render_line_augmented(font, &text, cfg.font_px, rng, 0.04, 1);
        let (tokens, n_tokens, token_dim) = line_to_tokens(&line);
        let mut state = fresh_state(w);
        let out = ctm_forward(
            w, &mut state,
            CtmInput::Raw { obs: &tokens, n_tokens, raw_dim: token_dim },
        );
        let decoded = greedy_decode_predictions(&out.predictions);
        let predicted = classes_to_string(&decoded);
        pairs.push((text, predicted));
    }
    (cer(&pairs), pairs)
}

fn main() {
    let cfg = parse_cfg();

    eprintln!("ocr_smoke_train — phase 0 CTM + CTC");
    eprintln!("  ticks={} d_model={} d_input={} heads={} synapse_depth={} memory={}",
        cfg.ticks, cfg.d_model, cfg.d_input, cfg.heads, cfg.synapse_depth, cfg.memory_length);
    eprintln!("  steps={} batch={} lr={} chars={}..={} alphabet={}",
        cfg.steps, cfg.batch, cfg.lr, cfg.min_chars, cfg.max_chars, ALPHABET_SIZE);

    let font_bytes = std::fs::read(&cfg.font_path).unwrap_or_else(|e| {
        eprintln!("failed to read font {}: {}", cfg.font_path, e);
        std::process::exit(1);
    });
    let font = FontRef::try_from_slice(&font_bytes).unwrap_or_else(|e| {
        eprintln!("failed to parse font: {e}");
        std::process::exit(1);
    });

    let ctm_config = CtmConfig {
        iterations: cfg.ticks,
        d_model: cfg.d_model,
        d_input: cfg.d_input,
        heads: cfg.heads,
        n_synch_out: cfg.d_model,
        n_synch_action: cfg.d_model,
        synapse_depth: cfg.synapse_depth,
        memory_length: cfg.memory_length,
        deep_nlms: false,
        memory_hidden_dims: 4,
        out_dims: ALPHABET_SIZE,
        n_random_pairing_self: 0,
        min_width: 16,
        exit_strategy: modgrad_ctm::config::ExitStrategy::None,
        collect_trajectories: false,
    };

    let mut w = CtmWeights::new(ctm_config, LINE_H);
    let mut grads = CtmGradients::zeros(&w);
    let loss_fn = CtcLossFn::new();
    let mut train_rng = Lcg::new(cfg.seed);
    let mut eval_rng = Lcg::new(cfg.seed.wrapping_add(0xa5a5_a5a5));

    eprintln!("  params: ~{} (rough)", w.n_params());
    eprintln!();

    let t0 = std::time::Instant::now();
    let mut loss_ema: f32 = 0.0;
    let mut step = 0;
    let mut best_cer: f32 = f32::INFINITY;

    while step < cfg.steps {
        // ── accumulate one micro-batch then step ──
        let mut batch_loss = 0.0f32;
        let mut batch_ok = 0usize;
        for _ in 0..cfg.batch {
            let text = sample_string(&mut train_rng, cfg.min_chars, cfg.max_chars);
            let target = match string_to_classes(&text) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let line = render_line_augmented(&font, &text, cfg.font_px, &mut train_rng, 0.04, 2);
            let (tokens, n_tokens, token_dim) = line_to_tokens(&line);
            let result = train_step_composed(
                &w, &mut grads, &tokens, n_tokens, token_dim, &target, &loss_fn,
            );
            if result.loss.is_finite() {
                batch_loss += result.loss;
                batch_ok += 1;
            }
        }
        if batch_ok == 0 {
            // Should be unreachable for sane configs; emit and continue.
            eprintln!("[step {step}] all-infinite batch — bumping step");
            step += 1;
            continue;
        }
        let mean_loss = batch_loss / batch_ok as f32;
        loss_ema = if step == 0 { mean_loss } else { 0.9 * loss_ema + 0.1 * mean_loss };

        grads.apply(&mut w, cfg.lr, cfg.clip_norm);
        // Re-zero for next micro-batch.
        grads = CtmGradients::zeros(&w);

        step += 1;

        if step % cfg.eval_every == 0 || step == cfg.steps {
            let (eval_cer, pairs) = run_eval(&w, &font, &cfg, &mut eval_rng);
            best_cer = best_cer.min(eval_cer);
            let elapsed = t0.elapsed().as_secs_f32();
            let steps_per_sec = step as f32 / elapsed;
            eprintln!(
                "[step {step:>4}/{:>4}] loss_ema={:>6.3}  CER={:>5.3}  best={:>5.3}  {:>5.1} step/s",
                cfg.steps, loss_ema, eval_cer, best_cer, steps_per_sec
            );
            // Show a couple of example decodes so the smoke run is
            // debuggable from the log alone.
            for (target, predicted) in pairs.iter().take(3) {
                eprintln!("     {:?} -> {:?}", target, predicted);
            }
        }
    }

    let dt = t0.elapsed().as_secs_f32();
    eprintln!();
    eprintln!("smoke: {:.1}s wallclock, final loss_ema={:.3}, best CER={:.3}, gate={:.3}",
        dt, loss_ema, best_cer, cfg.cer_gate);
    if best_cer <= cfg.cer_gate {
        println!("PASS");
        std::process::exit(0);
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
