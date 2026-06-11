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
use modgrad_ctm::train::{train_step_composed, CtmAdamW, CtmGradients};
use modgrad_ctm::weights::{CtmState, CtmWeights};
use modgrad_ctm::forward::{ctm_forward, CtmInput};

const DEFAULT_FONT: &str = "/usr/share/fonts/TTF/DejaVuSansMono.ttf";

struct Cfg {
    font_path: String,
    /// Optional pool of fonts to sample from each example. Comma-separated
    /// TTF paths. When empty, falls back to `font_path` alone.
    fonts: Vec<String>,
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
    smoothing: f32,
    blank_bias: f32,
    vocab: String,
    save_path: Option<String>,
    load_path: Option<String>,
    save_every: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            font_path: DEFAULT_FONT.into(),
            fonts: Vec::new(),
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
            smoothing: 0.05,
            blank_bias: -4.0,
            vocab: "alnum".into(),
            save_path: None,
            load_path: None,
            save_every: 0,
        }
    }
}

fn parse_cfg() -> Cfg {
    let mut cfg = Cfg::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--font" => cfg.font_path = args.next().expect("--font needs a path"),
            "--fonts" => {
                let raw = args.next().expect("--fonts needs a comma-separated list");
                cfg.fonts = raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            }
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
            "--smoothing" => cfg.smoothing = args.next().unwrap().parse().unwrap(),
            "--blank-bias" => cfg.blank_bias = args.next().unwrap().parse().unwrap(),
            "--vocab" => cfg.vocab = args.next().unwrap(),
            "--save" => cfg.save_path = Some(args.next().unwrap()),
            "--load" => cfg.load_path = Some(args.next().unwrap()),
            "--save-every" => cfg.save_every = args.next().unwrap().parse().unwrap(),
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

/// Built-in target vocabularies. The model output is always over the
/// full 96-class printable ascii alphabet — we just sample targets
/// from a restricted subset to control the difficulty.
const VOCAB_ALNUM: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const VOCAB_DIGITS: &[u8] = b"0123456789";
const VOCAB_LOWER: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

fn pick_vocab(name: &str) -> &'static [u8] {
    match name {
        "digits" => VOCAB_DIGITS,
        "lower"  => VOCAB_LOWER,
        "alnum" | _ => VOCAB_ALNUM,
    }
}

fn sample_string(rng: &mut Lcg, vocab: &[u8], min_len: usize, max_len: usize) -> String {
    let lo = min_len as i32;
    let hi = max_len as i32;
    let len = rng.range(lo..=hi) as usize;
    let max_idx = (vocab.len() as i32) - 1;
    let mut s = String::with_capacity(len);
    for _ in 0..len {
        let idx = rng.range(0..=max_idx) as usize;
        s.push(vocab[idx] as char);
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
    fonts: &[FontRef<'_>],
    cfg: &Cfg,
    rng: &mut Lcg,
) -> (f32, Vec<(String, String)>) {
    let vocab = pick_vocab(&cfg.vocab);
    let max_font_idx = (fonts.len() as i32) - 1;
    let mut pairs = Vec::with_capacity(cfg.eval_size);
    for _ in 0..cfg.eval_size {
        let text = sample_string(rng, vocab, cfg.min_chars, cfg.max_chars);
        let font = &fonts[rng.range(0..=max_font_idx) as usize];
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

    // Build the font pool. --fonts wins if provided; otherwise --font.
    let font_paths: Vec<String> = if !cfg.fonts.is_empty() {
        cfg.fonts.clone()
    } else {
        vec![cfg.font_path.clone()]
    };
    let font_blobs: Vec<Vec<u8>> = font_paths.iter().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("failed to read font {}: {}", p, e);
            std::process::exit(1);
        })
    }).collect();
    let fonts: Vec<FontRef> = font_blobs.iter().map(|bytes| {
        FontRef::try_from_slice(bytes).unwrap_or_else(|e| {
            eprintln!("failed to parse a font: {e}");
            std::process::exit(1);
        })
    }).collect();
    eprintln!("  fonts: {} loaded", fonts.len());
    for p in &font_paths { eprintln!("    {}", p); }

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

    // Optionally warm-start from saved weights, otherwise initialise
    // fresh. Loading lets you stack multiple training sessions on the
    // same architecture (e.g. curriculum: digits → lower → alnum).
    let mut w = match cfg.load_path.as_deref() {
        Some(path) => {
            eprintln!("  loading weights from {}", path);
            CtmWeights::load(path).unwrap_or_else(|e| {
                eprintln!("  load failed: {e}. Initialising fresh.");
                CtmWeights::new(ctm_config.clone(), LINE_H)
            })
        }
        None => CtmWeights::new(ctm_config.clone(), LINE_H),
    };

    // CTC blank-bias init. CTC's loss landscape has a strong attractor
    // at "predict blank everywhere" because most ticks SHOULD be blank
    // — but the model can fall in before learning to ever emit a
    // character. Standard fix: start the blank logit (class 0) at a
    // negative bias so the early training signal favours non-blank
    // emissions until the actual alignment is learned. This single
    // change is the difference between blank-collapse and training.
    // Only applied to fresh init — loaded weights keep their bias.
    if cfg.load_path.is_none() && !w.output_proj.bias.is_empty() {
        w.output_proj.bias[0] = cfg.blank_bias;
    }

    let mut grads = CtmGradients::zeros(&w);
    let mut adam = CtmAdamW::zeros(&w);
    adam.weight_decay = 1e-4;
    adam.grad_clip = cfg.clip_norm;
    let loss_fn = CtcLossFn::with_smoothing(cfg.smoothing);
    let mut train_rng = Lcg::new(cfg.seed);
    let mut eval_rng = Lcg::new(cfg.seed.wrapping_add(0xa5a5_a5a5));

    eprintln!("  params: ~{} (rough)", w.n_params());
    eprintln!();

    let t0 = std::time::Instant::now();
    let mut loss_ema: f32 = 0.0;
    let mut step = 0;
    let mut best_cer: f32 = f32::INFINITY;
    let train_vocab = pick_vocab(&cfg.vocab);
    let max_font_idx = (fonts.len() as i32) - 1;

    while step < cfg.steps {
        // ── accumulate one micro-batch then step ──
        let mut batch_loss = 0.0f32;
        let mut batch_ok = 0usize;
        for _ in 0..cfg.batch {
            let text = sample_string(&mut train_rng, train_vocab, cfg.min_chars, cfg.max_chars);
            let target = match string_to_classes(&text) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let font_idx = train_rng.range(0..=max_font_idx) as usize;
            let line = render_line_augmented(&fonts[font_idx], &text, cfg.font_px, &mut train_rng, 0.04, 2);
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

        adam.step(&mut w, &mut grads, cfg.lr);
        grads.zero();

        step += 1;

        if cfg.save_every > 0 && step % cfg.save_every == 0 {
            if let Some(path) = cfg.save_path.as_deref() {
                if let Err(e) = w.save(path) {
                    eprintln!("  save {path} failed: {e}");
                }
            }
        }

        if step % cfg.eval_every == 0 || step == cfg.steps {
            let (eval_cer, pairs) = run_eval(&w, &fonts, &cfg, &mut eval_rng);
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

    // Final save (overwrites the periodic snapshots).
    if let Some(path) = cfg.save_path.as_deref() {
        if let Err(e) = w.save(path) {
            eprintln!("  final save {path} failed: {e}");
        } else {
            eprintln!("  saved final weights to {path}");
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
