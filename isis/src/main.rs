//! isis — 8-region hierarchical CTM runtime built on the modgrad SDK.
//!
//! Commands:
//!   isis train model.bin [--multimodal] [--debug-port 4747]
//!   isis nc model.bin [--audio mic.wav] [--camera frames/] [--debug-port 4747]

use modgrad_ctm::graph::*;
use modgrad_training::trainer::StepHook;
use isis_runtime::nc_socket;

use clap::{Parser, Subcommand};
use std::io::{self, Write};

#[derive(Parser)]
#[command(name = "isis", version, about = "8-region hierarchical CTM — a neural computer")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train on staged byte curriculum (BPTT + AdamW)
    Train {
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        #[arg(short, long)]
        curriculum: Option<String>,
        #[arg(long)]
        multimodal: bool,
        #[arg(long)]
        images: Option<String>,
        #[arg(long)]
        audio: Option<String>,
        #[arg(long)]
        video: Option<String>,
        #[arg(long, default_value = "2.0")]
        video_fps: f32,
        /// Enable GPU dispatch (KFD/CUDA/Vulkan) for linear ops.
        #[arg(long)]
        gpu: bool,
        #[arg(long)]
        debug_port: Option<u16>,
    },
    /// Interactive neural computer
    Nc {
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        #[arg(short, long, default_value = "0.8")]
        temperature: f32,
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        #[arg(long)]
        audio: Option<String>,
        #[arg(long)]
        camera: Option<String>,
        #[arg(long, default_value = "2.0")]
        camera_fps: f32,
        #[arg(long)]
        audio_out: Option<String>,
        #[arg(long)]
        image_out: Option<String>,
        #[arg(long)]
        debug_port: Option<u16>,
    },
    /// Generate text from a trained model
    Generate {
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        /// Prompt text
        #[arg(default_value = "the ")]
        prompt: String,
        #[arg(short, long, default_value = "200")]
        max_tokens: usize,
        #[arg(short, long, default_value = "0.8")]
        temperature: f32,
        /// Frozen cerebellum (.safetensors or .onnx)
        #[arg(long)]
        frozen_cereb: Option<String>,
    },
    /// Run as a headless daemon (NC service on TCP port)
    Daemon {
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        #[arg(short, long, default_value = "4747")]
        port: u16,
    },
    /// Send a command to a running daemon
    Send {
        /// Text to inject
        text: String,
        #[arg(long, default_value = "127.0.0.1:4747")]
        addr: String,
    },
    /// Learn from raw data — no curriculum, no phases, just tokens
    Learn {
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        /// Directory, file(s), or .jsonl with token pairs to learn from.
        #[arg(required = true)]
        data: Vec<String>,
        #[arg(long, default_value = "32")]
        context: usize,
        /// Vocabulary size. 256 = raw bytes, 8192 = VQGAN visual tokens.
        #[arg(long, default_value = "256")]
        vocab: usize,
        /// GPU hybrid: stream weights per-call (PCIe x16).
        #[arg(long)]
        gpu: bool,
        /// GPU VRAM: keep weights in VRAM, zero PCIe during training (PCIe x4).
        #[arg(long)]
        vram: bool,
        /// Medium model (~37M params, d_model=256).
        #[arg(long)]
        medium: bool,
        /// Large model (~81M params, d_model=512).
        #[arg(long)]
        large: bool,
        /// Billion-scale (~223M params, d_model=1024).
        #[arg(long)]
        billion: bool,
        #[arg(long)]
        debug_port: Option<u16>,
        /// Path to frozen ONNX cerebellum model (e.g. backbone.onnx with hidden_states output).
        #[arg(long)]
        frozen_cereb: Option<String>,
    },
    /// Show available compute devices
    Devices,
    /// Train FFN cerebellum (standalone feedforward language model).
    /// This is phase 1 of the architecture: train language storage first,
    /// then freeze and layer CTM on top.
    LearnFfn {
        #[arg(default_value = "cerebellum.bin")]
        checkpoint: String,
        #[arg(required = true)]
        data: Vec<String>,
        #[arg(long, default_value = "64")]
        context: usize,
        #[arg(long, default_value = "256")]
        vocab: usize,
        /// Use GPU for training.
        #[arg(long)]
        gpu: bool,
        /// Small model: ~5M params (sanity check).
        #[arg(long)]
        small: bool,
        /// Medium model: ~50M params.
        #[arg(long)]
        medium: bool,
        /// Large model: ~200M params (default, real language learning).
        #[arg(long)]
        large: bool,
        /// XL model: ~500M params (tight on 8GB VRAM).
        #[arg(long)]
        xl: bool,
        /// Learning rate (defaults by size).
        #[arg(long)]
        lr: Option<f32>,
        /// Stop training after N wall-clock seconds. Intended for the
        /// autoresearch workflow (fixed per-experiment budget).
        #[arg(long)]
        budget: Option<u64>,
        /// Validation file for end-of-run val_bpb. When set, autoresearch
        /// summary block is printed to stderr after training. Kept
        /// optional so non-autoresearch runs skip the eval cost.
        #[arg(long)]
        val_data: Option<String>,
    },
    /// Evaluate a checkpoint — compute val_bpb on held-out data and print
    /// the autoresearch-compatible summary block to stderr. Paired with
    /// `learn-ffn --budget` for the autonomous research workflow.
    Eval {
        #[arg(default_value = "cerebellum.bin")]
        checkpoint: String,
        /// File(s) of validation bytes. Concatenated, truncated to --max-bytes.
        #[arg(required = true)]
        data: Vec<String>,
        /// Context window for forward passes. Must match the training context.
        #[arg(long, default_value = "64")]
        context: usize,
        /// Cap on evaluation bytes — trims wall-clock. 256 KiB is plenty for
        /// a statistically stable val_bpb and keeps eval under a few seconds.
        #[arg(long, default_value = "262144")]
        max_bytes: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train { checkpoint, curriculum, multimodal, images, audio, video, video_fps, gpu, debug_port } => {
            if gpu { modgrad_compute::neuron::enable_gpu(); }
            develop_staged(&checkpoint, curriculum.as_deref(), multimodal,
                images.as_deref(), audio.as_deref(), video.as_deref(), video_fps, debug_port);
        }
        Commands::Nc { checkpoint, temperature, max_tokens, audio, camera, camera_fps, audio_out, image_out, debug_port } => {
            run_nc(&checkpoint, temperature, max_tokens,
                audio.as_deref(), camera.as_deref(), camera_fps,
                audio_out.as_deref(), image_out.as_deref(), debug_port);
        }
        Commands::Generate { checkpoint, prompt, max_tokens, temperature, frozen_cereb } => {
            run_generate(&checkpoint, &prompt, max_tokens, temperature, frozen_cereb.as_deref());
        }
        Commands::Learn { checkpoint, data, context, vocab, gpu, vram, medium, large, billion, debug_port, frozen_cereb } => {
            if gpu || vram {
                modgrad_compute::neuron::enable_gpu();
            }
            if vram {
                // Full VRAM: all activations + weights cached in GPU memory.
                // Zero PCIe in the inner loop.
                let _ = modgrad_compute::backend::set_backend(
                    Box::new(modgrad_compute::backend::VramGpuBackend::new(512))
                );
            } else if gpu {
                // Hybrid/MegaTrain: weights stream through PCIe, GPU for compute.
                let _ = modgrad_compute::backend::set_backend(
                    Box::new(modgrad_compute::backend::HybridGpuBackend::new())
                );
            }
            learn(&checkpoint, &data, context, vocab, medium, large, billion, debug_port, frozen_cereb.as_deref());
        }
        Commands::Daemon { checkpoint, port } => {
            run_daemon(&checkpoint, port);
        }
        Commands::Send { text, addr } => {
            send_command(&text, &addr);
        }
        Commands::Devices => {
            show_devices();
        }
        Commands::LearnFfn { checkpoint, data, context, vocab, gpu, small, medium, large, xl, lr, budget, val_data } => {
            if gpu {
                modgrad_compute::neuron::enable_gpu();
                let _ = modgrad_compute::backend::set_backend(
                    Box::new(modgrad_compute::backend::VramGpuBackend::new(1024))
                );
            }
            learn_ffn(&checkpoint, &data, context, vocab, gpu, small, medium, large, xl, lr, budget, val_data.as_deref());
        }
        Commands::Eval { checkpoint, data, context, max_bytes } => {
            eval_ffn(&checkpoint, &data, context, max_bytes);
        }
    }
}

// ─── FFN Cerebellum Training ──────────────────────────────

fn learn_ffn(
    save_path: &str,
    data_paths: &[String],
    context_len: usize,
    vocab: usize,
    gpu: bool,
    small: bool,
    medium: bool,
    large: bool,
    xl: bool,
    lr_override: Option<f32>,
    budget_secs: Option<u64>,
    val_data_path: Option<&str>,
) {
    use modgrad_ffn::{FfnConfig, FfnWeights, FfnAdamW, FfnGradients, ffn_train_step};

    // Load data
    let mut all_tokens: Vec<usize> = Vec::new();
    for path in data_paths {
        if let Ok(data) = std::fs::read(path) {
            eprintln!("  + {path} ({} bytes)", data.len());
            for &b in &data { all_tokens.push(b as usize); }
        }
    }
    if all_tokens.is_empty() {
        eprintln!("No data found.");
        return;
    }
    eprintln!("Data: {} tokens, vocab: {}", all_tokens.len(), vocab);

    // Model size
    let cfg = if xl {
        FfnConfig::xl(vocab)
    } else if small {
        FfnConfig::small(vocab)
    } else if medium {
        FfnConfig::medium(vocab)
    } else {
        // Default: large (200M params — real language learning)
        let _ = large;
        FfnConfig::large(vocab)
    };

    // Try to resume from a CheckpointBundle. If the file exists but is in
    // the legacy two-file format (cerebellum.bin + cerebellum.bin.opt),
    // fall back to that path — preserves backward compatibility during
    // the rollout.
    use modgrad_training::{CheckpointBundle, BasicMeta};
    const MODEL_KIND: &str = "ffn-cerebellum";

    let (w, opt, resumed_meta) = if std::path::Path::new(save_path).exists() {
        match CheckpointBundle::<FfnWeights, FfnAdamW>::load(save_path, MODEL_KIND) {
            Ok(bundle) => {
                eprintln!("Resumed {save_path} — step {}, {} tokens seen",
                    bundle.meta.step, bundle.meta.tokens_seen);
                (bundle.model, bundle.optimizer, bundle.meta)
            }
            Err(e) => {
                // Probably a legacy two-file checkpoint. Try loading the
                // individual files; on success we'll migrate on the next
                // save.
                eprintln!("  (not a checkpoint bundle: {e})");
                eprintln!("  trying legacy two-file format...");
                let w = FfnWeights::load(save_path).expect("failed to load legacy weights");
                let opt_legacy = save_path.replace(".bin", ".opt.bin")
                                           .replace(".json", ".opt.json");
                let opt = if std::path::Path::new(&opt_legacy).exists() {
                    FfnAdamW::load(&opt_legacy).unwrap_or_else(|_| FfnAdamW::new(&w))
                } else {
                    FfnAdamW::new(&w)
                };
                (w, opt, BasicMeta::default())
            }
        }
    } else {
        eprintln!("Creating FFN cerebellum...");
        let w = FfnWeights::new(cfg);
        let opt = FfnAdamW::new(&w);
        (w, opt, BasicMeta::default())
    };
    w.print_summary();

    let lr = lr_override.unwrap_or_else(|| {
        if w.n_params() > 100_000_000 { 1e-4 }
        else if w.n_params() > 20_000_000 { 3e-4 }
        else { 1e-3 }
    });
    let mut opt = opt;
    opt = opt.with_lr(lr);

    // NOTE: VRAM-resident AdamW path is available via opt.enable_vram(&w)
    // but turns out to be net-slower than the cache-based path because
    // forward/backward still use prepare_weights' own VRAM cache — every
    // step we then pay 2× weight-sized BAR transfers to keep the mirror
    // and the cache in sync. Leaving it off until try_matmul can dispatch
    // on external VA pointers (bypassing the cache).
    let _ = gpu;  // silences unused-warning on non-GPU builds

    // ─── Runtime-framework-driven training loop ───
    //
    // Replaces the hand-rolled while(running) / Ctrl+C / manual logging
    // cadence with modgrad_training::TrainerLoop. All tunables sit on
    // TrainerConfig with defaults; step logic stays in the closure.
    //
    // Both closures (step + save) need mutable access to the same state.
    // RefCell gives us sequential interior mutability — the two closures
    // never run concurrently, so `borrow_mut` can never panic here.
    use modgrad_training::{TrainerConfig, TrainerLoop, StepReport};
    use std::cell::{Cell, RefCell};

    let grads = FfnGradients::zeros(&w);
    // Bundle w + opt + grads into one mutable unit so both closures
    // can reach them through `state.borrow_mut()` / `state.borrow()`.
    let state = RefCell::new((w, opt, grads));
    let offset = RefCell::new(0usize);
    let n_data = all_tokens.len();
    // Live token counter — resumed value + per-step deltas. save_fn reads
    // this instead of the snapshotted `resumed_meta.tokens_seen`, so
    // each save records an up-to-date count.
    let tokens_seen = Cell::new(resumed_meta.tokens_seen);
    // Start-of-run wall clock for the elapsed_secs metadata.
    let run_started = std::time::Instant::now();

    if let Some(secs) = budget_secs {
        eprintln!("\nTraining FFN... (lr={:.1e}, budget {}s, Ctrl+C to save early)\n", lr, secs);
    } else {
        eprintln!("\nTraining FFN... (lr={:.1e}, Ctrl+C to save and stop)\n", lr);
    }

    let cfg = TrainerConfig {
        max_steps: usize::MAX,   // run until SIGINT, `step_fn` returns None, or budget fires
        log_every: 100,
        save_every: Some(5_000),
        max_elapsed: budget_secs.map(std::time::Duration::from_secs),
        ..Default::default()
    };

    let step_fn = |_step: usize| -> Option<StepReport> {
        let mut o = offset.borrow_mut();
        loop {
            let end = (*o + context_len + 1).min(n_data);
            if end - *o < 2 {
                *o = 0;
                eprintln!("  --- epoch complete ---");
                continue;
            }
            let chunk = &all_tokens[*o..end];
            let mut s = state.borrow_mut();
            let (w, opt, grads) = &mut *s;
            let (loss, acc) = ffn_train_step(w, opt, grads, chunk);
            *o += context_len;
            // Count the tokens we just stepped over — one chunk = context_len
            // targets (we train one prediction per position in [0, context_len)).
            tokens_seen.set(tokens_seen.get() + context_len as u64);
            let progress = (*o as f32 / n_data as f32).min(1.0);
            return Some(StepReport::new(loss).with_accuracy(acc).with_progress(progress));
        }
    };

    // Save closure: delegate to `save_training_checkpoint`, which fills
    // schema / timestamp defaults. Resume pulls the same bundle back on
    // next launch (see earlier in this fn).
    let save_fn = || -> Result<(), Box<dyn std::error::Error>> {
        let s = state.borrow();
        modgrad_training::save_training_checkpoint(
            save_path,
            MODEL_KIND,
            &s.0,
            &s.1,
            s.1.step as u64,
            tokens_seen.get(),
            resumed_meta.elapsed_secs + run_started.elapsed().as_secs(),
        )?;
        Ok(())
    };

    let report = TrainerLoop::new(cfg).run(step_fn, save_fn);
    let training_seconds = run_started.elapsed().as_secs_f32();
    eprintln!("\nSaved to {save_path}  ({} steps, best {:.3} @ step {})",
        report.steps_completed, report.best_avg_loss, report.best_step);
    if report.stopped_by_budget {
        eprintln!("  (stopped by --budget)");
    }

    // Autoresearch summary: only printed when the caller asked for it
    // (via --val-data). The summary's val_bpb is the ground truth the
    // driving agent greps to decide keep/revert.
    if let Some(val_path) = val_data_path {
        let s = state.borrow();
        let w = &s.0;
        let mut val_bytes = std::fs::read(val_path)
            .unwrap_or_else(|e| { eprintln!("Failed to read {val_path}: {e}"); std::process::exit(1); });
        // Cap read at EVAL_MAX_BYTES so a huge val file doesn't eat
        // multiple minutes past the training budget.
        val_bytes.truncate(EVAL_MAX_BYTES);
        let val_tokens: Vec<usize> = val_bytes.iter().map(|&b| b as usize).collect();
        match compute_ffn_val_bpb(w, &val_tokens, context_len) {
            Ok((val_bpb, n_eval)) => {
                let total_seconds = run_started.elapsed().as_secs_f32();
                modgrad_training::AutoresearchSummary {
                    val_bpb,
                    training_seconds,
                    total_seconds,
                    peak_vram_mb: 0.0,
                    mfu_percent: 0.0,
                    total_tokens_m: (tokens_seen.get() as f32) / 1.0e6,
                    num_steps: report.steps_completed as u64,
                    num_params_m: (w.n_params() as f32) / 1.0e6,
                }.print();
                eprintln!("  (val_bpb over {n_eval} positions, {} bytes of {val_path}, ctx {context_len})",
                    val_bytes.len());
            }
            Err(msg) => {
                eprintln!("val_bpb eval failed: {msg}");
                std::process::exit(1);
            }
        }
    }
}

/// Thin adapter: `modgrad_training::metrics::compute_bpb` with an FFN
/// forward closure. Lives in the binary because `modgrad-training` has
/// no dependency on `modgrad-ffn` (and shouldn't gain one — keeps the
/// metric generic).
fn compute_ffn_val_bpb(
    w: &modgrad_ffn::FfnWeights,
    val_tokens: &[usize],
    context_len: usize,
) -> Result<(f32, usize), String> {
    modgrad_training::metrics::compute_bpb(val_tokens, context_len, |chunk| {
        let (logits, _cache) = modgrad_ffn::ffn_forward(w, chunk);
        logits
    })
}

/// Shared cap on end-of-run eval reads — same as `Eval`'s default, kept
/// here so `learn-ffn`'s post-training eval can't OOM or balloon the
/// total_seconds field on a huge val.txt.
const EVAL_MAX_BYTES: usize = 262_144;

/// `isis eval <checkpoint> <data…>` — load an FFN CheckpointBundle, compute
/// val_bpb on the concatenated data, print the autoresearch summary block.
/// For CTM/RegionalWeights checkpoints use a future dedicated subcommand;
/// this one is intentionally FFN-only to keep the eval harness honest.
fn eval_ffn(checkpoint: &str, data_paths: &[String], context_len: usize, max_bytes: usize) {
    use modgrad_training::{CheckpointBundle, AutoresearchSummary};
    use modgrad_ffn::{FfnWeights, FfnAdamW};
    const MODEL_KIND: &str = "ffn-cerebellum";

    let t_start = std::time::Instant::now();

    let bundle: CheckpointBundle<FfnWeights, FfnAdamW> =
        CheckpointBundle::load(checkpoint, MODEL_KIND).unwrap_or_else(|e| {
            eprintln!("Failed to load {checkpoint} as FFN bundle: {e}");
            std::process::exit(1);
        });
    let w = bundle.model;

    let mut val_bytes: Vec<u8> = Vec::new();
    for p in data_paths {
        match std::fs::read(p) {
            Ok(b) => val_bytes.extend_from_slice(&b),
            Err(e) => { eprintln!("Failed to read {p}: {e}"); std::process::exit(1); }
        }
        if val_bytes.len() >= max_bytes { break; }
    }
    val_bytes.truncate(max_bytes);
    let val_tokens: Vec<usize> = val_bytes.iter().map(|&b| b as usize).collect();

    let (val_bpb, n_positions) = match compute_ffn_val_bpb(&w, &val_tokens, context_len) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("val_bpb eval failed: {msg}");
            std::process::exit(1);
        }
    };
    let total_seconds = t_start.elapsed().as_secs_f32();

    // `total_tokens_m` in the autoresearch contract means *training*
    // tokens — for a standalone eval with no training, emit 0 to match
    // the semantic the driving agent expects.
    AutoresearchSummary {
        val_bpb,
        training_seconds: 0.0,
        total_seconds,
        peak_vram_mb: 0.0,
        mfu_percent: 0.0,
        total_tokens_m: 0.0,
        num_steps: 0,
        num_params_m: (w.n_params() as f32) / 1.0e6,
    }.print();
    eprintln!("  (eval over {n_positions} target positions, {} bytes, ctx {context_len})",
        val_bytes.len());
}

// ─── Generate ─────────────────────────────────────────────

fn run_generate(checkpoint: &str, prompt: &str, max_tokens: usize, temperature: f32, frozen_cereb_path: Option<&str>) {
    let mut w = RegionalWeights::load(checkpoint)
        .unwrap_or_else(|e| { eprintln!("Failed to load {checkpoint}: {e}"); std::process::exit(1); });

    // Load frozen cerebellum if specified
    let mut frozen: Option<Box<dyn modgrad_ctm::cerebellum::FrozenCerebellum>> =
        if let Some(path) = frozen_cereb_path {
            use modgrad_ctm::cerebellum::FrozenCerebellum;
            if path.ends_with(".safetensors") {
                let cfg = modgrad_ctm::frozen_transformer::TransformerConfig::qwen2_0_5b();
                let cereb = modgrad_ctm::frozen_transformer::FrozenTransformer::load(path, cfg)
                    .unwrap_or_else(|e| { eprintln!("Failed: {e}"); std::process::exit(1); });
                let hd = cereb.hidden_dim();
                let nl = cereb.n_layers();
                // Only configure projection if not already present in checkpoint
                if w.cereb_projection.is_none() {
                    w = w.with_frozen_cerebellum(hd, nl);
                }
                eprintln!("Cerebellum: {} (hidden_dim={}, layers={})", path, hd, nl);
                Some(Box::new(cereb))
            } else {
                #[cfg(feature = "onnx")]
                {
                    let cereb = isis_runtime::onnx_cerebellum::OnnxCerebellum::load(path)
                        .unwrap_or_else(|e| { eprintln!("Failed: {e}"); std::process::exit(1); });
                    let hd = cereb.hidden_dim();
                    let nl = cereb.n_layers();
                    if w.cereb_projection.is_none() {
                        w = w.with_frozen_cerebellum(hd, nl);
                    }
                    eprintln!("Cerebellum: {} (hidden_dim={}, layers={})", path, hd, nl);
                    Some(Box::new(cereb))
                }
                #[cfg(not(feature = "onnx"))]
                { eprintln!("ONNX requires --features onnx"); std::process::exit(1); }
            }
        } else {
            None
        };

    w.print_summary();

    // Generate with cerebellum
    let prompt_bytes = prompt.as_bytes();
    let mut nc = NeuralComputer::new(w);

    // Feed prompt through NC + cerebellum
    if let Some(ref mut fc) = frozen {
        use modgrad_ctm::cerebellum::blended_hidden_at;
        let token_ids: Vec<i64> = prompt_bytes.iter().map(|&b| b as i64).collect();
        let cache = fc.encode_context_layers(&token_ids);

        for (i, &b) in prompt_bytes.iter().enumerate() {
            let logits = nc.step(b as usize);
            // Inject cerebellum hidden state into the NC's region outputs
            if let Some(ref proj) = nc.weights.cereb_projection {
                let cereb_idx = nc.weights.config.region_names.iter()
                    .position(|n| n.contains("cerebellum")).unwrap_or(4);
                let hidden = blended_hidden_at(&cache, proj, i);
                if !hidden.is_empty() {
                    let d_model = nc.weights.config.regions[cereb_idx].d_model;
                    let projected = proj.project_out(&hidden);
                    let blend = nc.weights.cereb_blend_logit
                        .map(|l| 1.0 / (1.0 + (-l).exp()))
                        .unwrap_or(0.1);
                    for j in 0..d_model.min(projected.len()) {
                        nc.state.region_outputs[cereb_idx][j] += blend * projected[j];
                    }
                }
            }
            let _ = logits;
        }
    } else {
        for &b in prompt_bytes {
            nc.step(b as usize);
        }
    }

    // Generate
    print!("{prompt}");
    let mut prev_logits = nc.step(prompt_bytes.last().copied().unwrap_or(b' ') as usize);
    for _ in 0..max_tokens {
        let next = nc.sample(&prev_logits, temperature);
        if next < 256 {
            print!("{}", next as u8 as char);
        }
        // For generation, we run without cerebellum (autoregressive —
        // we'd need to re-encode the growing sequence each step, which
        // is expensive. The cortex has already been primed by the prompt.)
        prev_logits = nc.step(next);
    }
    println!();
}

// ─── Daemon ───────────────────────────────────────────────

fn run_daemon(checkpoint: &str, port: u16) {
    let w = if std::path::Path::new(checkpoint).exists() {
        eprintln!("Loading {checkpoint}...");
        RegionalWeights::load(checkpoint)
            .unwrap_or_else(|e| { eprintln!("Failed: {e}"); std::process::exit(1); })
    } else {
        eprintln!("No checkpoint at {checkpoint}, creating fresh 8-region model...");
        let cfg = RegionalConfig::eight_region(32, 256, 2);
        RegionalWeights::new(cfg)
    };
    w.print_summary();

    let mut nc = NeuralComputer::new(w);

    // Start debug server (this IS the daemon — accepts commands via the debug protocol)
    let view = nc_socket::NcDebugView::from_nc(&nc);
    let view = std::sync::Arc::new(std::sync::Mutex::new(view));
    let handle = nc_socket::start_debug_server(port, view.clone());

    eprintln!("Daemon running on port {port}. Ctrl+C to stop.");
    eprintln!("Connect with: modgrad-debugger 127.0.0.1:{port}");
    eprintln!("Or send text: isis send \"hello world\" --addr 127.0.0.1:{port}");

    // Block forever, updating state when debug clients inject tokens
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nShutting down...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    }).ok();

    while running.load(std::sync::atomic::Ordering::SeqCst) {
        // Check for injected tokens from debug clients
        if let Some(event) = handle.poll_control() {
            match event {
                nc_socket::DebugEvent::Inject(tokens) => {
                    let response = nc.act(&tokens, 256, 0.8);
                    // Update view for debugger
                    if let Ok(mut v) = view.try_lock() {
                        *v = nc_socket::NcDebugView::from_nc(&nc);
                    }
                    // Print response as text
                    for &t in &response {
                        if t < 256 { print!("{}", t as u8 as char); }
                    }
                    io::stdout().flush().ok();
                }
                nc_socket::DebugEvent::Pause | nc_socket::DebugEvent::Resume => {}
                nc_socket::DebugEvent::Step(token) => {
                    nc.step(token);
                    if let Ok(mut v) = view.try_lock() {
                        *v = nc_socket::NcDebugView::from_nc(&nc);
                    }
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Save on exit
    if let Err(e) = nc.weights.save(checkpoint) {
        eprintln!("Save failed: {e}");
    } else {
        eprintln!("Saved to {checkpoint}");
    }
}

// ─── Send ─────────────────────────────────────────────────

fn send_command(text: &str, addr: &str) {
    use std::io::{Read, Write as IoWrite};
    use std::net::TcpStream;

    let mut stream = match TcpStream::connect(addr) {
        Ok(s) => s,
        Err(e) => { eprintln!("Can't connect to {addr}: {e}"); std::process::exit(1); }
    };
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).ok();

    // Send InjectText request via debug protocol
    let req = nc_socket::DebugRequest::InjectText { text: text.to_string() };
    let data = bincode::serialize(&req).expect("serialize failed");
    let len = data.len() as u32;
    stream.write_all(&len.to_le_bytes()).ok();
    stream.write_all(&data).ok();
    stream.flush().ok();

    // Read response
    let mut len_buf = [0u8; 4];
    if stream.read_exact(&mut len_buf).is_ok() {
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        if stream.read_exact(&mut buf).is_ok() {
            if let Ok(resp) = bincode::deserialize::<nc_socket::DebugResponse>(&buf) {
                match resp {
                    nc_socket::DebugResponse::Ok => eprintln!("Sent."),
                    nc_socket::DebugResponse::Error { msg } => eprintln!("Error: {msg}"),
                    _ => eprintln!("Response: {resp:?}"),
                }
            }
        }
    }
}

// ─── Devices ──────────────────────────────────────────────

fn show_devices() {
    eprintln!("Compute devices:");
    eprintln!("  CPU: available (rayon, {} threads)", rayon::current_num_threads());

    #[cfg(feature = "cuda")]
    eprintln!("  CUDA: enabled");
    #[cfg(not(feature = "cuda"))]
    eprintln!("  CUDA: disabled (build with --features cuda)");

    // Check KFD (AMD GPU)
    if std::path::Path::new("/dev/kfd").exists() {
        eprintln!("  AMD KFD: /dev/kfd present");
    } else {
        eprintln!("  AMD KFD: not available");
    }
}
/// Extract a JSON array of integers from a line by key name.
/// Minimal parser — avoids pulling in serde_json for a simple pattern.
fn extract_json_array(line: &str, key: &str) -> Option<Vec<usize>> {
    let pattern = format!("\"{}\":", key);
    let start = line.find(&pattern)?;
    let after_key = &line[start + pattern.len()..];
    let bracket_start = after_key.find('[')?;
    let bracket_end = after_key.find(']')?;
    let inner = &after_key[bracket_start + 1..bracket_end];
    let tokens: Vec<usize> = inner.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if tokens.is_empty() { None } else { Some(tokens) }
}

fn generate_multimodal_pairs() -> Vec<Vec<usize>> {
    // graph types imported at crate level
    use modgrad_codec::vqvae::VqVae;
    use modgrad_codec::audio_codec::AudioCodec;

    let vae = VqVae::new(4096, 64);
    let audio_codec = AudioCodec::new(4096, 64, 24000);
    let mut pairs = Vec::new();

    // ── Text → Image pairs (synthetic "images" with CIFAR-10 class names) ──
    let class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"];

    for (label, name) in class_names.iter().enumerate() {
        // Generate synthetic pixel patterns per class (deterministic from label)
        for variant in 0..20 {
            let mut pixels = vec![0.0f32; 3072];
            let seed = (label * 100 + variant) as f32;
            for i in 0..3072 {
                pixels[i] = ((seed + i as f32 * 0.1).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
            }
            let codes = vae.tokenize(&pixels);
            let img_tokens = image_codes_to_tokens(&codes);

            // "a picture of a cat " <img> [codes] </img>
            let mut seq = text_to_tokens(format!("a picture of a {name} ").as_bytes());
            seq.extend(&img_tokens);
            pairs.push(seq);

            // <img> [codes] </img> " this is a cat"
            let mut seq = img_tokens.clone();
            seq.extend(text_to_tokens(format!(" this is a {name}").as_bytes()));
            pairs.push(seq);

            // Describe and show: "here is a cat: " <img> ... </img> " it has fur"
            let descriptions = [
                format!("{name} in a photo"),
                format!("image of {name}"),
                format!("you can see a {name} here"),
            ];
            let desc = &descriptions[variant % descriptions.len()];
            let mut seq = text_to_tokens(desc.as_bytes());
            seq.push(b' ' as usize);
            seq.extend(&img_tokens);
            pairs.push(seq);
        }
    }

    // ── Text → Audio pairs (synthetic waveforms) ──
    let audio_descriptions = [
        ("a sine tone", 440.0),
        ("a low hum", 120.0),
        ("a high pitch", 2000.0),
        ("a beep", 800.0),
        ("silence", 0.0),
    ];

    for (desc, freq) in &audio_descriptions {
        // Generate 0.5 seconds of synthetic audio at 24kHz
        let n_samples = 12000;
        let waveform: Vec<f32> = (0..n_samples).map(|i| {
            if *freq > 0.0 {
                (2.0 * std::f32::consts::PI * freq * i as f32 / 24000.0).sin() * 0.5
            } else {
                0.0
            }
        }).collect();

        let codes = audio_codec.tokenize(&waveform);
        let aud_tokens = audio_codes_to_tokens(&codes);

        // "a sine tone " <aud> [codes] </aud>
        let mut seq = text_to_tokens(format!("{desc} ").as_bytes());
        seq.extend(&aud_tokens);
        pairs.push(seq);

        // <aud> [codes] </aud> " that was a sine tone"
        let mut seq = aud_tokens.clone();
        seq.extend(text_to_tokens(format!(" that was {desc}").as_bytes()));
        pairs.push(seq);
    }

    // ── Text → Action pairs (GUI interaction patterns) ──
    let action_patterns: Vec<(&str, Vec<usize>)> = vec![
        ("click the center of the screen",
            action_click(0.5, 0.5)),
        ("click the top left corner",
            action_click(0.0, 0.0)),
        ("click the bottom right",
            action_click(1.0, 1.0)),
        ("move the mouse to the middle",
            action_mouse_move(0.5, 0.5)),
        ("type hello world",
            action_type_text("hello world")),
        ("press enter",
            action_key(ACT_KEY_ENTER)),
        ("press escape",
            action_key(ACT_KEY_ESCAPE)),
        ("press tab",
            action_key(ACT_KEY_TAB)),
        ("scroll up",
            action_key(ACT_SCROLL_UP)),
        ("scroll down",
            action_key(ACT_SCROLL_DOWN)),
        ("press ctrl c",
            action_modified_key(ACT_KEY_CTRL, b'c')),
        ("press ctrl v to paste",
            action_modified_key(ACT_KEY_CTRL, b'v')),
        ("open a new tab",
            action_modified_key(ACT_KEY_CTRL, b't')),
        ("close the window",
            action_modified_key(ACT_KEY_ALT, b'F')),  // alt+F4 approximation
        ("navigate up",
            action_key(ACT_KEY_UP)),
        ("navigate down",
            action_key(ACT_KEY_DOWN)),
    ];

    for (desc, action) in &action_patterns {
        // "click the center " <act> left_click 0.5 0.5 </act>
        let mut seq = text_to_tokens(format!("{desc} ").as_bytes());
        seq.extend(action);
        pairs.push(seq);

        // Variant: "please {action}" for instruction-following
        let mut seq = text_to_tokens(format!("please {desc} ").as_bytes());
        seq.extend(action);
        pairs.push(seq);
    }

    // ── Conversation patterns (text ↔ text with multimodal context) ──
    let conversations = [
        "what do you see? a cat in the image",
        "describe the sound. it is a high pitched tone",
        "what should i click? click the button in the center",
        "how does it look? the image shows a red car",
        "repeat after me: hello. hello",
    ];
    for conv in &conversations {
        for repeat in 0..5 {
            let mut seq = text_to_tokens(conv.as_bytes());
            // Add slight variation with a trailing space count
            for _ in 0..repeat { seq.push(b' ' as usize); }
            pairs.push(seq);
        }
    }

    // ── Multimodal chains (text → image → text → action) ──
    for (label, name) in class_names.iter().enumerate() {
        let mut pixels = vec![0.0f32; 3072];
        for i in 0..3072 { pixels[i] = ((label as f32 + i as f32 * 0.07).sin() * 0.5 + 0.5).clamp(0.0, 1.0); }
        let codes = vae.tokenize(&pixels);
        let img_tokens = image_codes_to_tokens(&codes);

        // "show me a {name}" → <img> → "now click on it" → <act>
        let mut seq = text_to_tokens(format!("show me a {name} ").as_bytes());
        seq.extend(&img_tokens);
        seq.extend(text_to_tokens(b" now click on it "));
        seq.extend(&action_click(0.5, 0.5));
        pairs.push(seq);
    }

    eprintln!("  Generated {} multimodal paired sequences", pairs.len());
    pairs
}

fn develop_staged(
    save_path: &str,
    curriculum_path: Option<&str>,
    multimodal: bool,
    images_path: Option<&str>,
    audio_path: Option<&str>,
    video_path: Option<&str>,
    video_fps: f32,
    debug_port: Option<u16>,
) {
    // graph types imported at crate level
    use isis_runtime::curriculum;

    // Model size from filename
    let (embed_dim, n_regions, ticks, context_len) = if save_path.contains("large") {
        (128, 8, 4, 128)
    } else if save_path.contains("medium") {
        (64, 8, 3, 64)
    } else if save_path.contains("tiny") {
        (16, 4, 2, 16)
    } else {
        (32, 8, 2, 32) // small (default)
    };
    let vocab_size = if multimodal { VOCAB_MULTIMODAL } else { VOCAB_TEXT };

    // Load model + optimizer via CheckpointBundle (canonical format).
    // On legacy two-file layouts, fall back to RegionalWeights::load +
    // the .opt.bin sidecar so older checkpoints keep working. Next save
    // writes a bundle regardless.
    use modgrad_training::{CheckpointBundle, BasicMeta};
    const MODEL_KIND_CTM: &str = "ctm-regional";
    let legacy_opt_path = save_path
        .replace(".bin", ".opt.bin")
        .replace(".json", ".opt.bin");

    let (mut w, mut opt, resumed_meta_ctm) = if std::path::Path::new(save_path).exists() {
        match CheckpointBundle::<RegionalWeights, RegionalAdamW>::load(save_path, MODEL_KIND_CTM) {
            Ok(bundle) => {
                eprintln!("Resumed {save_path} — step {}, {} tokens seen",
                    bundle.meta.step, bundle.meta.tokens_seen);
                (bundle.model, bundle.optimizer, bundle.meta)
            }
            Err(e) => {
                eprintln!("  (not a CheckpointBundle: {e})");
                eprintln!("  falling back to legacy two-file checkpoint...");
                let w = RegionalWeights::load(save_path).expect("failed to load model");
                let opt = if std::path::Path::new(&legacy_opt_path).exists() {
                    RegionalAdamW::load(&legacy_opt_path).unwrap_or_else(|_| RegionalAdamW::new(&w))
                } else {
                    RegionalAdamW::new(&w).with_lr(3e-3).with_wd(0.001).with_clip(5.0)
                };
                (w, opt, BasicMeta::default())
            }
        }
    } else {
        let cfg = match (n_regions, multimodal) {
            (4, true) => RegionalConfig::four_region_multimodal(embed_dim, ticks),
            (4, false) => RegionalConfig::four_region(embed_dim, vocab_size, ticks),
            (_, true) => RegionalConfig::eight_region_multimodal(embed_dim, ticks),
            (_, false) => RegionalConfig::eight_region(embed_dim, vocab_size, ticks),
        };
        let w = RegionalWeights::new(cfg);
        let opt = RegionalAdamW::new(&w).with_lr(3e-3).with_wd(0.001).with_clip(5.0);
        (w, opt, BasicMeta::default())
    };
    w.print_summary();
    eprintln!("  AdamW: lr={}, wd={}, clip={}, step={}",
        opt.lr, opt.weight_decay, opt.grad_clip, opt.step);

    // Wall clock for meta.elapsed_secs; added to resumed_meta_ctm.elapsed_secs.
    let run_started = std::time::Instant::now();

    // Ctrl+C handler for clean save
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nCtrl+C — saving and exiting...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    }).ok();

    // Debug server — lets the debugger watch training live
    let debug_nc: Option<(
        std::sync::Arc<std::sync::Mutex<isis_runtime::nc_socket::NcDebugView>>,
    )> = if let Some(port) = debug_port {
        use isis_runtime::nc_socket;
        // Create a temporary NC view from the weights for the debug server
        let nc_tmp = NeuralComputer::new(w.clone());
        let view = nc_socket::NcDebugView::from_nc(&nc_tmp);
        let view = std::sync::Arc::new(std::sync::Mutex::new(view));
        let _handle = nc_socket::start_debug_server(port, view.clone());
        Some((view,))
    } else {
        None
    };

    // Helper: update debug view during training
    let update_train_debug = |w: &RegionalWeights, step: usize, _loss: f32,
                               debug: &Option<(std::sync::Arc<std::sync::Mutex<isis_runtime::nc_socket::NcDebugView>>,)>| {
        if let Some((view,)) = debug {
            if step % 10 == 0 { // update every 10 steps to avoid overhead
                if let Ok(mut guard) = view.try_lock() {
                    // Update region params (weights may have changed)
                    guard.region_params = w.regions.iter().map(|r| r.n_params()).collect();
                    guard.total_params = w.n_params();
                    // Store step count in history for the debugger to read
                    guard.history = vec![step];
                }
            }
        }
    };

    // Load external curriculum if provided
    let external_data = curriculum_path.and_then(|path| {
        eprintln!("Loading external curriculum from {path}...");
        match curriculum::load_external(path) {
            Ok(data) => {
                eprintln!("  {} items loaded", data.len());
                Some(data)
            }
            Err(e) => {
                eprintln!("  Warning: {e}, using built-in only");
                None
            }
        }
    });

    let max_steps_per_phase = 2000;
    let reps = 5;

    // Try to load real-world text for later phases
    let real_text = {
        let candidates = [
            "train_climbmix_5m.txt", "train_climbmix.txt",
            "train_large.txt", "train_stories.txt",
        ];
        let mut text = Vec::new();
        for c in &candidates {
            let path = format!("{}/{c}", std::path::Path::new(save_path).parent()
                .unwrap_or(std::path::Path::new(".")).display());
            if let Ok(data) = std::fs::read(&path) {
                eprintln!("Loaded real text: {path} ({} bytes)", data.len());
                text = data;
                break;
            }
        }
        text
    };

    // Load multimodal data if requested
    let image_tokens: Vec<Vec<usize>> = if multimodal {
        if let Some(img_dir) = images_path {
            load_image_tokens(img_dir)
        } else {
            eprintln!("  No --images path, generating synthetic image tokens");
            // Synthetic: random VQ codes for testing the pipeline
            (0..100).map(|i| {
                let mut codes = vec![TOKEN_IMG_START];
                for j in 0..64 { codes.push(TOKEN_IMG_OFFSET + (i * 64 + j) % TOKEN_IMG_CODES); }
                codes.push(TOKEN_IMG_END);
                codes
            }).collect()
        }
    } else {
        Vec::new()
    };

    let audio_tokens: Vec<Vec<usize>> = if multimodal {
        if let Some(aud_dir) = audio_path {
            load_audio_tokens(aud_dir)
        } else {
            eprintln!("  No --audio path, generating synthetic audio tokens");
            (0..50).map(|i| {
                let mut codes = vec![TOKEN_AUD_START];
                for j in 0..75 { codes.push(TOKEN_AUD_OFFSET + (i * 75 + j) % TOKEN_AUD_CODES); }
                codes.push(TOKEN_AUD_END);
                codes
            }).collect()
        }
    } else {
        Vec::new()
    };

    // Load video data: each video → one token sequence with timestamps
    let video_tokens: Vec<Vec<usize>> = if multimodal {
        if let Some(vid_path) = video_path {
            load_video_tokens(vid_path, video_fps)
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    // Generate synthetic paired training data (text↔image, text↔audio, text↔action)
    let paired_data: Vec<Vec<usize>> = if multimodal {
        generate_multimodal_pairs()
    } else {
        Vec::new()
    };

    if multimodal {
        eprintln!("  Multimodal: {} images, {} audio, {} video, {} paired sequences",
            image_tokens.len(), audio_tokens.len(), video_tokens.len(), paired_data.len());
    }

    let mut total_tokens = opt.step as u64;

    for phase in 0..curriculum::NUM_PHASES {
        if !running.load(std::sync::atomic::Ordering::SeqCst) { break; }

        let (phase_name, mut phase_data) = curriculum::generate(phase, reps);

        if let Some(ref ext) = external_data {
            for (p, data) in ext {
                if *p == phase {
                    phase_data.extend_from_slice(data);
                }
            }
        }

        // Mix in real text from phase 0 — the model needs real English
        if !real_text.is_empty() {
            let limit = match phase {
                0..=2 => (100_000).min(real_text.len()),   // 100KB early
                3..=4 => (500_000).min(real_text.len()),   // 500KB mid
                _ => (2_000_000).min(real_text.len()),      // 2MB late
            };
            phase_data.extend_from_slice(&real_text[..limit]);
            eprintln!("  + {limit} bytes of real text mixed in");
        }

        eprintln!("\n============================================================");
        eprintln!("=== PHASE {phase}: {phase_name} ({} bytes) ===", phase_data.len());
        eprintln!("  Mastery threshold: {:.1}, streak needed: {}",
            curriculum::MASTERY_THRESHOLDS[phase], curriculum::MASTERY_STREAK);

        // Build training sequences as token indices
        let mut sequences: Vec<Vec<usize>> = phase_data.chunks(context_len)
            .map(|c| c.iter().map(|&b| b as usize).collect())
            .collect();

        // In later phases with multimodal, mix in paired data + raw media.
        // Gradual introduction: phase 5 = 10% paired, phase 6 = 25%, phase 7 = 50%.
        // This prevents loss spikes from suddenly seeing unknown token ranges.
        if multimodal && phase >= 5 {
            let n_text = sequences.len();
            let ratio = match phase {
                5 => 10,  // 1 multimodal per 10 text sequences
                6 => 4,   // 1 per 4
                _ => 2,   // 1 per 2
            };
            let n_pairs = (n_text / ratio).min(paired_data.len());
            for i in 0..n_pairs {
                sequences.push(paired_data[i].clone());
            }

            // Raw media sequences only in final phases
            if phase >= 6 {
                let n_img = image_tokens.len();
                let n_aud = audio_tokens.len();
                let n_vid = video_tokens.len();
                let n_media = n_img + n_aud + n_vid;
                for i in 0..(n_text / 4).min(n_media.max(1)) {
                    if i < n_img {
                        sequences.push(image_tokens[i % n_img].clone());
                    } else if i < n_img + n_aud {
                        sequences.push(audio_tokens[(i - n_img) % n_aud.max(1)].clone());
                    } else if n_vid > 0 {
                        sequences.push(video_tokens[(i - n_img - n_aud) % n_vid].clone());
                    }
                }
            }
        }

        if sequences.is_empty() { continue; }

        let mut mastery_streak = 0;
        let mut step = 0;
        let mut phase_losses = Vec::new();
        let threshold = curriculum::MASTERY_THRESHOLDS[phase];

        let mut grads = RegionalGradients::zeros(&w); // allocate once, reuse

        // Dream hook: replay random positions from current phase data
        let seqs_ref = &sequences;
        let mut dream_counter = 0u64;
        let mut dream = modgrad_training::dream::DreamHook::new(20, 0.3,
            |w: &mut RegionalWeights, lr: f32| {
                dream_counter += 1;
                let seq = &seqs_ref[(dream_counter as usize * 7919) % seqs_ref.len()];
                if seq.len() > 9 {
                    let pos = ((dream_counter * 7919) as usize) % (seq.len() - 9);
                    let mut dg = RegionalGradients::zeros(w);
                    dream_step(w, &mut dg, seq[pos], &seq[pos+1..pos+9], 8, 1.0);
                    dg.apply(w, lr, 1.0);
                }
            });

        while step < max_steps_per_phase {
            if !running.load(std::sync::atomic::Ordering::SeqCst) { break; }

            let chunk_idx = step % sequences.len();
            let seq = &sequences[chunk_idx];
            if seq.len() < 2 { step += 1; continue; }

            grads.zero(); // reuse buffer — no allocation
            let mut chunk_loss = 0.0f32;
            let mut chunk_correct = 0usize;
            let n_tokens = seq.len() - 1;

            for pos in 0..n_tokens {
                let token = seq[pos];
                let target = seq[pos + 1];
                let (loss, pred) = regional_train_token(&w, &mut grads, token, target);
                chunk_loss += loss;
                if pred == target { chunk_correct += 1; }
            }

            // AdamW step (no gradient averaging — AdamW adapts to gradient scale)
            opt.step(&mut w, &mut grads);

            // Dream phase
            dream.after_step(&mut w, step, opt.lr);

            chunk_loss /= n_tokens as f32;
            phase_losses.push(chunk_loss);
            total_tokens += n_tokens as u64;
            update_train_debug(&w, step, chunk_loss, &debug_nc);

            if chunk_loss < threshold {
                mastery_streak += 1;
            } else {
                mastery_streak = 0;
            }

            if step % 50 == 0 || mastery_streak >= curriculum::MASTERY_STREAK {
                let recent: f32 = if phase_losses.len() >= 10 {
                    phase_losses[phase_losses.len()-10..].iter().sum::<f32>() / 10.0
                } else {
                    phase_losses.iter().sum::<f32>() / phase_losses.len() as f32
                };
                eprintln!("  step {step:4}: loss={chunk_loss:.3} avg10={recent:.3} acc={chunk_correct}/{n_tokens} streak={mastery_streak}/{}",
                    curriculum::MASTERY_STREAK);
            }

            if mastery_streak >= curriculum::MASTERY_STREAK {
                eprintln!("  MASTERED phase {phase} ({phase_name}) at step {step}!");
                break;
            }

            step += 1;
        }

        if mastery_streak < curriculum::MASTERY_STREAK {
            let avg: f32 = phase_losses.iter().sum::<f32>() / phase_losses.len().max(1) as f32;
            eprintln!("  Phase {phase} not mastered after {max_steps_per_phase} steps (avg loss={avg:.3})");
        }

        // Save checkpoint after each phase as one atomic CheckpointBundle.
        modgrad_training::save_training_checkpoint(
            save_path,
            MODEL_KIND_CTM,
            &w,
            &opt,
            opt.step as u64,
            total_tokens,
            resumed_meta_ctm.elapsed_secs + run_started.elapsed().as_secs(),
        ).expect("failed to save checkpoint");
        eprintln!("  Checkpoint saved to {save_path} ({total_tokens} tokens trained)");
    }

    // Final generation test
    eprintln!("\n=== Generation test ===");
    for prompt in &[b"the " as &[u8], b"0x" as &[u8], b"fn " as &[u8], b"ssh " as &[u8]] {
        let mut generated = prompt.to_vec();
        let mut state = RegionalState::new(&w);
        // Feed prompt
        for &b in prompt.iter() {
            let obs = w.embed(b as usize).to_vec();
            let _ = regional_forward(&w, &mut state, &obs);
        }
        // Generate
        for _ in 0..30 {
            let obs = w.embed(*generated.last().unwrap() as usize).to_vec();
            let out = regional_forward(&w, &mut state, &obs);
            if let Some(logits) = out.predictions.last() {
                let next = logits.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as u8).unwrap_or(b' ');
                generated.push(next);
            }
        }
        let prompt_str = std::str::from_utf8(prompt).unwrap_or("?");
        let output_str = String::from_utf8_lossy(&generated[prompt.len()..]);
        eprintln!("  \"{prompt_str}\" -> \"{output_str}\"");
    }

    modgrad_training::save_training_checkpoint(
        save_path,
        MODEL_KIND_CTM,
        &w,
        &opt,
        opt.step as u64,
        total_tokens,
        resumed_meta_ctm.elapsed_secs + run_started.elapsed().as_secs(),
    ).expect("failed to save final checkpoint");
    let size = std::fs::metadata(save_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("\nFinal save to {save_path} ({size} bytes, {total_tokens} tokens)");
}

/// Learn from raw bytes. No curriculum, no phases, no graduation.
/// Reads every file in the given paths as a byte stream and predicts next bytes.
/// The model's exit gate, regional specialization, and sync dynamics
/// self-organize around whatever structure exists in the data.
fn learn(
    save_path: &str,
    data_paths: &[String],
    context_len: usize,
    vocab: usize,
    medium: bool,
    large: bool,
    billion: bool,
    debug_port: Option<u16>,
    frozen_cereb_path: Option<&str>,
) {
    // Gather all data as token sequences.
    // Two modes:
    //   - .jsonl files: read {input_tokens, output_tokens} pairs, concatenate into sequences
    //   - everything else: read as raw bytes (each byte is a token 0-255)
    let mut all_tokens: Vec<usize> = Vec::new();

    for path in data_paths {
        let p = std::path::Path::new(path);

        if p.extension().map_or(false, |e| e == "jsonl") {
            // JSONL: each line has input_tokens + output_tokens
            if let Ok(text) = std::fs::read_to_string(p) {
                let mut n_samples = 0usize;
                for line in text.lines() {
                    if line.is_empty() { continue; }
                    // Minimal JSON parsing — extract token arrays
                    if let (Some(input), Some(output)) = (
                        extract_json_array(line, "input_tokens"),
                        extract_json_array(line, "output_tokens"),
                    ) {
                        all_tokens.extend_from_slice(&input);
                        all_tokens.extend_from_slice(&output);
                        n_samples += 1;
                    }
                }
                eprintln!("  + {path} ({n_samples} token pairs)");
            }
        } else if p.is_dir() {
            if let Ok(entries) = std::fs::read_dir(p) {
                let mut files: Vec<_> = entries.filter_map(|e| e.ok())
                    .filter(|e| e.path().is_file())
                    .collect();
                files.sort_by_key(|e| e.path());
                for entry in files {
                    let ep = entry.path();
                    if ep.extension().map_or(false, |e| e == "jsonl") {
                        // Recurse into JSONL files in directory
                        if let Ok(text) = std::fs::read_to_string(&ep) {
                            let mut n = 0usize;
                            for line in text.lines() {
                                if line.is_empty() { continue; }
                                if let (Some(input), Some(output)) = (
                                    extract_json_array(line, "input_tokens"),
                                    extract_json_array(line, "output_tokens"),
                                ) {
                                    all_tokens.extend_from_slice(&input);
                                    all_tokens.extend_from_slice(&output);
                                    n += 1;
                                }
                            }
                            eprintln!("  + {} ({n} token pairs)", ep.display());
                        }
                    } else if let Ok(data) = std::fs::read(&ep) {
                        eprintln!("  + {} ({} bytes)", ep.display(), data.len());
                        for &b in &data { all_tokens.push(b as usize); }
                    }
                }
            }
        } else if let Ok(data) = std::fs::read(path) {
            eprintln!("  + {path} ({} bytes)", data.len());
            for &b in &data { all_tokens.push(b as usize); }
        }
    }

    if all_tokens.is_empty() {
        eprintln!("No data found. Provide files, directories, or .jsonl token pairs.");
        return;
    }

    eprintln!("Data: {} tokens, vocab: {}", all_tokens.len(), vocab);

    // Model size from filename (same convention as develop_staged)
    let (embed_dim, n_regions, ticks) = if save_path.contains("large") {
        (128, 8, 4)
    } else if save_path.contains("medium") {
        (64, 8, 3)
    } else if save_path.contains("tiny") {
        (16, 4, 2)
    } else {
        (32, 8, 2)
    };

    // ─── Resume or create ───
    //
    // Try CheckpointBundle first — it's the new canonical save format.
    // If the file exists but isn't a bundle (pre-migration two-file
    // checkpoint), fall back to loading RegionalWeights + RegionalAdamW
    // individually. The next save always writes a bundle regardless,
    // so the legacy path is a one-shot migration.
    use modgrad_training::{CheckpointBundle, BasicMeta};
    const MODEL_KIND_CTM: &str = "ctm-regional";

    let (maybe_opt, resumed_meta_ctm, mut w) = if std::path::Path::new(save_path).exists() {
        match CheckpointBundle::<RegionalWeights, RegionalAdamW>::load(save_path, MODEL_KIND_CTM) {
            Ok(bundle) => {
                eprintln!("Resumed {save_path} — step {}, {} tokens seen",
                    bundle.meta.step, bundle.meta.tokens_seen);
                (Some(bundle.optimizer), bundle.meta, bundle.model)
            }
            Err(e) => {
                eprintln!("  (not a CheckpointBundle: {e})");
                eprintln!("  falling back to legacy two-file checkpoint...");
                let w = RegionalWeights::load(save_path).expect("failed to load");
                (None, BasicMeta::default(), w)
            }
        }
    } else {
        let cfg = if billion {
            eprintln!("Creating billion-scale model (d_model=1024, ~1B params)...");
            RegionalConfig::eight_region_billion(embed_dim, vocab, ticks)
        } else if large {
            eprintln!("Creating large model (d_model=512, ~81M params)...");
            RegionalConfig::eight_region_large(embed_dim, vocab, ticks)
        } else if medium {
            eprintln!("Creating medium model (d_model=256, ~55M params)...");
            RegionalConfig::eight_region_medium(embed_dim, vocab, ticks)
        } else if n_regions <= 4 {
            RegionalConfig::four_region(embed_dim, vocab, ticks)
        } else {
            RegionalConfig::eight_region(embed_dim, vocab, ticks)
        };
        (None, BasicMeta::default(), RegionalWeights::new(cfg))
    };
    // Frozen cerebellum: load from safetensors (native) or ONNX
    let frozen_cereb: Option<Box<dyn modgrad_ctm::cerebellum::FrozenCerebellum>> =
        if let Some(path) = frozen_cereb_path {
            use modgrad_ctm::cerebellum::FrozenCerebellum;

            if path.ends_with(".safetensors") {
                // Native safetensors loader — no external runtime needed
                let cfg = modgrad_ctm::frozen_transformer::TransformerConfig::qwen2_0_5b();
                let cereb = modgrad_ctm::frozen_transformer::FrozenTransformer::load(path, cfg)
                    .unwrap_or_else(|e| { eprintln!("Failed to load safetensors: {e}"); std::process::exit(1); });
                let hd = cereb.hidden_dim();
                let nl = cereb.n_layers();
                w = w.with_frozen_cerebellum(hd, nl);
                eprintln!("Frozen cerebellum (native): {} (hidden_dim={}, layers={})", path, hd, nl);
                Some(Box::new(cereb) as Box<dyn FrozenCerebellum>)
            } else if path.ends_with(".onnx") {
                #[cfg(feature = "onnx")]
                {
                    let cereb = isis_runtime::onnx_cerebellum::OnnxCerebellum::load(path)
                        .unwrap_or_else(|e| { eprintln!("Failed to load ONNX: {e}"); std::process::exit(1); });
                    let hd = cereb.hidden_dim();
                    let nl = cereb.n_layers();
                    w = w.with_frozen_cerebellum(hd, nl);
                    eprintln!("Frozen cerebellum (ONNX): {} (hidden_dim={}, layers={})", path, hd, nl);
                    Some(Box::new(cereb) as Box<dyn FrozenCerebellum>)
                }
                #[cfg(not(feature = "onnx"))]
                {
                    eprintln!("ERROR: .onnx files require --features onnx. Use .safetensors instead.");
                    std::process::exit(1);
                }
            } else {
                eprintln!("ERROR: --frozen-cereb supports .safetensors or .onnx files");
                std::process::exit(1);
            }
        } else {
            None
        };

    w.print_summary();

    // Optimizer: bundle path already loaded it; legacy path tries the
    // .opt.bin sidecar; fresh-create tunes LR/clip by model size.
    let opt = if let Some(o) = maybe_opt {
        o
    } else {
        let legacy_opt = if save_path.ends_with(".bin") {
            save_path.replace(".bin", ".opt.bin")
        } else if save_path.ends_with(".json") {
            save_path.replace(".json", ".opt.json")
        } else {
            format!("{save_path}.opt")
        };
        if std::path::Path::new(&legacy_opt).exists() {
            RegionalAdamW::load(&legacy_opt).unwrap_or_else(|_| RegionalAdamW::new(&w))
        } else {
            let (lr, clip) = if w.n_params() > 50_000_000 { (3e-4, 1.0) }
                             else if w.n_params() > 10_000_000 { (1e-3, 2.0) }
                             else { (3e-3, 5.0) };
            RegionalAdamW::new(&w).with_lr(lr).with_wd(0.001).with_clip(clip)
        }
    };

    // Debug server — Arc<Mutex<>> so the step closure can reach it later.
    let debug_nc: Option<std::sync::Arc<std::sync::Mutex<isis_runtime::nc_socket::NcDebugView>>> =
        if let Some(port) = debug_port {
            let nc_tmp = NeuralComputer::new(w.clone());
            let view = nc_socket::NcDebugView::from_nc(&nc_tmp);
            let view = std::sync::Arc::new(std::sync::Mutex::new(view));
            let _handle = nc_socket::start_debug_server(port, view.clone());
            eprintln!("Debugger on port {port}");
            Some(view)
        } else {
            None
        };

    // ─── Runtime-framework-driven training loop ───
    //
    // Same shape as learn_ffn but richer state:
    //   (w, opt, grads, workspace, frozen_cereb) ← all per-step mutable
    //   offset, tokens_since_report, total_tokens, dream_counter ← scalar mut
    //
    // Everything goes in RefCells behind the closures. Ctrl+C, logging
    // cadence, checkpoint cadence all come from TrainerLoop.
    use modgrad_training::{TrainerConfig, TrainerLoop, StepReport};
    use std::cell::{Cell, RefCell};

    let grads = RegionalGradients::zeros(&w);
    let workspace = TrainWorkspace::new(&w);
    // Resumed token count comes from the bundle (legacy path: defaults to 0).
    let initial_total_tokens = resumed_meta_ctm.tokens_seen;

    // Bundle the big mutables.  `frozen_cereb` is Option<Box<dyn ...>>; when
    // present every step borrows it mutably for `encode_context_layers`.
    let state = RefCell::new((w, opt, grads, workspace, frozen_cereb));
    let offset = Cell::new(0usize);
    let tokens_since_report = Cell::new(0usize);
    let total_tokens = Cell::new(initial_total_tokens);
    let dream_counter = Cell::new(0u64);
    let tokens_ref: &Vec<usize> = &all_tokens;
    let n_data = all_tokens.len();
    let run_started = std::time::Instant::now();

    eprintln!("\nLearning... (Ctrl+C to save and stop)\n");

    let cfg = TrainerConfig {
        max_steps: usize::MAX,
        log_every: 100,
        save_every: Some(5_000),
        ..Default::default()
    };

    let step_fn = |step_idx: usize| -> Option<StepReport> {
        loop {
            let o = offset.get();
            let end = (o + context_len + 1).min(n_data);
            if end - o < 2 {
                offset.set(0);
                eprintln!("  --- epoch complete ---");
                continue;
            }
            let chunk = &all_tokens[o..end];
            let n = chunk.len() - 1;

            let mut s = state.borrow_mut();
            let (w, opt, grads, workspace, frozen_cereb) = &mut *s;

            grads.zero();
            let mut chunk_loss = 0.0f32;
            let mut chunk_correct = 0usize;

            // Encode full context through frozen cerebellum ONCE (amortised).
            let cereb_cache = if let Some(fc) = frozen_cereb.as_mut() {
                let token_ids: Vec<i64> = chunk.iter().map(|&t| t as i64).collect();
                Some(fc.encode_context_layers(&token_ids))
            } else {
                None
            };

            for pos in 0..n {
                let (loss, pred) = if let (Some(cache), Some(proj)) =
                    (&cereb_cache, &w.cereb_projection) {
                    use modgrad_ctm::cerebellum::blended_hidden_at;
                    let hidden = blended_hidden_at(cache, proj, pos);
                    if hidden.is_empty() {
                        regional_train_token_fast(w, grads, workspace, chunk[pos], chunk[pos + 1])
                    } else {
                        regional_train_token_with_cereb(w, grads, chunk[pos], chunk[pos + 1], &hidden)
                    }
                } else {
                    regional_train_token_fast(w, grads, workspace, chunk[pos], chunk[pos + 1])
                };
                chunk_loss += loss;
                if pred == chunk[pos + 1] { chunk_correct += 1; }
            }

            opt.step(w, grads);
            modgrad_device::kfd::accel::invalidate_cache();

            // Dream phase: replay a random position every 20 steps.
            if step_idx > 0 && step_idx % 20 == 0 && tokens_ref.len() > 10 {
                let dc = dream_counter.get() + 1;
                dream_counter.set(dc);
                let pos = ((dc * 7919) as usize) % (tokens_ref.len() - 9);
                let mut dg = RegionalGradients::zeros(w);
                dream_step(w, &mut dg, tokens_ref[pos], &tokens_ref[pos+1..pos+9], 8, 1.0);
                dg.apply(w, opt.lr * 0.3, 1.0);
            }

            // Debug view snapshot (every 100 steps, cheap inference).
            if step_idx % 100 == 0 {
                if let Some(ref view) = debug_nc {
                    if let Ok(mut guard) = view.try_lock() {
                        guard.history = vec![step_idx];
                        let saved_router = w.router.take();
                        let mut ss = RegionalState::new(w);
                        let obs = w.embed(chunk[n.saturating_sub(1)]);
                        let snap = regional_forward(w, &mut ss, obs);
                        w.router = saved_router;
                        guard.region_activations = snap.region_activations;
                        guard.global_sync = snap.global_sync;
                        guard.exit_lambdas = snap.exit_lambdas;
                        guard.ticks_used = snap.ticks_used;
                    }
                }
            }

            offset.set(o + context_len);
            tokens_since_report.set(tokens_since_report.get() + n);
            total_tokens.set(total_tokens.get() + n as u64);

            let avg_loss = chunk_loss / n as f32;
            let acc = chunk_correct as f32 / n as f32;
            let progress = ((o + context_len) as f32 / n_data as f32).min(1.0);
            let tokens = total_tokens.get();
            return Some(StepReport::new(avg_loss)
                .with_accuracy(acc)
                .with_progress(progress)
                .with_extra("tokens", tokens as f32));
        }
    };

    // Save closure: delegate to the helper — single source of truth
    // for CURRENT_SCHEMA + timestamp_unix + default loss fields.
    let save_fn = || -> Result<(), Box<dyn std::error::Error>> {
        let s = state.borrow();
        modgrad_training::save_training_checkpoint(
            save_path,
            MODEL_KIND_CTM,
            &s.0,
            &s.1,
            s.1.step as u64,
            total_tokens.get(),
            resumed_meta_ctm.elapsed_secs + run_started.elapsed().as_secs(),
        )?;
        Ok(())
    };

    let report = TrainerLoop::new(cfg).run(step_fn, save_fn);
    eprintln!("\nSaved to {save_path}  ({} steps, {} tokens total)",
        report.steps_completed, total_tokens.get());
}

/// Sync diversity diagnostic: measure how different sync patterns are across prompts.
fn load_image_tokens(path: &str) -> Vec<Vec<usize>> {
    // graph types imported at crate level
    use modgrad_codec::vqvae::VqVae;

    let vae = VqVae::new(4096, 64);
    let mut result = Vec::new();

    // Try CIFAR-10 binary format first
    if let Ok(data) = std::fs::read(path) {
        if data.len() > 3073 {
            // CIFAR-10 binary: each record = 1 label + 3072 pixels (32×32×3)
            let n_images = data.len() / 3073;
            eprintln!("  Loading {n_images} CIFAR-10 images from {path}");
            for i in 0..n_images.min(1000) {
                let offset = i * 3073 + 1; // skip label byte
                let pixels: Vec<f32> = data[offset..offset + 3072]
                    .iter().map(|&b| b as f32 / 255.0).collect();
                let codes = vae.tokenize(&pixels);
                result.push(image_codes_to_tokens(&codes));
            }
            return result;
        }
    }

    // Try directory of raw pixel files
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().map(|e| e == "bin" || e == "raw").unwrap_or(false) {
                if let Ok(data) = std::fs::read(&p) {
                    if data.len() >= 3072 {
                        let pixels: Vec<f32> = data[..3072]
                            .iter().map(|&b| b as f32 / 255.0).collect();
                        let codes = VqVae::new(4096, 64).tokenize(&pixels);
                        result.push(image_codes_to_tokens(&codes));
                    }
                }
            }
        }
    }

    eprintln!("  Loaded {} image token sequences from {path}", result.len());
    result
}

/// Load audio WAV files, tokenize with audio codec, return token sequences.
fn load_audio_tokens(path: &str) -> Vec<Vec<usize>> {
    // graph types imported at crate level
    use modgrad_codec::audio_codec::AudioCodec;

    let codec = AudioCodec::new_24khz();
    let mut result = Vec::new();

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().map(|e| e == "wav" || e == "raw").unwrap_or(false) {
                if let Ok(data) = std::fs::read(&p) {
                    // Simple WAV parser: skip 44-byte header, assume 16-bit PCM mono
                    let samples: Vec<f32> = if data.len() > 44 && &data[..4] == b"RIFF" {
                        data[44..].chunks_exact(2).map(|c| {
                            i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0
                        }).collect()
                    } else {
                        // Raw f32 samples
                        data.chunks_exact(4).map(|c| {
                            f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                        }).collect()
                    };

                    if samples.len() > 320 {
                        let codes = codec.tokenize(&samples);
                        // Split long audio into chunks of ~2 seconds (150 codes)
                        for chunk in codes.chunks(150) {
                            result.push(audio_codes_to_tokens(chunk));
                        }
                    }
                }
            }
        }
    }

    eprintln!("  Loaded {} audio token sequences from {path}", result.len());
    result
}

/// Interactive Neural Computer mode.
/// The model IS the running computer — computation, memory, and I/O
/// unified in the CTM's latent runtime state.
fn run_nc(
    checkpoint: &str, temperature: f32, max_tokens: usize,
    audio_in: Option<&str>, camera_in: Option<&str>, camera_fps: f32,
    audio_out: Option<&str>, image_out: Option<&str>,
    debug_port: Option<u16>,
) {
    // graph types imported at crate level
    use isis_runtime::nc_io;

    let nc = if std::path::Path::new(checkpoint).exists() {
        eprintln!("Loading neural computer from {checkpoint}...");
        NeuralComputer::load(checkpoint).expect("failed to load")
    } else {
        eprintln!("No checkpoint at {checkpoint}, creating fresh NC...");
        let cfg = RegionalConfig::eight_region_multimodal(32, 2);
        let w = RegionalWeights::new(cfg);
        NeuralComputer::new(w)
    };
    let mut nc = nc;
    nc.weights.print_summary();

    // Start debug server if requested
    let debug_view: Option<std::sync::Arc<std::sync::Mutex<isis_runtime::nc_socket::NcDebugView>>> =
        if let Some(port) = debug_port {
            use isis_runtime::nc_socket;
            let view = nc_socket::NcDebugView::from_nc(&nc);
            let view = std::sync::Arc::new(std::sync::Mutex::new(view));
            let _handle = nc_socket::start_debug_server(port, view.clone());
            Some(view)
        } else {
            None
        };

    // Helper: update debug view after NC state changes
    let update_debug = |nc: &NeuralComputer, view: &Option<std::sync::Arc<std::sync::Mutex<isis_runtime::nc_socket::NcDebugView>>>| {
        if let Some(v) = view {
            if let Ok(mut guard) = v.try_lock() {
                *guard = isis_runtime::nc_socket::NcDebugView::from_nc(nc);
            }
        }
    };

    // If audio or camera inputs provided, run in streaming mode
    if audio_in.is_some() || camera_in.is_some() {
        let (tx, rx) = std::sync::mpsc::channel();

        // Spawn I/O threads
        let mut handles = Vec::new();
        if let Some(path) = audio_in {
            eprintln!("  Audio input: {path}");
            handles.push(nc_io::audio_input_thread(path, tx.clone()));
        }
        if let Some(path) = camera_in {
            eprintln!("  Camera input: {path} at {camera_fps}fps");
            handles.push(nc_io::camera_input_thread(path, camera_fps, tx.clone()));
        }

        // Keyboard input thread (so text still works alongside audio/camera)
        let tx_kb = tx.clone();
        handles.push(std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                if stdin.read_line(&mut line).unwrap_or(0) == 0 { break; }
                let input = line.trim().to_string();
                if input == "/quit" || input == "/q" {
                    tx_kb.send(nc_io::NcInput::Quit).ok();
                    break;
                }
                if tx_kb.send(nc_io::NcInput::Text(input)).is_err() { break; }
            }
        }));

        drop(tx); // close our copy so rx disconnects when all senders done

        let config = nc_io::NcStreamConfig {
            temperature,
            max_response: max_tokens,
            audio_out: audio_out.map(|s| s.to_string()),
            image_out: image_out.map(|s| s.to_string()),
        };

        nc_io::nc_stream_loop(&mut nc, rx, config);

        for h in handles { h.join().ok(); }
        eprintln!("NC shutdown.");
        return;
    }

    eprintln!("Neural Computer ready. Type text, or commands:");
    eprintln!("  /click <x> <y>     — mouse click at normalized coords");
    eprintln!("  /move <x> <y>      — mouse move");
    eprintln!("  /key <name>        — special key (enter, tab, esc, up, down, left, right)");
    eprintln!("  /ctrl <char>       — ctrl+key combo");
    eprintln!("  /state             — show NC state summary");
    eprintln!("  /save <path>       — save checkpoint");
    eprintln!("  /quit              — exit");
    eprintln!();

    let stdin = std::io::stdin();
    let mut line = String::new();

    loop {
        eprint!("nc> ");
        std::io::Write::flush(&mut std::io::stderr()).ok();
        line.clear();
        if stdin.read_line(&mut line).unwrap_or(0) == 0 { break; }
        let input = line.trim();
        if input.is_empty() { continue; }

        if input.starts_with('/') {
            let parts: Vec<&str> = input.splitn(3, ' ').collect();
            match parts[0] {
                "/click" if parts.len() >= 3 => {
                    let x: f32 = parts[1].parse().unwrap_or(0.5);
                    let y: f32 = parts[2].parse().unwrap_or(0.5);
                    let action = action_click(x, y);
                    let response = nc.act(&action, max_tokens, temperature);
                    print_nc_response(&response);
                }
                "/move" if parts.len() >= 3 => {
                    let x: f32 = parts[1].parse().unwrap_or(0.5);
                    let y: f32 = parts[2].parse().unwrap_or(0.5);
                    let action = action_mouse_move(x, y);
                    let response = nc.act(&action, max_tokens, temperature);
                    print_nc_response(&response);
                }
                "/key" if parts.len() >= 2 => {
                    let key = match parts[1] {
                        "enter" => ACT_KEY_ENTER,
                        "backspace" | "bs" => ACT_KEY_BACKSPACE,
                        "tab" => ACT_KEY_TAB,
                        "esc" | "escape" => ACT_KEY_ESCAPE,
                        "up" => ACT_KEY_UP,
                        "down" => ACT_KEY_DOWN,
                        "left" => ACT_KEY_LEFT,
                        "right" => ACT_KEY_RIGHT,
                        _ => {
                            eprintln!("  unknown key: {}", parts[1]);
                            continue;
                        }
                    };
                    let action = action_key(key);
                    let response = nc.act(&action, max_tokens, temperature);
                    print_nc_response(&response);
                }
                "/ctrl" if parts.len() >= 2 => {
                    let ch = parts[1].as_bytes().first().copied().unwrap_or(b'c');
                    let action = action_modified_key(ACT_KEY_CTRL, ch);
                    let response = nc.act(&action, max_tokens, temperature);
                    print_nc_response(&response);
                }
                "/state" => {
                    eprintln!("  history: {} tokens", nc.history.len());
                    eprintln!("  regions: {}", nc.weights.config.regions.len());
                    eprintln!("  params: {}", nc.weights.n_params());
                    // Show last 20 tokens
                    let tail: Vec<usize> = nc.history.iter().rev().take(20).copied().collect();
                    eprintln!("  last tokens: {:?}", tail.into_iter().rev().collect::<Vec<_>>());
                }
                "/save" if parts.len() >= 2 => {
                    match nc.weights.save(parts[1]) {
                        Ok(_) => eprintln!("  saved to {}", parts[1]),
                        Err(e) => eprintln!("  save failed: {e}"),
                    }
                }
                "/quit" | "/exit" | "/q" => break,
                _ => eprintln!("  unknown command: {input}"),
            }
        } else {
            // Text input — chat mode
            let response = nc.chat(input, max_tokens, temperature);
            if response.is_empty() {
                eprintln!("  (no text response)");
            } else {
                println!("{response}");
            }
        }
        // Update debug view after every interaction
        update_debug(&nc, &debug_view);
    }
    eprintln!("NC shutdown.");
}

fn print_nc_response(tokens: &[usize]) {
    // graph types imported at crate level
    // Decode response tokens into human-readable form
    let mut text_buf = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i];
        match t {
            0..=255 => text_buf.push(t as u8),
            TOKEN_IMG_START => { flush_text(&mut text_buf); eprint!("[img:"); }
            TOKEN_IMG_END => eprint!("]"),
            TOKEN_AUD_START => { flush_text(&mut text_buf); eprint!("[aud:"); }
            TOKEN_AUD_END => eprint!("]"),
            TOKEN_VID_START => { flush_text(&mut text_buf); eprint!("[vid:"); }
            TOKEN_VID_END => eprint!("]"),
            ACT_START => { flush_text(&mut text_buf); eprint!("[act:"); }
            ACT_END => eprint!("]"),
            t if t >= TOKEN_TS_OFFSET && t < TOKEN_TS_OFFSET + TOKEN_TS_COUNT => {
                flush_text(&mut text_buf);
                let secs = (t - TOKEN_TS_OFFSET) as f32 * 0.5;
                eprint!("<{secs:.1}s>");
            }
            t if t >= TOKEN_IMG_OFFSET && t < TOKEN_IMG_OFFSET + TOKEN_IMG_CODES => {
                eprint!("{}", (t - TOKEN_IMG_OFFSET));
                if i + 1 < tokens.len() && tokens[i+1] >= TOKEN_IMG_OFFSET
                    && tokens[i+1] < TOKEN_IMG_OFFSET + TOKEN_IMG_CODES { eprint!(","); }
            }
            t if t >= TOKEN_COORD_OFFSET && t < TOKEN_COORD_OFFSET + TOKEN_COORD_COUNT => {
                let v = (t - TOKEN_COORD_OFFSET) as f32 / 255.0;
                eprint!("{v:.2}");
            }
            _ => eprint!("?{t}"),
        }
        i += 1;
    }
    flush_text(&mut text_buf);
    eprintln!();
}

fn flush_text(buf: &mut Vec<u8>) {
    if !buf.is_empty() {
        eprint!("{}", String::from_utf8_lossy(buf));
        buf.clear();
    }
}

/// Load videos from a directory of subdirectories.
/// Each subdirectory = one video, containing frame files + optional audio.wav.
fn load_video_tokens(path: &str, _fps: f32) -> Vec<Vec<usize>> {
    let result: Vec<Vec<usize>> = Vec::new();

    // Each subdirectory = one video with frame files + optional audio.wav
    // TODO: implement with modgrad_codec when needed
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                eprintln!("  video loading not yet implemented for {}", entry.path().display());
            }
        }
    }

    eprintln!("  Loaded {} video token sequences from {path}", result.len());
    result
}

