//! Maze solving benchmark for CTM.
//!
//! Pipeline: maze pixels → Encoder (VisualCortex) → TokenInput → Brain (CTM) → StepwiseCE
//!
//! Uses the SDK trait system: Encoder + Brain + LossFn compose cleanly.
//! No ad-hoc training code — the same path works for any modality.
//!
//! Usage:
//!   mazes --size 21 --ticks 16 --steps 10000
//!   mazes --size 39 --ticks 32 --steps 50000 --route-len 50

mod maze_gen;
mod wall_probe;

use maze_gen::*;
use modgrad_ctm::config::{CtmConfig, ExitStrategy};
use modgrad_ctm::weights::CtmWeights;
use modgrad_ctm::train::{Ctm, accumulate_gradients};
use modgrad_ctm::graph::{
    RegionalWeights, RegionalGradients, RegionalState,
    RegionalAdamW, RegionalBrain, regional_forward,
};
use modgrad_codec::retina::{LsdConfig, VisualCortex};
use modgrad_codec::genome::Genome;
use modgrad_traits::{Brain, Encoder, LossFn, StepwiseCE, Imagination, TokenInput};

/// Maze action vocabulary: Up / Down / Left / Right / Wait.
/// Task semantics live at the construction site per the SDK neutrality rule —
/// `StepwiseCE` itself just sees `n_classes = 5`.
const MAZE_N_CLASSES: usize = 5;
/// Curriculum lookahead for maze routes: pull the training frontier two
/// cells past the correct prefix each step. Small enough that early-training
/// gradient stays focused on the start of the route.
const MAZE_LOOKAHEAD: usize = 5;

fn maze_loss() -> StepwiseCE {
    StepwiseCE { n_classes: MAZE_N_CLASSES, lookahead: MAZE_LOOKAHEAD }
}

/// Construct a retina for maze-sized input. If `--pretrained-retina
/// PATH` was passed, load weights from that path and reshape the
/// `input_h`/`input_w` fields for this call site's dimensions (Conv2d
/// weights are dimension-independent; only those two fields need to
/// match the actual input). Otherwise fresh `::preserve_spatial`.
fn make_encoder(h: usize, w: usize) -> VisualCortex {
    if let Some(path) = pretrained_retina_path() {
        let mut r = VisualCortex::load(path)
            .unwrap_or_else(|e| panic!("--pretrained-retina: failed to load {path}: {e}"));
        r.input_h = h;
        r.input_w = w;
        eprintln!("Encoder: pretrained retina loaded from {path} (reshape to {h}×{w})");
        r
    } else {
        VisualCortex::preserve_spatial(h, w)
    }
}

/// Dispatch between the full retina encode and the --no-retina
/// bypass. Called from every training/eval step; flag is read once
/// from the global AtomicBool so no parameter plumbing.
fn encode_maybe_bypass(
    encoder: &VisualCortex,
    pixels: &[f32],
    in_h: usize,
    in_w: usize,
) -> TokenInput {
    if retina_bypass_enabled() {
        let (_, _, _, _, _, _, h_tok, w_tok) = encoder.stage_dims();
        let token_dim = encoder.token_dim();
        TokenInput {
            tokens: bypass_tokens(pixels, in_h, in_w, h_tok, w_tok, token_dim),
            n_tokens: h_tok * w_tok,
            token_dim,
        }
    } else {
        encoder.encode(pixels)
    }
}

/// Multi-scale encoding: concatenate V4, V2, V1 streams into a single
/// flat `TokenInput.tokens` buffer that the brain slices per
/// connection via `obs_scale_slice`. Layout matches
/// `Encoder::token_dims()` order: V4 first, then V2, then V1.
fn encode_multiscale_concat(
    encoder: &VisualCortex,
    pixels: &[f32],
) -> TokenInput {
    let multi = encoder.encode_multiscale(pixels);
    let mut tokens = Vec::with_capacity(
        multi.scales.iter().map(|s| s.tokens.len()).sum(),
    );
    for s in &multi.scales {
        tokens.extend_from_slice(&s.tokens);
    }
    // Brain doesn't introspect n_tokens / token_dim from the
    // observation buffer — it uses obs_scale_dims. Pick scale-0
    // (V4) shape as a representative for the TokenInput contract.
    TokenInput {
        n_tokens: multi.scales[0].n_tokens,
        token_dim: multi.scales[0].token_dim,
        tokens,
    }
}

fn main() {
    // Print the backend registry's probe result at startup. With
    // `--features kfd` on gfx1102, this should show `kfd` registered
    // ahead of `cpu`. Without the feature, CPU only — lets users
    // confirm a GPU build actually wired GPU dispatch.
    {
        let reg = modgrad_device::backend::registry();
        let names: Vec<&'static str> = reg.iter().map(|b| b.name()).collect();
        eprintln!("backends registered: [{}]", names.join(", "));
    }

    let args: Vec<String> = std::env::args().collect();
    let mut maze_size = 21usize;
    let mut ticks = 16usize;
    let mut steps = 10000usize;
    let mut route_len = 20usize;
    let mut d_model = 256usize;
    let mut lr = 3e-4f32;
    let mut seed = 42u64;
    let mut adaptive = true;
    let mut batch_size = 8usize;
    let mut imagination = false;
    let mut brain = false;
    let mut rl = false;
    let mut pain_mode = false;
    let mut plural_mode = false;
    let mut csv_mode = false;
    let mut cereb_size = 0usize; // 0 = use preset default
    let mut frozen_cereb = false;
    let mut autoresearch_summary = false;
    let mut budget_secs: Option<u64> = None;
    let mut train_bank_path: Option<String> = None;
    let mut test_bank_path: Option<String> = None;
    let mut hebbian_epochs = 0usize;
    let mut hebbian_samples = 500usize;
    let mut hebbian_lr = 2e-4f32;
    let mut dream_epochs = 0usize;
    let mut dream_sparsity_k = 8usize;
    let mut ood_size = 0usize;
    // --lsd-integration F: if <1.0 AND --dream-epochs>0, route dream
    // pretraining through VisualCortex::lsd with this integration. 1.0
    // falls through to the legacy permanent train_dream path.
    let mut lsd_integration = 1.0f32;
    // --live-viz DIR --live-every N: every N steps, forward-pass a
    // fixed probe maze through the current retina and dump a combined
    // input+V1+V2+V4 PPM panel so `feh --reload 1 DIR/combined.ppm`
    // shows the cortex updating live.
    let mut live_viz_dir: Option<String> = None;
    let mut live_every = 25usize;
    // --sdf: encode BFS wall-distance into pixel luminance instead of
    // flat path=1.0. Same tensor shape; only the input distribution
    // changes. A/B this against baseline to see if the retina was
    // information-bottlenecked by the binary wall/path encoding.
    let mut use_sdf = false;
    // --no-retina: skip the VisualCortex conv stack, feed avg-pooled
    // raw pixels into the brain. Binary A/B on whether the retina is
    // net-positive vs neutral vs net-negative on the task.
    let mut no_retina = false;
    // --substrate-log=PATH: every training step, append one CSV row
    // with substrate observability (cpu freq ratio, max temp,
    // throttle delta, governor, epp) alongside the step's loss/acc.
    // Opt-in so mazes stays pure when this isn't wanted — the only
    // cost when enabled is a handful of small sysfs reads per step.
    let mut substrate_log: Option<String> = None;
    // --pretrained-retina PATH: load V1/V2/V4 weights pretrained on
    // natural images (e.g. STL-10 via `pretrain_retina`) instead of
    // random init. Conv2d weights are input-size-independent, so the
    // pretrained weights drop in at any maze_size.
    let mut pretrained_retina: Option<String> = None;
    // --ghl-retina: enable GHL (Global-guided Hebbian Learning) updates
    // on V2/V4 during training (arXiv:2601.21367). Requires --brain
    // because we need `regional_train_step_generic`'s d_observation
    // return to backprop into the retina. When off, retina weights
    // are whatever we started with (random or pretrained).
    let mut ghl_retina = false;
    let mut ghl_lr = 1e-3f32;
    let mut ghl_tau = 1.0f32;
    // --topdown: enable brain→retina control flow. Pass 1 runs a
    // regional_forward to populate state.region_outputs; the
    // `attention` region's activated state is then deterministically
    // projected to a per-spatial-position gain mask (sigmoid over
    // tiled attention vector). Pass 2 re-encodes pixels with the gain
    // applied to V4 tokens and is the pass that backprops. Demonstrates
    // "brain's executive attention steers perception" using the
    // existing attention region rather than adding new top-down
    // weights. --brain required.
    let mut topdown = false;
    // --topdown-alpha F: scalar gate for the top-down gain mask. The
    // gate `gain = 1 + α · (sigmoid(attn) − 0.5)` lives on the attn→V*
    // pathway. α is read from `Genome::pathway_gates.topdown_alpha`
    // (defaults to 0 → identity), but this CLI flag overrides the
    // genome-derived value so users can A/B test without rebuilding the
    // genome. `Some(0.0)` makes top-down a no-op even with --topdown
    // on; `Some(1.0)` gives the original full-strength sigmoid.
    let mut topdown_alpha_override: Option<f32> = None;

    // --export DIR: after training, write the trained brain to DIR in the
    // wasm engine's JSON format (`brain_solver_weights.json` +
    // `brain_solver_reference.json`). Brain mode only; no effect on a
    // single-CTM run. Does NOT change training — pure post-training save.
    let mut export_dir: Option<String> = None;

    // --multiscale: brain regions wire to V1 / V2 / V4 directly
    // instead of all consuming V4. INPUT/MOTOR get V4 (categorical),
    // ATTENTION gets V2 (mid-contour), MOTOR/CEREBELLUM get V1
    // (fine spatial). Mirrors real cortical wiring where V1 has
    // direct projections to many areas, not only via the V2→V4 path.
    let mut multiscale = false;

    // --wall-probe: run the wall-probe gate harness (no training). Builds
    // the brain with the spatial readout enabled on the OUTPUT region and
    // measures whether walls are linearly decodable from global_sync (A)
    // vs the spatial readout (C), with a pos-encoding ablation.
    let mut wall_probe = false;
    let mut probe_samples = 2000usize;
    // --vin-readout: add the VIN (Value-Iteration-Network) ego-centric
    // readout as an alternative path in the wall-probe (read move at the
    // agent's own cell, no global pooling). Keeps the legacy readout intact.
    let mut vin_readout = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => { maze_size = args[i+1].parse().unwrap(); i += 2; }
            "--ticks" => { ticks = args[i+1].parse().unwrap(); i += 2; }
            "--steps" => { steps = args[i+1].parse().unwrap(); i += 2; }
            "--route-len" => { route_len = args[i+1].parse().unwrap(); i += 2; }
            "--d-model" => { d_model = args[i+1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i+1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i+1].parse().unwrap(); i += 2; }
            "--batch" => { batch_size = args[i+1].parse().unwrap(); i += 2; }
            "--no-adaptive" => { adaptive = false; i += 1; }
            "--imagination" => { imagination = true; i += 1; }
            "--brain" => { brain = true; i += 1; }
            "--rl" => { rl = true; brain = true; i += 1; }
            "--pain" => { pain_mode = true; i += 1; }
            "--plural" => { plural_mode = true; pain_mode = true; i += 1; }
            "--csv" => { csv_mode = true; i += 1; }
            "--cereb-size" => { cereb_size = args[i+1].parse().unwrap(); i += 2; }
            "--frozen-cereb" => { frozen_cereb = true; i += 1; }
            "--autoresearch-summary" => { autoresearch_summary = true; i += 1; }
            "--budget" => { budget_secs = Some(args[i+1].parse().unwrap()); i += 2; }
            "--load-mazes" => { train_bank_path = Some(args[i+1].clone()); i += 2; }
            "--load-mazes-test" => { test_bank_path = Some(args[i+1].clone()); i += 2; }
            "--hebbian-epochs" => { hebbian_epochs = args[i+1].parse().unwrap(); i += 2; }
            "--hebbian-samples" => { hebbian_samples = args[i+1].parse().unwrap(); i += 2; }
            "--hebbian-lr" => { hebbian_lr = args[i+1].parse().unwrap(); i += 2; }
            "--dream-epochs" => { dream_epochs = args[i+1].parse().unwrap(); i += 2; }
            "--dream-sparsity-k" => { dream_sparsity_k = args[i+1].parse().unwrap(); i += 2; }
            "--ood-size" => { ood_size = args[i+1].parse().unwrap(); i += 2; }
            "--lsd-integration" => { lsd_integration = args[i+1].parse().unwrap(); i += 2; }
            "--live-viz" => { live_viz_dir = Some(args[i+1].clone()); i += 2; }
            "--live-every" => { live_every = args[i+1].parse().unwrap(); i += 2; }
            "--sdf" => { use_sdf = true; i += 1; }
            "--no-retina" => { no_retina = true; i += 1; }
            "--substrate-log" => { substrate_log = Some(args[i+1].clone()); i += 2; }
            "--pretrained-retina" => { pretrained_retina = Some(args[i+1].clone()); i += 2; }
            "--ghl-retina" => { ghl_retina = true; i += 1; }
            "--ghl-lr" => { ghl_lr = args[i+1].parse().unwrap(); i += 2; }
            "--ghl-tau" => { ghl_tau = args[i+1].parse().unwrap(); i += 2; }
            "--topdown" => { topdown = true; i += 1; }
            "--topdown-alpha" => { topdown_alpha_override = Some(args[i+1].parse().unwrap()); i += 2; }
            "--multiscale" => { multiscale = true; i += 1; }
            "--wall-probe" => { wall_probe = true; i += 1; }
            "--probe-samples" => { probe_samples = args[i+1].parse().unwrap(); i += 2; }
            "--vin-readout" => { vin_readout = true; i += 1; }
            "--export" => { export_dir = Some(args[i+1].clone()); i += 2; }
            "--help" | "-h" => {
                eprintln!("Usage: mazes [--size N] [--ticks N] [--steps N] [--route-len N]");
                eprintln!("             [--d-model N] [--lr F] [--batch N] [--no-adaptive]");
                eprintln!("             [--imagination] [--brain] [--pain] [--plural] [--csv]");
                eprintln!("             [--cereb-size N] [--frozen-cereb]");
                eprintln!("             [--dream-epochs N] [--lsd-integration F (default 1.0)]");
                return;
            }
            _ => { i += 1; }
        }
    }

    let maze_size = maze_size | 1;

    // Pixel rendering mode is a one-time process-wide setting. The
    // helper `render_input(&maze)` dispatches to `render_maze_sdf` if
    // this flag is on, otherwise the standard `render_maze`.
    set_render_mode_sdf(use_sdf);
    set_retina_bypass(no_retina);
    set_pretrained_retina_path(pretrained_retina.clone());
    if use_sdf {
        eprintln!("Input encoding: SDF (wall-distance normalised into luminance)");
    }
    if no_retina {
        eprintln!("Encoder: BYPASSED (avg-pool raw pixels → brain; no conv stack)");
    }

    // Optionally load external maze banks (e.g., Sakana PNGs exported via the
    // /tmp/export_sakana_mazes.py script). When loaded, the training loop
    // samples from the bank instead of generating fresh mazes.
    let train_bank: Option<Vec<Maze>> = train_bank_path.as_deref().map(|p| {
        let m = load_maze_bank(p).expect("failed to load train maze bank");
        eprintln!("Loaded {} train mazes from {}", m.len(), p);
        m
    });
    let test_bank: Option<Vec<Maze>> = test_bank_path.as_deref().map(|p| {
        let m = load_maze_bank(p).expect("failed to load test maze bank");
        eprintln!("Loaded {} test mazes from {}", m.len(), p);
        m
    });
    // If a bank is loaded and matches a size, override --size so shapes line up.
    let maze_size = if let Some(ref b) = train_bank {
        b[0].grid_size
    } else { maze_size };

    if wall_probe {
        wall_probe::run_wall_probe(maze_size, ticks, probe_samples, seed, vin_readout);
        return;
    }
    if rl {
        run_rl(maze_size, ticks, steps, route_len, lr, seed, export_dir);
        return;
    }
    if brain {
        run_brain(maze_size, ticks, steps, route_len, lr, seed, batch_size, imagination, pain_mode, plural_mode, csv_mode, cereb_size, frozen_cereb, autoresearch_summary, budget_secs, train_bank, test_bank, ghl_retina, ghl_lr, ghl_tau, topdown, topdown_alpha_override, multiscale, export_dir);
        return;
    }

    eprintln!("Maze benchmark (single CTM): size={maze_size} ticks={ticks} d_model={d_model} \
               route_len={route_len} batch={batch_size}");

    // ── Encoder: visual retina → spatial tokens ──
    let mut encoder = make_encoder(maze_size, maze_size);
    let token_dim = encoder.token_dim();

    // Probe token count with dummy image
    let dummy = vec![0.0f32; 3 * maze_size * maze_size];
    let probe = encoder.encode(&dummy);
    eprintln!("Encoder: {maze_size}×{maze_size} pixels → {} spatial tokens × {token_dim}-dim",
        probe.n_tokens);

    // ── Optional Hebbian pretraining of cortex (V2, V4) ──
    // V1 stays fixed (evolved). V2/V4 learn unsupervised sparse-coding on
    // the task pixel distribution before the brain sees anything.
    if hebbian_epochs > 0 {
        let t0 = std::time::Instant::now();
        let mut pretrain_rng = MazeRng::new(seed ^ 0xA5A5);
        let mut bank: Vec<Vec<f32>> = Vec::with_capacity(hebbian_samples);
        while bank.len() < hebbian_samples {
            let m = generate_maze(maze_size, &mut pretrain_rng);
            if m.path_length < 3 { continue; }
            bank.push(render_input(&m));
        }
        let refs: Vec<&[f32]> = bank.iter().map(|v| v.as_slice()).collect();
        eprintln!("Hebbian pretraining: {hebbian_samples} mazes × {hebbian_epochs} epochs (lr={hebbian_lr})");
        encoder.train_hebbian(&refs, hebbian_epochs, hebbian_lr);
        eprintln!("Hebbian pretraining done in {:.1}s", t0.elapsed().as_secs_f32());
    }

    // ── Optional dream pretraining (Hoel 2021) ──
    // Delegates to VisualCortex::train_dream: synthesize N pseudo-images
    // via V4→V2→V1 adjoint projection, then Hebbian-train V2/V4 on those.
    // Combine with --hebbian-epochs to first seed with a real-data prior
    // then refine with dream augmentation.
    if dream_epochs > 0 {
        let t0 = std::time::Instant::now();
        if lsd_integration < 1.0 {
            // LSD path: dream synthesis with a wear-off window. The cortex
            // tours the dream manifold then interpolates back toward pre-trip
            // weights by (1 - integration). integration=0.0 fully reverts,
            // integration=1.0 would equal the legacy train_dream branch.
            eprintln!("LSD pretraining: {hebbian_samples} dreams × {dream_epochs} epochs \
                       (lr={hebbian_lr}, sparsity_k={dream_sparsity_k}, integration={lsd_integration:.2})");
            let report = encoder.lsd(LsdConfig {
                dose: dream_sparsity_k,
                duration: hebbian_samples,
                epochs: dream_epochs,
                lr: hebbian_lr,
                plasticity_boost: 1.0,
                integration: lsd_integration,
                seed,
            });
            eprintln!("  trip: peak v2={:.4} v4={:.4} / post v2={:.4} v4={:.4}",
                report.peak_v2_delta, report.peak_v4_delta,
                report.post_v2_delta, report.post_v4_delta);
        } else {
            eprintln!("Dream pretraining: {hebbian_samples} synthesized × {dream_epochs} epochs \
                       (lr={hebbian_lr}, sparsity_k={dream_sparsity_k})");
            encoder.train_dream(hebbian_samples, dream_epochs, hebbian_lr,
                                dream_sparsity_k, seed);
        }
        eprintln!("Dream pretraining done in {:.1}s", t0.elapsed().as_secs_f32());
    }

    // ── Brain: CTM ──
    let out_dims = route_len * N_DIRECTIONS;
    let exit_strategy = if adaptive {
        ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.99 }
    } else {
        ExitStrategy::None
    };

    let cfg = CtmConfig {
        iterations: ticks,
        d_model,
        d_input: token_dim.min(d_model / 2).max(32),
        heads: 4,
        n_synch_out: d_model,
        n_synch_action: d_model,
        synapse_depth: 4,
        memory_length: 16,
        deep_nlms: true,
        memory_hidden_dims: 8,
        out_dims,
        n_random_pairing_self: 0,
        min_width: 16,
        exit_strategy,
        collect_trajectories: false,
        spatial: None,
    };

    let mut w = CtmWeights::new(cfg, token_dim);
    eprintln!("Brain: d_model={d_model} d_input={} ticks={ticks} out={out_dims} params={}",
        w.config.d_input, w.n_params());

    // ── Loss: stepwise CE with auto-curriculum (maze-configured) ──
    let base_loss = maze_loss();
    let imagination_loss = Imagination::new(maze_loss());
    let loss_fn: &dyn LossFn<Target = [usize]> = if imagination {
        eprintln!("Loss: Imagination<StepwiseCE> (ratio=0.5)");
        &imagination_loss
    } else {
        eprintln!("Loss: StepwiseCE (Sakana baseline)");
        &base_loss
    };

    // ── Training ──
    let t_train_start = std::time::Instant::now();
    let budget = budget_secs.map(std::time::Duration::from_secs);
    let mut rng = MazeRng::new(seed);
    let mut loss_history = Vec::new();
    let mut acc_history = Vec::new();
    let mut step = 0usize;

    // Substrate telemetry log: one CSV row per training step. Records
    // (step, loss, acc, freq_ratio, freq_khz, temp_c, throttle_delta,
    // governor, epp). Opt-in via --substrate-log=PATH. Dataset for a
    // later learned clock controller — see crates/modgrad-substrate.
    use std::io::Write as _;
    let (mut substrate_writer, mut prev_throttle) = if let Some(path) = &substrate_log {
        let mut f = std::fs::File::create(path)
            .expect("--substrate-log: failed to open file");
        writeln!(
            f,
            "step,loss,acc,freq_ratio,mean_freq_khz,max_temp_c,throttle_delta,governor,epp"
        )
        .expect("--substrate-log: write header");
        let baseline = modgrad_substrate::Snapshot::take()
            .map(|s| s.throttle_total())
            .unwrap_or(0);
        (Some(f), baseline)
    } else {
        (None, 0u64)
    };

    // Live-viz setup: freeze one probe maze, take its pixels, make a
    // directory. At every --live-every step, we'll re-render the
    // current retina's response to this fixed probe so `feh --reload 1`
    // shows drift. Same probe across steps so visual change ↔ weight change.
    let (live_probe_pixels, live_probe_h, live_probe_w, live_dir) = if let Some(dir) = &live_viz_dir {
        std::fs::create_dir_all(dir).expect("live-viz out dir");
        let mut probe_rng = MazeRng::new(seed ^ 0xC0FFEE);
        let probe = loop {
            let m = generate_maze(maze_size, &mut probe_rng);
            if m.path_length >= 3 { break m; }
        };
        let px = render_input(&probe);
        (Some(px), maze_size, maze_size, Some(dir.clone()))
    } else {
        (None, 0, 0, None)
    };

    while step < steps {
        // Budget check — stops at the batch boundary, not mid-batch,
        // to keep the gradient update atomic.
        if let Some(d) = budget {
            if t_train_start.elapsed() >= d {
                eprintln!("(--budget {}s exhausted at step {step})", d.as_secs());
                break;
            }
        }
        let mut batch_grads = Ctm::zero_gradients(&w);
        let mut batch_loss = 0.0f32;
        let mut batch_correct = 0usize;
        let mut batch_total = 0usize;
        let mut batch_valid = 0usize;

        for _ in 0..batch_size {
            let maze = generate_maze(maze_size, &mut rng);
            if maze.path_length < 3 { continue; }

            // Encode: pixels → TokenInput (bypasses retina if --no-retina)
            let pixels = render_input(&maze);
            let input = encode_maybe_bypass(&encoder, &pixels, maze_size, maze_size);

            // Truncate/pad route
            let mut route = maze.route.clone();
            route.truncate(route_len);
            while route.len() < route_len { route.push(DIR_WAIT); }

            // Forward (Brain trait)
            let state = Ctm::init_state(&w);
            let (output, _state, cache) = Ctm::forward_cached(&w, state, &input);

            // Loss (LossFn trait)
            let (loss, d_preds) = loss_fn.compute(
                &output.predictions, &output.certainties, &route);

            // Backward (Brain trait)
            let sample_grads = Ctm::backward(&w, cache, &d_preds);

            // Accumulate
            accumulate_gradients(&mut batch_grads, &sample_grads);
            batch_loss += loss;
            batch_valid += 1;

            // Accuracy: check last tick prediction per position
            if let Some(pred) = output.predictions.last() {
                for pos in 0..route_len.min(maze.path_length) {
                    let offset = pos * N_DIRECTIONS;
                    let p = pred[offset..offset + N_DIRECTIONS].iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i).unwrap_or(0);
                    if p == route[pos] { batch_correct += 1; }
                    batch_total += 1;
                }
            }
        }

        if batch_valid > 0 {
            // Apply (Brain trait)
            let effective_lr = lr / batch_valid as f32;
            Ctm::apply_gradients(&mut w, &mut batch_grads, effective_lr, 5.0);

            let avg_loss = batch_loss / batch_valid as f32;
            let avg_acc = batch_correct as f32 / batch_total.max(1) as f32;
            loss_history.push(avg_loss);
            acc_history.push(avg_acc);
        }

        step += 1;

        // Substrate telemetry — one row per step when --substrate-log
        // is set. Kept close to the step boundary so throttle_delta
        // attributes cleanly to what this step did. Errors reading
        // sysfs degrade to NaN/0 fields rather than aborting training.
        if let Some(ref mut f) = substrate_writer {
            let loss = loss_history.last().copied().unwrap_or(f32::NAN);
            let acc = acc_history.last().copied().unwrap_or(f32::NAN);
            let (ratio, freq_khz, temp, throttle_delta, governor) =
                if let Ok(s) = modgrad_substrate::Snapshot::take() {
                    let t = s.throttle_total();
                    let delta = t.saturating_sub(prev_throttle);
                    prev_throttle = t;
                    (
                        s.mean_freq_ratio().unwrap_or(f32::NAN),
                        s.mean_freq_khz().unwrap_or(0),
                        s.max_temp_c().unwrap_or(f32::NAN),
                        delta,
                        s.governor.clone(),
                    )
                } else {
                    (f32::NAN, 0, f32::NAN, 0, String::from("error"))
                };
            let epp = modgrad_substrate::cpu_energy_performance_preference(0)
                .unwrap_or_else(|_| String::from("n/a"));
            let _ = writeln!(
                f,
                "{step},{loss:.6},{acc:.6},{ratio:.3},{freq_khz},{temp:.1},{throttle_delta},{governor},{epp}"
            );
        }

        if step % 100 == 0 || step == steps {
            let window = 50.min(loss_history.len());
            if window > 0 {
                let smooth_loss: f32 = loss_history[loss_history.len()-window..].iter().sum::<f32>() / window as f32;
                let smooth_acc: f32 = acc_history[acc_history.len()-window..].iter().sum::<f32>() / window as f32;
                eprintln!("step {step:5}: loss={smooth_loss:.3} route_acc={:.1}%",
                    smooth_acc * 100.0);
            }
        }

        // Live viz: forward the probe maze through current retina and
        // dump combined input+V1+V2+V4 PPM. Fixed spatial probe, so any
        // visual change is weight change. feh --reload 1 picks up the
        // new file automatically.
        if let (Some(dir), Some(px)) = (&live_dir, &live_probe_pixels) {
            if step % live_every == 0 {
                let path = format!("{}/combined.ppm", dir);
                if let Err(e) = live_viz::dump_combined(&encoder, px, live_probe_h, live_probe_w, &path) {
                    eprintln!("live_viz write failed: {e}");
                }
            }
        }
    }

    // ── Evaluation ──
    let training_seconds = t_train_start.elapsed().as_secs_f32();
    eprintln!("\n--- Evaluation (200 mazes, size={maze_size}) ---");
    let id_stats = eval(&w, &encoder, loss_fn, maze_size, route_len, seed + 999,
         autoresearch_summary, training_seconds, step, w.n_params());

    // ── OOD evaluation: larger mazes than training, same brain weights ──
    // The Hoel generalization test: if regularization (dreams, etc.) helps,
    // the gap between in-distribution and OOD accuracy should shrink.
    if ood_size > 0 && ood_size != maze_size {
        let ood_size = ood_size | 1;
        eprintln!("\n--- OOD Evaluation (200 mazes, size={ood_size}) ---");
        let ood_encoder = make_encoder(ood_size, ood_size);
        let ood_stats = eval(&w, &ood_encoder, loss_fn, ood_size, route_len, seed + 1999,
             false, training_seconds, step, w.n_params());
        eprintln!("\n--- Generalization gap (ID={maze_size} → OOD={ood_size}) ---");
        eprintln!("First step:  ID={:.1}% OOD={:.1}% gap={:.1} pp",
            id_stats.first_step_acc * 100.0,
            ood_stats.first_step_acc * 100.0,
            (id_stats.first_step_acc - ood_stats.first_step_acc) * 100.0);
        eprintln!("Per-step:    ID={:.1}% OOD={:.1}% gap={:.1} pp",
            id_stats.per_step_acc * 100.0,
            ood_stats.per_step_acc * 100.0,
            (id_stats.per_step_acc - ood_stats.per_step_acc) * 100.0);
        eprintln!("Prefix len:  ID={:.2} OOD={:.2} gap={:.2}",
            id_stats.avg_prefix, ood_stats.avg_prefix,
            id_stats.avg_prefix - ood_stats.avg_prefix);
    }
}

// ═══════════════════════════════════════════════════════════════
// Live visualisation: render retina forward-pass for one probe maze
// as a single combined PPM so `feh --reload 1 /path/combined.ppm`
// shows the cortex updating during training. Inline so mazes stays
// standalone.
// ═══════════════════════════════════════════════════════════════

mod live_viz {
    use modgrad_codec::retina::VisualCortex;
    use std::io::Write;

    fn leaky_relu(x: &mut [f32]) {
        for v in x.iter_mut() { if *v < 0.0 { *v *= 0.1; } }
    }

    fn layer_norm_map(data: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; h * w];
        for c in 0..channels {
            for i in 0..h * w { m[i] += data[c * h * w + i].powi(2); }
        }
        for v in m.iter_mut() { *v = v.sqrt(); }
        m
    }

    fn heatmap_rgb(vals: &[f32], h: usize, w: usize, us: usize) -> Vec<u8> {
        let (mn, mx) = vals.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
            |(a, b), &v| (a.min(v), b.max(v)));
        let span = (mx - mn).max(1e-6);
        let oh = h * us;
        let ow = w * us;
        let mut out = vec![0u8; 3 * oh * ow];
        for y in 0..h {
            for x in 0..w {
                let v = (vals[y * w + x] - mn) / span;
                let r = (v * 255.0) as u8;
                let b = ((1.0 - v) * 255.0) as u8;
                let g = ((0.5 - (v - 0.5).abs()) * 200.0) as u8;
                for dy in 0..us {
                    for dx in 0..us {
                        let off = ((y * us + dy) * ow + (x * us + dx)) * 3;
                        out[off] = r;
                        out[off + 1] = g;
                        out[off + 2] = b;
                    }
                }
            }
        }
        out
    }

    fn input_rgb(px: &[f32], h: usize, w: usize, us: usize) -> Vec<u8> {
        let oh = h * us;
        let ow = w * us;
        let mut out = vec![0u8; 3 * oh * ow];
        for y in 0..h {
            for x in 0..w {
                let r = (px[y * w + x] * 255.0).clamp(0.0, 255.0) as u8;
                let g = (px[h * w + y * w + x] * 255.0).clamp(0.0, 255.0) as u8;
                let b = (px[2 * h * w + y * w + x] * 255.0).clamp(0.0, 255.0) as u8;
                for dy in 0..us {
                    for dx in 0..us {
                        let off = ((y * us + dy) * ow + (x * us + dx)) * 3;
                        out[off] = r;
                        out[off + 1] = g;
                        out[off + 2] = b;
                    }
                }
            }
        }
        out
    }

    fn write_ppm_atomic(path: &str, rgb: &[u8], h: usize, w: usize) -> std::io::Result<()> {
        // Write to .tmp then rename — feh never sees a half-written file.
        let tmp = format!("{path}.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            writeln!(f, "P6\n{w} {h}\n255")?;
            f.write_all(rgb)?;
        }
        std::fs::rename(&tmp, path)
    }

    /// Forward `pixels` through `retina` and dump one combined panel to `path`.
    pub fn dump_combined(
        retina: &VisualCortex,
        pixels: &[f32],
        in_h: usize,
        in_w: usize,
        path: &str,
    ) -> std::io::Result<()> {
        let (mut v1, h1, w1) = retina.v1.forward(pixels, 1, in_h, in_w);
        leaky_relu(&mut v1);
        let (mut v2, h2, w2) = retina.v2.forward(&v1, 1, h1, w1);
        leaky_relu(&mut v2);
        let (mut v4, h4, w4) = retina.v4.forward(&v2, 1, h2, w2);
        leaky_relu(&mut v4);

        let us_in = 16usize;
        let panel_h = in_h * us_in;
        let us1 = (panel_h / h1).max(1);
        let us2 = (panel_h / h2).max(1);
        let us4 = (panel_h / h4).max(1);

        let input_px = input_rgb(pixels, in_h, in_w, us_in);
        let v1_px = heatmap_rgb(&layer_norm_map(&v1, retina.v1.out_channels, h1, w1), h1, w1, us1);
        let v2_px = heatmap_rgb(&layer_norm_map(&v2, retina.v2.out_channels, h2, w2), h2, w2, us2);
        let v4_px = heatmap_rgb(&layer_norm_map(&v4, retina.v4.out_channels, h4, w4), h4, w4, us4);

        let iw = in_w * us_in;
        let w1p = w1 * us1;
        let w2p = w2 * us2;
        let w4p = w4 * us4;
        let h1p = h1 * us1;
        let h2p = h2 * us2;
        let h4p = h4 * us4;
        let border = 6usize;
        let total_w = iw + w1p + w2p + w4p + 5 * border;
        let total_h = panel_h + 2 * border;
        let mut canvas = vec![40u8; 3 * total_h * total_w];

        let blit = |canvas: &mut Vec<u8>, src: &[u8], sh: usize, sw: usize, ox: usize| {
            for y in 0..sh {
                for x in 0..sw {
                    let s = (y * sw + x) * 3;
                    let d = ((border + y) * total_w + (ox + x)) * 3;
                    canvas[d] = src[s];
                    canvas[d + 1] = src[s + 1];
                    canvas[d + 2] = src[s + 2];
                }
            }
        };

        let mut ox = border;
        blit(&mut canvas, &input_px, panel_h, iw, ox); ox += iw + border;
        blit(&mut canvas, &v1_px,    h1p,    w1p, ox); ox += w1p + border;
        blit(&mut canvas, &v2_px,    h2p,    w2p, ox); ox += w2p + border;
        blit(&mut canvas, &v4_px,    h4p,    w4p, ox);

        write_ppm_atomic(path, &canvas, total_h, total_w)
    }
}

/// Eval summary returned by `eval` so callers can compute gaps between
/// in-distribution and out-of-distribution runs (Hoel generalization test).
#[derive(Debug, Clone, Copy)]
struct EvalStats {
    first_step_acc: f32,
    per_step_acc: f32,
    avg_prefix: f32,
}

fn eval(
    w: &CtmWeights, encoder: &VisualCortex, _loss_fn: &dyn LossFn<Target = [usize]>,
    maze_size: usize, route_len: usize, seed: u64,
    autoresearch_summary: bool, training_seconds: f32,
    num_steps: usize, num_params: usize,
) -> EvalStats {
    let t_eval_start = std::time::Instant::now();
    let mut rng = MazeRng::new(seed);
    let n_eval = 200;
    let mut first_correct = 0usize;
    let mut route_correct = 0usize;
    let mut route_total = 0usize;
    let mut prefix_lengths = Vec::new();
    let mut total_ticks = 0usize;
    let mut n_valid = 0usize;

    for _ in 0..n_eval {
        let maze = generate_maze(maze_size, &mut rng);
        if maze.path_length < 3 { continue; }

        let pixels = render_input(&maze);
        let input = encode_maybe_bypass(encoder, &pixels, maze_size, maze_size);

        let mut route = maze.route.clone();
        route.truncate(route_len);
        while route.len() < route_len { route.push(DIR_WAIT); }

        let state = Ctm::init_state(w);
        let (output, _state) = Ctm::forward(w, state, &input);
        total_ticks += output.predictions.len();

        let pred = output.predictions.last().unwrap();

        // First step
        let first_logits = &pred[0..N_DIRECTIONS];
        let first_pred = first_logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if first_pred == route[0] { first_correct += 1; }

        // Correct prefix length
        let eval_len = route_len.min(maze.path_length);
        let mut prefix = 0usize;
        for pos in 0..eval_len {
            let offset = pos * N_DIRECTIONS;
            let logits = &pred[offset..offset + N_DIRECTIONS];
            let p = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if p == route[pos] {
                route_correct += 1;
                if pos == prefix { prefix += 1; }
            }
            route_total += 1;
        }
        prefix_lengths.push(prefix);
        n_valid += 1;
    }

    let avg_prefix = prefix_lengths.iter().sum::<usize>() as f32 / n_valid.max(1) as f32;
    let avg_ticks = total_ticks as f32 / n_valid.max(1) as f32;

    let first_step_acc = first_correct as f32 / n_valid.max(1) as f32;
    eprintln!("First step acc:     {first_correct}/{n_valid} ({:.1}%)",
        first_step_acc * 100.0);
    eprintln!("Per-step acc:       {route_correct}/{route_total} ({:.1}%)",
        if route_total > 0 { route_correct as f32 / route_total as f32 * 100.0 } else { 0.0 });
    eprintln!("Avg correct prefix: {avg_prefix:.1} steps (of {route_len})");
    eprintln!("Avg ticks used:     {avg_ticks:.1}");

    // Autoresearch contract: grep-parseable summary block on stderr so the
    // same driving-agent workflow that iterates FFN on val_bpb can iterate
    // on maze routing. `val_bpb` is the field name the agent already looks
    // for; we map it to `1 - first_step_acc` (lower-is-better, in [0,1],
    // 0 = perfect routing). Not literally bits-per-byte, but the agent
    // doesn't care about the underlying unit — only the direction.
    if autoresearch_summary {
        let total_seconds = training_seconds + t_eval_start.elapsed().as_secs_f32();
        modgrad_training::AutoresearchSummary {
            val_bpb: 1.0 - first_step_acc,
            training_seconds,
            total_seconds,
            peak_vram_mb: 0.0,
            mfu_percent: 0.0,
            total_tokens_m: 0.0,
            num_steps: num_steps as u64,
            num_params_m: (num_params as f32) / 1.0e6,
        }.print();
        eprintln!("  (task=mazes size={maze_size} route_len={route_len}, \
                   val_bpb = 1 - first_step_acc = {:.6})",
                   1.0 - first_step_acc);
    }

    let per_step_acc = if route_total > 0 {
        route_correct as f32 / route_total as f32
    } else { 0.0 };
    EvalStats {
        first_step_acc,
        per_step_acc,
        avg_prefix,
    }
}

// ═══════════════════════════════════════════════════════════════
// CLOSED-LOOP REINFORCE MODE  (--rl)
//
// The behaviour-cloning path (run_brain) shows the agent ONE image with
// itself pinned at the start, and trains it to copy the BFS-optimal route.
// It never sees itself anywhere else, never acts, never learns from a
// mistake — so it memorises one route and collapses to a down/right prior.
//
// This path closes the loop: the agent ACTS in the maze, the board is
// re-rendered with it at its new cell, and learning is driven by OUTCOME
// (reach goal = +1, bump a wall = penalty) via REINFORCE — policy gradient
// through the brain's own backward. No expert, no fixed start. It learns
// the rule ("step to the open neighbour toward the goal") that generalises.
// ═══════════════════════════════════════════════════════════════

fn rl_softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum::<f32>().max(1e-9);
    for v in &mut e { *v /= s; }
    e
}

fn rl_sample(probs: &[f32], rng: &mut MazeRng) -> usize {
    let u = rng.range(1_000_000) as f32 / 1_000_000.0;
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc { return i; }
    }
    probs.len().saturating_sub(1)
}

/// BFS distance-to-goal for every open cell (i32::MAX = wall/unreachable).
/// Used only as a dense reward *potential* — the agent is rewarded for getting
/// closer, but still has to discover HOW. It's an outcome signal, not a move label.
fn rl_dist(maze: &Maze) -> Vec<i32> {
    let s = maze.grid_size;
    let mut d = vec![i32::MAX; s * s];
    let (gr, gc) = maze.end;
    d[gr * s + gc] = 0;
    let mut q = std::collections::VecDeque::new();
    q.push_back((gr, gc));
    let dirs = [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)];
    while let Some((r, c)) = q.pop_front() {
        for (dr, dc) in dirs {
            let (nr, nc) = (r as i64 + dr, c as i64 + dc);
            if nr < 0 || nr >= s as i64 || nc < 0 || nc >= s as i64 { continue; }
            let (nr, nc) = (nr as usize, nc as usize);
            if maze.grid[nr * s + nc] || d[nr * s + nc] != i32::MAX { continue; }
            d[nr * s + nc] = d[r * s + c] + 1;
            q.push_back((nr, nc));
        }
    }
    d
}

/// Environment transition with dense reward shaping. Returns (new_pos, reward, done).
/// DIR: U D L R WAIT. Reward = progress toward goal (Δdistance) − small step cost;
/// wall/edge bump is penalised; reaching the goal pays a terminal bonus.
fn rl_step(maze: &Maze, pos: (usize, usize), action: usize, dist: &[i32]) -> ((usize, usize), f32, bool) {
    let sz = maze.grid_size;
    let s = sz as i64;
    let (r, c) = (pos.0 as i64, pos.1 as i64);
    let (nr, nc) = match action {
        0 => (r - 1, c),
        1 => (r + 1, c),
        2 => (r, c - 1),
        3 => (r, c + 1),
        _ => return (pos, -0.3, false), // WAIT — wasted step
    };
    if nr < 0 || nr >= s || nc < 0 || nc >= s {
        return (pos, -0.3, false); // off the board — bump
    }
    let (nr, nc) = (nr as usize, nc as usize);
    if maze.grid[nr * sz + nc] {
        return (pos, -0.3, false); // wall — bump, stay put
    }
    if (nr, nc) == maze.end {
        return ((nr, nc), 5.0, true); // reached the goal
    }
    // dense shaping: +1 per cell closer, −1 per cell farther, minus a tiny time cost
    let progress = (dist[pos.0 * sz + pos.1] - dist[nr * sz + nc]) as f32;
    ((nr, nc), progress - 0.05, false)
}

#[allow(clippy::too_many_arguments)]
fn run_rl(
    maze_size: usize, ticks: usize, steps: usize, route_len: usize,
    lr: f32, seed: u64, export_dir: Option<String>,
) {
    use modgrad_codec::genome::{ExpressedBrain, Genome};
    let out_dims = route_len * N_DIRECTIONS;
    let genome = Genome::blank_slate(maze_size, maze_size, out_dims, ticks);
    let ExpressedBrain { cortex, config } = genome.express();
    let encoder = cortex;
    let mut w = RegionalWeights::new(config);
    w.print_summary();
    // No AdamW: RL's landscape is non-stationary, so momentum points in stale
    // directions and adaptive LR is far less load-bearing than in SFT (Mukherjee
    // et al. 2026, "Do We Need Adam?"). We use the SDK's plain SGD apply — which
    // is also what `three_factor`'s ΔW = θ·advantage·trace amounts to: a
    // momentum-free, eligibility-weighted update. SGD wants a much larger LR.
    let mut rng = MazeRng::new(seed);

    let gamma = 0.95f32;
    let max_ep = maze_size * maze_size; // step budget per episode
    let mut baseline = 0.0f32;          // EMA of episode return (variance reduction)

    // rolling 100-episode metrics
    let (mut solved, mut wall_moves, mut total_moves, mut updir_moves) = (0usize, 0usize, 0usize, 0usize);

    eprintln!("RL: closed-loop REINFORCE — {maze_size}x{maze_size}, ticks={ticks}, lr={lr}, steps={steps}");

    for step in 1..=steps {
        let maze = generate_maze(maze_size, &mut rng);
        if maze.path_length < 3 { continue; }
        let goal = maze.end;
        let dist = rl_dist(&maze);
        let mut pos = maze.start;
        let mut traj: Vec<(Vec<f32>, usize, f32)> = Vec::new(); // (pixels, action, reward)
        let mut reached = false;
        // The CTM's continuous-thought state is carried across the WHOLE episode
        // — the brain integrates every observation it's seen, instead of waking
        // up blank each step. This is the recurrence the CTM exists for.
        let mut state = RegionalBrain::init_state(&w);

        // ---- rollout: act in the env under the current policy ----
        for _t in 0..max_ep {
            let pixels = render_maze_with_agent(&maze, pos);
            let input = encoder.encode(&pixels);
            let (output, new_state, _c) = RegionalBrain::forward_cached(&w, state, &input);
            state = new_state; // carry the thought forward
            let last = output.predictions.last().expect("no ticks");
            let probs = rl_softmax(&last[0..N_DIRECTIONS]);
            let action = rl_sample(&probs, &mut rng);

            total_moves += 1;
            if action == 0 || action == 2 { updir_moves += 1; } // up or left = "against the prior"
            let (npos, reward, done) = rl_step(&maze, pos, action, &dist);
            if npos == pos && action != 4 { wall_moves += 1; }
            traj.push((pixels, action, reward));
            pos = npos;
            if done { reached = pos == goal; break; }
        }
        if reached { solved += 1; }

        // ---- returns (discounted) ----
        let n = traj.len();
        let mut returns = vec![0.0f32; n];
        let mut g = 0.0f32;
        for t in (0..n).rev() { g = traj[t].2 + gamma * g; returns[t] = g; }

        // ---- GRPO-style advantage: A = (R - mean) / std. The mean is the
        // baseline (variance reduction); the std-normalization keeps the
        // gradient scale ~unit so momentum-free SGD takes well-conditioned
        // steps. This is the standard RL normalization, not a hand-tuned hack. ----
        let mean = returns.iter().sum::<f32>() / n.max(1) as f32;
        let var = returns.iter().map(|r| (r - mean) * (r - mean)).sum::<f32>() / n.max(1) as f32;
        let std = var.sqrt().max(1e-3);
        let advs: Vec<f32> = returns.iter().map(|r| (r - mean) / std).collect();

        // ---- REINFORCE updates: re-forward under current w, policy gradient.
        // We apply per timestep but scale the LR by 1/n, so the episode's
        // correlated steps sum to ONE effective update on the mean gradient
        // (standard REINFORCE is one update per episode; this approximates it
        // without needing a RegionalGradients accumulator). ----
        let eff_lr = lr / n.max(1) as f32;
        // replay the episode forward carrying state (truncated BPTT: state is
        // carried so each decision is memory-conditioned, gradient flows through
        // the single step). Forward every step to keep the state sequence intact.
        let mut up_state = RegionalBrain::init_state(&w);
        for t in 0..n {
            let (pixels, action, _r) = &traj[t];
            let input = encoder.encode(pixels);
            let (output, new_state, cache) = RegionalBrain::forward_cached(&w, up_state, &input);
            up_state = new_state;
            let adv = advs[t];
            if adv.abs() < 1e-4 { continue; }
            let k = output.predictions.len();
            let last = &output.predictions[k - 1];
            let probs = rl_softmax(&last[0..N_DIRECTIONS]);
            // d(-adv*logπ)/dlogit_i = -adv*(1{i=a} - p_i), placed on the committed tick
            let mut d_preds = vec![vec![0.0f32; last.len()]; k];
            for i in 0..N_DIRECTIONS {
                let ind = if i == *action { 1.0 } else { 0.0 };
                d_preds[k - 1][i] = -adv * (ind - probs[i]);
            }
            let mut grads = RegionalBrain::backward(&w, cache, &d_preds);
            // SGD step via the SDK's own apply (momentum-free), clip-norm 5.
            RegionalBrain::apply_gradients(&mut w, &mut grads, eff_lr, 5.0);
        }

        baseline = 0.95 * baseline + 0.05 * returns.first().copied().unwrap_or(0.0);

        if step % 100 == 0 {
            let solv = solved as f32;
            let wall = wall_moves as f32 / total_moves.max(1) as f32 * 100.0;
            let updir = updir_moves as f32 / total_moves.max(1) as f32 * 100.0;
            eprintln!(
                "rl {step:5}: solved={solv:.0}/100  wall_bumps={wall:.0}%  up|left_moves={updir:.0}%  avg_ep_len={:.1}  baseline={baseline:.3}",
                total_moves as f32 / 100.0
            );
            solved = 0; wall_moves = 0; total_moves = 0; updir_moves = 0;
        }
    }

    if let Some(dir) = export_dir {
        eprintln!("RL: (export to {dir} not yet wired for the RL path)");
    }
}

// ═══════════════════════════════════════════════════════════════
// 8-REGION BRAIN MODE
// ═══════════════════════════════════════════════════════════════

fn run_brain(
    maze_size: usize, ticks: usize, steps: usize, route_len: usize,
    lr: f32, seed: u64, batch_size: usize, imagination: bool, pain_mode: bool,
    plural_mode: bool, csv_mode: bool, cereb_size: usize, frozen_cereb: bool,
    autoresearch_summary: bool, budget_secs: Option<u64>,
    train_bank: Option<Vec<Maze>>, test_bank: Option<Vec<Maze>>,
    ghl_retina: bool, ghl_lr: f32, ghl_tau: f32,
    topdown: bool, topdown_alpha_override: Option<f32>,
    multiscale: bool,
    export_dir: Option<String>,
) {
    let t_train_start = std::time::Instant::now();
    let budget = budget_secs.map(std::time::Duration::from_secs);
    use std::io::Write;
    use modgrad_ctm::bio::dream;
    use modgrad_ctm::memory::episodic::EpisodicConfig;
    use modgrad_ctm::graph::AuxLossConfig;
    use modgrad_ctm::organism::{Organism, OrganismConfig};
    use modgrad_ctm::bio::pain::PainConfig;
    use modgrad_ctm::cerebellum::FrozenCerebellum;

    let out_dims = route_len * N_DIRECTIONS;

    // Express the brain from a Genome. Pretrained-retina path
    // overrides the encoder; the connectome (still genome-derived)
    // is unaffected. Keep `genome` bound — its `pathway_gates`
    // scalars are read at the per-sample top-down/subcortical sites.
    let genome = if multiscale {
        Genome::default_human_visual(maze_size, maze_size, out_dims, ticks)
    } else {
        Genome::blank_slate(maze_size, maze_size, out_dims, ticks)
    };
    let modgrad_codec::genome::ExpressedBrain { cortex: expressed_cortex, config: expressed_cfg } =
        genome.express();
    // Resolve the top-down gate: CLI override beats genome default.
    // `0.0` → top-down is identity (no-op), preserving the zero-init
    // property even when `--topdown` is passed.
    let topdown_alpha: f32 = topdown_alpha_override
        .unwrap_or(genome.pathway_gates.topdown_alpha);
    let mut encoder = if pretrained_retina_path().is_some() {
        make_encoder(maze_size, maze_size)
    } else {
        expressed_cortex
    };
    let token_dim = encoder.token_dim();
    let mut cfg = expressed_cfg;
    if multiscale {
        eprintln!("[multiscale] obs_scale_dims = {:?} (V4, V2, V1)", cfg.obs_scale_dims);
    }
    let _ = token_dim;

    // Cerebellum: 64 neurons × 16 ticks is the sweet spot for this scale.
    if pain_mode {
        let cereb_idx = cfg.region_names.iter()
            .position(|n| n.contains("cerebellum")).unwrap_or(4);
        let cs = if cereb_size > 0 { cereb_size } else { 64 };
        cfg.regions[cereb_idx] = modgrad_ctm::config::CtmConfig::region(
            "cerebellum", cs, cs / 2, 8, false, ticks,
            modgrad_ctm::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 },
        );
        cfg.aux_losses = AuxLossConfig {
            cerebellar_prediction: true,
            hippocampal_contrastive: true,
            bg_temporal_difference: true,
            aux_weight: 0.1,
        };
        eprintln!("Cerebellum: {cs} neurons × {ticks} ticks (aux losses enabled)");
    } else if cereb_size > 0 {
        let cereb_idx = cfg.region_names.iter()
            .position(|n| n.contains("cerebellum")).unwrap_or(4);
        cfg.regions[cereb_idx] = modgrad_ctm::config::CtmConfig::region(
            "cerebellum", cereb_size, cereb_size / 2, 8, false, ticks,
            modgrad_ctm::config::ExitStrategy::AdaptiveGate { beta: 0.05, threshold: 0.99 },
        );
        eprintln!("Cerebellum: {cereb_size} neurons (no aux losses)");
    }
    let mut w = RegionalWeights::new(cfg);

    // Frozen cerebellum: RandomExpansion as biological cerebellar model
    let mut frozen_model: Option<modgrad_ctm::cerebellum::RandomExpansion> = if frozen_cereb {
        let cereb_idx = w.config.region_names.iter()
            .position(|n| n.contains("cerebellum")).unwrap_or(4);
        let cortex_dim = w.regions[cereb_idx].config.d_model;
        let expansion = 4; // biological 4:1 granule:mossy ratio
        let frozen = modgrad_ctm::cerebellum::RandomExpansion::new(cortex_dim, expansion, seed);
        let hd = frozen.hidden_dim();
        w = w.with_frozen_cerebellum(hd, 1); // RandomExpansion: single layer
        eprintln!("Frozen cerebellum: RandomExpansion hidden_dim={hd} (4:1 expansion)");
        Some(frozen)
    } else {
        None
    };

    w.print_summary();

    let base_loss = maze_loss();
    let imagination_loss = Imagination::new(maze_loss());
    let loss_fn: &dyn LossFn<Target = [usize]> = if imagination {
        eprintln!("Loss: Imagination<StepwiseCE>");
        &imagination_loss
    } else {
        eprintln!("Loss: StepwiseCE (baseline)");
        &base_loss
    };

    if pain_mode && !plural_mode {
        eprintln!("Pain v4: surprise + priming + adaptive focus + dream replay + consolidation");
    }
    if plural_mode {
        eprintln!("Plural: pressure-driven personality splitting + salience switching");
    }

    let mut opt = RegionalAdamW::new(&w).with_lr(lr).with_clip(5.0);
    let mut rng = MazeRng::new(seed);
    let mut loss_history = Vec::new();
    let mut acc_history = Vec::new();

    let d_model = w.config.regions[0].d_model;
    let n_regions = w.config.regions.len();
    let pain_warmup = steps / 10;

    // Cache attention region index — used every sample when --topdown
    // is on. Fallback to 1 (attention is typically the second region
    // after `input`).
    let attn_region_idx: usize = w.config.region_names.iter()
        .position(|n| n.contains("attention"))
        .unwrap_or(1);
    if topdown {
        eprintln!(
            "Top-down: attention region [{}] = '{}' gates V4 tokens (2× forward/sample) \
             alpha={:.3}{}",
            attn_region_idx,
            w.config.region_names.get(attn_region_idx).map(|s| s.as_str()).unwrap_or("?"),
            topdown_alpha,
            if topdown_alpha == 0.0 { " — gain=1.0 (identity, no-op)" } else { "" },
        );
    }

    // Organism composes all pain/memory/dream/plural state
    let mut org = Organism::new(OrganismConfig {
        n_regions,
        d_model,
        max_ticks: ticks,
        pain: PainConfig::default(),
        episodic: EpisodicConfig {
            capacity: 512,
            max_ticks: ticks,
            d_model,
            min_ticks_for_storage: 1,
            min_surprise: 0.0,
            retrieval_threshold: 0.5,
            consolidation_threshold: 0.85,
            semantic_collapse_retrievals: 5,
            strength_decay: 0.95,
        },
        warmup_steps: pain_warmup,
        n_positions: route_len,
        plural: plural_mode,
        max_personalities: 4,
        red_threshold_for_split: 50,
        pain_focus_decay: 0.95,
        overcoming_threshold: -0.2,
    });

    let mut csv_writer = if csv_mode {
        let mut f = std::fs::File::create("maze_telemetry.csv")
            .expect("failed to create maze_telemetry.csv");
        writeln!(f, "step,loss,route_acc,pressure,dopamine,serotonin,norepinephrine,lr,first_step_acc,memory_count")
            .expect("failed to write CSV header");
        Some(f)
    } else {
        None
    };

    let mut first_step_hits = 0usize;
    let mut first_step_total = 0usize;
    let mut steps_completed = 0usize;

    for step in 1..=steps {
        if let Some(d) = budget {
            if t_train_start.elapsed() >= d {
                eprintln!("(--budget {}s exhausted at step {step})", d.as_secs());
                break;
            }
        }
        steps_completed = step;
        let mut grads = RegionalGradients::zeros(&w);
        let mut batch_loss = 0.0f32;
        let mut batch_correct = 0usize;
        let mut batch_total = 0usize;
        let mut batch_valid = 0usize;
        let mut batch_first_correct = 0usize;
        let mut batch_first_total = 0usize;

        // Organism handles plural switching check
        if pain_mode { org.begin_step(); }

        for _ in 0..batch_size {
            let maze: Maze = if let Some(ref bank) = train_bank {
                bank[rng.range(bank.len())].clone()
            } else {
                generate_maze(maze_size, &mut rng)
            };
            if maze.path_length < 3 { continue; }

            let pixels = render_input(&maze);
            let input = if multiscale {
                encode_multiscale_concat(&encoder, &pixels)
            } else {
                encoder.encode(&pixels)
            };

            let mut route = maze.route.clone();
            route.truncate(route_len);
            while route.len() < route_len { route.push(DIR_WAIT); }

            // Organism handles episodic retrieval + valence
            let query_len = d_model.min(input.tokens.len());
            let before = if pain_mode {
                org.before_sample(&input.tokens[..query_len])
            } else {
                org.before_sample(&[])
            };

            // Prime initial state from retrieval
            let mut state = RegionalBrain::init_state(&w);
            if let Some(ref retrieval) = before.retrieval {
                if retrieval.n_matches > 0 {
                    let hippo_idx = w.config.region_names.iter()
                        .position(|n| n.contains("hippocampus"))
                        .unwrap_or(7);
                    let blend = retrieval.best_similarity * 0.3;
                    dream::prime_state(&mut state.region_outputs, retrieval, blend, hippo_idx);
                }
            }

            // ── Optional top-down pass ─────────────────────────────
            // When --topdown is on, the `attention` region's output from
            // a scratch forward pass projects to a per-spatial-position
            // gain mask, which is then applied to the V4 tokens in the
            // *actual* training pass. With --multiscale, the same gain
            // is applied at V1, V2, AND V4 — mirroring biology's V4→V2
            // and V2→V1 back-projections. Costs 2× forward per sample.
            let input = if topdown && !retina_bypass_enabled() {
                let scratch_state = RegionalBrain::init_state(&w);
                let (_, state_after, _) = if let Some(ref mut frozen) = frozen_model {
                    RegionalBrain::forward_cached_frozen(&w, scratch_state, &input, frozen)
                } else {
                    RegionalBrain::forward_cached(&w, scratch_state, &input)
                };
                let attn = &state_after.region_outputs[attn_region_idx];
                let n_pos = input.n_tokens;
                let mut gain = vec![0.0f32; n_pos];
                let al = attn.len().max(1);
                // Gated sigmoid centred at 1.0: `gain = 1 + α·(σ − 0.5)`.
                // α = 0 → gain ≡ 1 (identity, top-down off even with --topdown).
                // α = 1 → range [0.5, 1.5], the original full-strength sigmoid
                //         shifted to be centred on no-op.
                // This matches `PathwayGates`' meditation-ring zero-init
                // contract: every new pathway starts as a no-op.
                for i in 0..n_pos {
                    let v = attn[i % al];
                    let s = 1.0 / (1.0 + (-v).exp());
                    gain[i] = 1.0 + topdown_alpha * (s - 0.5);
                }
                if multiscale {
                    // Apply the same gain at V1, V2, V4 — multistage
                    // top-down. Concatenate V4/V2/V1 streams the same
                    // way encode_multiscale_concat does.
                    let multi = encoder.encode_gated_multistage(&pixels, &gain);
                    let mut tokens = Vec::with_capacity(
                        multi.scales.iter().map(|s| s.tokens.len()).sum(),
                    );
                    for s in &multi.scales { tokens.extend_from_slice(&s.tokens); }
                    TokenInput {
                        n_tokens: multi.scales[0].n_tokens,
                        token_dim: multi.scales[0].token_dim,
                        tokens,
                    }
                } else {
                    encoder.encode_gated(&pixels, &gain)
                }
            } else {
                input
            };

            let (output, _state, cache) = if let Some(ref mut frozen) = frozen_model {
                RegionalBrain::forward_cached_frozen(&w, state, &input, frozen)
            } else {
                RegionalBrain::forward_cached(&w, state, &input)
            };
            let (loss, d_preds) = loss_fn.compute(
                &output.predictions, &output.certainties, &route);
            let sample_grads = if ghl_retina {
                let (g, d_obs) =
                    RegionalBrain::backward_with_input_grad(&w, cache, &d_preds);
                // d_obs is the gradient w.r.t. the flattened observation
                // (= `TokenInput.tokens`, shape `[n_tokens × token_dim]`),
                // which is exactly the `d_tokens` layout `ghl_step` wants.
                encoder.ghl_step(&pixels, &d_obs, ghl_lr, ghl_tau);
                g
            } else {
                RegionalBrain::backward(&w, cache, &d_preds)
            };

            // Accumulate gradients
            let add = |d: &mut [f32], s: &[f32]| {
                for (d, s) in d.iter_mut().zip(s) { *d += s; }
            };
            for r in 0..grads.region_grads.len() {
                add(&mut grads.region_grads[r].nlm_s1_w, &sample_grads.region_grads[r].nlm_s1_w);
                add(&mut grads.region_grads[r].nlm_s1_b, &sample_grads.region_grads[r].nlm_s1_b);
                add(&mut grads.region_grads[r].kv_proj_w, &sample_grads.region_grads[r].kv_proj_w);
                add(&mut grads.region_grads[r].kv_proj_b, &sample_grads.region_grads[r].kv_proj_b);
                add(&mut grads.region_grads[r].q_proj_w, &sample_grads.region_grads[r].q_proj_w);
                add(&mut grads.region_grads[r].q_proj_b, &sample_grads.region_grads[r].q_proj_b);
                add(&mut grads.region_grads[r].mha_in_w, &sample_grads.region_grads[r].mha_in_w);
                add(&mut grads.region_grads[r].mha_in_b, &sample_grads.region_grads[r].mha_in_b);
                add(&mut grads.region_grads[r].mha_out_w, &sample_grads.region_grads[r].mha_out_w);
                add(&mut grads.region_grads[r].mha_out_b, &sample_grads.region_grads[r].mha_out_b);
                add(&mut grads.region_grads[r].out_proj_w, &sample_grads.region_grads[r].out_proj_w);
                add(&mut grads.region_grads[r].out_proj_b, &sample_grads.region_grads[r].out_proj_b);
            }
            add(&mut grads.output_proj_dw, &sample_grads.output_proj_dw);
            add(&mut grads.output_proj_db, &sample_grads.output_proj_db);
            for ci in 0..grads.connection_dw.len() {
                add(&mut grads.connection_dw[ci], &sample_grads.connection_dw[ci]);
                add(&mut grads.connection_db[ci], &sample_grads.connection_db[ci]);
            }

            batch_loss += loss;
            batch_valid += 1;

            // Per-position accuracy — organism handles pain focus
            let mut sample_correct = 0usize;
            let mut sample_total = 0usize;

            if let Some(pred) = output.predictions.last() {
                for pos in 0..route_len.min(maze.path_length) {
                    let off = pos * N_DIRECTIONS;
                    if off + N_DIRECTIONS <= pred.len() {
                        let p = pred[off..off + N_DIRECTIONS].iter().enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(i, _)| i).unwrap_or(0);
                        let correct = p == route[pos];
                        if correct { sample_correct += 1; }
                        sample_total += 1;

                        if pos == 0 {
                            batch_first_total += 1;
                            if correct { batch_first_correct += 1; }
                        }

                        if pain_mode {
                            let confidence = if correct {
                                let logits = &pred[off..off + N_DIRECTIONS];
                                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                let sum_exp: f32 = logits.iter().map(|l| (l - max_l).exp()).sum();
                                (max_l - sum_exp.ln()).exp().clamp(0.0, 1.0)
                            } else {
                                0.5
                            };
                            org.after_position(pos, correct, confidence, before.retrieval_valence);
                        }
                    }
                }
            }
            batch_correct += sample_correct;
            batch_total += sample_total;

            // Organism handles episodic storage with valence
            if pain_mode {
                org.after_sample(loss, sample_correct, sample_total, &input.tokens);
            }
        }

        first_step_hits += batch_first_correct;
        first_step_total += batch_first_total;

        if batch_valid > 0 {
            let avg_loss = batch_loss / batch_valid as f32;
            let avg_acc = batch_correct as f32 / batch_total.max(1) as f32;

            // Organism handles LR scaling, sleep, dream, splitting
            if pain_mode {
                let batch = org.after_batch(avg_loss, Some(&|_idx, key| {
                    let mut dream_state = RegionalState::new(&w);
                    let obs = if key.len() >= w.config.raw_obs_dim {
                        key[..w.config.raw_obs_dim].to_vec()
                    } else {
                        let mut o = key.to_vec();
                        o.resize(w.config.raw_obs_dim, 0.0);
                        o
                    };
                    let output = regional_forward(&w, &mut dream_state, &obs);
                    let loss = if let Some(pred) = output.predictions.last() {
                        let max_l = pred.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let sum_exp: f32 = pred.iter().map(|l| (l - max_l).exp()).sum();
                        (sum_exp.ln() + max_l).abs()
                    } else {
                        avg_loss
                    };
                    let was_correct = loss < avg_loss;
                    (loss, was_correct)
                }));

                opt.lr = lr * batch.lr_scale;

                if batch.slept && step % 200 == 0 {
                    eprintln!("    [sleep] overcame={} regressed={} merges={} mem={}",
                        batch.dream.as_ref().map_or(0, |d| d.overcame),
                        batch.dream.as_ref().map_or(0, |d| d.regressed),
                        batch.merges, batch.memory_count);
                }
                if batch.did_split {
                    eprintln!("    [SPLIT] personality forking → '{}' (n={})",
                        batch.active_name, batch.n_personalities);
                }

                // CSV telemetry
                if let Some(ref mut f) = csv_writer {
                    let first_acc = if batch_first_total > 0 {
                        batch_first_correct as f32 / batch_first_total as f32
                    } else { 0.0 };
                    writeln!(f, "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
                        step, avg_loss, avg_acc, batch.pressure, batch.dopamine,
                        batch.serotonin, batch.norepinephrine, opt.lr, first_acc,
                        batch.memory_count)
                        .expect("failed to write CSV row");
                }
            } else {
                // CSV telemetry (no pain)
                if let Some(ref mut f) = csv_writer {
                    let first_acc = if batch_first_total > 0 {
                        batch_first_correct as f32 / batch_first_total as f32
                    } else { 0.0 };
                    writeln!(f, "{},{:.6},{:.6},0,0,0,0,{:.6},{:.6},0",
                        step, avg_loss, avg_acc, lr, first_acc)
                        .expect("failed to write CSV row");
                }
            }

            opt.step(&mut w, &mut grads);
            loss_history.push(avg_loss);
            acc_history.push(avg_acc);
        }

        if step % 100 == 0 || step == steps {
            let window = 50.min(loss_history.len());
            if window > 0 {
                let smooth_loss: f32 = loss_history[loss_history.len()-window..].iter().sum::<f32>() / window as f32;
                let smooth_acc: f32 = acc_history[acc_history.len()-window..].iter().sum::<f32>() / window as f32;
                let first_pct = if first_step_total > 0 {
                    first_step_hits as f32 / first_step_total as f32 * 100.0
                } else { 0.0 };
                if pain_mode {
                    let report = org.report();
                    eprintln!("step {step:5}: loss={smooth_loss:.3} route={:.1}% first={first_pct:.1}% | {report} lr={:.5}",
                        smooth_acc * 100.0, opt.lr);
                } else {
                    eprintln!("step {step:5}: loss={smooth_loss:.3} route_acc={:.1}% first={first_pct:.1}%",
                        smooth_acc * 100.0);
                }
                first_step_hits = 0;
                first_step_total = 0;
            }
        }
    }

    // Evaluation
    eprintln!("\n--- Evaluation (200 mazes) ---");
    let mut eval_rng = MazeRng::new(seed + 999);
    let mut first_correct = 0usize;
    let mut route_correct = 0usize;
    let mut route_total = 0usize;
    let mut prefix_lengths = Vec::new();
    let mut n_valid = 0usize;

    // Use held-out test bank if provided; otherwise sample the training
    // bank (or generate fresh mazes). Keeping eval distinct from training
    // data matters when a fixed bank replaces the online generator.
    let eval_source: Option<&[Maze]> = test_bank.as_deref().or(train_bank.as_deref());

    for eval_i in 0..200 {
        let maze: Maze = if let Some(bank) = eval_source {
            bank[eval_i % bank.len()].clone()
        } else {
            generate_maze(maze_size, &mut eval_rng)
        };
        if maze.path_length < 3 { continue; }

        let pixels = render_input(&maze);
        let input = if multiscale {
            encode_multiscale_concat(&encoder, &pixels)
        } else {
            encoder.encode(&pixels)
        };
        let obs: Vec<f32> = input.tokens.clone();

        let mut route = maze.route.clone();
        route.truncate(route_len);
        while route.len() < route_len { route.push(DIR_WAIT); }

        let mut state = RegionalState::new(&w);
        let output = regional_forward(&w, &mut state, &obs);
        let pred = output.predictions.last().unwrap();

        let first_logits = &pred[0..N_DIRECTIONS.min(pred.len())];
        let first_pred = first_logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if first_pred == route[0] { first_correct += 1; }

        let eval_len = route_len.min(maze.path_length);
        let mut prefix = 0usize;
        for pos in 0..eval_len {
            let offset = pos * N_DIRECTIONS;
            if offset + N_DIRECTIONS > pred.len() { break; }
            let logits = &pred[offset..offset + N_DIRECTIONS];
            let p = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if p == route[pos] {
                route_correct += 1;
                if pos == prefix { prefix += 1; }
            }
            route_total += 1;
        }
        prefix_lengths.push(prefix);
        n_valid += 1;
    }

    let avg_prefix = prefix_lengths.iter().sum::<usize>() as f32 / n_valid.max(1) as f32;
    let first_step_acc = first_correct as f32 / n_valid.max(1) as f32;
    eprintln!("First step acc:     {first_correct}/{n_valid} ({:.1}%)",
        first_step_acc * 100.0);
    eprintln!("Per-step acc:       {route_correct}/{route_total} ({:.1}%)",
        if route_total > 0 { route_correct as f32 / route_total as f32 * 100.0 } else { 0.0 });
    eprintln!("Avg correct prefix: {avg_prefix:.1} steps (of {route_len})");

    // Same autoresearch summary shape as run_single — see that function
    // for the `val_bpb = 1 - first_step_acc` rationale.
    if autoresearch_summary {
        let total_seconds = t_train_start.elapsed().as_secs_f32();
        modgrad_training::AutoresearchSummary {
            val_bpb: 1.0 - first_step_acc,
            training_seconds: total_seconds,  // no separate eval start — brain mode interleaves
            total_seconds,
            peak_vram_mb: 0.0,
            mfu_percent: 0.0,
            total_tokens_m: 0.0,
            num_steps: steps_completed as u64,
            num_params_m: (w.n_params() as f32) / 1.0e6,
        }.print();
        eprintln!("  (task=mazes-brain size={maze_size} route_len={route_len}, \
                   val_bpb = 1 - first_step_acc = {:.6})",
                   1.0 - first_step_acc);
    }

    // ── Export the trained brain in the wasm-engine JSON format ──
    // Pure post-training save: training above is unchanged. Skipped
    // unless --export DIR was passed. The exported observation path
    // uses the plain (non-multiscale) retina, so refuse to export a
    // multiscale brain — its token layout would not match the engine's
    // `spatial_tokens` consumer.
    if let Some(dir) = export_dir {
        if multiscale {
            eprintln!("--export: refusing to export a --multiscale brain \
                       (engine consumes plain spatial_tokens); skipping save.");
        } else {
            export_brain::export(
                &dir, &w, &encoder, maze_size, ticks, route_len,
                out_dims, first_step_acc,
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// EXPORT: save trained brain in the wasm engine's JSON format.
//
// Writes:
//   brain_solver_weights.json   = { cortex: VisualCortex, regional: RegionalWeights (embeddings stripped) }
//   brain_solver_reference.json = config echo + held-out accuracy + oracle traces
//
// The oracle replicates `regional_forward`'s outer loop so it can
// capture per-outer-tick {prediction, region_activations, global_sync,
// exit_lambda}, then cross-checks the replicated final-tick predictions
// against the real `regional_forward` to guarantee the trace is faithful.
// Mirrors examples/mazes/src/bin/brain_oracle.rs exactly.
// ═══════════════════════════════════════════════════════════════

mod export_brain {
    use crate::{MAZE_N_CLASSES, N_DIRECTIONS};
    use modgrad_codec::retina::VisualCortex;
    use modgrad_ctm::graph::{RegionalWeights, RegionalState, regional_forward};
    use modgrad_ctm::forward::{ctm_forward, CtmInput};
    use serde::Serialize;
    use serde_json::Value;

    // Four fixed deterministic 9×9 mazes (1 = wall, 0 = open), row-major.
    // Identical to brain_oracle::fixed_mazes so the engine test bank lines up.
    fn fixed_mazes(size: usize) -> Vec<(Vec<u8>, (usize, usize), (usize, usize))> {
        let parse = |rows: &[&str]| -> Vec<u8> {
            let mut g = vec![0u8; size * size];
            for (r, row) in rows.iter().enumerate() {
                for (c, ch) in row.chars().enumerate() {
                    g[r * size + c] = if ch == '#' { 1 } else { 0 };
                }
            }
            g
        };
        let m0 = parse(&[
            "#########", "#.......#", "#.#####.#", "#.#...#.#", "#.#.#.#.#",
            "#.#.#.#.#", "#...#...#", "#.#####.#", "#########",
        ]);
        let m1 = parse(&[
            "#########", "#.#.....#", "#.#.###.#", "#...#...#", "###.#.###",
            "#...#...#", "#.###.#.#", "#.....#.#", "#########",
        ]);
        let m2 = parse(&[
            "#########", "#.......#", "#######.#", "#.......#", "#.#######",
            "#.......#", "#######.#", "#.......#", "#########",
        ]);
        let m3 = parse(&[
            "#########", "#...#...#", "#.#.#.#.#", "#.#...#.#", "#.#####.#",
            "#.....#.#", "#####.#.#", "#.......#", "#########",
        ]);
        vec![
            (m0, (1, 1), (7, 7)),
            (m1, (1, 1), (7, 7)),
            (m2, (1, 1), (7, 7)),
            (m3, (1, 1), (7, 7)),
        ]
    }

    // Render a grid+agent+goal to RGB pixels [3 × H × W] CHW using the
    // EXACT scheme `render_maze` (the proven recipe's renderer, SDF off):
    //   wall  → (0,0,0)   open  → (1,1,1)
    //   agent → (1,0,0)   goal  → (0,1,0)
    // Channel-major [R plane][G plane][B plane].
    fn render_pixels(
        grid: &[u8], size: usize, agent: (usize, usize), goal: (usize, usize),
    ) -> Vec<f32> {
        let n = size * size;
        let mut px = vec![0.0f32; 3 * n];
        // open cells white first
        for idx in 0..n {
            if grid[idx] == 0 {
                px[idx] = 1.0; px[n + idx] = 1.0; px[2 * n + idx] = 1.0;
            }
        }
        // agent: red (overwrites)
        let ai = agent.0 * size + agent.1;
        px[ai] = 1.0; px[n + ai] = 0.0; px[2 * n + ai] = 0.0;
        // goal: green (overwrites)
        let gi = goal.0 * size + goal.1;
        px[gi] = 0.0; px[n + gi] = 1.0; px[2 * n + gi] = 0.0;
        px
    }

    struct TraceOut {
        predictions: Vec<Vec<f32>>,
        region_acts: Vec<Vec<Vec<f32>>>,
        global_sync: Vec<Vec<f32>>,
        exit_lambdas: Vec<f32>,
        ticks_used: usize,
    }

    // Faithful replica of regional_forward with per-outer-tick capture.
    // Mirrors modgrad_ctm::graph::regional_forward operation-for-operation
    // (fixed-connection path, no router), sequential so the FP order is
    // deterministic. Uses the real SDK ctm_forward per region.
    fn regional_forward_traced(
        w: &RegionalWeights, state: &mut RegionalState, observation: &[f32],
    ) -> TraceOut {
        let cfg = &w.config;
        let n_regions = cfg.regions.len();

        let mut predictions: Vec<Vec<f32>> = Vec::new();
        let mut region_acts: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut global_sync_trace: Vec<Vec<f32>> = Vec::new();
        let mut exit_lambdas: Vec<f32> = Vec::new();
        let mut exit_cdf = 0.0f32;
        let mut survival = 1.0f32;

        let obs_projected = w.obs_proj.forward(observation);
        let total_neurons: usize = cfg.regions.iter().map(|r| r.d_model).sum();
        let n_sync = cfg.n_global_sync;

        let mut prev_outputs: Vec<Vec<f32>> = state.region_outputs.clone();

        for _outer_tick in 0..cfg.outer_ticks {
            // Phase A: build each region's observation via fixed connections.
            let region_obs: Vec<Vec<f32>> = (0..n_regions).map(|r| {
                let mut slot: Vec<f32> = Vec::new();
                for (ci, conn) in cfg.connections.iter().enumerate() {
                    if conn.to == r {
                        let mut src = Vec::new();
                        for &from_idx in &conn.from {
                            src.extend_from_slice(&prev_outputs[from_idx]);
                        }
                        if conn.receives_observation {
                            src.extend_from_slice(observation);
                        }
                        let projected = w.connection_synapses[ci].forward(&src);
                        if slot.is_empty() {
                            slot = projected;
                        } else {
                            let nmin = slot.len().min(projected.len());
                            for i in 0..nmin { slot[i] += projected[i]; }
                        }
                    }
                }
                if slot.is_empty() { obs_projected.clone() } else { slot }
            }).collect();

            // Phase B: run each region's CTM (sequential).
            let mut results: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
            for r in 0..n_regions {
                let d_input = w.regions[r].config.d_input;
                let _out = ctm_forward(&w.regions[r], &mut state.region_states[r], CtmInput::Raw {
                    obs: &region_obs[r], n_tokens: 1, raw_dim: d_input,
                });
                results.push(state.region_states[r].activated.clone());
            }

            // Phase C: commit outputs.
            for r in 0..n_regions {
                state.region_outputs[r] = results[r].clone();
            }
            prev_outputs = state.region_outputs.clone();

            // Phase 3: global sync.
            let mut all_act = vec![0.0f32; total_neurons];
            {
                let mut offset = 0;
                for r in 0..n_regions {
                    let d = state.region_outputs[r].len();
                    all_act[offset..offset + d].copy_from_slice(&state.region_outputs[r]);
                    offset += d;
                }
            }
            for i in 0..n_sync {
                let l = w.global_sync_left[i];
                let rr = w.global_sync_right[i];
                if l < all_act.len() && rr < all_act.len() {
                    let pw = all_act[l] * all_act[rr];
                    let decay = (-w.global_decay[i].clamp(0.0, 15.0)).exp();
                    state.global_alpha[i] = decay * state.global_alpha[i] + pw;
                    state.global_beta[i] = decay * state.global_beta[i] + 1.0;
                }
            }
            let mut gs_buf = vec![0.0f32; n_sync];
            for i in 0..n_sync {
                gs_buf[i] = state.global_alpha[i] / state.global_beta[i].sqrt().max(1e-8);
            }

            // Phase 4: output prediction.
            let prediction = w.output_proj.forward(&gs_buf);

            predictions.push(prediction);
            region_acts.push(state.region_outputs.clone());
            global_sync_trace.push(gs_buf.clone());

            // Phase 5: AdaptiveGate outer exit.
            if let Some(ref gate) = w.outer_exit_gate {
                let gate_logit = gate.forward(&gs_buf);
                let lambda = 1.0 / (1.0 + (-gate_logit[0]).exp());
                exit_lambdas.push(lambda);
                let p_exit = lambda * survival;
                exit_cdf += p_exit;
                survival *= 1.0 - lambda;
                if exit_cdf > 0.99 { break; }
            }
        }

        let ticks_used = predictions.len();
        TraceOut { predictions, region_acts, global_sync: global_sync_trace, exit_lambdas, ticks_used }
    }

    #[derive(Serialize)]
    struct TickTrace {
        prediction: Vec<f32>,
        region_activations: Vec<Vec<f32>>,
        global_sync: Vec<f32>,
        exit_lambda: Option<f32>,
    }

    #[derive(Serialize)]
    struct MazeSample {
        grid: Vec<u8>,
        agent: [usize; 2],
        goal: [usize; 2],
        pixels: Vec<f32>,
        observation: Vec<f32>,
        n_tokens: usize,
        token_dim: usize,
        ticks_used: usize,
        ticks: Vec<TickTrace>,
    }

    #[derive(Serialize)]
    struct RegionInfo {
        name: String,
        d_model: usize,
        d_input: usize,
        memory_length: usize,
        iterations: usize,
    }

    #[derive(Serialize)]
    struct ConnectionInfo {
        from: Vec<usize>,
        to: usize,
        receives_observation: bool,
        observation_scale: usize,
    }

    #[derive(Serialize)]
    struct PixelSpec {
        layout: String,
        channels: usize,
        height: usize,
        width: usize,
        sdf: bool,
        scheme: String,
    }

    #[derive(Serialize)]
    struct Reference {
        task: String,
        size: usize,
        ticks: usize,
        out_dims: usize,
        route_len: usize,
        n_classes: usize,
        n_directions: usize,
        raw_obs_dim: usize,
        obs_scale_dims: Vec<usize>,
        n_global_sync: usize,
        region_names: Vec<String>,
        regions: Vec<RegionInfo>,
        connections: Vec<ConnectionInfo>,
        heldout_first_move_acc: f32,
        pixel_spec: PixelSpec,
        samples: Vec<MazeSample>,
    }

    pub fn export(
        out_dir: &str,
        w: &RegionalWeights,
        cortex: &VisualCortex,
        size: usize,
        ticks: usize,
        route_len: usize,
        out_dims: usize,
        heldout_first_move_acc: f32,
    ) {
        std::fs::create_dir_all(out_dir).expect("--export: create out dir");
        let cfg = &w.config;

        let region_infos: Vec<RegionInfo> = cfg.region_names.iter().zip(&cfg.regions)
            .map(|(name, rc)| RegionInfo {
                name: name.clone(), d_model: rc.d_model, d_input: rc.d_input,
                memory_length: rc.memory_length, iterations: rc.iterations,
            }).collect();
        let conn_infos: Vec<ConnectionInfo> = cfg.connections.iter().map(|c| ConnectionInfo {
            from: c.from.clone(), to: c.to,
            receives_observation: c.receives_observation,
            observation_scale: c.observation_scale,
        }).collect();

        // Oracle: run the trained retina + brain on each fixed maze.
        let mazes = fixed_mazes(size);
        let mut samples: Vec<MazeSample> = Vec::new();
        for (mi, (grid, agent, goal)) in mazes.iter().enumerate() {
            let pixels = render_pixels(grid, size, *agent, *goal);
            let (tokens, n_tokens, token_dim) = cortex.spatial_tokens(&pixels);

            let mut traced_state = RegionalState::new(w);
            let traced = regional_forward_traced(w, &mut traced_state, &tokens);

            // Cross-check the traced predictions against the real
            // regional_forward to guarantee the trace is faithful.
            let mut real_state = RegionalState::new(w);
            let real_out = regional_forward(w, &mut real_state, &tokens);
            assert_eq!(real_out.predictions.len(), traced.predictions.len(),
                "export maze {mi}: tick count mismatch");
            let mut max_d = 0.0f32;
            for (t, (rp, tp)) in real_out.predictions.iter().zip(&traced.predictions).enumerate() {
                for (j, (&a, &b)) in rp.iter().zip(tp).enumerate() {
                    let d = (a - b).abs();
                    if d > max_d { max_d = d; }
                    assert!(d < 1e-4,
                        "export maze {mi} tick {t} pred[{j}]: traced {b} vs real {a}");
                }
            }
            eprintln!("--export: oracle maze {mi} cross-check OK \
                       (max |Δpred| = {max_d:.2e}, ticks_used={})", traced.ticks_used);

            let tick_traces: Vec<TickTrace> = (0..traced.ticks_used).map(|t| TickTrace {
                prediction: traced.predictions[t].clone(),
                region_activations: traced.region_acts[t].clone(),
                global_sync: traced.global_sync[t].clone(),
                exit_lambda: traced.exit_lambdas.get(t).copied(),
            }).collect();

            samples.push(MazeSample {
                grid: grid.clone(),
                agent: [agent.0, agent.1],
                goal: [goal.0, goal.1],
                pixels,
                observation: tokens,
                n_tokens, token_dim,
                ticks_used: traced.ticks_used,
                ticks: tick_traces,
            });
        }

        let reference = Reference {
            task: "brain_maze_solver".to_string(),
            size, ticks, out_dims, route_len,
            n_classes: MAZE_N_CLASSES,
            n_directions: N_DIRECTIONS,
            raw_obs_dim: cfg.raw_obs_dim,
            obs_scale_dims: cfg.obs_scale_dims.clone(),
            n_global_sync: cfg.n_global_sync,
            region_names: cfg.region_names.clone(),
            regions: region_infos,
            connections: conn_infos,
            heldout_first_move_acc,
            pixel_spec: PixelSpec {
                layout: "CHW [3 x H x W]".to_string(),
                channels: 3, height: size, width: size,
                sdf: false,
                scheme: "wall=(0,0,0) open=(1,1,1) agent/start=(1,0,0) goal/end=(0,1,0)".to_string(),
            },
            samples,
        };

        // Serialize weights MINUS the (huge, unused) embeddings table.
        let mut weights_val: Value = serde_json::to_value(w).expect("serialize regional weights");
        if let Some(obj) = weights_val.as_object_mut() {
            obj.remove("embeddings");
        }
        let cortex_val: Value = serde_json::to_value(cortex).expect("serialize cortex");

        #[derive(Serialize)]
        struct BrainWeights { cortex: Value, regional: Value }
        let brain_weights = BrainWeights { cortex: cortex_val, regional: weights_val };

        let w_path = format!("{out_dir}/brain_solver_weights.json");
        let r_path = format!("{out_dir}/brain_solver_reference.json");
        let w_str = serde_json::to_string(&brain_weights).expect("serialize brain_weights");
        let r_str = serde_json::to_string(&reference).expect("serialize reference");
        std::fs::write(&w_path, &w_str).expect("write brain_solver_weights.json");
        std::fs::write(&r_path, &r_str).expect("write brain_solver_reference.json");

        eprintln!("--export: wrote {} ({} bytes) + {} ({} bytes)",
            w_path, w_str.len(), r_path, r_str.len());
        eprintln!("--export: held-out first-move acc embedded = {:.1}%",
            heldout_first_move_acc * 100.0);
    }
}

