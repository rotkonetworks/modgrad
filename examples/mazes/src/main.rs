//! Maze solving benchmark for CTM.
//!
//! Pipeline: maze pixels → Encoder (VisualRetina) → TokenInput → Brain (CTM) → RouteLoss
//!
//! Uses the SDK trait system: Encoder + Brain + LossFn compose cleanly.
//! No ad-hoc training code — the same path works for any modality.
//!
//! Usage:
//!   mazes --size 21 --ticks 16 --steps 10000
//!   mazes --size 39 --ticks 32 --steps 50000 --route-len 50

mod maze_gen;

use maze_gen::*;
use modgrad_ctm::config::{CtmConfig, ExitStrategy};
use modgrad_ctm::weights::{CtmWeights, CtmState};
use modgrad_ctm::train::{Ctm, CtmGradients, accumulate_gradients};
use modgrad_ctm::graph::{
    RegionalConfig, RegionalWeights, RegionalGradients, RegionalState,
    RegionalAdamW, RegionalBrain, regional_forward,
};
use modgrad_codec::retina::VisualRetina;
use modgrad_traits::{Brain, Encoder, LossFn, RouteLoss, Imagination};

fn main() {
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
            "--help" | "-h" => {
                eprintln!("Usage: mazes [--size N] [--ticks N] [--steps N] [--route-len N]");
                eprintln!("             [--d-model N] [--lr F] [--batch N] [--no-adaptive]");
                eprintln!("             [--imagination] [--brain] [--pain] [--plural] [--csv]");
                eprintln!("             [--cereb-size N] [--frozen-cereb]");
                return;
            }
            _ => { i += 1; }
        }
    }

    let maze_size = maze_size | 1;

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

    if brain {
        run_brain(maze_size, ticks, steps, route_len, lr, seed, batch_size, imagination, pain_mode, plural_mode, csv_mode, cereb_size, frozen_cereb, autoresearch_summary, budget_secs, train_bank, test_bank);
        return;
    }

    eprintln!("Maze benchmark (single CTM): size={maze_size} ticks={ticks} d_model={d_model} \
               route_len={route_len} batch={batch_size}");

    // ── Encoder: visual retina → spatial tokens ──
    let mut encoder = VisualRetina::maze(maze_size, maze_size);
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
            bank.push(render_maze(&m));
        }
        let refs: Vec<&[f32]> = bank.iter().map(|v| v.as_slice()).collect();
        eprintln!("Hebbian pretraining: {hebbian_samples} mazes × {hebbian_epochs} epochs (lr={hebbian_lr})");
        encoder.train_hebbian(&refs, hebbian_epochs, hebbian_lr);
        eprintln!("Hebbian pretraining done in {:.1}s", t0.elapsed().as_secs_f32());
    }

    // ── Optional dream pretraining (Hoel 2021) ──
    // Synthesizes top-down pixels via V4^T→V2^T→V1^T (sparse noise at V4,
    // adjoint down through the cortex). Runs the same Hebbian update on
    // these dream-pixels as on real mazes, but since the source is the
    // cortex itself the training data is out-of-distribution relative
    // to the task — the overfitted-brain hypothesis's proposed
    // regularization mechanism.
    //
    // Bootstrap note: when V2/V4 are random, dreams are near-random too.
    // That's fine under Hoel — the sparsity + top-down stochasticity do
    // the regularization work even without a well-formed cortex prior.
    // Combining with --hebbian-epochs gives the cortex a real prior
    // first, then dream refinement.
    if dream_epochs > 0 {
        let t0 = std::time::Instant::now();
        let mut bank: Vec<Vec<f32>> = Vec::with_capacity(hebbian_samples);
        for i in 0..hebbian_samples {
            let dseed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(0xD5EA_D5EA)
                .wrapping_add(i as u64);
            bank.push(encoder.dream_pixel(dseed, dream_sparsity_k));
        }
        let refs: Vec<&[f32]> = bank.iter().map(|v| v.as_slice()).collect();
        eprintln!("Dream pretraining: {hebbian_samples} synthesized × {dream_epochs} epochs \
                   (lr={hebbian_lr}, sparsity_k={dream_sparsity_k})");
        encoder.train_hebbian(&refs, dream_epochs, hebbian_lr);
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
    };

    let mut w = CtmWeights::new(cfg, token_dim);
    eprintln!("Brain: d_model={d_model} d_input={} ticks={ticks} out={out_dims} params={}",
        w.config.d_input, w.n_params());

    // ── Loss: route prediction with auto-curriculum ──
    let base_loss = RouteLoss::maze();
    let imagination_loss = Imagination::new(RouteLoss::maze());
    let loss_fn: &dyn LossFn<Target = [usize]> = if imagination {
        eprintln!("Loss: Imagination<RouteLoss> (ratio=0.5)");
        &imagination_loss
    } else {
        eprintln!("Loss: RouteLoss (Sakana baseline)");
        &base_loss
    };

    // ── Training ──
    let t_train_start = std::time::Instant::now();
    let budget = budget_secs.map(std::time::Duration::from_secs);
    let mut rng = MazeRng::new(seed);
    let mut loss_history = Vec::new();
    let mut acc_history = Vec::new();
    let mut step = 0usize;

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

            // Encode: pixels → TokenInput
            let pixels = render_maze(&maze);
            let input = encoder.encode(&pixels);

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

        if step % 100 == 0 || step == steps {
            let window = 50.min(loss_history.len());
            if window > 0 {
                let smooth_loss: f32 = loss_history[loss_history.len()-window..].iter().sum::<f32>() / window as f32;
                let smooth_acc: f32 = acc_history[acc_history.len()-window..].iter().sum::<f32>() / window as f32;
                eprintln!("step {step:5}: loss={smooth_loss:.3} route_acc={:.1}%",
                    smooth_acc * 100.0);
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
        let ood_encoder = VisualRetina::maze(ood_size, ood_size);
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

/// Eval summary returned by `eval` so callers can compute gaps between
/// in-distribution and out-of-distribution runs (Hoel generalization test).
#[derive(Debug, Clone, Copy)]
struct EvalStats {
    first_step_acc: f32,
    per_step_acc: f32,
    avg_prefix: f32,
}

fn eval(
    w: &CtmWeights, encoder: &VisualRetina, _loss_fn: &dyn LossFn<Target = [usize]>,
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

        let pixels = render_maze(&maze);
        let input = encoder.encode(&pixels);

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
// 8-REGION BRAIN MODE
// ═══════════════════════════════════════════════════════════════

fn run_brain(
    maze_size: usize, ticks: usize, steps: usize, route_len: usize,
    lr: f32, seed: u64, batch_size: usize, imagination: bool, pain_mode: bool,
    plural_mode: bool, csv_mode: bool, cereb_size: usize, frozen_cereb: bool,
    autoresearch_summary: bool, budget_secs: Option<u64>,
    train_bank: Option<Vec<Maze>>, test_bank: Option<Vec<Maze>>,
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

    let encoder = VisualRetina::maze(maze_size, maze_size);
    let token_dim = encoder.token_dim();
    let out_dims = route_len * N_DIRECTIONS;

    let mut cfg = RegionalConfig::eight_region_small(token_dim, out_dims, ticks);

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

    let base_loss = RouteLoss::maze();
    let imagination_loss = Imagination::new(RouteLoss::maze());
    let loss_fn: &dyn LossFn<Target = [usize]> = if imagination {
        eprintln!("Loss: Imagination<RouteLoss>");
        &imagination_loss
    } else {
        eprintln!("Loss: RouteLoss (baseline)");
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

            let pixels = render_maze(&maze);
            let input = encoder.encode(&pixels);

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

            let (output, _state, cache) = if let Some(ref mut frozen) = frozen_model {
                RegionalBrain::forward_cached_frozen(&w, state, &input, frozen)
            } else {
                RegionalBrain::forward_cached(&w, state, &input)
            };
            let (loss, d_preds) = loss_fn.compute(
                &output.predictions, &output.certainties, &route);
            let sample_grads = RegionalBrain::backward(&w, cache, &d_preds);

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

        let pixels = render_maze(&maze);
        let input = encoder.encode(&pixels);
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
}

