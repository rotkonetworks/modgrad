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
use modgrad_traits::{Brain, Encoder, LossFn, RouteLoss, Imagination, TokenInput};

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
            "--help" | "-h" => {
                eprintln!("Usage: mazes [--size N] [--ticks N] [--steps N] [--route-len N]");
                eprintln!("             [--d-model N] [--lr F] [--batch N] [--no-adaptive]");
                eprintln!("             [--imagination] [--brain] [--pain] [--plural] [--csv]");
                eprintln!("             [--cereb-size N]");
                return;
            }
            _ => { i += 1; }
        }
    }

    let maze_size = maze_size | 1;

    if brain {
        run_brain(maze_size, ticks, steps, route_len, lr, seed, batch_size, imagination, pain_mode, plural_mode, csv_mode, cereb_size);
        return;
    }

    eprintln!("Maze benchmark (single CTM): size={maze_size} ticks={ticks} d_model={d_model} \
               route_len={route_len} batch={batch_size}");

    // ── Encoder: visual retina → spatial tokens ──
    let encoder = VisualRetina::maze(maze_size, maze_size);
    let token_dim = encoder.token_dim();

    // Probe token count with dummy image
    let dummy = vec![0.0f32; 3 * maze_size * maze_size];
    let probe = encoder.encode(&dummy);
    eprintln!("Encoder: {maze_size}×{maze_size} pixels → {} spatial tokens × {token_dim}-dim",
        probe.n_tokens);

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
    let mut rng = MazeRng::new(seed);
    let mut loss_history = Vec::new();
    let mut acc_history = Vec::new();
    let mut step = 0usize;

    while step < steps {
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
    eprintln!("\n--- Evaluation (200 mazes) ---");
    eval(&w, &encoder, loss_fn, maze_size, route_len, seed + 999);
}

fn eval(
    w: &CtmWeights, encoder: &VisualRetina, _loss_fn: &dyn LossFn<Target = [usize]>,
    maze_size: usize, route_len: usize, seed: u64,
) {
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

    eprintln!("First step acc:     {first_correct}/{n_valid} ({:.1}%)",
        first_correct as f32 / n_valid.max(1) as f32 * 100.0);
    eprintln!("Per-step acc:       {route_correct}/{route_total} ({:.1}%)",
        if route_total > 0 { route_correct as f32 / route_total as f32 * 100.0 } else { 0.0 });
    eprintln!("Avg correct prefix: {avg_prefix:.1} steps (of {route_len})");
    eprintln!("Avg ticks used:     {avg_ticks:.1}");
}

// ═══════════════════════════════════════════════════════════════
// 8-REGION BRAIN MODE
// ═══════════════════════════════════════════════════════════════

fn run_brain(
    maze_size: usize, ticks: usize, steps: usize, route_len: usize,
    lr: f32, seed: u64, batch_size: usize, imagination: bool, pain_mode: bool,
    plural_mode: bool, csv_mode: bool, cereb_size: usize,
) {
    use std::io::Write;
    use modgrad_ctm::bio::homeostasis::Homeostasis;
    use modgrad_ctm::bio::neuromod::Neuromodulators;
    use modgrad_ctm::bio::pain::{self, PainConfig};
    use modgrad_ctm::bio::dream;
    use modgrad_ctm::memory::episodic::{self, EpisodicConfig, EpisodicMemory, ValenceReceipt};
    use modgrad_ctm::graph::AuxLossConfig;
    use modgrad_ctm::plural::{self, PluralSystem, SwitchPolicy, SwitchTrigger};

    let encoder = VisualRetina::maze(maze_size, maze_size);
    let token_dim = encoder.token_dim();
    let out_dims = route_len * N_DIRECTIONS;

    let mut cfg = RegionalConfig::eight_region_small(token_dim, out_dims, ticks);

    // Cerebellum: 64 neurons × 16 ticks is the sweet spot for this scale.
    // Tight recurrent forward model that trains well via aux losses.
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
    let pain_cfg = PainConfig::default();
    let pain_warmup = steps / 10;
    let episodic_cfg = EpisodicConfig {
        capacity: 512,
        max_ticks: ticks,
        d_model,
        min_ticks_for_storage: 1,
        min_surprise: 0.0,
        retrieval_threshold: 0.5,
        consolidation_threshold: 0.85,
        semantic_collapse_retrievals: 5,
        strength_decay: 0.95,
    };

    // Plural system: wraps homeostasis/neuromod/memory per personality
    let mut plural_sys = if plural_mode {
        let mut sys = PluralSystem::new("primary", n_regions, episodic_cfg.clone());
        sys.switch_policy = SwitchPolicy::Salience;
        sys.permeability = 0.3;
        Some(sys)
    } else {
        None
    };

    // Standalone pain state (used when plural_mode is off)
    let mut homeostasis = Homeostasis::default();
    let mut neuromod = Neuromodulators::default();
    let mut memory = EpisodicMemory::new(episodic_cfg.clone());

    // Shared across modes
    let mut batch_baseline = pain::LossBaseline::new(0.95);
    let mut pain_focus = dream::AdaptivePainFocus::new(route_len, 0.95);
    let mut prev_loss = 0.0f32;

    // Plural splitting state
    let mut steps_in_red = 0usize;
    let max_personalities = 4;
    let red_threshold_for_split = 50; // steps in red before splitting

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

    // #4: Curriculum — organism self-paces (currently logs zone, doesn't change maze size
    // because the encoder is fixed to maze_size. Future: use resizable encoder.)
    let current_maze_size = maze_size;

    for step in 1..=steps {
        let mut grads = RegionalGradients::zeros(&w);
        let mut batch_loss = 0.0f32;
        let mut batch_correct = 0usize;
        let mut batch_total = 0usize;
        let mut batch_valid = 0usize;
        let mut batch_first_correct = 0usize;
        let mut batch_first_total = 0usize;

        let effective_maze_size = current_maze_size;

        // Plural: check if we should switch personality before this batch
        if plural_mode && step > pain_warmup {
            if let Some(ref mut sys) = plural_sys {
                // Use batch baseline surprise as proxy for global sync
                let sync_proxy = vec![batch_baseline.expected(); 16];
                let motor_proxy = vec![prev_loss; 4];
                if let Some(target) = plural::should_switch(sys, &sync_proxy, &motor_proxy) {
                    let claim = plural::evaluate_claims(sys, &sync_proxy, &motor_proxy)
                        .into_iter().find(|(id, _)| *id == target)
                        .map(|(_, c)| c).unwrap_or(0.0);
                    let temp = std::mem::replace(sys, PluralSystem::new("_", n_regions, episodic_cfg.clone()));
                    *sys = plural::switch(temp, target, SwitchTrigger::Salience { claim });
                }
            }
        }

        for _ in 0..batch_size {
            let maze = generate_maze(effective_maze_size, &mut rng);
            if maze.path_length < 3 { continue; }

            let pixels = render_maze(&maze);
            let input = encoder.encode(&pixels);

            let mut route = maze.route.clone();
            route.truncate(route_len);
            while route.len() < route_len { route.push(DIR_WAIT); }

            // Get active personality's state (or standalone state)
            let (h, n, mem) = if let Some(ref mut sys) = plural_sys {
                let active = &mut sys.personalities[sys.active];
                (&mut active.homeostasis, &mut active.neuromod, &mut active.memory)
            } else {
                (&mut homeostasis, &mut neuromod, &mut memory)
            };

            // #2: Episodic retrieval + state priming
            let mut retrieval_valence = 0.0f32;
            let mut retrieval_result = None;
            if pain_mode && step > pain_warmup && mem.count > 0 {
                let query_len = d_model.min(input.tokens.len());
                let query = &input.tokens[..query_len];
                let result = episodic::retrieve(mem, query);
                if result.n_matches > 0 {
                    retrieval_valence = result.blended_valence;
                    pain::on_retrieval(h, n, retrieval_valence, &pain_cfg);
                    retrieval_result = Some(result);
                }
            }

            // #2: Prime the initial state from retrieval
            let mut state = RegionalBrain::init_state(&w);
            if let Some(ref retrieval) = retrieval_result {
                // Prime hippocampus region (index 7 in eight_region_small)
                let hippo_idx = w.config.region_names.iter()
                    .position(|n| n.contains("hippocampus"))
                    .unwrap_or(7);
                let blend = retrieval.best_similarity * 0.3; // scale blend by match quality
                dream::prime_state(&mut state.region_outputs, retrieval, blend, hippo_idx);
            }

            let (output, _state, cache) = RegionalBrain::forward_cached(&w, state, &input);

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

            // Per-position accuracy with adaptive pain focus
            let mut sample_correct = 0usize;
            let mut sample_total = 0usize;
            let mut position_valences = Vec::new();

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

                        // #5: Adaptive pain focus — weight from learned failure rates
                        if pain_mode && step > pain_warmup {
                            pain_focus.update(pos, correct);
                            let pos_weight = pain_focus.weight(pos);
                            let pos_loss = if correct { 0.0 } else { 1.0 } * pos_weight;

                            let confidence = if correct {
                                let logits = &pred[off..off + N_DIRECTIONS];
                                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                let sum_exp: f32 = logits.iter().map(|l| (l - max_l).exp()).sum();
                                (max_l - sum_exp.ln()).exp().clamp(0.0, 1.0)
                            } else {
                                0.5
                            };

                            // #7: Use per-batch baseline for surprise, accumulate per position
                            let surprise = batch_baseline.update(pos_loss);

                            let resp = pain::on_prediction(
                                h, n, surprise, pos_loss, confidence, correct, &pain_cfg,
                            );
                            position_valences.push(resp.valence_for_storage);

                            if correct && retrieval_valence < -0.2 {
                                pain::on_overcoming(
                                    h, n, retrieval_valence, true, &pain_cfg,
                                );
                            }
                        }
                    }
                }
            }
            batch_correct += sample_correct;
            batch_total += sample_total;

            // Store to episodic memory
            if pain_mode && step > pain_warmup {
                let avg_valence = if position_valences.is_empty() { 0.0 }
                    else { position_valences.iter().sum::<f32>() / position_valences.len() as f32 };
                let acc = if sample_total > 0 { sample_correct as f32 / sample_total as f32 } else { 0.0 };

                let traj_len = d_model * ticks;
                let mut traj = vec![0.0f32; traj_len];
                let copy_len = traj_len.min(input.tokens.len());
                traj[..copy_len].copy_from_slice(&input.tokens[..copy_len]);

                let cert = vec![[1.0 - acc, acc]; ticks];
                let receipt = ValenceReceipt {
                    valence: avg_valence,
                    loss: loss / route_len as f32,
                    confidence: acc,
                    correct: acc > 0.5,
                };
                let taken = std::mem::replace(mem, EpisodicMemory::new(episodic_cfg.clone()));
                let (m, _) = episodic::store_with_valence(
                    taken, &traj, &cert, &[], ticks, loss, Some(receipt),
                );
                *mem = m;
            } else if pain_mode {
                batch_baseline.update(loss / route_len as f32);
            }
        }

        first_step_hits += batch_first_correct;
        first_step_total += batch_first_total;

        if batch_valid > 0 {
            // Get active state for post-batch operations
            let (h, n, mem) = if let Some(ref mut sys) = plural_sys {
                let active = &mut sys.personalities[sys.active];
                (&mut active.homeostasis, &mut active.neuromod, &mut active.memory)
            } else {
                (&mut homeostasis, &mut neuromod, &mut memory)
            };

            if pain_mode && step > pain_warmup {
                opt.lr = lr * pain::lr_scale(n);
            }

            opt.step(&mut w, &mut grads);

            let avg_loss = batch_loss / batch_valid as f32;
            let avg_acc = batch_correct as f32 / batch_total.max(1) as f32;
            loss_history.push(avg_loss);
            acc_history.push(avg_acc);
            prev_loss = avg_loss;

            if pain_mode {
                h.tick_from_ctm(avg_loss, true, avg_loss);
            }

            // CSV telemetry
            if let Some(ref mut f) = csv_writer {
                let first_acc = if batch_first_total > 0 {
                    batch_first_correct as f32 / batch_first_total as f32
                } else { 0.0 };
                let (pressure, dopamine, serotonin, norepinephrine, current_lr) = if pain_mode {
                    (h.pressure, n.dopamine, n.serotonin, n.norepinephrine, opt.lr)
                } else {
                    (0.0, 0.0, 0.0, 0.0, lr)
                };
                let mem_count = if pain_mode { mem.count } else { 0 };
                writeln!(f, "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
                    step, avg_loss, avg_acc, pressure, dopamine, serotonin,
                    norepinephrine, current_lr, first_acc, mem_count)
                    .expect("failed to write CSV row");
            }

            // Adaptive sleep
            if pain_mode && step > pain_warmup && h.should_sleep() {
                let quality = if h.must_sleep() { 1.0 } else { 0.6 };
                h.on_sleep(quality);

                // Consolidate
                let taken = std::mem::replace(mem, EpisodicMemory::new(episodic_cfg.clone()));
                let (m, cons) = episodic::consolidate(taken);

                // Dream replay
                let (m, dream_result) = dream::dream_replay(
                    m, h, n,
                    &|_idx, key| {
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
                            prev_loss
                        };
                        let was_correct = loss < prev_loss;
                        (loss, was_correct)
                    },
                    10, &pain_cfg,
                );
                *mem = m;

                if step % 200 == 0 {
                    eprintln!("    [sleep] quality={quality:.1} merges={} overcame={} regressed={} mem={}",
                        cons.merges, dream_result.overcame, dream_result.regressed, mem.count);
                }
            }

            // Plural: pressure-driven splitting
            if plural_mode && step > pain_warmup {
                if h.should_sleep() || matches!(h.zone(), modgrad_ctm::bio::homeostasis::SleepZone::Red) {
                    steps_in_red += 1;
                } else {
                    steps_in_red = steps_in_red.saturating_sub(1);
                }
            }
        }

        // Plural: fork when sustained red — the unified self can't cope
        if plural_mode && step > pain_warmup && steps_in_red >= red_threshold_for_split {
            if let Some(ref mut sys) = plural_sys {
                if sys.personalities.len() < max_personalities {
                    let n_alters = sys.personalities.len();
                    let name = format!("alter_{n_alters}");
                    eprintln!("    [SPLIT] personality forking → '{}' (sustained red for {} steps)",
                        name, steps_in_red);

                    // Fork from active
                    let temp = std::mem::replace(sys, PluralSystem::new("_", n_regions, episodic_cfg.clone()));
                    let mut forked = plural::fork_active(temp, &name);

                    // Differentiate: opposite temperament to handle what parent can't
                    let new_id = forked.personalities.len() - 1;
                    let parent = &forked.personalities[forked.active];
                    let parent_ne = parent.neuromod.norepinephrine;
                    let parent_da = parent.neuromod.dopamine;
                    let new_p = &mut forked.personalities[new_id];

                    // If parent is high-anxiety, new alter is calm explorer
                    // If parent is low-arousal, new alter is vigilant
                    if parent_ne > 1.0 {
                        new_p.neuromod.norepinephrine = 0.3;
                        new_p.neuromod.dopamine = 2.0;
                        new_p.neuromod.curiosity = 1.5;
                        new_p.neuromod.serotonin = 1.5;
                    } else {
                        new_p.neuromod.norepinephrine = 1.5;
                        new_p.neuromod.dopamine = 0.8;
                        new_p.neuromod.curiosity = 0.3;
                        new_p.neuromod.serotonin = 0.8;
                    }

                    // New personality is co-conscious with all others
                    forked.co_conscious = (0..forked.personalities.len()).collect();

                    *sys = forked;
                    steps_in_red = 0;
                }
            }
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
                    let weakest = pain_focus.weakest_position();
                    let (p, da, sht, ne, mc) = if let Some(ref sys) = plural_sys {
                        let a = &sys.personalities[sys.active];
                        (a.homeostasis.pressure, a.neuromod.dopamine, a.neuromod.serotonin,
                         a.neuromod.norepinephrine, a.memory.count)
                    } else {
                        (homeostasis.pressure, neuromod.dopamine, neuromod.serotonin,
                         neuromod.norepinephrine, memory.count)
                    };
                    let plural_info = if let Some(ref sys) = plural_sys {
                        format!(" pers={}/{} \"{}\"", sys.active, sys.personalities.len(),
                            sys.personalities[sys.active].name)
                    } else {
                        String::new()
                    };
                    eprintln!("step {step:5}: loss={smooth_loss:.3} route={:.1}% first={first_pct:.1}% | p={p:.2} DA={da:.2} 5HT={sht:.2} NE={ne:.2} lr={:.5} mem={mc} weak=pos{weakest}{plural_info}",
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

    for _ in 0..200 {
        let maze = generate_maze(maze_size, &mut eval_rng);
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
    eprintln!("First step acc:     {first_correct}/{n_valid} ({:.1}%)",
        first_correct as f32 / n_valid.max(1) as f32 * 100.0);
    eprintln!("Per-step acc:       {route_correct}/{route_total} ({:.1}%)",
        if route_total > 0 { route_correct as f32 / route_total as f32 * 100.0 } else { 0.0 });
    eprintln!("Avg correct prefix: {avg_prefix:.1} steps (of {route_len})");
}

