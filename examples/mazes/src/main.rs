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
use modgrad_codec::retina::VisualRetina;
use modgrad_traits::{Brain, Encoder, LossFn, RouteLoss, TokenInput};

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
            "--help" | "-h" => {
                eprintln!("Usage: mazes [--size N] [--ticks N] [--steps N] [--route-len N]");
                eprintln!("             [--d-model N] [--lr F] [--batch N] [--no-adaptive]");
                return;
            }
            _ => { i += 1; }
        }
    }

    let maze_size = maze_size | 1;
    eprintln!("Maze benchmark: size={maze_size} ticks={ticks} d_model={d_model} \
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
    let loss_fn = RouteLoss::maze();

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
            Ctm::apply_gradients(&mut w, &batch_grads, effective_lr, 5.0);

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
    eval(&w, &encoder, &loss_fn, maze_size, route_len, seed + 999);
}

fn eval(
    w: &CtmWeights, encoder: &VisualRetina, _loss_fn: &RouteLoss,
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

