//! MILESTONE 2 trainer/exporter: full 8-region brain WITH visual retina,
//! TRAINED to navigate 9x9 mazes (next-move prediction toward goal).
//!
//! Same exact brain the parity-validated engine consumes:
//! `Genome::blank_slate(9,9,4,8).express()` (EightRegionSmall + VisualCortex
//! retina, out_dims=4, outer_ticks=8). The forward path is
//! maze → render_pixels → cortex.spatial_tokens → regional_forward — the
//! retina is the whole point, NOT a flat hand-encoding.
//!
//! Training: grad-accumulation mini-batching with `regional_train_step` +
//! `RegionalAdamW` (same API as `brain_export.rs`), BFS optimal-move labels,
//! lr decay. Metrics: max-certainty move accuracy + greedy solve rate.
//!
//! Export is byte-for-byte the SAME path as `brain_oracle.rs`: trained
//! `brain_weights.json` (cortex + RegionalWeights MINUS embeddings) +
//! `brain_reference.json` (config echo + per-maze pixels/observation +
//! per-outer-tick traces). Plus curated demo mazes the trained brain solves
//! and oracle traces, so the live demo always works.
//!
//! Usage:
//!   cargo run -p mazes --bin brain_train --release [--features rocm] -- <out_dir>
//!
//! GPU (rocm) is attempted by the caller; if it crashes, fall back to CPU.

#![allow(dead_code)]

#[path = "../maze_gen.rs"]
mod maze_gen;
use maze_gen::{Maze, MazeRng, generate_maze, DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT};

use modgrad_codec::genome::{Genome, ExpressedBrain};
use modgrad_codec::retina::VisualCortex;
use modgrad_ctm::graph::{
    RegionalWeights, RegionalState, RegionalGradients, RegionalAdamW,
    regional_forward, regional_train_step,
};
use modgrad_ctm::forward::{ctm_forward, CtmInput};
use serde::Serialize;
use serde_json::Value;
use std::collections::VecDeque;

const SIZE: usize = 9;
const OUT_DIMS: usize = 4;
const TICKS: usize = 8;
const BATCH: usize = 16;
const SEED: u64 = 42;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

// ─── Maze rendering → RGB pixels [3 × H × W] (verbatim from brain_oracle) ──
//
//   wall  → (0.1, 0.1, 0.1)
//   open  → (0.8, 0.8, 0.8)
//   agent → (0.1, 0.4, 0.9)
//   goal  → (0.9, 0.3, 0.1)
//
// Layout is CHW: channel-major, [R plane (H×W)][G plane][B plane].
fn render_pixels(grid: &[u8], size: usize, agent: (usize, usize), goal: (usize, usize)) -> Vec<f32> {
    let n = size * size;
    let mut px = vec![0.0f32; 3 * n];
    for r in 0..size {
        for c in 0..size {
            let idx = r * size + c;
            let (rr, gg, bb) = if (r, c) == agent {
                (0.1, 0.4, 0.9)
            } else if (r, c) == goal {
                (0.9, 0.3, 0.1)
            } else if grid[idx] != 0 {
                (0.1, 0.1, 0.1)
            } else {
                (0.8, 0.8, 0.8)
            };
            px[idx] = rr;
            px[n + idx] = gg;
            px[2 * n + idx] = bb;
        }
    }
    px
}

/// Render a `Maze` (grid is `Vec<bool>`, true=wall) at `agent`.
fn render_maze_pixels(maze: &Maze, agent: (usize, usize)) -> Vec<f32> {
    let grid: Vec<u8> = maze.grid.iter().map(|&b| b as u8).collect();
    render_pixels(&grid, maze.grid_size, agent, maze.end)
}

// ─── BFS optimal-move labels (verbatim from export.rs / brain_export.rs) ──

fn dist_to_goal(grid: &[bool], size: usize, end: (usize, usize)) -> Vec<u32> {
    let mut dist = vec![u32::MAX; size * size];
    let mut q = VecDeque::new();
    dist[end.0 * size + end.1] = 0;
    q.push_back(end);
    while let Some((r, c)) = q.pop_front() {
        let d = dist[r * size + c];
        for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nc < 0 || nr >= size as i32 || nc >= size as i32 { continue; }
            let (nr, nc) = (nr as usize, nc as usize);
            let idx = nr * size + nc;
            if !grid[idx] && dist[idx] == u32::MAX {
                dist[idx] = d + 1;
                q.push_back((nr, nc));
            }
        }
    }
    dist
}

fn optimal_move(dist: &[u32], grid: &[bool], size: usize, cell: (usize, usize)) -> Option<usize> {
    let (r, c) = cell;
    if dist[r * size + c] == 0 { return None; }
    let cand = [
        (r.wrapping_sub(1), c, DIR_UP),
        (r + 1, c, DIR_DOWN),
        (r, c.wrapping_sub(1), DIR_LEFT),
        (r, c + 1, DIR_RIGHT),
    ];
    let mut best: Option<(u32, usize)> = None;
    for (nr, nc, dir) in cand {
        if nr >= size || nc >= size { continue; }
        let idx = nr * size + nc;
        if grid[idx] { continue; }
        let d = dist[idx];
        if d == u32::MAX { continue; }
        if best.map_or(true, |(bd, _)| d < bd) { best = Some((d, dir)); }
    }
    best.map(|(_, dir)| dir)
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
}

fn committed_pred<'a>(out: &'a modgrad_ctm::graph::RegionalOutput) -> &'a [f32] {
    let t = out.ticks_used.saturating_sub(1).min(out.predictions.len().saturating_sub(1));
    &out.predictions[t]
}

fn step_cell(cell: (usize, usize), dir: usize) -> (usize, usize) {
    let (r, c) = cell;
    match dir {
        DIR_UP => (r.wrapping_sub(1), c),
        DIR_DOWN => (r + 1, c),
        DIR_LEFT => (r, c.wrapping_sub(1)),
        _ => (r, c + 1),
    }
}

fn random_open_cell(maze: &Maze, dist: &[u32], rng: &mut MazeRng) -> Option<(usize, usize)> {
    let s = maze.grid_size;
    let cells: Vec<(usize, usize)> = (0..s * s)
        .filter(|&i| !maze.grid[i] && dist[i] != u32::MAX && dist[i] != 0)
        .map(|i| (i / s, i % s))
        .collect();
    if cells.is_empty() { return None; }
    Some(cells[rng.range(cells.len())])
}

/// One retina+brain forward at `agent`, returning the committed move.
fn predict_move(w: &RegionalWeights, cortex: &VisualCortex, maze: &Maze, agent: (usize, usize)) -> usize {
    let pixels = render_maze_pixels(maze, agent);
    let (obs, _n, _d) = cortex.spatial_tokens(&pixels);
    let mut st = RegionalState::new(w);
    let out = regional_forward(w, &mut st, &obs);
    argmax(committed_pred(&out))
}

/// Greedy rollout from start; true if the brain reaches the goal.
fn solves(w: &RegionalWeights, cortex: &VisualCortex, maze: &Maze) -> bool {
    let s = maze.grid_size;
    let mut agent = maze.start;
    let budget = maze.path_length * 4 + 8;
    for _ in 0..budget {
        if agent == maze.end { return true; }
        let dir = predict_move(w, cortex, maze, agent);
        let nx = step_cell(agent, dir);
        if nx.0 >= s || nx.1 >= s || maze.grid[nx.0 * s + nx.1] { return false; }
        agent = nx;
    }
    agent == maze.end
}

/// Held-out metrics: optimal-move accuracy + greedy solve rate.
fn quick_eval(w: &RegionalWeights, cortex: &VisualCortex, n_mazes: usize, seed: u64) -> (f32, f32) {
    let mut erng = MazeRng::new(seed);
    let (mut mv_c, mut mv_t) = (0usize, 0usize);
    let mut solved = 0usize;
    for _ in 0..n_mazes {
        let maze = generate_maze(SIZE, &mut erng);
        let dist = dist_to_goal(&maze.grid, SIZE, maze.end);
        for _ in 0..4 {
            if let Some(cell) = random_open_cell(&maze, &dist, &mut erng) {
                if let Some(tgt) = optimal_move(&dist, &maze.grid, SIZE, cell) {
                    if predict_move(w, cortex, &maze, cell) == tgt { mv_c += 1; }
                    mv_t += 1;
                }
            }
        }
        if solves(w, cortex, &maze) { solved += 1; }
    }
    (mv_c as f32 / mv_t.max(1) as f32, solved as f32 / n_mazes as f32)
}

// ─── Faithful replica of regional_forward with per-tick capture ─────
// (verbatim from brain_oracle.rs — guarantees the dumped traces match the
//  engine, which is bit-exact against this replica.)
struct TraceOut {
    predictions: Vec<Vec<f32>>,
    region_acts: Vec<Vec<Vec<f32>>>,
    global_sync: Vec<Vec<f32>>,
    exit_lambdas: Vec<f32>,
    ticks_used: usize,
}

fn regional_forward_traced(
    w: &RegionalWeights,
    state: &mut RegionalState,
    observation: &[f32],
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

        let mut results: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
        for r in 0..n_regions {
            let d_input = w.regions[r].config.d_input;
            let _out = ctm_forward(&w.regions[r], &mut state.region_states[r], CtmInput::Raw {
                obs: &region_obs[r], n_tokens: 1, raw_dim: d_input,
            });
            results.push(state.region_states[r].activated.clone());
        }

        for r in 0..n_regions {
            state.region_outputs[r] = results[r].clone();
        }
        prev_outputs = state.region_outputs.clone();

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

        let prediction = w.output_proj.forward(&gs_buf);

        predictions.push(prediction);
        region_acts.push(state.region_outputs.clone());
        global_sync_trace.push(gs_buf.clone());

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

// ─── Serializable reference (verbatim schema from brain_oracle.rs) ──────

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

/// Curated demo maze the trained brain actually solves (greedy rollout).
#[derive(Serialize)]
struct DemoMaze {
    grid: Vec<u8>,
    start: [usize; 2],
    end: [usize; 2],
    path_length: usize,
}

#[derive(Serialize)]
struct Reference {
    task: String,
    size: usize,
    ticks: usize,
    out_dims: usize,
    raw_obs_dim: usize,
    obs_scale_dims: Vec<usize>,
    n_global_sync: usize,
    move_acc: f32,
    solve_rate: f32,
    region_names: Vec<String>,
    regions: Vec<RegionInfo>,
    connections: Vec<ConnectionInfo>,
    // Engine parity samples (the brain.rs test reads `samples`).
    samples: Vec<MazeSample>,
    // Curated demo mazes + oracle traces for the live /play demo.
    demo_mazes: Vec<DemoMaze>,
    oracle: Vec<MazeSample>,
}

/// Build a full traced MazeSample for one (maze, agent) via the retina.
fn build_sample(w: &RegionalWeights, cortex: &VisualCortex, grid: &[u8], agent: (usize, usize), goal: (usize, usize)) -> MazeSample {
    let pixels = render_pixels(grid, SIZE, agent, goal);
    let (tokens, n_tokens, token_dim) = cortex.spatial_tokens(&pixels);

    let mut traced_state = RegionalState::new(w);
    let traced = regional_forward_traced(w, &mut traced_state, &tokens);

    // Cross-check against the real regional_forward (predictions only).
    let mut real_state = RegionalState::new(w);
    let real_out = regional_forward(w, &mut real_state, &tokens);
    assert_eq!(real_out.predictions.len(), traced.predictions.len(),
        "tick count mismatch real {} vs traced {}",
        real_out.predictions.len(), traced.predictions.len());
    for (t, (rp, tp)) in real_out.predictions.iter().zip(&traced.predictions).enumerate() {
        for (j, (&a, &b)) in rp.iter().zip(tp).enumerate() {
            let d = (a - b).abs();
            assert!(d < 1e-4, "tick {t} pred[{j}]: traced {b} vs real {a} (|d|={d})");
        }
    }

    let ticks: Vec<TickTrace> = (0..traced.ticks_used).map(|t| TickTrace {
        prediction: traced.predictions[t].clone(),
        region_activations: traced.region_acts[t].clone(),
        global_sync: traced.global_sync[t].clone(),
        exit_lambda: traced.exit_lambdas.get(t).copied(),
    }).collect();

    MazeSample {
        grid: grid.to_vec(),
        agent: [agent.0, agent.1],
        goal: [goal.0, goal.1],
        pixels,
        observation: tokens,
        n_tokens, token_dim,
        ticks_used: traced.ticks_used,
        ticks,
    }
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let batches = env_usize("BATCHES", 1200);
    let lr = env_f32("LR", 1.5e-3);

    // Build the brain from the real genome API — EXACT engine config.
    let genome = Genome::blank_slate(SIZE, SIZE, OUT_DIMS, TICKS);
    let ExpressedBrain { cortex, mut config } = genome.express();

    eprintln!(
        "blank_slate({SIZE},{SIZE},{OUT_DIMS},{TICKS}) → {} regions, n_global_sync={}, out_dims={}, raw_obs_dim={} (per-token), obs_scale_dims={:?} (PRE-FIX)",
        config.regions.len(), config.n_global_sync, config.out_dims,
        config.raw_obs_dim, config.obs_scale_dims,
    );

    // ── OBSERVATION-BOTTLENECK FIX ─────────────────────────────────────
    // `express()` sets raw_obs_dim = token_dim (128) — the PER-TOKEN dim —
    // but the VisualCortex retina emits a FLAT observation of
    // n_tokens × token_dim (25 × 128 = 3200) floats for a 9×9 maze, and
    // `regional_forward` feeds that ENTIRE flat vector into obs_proj and
    // (via `src.extend_from_slice(observation)`) into every
    // `receives_observation` connection synapse. With raw_obs_dim=128 those
    // projections read only the first 128 floats = a single retina token =
    // the maze's top-left corner, so the brain is blind to everything else.
    //
    // Fix: widen raw_obs_dim to the TRUE flattened retina dim and make it a
    // single observation scale spanning the whole vector. `RegionalWeights::new`
    // then sizes obs_proj as (3200 → regions[0].d_input) and each
    // receives_observation synapse as (Σ source d_models + 3200 → target d_input).
    // Nothing else (region dims, connections, etc.) changes: region d_input is
    // independent of raw_obs_dim in eight_region_small, and obs_scale_slice(0)
    // returns (0, 3200) so the synapse-side concat covers the full observation.
    let true_obs_dim = {
        let probe = render_pixels(&vec![0u8; SIZE * SIZE], SIZE, (1, 1), (7, 7));
        let (obs, n_tokens, token_dim) = cortex.spatial_tokens(&probe);
        assert_eq!(token_dim, config.raw_obs_dim,
            "retina token_dim {token_dim} != expressed raw_obs_dim {} (per-token)", config.raw_obs_dim);
        assert_eq!(obs.len(), n_tokens * token_dim, "retina obs length");
        obs.len()
    };
    config.raw_obs_dim = true_obs_dim;
    config.obs_scale_dims = vec![true_obs_dim];
    eprintln!(
        "FIXED raw_obs_dim={} obs_scale_dims={:?} (full flattened retina, single scale)",
        config.raw_obs_dim, config.obs_scale_dims,
    );

    // Snapshot config metadata for the reference echo (before move into weights).
    let region_infos: Vec<RegionInfo> = config.region_names.iter().zip(&config.regions)
        .map(|(name, rc)| RegionInfo {
            name: name.clone(), d_model: rc.d_model, d_input: rc.d_input,
            memory_length: rc.memory_length, iterations: rc.iterations,
        }).collect();
    let conn_infos: Vec<ConnectionInfo> = config.connections.iter().map(|c| ConnectionInfo {
        from: c.from.clone(), to: c.to,
        receives_observation: c.receives_observation,
        observation_scale: c.observation_scale,
    }).collect();
    let region_names = config.region_names.clone();
    let raw_obs_dim = config.raw_obs_dim;
    let obs_scale_dims = config.obs_scale_dims.clone();
    let n_global_sync = config.n_global_sync;

    let mut w = RegionalWeights::new(config);
    eprintln!("brain params: {}", w.n_params());

    // Probe the retina output dim once (so obs feeding train_step is right).
    // Post-fix invariant: the FLAT observation length == raw_obs_dim, and
    // obs_proj is sized to read all of it.
    {
        let probe = render_pixels(&vec![0u8; SIZE * SIZE], SIZE, (1, 1), (7, 7));
        let (obs, n_tokens, token_dim) = cortex.spatial_tokens(&probe);
        eprintln!("retina: n_tokens={n_tokens} token_dim={token_dim} obs_len={} (raw_obs_dim={raw_obs_dim})",
            obs.len());
        assert_eq!(obs.len(), n_tokens * token_dim, "retina obs length");
        assert_eq!(obs.len(), raw_obs_dim,
            "flat retina obs len {} != config raw_obs_dim {raw_obs_dim} (fix not applied)", obs.len());
        assert_eq!(w.obs_proj.in_dim, raw_obs_dim,
            "obs_proj in_dim {} != raw_obs_dim {raw_obs_dim}", w.obs_proj.in_dim);
    }

    // ── Train: grad-accumulation mini-batch, BFS optimal-move labels. ──
    eprintln!("training: batch={BATCH}x{batches} lr={lr}");
    let mut opt = RegionalAdamW::new(&w).with_lr(lr).with_clip(5.0);
    let mut rng = MazeRng::new(SEED);
    let t0 = std::time::Instant::now();

    for b in 0..batches {
        let mut grads = RegionalGradients::zeros(&w);
        let mut filled = 0;
        let mut batch_loss = 0.0f32;
        while filled < BATCH {
            let maze = generate_maze(SIZE, &mut rng);
            let dist = dist_to_goal(&maze.grid, SIZE, maze.end);
            let Some(cell) = random_open_cell(&maze, &dist, &mut rng) else { continue };
            let Some(target) = optimal_move(&dist, &maze.grid, SIZE, cell) else { continue };
            let pixels = render_maze_pixels(&maze, cell);
            let (obs, _n, _d) = cortex.spatial_tokens(&pixels);
            let (loss, _pred) = regional_train_step(&w, &mut grads, &obs, target);
            batch_loss += loss;
            filled += 1;
        }
        opt.lr = lr * (1.0 - 0.9 * b as f32 / batches as f32);
        opt.step(&mut w, &mut grads);

        if b % 50 == 0 || b == batches - 1 {
            let (mv, solve) = quick_eval(&w, &cortex, 40, 9999);
            eprintln!(
                "  batch {b:4}/{batches}: loss={:.3}  move_acc={:.1}%  solve_rate={:.1}%  ({:.0}s)",
                batch_loss / BATCH as f32, mv * 100.0, solve * 100.0, t0.elapsed().as_secs_f32()
            );
        }
    }

    let (move_acc, solve_rate) = quick_eval(&w, &cortex, 300, 12345);
    eprintln!(
        "FINAL: move_acc={:.1}%  solve_rate={:.1}%  ({:.0}s total)",
        move_acc * 100.0, solve_rate * 100.0, t0.elapsed().as_secs_f32()
    );

    // ── Parity samples: same 4 fixed mazes brain_oracle used (start cell). ──
    // These give the engine's brain.rs test deterministic samples to match.
    let parse = |rows: [&str; SIZE]| -> Vec<u8> {
        let mut g = vec![0u8; SIZE * SIZE];
        for (r, row) in rows.iter().enumerate() {
            for (c, ch) in row.chars().enumerate() {
                g[r * SIZE + c] = if ch == '#' { 1 } else { 0 };
            }
        }
        g
    };
    let fixed: Vec<(Vec<u8>, (usize, usize), (usize, usize))> = vec![
        (parse([
            "#########","#.......#","#.#####.#","#.#...#.#","#.#.#.#.#",
            "#.#.#.#.#","#...#...#","#.#####.#","#########",
        ]), (1, 1), (7, 7)),
        (parse([
            "#########","#.#.....#","#.#.###.#","#...#...#","###.#.###",
            "#...#...#","#.###.#.#","#.....#.#","#########",
        ]), (1, 1), (7, 7)),
        (parse([
            "#########","#.......#","#######.#","#.......#","#.#######",
            "#.......#","#######.#","#.......#","#########",
        ]), (1, 1), (7, 7)),
        (parse([
            "#########","#...#...#","#.#.#.#.#","#.#...#.#","#.#####.#",
            "#.....#.#","#####.#.#","#.......#","#########",
        ]), (1, 1), (7, 7)),
    ];
    let samples: Vec<MazeSample> = fixed.iter()
        .map(|(grid, agent, goal)| build_sample(&w, &cortex, grid, *agent, *goal))
        .collect();

    // ── Curate demo mazes the trained brain SOLVES + oracle traces. ──
    let mut crng = MazeRng::new(777);
    let mut demo_mazes: Vec<DemoMaze> = Vec::new();
    let mut oracle: Vec<MazeSample> = Vec::new();
    let mut tries = 0;
    while demo_mazes.len() < 16 && tries < 20000 {
        tries += 1;
        let maze = generate_maze(SIZE, &mut crng);
        if maze.path_length < 3 { continue; }
        if !solves(&w, &cortex, &maze) { continue; }
        if oracle.len() < 6 {
            let grid: Vec<u8> = maze.grid.iter().map(|&b| b as u8).collect();
            oracle.push(build_sample(&w, &cortex, &grid, maze.start, maze.end));
        }
        demo_mazes.push(DemoMaze {
            grid: maze.grid.iter().map(|&b| b as u8).collect(),
            start: [maze.start.0, maze.start.1],
            end: [maze.end.0, maze.end.1],
            path_length: maze.path_length,
        });
    }
    eprintln!("curated {} demo mazes ({} oracle traces) in {tries} tries",
        demo_mazes.len(), oracle.len());

    let reference = Reference {
        task: "brain_maze".to_string(),
        size: SIZE, ticks: TICKS, out_dims: OUT_DIMS,
        raw_obs_dim, obs_scale_dims, n_global_sync,
        move_acc, solve_rate,
        region_names, regions: region_infos, connections: conn_infos,
        samples, demo_mazes, oracle,
    };

    // ── Serialize weights MINUS embeddings (verbatim stripping). ──
    let mut weights_val: Value = serde_json::to_value(&w).expect("serialize weights to Value");
    if let Some(obj) = weights_val.as_object_mut() {
        obj.remove("embeddings");
    }
    let cortex_val: Value = serde_json::to_value(&cortex).expect("serialize cortex");

    #[derive(Serialize)]
    struct BrainWeights {
        cortex: Value,
        regional: Value,
    }
    let brain_weights = BrainWeights { cortex: cortex_val, regional: weights_val };

    let w_path = format!("{out_dir}/brain_weights.json");
    let r_path = format!("{out_dir}/brain_reference.json");
    let w_str = serde_json::to_string(&brain_weights).expect("serialize brain_weights");
    std::fs::write(&w_path, &w_str).expect("write brain_weights.json");
    std::fs::write(&r_path, serde_json::to_string(&reference).expect("serialize reference"))
        .expect("write brain_reference.json");

    eprintln!("wrote {} ({} bytes) + {}", w_path, w_str.len(), r_path);
}
