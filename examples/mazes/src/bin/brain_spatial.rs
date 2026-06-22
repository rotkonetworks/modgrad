//! SPATIAL-input trainer: the SAME parity-validated 8-region brain
//! (`Genome::blank_slate(s,s,4,8).express()` → EightRegionSmall, out_dims=4,
//! outer_ticks=8) as `brain_flat.rs`, but the INPUT region (region 0) is
//! reconfigured to be SPATIAL: instead of flattening the maze into one
//! d_input vector, it attends over the maze cells as `size*size` tokens of
//! `RAW_DIM=9` features each — the EXACT mechanism the single CTM uses to
//! solve (`ctm_forward` with `n_tokens = num_cells`, per-cell attention).
//!
//! Make-or-break question: does a brain whose INPUT region attends over
//! cells (instead of flattening) actually learn to navigate?
//!
//! Forward path: maze → encode (size*size tokens × 9 raw features) →
//! regional_forward. The INPUT region runs `CtmInput::Raw { n_tokens=cells,
//! raw_dim=9 }` over the RAW observation (its kv_proj is sized 9 → d_input);
//! the other 7 regions stay flat (n_tokens=1) and compose over INPUT's
//! single sync-vector output. NO retina, NO cortex.
//!
//! Training: grad-accumulation mini-batching with `regional_train_step` +
//! `RegionalAdamW`, BFS optimal-move labels, lr decay. Metrics: max-certainty
//! move accuracy + greedy solve rate. Progress is logged to stderr AND a
//! per-run progress file so we never lose it to a watchdog kill.
//!
//! Env knobs:
//!   SIZE     odd grid size (default 7; 7×7 is where the single CTM gets 59%)
//!   BATCHES  number of grad-accum batches (default 1200)
//!   LR       initial learning rate (default 1.5e-3)
//!   TICKS    outer ticks (default 8)
//!   LOG      progress log path (default /tmp/brain_spatial_<size>.log)
//!
//! Usage: SIZE=7 cargo run -p mazes --bin brain_spatial --release

#![allow(dead_code)]

#[path = "../maze_gen.rs"]
mod maze_gen;
use maze_gen::{Maze, MazeRng, generate_maze, DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT};

use modgrad_codec::genome::{Genome, ExpressedBrain};
use modgrad_ctm::graph::{
    RegionalWeights, RegionalState, RegionalGradients, RegionalAdamW,
    regional_forward, regional_train_step,
};
use std::collections::VecDeque;
use std::io::Write;

const OUT_DIMS: usize = 4;   // UP, DOWN, LEFT, RIGHT
const TICKS: usize = 8;      // outer ticks — same as the engine-validated brain
// per cell: [is_wall, is_agent, is_goal, wall_up, wall_down, wall_left, wall_right, x_norm, y_norm]
const RAW_DIM: usize = 9;
const BATCH: usize = 16;     // grad-accumulation batch
const SEED: u64 = 42;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

// ─── BFS optimal-move labels (verbatim from export.rs) ──

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

/// SPATIAL encoding: `size*size` tokens × RAW_DIM features, laid out as
/// `size*size` contiguous tokens (row-major cell order), each token:
///   [is_wall, is_agent, is_goal, wall_up, wall_down, wall_left, wall_right, x_norm, y_norm]
/// where x_norm = c/(s-1), y_norm = r/(s-1). OOB neighbors count as walls.
/// This is the SAME per-cell encoding as the single CTM (export.rs `encode`),
/// kept as `size*size` tokens of 9 (NOT flattened into one vector). The flat
/// vector layout is identical bytes; the difference is purely how the INPUT
/// region consumes it (n_tokens=cells via attention vs n_tokens=1 flatten).
fn encode(maze: &Maze, agent: (usize, usize)) -> Vec<f32> {
    let s = maze.grid_size;
    let mut t = vec![0.0f32; s * s * RAW_DIM];
    let denom = (s - 1) as f32;
    let wall = |r: i32, c: i32| -> f32 {
        if r < 0 || c < 0 || r >= s as i32 || c >= s as i32 { 1.0 } // OOB = wall
        else if maze.grid[r as usize * s + c as usize] { 1.0 } else { 0.0 }
    };
    for r in 0..s {
        for c in 0..s {
            let i = (r * s + c) * RAW_DIM;
            let (ri, ci) = (r as i32, c as i32);
            t[i] = wall(ri, ci);
            t[i + 1] = if (r, c) == agent { 1.0 } else { 0.0 };
            t[i + 2] = if (r, c) == maze.end { 1.0 } else { 0.0 };
            t[i + 3] = wall(ri - 1, ci); // up blocked
            t[i + 4] = wall(ri + 1, ci); // down blocked
            t[i + 5] = wall(ri, ci - 1); // left blocked
            t[i + 6] = wall(ri, ci + 1); // right blocked
            t[i + 7] = c as f32 / denom; // x_norm
            t[i + 8] = r as f32 / denom; // y_norm
        }
    }
    t
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

/// One brain forward at `agent`, returning the committed move.
fn predict_move(w: &RegionalWeights, maze: &Maze, agent: (usize, usize)) -> usize {
    let obs = encode(maze, agent);
    let mut st = RegionalState::new(w);
    let out = regional_forward(w, &mut st, &obs);
    argmax(committed_pred(&out))
}

/// Greedy rollout from start; true if the brain reaches the goal.
fn solves(w: &RegionalWeights, maze: &Maze) -> bool {
    let s = maze.grid_size;
    let mut agent = maze.start;
    let budget = maze.path_length * 4 + 8;
    for _ in 0..budget {
        if agent == maze.end { return true; }
        let dir = predict_move(w, maze, agent);
        let nx = step_cell(agent, dir);
        if nx.0 >= s || nx.1 >= s || maze.grid[nx.0 * s + nx.1] { return false; }
        agent = nx;
    }
    agent == maze.end
}

/// Held-out metrics: optimal-move accuracy + greedy solve rate.
fn quick_eval(w: &RegionalWeights, size: usize, n_mazes: usize, seed: u64) -> (f32, f32) {
    let mut erng = MazeRng::new(seed);
    let (mut mv_c, mut mv_t) = (0usize, 0usize);
    let mut solved = 0usize;
    for _ in 0..n_mazes {
        let maze = generate_maze(size, &mut erng);
        let dist = dist_to_goal(&maze.grid, size, maze.end);
        for _ in 0..4 {
            if let Some(cell) = random_open_cell(&maze, &dist, &mut erng) {
                if let Some(tgt) = optimal_move(&dist, &maze.grid, size, cell) {
                    if predict_move(w, &maze, cell) == tgt { mv_c += 1; }
                    mv_t += 1;
                }
            }
        }
        if solves(w, &maze) { solved += 1; }
    }
    (mv_c as f32 / mv_t.max(1) as f32, solved as f32 / n_mazes as f32)
}

fn main() {
    let size = env_usize("SIZE", 7);            // 7×7 default (CTM gets 59% here)
    let batches = env_usize("BATCHES", 1200);
    let lr = env_f32("LR", 1.5e-3);
    let ticks = env_usize("TICKS", TICKS);
    let n_cells = size * size;
    let od = n_cells * RAW_DIM;                 // flat obs length (= cells × 9)
    let log_path = std::env::var("LOG")
        .unwrap_or_else(|_| format!("/tmp/brain_spatial_{size}.log"));
    let mut logf = std::fs::File::create(&log_path).expect("create log");
    let mut log = |s: &str| {
        eprintln!("{s}");
        let _ = writeln!(logf, "{s}");
        let _ = logf.flush();
    };

    // Build the brain from the real genome API — EXACT engine config (sans retina).
    let genome = Genome::blank_slate(size, size, OUT_DIMS, ticks);
    let ExpressedBrain { cortex: _cortex, mut config } = genome.express();

    log(&format!(
        "blank_slate({size},{size},{OUT_DIMS},{ticks}) → {} regions, n_global_sync={}, out_dims={}, raw_obs_dim={} (PRE-FIX). SPATIAL obs: {n_cells} tokens × {RAW_DIM}",
        config.regions.len(), config.n_global_sync, config.out_dims, config.raw_obs_dim,
    ));

    // ── OBSERVATION FIX: raw_obs_dim = full flat encoding length (so obs_proj
    // and any flat fallback see the whole encoding). The INPUT region will
    // bypass this and attend over the raw tokens directly.
    config.raw_obs_dim = od;
    config.obs_scale_dims = vec![od];

    // ── SPATIAL INPUT REGION: reconfigure region 0 to attend over the maze
    // cells as `n_cells` tokens × RAW_DIM. RegionalWeights::new sizes its
    // kv_proj as RAW_DIM → d_input because spatial = Some((_, RAW_DIM)).
    let input_d_input = config.regions[0].d_input;
    config.regions[0].spatial = Some((n_cells, RAW_DIM));
    log(&format!(
        "INPUT region spatial = Some(({n_cells}, {RAW_DIM})); input d_model={}, d_input={input_d_input} (kv_proj now {RAW_DIM}→{input_d_input})",
        config.regions[0].d_model,
    ));

    let mut w = RegionalWeights::new(config);
    log(&format!("brain params: {}", w.n_params()));

    // Invariant: the spatial INPUT region's kv_proj reads RAW_DIM per token.
    {
        let probe = generate_maze(size, &mut MazeRng::new(1));
        let obs = encode(&probe, probe.start);
        assert_eq!(obs.len(), od, "spatial encode len {} != obs_dim {od}", obs.len());
        assert_eq!(w.regions[0].kv_proj.in_dim, RAW_DIM,
            "INPUT kv_proj.in_dim {} != RAW_DIM {RAW_DIM} (spatial build not applied)",
            w.regions[0].kv_proj.in_dim);
        log(&format!("INPUT kv_proj.in_dim={} (== RAW_DIM {RAW_DIM}, verified)",
            w.regions[0].kv_proj.in_dim));
    }

    // ── Train: grad-accumulation mini-batch, BFS optimal-move labels. ──
    log(&format!("training: size={size} batch={BATCH}x{batches} lr={lr}"));
    let mut opt = RegionalAdamW::new(&w).with_lr(lr).with_clip(5.0);
    let mut rng = MazeRng::new(SEED);
    let t0 = std::time::Instant::now();
    let mut best_solve = 0.0f32;
    let mut best_move = 0.0f32;

    for b in 0..batches {
        let mut grads = RegionalGradients::zeros(&w);
        let mut filled = 0;
        let mut batch_loss = 0.0f32;
        while filled < BATCH {
            let maze = generate_maze(size, &mut rng);
            let dist = dist_to_goal(&maze.grid, size, maze.end);
            let Some(cell) = random_open_cell(&maze, &dist, &mut rng) else { continue };
            let Some(target) = optimal_move(&dist, &maze.grid, size, cell) else { continue };
            let obs = encode(&maze, cell);
            let (loss, _pred) = regional_train_step(&w, &mut grads, &obs, target);
            batch_loss += loss;
            filled += 1;
        }
        opt.lr = lr * (1.0 - 0.9 * b as f32 / batches as f32);
        opt.step(&mut w, &mut grads);

        if b % 25 == 0 || b == batches - 1 {
            let (mv, solve) = quick_eval(&w, size, 40, 9999);
            if solve > best_solve { best_solve = solve; }
            if mv > best_move { best_move = mv; }
            log(&format!(
                "  batch {b:4}/{batches}: loss={:.3}  move_acc={:.1}%  solve_rate={:.1}%  (best mv {:.1}% / solve {:.1}%, {:.0}s)",
                batch_loss / BATCH as f32, mv * 100.0, solve * 100.0,
                best_move * 100.0, best_solve * 100.0, t0.elapsed().as_secs_f32()
            ));
        }
    }

    let (move_acc, solve_rate) = quick_eval(&w, size, 300, 12345);
    log(&format!(
        "FINAL size={size}: move_acc={:.1}%  solve_rate={:.1}%  (best mv {:.1}% / solve {:.1}%, {:.0}s total)",
        move_acc * 100.0, solve_rate * 100.0,
        best_move * 100.0, best_solve * 100.0, t0.elapsed().as_secs_f32()
    ));
}
