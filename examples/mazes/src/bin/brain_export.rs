//! De-risk probe: can the 8-region brain (`RegionalConfig::eight_region_small`)
//! learn to navigate mazes from a FLAT direct encoding (no VisualCortex retina)?
//!
//! Framing: NEXT-MOVE prediction. The brain sees a flat observation vector of
//! per-cell features (wall / agent / goal + the 4 neighbor-wall bits + x,y) and
//! must output the BFS-optimal move (UP/DOWN/LEFT/RIGHT = out_dims 4) from the
//! agent's current cell. Trained with grad-accumulation mini-batching, same
//! pattern as the single-CTM `export.rs`. Metrics are the REAL ones:
//! max-certainty move accuracy + greedy solve rate on held-out mazes.
//!
//! Usage: cargo run -p mazes --bin brain_export --release

#![allow(dead_code)]

#[path = "../maze_gen.rs"]
mod maze_gen;
use maze_gen::{Maze, MazeRng, generate_maze, DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT};

use modgrad_ctm::graph::{
    RegionalConfig, RegionalWeights, RegionalState, RegionalGradients,
    RegionalAdamW, regional_forward, regional_train_step,
};
use std::collections::VecDeque;

const SIZE: usize = 7;       // odd grid (7x7 = 49 cells)
const RAW_DIM: usize = 7;    // per-cell features (see encode)
const OUT_DIMS: usize = 4;   // UP, DOWN, LEFT, RIGHT
const BATCH: usize = 16;     // grad-accumulation batch
const SEED: u64 = 42;

// Env-overridable so we can A/B test (ticks / region size / steps / lr)
// without editing constants. Defaults reproduce the first probe run.
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn obs_dim() -> usize { SIZE * SIZE * RAW_DIM }

/// BFS distance-to-goal over open cells (u32::MAX = unreachable/wall).
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

/// Optimal next move: the open neighbor with smallest dist-to-goal.
fn optimal_move(dist: &[u32], grid: &[bool], size: usize, cell: (usize, usize)) -> Option<usize> {
    let (r, c) = cell;
    if dist[r * size + c] == 0 { return None; } // at goal
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

/// FLAT encoding: [size*size*RAW_DIM]. Per cell:
/// [is_wall, is_agent, is_goal, wall_up, wall_down, wall_left, wall_right].
/// The 4 neighbor-wall bits (local connectivity) were what made the single
/// CTM learn — we keep them. OOB neighbors count as walls.
fn encode(maze: &Maze, agent: (usize, usize)) -> Vec<f32> {
    let s = maze.grid_size;
    let mut t = vec![0.0f32; s * s * RAW_DIM];
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
        }
    }
    t
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
}

/// Committed prediction: the last outer tick that actually ran. The regional
/// brain's RegionalOutput exposes `ticks_used` (early-exit aware) but not a
/// per-tick certainty vector, so we read the final committed tick.
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

/// Greedy rollout from start; true if the brain reaches the goal.
fn solves(w: &RegionalWeights, maze: &Maze) -> bool {
    let s = maze.grid_size;
    let mut agent = maze.start;
    let budget = maze.path_length * 4 + 8;
    for _ in 0..budget {
        if agent == maze.end { return true; }
        let obs = encode(maze, agent);
        let mut st = RegionalState::new(w);
        let out = regional_forward(w, &mut st, &obs);
        let dir = argmax(committed_pred(&out));
        let nx = step_cell(agent, dir);
        if nx.0 >= s || nx.1 >= s || maze.grid[nx.0 * s + nx.1] { return false; }
        agent = nx;
    }
    agent == maze.end
}

/// Held-out metrics: optimal-move accuracy (max-certainty readout) +
/// full-maze greedy solve rate.
fn quick_eval(w: &RegionalWeights, n_mazes: usize, seed: u64) -> (f32, f32) {
    let mut erng = MazeRng::new(seed);
    let (mut mv_c, mut mv_t) = (0usize, 0usize);
    let mut solved = 0usize;
    for _ in 0..n_mazes {
        let maze = generate_maze(SIZE, &mut erng);
        let dist = dist_to_goal(&maze.grid, SIZE, maze.end);
        for _ in 0..4 {
            if let Some(cell) = random_open_cell(&maze, &dist, &mut erng) {
                if let Some(tgt) = optimal_move(&dist, &maze.grid, SIZE, cell) {
                    let obs = encode(&maze, cell);
                    let mut st = RegionalState::new(w);
                    let out = regional_forward(w, &mut st, &obs);
                    if argmax(committed_pred(&out)) == tgt { mv_c += 1; }
                    mv_t += 1;
                }
            }
        }
        if solves(w, &maze) { solved += 1; }
    }
    (mv_c as f32 / mv_t.max(1) as f32, solved as f32 / n_mazes as f32)
}

fn main() {
    let od = obs_dim();
    let ticks = env_usize("TICKS", 12);
    let batches = env_usize("BATCHES", 1200);
    let lr = env_f32("LR", 1.5e-3);
    let medium = env_usize("MEDIUM", 0) != 0;
    eprintln!(
        "Brain maze (FLAT encoding): size={SIZE} ticks={ticks} raw_dim={RAW_DIM} \
         obs_dim={od} out_dims={OUT_DIMS} batch={BATCH}x{batches} lr={lr} medium={medium}"
    );

    let cfg = if medium {
        RegionalConfig::eight_region_medium(od, OUT_DIMS, ticks)
    } else {
        RegionalConfig::eight_region_small(od, OUT_DIMS, ticks)
    };
    let total_params: usize = cfg.regions.iter().map(|r| r.d_model).sum();
    eprintln!(
        "config: {} regions={} total_neurons={} n_global_sync={} router={} outer_ticks={}",
        if medium { "eight_region_medium" } else { "eight_region_small" },
        cfg.regions.len(), total_params, cfg.n_global_sync,
        cfg.router.is_some(), cfg.outer_ticks
    );

    let mut w = RegionalWeights::new(cfg);
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
            let obs = encode(&maze, cell);
            let (loss, _pred) = regional_train_step(&w, &mut grads, &obs, target); // accumulates
            batch_loss += loss;
            filled += 1;
        }
        // Linear LR decay to 10% — the fixed-LR run stayed noisy late.
        opt.lr = lr * (1.0 - 0.9 * b as f32 / batches as f32);
        opt.step(&mut w, &mut grads);

        if b % 50 == 0 || b == batches - 1 {
            let (mv, solve) = quick_eval(&w, 40, 9999);
            let secs = t0.elapsed().as_secs_f32();
            eprintln!(
                "  batch {b:4}/{batches}: loss={:.3}  move_acc={:.1}%  solve_rate={:.1}%  ({:.0}s)",
                batch_loss / BATCH as f32, mv * 100.0, solve * 100.0, secs
            );
        }
    }

    let (mv, solve) = quick_eval(&w, 300, 12345);
    eprintln!(
        "FINAL: move_acc={:.1}%  solve_rate={:.1}%  ({:.0}s total)",
        mv * 100.0, solve * 100.0, t0.elapsed().as_secs_f32()
    );
}
