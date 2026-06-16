//! Maze-navigation exporter for the modgrad.com `/play` demo.
//!
//! Trains a single CTM to predict the next move toward the goal from direct
//! cell tokens (no retina), so the whole inference path is a plain
//! `ctm_forward` that the wasm reimplementation can mirror. Reports per-move
//! accuracy AND full-maze solve rate (greedy rollout from start to goal).
//!
//! Usage: cargo run -p mazes --bin export --release -- <out_dir>

#![allow(dead_code)]

#[path = "../maze_gen.rs"]
mod maze_gen;
use maze_gen::{Maze, MazeRng, generate_maze, DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT};

use modgrad_ctm::config::{CtmConfig, ExitStrategy};
use modgrad_ctm::weights::{CtmWeights, CtmState};
use modgrad_ctm::train::{train_step, CtmGradients};
use modgrad_ctm::forward::{ctm_forward, CtmInput};
use std::collections::VecDeque;

const SIZE: usize = 11;     // odd grid (11x11 = 121 cells/tokens)
const TICKS: usize = 12;
const D_MODEL: usize = 128;
const D_INPUT: usize = 64;
const RAW_DIM: usize = 5;   // [is_wall, is_agent, is_goal, x_norm, y_norm]
const OUT_DIMS: usize = 4;  // UP, DOWN, LEFT, RIGHT
const STEPS: usize = 40000;
const LR: f32 = 1.0e-3;
const SEED: u64 = 42;

fn make_config() -> CtmConfig {
    CtmConfig {
        iterations: TICKS,
        d_model: D_MODEL,
        d_input: D_INPUT,
        heads: 4,
        n_synch_out: D_MODEL,
        n_synch_action: D_MODEL,
        synapse_depth: 2,
        memory_length: 8,
        deep_nlms: true,
        memory_hidden_dims: 8,
        out_dims: OUT_DIMS,
        n_random_pairing_self: 0,
        min_width: 8,
        exit_strategy: ExitStrategy::None,
        collect_trajectories: true,
    }
}

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

/// Encode (maze, agent) into [n_tokens * RAW_DIM] direct cell tokens.
fn encode(maze: &Maze, agent: (usize, usize)) -> Vec<f32> {
    let s = maze.grid_size;
    let mut t = vec![0.0f32; s * s * RAW_DIM];
    let denom = (s - 1) as f32;
    for r in 0..s {
        for c in 0..s {
            let i = (r * s + c) * RAW_DIM;
            t[i] = if maze.grid[r * s + c] { 1.0 } else { 0.0 };
            t[i + 1] = if (r, c) == agent { 1.0 } else { 0.0 };
            t[i + 2] = if (r, c) == maze.end { 1.0 } else { 0.0 };
            t[i + 3] = c as f32 / denom;
            t[i + 4] = r as f32 / denom;
        }
    }
    t
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
}
fn commit_tick(c: &[[f32; 2]]) -> usize {
    (0..c.len()).max_by(|&a, &b| c[a][1].partial_cmp(&c[b][1]).unwrap()).unwrap_or(0)
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
fn solves(w: &CtmWeights, maze: &Maze) -> bool {
    let s = maze.grid_size;
    let n = s * s;
    let mut agent = maze.start;
    let budget = maze.path_length * 4 + 8;
    for _ in 0..budget {
        if agent == maze.end { return true; }
        let obs = encode(maze, agent);
        let mut st = CtmState::new(w);
        let out = ctm_forward(w, &mut st, CtmInput::Raw { obs: &obs, n_tokens: n, raw_dim: RAW_DIM });
        let dir = argmax(&out.predictions[commit_tick(&out.certainties)]);
        let nx = step_cell(agent, dir);
        if nx.0 >= s || nx.1 >= s || maze.grid[nx.0 * s + nx.1] { return false; }
        agent = nx;
    }
    agent == maze.end
}

fn main() {
    let _out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let n_tokens = SIZE * SIZE;

    eprintln!("Maze CTM: size={SIZE} ticks={TICKS} d_model={D_MODEL} tokens={n_tokens} steps={STEPS}");
    let mut w = CtmWeights::new(make_config(), RAW_DIM);
    let mut grads = CtmGradients::zeros(&w);
    let mut rng = MazeRng::new(SEED);

    let mut losses: Vec<f32> = Vec::new();
    let mut corr: Vec<f32> = Vec::new();
    for step in 0..STEPS {
        let maze = generate_maze(SIZE, &mut rng);
        let dist = dist_to_goal(&maze.grid, SIZE, maze.end);
        let Some(cell) = random_open_cell(&maze, &dist, &mut rng) else { continue };
        let Some(target) = optimal_move(&dist, &maze.grid, SIZE, cell) else { continue };
        let obs = encode(&maze, cell);
        grads.zero();
        let lr = LR * (1.0 - 0.9 * step as f32 / STEPS as f32);
        let r = train_step(&w, &mut grads, &obs, n_tokens, RAW_DIM, target);
        grads.apply(&mut w, lr, 5.0);
        losses.push(r.loss);
        corr.push(if r.prediction == target { 1.0 } else { 0.0 });
        if step % 2000 == 0 || step == STEPS - 1 {
            let win = 500.min(losses.len());
            let l: f32 = losses[losses.len() - win..].iter().sum::<f32>() / win as f32;
            let a: f32 = corr[corr.len() - win..].iter().sum::<f32>() / win as f32;
            eprintln!("  step {step:6}: loss={l:.3} move_acc={:.1}%", a * 100.0);
        }
    }

    // Eval: per-move accuracy + full-maze solve rate on fresh mazes.
    let mut erng = MazeRng::new(SEED.wrapping_add(12345));
    let (mut mv_c, mut mv_t) = (0usize, 0usize);
    let mut solved = 0usize;
    let n_mazes = 200usize;
    for _ in 0..n_mazes {
        let maze = generate_maze(SIZE, &mut erng);
        let dist = dist_to_goal(&maze.grid, SIZE, maze.end);
        for _ in 0..5 {
            if let Some(cell) = random_open_cell(&maze, &dist, &mut erng) {
                if let Some(tgt) = optimal_move(&dist, &maze.grid, SIZE, cell) {
                    let obs = encode(&maze, cell);
                    let mut st = CtmState::new(&w);
                    let out = ctm_forward(&w, &mut st, CtmInput::Raw { obs: &obs, n_tokens, raw_dim: RAW_DIM });
                    if argmax(&out.predictions[commit_tick(&out.certainties)]) == tgt { mv_c += 1; }
                    mv_t += 1;
                }
            }
        }
        if solves(&w, &maze) { solved += 1; }
    }
    eprintln!("eval: move_acc={:.1}%  solve_rate={:.1}%  (n={n_mazes})",
        mv_c as f32 / mv_t.max(1) as f32 * 100.0,
        solved as f32 / n_mazes as f32 * 100.0);
}
