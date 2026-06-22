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
use modgrad_ctm::forward::{ctm_forward, ctm_forward_with_attn_trace, CtmInput};
use serde::Serialize;
use std::collections::VecDeque;

const SIZE: usize = 9;      // odd grid (9x9) — harder than 7x7, solvable enough to stay snappy
const TICKS: usize = 8;     // fewer ticks: lean on model capacity, keep the demo snappy
const D_MODEL: usize = 256; // bigger model: the capacity that broke the floor on 11x11
const D_INPUT: usize = 128;
// per cell: [is_wall, is_agent, is_goal, wall_up, wall_down, wall_left, wall_right, x, y]
const RAW_DIM: usize = 9;
const OUT_DIMS: usize = 4;  // UP, DOWN, LEFT, RIGHT
const BATCH: usize = 16;    // grad-accumulation batch (stabilizes fresh-maze SGD)
const BATCHES: usize = 1800; // plateaus at ~25% solve by ~850; cap here for a clean fast finish + export
const LR: f32 = 2.0e-3;
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
        spatial: None,
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
/// Each cell carries its own wall flag, agent/goal flags, the 4 neighbor-wall
/// bits (local connectivity = which moves are blocked), and normalized (x, y).
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
            t[i + 7] = c as f32 / denom;
            t[i + 8] = r as f32 / denom;
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

/// True held-out metrics using the max-certainty readout: optimal-move
/// accuracy + full-maze greedy solve rate.
fn quick_eval(w: &CtmWeights, n_mazes: usize, seed: u64) -> (f32, f32) {
    let n = SIZE * SIZE;
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
                    let mut st = CtmState::new(w);
                    let out = ctm_forward(w, &mut st, CtmInput::Raw { obs: &obs, n_tokens: n, raw_dim: RAW_DIM });
                    if argmax(&out.predictions[commit_tick(&out.certainties)]) == tgt { mv_c += 1; }
                    mv_t += 1;
                }
            }
        }
        if solves(w, &maze) { solved += 1; }
    }
    (mv_c as f32 / mv_t.max(1) as f32, solved as f32 / n_mazes as f32)
}

// ── Artifact export (weights + curated demo mazes + oracle traces) ──

#[derive(Serialize)]
struct TickTrace {
    prediction: Vec<f32>,      // [out_dims] move logits
    certainty: [f32; 2],
    activations: Vec<f32>,     // [d_model] neuron pool this tick
    attention: Vec<Vec<f32>>,  // [heads][n_tokens] over maze cells
}

#[derive(Serialize)]
struct OracleSample {
    grid: Vec<u8>,
    start: [usize; 2],
    end: [usize; 2],
    agent: [usize; 2],
    optimal: usize,
    predicted: usize,
    commit_tick: usize,
    ticks_used: usize,
    ticks: Vec<TickTrace>,
}

#[derive(Serialize)]
struct DemoMaze {
    grid: Vec<u8>,             // 1=wall 0=open, row-major size*size
    start: [usize; 2],
    end: [usize; 2],
    path_length: usize,
}

#[derive(Serialize)]
struct Reference {
    task: String,
    size: usize,
    ticks: usize,
    d_model: usize,
    raw_dim: usize,
    out_dims: usize,
    heads: usize,
    move_acc: f32,
    solve_rate: f32,
    mazes: Vec<DemoMaze>,
    oracle: Vec<OracleSample>,
}

/// One forward pass at `agent`, capturing the full per-tick observable state
/// (predictions, certainty, neuron activations, attention over cells).
fn trace_at(w: &CtmWeights, maze: &Maze, agent: (usize, usize)) -> OracleSample {
    let s = maze.grid_size;
    let n = s * s;
    let d = w.config.d_model;
    let obs = encode(maze, agent);
    let mut st = CtmState::new(w);
    let (out, attn) = ctm_forward_with_attn_trace(
        w, &mut st, CtmInput::Raw { obs: &obs, n_tokens: n, raw_dim: RAW_DIM });
    let ct = commit_tick(&out.certainties);
    let mut ticks = Vec::with_capacity(out.ticks_used);
    for t in 0..out.ticks_used {
        let activations = if out.trajectory.len() >= (t + 1) * d {
            out.trajectory[t * d..(t + 1) * d].to_vec()
        } else { Vec::new() };
        ticks.push(TickTrace {
            prediction: out.predictions[t].clone(),
            certainty: out.certainties[t],
            activations,
            attention: attn.get(t).cloned().unwrap_or_default(),
        });
    }
    let dist = dist_to_goal(&maze.grid, s, maze.end);
    OracleSample {
        grid: maze.grid.iter().map(|&b| b as u8).collect(),
        start: [maze.start.0, maze.start.1],
        end: [maze.end.0, maze.end.1],
        agent: [agent.0, agent.1],
        optimal: optimal_move(&dist, &maze.grid, s, agent).unwrap_or(99),
        predicted: argmax(&out.predictions[ct]),
        commit_tick: ct,
        ticks_used: out.ticks_used,
        ticks,
    }
}

fn export_artifacts(w: &CtmWeights, out_dir: &str, move_acc: f32, solve_rate: f32) {
    // Curate demo mazes the brain actually solves, so the demo always works.
    let mut crng = MazeRng::new(777);
    let mut mazes = Vec::new();
    let mut oracle = Vec::new();
    let mut tries = 0;
    while mazes.len() < 16 && tries < 6000 {
        tries += 1;
        let maze = generate_maze(SIZE, &mut crng);
        if !solves(w, &maze) { continue; }
        if oracle.len() < 6 { oracle.push(trace_at(w, &maze, maze.start)); }
        mazes.push(DemoMaze {
            grid: maze.grid.iter().map(|&b| b as u8).collect(),
            start: [maze.start.0, maze.start.1],
            end: [maze.end.0, maze.end.1],
            path_length: maze.path_length,
        });
    }
    let reference = Reference {
        task: "maze".to_string(),
        size: SIZE, ticks: TICKS, d_model: D_MODEL, raw_dim: RAW_DIM,
        out_dims: OUT_DIMS, heads: 4, move_acc, solve_rate, mazes, oracle,
    };
    std::fs::write(format!("{out_dir}/maze_weights.json"),
        serde_json::to_string(w).unwrap()).expect("write weights");
    std::fs::write(format!("{out_dir}/maze_reference.json"),
        serde_json::to_string(&reference).unwrap()).expect("write reference");
    eprintln!("wrote maze_weights.json + maze_reference.json ({} demo mazes, {} oracle traces)",
        reference.mazes.len(), reference.oracle.len());
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let n_tokens = SIZE * SIZE;

    eprintln!("Maze CTM: size={SIZE} ticks={TICKS} d_model={D_MODEL} tokens={n_tokens} batch={BATCH}x{BATCHES}");
    let mut w = CtmWeights::new(make_config(), RAW_DIM);
    let mut grads = CtmGradients::zeros(&w);
    let mut rng = MazeRng::new(SEED);

    for b in 0..BATCHES {
        grads.zero();
        let mut filled = 0;
        while filled < BATCH {
            let maze = generate_maze(SIZE, &mut rng);
            let dist = dist_to_goal(&maze.grid, SIZE, maze.end);
            let Some(cell) = random_open_cell(&maze, &dist, &mut rng) else { continue };
            let Some(target) = optimal_move(&dist, &maze.grid, SIZE, cell) else { continue };
            let obs = encode(&maze, cell);
            train_step(&w, &mut grads, &obs, n_tokens, RAW_DIM, target); // accumulates
            filled += 1;
        }
        let lr = LR * (1.0 - 0.9 * b as f32 / BATCHES as f32);
        grads.apply(&mut w, lr, 5.0); // norm-clip the accumulated batch gradient
        if b % 50 == 0 || b == BATCHES - 1 {
            let (mv, solve) = quick_eval(&w, 40, 9999);
            eprintln!("  batch {b:4}/{BATCHES}: move_acc={:.1}%  solve_rate={:.1}%",
                mv * 100.0, solve * 100.0);
        }
    }

    let (mv, solve) = quick_eval(&w, 300, 12345);
    eprintln!("FINAL: move_acc={:.1}%  solve_rate={:.1}%", mv * 100.0, solve * 100.0);
    export_artifacts(&w, &out_dir, mv, solve);
}
