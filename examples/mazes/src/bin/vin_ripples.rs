//! Measure entropy-gated value iteration ("ripples ↑ under uncertainty")
//! against fixed-depth iteration on the SAME trained planner. No retraining —
//! this isolates the effect of WHEN the planner stops sweeping.
//!
//!   fixed:    every step runs `iters_for_size(s)` Bellman sweeps.
//!   adaptive: each step sweeps until the move-entropy ≤ floor (min..max),
//!             so hard junctions get more "ripples", straightaways get fewer.
//!
//! Reports closed-loop solve rate, wall-hits, and the AVERAGE sweeps/step
//! (the compute the adaptive policy actually spent) at 9/11/13.
//!
//! Usage: cargo run -p mazes --bin vin_ripples --release -- --in <vin.json>

use std::collections::VecDeque;

use modgrad_ctm::vin::{VinReadout, DIR_OFFSETS};

#[path = "../maze_gen.rs"]
mod maze_gen;
use maze_gen::{generate_maze, Maze, MazeRng};

fn bfs_dist_to_goal(maze: &Maze) -> Vec<u32> {
    let s = maze.grid_size;
    let mut dist = vec![u32::MAX; s * s];
    let mut q = VecDeque::new();
    let (er, ec) = maze.end;
    dist[er * s + ec] = 0;
    q.push_back((er, ec));
    while let Some((r, c)) = q.pop_front() {
        let d = dist[r * s + c];
        for (dr, dc) in DIR_OFFSETS {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nc < 0 || nr >= s as i32 || nc >= s as i32 {
                continue;
            }
            let (nr, nc) = (nr as usize, nc as usize);
            let idx = nr * s + nc;
            if maze.grid[idx] {
                continue;
            }
            if dist[idx] == u32::MAX {
                dist[idx] = d + 1;
                q.push_back((nr, nc));
            }
        }
    }
    dist
}

fn optimal_move(maze: &Maze, dist: &[u32], cell: (usize, usize)) -> Option<usize> {
    let s = maze.grid_size;
    let (r, c) = cell;
    let here = dist[r * s + c];
    if here == 0 || here == u32::MAX {
        return None;
    }
    let mut best_dir = None;
    let mut best_d = u32::MAX;
    for (di, (dr, dc)) in DIR_OFFSETS.iter().enumerate() {
        let nr = r as i32 + dr;
        let nc = c as i32 + dc;
        if nr < 0 || nc < 0 || nr >= s as i32 || nc >= s as i32 {
            continue;
        }
        let idx = nr as usize * s + nc as usize;
        if maze.grid[idx] {
            continue;
        }
        if dist[idx] < best_d {
            best_d = dist[idx];
            best_dir = Some(di);
        }
    }
    match best_dir {
        Some(di) if best_d < here => Some(di),
        _ => None,
    }
}

fn encode_tokens(maze: &Maze, raw_dim: usize) -> Vec<f32> {
    let s = maze.grid_size;
    let (er, ec) = maze.end;
    let goal = er * s + ec;
    let mut toks = vec![0.0f32; s * s * raw_dim];
    for cell in 0..s * s {
        let off = cell * raw_dim;
        toks[off] = if maze.grid[cell] { 0.0 } else { 1.0 };
        if raw_dim >= 2 {
            toks[off + 1] = if cell == goal { 1.0 } else { 0.0 };
        }
        if raw_dim >= 3 {
            toks[off + 2] = 1.0;
        }
    }
    toks
}

fn argmax4(l: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for i in 0..4 {
        if l[i] > bv {
            bv = l[i];
            best = i;
        }
    }
    best
}

fn iters_for_size(s: usize) -> usize {
    (2 * s + 2).clamp(16, 48)
}

/// Closed-loop solve. `adaptive` switches between fixed and entropy-gated
/// sweeps. Returns (solved, wall_hits, total_sweeps, steps).
fn solve(
    vin: &VinReadout,
    maze: &Maze,
    raw_dim: usize,
    adaptive: bool,
    min_it: usize,
    max_it: usize,
    floor: f32,
) -> (bool, u32, u64, u32) {
    let s = maze.grid_size;
    let toks = encode_tokens(maze, raw_dim);
    let fixed_it = iters_for_size(s);
    let mut pos = maze.start;
    let budget = (4 * maze.path_length.max(1)).max(40) as u32;
    let (mut wall_hits, mut steps, mut sweeps) = (0u32, 0u32, 0u64);
    let mut recent: VecDeque<(usize, usize)> = VecDeque::with_capacity(8);
    while steps < budget {
        if pos == maze.end {
            return (true, wall_hits, sweeps, steps);
        }
        let (out, used) = if adaptive {
            vin.forward_adaptive(&toks, s, s, Some(pos), min_it, max_it, floor)
        } else {
            let mut v = vin.clone();
            v.config.iters = fixed_it;
            v.config.max_iters = v.config.max_iters.max(fixed_it);
            (v.forward(&toks, s, s, Some(pos)), fixed_it)
        };
        sweeps += used as u64;
        let (dr, dc) = DIR_OFFSETS[argmax4(&out.move_logits)];
        let nr = pos.0 as i32 + dr;
        let nc = pos.1 as i32 + dc;
        steps += 1;
        if nr < 0 || nc < 0 || nr >= s as i32 || nc >= s as i32 || maze.grid[nr as usize * s + nc as usize] {
            wall_hits += 1;
            continue;
        }
        pos = (nr as usize, nc as usize);
        recent.push_back(pos);
        if recent.len() > 6 {
            recent.pop_front();
        }
        if recent.iter().filter(|&&p| p == pos).count() >= 3 {
            return (false, wall_hits, sweeps, steps);
        }
    }
    (pos == maze.end, wall_hits, sweeps, steps)
}

fn eval(vin: &VinReadout, raw_dim: usize, size: usize, n: usize, adaptive: bool) -> (f32, f32, f32) {
    // ADDITIVE ripples: never sweep FEWER than the trained depth (so the move
    // head always sees an in-distribution value field), only ADD sweeps while
    // the move is still uncertain. "ripples ↑ under uncertainty", never ↓.
    let min_it = iters_for_size(size);
    let max_it = iters_for_size(size) * 2;
    let floor = 0.3f32; // keep sweeping only while genuinely uncertain
    let (mut solved, mut wall_hits, mut total_sweeps, mut total_steps) = (0usize, 0u32, 0u64, 0u64);
    for i in 0..n {
        let mut rng = MazeRng::new(9_700_000 + size as u64 * 10_000 + i as u64);
        let maze = generate_maze(size, &mut rng);
        let _ = bfs_dist_to_goal(&maze); // (kept for parity with trainer eval path)
        let (s, wh, sw, st) = solve(vin, &maze, raw_dim, adaptive, min_it, max_it, floor);
        solved += s as usize;
        wall_hits += wh;
        total_sweeps += sw;
        total_steps += st.max(1) as u64;
    }
    (
        solved as f32 / n as f32,
        wall_hits as f32 / n as f32,
        total_sweeps as f32 / total_steps as f32, // avg sweeps/step
    )
}

fn arg_val<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_path = arg_val::<String>(&args, "--in").expect("--in <vin.json> required");
    let n = arg_val::<usize>(&args, "--n").unwrap_or(300);
    let json = std::fs::read_to_string(&in_path).unwrap_or_else(|e| panic!("read {in_path}: {e}"));
    let vin: VinReadout = serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {in_path}: {e}"));
    let raw_dim = vin.raw_dim;
    // ensure dist fn is considered used by the optimizer
    let _ = optimal_move;

    println!("entropy-gated ripples vs fixed iteration — {in_path} ({n} mazes/size)\n");
    println!("  size  | policy   | solve  | wall_hits | sweeps/step");
    for &size in &[9usize, 11, 13] {
        let (sf, wf, cf) = eval(&vin, raw_dim, size, n, false);
        let (sa, wa, ca) = eval(&vin, raw_dim, size, n, true);
        println!(
            "  {:>2}x{:<2} | fixed    | {:5.1}% | {:8.2} | {:6.1}\n         | adaptive | {:5.1}% | {:8.2} | {:6.1}   (floor 0.5, {}..{} sweeps)",
            size, size, 100.0 * sf, wf, cf, 100.0 * sa, wa, ca, 6, (iters_for_size(size) * 2).max(32),
        );
    }
}
