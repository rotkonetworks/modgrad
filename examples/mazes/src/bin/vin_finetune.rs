//! Fine-tune the folded planner (Milestone 2 of "VIN into the brain").
//!
//! The standalone VIN is trained by single-step behaviour cloning: from a
//! random reachable cell, match the BFS-optimal move. That is exactly the
//! objective our root-cause notes blame for closed-loop FAILURE — the planner
//! imitates one step well but compounds errors and gets stuck in loops on
//! bigger mazes (hence the bio-escape hacks in /play). This trainer attacks
//! that directly, WITHOUT any oracle at inference:
//!
//!   1. WARM-START from the already-trained VIN (we fine-tune, not restart).
//!   2. DAgger — roll the CURRENT planner out closed-loop and train on the
//!      states IT actually visits (including the wall-hits and loop cells),
//!      labelled by the BFS expert. This is the standard fix for compounding
//!      behaviour-cloning error: learn on your own visited distribution.
//!   3. MULTI-SIZE CURRICULUM (9/11/13) with DIAMETER-SCALED value-iteration
//!      rounds, so the planner learns to propagate value far enough on larger
//!      grids — the zero-shot generalization we want.
//!
//! Eval is the honest closed-loop solve rate on held-out seeds at 9/11/13,
//! reported before and after, so the improvement (or lack of it) is visible.
//!
//! Usage:
//!   cargo run -p mazes --bin vin_finetune --release -- \
//!     --in  <vin_solver_weights.json> \
//!     --out <vin_planner_v2_weights.json> \
//!     [--batches 2000] [--lr 3e-3] [--batch 32]

use std::collections::VecDeque;

use modgrad_ctm::vin::{VinGradients, VinReadout, DIR_OFFSETS};

#[path = "../maze_gen.rs"]
mod maze_gen;
use maze_gen::{generate_maze, Maze, MazeRng};

// ── BFS distance-to-goal over OPEN cells (true = wall). u32::MAX = unreachable.
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

/// BFS-optimal move (U,D,L,R index) from `cell`: the open neighbour with the
/// smallest dist-to-goal. None at the goal / trapped cells.
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
        let (nr, nc) = (nr as usize, nc as usize);
        let idx = nr * s + nc;
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

/// Honest per-cell tokens from the maze image only: [is_open, is_goal, bias].
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

fn softmax4(l: &[f32]) -> [f32; 4] {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e = [0.0f32; 4];
    let mut s = 0.0;
    for i in 0..4 {
        e[i] = (l[i] - m).exp();
        s += e[i];
    }
    let inv = 1.0 / s.max(1e-20);
    for v in &mut e {
        *v *= inv;
    }
    e
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

fn reachable_cells(maze: &Maze, dist: &[u32]) -> Vec<(usize, usize)> {
    let s = maze.grid_size;
    let mut out = Vec::new();
    for r in 0..s {
        for c in 0..s {
            let d = dist[r * s + c];
            if d != u32::MAX && d != 0 {
                out.push((r, c));
            }
        }
    }
    out
}

/// Value-iteration rounds for a maze of side `s`: scale with the grid so value
/// reaches the agent from the goal on larger boards (~2·diameter sweeps).
fn iters_for_size(s: usize) -> usize {
    (2 * s + 2).clamp(16, 48)
}

/// Closed-loop solve with the current planner. Returns (solved, wall_hits).
fn closed_loop_solve(vin: &VinReadout, maze: &Maze, raw_dim: usize) -> (bool, u32) {
    let s = maze.grid_size;
    let toks = encode_tokens(maze, raw_dim);
    let mut pos = maze.start;
    let budget = (4 * maze.path_length.max(1)).max(40) as u32;
    let (mut wall_hits, mut steps) = (0u32, 0u32);
    let mut recent: VecDeque<(usize, usize)> = VecDeque::with_capacity(8);
    while steps < budget {
        if pos == maze.end {
            return (true, wall_hits);
        }
        let out = vin.forward(&toks, s, s, Some(pos));
        let (dr, dc) = DIR_OFFSETS[argmax4(&out.move_logits)];
        let nr = pos.0 as i32 + dr;
        let nc = pos.1 as i32 + dc;
        steps += 1;
        if nr < 0 || nc < 0 || nr >= s as i32 || nc >= s as i32 {
            wall_hits += 1;
            continue;
        }
        let (nr, nc) = (nr as usize, nc as usize);
        if maze.grid[nr * s + nc] {
            wall_hits += 1;
            continue;
        }
        pos = (nr, nc);
        recent.push_back(pos);
        if recent.len() > 6 {
            recent.pop_front();
        }
        if recent.iter().filter(|&&p| p == pos).count() >= 3 {
            return (false, wall_hits);
        }
    }
    (pos == maze.end, wall_hits)
}

/// DAgger: roll the planner out closed-loop and return the cells it VISITS
/// (the planner's own state distribution — including wall-bumps and loop
/// cells), so we can train on them with expert labels. Capped at the budget.
fn rollout_visited(vin: &VinReadout, maze: &Maze, raw_dim: usize) -> Vec<(usize, usize)> {
    let s = maze.grid_size;
    let toks = encode_tokens(maze, raw_dim);
    let mut pos = maze.start;
    let budget = (4 * maze.path_length.max(1)).max(40) as u32;
    let mut visited = vec![pos];
    let mut recent: VecDeque<(usize, usize)> = VecDeque::with_capacity(8);
    let mut steps = 0u32;
    while steps < budget && pos != maze.end {
        let out = vin.forward(&toks, s, s, Some(pos));
        let (dr, dc) = DIR_OFFSETS[argmax4(&out.move_logits)];
        let nr = pos.0 as i32 + dr;
        let nc = pos.1 as i32 + dc;
        steps += 1;
        // record the CURRENT cell regardless of whether the move is legal —
        // a wall-bump means "you should have moved differently HERE".
        if nr < 0 || nc < 0 || nr >= s as i32 || nc >= s as i32 || maze.grid[nr as usize * s + nc as usize] {
            continue;
        }
        pos = (nr as usize, nc as usize);
        visited.push(pos);
        recent.push_back(pos);
        if recent.len() > 6 {
            recent.pop_front();
        }
        if recent.iter().filter(|&&p| p == pos).count() >= 3 {
            break; // stuck — stop accumulating this trajectory
        }
    }
    visited
}

/// Reverse-replay trace (#2, Foster & Wilson credit assignment): roll the
/// planner out and return the (cell, chosen_move) sequence plus whether the
/// rollout FAILED (looped / ran out of budget without reaching the goal).
/// On failure the caller replays BACKWARD from the failure, focusing learning
/// on the cells nearest where it went wrong.
fn rollout_trace(vin: &VinReadout, maze: &Maze, raw_dim: usize) -> (Vec<((usize, usize), usize)>, bool) {
    let s = maze.grid_size;
    let toks = encode_tokens(maze, raw_dim);
    let mut pos = maze.start;
    let budget = (4 * maze.path_length.max(1)).max(40) as u32;
    let mut trace = Vec::new();
    let mut recent: VecDeque<(usize, usize)> = VecDeque::with_capacity(8);
    let mut steps = 0u32;
    while steps < budget {
        if pos == maze.end {
            return (trace, false);
        }
        let out = vin.forward(&toks, s, s, Some(pos));
        let mv = argmax4(&out.move_logits);
        trace.push((pos, mv));
        let (dr, dc) = DIR_OFFSETS[mv];
        let nr = pos.0 as i32 + dr;
        let nc = pos.1 as i32 + dc;
        steps += 1;
        if nr < 0 || nc < 0 || nr >= s as i32 || nc >= s as i32 || maze.grid[nr as usize * s + nc as usize] {
            continue; // wall-bump: stay, but the mistake is recorded at `pos`
        }
        pos = (nr as usize, nc as usize);
        recent.push_back(pos);
        if recent.len() > 6 {
            recent.pop_front();
        }
        if recent.iter().filter(|&&p| p == pos).count() >= 3 {
            return (trace, true); // stuck in a loop = failure
        }
    }
    (trace, pos != maze.end)
}

struct EvalResult {
    solve_rate: f32,
    move_acc: f32,
    avg_wall_hits: f32,
}

/// Held-out eval at `size` with diameter-scaled iterations.
fn evaluate(vin: &VinReadout, size: usize, raw_dim: usize, n_mazes: usize, seed_base: u64) -> EvalResult {
    let mut vin = vin.clone();
    let it = iters_for_size(size);
    vin.config.iters = it;
    vin.config.max_iters = vin.config.max_iters.max(it);
    let (mut solved, mut wall_hits, mut correct, mut sampled) = (0usize, 0u32, 0usize, 0usize);
    for i in 0..n_mazes {
        let mut rng = MazeRng::new(seed_base.wrapping_add(i as u64));
        let maze = generate_maze(size, &mut rng);
        let dist = bfs_dist_to_goal(&maze);
        let toks = encode_tokens(&maze, raw_dim);
        for &cell in &reachable_cells(&maze, &dist) {
            if let Some(label) = optimal_move(&maze, &dist, cell) {
                let out = vin.forward(&toks, size, size, Some(cell));
                if argmax4(&out.move_logits) == label {
                    correct += 1;
                }
                sampled += 1;
            }
        }
        let (s, wh) = closed_loop_solve(&vin, &maze, raw_dim);
        solved += s as usize;
        wall_hits += wh;
    }
    EvalResult {
        solve_rate: solved as f32 / n_mazes as f32,
        move_acc: if sampled > 0 { correct as f32 / sampled as f32 } else { 0.0 },
        avg_wall_hits: wall_hits as f32 / n_mazes as f32,
    }
}

fn eval_block(tag: &str, vin: &VinReadout, raw_dim: usize) {
    println!("  {tag:<8} size |  solve  | move_acc | wall_hits/maze");
    for &size in &[9usize, 11, 13] {
        let ev = evaluate(vin, size, raw_dim, 200, 9_700_000 + size as u64 * 10_000);
        println!(
            "  {:>8} {:>3}x{:<2} | {:5.1}% | {:6.1}% | {:.2}",
            "", size, size, 100.0 * ev.solve_rate, 100.0 * ev.move_acc, ev.avg_wall_hits
        );
    }
}

fn arg_val<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

/// splitmix64 — deterministic per-seed hash for sampling the training depth.
fn splitmix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = arg_val::<String>(&args, "--out")
        .expect("--out <vin_planner_v2_weights.json> required");
    let n_batches = arg_val::<usize>(&args, "--batches").unwrap_or(2000);
    let lr = arg_val::<f32>(&args, "--lr").unwrap_or(3e-3);
    let batch = arg_val::<usize>(&args, "--batch").unwrap_or(32);
    let eval_every = arg_val::<usize>(&args, "--eval-every").unwrap_or(250);
    let random_depth = args.iter().any(|a| a == "--random-depth");
    let mistake_replay = args.iter().any(|a| a == "--mistake-replay");

    // ── warm-start from a trained VIN, OR init fresh (--scratch) for scaling ──
    let mut vin: VinReadout = match arg_val::<String>(&args, "--in") {
        Some(in_path) => {
            let json = std::fs::read_to_string(&in_path)
                .unwrap_or_else(|e| panic!("read {in_path}: {e}"));
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse VIN {in_path}: {e}"))
        }
        None => {
            // from-scratch (Phase-4 scale): bigger value_dim, fresh weights.
            let value_dim = arg_val::<usize>(&args, "--value-dim").unwrap_or(16);
            let cfg = modgrad_ctm::vin::VinConfig {
                value_dim,
                iters: iters_for_size(9),
                max_iters: iters_for_size(13) * 2,
                softmax_temp: 0.5,
                highway_gate: true,
                n_dirs: 4,
            };
            println!("(from scratch — value_dim={value_dim})");
            VinReadout::new(cfg, 3)
        }
    };
    let raw_dim = vin.raw_dim;
    println!(
        "fine-tune (DAgger + multi-size curriculum){}\n\
         config: value_dim={} base_iters={} temp={} raw_dim={} params={}\n\
         batches={n_batches} lr={lr} batch={batch}  sizes=[9,11,13] iters=2·s+2",
        format!("{}{}", if random_depth { " + random-depth" } else { "" }, if mistake_replay { " + mistake-replay" } else { "" }),
        vin.config.value_dim, vin.config.iters, vin.config.softmax_temp, raw_dim, vin.num_params(),
    );

    println!("\n── baseline (before fine-tune) ──");
    eval_block("before", &vin, raw_dim);
    println!();

    // curriculum sizes; cycle so each batch mixes scales.
    const SIZES: [usize; 3] = [9, 11, 13];
    let mut train_seed = 1u64;
    println!("  batch |  size | trainCE | move_acc | (running)");
    for b in 0..n_batches {
        let size = SIZES[b % SIZES.len()];
        let base_it = iters_for_size(size);

        let mut grads = VinGradients::zeros(&vin);
        let (mut loss, mut correct, mut filled) = (0.0f32, 0usize, 0usize);

        while filled < batch {
            let mut rng = MazeRng::new(train_seed);
            train_seed = train_seed.wrapping_add(1);
            let maze = generate_maze(size, &mut rng);
            let dist = bfs_dist_to_goal(&maze);
            if reachable_cells(&maze, &dist).is_empty() {
                continue;
            }
            let toks = encode_tokens(&maze, raw_dim);

            // ANYTIME / random-depth training (#22): sample the sweep count
            // per maze so the move head learns to be correct (and confident
            // ONLY when correct) at ANY depth. Then at inference, low
            // move-entropy genuinely means "value has arrived" → entropy-gated
            // ripples can stop early on easy cells and sweep more at junctions.
            let depth = if random_depth {
                let h = splitmix(train_seed);
                6 + (h as usize % (base_it.saturating_sub(6) + 1))
            } else {
                base_it
            };
            vin.config.iters = depth;
            vin.config.max_iters = vin.config.max_iters.max(depth);

            // DAgger: train on the planner's OWN visited cells (its mistakes
            // and loops), labelled by the BFS expert. Falls back to start.
            // With --mistake-replay (#2): focus on the cells where the planner
            // ERRED, and on a failed rollout order them BACKWARD from the
            // failure (reverse-replay credit assignment).
            let (candidates, front_loaded) = if mistake_replay {
                let (trace, failed) = rollout_trace(&vin, &maze, raw_dim);
                let mut mistakes: Vec<(usize, usize)> = trace
                    .iter()
                    .filter(|(cell, mv)| optimal_move(&maze, &dist, *cell).is_some_and(|e| e != *mv))
                    .map(|(cell, _)| *cell)
                    .collect();
                if failed {
                    mistakes.reverse(); // credit from the failure point backward
                }
                if mistakes.is_empty() {
                    (trace.iter().map(|(c, _)| *c).collect::<Vec<_>>(), false)
                } else {
                    (mistakes, true)
                }
            } else {
                (rollout_visited(&vin, &maze, raw_dim), false)
            };
            if candidates.is_empty() {
                continue;
            }
            let per_maze = 4.min(batch - filled);
            for k in 0..per_maze {
                // front_loaded → take the highest-credit cells in order;
                // otherwise spread deterministically across the trajectory.
                let cell = if front_loaded {
                    candidates[k % candidates.len()]
                } else {
                    candidates[(train_seed as usize + k * 7) % candidates.len()]
                };
                let label = match optimal_move(&maze, &dist, cell) {
                    Some(l) => l,
                    None => continue,
                };
                let (out, cache) = vin.forward_train(&toks, size, size, cell);
                let probs = softmax4(&out.move_logits);
                loss += -(probs[label].max(1e-20)).ln();
                if argmax4(&out.move_logits) == label {
                    correct += 1;
                }
                let mut d_logits = vec![0.0f32; 4];
                for i in 0..4 {
                    d_logits[i] = probs[i] - if i == label { 1.0 } else { 0.0 };
                }
                grads.add(&vin.backward(&cache, &toks, size, size, &d_logits));
                filled += 1;
                if filled >= batch {
                    break;
                }
            }
        }

        // gentle warmdown over the back half (consolidate, don't spike out).
        let frac = b as f32 / n_batches.max(1) as f32;
        let lr_scale = if frac < 0.5 {
            1.0
        } else {
            let t = (frac - 0.5) / 0.5;
            0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
        };
        vin.apply_grads(&grads, lr_scale * lr / batch as f32);

        if b % eval_every == 0 || b == n_batches - 1 {
            println!(
                "  {:5} | {:>3}x{:<2}| {:7.4} | {:6.1}% |",
                b, size, size, loss / batch as f32, 100.0 * correct as f32 / batch as f32,
            );
        }
    }

    // restore a canonical base iter count in the saved config (eval/inference
    // can still override); use the 9x9 setting as the stored default.
    vin.config.iters = iters_for_size(9);
    vin.config.max_iters = vin.config.max_iters.max(vin.config.iters);

    println!("\n── after fine-tune ──");
    eval_block("after", &vin, raw_dim);

    // ── export ────────────────────────────────────────────────────────────
    let out_json = serde_json::to_string(&vin).expect("serialize VIN");
    std::fs::write(&out_path, &out_json).unwrap_or_else(|e| panic!("write {out_path}: {e}"));
    // self-verify it reloads
    let _check: VinReadout = serde_json::from_str(&out_json).expect("re-deserialize exported VIN");
    println!("\nexported fine-tuned planner → {out_path} ({} bytes)", out_json.len());
}
