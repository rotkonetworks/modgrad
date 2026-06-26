//! Train the learned Value-Iteration-Network (VIN) readout to SOLVE mazes
//! from the maze image ALONE — no solver, no solved-distance, no route fed
//! as input. The honest empirical test of whether the planner works.
//!
//! TOKEN ENCODING (honest perception): per cell, raw_dim features built ONLY
//! from what is visible in the maze image:
//!   raw_dim=3 → [is_open, is_goal, bias=1.0]
//! The agent's current cell is passed EXPLICITLY to the VIN via
//! `forward_train(.., agent_cell)`. Tokens contain NO agent position, NO
//! distance-to-goal, NO route, NO value — only walls + goal. (Knowing the
//! walls and goal = legitimate perception; feeding solved distances or the
//! optimal path = cheating, which we do NOT do.)
//!
//! LABELS (supervised, à la Tamar/CTM): BFS distance-to-goal over open cells;
//! the optimal move from a cell = the in-grid open neighbour (U/D/L/R) with
//! the smallest distance-to-goal (ties: first in U,D,L,R order). These labels
//! are used ONLY to form the cross-entropy target — they are NOT fed into the
//! tokens.
//!
//! EVAL — closed-loop SOLVE RATE on held-out fresh mazes: start at maze.start,
//! repeatedly run the VIN at the agent's CURRENT cell, argmax the move logits,
//! step; wall/edge = wall-hit (stay); solved when end reached; fail on budget
//! or oscillation. solve_rate = solved / total.

use std::collections::VecDeque;

use modgrad_ctm::vin::{VinConfig, VinGradients, VinReadout, DIR_OFFSETS};

#[path = "../maze_gen.rs"]
mod maze_gen;
use maze_gen::{generate_maze, Maze, MazeRng};

// ── BFS distance-to-goal over OPEN cells (true=wall). u32::MAX = unreachable.
fn bfs_dist_to_goal(maze: &Maze) -> Vec<u32> {
    let s = maze.grid_size;
    let n = s * s;
    let mut dist = vec![u32::MAX; n];
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
                continue; // wall
            }
            if dist[idx] == u32::MAX {
                dist[idx] = d + 1;
                q.push_back((nr, nc));
            }
        }
    }
    dist
}

/// Optimal move (index into DIR_OFFSETS = U,D,L,R) from `cell` toward goal:
/// the in-grid open neighbour with the smallest dist-to-goal. None if no
/// strictly-improving open neighbour (e.g. cell == goal or trapped).
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
        let d = dist[idx];
        if d < best_d {
            best_d = d;
            best_dir = Some(di);
        }
    }
    // Only return a move that strictly improves (best_d < here). With BFS this
    // is guaranteed for any reachable non-goal cell that has an open neighbour.
    match best_dir {
        Some(di) if best_d < here => Some(di),
        _ => None,
    }
}

/// HONEST token encoding: per cell raw_dim features from the maze image only.
/// raw_dim=2 → [is_open, is_goal]; raw_dim=3 → +[bias=1.0].
fn encode_tokens(maze: &Maze, raw_dim: usize) -> Vec<f32> {
    let s = maze.grid_size;
    let n = s * s;
    let (er, ec) = maze.end;
    let goal = er * s + ec;
    let mut toks = vec![0.0f32; n * raw_dim];
    for cell in 0..n {
        let off = cell * raw_dim;
        let is_open = if maze.grid[cell] { 0.0 } else { 1.0 };
        toks[off] = is_open;
        if raw_dim >= 2 {
            toks[off + 1] = if cell == goal { 1.0 } else { 0.0 };
        }
        if raw_dim >= 3 {
            toks[off + 2] = 1.0; // constant bias channel
        }
    }
    toks
}

fn softmax4(logits: &[f32]) -> [f32; 4] {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e = [0.0f32; 4];
    let mut s = 0.0;
    for i in 0..4 {
        e[i] = (logits[i] - m).exp();
        s += e[i];
    }
    let inv = 1.0 / s.max(1e-20);
    for i in 0..4 {
        e[i] *= inv;
    }
    e
}

fn argmax4(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for i in 0..4 {
        if logits[i] > bv {
            bv = logits[i];
            best = i;
        }
    }
    best
}

/// All reachable open non-goal cells (cells from which an optimal move exists).
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

/// Closed-loop solve: start at maze.start, argmax moves, step. Returns
/// (solved, wall_hits, steps).
fn closed_loop_solve(vin: &VinReadout, maze: &Maze, raw_dim: usize) -> (bool, u32, u32) {
    let s = maze.grid_size;
    let toks = encode_tokens(maze, raw_dim);
    let mut pos = maze.start;
    let budget = (4 * maze.path_length.max(1)).max(40) as u32;
    let mut wall_hits = 0u32;
    let mut steps = 0u32;
    // Short-oscillation detection: remember the last few positions.
    let mut recent: VecDeque<(usize, usize)> = VecDeque::with_capacity(8);
    while steps < budget {
        if pos == maze.end {
            return (true, wall_hits, steps);
        }
        let out = vin.forward(&toks, s, s, Some(pos));
        let dir = argmax4(&out.move_logits);
        let (dr, dc) = DIR_OFFSETS[dir];
        let nr = pos.0 as i32 + dr;
        let nc = pos.1 as i32 + dc;
        steps += 1;
        if nr < 0 || nc < 0 || nr >= s as i32 || nc >= s as i32 {
            wall_hits += 1;
            continue; // edge: stay
        }
        let (nr, nc) = (nr as usize, nc as usize);
        if maze.grid[nr * s + nc] {
            wall_hits += 1;
            continue; // wall: stay
        }
        pos = (nr, nc);
        // Oscillation guard: if we've been at this exact cell 3+ times in the
        // recent window, we're stuck in a loop → fail early.
        recent.push_back(pos);
        if recent.len() > 6 {
            recent.pop_front();
        }
        let count = recent.iter().filter(|&&p| p == pos).count();
        if count >= 3 {
            return (false, wall_hits, steps);
        }
    }
    (pos == maze.end, wall_hits, steps)
}

/// Held-out eval: solve_rate + per-move accuracy (argmax vs BFS-optimal).
struct EvalResult {
    solve_rate: f32,
    move_acc: f32,
    avg_wall_hits: f32,
}

fn evaluate(
    vin: &VinReadout,
    size: usize,
    raw_dim: usize,
    n_mazes: usize,
    seed_base: u64,
    forward_iters_override: Option<usize>,
) -> EvalResult {
    // Optionally run forward with a different iter count (generalization).
    let mut vin = vin.clone();
    if let Some(it) = forward_iters_override {
        vin.config.iters = it;
        vin.config.max_iters = vin.config.max_iters.max(it);
    }
    let mut solved = 0usize;
    let mut total_wall_hits = 0u32;
    let mut correct = 0usize;
    let mut sampled = 0usize;
    for i in 0..n_mazes {
        let mut rng = MazeRng::new(seed_base.wrapping_add(i as u64));
        let maze = generate_maze(size, &mut rng);
        let dist = bfs_dist_to_goal(&maze);
        let toks = encode_tokens(&maze, raw_dim);
        // per-move accuracy over all reachable cells
        for &cell in &reachable_cells(&maze, &dist) {
            if let Some(label) = optimal_move(&maze, &dist, cell) {
                let out = vin.forward(&toks, size, size, Some(cell));
                if argmax4(&out.move_logits) == label {
                    correct += 1;
                }
                sampled += 1;
            }
        }
        let (s, wh, _) = closed_loop_solve(&vin, &maze, raw_dim);
        if s {
            solved += 1;
        }
        total_wall_hits += wh;
    }
    EvalResult {
        solve_rate: solved as f32 / n_mazes as f32,
        move_acc: if sampled > 0 {
            correct as f32 / sampled as f32
        } else {
            0.0
        },
        avg_wall_hits: total_wall_hits as f32 / n_mazes as f32,
    }
}

#[derive(Clone)]
struct TrainCfg {
    value_dim: usize,
    iters: usize,
    softmax_temp: f32,
    lr: f32,
    batch: usize,
    raw_dim: usize,
    size: usize,
    n_batches: usize,
    eval_every: usize,
    eval_mazes: usize,
}

fn train(cfg: &TrainCfg) -> VinReadout {
    let vcfg = VinConfig {
        value_dim: cfg.value_dim,
        iters: cfg.iters,
        max_iters: cfg.iters.max(20),
        softmax_temp: cfg.softmax_temp,
        highway_gate: true,
        n_dirs: 4,
    };
    let mut vin = VinReadout::new(vcfg, cfg.raw_dim);
    println!(
        "config: value_dim={} iters={} temp={} lr={} batch={} raw_dim={} size={} params={}",
        cfg.value_dim,
        cfg.iters,
        cfg.softmax_temp,
        cfg.lr,
        cfg.batch,
        cfg.raw_dim,
        cfg.size,
        vin.num_params()
    );
    println!("  batch | trainCE | move_acc | solve_rate | wall_hits/maze");

    // Training mazes use seeds disjoint from eval seeds (eval uses 9_000_000+).
    let mut train_seed = 1u64;

    for b in 0..cfg.n_batches {
        let mut grads = VinGradients::zeros(&vin);
        let mut batch_loss = 0.0f32;
        let mut batch_correct = 0usize;
        let mut filled = 0usize;
        while filled < cfg.batch {
            let mut rng = MazeRng::new(train_seed);
            train_seed = train_seed.wrapping_add(1);
            let maze = generate_maze(cfg.size, &mut rng);
            let dist = bfs_dist_to_goal(&maze);
            let cells = reachable_cells(&maze, &dist);
            if cells.is_empty() {
                continue;
            }
            let toks = encode_tokens(&maze, cfg.raw_dim);
            // Sample a handful of cells per maze (broad coverage), but cap so
            // one maze can't dominate a batch.
            let per_maze = 4.min(cfg.batch - filled);
            for k in 0..per_maze {
                // deterministic-ish spread across the cell list
                let cell = cells[(train_seed as usize + k * 7) % cells.len()];
                let label = match optimal_move(&maze, &dist, cell) {
                    Some(l) => l,
                    None => continue,
                };
                let (out, cache) = vin.forward_train(&toks, cfg.size, cfg.size, cell);
                let probs = softmax4(&out.move_logits);
                // cross-entropy loss
                batch_loss += -(probs[label].max(1e-20)).ln();
                if argmax4(&out.move_logits) == label {
                    batch_correct += 1;
                }
                // upstream grad = softmax - onehot
                let mut d_logits = vec![0.0f32; 4];
                for i in 0..4 {
                    d_logits[i] = probs[i] - if i == label { 1.0 } else { 0.0 };
                }
                let g = vin.backward(&cache, &toks, cfg.size, cfg.size, &d_logits);
                grads.add(&g);
                filled += 1;
                if filled >= cfg.batch {
                    break;
                }
            }
        }
        // mean over batch, with a cosine-style warmdown over the back half of
        // training so the lr=1e-2 schedule consolidates the ~90% solve regime
        // instead of spiking out of it late (observed at the flat lr).
        let frac = b as f32 / cfg.n_batches.max(1) as f32;
        let lr_scale = if frac < 0.5 {
            1.0
        } else {
            // cosine from 1.0 down to 0.1 over the back half
            let t = (frac - 0.5) / 0.5;
            0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
        };
        vin.apply_grads(&grads, lr_scale * cfg.lr / cfg.batch as f32);

        if b % cfg.eval_every == 0 || b == cfg.n_batches - 1 {
            let ev = evaluate(&vin, cfg.size, cfg.raw_dim, cfg.eval_mazes, 9_000_000, None);
            println!(
                "  {:5} | {:7.4} | {:7.1}% | {:8.1}% | {:.2}",
                b,
                batch_loss / cfg.batch as f32,
                100.0 * batch_correct as f32 / cfg.batch as f32,
                100.0 * ev.solve_rate,
                ev.avg_wall_hits,
            );
        }
    }
    vin
}

/// Pull a `--flag <value>` argument, parsed.
fn arg_val<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("main");

    // ── verify: deserialize an exported weights JSON and run one forward ──
    if mode == "verify" {
        let path = args
            .iter()
            .position(|a| a == "--verify")
            .and_then(|i| args.get(i + 1))
            .cloned()
            .or_else(|| args.get(2).cloned())
            .expect("usage: vin_train verify --verify <json>");
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"));
        let vin: VinReadout = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("deserialize {path}: {e}"));
        // One forward on a trivial 9x9 to confirm 4 move logits come out.
        let s = 9usize;
        let toks = vec![1.0f32; s * s * vin.raw_dim];
        let out = vin.forward(&toks, s, s, Some((4, 4)));
        assert_eq!(out.move_logits.len(), 4, "expected 4 move logits");
        println!(
            "VERIFY OK: {path} deserialized; raw_dim={} value_dim={} iters={} → move_logits={:?}",
            vin.raw_dim, vin.config.value_dim, vin.config.iters, out.move_logits
        );
        return;
    }

    if mode == "sweep" {
        // Short sweep to find a promising config, then the caller runs `main`.
        let value_dims = [8usize, 16];
        let iters_set = [14usize, 20];
        let temps = [0.3f32, 0.5];
        let lrs = [3e-3f32, 1e-2];
        let batch = 32;
        println!("=== SWEEP (short, 400 batches each) on 9x9 ===");
        for &vd in &value_dims {
            for &it in &iters_set {
                for &tp in &temps {
                    for &lr in &lrs {
                        let cfg = TrainCfg {
                            value_dim: vd,
                            iters: it,
                            softmax_temp: tp,
                            lr,
                            batch,
                            raw_dim: 3,
                            size: 9,
                            n_batches: 400,
                            eval_every: 399,
                            eval_mazes: 60,
                        };
                        let vin = train(&cfg);
                        let ev = evaluate(&vin, 9, 3, 100, 9_500_000, None);
                        println!(
                            ">>> vd={vd} it={it} tp={tp} lr={lr} :: solve={:.1}% move_acc={:.1}%\n",
                            100.0 * ev.solve_rate,
                            100.0 * ev.move_acc
                        );
                    }
                }
            }
        }
        return;
    }

    // ── main: train the best config long, then eval + generalization ──
    let value_dim = arg_val(&args, "--value-dim").unwrap_or(16usize);
    let iters = arg_val(&args, "--iters").unwrap_or(20usize);
    let temp = arg_val(&args, "--temp").unwrap_or(0.5f32);
    let lr = arg_val(&args, "--lr").unwrap_or(1e-2f32);
    let n_batches = arg_val(&args, "--batches").unwrap_or(4000usize);
    let export_path: Option<String> = args
        .iter()
        .position(|a| a == "--export")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let cfg = TrainCfg {
        value_dim,
        iters,
        softmax_temp: temp,
        lr,
        batch: 32,
        raw_dim: 3,
        size: 9,
        n_batches,
        eval_every: 100,
        eval_mazes: 100,
    };

    println!("=== TRAIN 9x9 (honest: tokens = [is_open, is_goal, bias] only) ===");
    let vin = train(&cfg);

    println!("\n=== FINAL HELD-OUT EVAL (fresh seeds) ===");
    let final9 = evaluate(&vin, 9, cfg.raw_dim, 200, 9_700_000, None);
    println!(
        "9x9   : solve_rate={:.1}%  move_acc={:.1}%  wall_hits/maze={:.2}",
        100.0 * final9.solve_rate,
        100.0 * final9.move_acc,
        final9.avg_wall_hits
    );

    // Generalization: same trained VIN, bigger mazes, iters scaled to diameter.
    // 11x11 diameter ~ 20, 13x13 ~ 24.
    let g11 = evaluate(&vin, 11, cfg.raw_dim, 200, 9_710_000, Some(24));
    println!(
        "11x11 : solve_rate={:.1}%  move_acc={:.1}%  wall_hits/maze={:.2}  (iters=24)",
        100.0 * g11.solve_rate,
        100.0 * g11.move_acc,
        g11.avg_wall_hits
    );
    let g13 = evaluate(&vin, 13, cfg.raw_dim, 200, 9_720_000, Some(30));
    println!(
        "13x13 : solve_rate={:.1}%  move_acc={:.1}%  wall_hits/maze={:.2}  (iters=30)",
        100.0 * g13.solve_rate,
        100.0 * g13.move_acc,
        g13.avg_wall_hits
    );

    // ── Export trained weights + a companion reference JSON ──────────────
    if let Some(path) = export_path {
        let json = serde_json::to_string(&vin).expect("serialize VinReadout");
        std::fs::write(&path, &json)
            .unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("\nexported weights → {path} ({} bytes)", json.len());

        // Companion reference next to the weights file.
        let ref_path = std::path::Path::new(&path)
            .with_file_name("vin_solver_reference.json");
        let reference = serde_json::json!({
            "value_dim": vin.config.value_dim,
            "iters": vin.config.iters,
            "max_iters": vin.config.max_iters,
            "softmax_temp": vin.config.softmax_temp,
            "highway_gate": vin.config.highway_gate,
            "n_dirs": vin.config.n_dirs,
            "raw_dim": vin.raw_dim,
            "token_encoding": "per-cell [is_open, is_goal, bias=1.0]",
            "dir_order": "U,D,L,R",
            "final_solve_9x9": final9.solve_rate,
            "move_acc": final9.move_acc,
        });
        let ref_json = serde_json::to_string_pretty(&reference)
            .expect("serialize reference");
        std::fs::write(&ref_path, &ref_json)
            .unwrap_or_else(|e| panic!("write {}: {e}", ref_path.display()));
        println!(
            "exported reference → {} ({} bytes)",
            ref_path.display(),
            ref_json.len()
        );

        // Self-verify: deserialize the just-written file and run one forward.
        let back: VinReadout =
            serde_json::from_str(&json).expect("re-deserialize exported weights");
        let toks = vec![1.0f32; 9 * 9 * back.raw_dim];
        let out = back.forward(&toks, 9, 9, Some((4, 4)));
        assert_eq!(out.move_logits.len(), 4);
        println!(
            "self-verify OK: re-deserialized + forward → 4 move logits {:?}",
            out.move_logits
        );
    }
}
