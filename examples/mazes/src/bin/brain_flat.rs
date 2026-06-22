//! FLAT-encoding trainer/exporter: the SAME parity-validated 8-region brain
//! (`Genome::blank_slate(9,9,4,8).express()` → EightRegionSmall, out_dims=4,
//! outer_ticks=8) but with the VisualCortex retina DROPPED and replaced by a
//! flat per-cell wall encoding. The de-risk (`brain_export.rs`) showed the
//! 8-region brain can navigate from this flat encoding (~78% move-acc / ~15%
//! solve at 7×7); this reproduces it for 9×9 and exports for the wasm engine.
//!
//! Forward path is: maze → encode (flat 729-dim obs) → regional_forward.
//! NO retina, NO cortex. The brain runs n_tokens=1 and consumes the whole
//! 729-dim flat vector through `obs_proj` and every `receives_observation`
//! connection synapse (same obs-bottleneck fix the retina version used, but
//! here raw_obs_dim is the flat encoding dim, not the flattened retina).
//!
//! Training: grad-accumulation mini-batching with `regional_train_step` +
//! `RegionalAdamW`, BFS optimal-move labels, lr decay. Metrics: max-certainty
//! move accuracy + greedy solve rate.
//!
//! Export (to <out_dir>): `brain_flat_weights.json` (RegionalWeights MINUS
//! embeddings; NO cortex/retina) + `brain_flat_reference.json` (config echo +
//! metrics + curated demo mazes the brain solves + oracle per-tick traces).
//!
//! Usage: cargo run -p mazes --bin brain_flat --release -- <out_dir>

#![allow(dead_code)]

#[path = "../maze_gen.rs"]
mod maze_gen;
use maze_gen::{Maze, MazeRng, generate_maze, DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT};

use modgrad_codec::genome::{Genome, ExpressedBrain};
use modgrad_ctm::graph::{
    RegionalWeights, RegionalState, RegionalGradients, RegionalAdamW,
    regional_forward, regional_train_step,
};
use serde::Serialize;
use serde_json::Value;
use std::collections::VecDeque;

const SIZE: usize = 9;       // odd grid (9x9 = 81 cells)
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

fn obs_dim() -> usize { SIZE * SIZE * RAW_DIM } // 81 * 9 = 729

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

/// FLAT encoding: [size*size*RAW_DIM] concatenated cell-by-cell (row-major).
/// Per cell, in order:
///   [is_wall, is_agent, is_goal, wall_up, wall_down, wall_left, wall_right, x_norm, y_norm]
/// where x_norm = c/(s-1), y_norm = r/(s-1). OOB neighbors count as walls.
/// THIS ORDER MUST BE MIRRORED EXACTLY by the wasm/JS encoder.
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
                    if predict_move(w, &maze, cell) == tgt { mv_c += 1; }
                    mv_t += 1;
                }
            }
        }
        if solves(w, &maze) { solved += 1; }
    }
    (mv_c as f32 / mv_t.max(1) as f32, solved as f32 / n_mazes as f32)
}

// ─── Faithful replica of regional_forward with per-tick capture ─────
// (verbatim from brain_train.rs — guarantees dumped traces match the
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
    use modgrad_ctm::forward::{ctm_forward, CtmInput};
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

// ─── Serializable reference schema ──────

#[derive(Serialize)]
struct TickTrace {
    prediction: Vec<f32>,
    region_activations: Vec<Vec<f32>>,
    global_sync: Vec<f32>,
    exit_lambda: Option<f32>,
}

/// One traced (maze, agent) sample with the flat observation. The oracle
/// traces store the flat `observation` so the demo can show per-tick state.
#[derive(Serialize)]
struct MazeSample {
    grid: Vec<u8>,
    agent: [usize; 2],
    goal: [usize; 2],
    observation: Vec<f32>,   // flat [size*size*RAW_DIM]
    raw_dim: usize,
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
    encoding: String,
    move_acc: f32,
    solve_rate: f32,
    region_names: Vec<String>,
    regions: Vec<RegionInfo>,
    connections: Vec<ConnectionInfo>,
    demo_mazes: Vec<DemoMaze>,
    oracle: Vec<MazeSample>,
}

/// Build a full traced MazeSample for one (maze, agent) via the flat encoding.
fn build_sample(w: &RegionalWeights, grid: &[u8], size: usize, agent: (usize, usize), goal: (usize, usize)) -> MazeSample {
    // Reconstruct a Maze just for encode() (only grid/grid_size/end are read).
    let maze = Maze {
        grid: grid.iter().map(|&b| b != 0).collect(),
        grid_size: size,
        start: agent,
        end: goal,
        route: Vec::new(),
        path_length: 0,
    };
    let obs = encode(&maze, agent);

    let mut traced_state = RegionalState::new(w);
    let traced = regional_forward_traced(w, &mut traced_state, &obs);

    // Cross-check against the real regional_forward (predictions only).
    let mut real_state = RegionalState::new(w);
    let real_out = regional_forward(w, &mut real_state, &obs);
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
        observation: obs,
        raw_dim: RAW_DIM,
        ticks_used: traced.ticks_used,
        ticks,
    }
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let batches = env_usize("BATCHES", 1500);
    let lr = env_f32("LR", 1.5e-3);
    let ticks = env_usize("TICKS", TICKS); // outer_ticks; engine reads this from JSON
    let od = obs_dim();

    // Build the brain from the real genome API — EXACT engine config (sans retina).
    let genome = Genome::blank_slate(SIZE, SIZE, OUT_DIMS, ticks);
    let ExpressedBrain { cortex: _cortex, mut config } = genome.express();

    eprintln!(
        "blank_slate({SIZE},{SIZE},{OUT_DIMS},{ticks}) → {} regions, n_global_sync={}, out_dims={}, raw_obs_dim={} (per-token), obs_scale_dims={:?} (PRE-FIX). FLAT obs_dim={od}",
        config.regions.len(), config.n_global_sync, config.out_dims,
        config.raw_obs_dim, config.obs_scale_dims,
    );

    // ── OBSERVATION FIX (flat encoding) ────────────────────────────────
    // `express()` sets raw_obs_dim = retina token_dim (128). We DROP the
    // retina and feed a FLAT 729-dim per-cell wall encoding instead. So set
    // raw_obs_dim = 729 and obs_scale_dims = [729] BEFORE RegionalWeights::new,
    // so obs_proj is sized (729 → regions[0].d_input) and every
    // receives_observation synapse reads the whole 729-dim observation.
    config.raw_obs_dim = od;
    config.obs_scale_dims = vec![od];
    eprintln!(
        "FIXED raw_obs_dim={} obs_scale_dims={:?} (full flat encoding, single scale)",
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

    // Post-fix invariant: obs_proj reads the full 729-dim flat observation.
    {
        let probe = generate_maze(SIZE, &mut MazeRng::new(1));
        let obs = encode(&probe, probe.start);
        assert_eq!(obs.len(), od, "flat encode len {} != obs_dim {od}", obs.len());
        assert_eq!(w.obs_proj.in_dim, od,
            "obs_proj in_dim {} != obs_dim {od} (fix not applied)", w.obs_proj.in_dim);
        eprintln!("obs_proj.in_dim={} (verified == {od})", w.obs_proj.in_dim);
    }

    // ── Train: grad-accumulation mini-batch, BFS optimal-move labels. ──
    eprintln!("training: batch={BATCH}x{batches} lr={lr}");
    let mut opt = RegionalAdamW::new(&w).with_lr(lr).with_clip(5.0);
    let mut rng = MazeRng::new(SEED);
    let t0 = std::time::Instant::now();
    let mut best_solve = 0.0f32;

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
            let (loss, _pred) = regional_train_step(&w, &mut grads, &obs, target);
            batch_loss += loss;
            filled += 1;
        }
        opt.lr = lr * (1.0 - 0.9 * b as f32 / batches as f32);
        opt.step(&mut w, &mut grads);

        if b % 50 == 0 || b == batches - 1 {
            let (mv, solve) = quick_eval(&w, 40, 9999);
            if solve > best_solve { best_solve = solve; }
            eprintln!(
                "  batch {b:4}/{batches}: loss={:.3}  move_acc={:.1}%  solve_rate={:.1}%  (best {:.1}%, {:.0}s)",
                batch_loss / BATCH as f32, mv * 100.0, solve * 100.0, best_solve * 100.0, t0.elapsed().as_secs_f32()
            );
        }
    }

    let (move_acc, solve_rate) = quick_eval(&w, 300, 12345);
    eprintln!(
        "FINAL: move_acc={:.1}%  solve_rate={:.1}%  ({:.0}s total)",
        move_acc * 100.0, solve_rate * 100.0, t0.elapsed().as_secs_f32()
    );

    // ── Curate demo mazes the trained brain SOLVES + oracle traces. ──
    let mut crng = MazeRng::new(777);
    let mut demo_mazes: Vec<DemoMaze> = Vec::new();
    let mut oracle: Vec<MazeSample> = Vec::new();
    let mut tries = 0;
    while demo_mazes.len() < 16 && tries < 40000 {
        tries += 1;
        let maze = generate_maze(SIZE, &mut crng);
        if maze.path_length < 3 { continue; }
        if !solves(&w, &maze) { continue; }
        let grid: Vec<u8> = maze.grid.iter().map(|&b| b as u8).collect();
        if oracle.len() < 6 {
            oracle.push(build_sample(&w, &grid, SIZE, maze.start, maze.end));
        }
        demo_mazes.push(DemoMaze {
            grid,
            start: [maze.start.0, maze.start.1],
            end: [maze.end.0, maze.end.1],
            path_length: maze.path_length,
        });
    }
    eprintln!("curated {} demo mazes ({} oracle traces) in {tries} tries",
        demo_mazes.len(), oracle.len());

    let reference = Reference {
        task: "brain_maze_flat".to_string(),
        size: SIZE, ticks, out_dims: OUT_DIMS,
        raw_obs_dim, obs_scale_dims, n_global_sync,
        encoding: "flat_per_cell_walls_v9".to_string(),
        move_acc, solve_rate,
        region_names, regions: region_infos, connections: conn_infos,
        demo_mazes, oracle,
    };

    // ── Serialize weights MINUS embeddings; NO cortex. ──
    let mut weights_val: Value = serde_json::to_value(&w).expect("serialize weights to Value");
    if let Some(obj) = weights_val.as_object_mut() {
        obj.remove("embeddings");
    }

    #[derive(Serialize)]
    struct BrainWeights {
        // No `cortex` field — the engine must treat it as Option/default.
        regional: Value,
    }
    let brain_weights = BrainWeights { regional: weights_val };

    let w_path = format!("{out_dir}/brain_flat_weights.json");
    let r_path = format!("{out_dir}/brain_flat_reference.json");
    let w_str = serde_json::to_string(&brain_weights).expect("serialize brain_flat_weights");
    std::fs::write(&w_path, &w_str).expect("write brain_flat_weights.json");
    std::fs::write(&r_path, serde_json::to_string(&reference).expect("serialize reference"))
        .expect("write brain_flat_reference.json");

    eprintln!("wrote {} ({} bytes) + {}", w_path, w_str.len(), r_path);
}
