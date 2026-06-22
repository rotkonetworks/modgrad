//! THE FAIR TEST — a REAL-capacity 8-region brain with spatial attention.
//!
//! Every prior maze failure used either toy capacity (32-neuron cortical
//! regions, 160 neurons total) OR no spatial attention — never both fixed
//! at once. The single CTM that solves 7×7 (≈59%) has 256 neurons in ONE
//! pool; the old "small" brain raced it with eight starved stubs.
//!
//! This trainer fixes BOTH:
//!   • REAL capacity — every region is REBUILT via `CtmConfig::region(...)`
//!     (so derived heads/synch/synapse-depth recompute consistently):
//!       cortical (input, attention, output, motor): d_model=384, d_input=128
//!       subcortical (cerebellum, basal_ganglia, insula, hippocampus):
//!                                                    d_model=96,  d_input=64
//!     → ~1700 neurons total (vs the old 160).
//!   • SPATIAL attention on the two regions closest to the maze and the
//!     move head: INPUT (region 0, sees the maze) and OUTPUT (region 2,
//!     feeds the `output_proj(global_sync)` move readout). Both attend over
//!     `size*size` tokens × 9 raw per-cell features — the EXACT mechanism
//!     the single CTM uses to localize. The other 6 regions stay flat.
//!   • n_global_sync = total_neurons.min(GSYNC) (default 512).
//!
//! Encoding is byte-identical to the single CTM (`export.rs::encode`):
//!   per cell: [is_wall, is_agent, is_goal, wall_up, wall_down, wall_left,
//!             wall_right, x_norm, y_norm]; OOB neighbors = wall.
//!
//! Training: grad-accumulation mini-batch with `regional_train_step` +
//! `RegionalAdamW`, BFS optimal-move labels, lr decay. Metrics: max-certainty
//! move accuracy + greedy solve rate, plus the majority-move baseline for
//! context. Progress logged to stderr AND a per-run file to survive a
//! watchdog kill.
//!
//! Env knobs:
//!   SIZE      odd grid size (default 7)
//!   BATCHES   grad-accum batches (default 1500)
//!   LR        initial learning rate (default 1.5e-3)
//!   TICKS     outer ticks (default 8)
//!   DCORT     cortical d_model (default 384)
//!   DICORT    cortical d_input (default 128)
//!   DSUB      subcortical d_model (default 96)
//!   DISUB     subcortical d_input (default 64)
//!   GSYNC     global-sync cap (default 512)
//!   EVERY     eval cadence in batches (default 25)
//!   LOG       progress log path (default /tmp/brain_real_<size>.log)
//!
//! Usage: SIZE=7 cargo run -p mazes --bin brain_real --release

#![allow(dead_code)]

#[path = "../maze_gen.rs"]
mod maze_gen;
use maze_gen::{Maze, MazeRng, generate_maze, DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT};

use modgrad_codec::genome::{Genome, ExpressedBrain};
use modgrad_ctm::config::{CtmConfig, ExitStrategy};
use modgrad_ctm::graph::{
    RegionalWeights, RegionalState, RegionalGradients, RegionalAdamW,
    regional_forward, regional_train_step,
};
use std::collections::VecDeque;
use std::io::Write;

const OUT_DIMS: usize = 4;   // UP, DOWN, LEFT, RIGHT
const TICKS: usize = 8;
const RAW_DIM: usize = 9;    // per-cell features (== single-CTM encode)
const BATCH: usize = 16;     // grad-accumulation batch
const SEED: u64 = 42;

// Region indices (eight_region_small order).
const INPUT: usize = 0;
const ATTENTION: usize = 1;
const OUTPUT: usize = 2;
const MOTOR: usize = 3;
const CEREBELLUM: usize = 4;
const BASAL_GANGLIA: usize = 5;
const INSULA: usize = 6;
const HIPPOCAMPUS: usize = 7;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

// ─── BFS optimal-move labels (verbatim from export.rs / brain_spatial.rs) ──

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

/// SPATIAL encoding: `size*size` tokens × RAW_DIM features, row-major cell
/// order, each token: [is_wall, is_agent, is_goal, wall_up, wall_down,
/// wall_left, wall_right, x_norm, y_norm]. OOB neighbors = wall. Byte-
/// identical to the single CTM (export.rs `encode`).
fn encode(maze: &Maze, agent: (usize, usize)) -> Vec<f32> {
    let s = maze.grid_size;
    let mut t = vec![0.0f32; s * s * RAW_DIM];
    let denom = (s - 1) as f32;
    let wall = |r: i32, c: i32| -> f32 {
        if r < 0 || c < 0 || r >= s as i32 || c >= s as i32 { 1.0 }
        else if maze.grid[r as usize * s + c as usize] { 1.0 } else { 0.0 }
    };
    for r in 0..s {
        for c in 0..s {
            let i = (r * s + c) * RAW_DIM;
            let (ri, ci) = (r as i32, c as i32);
            t[i] = wall(ri, ci);
            t[i + 1] = if (r, c) == agent { 1.0 } else { 0.0 };
            t[i + 2] = if (r, c) == maze.end { 1.0 } else { 0.0 };
            t[i + 3] = wall(ri - 1, ci);
            t[i + 4] = wall(ri + 1, ci);
            t[i + 5] = wall(ri, ci - 1);
            t[i + 6] = wall(ri, ci + 1);
            t[i + 7] = c as f32 / denom;
            t[i + 8] = r as f32 / denom;
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

fn predict_move(w: &RegionalWeights, maze: &Maze, agent: (usize, usize)) -> usize {
    let obs = encode(maze, agent);
    let mut st = RegionalState::new(w);
    let out = regional_forward(w, &mut st, &obs);
    argmax(committed_pred(&out))
}

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

/// Held-out metrics: optimal-move accuracy, greedy solve rate, and the
/// majority-move baseline (frequency of the most common optimal move) so
/// move_acc can be read against the right zero-line.
fn quick_eval(w: &RegionalWeights, size: usize, n_mazes: usize, seed: u64) -> (f32, f32, f32) {
    let mut erng = MazeRng::new(seed);
    let (mut mv_c, mut mv_t) = (0usize, 0usize);
    let mut solved = 0usize;
    let mut label_counts = [0usize; OUT_DIMS];
    for _ in 0..n_mazes {
        let maze = generate_maze(size, &mut erng);
        let dist = dist_to_goal(&maze.grid, size, maze.end);
        for _ in 0..4 {
            if let Some(cell) = random_open_cell(&maze, &dist, &mut erng) {
                if let Some(tgt) = optimal_move(&dist, &maze.grid, size, cell) {
                    label_counts[tgt] += 1;
                    if predict_move(w, &maze, cell) == tgt { mv_c += 1; }
                    mv_t += 1;
                }
            }
        }
        if solves(w, &maze) { solved += 1; }
    }
    let baseline = *label_counts.iter().max().unwrap() as f32 / mv_t.max(1) as f32;
    (mv_c as f32 / mv_t.max(1) as f32, solved as f32 / n_mazes as f32, baseline)
}

/// Diagnostic: does the OUTPUT region's activated state actually VARY with
/// agent position, or is it washed out? Returns the mean per-neuron stddev
/// of OUTPUT's activated vector across several agent positions in one maze.
fn output_region_variation(w: &RegionalWeights, size: usize, seed: u64) -> f32 {
    let mut rng = MazeRng::new(seed);
    let maze = generate_maze(size, &mut rng);
    let dist = dist_to_goal(&maze.grid, size, maze.end);
    let cells: Vec<(usize, usize)> = (0..size * size)
        .filter(|&i| !maze.grid[i] && dist[i] != u32::MAX)
        .map(|i| (i / size, i % size))
        .collect();
    let sample: Vec<(usize, usize)> = cells.iter().copied().take(12).collect();
    if sample.len() < 2 { return 0.0; }
    let mut acts: Vec<Vec<f32>> = Vec::new();
    for &cell in &sample {
        let obs = encode(&maze, cell);
        let mut st = RegionalState::new(w);
        let _ = regional_forward(w, &mut st, &obs);
        acts.push(st.region_outputs[OUTPUT].clone());
    }
    let d = acts[0].len();
    let mut total_std = 0.0f32;
    for j in 0..d {
        let mean: f32 = acts.iter().map(|a| a[j]).sum::<f32>() / acts.len() as f32;
        let var: f32 = acts.iter().map(|a| (a[j] - mean).powi(2)).sum::<f32>() / acts.len() as f32;
        total_std += var.sqrt();
    }
    total_std / d as f32
}

/// Mean per-element stddev of a set of vectors across positions.
fn mean_std(vecs: &[Vec<f32>]) -> f32 {
    if vecs.len() < 2 || vecs[0].is_empty() { return 0.0; }
    let d = vecs[0].len();
    let mut total = 0.0f32;
    for j in 0..d {
        let mean: f32 = vecs.iter().map(|v| v[j]).sum::<f32>() / vecs.len() as f32;
        let var: f32 = vecs.iter().map(|v| (v[j] - mean).powi(2)).sum::<f32>() / vecs.len() as f32;
        total += var.sqrt();
    }
    total / d as f32
}

/// Diagnostic for the readout fix: does the OUTPUT region's OWN sync_out
/// (the EXACT input the local move head consumes) vary with agent position,
/// or does the per-region synch pooling wash out the localized signal that
/// `activated` clearly carries? Compares, across positions in one maze, the
/// mean per-element stddev of (a) OUTPUT.activated and (b) OUTPUT.sync_out,
/// plus the spread of the resulting move logits.
fn output_sync_variation(w: &RegionalWeights, size: usize, seed: u64) -> (f32, f32, f32) {
    let mut rng = MazeRng::new(seed);
    let maze = generate_maze(size, &mut rng);
    let dist = dist_to_goal(&maze.grid, size, maze.end);
    let cells: Vec<(usize, usize)> = (0..size * size)
        .filter(|&i| !maze.grid[i] && dist[i] != u32::MAX)
        .map(|i| (i / size, i % size))
        .take(16)
        .collect();
    if cells.len() < 2 { return (0.0, 0.0, 0.0); }
    let mut act: Vec<Vec<f32>> = Vec::new();
    let mut syn: Vec<Vec<f32>> = Vec::new();
    let mut logits: Vec<Vec<f32>> = Vec::new();
    for &cell in &cells {
        let obs = encode(&maze, cell);
        let mut st = RegionalState::new(w);
        let out = regional_forward(w, &mut st, &obs);
        act.push(st.region_outputs[OUTPUT].clone());
        if let Some(s) = out.output_local_sync.clone() { syn.push(s); }
        logits.push(committed_pred(&out).to_vec());
    }
    (mean_std(&act), mean_std(&syn), mean_std(&logits))
}

fn main() {
    let size = env_usize("SIZE", 7);
    let batches = env_usize("BATCHES", 1500);
    let lr = env_f32("LR", 1.5e-3);
    let ticks = env_usize("TICKS", TICKS);
    let d_cort = env_usize("DCORT", 384);
    let di_cort = env_usize("DICORT", 128);
    let d_sub = env_usize("DSUB", 96);
    let di_sub = env_usize("DISUB", 64);
    let gsync_cap = env_usize("GSYNC", 512);
    let every = env_usize("EVERY", 25);
    let evaln = env_usize("EVALN", 40);
    let n_cells = size * size;
    let od = n_cells * RAW_DIM;
    let log_path = std::env::var("LOG")
        .unwrap_or_else(|_| format!("/tmp/brain_real_{size}.log"));
    let mut logf = std::fs::File::create(&log_path).expect("create log");
    let mut log = |s: &str| {
        eprintln!("{s}");
        let _ = writeln!(logf, "{s}");
        let _ = logf.flush();
    };

    // Start from the real genome API (EightRegionSmall), then REBUILD every
    // region at real capacity so all derived params recompute consistently.
    let genome = Genome::blank_slate(size, size, OUT_DIMS, ticks);
    let ExpressedBrain { cortex: _cortex, mut config } = genome.express();

    log(&format!(
        "blank_slate({size},{size},{OUT_DIMS},{ticks}) → {} regions (PRE-REBUILD). \
         old total_neurons={}, old n_global_sync={}",
        config.regions.len(),
        config.regions.iter().map(|r| r.d_model).sum::<usize>(),
        config.n_global_sync,
    ));

    // Per-region rebuild. Cortical = big, subcortical = medium. Memory/NLM
    // depth and exit betas mirror eight_region_small's intent.
    let beta = |b: f32| ExitStrategy::AdaptiveGate { beta: b, threshold: 0.99 };
    let mut rebuild = |idx: usize, name: &str, d_model: usize, d_input: usize,
                       mem: usize, deep: bool, b: f32| {
        config.regions[idx] = CtmConfig::region(name, d_model, d_input, mem, deep, ticks, beta(b));
    };
    rebuild(INPUT,         "input",         d_cort, di_cort, 8,  false, 0.05);
    rebuild(ATTENTION,     "attention",     d_cort, di_cort, 8,  true,  0.10);
    rebuild(OUTPUT,        "output",        d_cort, di_cort, 16, true,  0.10);
    rebuild(MOTOR,         "motor",         d_cort, di_cort, 8,  false, 0.05);
    rebuild(CEREBELLUM,    "cerebellum",    d_sub,  di_sub,  4,  false, 0.05);
    rebuild(BASAL_GANGLIA, "basal_ganglia", d_sub,  di_sub,  4,  false, 0.10);
    rebuild(INSULA,        "insula",        d_sub,  di_sub,  4,  false, 0.05);
    rebuild(HIPPOCAMPUS,   "hippocampus",   d_sub,  di_sub,  8,  true,  0.15);

    // Observation: raw_obs_dim = full flat encoding length (flat regions /
    // obs_proj see the whole encoding); spatial regions bypass this.
    config.raw_obs_dim = od;
    config.obs_scale_dims = vec![od];

    // SPATIAL on INPUT (sees the maze) AND OUTPUT (closest to the move head).
    // Both attend over n_cells tokens × RAW_DIM; kv_proj sized RAW_DIM→d_input.
    config.regions[INPUT].spatial = Some((n_cells, RAW_DIM));
    config.regions[OUTPUT].spatial = Some((n_cells, RAW_DIM));

    // Recompute global sync over the real neuron count.
    let total_neurons: usize = config.regions.iter().map(|r| r.d_model).sum();
    config.n_global_sync = total_neurons.min(gsync_cap);

    // READOUT FIX (gated): read the move DIRECTLY from the OUTPUT region's
    // own sync_out via a learned output-local head, bypassing global-sync
    // pooling. READOUT=1 enables it; default 0 keeps the legacy global-sync
    // readout for an apples-to-apples baseline.
    let readout_local = env_usize("READOUT", 0) != 0;
    if readout_local {
        config.output_local_region = Some(OUTPUT);
    }

    log(&format!(
        "REBUILT: cortical d_model={d_cort}/d_input={di_cort}, subcortical d_model={d_sub}/d_input={di_sub}; \
         total_neurons={total_neurons}, n_global_sync={}; SPATIAL=[input,output] ({n_cells} tokens × {RAW_DIM})",
        config.n_global_sync,
    ));

    let mut w = RegionalWeights::new(config);
    log(&format!(
        "brain params: {}  READOUT={}  (output_local_region={:?}, local_head={})",
        w.n_params(), readout_local as u8,
        w.config.output_local_region,
        w.output_local_head.is_some(),
    ));

    // Invariants: both spatial regions read RAW_DIM per token.
    {
        let probe = generate_maze(size, &mut MazeRng::new(1));
        let obs = encode(&probe, probe.start);
        assert_eq!(obs.len(), od, "encode len {} != obs_dim {od}", obs.len());
        assert_eq!(w.regions[INPUT].kv_proj.in_dim, RAW_DIM,
            "INPUT kv_proj.in_dim {} != RAW_DIM {RAW_DIM}", w.regions[INPUT].kv_proj.in_dim);
        assert_eq!(w.regions[OUTPUT].kv_proj.in_dim, RAW_DIM,
            "OUTPUT kv_proj.in_dim {} != RAW_DIM {RAW_DIM}", w.regions[OUTPUT].kv_proj.in_dim);
        log(&format!(
            "verified: INPUT.kv_proj.in_dim={}, OUTPUT.kv_proj.in_dim={} (== RAW_DIM {RAW_DIM})",
            w.regions[INPUT].kv_proj.in_dim, w.regions[OUTPUT].kv_proj.in_dim,
        ));
    }

    // DIAG: probe whether the OUTPUT-local head's INPUT (region sync_out)
    // varies with position the way OUTPUT.activated does — the where-it-dies
    // question. Runs on the fresh brain AND after a short warmup.
    if env_usize("DIAG", 0) != 0 {
        let (a0, s0, l0) = output_sync_variation(&w, size, 4242);
        log(&format!(
            "DIAG(init): OUTPUT.activated_var={a0:.4}  OUTPUT.sync_out_var={s0:.4}  \
             move_logit_var={l0:.4}"
        ));
        // brief warmup so the head/region are not at init
        let mut opt0 = RegionalAdamW::new(&w).with_lr(lr).with_clip(5.0);
        let mut rng0 = MazeRng::new(SEED);
        for _ in 0..40 {
            let mut grads = RegionalGradients::zeros(&w);
            let mut filled = 0;
            while filled < BATCH {
                let maze = generate_maze(size, &mut rng0);
                let dist = dist_to_goal(&maze.grid, size, maze.end);
                let Some(cell) = random_open_cell(&maze, &dist, &mut rng0) else { continue };
                let Some(target) = optimal_move(&dist, &maze.grid, size, cell) else { continue };
                let obs = encode(&maze, cell);
                let _ = regional_train_step(&w, &mut grads, &obs, target);
                filled += 1;
            }
            opt0.step(&mut w, &mut grads);
        }
        let (a1, s1, l1) = output_sync_variation(&w, size, 4242);
        log(&format!(
            "DIAG(warm40): OUTPUT.activated_var={a1:.4}  OUTPUT.sync_out_var={s1:.4}  \
             move_logit_var={l1:.4}"
        ));
        return;
    }

    log(&format!("training: size={size} batch={BATCH}x{batches} lr={lr} eval-every={every}"));
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

        if b % every == 0 || b == batches - 1 {
            let (mv, solve, base) = quick_eval(&w, size, evaln, 9999);
            let var = output_region_variation(&w, size, 7777);
            if solve > best_solve { best_solve = solve; }
            if mv > best_move { best_move = mv; }
            log(&format!(
                "  batch {b:4}/{batches}: loss={:.3}  move_acc={:.1}%  solve={:.1}%  \
                 (base={:.1}%, best mv {:.1}%/solve {:.1}%, out_var={:.4}, {:.0}s)",
                batch_loss / BATCH as f32, mv * 100.0, solve * 100.0, base * 100.0,
                best_move * 100.0, best_solve * 100.0, var, t0.elapsed().as_secs_f32()
            ));
        }
    }

    let (move_acc, solve_rate, base) = quick_eval(&w, size, 300, 12345);
    let var = output_region_variation(&w, size, 7777);
    log(&format!(
        "FINAL size={size}: move_acc={:.1}%  solve_rate={:.1}%  baseline={:.1}%  \
         (best mv {:.1}%/solve {:.1}%, out_var={:.4}, {:.0}s total)",
        move_acc * 100.0, solve_rate * 100.0, base * 100.0,
        best_move * 100.0, best_solve * 100.0, var, t0.elapsed().as_secs_f32()
    ));
}
