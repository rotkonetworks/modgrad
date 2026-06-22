//! MILESTONE 1 oracle exporter: full 8-region brain WITH visual retina.
//!
//! Builds the brain via the real `Genome::blank_slate(9,9,4,8).express()`
//! API (random init — fine, parity only needs the SAME weights run through
//! two implementations), renders fixed 9x9 mazes to RGB pixel tensors,
//! runs retina (`spatial_tokens`) → flat observation → `regional_forward`,
//! and dumps per-outer-tick {prediction, per-region activations,
//! global_sync, exit_lambda} so the dep-free engine can be proven
//! bit-exact against it.
//!
//! This binary REPLICATES the `regional_forward` outer loop (using the
//! real SDK `ctm_forward` + the real weights) so it can capture per-tick
//! traces the public `RegionalOutput` doesn't expose. It cross-checks
//! its replicated final-tick predictions against the real
//! `regional_forward` to guarantee the trace is faithful.
//!
//! Usage: cargo run -p mazes --bin brain_oracle --release -- <out_dir>
//! (CPU only — do NOT pass --features rocm)

#![allow(dead_code)]

use modgrad_codec::genome::{Genome, ExpressedBrain};
use modgrad_ctm::graph::{RegionalWeights, RegionalState, regional_forward};
use modgrad_ctm::forward::{ctm_forward, CtmInput};
use serde::Serialize;
use serde_json::Value;

const SIZE: usize = 9;
const OUT_DIMS: usize = 4;
const TICKS: usize = 8;

// ─── Maze rendering → RGB pixels [3 × H × W] ────────────────────────
//
// Deterministic, fixed color scheme. The engine test reconstructs the
// exact same pixels from the dumped `grid`/`agent`/`goal` (we also dump
// the raw pixels AND the flat observation, so the test can pick either).
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
            px[idx] = rr;             // R plane
            px[n + idx] = gg;         // G plane
            px[2 * n + idx] = bb;     // B plane
        }
    }
    px
}

/// Four fixed deterministic 9x9 mazes (1 = wall, 0 = open), row-major.
/// Hand-laid so each has an open agent cell and an open goal cell.
fn fixed_mazes() -> Vec<(Vec<u8>, (usize, usize), (usize, usize))> {
    // Helper to build a maze from an ASCII pattern.
    let parse = |rows: [&str; SIZE]| -> Vec<u8> {
        let mut g = vec![0u8; SIZE * SIZE];
        for (r, row) in rows.iter().enumerate() {
            for (c, ch) in row.chars().enumerate() {
                g[r * SIZE + c] = if ch == '#' { 1 } else { 0 };
            }
        }
        g
    };

    let m0 = parse([
        "#########",
        "#.......#",
        "#.#####.#",
        "#.#...#.#",
        "#.#.#.#.#",
        "#.#.#.#.#",
        "#...#...#",
        "#.#####.#",
        "#########",
    ]);
    let m1 = parse([
        "#########",
        "#.#.....#",
        "#.#.###.#",
        "#...#...#",
        "###.#.###",
        "#...#...#",
        "#.###.#.#",
        "#.....#.#",
        "#########",
    ]);
    let m2 = parse([
        "#########",
        "#.......#",
        "#######.#",
        "#.......#",
        "#.#######",
        "#.......#",
        "#######.#",
        "#.......#",
        "#########",
    ]);
    let m3 = parse([
        "#########",
        "#...#...#",
        "#.#.#.#.#",
        "#.#...#.#",
        "#.#####.#",
        "#.....#.#",
        "#####.#.#",
        "#.......#",
        "#########",
    ]);

    vec![
        (m0, (1, 1), (7, 7)),
        (m1, (1, 1), (7, 7)),
        (m2, (1, 1), (7, 7)),
        (m3, (1, 1), (7, 7)),
    ]
}

// ─── Faithful replica of regional_forward with per-tick capture ─────
//
// Mirrors `modgrad_ctm::graph::regional_forward` operation-for-operation
// (fixed-connection path, no router), but records the per-outer-tick
// region activations, global_sync, prediction and exit lambda. Uses the
// real SDK `ctm_forward` for each region. Sequential (no rayon) so the
// floating-point accumulation order is fully deterministic.
struct TraceOut {
    predictions: Vec<Vec<f32>>,         // [tick][out_dims]
    region_acts: Vec<Vec<Vec<f32>>>,    // [tick][region][d_model]
    global_sync: Vec<Vec<f32>>,         // [tick][n_global_sync]
    exit_lambdas: Vec<f32>,             // [tick] (may be shorter on early exit)
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

    // Mirror the SDK's double-buffer semantics: `prev_outputs` is the
    // PREVIOUS tick's committed `region_outputs`. The SDK initializes
    // both `region_outputs` and `prev_outputs` to each region's
    // `start_activated`, so at tick 0 prev == the start states.
    let mut prev_outputs: Vec<Vec<f32>> = state.region_outputs.clone();

    for _outer_tick in 0..cfg.outer_ticks {

        // Phase A: build each region's observation via fixed connections.
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
                    // merge_into_region_obs: assign on empty, else add.
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

        // Phase B: run each region's CTM (sequential).
        let mut results: Vec<Vec<f32>> = Vec::with_capacity(n_regions);
        for r in 0..n_regions {
            let d_input = w.regions[r].config.d_input;
            let _out = ctm_forward(&w.regions[r], &mut state.region_states[r], CtmInput::Raw {
                obs: &region_obs[r], n_tokens: 1, raw_dim: d_input,
            });
            results.push(state.region_states[r].activated.clone());
        }

        // Phase C: commit outputs.
        for r in 0..n_regions {
            state.region_outputs[r] = results[r].clone();
        }
        // Next tick's `prev` is this tick's committed outputs.
        prev_outputs = state.region_outputs.clone();

        // Phase 3: global sync.
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

        // Phase 4: output prediction.
        let prediction = w.output_proj.forward(&gs_buf);

        // Record trace.
        predictions.push(prediction);
        region_acts.push(state.region_outputs.clone());
        global_sync_trace.push(gs_buf.clone());

        // Phase 5: AdaptiveGate exit.
        if let Some(ref gate) = w.outer_exit_gate {
            let gate_logit = gate.forward(&gs_buf);
            let lambda = 1.0 / (1.0 + (-gate_logit[0]).exp());
            exit_lambdas.push(lambda);
            let p_exit = lambda * survival;
            exit_cdf += p_exit;
            survival *= 1.0 - lambda;
            // threshold is 0.99 in this config.
            if exit_cdf > 0.99 { break; }
        }
    }

    let ticks_used = predictions.len();
    TraceOut { predictions, region_acts, global_sync: global_sync_trace, exit_lambdas, ticks_used }
}

// ─── Serializable reference ─────────────────────────────────────────

#[derive(Serialize)]
struct TickTrace {
    prediction: Vec<f32>,
    region_activations: Vec<Vec<f32>>, // [region][d_model]
    global_sync: Vec<f32>,
    exit_lambda: Option<f32>,
}

#[derive(Serialize)]
struct MazeSample {
    grid: Vec<u8>,
    agent: [usize; 2],
    goal: [usize; 2],
    pixels: Vec<f32>,        // [3 × SIZE × SIZE] CHW
    observation: Vec<f32>,   // flat retina spatial tokens [n_tokens × token_dim]
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

#[derive(Serialize)]
struct Reference {
    task: String,
    size: usize,
    ticks: usize,
    out_dims: usize,
    raw_obs_dim: usize,
    obs_scale_dims: Vec<usize>,
    n_global_sync: usize,
    region_names: Vec<String>,
    regions: Vec<RegionInfo>,
    connections: Vec<ConnectionInfo>,
    samples: Vec<MazeSample>,
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());

    // Build the brain from the real genome API.
    let genome = Genome::blank_slate(SIZE, SIZE, OUT_DIMS, TICKS);
    let ExpressedBrain { cortex, config } = genome.express();

    eprintln!(
        "blank_slate({SIZE},{SIZE},{OUT_DIMS},{TICKS}) → {} regions, n_global_sync={}, out_dims={}, raw_obs_dim={}, obs_scale_dims={:?}",
        config.regions.len(), config.n_global_sync, config.out_dims,
        config.raw_obs_dim, config.obs_scale_dims,
    );
    for (i, (name, rc)) in config.region_names.iter().zip(&config.regions).enumerate() {
        eprintln!(
            "  region[{i}] {name}: d_model={} d_input={} mem={} heads={} iters={} deep_nlms={} exit={:?}",
            rc.d_model, rc.d_input, rc.memory_length, rc.heads, rc.iterations, rc.deep_nlms, rc.exit_strategy,
        );
    }
    for (i, c) in config.connections.iter().enumerate() {
        eprintln!("  conn[{i}] from={:?} → {} recv_obs={} scale={}",
            c.from, c.to, c.receives_observation, c.observation_scale);
    }

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

    // Build weights (FIXED seed inside RegionalWeights::new).
    let w = RegionalWeights::new(config);
    eprintln!("brain params: {}", w.n_params());

    // Run the retina + brain on each fixed maze.
    let mazes = fixed_mazes();
    let mut samples: Vec<MazeSample> = Vec::new();

    for (mi, (grid, agent, goal)) in mazes.iter().enumerate() {
        let pixels = render_pixels(grid, SIZE, *agent, *goal);

        // Retina → spatial tokens → flat observation.
        let (tokens, n_tokens, token_dim) = cortex.spatial_tokens(&pixels);
        eprintln!("maze {mi}: retina → n_tokens={n_tokens} token_dim={token_dim} (obs len {})", tokens.len());

        // Faithful traced replica.
        let mut traced_state = RegionalState::new(&w);
        let traced = regional_forward_traced(&w, &mut traced_state, &tokens);

        // Cross-check against the real regional_forward (predictions only).
        let mut real_state = RegionalState::new(&w);
        let real_out = regional_forward(&w, &mut real_state, &tokens);
        assert_eq!(real_out.predictions.len(), traced.predictions.len(),
            "maze {mi}: tick count mismatch real {} vs traced {}",
            real_out.predictions.len(), traced.predictions.len());
        let mut max_d = 0.0f32;
        for (t, (rp, tp)) in real_out.predictions.iter().zip(&traced.predictions).enumerate() {
            for (j, (&a, &b)) in rp.iter().zip(tp).enumerate() {
                let d = (a - b).abs();
                if d > max_d { max_d = d; }
                assert!(d < 1e-4,
                    "maze {mi} tick {t} pred[{j}]: traced {b} vs real {a} (|d|={d})");
            }
        }
        // Also confirm the real final-tick global_sync matches our trace.
        {
            let last = traced.global_sync.last().unwrap();
            for (j, (&a, &b)) in real_out.global_sync.iter().zip(last).enumerate() {
                assert!((a - b).abs() < 1e-4,
                    "maze {mi} final gs[{j}]: traced {b} vs real {a}");
            }
        }
        eprintln!("  cross-check OK (max |Δpred| traced-vs-real = {max_d:.2e}), ticks_used={}", traced.ticks_used);

        let ticks: Vec<TickTrace> = (0..traced.ticks_used).map(|t| TickTrace {
            prediction: traced.predictions[t].clone(),
            region_activations: traced.region_acts[t].clone(),
            global_sync: traced.global_sync[t].clone(),
            exit_lambda: traced.exit_lambdas.get(t).copied(),
        }).collect();

        samples.push(MazeSample {
            grid: grid.clone(),
            agent: [agent.0, agent.1],
            goal: [goal.0, goal.1],
            pixels,
            observation: tokens,
            n_tokens, token_dim,
            ticks_used: traced.ticks_used,
            ticks,
        });
    }

    let reference = Reference {
        task: "brain_maze".to_string(),
        size: SIZE, ticks: TICKS, out_dims: OUT_DIMS,
        raw_obs_dim, obs_scale_dims, n_global_sync,
        region_names, regions: region_infos, connections: conn_infos,
        samples,
    };

    // ── Serialize weights MINUS the (huge, unused) embeddings table. ──
    let mut weights_val: Value = serde_json::to_value(&w).expect("serialize weights to Value");
    if let Some(obj) = weights_val.as_object_mut() {
        obj.remove("embeddings");
    }
    // Also drop the retina/cortex into a sibling object so the engine can
    // reproduce the observation if it prefers pixels over the dumped obs.
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
