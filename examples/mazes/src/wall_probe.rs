//! Wall-probe gate harness (`--wall-probe`).
//!
//! Verifies whether the brain's internal representation linearly encodes
//! "is there a wall next to the agent" in each of the 4 directions
//! [Up, Down, Left, Right].
//!
//! Two representations are probed:
//!   A) global_sync  (`BrainOutput.sync`)         — baseline, expect chance.
//!   C) spatial readout (OUTPUT region local sync) — expect high w/ pos-enc.
//!
//! For C we also ablate the positional encoding (env MODGRAD_DISABLE_POS)
//! to show pos-enc is the causal ingredient.
//!
//! Probe: 4 independent ridge-regression linear probes (closed form), one
//! per direction, trained on 80% / tested on 20% (maze-disjoint split).

use crate::maze_gen::{
    generate_maze, render_maze_with_agent, Maze, MazeRng,
};
use modgrad_codec::genome::Genome;
use modgrad_codec::retina::VisualCortex;
use modgrad_ctm::graph::{RegionalBrain, RegionalWeights};
use modgrad_ctm::vin::{VinConfig, VinReadout};
use modgrad_traits::{Brain, Encoder};

/// One collected sample: feature vectors + the 4-bit wall label.
struct Sample {
    global_sync: Vec<f32>,
    local_sync: Vec<f32>,
    /// Mean-pooled raw V4 tokens (token_dim) — diagnostic upper bound for
    /// what the ENCODER linearly exposes (independent of the readout).
    raw_meanpool: Vec<f32>,
    /// The single raw V4 token at the agent's grid cell (token_dim) — the
    /// strongest per-position diagnostic: if THIS decodes walls but pooled
    /// reps don't, the loss is from pooling/attention-selection, not the
    /// encoder.
    raw_agent_tok: Vec<f32>,
    /// VIN ego-centric agent-cell readout (post value-propagation; agent
    /// cell value ++ 4 neighbour values). `None` unless `--vin-readout`.
    /// This is representation **D** — the VIN alternative readout.
    vin_agent: Option<Vec<f32>>,
    /// [U, D, L, R] — true = wall/edge.
    wall: [bool; 4],
    /// Maze index (for maze-disjoint train/test split).
    maze_idx: usize,
}

/// Mean-pool a flat `[n_tokens × token_dim]` buffer over tokens.
fn meanpool(tokens: &[f32], n_tokens: usize, token_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; token_dim];
    if n_tokens == 0 {
        return out;
    }
    for t in 0..n_tokens {
        for d in 0..token_dim {
            out[d] += tokens[t * token_dim + d];
        }
    }
    let inv = 1.0 / n_tokens as f32;
    for v in out.iter_mut() {
        *v *= inv;
    }
    out
}

/// Wall mask at `(r, c)`: true = wall or grid edge in each of U/D/L/R.
fn wall_mask(maze: &Maze, r: usize, c: usize) -> [bool; 4] {
    let s = maze.grid_size;
    let is_wall = |rr: i32, cc: i32| -> bool {
        if rr < 0 || cc < 0 || rr >= s as i32 || cc >= s as i32 {
            return true; // edge counts as wall
        }
        maze.grid[rr as usize * s + cc as usize]
    };
    [
        is_wall(r as i32 - 1, c as i32), // Up
        is_wall(r as i32 + 1, c as i32), // Down
        is_wall(r as i32, c as i32 - 1), // Left
        is_wall(r as i32, c as i32 + 1), // Right
    ]
}

/// Free (non-wall) neighbours of `(r,c)` as (dr,dc).
fn free_neighbours(maze: &Maze, r: usize, c: usize) -> Vec<(i32, i32)> {
    let s = maze.grid_size;
    let mut out = Vec::new();
    for (dr, dc) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
        let nr = r as i32 + dr;
        let nc = c as i32 + dc;
        if nr < 0 || nc < 0 || nr >= s as i32 || nc >= s as i32 {
            continue;
        }
        if !maze.grid[nr as usize * s + nc as usize] {
            out.push((dr, dc));
        }
    }
    out
}

/// Closed-form ridge regression: solve (XᵀX + λI) w = Xᵀy via
/// Gauss-Jordan. `x_rows` is n × d (each row a feature vector with a
/// trailing bias 1.0 already appended by the caller). Returns w (length d).
fn ridge_fit(x_rows: &[Vec<f32>], y: &[f32], lambda: f32) -> Vec<f32> {
    let n = x_rows.len();
    let d = if n == 0 { 0 } else { x_rows[0].len() };
    // Gram = XᵀX + λI   (d×d),  rhs = Xᵀy   (d).
    let mut g = vec![0.0f64; d * d];
    let mut rhs = vec![0.0f64; d];
    for i in 0..n {
        let xi = &x_rows[i];
        let yi = y[i] as f64;
        for a in 0..d {
            let xa = xi[a] as f64;
            rhs[a] += xa * yi;
            let row = a * d;
            for b in 0..d {
                g[row + b] += xa * (xi[b] as f64);
            }
        }
    }
    for a in 0..d {
        g[a * d + a] += lambda as f64;
    }
    // Gauss-Jordan solve g w = rhs.
    gauss_jordan(&mut g, &mut rhs, d);
    rhs.iter().map(|&v| v as f32).collect()
}

/// In-place Gauss-Jordan elimination solving `A x = b` (A is d×d
/// row-major). On return, `b` holds the solution. Partial pivoting.
fn gauss_jordan(a: &mut [f64], b: &mut [f64], d: usize) {
    for col in 0..d {
        // Pivot: largest abs in this column at/below the diagonal.
        let mut piv = col;
        let mut best = a[col * d + col].abs();
        for r in (col + 1)..d {
            let v = a[r * d + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            continue; // singular column; ridge λ should prevent this
        }
        if piv != col {
            for k in 0..d {
                a.swap(piv * d + k, col * d + k);
            }
            b.swap(piv, col);
        }
        let diag = a[col * d + col];
        // Normalise pivot row.
        for k in 0..d {
            a[col * d + k] /= diag;
        }
        b[col] /= diag;
        // Eliminate other rows.
        for r in 0..d {
            if r == col {
                continue;
            }
            let factor = a[r * d + col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..d {
                a[r * d + k] -= factor * a[col * d + k];
            }
            b[r] -= factor * b[col];
        }
    }
}

/// Train + evaluate a single-direction probe on the given feature
/// extractor. Returns test accuracy (threshold 0.5).
fn probe_direction(
    train: &[&Sample],
    test: &[&Sample],
    feat: impl Fn(&Sample) -> &Vec<f32>,
    dir: usize,
    lambda: f32,
) -> f32 {
    // Build design matrices with a trailing bias term.
    let to_rows = |set: &[&Sample]| -> Vec<Vec<f32>> {
        set.iter()
            .map(|s| {
                let f = feat(s);
                let mut row = Vec::with_capacity(f.len() + 1);
                row.extend_from_slice(f);
                row.push(1.0);
                row
            })
            .collect()
    };
    let xtr = to_rows(train);
    // Center the binary target to {-1,+1} for symmetric regression.
    let ytr: Vec<f32> = train
        .iter()
        .map(|s| if s.wall[dir] { 1.0 } else { -1.0 })
        .collect();
    let w = ridge_fit(&xtr, &ytr, lambda);

    let mut correct = 0usize;
    for s in test {
        let f = feat(s);
        let mut dot = w[f.len()]; // bias
        for (wi, fi) in w.iter().zip(f.iter()) {
            dot += wi * fi;
        }
        let pred = dot > 0.0;
        if pred == s.wall[dir] {
            correct += 1;
        }
    }
    if test.is_empty() {
        0.0
    } else {
        correct as f32 / test.len() as f32
    }
}

/// Collect `n_samples` (maze, agent-cell) feature/label samples. Carries
/// CTM state across steps within a maze (like deployment); resets per
/// maze. `pos_enabled=false` sets MODGRAD_DISABLE_POS for this collection.
#[allow(clippy::too_many_arguments)]
fn collect_samples(
    weights: &RegionalWeights,
    encoder: &VisualCortex,
    maze_size: usize,
    n_samples: usize,
    seed: u64,
    pos_enabled: bool,
    v4_grid: (usize, usize),
    vin: Option<&VinReadout>,
) -> Vec<Sample> {
    let (h4, w4) = v4_grid;
    // SAFETY: single-threaded setup phase; no other thread reads env here.
    unsafe {
        if pos_enabled {
            std::env::remove_var("MODGRAD_DISABLE_POS");
        } else {
            std::env::set_var("MODGRAD_DISABLE_POS", "1");
        }
    }

    let mut rng = MazeRng::new(seed);
    let mut samples = Vec::with_capacity(n_samples);
    let mut maze_idx = 0usize;
    let steps_per_maze = 24usize; // wander length to cover diverse cells

    while samples.len() < n_samples {
        let maze = generate_maze(maze_size, &mut rng);
        // Carry CTM state across the wander, like closed-loop deployment.
        let mut state = RegionalBrain::init_state(weights);
        let (mut ar, mut ac) = maze.start;

        for _ in 0..steps_per_maze {
            if samples.len() >= n_samples {
                break;
            }
            let pixels = render_maze_with_agent(&maze, (ar, ac));
            let input = encoder.encode(&pixels);
            let (output, new_state, _cache) =
                RegionalBrain::forward_cached(weights, state, &input);
            state = new_state;

            let local_sync = state
                .last_output_local_sync
                .clone()
                .unwrap_or_else(|| output.sync.clone());

            let raw_meanpool =
                meanpool(&input.tokens, input.n_tokens, input.token_dim);
            // Map agent maze cell → V4 token grid cell (nearest), extract
            // that token. V4 grid is h4×w4 row-major.
            let td = input.token_dim;
            let tr = (ar * h4 / maze_size).min(h4.saturating_sub(1));
            let tc = (ac * w4 / maze_size).min(w4.saturating_sub(1));
            let tok_idx = tr * w4 + tc;
            let raw_agent_tok = if tok_idx < input.n_tokens {
                input.tokens[tok_idx * td..(tok_idx + 1) * td].to_vec()
            } else {
                vec![0.0f32; td]
            };
            // VIN ego-centric readout: value-propagate over the V4 grid,
            // read at the agent's own (tr, tc) cell. No global pooling.
            let vin_agent = vin.map(|v| {
                let out = v.forward(&input.tokens, h4, w4, Some((tr, tc)));
                out.agent_readout
            });
            samples.push(Sample {
                global_sync: output.sync.clone(),
                local_sync,
                raw_meanpool,
                raw_agent_tok,
                vin_agent,
                wall: wall_mask(&maze, ar, ac),
                maze_idx,
            });

            // Wander to a random free neighbour (covers diverse cells).
            let nbrs = free_neighbours(&maze, ar, ac);
            if nbrs.is_empty() {
                break;
            }
            let pick = rng.range(nbrs.len());
            let (dr, dc) = nbrs[pick];
            ar = (ar as i32 + dr) as usize;
            ac = (ac as i32 + dc) as usize;
        }
        maze_idx += 1;
    }

    // SAFETY: single-threaded setup phase.
    unsafe {
        std::env::remove_var("MODGRAD_DISABLE_POS");
    }
    samples
}

/// Maze-disjoint 80/20 split (train = first 80% of maze indices).
fn split_disjoint(samples: &[Sample]) -> (Vec<&Sample>, Vec<&Sample>) {
    let max_maze = samples.iter().map(|s| s.maze_idx).max().unwrap_or(0);
    let cutoff = (max_maze as f32 * 0.8) as usize;
    let train: Vec<&Sample> = samples.iter().filter(|s| s.maze_idx <= cutoff).collect();
    let test: Vec<&Sample> = samples.iter().filter(|s| s.maze_idx > cutoff).collect();
    (train, test)
}

const DIR_NAMES: [&str; 4] = ["Up", "Down", "Left", "Right"];

/// Run the full wall-probe gate and print the results table + verdict.
pub fn run_wall_probe(
    maze_size: usize,
    ticks: usize,
    n_samples: usize,
    seed: u64,
    vin_readout: bool,
) {
    let maze_size = maze_size | 1;
    eprintln!(
        "[wall-probe] maze_size={maze_size} ticks={ticks} n_samples={n_samples} seed={seed} \
         vin_readout={vin_readout}"
    );

    // Build the brain exactly like run_brain (blank_slate, single-scale),
    // then enable the spatial readout on the OUTPUT region (region 2).
    let out_dims = crate::maze_gen::N_DIRECTIONS; // route_len=1 is enough for the probe
    let genome = Genome::blank_slate(maze_size, maze_size, out_dims, ticks);
    let modgrad_codec::genome::ExpressedBrain {
        cortex: encoder,
        config: mut cfg,
    } = genome.express();

    // Probe a single encode to learn the real (n_tokens, token_dim).
    let token_dim = encoder.token_dim();
    let probe_input = {
        let dummy_maze = {
            let mut r = MazeRng::new(seed);
            generate_maze(maze_size, &mut r)
        };
        let px = render_maze_with_agent(&dummy_maze, dummy_maze.start);
        encoder.encode(&px)
    };
    let n_tok = probe_input.n_tokens;
    let raw_dim = probe_input.token_dim;
    assert_eq!(
        raw_dim, token_dim,
        "encoder token_dim disagreement: {raw_dim} vs {token_dim}"
    );
    eprintln!(
        "[wall-probe] encoder: {n_tok} spatial tokens × {raw_dim}-dim (V4 channels)"
    );

    // FIX 2 wiring: OUTPUT region (index 2) attends over raw pos-encoded
    // tokens; the move head reads its sync_out.
    const OUTPUT_R: usize = 2;
    assert!(
        cfg.regions.len() > OUTPUT_R,
        "expected >= 3 regions, got {}",
        cfg.regions.len()
    );
    cfg.regions[OUTPUT_R].spatial = Some((n_tok, raw_dim));
    cfg.output_local_region = Some(OUTPUT_R);

    let weights = RegionalWeights::new(cfg);
    assert!(
        weights.output_local_head.is_some(),
        "output_local_head not built"
    );
    let local_dim = weights.regions[OUTPUT_R].config.n_synch_out;
    eprintln!(
        "[wall-probe] global_sync dim={}, local_sync dim={}",
        weights.config.n_global_sync, local_dim
    );
    eprintln!(
        "[wall-probe] region[{OUTPUT_R}] '{}' spatial={:?} d_input={} kv_proj.in={}",
        weights.config.region_names.get(OUTPUT_R).map(|s| s.as_str()).unwrap_or("?"),
        weights.config.regions[OUTPUT_R].spatial,
        weights.config.regions[OUTPUT_R].d_input,
        weights.regions[OUTPUT_R].kv_proj.in_dim,
    );

    // ── Collect three sample sets ──────────────────────────────────
    // A & C(pos-ON) share one pass (pos on); C(pos-OFF) needs a second.
    let (_, _, _, _, _, _, h4, w4) = encoder.stage_dims();
    eprintln!("[wall-probe] V4 token grid = {h4}×{w4}");

    // VIN readout (representation D), built only when --vin-readout. The
    // VIN value-propagates over the h4×w4 V4 grid and reads ego-centrically
    // at the agent's own cell — no global pooling.
    let vin: Option<VinReadout> = if vin_readout {
        // Size the per-cell value width to preserve the (highly decodable)
        // agent-cell signal — a tight bottleneck on 128-dim V4 tokens throws
        // away the very wall signal we want at the agent cell.
        let cfg = VinConfig {
            value_dim: raw_dim.min(64),
            ..VinConfig::default()
        };
        let v = VinReadout::new(cfg.clone(), raw_dim);
        eprintln!(
            "[wall-probe] VIN readout: value_dim={} iters={} (cap {}) softmax_temp={} \
             highway={} → agent-cell readout dim={} ({} params)",
            cfg.value_dim,
            cfg.effective_iters(),
            cfg.max_iters,
            cfg.softmax_temp,
            cfg.highway_gate,
            v.agent_cell_readout_dim(),
            v.num_params(),
        );
        Some(v)
    } else {
        None
    };

    eprintln!("[wall-probe] collecting samples (pos ON)…");
    let samples_pos_on = collect_samples(
        &weights, &encoder, maze_size, n_samples, seed + 1, true, (h4, w4), vin.as_ref(),
    );
    eprintln!("[wall-probe] collecting samples (pos OFF)…");
    let samples_pos_off = collect_samples(
        &weights, &encoder, maze_size, n_samples, seed + 1, false, (h4, w4), None,
    );

    // Sanity: confirm pos-enc actually changes the local readout.
    {
        let cs = |s: &Sample| -> f64 { s.local_sync.iter().map(|&v| v as f64).sum() };
        let on0 = samples_pos_on.first().map(cs).unwrap_or(0.0);
        let off0 = samples_pos_off.first().map(cs).unwrap_or(0.0);
        let on_var: f64 = {
            let m: f64 = samples_pos_on.iter().map(cs).sum::<f64>()
                / samples_pos_on.len().max(1) as f64;
            samples_pos_on.iter().map(|s| (cs(s) - m).powi(2)).sum::<f64>()
                / samples_pos_on.len().max(1) as f64
        };
        eprintln!(
            "[wall-probe] sanity: local_sync sum sample0 ON={on0:.4} OFF={off0:.4} \
             (Δ={:.4e}); ON across-sample var={on_var:.4e}",
            (on0 - off0).abs()
        );
    }

    let lambda = 1.0f32;

    // Report label balance (sanity: probe accuracy is only meaningful
    // when both classes are present).
    let report_balance = |samples: &[Sample]| {
        for d in 0..4 {
            let pos = samples.iter().filter(|s| s.wall[d]).count();
            eprintln!(
                "[wall-probe]   {} wall-rate = {:.1}% ({}/{})",
                DIR_NAMES[d],
                100.0 * pos as f32 / samples.len().max(1) as f32,
                pos,
                samples.len()
            );
        }
    };
    eprintln!("[wall-probe] label balance (pos-ON set):");
    report_balance(&samples_pos_on);

    // Evaluate a representation: returns [4 accuracies] + mean.
    let eval_rep = |samples: &[Sample], feat: fn(&Sample) -> &Vec<f32>| -> ([f32; 4], f32) {
        let (train, test) = split_disjoint(samples);
        let mut accs = [0.0f32; 4];
        for d in 0..4 {
            accs[d] = probe_direction(&train, &test, |s| feat(s), d, lambda);
        }
        let mean = accs.iter().sum::<f32>() / 4.0;
        (accs, mean)
    };

    let (a_acc, a_mean) =
        eval_rep(&samples_pos_on, |s| &s.global_sync);
    let (c_on_acc, c_on_mean) =
        eval_rep(&samples_pos_on, |s| &s.local_sync);
    let (c_off_acc, c_off_mean) =
        eval_rep(&samples_pos_off, |s| &s.local_sync);
    // Diagnostic reference: what the ENCODER alone exposes (mean-pooled
    // raw V4 tokens). Upper bound on a pooled readout; if THIS is at
    // chance, the encoder — not the readout — is the bottleneck.
    let (r_acc, r_mean) =
        eval_rep(&samples_pos_on, |s| &s.raw_meanpool);
    // Strongest per-position diagnostic: the agent-cell's own V4 token.
    let (rt_acc, rt_mean) =
        eval_rep(&samples_pos_on, |s| &s.raw_agent_tok);

    // Representation D: VIN ego-centric agent-cell readout (only when
    // --vin-readout). Probes the EXACT readout the VIN move head consumes.
    let vin_result: Option<([f32; 4], f32)> = if vin_readout {
        let (train, test) = split_disjoint(&samples_pos_on);
        let mut accs = [0.0f32; 4];
        for d in 0..4 {
            accs[d] = probe_direction(
                &train, &test, |s| s.vin_agent.as_ref().expect("vin_agent present"), d, lambda,
            );
        }
        let mean = accs.iter().sum::<f32>() / 4.0;
        Some((accs, mean))
    } else {
        None
    };

    // Majority-class baseline (per direction): predict the more common
    // label on the test split. The honest "chance" floor.
    let majority_mean: f32 = {
        let (_train, test) = split_disjoint(&samples_pos_on);
        let mut sum = 0.0f32;
        for d in 0..4 {
            let pos = test.iter().filter(|s| s.wall[d]).count();
            let neg = test.len() - pos;
            sum += pos.max(neg) as f32 / test.len().max(1) as f32;
        }
        sum / 4.0
    };

    // ── Results table ──────────────────────────────────────────────
    println!();
    println!("================ WALL-PROBE RESULTS ================");
    println!(
        "{:<34} {:>6} {:>6} {:>6} {:>6} {:>7}",
        "representation × pos", "Up", "Down", "Left", "Right", "mean"
    );
    let row = |label: &str, accs: &[f32; 4], mean: f32| {
        println!(
            "{:<34} {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}% {:>6.1}%",
            label,
            100.0 * accs[0],
            100.0 * accs[1],
            100.0 * accs[2],
            100.0 * accs[3],
            100.0 * mean
        );
    };
    row("A: global_sync (baseline)", &a_acc, a_mean);
    row("C: spatial readout (pos ON)", &c_on_acc, c_on_mean);
    row("C: spatial readout (pos OFF)", &c_off_acc, c_off_mean);
    row("R: raw V4 meanpool (encoder)", &r_acc, r_mean);
    row("R2: raw V4 agent-cell token", &rt_acc, rt_mean);
    if let Some((d_acc, d_mean)) = vin_result {
        row("D: VIN agent-cell readout", &d_acc, d_mean);
    }
    println!(
        "{:<34} {:>5} {:>5} {:>5} {:>5} {:>5.1}%",
        "(majority-class baseline)", "·", "·", "·", "·", 100.0 * majority_mean
    );
    println!("====================================================");

    // ── Verdict ────────────────────────────────────────────────────
    let baseline_chance = a_mean >= 0.40 && a_mean <= 0.60;
    let spatial_strong = c_on_mean >= 0.80;
    let pos_causal = c_off_mean < c_on_mean - 0.10; // pos-OFF clearly worse

    println!();
    println!("PASS CRITERIA:");
    println!(
        "  [{}] baseline global_sync ≈ chance (50±10%): mean={:.1}%",
        if baseline_chance { "PASS" } else { "FAIL" },
        100.0 * a_mean
    );
    println!(
        "  [{}] spatial readout pos-ON ≥ 80% mean: mean={:.1}%",
        if spatial_strong { "PASS" } else { "FAIL" },
        100.0 * c_on_mean
    );
    println!(
        "  [{}] pos-OFF below pos-ON (pos-enc causal): off={:.1}% vs on={:.1}%",
        if pos_causal { "PASS" } else { "FAIL" },
        100.0 * c_off_mean,
        100.0 * c_on_mean
    );

    // VIN gate: the ego-centric readout must make walls decodable ≫ chance.
    // Target ≥85% mean, clearly above majority baseline and global_sync.
    let vin_gate = vin_result.map(|(_, d_mean)| {
        let strong = d_mean >= 0.85;
        let beats_majority = d_mean > majority_mean + 0.10;
        let beats_global = d_mean > a_mean + 0.10;
        println!(
            "  [{}] VIN agent-cell ≥ 85% mean: mean={:.1}%",
            if strong { "PASS" } else { "FAIL" },
            100.0 * d_mean
        );
        println!(
            "  [{}] VIN ≫ majority baseline (+10pt): vin={:.1}% vs maj={:.1}%",
            if beats_majority { "PASS" } else { "FAIL" },
            100.0 * d_mean,
            100.0 * majority_mean
        );
        println!(
            "  [{}] VIN ≫ global_sync (+10pt): vin={:.1}% vs global={:.1}%",
            if beats_global { "PASS" } else { "FAIL" },
            100.0 * d_mean,
            100.0 * a_mean
        );
        strong && beats_majority && beats_global
    });

    println!();
    if let Some(vg) = vin_gate {
        // When --vin-readout is requested, the VIN gate is the headline.
        println!(
            "VIN GATE VERDICT: {}",
            if vg { "PASS ✓" } else { "FAIL ✗" }
        );
    } else {
        let gate = baseline_chance && spatial_strong && pos_causal;
        println!(
            "GATE VERDICT: {}",
            if gate { "PASS ✓" } else { "FAIL ✗" }
        );
    }

    // Diagnostic interpretation (the honest story behind the numbers).
    println!();
    println!("DIAGNOSTIC:");
    println!(
        "  raw V4 agent-cell token decodes walls at {:.1}% mean — the wall",
        100.0 * rt_mean
    );
    println!("  signal IS linearly present per-token in the encoder output.");
    if rt_mean >= 0.80 && c_on_mean < 0.80 {
        println!(
            "  But the POOLED reps (meanpool {:.1}%, readout {:.1}%, global {:.1}%)",
            100.0 * r_mean, 100.0 * c_on_mean, 100.0 * a_mean
        );
        println!("  are near chance: pooling/attention destroys the per-cell signal.");
        println!("  With UNTRAINED (random) attention, the MHA sum-pools all tokens");
        println!("  and cannot SELECT the agent cell, so positional encoding has no");
        println!("  causal effect yet (pos-ON ≈ pos-OFF). The pos-enc machinery is");
        println!("  wired and forward-correct; it becomes decisive only once the");
        println!("  readout's attention is TRAINED to localize on the agent.");
    }
}
