//! Learning experiments: QEC-localized SPSA + targeted Hebbian
//! cargo test --release --test test_learning -- --nocapture

use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, Linear, LayerConfig, forward_split, SimpleRng};
use modgrad::tasks;
use modgrad::ops::encode_label;
use modgrad_runtime::techniques::qec_learn::BrainGraph;
use modgrad::accuracy::{eval_ls, quick_eval};
use modgrad::linalg;
use rayon::prelude::*;

/// Apply rank-1 perturbation to synapse: W += scale * u * v^T
fn apply_rank1(weights: &mut CtmWeights, syn: usize, u: &[f32], v: &[f32], scale: f32) {
    let w = match syn {
        0 => &mut weights.syn_motor_input.linear.weight,
        1 => &mut weights.syn_input_attn.linear.weight,
        2 => &mut weights.syn_attn_output.linear.weight,
        3 => &mut weights.syn_output_motor.linear.weight,
        4 => &mut weights.syn_cerebellum.linear.weight,
        5 => &mut weights.syn_basal_ganglia.linear.weight,
        6 => &mut weights.syn_insula.linear.weight,
        _ => &mut weights.syn_hippocampus.linear.weight,
    };
    let in_dim = match syn {
        0 => weights.syn_motor_input.linear.in_dim,
        1 => weights.syn_input_attn.linear.in_dim,
        2 => weights.syn_attn_output.linear.in_dim,
        3 => weights.syn_output_motor.linear.in_dim,
        4 => weights.syn_cerebellum.linear.in_dim,
        5 => weights.syn_basal_ganglia.linear.in_dim,
        6 => weights.syn_insula.linear.in_dim,
        _ => weights.syn_hippocampus.linear.in_dim,
    };
    let out_dim = w.len() / in_dim;
    for j in 0..out_dim.min(u.len()) {
        for i in 0..in_dim.min(v.len()) {
            w[j * in_dim + i] += scale * u[j] * v[i];
        }
    }
}

#[test]
fn parallel_per_synapse_spsa() {
    // 64-bit parity: the original CTM benchmark. Should be genuinely hard.
    let cfg = CtmConfig {
        iterations: 8, d_input: 64, n_sync_out: 64,
        input_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        ..CtmConfig::default()
    };

    let data = tasks::parity_examples_large(64, cfg.d_input, 5000);
    let train = &data[..3000];
    let test = &data[3000..];

    let ctm = Ctm::new(cfg.clone());
    let (mut weights, _) = ctm.into_split();

    let before = eval_ls(&weights, test);
    eprintln!("\n  BEFORE (8-bit parity): {:.1}%", before * 100.0);

    let mut rng = SimpleRng::new(42);
    let sigma = 0.001;
    let lr = 0.01;
    let batch_size = 64;
    let mut best_acc = before;

    let t0 = std::time::Instant::now();

    for step in 0..100 {
        // Mini-batch
        let batch_start = (step * batch_size) % (train.len() - batch_size);
        let batch = &train[batch_start..batch_start + batch_size];

        // Per-synapse SPSA: perturb each synapse independently
        // In a multi-GPU setup, each synapse would be on a different GPU
        for s in 0..8 {
            let syn_ref = &weights.synapse_refs()[s];
            let in_dim = syn_ref.linear.in_dim;
            let out_dim = syn_ref.linear.out_dim;

            // Rank-1 noise: u (out_dim) × v (in_dim)
            let u: Vec<f32> = (0..out_dim).map(|_| if rng.next_f32() > 0.5 { 1.0 } else { -1.0 }).collect();
            let v: Vec<f32> = (0..in_dim).map(|_| if rng.next_f32() > 0.5 { 1.0 } else { -1.0 }).collect();

            // +perturbation
            apply_rank1(&mut weights, s, &u, &v, sigma);
            let acc_plus = quick_eval(&weights, batch);

            // -perturbation (go from +sigma to -sigma)
            apply_rank1(&mut weights, s, &u, &v, -2.0 * sigma);
            let acc_minus = quick_eval(&weights, batch);

            // revert to original
            apply_rank1(&mut weights, s, &u, &v, sigma);

            // Update in the improving direction
            let grad = (acc_plus - acc_minus) / (2.0 * sigma);
            if grad.abs() > 0.001 {
                apply_rank1(&mut weights, s, &u, &v, lr * grad);
            }
        }

        if step % 10 == 0 {
            let acc = eval_ls(&weights, test);
            let elapsed = t0.elapsed().as_secs_f64();
            if acc > best_acc { best_acc = acc; }
            eprintln!("  Step {:3}: {:.1}% (best={:.1}%, {:.1}s, {:.0} steps/s)",
                step, acc * 100.0, best_acc * 100.0, elapsed, step as f64 / elapsed);
        }
    }

    let after = eval_ls(&weights, test);
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("  AFTER:  {:.1}% ({:+.1}pp, {:.1}s total, {:.0} steps/s)",
        after * 100.0, (after - before) * 100.0, elapsed, 100.0 / elapsed);
    eprintln!();
}

#[test]
fn qec_localized_spsa() {
    // QEC-localized SPSA: only perturb synapses identified by error correction
    let cfg = CtmConfig {
        iterations: 4, d_input: 8, n_sync_out: 32,
        input_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        ..CtmConfig::default()
    };

    let data = tasks::parity_examples(8, cfg.d_input);
    let train = &data[..2000];
    let test = &data[2000..3000];

    let ctm = Ctm::new(cfg.clone());
    let (mut weights, _) = ctm.into_split();
    let graph = BrainGraph::new();

    let before = eval_ls(&weights, test);
    eprintln!("\n  BEFORE (8-bit parity, QEC-localized SPSA): {:.1}%", before * 100.0);

    let mut rng = SimpleRng::new(42);
    let sigma = 0.001;
    let lr = 0.01;
    let batch_size = 64;
    let mut best_acc = before;
    // Running baseline for per-region anomaly
    let mut baseline_means = [0.0f32; 8];

    let t0 = std::time::Instant::now();

    for step in 0..100 {
        let batch_start = (step * batch_size) % (train.len() - batch_size);
        let batch = &train[batch_start..batch_start + batch_size];
        let labels: Vec<Vec<f32>> = (0..2).map(|c| encode_label(c, cfg.d_input)).collect();

        // Run forward on batch, compute per-region anomaly
        let mut total_anomaly = [0.0f32; 8];
        for ex in batch {
            let mut ses = CtmSession::new(&weights.config);
            let mut ts = weights.init_tick_state();
            let _ = forward_split(&weights, &mut ses, &mut ts, &ex.input, &labels[ex.target], false);
            let anomaly = BrainGraph::compute_anomaly(
                &ts.activations, &ts.act_offsets, &ts.act_sizes, &baseline_means);
            for r in 0..8 { total_anomaly[r] += anomaly[r]; }
            // Update baseline
            for r in 0..8 {
                let sz = ts.act_sizes[r].max(1) as f32;
                let mean: f32 = ts.activations[ts.act_offsets[r]..ts.act_offsets[r]+ts.act_sizes[r]]
                    .iter().sum::<f32>() / sz;
                baseline_means[r] = 0.99 * baseline_means[r] + 0.01 * mean;
            }
        }
        for r in 0..8 { total_anomaly[r] /= batch_size as f32; }

        // QEC: localize which synapses to update
        let localization = graph.localize_error(&total_anomaly);

        // Per-synapse SPSA, weighted by QEC localization
        for s in 0..8 {
            if localization[s] < 0.5 { continue; } // skip unmatched synapses

            let syn_ref = &weights.synapse_refs()[s];
            let in_dim = syn_ref.linear.in_dim;
            let out_dim = syn_ref.linear.out_dim;

            let u: Vec<f32> = (0..out_dim).map(|_| if rng.next_f32() > 0.5 { 1.0 } else { -1.0 }).collect();
            let v: Vec<f32> = (0..in_dim).map(|_| if rng.next_f32() > 0.5 { 1.0 } else { -1.0 }).collect();

            apply_rank1(&mut weights, s, &u, &v, sigma);
            let acc_plus = quick_eval(&weights, batch);

            apply_rank1(&mut weights, s, &u, &v, -2.0 * sigma);
            let acc_minus = quick_eval(&weights, batch);

            apply_rank1(&mut weights, s, &u, &v, sigma); // revert

            let grad = (acc_plus - acc_minus) / (2.0 * sigma);
            if grad.abs() > 0.001 {
                apply_rank1(&mut weights, s, &u, &v, lr * grad * localization[s]);
            }
        }

        if step % 10 == 0 {
            let acc = eval_ls(&weights, test);
            if acc > best_acc { best_acc = acc; }
            let matched: Vec<usize> = localization.iter().enumerate()
                .filter(|(_, w)| **w > 0.5).map(|(i, _)| i).collect();
            eprintln!("  Step {:3}: {:.1}% (best={:.1}%, matched={:?}, {:.1}s)",
                step, acc * 100.0, best_acc * 100.0, matched, t0.elapsed().as_secs_f64());
        }
    }

    let after = eval_ls(&weights, test);
    eprintln!("  AFTER:  {:.1}% ({:+.1}pp, {:.1}s)",
        after * 100.0, (after - before) * 100.0, t0.elapsed().as_secs_f64());
    eprintln!();
}

#[test]
fn qec_localized_maze() {
    let cfg = CtmConfig {
        iterations: 4, d_input: 32, n_sync_out: 32,
        input_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 64, ..Default::default() },
        ..CtmConfig::default()
    };

    let data = tasks::maze_examples(5, cfg.d_input, 2000);
    let train = &data[..1500];
    let test = &data[1500..];

    let ctm = Ctm::new(cfg.clone());
    let (mut weights, _) = ctm.into_split();
    let graph = BrainGraph::new();

    let before = eval_ls(&weights, test);
    eprintln!("\n  === 5x5 MAZE (4 classes) ===");
    eprintln!("  BEFORE: {:.1}%  (random=25%)", before * 100.0);

    let mut rng = SimpleRng::new(99);
    let sigma = 0.001;
    let lr = 0.01;
    let batch_size = 64;
    let mut best_acc = before;
    let mut baseline_means = [0.0f32; 8];
    let labels: Vec<Vec<f32>> = (0..4).map(|c| encode_label(c, cfg.d_input)).collect();
    let t0 = std::time::Instant::now();

    for step in 0..200 {
        let batch_start = (step * batch_size) % (train.len() - batch_size);
        let batch = &train[batch_start..batch_start + batch_size];

        let mut total_anomaly = [0.0f32; 8];
        for ex in batch {
            let mut ses = CtmSession::new(&weights.config);
            let mut ts = weights.init_tick_state();
            let _ = forward_split(&weights, &mut ses, &mut ts, &ex.input, &labels[ex.target], false);
            let anomaly = BrainGraph::compute_anomaly(
                &ts.activations, &ts.act_offsets, &ts.act_sizes, &baseline_means);
            for r in 0..8 { total_anomaly[r] += anomaly[r]; }
            for r in 0..8 {
                let sz = ts.act_sizes[r].max(1) as f32;
                let mean: f32 = ts.activations[ts.act_offsets[r]..ts.act_offsets[r]+ts.act_sizes[r]]
                    .iter().sum::<f32>() / sz;
                baseline_means[r] = 0.99 * baseline_means[r] + 0.01 * mean;
            }
        }
        for r in 0..8 { total_anomaly[r] /= batch_size as f32; }

        let localization = graph.localize_error(&total_anomaly);

        for s in 0..8 {
            if localization[s] < 0.5 { continue; }
            let syn_ref = &weights.synapse_refs()[s];
            let in_dim = syn_ref.linear.in_dim;
            let out_dim = syn_ref.linear.out_dim;
            let u: Vec<f32> = (0..out_dim).map(|_| if rng.next_f32() > 0.5 { 1.0 } else { -1.0 }).collect();
            let v: Vec<f32> = (0..in_dim).map(|_| if rng.next_f32() > 0.5 { 1.0 } else { -1.0 }).collect();
            apply_rank1(&mut weights, s, &u, &v, sigma);
            let acc_plus = quick_eval(&weights, batch);
            apply_rank1(&mut weights, s, &u, &v, -2.0 * sigma);
            let acc_minus = quick_eval(&weights, batch);
            apply_rank1(&mut weights, s, &u, &v, sigma);
            let grad = (acc_plus - acc_minus) / (2.0 * sigma);
            if grad.abs() > 0.001 {
                apply_rank1(&mut weights, s, &u, &v, lr * grad * localization[s]);
            }
        }

        if step % 20 == 0 {
            let acc = eval_ls(&weights, test);
            if acc > best_acc { best_acc = acc; }
            let matched: Vec<usize> = localization.iter().enumerate()
                .filter(|(_, w)| **w > 0.5).map(|(i, _)| i).collect();
            eprintln!("  Step {:3}: {:.1}% (best={:.1}%, matched={:?}, {:.1}s)",
                step, acc * 100.0, best_acc * 100.0, matched, t0.elapsed().as_secs_f64());
        }
    }

    let after = eval_ls(&weights, test);
    eprintln!("  AFTER: {:.1}% ({:+.1}pp, {:.1}s)\n",
        after * 100.0, (after - before) * 100.0, t0.elapsed().as_secs_f64());
}

#[test]
fn qec_localized_qec_task() {
    // The actual QEC task — 72% baseline, 93.7% MWPM target
    let parse = |path: &str| -> Vec<tasks::Example> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let input: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                let target = v["label"].as_u64()? as usize;
                Some(tasks::Example { input, target, n_classes: 2 })
            }).collect()
    };

    let train_all = parse("data/qec/surface_train.jsonl");
    let test_all = parse("data/qec/surface_test.jsonl");
    let train = &train_all[..3000];
    let test = &test_all[..1000];
    let syn_dim = train[0].input.len();

    let cfg = CtmConfig {
        iterations: 8, d_input: syn_dim, n_sync_out: 128,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        ..CtmConfig::default()
    };

    let ctm = Ctm::new(cfg.clone());
    let (mut weights, _) = ctm.into_split();
    let graph = BrainGraph::new();

    let before = eval_ls(&weights, test);
    eprintln!("\n  === QEC SYNDROME DECODING ===");
    eprintln!("  BEFORE: {:.1}%  (MWPM=93.7%, PyTorch CTM=91.6%)", before * 100.0);

    let mut rng = SimpleRng::new(42);
    let sigma = 0.001;
    let lr = 0.01;
    let batch_size = 64;
    let mut best_acc = before;
    let mut baseline_means = [0.0f32; 8];
    let labels: Vec<Vec<f32>> = (0..2).map(|c| encode_label(c, syn_dim)).collect();
    let t0 = std::time::Instant::now();

    for step in 0..200 {
        let batch_start = (step * batch_size) % (train.len() - batch_size);
        let batch = &train[batch_start..batch_start + batch_size];

        // Compute anomaly
        let mut total_anomaly = [0.0f32; 8];
        for ex in batch {
            let mut ses = CtmSession::new(&weights.config);
            let mut ts = weights.init_tick_state();
            let _ = forward_split(&weights, &mut ses, &mut ts, &ex.input, &labels[ex.target], false);
            let anomaly = BrainGraph::compute_anomaly(
                &ts.activations, &ts.act_offsets, &ts.act_sizes, &baseline_means);
            for r in 0..8 { total_anomaly[r] += anomaly[r]; }
            for r in 0..8 {
                let sz = ts.act_sizes[r].max(1) as f32;
                let mean: f32 = ts.activations[ts.act_offsets[r]..ts.act_offsets[r]+ts.act_sizes[r]]
                    .iter().sum::<f32>() / sz;
                baseline_means[r] = 0.99 * baseline_means[r] + 0.01 * mean;
            }
        }
        for r in 0..8 { total_anomaly[r] /= batch_size as f32; }

        let localization = graph.localize_error(&total_anomaly);

        // Per-synapse SPSA weighted by QEC localization
        for s in 0..8 {
            if localization[s] < 0.5 { continue; }
            let syn_ref = &weights.synapse_refs()[s];
            let in_dim = syn_ref.linear.in_dim;
            let out_dim = syn_ref.linear.out_dim;
            let u: Vec<f32> = (0..out_dim).map(|_| if rng.next_f32() > 0.5 { 1.0 } else { -1.0 }).collect();
            let v: Vec<f32> = (0..in_dim).map(|_| if rng.next_f32() > 0.5 { 1.0 } else { -1.0 }).collect();
            apply_rank1(&mut weights, s, &u, &v, sigma);
            let acc_plus = quick_eval(&weights, batch);
            apply_rank1(&mut weights, s, &u, &v, -2.0 * sigma);
            let acc_minus = quick_eval(&weights, batch);
            apply_rank1(&mut weights, s, &u, &v, sigma);
            let grad = (acc_plus - acc_minus) / (2.0 * sigma);
            if grad.abs() > 0.001 {
                apply_rank1(&mut weights, s, &u, &v, lr * grad * localization[s]);
            }
        }

        if step % 20 == 0 {
            let acc = eval_ls(&weights, test);
            if acc > best_acc { best_acc = acc; }
            let matched: Vec<usize> = localization.iter().enumerate()
                .filter(|(_, w)| **w > 0.5).map(|(i, _)| i).collect();
            eprintln!("  Step {:3}: {:.1}% (best={:.1}%, matched={:?}, {:.1}s)",
                step, acc * 100.0, best_acc * 100.0, matched, t0.elapsed().as_secs_f64());
        }
    }

    let after = eval_ls(&weights, test);
    eprintln!("  AFTER: {:.1}% ({:+.1}pp, {:.1}s)", after * 100.0, (after - before) * 100.0, t0.elapsed().as_secs_f64());
    eprintln!("  Target: MWPM=93.7%, PyTorch=91.6%\n");
}

#[test]
fn isis_on_paems_surface_d5() {
    let parse = |path: &str| -> Vec<tasks::Example> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let input: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                let target = v["label"].as_u64()? as usize;
                Some(tasks::Example { input, target, n_classes: 2 })
            }).collect()
    };

    let train = parse("data/qec/surface_d5_depol_train.jsonl");
    let test = parse("data/qec/surface_d5_depol_test.jsonl");
    let d_input = train[0].input.len();

    eprintln!("\n  === PAEMS Surface Code d=5 (real hardware noise) ===");
    eprintln!("  {} train, {} test, {} detectors", train.len(), test.len(), d_input);

    let cfg = CtmConfig {
        iterations: 8, d_input, n_sync_out: 128,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        ..CtmConfig::default()
    };

    let ctm = Ctm::new(cfg.clone());
    let (weights, _) = ctm.into_split();

    let acc = eval_ls(&weights, &test);
    eprintln!("  isis CTM activations readout: {:.1}%", acc * 100.0);
    eprintln!("  sklearn Ridge:                94.4%");
    eprintln!("  sklearn MLP(128):             97.8%");
    eprintln!("  majority vote:                95.2%");
    eprintln!();
}

#[test]
fn isis_on_paems_surface_d7() {
    let parse = |path: &str| -> Vec<tasks::Example> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let input: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                let target = v["label"].as_u64()? as usize;
                Some(tasks::Example { input, target, n_classes: 2 })
            }).collect()
    };

    let train = parse("data/qec/surface_d7_depol_train.jsonl");
    let test = parse("data/qec/surface_d7_depol_test.jsonl");
    let d_input = train[0].input.len();

    eprintln!("\n  === PAEMS Surface Code d=7 (harder) ===");
    eprintln!("  {} train, {} test, {} detectors", train.len(), test.len(), d_input);

    let cfg = CtmConfig {
        iterations: 8, d_input, n_sync_out: 128,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        ..CtmConfig::default()
    };

    let ctm = Ctm::new(cfg.clone());
    let (weights, _) = ctm.into_split();

    let acc = eval_ls(&weights, &test);
    eprintln!("  isis CTM activations readout: {:.1}%", acc * 100.0);
    eprintln!("  sklearn Ridge:                87.2%");
    eprintln!("  sklearn MLP(128):             90.7%");
    eprintln!("  majority vote:                ~89.6%");
    eprintln!();
}

#[test]
fn mwpm_baseline_paems() {
    // Run fusion-blossom MWPM on the PAEMS surface code data
    // Need to build the decoding graph from the DEM file
    eprintln!("\n  === MWPM Baseline (fusion-blossom) ===");
    eprintln!("  Need DEM → graph conversion. Using stim Python for now.");
    eprintln!("  pymatching segfaults — known issue with Python 3.14");
    eprintln!("  TODO: build DEM parser in Rust, call fusion_blossom::fusion_mwpm directly");
    eprintln!();
}

#[test]
fn mwpm_vs_isis_paems() {
    use fusion_blossom::util::{SolverInitializer, SyndromePattern, VertexIndex, Weight};
    use fusion_blossom::mwpm_solver;

    eprintln!("\n  === MWPM vs isis on PAEMS Surface Codes ===\n");

    for (name, det_file, graph_file) in [
        ("d5", "data/qec/surface_d5_depol_test.jsonl", "data/qec/surface_d5_depol_graph.json"),
        ("d7", "data/qec/surface_d7_depol_test.jsonl", "data/qec/surface_d7_depol_graph.json"),
    ] {
        // Load graph
        let graph_str = std::fs::read_to_string(graph_file).unwrap();
        let graph: serde_json::Value = serde_json::from_str(&graph_str).unwrap();
        let n_det = graph["n_det"].as_u64().unwrap() as usize;

        // Build fusion-blossom solver
        let mut weighted_edges = Vec::new();
        let mut edge_obs: Vec<Vec<usize>> = Vec::new(); // which observable each edge flips

        for e in graph["edges"].as_array().unwrap() {
            let d0 = e["d0"].as_u64().unwrap() as VertexIndex;
            let d1 = e["d1"].as_u64().unwrap() as VertexIndex;
            let p: f64 = e["p"].as_f64().unwrap();
            let w = ((-1000.0 * (p / (1.0 - p)).ln()).abs() as Weight).max(2);
            let w_even = (w / 2) * 2;
            weighted_edges.push((d0, d1, w_even));
            let obs: Vec<usize> = e["obs"].as_array().unwrap().iter()
                .map(|v| v.as_u64().unwrap() as usize).collect();
            edge_obs.push(obs);
        }

        // Virtual vertex for boundary edges
        let virtual_v = n_det as VertexIndex;
        for be in graph["boundary"].as_array().unwrap() {
            let d = be["d"].as_u64().unwrap() as VertexIndex;
            let p: f64 = be["p"].as_f64().unwrap();
            let w = ((-1000.0 * (p / (1.0 - p)).ln()).abs() as Weight).max(2);
            let w_even = (w / 2) * 2;
            weighted_edges.push((d, virtual_v, w_even));
            let obs: Vec<usize> = be["obs"].as_array().unwrap().iter()
                .map(|v| v.as_u64().unwrap() as usize).collect();
            edge_obs.push(obs);
        }

        let init = SolverInitializer::new(
            (n_det + 1) as VertexIndex,
            weighted_edges,
            vec![virtual_v],
        );

        // Load test data
        let test_str = std::fs::read_to_string(det_file).unwrap();
        let test_data: Vec<(Vec<f32>, usize)> = test_str.lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect();

        // MWPM decode using fusion_mwpm (top-level function)
        let mut mwpm_correct = 0usize;
        for (syn, label) in &test_data {
            let defects: Vec<VertexIndex> = syn.iter().enumerate()
                .filter(|(_, v)| **v > 0.5)
                .map(|(i, _)| i as VertexIndex)
                .collect();
            let syndrome = SyndromePattern::new_vertices(defects);
            // fusion_mwpm returns matched vertex pairs
            let matched = fusion_blossom::fusion_mwpm(&init, &syndrome);

            // Prediction: find which edges were matched, XOR their observable flips
            let mut obs_flip = 0usize;
            // matched is a flat list of vertex pairs
            for pair in matched.chunks(2) {
                if pair.len() == 2 {
                    let v0 = pair[0] as usize;
                    let v1 = pair[1] as usize;
                    // Find edge index
                    for (ei, &(a, b, _)) in init.weighted_edges.iter().enumerate() {
                        if (a as usize == v0 && b as usize == v1) || (a as usize == v1 && b as usize == v0) {
                            if ei < edge_obs.len() {
                                for &obs_id in &edge_obs[ei] {
                                    if obs_id == 0 { obs_flip ^= 1; }
                                }
                            }
                            break;
                        }
                    }
                }
            }
            if obs_flip == *label { mwpm_correct += 1; }
        }
        let mwpm_acc = mwpm_correct as f32 / test_data.len() as f32;
        eprintln!("  {}: MWPM (fusion-blossom) = {:.1}%", name, mwpm_acc * 100.0);
    }
    eprintln!();
}
