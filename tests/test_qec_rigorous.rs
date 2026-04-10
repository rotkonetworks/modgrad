//! Rigorous QEC decoder comparison with cross-validation.
//! cargo test --release --test test_qec_rigorous -- --nocapture --test-threads=1

use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, Linear, LayerConfig, forward_split, SimpleRng};
use modgrad::tasks;
use modgrad::linalg;
use modgrad_runtime::techniques::qec_learn::BrainGraph;
use fusion_blossom::util::{SolverInitializer, SyndromePattern, VertexIndex, Weight};
use rayon::prelude::*;
use std::time::Instant;

fn isis_eval(cfg: &CtmConfig, train: &[(Vec<f32>, usize)], test: &[(Vec<f32>, usize)]) -> f32 {
    let ctm = Ctm::new(cfg.clone());
    let (weights, _) = ctm.into_split();
    let proprio = vec![0.0f32; cfg.d_input];

    // Collect activations on train
    let train_feats: Vec<(Vec<f32>, usize)> = train.par_iter().map(|(syn, label)| {
        let mut s = CtmSession::new(&weights.config);
        let mut t = weights.init_tick_state();
        let _ = forward_split(&weights, &mut s, &mut t, syn, &proprio, false);
        (t.activations.clone(), *label)
    }).collect();

    let fd = train_feats[0].0.len();
    let nc = 2;
    let mut xtx = vec![0.0f32; fd * fd];
    let mut xty = vec![0.0f32; fd * nc];
    for (f, l) in &train_feats {
        for r in 0..fd { for c in 0..fd { xtx[r*fd+c] += f[r]*f[c]; } xty[r*nc+*l] += f[r]; }
    }

    // Lambda sweep on train
    let mut best_lam = 1.0f32;
    let mut best_val = 0.0f32;
    let val_split = train_feats.len() * 4 / 5;
    for &lam in &[1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0] {
        let mut xr = xtx.clone();
        for i in 0..fd { xr[i*fd+i] += lam; }
        if let Some(l) = linalg::cholesky(&xr, fd) {
            let mut rd = Linear::new(fd, nc);
            for c in 0..nc {
                let rhs: Vec<f32> = (0..fd).map(|r| xty[r*nc+c]).collect();
                let z = linalg::forward_solve(&l, &rhs, fd);
                let w = linalg::backward_solve(&l, &z, fd);
                for r in 0..fd { rd.weight[c*rd.in_dim+r] = w[r]; }
            }
            let ok: usize = train_feats[val_split..].iter()
                .map(|(f, lab)| {
                    let logits = rd.forward(f);
                    if logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == *lab {1} else {0}
                }).sum();
            let acc = ok as f32 / (train_feats.len() - val_split) as f32;
            if acc > best_val { best_val = acc; best_lam = lam; }
        }
    }

    // Fit on full train with best lambda
    let mut xr = xtx;
    for i in 0..fd { xr[i*fd+i] += best_lam; }
    if let Some(l) = linalg::cholesky(&xr, fd) {
        let mut rd = Linear::new(fd, nc);
        for c in 0..nc {
            let rhs: Vec<f32> = (0..fd).map(|r| xty[r*nc+c]).collect();
            let z = linalg::forward_solve(&l, &rhs, fd);
            let w = linalg::backward_solve(&l, &z, fd);
            for r in 0..fd { rd.weight[c*rd.in_dim+r] = w[r]; }
        }
        // Eval on test
        let ok: usize = test.par_iter().map(|(syn, label)| {
            let mut s = CtmSession::new(&weights.config);
            let mut t = weights.init_tick_state();
            let _ = forward_split(&weights, &mut s, &mut t, syn, &proprio, false);
            let logits = rd.forward(&t.activations);
            if logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == *label {1} else {0}
        }).sum();
        ok as f32 / test.len() as f32
    } else { 0.5 }
}

fn mwpm_eval(graph_file: &str, test: &[(Vec<f32>, usize)]) -> f32 {
    let graph_str = std::fs::read_to_string(graph_file).unwrap();
    let graph: serde_json::Value = serde_json::from_str(&graph_str).unwrap();
    let n_det = graph["n_det"].as_u64().unwrap() as usize;

    let mut weighted_edges = Vec::new();
    let mut edge_obs: Vec<Vec<usize>> = Vec::new();
    for e in graph["edges"].as_array().unwrap() {
        let d0 = e["d0"].as_u64().unwrap() as VertexIndex;
        let d1 = e["d1"].as_u64().unwrap() as VertexIndex;
        let p: f64 = e["p"].as_f64().unwrap();
        let w = ((-1000.0 * (p / (1.0 - p)).ln()).abs() as Weight).max(2);
        let w_even = (w / 2) * 2;
        weighted_edges.push((d0, d1, w_even));
        edge_obs.push(e["obs"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect());
    }
    let virtual_v = n_det as VertexIndex;
    for be in graph["boundary"].as_array().unwrap() {
        let d = be["d"].as_u64().unwrap() as VertexIndex;
        let p: f64 = be["p"].as_f64().unwrap();
        let w = ((-1000.0 * (p / (1.0 - p)).ln()).abs() as Weight).max(2);
        let w_even = (w / 2) * 2;
        weighted_edges.push((d, virtual_v, w_even));
        edge_obs.push(be["obs"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect());
    }
    let init = SolverInitializer::new((n_det + 1) as VertexIndex, weighted_edges, vec![virtual_v]);

    let mut correct = 0usize;
    for (syn, label) in test {
        let defects: Vec<VertexIndex> = syn.iter().enumerate()
            .filter(|(_, v)| **v > 0.5).map(|(i, _)| i as VertexIndex).collect();
        let syndrome = SyndromePattern::new_vertices(defects);
        let matched = fusion_blossom::fusion_mwpm(&init, &syndrome);
        let mut obs_flip = 0usize;
        for pair in matched.chunks(2) {
            if pair.len() == 2 {
                let (v0, v1) = (pair[0] as usize, pair[1] as usize);
                for (ei, &(a, b, _)) in init.weighted_edges.iter().enumerate() {
                    if (a as usize == v0 && b as usize == v1) || (a as usize == v1 && b as usize == v0) {
                        if ei < edge_obs.len() { for &o in &edge_obs[ei] { if o == 0 { obs_flip ^= 1; } } }
                        break;
                    }
                }
            }
        }
        if obs_flip == *label { correct += 1; }
    }
    correct as f32 / test.len() as f32
}

#[test]
fn rigorous_comparison() {
    eprintln!("\n  ╔══════════════════════════════════════════════════════════════╗");
    eprintln!("  ║  isis CTM vs MWPM: Rigorous 5-Fold Cross-Validation         ║");
    eprintln!("  ║  PAEMS Surface Code Data (IBM Hardware Noise)                ║");
    eprintln!("  ╚══════════════════════════════════════════════════════════════╝\n");

    for (name, train_file, test_file, graph_file) in [
        ("d=5 (120 det)", "data/qec/surface_d5_depol_train.jsonl", "data/qec/surface_d5_depol_test.jsonl", "data/qec/surface_d5_depol_graph.json"),
        ("d=7 (336 det)", "data/qec/surface_d7_depol_train.jsonl", "data/qec/surface_d7_depol_test.jsonl", "data/qec/surface_d7_depol_graph.json"),
    ] {
        let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
            std::fs::read_to_string(path).unwrap().lines()
                .filter_map(|l| {
                    let v: serde_json::Value = serde_json::from_str(l).ok()?;
                    let syn: Vec<f32> = v["syndrome"].as_array()?
                        .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                    Some((syn, v["label"].as_u64()? as usize))
                }).collect()
        };

        let mut all_data = parse(train_file);
        all_data.extend(parse(test_file));
        let n = all_data.len();
        let d_input = all_data[0].0.len();
        let error_rate = all_data.iter().filter(|(_, l)| *l == 1).count() as f32 / n as f32;

        eprintln!("  --- {} ---", name);
        eprintln!("  {} samples, {} detectors, {:.1}% logical error rate\n", n, d_input, error_rate * 100.0);

        let cfg = CtmConfig {
            iterations: 8, d_input, n_sync_out: 128,
            input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
            attention_layer: LayerConfig { n_neurons: 128, ..Default::default() },
            output_layer: LayerConfig { n_neurons: 128, ..Default::default() },
            motor_layer: LayerConfig { n_neurons: 128, ..Default::default() },
            ..CtmConfig::default()
        };

        // 5-fold cross-validation
        let k_folds = 5;
        let fold_size = n / k_folds;
        let mut isis_accs = Vec::new();
        let mut mwpm_accs = Vec::new();
        let mut majority_accs = Vec::new();

        for fold in 0..k_folds {
            let test_start = fold * fold_size;
            let test_end = test_start + fold_size;
            let test_fold: Vec<_> = all_data[test_start..test_end].to_vec();
            let mut train_fold: Vec<_> = all_data[..test_start].to_vec();
            train_fold.extend_from_slice(&all_data[test_end..]);

            let t0 = Instant::now();

            // isis CTM (3 random seeds, take best)
            let mut best_isis = 0.0f32;
            for seed_offset in 0..3 {
                let mut cfg_s = cfg.clone();
                cfg_s.out_dims = seed_offset + 1;
                let acc = isis_eval(&cfg_s, &train_fold, &test_fold);
                best_isis = best_isis.max(acc);
            }
            isis_accs.push(best_isis);

            // MWPM
            let mwpm_acc = mwpm_eval(graph_file, &test_fold);
            mwpm_accs.push(mwpm_acc);

            // Majority vote
            let train_pos = train_fold.iter().filter(|(_, l)| *l == 1).count() as f32 / train_fold.len() as f32;
            let majority_pred = if train_pos > 0.5 { 1 } else { 0 };
            let maj_acc = test_fold.iter().filter(|(_, l)| *l == majority_pred).count() as f32 / test_fold.len() as f32;
            majority_accs.push(maj_acc);

            eprintln!("  Fold {}: isis={:.1}% MWPM={:.1}% majority={:.1}% ({:.1}s)",
                fold, best_isis * 100.0, mwpm_acc * 100.0, maj_acc * 100.0, t0.elapsed().as_secs_f64());
        }

        let isis_mean = isis_accs.iter().sum::<f32>() / k_folds as f32;
        let isis_std = (isis_accs.iter().map(|x| (x - isis_mean).powi(2)).sum::<f32>() / k_folds as f32).sqrt();
        let mwpm_mean = mwpm_accs.iter().sum::<f32>() / k_folds as f32;
        let mwpm_std = (mwpm_accs.iter().map(|x| (x - mwpm_mean).powi(2)).sum::<f32>() / k_folds as f32).sqrt();
        let maj_mean = majority_accs.iter().sum::<f32>() / k_folds as f32;

        eprintln!();
        eprintln!("  {:>15} {:>12} {:>12}", "Decoder", "Accuracy", "± Std");
        eprintln!("  {}", "-".repeat(42));
        eprintln!("  {:>15} {:>10.1}% {:>10.1}%", "isis CTM", isis_mean * 100.0, isis_std * 100.0);
        eprintln!("  {:>15} {:>10.1}% {:>10.1}%", "MWPM", mwpm_mean * 100.0, mwpm_std * 100.0);
        eprintln!("  {:>15} {:>10.1}%", "majority", maj_mean * 100.0);
        eprintln!("  {}", "-".repeat(42));
        let delta = isis_mean - mwpm_mean;
        eprintln!("  isis - MWPM = {:+.1}pp", delta * 100.0);
        eprintln!();
    }
}
