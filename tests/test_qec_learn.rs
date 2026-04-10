//! SPSA learning on sweet-spot QEC data.
//! Target: close the gap from 64% toward MWPM 87%.
//! cargo test --release --test test_qec_learn -- --nocapture --test-threads=1

use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, Linear, LayerConfig, forward_split, SimpleRng};
use modgrad::linalg;
use rayon::prelude::*;
use std::time::Instant;

fn parse(path: &str) -> Vec<(Vec<f32>, usize)> {
    std::fs::read_to_string(path).unwrap().lines()
        .filter_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).ok()?;
            let syn: Vec<f32> = v["syndrome"].as_array()?.iter()
                .filter_map(|x| Some(x.as_f64()? as f32)).collect();
            Some((syn, v["label"].as_u64()? as usize))
        }).collect()
}

fn eval(weights: &CtmWeights, train: &[(Vec<f32>, usize)], test: &[(Vec<f32>, usize)]) -> f32 {
    let proprio = vec![0.0f32; weights.config.d_input];
    let train_feats: Vec<(Vec<f32>, usize)> = train.par_iter().map(|(s, l)| {
        let mut ses = CtmSession::new(&weights.config);
        let mut t = weights.init_tick_state();
        let _ = forward_split(weights, &mut ses, &mut t, s, &proprio, false);
        (t.activations.clone(), *l)
    }).collect();

    let fd = train_feats[0].0.len();
    let mut xtx = vec![0.0f32; fd * fd];
    let mut xty = vec![0.0f32; fd * 2];
    for (f, l) in &train_feats {
        for r in 0..fd { for c in 0..fd { xtx[r*fd+c] += f[r]*f[c]; } xty[r*2+*l] += f[r]; }
    }

    let mut best_lam = 1.0f32;
    let mut best_v = 0.0f32;
    let vs = train_feats.len() * 4 / 5;
    for &lam in &[1e-4, 1e-2, 0.1, 1.0, 10.0] {
        let mut xr = xtx.clone();
        for i in 0..fd { xr[i*fd+i] += lam; }
        if let Some(l) = linalg::cholesky(&xr, fd) {
            let mut rd = Linear::new(fd, 2);
            for c in 0..2 {
                let rhs: Vec<f32> = (0..fd).map(|r| xty[r*2+c]).collect();
                let z = linalg::forward_solve(&l, &rhs, fd);
                let w = linalg::backward_solve(&l, &z, fd);
                for r in 0..fd { rd.weight[c*rd.in_dim+r] = w[r]; }
            }
            let ok: usize = train_feats[vs..].iter()
                .map(|(f,l)| if rd.forward(f).iter().enumerate()
                    .max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == *l {1} else {0}).sum();
            let a = ok as f32 / (train_feats.len() - vs) as f32;
            if a > best_v { best_v = a; best_lam = lam; }
        }
    }

    let mut xr = xtx;
    for i in 0..fd { xr[i*fd+i] += best_lam; }
    if let Some(l) = linalg::cholesky(&xr, fd) {
        let mut rd = Linear::new(fd, 2);
        for c in 0..2 {
            let rhs: Vec<f32> = (0..fd).map(|r| xty[r*2+c]).collect();
            let z = linalg::forward_solve(&l, &rhs, fd);
            let w = linalg::backward_solve(&l, &z, fd);
            for r in 0..fd { rd.weight[c*rd.in_dim+r] = w[r]; }
        }
        let ok: usize = test.par_iter().map(|(s, l)| {
            let mut ses = CtmSession::new(&weights.config);
            let mut t = weights.init_tick_state();
            let _ = forward_split(weights, &mut ses, &mut t, s, &proprio, false);
            if rd.forward(&t.activations).iter().enumerate()
                .max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == *l {1} else {0}
        }).sum();
        ok as f32 / test.len() as f32
    } else { 0.5 }
}

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
    let in_dim = w.len() / u.len().max(1);
    for j in 0..u.len() {
        for i in 0..in_dim.min(v.len()) {
            w[j * in_dim + i] += scale * u[j] * v[i];
        }
    }
}

fn quick_batch_acc(weights: &CtmWeights, batch: &[(Vec<f32>, usize)]) -> f32 {
    let proprio = vec![0.0f32; weights.config.d_input];
    let ok: usize = batch.par_iter().map(|(s, l)| {
        let mut ses = CtmSession::new(&weights.config);
        let mut t = weights.init_tick_state();
        let _ = forward_split(weights, &mut ses, &mut t, s, &proprio, false);
        // Use output region mean as binary classifier
        let out_start = t.act_offsets[2];
        let out_size = t.act_sizes[2];
        let sum: f32 = t.activations[out_start..out_start+out_size].iter().sum();
        if (if sum > 0.0 {1} else {0}) == *l {1} else {0}
    }).sum();
    ok as f32 / batch.len() as f32
}

#[test]
fn spsa_learn_qec() {
    let train_all = parse("data/qec/sweet_d5_p2_train.jsonl");
    let test = parse("data/qec/sweet_d5_p2_test.jsonl");

    // Use 5K subset for SPSA eval (fast), full 50K for final LS readout
    let spsa_train = &train_all[..5000];
    let ls_train = &train_all[..10000]; // 10K for LS (50K too slow for Cholesky on 544d)
    let test_sub = &test[..2000]; // 2K test for speed

    let cfg = CtmConfig {
        iterations: 8, d_input: 120, n_sync_out: 128,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        ..CtmConfig::default()
    };

    let ctm = Ctm::new(cfg.clone());
    let (mut weights, _) = ctm.into_split();

    let before = eval(&weights, ls_train, test_sub);
    eprintln!("\n  ╔═══════════════════════════════════════════════╗");
    eprintln!("  ║  SPSA Learning on QEC d=5 p=2%                ║");
    eprintln!("  ║  Target: MWPM 86.7%, Floor: majority 62.4%    ║");
    eprintln!("  ╚═══════════════════════════════════════════════╝");
    eprintln!("\n  BEFORE: {:.1}%\n", before * 100.0);

    let mut rng = SimpleRng::new(42);
    let sigma = 0.002;
    let lr = 0.02;
    let batch_size = 128;
    let mut best_acc = before;
    let t0 = Instant::now();

    for step in 0..200 {
        let batch_start = (step * batch_size) % (spsa_train.len() - batch_size);
        let batch = &spsa_train[batch_start..batch_start + batch_size];

        // Per-synapse SPSA
        for s in 0..8 {
            let syn_ref = &weights.synapse_refs()[s];
            let out_dim = syn_ref.linear.out_dim;
            let in_dim = syn_ref.linear.in_dim;

            let u: Vec<f32> = (0..out_dim).map(|_| if rng.next_f32() > 0.5 { 1.0 } else { -1.0 }).collect();
            let v: Vec<f32> = (0..in_dim).map(|_| if rng.next_f32() > 0.5 { 1.0 } else { -1.0 }).collect();

            apply_rank1(&mut weights, s, &u, &v, sigma);
            let acc_plus = quick_batch_acc(&weights, batch);
            apply_rank1(&mut weights, s, &u, &v, -2.0 * sigma);
            let acc_minus = quick_batch_acc(&weights, batch);
            apply_rank1(&mut weights, s, &u, &v, sigma); // revert

            let grad = (acc_plus - acc_minus) / (2.0 * sigma);
            if grad.abs() > 0.001 {
                apply_rank1(&mut weights, s, &u, &v, lr * grad);
            }
        }

        if step % 25 == 0 {
            let acc = eval(&weights, ls_train, test_sub);
            if acc > best_acc { best_acc = acc; }
            eprintln!("  Step {:3}: {:.1}% (best={:.1}%, {:.0}s)",
                step, acc * 100.0, best_acc * 100.0, t0.elapsed().as_secs_f64());
        }
    }

    let after = eval(&weights, ls_train, test_sub);
    eprintln!("\n  AFTER:  {:.1}% ({:+.1}pp, {:.0}s)", after * 100.0, (after - before) * 100.0, t0.elapsed().as_secs_f64());
    eprintln!("  MWPM:   86.7%");
    eprintln!("  Gap:    {:.1}pp\n", (0.867 - after) * 100.0);
}
