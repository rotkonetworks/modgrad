//! EBT: Energy-Based Transformer training on QEC.
//! cargo test --release --test transformer_train -- --nocapture

use modgrad_compute::tensor::*;

#[test]
fn ebt_qec() {
    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let train = parse("data/qec/surface_train.jsonl");
    let test = parse("data/qec/surface_test.jsonl");
    let d_in = train[0].0.len();
    let n_cls = 2;

    eprintln!("\n  EBT on QEC (d_in={}, {} train, {} test)", d_in, train.len(), test.len());

    let mut model = MiniCtm::new(d_in, 128, 4, n_cls);
    let mut opt = Adam::new(model.param_count(), 0.001);

    // Phase 1: Standard backprop (train syn + nlm + out + energy together)
    eprintln!("  Phase 1: Standard training...");
    for epoch in 0..10 {
        for chunk in train.chunks(64) {
            let x = Tensor {
                data: chunk.iter().flat_map(|(s, _)| s.iter().copied()).collect(),
                rows: chunk.len(), cols: d_in,
            };
            let targets: Vec<usize> = chunk.iter().map(|(_, l)| *l).collect();
            let (_, grads) = model.forward_backward(&x, &targets);
            model.apply_grads(&grads, &mut opt);
        }
        if epoch % 3 == 0 {
            let acc = eval_acc(&model, &test, d_in, n_cls);
            eprintln!("    epoch {:2}: {:.1}%", epoch, acc * 100.0);
        }
    }

    let std_acc = eval_acc(&model, &test, d_in, n_cls);
    eprintln!("    Standard final: {:.1}%\n", std_acc * 100.0);

    // Phase 2: Train energy head (contrastive — low for correct, high for wrong)
    eprintln!("  Phase 2: Energy head training...");
    let mut e_opt = Adam::new(model.energy_w.len() + model.energy_b.len(), 0.01);

    for epoch in 0..10 {
        for chunk in train.chunks(64) {
            let x = Tensor {
                data: chunk.iter().flat_map(|(s, _)| s.iter().copied()).collect(),
                rows: chunk.len(), cols: d_in,
            };
            let targets: Vec<usize> = chunk.iter().map(|(_, l)| *l).collect();
            let batch = chunk.len();
            let h = model.tick_loop(&x);

            // True and fake one-hot predictions
            let mut y_true = Tensor::zeros(batch, n_cls);
            let mut y_fake = Tensor::zeros(batch, n_cls);
            for (b, &t) in targets.iter().enumerate() {
                y_true.data[b * n_cls + t] = 1.0;
                y_fake.data[b * n_cls + (1 - t)] = 1.0;
            }

            // Finite-difference gradient on energy params
            let eps = 0.001;
            let margin = 1.0;

            let base_loss = |m: &MiniCtm| -> f32 {
                let et = m.energy(&h, &y_true);
                let ef = m.energy(&h, &y_fake);
                (0..batch).map(|b| (et[b] - ef[b] + margin).max(0.0)).sum::<f32>() / batch as f32
            };

            let mut grads_w = vec![0.0f32; model.energy_w.data.len()];
            for i in 0..grads_w.len() {
                let orig = model.energy_w.data[i];
                model.energy_w.data[i] = orig + eps;
                let lp = base_loss(&model);
                model.energy_w.data[i] = orig - eps;
                let lm = base_loss(&model);
                model.energy_w.data[i] = orig;
                grads_w[i] = (lp - lm) / (2.0 * eps);
            }
            let mut grads_b = vec![0.0f32; model.energy_b.len()];
            for i in 0..grads_b.len() {
                let orig = model.energy_b[i];
                model.energy_b[i] = orig + eps;
                let lp = base_loss(&model);
                model.energy_b[i] = orig - eps;
                let lm = base_loss(&model);
                model.energy_b[i] = orig;
                grads_b[i] = (lp - lm) / (2.0 * eps);
            }

            let mut params: Vec<f32> = model.energy_w.data.clone();
            params.extend_from_slice(&model.energy_b);
            let mut g: Vec<f32> = grads_w;
            g.extend_from_slice(&grads_b);
            e_opt.step(&mut params, &g);
            let n = model.energy_w.len();
            model.energy_w.data.copy_from_slice(&params[..n]);
            model.energy_b.copy_from_slice(&params[n..]);
        }
        if epoch % 3 == 0 {
            eprintln!("    epoch {:2}: energy trained", epoch);
        }
    }

    // Phase 3: Compare standard vs EBT inference
    eprintln!("\n  Phase 3: EBT inference...");
    let mut ebt_correct = 0usize;
    for chunk in test.chunks(256) {
        let x = Tensor {
            data: chunk.iter().flat_map(|(s, _)| s.iter().copied()).collect(),
            rows: chunk.len(), cols: d_in,
        };
        let (pred, _energies) = model.forward_ebt(&x, 30, 0.5);
        for (b, (_, label)) in chunk.iter().enumerate() {
            let p = if pred.data[b * n_cls] > pred.data[b * n_cls + 1] { 0 } else { 1 };
            if p == *label { ebt_correct += 1; }
        }
    }
    let ebt_acc = ebt_correct as f32 / test.len() as f32;

    eprintln!("  Results:");
    eprintln!("    Standard:  {:.1}%", std_acc * 100.0);
    eprintln!("    EBT:       {:.1}%", ebt_acc * 100.0);
    eprintln!("    Delta:     {:+.1}pp\n", (ebt_acc - std_acc) * 100.0);
}

fn eval_acc(model: &MiniCtm, data: &[(Vec<f32>, usize)], d_in: usize, n_cls: usize) -> f32 {
    let mut correct = 0usize;
    for chunk in data.chunks(256) {
        let x = Tensor {
            data: chunk.iter().flat_map(|(s, _)| s.iter().copied()).collect(),
            rows: chunk.len(), cols: d_in,
        };
        let logits = model.forward(&x);
        for (b, (_, label)) in chunk.iter().enumerate() {
            if (if logits.data[b * n_cls] > logits.data[b * n_cls + 1] { 0 } else { 1 }) == *label {
                correct += 1;
            }
        }
    }
    correct as f32 / data.len() as f32
}
