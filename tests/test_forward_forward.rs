//! Test: MiniCTM with backprop in pure Rust.
//! Does it match PyTorch's 91.6% on QEC?
//! cargo test --release --test test_forward_forward -- --nocapture

use modgrad_compute::tensor::*;

#[test]
fn mini_ctm_qec() {
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

    eprintln!("\n  Rust MiniCTM backprop on QEC (d_in={}, {} train, {} test)",
        d_in, train.len(), test.len());

    let mut model = MiniCtm::new(d_in, 128, 4, 2);
    let mut opt = Adam::new(model.param_count(), 0.001);
    let batch_size = 64;

    for epoch in 0..20 {
        let mut total_loss = 0.0f32;
        let mut n_batches = 0;

        for chunk in train.chunks(batch_size) {
            let x = Tensor {
                data: chunk.iter().flat_map(|(s, _)| s.iter().copied()).collect(),
                rows: chunk.len(), cols: d_in,
            };
            let targets: Vec<usize> = chunk.iter().map(|(_, l)| *l).collect();

            let (loss, grads) = model.forward_backward(&x, &targets);
            model.apply_grads(&grads, &mut opt);
            total_loss += loss;
            n_batches += 1;
        }

        let mut correct = 0usize;
        for chunk in test.chunks(256) {
            let x = Tensor {
                data: chunk.iter().flat_map(|(s, _)| s.iter().copied()).collect(),
                rows: chunk.len(), cols: d_in,
            };
            let logits = model.forward(&x);
            for (b, (_, label)) in chunk.iter().enumerate() {
                let pred = if logits.data[b * 2] > logits.data[b * 2 + 1] { 0 } else { 1 };
                if pred == *label { correct += 1; }
            }
        }

        let acc = correct as f32 / test.len() as f32;
        eprintln!("  epoch {:2}: loss={:.4} acc={:.1}%",
            epoch, total_loss / n_batches as f32, acc * 100.0);
    }
    eprintln!();
}
