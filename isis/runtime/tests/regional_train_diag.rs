//! Diagnostic tests: per-region gradient L2 norms per epoch.
//! Measurement only — not for CI.

use isis_runtime::regional::*;

fn fmt_norms(norms: &[f32]) -> String {
    let parts: Vec<String> = norms.iter().map(|n| format!("{:.4e}", n)).collect();
    format!("[{}]", parts.join(", "))
}

#[test]
fn four_region_bptt_diagnostic() {
    let cfg = RegionalConfig::four_region(16, 4, 4);
    eprintln!("  region names: {:?}", cfg.region_names);
    eprintln!("  connections:");
    for (i, c) in cfg.connections.iter().enumerate() {
        eprintln!("    {i}: from={:?} to={} recv_obs={}", c.from, c.to, c.receives_observation);
    }
    let mut w = RegionalWeights::new(cfg);
    eprintln!("  params: {}", w.n_params());

    let examples: Vec<(Vec<f32>, usize)> = vec![
        (vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
        (vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1),
        (vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2),
        (vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3),
    ];

    let lr = 0.01;
    let clip = 5.0;

    for epoch in 0..20 {
        let mut epoch_loss = 0.0;
        // Accumulate per-region norms across the 4 examples (sum, averaged at end).
        let n_regions = w.config.regions.len();
        let n_conns = w.config.connections.len();
        let mut sum_region_norms = vec![0.0f64; n_regions];
        let mut sum_conn_norms = vec![0.0f64; n_conns];

        for (obs, target) in &examples {
            let mut grads = RegionalGradients::zeros(&w);
            let (loss, _pred) = regional_train_step(&w, &mut grads, obs, *target);
            let rn = grads.region_norms();
            let cn = grads.connection_norms();
            for (i, n) in rn.iter().enumerate() { sum_region_norms[i] += *n as f64; }
            for (i, n) in cn.iter().enumerate() { sum_conn_norms[i] += *n as f64; }
            grads.apply(&mut w, lr, clip);
            epoch_loss += loss;
        }

        epoch_loss /= examples.len() as f32;
        let mean_region: Vec<f32> = sum_region_norms.iter()
            .map(|s| (*s / examples.len() as f64) as f32).collect();
        let mean_conn: Vec<f32> = sum_conn_norms.iter()
            .map(|s| (*s / examples.len() as f64) as f32).collect();
        eprintln!("epoch {:3}: loss={:.4} region_norms={} conn_norms={}",
            epoch, epoch_loss, fmt_norms(&mean_region), fmt_norms(&mean_conn));
    }
}

#[test]
fn eight_region_bptt_diagnostic() {
    let cfg = RegionalConfig::eight_region(16, 4, 4);
    eprintln!("  region names: {:?}", cfg.region_names);
    eprintln!("  connections:");
    for (i, c) in cfg.connections.iter().enumerate() {
        eprintln!("    {i}: from={:?} to={} recv_obs={}", c.from, c.to, c.receives_observation);
    }
    let mut w = RegionalWeights::new(cfg);
    eprintln!("  8-region params: {}", w.n_params());

    let examples: Vec<(Vec<f32>, usize)> = vec![
        (vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
        (vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1),
        (vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2),
        (vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3),
    ];

    let lr = 0.001;
    let clip = 2.0;

    for epoch in 0..20 {
        let mut epoch_loss = 0.0;
        let n_regions = w.config.regions.len();
        let mut sum_region_norms = vec![0.0f64; n_regions];
        let mut sum_to_route = vec![0.0f64; n_regions];
        let mut sum_from_route = vec![0.0f64; n_regions];
        let mut sum_route_proj = 0.0f64;
        let mut sum_tick_embed = 0.0f64;

        for (obs, target) in &examples {
            let mut grads = RegionalGradients::zeros(&w);
            let (loss, _pred) = regional_train_step(&w, &mut grads, obs, *target);
            let rn = grads.region_norms();
            let (tr, fr, rp, te) = grads.router_norms();
            for (i, n) in rn.iter().enumerate() { sum_region_norms[i] += *n as f64; }
            for (i, n) in tr.iter().enumerate() { sum_to_route[i] += *n as f64; }
            for (i, n) in fr.iter().enumerate() { sum_from_route[i] += *n as f64; }
            sum_route_proj += rp as f64;
            sum_tick_embed += te as f64;
            grads.apply(&mut w, lr, clip);
            epoch_loss += loss;
        }

        epoch_loss /= examples.len() as f32;
        let m = |v: &[f64]| v.iter().map(|s| (*s / examples.len() as f64) as f32).collect::<Vec<f32>>();
        eprintln!(
            "epoch {:3}: loss={:.4} region={} to_route={} from_route={} route_proj={:.4e} tick_embed={:.4e}",
            epoch, epoch_loss,
            fmt_norms(&m(&sum_region_norms)),
            fmt_norms(&m(&sum_to_route)),
            fmt_norms(&m(&sum_from_route)),
            sum_route_proj / examples.len() as f64,
            sum_tick_embed / examples.len() as f64,
        );
    }
}
