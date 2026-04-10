//! Depth-2 synapse scaling test.
//! Compares depth-1 vs depth-2 at different neuron counts.
//! cargo test --release --test depth2_scaling -- --nocapture

use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, LayerConfig};
use modgrad::tasks;
use modgrad::accuracy::eval_ls;

#[test]
fn depth1_vs_depth2_scaling() {
    let d_input = 64;
    let data = tasks::maze_examples(7, d_input, 4000);
    let train = &data[..3000];
    let test = &data[3000..];

    eprintln!("\n  === Depth-1 vs Depth-2 Synapse Scaling (7×7 maze) ===\n");
    eprintln!("  {:>12} {:>6} {:>8} {:>8} {:>10} {:>10} {:>10}",
        "config", "depth", "neurons", "params", "features", "LS test%", "time(s)");

    for &(n, label) in &[(64, "small"), (128, "medium"), (256, "large"), (512, "2K")] {
        for &depth in &[1, 2] {
            let cfg = CtmConfig {
                iterations: 8, d_input, n_sync_out: 64,
                synapse_depth: depth,
                input_layer: LayerConfig { n_neurons: n, ..Default::default() },
                attention_layer: LayerConfig { n_neurons: n / 2, ..Default::default() },
                output_layer: LayerConfig { n_neurons: n / 2, ..Default::default() },
                motor_layer: LayerConfig { n_neurons: 4, ..Default::default() },
                ..CtmConfig::default()
            };

            let ctm = Ctm::new(cfg.clone());
            let (weights, _) = ctm.into_split();

            let total_neurons: usize = [
                cfg.input_layer.n_neurons, cfg.attention_layer.n_neurons,
                cfg.output_layer.n_neurons, cfg.motor_layer.n_neurons,
                cfg.cerebellum_layer.n_neurons, cfg.basal_ganglia_layer.n_neurons,
                cfg.insula_layer.n_neurons, cfg.hippocampus_layer.n_neurons,
            ].iter().sum();

            let total_params: usize = weights.synapse_refs().iter()
                .map(|s| {
                    let p1 = s.linear.weight.len();
                    let p2 = s.linear2.as_ref().map(|l| l.weight.len()).unwrap_or(0);
                    p1 + p2
                }).sum();

            let act_dim = weights.init_tick_state().activations.len();

            let t0 = std::time::Instant::now();
            let acc = eval_ls(&weights, test);
            let elapsed = t0.elapsed().as_secs_f64();

            // Verify depth
            let actual_depth = weights.syn_motor_input.depth();

            eprintln!("  {:>12} {:>4}d{} {:>8} {:>8} {:>10} {:>9.1}% {:>10.1}",
                label, actual_depth, if actual_depth == depth { "" } else { "!" },
                total_neurons, total_params, act_dim,
                acc * 100.0, elapsed);
        }
    }
    eprintln!();
}

#[test]
fn depth_scaling_15x15() {
    let d_input = 256;
    let data = tasks::maze_examples(15, d_input, 3000);
    let test = &data[2000..];

    eprintln!("\n  === 15×15 Maze Scaling ===\n");
    eprintln!("  {:>8} {:>6} {:>8} {:>10} {:>10}",
        "neurons", "depth", "params", "LS test%", "time(s)");

    for &n in &[128, 256, 512] {
        for &depth in &[1, 2] {
            let cfg = CtmConfig {
                iterations: 8, d_input, n_sync_out: 64,
                synapse_depth: depth,
                input_layer: LayerConfig { n_neurons: n, ..Default::default() },
                attention_layer: LayerConfig { n_neurons: n / 2, ..Default::default() },
                output_layer: LayerConfig { n_neurons: n / 2, ..Default::default() },
                motor_layer: LayerConfig { n_neurons: 4, ..Default::default() },
                ..CtmConfig::default()
            };
            let ctm = Ctm::new(cfg.clone());
            let (weights, _) = ctm.into_split();
            let total_params: usize = weights.synapse_refs().iter()
                .map(|s| s.linear.weight.len() + s.linear2.as_ref().map(|l| l.weight.len()).unwrap_or(0)).sum();
            let t0 = std::time::Instant::now();
            let acc = eval_ls(&weights, test);
            eprintln!("  {:>8} {:>4}d {:>8} {:>9.1}% {:>10.1}",
                cfg.input_layer.n_neurons + cfg.attention_layer.n_neurons + cfg.output_layer.n_neurons + 4,
                depth, total_params, acc * 100.0, t0.elapsed().as_secs_f64());
        }
    }
    eprintln!();
}
