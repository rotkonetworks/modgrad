//! Scaling test: how fast is forward_split at different neuron counts?
//! cargo test --release --test scale_test -- --nocapture

use modgrad::ctm::{Ctm, CtmConfig, CtmSession, LayerConfig, forward_split};

#[test]
fn forward_pass_scaling() {
    eprintln!("\n  === Forward pass timing vs neuron count ===\n");
    eprintln!("  {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "neurons", "params", "1 fwd (ms)", "100 fwd", "samples/s", "bottleneck");

    for &n in &[64, 128, 256, 512, 1024] {
        let cfg = CtmConfig {
            iterations: 8, d_input: 64, n_sync_out: 64,
            input_layer: LayerConfig { n_neurons: n, ..Default::default() },
            attention_layer: LayerConfig { n_neurons: n / 2, ..Default::default() },
            output_layer: LayerConfig { n_neurons: n / 2, ..Default::default() },
            motor_layer: LayerConfig { n_neurons: 4, ..Default::default() },
            ..CtmConfig::default()
        };

        let ctm = Ctm::new(cfg.clone());
        let (weights, _) = ctm.into_split();
        let proprio = vec![0.0f32; 64];
        let input = vec![0.5f32; 64];

        let total_neurons = n + n/2 + n/2 + 4
            + cfg.cerebellum_layer.n_neurons
            + cfg.basal_ganglia_layer.n_neurons
            + cfg.insula_layer.n_neurons
            + cfg.hippocampus_layer.n_neurons;
        let total_params: usize = weights.synapse_refs().iter()
            .map(|s| s.linear.weight.len()).sum();

        // Warmup
        for _ in 0..3 {
            let mut s = CtmSession::new(&weights.config);
            let mut t = weights.init_tick_state();
            let _ = forward_split(&weights, &mut s, &mut t, &input, &proprio, false);
        }

        // Time 100 forward passes
        let t0 = std::time::Instant::now();
        let n_runs = 100;
        for _ in 0..n_runs {
            let mut s = CtmSession::new(&weights.config);
            let mut t = weights.init_tick_state();
            let _ = forward_split(&weights, &mut s, &mut t, &input, &proprio, false);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_fwd_ms = elapsed / n_runs as f64 * 1000.0;
        let samples_per_sec = n_runs as f64 / elapsed;

        let bottleneck = if per_fwd_ms < 1.0 { "CPU OK" }
            else if per_fwd_ms < 10.0 { "CPU marginal" }
            else { "NEEDS GPU" };

        eprintln!("  {:>8} {:>8} {:>10.2} {:>8.1}s {:>10.0} {:>10}",
            total_neurons, total_params, per_fwd_ms, elapsed, samples_per_sec, bottleneck);
    }
    eprintln!();

    // Wake/sleep throughput estimate
    eprintln!("  Wake/sleep cycle needs ~4000 wake + ~768 sleep = ~4768 forward passes");
    eprintln!("  At 100 samples/s = 48s per cycle (10 cycles = 8 min)");
    eprintln!("  At 10 samples/s  = 480s per cycle (10 cycles = 80 min) → need GPU");
    eprintln!();
}
