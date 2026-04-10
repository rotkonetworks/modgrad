//! Verify GPU NLM (superlinear kernel) matches CPU NLM numerically.
//! run: cargo test --release --test kfd_nlm_gpu_verify -- --nocapture --test-threads=1

use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, CtmTickState, NeuronLayerWeights, forward_split};
use modgrad::gpu_accel::{GpuWeightCache, GpuNlmCache, gpu_nlm_step};
use modgrad::kfd::{self, HsaDevice};
use modgrad::kfd::dispatch::KernArgs;
use modgrad::tabula_rasa::Dna;
use std::sync::Arc;

/// Test 1: Raw superlinear kernel dispatch matches CPU SuperLinear::forward.
#[test]
fn verify_superlinear_kernel() {
    if !kfd::is_available() { eprintln!("skip: no kfd"); return; }
    let mut dev = match HsaDevice::open() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: {e}"); return; }
    };

    // Create a SuperLinear and test GPU vs CPU
    use modgrad::ctm::SuperLinear;

    for &(n, inp, out) in &[(64, 8, 128), (128, 16, 256), (8, 4, 2), (1504, 8, 128)] {
        let sl = SuperLinear::new(n, inp, out);
        let trace: Vec<f32> = (0..n * inp).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();

        // CPU reference
        let cpu_out = sl.forward(&trace);

        // GPU dispatch
        let w_buf = dev.upload_f32(&sl.weights).unwrap();
        let b_buf = dev.upload_f32(&sl.biases).unwrap();
        let x_buf = dev.upload_f32(&trace).unwrap();
        let y_buf = dev.alloc_output(n * out * 4 + 64).unwrap();

        let ok = dev.dispatch_superlinear_enqueue(&w_buf, &b_buf, &x_buf, &y_buf,
                                                   n as u32, out as u32, inp as u32);
        if !ok { eprintln!("skip: superlinear_fwd kernel not loaded"); return; }
        assert!(dev.submit_wait(10_000), "GPU superlinear timeout");

        let gpu_out = unsafe {
            std::slice::from_raw_parts(y_buf.cpu_ptr as *const f32, n * out)
        };

        // Compare
        let max_err: f32 = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let mean_err: f32 = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).sum::<f32>() / cpu_out.len() as f32;

        println!("  SuperLinear({},{},{}) max_err={:.6} mean_err={:.8}", n, inp, out, max_err, mean_err);

        // FP32 tolerance: rcp_iflag_f32 introduces small division error
        assert!(max_err < 0.01, "SuperLinear({},{},{}) max_err={} too large!", n, inp, out, max_err);
        assert!(!gpu_out.iter().any(|x| x.is_nan()), "NaN in GPU output");
    }
    println!("  PASS: all SuperLinear GPU outputs match CPU");
}

/// Test 2: Full gpu_nlm_step matches CPU NeuronLayerWeights::step for all 8 regions.
#[test]
fn verify_gpu_nlm_step() {
    if !kfd::is_available() { eprintln!("skip: no kfd"); return; }
    let mut dev = match HsaDevice::open() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: {e}"); return; }
    };

    let cfg = CtmConfig {
        iterations: 2,
        ..Dna::large().ctm
    };
    let ctm = Ctm::new(cfg.clone());
    let (weights, _) = ctm.into_split();

    let nlm = match GpuNlmCache::new(&dev, &weights) {
        Ok(c) => c,
        Err(e) => { eprintln!("skip: NLM cache failed: {e}"); return; }
    };

    let region_weights: [&NeuronLayerWeights; 8] = [
        &weights.input_region, &weights.attention_region,
        &weights.output_region, &weights.motor_region,
        &weights.cerebellum_region, &weights.basal_ganglia_region,
        &weights.insula_region, &weights.hippocampus_region,
    ];

    let region_names = ["input", "attention", "output", "motor",
                        "cerebellum", "basal_ganglia", "insula", "hippocampus"];

    // Create dummy signals (synapse outputs)
    let mut cpu_signals: [Vec<f32>; 8] = Default::default();
    for r in 0..8 {
        let n = region_weights[r].config.n_neurons;
        cpu_signals[r] = (0..n).map(|i| ((i * 13 + r * 7 + 5) % 100) as f32 / 100.0 - 0.5).collect();
    }

    // CPU path
    let tick_state = weights.init_tick_state();
    let mut cpu_traces: [Vec<f32>; 8] = Default::default();
    for r in 0..8 {
        cpu_traces[r] = tick_state.trace(r).to_vec();
    }
    let mut cpu_traces_copy = cpu_traces.clone();

    let mut cpu_acts: [Vec<f32>; 8] = Default::default();
    for r in 0..8 {
        cpu_acts[r] = region_weights[r].step(&cpu_signals[r], &mut cpu_traces_copy[r]);
    }

    // GPU path
    let sigs: [&[f32]; 8] = [
        &cpu_signals[0], &cpu_signals[1], &cpu_signals[2], &cpu_signals[3],
        &cpu_signals[4], &cpu_signals[5], &cpu_signals[6], &cpu_signals[7],
    ];
    let gpu_acts = gpu_nlm_step(&mut dev, &nlm, &region_weights, &sigs, &mut cpu_traces);

    // Compare per region
    println!();
    let mut overall_max = 0.0f32;
    for r in 0..8 {
        let max_err: f32 = cpu_acts[r].iter().zip(gpu_acts[r].iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let mean_err: f32 = cpu_acts[r].iter().zip(gpu_acts[r].iter())
            .map(|(a, b)| (a - b).abs()).sum::<f32>() / cpu_acts[r].len().max(1) as f32;
        overall_max = overall_max.max(max_err);

        let status = if max_err < 0.001 { "EXACT" }
            else if max_err < 0.01 { "MATCH" }
            else if max_err < 0.1 { "CLOSE" }
            else { "DIVERGED" };

        println!("  region {:>14} (n={:>4}): max_err={:.6} mean_err={:.8} [{}]",
            region_names[r], region_weights[r].config.n_neurons, max_err, mean_err, status);
    }
    println!();
    println!("  overall max_err: {:.6}", overall_max);

    // Allow some tolerance for FP32 reduction order differences
    assert!(overall_max < 0.05, "GPU NLM diverged too much: max_err={}", overall_max);

    // Verify no NaN/Inf
    for r in 0..8 {
        assert!(!gpu_acts[r].iter().any(|x| x.is_nan()), "NaN in GPU NLM region {}", r);
        assert!(!gpu_acts[r].iter().any(|x| x.is_infinite()), "Inf in GPU NLM region {}", r);
    }
    println!("  PASS: GPU NLM matches CPU across all 8 regions");
}
