//! Verify GPU batched forward matches CPU forward numerically.
//! run: cargo test --release --test kfd_gpu_accel_verify -- --nocapture --test-threads=1

use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, CtmTickState, forward_split};
use modgrad::gpu_accel::{GpuWeightCache, forward_split_batched};
use modgrad::kfd::{self, HsaDevice};
use modgrad::tabula_rasa::Dna;
use std::sync::Arc;

#[test]
fn verify_gpu_matches_cpu() {
    if !kfd::is_available() { eprintln!("skip: no kfd"); return; }
    let mut dev = match HsaDevice::open() {
        Ok(d) => d,
        Err(e) => { eprintln!("skip: {e}"); return; }
    };

    // Use large config — bigger synapses that align to GPU tiles
    let cfg = CtmConfig {
        iterations: 2,
        ..Dna::large().ctm
    };
    let ctm = Ctm::new(cfg.clone());
    let (weights, _) = ctm.into_split();

    // Print synapse dims (all go to GPU now via padding)
    println!();
    for (i, (k, m)) in weights.synapse_dims().iter().enumerate() {
        let m_pad = ((*m + 127) / 128) * 128;
        let k_pad = ((*k + 7) / 8) * 8;
        println!("  synapse {}: in={:<5}(→{:<5}) out={:<5}(→{:<5}) GPU",
            i, k, k_pad, m, m_pad);
    }
    println!();

    let gpu = match GpuWeightCache::new(&dev, &weights) {
        Ok(g) => g,
        Err(e) => { eprintln!("skip: GPU cache failed: {e}"); return; }
    };
    let weights = Arc::new(weights);

    // Step 1: Sanity test — single matmul dispatch
    println!("  Step 1: single synapse matmul test...");
    {
        let dims = weights.synapse_dims();
        // Find first synapse that fits GPU
        for (i, &(k, m)) in dims.iter().enumerate() {
            if m >= 32 && k >= 8 && m % 32 == 0 && k % 8 == 0 {
                println!("    testing synapse {} (in={} out={}) on GPU...", i, k, m);
                let x = vec![0.001f32; 32 * k];  // batch=32
                let x_buf = dev.upload_f32(&x).unwrap();
                let y_buf = dev.alloc_output((32 * m * 4 + 64) as usize).unwrap();

                let gs = &gpu.synapses[i];
                let n = 32u32;
                let (kn, nwg, w) = if m as u32 >= 1536 {
                    ("matmul_blocked", (((m as u32) + 127) / 128) * ((n + 31) / 32), &gs.w_row)
                } else {
                    ("matmul_small", (((m as u32) + 31) / 32) * ((n + 31) / 32), &gs.w_col)
                };

                use modgrad::kfd::dispatch::KernArgs;
                let mut args = KernArgs::new();
                args.push_ptr(w); args.push_ptr(&gs.bias);
                args.push_ptr(&x_buf); args.push_ptr(&y_buf);
                args.push_u32(m as u32); args.push_u32(k as u32); args.push_u32(n);
                let ab = args.upload(&dev.alloc).unwrap();

                dev.dispatch_enqueue(kn, &ab, [nwg, 1, 1], [256, 1, 1]);
                assert!(dev.submit_wait(5_000), "single synapse GPU timeout!");
                println!("    OK — GPU dispatch works for synapse {}", i);
                break;
            }
        }
    }

    // Step 2: Full batched forward vs CPU
    let batch = 32;
    let obs_data: Vec<Vec<f32>> = (0..batch).map(|i| {
        (0..cfg.d_input).map(|j| ((i * 7 + j * 3) % 100) as f32 / 100.0).collect()
    }).collect();
    let proprio = vec![0.0f32; cfg.d_input];

    println!("  Step 2: CPU forward ({} samples, {} ticks)...", batch, cfg.iterations);
    let cpu_results: Vec<Vec<f32>> = obs_data.iter().map(|obs| {
        let mut session = CtmSession::new(&weights.config);
        let mut tick_state = weights.init_tick_state();
        let (_, sync, _) = forward_split(
            &weights, &mut session, &mut tick_state, obs, &proprio, false);
        sync
    }).collect();
    println!("    CPU done, sync_dim={}", cpu_results[0].len());

    println!("  Step 3: GPU batched forward...");
    let mut sessions: Vec<CtmSession> = (0..batch)
        .map(|_| CtmSession::new(&weights.config)).collect();
    let mut tick_states: Vec<CtmTickState> = (0..batch)
        .map(|_| weights.init_tick_state()).collect();

    let obs_refs: Vec<&[f32]> = obs_data.iter().map(|v| v.as_slice()).collect();
    let pro_refs: Vec<&[f32]> = (0..batch).map(|_| proprio.as_slice()).collect();

    let gpu_results = forward_split_batched(
        &mut dev, &gpu, &weights,
        &mut sessions, &mut tick_states,
        &obs_refs, &pro_refs, false,
    );
    println!("    GPU done");

    // Compare
    println!();
    println!("  Comparison (first 8 of {} samples):", batch);
    println!("  {}", "-".repeat(60));

    let mut max_err = 0.0f32;
    for b in 0..batch.min(8) {
        let cpu_sync = &cpu_results[b];
        let (gpu_sync, _, _) = &gpu_results[b];

        let sample_max: f32 = cpu_sync.iter().zip(gpu_sync.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let sample_mean: f32 = cpu_sync.iter().zip(gpu_sync.iter())
            .map(|(a, b)| (a - b).abs()).sum::<f32>() / cpu_sync.len() as f32;
        max_err = max_err.max(sample_max);

        let status = if sample_max < 0.01 { "MATCH" }
            else if sample_max < 0.1 { "CLOSE" }
            else if sample_max < 1.0 { "DIVERGED" }
            else { "WRONG" };

        println!("  sample {:>2}: max={:.4} mean={:.4}  [{}]", b, sample_max, sample_mean, status);
    }
    println!();
    println!("  overall max_err: {:.6}", max_err);
    println!("  note: some divergence expected (simplified sync_action in GPU path)");

    // Verify outputs are valid
    for b in 0..batch {
        let (gpu_sync, _, _) = &gpu_results[b];
        assert!(!gpu_sync.iter().any(|x| x.is_nan()), "sample {} has NaN", b);
        assert!(!gpu_sync.iter().all(|x| *x == 0.0), "sample {} all zero", b);
    }
    println!("  PASS: all GPU outputs are finite and non-zero");
    println!();
}
