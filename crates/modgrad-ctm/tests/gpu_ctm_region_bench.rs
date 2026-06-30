//! M1 de-risk for the batched-resident GPU training path: does the REAL CTM
//! region op (not just raw matmul) run faster batched on ROCm than the
//! per-sample loop? If yes, the batched cascade approach is sound and the
//! backward is worth building; if not, the per-tick host-bouncing must be made
//! resident first. Run:
//!   HSA_OVERRIDE_GFX_VERSION=11.0.0 cargo test -p modgrad-ctm --release \
//!     --features rocm --test gpu_ctm_region_bench -- --ignored --nocapture

#[cfg(feature = "rocm")]
#[test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture"]
fn bench_ctm_region_batched_vs_scalar_rocm() {
    use modgrad_ctm::config::{CtmConfig, ExitStrategy};
    use modgrad_ctm::forward::{ctm_forward_typed_batched, ctm_forward_typed_with_cache};
    use modgrad_ctm::weights::CtmWeights;
    use modgrad_ctm::weights::CtmWeightsTyped;
    use modgrad_device::backend::tensor::Rocm;
    use std::time::Instant;

    // A cortex-sized region (the eight_region_large cortex: d_model 512).
    let d_model = 512usize;
    let d_input = 128usize;
    let ticks = 8usize;
    let raw = d_input;
    let batch = 256usize;
    let cfg = CtmConfig::region("bench", d_model, d_input, 8, true, ticks, ExitStrategy::None);

    let w = CtmWeights::new(cfg.clone(), raw);
    let typed = CtmWeightsTyped::<Rocm>::from_untyped(&w).expect("upload region to ROCm");
    let obs: Vec<f32> = (0..batch * raw)
        .map(|i| (((i * 7 % 13) as f32) - 6.0) * 0.05)
        .collect();

    // ── batched forward (one call for the whole batch) ──
    let mut act_b: Vec<f32> = (0..batch).flat_map(|_| w.start_activated.clone()).collect();
    let mut tr_b: Vec<f32> = (0..batch).flat_map(|_| w.start_trace.clone()).collect();
    // warm (JIT kernels)
    let _ = ctm_forward_typed_batched::<Rocm>(&typed, &mut act_b.clone(), &mut tr_b.clone(), &obs, batch).unwrap();
    let iters = 10;
    let t = Instant::now();
    for _ in 0..iters {
        let mut a = act_b.clone();
        let mut tr = tr_b.clone();
        let _ = ctm_forward_typed_batched::<Rocm>(&typed, &mut a, &mut tr, &obs, batch).unwrap();
    }
    let batched_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // ── per-sample loop (batch scalar calls) ──
    // warm
    {
        let mut a = w.start_activated.clone();
        let mut tr = w.start_trace.clone();
        let _ = ctm_forward_typed_with_cache::<Rocm>(&typed, &mut a, &mut tr, &obs[0..raw]).unwrap();
    }
    let t = Instant::now();
    for _ in 0..iters {
        for b in 0..batch {
            let mut a = w.start_activated.clone();
            let mut tr = w.start_trace.clone();
            let _ = ctm_forward_typed_with_cache::<Rocm>(
                &typed, &mut a, &mut tr, &obs[b * raw..(b + 1) * raw],
            )
            .unwrap();
        }
    }
    let scalar_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    println!(
        "\n  CTM region d_model={d_model} ticks={ticks} batch={batch} (ROCm):\n  \
         batched  : {batched_ms:8.2} ms/batch  ({:6.3} ms/sample)\n  \
         per-sample: {scalar_ms:8.2} ms/batch  ({:6.3} ms/sample)\n  \
         speedup  : {:.1}x\n",
        batched_ms / batch as f64,
        scalar_ms / batch as f64,
        scalar_ms / batched_ms,
    );
}
