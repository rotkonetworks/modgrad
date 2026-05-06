//! Find the GPU/CPU crossover for the regional brain forward.
//!
//! Path C version: measures `regional_forward_typed::<Cpu>` vs
//! `regional_forward_typed::<Rocm>` across multiple cortex sizes.
//! Same architectural question as the original *_resident bench
//! ("at what model size does the GPU win") but on the typed cascade.
//!
//! Five preset sizes (single-scale only — typed forward doesn't yet
//! route observation through `obs_scale_slice`, so multi-scale brains
//! are skipped):
//!   - `small`       — cortex d_model=128 (replaces bench_maze_resident)
//!   - `medium`      — cortex d_model=512 (replaces bench_maze_resident_medium, single-scale variant)
//!   - `billion`     — cortex d_model=1024
//!   - `mega-2048`   — cortex d_model=2048
//!   - `mega-2560`   — cortex d_model=2560 (largest fitting on 8 GB)
//!
//! Run with:
//!   cargo run -p bench_brain_crossover --release --features rocm
//!
//! On no-HIP hosts: cleanly returns "skipping". The Cpu baseline still
//! runs but the Cpu/Rocm comparison line is omitted.

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("bench_brain_crossover requires --features rocm; skipping");
}

#[cfg(feature = "rocm")]
fn main() {
    use std::time::Instant;
    use modgrad_ctm::graph::{RegionalConfig, RegionalWeights, RegionalWeightsTyped};
    use modgrad_device::backend::tensor::{Cpu, Rocm};

    const N_ITERS: usize = 300;
    const N_WARMUP: usize = 4;
    const TICKS: usize = 2;
    const OBS_DIM: usize = 224;
    const OUT_DIMS: usize = 64;

    eprintln!("bench_brain_crossover [typed]: n_iters={N_ITERS}\n");

    bench_preset("small", "d_model=128", |obs_dim| {
        let mut cfg = RegionalConfig::eight_region_small(obs_dim, OUT_DIMS, TICKS);
        cfg.router = None;
        cfg
    }, OBS_DIM, N_ITERS, N_WARMUP);

    bench_preset("medium", "d_model=512", |obs_dim| {
        let mut cfg = RegionalConfig::eight_region_medium(obs_dim, OUT_DIMS, TICKS);
        cfg.router = None;
        cfg
    }, OBS_DIM, N_ITERS, N_WARMUP);

    bench_preset("billion", "d_model=1024", |obs_dim| {
        let mut cfg = RegionalConfig::eight_region_billion(obs_dim, OUT_DIMS, TICKS);
        cfg.router = None;
        cfg
    }, OBS_DIM, N_ITERS, N_WARMUP);

    bench_preset("mega-2048", "d_model=2048", |obs_dim| {
        let mut cfg = RegionalConfig::eight_region_mega(obs_dim, OUT_DIMS, TICKS, 2048);
        cfg.router = None;
        cfg
    }, OBS_DIM, 100, 2);

    bench_preset("mega-2560", "d_model=2560", |obs_dim| {
        let mut cfg = RegionalConfig::eight_region_mega(obs_dim, OUT_DIMS, TICKS, 2560);
        cfg.router = None;
        cfg
    }, OBS_DIM, 50, 2);

    eprintln!("\n────────────────────────────────────────────");
    eprintln!("Path C cascade comparison: Cpu vs Rocm Tensor<D>");
    eprintln!("backends, same code, same allocations. Per-region matvec");
    eprintln!("compute scales as cortex_d_model²; per-dispatch overhead");
    eprintln!("is constant. Crossover (Rocm ≥ Cpu) is whichever preset");
    eprintln!("first flips from CPU WINS to ROCM WINS.");
    eprintln!("────────────────────────────────────────────");

    fn bench_preset(
        name: &str,
        size_label: &str,
        cfg_fn: impl FnOnce(usize) -> RegionalConfig,
        obs_dim: usize,
        n_iters: usize,
        n_warmup: usize,
    ) {
        eprintln!("=== preset {name} ({size_label}) ===");

        let cfg = cfg_fn(obs_dim);
        let weights = RegionalWeights::new(cfg);
        eprintln!("  params  = {}", weights.n_params());
        eprintln!("  regions = {}, conns = {}",
            weights.regions.len(), weights.connection_synapses.len());

        let observation: Vec<f32> = (0..obs_dim)
            .map(|i| (i as f32 * 0.013 - 0.21).sin())
            .collect();

        // ── Cpu typed path ────────────────────────────────────
        let typed_cpu = RegionalWeightsTyped::<Cpu>::from_untyped(&weights)
            .expect("typed Cpu lift");
        let n_sync = typed_cpu.config.n_global_sync;
        let mut act_cpu: Vec<Vec<f32>> = weights.regions.iter()
            .map(|r| r.start_activated.clone()).collect();
        let mut trc_cpu: Vec<Vec<f32>> = weights.regions.iter()
            .map(|r| r.start_trace.clone()).collect();
        let mut alpha_cpu = vec![0.0f32; n_sync];
        let mut beta_cpu = vec![1.0f32; n_sync];
        for _ in 0..n_warmup {
            let _ = typed_cpu.regional_forward_typed(
                &observation, &mut act_cpu, &mut trc_cpu,
                &mut alpha_cpu, &mut beta_cpu,
            ).expect("Cpu warmup");
        }
        let t = Instant::now();
        for _ in 0..n_iters {
            let _ = typed_cpu.regional_forward_typed(
                &observation, &mut act_cpu, &mut trc_cpu,
                &mut alpha_cpu, &mut beta_cpu,
            ).expect("Cpu forward");
        }
        let dt_cpu = t.elapsed();
        let per_iter_cpu = dt_cpu.as_nanos() as f64 / n_iters as f64 / 1000.0;
        eprintln!("  Cpu  wall = {dt_cpu:?}  per-iter = {per_iter_cpu:.2} µs");

        // ── Rocm typed path ───────────────────────────────────
        let typed_rocm = match RegionalWeightsTyped::<Rocm>::from_untyped(&weights) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  Rocm: lift failed ({e}) — skipping Rocm phase");
                return;
            }
        };
        let mut act_r: Vec<Vec<f32>> = weights.regions.iter()
            .map(|r| r.start_activated.clone()).collect();
        let mut trc_r: Vec<Vec<f32>> = weights.regions.iter()
            .map(|r| r.start_trace.clone()).collect();
        let mut alpha_r = vec![0.0f32; n_sync];
        let mut beta_r = vec![1.0f32; n_sync];
        for _ in 0..n_warmup {
            let _ = typed_rocm.regional_forward_typed(
                &observation, &mut act_r, &mut trc_r,
                &mut alpha_r, &mut beta_r,
            ).expect("Rocm warmup");
        }
        let t = Instant::now();
        for _ in 0..n_iters {
            let _ = typed_rocm.regional_forward_typed(
                &observation, &mut act_r, &mut trc_r,
                &mut alpha_r, &mut beta_r,
            ).expect("Rocm forward");
        }
        let dt_rocm = t.elapsed();
        let per_iter_rocm = dt_rocm.as_nanos() as f64 / n_iters as f64 / 1000.0;
        eprintln!("  Rocm wall = {dt_rocm:?}  per-iter = {per_iter_rocm:.2} µs");

        let speedup = per_iter_cpu / per_iter_rocm;
        let verdict = if speedup > 1.0 {
            format!("ROCM WINS by {speedup:.2}×")
        } else {
            format!("CPU WINS — Rocm is {:.2}× slower", 1.0 / speedup)
        };
        eprintln!("  → {verdict}\n");
    }
}
