//! Find the GPU/CPU crossover for the regional brain forward.
//!
//! `bench_maze_resident_medium` showed GPU is 3× slower than CPU at
//! d_model=512 — per-dispatch MIOpen overhead dominates. Small brains
//! aren't where GPU wins. This bench pushes to LARGER models where
//! compute >> dispatch overhead and resident should start winning.
//!
//! Three preset sizes:
//!   - `billion`     — cortex d_model=1024 (~2× compute over medium)
//!   - `mega-2048`   — cortex d_model=2048 (~8× compute, custom inline)
//!   - `mega-4096`   — cortex d_model=4096 (~32× compute)
//!
//! Larger d_model means each per-dispatch matvec amortizes more compute
//! against the same ~5µs MIOpen overhead. Crossover hardware-dependent;
//! on gfx1102 we expect somewhere between 1024 and 4096.
//!
//! Run with:
//!   cargo run -p bench_brain_crossover --release --features rocm
//!
//! On no-HIP hosts: cleanly returns "skipping". The host-only baseline
//! still runs but the comparison line is omitted.

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("bench_brain_crossover requires --features rocm; skipping");
}

#[cfg(feature = "rocm")]
fn main() {
    use std::time::Instant;
    use modgrad_ctm::graph::{RegionalBrain, RegionalConfig, RegionalWeights};
    use modgrad_ctm::resident::RegionalResidentCache;
    use modgrad_traits::{Brain, TokenInput};
    use modgrad_device::backend::HipBatch;

    const N_ITERS: usize = 300;
    const N_WARMUP: usize = 4;
    const TICKS: usize = 2;
    const OBS_DIM: usize = 224;
    const OUT_DIMS: usize = 64;

    eprintln!("bench_brain_crossover: n_iters={N_ITERS}\n");

    // ── medium (d_model=512) — known to lose, included as reference ──
    bench_preset("medium", "d_model=512", |obs_dim| {
        let scales: Vec<usize> = vec![128, 64, 32];
        assert_eq!(scales.iter().sum::<usize>(), obs_dim);
        let mut cfg = RegionalConfig::eight_region_medium_multiscale(
            &scales, OUT_DIMS, TICKS);
        cfg.router = None;
        cfg
    }, OBS_DIM, N_ITERS, N_WARMUP);

    // ── billion (d_model=1024) — 4× compute over medium ────────
    bench_preset("billion", "d_model=1024", |obs_dim| {
        let mut cfg = RegionalConfig::eight_region_billion(obs_dim, OUT_DIMS, TICKS);
        cfg.router = None;
        cfg
    }, OBS_DIM, N_ITERS, N_WARMUP);

    eprintln!("\n────────────────────────────────────────────");
    eprintln!("Read: if billion's 'speedup' is < 1.0×, the brain at");
    eprintln!("d_model=1024 still isn't enough compute per dispatch to");
    eprintln!("amortize MIOpen overhead. Crossover lives above 1024 for");
    eprintln!("this brain shape; the answer is 'larger brains' or");
    eprintln!("'batched cross-region kernels' (multi-week project).");
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

        let input = TokenInput {
            tokens: (0..obs_dim)
                .map(|i| (i as f32 * 0.013 - 0.21).sin())
                .collect(),
            n_tokens: 1,
            token_dim: obs_dim,
        };

        // Host
        let mut host_state = <RegionalBrain as Brain>::init_state(&weights);
        for _ in 0..n_warmup {
            let (_, s) = <RegionalBrain as Brain>::forward(&weights, host_state, &input);
            host_state = s;
        }
        let t = Instant::now();
        for _ in 0..n_iters {
            let (_, s) = <RegionalBrain as Brain>::forward(&weights, host_state, &input);
            host_state = s;
        }
        let dt_host = t.elapsed();
        let per_iter_host = dt_host.as_nanos() as f64 / n_iters as f64 / 1000.0;
        eprintln!("  host     wall = {dt_host:?}  per-iter = {per_iter_host:.2} µs");

        // Resident
        let cache = match RegionalResidentCache::from_weights(&weights) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  resident: cache build failed ({e}) — skipping resident phase");
                return;
            }
        };
        let batch = HipBatch::new();

        for _ in 0..n_warmup {
            let fresh = cache.fresh(&weights).expect("fresh");
            let _ = RegionalBrain::forward_cached_resident(&weights, fresh, &batch, &input)
                .expect("warmup");
        }
        batch.flush().expect("warmup flush");

        let t = Instant::now();
        for _ in 0..n_iters {
            let fresh = cache.fresh(&weights).expect("fresh");
            let _ = RegionalBrain::forward_cached_resident(&weights, fresh, &batch, &input)
                .expect("resident forward");
        }
        batch.flush().expect("flush");
        let dt_res = t.elapsed();
        let per_iter_res = dt_res.as_nanos() as f64 / n_iters as f64 / 1000.0;
        eprintln!("  resident wall = {dt_res:?}  per-iter = {per_iter_res:.2} µs");

        let speedup = per_iter_host / per_iter_res;
        let verdict = if speedup > 1.0 {
            format!("RESIDENT WINS by {speedup:.2}×")
        } else {
            format!("HOST WINS — resident is {:.2}× slower", 1.0 / speedup)
        };
        eprintln!("  → {verdict}\n");
    }
}
