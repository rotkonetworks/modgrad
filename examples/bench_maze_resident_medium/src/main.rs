//! Real-shape maze brain residency benchmark — MEDIUM topology
//! (`eight_region_medium_multiscale`, cortical d_model=512). Sister
//! bench to `bench_maze_resident` (small, d_model=64).
//!
//! Why a separate medium bench. On gfx1102 the small variant runs the
//! resident path ~58× SLOWER than the host path even at 99% rocm-smi
//! sustained: the per-dispatch hipBLAS sgemv on a 64×64 matvec is
//! ~600 ns of compute under ~tens of µs of FFI / sync overhead, so
//! the GPU is busy but starved. The medium shape grows the
//! per-dispatch matvec to 512×512 (~64× the flops); the synthetic
//! `bench_brain_resident` proved 8.17× speedup at that regime in a
//! tight chain. This bench runs the *real* `RegionalBrain` chain at
//! that shape — connection synapses + per-region inner CTM dispatches
//! + global-sync + output-projection — to see whether residency wins
//! once the real maze brain sits at d_model=512.
//!
//! Why we clear `router`. `eight_region_medium_multiscale` enables
//! the learned MoS-style router by default. `forward_cached_resident`
//! falls back to host whenever `weights.router.is_some()` (graph.rs
//! ~4087) because the router's matvecs are not in the resident
//! `connection_synapses` inventory. Routing through the resident path
//! is a separate iteration; for an apples-to-apples residency check
//! we drop the router on the floor before building weights. The host
//! baseline runs against the same router-disabled config so the
//! comparison is fair (host vs resident, both no router).
//!
//! Why we ignore the multi-scale dims for `obs_proj`. The encoder
//! contract feeds a flat `tokens` buffer of length ≥ `raw_obs_dim`
//! (= sum of `obs_scale_dims`); each connection synapse slices into
//! that buffer by `obs_scale_slice(scale)`. We wire
//! `obs_scale_dims = [128, 64, 32]` (V4/V2/V1, matching
//! `Encoder::token_dims()` ordering for a vision encoder of this
//! shape — see `eight_region_medium_multiscale` doc comment) and feed
//! `tokens` of length `raw_obs_dim = 224` so all three scale slices
//! land in-bounds. No real encoder is wired; the bench is measuring
//! the brain dispatch chain, not the encoder.
//!
//! Why the iter count. matvec on gfx1102 needs sustained load
//! (≥ ~10 s per `feedback_gpu_perf_reality.md`) before the SMU pins
//! the boost rail. Per-iter on medium is much heavier than small —
//! 8 connection synapses + 8 per-region inner CTM forwards × 2 outer
//! ticks + global-sync + output-proj per tick, with each connection
//! synapse + exit gate + output-proj individually flushing to host
//! inside the iter — so a few-hundred iter count already buys the
//! SMU pin-up budget. Default `N_ITERS = 1500` keeps the resident
//! phase comfortably ≥ 15 s on gfx1102. Pass a larger N as `argv[1]`
//! for a longer steady-state observation.
//!
//! Run:
//!   cargo run --release --features rocm -p bench_maze_resident_medium
//!   cargo run --release --features rocm -p bench_maze_resident_medium -- 3000
//! Watch in another terminal during the resident phase:
//!   watch -n 0.5 rocm-smi --showuse

#[cfg(feature = "rocm")]
fn main() {
    use std::time::Instant;
    use modgrad_ctm::graph::{RegionalBrain, RegionalConfig, RegionalWeights};
    use modgrad_ctm::resident::RegionalResidentCache;
    use modgrad_traits::{Brain, TokenInput};
    use modgrad_device::backend::HipBatch;

    // ── Brain shape ────────────────────────────────────────────
    // Multi-scale obs_scale_dims matches Encoder::token_dims() order
    // for VisualCortex: [V4, V2, V1] (high-level → low-level).
    // raw_obs_dim = sum = 224. Cortical regions run d_model=512.
    let obs_scale_dims: Vec<usize> = vec![128, 64, 32];
    let raw_obs_dim: usize = obs_scale_dims.iter().sum();
    let out_dims: usize = 64;
    let ticks: usize = 2;

    // ── CLI ────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let n_iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_500);
    eprintln!("bench_maze_resident_medium: n_iters = {n_iters}");
    eprintln!("                            shape   = eight_region_medium_multiscale");
    eprintln!("                            scales  = {obs_scale_dims:?} (V4/V2/V1)");
    eprintln!("                            raw_obs = {raw_obs_dim}, out_dims={out_dims}, ticks={ticks}");

    // ── Build brain ────────────────────────────────────────────
    // Disable router before constructing weights — the resident
    // path falls back to host when `router.is_some()` (graph.rs
    // ~4087). For the host baseline to be a fair comparison we use
    // the same router-disabled config in both paths.
    let mut cfg = RegionalConfig::eight_region_medium_multiscale(
        &obs_scale_dims, out_dims, ticks);
    cfg.router = None;

    let weights = RegionalWeights::new(cfg);
    eprintln!("                            params  = {}", weights.n_params());
    eprintln!("                            regions = {}", weights.regions.len());
    eprintln!("                            conns   = {}", weights.connection_synapses.len());

    // Flat tokens of length exactly raw_obs_dim — obs_proj.forward
    // reads in_dim=raw_obs_dim, and connection synapses slice by
    // obs_scale_slice into the same buffer.
    let input = TokenInput {
        tokens: (0..raw_obs_dim)
            .map(|i| (i as f32 * 0.013 - 0.21).sin())
            .collect(),
        n_tokens: 1,
        token_dim: raw_obs_dim,
    };

    // ── Host path baseline ─────────────────────────────────────
    // RegionalBrain::forward currently constructs a fresh
    // RegionalState::new(weights) every call (graph.rs KNOWN
    // REGRESSION) — we still thread the returned state to mirror
    // real call-site shape.
    eprintln!("\n[host] warmup …");
    let mut host_state = <RegionalBrain as Brain>::init_state(&weights);
    for _ in 0..4 {
        let (_, s) = <RegionalBrain as Brain>::forward(&weights, host_state, &input);
        host_state = s;
    }

    eprintln!("[host] running {n_iters} forwards …");
    let t_host = Instant::now();
    for _ in 0..n_iters {
        let (_, s) = <RegionalBrain as Brain>::forward(&weights, host_state, &input);
        host_state = s;
    }
    let dt_host = t_host.elapsed();
    let per_iter_host_us = dt_host.as_nanos() as f64 / n_iters as f64 / 1000.0;
    eprintln!("[host] wall    = {dt_host:?}");
    eprintln!("[host] per-iter = {per_iter_host_us:.2} µs");

    // ── Resident path ──────────────────────────────────────────
    let cache = match RegionalResidentCache::from_weights(&weights) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("RegionalResidentCache::from_weights failed: {e}");
            eprintln!("(Likely no HIP runtime — bench requires GPU.)");
            return;
        }
    };
    eprintln!("\n[resident] cache built: {} top-level Linears", cache.n_linears());

    let batch = HipBatch::new();

    eprintln!("\n>>> WATCH `rocm-smi --showuse` NOW <<<");
    eprintln!("    Resident phase starts in 2 s …");
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Warmup: pins SMU + primes hipBLAS handles.
    eprintln!("[resident] warmup …");
    for _ in 0..4 {
        let fresh = cache.fresh(&weights).expect("fresh");
        let _ = RegionalBrain::forward_cached_resident(&weights, fresh, &batch, &input)
            .expect("warmup forward_cached_resident");
    }
    batch.flush().expect("warmup flush");

    eprintln!("[resident] running {n_iters} forwards …");
    let t_res = Instant::now();
    for _ in 0..n_iters {
        let fresh = cache.fresh(&weights).expect("fresh");
        let _ = RegionalBrain::forward_cached_resident(&weights, fresh, &batch, &input)
            .expect("forward_cached_resident");
    }
    batch.flush().expect("final flush");
    let dt_res = t_res.elapsed();
    let per_iter_res_us = dt_res.as_nanos() as f64 / n_iters as f64 / 1000.0;
    eprintln!("[resident] wall    = {dt_res:?}");
    eprintln!("[resident] per-iter = {per_iter_res_us:.2} µs");

    // ── Summary ────────────────────────────────────────────────
    let speedup = per_iter_host_us / per_iter_res_us;
    eprintln!("\n────────────────────────────────────────────");
    eprintln!("   real maze brain (eight_region_medium_multiscale, d_model=512, ticks={ticks}):");
    eprintln!("     host     = {per_iter_host_us:>10.2} µs / iter");
    eprintln!("     resident = {per_iter_res_us:>10.2} µs / iter");
    eprintln!("     speedup  = {speedup:.2}× (host vs resident)");
    eprintln!("   resident phase wall = {:.2} s", dt_res.as_secs_f64());
    if dt_res.as_secs_f64() < 15.0 {
        eprintln!("   [warn] resident phase < 15 s — re-run with larger N_ITERS to give the SMU");
        eprintln!("          time to pin to boost (see feedback_gpu_perf_reality.md).");
    } else {
        eprintln!("   [ok]   resident phase >= 15 s — long enough for SMU pin-up + rocm-smi window.");
    }
    if speedup > 1.0 {
        eprintln!("   [win]  resident faster than host on the medium shape.");
    } else {
        eprintln!("   [loss] even at d_model=512 the per-dispatch host bounces in");
        eprintln!("          forward_cached_resident dominate — eliminating the");
        eprintln!("          activation/MHA bounces is the next limit.");
    }
    eprintln!("────────────────────────────────────────────");
}

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("bench_maze_resident_medium requires --features rocm; building without GPU is a no-op");
}
