//! Real-shape maze brain residency benchmark. Runs
//! `RegionalBrain::forward_cached_resident` on `eight_region_small`
//! for N iters under one `HipBatch`; reports per-iter timing for the
//! host and resident paths and tells the user to watch
//! `rocm-smi --showuse` during the resident phase.
//!
//! See `bench_brain_resident` (in modgrad-compute/examples) for the
//! synthetic 13-matvec ceiling on this hardware (8.17× speedup,
//! 98–100% rocm-smi sustained). This bench validates the same
//! architecture on the real `RegionalBrain` shape — many more
//! matvecs per iter, plus per-region inner CTM dispatches and the
//! global-sync / output-projection tail.
//!
//! Why the iter count is high. matvec on gfx1102 needs sustained
//! load (≥ ~10 s per `feedback_gpu_perf_reality.md`) before the SMU
//! pins the boost rail; the default `N_ITERS` is sized so the
//! resident phase runs ≥ 15 s and the user has a real window to
//! alt-tab to `rocm-smi --showuse` and observe steady-state
//! utilization rather than a transient spike.
//!
//! Run:
//!   cargo run --release --features rocm -p bench_maze_resident
//!   cargo run --release --features rocm -p bench_maze_resident -- 80000
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
    // Mirrors the in-tree `forward_cached_resident_matches_host`
    // test fixture (graph.rs), so the resident path here exercises
    // the same code that has the bit-for-bit (1e-3) parity check.
    //
    //   token_dim × n_tokens = 64 floats of input per call;
    //   raw_obs_dim = 16 (obs_proj reads first 16, connections
    //   slice into [s..s+len] within the same buffer);
    //   out_dims = 64 (output_proj fan-out);
    //   ticks    = 2 (outer ticks; each fires the full 8-region
    //               connection synapse + per-region inner CTM
    //               + global-sync + output-proj chain).
    let token_dim: usize = 16;
    let n_tokens: usize = 4;
    let out_dims: usize = 64;
    let ticks: usize = 2;

    // ── CLI ────────────────────────────────────────────────────
    // Default sized for ≥ 60 s resident phase on RX 7600M XT.
    // Empirically (2026-04-25, gfx1102) `forward_cached_resident`
    // on `eight_region_small` runs ~30 ms / iter — the per-iter
    // chain is dispatch-bound (8 connection synapses + 8 per-region
    // inner CTM forwards × 2 outer ticks + global-sync + output-
    // proj per tick), with each connection synapse + exit gate +
    // output-proj individually flushing to host inside the iter.
    // 2000 iters → ~60 s resident wall, comfortable rocm-smi window
    // and ≥ 15 s SMU pin-up budget. Pass a larger N as `argv[1]`
    // for a longer steady-state observation.
    let args: Vec<String> = std::env::args().collect();
    let n_iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2_000);
    eprintln!("bench_maze_resident: n_iters = {n_iters}");
    eprintln!("                     shape    = eight_region_small \
        (obs_dim={token_dim}, out_dims={out_dims}, ticks={ticks})");

    // ── Build brain + input ────────────────────────────────────
    let cfg = RegionalConfig::eight_region_small(token_dim, out_dims, ticks);
    let weights = RegionalWeights::new(cfg);
    eprintln!("                     params   = {}", weights.n_params());
    eprintln!("                     regions  = {}", weights.regions.len());
    eprintln!("                     conns    = {}", weights.connection_synapses.len());

    let input = TokenInput {
        tokens: (0..n_tokens * token_dim)
            .map(|i| (i as f32 * 0.013 - 0.21).sin())
            .collect(),
        n_tokens,
        token_dim,
    };

    // ── Host path baseline ─────────────────────────────────────
    // Brain::forward is value-shaped (state in, state out), and
    // RegionalBrain's impl currently ignores the passed-in state
    // (graph.rs:4351 KNOWN REGRESSION) — it constructs a fresh
    // RegionalState::new(weights) each call. We thread the returned
    // state anyway to mirror real call-site shape.
    eprintln!("\n[host] warmup …");
    let mut host_state = <RegionalBrain as Brain>::init_state(&weights);
    for _ in 0..16 {
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
    for _ in 0..16 {
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
    eprintln!("   real maze brain (eight_region_small, ticks={ticks}):");
    eprintln!("     host     = {per_iter_host_us:>8.2} µs / iter");
    eprintln!("     resident = {per_iter_res_us:>8.2} µs / iter");
    eprintln!("     speedup  = {speedup:.2}× (host vs resident)");
    eprintln!("   resident phase wall = {:.2} s", dt_res.as_secs_f64());
    if dt_res.as_secs_f64() < 15.0 {
        eprintln!("   [warn] resident phase < 15 s — re-run with larger N_ITERS to give the SMU");
        eprintln!("          time to pin to boost (see feedback_gpu_perf_reality.md).");
    } else {
        eprintln!("   [ok]   resident phase >= 15 s — long enough for SMU pin-up + rocm-smi window.");
    }
    eprintln!("────────────────────────────────────────────");
}

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("bench_maze_resident requires --features rocm; building without GPU is a no-op");
}
