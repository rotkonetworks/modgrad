//! Brain-shaped residency ceiling bench.
//!
//! Chains many `LinearResident::forward` calls back-to-back with all
//! activations as device-resident `GpuVec::Hip`. The shapes mirror a
//! single forward of the maze `RegionalBrain` at `--multiscale`:
//!
//!   - 4 "small" matvecs (64 → 64)   — per-region kv_proj / q_proj /
//!                                      mha_in_proj / mha_out_proj.
//!   - 8 "medium" matvecs (128 → 64) — connection synapses (concat
//!                                      of upstream region outputs →
//!                                      downstream region d_input).
//!   - 1 "wide"  matvec  (64 → 256)  — output_proj (synch_size_out →
//!                                      out_dims).
//!
//! These shape counts are illustrative — a real maze brain forward
//! issues several hundred matvecs per outer-tick. The point of this
//! bench is the *ceiling*: how high does `rocm-smi` go when activations
//! never leave the device through a brain-shaped pipeline of matvecs?
//!
//! Why N_ITERS is high. Per `memory/feedback_gpu_perf_reality.md`:
//!   - matmul beats matvec by ~85× on this part (RX 7600M XT, gfx1102),
//!     so saturating utilization with matvec-only requires sustained
//!     load to keep the SMU at boost clocks;
//!   - the SMU only pins to its boost rail after ~10 s of sustained
//!     load — bursts under that window silently run at 0.52× clock.
//!   We therefore size the resident loop to run ≥15 s by default so
//!   the user has a real window to alt-tab to `rocm-smi --showuse`
//!   and observe steady-state utilization, not a transient spike.
//!
//! Run:
//!   cargo run --release --features rocm \
//!        -p modgrad-compute --example bench_brain_resident -- 150000
//!
//! In another terminal during the run:
//!   watch -n 0.5 rocm-smi --showuse
//!
//! Args: `n_iters` (default 150_000 — sized so the resident phase
//! runs ~15 s on RX 7600M XT at ~105 µs/iter, giving the SMU time
//! to pin to boost rails AND giving the user a window to alt-tab
//! and observe `rocm-smi --showuse`).

#[cfg(feature = "rocm")]
fn main() {
    use modgrad_compute::backend::GpuVec;
    use modgrad_compute::neuron::{Linear, LinearResident, SimpleRng};
    use modgrad_device::backend::HipBatch;
    use std::time::Instant;

    // ── Shape constants — pinned, brain-shaped ───────────────────
    const SMALL_DIM: usize = 64;     // kv_proj / q_proj / mha_*_proj
    const MED_IN: usize = 128;       // concat(upstream region outputs)
    const MED_OUT: usize = 64;       // d_input of downstream region
    const WIDE_IN: usize = 64;       // synch_size_out
    const WIDE_OUT: usize = 256;     // out_dims

    const N_SMALL: usize = 4;
    const N_MEDIUM: usize = 8;
    const N_WIDE: usize = 1;

    let args: Vec<String> = std::env::args().collect();
    let n_iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(150_000);

    let total_matvecs_per_iter = N_SMALL + N_MEDIUM + N_WIDE;
    eprintln!("bench_brain_resident: chain = {N_SMALL}×({SMALL_DIM}→{SMALL_DIM}) \
        + {N_MEDIUM}×({MED_IN}→{MED_OUT}) + {N_WIDE}×({WIDE_IN}→{WIDE_OUT})");
    eprintln!("                      {total_matvecs_per_iter} matvecs / iter,  n_iters = {n_iters}");
    eprintln!("                      total dispatches (resident path) = {}",
        n_iters * total_matvecs_per_iter);

    // ── Build the synthetic Linears ──────────────────────────────
    let smalls: Vec<Linear> =
        (0..N_SMALL).map(|_| Linear::new(SMALL_DIM, SMALL_DIM)).collect();
    let mediums: Vec<Linear> =
        (0..N_MEDIUM).map(|_| Linear::new(MED_IN, MED_OUT)).collect();
    let wide = Linear::new(WIDE_IN, WIDE_OUT);

    // ── Seed inputs (host-side, used to prime both paths) ────────
    let mut rng = SimpleRng::new(0xB1A1_BEEF);
    let host_x_small: Vec<f32> =
        (0..SMALL_DIM).map(|_| rng.next_normal()).collect();
    let host_x_med: Vec<f32> =
        (0..MED_IN).map(|_| rng.next_normal()).collect();

    // ── HOST-ONLY PATH ───────────────────────────────────────────
    // Two-buffer ping-pong per shape so back-to-back forward_into's
    // can chain output-of-N → input-of-N+1 without re-allocation.
    eprintln!("\n[host] running {n_iters} brain-equivalent iters …");
    let mut h_small_a = host_x_small.clone();
    let mut h_small_b = vec![0.0f32; SMALL_DIM];
    let h_med_in = host_x_med.clone();
    let mut h_med_out = vec![0.0f32; MED_OUT];
    let mut h_wide_out = vec![0.0f32; WIDE_OUT];

    let t_host_start = Instant::now();
    for _ in 0..n_iters {
        // small chain: a → b → a → b → a   (output ends in h_small_a)
        for (i, lin) in smalls.iter().enumerate() {
            if i % 2 == 0 {
                lin.forward_into(&h_small_a, &mut h_small_b);
            } else {
                lin.forward_into(&h_small_b, &mut h_small_a);
            }
        }
        // medium chain: each reads persistent h_med_in (128) → h_med_out (64).
        // Real-world the 128 input would be a concat of upstream activations;
        // for ceiling-bench purposes we keep h_med_in primed once.
        for lin in mediums.iter() {
            lin.forward_into(&h_med_in, &mut h_med_out);
        }
        // wide tail: 64 → 256
        wide.forward_into(&h_small_a, &mut h_wide_out);
    }
    let dt_host = t_host_start.elapsed();
    let per_iter_host_us = dt_host.as_nanos() as f64 / n_iters as f64 / 1000.0;
    let per_matvec_host_us = per_iter_host_us / total_matvecs_per_iter as f64;
    eprintln!("[host]     wall   = {dt_host:?}");
    eprintln!("[host]     per-iter = {per_iter_host_us:.2} µs   per-matvec = {per_matvec_host_us:.2} µs");

    // ── RESIDENT PATH ────────────────────────────────────────────
    // Upload weights once.
    let smalls_dev: Vec<LinearResident> = match smalls
        .iter()
        .map(LinearResident::from_linear)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!("LinearResident init failed for smalls: {e}");
            eprintln!("(Likely no HIP runtime available — bench requires GPU.)");
            return;
        }
    };
    let mediums_dev: Vec<LinearResident> = match mediums
        .iter()
        .map(LinearResident::from_linear)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!("LinearResident init failed for mediums: {e}");
            return;
        }
    };
    let wide_dev = match LinearResident::from_linear(&wide) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("LinearResident init failed for wide: {e}");
            return;
        }
    };

    // Pre-allocate all on-device activation buffers. The chain layout:
    //   small_a (64) ←─┐
    //                  ├── ping-pong for small chain
    //   small_b (64) ←─┘
    //   med_in  (128) — primed once, reused for all 8 medium matvecs
    //   med_out (64)  — overwritten by each medium matvec
    //   wide_out (256) — final tail
    let mut sa_dev = GpuVec::try_hip(SMALL_DIM).expect("alloc small_a");
    let mut sb_dev = GpuVec::try_hip(SMALL_DIM).expect("alloc small_b");
    let mut min_dev = GpuVec::try_hip(MED_IN).expect("alloc med_in");
    let mut mout_dev = GpuVec::try_hip(MED_OUT).expect("alloc med_out");
    let mut wout_dev = GpuVec::try_hip(WIDE_OUT).expect("alloc wide_out");
    sa_dev.copy_from(&host_x_small);
    min_dev.copy_from(&host_x_med);

    // The HipBatch RAII guard is the only thing that can guarantee the
    // queue gets drained — see `memory/feedback_hip_queue_overflow.md`
    // for the incident that motivated it. LinearResident::forward takes
    // `&HipBatch` so calling it without one is a compile error. Drop
    // runs a final sync; intermediate syncs every 256 dispatches.
    let batch = HipBatch::new();

    // Warmup — first calls may JIT or prime hipBLAS handles.
    eprintln!("\n[resident] warmup …");
    for _ in 0..16 {
        for (i, lin) in smalls_dev.iter().enumerate() {
            if i % 2 == 0 {
                lin.forward(&batch, &sa_dev, &mut sb_dev).expect("warmup small a→b");
            } else {
                lin.forward(&batch, &sb_dev, &mut sa_dev).expect("warmup small b→a");
            }
        }
        for lin in mediums_dev.iter() {
            lin.forward(&batch, &min_dev, &mut mout_dev).expect("warmup medium");
        }
        wide_dev.forward(&batch, &sa_dev, &mut wout_dev).expect("warmup wide");
    }
    batch.flush().expect("warmup flush");

    eprintln!("[resident] running {n_iters} brain-equivalent iters …");
    eprintln!("[resident] watch `rocm-smi --showuse` in another terminal NOW");
    let t_res_start = Instant::now();
    for _ in 0..n_iters {
        // small chain: sa → sb → sa → sb → sa  (output ends in sa)
        for (i, lin) in smalls_dev.iter().enumerate() {
            if i % 2 == 0 {
                lin.forward(&batch, &sa_dev, &mut sb_dev).expect("small a→b");
            } else {
                lin.forward(&batch, &sb_dev, &mut sa_dev).expect("small b→a");
            }
        }
        // medium chain: min (128, primed) → mout (64), eight times.
        for lin in mediums_dev.iter() {
            lin.forward(&batch, &min_dev, &mut mout_dev).expect("medium");
        }
        // wide tail: sa (64) → wout (256)
        wide_dev.forward(&batch, &sa_dev, &mut wout_dev).expect("wide");
    }
    batch.flush().expect("final flush");
    let dt_res = t_res_start.elapsed();
    let per_iter_res_us = dt_res.as_nanos() as f64 / n_iters as f64 / 1000.0;
    let per_matvec_res_us = per_iter_res_us / total_matvecs_per_iter as f64;

    eprintln!("\n[resident] wall   = {dt_res:?}");
    eprintln!("[resident] per-iter = {per_iter_res_us:.2} µs   per-matvec = {per_matvec_res_us:.2} µs");

    // ── Summary ──────────────────────────────────────────────────
    let speedup = per_iter_host_us / per_iter_res_us;
    eprintln!("\n────────────────────────────────────────────");
    eprintln!("   brain-forward equivalent ({} matvecs):", total_matvecs_per_iter);
    eprintln!("     host     = {per_iter_host_us:>8.2} µs / iter");
    eprintln!("     resident = {per_iter_res_us:>8.2} µs / iter");
    eprintln!("     speedup  = {speedup:.2}× (host vs resident)");
    eprintln!("   per-matvec dispatch (resident) = {per_matvec_res_us:.2} µs");
    eprintln!("   resident phase wall            = {:.2} s", dt_res.as_secs_f64());
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
    eprintln!("bench_brain_resident requires --features rocm; building without GPU is a no-op");
}
