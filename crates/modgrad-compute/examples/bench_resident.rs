//! Compare host-slice `Linear::forward_into` (current default in
//! `RegionalBrain`) vs device-resident `LinearResident::forward` on
//! a tight inner loop. Demonstrates the GPU-residency win:
//!
//!   `cargo run --release -p modgrad-compute --features rocm \
//!        --example bench_resident -- 1024 512 10000`
//!
//! Args: `out_dim in_dim n_iters` (defaults: 1024 512 10000).
//! Watch via: `rocm-smi --showuse` in another terminal.
//!
//! Expected on RX 7600M XT (gfx1102):
//!   - Host path: 4-10% GPU util, dominated by hipMemcpy round-trips
//!   - Resident path: 50%+ GPU util, hipblasSgemv back-to-back

#[cfg(feature = "rocm")]
fn main() {
    use modgrad_compute::backend::GpuVec;
    use modgrad_compute::neuron::{Linear, LinearResident, SimpleRng};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let out_dim: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let in_dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let n_iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    eprintln!("bench: out_dim={out_dim} in_dim={in_dim} n_iters={n_iters}");

    let lin = Linear::new(in_dim, out_dim);
    let mut rng = SimpleRng::new(0xCAFE);
    let host_x: Vec<f32> = (0..in_dim).map(|_| rng.next_normal()).collect();

    // ── Host-slice path (current default — round-trip every call) ──
    let mut host_y = vec![0.0f32; out_dim];
    let t0 = Instant::now();
    for _ in 0..n_iters {
        lin.forward_into(&host_x, &mut host_y);
    }
    let dt_host = t0.elapsed();
    let per_call_host_us = dt_host.as_nanos() as f64 / n_iters as f64 / 1000.0;
    eprintln!("[host]     {n_iters} forwards in {dt_host:?}  ({per_call_host_us:.2} µs/call)");

    // ── Device-resident path (weights uploaded once) ──
    let resident = match LinearResident::from_linear(&lin) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("LinearResident init failed: {e}");
            eprintln!("(Likely no HIP runtime available — bench requires GPU.)");
            return;
        }
    };
    let mut x_dev = GpuVec::try_hip(in_dim).expect("alloc x");
    x_dev.copy_from(&host_x);
    let mut out_dev = GpuVec::try_hip(out_dim).expect("alloc out");

    // The HipBatch RAII guard is the only thing that can guarantee
    // the queue gets drained — see `feedback_hip_queue_overflow.md`
    // for the incident that motivated it. LinearResident::forward
    // takes `&HipBatch` so calling it without one is a compile
    // error. Drop runs a final sync. Default cadence: every 256
    // dispatches.
    let batch = modgrad_device::backend::HipBatch::new();

    // Warmup — first call may JIT-compile or load kernel module.
    for _ in 0..16 {
        resident.forward(&batch, &x_dev, &mut out_dev).expect("warmup");
    }
    batch.flush().expect("warmup flush");

    let t1 = Instant::now();
    for _ in 0..n_iters {
        resident.forward(&batch, &x_dev, &mut out_dev).expect("forward");
    }
    batch.flush().expect("final flush");
    let dt_resident = t1.elapsed();
    let per_call_resident_us = dt_resident.as_nanos() as f64 / n_iters as f64 / 1000.0;
    eprintln!("[resident] {n_iters} forwards in {dt_resident:?}  ({per_call_resident_us:.2} µs/call)");

    let speedup = per_call_host_us / per_call_resident_us;
    eprintln!("\nspeedup: {speedup:.2}× (host vs resident)");
}

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("bench_resident requires --features rocm; building without GPU is a no-op");
}
