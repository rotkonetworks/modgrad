//! Parity tests for `Op::SuperLinearBwdDwResident` / `SuperLinearBwdDxResident`.
//!
//! Phase 3 of the brain-on-GPU plan: the per-neuron NLM backward was
//! the single hottest training-loop bucket (36.7%) because the
//! existing `Op::SuperLinearBwdDw` / `SuperLinearBwdDx` only had CPU
//! dispatch. These tests validate the new strided-batched-SGEMM ROCm
//! kernels (`super_linear_bwd_dw_resident_batched_f32` etc.) match
//! the CPU reference to within fp32 noise.
//!
//! No-op when the ROCm runtime isn't available, like the rest of the
//! resident-op test suite.

#![cfg(feature = "rocm")]

use modgrad_device::backend::{Backend, BackendError, HipBuffer, Op, RocmBackend};

const ABS_TOL: f32 = 1e-3;
const REL_TOL: f32 = 1e-3;

fn assert_close(a: &[f32], b: &[f32], name: &str) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let scale = av.abs().max(bv.abs()).max(1.0);
        if diff > ABS_TOL && diff / scale > REL_TOL {
            panic!("{name}[{i}] divergence: gpu={av} cpu={bv} (|Δ| = {diff})");
        }
    }
}

fn upload(host: &[f32]) -> Result<HipBuffer, BackendError> {
    let buf = HipBuffer::new(host.len() * 4)?;
    buf.copy_from_host(host)?;
    Ok(buf)
}

fn alloc_out(n: usize) -> Result<HipBuffer, BackendError> {
    HipBuffer::new(n * 4)
}

// ─── Host references (mirror cpu.rs::super_linear_bwd_dw / dx) ──────

fn host_bwd_dw(
    d_out: &[f32], trace: &[f32], d_weights: &mut [f32],
    n_neurons: usize, in_per: usize, out_per: usize,
) {
    for n in 0..n_neurons {
        let t = &trace[n * in_per..(n + 1) * in_per];
        let w_off = n * out_per * in_per;
        let o_off = n * out_per;
        for i in 0..out_per {
            let d = d_out[o_off + i];
            for k in 0..in_per {
                d_weights[w_off + i * in_per + k] += d * t[k];
            }
        }
    }
}

fn host_bwd_dx(
    d_out: &[f32], weights: &[f32], d_trace: &mut [f32],
    n_neurons: usize, in_per: usize, out_per: usize,
) {
    for n in 0..n_neurons {
        let w_off = n * out_per * in_per;
        let o_off = n * out_per;
        let t_off = n * in_per;
        for k in 0..in_per {
            let mut acc = 0.0f32;
            for i in 0..out_per {
                acc += weights[w_off + i * in_per + k] * d_out[o_off + i];
            }
            d_trace[t_off + k] = acc;
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[test]
fn super_linear_bwd_dw_resident_matches_cpu() -> Result<(), BackendError> {
    let Some(backend) = RocmBackend::try_new() else {
        eprintln!("super_linear_bwd_dw_resident: no ROCm runtime; skipping");
        return Ok(());
    };

    let n_neurons = 32;
    let in_per = 24;
    let out_per = 16;

    // Deterministic ramp/sine inputs — reproducible across runs.
    let d_out: Vec<f32> = (0..n_neurons * out_per)
        .map(|i| ((i as f32) * 0.013 - 0.5).sin())
        .collect();
    let trace: Vec<f32> = (0..n_neurons * in_per)
        .map(|i| ((i as f32) * 0.021 + 0.3).cos())
        .collect();
    // Pre-existing dW so we test accumulation (beta=1.0).
    let mut d_w_init: Vec<f32> = (0..n_neurons * out_per * in_per)
        .map(|i| ((i as f32) * 0.001).sin() * 0.05)
        .collect();
    let mut d_w_host = d_w_init.clone();
    host_bwd_dw(&d_out, &trace, &mut d_w_host, n_neurons, in_per, out_per);

    let d_out_dev = upload(&d_out)?;
    let trace_dev = upload(&trace)?;
    let dw_dev = alloc_out(n_neurons * out_per * in_per)?;
    dw_dev.copy_from_host(&d_w_init)?;

    let mut op = Op::SuperLinearBwdDwResident {
        d_out_dev:    d_out_dev.device_ptr() as *const f32,
        trace_dev:    trace_dev.device_ptr() as *const f32,
        d_weight_dev: dw_dev.device_ptr() as *mut f32,
        n_neurons, in_per, out_per,
    };
    backend.dispatch(&mut op)?;

    let mut d_w_gpu = vec![0.0f32; n_neurons * out_per * in_per];
    dw_dev.copy_to_host(&mut d_w_gpu);
    assert_close(&d_w_gpu, &d_w_host, "bwd_dw");
    let _ = &mut d_w_init;
    Ok(())
}

#[test]
fn super_linear_bwd_dx_resident_matches_cpu() -> Result<(), BackendError> {
    let Some(backend) = RocmBackend::try_new() else {
        eprintln!("super_linear_bwd_dx_resident: no ROCm runtime; skipping");
        return Ok(());
    };

    let n_neurons = 32;
    let in_per = 24;
    let out_per = 16;

    let d_out: Vec<f32> = (0..n_neurons * out_per)
        .map(|i| ((i as f32) * 0.011 + 0.1).sin() * 0.5)
        .collect();
    let weights: Vec<f32> = (0..n_neurons * out_per * in_per)
        .map(|i| ((i as f32) * 0.007 - 0.3).cos() * 0.1)
        .collect();
    let mut d_trace_host = vec![0.0f32; n_neurons * in_per];
    host_bwd_dx(&d_out, &weights, &mut d_trace_host, n_neurons, in_per, out_per);

    let d_out_dev = upload(&d_out)?;
    let weight_dev = upload(&weights)?;
    let dt_dev = alloc_out(n_neurons * in_per)?;

    let mut op = Op::SuperLinearBwdDxResident {
        d_out_dev:   d_out_dev.device_ptr() as *const f32,
        weight_dev:  weight_dev.device_ptr() as *const f32,
        d_trace_dev: dt_dev.device_ptr() as *mut f32,
        n_neurons, in_per, out_per,
    };
    backend.dispatch(&mut op)?;

    let mut d_trace_gpu = vec![0.0f32; n_neurons * in_per];
    dt_dev.copy_to_host(&mut d_trace_gpu);
    assert_close(&d_trace_gpu, &d_trace_host, "bwd_dx");
    Ok(())
}
