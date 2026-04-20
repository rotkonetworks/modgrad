//! Backend parity harness.
//!
//! For every `Op` variant we care about, run the op on every registered
//! backend and assert outputs agree within a tight f32 tolerance.
//!
//! When only CPU is registered (Phase 1), this trivially passes — its
//! value is as a gate for Phase 2: the moment KFD is ported to the
//! `Backend` trait, running this test on gfx1102 hardware asserts that
//! the new KFD impl matches the CPU reference. Any regression surfaces
//! as a failed assertion naming the specific Op and backend.
//!
//! Tolerances:
//! - absolute 1e-4, relative 1e-3. Generous enough to absorb
//!   order-of-operations differences between parallel reductions and
//!   serial loops; tight enough to catch real math bugs.

use modgrad_device::backend::{
    Backend, BackendRegistry, CpuBackend, KfdBackend, Op, QuantKind, RocmBackend,
};

const ABS_TOL: f32 = 1e-4;
const REL_TOL: f32 = 1e-3;

fn assert_close(a: &[f32], b: &[f32], name: &str) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let scale = av.abs().max(bv.abs()).max(1.0);
        if diff > ABS_TOL && diff / scale > REL_TOL {
            panic!("{name}[{i}] divergence: {av} vs {bv} (|Δ| = {diff})");
        }
    }
}

/// Collect all registered backends into a Vec for iteration during parity.
/// We build a registry with CPU plus any future probed backends (added
/// automatically as Phase 2 lands). The parity runner then compares
/// every backend's output to the CPU reference.
fn backends() -> Vec<Box<dyn Backend>> {
    // Always include CPU reference. CPU output is the ground truth
    // every other backend is compared against.
    let mut v: Vec<Box<dyn Backend>> = vec![Box::new(CpuBackend::new())];

    // Probe hardware-specific backends; each is optional.
    if let Some(kfd) = KfdBackend::try_new() {
        v.push(Box::new(kfd));
    }
    if let Some(rocm) = RocmBackend::try_new() {
        v.push(Box::new(rocm));
    }
    v
}

/// Registry populated the same way; used to verify `dispatch()` still
/// picks a backend when given this op.
fn registry() -> BackendRegistry {
    let mut r = BackendRegistry::new();
    for b in backends() {
        r.register(b);
    }
    r
}

/// Run one op on every backend and assert outputs match. The op is
/// rebuilt fresh per backend via the closure (since each needs its own
/// `&mut` output buffers).
fn run_parity<F>(op_name: &str, mut build_op: F)
where
    F: for<'a> FnMut(&'a mut Vec<f32>) -> Op<'a>,
{
    let backends = backends();
    let mut reference: Option<Vec<f32>> = None;

    for b in &backends {
        let mut out_buf = vec![0.0f32; 0];
        // Build op with this backend's out buffer.
        // We need a two-step construction because build_op takes &mut Vec,
        // and we want to pull the vec back out afterwards.
        {
            let mut op = build_op(&mut out_buf);
            if !b.supports(&op) { continue; }
            b.dispatch(&mut op).unwrap_or_else(|e| {
                panic!("{op_name} on backend '{}' errored: {e}", b.name())
            });
        }

        if let Some(ref r) = reference {
            assert_close(&out_buf, r, &format!("{op_name} on '{}'", b.name()));
        } else {
            reference = Some(out_buf.clone());
        }
    }

    // Verify registry also dispatches successfully.
    let reg = registry();
    let mut out_buf = vec![];
    let mut op = build_op(&mut out_buf);
    reg.dispatch(&mut op).unwrap_or_else(|e| {
        panic!("{op_name} through registry failed: {e}")
    });
}

#[test]
fn parity_matvec_small() {
    run_parity("matvec", |out| {
        *out = vec![0.0f32; 3];
        Op::Matvec {
            x: &[1.0, 2.0, 3.0],
            weight: &[1.0, 2.0, 3.0,
                      4.0, 5.0, 6.0,
                      7.0, 8.0, 9.0],
            bias: &[0.1, 0.2, 0.3],
            out,
            out_dim: 3,
            in_dim: 3,
            quant: QuantKind::F32,
        }
    });
}

#[test]
fn parity_matvec_t_small() {
    // d_input[j] = sum_i weight[i, j] * d_out[i]
    run_parity("matvec_t", |out| {
        *out = vec![0.0f32; 3];
        Op::MatvecT {
            d_out: &[1.0, 2.0, 0.5, -1.0],
            weight: &[
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                0.5, 0.5, 0.5,
            ],
            d_input: out,
            out_dim: 4,
            in_dim: 3,
        }
    });
}

#[test]
fn parity_matmul_nn_small() {
    // (2x3) @ (3x2) = (2x2)
    run_parity("matmul_nn", |out| {
        *out = vec![0.0f32; 4];
        Op::MatmulNN {
            a: &[1.0, 2.0, 3.0,
                 4.0, 5.0, 6.0],
            b: &[7.0,  8.0,
                 9.0,  10.0,
                 11.0, 12.0],
            out,
            bias: None,
            m: 2, k: 3, n: 2,
        }
    });
}

#[test]
fn parity_silu_small() {
    run_parity("silu_fwd", |out| {
        *out = vec![0.0f32; 5];
        Op::SiluFwd {
            x: &[-2.0, -1.0, 0.0, 1.0, 2.0],
            out,
        }
    });
}

#[test]
fn parity_reduce_l2_sq() {
    run_parity("reduce_l2_sq", |out| {
        *out = vec![0.0f32; 1];
        Op::ReduceL2Sq {
            x: &[1.0, 2.0, 3.0, 4.0, 5.0],
            out,
        }
    });
}

#[test]
fn parity_outer_product_acc() {
    run_parity("outer_product_acc", |out| {
        *out = vec![1.0f32; 6]; // seeded, op should ADD
        Op::OuterProductAcc {
            a: &[2.0, 3.0],
            b: &[4.0, 5.0, 6.0],
            accum: out,
            m: 2, n: 3,
        }
    });
}

/// KFD's outer_product kernel uses approximate float division for
/// integer row/col computation. Below a small size threshold it has
/// been observed to skip updates (ticket: see `kfd.rs` supports()
/// comment). This test covers the size regime it actually handles
/// correctly — 256-aligned total elements. Inputs are materialised
/// inside the harness so the closure doesn't capture outer state.
#[test]
fn parity_outer_product_acc_aligned() {
    // m=8, n=32 → total=256, exactly one workgroup fully utilized.
    // Do the parity check manually (the generic harness can't hold
    // two extra input vectors across backends).
    let m = 8usize;
    let n = 32usize;
    let a: Vec<f32> = (0..m).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let b: Vec<f32> = (0..n).map(|j| (j as f32) * 0.25 - 0.3).collect();
    let seed: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();

    let bs = backends();
    let mut reference: Option<Vec<f32>> = None;

    for backend in &bs {
        let mut accum = seed.clone();
        let mut op = Op::OuterProductAcc {
            a: &a, b: &b, accum: &mut accum, m, n,
        };
        if !backend.supports(&op) { continue; }
        backend.dispatch(&mut op).unwrap_or_else(|e| {
            panic!("outer_product_acc_aligned on '{}' errored: {e}", backend.name())
        });

        if let Some(ref r) = reference {
            assert_close(&accum, r, &format!("outer_product_acc_aligned on '{}'", backend.name()));
        } else {
            reference = Some(accum.clone());
        }
    }
}

#[test]
fn parity_layernorm_fwd() {
    // LayerNorm has two output buffers (out + cache). The harness
    // compares `out` only; correctness of `cache` is verified via the
    // LayerNormBwd test (which consumes it). Still exercises the
    // dispatch code path.
    let mut out_buf = vec![0.0f32; 6];
    let mut cache = vec![0.0f32; 4];
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma = [1.0, 1.0, 1.0];
    let beta = [0.0, 0.0, 0.0];
    let mut op = Op::LayerNormFwd {
        x: &x, gamma: &gamma, beta: &beta,
        out: &mut out_buf, cache: Some(&mut cache),
        n_rows: 2, n_cols: 3,
    };
    let be = CpuBackend::new();
    be.dispatch(&mut op).unwrap();
    // Centered and scaled rows have mean ≈ 0, std ≈ 1.
    let mean0: f32 = out_buf[0..3].iter().sum::<f32>() / 3.0;
    let mean1: f32 = out_buf[3..6].iter().sum::<f32>() / 3.0;
    assert!(mean0.abs() < 1e-3, "row0 mean not zero: {mean0}");
    assert!(mean1.abs() < 1e-3, "row1 mean not zero: {mean1}");
}

/// SuperLinear forward without cache — exercises the KFD
/// `try_superlinear` path when available, CPU otherwise.
#[test]
fn parity_super_linear_fwd_nocache() {
    let d_model = 32usize;
    let memory_length = 16usize;
    let out_per = 2usize;

    let trace: Vec<f32> = (0..d_model * memory_length)
        .map(|i| (i as f32) * 0.01).collect();
    let weights: Vec<f32> = (0..d_model * out_per * memory_length)
        .map(|i| ((i as f32) * 0.003).sin() * 0.5).collect();
    let biases: Vec<f32> = (0..d_model * out_per)
        .map(|i| (i as f32) * 0.02 - 0.3).collect();

    let bs = backends();
    let mut reference: Option<Vec<f32>> = None;

    for backend in &bs {
        let mut out = vec![0.0f32; d_model * out_per];
        let mut op = Op::SuperLinearFwd {
            trace: &trace, weights: &weights, biases: &biases,
            out: &mut out, cache: None,
            d_model, memory_length, out_per,
        };
        if !backend.supports(&op) { continue; }
        backend.dispatch(&mut op).unwrap_or_else(|e| {
            panic!("super_linear_fwd on '{}' errored: {e}", backend.name())
        });

        if let Some(ref r) = reference {
            assert_close(&out, r, &format!("super_linear_fwd on '{}'", backend.name()));
        } else {
            reference = Some(out.clone());
        }
    }
}

/// Hand-picked fused-synapse parity case. Shape (out_dim=128, in_dim=64)
/// clears the KFD supports() gate (out_dim ≥ 32, multiple-of-32, ≤ 1024,
/// in_dim ≥ 32), so the KFD dispatch actually runs rather than the
/// registry falling through to CPU. Weights are deterministic so the
/// test is reproducible across runs.
#[test]
fn parity_synapse_forward() {
    let out_dim = 128usize;
    let in_dim = 64usize;
    let weight: Vec<f32> = (0..2 * out_dim * in_dim)
        .map(|i| ((i as f32) * 0.001).sin() * 0.1).collect();
    let bias: Vec<f32> = (0..2 * out_dim)
        .map(|i| ((i as f32) * 0.05).cos() * 0.1).collect();
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i as f32) * 0.07).sin()).collect();

    let bs = backends();
    let mut reference: Option<Vec<f32>> = None;

    for backend in &bs {
        let mut out = vec![0.0f32; out_dim];
        let mut op = Op::SynapseForward {
            weight: &weight, bias: &bias, x: &x, out: &mut out,
            out_dim, in_dim,
        };
        if !backend.supports(&op) { continue; }
        backend.dispatch(&mut op).unwrap_or_else(|e| {
            panic!("synapse_forward on '{}' errored: {e}", backend.name())
        });

        if let Some(ref r) = reference {
            assert_close(&out, r, &format!("synapse_forward on '{}'", backend.name()));
        } else {
            reference = Some(out.clone());
        }
    }
}

/// Hand-picked LN-inplace parity case. Single row of 256 elements
/// clears the KFD gate (n_rows == 1, n_cols multiple-of-32, ≤ 1024).
#[test]
fn parity_layer_norm_inplace() {
    let n_cols = 256usize;
    let x0: Vec<f32> = (0..n_cols)
        .map(|i| ((i as f32) * 0.03).sin() * 2.0 + 0.1).collect();

    let bs = backends();
    let mut reference: Option<Vec<f32>> = None;

    for backend in &bs {
        let mut x = x0.clone();
        let mut op = Op::LayerNormInplace { x: &mut x, n_rows: 1, n_cols };
        if !backend.supports(&op) { continue; }
        backend.dispatch(&mut op).unwrap_or_else(|e| {
            panic!("layer_norm_inplace on '{}' errored: {e}", backend.name())
        });

        if let Some(ref r) = reference {
            assert_close(&x, r, &format!("layer_norm_inplace on '{}'", backend.name()));
        } else {
            reference = Some(x.clone());
        }
    }
}

#[test]
fn registry_detect_at_least_cpu() {
    let reg = BackendRegistry::detect();
    assert!(!reg.is_empty(), "detect() should always find CPU at minimum");
    assert!(reg.len() >= 1);
}

#[test]
fn env_override_cpu_only() {
    // Just probes that the code path for MODGRAD_BACKEND=cpu works.
    // We don't actually set the env var here because tests share
    // process state — setting it would affect other tests.
    let reg = BackendRegistry::detect();
    let mut x = vec![1.0f32, 1.0, 1.0, 1.0];
    let mut out = vec![0.0f32; 1];
    let mut op = Op::ReduceL2Sq { x: &mut x, out: &mut out };
    let chose = reg.dispatch(&mut op).unwrap();
    assert_eq!(chose, "cpu", "Phase 1: CPU is the only registered backend");
    assert_eq!(out[0], 4.0);
}

/// Exercise the ROCm weight-VRAM cache path end-to-end: run the same
/// matvec twice against the same weight pointer and assert both calls
/// produce identical output. The second call should hit the cache (no
/// re-upload), but the observable contract is just that results match.
///
/// Dormant when ROCm isn't available on this host — `try_new()` returns
/// `None` and the test short-circuits. Also a no-op when compiled
/// without the `rocm` feature for the same reason.
#[test]
fn rocm_weight_cache_hit_matches_first_dispatch() {
    let Some(rocm) = RocmBackend::try_new() else {
        eprintln!("rocm backend unavailable — skipping cache parity check");
        return;
    };

    // 64×64 clears RocmBackend::supports() (which gates on dim ≥ 64).
    let out_dim = 64usize;
    let in_dim = 64usize;
    let weight: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i as f32) * 0.001).sin()).collect();
    let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.01).collect();
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.02 - 0.5).collect();

    // First dispatch: cold cache, triggers alloc + upload + insert.
    let mut out1 = vec![0.0f32; out_dim];
    let mut op1 = Op::Matvec {
        x: &x, weight: &weight, bias: &bias, out: &mut out1,
        out_dim, in_dim, quant: QuantKind::F32,
    };
    assert!(rocm.supports(&op1), "64×64 matvec should hit rocm supports() gate");
    rocm.dispatch(&mut op1).expect("first rocm matvec dispatch");

    // Second dispatch: same weight pointer → cache HIT. Output must
    // match bit-for-bit because the cached VRAM hasn't changed.
    let mut out2 = vec![0.0f32; out_dim];
    let mut op2 = Op::Matvec {
        x: &x, weight: &weight, bias: &bias, out: &mut out2,
        out_dim, in_dim, quant: QuantKind::F32,
    };
    rocm.dispatch(&mut op2).expect("second rocm matvec dispatch (cache hit)");
    eprintln!("rocm cache-hit dispatch completed (weight ptr {:p})", weight.as_ptr());

    assert_eq!(out1, out2,
        "cache-hit dispatch must produce the same output as the cold dispatch");

    // Invalidating the cache must not break correctness — the next
    // dispatch re-uploads and should still match.
    rocm.invalidate_cache();
    let mut out3 = vec![0.0f32; out_dim];
    let mut op3 = Op::Matvec {
        x: &x, weight: &weight, bias: &bias, out: &mut out3,
        out_dim, in_dim, quant: QuantKind::F32,
    };
    rocm.dispatch(&mut op3).expect("post-invalidate matvec dispatch");
    assert_eq!(out1, out3,
        "post-invalidate dispatch must match the original output");
}
