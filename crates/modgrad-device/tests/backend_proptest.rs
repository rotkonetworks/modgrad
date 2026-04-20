//! Property-based parity tests across backends.
//!
//! For each op with random shape + data, verify every registered
//! backend produces outputs within 1e-3 absolute / 1e-2 relative of
//! the CPU reference. Tolerances are looser than `backend_parity.rs`
//! because generated shapes hit a wider range of numerical paths
//! (bigger matrices accumulate more rounding).
//!
//! Test-count is kept modest (32 cases per property) so CI runs in
//! seconds, not minutes. Shape ranges are conservative — the goal is
//! catching layout/bounds bugs, not stress-testing precision.

use modgrad_device::backend::{
    AdamWArgs, Backend, CpuBackend, KfdBackend, Op, QuantKind, RocmBackend,
};
use proptest::prelude::*;

// Slightly looser than `backend_parity.rs` — random shapes hit more
// accumulation paths and f32 rounding diverges more at scale.
const ABS_TOL: f32 = 1e-3;
const REL_TOL: f32 = 1e-2;

fn assert_close(a: &[f32], b: &[f32], ctx: &str) -> Result<(), TestCaseError> {
    prop_assert_eq!(a.len(), b.len(), "{}: length mismatch", ctx);
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let scale = av.abs().max(bv.abs()).max(1.0);
        prop_assert!(
            diff <= ABS_TOL || diff / scale <= REL_TOL,
            "{}[{}] diverged: {} vs {} (|Δ| = {}, scale = {})",
            ctx, i, av, bv, diff, scale,
        );
    }
    Ok(())
}

fn backends() -> Vec<Box<dyn Backend>> {
    let mut v: Vec<Box<dyn Backend>> = vec![Box::new(CpuBackend::new())];
    if let Some(kfd) = KfdBackend::try_new() {
        // KFD maintains a global weight cache keyed by pointer. When
        // sequential proptests dispatch with different weight slices
        // (as they do here with fresh-allocated vectors per case),
        // stale cache entries from prior cases can cross-contaminate
        // a later dispatch. Invalidate on registration so each
        // property starts clean.
        kfd.invalidate_cache();
        v.push(Box::new(kfd));
    }
    if let Some(rocm) = RocmBackend::try_new() {
        v.push(Box::new(rocm));
    }
    v
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        .. ProptestConfig::default()
    })]

    // ─── Matvec ──────────────────────────────────────────────
    // Random out_dim × in_dim, bounded values. Covers the common
    // Linear-layer shape space.
    #[test]
    fn prop_matvec_parity(
        out_dim in 1usize..=16,
        in_dim  in 1usize..=32,
        seed    in 0u64..u64::MAX,
    ) {
        let rng = |i: usize| -> f32 {
            let h = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as u32;
            ((h as f32) / (u32::MAX as f32)) * 4.0 - 2.0
        };
        let x: Vec<f32> = (0..in_dim).map(rng).collect();
        let weight: Vec<f32> = (0..out_dim * in_dim).map(|i| rng(i + 1000)).collect();
        let bias: Vec<f32> = (0..out_dim).map(|i| rng(i + 2000)).collect();

        let bs = backends();
        let mut reference: Option<Vec<f32>> = None;

        for backend in &bs {
            let mut out = vec![0.0f32; out_dim];
            let mut op = Op::Matvec {
                x: &x, weight: &weight, bias: &bias, out: &mut out,
                out_dim, in_dim, quant: QuantKind::F32,
            };
            if !backend.supports(&op) { continue; }
            backend.dispatch(&mut op).map_err(|e|
                TestCaseError::fail(format!("matvec on '{}' errored: {e}", backend.name())))?;

            if let Some(ref r) = reference {
                assert_close(&out, r, &format!("matvec on '{}'", backend.name()))?;
            } else {
                reference = Some(out);
            }
        }
    }

    // ─── Matmul NN ───────────────────────────────────────────
    // Small shapes so KFD's 32/8/128 alignment excludes it via supports();
    // CPU vs ROCm (if available) provides the agreement check.
    #[test]
    fn prop_matmul_nn_parity(
        m in 1usize..=12,
        k in 1usize..=16,
        n in 1usize..=12,
        seed in 0u64..u64::MAX,
    ) {
        let rng = |i: usize| -> f32 {
            let h = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as u32;
            ((h as f32) / (u32::MAX as f32)) * 4.0 - 2.0
        };
        let a: Vec<f32> = (0..m * k).map(rng).collect();
        let b: Vec<f32> = (0..k * n).map(|i| rng(i + 1000)).collect();

        let bs = backends();
        let mut reference: Option<Vec<f32>> = None;

        for backend in &bs {
            let mut out = vec![0.0f32; m * n];
            let mut op = Op::MatmulNN {
                a: &a, b: &b, out: &mut out, bias: None,
                m, k, n,
            };
            if !backend.supports(&op) { continue; }
            backend.dispatch(&mut op).map_err(|e|
                TestCaseError::fail(format!("matmul on '{}' errored: {e}", backend.name())))?;

            if let Some(ref r) = reference {
                assert_close(&out, r, &format!("matmul on '{}'", backend.name()))?;
            } else {
                reference = Some(out);
            }
        }
    }

    // ─── OuterProductAcc ─────────────────────────────────────
    // Shapes kept above the KFD 256-element threshold so both CPU and
    // KFD actually run. Below that KFD declines and we'd only be
    // verifying CPU against itself.
    #[test]
    fn prop_outer_product_acc_parity(
        m in 8usize..=32,
        n in 8usize..=32,
        seed in 0u64..u64::MAX,
    ) {
        prop_assume!(m * n >= 256); // honor KFD size gate
        let rng = |i: usize| -> f32 {
            let h = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as u32;
            ((h as f32) / (u32::MAX as f32)) * 4.0 - 2.0
        };
        let a: Vec<f32> = (0..m).map(rng).collect();
        let b: Vec<f32> = (0..n).map(|i| rng(i + 1000)).collect();
        let seed_buf: Vec<f32> = (0..m * n).map(|i| rng(i + 2000)).collect();

        let bs = backends();
        let mut reference: Option<Vec<f32>> = None;

        for backend in &bs {
            let mut accum = seed_buf.clone();
            let mut op = Op::OuterProductAcc {
                a: &a, b: &b, accum: &mut accum, m, n,
            };
            if !backend.supports(&op) { continue; }
            backend.dispatch(&mut op).map_err(|e|
                TestCaseError::fail(format!("outer_product_acc on '{}' errored: {e}", backend.name())))?;

            if let Some(ref r) = reference {
                assert_close(&accum, r, &format!("outer_product_acc on '{}'", backend.name()))?;
            } else {
                reference = Some(accum);
            }
        }
    }

    // ─── SuperLinearFwd (cache=None, inference path) ─────────
    #[test]
    fn prop_super_linear_fwd_nocache(
        d_model in 1usize..=32,
        memory_length in 1usize..=16,
        out_per in 1usize..=4,
        seed in 0u64..u64::MAX,
    ) {
        let rng = |i: usize| -> f32 {
            let h = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as u32;
            ((h as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        };
        let trace: Vec<f32> = (0..d_model * memory_length).map(rng).collect();
        let weights: Vec<f32> = (0..d_model * out_per * memory_length).map(|i| rng(i + 1000)).collect();
        let biases: Vec<f32> = (0..d_model * out_per).map(|i| rng(i + 2000)).collect();

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
            backend.dispatch(&mut op).map_err(|e|
                TestCaseError::fail(format!("super_linear_fwd on '{}' errored: {e}", backend.name())))?;

            if let Some(ref r) = reference {
                assert_close(&out, r, &format!("super_linear_fwd on '{}'", backend.name()))?;
            } else {
                reference = Some(out);
            }
        }
    }

    // ─── SiluFwd ─────────────────────────────────────────────
    // Elementwise; no index math. Expect universal agreement.
    #[test]
    fn prop_silu_fwd_parity(
        len in 1usize..=256,
        seed in 0u64..u64::MAX,
    ) {
        let rng = |i: usize| -> f32 {
            let h = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as u32;
            ((h as f32) / (u32::MAX as f32)) * 8.0 - 4.0
        };
        let x: Vec<f32> = (0..len).map(rng).collect();

        let bs = backends();
        let mut reference: Option<Vec<f32>> = None;

        for backend in &bs {
            let mut out = vec![0.0f32; len];
            let mut op = Op::SiluFwd { x: &x, out: &mut out };
            if !backend.supports(&op) { continue; }
            backend.dispatch(&mut op).map_err(|e|
                TestCaseError::fail(format!("silu_fwd on '{}' errored: {e}", backend.name())))?;

            if let Some(ref r) = reference {
                assert_close(&out, r, &format!("silu_fwd on '{}'", backend.name()))?;
            } else {
                reference = Some(out);
            }
        }
    }

    // ─── SiluBwd ─────────────────────────────────────────────
    #[test]
    fn prop_silu_bwd_parity(
        len in 1usize..=256,
        seed in 0u64..u64::MAX,
    ) {
        let rng = |i: usize| -> f32 {
            let h = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as u32;
            ((h as f32) / (u32::MAX as f32)) * 4.0 - 2.0
        };
        let x: Vec<f32> = (0..len).map(rng).collect();
        let d_out: Vec<f32> = (0..len).map(|i| rng(i + 1000)).collect();

        let bs = backends();
        let mut reference: Option<Vec<f32>> = None;

        for backend in &bs {
            let mut d_x = vec![0.0f32; len];
            let mut op = Op::SiluBwd { d_out: &d_out, x: &x, d_x: &mut d_x };
            if !backend.supports(&op) { continue; }
            backend.dispatch(&mut op).map_err(|e|
                TestCaseError::fail(format!("silu_bwd on '{}' errored: {e}", backend.name())))?;

            if let Some(ref r) = reference {
                assert_close(&d_x, r, &format!("silu_bwd on '{}'", backend.name()))?;
            } else {
                reference = Some(d_x);
            }
        }
    }

    // ─── GluFwd ──────────────────────────────────────────────
    // Input splits into value/gate halves; output is half-length.
    #[test]
    fn prop_glu_fwd_parity(
        half in 1usize..=128,
        seed in 0u64..u64::MAX,
    ) {
        let rng = |i: usize| -> f32 {
            let h = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as u32;
            ((h as f32) / (u32::MAX as f32)) * 4.0 - 2.0
        };
        let x: Vec<f32> = (0..2 * half).map(rng).collect();

        let bs = backends();
        let mut reference: Option<Vec<f32>> = None;

        for backend in &bs {
            let mut out = vec![0.0f32; half];
            let mut op = Op::GluFwd { x: &x, out: &mut out };
            if !backend.supports(&op) { continue; }
            backend.dispatch(&mut op).map_err(|e|
                TestCaseError::fail(format!("glu_fwd on '{}' errored: {e}", backend.name())))?;

            if let Some(ref r) = reference {
                assert_close(&out, r, &format!("glu_fwd on '{}'", backend.name()))?;
            } else {
                reference = Some(out);
            }
        }
    }

    // ─── AdamW ───────────────────────────────────────────────
    // Full optimizer step. Tolerance loosened (5e-3 abs) because
    // bias-correction and sqrt stack enough rounding that backend
    // differences accumulate mildly.
    #[test]
    fn prop_adamw_parity(
        len in 1usize..=128,
        step in 1u32..=16,
        seed in 0u64..u64::MAX,
    ) {
        let rng = |i: usize| -> f32 {
            let h = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as u32;
            ((h as f32) / (u32::MAX as f32)) * 1.0 - 0.5
        };
        let w0: Vec<f32> = (0..len).map(rng).collect();
        let g: Vec<f32> = (0..len).map(|i| rng(i + 1000)).collect();
        let m0: Vec<f32> = (0..len).map(|i| rng(i + 2000) * 0.1).collect();
        let v0: Vec<f32> = (0..len).map(|i| rng(i + 3000).abs() * 0.01).collect();

        let bs = backends();
        let mut reference: Option<Vec<f32>> = None;

        for backend in &bs {
            let mut w = w0.clone();
            let mut m = m0.clone();
            let mut v = v0.clone();
            let bc1_inv = 1.0 / (1.0 - 0.9_f32.powi(step as i32));
            let bc2_inv = 1.0 / (1.0 - 0.999_f32.powi(step as i32));
            let mut op = Op::AdamW(AdamWArgs {
                w: &mut w, g: &g, m: &mut m, v: &mut v,
                lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8,
                weight_decay: 0.0, bc1_inv, bc2_inv,
            });
            if !backend.supports(&op) { continue; }
            backend.dispatch(&mut op).map_err(|e|
                TestCaseError::fail(format!("adamw on '{}' errored: {e}", backend.name())))?;

            if let Some(ref r) = reference {
                assert_close(&w, r, &format!("adamw on '{}'", backend.name()))?;
            } else {
                reference = Some(w);
            }
        }
    }

    // ─── MatvecT (backward of Linear) ────────────────────────
    // d_input[j] = sum_i weight[i, j] * d_out[i]. Gate mirrors Matvec.
    #[test]
    fn prop_matvec_t_parity(
        out_dim in 1usize..=64,
        in_dim  in 1usize..=64,
        seed    in 0u64..u64::MAX,
    ) {
        let rng = |i: usize| -> f32 {
            let h = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as u32;
            ((h as f32) / (u32::MAX as f32)) * 4.0 - 2.0
        };
        let d_out: Vec<f32> = (0..out_dim).map(rng).collect();
        let weight: Vec<f32> = (0..out_dim * in_dim).map(|i| rng(i + 1000)).collect();

        let bs = backends();
        let mut reference: Option<Vec<f32>> = None;

        for backend in &bs {
            let mut d_input = vec![0.0f32; in_dim];
            let mut op = Op::MatvecT {
                d_out: &d_out, weight: &weight, d_input: &mut d_input,
                out_dim, in_dim,
            };
            if !backend.supports(&op) { continue; }
            backend.dispatch(&mut op).map_err(|e|
                TestCaseError::fail(format!("matvec_t on '{}' errored: {e}", backend.name())))?;

            if let Some(ref r) = reference {
                assert_close(&d_input, r, &format!("matvec_t on '{}'", backend.name()))?;
            } else {
                reference = Some(d_input);
            }
        }
    }

    // ─── TraceShiftFwd ───────────────────────────────────────
    // In-place rotation of per-neuron memory. Non-trivial because it
    // mutates `trace` (both a new value is pushed and old values shift).
    #[test]
    fn prop_trace_shift_fwd_parity(
        d_model in 1usize..=64,
        memory_length in 2usize..=32,
        seed in 0u64..u64::MAX,
    ) {
        let rng = |i: usize| -> f32 {
            let h = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as u32;
            ((h as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        };
        let trace_init: Vec<f32> = (0..d_model * memory_length).map(rng).collect();
        let new_val: Vec<f32> = (0..d_model).map(|i| rng(i + 1000)).collect();

        let bs = backends();
        let mut reference: Option<Vec<f32>> = None;

        for backend in &bs {
            let mut trace = trace_init.clone();
            let mut op = Op::TraceShiftFwd {
                trace: &mut trace, new_val: &new_val,
                d_model, memory_length,
            };
            if !backend.supports(&op) { continue; }
            backend.dispatch(&mut op).map_err(|e|
                TestCaseError::fail(format!("trace_shift on '{}' errored: {e}", backend.name())))?;

            if let Some(ref r) = reference {
                assert_close(&trace, r, &format!("trace_shift on '{}'", backend.name()))?;
            } else {
                reference = Some(trace);
            }
        }
    }

    // ─── ReduceL2Sq ──────────────────────────────────────────
    // Sum-of-squares reduction. Scalar output; tolerance naturally
    // grows with len (parallel reduction ordering varies).
    #[test]
    fn prop_reduce_l2_sq_parity(
        len in 1usize..=1024,
        seed in 0u64..u64::MAX,
    ) {
        let rng = |i: usize| -> f32 {
            let h = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as u32;
            ((h as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..len).map(rng).collect();

        let bs = backends();
        let mut reference: Option<Vec<f32>> = None;

        for backend in &bs {
            let mut out = vec![0.0f32; 1];
            let mut op = Op::ReduceL2Sq { x: &x, out: &mut out };
            if !backend.supports(&op) { continue; }
            backend.dispatch(&mut op).map_err(|e|
                TestCaseError::fail(format!("reduce_l2_sq on '{}' errored: {e}", backend.name())))?;

            if let Some(ref r) = reference {
                assert_close(&out, r, &format!("reduce_l2_sq on '{}'", backend.name()))?;
            } else {
                reference = Some(out);
            }
        }
    }
}
