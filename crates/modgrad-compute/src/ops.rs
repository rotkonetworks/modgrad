//! Math primitives — pure functions with no state.
//!
//! Extracted from ctm.rs (dot, matvec) and re-exports from neuron.rs
//! (glu, layer_norm, concat, maybe_broadcast, simple_attention).

// Re-export neuron-level math primitives so callers can use ops:: for all of them

// ─── Label encoding ────────────────────────────────────────

/// Encode a classification label as a proprioception vector.
///
/// Each class gets a deterministic pseudo-random direction in `d`-dimensional
/// space. In high dimensions random Gaussian vectors are nearly orthogonal,
/// giving natural class separation without a learned embedding table.
///
/// Properties:
///   - Deterministic: same `(class, d)` always yields the same vector.
///   - Unit-variance: E[||v||²] = 1 regardless of `d`.
///   - Near-orthogonal: E[cos(v_i, v_j)] ≈ 0 for i ≠ j when d ≥ 32.
pub fn encode_label(class: usize, d: usize) -> Vec<f32> {
    // Golden-ratio hash spreads classes uniformly across seed space.
    let seed = (class as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut rng = super::neuron::SimpleRng::new(seed);
    let scale = 1.0 / (d as f32).sqrt();
    (0..d).map(|_| rng.next_normal() * scale).collect()
}

// ─── Dot product ───────────────────────────────────────────

/// Dot product — 16-wide unrolled for AVX-512 auto-vectorization.
/// With -C target-cpu=native on AVX-512 hardware, LLVM emits
/// vfmadd231ps zmm (512-bit fused multiply-add, 16 floats/cycle).
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    // 16-wide accumulation — maps to 512-bit AVX-512 registers.
    // Four accumulators to hide FMA latency (4 cycles on Zen4/SPR).
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let chunks = n / 16;
    for i in 0..chunks {
        let j = i * 16;
        // SAFETY: j + 15 < chunks * 16 ≤ n = a.len() = b.len() (from debug_assert above)
        unsafe {
            s0 += *a.get_unchecked(j)      * *b.get_unchecked(j);
            s1 += *a.get_unchecked(j + 1)  * *b.get_unchecked(j + 1);
            s2 += *a.get_unchecked(j + 2)  * *b.get_unchecked(j + 2);
            s3 += *a.get_unchecked(j + 3)  * *b.get_unchecked(j + 3);
            s0 += *a.get_unchecked(j + 4)  * *b.get_unchecked(j + 4);
            s1 += *a.get_unchecked(j + 5)  * *b.get_unchecked(j + 5);
            s2 += *a.get_unchecked(j + 6)  * *b.get_unchecked(j + 6);
            s3 += *a.get_unchecked(j + 7)  * *b.get_unchecked(j + 7);
            s0 += *a.get_unchecked(j + 8)  * *b.get_unchecked(j + 8);
            s1 += *a.get_unchecked(j + 9)  * *b.get_unchecked(j + 9);
            s2 += *a.get_unchecked(j + 10) * *b.get_unchecked(j + 10);
            s3 += *a.get_unchecked(j + 11) * *b.get_unchecked(j + 11);
            s0 += *a.get_unchecked(j + 12) * *b.get_unchecked(j + 12);
            s1 += *a.get_unchecked(j + 13) * *b.get_unchecked(j + 13);
            s2 += *a.get_unchecked(j + 14) * *b.get_unchecked(j + 14);
            s3 += *a.get_unchecked(j + 15) * *b.get_unchecked(j + 15);
        }
    }
    for i in (chunks * 16)..n {
        s0 += a[i] * b[i];
    }
    (s0 + s1) + (s2 + s3)
}

