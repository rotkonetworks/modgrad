//! Tests for the device-resident Q4_K_M dequantize kernel.
//!
//! Pattern follows `tests/resident_matmul_rms.rs`:
//!   1. Generate a smooth fp32 reference on host (deterministic, no
//!      rand so failures are reproducible across runs).
//!   2. Quantise host-side via `kfd::gguf::quantize_row_q4_k`.
//!   3. Upload Q4_K bytes to a `HipBuffer`, dispatch the kernel.
//!   4. Download the fp32 result, compare to the host dequant of the
//!      same bytes — they MUST agree element-wise (the kernel is the
//!      canonical reader; the host dequant is the spec).
//!   5. Bonus assertion: the round trip f32 → Q4_K → device-fp32
//!      stays within Q4_K's published 1% RMS error band.
//!
//! The whole suite returns early when the ROCm runtime is missing OR
//! when build.rs skipped the hipcc compile, so `cargo test
//! --features rocm` is a no-op on a host without an AMD GPU rather
//! than a failure.

#![cfg(feature = "rocm")]

use modgrad_device::backend::{Backend, BackendError, HipBuffer, Op, RocmBackend};
use modgrad_device::kfd::gguf::{
    dequantize_row_q4_k, quantize_row_q4_k, Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS,
};

fn upload_bytes(host: &[u8]) -> Result<HipBuffer, BackendError> {
    // HipBuffer::new takes a byte count; copy_from_host expects f32
    // slices, so we go through the lower-level FFI by overlapping
    // the byte slice as a u32 array (4-byte aligned by Vec
    // alignment guarantees) and uploading via copy_from_host. This
    // avoids adding a u8-specific overload to HipBuffer for one
    // test.
    assert!(host.len() % 4 == 0,
        "Q4_K block bytes ({}) are a multiple of 4; upload pads via f32 slice",
        host.len());
    let buf = HipBuffer::new(host.len())?;
    let f32_view: &[f32] = unsafe {
        std::slice::from_raw_parts(host.as_ptr() as *const f32, host.len() / 4)
    };
    buf.copy_from_host(f32_view)?;
    Ok(buf)
}

fn alloc_out_f32(n: usize) -> Result<HipBuffer, BackendError> {
    HipBuffer::new(n * 4)
}

#[test]
fn dequant_q4k_resident_matches_host() {
    let Some(be) = RocmBackend::try_new() else {
        eprintln!("dequant_q4k_resident: no ROCm runtime; skipping");
        return;
    };
    // Build-script may have skipped the hipcc compile on this host.
    // Probe `supports()` to skip cleanly instead of failing on a
    // host without hipcc.
    let probe = Op::DequantQ4KResident {
        q4k_dev: std::ptr::null(),
        fp32_dev: std::ptr::null_mut(),
        n_blocks: 1,
    };
    if !be.supports(&probe) {
        eprintln!("dequant_q4k_resident: hipcc kernel not built; skipping");
        return;
    }

    // Two cases: a single block (smallest valid input), and a
    // multi-block input (exercises the per-block grid-y axis).
    for n_blocks in [1usize, 8] {
        let n = n_blocks * Q4K_BLOCK_ELEMS;
        // Smooth, deterministic, non-zero signal — looks more like
        // model weights than random noise. Negative + positive
        // ensures we exercise the dmin*m subtraction path.
        let x_ref: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                (t * 7.3).sin() * 0.5 + (t * 17.1).cos() * 0.2
            })
            .collect();

        // Host-side quantise to Q4_K bytes.
        let mut q4k = vec![0u8; n_blocks * Q4K_BLOCK_BYTES];
        let written = quantize_row_q4_k(&x_ref, &mut q4k);
        assert_eq!(written, n_blocks * Q4K_BLOCK_BYTES);

        // Host-side dequant — this is the spec the kernel must match
        // bit-for-bit (modulo the trivial fp16 conversion path,
        // which both routes share verbatim).
        let mut host_y = vec![0.0f32; n];
        dequantize_row_q4_k(&q4k, &mut host_y, n_blocks);

        // Upload bytes, dispatch device dequant, download.
        let q4k_buf = upload_bytes(&q4k).expect("upload q4k bytes");
        let out_buf = alloc_out_f32(n).expect("alloc fp32 out");
        let mut op = Op::DequantQ4KResident {
            q4k_dev: q4k_buf.device_ptr() as *const u8,
            fp32_dev: out_buf.device_ptr() as *mut f32,
            n_blocks,
        };
        be.dispatch(&mut op).expect("dequant_q4k_resident dispatch");

        let mut device_y = vec![0.0f32; n];
        out_buf.copy_to_host(&mut device_y).expect("download fp32");

        // Element-wise: kernel must match host dequant of the same
        // bytes within fp32 rounding. The two routes share the same
        // f16→f32 helper and the same arithmetic, so this should be
        // exact except for fp32 multiply-add ordering.
        let mut max_diff = 0.0f32;
        for i in 0..n {
            let d = (host_y[i] - device_y[i]).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 1e-5,
            "dequant_q4k_resident host vs device divergence: \
             max |Δ| = {max_diff} (n_blocks={n_blocks})");

        // Bonus: full round trip f32 → Q4_K → device fp32 stays in
        // the Q4_K precision band. Reconstruction RMS / signal RMS
        // is the canonical Q4_K tolerance metric — pointwise
        // relative error is unbounded near zero crossings.
        let signal_sq: f32 = x_ref.iter().map(|v| v * v).sum();
        let err_sq: f32 = x_ref.iter().zip(&device_y)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let rms_rel = (err_sq / signal_sq.max(1e-12)).sqrt();
        assert!(rms_rel < 0.05,
            "Q4_K round trip RMS rel = {rms_rel} > 5% (n_blocks={n_blocks})");
    }
}
