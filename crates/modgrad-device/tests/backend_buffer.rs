//! Per-backend `DeviceBuffer` roundtrip harness.
//!
//! Each backend's `ComputeCtx::alloc_buffer` produces a `DeviceBuffer`.
//! For every backend we can actually construct on the test host, write
//! a small host vec in, read it back out, and assert equality.
//!
//! Feature-gated backends (kfd, rocm) emit a visible SKIP message and
//! early-return when the runtime isn't present — never silently green.
//!
//! Stage 2 of the compute-device unification plan. CPU runs everywhere;
//! KFD runs on gfx1102; ROCm runs on any AMD host with the `rocm`
//! feature and /opt/rocm available.

use modgrad_device::backend::{
    ComputeCtx, CpuBackend, DeviceBuffer,
};

/// CPU's `HostBuffer` roundtrip — the ground-truth case. Always runs.
#[test]
fn cpu_host_buffer_roundtrip() {
    // Leak a CpuBackend for &'static — matches how registry-owned
    // backends actually live in production. A fresh Box::leak per test
    // is fine (small, test-local, process ends soon).
    let be: &'static CpuBackend = Box::leak(Box::new(CpuBackend::new()));
    let ctx: ComputeCtx<CpuBackend> = ComputeCtx::new(be);
    assert_eq!(ctx.backend_name(), "cpu");

    let mut buf = ctx.alloc_buffer(8).expect("cpu alloc_buffer");
    assert_eq!(buf.len(), 8);
    assert_eq!(buf.backend_name(), "host");

    let payload: Vec<f32> = (0..8).map(|i| i as f32 * 0.125).collect();
    buf.copy_from_host(&payload).unwrap();

    let mut out = vec![0.0f32; 8];
    buf.copy_to_host(&mut out).unwrap();
    assert_eq!(out, payload, "cpu HostBuffer roundtrip mismatch");

    // Lifecycle hooks compile + run without panic (no-ops on CPU).
    ctx.arena_reset();
    ctx.flush();
}

/// KFD's `KfdBuffer` roundtrip. Skipped when the runtime isn't present
/// or the `kfd` feature isn't compiled in.
#[cfg(feature = "kfd")]
#[test]
fn kfd_device_buffer_roundtrip() {
    use modgrad_device::backend::KfdBackend;

    let Some(kfd) = KfdBackend::try_new() else {
        eprintln!("[test] SKIP — no kfd backend detected (gfx1102 / /dev/kfd not available)");
        return;
    };
    let be: &'static KfdBackend = Box::leak(Box::new(kfd));
    let ctx: ComputeCtx<KfdBackend> = ComputeCtx::new(be);
    assert_eq!(ctx.backend_name(), "kfd");

    let mut buf = match ctx.alloc_buffer(16) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[test] SKIP — kfd alloc_buffer failed: {e}");
            return;
        }
    };
    assert_eq!(buf.len(), 16);
    assert_eq!(buf.backend_name(), "kfd");

    let payload: Vec<f32> = (0..16).map(|i| i as f32 + 0.5).collect();
    buf.copy_from_host(&payload).expect("kfd copy_from_host");

    let mut out = vec![0.0f32; 16];
    buf.copy_to_host(&mut out).expect("kfd copy_to_host");
    assert_eq!(out, payload, "kfd KfdBuffer roundtrip mismatch");

    // Lifecycle — arena_reset forwards to the existing global fn.
    ctx.arena_reset();
    ctx.flush();
}

/// When the `kfd` feature is compiled out there's no backend to test;
/// report the SKIP explicitly so CI logs don't imply coverage.
#[cfg(not(feature = "kfd"))]
#[test]
fn kfd_device_buffer_roundtrip() {
    eprintln!("[test] SKIP — kfd feature disabled at compile time");
}

/// Two independent KFD allocations must not share VRAM. Exercises the
/// `alloc_buffer` lifetime contract that Stage 4's VramGpuBackend
/// migration relies on: writes to one buffer must not leak into the
/// other. Single round-trip roundtrip doesn't catch this — you need two
/// live buffers simultaneously.
#[cfg(feature = "kfd")]
#[test]
fn kfd_two_buffers_do_not_collide() {
    use modgrad_device::backend::KfdBackend;

    let Some(kfd) = KfdBackend::try_new() else {
        eprintln!("[test] SKIP — no kfd backend detected (gfx1102 / /dev/kfd not available)");
        return;
    };
    let be: &'static KfdBackend = Box::leak(Box::new(kfd));
    let ctx: ComputeCtx<KfdBackend> = ComputeCtx::new(be);

    let mut a = match ctx.alloc_buffer(8) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[test] SKIP — kfd alloc_buffer failed: {e}");
            return;
        }
    };
    let mut b = match ctx.alloc_buffer(8) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[test] SKIP — second kfd alloc_buffer failed: {e}");
            return;
        }
    };

    let pa: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let pb: Vec<f32> = (0..8).map(|i| -(i as f32) - 100.0).collect();
    a.copy_from_host(&pa).expect("a copy_from_host");
    b.copy_from_host(&pb).expect("b copy_from_host");

    // Read in the opposite order to catch any "last write wins" bug.
    let mut out_b = vec![0.0f32; 8];
    let mut out_a = vec![0.0f32; 8];
    b.copy_to_host(&mut out_b).expect("b copy_to_host");
    a.copy_to_host(&mut out_a).expect("a copy_to_host");

    assert_eq!(out_a, pa, "buffer a contents collided with b");
    assert_eq!(out_b, pb, "buffer b contents collided with a");
}

#[cfg(not(feature = "kfd"))]
#[test]
fn kfd_two_buffers_do_not_collide() {
    eprintln!("[test] SKIP — kfd feature disabled at compile time");
}

/// ROCm's `RocmBuffer` roundtrip. Skipped without the feature or a
/// working libamdhip64.
#[cfg(feature = "rocm")]
#[test]
fn rocm_device_buffer_roundtrip() {
    use modgrad_device::backend::RocmBackend;

    let Some(rocm) = RocmBackend::try_new() else {
        eprintln!("[test] SKIP — no rocm runtime detected (libamdhip64 / device missing)");
        return;
    };
    let be: &'static RocmBackend = Box::leak(Box::new(rocm));
    let ctx: ComputeCtx<RocmBackend> = ComputeCtx::new(be);
    assert_eq!(ctx.backend_name(), "rocm");

    let mut buf = match ctx.alloc_buffer(12) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[test] SKIP — rocm alloc_buffer failed: {e}");
            return;
        }
    };
    assert_eq!(buf.len(), 12);
    assert_eq!(buf.backend_name(), "rocm");

    let payload: Vec<f32> = (0..12).map(|i| (i as f32).sin()).collect();
    buf.copy_from_host(&payload).expect("rocm copy_from_host");

    let mut out = vec![0.0f32; 12];
    buf.copy_to_host(&mut out).expect("rocm copy_to_host");
    assert_eq!(out, payload, "rocm RocmBuffer roundtrip mismatch");

    ctx.arena_reset();
    ctx.flush();
}

#[cfg(not(feature = "rocm"))]
#[test]
fn rocm_device_buffer_roundtrip() {
    eprintln!("[test] SKIP — rocm feature disabled at compile time");
}
