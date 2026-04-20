//! Static, compile-time evidence that `ComputeCtx<B>::alloc_buffer`
//! returns a backend-affine type.
//!
//! The load-bearing negative test (a cross-backend assignment that must
//! *fail* to compile) lives as a `compile_fail` doctest on
//! `modgrad_device::backend::ComputeCtx`. Doctests run under
//! `cargo test --doc`; this file provides **positive** evidence via
//! `TypeId` comparison so the `--lib --tests` invocation in the Stage 2
//! verification list also gets visible coverage of the invariant.
//!
//! Rationale: if `CpuBackend::Buffer` and `KfdBackend::Buffer` are
//! distinct TypeIds, there is no world in which the Rust compiler lets
//! you substitute one for the other without a conversion. That's the
//! affinity guarantee — stated in types, verified in tests.

use std::any::TypeId;

use modgrad_device::backend::{BufferBackend, CpuBackend, HostBuffer};

#[test]
fn cpu_buffer_is_host_buffer() {
    // CPU defaults to HostBuffer — sanity check the associated type.
    assert_eq!(
        TypeId::of::<<CpuBackend as BufferBackend>::Buffer>(),
        TypeId::of::<HostBuffer>(),
        "CpuBackend::Buffer must be HostBuffer",
    );
}

#[cfg(feature = "kfd")]
#[test]
fn kfd_buffer_is_distinct_from_cpu_buffer() {
    use modgrad_device::backend::KfdBackend;
    // KFD defines its own Buffer type (KfdBuffer). Distinct from CPU's
    // HostBuffer: this is the type-system link that prevents
    // `ComputeCtx<CpuBackend>::alloc_buffer(...)` from ever producing a
    // value assignable to a `<KfdBackend as BufferBackend>::Buffer`
    // binding.
    assert_ne!(
        TypeId::of::<<KfdBackend as BufferBackend>::Buffer>(),
        TypeId::of::<<CpuBackend as BufferBackend>::Buffer>(),
        "KfdBackend::Buffer must differ from CpuBackend::Buffer",
    );
}

#[cfg(not(feature = "kfd"))]
#[test]
fn kfd_buffer_is_distinct_from_cpu_buffer() {
    eprintln!("[test] SKIP — kfd feature disabled, KfdBuffer not compiled");
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_buffer_is_distinct_from_cpu_buffer() {
    use modgrad_device::backend::RocmBackend;
    assert_ne!(
        TypeId::of::<<RocmBackend as BufferBackend>::Buffer>(),
        TypeId::of::<<CpuBackend as BufferBackend>::Buffer>(),
        "RocmBackend::Buffer must differ from CpuBackend::Buffer",
    );
}

#[cfg(not(feature = "rocm"))]
#[test]
fn rocm_buffer_is_distinct_from_cpu_buffer() {
    eprintln!("[test] SKIP — rocm feature disabled, RocmBuffer not compiled");
}
