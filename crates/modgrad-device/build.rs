//! Build-script helpers for the `modgrad-device` backends.
//!
//! When the `rocm` feature is enabled, we need the linker to find
//! `libamdhip64.so` and `libhipblas.so`. They live under
//! `/opt/rocm/lib` on every ROCm install (Ubuntu/Debian packages,
//! the upstream tarball, and the `rocm-*` rpms all use that path).
//! The bare `#[link(name = "amdhip64")]` attributes in
//! `src/backend/rocm.rs` rely on the linker's default search order,
//! which does NOT include `/opt/rocm/lib`. So without this script
//! the build fails with `unable to find library -lamdhip64`, which
//! is the issue session 2026-04-21 hit until the env var
//! `RUSTFLAGS="-L /opt/rocm/lib"` was set.
//!
//! This script does the equivalent at build time with zero caller
//! setup — `cargo build -p mazes --features rocm` just works.
//!
//! The `ROCM_PATH` environment variable (honored by hip and rocm
//! upstream) overrides the default `/opt/rocm` location for users
//! with a non-standard install (e.g. a project-local ROCm build).

fn main() {
    // Only emit link-search when the rocm feature is active. Without
    // this guard, CPU-only and CUDA-only builds would pick up the
    // -L flag needlessly (harmless on paper but dirty).
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    if std::env::var("CARGO_FEATURE_ROCM").is_ok() {
        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        println!("cargo:rustc-link-search=native={rocm}/lib");
        // Runtime RPATH so `cargo run` and the installed binary can
        // find the libs without needing LD_LIBRARY_PATH. If the user
        // prefers to avoid baked-in RPATH they can set
        // MODGRAD_NO_RPATH=1 in their environment.
        if std::env::var("MODGRAD_NO_RPATH").is_err() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm}/lib");
        }
    }
}
