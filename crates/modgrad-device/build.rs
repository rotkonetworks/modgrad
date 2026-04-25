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
//!
//! Custom hipcc kernels: when the `rocm` feature is on AND
//! `/opt/rocm/bin/hipcc` is present, this script compiles
//! `kernels/rms_norm.hip` to a host-side .o and links it into the
//! final binary. The Rust side gates the FFI declaration and the
//! corresponding dispatch arm on a custom cfg
//! (`modgrad_hipcc_kernels`) we set here, so hosts without hipcc
//! still pass `cargo check` cleanly — they just don't get the
//! RmsNormResident path (returns `Unsupported`).

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-changed=kernels/rms_norm.hip");

    // Declare the custom cfg so rustc (1.80+ checking) doesn't warn.
    println!("cargo:rustc-check-cfg=cfg(modgrad_hipcc_kernels)");

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

        // Best-effort hipcc kernel compilation. Failure here is NOT a
        // build error — hosts without hipcc (or with mismatched ROCm)
        // still need `cargo check` to succeed, and the only feature
        // they lose is the RmsNormResident dispatch (which gracefully
        // surfaces as `BackendError::Unsupported`). We log the reason
        // via `cargo:warning=` so anyone *expecting* the kernel to
        // build sees why it didn't.
        compile_hipcc_kernels(&rocm);
    }
}

fn compile_hipcc_kernels(rocm: &str) {
    let hipcc = format!("{rocm}/bin/hipcc");
    if !Path::new(&hipcc).exists() {
        println!(
            "cargo:warning=modgrad-device: hipcc not found at {hipcc}; \
             RmsNormResident will return Unsupported. Set ROCM_PATH if your \
             ROCm install is elsewhere."
        );
        return;
    }

    let kernel_src = "kernels/rms_norm.hip";
    if !Path::new(kernel_src).exists() {
        println!(
            "cargo:warning=modgrad-device: {kernel_src} missing; \
             skipping hipcc kernel compile."
        );
        return;
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set by cargo");
    let kernel_obj = format!("{out_dir}/rms_norm.o");

    // gfx1102 is this project's RDNA3 target (RX 7600M XT). Bump
    // `--offload-arch` if the build needs to retarget — there's no
    // multi-arch fat-binary need yet because every consumer is
    // co-located with this exact GPU.
    let status = Command::new(&hipcc)
        .args([
            "--offload-arch=gfx1102",
            "-O3",
            "-fPIC",
            "-c",
            kernel_src,
            "-o",
            &kernel_obj,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            // Link the .o directly into every binary. `rustc-link-arg`
            // is positional and propagates to all final-link artefacts
            // in this crate (lib, tests, examples) which is exactly
            // what we need.
            println!("cargo:rustc-link-arg={kernel_obj}");
            // The .o references hipLaunchKernel / __hipRegisterFat...
            // from libamdhip64. `cargo:rustc-link-lib` would only
            // propagate to crates that link against the lib's
            // `#[link]` attributes, missing the example/test binaries
            // in the same package that don't take the rocm FFI code
            // path (e.g. `examples/miopen_probe.rs`). Using the
            // positional `rustc-link-arg=-lamdhip64` form pushes the
            // dependency through to every final binary instead.
            println!("cargo:rustc-link-arg=-lamdhip64");
            println!("cargo:rustc-cfg=modgrad_hipcc_kernels");
        }
        Ok(s) => {
            println!(
                "cargo:warning=modgrad-device: hipcc exited non-zero ({s}) \
                 while compiling {kernel_src}; RmsNormResident will return \
                 Unsupported."
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=modgrad-device: failed to invoke hipcc ({e}); \
                 RmsNormResident will return Unsupported."
            );
        }
    }
}
