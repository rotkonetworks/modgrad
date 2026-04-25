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
            // Archive the .o into a static lib so `rustc-link-lib`
            // can be used. `rustc-link-arg=path/to/.o` only attaches
            // to THIS crate's final-link artefacts and does NOT
            // propagate to downstream crates that depend on
            // modgrad-device — `cargo test -p modgrad-ctm` would
            // then fail with `undefined symbol: launch_rms_norm`.
            // `rustc-link-lib=static=<name>` propagates transitively
            // through cargo's link-graph.
            let kernel_lib = format!("{out_dir}/libmodgrad_kernels.a");
            let ar_status = Command::new("ar")
                .args(["rcs", &kernel_lib, &kernel_obj])
                .status();
            let ar_ok = matches!(ar_status, Ok(s) if s.success());
            if !ar_ok {
                println!(
                    "cargo:warning=modgrad-device: failed to archive \
                     {kernel_obj} into {kernel_lib} via `ar`; \
                     RmsNormResident will return Unsupported."
                );
                return;
            }
            println!("cargo:rustc-link-search=native={out_dir}");
            println!("cargo:rustc-link-lib=static=modgrad_kernels");
            // The kernel .o references hipLaunchKernel /
            // __hipRegisterFat... from libamdhip64. The
            // `#[link(name="amdhip64")]` attribute in
            // src/backend/rocm.rs propagates the dynamic-lib link
            // request through cargo to dependents — but only for
            // crates that pull in the rocm FFI code path.
            // Examples like `miopen_probe` don't, so we also emit
            // a dylib link here as a belt-and-braces. cargo
            // propagates rustc-link-lib=dylib transitively.
            println!("cargo:rustc-link-lib=dylib=amdhip64");
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
