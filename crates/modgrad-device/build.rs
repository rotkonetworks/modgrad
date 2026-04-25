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
    println!("cargo:rerun-if-changed=kernels/dequant_q4k.hip");

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
        // they lose is the resident dispatches that depend on the
        // missing kernel (which gracefully surface as
        // `BackendError::Unsupported`). We log the reason via
        // `cargo:warning=` so anyone *expecting* the kernels to
        // build sees why they didn't.
        compile_hipcc_kernels(&rocm);
    }
}

fn compile_hipcc_kernels(rocm: &str) {
    let hipcc = format!("{rocm}/bin/hipcc");
    if !Path::new(&hipcc).exists() {
        println!(
            "cargo:warning=modgrad-device: hipcc not found at {hipcc}; \
             custom kernels will return Unsupported. Set ROCM_PATH if your \
             ROCm install is elsewhere."
        );
        return;
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set by cargo");

    // Compile every `.hip` kernel in `kernels/` into its own .o, then
    // archive them all into a single static lib that propagates via
    // `rustc-link-lib=static=`. Adding a new kernel is just dropping
    // a new file into `kernels/` and a `rerun-if-changed=` line above.
    let kernel_sources = ["kernels/rms_norm.hip", "kernels/dequant_q4k.hip"];

    let mut object_files: Vec<String> = Vec::with_capacity(kernel_sources.len());
    for src in kernel_sources {
        if !Path::new(src).exists() {
            println!(
                "cargo:warning=modgrad-device: {src} missing; \
                 skipping hipcc kernel compile."
            );
            return;
        }
        // Strip leading `kernels/` and `.hip` to derive an object
        // name. Matches the previous convention of `<name>.o`.
        let stem = Path::new(src)
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("kernel src has utf-8 file stem");
        let obj = format!("{out_dir}/{stem}.o");
        let status = Command::new(&hipcc)
            .args([
                "--offload-arch=gfx1102",
                "-O3",
                "-fPIC",
                "-c",
                src,
                "-o",
                &obj,
            ])
            .status();
        match status {
            Ok(s) if s.success() => object_files.push(obj),
            Ok(s) => {
                println!(
                    "cargo:warning=modgrad-device: hipcc exited non-zero ({s}) \
                     while compiling {src}; custom kernels will return Unsupported."
                );
                return;
            }
            Err(e) => {
                println!(
                    "cargo:warning=modgrad-device: failed to invoke hipcc ({e}) \
                     while compiling {src}; custom kernels will return Unsupported."
                );
                return;
            }
        }
    }

    // Archive every .o into one static lib so `rustc-link-lib=static=<name>`
    // can pull them in transitively. `rustc-link-arg=path/to/.o` only
    // attaches to THIS crate's final-link artefacts and does NOT
    // propagate to downstream crates that depend on modgrad-device —
    // `cargo test -p modgrad-compute` would then fail with
    // `undefined symbol: launch_rms_norm`. The static archive avoids
    // that by riding cargo's link-graph the way every other Rust
    // C-stub-ffi crate does.
    let kernel_lib = format!("{out_dir}/libmodgrad_kernels.a");
    let mut ar_args: Vec<String> = vec!["rcs".into(), kernel_lib.clone()];
    ar_args.extend(object_files.iter().cloned());
    let ar_status = Command::new("ar").args(&ar_args).status();
    let ar_ok = matches!(ar_status, Ok(s) if s.success());
    if !ar_ok {
        println!(
            "cargo:warning=modgrad-device: failed to archive kernel objects \
             into {kernel_lib} via `ar`; custom kernels will return Unsupported."
        );
        return;
    }
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=modgrad_kernels");
    // The kernel .o files reference hipLaunchKernel / __hipRegisterFat…
    // from libamdhip64. The `#[link(name="amdhip64")]` attribute in
    // src/backend/rocm.rs propagates the dynamic-lib link request
    // through cargo to dependents — but only for crates that pull
    // in the rocm FFI code path. Belt-and-braces dylib link too.
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-cfg=modgrad_hipcc_kernels");
}
