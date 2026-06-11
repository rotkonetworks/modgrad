//! The HIP kernels are compiled + linked by `modgrad-device`'s build script;
//! that build sets the `modgrad_hipcc_kernels` cfg for ITS OWN compilation, but
//! custom cfgs don't cross crate boundaries. So this crate mirrors the same
//! hipcc-presence check to gate the rocm_gemma module identically.
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rustc-check-cfg=cfg(modgrad_hipcc_kernels)");

    if std::env::var("CARGO_FEATURE_ROCM").is_ok() {
        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        if Path::new(&format!("{rocm}/bin/hipcc")).exists() {
            println!("cargo:rustc-cfg=modgrad_hipcc_kernels");
        }
    }
}
