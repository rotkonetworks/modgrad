//! Scoping probe — MIOpen FFI smoke test (delete or keep).
//!
//! Confirms that `libMIOpen.so` is present, links, and exposes the calls we
//! plan to use for Phase 5b limits #2 (LayerNorm) #3 (Softmax) #4 (GLU).
//! Calls only `miopenCreate` / `miopenGetVersion` / `miopenDestroy` — no
//! tensor work, no GPU memory. The point is to prove the library is wired,
//! not to exercise it.
//!
//! Run: cargo run --release --features rocm --example miopen_probe
//! (set RUSTFLAGS='-L /opt/rocm/lib' when rocm feature is on)
//!
//! This file is NOT production code. Once we commit to the MIOpen wiring
//! path, the FFI block here moves into src/backend/rocm.rs alongside the
//! existing hipblas bindings, and this example is removed.

#[cfg(feature = "rocm")]
#[allow(non_camel_case_types, non_upper_case_globals)]
mod probe {
    use std::ffi::{c_int, c_void};

    pub type miopenHandle_t = *mut c_void;
    pub type miopenStatus_t = c_int;
    pub const miopenStatusSuccess: miopenStatus_t = 0;

    #[link(name = "MIOpen")]
    unsafe extern "C" {
        pub fn miopenCreate(handle: *mut miopenHandle_t) -> miopenStatus_t;
        pub fn miopenDestroy(handle: miopenHandle_t) -> miopenStatus_t;
        pub fn miopenGetVersion(major: *mut usize, minor: *mut usize, patch: *mut usize) -> miopenStatus_t;
    }

    pub fn run() -> Result<(), String> {
        let mut major: usize = 0;
        let mut minor: usize = 0;
        let mut patch: usize = 0;
        let st = unsafe { miopenGetVersion(&mut major, &mut minor, &mut patch) };
        if st != miopenStatusSuccess {
            return Err(format!("miopenGetVersion -> status {st}"));
        }
        println!("MIOpen runtime version: {major}.{minor}.{patch}");

        let mut handle: miopenHandle_t = std::ptr::null_mut();
        let st = unsafe { miopenCreate(&mut handle) };
        if st != miopenStatusSuccess {
            return Err(format!("miopenCreate -> status {st}"));
        }
        if handle.is_null() {
            return Err("miopenCreate returned null handle with success status".into());
        }
        println!("miopenCreate ok: handle={handle:?}");

        let st = unsafe { miopenDestroy(handle) };
        if st != miopenStatusSuccess {
            return Err(format!("miopenDestroy -> status {st}"));
        }
        println!("miopenDestroy ok — MIOpen FFI links and round-trips.");
        Ok(())
    }
}

#[cfg(feature = "rocm")]
fn main() {
    match probe::run() {
        Ok(()) => println!("OK: MIOpen probe succeeded."),
        Err(e) => {
            eprintln!("FAIL: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("This probe requires --features rocm.");
    std::process::exit(2);
}
