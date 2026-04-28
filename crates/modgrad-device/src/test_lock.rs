//! Shared lock for HIP-using tests across the modgrad workspace.
//!
//! Multiple crates run HIP tests that share the device's default stream;
//! without coordination they collide under `cargo test --lib`'s default
//! thread parallelism. Local per-crate `static HIP_TEST_LOCK: Mutex<()>`
//! workarounds don't interlock across crate boundaries — this module
//! provides one process-wide mutex that any HIP-using test in any crate
//! can acquire.
//!
//! Gated behind the `test-utils` feature so non-test builds pay nothing.
//! Downstream crates opt in via:
//!
//! ```toml
//! [dev-dependencies]
//! modgrad-device = { path = "...", features = ["test-utils"] }
//! ```
//!
//! Usage in any test:
//!
//! ```ignore
//! #[test]
//! fn my_hip_test() {
//!     let _guard = modgrad_device::test_lock::hip_test_lock();
//!     // ... HIP-using code; guard released at end of scope ...
//! }
//! ```
//!
//! The guard is a `MutexGuard<'static, ()>` that holds the lock until it
//! goes out of scope, so don't drop it explicitly until you're done with
//! HIP. The single shared `Mutex` is process-wide.

use std::sync::{Mutex, MutexGuard, OnceLock};

static HIP_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Acquire the shared HIP test lock. Blocks until any other holder
/// releases. Returns a guard; HIP-using code should run while holding
/// the guard.
///
/// **Poison recovery is intentional.** If a prior holder panicked,
/// `Mutex::lock` returns `Err(PoisonError)`. For a generic data-carrying
/// mutex this is a correctness signal. For *this* mutex — which guards
/// no Rust-level state, only the implicit serialization of HIP runtime
/// calls — a panic in one test should not permanently break HIP testing
/// for every subsequent test in the run. We therefore unconditionally
/// extract the inner guard via `PoisonError::into_inner`. Do NOT copy
/// this pattern for mutexes that protect actual state.
pub fn hip_test_lock() -> MutexGuard<'static, ()> {
    let mtx = HIP_TEST_LOCK.get_or_init(|| Mutex::new(()));
    match mtx.lock() {
        Ok(g) => g,
        Err(poison) => poison.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::hip_test_lock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    /// Four threads contend for the lock; assert no two are ever inside
    /// the critical section simultaneously, and that all four eventually
    /// complete. This exercises the OnceLock init path and the mutual-
    /// exclusion guarantee.
    #[test]
    fn lock_is_exclusive() {
        let counter = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let c = counter.clone();
            let a = active.clone();
            handles.push(thread::spawn(move || {
                let _guard = hip_test_lock();
                let in_critical = a.fetch_add(1, Ordering::SeqCst);
                assert_eq!(in_critical, 0, "lock not exclusive");
                thread::sleep(Duration::from_millis(10));
                a.fetch_sub(1, Ordering::SeqCst);
                c.fetch_add(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }

    /// Acquiring the lock twice in sequence (after release) should
    /// always succeed. The poison-recovery branch is exercised
    /// structurally — inducing an actual panic-while-holding without
    /// disturbing other tests is fragile, so we settle for verifying
    /// that the happy path works and trust the `match` arms by review.
    #[test]
    fn lock_acquires_in_sequence() {
        {
            let _g = hip_test_lock();
        }
        {
            let _g = hip_test_lock();
        }
    }
}
