//! Threading prelude shim.
//!
//! `wasm32-unknown-unknown` has no threads, so rayon cannot run there.
//! On native we simply re-export rayon's prelude, so every `par_iter` /
//! `par_iter_mut` / `into_par_iter` / `par_chunks_mut` call site keeps
//! running in parallel exactly as before — native is byte-for-byte
//! unchanged.
//!
//! On wasm we provide extension traits with the same method names that
//! return *serial* std iterators (`iter` / `iter_mut` / `into_iter` /
//! `chunks_mut`). Because the std iterators support the same downstream
//! adapters (`map`, `flat_map`, `filter_map`, `enumerate`, `for_each`,
//! `collect`, `sum`, …), the call sites compile unchanged and just run
//! single-threaded.
//!
//! When the `wasm-threads` feature is enabled (the parallel browser build,
//! compiled with +atomics + build-std and a wasm-bindgen-rayon thread pool),
//! wasm32 ALSO re-exports rayon's real prelude — the call sites then run in
//! parallel across SharedArrayBuffer-backed Web Workers, exactly like native.
//! The serial shim below is compiled only on wasm32 WITHOUT that feature.

// Real rayon: native always, or wasm32 with the `wasm-threads` feature.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
pub use rayon::prelude::*;

/// Thread-count helper. Mirrors `rayon::current_num_threads()` whenever real
/// rayon is linked (native, or wasm32 + `wasm-threads`); always 1 on the
/// serial wasm build. Call sites use this instead of the fully-qualified
/// `rayon::current_num_threads()` so they compile on every target.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
#[inline]
pub fn current_num_threads() -> usize {
    rayon::current_num_threads()
}
#[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
#[inline]
pub fn current_num_threads() -> usize {
    1
}

#[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
pub use wasm_serial::*;

#[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
mod wasm_serial {
    /// Serial stand-in for `rayon`'s `par_iter`/`par_iter_mut`.
    pub trait ParIterShim {
        type Iter<'a>
        where
            Self: 'a;
        type IterMut<'a>
        where
            Self: 'a;
        fn par_iter(&self) -> Self::Iter<'_>;
        fn par_iter_mut(&mut self) -> Self::IterMut<'_>;
    }

    impl<T> ParIterShim for [T] {
        type Iter<'a>
            = core::slice::Iter<'a, T>
        where
            T: 'a;
        type IterMut<'a>
            = core::slice::IterMut<'a, T>
        where
            T: 'a;
        fn par_iter(&self) -> Self::Iter<'_> {
            self.iter()
        }
        fn par_iter_mut(&mut self) -> Self::IterMut<'_> {
            self.iter_mut()
        }
    }

    impl<T> ParIterShim for Vec<T> {
        type Iter<'a>
            = core::slice::Iter<'a, T>
        where
            T: 'a;
        type IterMut<'a>
            = core::slice::IterMut<'a, T>
        where
            T: 'a;
        fn par_iter(&self) -> Self::Iter<'_> {
            self.iter()
        }
        fn par_iter_mut(&mut self) -> Self::IterMut<'_> {
            self.iter_mut()
        }
    }

    /// Serial stand-in for `rayon`'s `par_chunks_mut`.
    pub trait ParChunksShim<T> {
        fn par_chunks_mut(&mut self, n: usize) -> core::slice::ChunksMut<'_, T>;
    }

    impl<T> ParChunksShim<T> for [T] {
        fn par_chunks_mut(&mut self, n: usize) -> core::slice::ChunksMut<'_, T> {
            self.chunks_mut(n)
        }
    }

    /// Serial stand-in for `rayon`'s `into_par_iter`.
    pub trait IntoParIterShim {
        type Iter: Iterator;
        fn into_par_iter(self) -> Self::Iter;
    }

    impl<T> IntoParIterShim for Vec<T> {
        type Iter = std::vec::IntoIter<T>;
        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }

    impl IntoParIterShim for core::ops::Range<usize> {
        type Iter = core::ops::Range<usize>;
        fn into_par_iter(self) -> Self::Iter {
            self
        }
    }
}
