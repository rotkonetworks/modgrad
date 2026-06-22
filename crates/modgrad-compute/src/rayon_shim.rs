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

#[cfg(not(target_arch = "wasm32"))]
pub use rayon::prelude::*;

/// Thread-count helper. Mirrors `rayon::current_num_threads()` on native;
/// always 1 on wasm (single-threaded). Call sites use this instead of the
/// fully-qualified `rayon::current_num_threads()` so they compile on both.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn current_num_threads() -> usize {
    rayon::current_num_threads()
}
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn current_num_threads() -> usize {
    1
}

#[cfg(target_arch = "wasm32")]
pub use wasm_serial::*;

#[cfg(target_arch = "wasm32")]
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
