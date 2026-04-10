//! Typed token sequences — dimension-safe wrappers for the Brain interface.
//!
//! Replaces raw `(&[f32], n_tokens, token_dim)` triples with a checked
//! `Tokens` type that validates dimensions on construction. Passing
//! mismatched dimensions panics at the boundary, not silently corrupts
//! deep inside a matmul.

use std::fmt;

/// A sequence of token vectors, each of dimension `dim`.
///
/// Invariant: `data.len() == n * dim` (checked on construction).
#[derive(Clone)]
pub struct Tokens {
    data: Vec<f32>,
    n: usize,
    dim: usize,
}

impl Tokens {
    /// Create a token sequence. Panics if `data.len() != n * dim`.
    pub fn new(data: Vec<f32>, n: usize, dim: usize) -> Self {
        assert_eq!(data.len(), n * dim,
            "Tokens: data.len()={} != n={} × dim={} = {}",
            data.len(), n, dim, n * dim);
        Self { data, n, dim }
    }

    /// Create from a flat slice (copies).
    pub fn from_slice(data: &[f32], n: usize, dim: usize) -> Self {
        Self::new(data.to_vec(), n, dim)
    }

    /// Empty sequence with given dimension.
    pub fn empty(dim: usize) -> Self {
        Self { data: Vec::new(), n: 0, dim }
    }

    /// Number of tokens.
    #[inline] pub fn n(&self) -> usize { self.n }

    /// Dimension per token.
    #[inline] pub fn dim(&self) -> usize { self.dim }

    /// Total floats.
    #[inline] pub fn len(&self) -> usize { self.data.len() }

    /// Is empty?
    #[inline] pub fn is_empty(&self) -> bool { self.n == 0 }

    /// Raw flat data.
    #[inline] pub fn data(&self) -> &[f32] { &self.data }

    /// Mutable raw flat data.
    #[inline] pub fn data_mut(&mut self) -> &mut [f32] { &mut self.data }

    /// Consume into owned Vec.
    pub fn into_data(self) -> Vec<f32> { self.data }

    /// Token at index i: returns &[dim].
    #[inline]
    pub fn token(&self, i: usize) -> &[f32] {
        debug_assert!(i < self.n, "token index {} >= n {}", i, self.n);
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Mutable token at index i.
    #[inline]
    pub fn token_mut(&mut self, i: usize) -> &mut [f32] {
        debug_assert!(i < self.n);
        let d = self.dim;
        &mut self.data[i * d..(i + 1) * d]
    }

    /// Append tokens from another sequence. Panics on dim mismatch.
    pub fn extend(&mut self, other: &Tokens) {
        assert_eq!(self.dim, other.dim,
            "Tokens::extend: dim mismatch {} vs {}", self.dim, other.dim);
        self.data.extend_from_slice(&other.data);
        self.n += other.n;
    }

    /// Concatenate multiple sequences. All must share the same dim.
    pub fn concat(parts: &[&Tokens]) -> Self {
        if parts.is_empty() { return Self::empty(0); }
        let dim = parts[0].dim;
        let total_n: usize = parts.iter().map(|p| p.n).sum();
        let mut data = Vec::with_capacity(total_n * dim);
        for part in parts {
            assert_eq!(part.dim, dim, "Tokens::concat: dim mismatch {} vs {}", part.dim, dim);
            data.extend_from_slice(&part.data);
        }
        Self { data, n: total_n, dim }
    }

    /// Slice a range of tokens [start..end).
    pub fn slice(&self, start: usize, end: usize) -> Tokens {
        assert!(end <= self.n && start <= end);
        let d = self.dim;
        Tokens::new(self.data[start * d..end * d].to_vec(), end - start, d)
    }
}

/// Borrowed view into a token sequence — zero-copy.
/// Use this in hot paths to avoid allocation.
#[derive(Clone, Copy)]
pub struct TokensRef<'a> {
    data: &'a [f32],
    n: usize,
    dim: usize,
}

impl<'a> TokensRef<'a> {
    /// Borrow from a Tokens.
    pub fn from_tokens(t: &'a Tokens) -> Self {
        Self { data: t.data(), n: t.n(), dim: t.dim() }
    }

    /// Borrow from a raw slice. Panics on dim mismatch.
    pub fn from_slice(data: &'a [f32], n: usize, dim: usize) -> Self {
        assert_eq!(data.len(), n * dim,
            "TokensRef: data.len()={} != n={} × dim={}", data.len(), n, dim);
        Self { data, n, dim }
    }

    #[inline] pub fn n(&self) -> usize { self.n }
    #[inline] pub fn dim(&self) -> usize { self.dim }
    #[inline] pub fn data(&self) -> &[f32] { self.data }

    #[inline]
    pub fn token(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Convert to owned.
    pub fn to_owned(&self) -> Tokens {
        Tokens::new(self.data.to_vec(), self.n, self.dim)
    }
}

impl<'a> fmt::Debug for TokensRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TokensRef(n={}, dim={})", self.n, self.dim)
    }
}

impl Tokens {
    /// Borrow as a TokensRef — zero-copy view.
    pub fn as_ref(&self) -> TokensRef<'_> {
        TokensRef::from_tokens(self)
    }
}

impl fmt::Debug for Tokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tokens(n={}, dim={})", self.n, self.dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_and_access() {
        let t = Tokens::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert_eq!(t.n(), 2);
        assert_eq!(t.dim(), 3);
        assert_eq!(t.token(0), &[1.0, 2.0, 3.0]);
        assert_eq!(t.token(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "data.len()=5 != n=2 × dim=3 = 6")]
    fn rejects_mismatched_dims() {
        Tokens::new(vec![0.0; 5], 2, 3);
    }

    #[test]
    fn concat_works() {
        let a = Tokens::new(vec![1.0, 2.0], 1, 2);
        let b = Tokens::new(vec![3.0, 4.0, 5.0, 6.0], 2, 2);
        let c = Tokens::concat(&[&a, &b]);
        assert_eq!(c.n(), 3);
        assert_eq!(c.token(2), &[5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "dim mismatch")]
    fn concat_rejects_dim_mismatch() {
        let a = Tokens::new(vec![1.0, 2.0], 1, 2);
        let b = Tokens::new(vec![3.0, 4.0, 5.0], 1, 3);
        Tokens::concat(&[&a, &b]);
    }

    #[test]
    fn slice_works() {
        let t = Tokens::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let s = t.slice(1, 3);
        assert_eq!(s.n(), 2);
        assert_eq!(s.token(0), &[3.0, 4.0]);
    }
}
