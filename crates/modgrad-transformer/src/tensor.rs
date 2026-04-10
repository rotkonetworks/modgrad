//! Typed tensor: compile-time dimension tagging.
//!
//! `Tensor2<R, C>` is a row-major matrix where R and C are dimension newtypes.
//! This prevents passing the wrong projection matrix to the wrong operation.

use std::marker::PhantomData;

/// A 2D row-major tensor tagged with dimension newtypes.
///
/// The phantom types R and C carry the semantic meaning (e.g., ModelDim, HeadDim)
/// but the actual shape is stored as plain usize for runtime flexibility.
#[derive(Debug, Clone)]
pub struct Tensor2<R, C> {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    _phantom: PhantomData<(R, C)>,
}

impl<R, C> Tensor2<R, C> {
    /// Create a new tensor, verifying data length matches shape.
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Option<Self> {
        if data.len() == rows * cols {
            Some(Self { data, rows, cols, _phantom: PhantomData })
        } else {
            None
        }
    }

    /// Create a zero-filled tensor.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
            _phantom: PhantomData,
        }
    }

    /// Row slice (no copy).
    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Mutable row slice.
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Full data as slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] { &self.data }

    /// Full data as mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] { &mut self.data }
}

/// A 1D vector tagged with a dimension newtype.
#[derive(Debug, Clone)]
pub struct Tensor1<D> {
    pub data: Vec<f32>,
    _phantom: PhantomData<D>,
}

impl<D> Tensor1<D> {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data, _phantom: PhantomData }
    }

    pub fn zeros(len: usize) -> Self {
        Self { data: vec![0.0; len], _phantom: PhantomData }
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] { &self.data }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] { &mut self.data }

    #[inline]
    pub fn len(&self) -> usize { self.data.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

/// Bias-free weight matrix for transformer linears.
/// Distinct from the existing `Linear` (which has bias) in ctm.rs.
pub type WeightMatrix<R, C> = Tensor2<R, C>;
