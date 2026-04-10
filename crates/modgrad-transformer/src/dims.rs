//! Zero-cost dimension newtypes.
//!
//! Prevents the silent dimension-swap bugs that plague tensor code.
//! Each newtype wraps a usize and is accessed via `.get()`.

macro_rules! dim_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
        pub struct $name(usize);

        impl $name {
            #[inline(always)]
            pub const fn new(v: usize) -> Self { Self(v) }
            #[inline(always)]
            pub const fn get(self) -> usize { self.0 }
        }

        impl From<usize> for $name {
            fn from(v: usize) -> Self { Self(v) }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

dim_newtype!(
    /// Dimension of each attention head (typically 64 or 128).
    HeadDim
);
dim_newtype!(
    /// Maximum or current sequence length.
    SeqLen
);
dim_newtype!(
    /// Number of query attention heads.
    NumHeads
);
dim_newtype!(
    /// Number of key/value heads (for GQA; may be < NumHeads).
    NumKvHeads
);
dim_newtype!(
    /// Model hidden dimension (embedding size).
    ModelDim
);
dim_newtype!(
    /// Vocabulary size.
    VocabSize
);
dim_newtype!(
    /// Total number of transformer layers.
    NumLayers
);
dim_newtype!(
    /// MLP intermediate dimension.
    MlpDim
);

/// Layer index, bounded by a NumLayers.
/// Construction is fallible: `LayerIdx::new(5, NumLayers::new(12))` succeeds,
/// `LayerIdx::new(12, NumLayers::new(12))` returns None.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerIdx {
    idx: usize,
    total: usize,
}

impl LayerIdx {
    /// Create a layer index, returning None if `idx >= total.get()`.
    pub fn new(idx: usize, total: NumLayers) -> Option<Self> {
        if idx < total.get() {
            Some(Self { idx, total: total.get() })
        } else {
            None
        }
    }

    /// Create without bounds check (for internal iteration).
    ///
    /// # Safety (logical)
    /// Caller must ensure `idx < total.get()`.
    #[inline(always)]
    pub(crate) fn new_unchecked(idx: usize, total: NumLayers) -> Self {
        debug_assert!(idx < total.get());
        Self { idx, total: total.get() }
    }

    #[inline(always)]
    pub fn get(self) -> usize { self.idx }

    #[inline(always)]
    pub fn total(self) -> usize { self.total }

    /// Fractional position in the stack: 0.0 for first layer, approaches 1.0 for last.
    #[inline]
    pub fn frac(self) -> f32 {
        self.idx as f32 / (self.total - 1).max(1) as f32
    }

    /// Whether this is the midpoint layer (for backout caching).
    #[inline]
    pub fn is_midpoint(self) -> bool {
        self.idx == self.total / 2
    }
}
