//! KV cache with typestate enforcement.
//!
//! State machine: Empty → Prefilled → Decoding
//! Transitions are compile-time enforced — no runtime Option guards.
//!
//! Also stores previous-token embedding for smear mixing.

use std::marker::PhantomData;
use super::dims::*;

// ─── Typestate markers ──────────────────────────────────────

/// Cache is empty, no tokens processed.
pub struct Empty;
/// Prefill complete, KV entries populated for prompt.
pub struct Prefilled;
/// Autoregressive decoding in progress.
pub struct Decoding;

/// Per-layer KV storage.
#[derive(Debug, Clone)]
pub struct LayerKv {
    /// Key cache: [max_seq_len, kv_dim] row-major.
    pub k: Vec<f32>,
    /// Value cache: [max_seq_len, kv_dim] row-major.
    pub v: Vec<f32>,
    /// KV head dimension (num_kv_heads * head_dim).
    kv_dim: usize,
    /// Maximum sequence length.
    max_len: usize,
}

impl LayerKv {
    fn new(max_seq_len: usize, kv_dim: usize) -> Self {
        Self {
            k: vec![0.0; max_seq_len * kv_dim],
            v: vec![0.0; max_seq_len * kv_dim],
            kv_dim,
            max_len: max_seq_len,
        }
    }

    /// Write K/V for positions `start..start+seq_len`.
    pub fn write(&mut self, k_data: &[f32], v_data: &[f32], start: usize, seq_len: usize) {
        debug_assert!(start + seq_len <= self.max_len);
        let size = seq_len * self.kv_dim;
        let offset = start * self.kv_dim;
        self.k[offset..offset + size].copy_from_slice(&k_data[..size]);
        self.v[offset..offset + size].copy_from_slice(&v_data[..size]);
    }

    /// Read K for positions `0..len`.
    #[inline]
    pub fn k_slice(&self, len: usize) -> &[f32] {
        &self.k[..len * self.kv_dim]
    }

    /// Read V for positions `0..len`.
    #[inline]
    pub fn v_slice(&self, len: usize) -> &[f32] {
        &self.v[..len * self.kv_dim]
    }

    #[inline]
    pub fn kv_dim(&self) -> usize { self.kv_dim }
}

/// The KV cache, parameterized by typestate.
pub struct KvCache<State> {
    /// Per-layer K/V storage.
    pub layers: Vec<LayerKv>,
    /// Previous token's embedding (for smear). `[model_dim]`.
    pub prev_embedding: Vec<f32>,
    /// Current sequence length (number of tokens in cache).
    seq_len: usize,
    /// Max sequence length.
    max_seq_len: usize,
    _state: PhantomData<State>,
}

impl KvCache<Empty> {
    /// Allocate a new empty cache.
    pub fn new(
        num_layers: NumLayers,
        num_kv_heads: NumKvHeads,
        head_dim: HeadDim,
        model_dim: ModelDim,
        max_seq_len: SeqLen,
    ) -> Self {
        let kv_dim = num_kv_heads.get() * head_dim.get();
        let max_len = max_seq_len.get();
        let layers = (0..num_layers.get())
            .map(|_| LayerKv::new(max_len, kv_dim))
            .collect();

        Self {
            layers,
            prev_embedding: vec![0.0; model_dim.get()],
            seq_len: 0,
            max_seq_len: max_len,
            _state: PhantomData,
        }
    }

    /// Allocate cache with per-layer KV dimensions.
    /// `kv_dims[i]` = kv_dim for layer i. Supports varying head sizes.
    /// `kv_source[i]` = which layer to source KV from (for sharing).
    pub fn new_per_layer(
        kv_dims: &[usize],
        model_dim: ModelDim,
        max_seq_len: SeqLen,
    ) -> Self {
        let max_len = max_seq_len.get();
        let layers = kv_dims.iter()
            .map(|&kv_dim| LayerKv::new(max_len, kv_dim))
            .collect();

        Self {
            layers,
            prev_embedding: vec![0.0; model_dim.get()],
            seq_len: 0,
            max_seq_len: max_len,
            _state: PhantomData,
        }
    }

    /// Transition to Prefilled after processing the prompt.
    pub fn prefill(mut self, prompt_len: usize) -> KvCache<Prefilled> {
        self.seq_len = prompt_len;
        KvCache {
            layers: self.layers,
            prev_embedding: self.prev_embedding,
            seq_len: self.seq_len,
            max_seq_len: self.max_seq_len,
            _state: PhantomData,
        }
    }
}

impl KvCache<Prefilled> {
    /// Transition to Decoding mode.
    pub fn start_decode(self) -> KvCache<Decoding> {
        KvCache {
            layers: self.layers,
            prev_embedding: self.prev_embedding,
            seq_len: self.seq_len,
            max_seq_len: self.max_seq_len,
            _state: PhantomData,
        }
    }
}

impl KvCache<Decoding> {
    /// Advance by one token. Returns the position to write to.
    pub fn advance(&mut self) -> usize {
        let pos = self.seq_len;
        self.seq_len += 1;
        pos
    }

    /// Reset to empty for a new conversation.
    pub fn reset(self) -> KvCache<Empty> {
        let mut cache = KvCache {
            layers: self.layers,
            prev_embedding: self.prev_embedding,
            seq_len: 0,
            max_seq_len: self.max_seq_len,
            _state: PhantomData,
        };
        // Zero out for safety
        for layer in &mut cache.layers {
            layer.k.fill(0.0);
            layer.v.fill(0.0);
        }
        cache.prev_embedding.fill(0.0);
        cache
    }
}

/// Methods available in any state.
impl<S> KvCache<S> {
    #[inline]
    pub fn seq_len(&self) -> usize { self.seq_len }

    #[inline]
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }

    /// Mutable access to a layer's KV storage.
    #[inline]
    pub fn layer_mut(&mut self, layer: usize) -> &mut LayerKv {
        &mut self.layers[layer]
    }

    /// Immutable access to a layer's KV storage.
    #[inline]
    pub fn layer(&self, layer: usize) -> &LayerKv {
        &self.layers[layer]
    }

    /// Update the previous embedding (for smear).
    pub fn set_prev_embedding(&mut self, emb: &[f32]) {
        self.prev_embedding.copy_from_slice(emb);
    }
}
