//! Device-resident KV cache for the transformer's resident attention path.
//!
//! Pre-allocated `HipBuffer`s for K and V across (n_layers × n_kv_heads ×
//! max_seq × head_dim). Layout is **head-major**:
//!
//!   `K[layer][kv_h, t, i] = k_dev[layer * stride_layer
//!                                + kv_h * max_seq_len * head_dim
//!                                + t * head_dim
//!                                + i]`
//!
//! …where `stride_layer = n_kv_heads * max_seq_len * head_dim`.
//!
//! # Why head-major?
//!
//! The host KV cache (`crate::kv_cache::LayerKv`) uses
//! `[max_seq × kv_dim] = [max_seq × (n_kv_heads × head_dim)]` row-major,
//! i.e. token-major. Token-major makes "write the new token's K/V in
//! place" a single contiguous write. But it makes "compute attention
//! scores for head h across all cached tokens" a strided read — the
//! per-head K slice is `[attn_len × head_dim]` strided by `kv_dim` in
//! the underlying buffer. There is no resident matvec or matmul that
//! consumes strided weights; we'd have to transpose.
//!
//! Head-major flips the trade: writes become per-kv-head D2D copies of
//! `head_dim` floats (cheap, ~0.5 µs each, n_kv_heads of them — single-
//! digit microseconds total), but the per-head K/V slabs are then
//! contiguous and feed straight into `matvec_resident` (for K) and
//! `matmul_resident_tn` (for V·softmax). The forward win-regime is
//! decode/score, where the *score computation* is the hot path. So we
//! pay a small constant on the cache write to make the score path
//! a clean dispatch.
//!
//! This is *also* the layout the FlashAttention papers describe as
//! optimal for KV access during scoring; we are not inventing anything,
//! we are paying the rearrangement cost upfront so the per-head matvec
//! reads contiguous memory.
//!
//! # Lifecycle
//!
//! ```ignore
//! let kv_cache = KvCacheResident::new(n_layers, n_kv_heads, head_dim, max_seq_len)?;
//! // Per token, per layer:
//! kv_cache.write(layer, position, &k_dev, &v_dev)?;
//! // Read access for the scoring matvec:
//! let k_layer_dev = kv_cache.k_layer_ptr(layer);
//! let v_layer_dev = kv_cache.v_layer_ptr(layer);
//! ```
//!
//! Only available with `--features rocm`.

#![cfg(feature = "rocm")]

use modgrad_compute::backend::{GpuVec, ResidencyError};
use modgrad_device::backend::{HipBatch, HipBuffer};

/// Window-aware view into one (layer, kv_head)'s K or V slab. Returned
/// by [`KvCacheResident::k_slab_view`] / [`KvCacheResident::v_slab_view`].
/// Encodes "where to start reading" + "how many rows to read" for the
/// matvec/matmul call — handles linear caches and rolling caches
/// uniformly so the attention forward code stays branch-free on cache
/// shape.
#[derive(Debug, Clone, Copy)]
pub struct KvSlabView {
    /// Device pointer to the first row to read (already offset into the
    /// per-(layer, kv_head) slab).
    pub ptr: *const f32,
    /// Number of rows to read starting at `ptr`. Multiply by `head_dim`
    /// for the byte/element count.
    pub rows: usize,
}

#[inline]
fn slab_view_inner(slab_base: *const f32, head_dim: usize, slot_count: usize, start: usize, position: usize) -> KvSlabView {
    if position < slot_count {
        // Either a linear layer (slot_count == max_seq_len) or a rolling
        // layer whose ring isn't full yet — slot index equals position
        // either way, so the legacy linear-style offset is correct.
        let off = start * head_dim;
        KvSlabView {
            ptr: unsafe { slab_base.add(off) },
            rows: position + 1 - start,
        }
    } else {
        // Full rolling ring: every slot holds one of the most recent
        // `slot_count` positions. Reading the entire slab in slot order
        // is fine because attention is permutation-invariant in K/V
        // position. Caller's `start` is unused here.
        KvSlabView { ptr: slab_base, rows: slot_count }
    }
}

/// Device-resident KV cache. See module docs.
///
/// **Per-layer rolling support.** Each layer has its own `slots` capacity
/// (a row count). When `slots[layer] == max_seq_len` the layer behaves
/// linearly: position p writes to slot p, and prior tokens sit in slot
/// order (the default constructor wires this up). When `slots[layer] < max_seq_len`
/// (opt-in via `with_layer_slots`), the layer is a ring buffer of capacity
/// `slots[layer]`: position p writes to slot `p % slots[layer]`, evicting
/// the oldest entry. Combined with sliding-window attention this gives
/// per-layer KV memory ≈ `slots[layer] · head_dim · n_kv_heads` instead of
/// `max_seq_len · head_dim · n_kv_heads` — the actual memory win SWA
/// promises.
pub struct KvCacheResident {
    /// One `HipBuffer` per layer for K. Sized `n_kv_heads * slots[layer] *
    /// head_dim * 4` bytes — varies per layer when the rolling-cache
    /// constructor is used.
    pub k_dev: Vec<HipBuffer>,
    /// One `HipBuffer` per layer for V. Same shape as the matching `k_dev[layer]`.
    pub v_dev: Vec<HipBuffer>,
    /// Per-layer slot count. `slots[layer] == max_seq_len` means linear
    /// cache (default); `slots[layer] < max_seq_len` means ring buffer.
    pub slots: Vec<usize>,
    /// Previous-token embedding (for the smear path). `[model_dim]` on device.
    /// Allocated alongside the layers so the caller's smear pass can
    /// dispatch resident as well; safe to ignore if smear is host-side.
    pub prev_embedding_dev: HipBuffer,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    /// Number of tokens written so far. Updated by [`Self::extend`] /
    /// [`Self::set_seq_len`].
    seq_len: usize,
}

impl KvCacheResident {
    /// Allocate K and V buffers for every layer plus the smear scratch.
    /// Every layer is sized for `max_seq_len` slots — i.e. linear cache
    /// throughout, the default and historical behavior.
    /// Total VRAM cost: `2 * n_layers * n_kv_heads * max_seq_len * head_dim * 4`
    /// bytes plus `model_dim * 4` for the smear scratch.
    ///
    /// Use [`Self::with_layer_slots`] to opt into per-layer rolling caches
    /// (e.g. a SWA layer with window=128 only needs 128 slots, regardless
    /// of `max_seq_len`).
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        model_dim: usize,
    ) -> Result<Self, ResidencyError> {
        let slots = vec![max_seq_len; n_layers];
        Self::with_layer_slots(slots, n_kv_heads, head_dim, max_seq_len, model_dim)
    }

    /// Allocate K and V buffers per-layer with caller-specified slot
    /// counts. A layer with `layer_slots[i] < max_seq_len` is a ring
    /// buffer: position `p` writes to slot `p % layer_slots[i]`. Sized
    /// matching the SWA window of that layer gives the actual KV-memory
    /// savings of sliding-window attention.
    ///
    /// Constraints:
    /// - `layer_slots.len() == n_layers` (one entry per layer).
    /// - Each `layer_slots[i] > 0` and `<= max_seq_len`.
    /// - `max_seq_len` is still the cap on the *highest position index*
    ///   the cache will ever see — used to size the smear scratch and
    ///   bound the public `max_seq_len()` getter. Pass the same value
    ///   you would to `new`.
    pub fn with_layer_slots(
        layer_slots: Vec<usize>,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        model_dim: usize,
    ) -> Result<Self, ResidencyError> {
        let n_layers = layer_slots.len();
        debug_assert!(n_layers > 0, "n_layers > 0 required");
        for (li, &s) in layer_slots.iter().enumerate() {
            debug_assert!(s > 0,
                "layer {li} has 0 slots — every layer needs at least 1");
            debug_assert!(s <= max_seq_len,
                "layer {li} slots ({s}) exceeds max_seq_len ({max_seq_len})");
        }

        let mut k_dev = Vec::with_capacity(n_layers);
        let mut v_dev = Vec::with_capacity(n_layers);
        for &slots in &layer_slots {
            let layer_bytes = n_kv_heads * slots * head_dim * 4;
            let kb = HipBuffer::new(layer_bytes)?;
            // Zero the buffer so any "look at uninitialized cache slot"
            // bug surfaces as a deterministic zero rather than a HIP
            // sanitizer race.
            zero_hip_buffer(&kb, n_kv_heads * slots * head_dim)?;
            let vb = HipBuffer::new(layer_bytes)?;
            zero_hip_buffer(&vb, n_kv_heads * slots * head_dim)?;
            k_dev.push(kb);
            v_dev.push(vb);
        }
        let prev_embedding_dev = HipBuffer::new(model_dim * 4)?;
        zero_hip_buffer(&prev_embedding_dev, model_dim)?;
        Ok(Self {
            k_dev, v_dev, slots: layer_slots, prev_embedding_dev,
            n_layers, n_kv_heads, head_dim, max_seq_len,
            seq_len: 0,
        })
    }

    /// Number of tokens currently in the cache.
    #[inline]
    pub fn seq_len(&self) -> usize { self.seq_len }
    /// Maximum cache capacity in tokens.
    #[inline]
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }
    /// Number of KV heads (cached at construction).
    #[inline]
    pub fn n_kv_heads(&self) -> usize { self.n_kv_heads }
    /// Head dimension.
    #[inline]
    pub fn head_dim(&self) -> usize { self.head_dim }
    /// Number of layers.
    #[inline]
    pub fn n_layers(&self) -> usize { self.n_layers }

    /// Set the cache length explicitly. Use after a prefill that wrote
    /// tokens 0..N at once via direct buffer access.
    pub fn set_seq_len(&mut self, n: usize) {
        debug_assert!(n <= self.max_seq_len);
        self.seq_len = n;
    }

    /// Reset to length 0 for a new conversation. Does NOT zero the
    /// underlying buffers (deterministic dispatch only ever reads slots
    /// 0..seq_len, so unused slots are dead memory). Call
    /// [`Self::reset_zero`] for an explicit wipe.
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Reset and zero every KV buffer. Slow (one D2D memset per layer);
    /// only use when cache contents could leak into the next forward.
    pub fn reset_zero(&mut self) -> Result<(), ResidencyError> {
        for (buf, &slots) in self.k_dev.iter().zip(self.slots.iter()) {
            zero_hip_buffer(buf, self.n_kv_heads * slots * self.head_dim)?;
        }
        for (buf, &slots) in self.v_dev.iter().zip(self.slots.iter()) {
            zero_hip_buffer(buf, self.n_kv_heads * slots * self.head_dim)?;
        }
        self.seq_len = 0;
        Ok(())
    }

    /// Write the freshly-projected K and V for one token at `position`
    /// in `layer`. Both `k_dev` and `v_dev` are length `kv_dim =
    /// n_kv_heads * head_dim` and laid out per-head-then-element
    /// (matching the host K/V projection output).
    ///
    /// The destination slot is `position % self.slots[layer]`. For
    /// linear layers (`slots[layer] == max_seq_len`) and `position <
    /// max_seq_len` this is just `position` — same behavior as before.
    /// For rolling layers (`slots[layer] < max_seq_len`) the modulus
    /// implements the ring-buffer eviction.
    pub fn write(
        &mut self,
        batch: &HipBatch,
        layer: usize,
        position: usize,
        k_dev: &GpuVec,
        v_dev: &GpuVec,
    ) -> Result<(), ResidencyError> {
        debug_assert!(layer < self.n_layers);
        debug_assert!(position < self.max_seq_len);
        let kv_dim = self.n_kv_heads * self.head_dim;
        debug_assert_eq!(k_dev.len(), kv_dim);
        debug_assert_eq!(v_dev.len(), kv_dim);

        let k_buf = match k_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };
        let v_buf = match v_dev {
            GpuVec::Hip(b) => b,
            other => return Err(ResidencyError::WrongVariant {
                expected: "Hip", got: other.variant_name(),
            }),
        };

        let layer_slots = self.slots[layer];
        let slot = position % layer_slots;
        let head_bytes = self.head_dim * 4;
        for kv_h in 0..self.n_kv_heads {
            let src_off = kv_h * self.head_dim * 4;
            let dst_off = kv_h * layer_slots * self.head_dim * 4
                        + slot * self.head_dim * 4;
            // K
            unsafe {
                hip_memcpy_d2d(
                    self.k_dev[layer].device_ptr().byte_add(dst_off),
                    k_buf.device_ptr().byte_add(src_off),
                    head_bytes,
                )?;
            }
            batch.note_dispatch()?;
            // V
            unsafe {
                hip_memcpy_d2d(
                    self.v_dev[layer].device_ptr().byte_add(dst_off),
                    v_buf.device_ptr().byte_add(src_off),
                    head_bytes,
                )?;
            }
            batch.note_dispatch()?;
        }
        if position + 1 > self.seq_len {
            self.seq_len = position + 1;
        }
        Ok(())
    }

    /// Raw device pointer to the start of layer `layer`'s K cache.
    /// Caller can offset by `kv_h * max_seq_len * head_dim` to find a
    /// per-head `[max_seq_len × head_dim]` row-major slab.
    #[inline]
    pub fn k_layer_ptr(&self, layer: usize) -> *const f32 {
        self.k_dev[layer].device_ptr() as *const f32
    }

    /// Raw device pointer to the start of layer `layer`'s V cache.
    #[inline]
    pub fn v_layer_ptr(&self, layer: usize) -> *const f32 {
        self.v_dev[layer].device_ptr() as *const f32
    }

    /// Number of slots allocated for `layer`. Equals `max_seq_len` for
    /// linear layers (default constructor) and the SWA window size for
    /// layers that opted into a rolling cache.
    #[inline]
    pub fn layer_slots(&self, layer: usize) -> usize { self.slots[layer] }

    /// Compute a window-aware view into the K slab for one (layer, kv_head)
    /// over the logical position range `[start, position]` (inclusive on
    /// both ends — caller passes the most recent `position`, not `position
    /// + 1`).
    ///
    /// For **linear** layers (`slots[layer] == max_seq_len`) and for
    /// **rolling** layers while the ring isn't full yet (`position <
    /// slots[layer]`), the slab is still in slot=position order, so the
    /// view is `(slab_base + start * head_dim, position + 1 - start)`
    /// — exactly the legacy linear semantics.
    ///
    /// For a **full rolling ring** (`position >= slots[layer]`), all slots
    /// are populated and hold the most recent `slots[layer]` positions in
    /// some cyclic permutation. Attention is permutation-invariant in
    /// position order (softmax + V-sum don't care about ordering of K/V
    /// pairs), so reading the whole ring as one contiguous slab from
    /// offset 0 with `rows = slots[layer]` is correct — no unroll copy
    /// needed. Caller's `start` is ignored in this case (the entire ring
    /// IS the window).
    pub fn k_slab_view(&self, layer: usize, kv_head: usize, start: usize, position: usize) -> KvSlabView {
        slab_view_inner(self.k_slab_ptr(layer, kv_head), self.head_dim, self.slots[layer], start, position)
    }

    /// V-side analog of [`Self::k_slab_view`]. Same view semantics — for
    /// permutation invariance to hold, K and V must be read with the
    /// **same** `start`/`position` so each row's K and V correspond to
    /// the same logical position.
    pub fn v_slab_view(&self, layer: usize, kv_head: usize, start: usize, position: usize) -> KvSlabView {
        slab_view_inner(self.v_slab_ptr(layer, kv_head), self.head_dim, self.slots[layer], start, position)
    }

    /// Pointer to the K slab for a specific (layer, kv_head).
    /// Slab shape `[slots[layer] × head_dim]` row-major; for matvec
    /// scoring use `attn_len` rows and `head_dim` cols when the layer
    /// is linear, or refer to [`Self::k_slab_view`] for a window-aware
    /// (ptr, rows) pair that handles linear and rolling uniformly.
    #[inline]
    pub fn k_slab_ptr(&self, layer: usize, kv_head: usize) -> *const f32 {
        let off = kv_head * self.slots[layer] * self.head_dim;
        unsafe { (self.k_layer_ptr(layer)).add(off) }
    }

    /// Pointer to the V slab for a specific (layer, kv_head).
    /// Slab shape `[slots[layer] × head_dim]` row-major.
    #[inline]
    pub fn v_slab_ptr(&self, layer: usize, kv_head: usize) -> *const f32 {
        let off = kv_head * self.slots[layer] * self.head_dim;
        unsafe { (self.v_layer_ptr(layer)).add(off) }
    }
}

/// HIP D2D copy of `bytes` from `src` to `dst`. Uses the synchronous
/// `hipMemcpy` with the D2D direction; faster than going through host
/// staging. Wraps the FFI in a typed `Result` so callers don't have to
/// import `BackendError`.
///
/// # Safety
///
/// `src` must be a valid hip-device pointer with at least `bytes` bytes
/// of readable memory; `dst` must be a valid hip-device pointer with at
/// least `bytes` bytes of writable memory. Both must come from the same
/// HIP context.
unsafe fn hip_memcpy_d2d(
    dst: *mut std::os::raw::c_void,
    src: *const std::os::raw::c_void,
    bytes: usize,
) -> Result<(), ResidencyError> {
    use modgrad_device::backend::rocm::ffi;
    /// `hipMemcpyKind::hipMemcpyDeviceToDevice = 3` per the HIP runtime header.
    /// The public ffi module only exposes H2D and D2H constants today, so we
    /// pin the literal value here. Bumping ROCm versions should not change
    /// this enum value (it is API-stable as of HIP 5.0+).
    const HIP_MEMCPY_DEVICE_TO_DEVICE: std::os::raw::c_int = 3;
    let err = unsafe {
        ffi::hipMemcpy(dst, src, bytes, HIP_MEMCPY_DEVICE_TO_DEVICE)
    };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "hipMemcpy D2D ({bytes} bytes): {}", ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}

/// Zero-fill `n_floats` f32 elements at the start of `buf`. Goes through
/// a host-allocated zero vector and `copy_from_host` — the slow path,
/// but only run at construction (or explicit `reset_zero`), so the
/// inefficiency is amortised.
fn zero_hip_buffer(buf: &HipBuffer, n_floats: usize) -> Result<(), ResidencyError> {
    let zeros = vec![0.0f32; n_floats];
    buf.copy_from_host(&zeros)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Shared lock: `modgrad_device::test_lock::hip_test_lock()` —
    //! serialises HIP runtime tests across the workspace.
    use super::*;
    use modgrad_device::backend::rocm::ffi::runtime_available;

    #[test]
    fn kv_cache_alloc_zero_init() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let n_layers = 2;
        let n_kv = 4;
        let head_dim = 32;
        let max_seq = 16;
        let model_dim = 128;
        let cache = KvCacheResident::new(n_layers, n_kv, head_dim, max_seq, model_dim)
            .expect("alloc");
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.max_seq_len(), max_seq);
        assert_eq!(cache.n_kv_heads(), n_kv);
        assert_eq!(cache.head_dim(), head_dim);

        // Sanity: download layer 0's K and verify it's all zero.
        let total = n_kv * cache.layer_slots(0) * head_dim;
        let mut buf = vec![1.0f32; total];
        cache.k_dev[0].copy_to_host(&mut buf).expect("d2h");
        assert!(buf.iter().all(|&v| v == 0.0), "fresh cache should be zero");
    }

    #[test]
    fn rolling_cache_with_layer_slots_writes_modular_slot() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let n_kv = 1;
        let head_dim = 4;
        let max_seq = 16;
        let model_dim = 4;
        // Layer 0 linear (size = max_seq), layer 1 rolling (window = 3).
        let mut cache = KvCacheResident::with_layer_slots(
            vec![max_seq, 3], n_kv, head_dim, max_seq, model_dim,
        ).expect("alloc");
        assert_eq!(cache.layer_slots(0), max_seq);
        assert_eq!(cache.layer_slots(1), 3);

        let kv_dim = n_kv * head_dim;
        let batch = HipBatch::new();
        // Write 5 positions into the rolling layer 1; each carries a
        // unique scalar so we can detect what survived.
        for pos in 0..5 {
            let k_host: Vec<f32> = vec![100.0 + pos as f32; kv_dim];
            let v_host: Vec<f32> = vec![200.0 + pos as f32; kv_dim];
            let mut k_dev = GpuVec::try_hip(kv_dim).expect("alloc k");
            k_dev.copy_from(&k_host);
            let mut v_dev = GpuVec::try_hip(kv_dim).expect("alloc v");
            v_dev.copy_from(&v_host);
            cache.write(&batch, 1, pos, &k_dev, &v_dev).expect("write");
        }
        batch.flush().expect("flush");

        // Layer 1 ring (slots=3): position p went to slot p%3, evicting
        // the oldest. After writing 0..5: slot 0 holds position 3 (most
        // recent with p%3==0), slot 1 holds position 4, slot 2 holds
        // position 2.
        let total = n_kv * 3 * head_dim;
        let mut full = vec![0.0f32; total];
        cache.k_dev[1].copy_to_host(&mut full).expect("d2h");
        assert_eq!(full[0], 103.0, "slot 0 should hold position 3 after eviction");
        assert_eq!(full[head_dim], 104.0, "slot 1 should hold position 4");
        assert_eq!(full[2 * head_dim], 102.0, "slot 2 should hold position 2");
    }

    #[test]
    fn kv_cache_write_and_read() {
        let _guard = modgrad_device::test_lock::hip_test_lock();
        if !runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }
        let n_layers = 2;
        let n_kv = 2;
        let head_dim = 4;
        let max_seq = 8;
        let model_dim = 8;
        let mut cache = KvCacheResident::new(n_layers, n_kv, head_dim, max_seq, model_dim)
            .expect("alloc");
        let kv_dim = n_kv * head_dim;

        // Token at position=2, layer=1: write K = [10..18], V = [20..28].
        let k_host: Vec<f32> = (0..kv_dim).map(|i| 10.0 + i as f32).collect();
        let v_host: Vec<f32> = (0..kv_dim).map(|i| 20.0 + i as f32).collect();
        let mut k_dev = GpuVec::try_hip(kv_dim).expect("alloc k");
        k_dev.copy_from(&k_host);
        let mut v_dev = GpuVec::try_hip(kv_dim).expect("alloc v");
        v_dev.copy_from(&v_host);

        let batch = HipBatch::new();
        cache.write(&batch, 1, 2, &k_dev, &v_dev).expect("write");
        batch.flush().expect("flush");
        assert_eq!(cache.seq_len(), 3);

        // Download the layer-1 K and verify the head-major layout:
        // For kv_h=0, position 2 should have [10, 11, 12, 13].
        // For kv_h=1, position 2 should have [14, 15, 16, 17].
        let total = n_kv * max_seq * head_dim;
        let mut full = vec![0.0f32; total];
        cache.k_dev[1].copy_to_host(&mut full).expect("d2h");

        for kv_h in 0..n_kv {
            let off = kv_h * max_seq * head_dim + 2 * head_dim;
            let slot = &full[off..off + head_dim];
            let expected = &k_host[kv_h * head_dim..(kv_h + 1) * head_dim];
            assert_eq!(slot, expected,
                "K head-major write mismatch for kv_h={kv_h}");
        }

        // V symmetric.
        let mut full_v = vec![0.0f32; total];
        cache.v_dev[1].copy_to_host(&mut full_v).expect("d2h");
        for kv_h in 0..n_kv {
            let off = kv_h * max_seq * head_dim + 2 * head_dim;
            let slot = &full_v[off..off + head_dim];
            let expected = &v_host[kv_h * head_dim..(kv_h + 1) * head_dim];
            assert_eq!(slot, expected, "V head-major write mismatch");
        }
    }
}
