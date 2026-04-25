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

/// Device-resident KV cache. See module docs.
pub struct KvCacheResident {
    /// One `HipBuffer` per layer for K. Each is sized
    /// `n_kv_heads * max_seq_len * head_dim * 4` bytes.
    pub k_dev: Vec<HipBuffer>,
    /// One `HipBuffer` per layer for V. Same shape as `k_dev`.
    pub v_dev: Vec<HipBuffer>,
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
    /// Total VRAM cost: `2 * n_layers * n_kv_heads * max_seq_len * head_dim * 4`
    /// bytes plus `model_dim * 4` for the smear scratch.
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        model_dim: usize,
    ) -> Result<Self, ResidencyError> {
        let layer_bytes = n_kv_heads * max_seq_len * head_dim * 4;
        let mut k_dev = Vec::with_capacity(n_layers);
        let mut v_dev = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let kb = HipBuffer::new(layer_bytes)?;
            // Zero the buffer so any "look at uninitialized cache slot"
            // bug surfaces as a deterministic zero rather than a HIP
            // sanitizer race.
            zero_hip_buffer(&kb, n_kv_heads * max_seq_len * head_dim)?;
            let vb = HipBuffer::new(layer_bytes)?;
            zero_hip_buffer(&vb, n_kv_heads * max_seq_len * head_dim)?;
            k_dev.push(kb);
            v_dev.push(vb);
        }
        let prev_embedding_dev = HipBuffer::new(model_dim * 4)?;
        zero_hip_buffer(&prev_embedding_dev, model_dim)?;
        Ok(Self {
            k_dev, v_dev, prev_embedding_dev,
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
        for buf in &self.k_dev {
            zero_hip_buffer(buf, self.n_kv_heads * self.max_seq_len * self.head_dim)?;
        }
        for buf in &self.v_dev {
            zero_hip_buffer(buf, self.n_kv_heads * self.max_seq_len * self.head_dim)?;
        }
        self.seq_len = 0;
        Ok(())
    }

    /// Write the freshly-projected K and V for one token at `position`
    /// in `layer`. Both `k_dev` and `v_dev` are length `kv_dim =
    /// n_kv_heads * head_dim` and laid out per-head-then-element
    /// (matching the host K/V projection output).
    ///
    /// Per kv-head, copies `head_dim` floats from `k_dev[kv_h * head_dim..]`
    /// into `self.k_dev[layer]` at offset `kv_h * max_seq_len * head_dim
    /// + position * head_dim`. Same for V.
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

        let head_bytes = self.head_dim * 4;
        for kv_h in 0..self.n_kv_heads {
            let src_off = kv_h * self.head_dim * 4;
            let dst_off = kv_h * self.max_seq_len * self.head_dim * 4
                        + position * self.head_dim * 4;
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

    /// Pointer to the K slab for a specific (layer, kv_head).
    /// Slab shape `[max_seq_len × head_dim]` row-major; for matvec
    /// scoring use `attn_len` rows and `head_dim` cols.
    #[inline]
    pub fn k_slab_ptr(&self, layer: usize, kv_head: usize) -> *const f32 {
        let off = kv_head * self.max_seq_len * self.head_dim;
        unsafe { (self.k_layer_ptr(layer)).add(off) }
    }

    /// Pointer to the V slab for a specific (layer, kv_head).
    /// Slab shape `[max_seq_len × head_dim]` row-major.
    #[inline]
    pub fn v_slab_ptr(&self, layer: usize, kv_head: usize) -> *const f32 {
        let off = kv_head * self.max_seq_len * self.head_dim;
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
    use super::*;
    use modgrad_device::backend::rocm::ffi::runtime_available;
    use std::sync::Mutex;

    /// Serialise HIP runtime tests — see the matching note in
    /// `crate::resident::tests`.
    static HIP_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn kv_cache_alloc_zero_init() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
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
        let total = n_kv * max_seq * head_dim;
        let mut buf = vec![1.0f32; total];
        cache.k_dev[0].copy_to_host(&mut buf).expect("d2h");
        assert!(buf.iter().all(|&v| v == 0.0), "fresh cache should be zero");
    }

    #[test]
    fn kv_cache_write_and_read() {
        let _guard = HIP_TEST_LOCK.lock().unwrap();
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
