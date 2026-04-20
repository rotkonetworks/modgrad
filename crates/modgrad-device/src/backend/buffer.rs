//! Cross-backend buffer abstraction.
//!
//! [`DeviceBuffer`] is the opaque handle into a backend's memory. It is
//! deliberately **backend-affine**: a `KfdBuffer` only speaks to the KFD
//! dispatcher, a `RocmBuffer` only to ROCm. Cross-backend mixing must be
//! a compile-time error — that invariant is enforced by the associated
//! type `Backend::Buffer`, not by any runtime check.
//!
//! If a caller really wants cross-backend data movement they write the
//! host round-trip by hand (`copy_to_host` → `copy_from_host`) — loudly,
//! and in the caller's source, not hidden inside a dispatcher.
//!
//! The default Buffer type for Backend impls that don't have VRAM is
//! [`HostBuffer`]: a thin `Vec<f32>` newtype. CPU-only backends and
//! backends that haven't plumbed device allocation yet (CUDA, Vulkan at
//! the time of Stage 2) all use it.
//!
//! See `tasks/compute-device-unify.md` — "Committed policy: DeviceBuffer
//! is backend-affine" — for the governing decision.

use super::BackendError;

/// Opaque handle to a contiguous f32 region owned by exactly one backend.
///
/// Implementations must be `Send + Sync + 'static` so a buffer can cross
/// threads when dispatched through a shared `ComputeCtx`.
///
/// The `copy_*_host` methods are the single sanctioned way to move data
/// between a buffer and plain `&[f32]` / `&mut [f32]` — a backend is free
/// to implement them as a direct memcpy (CPU buffers, BAR-mapped VRAM) or
/// as a `hipMemcpy`-style DMA (ROCm). Callers should not assume anything
/// about cost; a copy_to_host on a KFD arena buffer may be a hot-path-
/// safe pointer deref, whereas on ROCm it's a PCIe round-trip.
pub trait DeviceBuffer: Send + Sync + 'static {
    /// Short name of the backend this buffer belongs to (matches
    /// `Backend::name()`). Used in error messages and telemetry.
    fn backend_name(&self) -> &'static str;

    /// Number of f32 elements the buffer can hold. This is the **logical**
    /// capacity — backends may over-allocate for alignment (KFD rounds to
    /// 4 KiB pages), but `len()` always reports the requested size.
    fn len(&self) -> usize;

    /// True when `len() == 0`. Convenience helper, mirrors `Vec::is_empty`.
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Copy `src` into the buffer starting at offset 0.
    ///
    /// `src.len()` must not exceed `self.len()`. A shorter `src` is
    /// permitted — trailing elements are left untouched.
    fn copy_from_host(&mut self, src: &[f32]) -> Result<(), BackendError>;

    /// Copy the buffer into `dst` starting at offset 0.
    ///
    /// `dst.len()` must not exceed `self.len()`. A shorter `dst` reads
    /// only the prefix of the buffer.
    fn copy_to_host(&self, dst: &mut [f32]) -> Result<(), BackendError>;
}

/// Default `DeviceBuffer` — a plain heap-allocated `Vec<f32>`.
///
/// Used by CPU backends and by any backend that hasn't plumbed a device-
/// resident allocation path yet (CUDA, Vulkan as of Stage 2). The
/// `backend_name()` is always `"host"`; the buffer is valid to pass
/// across backends *only* because it lives in host memory — but the
/// type system still enforces affinity via `Backend::Buffer`, so a
/// `HostBuffer` handed to a `ComputeCtx<KfdBackend>` still won't
/// compile unless `KfdBackend::Buffer == HostBuffer` (it isn't).
pub struct HostBuffer {
    data: Vec<f32>,
}

impl HostBuffer {
    /// Allocate `n` zeroed f32s on the host.
    pub fn new(n: usize) -> Self {
        Self { data: vec![0.0f32; n] }
    }

    /// Borrow the contents as `&[f32]`. Useful when a caller wants to
    /// feed a `HostBuffer` into an `Op`'s slice-typed argument without
    /// going through `copy_to_host`.
    pub fn as_slice(&self) -> &[f32] { &self.data }

    /// Mutable borrow, same rationale as `as_slice`.
    pub fn as_mut_slice(&mut self) -> &mut [f32] { &mut self.data }
}

impl DeviceBuffer for HostBuffer {
    fn backend_name(&self) -> &'static str { "host" }

    fn len(&self) -> usize { self.data.len() }

    fn copy_from_host(&mut self, src: &[f32]) -> Result<(), BackendError> {
        if src.len() > self.data.len() {
            return Err(BackendError::Runtime(format!(
                "HostBuffer::copy_from_host: src.len()={} > buffer.len()={}",
                src.len(), self.data.len(),
            )));
        }
        self.data[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [f32]) -> Result<(), BackendError> {
        if dst.len() > self.data.len() {
            return Err(BackendError::Runtime(format!(
                "HostBuffer::copy_to_host: dst.len()={} > buffer.len()={}",
                dst.len(), self.data.len(),
            )));
        }
        dst.copy_from_slice(&self.data[..dst.len()]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_buffer_roundtrip() {
        let mut b = HostBuffer::new(4);
        assert_eq!(b.len(), 4);
        assert_eq!(b.backend_name(), "host");
        b.copy_from_host(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut out = vec![0.0f32; 4];
        b.copy_to_host(&mut out).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn host_buffer_rejects_oversize_src() {
        let mut b = HostBuffer::new(2);
        let err = b.copy_from_host(&[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, BackendError::Runtime(_)));
    }

    #[test]
    fn host_buffer_rejects_oversize_dst() {
        let b = HostBuffer::new(2);
        let mut out = vec![0.0f32; 3];
        let err = b.copy_to_host(&mut out).unwrap_err();
        assert!(matches!(err, BackendError::Runtime(_)));
    }
}
