//! Device-located tensors: `CpuTensor<T>` lives in host RAM, `VramTensor<T>` lives
//! on the GPU. The types are distinct — a CPU-only op literally cannot accept a
//! VRAM tensor without an explicit `.download()` first, and vice versa.
//!
//! Why
//! ---
//! The older API takes `&[f32]` / `&mut [f32]` everywhere and the GPU path
//! decides at runtime whether to upload. That leads to:
//!   * hidden caches (prepare_weights keyed by ptr identity),
//!   * duplicated storage (VramMirror + prepare_weights cache),
//!   * per-call `gpu_enabled()` branches on the hot path,
//!   * and — most painfully — "works on CPU, silently CPU on GPU" bugs like
//!     the `adamw.co`-as-relocatable-ELF issue this session.
//!
//! With typed tensors, the location is part of the type and operations dispatch
//! on it at compile time. Training loops look like:
//!
//! ```ignore
//! let w = CpuTensor::<f32>::from_vec(vec![..], [out, in]).move_to_vram()?;
//! for step in .. {
//!     forward_backward_vram(&w, &batch, &mut grads)?;
//!     adamw_in_place(&w, &grads, &mut m, &mut v, ..)?;   // all VRAM
//! }
//! w.download()?.save(path)?;                             // one-time
//! ```
//!
//! Safety
//! ------
//! * `VramTensor<T>` owns a `GpuBuffer`. Drop unmaps the GPU mapping.
//! * Slices returned from `as_slice` / `as_mut_slice` are BAR-mapped. Reads
//!   and writes go straight to VRAM; no staging copy. Requires resizable
//!   BAR (already required by the rest of the SDK on this hardware).
//! * `as_mut_slice` takes `&mut self` so aliasing is impossible in safe
//!   Rust. Callers who want multiple mutable views across a training step
//!   must split the tensor or use distinct tensors.
//! * GPU kernels reading a tensor while the CPU holds an `&mut [T]` is a
//!   data race at the hardware level. Call `flush_gpu()` between kernel
//!   dispatch and CPU access.

use modgrad_device::kfd::memory::GpuBuffer;
use std::marker::PhantomData;

/// Element type usable inside a device tensor. Currently only `f32` is
/// wired up; additional types (f16, i8, etc.) plug in by implementing
/// this trait and adding kernel paths.
pub trait DeviceElem: Copy + Default + Send + Sync + 'static {
    /// Size in bytes of one element. Must match `std::mem::size_of::<Self>()`
    /// but exposed here for generic buffer sizing.
    const BYTE_SIZE: usize;
}

impl DeviceElem for f32 {
    const BYTE_SIZE: usize = 4;
}

/// Host-side tensor, row-major. Backed by a `Vec<T>`.
#[derive(Debug, Clone)]
pub struct CpuTensor<T: DeviceElem> {
    data: Vec<T>,
    dims: Vec<usize>,
}

impl<T: DeviceElem> CpuTensor<T> {
    /// Wrap an existing `Vec<T>` with a shape. Panics if `data.len()` doesn't
    /// equal the product of `dims`.
    pub fn from_vec(data: Vec<T>, dims: impl Into<Vec<usize>>) -> Self {
        let dims = dims.into();
        let expected: usize = dims.iter().product();
        assert_eq!(data.len(), expected,
            "CpuTensor::from_vec: data len {} ≠ product of dims {:?} = {}",
            data.len(), dims, expected);
        Self { data, dims }
    }

    /// Allocate a CPU tensor filled with `T::default()` (zeros for f32).
    pub fn zeros(dims: impl Into<Vec<usize>>) -> Self {
        let dims = dims.into();
        let n: usize = dims.iter().product();
        Self { data: vec![T::default(); n], dims }
    }

    pub fn dims(&self) -> &[usize] { &self.dims }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn as_slice(&self) -> &[T] { &self.data }
    pub fn as_mut_slice(&mut self) -> &mut [T] { &mut self.data }
    pub fn into_vec(self) -> Vec<T> { self.data }
}

/// GPU-resident tensor, row-major. Backed by a permanent VRAM allocation.
///
/// The `GpuBuffer` owns its VRAM mapping; dropping the tensor unmaps it.
/// `as_slice` / `as_mut_slice` give BAR-mapped direct-access views — reads
/// and writes go straight to VRAM via the resizable-BAR mapping, with no
/// staging copy.
pub struct VramTensor<T: DeviceElem> {
    buf: GpuBuffer,
    dims: Vec<usize>,
    len: usize,
    _marker: PhantomData<T>,
}

// Safe to share across threads: GpuBuffer is Send+Sync and the tensor adds no
// non-Send/Sync state.
unsafe impl<T: DeviceElem> Send for VramTensor<T> {}
unsafe impl<T: DeviceElem> Sync for VramTensor<T> {}

impl<T: DeviceElem> VramTensor<T> {
    /// Allocate a VRAM tensor of the given shape, zero-initialised.
    /// Returns `None` when no KFD GPU is available, or the alloc fails.
    pub fn zeros(dims: impl Into<Vec<usize>>) -> Option<Self> {
        let dims = dims.into();
        let n: usize = dims.iter().product();
        let bytes = (n * T::BYTE_SIZE) as u64;
        // Round up to 4 KiB page granularity — KFD alloc is page-sized.
        let cap = ((bytes + 4095) & !4095).max(4096);

        let buf = crate::alloc_device_vram(cap)?;
        // Zero the first `n` elements via BAR. `write_bytes` is safe here:
        // the buffer has CPU access (alloc_vram uses PUBLIC flag) and we own
        // the allocation, so no races.
        unsafe {
            let p = buf.cpu_ptr as *mut u8;
            std::ptr::write_bytes(p, 0, n * T::BYTE_SIZE);
        }
        Some(Self { buf, dims, len: n, _marker: PhantomData })
    }

    /// Allocate and upload from a `CpuTensor`. Equivalent to `zeros` then
    /// `as_mut_slice().copy_from_slice(...)` but fused into one alloc.
    pub fn from_cpu(cpu: &CpuTensor<T>) -> Option<Self> {
        let mut v = Self::zeros(cpu.dims().to_vec())?;
        v.as_mut_slice().copy_from_slice(cpu.as_slice());
        Some(v)
    }

    /// Download to a fresh `CpuTensor`. Read through the BAR mapping.
    pub fn download(&self) -> CpuTensor<T> {
        let mut out = CpuTensor::<T>::zeros(self.dims.clone());
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }

    /// Destructively take the VRAM buffer back as CPU data. Frees VRAM.
    pub fn into_cpu(self) -> CpuTensor<T> { self.download() }

    /// BAR-mapped read view. Safe as long as no kernel is writing this
    /// tensor in flight — caller must `flush_gpu()` before reading after
    /// a GPU write.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.buf.cpu_ptr as *const T, self.len) }
    }

    /// BAR-mapped mutable view. Requires `&mut self` so aliasing is
    /// prevented in safe Rust. Callers racing this against in-flight GPU
    /// kernels is a hardware-level data race — flush first.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.buf.cpu_ptr as *mut T, self.len) }
    }

    /// GPU virtual address for kernel dispatch. Not a safe pointer — only
    /// usable by KFD kernargs on the same device.
    pub fn va(&self) -> u64 { self.buf.va_addr }

    pub fn dims(&self) -> &[usize] { &self.dims }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl<T: DeviceElem> CpuTensor<T> {
    /// Upload this tensor to VRAM. Consumes `self`; the CPU Vec is dropped
    /// once the VRAM alloc succeeds. Returns `None` if GPU unavailable.
    pub fn move_to_vram(self) -> Option<VramTensor<T>> {
        VramTensor::from_cpu(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_tensor_zeros_and_shape() {
        let t: CpuTensor<f32> = CpuTensor::zeros([3, 4]);
        assert_eq!(t.dims(), &[3, 4]);
        assert_eq!(t.len(), 12);
        assert!(t.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    #[should_panic(expected = "data len 4 ≠ product of dims")]
    fn cpu_tensor_from_vec_shape_mismatch_panics() {
        let _ = CpuTensor::from_vec(vec![0.0f32; 4], [3, 4]);
    }

    #[test]
    fn vram_roundtrip_or_skip() {
        // Graceful when no GPU present — `zeros` returns None.
        let cpu = CpuTensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]);
        let vram = match cpu.clone().move_to_vram() {
            Some(v) => v,
            None => return, // no GPU on this host, skip.
        };
        assert_eq!(vram.dims(), &[2, 2]);
        assert_eq!(vram.len(), 4);
        let back = vram.download();
        assert_eq!(back.as_slice(), cpu.as_slice());
    }
}
