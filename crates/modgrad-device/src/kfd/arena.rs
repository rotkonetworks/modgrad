//! VRAM arena: pre-allocated GPU memory pool that hands out slices
//! usable from both GPU (full bandwidth) and CPU (BAR-mapped).
//!
//! The key insight: `GpuBuffer.cpu_ptr` is a BAR-mapped pointer to VRAM.
//! Slices backed by this pointer work as normal `&mut [f32]` on the CPU
//! side (reads/writes go through PCIe BAR), while the GPU accesses the
//! same memory at full VRAM bandwidth via `va_addr`.
//!
//! When the StreamEngine sees a pointer that falls inside the arena's
//! VRAM range, it skips upload/download and uses the GPU VA directly.
//! This eliminates ALL PCIe round-trips between chained GPU ops.

use super::memory::GpuBuffer;
use super::HsaDevice;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A VRAM arena that hands out f32 slices backed by GPU memory.
pub struct VramArena {
    buf: GpuBuffer,
    /// Next free offset in bytes.
    cursor: AtomicUsize,
    /// Total capacity in bytes.
    capacity: usize,
}

/// A handle to a VRAM-backed slice.
/// The slice can be used as `&[f32]`/`&mut [f32]` on CPU (BAR-mapped),
/// and the GPU VA is available for kernel dispatch.
#[derive(Clone, Copy)]
pub struct VramSlice {
    /// GPU virtual address of the start of this slice.
    pub va: u64,
    /// CPU-mapped pointer to the same memory.
    pub ptr: *mut f32,
    /// Number of f32 elements.
    pub len: usize,
}

unsafe impl Send for VramSlice {}
unsafe impl Sync for VramSlice {}

impl VramSlice {
    /// Get as CPU-accessible slice (reads/writes through PCIe BAR).
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get as mutable CPU-accessible slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Zero the buffer (through CPU BAR — small buffers only).
    pub fn zero(&self) {
        unsafe { std::ptr::write_bytes(self.ptr, 0, self.len); }
    }

    /// Copy from a CPU slice into VRAM (through BAR mapping).
    pub fn upload(&self, data: &[f32]) {
        assert!(data.len() <= self.len, "upload: data.len()={} > slice.len()={}", data.len(), self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, data.len());
        }
    }

    /// Copy from VRAM into a CPU Vec (through BAR mapping).
    pub fn download(&self, n: usize) -> Vec<f32> {
        assert!(n <= self.len);
        let mut out = vec![0.0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, out.as_mut_ptr(), n);
        }
        out
    }

    /// Copy from VRAM into an existing CPU slice.
    pub fn download_into(&self, dst: &mut [f32]) {
        let n = dst.len().min(self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, dst.as_mut_ptr(), n);
        }
    }
}

impl VramArena {
    /// Allocate a new VRAM arena of the given size in bytes.
    pub fn new(dev: &HsaDevice, size_bytes: usize) -> Option<Self> {
        let aligned = ((size_bytes + 4095) & !4095) as u64;
        let buf = dev.alloc.alloc_vram(aligned).ok()?;
        // Zero the entire arena
        unsafe { std::ptr::write_bytes(buf.cpu_ptr, 0, aligned as usize); }
        Some(VramArena {
            capacity: aligned as usize,
            buf,
            cursor: AtomicUsize::new(0),
        })
    }

    /// Allocate `n` f32 elements from the arena. Returns None if full.
    /// Alignment: 256-byte aligned for GPU coalescing.
    pub fn alloc(&self, n: usize) -> Option<VramSlice> {
        let bytes = n * 4;
        let aligned_bytes = (bytes + 255) & !255; // 256-byte align
        let offset = self.cursor.fetch_add(aligned_bytes, Ordering::Relaxed);
        if offset + aligned_bytes > self.capacity {
            return None;
        }
        Some(VramSlice {
            va: self.buf.va_addr + offset as u64,
            ptr: unsafe { (self.buf.cpu_ptr.add(offset)) as *mut f32 },
            len: n,
        })
    }

    /// Reset the arena (free all allocations). Not thread-safe.
    pub fn reset(&self) {
        self.cursor.store(0, Ordering::Relaxed);
    }

    /// Check if a CPU pointer falls within this arena's VRAM mapping.
    /// If so, return the corresponding GPU VA.
    pub fn resolve_va(&self, ptr: *const f32) -> Option<u64> {
        let p = ptr as usize;
        let base = self.buf.cpu_ptr as usize;
        let end = base + self.capacity;
        if p >= base && p < end {
            let offset = p - base;
            Some(self.buf.va_addr + offset as u64)
        } else {
            None
        }
    }

    /// Base GPU VA of the arena.
    pub fn base_va(&self) -> u64 { self.buf.va_addr }

    /// Base CPU pointer of the arena.
    pub fn base_ptr(&self) -> *mut u8 { self.buf.cpu_ptr }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize { self.capacity }

    /// Used bytes.
    pub fn used(&self) -> usize { self.cursor.load(Ordering::Relaxed) }
}
