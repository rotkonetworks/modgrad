//! GPU memory management via KFD ioctls.
//!
//! Three allocation types matching tinygrad's KFDIface.alloc():
//!   - VRAM: device-local, for weights and compute buffers
//!   - GTT: system memory visible to GPU, for ring buffers and signals
//!   - Userptr: user-allocated memory mapped to GPU

use super::ioctl;
use std::os::unix::io::RawFd;
use std::ptr;

const MAP_NORESERVE: libc::c_int = 0x4000;

/// A GPU-visible memory buffer with optional CPU-mapped view.
///
/// Safety invariants:
/// - `va_addr` is always a valid GPU virtual address while the buffer lives
/// - `cpu_ptr` is non-null IFF `has_cpu_access` is true
/// - `mmap_addr` is the original mmap reservation (may differ from cpu_ptr for private VRAM)
/// - `kfd_fd` must remain open for the lifetime of this buffer (enforced by Arc in GpuAllocator)
pub struct GpuBuffer {
    pub va_addr: u64,
    pub size: u64,
    pub handle: u64,
    pub cpu_ptr: *mut u8,
    /// Original mmap address for munmap on drop (may be PROT_NONE reservation for private VRAM)
    mmap_addr: *mut libc::c_void,
    mmap_size: usize,
    pub kfd_fd: RawFd,
    pub gpu_id: u32,
    flags: u32,
    has_cpu_access: bool,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    /// True if this buffer can be read/written from CPU.
    pub fn has_cpu_access(&self) -> bool { self.has_cpu_access }

    /// Panic message for null access.
    fn assert_cpu_access(&self) {
        assert!(self.has_cpu_access && !self.cpu_ptr.is_null(),
            "attempted CPU access on GPU-only buffer (flags=0x{:08x})", self.flags);
    }

    /// Write data to the buffer at the given byte offset.
    pub fn write(&self, offset: usize, data: &[u8]) {
        self.assert_cpu_access();
        assert!(offset + data.len() <= self.size as usize,
            "write out of bounds: offset={} len={} size={}", offset, data.len(), self.size);
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.cpu_ptr.add(offset), data.len());
        }
    }

    /// Read data from the buffer.
    pub fn read(&self, offset: usize, len: usize) -> Vec<u8> {
        self.assert_cpu_access();
        assert!(offset + len <= self.size as usize,
            "read out of bounds: offset={} len={} size={}", offset, len, self.size);
        let mut out = vec![0u8; len];
        unsafe {
            ptr::copy_nonoverlapping(self.cpu_ptr.add(offset), out.as_mut_ptr(), len);
        }
        out
    }

    /// Get a typed slice view of the buffer (requires CPU access).
    pub fn as_slice<T>(&self) -> &[T] {
        self.assert_cpu_access();
        let count = self.size as usize / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(self.cpu_ptr as *const T, count) }
    }

    /// Get a mutable typed slice view (requires &mut self to prevent aliasing).
    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        self.assert_cpu_access();
        let count = self.size as usize / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(self.cpu_ptr as *mut T, count) }
    }

    /// Write f32 slice into the buffer at byte offset.
    pub fn write_f32(&self, offset: usize, data: &[f32]) {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.write(offset, bytes);
    }

    /// Read f32 slice from the buffer.
    pub fn read_f32(&self, offset: usize, count: usize) -> Vec<f32> {
        self.assert_cpu_access();
        assert!(offset + count * 4 <= self.size as usize,
            "read_f32 out of bounds: offset={} count={} size={}", offset, count, self.size);
        let mut out = vec![0.0f32; count];
        unsafe {
            ptr::copy_nonoverlapping(
                self.cpu_ptr.add(offset),
                out.as_mut_ptr() as *mut u8,
                count * 4,
            );
        }
        out
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // Release GPU mapping and handle
        if self.handle != 0 {
            let gpu_ids = [self.gpu_id];
            let _ = ioctl::unmap_memory(self.kfd_fd, self.handle, &gpu_ids);
            let _ = ioctl::free_memory(self.kfd_fd, self.handle);
        }
        // Release CPU mmap (covers both CPU-mapped and VA-reservation-only cases)
        if !self.mmap_addr.is_null() && self.mmap_size > 0 {
            unsafe { libc::munmap(self.mmap_addr, self.mmap_size); }
        }
    }
}

/// GPU memory allocator.
pub struct GpuAllocator {
    pub kfd_fd: RawFd,
    pub drm_fd: RawFd,
    pub gpu_id: u32,
}

impl GpuAllocator {
    /// Allocate VRAM (device-local). CPU-visible through resizable BAR.
    pub fn alloc_vram(&self, size: u64) -> std::io::Result<GpuBuffer> {
        let flags = ioctl::ALLOC_MEM_FLAGS_VRAM
            | ioctl::ALLOC_MEM_FLAGS_WRITABLE
            | ioctl::ALLOC_MEM_FLAGS_EXECUTABLE
            | ioctl::ALLOC_MEM_FLAGS_PUBLIC
            | ioctl::ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
        self.alloc_internal(size, flags, false)
    }

    /// Allocate GTT (system memory visible to GPU). Uncached, coherent.
    /// Used for event pages, signal memory, control structures.
    pub fn alloc_gtt(&self, size: u64) -> std::io::Result<GpuBuffer> {
        let flags = ioctl::ALLOC_MEM_FLAGS_GTT
            | ioctl::ALLOC_MEM_FLAGS_WRITABLE
            | ioctl::ALLOC_MEM_FLAGS_EXECUTABLE
            | ioctl::ALLOC_MEM_FLAGS_NO_SUBSTITUTE
            | ioctl::ALLOC_MEM_FLAGS_COHERENT
            | ioctl::ALLOC_MEM_FLAGS_UNCACHED;
        self.alloc_internal(size, flags, false)
    }

    /// Allocate GTT with PUBLIC flag (CPU + GPU visible).
    /// Used for ring buffers and read/write pointers.
    pub fn alloc_gtt_public(&self, size: u64) -> std::io::Result<GpuBuffer> {
        let flags = ioctl::ALLOC_MEM_FLAGS_GTT
            | ioctl::ALLOC_MEM_FLAGS_WRITABLE
            | ioctl::ALLOC_MEM_FLAGS_EXECUTABLE
            | ioctl::ALLOC_MEM_FLAGS_PUBLIC
            | ioctl::ALLOC_MEM_FLAGS_NO_SUBSTITUTE
            | ioctl::ALLOC_MEM_FLAGS_COHERENT
            | ioctl::ALLOC_MEM_FLAGS_UNCACHED;
        self.alloc_internal(size, flags, false)
    }

    /// Allocate VRAM without PUBLIC flag (not CPU-accessible).
    /// For internal GPU buffers like EOP and ctx_save_restore.
    pub fn alloc_vram_private(&self, size: u64) -> std::io::Result<GpuBuffer> {
        let flags = ioctl::ALLOC_MEM_FLAGS_VRAM
            | ioctl::ALLOC_MEM_FLAGS_WRITABLE
            | ioctl::ALLOC_MEM_FLAGS_EXECUTABLE
            | ioctl::ALLOC_MEM_FLAGS_NO_SUBSTITUTE;

        // Reserve VA space (no CPU access needed)
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size as usize,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | MAP_NORESERVE,
                -1, 0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }

        #[cfg(debug_assertions)]
        eprintln!("        vram_private: size=0x{:x} flags=0x{:08x} va=0x{:x}", size, flags, addr as u64);
        let mem = ioctl::alloc_memory(
            self.kfd_fd, addr as u64, size, self.gpu_id, flags, 0,
        )?;
        #[cfg(debug_assertions)]
        eprintln!("        vram_private: handle=0x{:x} va=0x{:x}", mem.handle, mem.va_addr);

        // No CPU mmap for private VRAM — GPU-only buffer
        let gpu_ids = [self.gpu_id];
        ioctl::map_memory(self.kfd_fd, mem.handle, &gpu_ids)?;

        Ok(GpuBuffer {
            va_addr: mem.va_addr,
            size: mem.size,
            handle: mem.handle,
            cpu_ptr: ptr::null_mut(),
            mmap_addr: addr as *mut libc::c_void,
            mmap_size: size as usize,
            kfd_fd: self.kfd_fd,
            gpu_id: self.gpu_id,
            flags,
            has_cpu_access: false,
        })
    }

    /// Allocate userptr (user-managed memory mapped to GPU).
    /// With COHERENT + UNCACHED — for ring buffers and control structures.
    pub fn alloc_userptr(&self, size: u64) -> std::io::Result<GpuBuffer> {
        self.alloc_userptr_flags(size, ioctl::ALLOC_MEM_FLAGS_USERPTR
            | ioctl::ALLOC_MEM_FLAGS_WRITABLE
            | ioctl::ALLOC_MEM_FLAGS_EXECUTABLE
            | ioctl::ALLOC_MEM_FLAGS_NO_SUBSTITUTE
            | ioctl::ALLOC_MEM_FLAGS_COHERENT
            | ioctl::ALLOC_MEM_FLAGS_UNCACHED)
    }

    /// Allocate userptr with PUBLIC flag — for signal/scratch buffers.
    /// Matches tinygrad's alloc(cpu_access=True) = flags 0xF0000004.
    pub fn alloc_userptr_public(&self, size: u64) -> std::io::Result<GpuBuffer> {
        self.alloc_userptr_flags(size, ioctl::ALLOC_MEM_FLAGS_USERPTR
            | ioctl::ALLOC_MEM_FLAGS_WRITABLE
            | ioctl::ALLOC_MEM_FLAGS_EXECUTABLE
            | ioctl::ALLOC_MEM_FLAGS_PUBLIC
            | ioctl::ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
    }

    fn alloc_userptr_flags(&self, size: u64, flags: u32) -> std::io::Result<GpuBuffer> {

        // Userptr: mmap first, then tell KFD about it
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                -1, 0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }

        let mem = ioctl::alloc_memory(
            self.kfd_fd, addr as u64, size, self.gpu_id, flags, addr as u64,
        )?;

        let gpu_ids = [self.gpu_id];
        ioctl::map_memory(self.kfd_fd, mem.handle, &gpu_ids)?;

        Ok(GpuBuffer {
            va_addr: mem.va_addr,
            size: mem.size,
            handle: mem.handle,
            cpu_ptr: addr as *mut u8,
            mmap_addr: addr as *mut libc::c_void,
            mmap_size: size as usize,
            kfd_fd: self.kfd_fd,
            gpu_id: self.gpu_id,
            flags,
            has_cpu_access: true,
        })
    }

    fn alloc_internal(&self, size: u64, flags: u32, _is_userptr: bool) -> std::io::Result<GpuBuffer> {
        #[cfg(debug_assertions)]
        eprintln!("        alloc_internal: size=0x{:x} flags=0x{:08x}", size, flags);
        // Reserve VA space
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size as usize,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | MAP_NORESERVE,
                -1, 0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }

        // KFD alloc
        let mem = ioctl::alloc_memory(
            self.kfd_fd, addr as u64, size, self.gpu_id, flags, 0,
        )?;

        // mmap the allocation for CPU access
        let cpu_ptr = unsafe {
            libc::mmap(
                mem.va_addr as *mut libc::c_void,
                mem.size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                self.drm_fd,
                mem.mmap_offset as libc::off_t,
            )
        };
        if cpu_ptr == libc::MAP_FAILED {
            ioctl::free_memory(self.kfd_fd, mem.handle)?;
            return Err(std::io::Error::last_os_error());
        }

        // Map to GPU
        let gpu_ids = [self.gpu_id];
        ioctl::map_memory(self.kfd_fd, mem.handle, &gpu_ids)?;

        Ok(GpuBuffer {
            va_addr: mem.va_addr,
            size: mem.size,
            handle: mem.handle,
            cpu_ptr: cpu_ptr as *mut u8,
            mmap_addr: cpu_ptr as *mut libc::c_void,
            mmap_size: mem.size as usize,
            kfd_fd: self.kfd_fd,
            gpu_id: self.gpu_id,
            flags,
            has_cpu_access: true,
        })
    }
}
