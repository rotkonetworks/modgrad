//! Compute queue management: ring buffer, doorbell, PM4 submission.
//!
//! PM4 (Packet Manager 4) is the command packet format for GFX/compute.
//! We use QUEUE_TYPE_COMPUTE (raw PM4) rather than AQL for simplicity.

use super::ioctl;
use super::memory::{GpuAllocator, GpuBuffer};
use std::os::unix::io::RawFd;
use std::ptr;

/// PM4 packet header: type 3
const fn pkt3(opcode: u32, count: u32) -> u32 {
    (3 << 30) | ((count.wrapping_sub(1) & 0x3FFF) << 16) | (opcode << 8)
}

// ─── PM4 opcodes (GFX11/RDNA3) ─────────────────────────────

pub const PACKET3_NOP: u32 = 0x10;
pub const PACKET3_SET_SH_REG: u32 = 0x76;
pub const PACKET3_DISPATCH_DIRECT: u32 = 0x15;
pub const PACKET3_ACQUIRE_MEM: u32 = 0x58;
pub const PACKET3_RELEASE_MEM: u32 = 0x49;
pub const PACKET3_WAIT_REG_MEM: u32 = 0x3C;
pub const PACKET3_EVENT_WRITE: u32 = 0x46;

// SET_SH_REG base for GFX11
pub const SH_REG_BASE: u32 = 0x2C00;

// Compute shader registers (offsets from SH_REG_BASE)
// These are GFX11 (RDNA3) specific — from gc_11_0_0_offset.h
pub const COMPUTE_PGM_LO: u32 = 0x2E0C - SH_REG_BASE;
pub const COMPUTE_PGM_HI: u32 = 0x2E0D - SH_REG_BASE;
pub const COMPUTE_PGM_RSRC1: u32 = 0x2E12 - SH_REG_BASE;
pub const COMPUTE_PGM_RSRC2: u32 = 0x2E13 - SH_REG_BASE;
pub const COMPUTE_PGM_RSRC3: u32 = 0x2E28 - SH_REG_BASE;
pub const COMPUTE_TMPRING_SIZE: u32 = 0x2E18 - SH_REG_BASE;
pub const COMPUTE_USER_DATA_0: u32 = 0x2E40 - SH_REG_BASE;
pub const COMPUTE_RESOURCE_LIMITS: u32 = 0x2E15 - SH_REG_BASE;
pub const COMPUTE_START_X: u32 = 0x2E04 - SH_REG_BASE;
pub const COMPUTE_NUM_THREAD_X: u32 = 0x2E07 - SH_REG_BASE;
pub const COMPUTE_RESTART_X: u32 = 0x2E1B - SH_REG_BASE;

// DISPATCH_INITIATOR bits
pub const COMPUTE_SHADER_EN: u32 = 1 << 0;
pub const CS_W32_EN: u32 = 1 << 15; // wave32 mode (RDNA3)
pub const FORCE_START_AT_000: u32 = 1 << 2;

// ACQUIRE_MEM / RELEASE_MEM cache control (GFX11)
pub const GCR_GLI_INV_GL1: u32 = 1 << 0;
pub const GCR_GL2_INV: u32 = 1 << 14;
pub const GCR_GL2_WB: u32 = 1 << 15;
pub const GCR_GLM_INV: u32 = 1 << 5;
pub const GCR_GLM_WB: u32 = 1 << 4;
pub const GCR_GLV_INV: u32 = 1 << 9;
pub const GCR_GLK_INV: u32 = 1 << 12;
pub const GCR_SEQ_FORWARD: u32 = 1 << 16;

// RELEASE_MEM event types
pub const EVENT_TYPE_CACHE_FLUSH: u32 = 0x06;
pub const EVENT_TYPE_CS_PARTIAL_FLUSH: u32 = 0x04;
pub const DATA_SEL_VALUE_32BIT: u32 = 1;
pub const DATA_SEL_VALUE_64BIT: u32 = 2;
pub const INT_SEL_SEND_DATA_AFTER_WR_CONFIRM: u32 = 3;

// Wait conditions
pub const WAIT_REG_MEM_FUNCTION_GEQ: u32 = 5;
pub const WAIT_REG_MEM_MEM_SPACE_MEM: u32 = 1;

/// Ring buffer size (16MB, matching tinygrad production)
const RING_SIZE: u64 = 0x1000000;

/// Size for EOP (end-of-pipe) buffer
const EOP_SIZE: u64 = 0x1000;

/// ctx_save_restore size (ioctl field value, from sysfs cwsr_size)
const CTX_SAVE_SIZE: u64 = 0xAA4000;

/// Debug memory size: wave_count * 32, rounded to 64.
/// wave_count = max_waves_per_simd * simd_count = 16 * 64 = 1024
/// 1024 * 32 = 32768 = 0x8000
const DEBUG_MEM_SIZE: u64 = 0x8000;

/// Actual buffer allocation = cwsr_size + debug_memory, page-aligned.
/// Kernel validates the BO size matches this exactly (since 6.11).
const CTX_SAVE_ALLOC_SIZE: u64 = (CTX_SAVE_SIZE + DEBUG_MEM_SIZE + 0xFFF) & !0xFFF; // 0xAAC000

/// ctl_stack_size
const CTL_STACK_SIZE: u32 = 0x4000;

/// A PM4 compute queue on the GPU.
pub struct ComputeQueue {
    pub queue_id: u32,
    pub ring: GpuBuffer,
    pub rw_ptrs: GpuBuffer,
    doorbell_ptr: *mut u64,
    _doorbell_mmap: *mut libc::c_void,
    _doorbell_size: usize,
    _eop: GpuBuffer,
    _ctx_save: GpuBuffer,
    pub put: u64,
    kfd_fd: RawFd,
}

unsafe impl Send for ComputeQueue {}
unsafe impl Sync for ComputeQueue {}

impl ComputeQueue {
    /// Create a new PM4 compute queue.
    pub fn new(alloc: &GpuAllocator) -> std::io::Result<Self> {
        // Ring buffer (GTT+PUBLIC, flags 0xF6000002 matching tinygrad)
        let ring = alloc.alloc_gtt_public(RING_SIZE)
            .map_err(|e| { eprintln!("      ring alloc failed: {}", e); e })?;

        // Read/write pointer memory (GTT+PUBLIC, flags 0xF6000002)
        let rw_ptrs = alloc.alloc_gtt_public(0x100)
            .map_err(|e| { eprintln!("      rw_ptrs alloc failed: {}", e); e })?;

        // Zero the write/read pointers
        unsafe {
            ptr::write_bytes(rw_ptrs.cpu_ptr, 0, rw_ptrs.size as usize);
        }

        // EOP (VRAM private, flags 0xD0000001 matching tinygrad)
        let eop = alloc.alloc_vram_private(EOP_SIZE)
            .map_err(|e| { eprintln!("      eop alloc failed: {}", e); e })?;

        // ctx_save_restore: alloc CTX_SAVE_ALLOC_SIZE (cwsr + debug), ioctl field is CTX_SAVE_SIZE
        // VRAM private, flags 0xD0000001 matching tinygrad
        let ctx_save = alloc.alloc_vram_private(CTX_SAVE_ALLOC_SIZE)
            .map_err(|e| { eprintln!("      ctx_save alloc failed: {}", e); e })?;

        // Create queue via KFD
        let mut args = ioctl::CreateQueueArgs {
            ring_base_address: ring.va_addr,
            ring_size: RING_SIZE as u32,
            gpu_id: alloc.gpu_id,
            queue_type: ioctl::QUEUE_TYPE_COMPUTE,
            queue_percentage: ioctl::MAX_QUEUE_PERCENTAGE,
            queue_priority: 7,
            write_pointer_address: rw_ptrs.va_addr + 0x38,
            read_pointer_address: rw_ptrs.va_addr + 0x80,
            eop_buffer_address: eop.va_addr,
            eop_buffer_size: EOP_SIZE,
            ctx_save_restore_address: ctx_save.va_addr,
            ctx_save_restore_size: CTX_SAVE_SIZE as u32,
            ctl_stack_size: CTL_STACK_SIZE,
            ..Default::default()
        };
        ioctl::create_queue(alloc.kfd_fd, &mut args)?;

        // mmap doorbell
        let doorbell_offset = args.doorbell_offset;
        let doorbell_page = doorbell_offset & !0x1FFF;
        let doorbell_off_in_page = (doorbell_offset & 0x1FFF) as usize;
        let doorbell_size = 8192;
        let doorbell_mmap = unsafe {
            libc::mmap(
                ptr::null_mut(),
                doorbell_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                alloc.kfd_fd,
                doorbell_page as libc::off_t,
            )
        };
        if doorbell_mmap == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        let doorbell_ptr = unsafe {
            (doorbell_mmap as *mut u8).add(doorbell_off_in_page) as *mut u64
        };

        Ok(ComputeQueue {
            queue_id: args.queue_id,
            ring,
            rw_ptrs,
            doorbell_ptr,
            _doorbell_mmap: doorbell_mmap,
            _doorbell_size: doorbell_size,
            _eop: eop,
            _ctx_save: ctx_save,
            put: 0,
            kfd_fd: alloc.kfd_fd,
        })
    }

    /// Check ring buffer space. Panics if GPU hasn't consumed enough packets.
    fn check_ring_space(&self, dwords_needed: u64) {
        let ring_dwords = self.ring.size as u64 / 4;
        let read_ptr = unsafe {
            (self.rw_ptrs.cpu_ptr.add(0x80) as *const u64).read_volatile()
        };
        let used = self.put.wrapping_sub(read_ptr);
        assert!(used + dwords_needed < ring_dwords,
            "ring buffer overflow: put={} read={} need={} capacity={}",
            self.put, read_ptr, dwords_needed, ring_dwords);
    }

    /// Write a DWORD to the ring and advance.
    #[inline]
    fn q(&mut self, val: u32) {
        let ring_dwords = self.ring.size as u64 / 4;
        let offset = (self.put % ring_dwords) as usize;
        unsafe {
            let ptr = self.ring.cpu_ptr as *mut u32;
            ptr.add(offset).write_volatile(val);
        }
        self.put += 1;
    }

    /// Write a PM4 type 3 packet.
    fn pkt3(&mut self, opcode: u32, vals: &[u32]) {
        self.check_ring_space(1 + vals.len() as u64);
        self.q(pkt3(opcode, vals.len() as u32));
        for &v in vals {
            self.q(v);
        }
    }

    /// Write SET_SH_REG packet (set one or more shader registers).
    pub fn set_sh_reg(&mut self, offset: u32, values: &[u32]) {
        let mut data = vec![offset];
        data.extend_from_slice(values);
        self.pkt3(PACKET3_SET_SH_REG, &data);
    }

    /// Memory barrier before kernel dispatch.
    /// 7 data dwords on gfx11 (not 6!) — includes GCR_CNTL as 7th word.
    pub fn acquire_mem(&mut self) {
        self.pkt3(PACKET3_ACQUIRE_MEM, &[
            0,          // CP_COHER_CNTL
            0xFFFFFFFF, // COHER_SIZE lo
            0xFFFFFFFF, // COHER_SIZE hi
            0,          // COHER_BASE lo
            0,          // COHER_BASE hi
            0,          // POLL_INTERVAL
            0x000003F0, // GCR_CNTL (gfx11 requires this 7th dword)
        ]);
    }

    /// Write a completion signal + send interrupt to KFD.
    /// Must send TWO RELEASE_MEM packets (matching tinygrad):
    ///   1. Write signal value to signal_addr (with cache flush)
    ///   2. Write event_id to event mailbox (interrupt notification)
    /// Without the second packet, MES hangs on queue destruction.
    pub fn signal(&mut self, signal_addr: u64, signal_value: u32,
                  event_mailbox_ptr: u64, event_id: u32) {
        // Packet 1: write signal value WITH cache flush, NO interrupt
        self.pkt3(PACKET3_RELEASE_MEM, &[
            0x0070f514, // event_type=20, event_index=5, GCR cache flush
            0x20000000, // data_sel=1(32bit), int_sel=0(none)
            signal_addr as u32,
            (signal_addr >> 32) as u32,
            signal_value,
            0,
            0,
        ]);

        // Packet 2: cache flush + event mailbox + interrupt
        // by the time this executes, packet 1's signal write is done
        self.pkt3(PACKET3_RELEASE_MEM, &[
            0x0070f514, // event_type=20, event_index=5, GCR cache flush
            0x22000000, // data_sel=1(32bit), int_sel=2(interrupt after write)
            event_mailbox_ptr as u32,
            (event_mailbox_ptr >> 32) as u32,
            event_id,
            0,
            event_id, // ctxid = event_id
        ]);
    }

    /// Wait until a memory address contains >= expected value.
    pub fn wait_reg_mem(&mut self, addr: u64, expected: u32) {
        self.pkt3(PACKET3_WAIT_REG_MEM, &[
            (WAIT_REG_MEM_FUNCTION_GEQ) | (WAIT_REG_MEM_MEM_SPACE_MEM << 4),
            addr as u32,
            (addr >> 32) as u32,
            expected,
            0xFFFFFFFF, // mask
            4,          // poll interval
        ]);
    }

    /// Dispatch a compute kernel.
    ///
    /// `pgm_addr`: GPU virtual address of kernel code
    /// `rsrc1`, `rsrc2`, `rsrc3`: from kernel descriptor
    /// `kernargs_addr`: GPU virtual address of kernel arguments buffer
    /// `grid`: (global_x, global_y, global_z) in workitems
    /// `block`: (local_x, local_y, local_z) workgroup size
    pub fn dispatch(&mut self,
                    pgm_addr: u64, rsrc1: u32, rsrc2: u32, rsrc3: u32,
                    kernargs_addr: u64, scratch_addr: u64,
                    grid: [u32; 3], block: [u32; 3]) {
        self.dispatch_lds(pgm_addr, rsrc1, rsrc2, rsrc3, kernargs_addr, scratch_addr, grid, block, 0);
    }

    /// Dispatch with explicit LDS allocation (group_segment_fixed_size in bytes).
    pub fn dispatch_lds(&mut self,
                    pgm_addr: u64, rsrc1: u32, rsrc2: u32, rsrc3: u32,
                    kernargs_addr: u64, scratch_addr: u64,
                    grid: [u32; 3], block: [u32; 3],
                    lds_bytes: u32) {
        // Cache invalidate
        self.acquire_mem();

        // Program address (shifted right by 8 per AMD convention)
        self.set_sh_reg(COMPUTE_PGM_LO, &[
            (pgm_addr >> 8) as u32,
            (pgm_addr >> 40) as u32,
        ]);

        // Resource descriptors — set PRIV bit (1<<20) on rsrc1 for GFX11 (cwsr workaround)
        let rsrc1 = rsrc1 | (1 << 20);
        // Patch rsrc2 with LDS_SIZE (bits 15:23) in 128-dword (512-byte) granularity
        let lds_alloc = (lds_bytes + 511) / 512;
        let rsrc2 = (rsrc2 & !(0x1FF << 15)) | (lds_alloc << 15);
        self.set_sh_reg(COMPUTE_PGM_RSRC1, &[rsrc1, rsrc2]);
        self.set_sh_reg(COMPUTE_PGM_RSRC3, &[rsrc3]);

        // scratch ring: WAVESIZE=0 for kernels with no private segment
        // setting WAVESIZE>0 limits concurrent waves and causes hangs at 32+ WGs
        self.set_sh_reg(COMPUTE_TMPRING_SIZE, &[0x00000000]);

        // Scratch base address (required on gfx11 with has_scratch_base_registers)
        // Register 0x0210 = COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI
        self.set_sh_reg(0x0210, &[
            (scratch_addr >> 8) as u32,
            (scratch_addr >> 40) as u32,
        ]);

        // Restart counters
        self.set_sh_reg(COMPUTE_RESTART_X, &[0, 0, 0]);

        // Kernel arguments pointer
        self.set_sh_reg(COMPUTE_USER_DATA_0, &[
            kernargs_addr as u32,
            (kernargs_addr >> 32) as u32,
        ]);

        // Resource limits: 0 = no limit. The hardware manages per-CU resource
        // allocation and won't launch a WG unless all its waves can be allocated.
        // Barrier deadlocks only happen with manual limits that are too low.
        self.set_sh_reg(COMPUTE_RESOURCE_LIMITS, &[0]);

        // Start offsets + workgroup size (contiguous regs 0x204-0x20C)
        // Matches tinygrad: start_x/y/z, local_x/y/z, 0, 0
        self.set_sh_reg(COMPUTE_START_X, &[
            0, 0, 0,                       // start_x, start_y, start_z
            block[0], block[1], block[2],   // num_thread_x, num_thread_y, num_thread_z
            0, 0,                           // padding (perfcount_enable, etc.)
        ]);

        // DISPATCH_DIRECT: grid dimensions + initiator
        self.pkt3(PACKET3_DISPATCH_DIRECT, &[
            grid[0], grid[1], grid[2],
            COMPUTE_SHADER_EN | CS_W32_EN | FORCE_START_AT_000,
        ]);

        // CS_PARTIAL_FLUSH after dispatch (required, matches tinygrad)
        self.pkt3(PACKET3_EVENT_WRITE, &[0x0407]);
    }

    /// Submit all queued packets to the GPU by ringing the doorbell.
    pub fn submit(&mut self) {
        // Memory fence to ensure ring writes are visible
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

        // Update write pointer
        // Write pointer is at offset +0x38 in rw_ptrs buffer
        let wp = unsafe { self.rw_ptrs.cpu_ptr.add(0x38) as *mut u64 };
        unsafe { wp.write_volatile(self.put); }

        // Ring doorbell (u64 write, same as tinygrad)
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        unsafe { self.doorbell_ptr.write_volatile(self.put); }
    }

    /// Spin-wait until a 32-bit signal address contains the expected value.
    pub fn poll_signal(signal_ptr: *const u32, expected: u32, timeout_us: u64) -> bool {
        let start = std::time::Instant::now();
        loop {
            let val = unsafe { signal_ptr.read_volatile() };
            if val >= expected { return true; }
            if start.elapsed().as_micros() as u64 > timeout_us { return false; }
            std::hint::spin_loop();
        }
    }
}

impl Drop for ComputeQueue {
    fn drop(&mut self) {
        let _ = ioctl::destroy_queue(self.kfd_fd, self.queue_id);
        if !self._doorbell_mmap.is_null() {
            unsafe { libc::munmap(self._doorbell_mmap, self._doorbell_size); }
        }
    }
}
