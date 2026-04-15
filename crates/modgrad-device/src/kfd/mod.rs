//! Heterogeneous compute via KFD (Kernel Fusion Driver).
//!
//! KFD manages both CPU and GPU as HSA compute agents.
//! One driver, unified memory, automatic dispatch.
//!
//! Node 0: CPU (32 cores, AVX-512, DDR5)
//! Node 1: GPU (64 SIMDs, wave32, 8GB VRAM)
//!
//! Small ops → CPU (lower latency, AVX-512)
//! Large matmuls → GPU (higher throughput, 64 SIMDs)
//! Shared address space — zero-copy between CPU and GPU.
//!
//! Reference: tinygrad/tinygrad/runtime/ops_amd.py

pub mod ioctl;
pub mod memory;
pub mod queue;
pub mod dispatch;
pub mod compute;
pub mod accel;
pub mod stream;
pub mod arena;
pub mod dispatch_queue;
pub mod gguf;
pub mod inference;
pub mod quant_dot;

use memory::{GpuAllocator, GpuBuffer};
use queue::ComputeQueue;
use dispatch::{CodeObject, GpuProgram, KernArgs};
use std::os::unix::io::RawFd;

// ─── GPU kernel registry ───────────────────────────────────
//
// Kernels are pre-compiled code objects (.co) for specific GPU targets.
// To add a new GPU target (e.g. gfx1100, gfx1101):
//   1. Reassemble each .s file with: clang -mcpu=gfxNNNN ...
//   2. Add the .co files under kernels/gfxNNNN/
//   3. Add an entry to kernel_objects_for_target() below
//
// The ISA is identical across RDNA3 variants — only the ELF
// target triple changes. PM4 encoding is GFX11-wide.

/// Return compiled kernel code objects for the given gfx_target_version.
/// Falls back to gfx1102 for any unrecognized RDNA3 target.
fn kernel_objects_for_target(gfx_version: u32) -> Vec<&'static [u8]> {
    match gfx_version {
        // RDNA3: gfx1100 (Navi31), gfx1101 (Navi32), gfx1102 (Navi33)
        // sysfs gfx_target_version is DECIMAL: 110000, 110001, 110002
        // Currently only gfx1102 binaries are compiled.
        110000 | 110001 | 110002 => vec![
            include_bytes!("kernels/test_store.co"),
            include_bytes!("kernels/matvec_asm.co"),
            include_bytes!("kernels/matmul_blocked.co"),
            include_bytes!("kernels/matmul_small.co"),
            include_bytes!("kernels/superlinear.co"),
            include_bytes!("kernels/glu.co"),
            include_bytes!("kernels/silu.co"),
            include_bytes!("kernels/layer_norm.co"),
            include_bytes!("kernels/trace_shift.co"),
            include_bytes!("kernels/sync_update.co"),
            include_bytes!("kernels/matvec_tiled.co"),
            include_bytes!("kernels/ln_silu.co"),
            include_bytes!("kernels/matvec_q4k.co"),
        ],
        _ => {
            eprintln!("    warning: no pre-compiled kernels for gfx{:x}", gfx_version);
            vec![]
        }
    }
}

// ─── Node properties ────────────────────────────────────────

/// Properties of a KFD compute node (CPU or GPU).
#[derive(Debug, Default, Clone)]
pub struct NodeProperties {
    pub node_id: u32,
    pub gpu_id: u32,
    pub cpu_cores_count: u32,
    pub simd_count: u32,
    pub wave_front_size: u32,
    pub gfx_target_version: u32,
    pub max_waves_per_simd: u32,
    pub array_count: u32,
    pub cu_per_simd_array: u32,
    pub drm_render_minor: u32,
    pub cwsr_size: u32,
    pub max_engine_clk: u32,
    // Memory
    pub mem_size_bytes: u64,
    pub mem_width: u32,
    pub mem_clk_max: u32,
}

impl NodeProperties {
    pub fn is_cpu(&self) -> bool { self.cpu_cores_count > 0 }
    pub fn is_gpu(&self) -> bool { self.simd_count > 0 }

    /// Read properties from sysfs for a given node.
    fn read(node_id: u32) -> std::io::Result<Self> {
        let base = format!("/sys/devices/virtual/kfd/kfd/topology/nodes/{}", node_id);
        let props_path = format!("{}/properties", base);
        let gpu_id_path = format!("{}/gpu_id", base);
        let mem_path = format!("{}/mem_banks/0/properties", base);

        let gpu_id: u32 = std::fs::read_to_string(&gpu_id_path)
            .unwrap_or_default().trim().parse().unwrap_or(0);

        let props_text = std::fs::read_to_string(&props_path)?;
        let mut p = NodeProperties { node_id, gpu_id, ..Default::default() };

        for line in props_text.lines() {
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap_or("");
            let val: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            match key {
                "cpu_cores_count" => p.cpu_cores_count = val as u32,
                "simd_count" => p.simd_count = val as u32,
                "wave_front_size" => p.wave_front_size = val as u32,
                "gfx_target_version" => p.gfx_target_version = val as u32,
                "cwsr_size" => p.cwsr_size = val as u32,
                "max_waves_per_simd" => p.max_waves_per_simd = val as u32,
                "array_count" => p.array_count = val as u32,
                "cu_per_simd_array" => p.cu_per_simd_array = val as u32,
                "drm_render_minor" => p.drm_render_minor = val as u32,
                "max_engine_clk_ccompute" => p.max_engine_clk = val as u32,
                _ => {}
            }
        }

        // Read memory bank
        if let Ok(mem_text) = std::fs::read_to_string(&mem_path) {
            for line in mem_text.lines() {
                let mut parts = line.split_whitespace();
                let key = parts.next().unwrap_or("");
                let val: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                match key {
                    "size_in_bytes" => p.mem_size_bytes = val,
                    "width" => p.mem_width = val as u32,
                    "mem_clk_max" => p.mem_clk_max = val as u32,
                    _ => {}
                }
            }
        }

        Ok(p)
    }
}

// ─── Device specs & status ──────────────────────────────────

/// Static device specifications. Readable without opening the device
/// (from sysfs topology). Use `DeviceSpecs::probe()` for zero-cost discovery.
#[derive(Debug, Clone)]
pub struct DeviceSpecs {
    /// CPU node properties.
    pub cpu: NodeProperties,
    /// GPU node properties.
    pub gpu: NodeProperties,
    /// Total VRAM in bytes.
    pub vram_total: u64,
    /// VRAM bus width in bits (e.g. 256).
    pub vram_width: u32,
    /// VRAM max clock in MHz.
    pub vram_clock_mhz: u32,
    /// Peak VRAM bandwidth in GB/s (computed: width * clock * 2 / 8 / 1000).
    pub vram_bandwidth_gbps: f32,
    /// Total system RAM in bytes.
    pub ram_total: u64,
    /// GPU gfx target version (e.g. 0x110002 for gfx1102).
    pub gfx_version: u32,
    /// GPU compute units (simd_count * cu_per_simd_array, or simd_count if no CU info).
    pub compute_units: u32,
}

impl DeviceSpecs {
    /// Probe device specs from sysfs. Does NOT open /dev/kfd — safe to call
    /// even without permissions, just needs sysfs readable.
    pub fn probe() -> std::io::Result<Self> {
        let mut cpu = None;
        let mut gpu = None;
        for node in 0..16 {
            match NodeProperties::read(node) {
                Ok(p) if p.is_gpu() && gpu.is_none() => gpu = Some(p),
                Ok(p) if p.is_cpu() && cpu.is_none() => cpu = Some(p),
                _ => {}
            }
        }
        let cpu = cpu.ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound, "no CPU node in KFD topology"))?;
        let gpu = gpu.ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound, "no GPU node in KFD topology"))?;

        let vram_total = gpu.mem_size_bytes;
        let vram_width = gpu.mem_width;
        let vram_clock = gpu.mem_clk_max;
        // DDR effective: clock * 2 (double data rate)
        // Bandwidth = width_bits / 8 * effective_clock * 1e6 / 1e9
        let bw = (vram_width as f32 / 8.0) * (vram_clock as f32 * 2.0) / 1000.0;

        Ok(DeviceSpecs {
            vram_total,
            vram_width,
            vram_clock_mhz: vram_clock,
            vram_bandwidth_gbps: bw,
            ram_total: cpu.mem_size_bytes,
            gfx_version: gpu.gfx_target_version,
            compute_units: if gpu.cu_per_simd_array > 0 {
                gpu.simd_count * gpu.cu_per_simd_array
            } else {
                gpu.simd_count
            },
            cpu,
            gpu,
        })
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "GPU: gfx{:x}, {} CUs, {} GB VRAM (GDDR6-{} MHz, {}-bit, {:.0} GB/s)\n\
             CPU: {} cores, {} GB RAM",
            self.gfx_version,
            self.compute_units,
            self.vram_total / (1024 * 1024 * 1024),
            self.vram_clock_mhz,
            self.vram_width,
            self.vram_bandwidth_gbps,
            self.cpu.cpu_cores_count,
            self.ram_total / (1024 * 1024 * 1024),
        )
    }

    /// Can this model's weights fit in VRAM?
    /// `weight_bytes`: total weight size in bytes.
    /// Returns (fits, headroom_bytes).
    pub fn weights_fit(&self, weight_bytes: u64) -> (bool, i64) {
        let headroom = self.vram_total as i64 - weight_bytes as i64;
        (headroom > 0, headroom)
    }
}

/// Live device status. Requires an open KFD file descriptor.
#[derive(Debug, Clone)]
pub struct DeviceStatus {
    /// Currently available (free) VRAM in bytes.
    pub vram_available: u64,
    /// Total VRAM in bytes (from specs).
    pub vram_total: u64,
    /// VRAM used = total - available.
    pub vram_used: u64,
    /// Current shader clock in MHz (from sysfs, None if unreadable).
    pub sclk_mhz: Option<u32>,
}

impl DeviceStatus {
    /// Utilization ratio: 0.0 = empty, 1.0 = full.
    pub fn vram_utilization(&self) -> f32 {
        if self.vram_total == 0 { return 0.0; }
        self.vram_used as f32 / self.vram_total as f32
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let used_mb = self.vram_used / (1024 * 1024);
        let total_mb = self.vram_total / (1024 * 1024);
        let avail_mb = self.vram_available / (1024 * 1024);
        let clk = self.sclk_mhz.map(|m| format!("{} MHz", m))
            .unwrap_or_else(|| "unknown".into());
        format!(
            "VRAM: {} MB / {} MB ({} MB free, {:.0}% used), sclk: {}",
            used_mb, total_mb, avail_mb,
            self.vram_utilization() * 100.0,
            clk,
        )
    }
}

// ─── Unified HSA device ─────────────────────────────────────

/// Unified heterogeneous compute device via KFD.
///
/// Manages both CPU and GPU nodes. Allocates shared memory visible
/// to both. Dispatches compute to the appropriate agent based on
/// problem size.
pub struct HsaDevice {
    pub kfd_fd: RawFd,
    pub drm_fd: RawFd,
    pub cpu: NodeProperties,
    pub gpu: NodeProperties,
    pub alloc: GpuAllocator,
    /// GPU resources — these fields drop BEFORE _fd_guard (declaration order).
    _code_objects: Vec<CodeObject>,
    pub kernels: Option<std::collections::HashMap<String, dispatch::KernelEntry>>,
    pub queue: ComputeQueue,
    pub signal: GpuBuffer,
    pub event_page: GpuBuffer,
    pub scratch: GpuBuffer,
    signal_value: u32,
    pub vram_available: u64,
    pub queue_event_id: u32,
    pub queue_event_mailbox_ptr: u64,
    pub gpu_dispatch_threshold: usize,
    /// MUST be last field: closes kfd_fd/drm_fd after all GpuBuffers are dropped.
    _fd_guard: FdGuard,
}

impl HsaDevice {
    /// Open the HSA device. Discovers CPU + GPU from KFD topology.
    pub fn open() -> std::io::Result<Self> {
        let kfd_fd = unsafe { libc::open(b"/dev/kfd\0".as_ptr() as _, libc::O_RDWR) };
        if kfd_fd < 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                "/dev/kfd not available"));
        }

        let (major, minor) = ioctl::get_version(kfd_fd)?;

        // Discover all nodes
        let mut cpu = None;
        let mut gpu = None;
        for node in 0..16 {
            match NodeProperties::read(node) {
                Ok(p) if p.is_gpu() && gpu.is_none() => gpu = Some(p),
                Ok(p) if p.is_cpu() && cpu.is_none() => cpu = Some(p),
                _ => {}
            }
        }

        let cpu = cpu.ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound, "no CPU node in KFD topology"))?;
        let gpu = gpu.ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound, "no GPU node in KFD topology"))?;

        #[cfg(debug_assertions)]
        eprintln!("  KFD {}.{} — HSA topology:", major, minor);
        #[cfg(debug_assertions)]
        eprintln!("    CPU: {} cores, {} MHz, {} GB RAM (DDR{}-{}, {}-bit)",
            cpu.cpu_cores_count, cpu.max_engine_clk,
            cpu.mem_size_bytes / (1024 * 1024 * 1024),
            if cpu.mem_clk_max > 2400 { "5" } else { "4" },
            cpu.mem_clk_max, cpu.mem_width);
        #[cfg(debug_assertions)]
        eprintln!("    GPU: gfx{:x}, {} SIMDs, wave{}, {} GB VRAM (GDDR6-{}, {}-bit)",
            gpu.gfx_target_version, gpu.simd_count, gpu.wave_front_size,
            gpu.mem_size_bytes / (1024 * 1024 * 1024),
            gpu.mem_clk_max, gpu.mem_width);

        // Open DRM render node for GPU
        let drm_path = format!("/dev/dri/renderD{}\0", gpu.drm_render_minor);
        let drm_fd = unsafe { libc::open(drm_path.as_ptr() as _, libc::O_RDWR) };
        if drm_fd < 0 {
            unsafe { libc::close(kfd_fd); }
            return Err(std::io::Error::last_os_error());
        }

        // Acquire VM for GPU
        ioctl::acquire_vm(kfd_fd, drm_fd, gpu.gpu_id)
            .map_err(|e| { eprintln!("    acquire_vm failed: {}", e); e })?;

        // Runtime enable (required before queue creation on KFD >= 1.14)
        if major > 1 || (major == 1 && minor >= 14) {
            ioctl::runtime_enable(kfd_fd)
                .map_err(|e| { eprintln!("    runtime_enable failed: {}", e); e })?;
        }

        let vram_available = ioctl::available_memory(kfd_fd, gpu.gpu_id).unwrap_or(0);

        let alloc = GpuAllocator {
            kfd_fd,
            drm_fd,
            gpu_id: gpu.gpu_id,
        };

        // Allocate event page (GTT, 0x8000 bytes) — must exist before CREATE_EVENT
        let event_page = alloc.alloc_gtt(0x8000)?;

        // Create events before queue (KFD requirement)
        // First event with event_page_offset=handle allocates the event page
        let _first_event = ioctl::create_event(kfd_fd, ioctl::EVENT_SIGNAL, 0,
            event_page.handle)?;
        // Queue event (auto_reset=1) — used for interrupt signaling
        let queue_event = ioctl::create_event(kfd_fd, ioctl::EVENT_SIGNAL, 1, 0)?;
        let _mem_event = ioctl::create_event(kfd_fd, ioctl::EVENT_MEMORY, 0, 0)?;
        let _hw_event = ioctl::create_event(kfd_fd, ioctl::EVENT_HW_EXCEPTION, 0, 0)?;

        // Compute mailbox pointer: event_page_va + event_slot_index * 8
        let queue_event_mailbox_ptr = event_page.va_addr
            + queue_event.event_slot_index as u64 * 8;

        let queue = ComputeQueue::new(&alloc)
            .map_err(|e| { eprintln!("    queue creation failed: {}", e); e })?;
        // Scratch buffer for compute dispatch (2MB, VRAM)
        let scratch = alloc.alloc_vram(2 * 1024 * 1024)?;

        // Signal buffer: GPU writes via WRITE_DATA, CPU reads via volatile.
        // GTT+COHERENT+UNCACHED — system memory, immediately visible to CPU.
        // WRITE_DATA (not RELEASE_MEM) for the data write, RELEASE_MEM for interrupt only.
        let signal = alloc.alloc_gtt(4096)?;
        signal.write(0, &[0u8; 8]);

        // Load compiled GPU kernels for this target.
        // Each code object owns its GpuBuffer — store them all to keep code in VRAM.
        let mut code_objects: Vec<CodeObject> = Vec::new();
        for co_bytes in kernel_objects_for_target(gpu.gfx_target_version) {
            match CodeObject::load(&alloc, co_bytes) {
                Ok(co) => code_objects.push(co),
                Err(e) => eprintln!("    warning: kernel load failed: {}", e),
            }
        }
        // merge kernel entries (they reference code_addr which stays valid while code_objects lives)
        let mut kernel_map = std::collections::HashMap::new();
        for co in &code_objects {
            for (name, entry) in &co.kernels {
                kernel_map.insert(name.clone(), entry.clone());
            }
        }
        let kernels = if kernel_map.is_empty() { None } else {
            #[cfg(debug_assertions)]
            eprintln!("    kernels: {:?}", kernel_map.keys().collect::<Vec<_>>());
            Some(kernel_map)
        };

        // GPU dispatch threshold: ~64K FLOPs
        // Below this, CPU AVX-512 is faster due to dispatch overhead
        let gpu_dispatch_threshold = 64 * 1024;

        // lock GPU clocks to max for consistent performance
        let sysfs_device = format!("/sys/class/drm/renderD{}/device", gpu.drm_render_minor);
        let clocks_locked = FdGuard::lock_clocks(&sysfs_device);
        #[cfg(debug_assertions)]
        if clocks_locked {
            eprintln!("    gpu clocks: locked to max");
        } else {
            eprintln!("    gpu clocks: auto (no sysfs write access)");
        }

        let mut dev = HsaDevice {
            kfd_fd,
            drm_fd,
            cpu,
            gpu,
            alloc,
            _code_objects: code_objects,
            kernels,
            queue,
            signal,
            event_page,
            scratch,
            signal_value: 0,
            vram_available,
            queue_event_id: queue_event.event_id,
            queue_event_mailbox_ptr,
            gpu_dispatch_threshold,
            _fd_guard: FdGuard {
                kfd_fd, drm_fd,
                sysfs_device: if clocks_locked { Some(sysfs_device) } else { None },
            },
        };

        // Smoke-test: verify signal + multi-WG dispatch
        dev.signal_value = 0;
        unsafe { (dev.signal.cpu_ptr as *mut u32).write_volatile(0); }
        dev.signal_value += 1;
        dev.queue.signal(dev.signal.va_addr, dev.signal_value,
            dev.queue_event_mailbox_ptr, dev.queue_event_id);
        dev.queue.submit();
        if !dev.wait_gpu(1000) {
            eprintln!("    GPU signal: FAIL");
        }

        // Multi-WG test: dispatch matvec with increasing WG count
        for nwg in [1u32, 4, 16, 33] {
            let m = nwg * 256;
            let k = 64u32;
            let w_data = vec![0.01f32; m as usize * k as usize];
            let b_data = vec![0.0f32; m as usize];
            let x_data = vec![1.0f32; k as usize];
            let w_buf = match dev.upload_f32(&w_data) { Ok(b) => b, Err(_) => continue };
            let b_buf = match dev.upload_f32(&b_data) { Ok(b) => b, Err(_) => continue };
            let x_buf = match dev.upload_f32(&x_data) { Ok(b) => b, Err(_) => continue };
            let y_buf = match dev.alloc_output(m as usize * 4 + 64) { Ok(b) => b, Err(_) => continue };

            let mut args = KernArgs::new();
            args.push_ptr(&w_buf); args.push_ptr(&b_buf);
            args.push_ptr(&x_buf); args.push_ptr(&y_buf);
            args.push_u32(m); args.push_u32(k);
            let args_buf = match args.upload(&dev.alloc) { Ok(b) => b, Err(_) => continue };

            dev.dispatch_enqueue("matvec", &args_buf, [nwg, 1, 1], [256, 1, 1]);
            let ok = dev.submit_wait(2000);
            let y0 = if ok { unsafe { (y_buf.cpu_ptr as *const f32).read_volatile() } } else { f32::NAN };
            eprintln!("    matvec {}WG ({}x{}): {} y0={:.4}",
                nwg, m, k, if ok { "OK" } else { "HANG" }, y0);
            if !ok { break; }
        }

        Ok(dev)
    }

    /// Allocate a VRAM buffer and upload f32 data.
    pub fn upload_f32(&self, data: &[f32]) -> std::io::Result<GpuBuffer> {
        // +4096: extra page for RDNA3 global_load_b128 cache line prefetch
        // at allocation boundary (kernel prefetch reads 1 tile past last load)
        let size = ((data.len() * 4 + 4096 + 4095) & !4095) as u64;
        let buf = self.alloc.alloc_vram(size)?;
        buf.write_f32(0, data);
        Ok(buf)
    }

    /// Upload a row-major matrix as column-major (transposed) in VRAM.
    /// Input: data[rows][cols] row-major. Output: buf[cols][rows] in VRAM.
    /// This gives coalesced access when threads read consecutive row indices
    /// from the same column — critical for RDNA3's narrow 128-bit memory bus.
    pub fn upload_f32_col_major(&self, data: &[f32], rows: usize, cols: usize)
        -> std::io::Result<GpuBuffer>
    {
        assert_eq!(data.len(), rows * cols);
        let size = ((rows * cols * 4 + 4096 + 4095) & !4095) as u64;
        let buf = self.alloc.alloc_vram(size)?;
        let dst = unsafe {
            std::slice::from_raw_parts_mut(buf.cpu_ptr as *mut f32, rows * cols)
        };
        for r in 0..rows {
            for c in 0..cols {
                dst[c * rows + r] = data[r * cols + c];
            }
        }
        Ok(buf)
    }

    /// Allocate a VRAM buffer of given size (bytes), zeroed.
    pub fn alloc_output(&self, size_bytes: usize) -> std::io::Result<GpuBuffer> {
        let size = ((size_bytes + 4095) & !4095) as u64;
        let buf = self.alloc.alloc_vram(size)?;
        unsafe { std::ptr::write_bytes(buf.cpu_ptr, 0, size as usize); }
        Ok(buf)
    }

    /// dispatch a kernel and return a future (non-blocking).
    pub fn dispatch_async(&mut self, name: &str,
                          kernargs: &GpuBuffer,
                          grid: [u32; 3], block: [u32; 3]) -> Option<compute::GpuFuture> {
        let entry = match self.kernels.as_ref().and_then(|m| m.get(name)) {
            Some(e) => e.clone(),
            None => return None,
        };
        self.signal_value += 1;
        self.queue.dispatch_lds(
            entry.code_addr,
            entry.desc.pgm_rsrc1, entry.desc.pgm_rsrc2, entry.desc.pgm_rsrc3,
            kernargs.va_addr, self.scratch.va_addr,
            grid, block,
            entry.desc.group_segment_size,
        );
        self.queue.cache_wb();
        self.queue.signal(self.signal.va_addr, self.signal_value,
            self.queue_event_mailbox_ptr, self.queue_event_id);
        self.queue.submit();
        Some(compute::GpuFuture::new(&self.signal, self.signal_value))
    }

    /// Resolve a kernel by name. Cache the result for fast dispatch.
    pub fn resolve_kernel(&self, name: &str) -> Option<dispatch::KernelEntry> {
        self.kernels.as_ref()?.get(name).cloned()
    }

    /// Queue a dispatch with pre-resolved kernel. No HashMap lookup.
    pub fn dispatch_enqueue_resolved(&mut self,
                                      entry: &dispatch::KernelEntry,
                                      kernargs: &memory::GpuBuffer,
                                      grid: [u32; 3], block: [u32; 3]) {
        self.queue.dispatch_lds(
            entry.code_addr,
            entry.desc.pgm_rsrc1, entry.desc.pgm_rsrc2, entry.desc.pgm_rsrc3,
            kernargs.va_addr, self.scratch.va_addr,
            grid, block,
            entry.desc.group_segment_size,
        );
    }

    /// Queue a dispatch with raw kernargs VA.
    /// Same as dispatch_enqueue but takes a u64 VA instead of &GpuBuffer.
    /// Used by GpuQueue which manages its own kernarg slab.
    pub fn dispatch_enqueue_va(&mut self, name: &str,
                               kernargs_va: u64,
                               grid: [u32; 3], block: [u32; 3]) -> bool {
        let entry = match self.kernels.as_ref().and_then(|m| m.get(name)) {
            Some(e) => e.clone(),
            None => return false,
        };
        self.queue.dispatch_lds(
            entry.code_addr,
            entry.desc.pgm_rsrc1, entry.desc.pgm_rsrc2, entry.desc.pgm_rsrc3,
            kernargs_va, self.scratch.va_addr,
            grid, block,
            entry.desc.group_segment_size,
        );
        true
    }

    /// Queue a dispatch without signaling or submitting.
    /// Call `submit_wait()` after all dispatches are queued.
    pub fn dispatch_enqueue(&mut self, name: &str,
                            kernargs: &GpuBuffer,
                            grid: [u32; 3], block: [u32; 3]) -> bool {
        let entry = match self.kernels.as_ref().and_then(|m| m.get(name)) {
            Some(e) => e.clone(),
            None => return false,
        };
        self.queue.dispatch_lds(
            entry.code_addr,
            entry.desc.pgm_rsrc1, entry.desc.pgm_rsrc2, entry.desc.pgm_rsrc3,
            kernargs.va_addr, self.scratch.va_addr,
            grid, block,
            entry.desc.group_segment_size,
        );
        // CS_PARTIAL_FLUSH between dispatches (same as after dispatch in queue.rs)
        true
    }

    /// Signal + submit + wait.
    /// Call after dispatch_enqueue batch. Blocks until all dispatches complete.
    /// NOTE: cache_wb removed — it caused MES stalls on GFX11 after many
    /// sequential dispatches (same class of bug as RELEASE_MEM descheduling).
    /// The matvec kernel uses flat_store which is coherent through VMMU.
    pub fn submit_wait(&mut self, timeout_ms: u32) -> bool {
        self.signal_value += 1;
        self.queue.signal(self.signal.va_addr, self.signal_value,
            self.queue_event_mailbox_ptr, self.queue_event_id);
        self.queue.submit();
        self.wait_gpu(timeout_ms)
    }

    /// Submit all queued packets and return a future for the last signal.
    pub fn submit(&mut self) -> compute::GpuFuture {
        self.signal_value += 1;
        self.queue.cache_wb();
        self.queue.signal(self.signal.va_addr, self.signal_value,
            self.queue_event_mailbox_ptr, self.queue_event_id);
        self.queue.submit();
        compute::GpuFuture::new(&self.signal, self.signal_value)
    }

    /// Wait for the last signal via spin-polling.
    ///
    /// WRITE_DATA sets the signal value in GTT memory (coherent+uncached).
    /// We spin-poll with backoff until the value appears. This avoids
    /// RELEASE_MEM which causes MES queue descheduling on GFX11.
    pub fn wait_gpu(&mut self, timeout_ms: u32) -> bool {
        let sig_ptr = self.signal.cpu_ptr as *const u32;
        ComputeQueue::poll_signal(sig_ptr, self.signal_value,
            timeout_ms as u64 * 1000)
    }

    /// dispatch a kernel by name and wait for completion (blocking).
    pub fn dispatch_kernel(&mut self, name: &str,
                           kernargs: &GpuBuffer,
                           grid: [u32; 3], block: [u32; 3]) -> bool {
        let entry = match self.kernels.as_ref().and_then(|m| m.get(name)) {
            Some(e) => e.clone(),
            None => return false,
        };
        self.signal_value += 1;

        // Record put position before dispatch for ring dump
        let put_before = self.queue.put;

        self.queue.dispatch_lds(
            entry.code_addr,
            entry.desc.pgm_rsrc1,
            entry.desc.pgm_rsrc2,
            entry.desc.pgm_rsrc3,
            kernargs.va_addr,
            self.scratch.va_addr,
            grid, block,
            entry.desc.group_segment_size,
        );
        self.queue.cache_wb();
        self.queue.signal(self.signal.va_addr, self.signal_value,
            self.queue_event_mailbox_ptr, self.queue_event_id);

        // Dump ring contents before submit (debug)
        #[cfg(debug_assertions)]
        {
            let ring = self.queue.ring.as_slice::<u32>();
            let n = (self.queue.put - put_before) as usize;
            eprintln!("  dispatch PM4 ({} DWORDs, code=0x{:x} kargs=0x{:x}):",
                n, entry.code_addr, kernargs.va_addr);
            for i in 0..n {
                let idx = (put_before as usize + i) % (self.queue.ring.size as usize / 4);
                eprintln!("    [{:3}] 0x{:08x}", i, ring[idx]);
            }
        }

        self.queue.submit();
        self.wait_gpu(10_000)
    }

    /// Enqueue Y = X * W^T + B, selecting the optimal kernel for the shape.
    /// W is row-major [M×K], X is row-major [N×K], Y is [N×M].
    /// Uses matmul_blocked (TM=128) for large M, matmul_small (TM=32) for small M.
    ///
    /// SAFETY: M, K, N must be padded to tile boundaries BEFORE calling this.
    ///   - M must be a multiple of 128 (matmul_blocked) or 32 (matmul_small)
    ///   - K must be a multiple of 8
    ///   - N must be a multiple of 32
    ///   - All buffers (W, B, X, Y) must be allocated for the padded dimensions.
    /// Violating this will cause GPU OOB access → device hang → Xorg crash.
    /// Use `dispatch_matmul_safe` for automatic padding.
    pub fn dispatch_matmul_enqueue(&mut self,
                                    w_row: &GpuBuffer, w_col: &GpuBuffer,
                                    b: &GpuBuffer, x: &GpuBuffer, y: &GpuBuffer,
                                    m: u32, k: u32, n: u32) -> bool {
        // Guard: reject dimensions that would cause OOB
        if k % 8 != 0 || n % 32 != 0 {
            eprintln!("dispatch_matmul_enqueue: REJECTING unaligned dims M={} K={} N={}", m, k, n);
            return false;
        }
        if m >= 1536 && m % 128 != 0 {
            eprintln!("dispatch_matmul_enqueue: REJECTING M={} (not multiple of 128 for blocked)", m);
            return false;
        }
        if m < 1536 && m % 32 != 0 {
            eprintln!("dispatch_matmul_enqueue: REJECTING M={} (not multiple of 32 for small)", m);
            return false;
        }

        let (kernel, nwg) = if m >= 1536 {
            ("matmul_blocked", ((m + 127) / 128) * ((n + 31) / 32))
        } else {
            ("matmul_small", ((m + 31) / 32) * ((n + 31) / 32))
        };

        let w = if m >= 1536 { w_row } else { w_col };

        let mut args = KernArgs::new();
        args.push_ptr(w); args.push_ptr(b);
        args.push_ptr(x); args.push_ptr(y);
        args.push_u32(m); args.push_u32(k); args.push_u32(n);
        let args_buf = match args.upload(&self.alloc) {
            Ok(buf) => buf,
            Err(_) => return false,
        };

        self.dispatch_enqueue(kernel, &args_buf, [nwg, 1, 1], [256, 1, 1])
    }

    /// Enqueue a SuperLinear forward: Y[n*O+o] = B[n*O+o] + sum_k W[..]*X[..].
    /// N=n_neurons, O=out_per, K=in_per. All buffers sized for N*O*K etc.
    /// Returns false if kernel not loaded.
    pub fn dispatch_superlinear_enqueue(&mut self,
                                         w: &GpuBuffer, b: &GpuBuffer,
                                         x: &GpuBuffer, y: &GpuBuffer,
                                         n_neurons: u32, out_per: u32, in_per: u32) -> bool {
        let total = n_neurons * out_per;
        let nwg = (total + 255) / 256;

        let mut args = KernArgs::new();
        args.push_ptr(w); args.push_ptr(b);
        args.push_ptr(x); args.push_ptr(y);
        args.push_u32(n_neurons); args.push_u32(out_per); args.push_u32(in_per);
        let args_buf = match args.upload(&self.alloc) {
            Ok(buf) => buf,
            Err(_) => return false,
        };

        self.dispatch_enqueue("superlinear_fwd", &args_buf, [nwg, 1, 1], [256, 1, 1])
    }

    /// Dispatch a GpuProgram directly and wait for completion.
    pub fn dispatch_sync(&mut self, program: &GpuProgram,
                         kernargs: &GpuBuffer,
                         grid: [u32; 3], block: [u32; 3]) {
        self.signal_value += 1;
        program.dispatch(&mut self.queue, kernargs, self.scratch.va_addr,
                        grid, block, &self.signal, self.signal_value,
                        self.queue_event_mailbox_ptr, self.queue_event_id);
        if !self.wait_gpu(10_000) {
            eprintln!("WARNING: GPU dispatch timed out after 10s");
        }
    }

    /// Load a code object into GPU memory.
    pub fn load_program(&self, code_object: &[u8]) -> std::io::Result<GpuProgram> {
        GpuProgram::load(&self.alloc, code_object)
    }

    /// read current GPU shader clock in mhz from sysfs.
    pub fn current_sclk_mhz(&self) -> Option<u32> {
        let path = format!("/sys/class/drm/renderD{}/device/pp_dpm_sclk",
            self.gpu.drm_render_minor);
        let text = std::fs::read_to_string(&path).ok()?;
        // active level marked with *
        text.lines()
            .find(|l| l.contains('*'))
            .and_then(|l| {
                // "2: 2330Mhz *" → extract 2330
                l.split(':').nth(1)
                    .and_then(|s| s.trim().split('M').next())
                    .and_then(|s| s.trim().parse().ok())
            })
    }

    /// Should this operation go to GPU? Based on estimated FLOPs.
    pub fn should_use_gpu(&self, flops: usize) -> bool {
        flops >= self.gpu_dispatch_threshold
    }

    /// Get static device specifications (from the already-read topology).
    pub fn specs(&self) -> DeviceSpecs {
        let vram_total = self.gpu.mem_size_bytes;
        let vram_width = self.gpu.mem_width;
        let vram_clock = self.gpu.mem_clk_max;
        let bw = (vram_width as f32 / 8.0) * (vram_clock as f32 * 2.0) / 1000.0;

        DeviceSpecs {
            vram_total,
            vram_width,
            vram_clock_mhz: vram_clock,
            vram_bandwidth_gbps: bw,
            ram_total: self.cpu.mem_size_bytes,
            gfx_version: self.gpu.gfx_target_version,
            compute_units: if self.gpu.cu_per_simd_array > 0 {
                self.gpu.simd_count * self.gpu.cu_per_simd_array
            } else {
                self.gpu.simd_count
            },
            cpu: self.cpu.clone(),
            gpu: self.gpu.clone(),
        }
    }

    /// Query live device status (VRAM usage, clock speed).
    pub fn status(&self) -> DeviceStatus {
        let vram_available = ioctl::available_memory(self.kfd_fd, self.gpu.gpu_id)
            .unwrap_or(self.vram_available);
        let vram_total = self.gpu.mem_size_bytes;

        DeviceStatus {
            vram_available,
            vram_total,
            vram_used: vram_total.saturating_sub(vram_available),
            sclk_mhz: self.current_sclk_mhz(),
        }
    }
}

/// Guard that closes fds on drop. Placed as LAST field in HsaDevice
/// so it drops after all GpuBuffers (which need the fds for cleanup).
struct FdGuard {
    kfd_fd: RawFd,
    drm_fd: RawFd,
    /// sysfs device path for clock control (e.g. /sys/class/drm/renderD128/device)
    sysfs_device: Option<String>,
}

impl FdGuard {
    /// Lock GPU to peak performance (max clocks, no DVFS).
    fn lock_clocks(sysfs_device: &str) -> bool {
        let perf_path = format!("{}/power_dpm_force_performance_level", sysfs_device);
        // profile_peak forces max clocks immediately — no DVFS ramp needed
        std::fs::write(&perf_path, "profile_peak").is_ok()
    }

    fn unlock_clocks(sysfs_device: &str) {
        let perf_path = format!("{}/power_dpm_force_performance_level", sysfs_device);
        let _ = std::fs::write(&perf_path, "auto");
    }
}

impl Drop for FdGuard {
    fn drop(&mut self) {
        if let Some(ref path) = self.sysfs_device {
            FdGuard::unlock_clocks(path);
        }
        unsafe {
            libc::close(self.drm_fd);
            libc::close(self.kfd_fd);
        }
    }
}

/// Check if KFD is available on this system.
pub fn is_available() -> bool {
    std::path::Path::new("/dev/kfd").exists()
}
