//! Device mesh: multi-GPU abstraction for region-parallel CTM.
//!
//! The brain has 6 regions that compute independently per tick.
//! Each region can live on a different GPU. Communication between
//! regions is just activation vectors (~kilobytes per tick).
//!
//! Three parallelism axes:
//!   Region:  6 regions → up to 6 devices (or grouped)
//!   Tensor:  large weight matrices split across devices
//!   Data:    different sequences on different devices
//!
//! For isis, region parallelism is the natural fit:
//!   communication << computation at 512+ neurons per region.
//!
//! Design follows autoparallel-style device mesh:
//!   - DeviceId identifies a compute backend (CPU, GPU0, GPU1, ...)
//!   - DeviceMesh maps region names → device assignments
//!   - Each device owns its own command queue and buffers
//!   - Cross-device transfers are explicit (not hidden behind magic)

use std::collections::HashMap;

// ─── Device Abstraction ────────────────────────────────────

/// Identifies a compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceId {
    /// CPU (rayon thread pool).
    Cpu,
    /// GPU by index (0, 1, ...).
    Gpu(usize),
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceId::Cpu => write!(f, "cpu"),
            DeviceId::Gpu(i) => write!(f, "gpu:{i}"),
        }
    }
}

/// Information about a discovered device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub id: DeviceId,
    pub name: String,
    /// VRAM in bytes (0 for CPU).
    pub memory_bytes: u64,
    /// Estimated TFLOPS (single precision).
    pub tflops: f32,
}

/// Discover all available compute devices.
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    #[allow(unused_mut)]
    let mut devices = vec![DeviceInfo {
        id: DeviceId::Cpu,
        name: format!("CPU ({} threads)", rayon::current_num_threads()),
        memory_bytes: 0,
        tflops: 0.0, // unknown for CPU
    }];

    // Prefer CUDA (NVIDIA) if available, fall back to Vulkan (AMD/Intel/portable)
    #[cfg(feature = "cuda")]
    {
        if let Some(cuda_devices) = enumerate_cuda() {
            devices.extend(cuda_devices);
        }
    }

    // Only enumerate Vulkan if no CUDA devices found
    #[cfg(feature = "gpu")]
    {
        let has_gpu = devices.iter().any(|d| matches!(d.id, DeviceId::Gpu(_)));
        if !has_gpu {
            if let Some(gpu_devices) = enumerate_gpus() {
                devices.extend(gpu_devices);
            }
        }
    }

    devices
}

#[cfg(feature = "cuda")]
fn enumerate_cuda() -> Option<Vec<DeviceInfo>> {
    use cudarc::driver::CudaContext;

    // Gate on /dev/nvidia0 BEFORE touching cudarc. Its OnceLock-init panic
    // path doesn't reliably round-trip through catch_unwind in release
    // builds; better to never call into it on a non-NVIDIA host.
    if !std::path::Path::new("/dev/nvidia0").exists() { return None; }

    let n = std::panic::catch_unwind(|| CudaContext::device_count())
        .ok()
        .and_then(|r| r.ok())? as usize;
    if n == 0 { return None; }

    let mut infos = Vec::new();
    for i in 0..n {
        let ctx = match std::panic::catch_unwind(|| CudaContext::new(i)) {
            Ok(Ok(c)) => c,
            _ => continue,
        };
        let name = ctx.name().unwrap_or_else(|_| format!("CUDA device {i}"));
        let mem = ctx.total_mem().unwrap_or(0) as u64;
        infos.push(DeviceInfo {
            id: DeviceId::Gpu(i),
            name,
            memory_bytes: mem,
            tflops: 0.0,
        });
    }
    if infos.is_empty() { None } else { Some(infos) }
}

#[cfg(feature = "gpu")]
fn enumerate_gpus() -> Option<Vec<DeviceInfo>> {
    use ash::vk;
    use std::ffi::CStr;

    let entry = unsafe { ash::Entry::load().ok()? };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"isis-enumerate")
        .api_version(vk::make_api_version(0, 1, 2, 0));
    let inst_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&inst_info, None).ok()? };
    let phys_devices = unsafe { instance.enumerate_physical_devices().ok()? };

    let mut infos = Vec::new();
    for (i, &pd) in phys_devices.iter().enumerate() {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .to_string();
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };

        // Sum device-local heap sizes for VRAM estimate
        let mut vram = 0u64;
        for j in 0..mem_props.memory_heap_count as usize {
            let heap = mem_props.memory_heaps[j];
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                vram += heap.size;
            }
        }

        infos.push(DeviceInfo {
            id: DeviceId::Gpu(i),
            name,
            memory_bytes: vram,
            tflops: 0.0, // could estimate from device limits
        });
    }

    unsafe { instance.destroy_instance(None) };
    Some(infos)
}

// ─── Device Mesh ───────────────────────────────────────────

/// Maps brain regions to devices.
///
/// Default: everything on CPU.
/// With 1 GPU: all regions on GPU0, fallback to CPU for overflow.
/// With 2+ GPUs: cortical on GPU0, subcortical on GPU1, etc.
#[derive(Debug, Clone)]
pub struct DeviceMesh {
    /// Region name → assigned device.
    assignments: HashMap<String, DeviceId>,
    /// Available devices (discovered at startup).
    pub devices: Vec<DeviceInfo>,
    /// Default device for unassigned regions.
    pub default_device: DeviceId,
}

/// Standard brain region names.
pub const REGIONS: &[&str] = &[
    "input", "attention", "output", "motor",
    "cerebellum", "basal_ganglia",
    "insula", "hippocampus",
];

impl DeviceMesh {
    /// Create a mesh from discovered devices with automatic assignment.
    pub fn auto() -> Self {
        let devices = enumerate_devices();
        let n_gpus = devices.iter().filter(|d| matches!(d.id, DeviceId::Gpu(_))).count();

        let mut assignments = HashMap::new();
        let default_device;

        match n_gpus {
            0 => {
                // CPU only — all regions on CPU
                default_device = DeviceId::Cpu;
            }
            1 => {
                // Single GPU — everything on GPU0
                default_device = DeviceId::Gpu(0);
            }
            2 => {
                // 2 GPUs — cortical on GPU0, subcortical + motor on GPU1
                assignments.insert("input".into(), DeviceId::Gpu(0));
                assignments.insert("attention".into(), DeviceId::Gpu(0));
                assignments.insert("output".into(), DeviceId::Gpu(0));
                assignments.insert("motor".into(), DeviceId::Gpu(1));
                assignments.insert("cerebellum".into(), DeviceId::Gpu(1));
                assignments.insert("basal_ganglia".into(), DeviceId::Gpu(1));
                default_device = DeviceId::Gpu(0);
            }
            _ => {
                // 3+ GPUs — round-robin assignment
                let gpu_ids: Vec<DeviceId> = (0..n_gpus).map(DeviceId::Gpu).collect();
                for (i, &region) in REGIONS.iter().enumerate() {
                    assignments.insert(region.into(), gpu_ids[i % n_gpus]);
                }
                default_device = DeviceId::Gpu(0);
            }
        }

        Self { assignments, devices, default_device }
    }

    /// CPU-only mesh.
    pub fn cpu_only() -> Self {
        Self {
            assignments: HashMap::new(),
            devices: vec![DeviceInfo {
                id: DeviceId::Cpu,
                name: "CPU".into(),
                memory_bytes: 0,
                tflops: 0.0,
            }],
            default_device: DeviceId::Cpu,
        }
    }

    /// Manually assign a region to a device.
    pub fn assign(&mut self, region: &str, device: DeviceId) {
        self.assignments.insert(region.into(), device);
    }

    /// Get device for a region.
    pub fn device_for(&self, region: &str) -> DeviceId {
        self.assignments.get(region).copied().unwrap_or(self.default_device)
    }

    /// All unique devices in use.
    pub fn active_devices(&self) -> Vec<DeviceId> {
        let mut seen: Vec<DeviceId> = self.assignments.values().copied().collect();
        if !seen.contains(&self.default_device) {
            seen.push(self.default_device);
        }
        seen.sort_by_key(|d| match d { DeviceId::Cpu => 0, DeviceId::Gpu(i) => i + 1 });
        seen.dedup();
        seen
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let n_gpus = self.devices.iter().filter(|d| matches!(d.id, DeviceId::Gpu(_))).count();
        let mut lines = vec![format!("{} devices ({} GPU):", self.devices.len(), n_gpus)];
        for d in &self.devices {
            let vram = if d.memory_bytes > 0 {
                format!(" ({:.1} GB)", d.memory_bytes as f64 / 1e9)
            } else {
                String::new()
            };
            lines.push(format!("  {}: {}{vram}", d.id, d.name));
        }
        lines.push(String::new());
        lines.push("Region assignments:".into());
        for &region in REGIONS {
            let dev = self.device_for(region);
            lines.push(format!("  {region:16} → {dev}"));
        }
        lines.join("\n")
    }
}

// ─── Region Compute Handle ─────────────────────────────────

/// Per-region compute context. Holds device-local buffers and queue.
///
/// This is the key abstraction: each region gets its own RegionCompute,
/// which knows which device it lives on and can execute synapse.forward()
/// and region.step() on that device.
///
/// For CPU: just runs the computation inline.
/// For GPU: dispatches to the assigned device's queue.
#[derive(Debug)]
pub struct RegionCompute {
    pub region_name: String,
    pub device: DeviceId,
    // Future: per-device Vulkan queue, persistent buffers, etc.
    // For now, routes through the global GPU context.
}

impl RegionCompute {
    pub fn new(region_name: &str, device: DeviceId) -> Self {
        Self {
            region_name: region_name.into(),
            device,
        }
    }

    // TODO(workspace): forward_synapse needs crate::ctm::Synapse from the main crate.
    // Re-add once modgrad-device can depend on the Synapse type (or use a trait).
    // pub fn forward_synapse(&self, synapse_input: &[f32], synapse: &crate::ctm::Synapse) -> Vec<f32> {
    //     synapse.forward(synapse_input)
    // }
}

// ─── Cross-Device Transfer ─────────────────────────────────

/// Transfer activations between devices.
///
/// For same-device: zero-copy (just a reference).
/// For CPU↔GPU: host-visible mapped copy.
/// For GPU↔GPU: peer-to-peer if available, else stage through host.
///
/// At typical CTM scale (256-1024 neurons per region), this is
/// 1-4 KB per transfer — negligible compared to compute.
pub fn transfer(data: &[f32], _from: DeviceId, _to: DeviceId) -> Vec<f32> {
    // For now: all data is on host (CPU). The GPU compute kernels
    // upload/download per dispatch. True multi-GPU will need:
    //   - Persistent device-local buffers
    //   - Async transfers overlapped with compute
    //   - P2P for GPU-GPU (vkCmdCopyBuffer with peer memory)
    //
    // At 1-4 KB per activation vector, even PCIe copies are <1μs.
    // This is NOT the bottleneck. Compute is.
    data.to_vec()
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_only_mesh() {
        let mesh = DeviceMesh::cpu_only();
        assert_eq!(mesh.device_for("input"), DeviceId::Cpu);
        assert_eq!(mesh.device_for("nonexistent"), DeviceId::Cpu);
    }

    #[test]
    fn test_auto_mesh() {
        let mesh = DeviceMesh::auto();
        // Should at least have CPU
        assert!(!mesh.devices.is_empty());
        // All regions should have a device
        for &r in REGIONS {
            let _ = mesh.device_for(r);
        }
    }

    #[test]
    fn test_manual_assignment() {
        let mut mesh = DeviceMesh::cpu_only();
        mesh.assign("input", DeviceId::Gpu(0));
        assert_eq!(mesh.device_for("input"), DeviceId::Gpu(0));
        assert_eq!(mesh.device_for("output"), DeviceId::Cpu);
    }

    #[test]
    fn test_summary() {
        let mesh = DeviceMesh::cpu_only();
        let s = mesh.summary();
        assert!(s.contains("input"));
        assert!(s.contains("cpu"));
    }
}
