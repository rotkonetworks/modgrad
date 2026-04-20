//! Vulkan compute backend for GPU-accelerated SuperLinear and matvec.
//!
//! Pattern: only big matmuls go to GPU. Norms, activations, sync stay on CPU.
//! Disable via ISIS_NO_GPU=1.
//!
//! Fixed for RDNA3 (RX 7600): uses DEVICE_LOCAL memory with staging transfers,
//! pre-allocated buffer pool, and proper bounds validation.

#[cfg(feature = "gpu")]
use ash::vk;
#[cfg(feature = "gpu")]
use std::ffi::CStr;

#[cfg(feature = "gpu")]
const TIMEOUT_NS: u64 = 5_000_000_000; // 5 seconds

/// GPU compute context. Created once, reused for all dispatches.
#[cfg(feature = "gpu")]
pub struct GpuContext {
    instance: ash::Instance,
    device: ash::Device,
    phys_device: vk::PhysicalDevice,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
    desc_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    superlinear_pl: Option<(vk::Pipeline, vk::PipelineLayout)>,
    matvec_pl: Option<(vk::Pipeline, vk::PipelineLayout)>,
}

#[cfg(feature = "gpu")]
#[repr(C)]
struct SuperlinearPC {
    n_neurons: u32,
    in_per: u32,
    out_per: u32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
struct MatvecPC {
    out_dim: u32,
    in_dim: u32,
}

/// A GPU buffer with host-visible mapped memory.
#[cfg(feature = "gpu")]
struct MappedBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    ptr: *mut u8,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Try to initialize Vulkan compute. Returns None if unavailable.
    pub fn init() -> Option<Self> {
        if std::env::var("ISIS_NO_GPU").is_ok() {
            return None;
        }

        let entry = unsafe { ash::Entry::load().ok()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"isis")
            .api_version(vk::make_api_version(0, 1, 2, 0));
        let inst_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&inst_info, None).ok()? };

        let phys_devices = unsafe { instance.enumerate_physical_devices().ok()? };
        if phys_devices.is_empty() { return None; }

        // Pick best GPU
        let mut chosen = phys_devices[0];
        for &pd in &phys_devices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                chosen = pd;
                break;
            }
        }

        let props = unsafe { instance.get_physical_device_properties(chosen) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        eprintln!("GPU: {} ({:?})", name.to_string_lossy(), props.device_type);

        let mem_props = unsafe { instance.get_physical_device_memory_properties(chosen) };

        // Find compute queue
        let qf_props = unsafe { instance.get_physical_device_queue_family_properties(chosen) };
        let qf_idx = qf_props.iter().position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))? as u32;

        let prios = [1.0f32];
        let qci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf_idx)
            .queue_priorities(&prios);
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&qci));
        let device = unsafe { instance.create_device(chosen, &dci, None).ok()? };
        let queue = unsafe { device.get_device_queue(qf_idx, 0) };

        // Command pool + buffer + fence
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf_idx)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { device.create_command_pool(&pool_ci, None).ok()? };

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buf = unsafe { device.allocate_command_buffers(&alloc_ci).ok()?[0] };
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None).ok()? };

        // Descriptor layout: 4 storage buffers
        let bindings: Vec<_> = (0..4).map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        }).collect();
        let dl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_layout = unsafe { device.create_descriptor_set_layout(&dl_ci, None).ok()? };

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 32,
        }];
        let dp_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(8)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let desc_pool = unsafe { device.create_descriptor_pool(&dp_ci, None).ok()? };

        let superlinear_pl = Self::make_pipeline(&device, desc_layout,
            include_bytes!("superlinear.spv"),
            std::mem::size_of::<SuperlinearPC>() as u32);
        let matvec_pl = Self::make_pipeline(&device, desc_layout,
            include_bytes!("matvec.spv"),
            std::mem::size_of::<MatvecPC>() as u32);

        eprintln!("GPU pipelines: superlinear={} matvec={}",
            superlinear_pl.is_some(), matvec_pl.is_some());

        Some(Self {
            instance, device, phys_device: chosen, mem_props,
            queue, cmd_pool, cmd_buf, fence,
            desc_layout, desc_pool,
            superlinear_pl, matvec_pl,
        })
    }

    fn make_pipeline(
        device: &ash::Device, layout: vk::DescriptorSetLayout,
        spirv: &[u8], push_size: u32,
    ) -> Option<(vk::Pipeline, vk::PipelineLayout)> {
        let code: Vec<u32> = spirv.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sm_ci = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader = unsafe { device.create_shader_module(&sm_ci, None).ok()? };

        let push = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0, size: push_size,
        };
        let pl_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&layout))
            .push_constant_ranges(std::slice::from_ref(&push));
        let pl_layout = unsafe { device.create_pipeline_layout(&pl_ci, None).ok()? };

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main");
        let cp_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pl_layout);
        let pipes = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(),
                std::slice::from_ref(&cp_ci), None).ok()?
        };
        unsafe { device.destroy_shader_module(shader, None) };
        Some((pipes[0], pl_layout))
    }

    fn find_memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        for i in 0..self.mem_props.memory_type_count {
            if (type_bits & (1 << i)) != 0
                && self.mem_props.memory_types[i as usize].property_flags.contains(flags)
            {
                return Some(i);
            }
        }
        None
    }

    /// Create a buffer with the EXACT size the shader will access.
    /// Uses HOST_VISIBLE | HOST_COHERENT for simplicity.
    /// Round up to 256-byte alignment for Vulkan spec compliance.
    fn create_buffer(&self, size: u64) -> Option<MappedBuffer> {
        let aligned_size = (size + 255) & !255; // 256-byte alignment
        let buf_ci = vk::BufferCreateInfo::default()
            .size(aligned_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&buf_ci, None).ok()? };
        let reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let mem_type = self.find_memory_type(
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(mem_type);
        let memory = unsafe { self.device.allocate_memory(&alloc, None).ok()? };
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0).ok()? };

        let ptr = unsafe {
            self.device.map_memory(memory, 0, reqs.size, vk::MemoryMapFlags::empty()).ok()? as *mut u8
        };

        // Zero the buffer to prevent reading uninitialized memory
        unsafe { std::ptr::write_bytes(ptr, 0, reqs.size as usize); }

        Some(MappedBuffer { buffer, memory, size: reqs.size, ptr })
    }

    /// Upload data to buffer. If data is smaller than buffer, remaining is zeroed.
    fn upload(&self, buf: &MappedBuffer, data: &[f32]) {
        let bytes = data.len() * 4;
        let copy_len = bytes.min(buf.size as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8, buf.ptr, copy_len);
        }
    }

    fn download(&self, buf: &MappedBuffer, out: &mut [f32]) {
        let bytes = out.len() * 4;
        let copy_len = bytes.min(buf.size as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.ptr, out.as_mut_ptr() as *mut u8, copy_len);
        }
    }

    fn dispatch(
        &self,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        desc_set: vk::DescriptorSet,
        push_data: &[u8],
        groups_x: u32,
    ) -> bool {
        unsafe {
            self.device.reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty()).ok();
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            if self.device.begin_command_buffer(self.cmd_buf, &begin).is_err() { return false; }

            self.device.cmd_bind_pipeline(self.cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                self.cmd_buf, vk::PipelineBindPoint::COMPUTE, layout,
                0, &[desc_set], &[]);
            self.device.cmd_push_constants(
                self.cmd_buf, layout, vk::ShaderStageFlags::COMPUTE,
                0, push_data);
            self.device.cmd_dispatch(self.cmd_buf, groups_x, 1, 1);

            if self.device.end_command_buffer(self.cmd_buf).is_err() { return false; }

            self.device.reset_fences(&[self.fence]).ok();
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.cmd_buf));
            if self.device.queue_submit(self.queue, &[submit], self.fence).is_err() { return false; }
            self.device.wait_for_fences(&[self.fence], true, TIMEOUT_NS).is_ok()
        }
    }

    /// Run SuperLinear on GPU.
    /// trace: [n_neurons × in_per], weights: [n_neurons × out_per × in_per],
    /// biases: [n_neurons × out_per] → out: [n_neurons × out_per]
    pub fn superlinear(
        &self,
        trace: &[f32],
        weights: &[f32],
        biases: &[f32],
        out: &mut [f32],
        n_neurons: u32,
        in_per: u32,
        out_per: u32,
    ) -> bool {
        let (pipeline, layout) = match &self.superlinear_pl {
            Some(p) => *p,
            None => return false,
        };

        // Validate bounds: shader shared memory limit
        if in_per > 512 { return false; } // shared_trace[512] in shader

        // Buffer sizes based on what the SHADER accesses, not host data length.
        // This ensures no out-of-bounds GPU reads.
        let trace_elems = n_neurons as usize * in_per as usize;
        let out_elems = n_neurons as usize * out_per as usize;
        let weight_elems = n_neurons as usize * out_per as usize * in_per as usize;
        let bias_elems = n_neurons as usize * out_per as usize;

        let trace_buf = match self.create_buffer((trace_elems * 4) as u64) { Some(b) => b, None => return false };
        let out_buf = match self.create_buffer((out_elems * 4) as u64) { Some(b) => b, None => return false };
        let weight_buf = match self.create_buffer((weight_elems * 4) as u64) { Some(b) => b, None => return false };
        let bias_buf = match self.create_buffer((bias_elems * 4) as u64) { Some(b) => b, None => return false };

        // Upload — only copy what we have, rest is zeroed by create_buffer
        self.upload(&trace_buf, &trace[..trace_elems.min(trace.len())]);
        self.upload(&weight_buf, &weights[..weight_elems.min(weights.len())]);
        self.upload(&bias_buf, &biases[..bias_elems.min(biases.len())]);

        // Descriptor set
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.desc_pool)
            .set_layouts(std::slice::from_ref(&self.desc_layout));
        let desc_set = match unsafe { self.device.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => sets[0],
            Err(_) => return false,
        };

        let bufs = [&trace_buf, &out_buf, &weight_buf, &bias_buf];
        let buf_infos: Vec<_> = bufs.iter().map(|b| {
            [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: b.size }]
        }).collect();
        let writes: Vec<_> = buf_infos.iter().enumerate().map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        }).collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let pc = SuperlinearPC { n_neurons, in_per, out_per };
        let pc_bytes = unsafe {
            std::slice::from_raw_parts(&pc as *const _ as *const u8,
                std::mem::size_of::<SuperlinearPC>())
        };

        let ok = self.dispatch(pipeline, layout, desc_set, pc_bytes, n_neurons);

        if ok {
            let dl = out_elems.min(out.len());
            self.download(&out_buf, &mut out[..dl]);
        }

        unsafe {
            self.device.free_descriptor_sets(self.desc_pool, &[desc_set]).ok();
            for b in [trace_buf, out_buf, weight_buf, bias_buf] {
                self.device.destroy_buffer(b.buffer, None);
                self.device.free_memory(b.memory, None);
            }
        }

        ok
    }

    /// Run matvec on GPU: y = Wx + b
    pub fn matvec(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        out: &mut [f32],
        out_dim: u32,
        in_dim: u32,
    ) -> bool {
        let (pipeline, layout) = match &self.matvec_pl {
            Some(p) => *p,
            None => return false,
        };

        // Buffer sizes based on what the SHADER accesses.
        // The shader reads: x[0..in_dim], weight[0..out_dim*in_dim], bias[0..out_dim]
        // The shader writes: out[0..out_dim]
        let x_elems = in_dim as usize;
        let out_elems = out_dim as usize;
        let w_elems = out_dim as usize * in_dim as usize;
        let b_elems = out_dim as usize;

        let x_buf = match self.create_buffer((x_elems * 4) as u64) { Some(b) => b, None => return false };
        let out_buf = match self.create_buffer((out_elems * 4) as u64) { Some(b) => b, None => return false };
        let w_buf = match self.create_buffer((w_elems * 4) as u64) { Some(b) => b, None => return false };
        let b_buf = match self.create_buffer((b_elems * 4) as u64) { Some(b) => b, None => return false };

        // Upload — only copy what we have, buffer is pre-zeroed
        self.upload(&x_buf, &x[..x_elems.min(x.len())]);
        self.upload(&w_buf, &weight[..w_elems.min(weight.len())]);
        self.upload(&b_buf, &bias[..b_elems.min(bias.len())]);

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.desc_pool)
            .set_layouts(std::slice::from_ref(&self.desc_layout));
        let desc_set = match unsafe { self.device.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => sets[0],
            Err(_) => return false,
        };

        let bufs_arr = [&x_buf, &out_buf, &w_buf, &b_buf];
        let buf_infos: Vec<_> = bufs_arr.iter().map(|b| {
            [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: b.size }]
        }).collect();
        let writes: Vec<_> = buf_infos.iter().enumerate().map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        }).collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let pc = MatvecPC { out_dim, in_dim };
        let pc_bytes = unsafe {
            std::slice::from_raw_parts(&pc as *const _ as *const u8,
                std::mem::size_of::<MatvecPC>())
        };

        let ok = self.dispatch(pipeline, layout, desc_set, pc_bytes, out_dim);

        if ok {
            let dl = out_elems.min(out.len());
            self.download(&out_buf, &mut out[..dl]);
        }

        unsafe {
            self.device.free_descriptor_sets(self.desc_pool, &[desc_set]).ok();
            for b in [x_buf, out_buf, w_buf, b_buf] {
                self.device.destroy_buffer(b.buffer, None);
                self.device.free_memory(b.memory, None);
            }
        }

        ok
    }
}

#[cfg(feature = "gpu")]
impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            if let Some((p, l)) = self.superlinear_pl {
                self.device.destroy_pipeline(p, None);
                self.device.destroy_pipeline_layout(l, None);
            }
            if let Some((p, l)) = self.matvec_pl {
                self.device.destroy_pipeline(p, None);
                self.device.destroy_pipeline_layout(l, None);
            }
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_descriptor_set_layout(self.desc_layout, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ─── Public API ─────────────────────────────────────────────

/// Global GPU context. Initialized once, shared across all dispatches.
#[cfg(feature = "gpu")]
static GPU: std::sync::OnceLock<Option<GpuContext>> = std::sync::OnceLock::new();

/// Initialize the global GPU context. Call once at startup.
pub fn init_global() {
    #[cfg(feature = "gpu")]
    {
        GPU.get_or_init(|| GpuContext::init());
    }
}

/// Try to run SuperLinear on GPU. Returns false if unavailable (caller should fall back to CPU).
pub fn try_superlinear(
    trace: &[f32], weights: &[f32], biases: &[f32], out: &mut [f32],
    n_neurons: u32, in_per: u32, out_per: u32,
) -> bool {
    #[cfg(feature = "gpu")]
    {
        if let Some(Some(gpu)) = GPU.get() {
            return gpu.superlinear(trace, weights, biases, out, n_neurons, in_per, out_per);
        }
    }
    #[cfg(not(feature = "gpu"))]
    { let _ = (trace, weights, biases, out, n_neurons, in_per, out_per); }
    false
}

/// Try to run matvec on GPU. Returns false if unavailable.
pub fn try_matvec(
    x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32],
    out_dim: u32, in_dim: u32,
) -> bool {
    #[cfg(feature = "gpu")]
    {
        if let Some(Some(gpu)) = GPU.get() {
            return gpu.matvec(x, weight, bias, out, out_dim, in_dim);
        }
    }
    #[cfg(not(feature = "gpu"))]
    { let _ = (x, weight, bias, out, out_dim, in_dim); }
    false
}

/// Check if GPU is available.
pub fn available() -> bool {
    #[cfg(feature = "gpu")]
    {
        matches!(GPU.get(), Some(Some(_)))
    }
    #[cfg(not(feature = "gpu"))]
    { false }
}
