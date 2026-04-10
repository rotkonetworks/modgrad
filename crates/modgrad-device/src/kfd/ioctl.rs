//! Raw KFD ioctl bindings for AMD GPU access.
//!
//! Structs match /usr/include/linux/kfd_ioctl.h exactly.
//! No dependencies beyond libc.

use std::os::unix::io::RawFd;

// ─── ioctl direction bits ───────────────────────────────────

const IOC_NONE: u32 = 0;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;
const IOC_READWRITE: u32 = 3;

const AMDKFD_IOCTL_BASE: u32 = b'K' as u32;

const fn ioc(dir: u32, nr: u32, size: u32) -> u64 {
    ((dir as u64) << 30) | ((size as u64) << 16) | ((AMDKFD_IOCTL_BASE as u64) << 8) | (nr as u64)
}

const fn ior<T>(nr: u32) -> u64 { ioc(IOC_READ, nr, std::mem::size_of::<T>() as u32) }
const fn iow<T>(nr: u32) -> u64 { ioc(IOC_WRITE, nr, std::mem::size_of::<T>() as u32) }
const fn iowr<T>(nr: u32) -> u64 { ioc(IOC_READWRITE, nr, std::mem::size_of::<T>() as u32) }

unsafe fn kfd_ioctl<T>(fd: RawFd, request: u64, arg: &mut T) -> std::io::Result<()> { unsafe {
    let ret = libc::ioctl(fd, request as libc::c_ulong, arg as *mut T);
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}}

// ─── Memory allocation flags ────────────────────────────────

pub const ALLOC_MEM_FLAGS_VRAM: u32 = 1 << 0;
pub const ALLOC_MEM_FLAGS_GTT: u32 = 1 << 1;
pub const ALLOC_MEM_FLAGS_USERPTR: u32 = 1 << 2;
pub const ALLOC_MEM_FLAGS_DOORBELL: u32 = 1 << 3;
pub const ALLOC_MEM_FLAGS_MMIO_REMAP: u32 = 1 << 4;
pub const ALLOC_MEM_FLAGS_WRITABLE: u32 = 1 << 31;
pub const ALLOC_MEM_FLAGS_EXECUTABLE: u32 = 1 << 30;
pub const ALLOC_MEM_FLAGS_PUBLIC: u32 = 1 << 29;
pub const ALLOC_MEM_FLAGS_NO_SUBSTITUTE: u32 = 1 << 28;
pub const ALLOC_MEM_FLAGS_AQL_QUEUE_MEM: u32 = 1 << 27;
pub const ALLOC_MEM_FLAGS_COHERENT: u32 = 1 << 26;
pub const ALLOC_MEM_FLAGS_UNCACHED: u32 = 1 << 25;

// ─── Queue types ────────────────────────────────────────────

pub const QUEUE_TYPE_COMPUTE: u32 = 0;
pub const QUEUE_TYPE_SDMA: u32 = 1;
pub const QUEUE_TYPE_COMPUTE_AQL: u32 = 2;

pub const MAX_QUEUE_PERCENTAGE: u32 = 100;
pub const MAX_QUEUE_PRIORITY: u32 = 15;

// ─── Event types ────────────────────────────────────────────

pub const EVENT_SIGNAL: u32 = 0;
pub const EVENT_HW_EXCEPTION: u32 = 3;
pub const EVENT_MEMORY: u32 = 8;

// ─── Doorbell mmap ──────────────────────────────────────────

pub const MMAP_TYPE_DOORBELL: u64 = 0x3 << 62;

// ─── Structs ────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Default)]
pub struct GetVersionArgs {
    pub major_version: u32,
    pub minor_version: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct AcquireVmArgs {
    pub drm_fd: u32,
    pub gpu_id: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct CreateQueueArgs {
    pub ring_base_address: u64,
    pub write_pointer_address: u64,
    pub read_pointer_address: u64,
    pub doorbell_offset: u64,
    pub ring_size: u32,
    pub gpu_id: u32,
    pub queue_type: u32,
    pub queue_percentage: u32,
    pub queue_priority: u32,
    pub queue_id: u32,
    pub eop_buffer_address: u64,
    pub eop_buffer_size: u64,
    pub ctx_save_restore_address: u64,
    pub ctx_save_restore_size: u32,
    pub ctl_stack_size: u32,
    pub sdma_engine_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct DestroyQueueArgs {
    pub queue_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct AllocMemoryArgs {
    pub va_addr: u64,
    pub size: u64,
    pub handle: u64,
    pub mmap_offset: u64,
    pub gpu_id: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct FreeMemoryArgs {
    pub handle: u64,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct MapMemoryArgs {
    pub handle: u64,
    pub device_ids_array_ptr: u64,
    pub n_devices: u32,
    pub n_success: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct UnmapMemoryArgs {
    pub handle: u64,
    pub device_ids_array_ptr: u64,
    pub n_devices: u32,
    pub n_success: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct CreateEventArgs {
    pub event_page_offset: u64,
    pub event_trigger_data: u32,
    pub event_type: u32,
    pub auto_reset: u32,
    pub node_id: u32,
    pub event_id: u32,
    pub event_slot_index: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct DestroyEventArgs {
    pub event_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct SetEventArgs {
    pub event_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct WaitEventsArgs {
    pub events_ptr: u64,
    pub num_events: u32,
    pub wait_for_all: u32,
    pub timeout: u32,
    pub wait_result: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct RuntimeEnableArgs {
    pub r_debug: u64,
    pub mode_mask: u32,
    pub capabilities_mask: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct GetAvailableMemoryArgs {
    pub available: u64,
    pub gpu_id: u32,
    pub pad: u32,
}

// kfd_event_data — used in wait_events array
// Union of memory_exception / hw_exception / signal_event, followed by ext + event_id + pad
// We only need the signal case, which is simplest
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct EventData {
    // Union — signal_event_data is just u64 (last_event_age)
    // memory_exception_data is the largest at 32 bytes
    // We use raw bytes for the union
    pub union_data: [u8; 32],
    pub kfd_event_data_ext: u64,
    pub event_id: u32,
    pub pad: u32,
}

// ─── ioctl numbers ──────────────────────────────────────────

const IOC_GET_VERSION: u64 = ior::<GetVersionArgs>(0x01);
const IOC_CREATE_QUEUE: u64 = iowr::<CreateQueueArgs>(0x02);
const IOC_DESTROY_QUEUE: u64 = iowr::<DestroyQueueArgs>(0x03);
const IOC_CREATE_EVENT: u64 = iowr::<CreateEventArgs>(0x08);
const IOC_DESTROY_EVENT: u64 = iow::<DestroyEventArgs>(0x09);
const IOC_SET_EVENT: u64 = iow::<SetEventArgs>(0x0A);
const IOC_WAIT_EVENTS: u64 = iowr::<WaitEventsArgs>(0x0C);
const IOC_ACQUIRE_VM: u64 = iow::<AcquireVmArgs>(0x15);
const IOC_ALLOC_MEMORY: u64 = iowr::<AllocMemoryArgs>(0x16);
const IOC_FREE_MEMORY: u64 = iow::<FreeMemoryArgs>(0x17);
const IOC_MAP_MEMORY: u64 = iowr::<MapMemoryArgs>(0x18);
const IOC_UNMAP_MEMORY: u64 = iowr::<UnmapMemoryArgs>(0x19);
const IOC_AVAILABLE_MEMORY: u64 = iowr::<GetAvailableMemoryArgs>(0x23);
const IOC_RUNTIME_ENABLE: u64 = iowr::<RuntimeEnableArgs>(0x25);

// ─── Typed wrappers ─────────────────────────────────────────

pub fn get_version(fd: RawFd) -> std::io::Result<(u32, u32)> {
    let mut args = GetVersionArgs::default();
    unsafe { kfd_ioctl(fd, IOC_GET_VERSION, &mut args)?; }
    Ok((args.major_version, args.minor_version))
}

pub fn acquire_vm(fd: RawFd, drm_fd: RawFd, gpu_id: u32) -> std::io::Result<()> {
    let mut args = AcquireVmArgs { drm_fd: drm_fd as u32, gpu_id };
    unsafe { kfd_ioctl(fd, IOC_ACQUIRE_VM, &mut args) }
}

pub fn runtime_enable(fd: RawFd) -> std::io::Result<u32> {
    let mut args = RuntimeEnableArgs::default();
    unsafe { kfd_ioctl(fd, IOC_RUNTIME_ENABLE, &mut args)?; }
    #[cfg(debug_assertions)]
    eprintln!("      runtime_enable: caps=0x{:x}", args.capabilities_mask);
    Ok(args.capabilities_mask)
}

pub fn alloc_memory(fd: RawFd, va_addr: u64, size: u64, gpu_id: u32,
                    flags: u32, mmap_offset: u64) -> std::io::Result<AllocMemoryArgs> {
    let mut args = AllocMemoryArgs {
        va_addr, size, gpu_id, flags, mmap_offset, handle: 0,
    };
    unsafe { kfd_ioctl(fd, IOC_ALLOC_MEMORY, &mut args)?; }
    Ok(args)
}

pub fn free_memory(fd: RawFd, handle: u64) -> std::io::Result<()> {
    let mut args = FreeMemoryArgs { handle };
    unsafe { kfd_ioctl(fd, IOC_FREE_MEMORY, &mut args) }
}

pub fn map_memory(fd: RawFd, handle: u64, gpu_ids: &[u32]) -> std::io::Result<u32> {
    let mut args = MapMemoryArgs {
        handle,
        device_ids_array_ptr: gpu_ids.as_ptr() as u64,
        n_devices: gpu_ids.len() as u32,
        n_success: 0,
    };
    unsafe { kfd_ioctl(fd, IOC_MAP_MEMORY, &mut args)?; }
    Ok(args.n_success)
}

pub fn unmap_memory(fd: RawFd, handle: u64, gpu_ids: &[u32]) -> std::io::Result<()> {
    let mut args = UnmapMemoryArgs {
        handle,
        device_ids_array_ptr: gpu_ids.as_ptr() as u64,
        n_devices: gpu_ids.len() as u32,
        n_success: 0,
    };
    unsafe { kfd_ioctl(fd, IOC_UNMAP_MEMORY, &mut args)?; }
    Ok(())
}

pub fn create_queue(fd: RawFd, args: &mut CreateQueueArgs) -> std::io::Result<()> {
    unsafe { kfd_ioctl(fd, IOC_CREATE_QUEUE, args) }
}

pub fn destroy_queue(fd: RawFd, queue_id: u32) -> std::io::Result<()> {
    let mut args = DestroyQueueArgs { queue_id, pad: 0 };
    unsafe { kfd_ioctl(fd, IOC_DESTROY_QUEUE, &mut args) }
}

pub fn create_event(fd: RawFd, event_type: u32, auto_reset: u32,
                    event_page_offset: u64) -> std::io::Result<CreateEventArgs> {
    let mut args = CreateEventArgs {
        event_page_offset, event_type, auto_reset,
        ..Default::default()
    };
    unsafe { kfd_ioctl(fd, IOC_CREATE_EVENT, &mut args)?; }
    Ok(args)
}

pub fn set_event(fd: RawFd, event_id: u32) -> std::io::Result<()> {
    let mut args = SetEventArgs { event_id, pad: 0 };
    unsafe { kfd_ioctl(fd, IOC_SET_EVENT, &mut args) }
}

pub fn wait_events(fd: RawFd, events: &mut [EventData],
                   wait_for_all: bool, timeout_ms: u32) -> std::io::Result<u32> {
    let mut args = WaitEventsArgs {
        events_ptr: events.as_mut_ptr() as u64,
        num_events: events.len() as u32,
        wait_for_all: wait_for_all as u32,
        timeout: timeout_ms,
        wait_result: 0,
    };
    unsafe { kfd_ioctl(fd, IOC_WAIT_EVENTS, &mut args)?; }
    Ok(args.wait_result)
}

pub fn available_memory(fd: RawFd, gpu_id: u32) -> std::io::Result<u64> {
    let mut args = GetAvailableMemoryArgs { gpu_id, available: 0, pad: 0 };
    unsafe { kfd_ioctl(fd, IOC_AVAILABLE_MEMORY, &mut args)?; }
    Ok(args.available)
}
