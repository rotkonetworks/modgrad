//! Kernel dispatch: load code objects, set up kernel arguments, run.
//!
//! AMD code objects are ELF files containing multiple kernels.
//! Each kernel has a `.kd` descriptor in `.rodata` and ISA code in `.text`.
//! Symbol table maps kernel names to their descriptors.

use super::memory::{GpuAllocator, GpuBuffer};
use super::queue::ComputeQueue;
use std::collections::HashMap;

/// Parsed kernel descriptor from an AMD code object.
#[derive(Debug, Clone)]
pub struct KernelDescriptor {
    pub pgm_rsrc1: u32,
    pub pgm_rsrc2: u32,
    pub pgm_rsrc3: u32,
    pub kernel_code_entry_offset: i64,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
    pub kernarg_size: u32,
}

/// A single kernel within a loaded code object.
#[derive(Debug, Clone)]
pub struct KernelEntry {
    pub name: String,
    pub desc: KernelDescriptor,
    pub code_addr: u64,
}

/// A loaded code object in GPU memory, containing one or more kernels.
pub struct CodeObject {
    pub buf: GpuBuffer,
    pub kernels: HashMap<String, KernelEntry>,
}

impl CodeObject {
    /// Parse only (no GPU upload) — for diagnostics.
    pub fn parse_only(elf_data: &[u8]) -> usize {
        match parse_elf_kernels(elf_data, 0) {
            Some(v) => v.len(),
            None => 0,
        }
    }

    /// Load a code object ELF into GPU memory and parse all kernels.
    ///
    /// The ELF is loaded as a flat VA-indexed image: we allocate enough space
    /// for the highest VA in the ELF, then copy each LOAD segment to its VA offset.
    /// This way VA addresses in the code work correctly as offsets from gpu_base.
    pub fn load(alloc: &GpuAllocator, elf_data: &[u8]) -> std::io::Result<Self> {
        // Build VA-indexed image (like tinygrad's elf_loader)
        let image = build_va_image(elf_data)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData,
                "failed to parse ELF"))?;

        let size = ((image.len() + 4095) & !4095) as u64;
        // VRAM with PUBLIC — code lives in GPU memory, CPU-accessible via BAR for upload
        let buf = alloc.alloc_vram(size)?;
        buf.write(0, &image);

        let kernels = parse_elf_kernels(elf_data, buf.va_addr)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData,
                "failed to parse code object ELF"))?;

        let mut map = HashMap::new();
        for k in kernels {
            map.insert(k.name.clone(), k);
        }

        Ok(CodeObject { buf, kernels: map })
    }

    /// Get a kernel by name.
    pub fn kernel(&self, name: &str) -> Option<&KernelEntry> {
        self.kernels.get(name)
    }
}

/// A ready-to-dispatch GPU program (single kernel from a code object).
pub struct GpuProgram {
    pub code_buf: GpuBuffer,
    pub desc: KernelDescriptor,
    pub code_addr: u64,
}

impl GpuProgram {
    /// Load a single-kernel code object.
    pub fn load(alloc: &GpuAllocator, code_object: &[u8]) -> std::io::Result<Self> {
        let co = CodeObject::load(alloc, code_object)?;
        let first = co.kernels.values().next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData,
                "no kernels in code object"))?;
        Ok(GpuProgram {
            desc: first.desc.clone(),
            code_addr: first.code_addr,
            code_buf: co.buf,
        })
    }

    /// Dispatch and signal.
    pub fn dispatch(&self, queue: &mut ComputeQueue,
                    kernargs: &GpuBuffer, scratch_addr: u64,
                    grid: [u32; 3], block: [u32; 3],
                    signal: &GpuBuffer, signal_value: u32,
                    event_mailbox_ptr: u64, event_id: u32) {
        queue.dispatch_lds(
            self.code_addr,
            self.desc.pgm_rsrc1,
            self.desc.pgm_rsrc2,
            self.desc.pgm_rsrc3,
            kernargs.va_addr,
            scratch_addr,
            grid, block,
            self.desc.group_segment_size,
        );
        queue.signal(signal.va_addr, signal_value, event_mailbox_ptr, event_id);
        queue.submit();
    }
}

// ─── ELF parsing ────────────────────────────────────────────

/// ELF64 symbol types
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;

/// Parse all kernels from an ELF code object.
/// Each kernel has a `<name>.kd` OBJECT symbol in .rodata (the descriptor)
/// and a `<name>` FUNC symbol in .text (the code entry).
fn parse_elf_kernels(elf: &[u8], gpu_base: u64) -> Option<Vec<KernelEntry>> {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" { return None; }

    let e_shoff = r64(elf, 40) as usize;
    let e_shentsize = r16(elf, 58) as usize;
    let e_shnum = r16(elf, 60) as usize;
    let e_shstrndx = r16(elf, 62) as usize;

    // Get section name string table
    let shstrtab_sh = e_shoff + e_shstrndx * e_shentsize;
    let shstrtab_off = r64(elf, shstrtab_sh + 24) as usize;

    // Find section headers by name
    let mut symtab_off = 0usize;
    let mut symtab_size = 0usize;
    let mut symtab_entsize = 0usize;
    let mut symtab_link = 0usize;
    let mut rodata_off = 0usize;
    let mut rodata_addr = 0u64;

    for i in 0..e_shnum {
        let sh = e_shoff + i * e_shentsize;
        if sh + e_shentsize > elf.len() { break; }

        let sh_name_idx = r32(elf, sh) as usize;
        let sh_type = r32(elf, sh + 4);
        let sh_addr = r64(elf, sh + 16);
        let sh_offset = r64(elf, sh + 24) as usize;
        let sh_size = r64(elf, sh + 32) as usize;

        let name = read_str(elf, shstrtab_off + sh_name_idx);

        if sh_type == 2 { // SHT_SYMTAB
            symtab_off = sh_offset;
            symtab_size = sh_size;
            symtab_entsize = r64(elf, sh + 56) as usize;
            symtab_link = r32(elf, sh + 40) as usize;
        }
        if name == ".rodata" {
            rodata_off = sh_offset;
            rodata_addr = sh_addr;
        }
    }

    if symtab_entsize == 0 { return None; }

    // Get strtab
    let strtab_sh = e_shoff + symtab_link * e_shentsize;
    let strtab_off = r64(elf, strtab_sh + 24) as usize;

    // Parse symbols: collect .kd descriptors and FUNC addresses
    let mut kd_map: HashMap<String, (u64, usize)> = HashMap::new(); // name -> (vaddr, file_offset)
    let mut func_map: HashMap<String, u64> = HashMap::new(); // name -> vaddr

    let n_syms = symtab_size / symtab_entsize;
    for i in 0..n_syms {
        let sym = symtab_off + i * symtab_entsize;
        if sym + symtab_entsize > elf.len() { break; }

        let st_name = r32(elf, sym) as usize;
        let st_info = elf[sym + 4];
        let st_value = r64(elf, sym + 8);
        let _st_size = r64(elf, sym + 16);
        let st_type = st_info & 0xf;

        let name = read_str(elf, strtab_off + st_name);

        if st_type == STT_OBJECT && name.ends_with(".kd") {
            let kernel_name = name[..name.len() - 3].to_string();
            // .kd symbol value is the VA in the ELF. Since we upload VA-indexed image,
            // the file_off in the original ELF for reading the descriptor is rodata_off + (va - rodata_addr)
            // but the GPU offset is just the VA itself (since image is VA-indexed).
            let elf_file_off = rodata_off + (st_value - rodata_addr) as usize;
            kd_map.insert(kernel_name, (st_value, elf_file_off));
        } else if st_type == STT_FUNC && !name.starts_with("__clang") {
            func_map.insert(name.to_string(), st_value);
        }
    }

    // Match descriptors with code addresses
    let mut kernels = Vec::new();
    for (name, (kd_vaddr, kd_file_off)) in &kd_map {
        if kd_file_off + 64 > elf.len() { continue; }
        let kd = &elf[*kd_file_off..];

        // AMDHSA kernel descriptor layout (verified against tinygrad):
        //   +00: group_segment_fixed_size (u32)
        //   +04: private_segment_fixed_size (u32)
        //   +08: kernarg_size (u32)
        //   +0c: reserved (u32)
        //   +10: kernel_code_entry_byte_offset (i64)
        //   +18: reserved (20 bytes)
        //   +2c: compute_pgm_rsrc3 (u32)
        //   +30: compute_pgm_rsrc1 (u32)
        //   +34: compute_pgm_rsrc2 (u32)
        //   +38: kernel_code_properties (u16) + kernarg_preload (u16)
        let desc = KernelDescriptor {
            group_segment_size: u32::from_le_bytes(kd[0x00..0x04].try_into().unwrap()),
            private_segment_size: u32::from_le_bytes(kd[0x04..0x08].try_into().unwrap()),
            kernarg_size: u32::from_le_bytes(kd[0x08..0x0c].try_into().unwrap()),
            kernel_code_entry_offset: i64::from_le_bytes(kd[0x10..0x18].try_into().unwrap()),
            pgm_rsrc3: u32::from_le_bytes(kd[0x2c..0x30].try_into().unwrap()),
            pgm_rsrc1: u32::from_le_bytes(kd[0x30..0x34].try_into().unwrap()),
            pgm_rsrc2: u32::from_le_bytes(kd[0x34..0x38].try_into().unwrap()),
        };

        // Code address: since we upload VA-indexed image, gpu_base + kd_vaddr = descriptor addr.
        // kernel_code_entry_offset is relative to the descriptor's VA.
        let kd_gpu_addr = gpu_base + *kd_vaddr;
        let code_addr = (kd_gpu_addr as i64 + desc.kernel_code_entry_offset) as u64;

        eprintln!("    kernel '{}': kd_va=0x{:x} file_off=0x{:x} entry_off=0x{:x} code=0x{:x}",
            name, kd_vaddr, kd_file_off, desc.kernel_code_entry_offset, code_addr);

        kernels.push(KernelEntry {
            name: name.clone(),
            desc,
            code_addr,
        });
    }

    Some(kernels)
}

/// Build a VA-indexed image from an ELF: allocate buffer sized to max VA,
/// copy each LOAD segment to its VA offset. This way gpu_base + VA = correct GPU address.
fn build_va_image(elf: &[u8]) -> Option<Vec<u8>> {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" { return None; }

    let e_phoff = r64(elf, 32) as usize;
    let e_phentsize = r16(elf, 54) as usize;
    let e_phnum = r16(elf, 56) as usize;

    // Find max VA across all LOAD segments
    let mut max_va: usize = 0;
    for i in 0..e_phnum {
        let ph = e_phoff + i * e_phentsize;
        if ph + e_phentsize > elf.len() { break; }
        let p_type = r32(elf, ph);
        if p_type != 1 { continue; } // PT_LOAD = 1
        let p_vaddr = r64(elf, ph + 16) as usize;
        let p_memsz = r64(elf, ph + 40) as usize;
        max_va = max_va.max(p_vaddr + p_memsz);
    }

    if max_va == 0 { return None; }
    let mut image = vec![0u8; max_va];

    // Copy each LOAD segment to its VA position
    for i in 0..e_phnum {
        let ph = e_phoff + i * e_phentsize;
        if ph + e_phentsize > elf.len() { break; }
        let p_type = r32(elf, ph);
        if p_type != 1 { continue; }
        let p_offset = r64(elf, ph + 8) as usize;
        let p_vaddr = r64(elf, ph + 16) as usize;
        let p_filesz = r64(elf, ph + 32) as usize;
        let end = (p_offset + p_filesz).min(elf.len());
        let copy_len = end - p_offset;
        if p_vaddr + copy_len <= image.len() {
            image[p_vaddr..p_vaddr + copy_len].copy_from_slice(&elf[p_offset..p_offset + copy_len]);
        }
    }

    Some(image)
}

fn r16(data: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(data[off..off+2].try_into().unwrap())
}
fn r32(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(data[off..off+4].try_into().unwrap())
}
fn r64(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(data[off..off+8].try_into().unwrap())
}

fn read_str(data: &[u8], off: usize) -> &str {
    let end = data[off..].iter().position(|&b| b == 0).unwrap_or(0);
    std::str::from_utf8(&data[off..off + end]).unwrap_or("")
}

// ─── Kernel argument builder ────────────────────────────────

/// Helper to pack kernel arguments into a GPU buffer.
pub struct KernArgs {
    data: Vec<u8>,
}

impl KernArgs {
    pub fn new() -> Self {
        KernArgs { data: Vec::with_capacity(256) }
    }

    pub fn push_ptr(&mut self, buf: &GpuBuffer) {
        self.align(8);
        self.data.extend_from_slice(&buf.va_addr.to_le_bytes());
    }

    pub fn push_u64(&mut self, val: u64) {
        self.align(8);
        self.data.extend_from_slice(&val.to_le_bytes());
    }

    pub fn push_u32(&mut self, val: u32) {
        self.align(4);
        self.data.extend_from_slice(&val.to_le_bytes());
    }

    pub fn push_f32(&mut self, val: f32) {
        self.align(4);
        self.data.extend_from_slice(&val.to_le_bytes());
    }

    fn align(&mut self, alignment: usize) {
        let pad = (alignment - (self.data.len() % alignment)) % alignment;
        self.data.extend(std::iter::repeat(0u8).take(pad));
    }

    /// Fill OpenCL hidden arguments based on dispatch grid/block sizes.
    /// Must be called after all explicit args are pushed but before upload.
    /// Standard hidden arg layout (from clang OpenCL compiler):
    ///   +0x00 from end of explicit: hidden_block_count_x/y/z (3x u32)
    ///   +0x0c: hidden_group_size_x/y/z (3x u16)
    ///   +0x12: hidden_remainder_x/y/z (3x u16)
    ///   +0x18: (padding to 0x28 from end of explicit)
    ///   +0x28: hidden_global_offset_x/y/z (3x u64)
    ///   +0x40: hidden_grid_dims (u16)
    pub fn fill_hidden_args(&mut self, grid: [u32; 3], block: [u32; 3], kernarg_size: u32) {
        // Pad to the expected kernarg_size
        while self.data.len() < kernarg_size as usize {
            self.data.push(0);
        }

        // Block counts = grid / block (number of workgroups per dimension)
        let block_count = [
            if block[0] > 0 { grid[0] / block[0] } else { 0 },
            if block[1] > 0 { grid[1] / block[1] } else { 0 },
            if block[2] > 0 { grid[2] / block[2] } else { 0 },
        ];
        let remainder = [
            if block[0] > 0 { (grid[0] % block[0]) as u16 } else { 0 },
            if block[1] > 0 { (grid[1] % block[1]) as u16 } else { 0 },
            if block[2] > 0 { (grid[2] % block[2]) as u16 } else { 0 },
        ];
        let grid_dims: u16 = if grid[2] > 1 { 3 } else if grid[1] > 1 { 2 } else { 1 };

        // Write at standard offsets (relative to byte 0x28 in our matvec case)
        // These offsets come from the .note metadata: hidden_block_count_x at 0x28
        let base = 0x28usize; // first hidden arg offset for our kernels
        if base + 12 <= self.data.len() {
            self.data[base..base+4].copy_from_slice(&block_count[0].to_le_bytes());
            self.data[base+4..base+8].copy_from_slice(&block_count[1].to_le_bytes());
            self.data[base+8..base+12].copy_from_slice(&block_count[2].to_le_bytes());
        }
        let gs = base + 12; // hidden_group_size
        if gs + 6 <= self.data.len() {
            self.data[gs..gs+2].copy_from_slice(&(block[0] as u16).to_le_bytes());
            self.data[gs+2..gs+4].copy_from_slice(&(block[1] as u16).to_le_bytes());
            self.data[gs+4..gs+6].copy_from_slice(&(block[2] as u16).to_le_bytes());
        }
        let rm = gs + 6; // hidden_remainder
        if rm + 6 <= self.data.len() {
            self.data[rm..rm+2].copy_from_slice(&remainder[0].to_le_bytes());
            self.data[rm+2..rm+4].copy_from_slice(&remainder[1].to_le_bytes());
            self.data[rm+4..rm+6].copy_from_slice(&remainder[2].to_le_bytes());
        }
        // hidden_global_offset at 0x50
        // (all zeros — we don't use global offsets)
        // hidden_grid_dims at 0x68
        if 0x68 + 2 <= self.data.len() {
            self.data[0x68..0x6a].copy_from_slice(&grid_dims.to_le_bytes());
        }
    }

    pub fn upload(&self, alloc: &GpuAllocator) -> std::io::Result<GpuBuffer> {
        let size = ((self.data.len().max(256) + 4095) & !4095) as u64;
        // Kernargs must be GPU-readable (PUBLIC flag)
        let buf = alloc.alloc_userptr_public(size)?;
        buf.write(0, &self.data);
        Ok(buf)
    }
}
