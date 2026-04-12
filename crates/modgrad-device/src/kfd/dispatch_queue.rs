//! Batched GPU dispatch queue.
//!
//! The old StreamEngine did upload→dispatch→wait→download per op.
//! That's ~30us host overhead wrapping ~3us GPU work. Unusable.
//!
//! This module does it right:
//!   1. Allocate VRAM buffers once (VramBuf — opaque, CPU never reads)
//!   2. Upload inputs once before the batch
//!   3. Queue N kernel dispatches — each just writes PM4 packets (~100ns)
//!   4. Flush once — single signal + poll
//!   5. Download outputs once after the batch
//!
//! For a full training tick (12+ ops), this is:
//!   2 PCIe transfers + 1 sync vs. 24+ PCIe transfers + 12 syncs.

use super::memory::GpuBuffer;
use super::HsaDevice;
use std::collections::HashMap;

// ─── VRAM buffer handle ──────────────────────────────────────

/// Opaque VRAM buffer. GPU accesses at full bandwidth via `va`.
/// CPU reads/writes ONLY through explicit upload/download methods
/// which go through the GpuBuffer's BAR-mapped cpu_ptr in bulk.
pub struct VramBuf {
    pub(crate) buf: GpuBuffer,
    /// CPU-visible pointer (BAR-mapped). Use only for bulk upload/download.
    pub ptr: *mut f32,
    /// GPU virtual address.
    pub va: u64,
}

impl VramBuf {
    /// GPU virtual address (for kernargs).
    pub fn va(&self) -> u64 { self.va }

    /// Number of bytes.
    pub fn size(&self) -> usize { self.buf.size as usize }

    /// Upload f32 data from CPU → VRAM. Call BEFORE dispatching.
    pub fn upload(&self, data: &[f32]) {
        self.buf.write_f32(0, data);
    }

    /// Download f32 data from VRAM → CPU. Call AFTER flush().
    /// Uses volatile reads to bypass CPU cache and see GPU's writes.
    pub fn download(&self, n: usize) -> Vec<f32> {
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        let mut out = vec![0.0f32; n];
        let src = self.buf.cpu_ptr as *const f32;
        for i in 0..n {
            out[i] = unsafe { std::ptr::read_volatile(src.add(i)) };
        }
        out
    }

    /// Download into existing slice.
    pub fn download_into(&self, dst: &mut [f32]) {
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        let src = self.buf.cpu_ptr as *const f32;
        for i in 0..dst.len() {
            dst[i] = unsafe { std::ptr::read_volatile(src.add(i)) };
        }
    }

    /// Zero the buffer. Call BEFORE dispatching if needed.
    pub fn zero(&self) {
        unsafe { std::ptr::write_bytes(self.buf.cpu_ptr, 0, self.buf.size as usize); }
    }
}

// ─── Batched dispatch queue ──────────────────────────────────

/// Batched GPU dispatch queue.
///
/// Queues kernel dispatches without synchronizing. Each dispatch
/// writes PM4 packets to the ring buffer (~100ns host cost).
/// `flush()` submits all queued work and waits for completion.
///
/// Kernargs: each dispatch gets its own slot in a pre-allocated
/// args buffer so packets don't overwrite each other.
pub struct GpuQueue {
    /// Weight VRAM cache (pointer identity → VRAM buffer).
    weight_cache: HashMap<(usize, usize), WeightEntry>,
    /// Kernarg slab: pre-allocated VRAM, 256 bytes per dispatch slot.
    args_slab: Option<GpuBuffer>,
    args_slab_cap: usize,
    /// Next free slot in the args slab.
    args_cursor: usize,
    /// Number of dispatches queued since last flush.
    pending: usize,
}

struct WeightEntry {
    buf: GpuBuffer,
    w_bytes: usize,
}

/// Kernarg slot size — each dispatch gets this many bytes.
/// Must be ≥ max kernarg size (sync_update = 88 bytes) + alignment.
const ARGS_SLOT: usize = 256;

/// Max dispatches per batch before we need a bigger args slab.
const MAX_BATCH: usize = 256;

impl GpuQueue {
    pub fn new() -> Self {
        Self {
            weight_cache: HashMap::new(),
            args_slab: None,
            args_slab_cap: 0,
            args_cursor: 0,
            pending: 0,
        }
    }

    /// Allocate a VRAM buffer for activations/intermediates.
    pub fn alloc(&self, dev: &HsaDevice, n_floats: usize) -> Option<VramBuf> {
        let bytes = ((n_floats * 4 + 4095) & !4095) as u64;
        let buf = dev.alloc.alloc_vram(bytes).ok()?;
        let ptr = buf.cpu_ptr as *mut f32;
        let va = buf.va_addr;
        Some(VramBuf { buf, ptr, va })
    }

    /// Ensure args slab is allocated.
    fn ensure_args(&mut self, dev: &HsaDevice) -> bool {
        let needed = MAX_BATCH * ARGS_SLOT;
        if self.args_slab_cap < needed {
            self.args_slab = dev.alloc.alloc_vram(needed as u64).ok();
            self.args_slab_cap = if self.args_slab.is_some() { needed } else { 0 };
        }
        self.args_slab.is_some()
    }

    /// Get the next kernarg slot VA and advance cursor.
    fn next_args_slot(&mut self) -> Option<(u64, usize)> {
        if self.args_cursor >= MAX_BATCH { return None; }
        let slab = self.args_slab.as_ref()?;
        let offset = self.args_cursor * ARGS_SLOT;
        self.args_cursor += 1;
        Some((slab.va_addr + offset as u64, offset))
    }

    /// Write kernargs to the current slot.
    fn write_args(&self, offset: usize, data: &[u8]) {
        if let Some(ref slab) = self.args_slab {
            slab.write(offset, data);
        }
    }

    /// Get or upload weight+bias to VRAM. Cached by pointer identity.
    fn prepare_weights(&mut self, dev: &HsaDevice,
                       weight: &[f32], bias: &[f32]) -> Option<(u64, u64)> {
        let key = (weight.as_ptr() as usize, weight.len());
        if !self.weight_cache.contains_key(&key) {
            let w_bytes = weight.len() * 4;
            let b_bytes = bias.len() * 4;
            let total = ((w_bytes + b_bytes + 4095) & !4095) as u64;
            let buf = dev.alloc.alloc_vram(total).ok()?;
            buf.write_f32(0, weight);
            buf.write_f32(w_bytes, bias);
            self.weight_cache.insert(key, WeightEntry { buf, w_bytes });
        }
        let e = self.weight_cache.get(&key)?;
        Some((e.buf.va_addr, e.buf.va_addr + e.w_bytes as u64))
    }

    /// Invalidate weight cache. Call after optimizer step.
    pub fn invalidate_weights(&mut self) {
        self.weight_cache.clear();
    }

    /// Number of dispatches queued since last flush.
    pub fn pending(&self) -> usize { self.pending }

    // ─── Dispatch methods (just queue, no sync) ──────────────

    /// Queue tiled matvec: y = W*x + b. Full GPU utilization via
    /// WG-cooperative dot product. Falls back to naive matvec if
    /// tiled kernel not loaded.
    pub fn enqueue_matvec_tiled(&mut self, dev: &mut HsaDevice,
                                weight: &[f32], bias: &[f32],
                                x: &VramBuf, y: &VramBuf,
                                out_dim: usize, in_dim: usize) -> bool {
        if !self.ensure_args(dev) { return false; }
        let (w_va, b_va) = match self.prepare_weights(dev, weight, bias) {
            Some(v) => v, None => return false,
        };
        let (args_va, args_off) = match self.next_args_slot() {
            Some(v) => v, None => return false,
        };

        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&b_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&x.va().to_le_bytes());
        kargs[24..32].copy_from_slice(&y.va().to_le_bytes());
        kargs[32..36].copy_from_slice(&(out_dim as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&(in_dim as u32).to_le_bytes());
        self.write_args(args_off, &kargs[..40]);

        // Tiled: one WG per output row, 256 threads per WG
        let ok = dev.dispatch_enqueue_va("matvec_tiled", args_va,
            [out_dim as u32, 1, 1], [256, 1, 1]);
        if ok { self.pending += 1; }
        ok
    }

    /// Queue fused affine LayerNorm + SiLU. Single WG, n ≤ 1024.
    pub fn enqueue_ln_silu(&mut self, dev: &mut HsaDevice,
                           x: &VramBuf, gamma: &VramBuf, beta: &VramBuf,
                           n: usize) -> bool {
        if n > 1024 { return false; }
        if !self.ensure_args(dev) { return false; }
        let (args_va, args_off) = match self.next_args_slot() {
            Some(v) => v, None => return false,
        };
        let mut kargs = [0u8; 32];
        kargs[0..8].copy_from_slice(&x.va().to_le_bytes());
        kargs[8..16].copy_from_slice(&gamma.va().to_le_bytes());
        kargs[16..24].copy_from_slice(&beta.va().to_le_bytes());
        kargs[24..28].copy_from_slice(&(n as u32).to_le_bytes());
        self.write_args(args_off, &kargs[..28]);

        let ok = dev.dispatch_enqueue_va("ln_silu_fwd", args_va,
            [256, 1, 1], [256, 1, 1]);
        if ok { self.pending += 1; }
        ok
    }

    /// Queue Q4_K_M matvec: y = dequant(W_q4)*x + bias.
    /// Reads quantized weights directly — no dequantization step.
    /// W_q4 is the raw Q4_K_M data in VRAM, bias is f32.
    pub fn enqueue_matvec_q4k(&mut self, dev: &mut HsaDevice,
                               w_q4: &VramBuf, x: &VramBuf,
                               bias: &VramBuf, y: &VramBuf,
                               out_dim: usize, blocks_per_row: usize) -> bool {
        if !self.ensure_args(dev) { return false; }
        let (args_va, args_off) = match self.next_args_slot() {
            Some(v) => v, None => return false,
        };

        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&w_q4.va().to_le_bytes());
        kargs[8..16].copy_from_slice(&x.va().to_le_bytes());
        kargs[16..24].copy_from_slice(&bias.va().to_le_bytes());
        kargs[24..32].copy_from_slice(&y.va().to_le_bytes());
        kargs[32..36].copy_from_slice(&(out_dim as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&(blocks_per_row as u32).to_le_bytes());
        self.write_args(args_off, &kargs[..40]);

        // One WG per output row, 256 threads per WG
        let ok = dev.dispatch_enqueue_va("matvec_q4k", args_va,
            [out_dim as u32, 1, 1], [256, 1, 1]);
        if !ok {
            eprintln!("    matvec_q4k dispatch FAILED");
        }
        if ok { self.pending += 1; }
        ok
    }

    /// Queue matvec: y = W*x + b. All VAs are VRAM addresses.
    pub fn enqueue_matvec(&mut self, dev: &mut HsaDevice,
                          weight: &[f32], bias: &[f32],
                          x: &VramBuf, y: &VramBuf,
                          out_dim: usize, in_dim: usize) -> bool {
        if !self.ensure_args(dev) { return false; }
        let (w_va, b_va) = match self.prepare_weights(dev, weight, bias) {
            Some(v) => v, None => return false,
        };
        let (args_va, args_off) = match self.next_args_slot() {
            Some(v) => v, None => return false,
        };

        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&b_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&x.va().to_le_bytes());
        kargs[24..32].copy_from_slice(&y.va().to_le_bytes());
        kargs[32..36].copy_from_slice(&(out_dim as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&(in_dim as u32).to_le_bytes());
        self.write_args(args_off, &kargs[..40]);

        // Create a temporary GpuBuffer view for dispatch_enqueue
        let args_ref = self.args_slab.as_ref().unwrap();
        let fake_args = FakeArgsRef { va: args_va, _parent: args_ref };

        let nwg = (out_dim as u32 + 255) / 256;
        let ok = dev.dispatch_enqueue_va("matvec", args_va, [nwg, 1, 1], [256, 1, 1]);
        if ok { self.pending += 1; }
        ok
    }

    /// Queue superlinear: batched per-neuron matvec.
    pub fn enqueue_superlinear(&mut self, dev: &mut HsaDevice,
                               weights: &[f32], biases: &[f32],
                               trace: &VramBuf, y: &VramBuf,
                               n_neurons: usize, out_per: usize, in_per: usize) -> bool {
        if !self.ensure_args(dev) { return false; }
        let (w_va, b_va) = match self.prepare_weights(dev, weights, biases) {
            Some(v) => v, None => return false,
        };
        let (args_va, args_off) = match self.next_args_slot() {
            Some(v) => v, None => return false,
        };

        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&b_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&trace.va().to_le_bytes());
        kargs[24..32].copy_from_slice(&y.va().to_le_bytes());
        kargs[32..36].copy_from_slice(&(n_neurons as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&(out_per as u32).to_le_bytes());
        kargs[40..44].copy_from_slice(&(in_per as u32).to_le_bytes());
        self.write_args(args_off, &kargs[..44]);

        let total = (n_neurons * out_per) as u32;
        let nwg = (total + 255) / 256;
        let ok = dev.dispatch_enqueue_va("superlinear_fwd", args_va, [nwg, 1, 1], [256, 1, 1]);
        if ok { self.pending += 1; }
        ok
    }

    /// Queue GLU: output[i] = input[i] * sigmoid(input[n+i]).
    pub fn enqueue_glu(&mut self, dev: &mut HsaDevice,
                       input: &VramBuf, output: &VramBuf, n: usize) -> bool {
        if !self.ensure_args(dev) { return false; }
        let (args_va, args_off) = match self.next_args_slot() {
            Some(v) => v, None => return false,
        };
        let mut kargs = [0u8; 24];
        kargs[0..8].copy_from_slice(&input.va().to_le_bytes());
        kargs[8..16].copy_from_slice(&output.va().to_le_bytes());
        kargs[16..20].copy_from_slice(&(n as u32).to_le_bytes());
        self.write_args(args_off, &kargs[..20]);

        let nwg = (n as u32 + 255) / 256;
        let ok = dev.dispatch_enqueue_va("glu_fwd", args_va, [nwg, 1, 1], [256, 1, 1]);
        if ok { self.pending += 1; }
        ok
    }

    /// Queue SiLU in-place.
    pub fn enqueue_silu(&mut self, dev: &mut HsaDevice,
                        x: &VramBuf, n: usize) -> bool {
        if !self.ensure_args(dev) { return false; }
        let (args_va, args_off) = match self.next_args_slot() {
            Some(v) => v, None => return false,
        };
        let mut kargs = [0u8; 16];
        kargs[0..8].copy_from_slice(&x.va().to_le_bytes());
        kargs[8..12].copy_from_slice(&(n as u32).to_le_bytes());
        self.write_args(args_off, &kargs[..12]);

        let nwg = (n as u32 + 255) / 256;
        let ok = dev.dispatch_enqueue_va("silu_fwd", args_va, [nwg, 1, 1], [256, 1, 1]);
        if ok { self.pending += 1; }
        ok
    }

    /// Queue layer norm in-place (single WG, n ≤ 1024).
    pub fn enqueue_layer_norm(&mut self, dev: &mut HsaDevice,
                              x: &VramBuf, n: usize) -> bool {
        if n > 1024 { return false; }
        if !self.ensure_args(dev) { return false; }
        let (args_va, args_off) = match self.next_args_slot() {
            Some(v) => v, None => return false,
        };
        let mut kargs = [0u8; 16];
        kargs[0..8].copy_from_slice(&x.va().to_le_bytes());
        kargs[8..12].copy_from_slice(&(n as u32).to_le_bytes());
        self.write_args(args_off, &kargs[..12]);

        let ok = dev.dispatch_enqueue_va("layer_norm_fwd", args_va, [256, 1, 1], [256, 1, 1]);
        if ok { self.pending += 1; }
        ok
    }

    /// Queue trace shift.
    pub fn enqueue_trace_shift(&mut self, dev: &mut HsaDevice,
                               traces: &VramBuf, new_act: &VramBuf,
                               n_neurons: usize, memory_length: usize) -> bool {
        if !self.ensure_args(dev) { return false; }
        let (args_va, args_off) = match self.next_args_slot() {
            Some(v) => v, None => return false,
        };
        let mut kargs = [0u8; 24];
        kargs[0..8].copy_from_slice(&traces.va().to_le_bytes());
        kargs[8..16].copy_from_slice(&new_act.va().to_le_bytes());
        kargs[16..20].copy_from_slice(&(n_neurons as u32).to_le_bytes());
        kargs[20..24].copy_from_slice(&(memory_length as u32).to_le_bytes());
        self.write_args(args_off, &kargs[..24]);

        let nwg = (n_neurons as u32 + 255) / 256;
        let ok = dev.dispatch_enqueue_va("trace_shift_fwd", args_va, [nwg, 1, 1], [256, 1, 1]);
        if ok { self.pending += 1; }
        ok
    }

    /// Queue sync update.
    pub fn enqueue_sync_update(&mut self, dev: &mut HsaDevice,
                               alpha: &VramBuf, beta: &VramBuf,
                               act_left: &VramBuf, act_right: &VramBuf,
                               phases_left: Option<&VramBuf>,
                               phases_right: Option<&VramBuf>,
                               decay: &VramBuf, decay_shift: &VramBuf,
                               sync_out: &VramBuf,
                               dopamine: f32, n_pairs: usize,
                               initialized: bool) -> bool {
        if !self.ensure_args(dev) { return false; }
        let (args_va, args_off) = match self.next_args_slot() {
            Some(v) => v, None => return false,
        };

        let has_phase = phases_left.is_some() && phases_right.is_some();
        let pl_va = phases_left.map(|p| p.va()).unwrap_or(0);
        let pr_va = phases_right.map(|p| p.va()).unwrap_or(0);

        let mut kargs = [0u8; 88];
        kargs[0..8].copy_from_slice(&alpha.va().to_le_bytes());
        kargs[8..16].copy_from_slice(&beta.va().to_le_bytes());
        kargs[16..24].copy_from_slice(&act_left.va().to_le_bytes());
        kargs[24..32].copy_from_slice(&act_right.va().to_le_bytes());
        kargs[32..40].copy_from_slice(&pl_va.to_le_bytes());
        kargs[40..48].copy_from_slice(&pr_va.to_le_bytes());
        kargs[48..56].copy_from_slice(&decay.va().to_le_bytes());
        kargs[56..64].copy_from_slice(&decay_shift.va().to_le_bytes());
        kargs[64..68].copy_from_slice(&dopamine.to_le_bytes());
        kargs[68..72].copy_from_slice(&(n_pairs as u32).to_le_bytes());
        kargs[72..76].copy_from_slice(&(initialized as u32).to_le_bytes());
        kargs[76..80].copy_from_slice(&(has_phase as u32).to_le_bytes());
        kargs[80..88].copy_from_slice(&sync_out.va().to_le_bytes());
        self.write_args(args_off, &kargs);

        let nwg = (n_pairs as u32 + 255) / 256;
        let ok = dev.dispatch_enqueue_va("sync_update_fwd", args_va, [nwg, 1, 1], [256, 1, 1]);
        if ok { self.pending += 1; }
        ok
    }

    /// Submit all queued dispatches and wait for completion.
    /// Flushes GPU L2 cache so CPU can read results through BAR.
    /// Resets the args cursor for the next batch.
    pub fn flush(&mut self, dev: &mut HsaDevice) -> bool {
        if self.pending == 0 { return true; }
        // Cache writeback before signal so CPU sees GPU's writes
        dev.queue.cache_wb();
        let ok = dev.submit_wait(5000);
        self.args_cursor = 0;
        self.pending = 0;
        ok
    }
}

// Temporary: need dispatch_enqueue_va on HsaDevice that takes a raw VA
// instead of a &GpuBuffer reference.
struct FakeArgsRef<'a> {
    va: u64,
    _parent: &'a GpuBuffer,
}
