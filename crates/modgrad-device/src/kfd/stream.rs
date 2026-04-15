//! GPU compute engine with three modes:
//!
//! **Stream mode** (--gpu): Upload weights per-call through ping-pong buffers.
//!   Best for PCIe x16 where transfer overlaps compute.
//!
//! **VRAM mode** (--vram): Upload weights once, keep in VRAM permanently.
//!   Best for PCIe x4 where transfer dominates. Re-upload only after
//!   optimizer step (one bulk transfer vs thousands of per-op transfers).
//!
//! **Full GPU mode**: All operations dispatched to GPU with VRAM scratch
//!   buffers. Data stays in VRAM between ops — zero PCIe round-trips in
//!   the inner loop. Chained dispatch: matvec→GLU→SiLU→LayerNorm as
//!   sequential GPU kernels with a single submit_wait.

use super::memory::GpuBuffer;
use super::HsaDevice;
use std::collections::HashMap;

/// Cached weight entry in VRAM.
struct VramEntry {
    buf: GpuBuffer,       // W + bias contiguous
    w_bytes: usize,       // offset where bias starts
}

/// GPU compute engine.
pub struct StreamEngine {
    /// VRAM mode: cached weights keyed by (ptr, len).
    /// Empty in stream mode — weights upload per-call.
    vram_cache: HashMap<(usize, usize), VramEntry>,
    /// Whether to cache weights in VRAM (true) or stream per-call (false).
    pub vram_mode: bool,
    /// Two weight buffers for stream mode (A/B ping-pong).
    w_buf: [Option<GpuBuffer>; 2],
    w_cap: [usize; 2],
    active: usize,
    /// Shared input/output/args buffers.
    x_buf: Option<GpuBuffer>,
    x_cap: usize,
    y_buf: Option<GpuBuffer>,
    y_cap: usize,
    args_buf: Option<GpuBuffer>,

    // ─── Chained dispatch scratch buffers ───
    // These stay in VRAM between kernel dispatches so data never
    // round-trips through PCIe in the inner loop.

    /// Scratch A: intermediate VRAM buffer (e.g., matvec output for synapse chain)
    scratch_a: Option<GpuBuffer>,
    scratch_a_cap: usize,
    /// Scratch B: second intermediate (e.g., GLU output → SiLU → LayerNorm)
    scratch_b: Option<GpuBuffer>,
    scratch_b_cap: usize,
    /// Per-kernel args buffers for chained dispatch (avoids overwriting
    /// args while previous kernel is still reading them).
    chain_args: [Option<GpuBuffer>; 8],
}

impl StreamEngine {
    pub fn new() -> Self {
        Self {
            vram_cache: HashMap::new(),
            vram_mode: false,
            w_buf: [None, None],
            w_cap: [0, 0],
            active: 0,
            x_buf: None, x_cap: 0,
            y_buf: None, y_cap: 0,
            args_buf: None,
            scratch_a: None, scratch_a_cap: 0,
            scratch_b: None, scratch_b_cap: 0,
            chain_args: [None, None, None, None, None, None, None, None],
        }
    }

    /// Invalidate VRAM cache. Call after optimizer step updates CPU weights.
    pub fn invalidate(&mut self) {
        self.vram_cache.clear();
    }

    /// Ensure a weight buffer is large enough.
    fn ensure_w(&mut self, idx: usize, bytes: usize, dev: &HsaDevice) -> bool {
        if self.w_cap[idx] < bytes {
            let cap = ((bytes + 4095) & !4095) as u64;
            self.w_buf[idx] = dev.alloc.alloc_vram(cap).ok();
            self.w_cap[idx] = cap as usize;
        }
        self.w_buf[idx].is_some()
    }

    fn ensure_x(&mut self, bytes: usize, dev: &HsaDevice) -> bool {
        if self.x_cap < bytes {
            let cap = ((bytes + 4095) & !4095).max(4096) as u64;
            self.x_buf = dev.alloc.alloc_vram(cap).ok();
            self.x_cap = cap as usize;
        }
        self.x_buf.is_some()
    }

    fn ensure_y(&mut self, bytes: usize, dev: &mut HsaDevice) -> bool {
        if self.y_cap < bytes {
            let cap = ((bytes + 4095) & !4095).max(4096) as u64;
            self.y_buf = dev.alloc_output(cap as usize).ok();
            self.y_cap = cap as usize;
        }
        self.y_buf.is_some()
    }

    fn ensure_args(&mut self, dev: &HsaDevice) -> bool {
        if self.args_buf.is_none() {
            self.args_buf = dev.alloc.alloc_vram(4096).ok();
        }
        self.args_buf.is_some()
    }

    fn ensure_scratch_a(&mut self, bytes: usize, dev: &HsaDevice) -> bool {
        if self.scratch_a_cap < bytes {
            let cap = ((bytes + 4095) & !4095).max(4096) as u64;
            self.scratch_a = dev.alloc.alloc_vram(cap).ok();
            self.scratch_a_cap = cap as usize;
        }
        self.scratch_a.is_some()
    }

    fn ensure_scratch_b(&mut self, bytes: usize, dev: &HsaDevice) -> bool {
        if self.scratch_b_cap < bytes {
            let cap = ((bytes + 4095) & !4095).max(4096) as u64;
            self.scratch_b = dev.alloc.alloc_vram(cap).ok();
            self.scratch_b_cap = cap as usize;
        }
        self.scratch_b.is_some()
    }

    /// Ensure chain_args[idx] exists (one per kernel in a chain).
    fn ensure_chain_args(&mut self, idx: usize, dev: &HsaDevice) -> bool {
        if self.chain_args[idx].is_none() {
            // 256 bytes per args buffer — enough for any kernel's args
            self.chain_args[idx] = dev.alloc.alloc_vram(4096).ok();
        }
        self.chain_args[idx].is_some()
    }

    /// Get or upload weight+bias buffer. In VRAM mode, caches permanently.
    /// In stream mode, uploads to ping-pong buffer each call.
    /// Returns (w_va, bias_va).
    fn prepare_weights(
        &mut self, dev: &HsaDevice,
        weight_data: &[f32], bias_data: &[f32],
    ) -> Option<(u64, u64)> {
        let w_bytes = weight_data.len() * 4;
        let b_bytes = bias_data.len() * 4;

        if self.vram_mode {
            // VRAM mode: cache by pointer identity, upload once
            let key = (weight_data.as_ptr() as usize, weight_data.len());
            if !self.vram_cache.contains_key(&key) {
                let total = w_bytes + b_bytes;
                let cap = ((total + 4095) & !4095) as u64;
                let buf = dev.alloc.alloc_vram(cap).ok()?;
                buf.write_f32(0, weight_data);
                buf.write_f32(w_bytes, bias_data);
                self.vram_cache.insert(key, VramEntry { buf, w_bytes });
            }
            let entry = self.vram_cache.get(&key)?;
            Some((entry.buf.va_addr, entry.buf.va_addr + entry.w_bytes as u64))
        } else {
            // Stream mode: upload to ping-pong buffer
            let idx = self.active;
            self.ensure_w(idx, w_bytes + b_bytes, dev);
            let wbuf = self.w_buf[idx].as_ref()?;
            wbuf.write_f32(0, weight_data);
            wbuf.write_f32(w_bytes, bias_data);
            Some((wbuf.va_addr, wbuf.va_addr + w_bytes as u64))
        }
    }

    /// Dispatch a kernel with weight/bias/input/output.
    ///
    /// Key change from earlier: we queue the dispatch and submit+wait
    /// in the SAME call, but each dispatch gets its own kernargs buffer
    /// to avoid overwriting args while the GPU is still reading them.
    /// The submit_wait ensures the GPU finishes BEFORE we read back.
    fn dispatch_kernel(
        &mut self, dev: &mut HsaDevice,
        kernel: &str,
        weight_data: &[f32], bias_data: &[f32],
        x_data: &[f32],
        extra_args: &[u32],
        nwg: u32,
        out_count: usize,
    ) -> Option<Vec<f32>> {
        let (w_va, bias_va) = self.prepare_weights(dev, weight_data, bias_data)?;

        self.ensure_x(x_data.len() * 4, dev);
        self.ensure_y(out_count * 4, dev);
        self.ensure_args(dev);

        // Upload input
        let xbuf = self.x_buf.as_ref()?;
        xbuf.write_f32(0, x_data);

        // Zero output
        let ybuf = self.y_buf.as_ref()?;
        unsafe { std::ptr::write_bytes(ybuf.cpu_ptr, 0, out_count * 4); }

        // Build kernargs
        let args = self.args_buf.as_ref()?;
        let mut kargs = [0u8; 64];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&bias_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&xbuf.va_addr.to_le_bytes());
        kargs[24..32].copy_from_slice(&ybuf.va_addr.to_le_bytes());
        for (i, &v) in extra_args.iter().enumerate() {
            let off = 32 + i * 4;
            kargs[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        args.write(0, &kargs[..32 + extra_args.len() * 4]);

        // Ensure the ring buffer isn't too full before submitting
        // (the GPU needs time to process previous dispatches)
        let ring_usage = dev.queue.put;
        let ring_cap = dev.queue.ring.size as u64 / 4;
        if ring_usage > ring_cap / 2 {
            // Ring is getting full — sync to let GPU drain it
            dev.submit_wait(5000);
            // Reset ring position (GPU has processed everything)
            // Actually, put keeps incrementing and wraps with modulo.
            // The check_ring_space handles this. But if the read_ptr
            // hasn't advanced, we need to wait.
        }

        // Queue dispatch
        if !dev.dispatch_enqueue(kernel, args, [nwg, 1, 1], [256, 1, 1]) {
            return None;
        }

        if !dev.submit_wait(5000) {
            return None;
        }

        // Read back
        let y_slice = unsafe {
            std::slice::from_raw_parts(ybuf.cpu_ptr as *const f32, out_count)
        };
        let mut y = vec![0.0f32; out_count];
        y.copy_from_slice(y_slice);

        if !self.vram_mode { self.active = 1 - self.active; }
        Some(y)
    }

    /// y = W @ x + b (tiled: 1 WG per row, 256 threads cooperate via LDS)
    pub fn matvec(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32], bias: &[f32], x: &[f32],
        out_dim: usize, in_dim: usize,
    ) -> Option<Vec<f32>> {
        self.dispatch_kernel(dev, "matvec_tiled", weight, bias, x,
            &[out_dim as u32, in_dim as u32], out_dim as u32, out_dim)
    }

    /// y = W @ x + b, writing directly to caller's output slice.
    /// Avoids the Vec allocation + double copy of dispatch_kernel → try_matvec.
    pub fn matvec_into(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32], bias: &[f32], x: &[f32],
        out: &mut [f32], out_dim: usize, in_dim: usize,
    ) -> bool {
        let (w_va, bias_va) = match self.prepare_weights(dev, weight, bias) {
            Some(v) => v, None => return false,
        };

        if !self.ensure_x(x.len() * 4, dev) { return false; }
        if !self.ensure_y(out_dim * 4, dev) { return false; }
        if !self.ensure_args(dev) { return false; }

        // Upload input
        self.x_buf.as_ref().unwrap().write_f32(0, x);

        // Build kernargs
        let xva = self.x_buf.as_ref().unwrap().va_addr;
        let yva = self.y_buf.as_ref().unwrap().va_addr;
        let args = self.args_buf.as_ref().unwrap();
        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&bias_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&xva.to_le_bytes());
        kargs[24..32].copy_from_slice(&yva.to_le_bytes());
        kargs[32..36].copy_from_slice(&(out_dim as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&(in_dim as u32).to_le_bytes());
        args.write(0, &kargs[..40]);

        // Dispatch + wait
        if !dev.dispatch_enqueue("matvec_tiled", args, [out_dim as u32, 1, 1], [256, 1, 1]) {
            return false;
        }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        // Read back directly into caller's slice (single copy)
        let src = unsafe {
            std::slice::from_raw_parts(self.y_buf.as_ref().unwrap().cpu_ptr as *const f32, out_dim)
        };
        out[..out_dim].copy_from_slice(src);

        if !self.vram_mode { self.active = 1 - self.active; }
        true
    }

    // ═══════════════════════════════════════════════════════════════
    // Zero-copy VRAM dispatch — no upload, no download, no PCIe
    // ═══════════════════════════════════════════════════════════════
    //
    // These methods take GPU VAs directly. Data must already be in VRAM
    // (via arena allocation). Used when both input and output slices
    // are backed by the VRAM arena.

    /// Zero-copy matvec: y = W*x + b, all pointers are VRAM VAs.
    /// Weights are prepared via normal VRAM cache. x_va/y_va point
    /// into the arena. Dispatches and waits.
    pub fn matvec_zerocopy(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32], bias: &[f32],
        x_va: u64, y_va: u64,
        out_dim: usize, in_dim: usize,
    ) -> bool {
        let (w_va, bias_va) = match self.prepare_weights(dev, weight, bias) {
            Some(v) => v, None => return false,
        };
        if !self.ensure_chain_args(0, dev) { return false; }

        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&bias_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&x_va.to_le_bytes());
        kargs[24..32].copy_from_slice(&y_va.to_le_bytes());
        kargs[32..36].copy_from_slice(&(out_dim as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&(in_dim as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..40]);

        let nwg = (out_dim as u32 + 255) / 256;
        if !dev.dispatch_enqueue("matvec", args, [nwg, 1, 1], [256, 1, 1]) {
            return false;
        }
        dev.submit_wait(5000)
    }

    /// Zero-copy superlinear: all pointers are VRAM VAs.
    pub fn superlinear_zerocopy(
        &mut self, dev: &mut HsaDevice,
        weights: &[f32], biases: &[f32],
        trace_va: u64, y_va: u64,
        n_neurons: usize, out_per: usize, in_per: usize,
    ) -> bool {
        let (w_va, bias_va) = match self.prepare_weights(dev, weights, biases) {
            Some(v) => v, None => return false,
        };
        if !self.ensure_chain_args(0, dev) { return false; }

        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&bias_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&trace_va.to_le_bytes());
        kargs[24..32].copy_from_slice(&y_va.to_le_bytes());
        kargs[32..36].copy_from_slice(&(n_neurons as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&(out_per as u32).to_le_bytes());
        kargs[40..44].copy_from_slice(&(in_per as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..44]);

        let total = (n_neurons * out_per) as u32;
        let nwg = (total + 255) / 256;
        if !dev.dispatch_enqueue("superlinear_fwd", args, [nwg, 1, 1], [256, 1, 1]) {
            return false;
        }
        dev.submit_wait(5000)
    }

    /// Zero-copy GLU: input_va → output_va, both in VRAM.
    pub fn glu_zerocopy(
        &mut self, dev: &mut HsaDevice,
        input_va: u64, output_va: u64, n: usize,
    ) -> bool {
        if !self.ensure_chain_args(0, dev) { return false; }
        let mut kargs = [0u8; 24];
        kargs[0..8].copy_from_slice(&input_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&output_va.to_le_bytes());
        kargs[16..20].copy_from_slice(&(n as u32).to_le_bytes());
        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..20]);
        let nwg = (n as u32 + 255) / 256;
        if !dev.dispatch_enqueue("glu_fwd", args, [nwg, 1, 1], [256, 1, 1]) { return false; }
        dev.submit_wait(5000)
    }

    /// Zero-copy SiLU in-place on VRAM buffer.
    pub fn silu_zerocopy(&mut self, dev: &mut HsaDevice, x_va: u64, n: usize) -> bool {
        if !self.ensure_chain_args(0, dev) { return false; }
        let mut kargs = [0u8; 16];
        kargs[0..8].copy_from_slice(&x_va.to_le_bytes());
        kargs[8..12].copy_from_slice(&(n as u32).to_le_bytes());
        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..12]);
        let nwg = (n as u32 + 255) / 256;
        if !dev.dispatch_enqueue("silu_fwd", args, [nwg, 1, 1], [256, 1, 1]) { return false; }
        dev.submit_wait(5000)
    }

    /// Zero-copy LayerNorm in-place on VRAM buffer.
    pub fn layer_norm_zerocopy(&mut self, dev: &mut HsaDevice, x_va: u64, n: usize) -> bool {
        if n > 1024 { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }
        let mut kargs = [0u8; 16];
        kargs[0..8].copy_from_slice(&x_va.to_le_bytes());
        kargs[8..12].copy_from_slice(&(n as u32).to_le_bytes());
        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..12]);
        if !dev.dispatch_enqueue("layer_norm_fwd", args, [256, 1, 1], [256, 1, 1]) { return false; }
        dev.submit_wait(5000)
    }

    /// dx = W^T @ d_out (backward input gradient)
    pub fn matvec_t(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32], d_out: &[f32],
        out_dim: usize, in_dim: usize,
    ) -> Option<Vec<f32>> {
        let mut w_t = vec![0.0f32; in_dim * out_dim];
        for i in 0..out_dim {
            for j in 0..in_dim {
                w_t[j * out_dim + i] = weight[i * in_dim + j];
            }
        }
        let bias_zero = vec![0.0f32; in_dim];
        self.matvec(dev, &w_t, &bias_zero, d_out, in_dim, out_dim)
    }

    /// SuperLinear: batched per-neuron matvec.
    pub fn superlinear(
        &mut self, dev: &mut HsaDevice,
        weights: &[f32], biases: &[f32], trace: &[f32],
        n_neurons: usize, out_per: usize, in_per: usize,
    ) -> Option<Vec<f32>> {
        let total = (n_neurons * out_per) as u32;
        let nwg = (total + 255) / 256;
        self.dispatch_kernel(dev, "superlinear_fwd", weights, biases, trace,
            &[n_neurons as u32, out_per as u32, in_per as u32],
            nwg, n_neurons * out_per)
    }

    // ═══════════════════════════════════════════════════════════════
    // Chained GPU dispatch — data stays in VRAM between kernels
    // ═══════════════════════════════════════════════════════════════

    /// Write kernargs to chain_args[idx] and dispatch kernel.
    /// Returns true on success.
    fn chain_dispatch(
        &mut self, dev: &mut HsaDevice,
        idx: usize,
        kernel: &str,
        kargs: &[u8],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> bool {
        if !self.ensure_chain_args(idx, dev) { return false; }
        let args = self.chain_args[idx].as_ref().unwrap();
        args.write(0, kargs);
        dev.dispatch_enqueue(kernel, args, grid, block)
    }

    /// Fused synapse forward: matvec → GLU → SiLU → LayerNorm.
    /// All 4 kernels dispatch sequentially on GPU. Data stays in VRAM.
    /// Only ONE PCIe round-trip: upload x, download final result.
    ///
    /// Weight matrix is [2*out_dim x in_dim] (GLU doubles the output).
    /// Bias is [2*out_dim].
    /// Input x is [in_dim].
    /// Output is [out_dim] after GLU halving.
    pub fn synapse_forward(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32], bias: &[f32], x: &[f32],
        out_dim: usize, in_dim: usize,
    ) -> Option<Vec<f32>> {
        let matvec_out = out_dim * 2; // pre-GLU size

        // Prepare weights (upload or VRAM cache)
        let (w_va, bias_va) = self.prepare_weights(dev, weight, bias)?;

        // Ensure buffers:
        // x_buf: input [in_dim]
        // scratch_a: matvec output [2*out_dim] → GLU reads from here
        // scratch_b: GLU output [out_dim] → SiLU in-place → LayerNorm in-place → final output
        self.ensure_x(in_dim * 4, dev);
        self.ensure_scratch_a(matvec_out * 4, dev);
        self.ensure_scratch_b(out_dim * 4, dev);

        // Upload input x
        let x_va = self.x_buf.as_ref()?.va_addr;
        self.x_buf.as_ref()?.write_f32(0, x);

        let sa_va = self.scratch_a.as_ref()?.va_addr;
        let sb_va = self.scratch_b.as_ref()?.va_addr;

        // ─── Kernel 1: matvec → scratch_a ───
        // Args: W(ptr), b(ptr), x(ptr), y(ptr), out_dim, in_dim
        {
            let mut kargs = [0u8; 48];
            kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
            kargs[8..16].copy_from_slice(&bias_va.to_le_bytes());
            kargs[16..24].copy_from_slice(&x_va.to_le_bytes());
            kargs[24..32].copy_from_slice(&sa_va.to_le_bytes());
            kargs[32..36].copy_from_slice(&(matvec_out as u32).to_le_bytes());
            kargs[36..40].copy_from_slice(&(in_dim as u32).to_le_bytes());
            let nwg = (matvec_out as u32 + 255) / 256;
            if !self.chain_dispatch(dev, 0, "matvec", &kargs[..40], [nwg, 1, 1], [256, 1, 1]) {
                return None;
            }
        }

        // ─── Kernel 2: GLU scratch_a → scratch_b ───
        // Args: input(ptr), output(ptr), n
        {
            let mut kargs = [0u8; 24];
            kargs[0..8].copy_from_slice(&sa_va.to_le_bytes());
            kargs[8..16].copy_from_slice(&sb_va.to_le_bytes());
            kargs[16..20].copy_from_slice(&(out_dim as u32).to_le_bytes());
            let nwg = (out_dim as u32 + 255) / 256;
            if !self.chain_dispatch(dev, 1, "glu_fwd", &kargs[..20], [nwg, 1, 1], [256, 1, 1]) {
                return None;
            }
        }

        // ─── Kernel 3: SiLU in-place on scratch_b ───
        // Args: x(ptr), n
        {
            let mut kargs = [0u8; 16];
            kargs[0..8].copy_from_slice(&sb_va.to_le_bytes());
            kargs[8..12].copy_from_slice(&(out_dim as u32).to_le_bytes());
            let nwg = (out_dim as u32 + 255) / 256;
            if !self.chain_dispatch(dev, 2, "silu_fwd", &kargs[..12], [nwg, 1, 1], [256, 1, 1]) {
                return None;
            }
        }

        // ─── Kernel 4: LayerNorm in-place on scratch_b ───
        // Args: x(ptr), n
        // Single workgroup, 256 threads, uses LDS
        {
            let mut kargs = [0u8; 16];
            kargs[0..8].copy_from_slice(&sb_va.to_le_bytes());
            kargs[8..12].copy_from_slice(&(out_dim as u32).to_le_bytes());
            // Single WG for reduction
            if !self.chain_dispatch(dev, 3, "layer_norm_fwd", &kargs[..12], [256, 1, 1], [256, 1, 1]) {
                return None;
            }
        }

        // Single submit+wait for all 4 kernels
        if !dev.submit_wait(5000) {
            return None;
        }

        // Read back final result from scratch_b
        let result_slice = unsafe {
            std::slice::from_raw_parts(self.scratch_b.as_ref()?.cpu_ptr as *const f32, out_dim)
        };
        let mut result = vec![0.0f32; out_dim];
        result.copy_from_slice(result_slice);

        if !self.vram_mode { self.active = 1 - self.active; }
        Some(result)
    }

    /// GPU trace shift: for each neuron, shift trace left, append new activation.
    /// Upload traces + new_activations, run kernel, download updated traces.
    pub fn trace_shift(
        &mut self, dev: &mut HsaDevice,
        traces: &mut [f32], new_activations: &[f32],
        n_neurons: usize, memory_length: usize,
    ) -> bool {
        let trace_bytes = n_neurons * memory_length * 4;
        let act_bytes = n_neurons * 4;

        // Upload traces to scratch_a, new_activations to scratch_b
        if !self.ensure_scratch_a(trace_bytes, dev) { return false; }
        if !self.ensure_scratch_b(act_bytes, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }

        let sa = self.scratch_a.as_ref().unwrap();
        sa.write_f32(0, &traces[..n_neurons * memory_length]);
        let sb = self.scratch_b.as_ref().unwrap();
        sb.write_f32(0, &new_activations[..n_neurons]);

        let sa_va = sa.va_addr;
        let sb_va = sb.va_addr;

        // Args: traces_ptr, new_act_ptr, n_neurons, memory_length
        let mut kargs = [0u8; 24];
        kargs[0..8].copy_from_slice(&sa_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&sb_va.to_le_bytes());
        kargs[16..20].copy_from_slice(&(n_neurons as u32).to_le_bytes());
        kargs[20..24].copy_from_slice(&(memory_length as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs);

        let nwg = (n_neurons as u32 + 255) / 256;
        if !dev.dispatch_enqueue("trace_shift_fwd", args, [nwg, 1, 1], [256, 1, 1]) {
            return false;
        }
        if !dev.submit_wait(5000) { return false; }

        // Read back updated traces
        let result = unsafe {
            std::slice::from_raw_parts(sa.cpu_ptr as *const f32, n_neurons * memory_length)
        };
        traces[..n_neurons * memory_length].copy_from_slice(result);
        true
    }

    /// GPU sync update: phase-aware Hebbian binding.
    /// Uploads all arrays, runs kernel, downloads alpha/beta/sync_out.
    pub fn sync_update(
        &mut self, dev: &mut HsaDevice,
        alpha: &mut [f32], beta: &mut [f32],
        act_left: &[f32], act_right: &[f32],
        phases_left: &[f32], phases_right: &[f32],
        decay: &[f32], decay_shift: &[f32],
        dopamine: f32, n_pairs: usize, initialized: bool,
        sync_out: &mut [f32],
    ) -> bool {
        // We need 9 VRAM buffers for the kernel args. Use a flat allocation.
        // Total data: 8 arrays of n_pairs floats + 1 output
        let pair_bytes = n_pairs * 4;
        let total_bytes = pair_bytes * 9; // alpha, beta, aL, aR, pL, pR, decay, decay_shift, sync_out

        if !self.ensure_scratch_a(total_bytes, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }

        let base = self.scratch_a.as_ref().unwrap();
        let base_va = base.va_addr;

        // Layout in scratch_a: [alpha | beta | aL | aR | pL | pR | decay | decay_shift | sync_out]
        let offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let vas: Vec<u64> = offsets.iter().map(|&i| base_va + (i * pair_bytes) as u64).collect();

        // Upload all arrays
        base.write_f32(0 * pair_bytes, &alpha[..n_pairs]);
        base.write_f32(1 * pair_bytes, &beta[..n_pairs]);
        base.write_f32(2 * pair_bytes, &act_left[..n_pairs]);
        base.write_f32(3 * pair_bytes, &act_right[..n_pairs]);

        let has_phase = !phases_left.is_empty() && !phases_right.is_empty();
        if has_phase {
            base.write_f32(4 * pair_bytes, &phases_left[..n_pairs]);
            base.write_f32(5 * pair_bytes, &phases_right[..n_pairs]);
        }
        base.write_f32(6 * pair_bytes, &decay[..n_pairs]);
        base.write_f32(7 * pair_bytes, &decay_shift[..n_pairs]);
        // sync_out area will be written by kernel

        // Build kernargs (88 bytes):
        // 8 pointers + dopamine(f32) + n_pairs(u32) + initialized(u32) + has_phase(u32) + sync_out(ptr)
        let mut kargs = [0u8; 88];
        for i in 0..8 {
            let off = i * 8;
            kargs[off..off + 8].copy_from_slice(&vas[i].to_le_bytes());
        }
        kargs[64..68].copy_from_slice(&dopamine.to_le_bytes());
        kargs[68..72].copy_from_slice(&(n_pairs as u32).to_le_bytes());
        kargs[72..76].copy_from_slice(&(initialized as u32).to_le_bytes());
        kargs[76..80].copy_from_slice(&(has_phase as u32).to_le_bytes());
        kargs[80..88].copy_from_slice(&vas[8].to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs);

        let nwg = (n_pairs as u32 + 255) / 256;
        if !dev.dispatch_enqueue("sync_update_fwd", args, [nwg, 1, 1], [256, 1, 1]) {
            return false;
        }
        if !dev.submit_wait(5000) { return false; }

        // Read back alpha, beta, sync_out
        let result = unsafe {
            std::slice::from_raw_parts(base.cpu_ptr as *const f32, 9 * n_pairs)
        };
        alpha[..n_pairs].copy_from_slice(&result[0..n_pairs]);
        beta[..n_pairs].copy_from_slice(&result[n_pairs..2 * n_pairs]);
        sync_out[..n_pairs].copy_from_slice(&result[8 * n_pairs..9 * n_pairs]);
        true
    }

    /// Standalone GLU dispatch (for non-chained use).
    pub fn glu(
        &mut self, dev: &mut HsaDevice,
        input: &[f32], out_dim: usize,
    ) -> Option<Vec<f32>> {
        let in_bytes = input.len() * 4;
        let out_bytes = out_dim * 4;

        self.ensure_scratch_a(in_bytes, dev);
        self.ensure_scratch_b(out_bytes, dev);
        self.ensure_chain_args(0, dev);

        let sa = self.scratch_a.as_ref()?;
        sa.write_f32(0, input);
        let sa_va = sa.va_addr;
        let sb_va = self.scratch_b.as_ref()?.va_addr;

        let mut kargs = [0u8; 24];
        kargs[0..8].copy_from_slice(&sa_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&sb_va.to_le_bytes());
        kargs[16..20].copy_from_slice(&(out_dim as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref()?;
        args.write(0, &kargs[..20]);

        let nwg = (out_dim as u32 + 255) / 256;
        if !dev.dispatch_enqueue("glu_fwd", args, [nwg, 1, 1], [256, 1, 1]) {
            return None;
        }
        if !dev.submit_wait(5000) { return None; }

        let result = unsafe {
            std::slice::from_raw_parts(self.scratch_b.as_ref()?.cpu_ptr as *const f32, out_dim)
        };
        let mut out = vec![0.0f32; out_dim];
        out.copy_from_slice(result);
        Some(out)
    }

    /// Standalone SiLU dispatch.
    pub fn silu_inplace(
        &mut self, dev: &mut HsaDevice,
        x: &mut [f32],
    ) -> bool {
        let n = x.len();
        let bytes = n * 4;

        if !self.ensure_scratch_a(bytes, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }

        let sa = self.scratch_a.as_ref().unwrap();
        sa.write_f32(0, x);
        let sa_va = sa.va_addr;

        let mut kargs = [0u8; 16];
        kargs[0..8].copy_from_slice(&sa_va.to_le_bytes());
        kargs[8..12].copy_from_slice(&(n as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..12]);

        let nwg = (n as u32 + 255) / 256;
        if !dev.dispatch_enqueue("silu_fwd", args, [nwg, 1, 1], [256, 1, 1]) {
            return false;
        }
        if !dev.submit_wait(5000) { return false; }

        let result = unsafe {
            std::slice::from_raw_parts(sa.cpu_ptr as *const f32, n)
        };
        x[..n].copy_from_slice(result);
        true
    }

    /// Standalone layer norm dispatch.
    pub fn layer_norm_inplace(
        &mut self, dev: &mut HsaDevice,
        x: &mut [f32],
    ) -> bool {
        let n = x.len();
        if n > 1024 { return false; } // kernel handles up to 1024

        let bytes = n * 4;
        if !self.ensure_scratch_a(bytes, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }

        let sa = self.scratch_a.as_ref().unwrap();
        sa.write_f32(0, x);
        let sa_va = sa.va_addr;

        let mut kargs = [0u8; 16];
        kargs[0..8].copy_from_slice(&sa_va.to_le_bytes());
        kargs[8..12].copy_from_slice(&(n as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..12]);

        // Single WG, 256 threads
        if !dev.dispatch_enqueue("layer_norm_fwd", args, [256, 1, 1], [256, 1, 1]) {
            return false;
        }
        if !dev.submit_wait(5000) { return false; }

        let result = unsafe {
            std::slice::from_raw_parts(sa.cpu_ptr as *const f32, n)
        };
        x[..n].copy_from_slice(result);
        true
    }
}
