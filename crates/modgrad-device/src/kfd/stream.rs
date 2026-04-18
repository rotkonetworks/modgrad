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
    /// Cached kernel entries for hot-path dispatch (no HashMap lookup).
    cached_matvec_tiled: Option<super::dispatch::KernelEntry>,
    cached_superlinear: Option<super::dispatch::KernelEntry>,
    cached_outer_product: Option<super::dispatch::KernelEntry>,
    cached_sgd_update: Option<super::dispatch::KernelEntry>,
    cached_matvec_t_tiled: Option<super::dispatch::KernelEntry>,
    cached_sl_bwd_dw: Option<super::dispatch::KernelEntry>,
    cached_sl_bwd_dx: Option<super::dispatch::KernelEntry>,
    cached_reduce_l2: Option<super::dispatch::KernelEntry>,
    cached_adamw: Option<super::dispatch::KernelEntry>,
    cached_matmul_blocked: Option<super::dispatch::KernelEntry>,

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
            cached_matvec_tiled: None,
            cached_matvec_t_tiled: None,
            cached_superlinear: None,
            cached_outer_product: None,
            cached_sgd_update: None,
            cached_sl_bwd_dw: None,
            cached_sl_bwd_dx: None,
            cached_reduce_l2: None,
            cached_adamw: None,
            cached_matmul_blocked: None,
        }
    }

    /// Resolve and cache kernel entries for hot-path dispatch.
    pub fn cache_kernels(&mut self, dev: &HsaDevice) {
        if self.cached_matvec_tiled.is_none() {
            self.cached_matvec_tiled = dev.resolve_kernel("matvec_tiled");
        }
        if self.cached_superlinear.is_none() {
            self.cached_superlinear = dev.resolve_kernel("superlinear_fwd");
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
        let ybuf = self.y_buf.as_ref()?;

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

        // Queue dispatch
        if !dev.dispatch_enqueue(kernel, args, [nwg, 1, 1], [256, 1, 1]) {
            eprintln!("GPU dispatch failed: kernel='{}'", kernel);
            return None;
        }

        // Flush GPU L2 → VRAM so CPU can read through BAR
        // Required for kernels using global_store (matvec_tiled, superlinear)
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) {
            let read_ptr = unsafe {
                (dev.queue.rw_ptrs.cpu_ptr.add(0x80) as *const u64).read_volatile()
            };
            eprintln!("GPU dispatch timeout: kernel='{}' grid=[{},1,1] put={} read={}",
                kernel, nwg, dev.queue.put, read_ptr);
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

    /// y = W @ x + b
    pub fn matvec(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32], bias: &[f32], x: &[f32],
        out_dim: usize, in_dim: usize,
    ) -> Option<Vec<f32>> {
        // Tiled kernel: 1 WG per row × 256 threads cooperate on dot product.
        // Verified correct at all dimensions via remu emulator.
        self.dispatch_kernel(dev, "matvec_tiled", weight, bias, x,
            &[out_dim as u32, in_dim as u32], out_dim as u32, out_dim)
    }

    /// y = W @ x + b, writing directly to caller's output slice.
    /// Optimized path: cached kernel lookup, single copy readback.
    pub fn matvec_into(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32], bias: &[f32], x: &[f32],
        out: &mut [f32], out_dim: usize, in_dim: usize,
    ) -> bool {
        let (w_va, bias_va) = match self.prepare_weights(dev, weight, bias) {
            Some(v) => v, None => return false,
        };

        if !self.ensure_x(in_dim * 4, dev) { return false; }
        if !self.ensure_y(out_dim * 4, dev) { return false; }
        if !self.ensure_args(dev) { return false; }

        // Upload input through BAR (~0.3us for 2KB at d_model=512)
        self.x_buf.as_ref().unwrap().write_f32(0, &x[..in_dim]);

        // Build kernargs (40 bytes — fast, no allocation)
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

        // Dispatch tiled kernel: 1 WG per row, 256 threads/WG
        // Cached entry avoids HashMap lookup on every call
        if self.cached_matvec_tiled.is_none() {
            self.cached_matvec_tiled = dev.resolve_kernel("matvec_tiled");
        }
        match &self.cached_matvec_tiled {
            Some(entry) => {
                dev.dispatch_enqueue_resolved(entry, args, [out_dim as u32, 1, 1], [256, 1, 1]);
            }
            None => return false,
        }

        // Cache writeback + signal + wait
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        // Read back through BAR directly into caller's slice (~0.3us for 2KB)
        let src = unsafe {
            std::slice::from_raw_parts(self.y_buf.as_ref().unwrap().cpu_ptr as *const f32, out_dim)
        };
        out[..out_dim].copy_from_slice(src);

        if !self.vram_mode { self.active = 1 - self.active; }
        true
    }

    /// Y[n×m] = X[n×k] @ W^T[k×m] + B[m] using the matmul_blocked kernel.
    /// W is row-major [m×k] (Linear stores weight as [out × in]). X is row-major
    /// [n×k], Y row-major [n×m]. Bias broadcast across rows.
    ///
    /// Precondition (caller MUST validate — violations hang gfx1102):
    ///   m % 128 == 0, k % 8 == 0, n % 32 == 0
    ///   weight.len() >= m*k, bias.len() >= m
    ///   x.len() >= n*k, out.len() >= n*m
    ///
    /// Returns false on any upload / dispatch / wait failure; leaves `out`
    /// untouched. Does NOT verify preconditions — the accel layer does that.
    pub fn matmul_into(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32], bias: &[f32], x: &[f32],
        out: &mut [f32], n: usize, k: usize, m: usize,
    ) -> bool {
        let (w_va, bias_va) = match self.prepare_weights(dev, weight, bias) {
            Some(v) => v, None => return false,
        };

        let x_bytes = n * k * 4;
        let y_bytes = n * m * 4;
        if !self.ensure_x(x_bytes, dev) { return false; }
        if !self.ensure_y(y_bytes, dev) { return false; }
        if !self.ensure_args(dev) { return false; }

        // Upload X through BAR. At N=128, K=1024 that's 512KB — well under the
        // staging buffer's typical capacity, single memcpy.
        self.x_buf.as_ref().unwrap().write_f32(0, &x[..n * k]);

        // Build kernargs — layout mirrors the .s file's
        //   { W, B, X, Y, M, K, N } = 8+8+8+8+4+4+4 = 44 bytes
        let xva = self.x_buf.as_ref().unwrap().va_addr;
        let yva = self.y_buf.as_ref().unwrap().va_addr;
        let args = self.args_buf.as_ref().unwrap();
        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&bias_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&xva.to_le_bytes());
        kargs[24..32].copy_from_slice(&yva.to_le_bytes());
        kargs[32..36].copy_from_slice(&(m as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&(k as u32).to_le_bytes());
        kargs[40..44].copy_from_slice(&(n as u32).to_le_bytes());
        args.write(0, &kargs[..44]);

        // Grid: TM=128, TN=32 tiles — one WG per (m/128, n/32) tile pair.
        let nwg_m = ((m + 127) / 128) as u32;
        let nwg_n = ((n +  31) /  32) as u32;
        let total_wg = nwg_m * nwg_n;

        if self.cached_matmul_blocked.is_none() {
            self.cached_matmul_blocked = dev.resolve_kernel("matmul_blocked");
        }
        match &self.cached_matmul_blocked {
            Some(entry) => {
                dev.dispatch_enqueue_resolved(entry, args, [total_wg, 1, 1], [256, 1, 1]);
            }
            None => return false,
        }

        dev.queue.cache_wb();
        // 10s timeout: at 189M param FFN with M=5120 K=1024 N=128 the kernel
        // runs in single-digit ms; a 10s bound catches real hangs without
        // false-positive on any healthy dispatch.
        if !dev.submit_wait(10_000) { return false; }

        // Read Y back through BAR.
        let src = unsafe {
            std::slice::from_raw_parts(
                self.y_buf.as_ref().unwrap().cpu_ptr as *const f32,
                n * m,
            )
        };
        out[..n * m].copy_from_slice(src);

        if !self.vram_mode { self.active = 1 - self.active; }
        true
    }

    /// SuperLinear into caller's output slice.
    pub fn superlinear_into(
        &mut self, dev: &mut HsaDevice,
        weights: &[f32], biases: &[f32], trace: &[f32],
        out: &mut [f32], n_neurons: usize, out_per: usize, in_per: usize,
    ) -> bool {
        let (w_va, bias_va) = match self.prepare_weights(dev, weights, biases) {
            Some(v) => v, None => return false,
        };

        let total_in = n_neurons * in_per;
        let total_out = n_neurons * out_per;
        if !self.ensure_x(total_in * 4, dev) { return false; }
        if !self.ensure_y(total_out * 4, dev) { return false; }
        if !self.ensure_args(dev) { return false; }

        self.x_buf.as_ref().unwrap().write_f32(0, &trace[..total_in]);

        let xva = self.x_buf.as_ref().unwrap().va_addr;
        let yva = self.y_buf.as_ref().unwrap().va_addr;
        let args = self.args_buf.as_ref().unwrap();
        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&bias_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&xva.to_le_bytes());
        kargs[24..32].copy_from_slice(&yva.to_le_bytes());
        kargs[32..36].copy_from_slice(&(n_neurons as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&(out_per as u32).to_le_bytes());
        kargs[40..44].copy_from_slice(&(in_per as u32).to_le_bytes());
        args.write(0, &kargs[..44]);

        if self.cached_superlinear.is_none() {
            self.cached_superlinear = dev.resolve_kernel("superlinear_fwd");
        }

        let total = (n_neurons * out_per) as u32;
        let nwg = (total + 255) / 256;
        match &self.cached_superlinear {
            Some(entry) => {
                dev.dispatch_enqueue_resolved(entry, args, [nwg, 1, 1], [256, 1, 1]);
            }
            None => return false,
        }

        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        let src = unsafe {
            std::slice::from_raw_parts(self.y_buf.as_ref().unwrap().cpu_ptr as *const f32, total_out)
        };
        out[..total_out].copy_from_slice(src);

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
        if !dev.dispatch_enqueue("layer_norm_fwd", args, [1, 1, 1], [256, 1, 1]) { return false; }
        dev.submit_wait(5000)
    }

    /// dx = W^T @ d_out (backward input gradient)
    /// Dedicated tiled kernel — no CPU transpose, no allocation.
    pub fn matvec_t(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32], d_out: &[f32],
        out_dim: usize, in_dim: usize,
    ) -> Option<Vec<f32>> {
        let (w_va, _) = self.prepare_weights(dev, weight, &[])?;

        if !self.ensure_x(out_dim * 4, dev) { return None; }
        if !self.ensure_y(in_dim * 4, dev) { return None; }
        if !self.ensure_args(dev) { return None; }

        // Upload d_out
        self.x_buf.as_ref()?.write_f32(0, d_out);
        let dout_va = self.x_buf.as_ref()?.va_addr;
        let dx_va = self.y_buf.as_ref()?.va_addr;

        // kernargs: W(ptr), d_out(ptr), dx(ptr), out_dim, in_dim
        let args = self.args_buf.as_ref()?;
        let mut kargs = [0u8; 32];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&dout_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&dx_va.to_le_bytes());
        kargs[24..28].copy_from_slice(&(out_dim as u32).to_le_bytes());
        kargs[28..32].copy_from_slice(&(in_dim as u32).to_le_bytes());
        args.write(0, &kargs[..32]);

        // 1 WG per output element (in_dim WGs), 256 threads each
        if !dev.dispatch_enqueue("matvec_t_tiled", args,
            [in_dim as u32, 1, 1], [256, 1, 1]) {
            return None;
        }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return None; }

        let src = unsafe {
            std::slice::from_raw_parts(self.y_buf.as_ref()?.cpu_ptr as *const f32, in_dim)
        };
        let mut dx = vec![0.0f32; in_dim];
        dx.copy_from_slice(src);

        if !self.vram_mode { self.active = 1 - self.active; }
        Some(dx)
    }

    /// dx = W^T @ d_out — zero-copy, both d_out and dx are VRAM VAs.
    pub fn matvec_t_zerocopy(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32],
        dout_va: u64, dx_va: u64,
        out_dim: usize, in_dim: usize,
    ) -> bool {
        let (w_va, _) = match self.prepare_weights(dev, weight, &[]) {
            Some(v) => v, None => return false,
        };
        if !self.ensure_chain_args(0, dev) { return false; }

        let mut kargs = [0u8; 32];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&dout_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&dx_va.to_le_bytes());
        kargs[24..28].copy_from_slice(&(out_dim as u32).to_le_bytes());
        kargs[28..32].copy_from_slice(&(in_dim as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..32]);

        if self.cached_matvec_t_tiled.is_none() {
            self.cached_matvec_t_tiled = dev.resolve_kernel("matvec_t_tiled");
        }
        match &self.cached_matvec_t_tiled {
            Some(entry) => {
                dev.dispatch_enqueue_resolved(entry, args, [in_dim as u32, 1, 1], [256, 1, 1]);
            }
            None => return false,
        }
        dev.submit_wait(5000)
    }

    /// dx = W^T @ d_out — direct write to caller's slice (no Vec alloc).
    pub fn matvec_t_into(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32], d_out: &[f32], dx: &mut [f32],
        out_dim: usize, in_dim: usize,
    ) -> bool {
        let (w_va, _) = match self.prepare_weights(dev, weight, &[]) {
            Some(v) => v, None => return false,
        };

        if !self.ensure_x(out_dim * 4, dev) { return false; }
        if !self.ensure_y(in_dim * 4, dev) { return false; }
        if !self.ensure_args(dev) { return false; }

        // Upload d_out
        self.x_buf.as_ref().unwrap().write_f32(0, d_out);
        let dout_va = self.x_buf.as_ref().unwrap().va_addr;
        let dx_va = self.y_buf.as_ref().unwrap().va_addr;

        // kernargs: W(ptr), d_out(ptr), dx(ptr), out_dim, in_dim
        let args = self.args_buf.as_ref().unwrap();
        let mut kargs = [0u8; 32];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&dout_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&dx_va.to_le_bytes());
        kargs[24..28].copy_from_slice(&(out_dim as u32).to_le_bytes());
        kargs[28..32].copy_from_slice(&(in_dim as u32).to_le_bytes());
        args.write(0, &kargs[..32]);

        // 1 WG per output element (in_dim WGs), 256 threads each
        if self.cached_matvec_t_tiled.is_none() {
            self.cached_matvec_t_tiled = dev.resolve_kernel("matvec_t_tiled");
        }
        match &self.cached_matvec_t_tiled {
            Some(entry) => {
                dev.dispatch_enqueue_resolved(entry, args, [in_dim as u32, 1, 1], [256, 1, 1]);
            }
            None => return false,
        }

        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        // Read back through BAR directly into caller's slice
        let src = unsafe {
            std::slice::from_raw_parts(self.y_buf.as_ref().unwrap().cpu_ptr as *const f32, in_dim)
        };
        dx[..in_dim].copy_from_slice(src);

        if !self.vram_mode { self.active = 1 - self.active; }
        true
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
            if !self.chain_dispatch(dev, 3, "layer_norm_fwd", &kargs[..12], [1, 1, 1], [256, 1, 1]) {
                return None;
            }
        }

        // Flush L2 + submit+wait for all 4 kernels
        dev.queue.cache_wb();
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

    /// Fused synapse backward: silu_bwd → ln_bwd → matvec_t.
    /// All 3 kernels dispatch sequentially on GPU. Data stays in VRAM.
    /// Only ONE PCIe round-trip: upload cached activations + d_out,
    /// download d_input + d_ln + d_gamma + d_beta.
    ///
    /// This is the backward pass for the SynapseBlock (Linear → LN → SiLU)
    /// forward chain. The backward order is reversed: SiLU → LN → Linear.
    ///
    /// Returns (d_input, d_ln). Accumulates into d_gamma, d_beta.
    /// Caller uses d_ln for d_weight (outer product) and d_bias.
    ///
    /// Constraint: out_dim <= 256 (ln_bwd kernel is single-WG).
    pub fn synapse_backward(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32],
        d_out: &[f32],         // [out_dim] gradient from upstream
        pre_silu: &[f32],     // [out_dim] cached pre-SiLU activations
        normalized: &[f32],   // [out_dim] cached LN normalized values
        gamma: &[f32],        // [out_dim] LN gamma
        d_gamma: &mut [f32],  // [out_dim] accumulate LN gamma gradients
        d_beta: &mut [f32],   // [out_dim] accumulate LN beta gradients
        inv_std: f32,         // cached LN inverse std
        out_dim: usize,       // output dimension (post-LN/SiLU)
        in_dim: usize,        // input dimension (weight is [out_dim x in_dim])
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        if out_dim > 256 { return None; } // ln_bwd kernel limit

        // Prepare weight matrix (upload or VRAM cache)
        let (w_va, _) = self.prepare_weights(dev, weight, &[])?;

        // Buffer layout for 3-kernel chain:
        //   scratch_a: d_out [out_dim] → reused for d_ln [out_dim] after K2
        //   scratch_b: pre_silu [out_dim]
        //   chain_args[0..2]: kernargs for K1, K2, K3
        //   chain_args[3]: d_silu [out_dim] (K1 output → K2 input)
        //   chain_args[4]: normalized [out_dim]
        //   chain_args[5]: gamma [out_dim]
        //   chain_args[6]: d_gamma [out_dim] (accumulate, readback)
        //   chain_args[7]: d_beta [out_dim] (accumulate, readback)
        //   y_buf: d_input [in_dim] (K3 output, readback)
        let bytes = out_dim * 4;
        self.ensure_scratch_a(bytes, dev);
        self.ensure_scratch_b(bytes, dev);
        self.ensure_y(in_dim * 4, dev);
        for i in 0..8 {
            if !self.ensure_chain_args(i, dev) { return None; }
        }

        // Upload data to VRAM
        let sa = self.scratch_a.as_ref()?;
        sa.write_f32(0, d_out);
        let sa_va = sa.va_addr;

        let sb = self.scratch_b.as_ref()?;
        sb.write_f32(0, pre_silu);
        let sb_va = sb.va_addr;

        let dsilu_buf = self.chain_args[3].as_ref()?;
        let dsilu_va = dsilu_buf.va_addr;

        let norm_buf = self.chain_args[4].as_ref()?;
        norm_buf.write_f32(0, normalized);
        let norm_va = norm_buf.va_addr;

        let gamma_buf = self.chain_args[5].as_ref()?;
        gamma_buf.write_f32(0, gamma);
        let gamma_va = gamma_buf.va_addr;

        let dg_buf = self.chain_args[6].as_ref()?;
        dg_buf.write_f32(0, d_gamma);
        let dg_va = dg_buf.va_addr;

        let db_buf = self.chain_args[7].as_ref()?;
        db_buf.write_f32(0, d_beta);
        let db_va = db_buf.va_addr;

        let di_va = self.y_buf.as_ref()?.va_addr;

        let nwg = (out_dim as u32 + 255) / 256;

        // ─── Kernel 1: silu_bwd ───
        // d_silu[i] = d_out[i] * (s + x*s*(1-s))  where s = sigmoid(pre_silu[i])
        {
            let mut kargs = [0u8; 28];
            kargs[0..8].copy_from_slice(&sa_va.to_le_bytes());     // d_out
            kargs[8..16].copy_from_slice(&sb_va.to_le_bytes());    // pre_silu
            kargs[16..24].copy_from_slice(&dsilu_va.to_le_bytes()); // d_silu output
            kargs[24..28].copy_from_slice(&(out_dim as u32).to_le_bytes());
            if !self.chain_dispatch(dev, 0, "silu_bwd", &kargs[..28],
                [nwg, 1, 1], [256, 1, 1]) {
                return None;
            }
        }

        // ─── Kernel 2: ln_bwd ───
        // Reuse scratch_a for d_ln output (safe: K1 already consumed d_out)
        {
            let mut kargs = [0u8; 56];
            kargs[0..8].copy_from_slice(&dsilu_va.to_le_bytes());   // d_out (= d_silu)
            kargs[8..16].copy_from_slice(&norm_va.to_le_bytes());   // normalized
            kargs[16..24].copy_from_slice(&gamma_va.to_le_bytes()); // gamma
            kargs[24..32].copy_from_slice(&dg_va.to_le_bytes());    // d_gamma
            kargs[32..40].copy_from_slice(&db_va.to_le_bytes());    // d_beta
            kargs[40..48].copy_from_slice(&sa_va.to_le_bytes());    // d_ln → scratch_a
            kargs[48..52].copy_from_slice(&inv_std.to_le_bytes());
            kargs[52..56].copy_from_slice(&(out_dim as u32).to_le_bytes());
            if !self.chain_dispatch(dev, 1, "ln_bwd", &kargs[..56],
                [1, 1, 1], [256, 1, 1]) {
                return None;
            }
        }

        // ─── Kernel 3: matvec_t_tiled (d_input = W^T @ d_ln) ───
        {
            let mut kargs = [0u8; 32];
            kargs[0..8].copy_from_slice(&w_va.to_le_bytes());     // W
            kargs[8..16].copy_from_slice(&sa_va.to_le_bytes());   // d_ln (from K2)
            kargs[16..24].copy_from_slice(&di_va.to_le_bytes());  // d_input output
            kargs[24..28].copy_from_slice(&(out_dim as u32).to_le_bytes());
            kargs[28..32].copy_from_slice(&(in_dim as u32).to_le_bytes());
            if !self.chain_dispatch(dev, 2, "matvec_t_tiled", &kargs[..32],
                [in_dim as u32, 1, 1], [256, 1, 1]) {
                return None;
            }
        }

        // Flush L2 + submit+wait for all 3 kernels
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) {
            return None;
        }

        // Read back d_input from y_buf
        let src_di = unsafe {
            std::slice::from_raw_parts(self.y_buf.as_ref()?.cpu_ptr as *const f32, in_dim)
        };
        let mut d_input = vec![0.0f32; in_dim];
        d_input.copy_from_slice(src_di);

        // Read back d_ln from scratch_a (K2 wrote it there, K3 consumed it)
        // Caller needs d_ln for d_weight (outer product) and d_bias accumulation.
        let src_dln = unsafe {
            std::slice::from_raw_parts(self.scratch_a.as_ref()?.cpu_ptr as *const f32, out_dim)
        };
        let mut d_ln = vec![0.0f32; out_dim];
        d_ln.copy_from_slice(src_dln);

        // Read back accumulated d_gamma and d_beta
        let src_dg = unsafe {
            std::slice::from_raw_parts(self.chain_args[6].as_ref()?.cpu_ptr as *const f32, out_dim)
        };
        d_gamma[..out_dim].copy_from_slice(src_dg);
        let src_db = unsafe {
            std::slice::from_raw_parts(self.chain_args[7].as_ref()?.cpu_ptr as *const f32, out_dim)
        };
        d_beta[..out_dim].copy_from_slice(src_db);

        if !self.vram_mode { self.active = 1 - self.active; }
        Some((d_input, d_ln))
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
        dev.queue.cache_wb();
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
        dev.queue.cache_wb();
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

    /// GPU sync backward: scatter gradients from pairs back to neurons via atomic add.
    ///
    /// For each pair i:
    ///   scale = 1 / max(sqrt(beta[i]), 1e-8)
    ///   d_act[left[i]] += d_sync[i] * scale * activated[right[i]]
    ///   d_act[right[i]] += d_sync[i] * scale * activated[left[i]]
    ///
    /// `left` and `right` are neuron indices (u32). `d_act` is zero-initialized
    /// on GPU before dispatch, then read back into the provided slice.
    pub fn sync_backward(
        &mut self, dev: &mut HsaDevice,
        d_sync: &[f32], activated: &[f32], beta: &[f32],
        left: &[u32], right: &[u32],
        n_pairs: usize, d_model: usize,
        d_act: &mut [f32],
    ) -> bool {
        // Layout in scratch_a:
        //   [d_sync | beta | left | right | activated | d_act]
        //   n_pairs n_pairs n_pairs n_pairs  d_model    d_model  (all *4 bytes)
        let pair_bytes = n_pairs * 4;
        let model_bytes = d_model * 4;
        let total_bytes = pair_bytes * 4 + model_bytes * 2;

        if !self.ensure_scratch_a(total_bytes, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }

        let base = self.scratch_a.as_ref().unwrap();
        let base_va = base.va_addr;

        // Offsets (byte)
        let d_sync_off    = 0;
        let beta_off      = pair_bytes;
        let left_off      = pair_bytes * 2;
        let right_off     = pair_bytes * 3;
        let activated_off = pair_bytes * 4;
        let d_act_off     = pair_bytes * 4 + model_bytes;

        let d_sync_va    = base_va + d_sync_off as u64;
        let beta_va      = base_va + beta_off as u64;
        let left_va      = base_va + left_off as u64;
        let right_va     = base_va + right_off as u64;
        let activated_va = base_va + activated_off as u64;
        let d_act_va     = base_va + d_act_off as u64;

        // Upload f32 arrays
        base.write_f32(d_sync_off, &d_sync[..n_pairs]);
        base.write_f32(beta_off, &beta[..n_pairs]);
        // Write u32 index arrays as raw bytes
        let left_bytes = unsafe {
            std::slice::from_raw_parts(left.as_ptr() as *const u8, pair_bytes)
        };
        let right_bytes = unsafe {
            std::slice::from_raw_parts(right.as_ptr() as *const u8, pair_bytes)
        };
        base.write(left_off, left_bytes);
        base.write(right_off, right_bytes);
        base.write_f32(activated_off, &activated[..d_model]);

        // Zero-initialize d_act region
        let zeros = vec![0u8; model_bytes];
        base.write(d_act_off, &zeros);

        // Build kernargs (52 bytes, padded to 56):
        //   6 pointers (48 bytes) + n_pairs (u32, 4 bytes) = 52 bytes
        let mut kargs = [0u8; 56];
        kargs[0..8].copy_from_slice(&d_sync_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&activated_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&beta_va.to_le_bytes());
        kargs[24..32].copy_from_slice(&left_va.to_le_bytes());
        kargs[32..40].copy_from_slice(&right_va.to_le_bytes());
        kargs[40..48].copy_from_slice(&d_act_va.to_le_bytes());
        kargs[48..52].copy_from_slice(&(n_pairs as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..56]);

        let nwg = (n_pairs as u32 + 255) / 256;
        if !dev.dispatch_enqueue("sync_backward_scatter", args, [nwg, 1, 1], [256, 1, 1]) {
            return false;
        }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        // Read back d_act
        let result = unsafe {
            std::slice::from_raw_parts(
                (base.cpu_ptr as *const u8).add(d_act_off) as *const f32,
                d_model,
            )
        };
        d_act[..d_model].copy_from_slice(result);
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
        dev.queue.cache_wb();
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
        dev.queue.cache_wb();
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
        if n > 1024 { return false; }

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
        if !dev.dispatch_enqueue("layer_norm_fwd", args, [1, 1, 1], [256, 1, 1]) {
            return false;
        }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        let result = unsafe {
            std::slice::from_raw_parts(sa.cpu_ptr as *const f32, n)
        };
        x[..n].copy_from_slice(result);
        true
    }

    /// Outer product accumulate: dW[i*k+j] += d_out[i] * input[j]
    /// SiLU backward: d_input[i] = d_out[i] * (s + x*s*(1-s))
    pub fn silu_backward(
        &mut self, dev: &mut HsaDevice,
        d_out: &[f32], pre_silu: &[f32], d_input: &mut [f32],
    ) -> bool {
        let n = d_out.len();
        let bytes = n * 4;
        if !self.ensure_scratch_a(bytes, dev) { return false; }
        if !self.ensure_scratch_b(bytes, dev) { return false; }
        if !self.ensure_y(bytes, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }

        let sa = self.scratch_a.as_ref().unwrap();
        sa.write_f32(0, d_out);
        let sb = self.scratch_b.as_ref().unwrap();
        sb.write_f32(0, pre_silu);
        let ybuf = self.y_buf.as_ref().unwrap();
        let dy_va = ybuf.va_addr;

        let mut kargs = [0u8; 28];
        kargs[0..8].copy_from_slice(&sa.va_addr.to_le_bytes());
        kargs[8..16].copy_from_slice(&sb.va_addr.to_le_bytes());
        kargs[16..24].copy_from_slice(&dy_va.to_le_bytes());
        kargs[24..28].copy_from_slice(&(n as u32).to_le_bytes());
        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..28]);

        let nwg = (n as u32 + 255) / 256;
        if !dev.dispatch_enqueue("silu_bwd", args, [nwg, 1, 1], [256, 1, 1]) { return false; }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        let src = unsafe { std::slice::from_raw_parts(ybuf.cpu_ptr as *const f32, n) };
        d_input[..n].copy_from_slice(src);
        true
    }

    /// GLU backward: d_val and d_gate from d_out + cached input.
    pub fn glu_backward(
        &mut self, dev: &mut HsaDevice,
        d_out: &[f32], cached_input: &[f32], d_input: &mut [f32], n: usize,
    ) -> bool {
        let dout_bytes = n * 4;
        let input_bytes = 2 * n * 4;
        if !self.ensure_scratch_a(dout_bytes, dev) { return false; }
        if !self.ensure_scratch_b(input_bytes, dev) { return false; }
        if !self.ensure_y(input_bytes, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }

        let sa = self.scratch_a.as_ref().unwrap();
        sa.write_f32(0, d_out);
        let sb = self.scratch_b.as_ref().unwrap();
        sb.write_f32(0, cached_input);
        let ybuf = self.y_buf.as_ref().unwrap();
        let di_va = ybuf.va_addr;

        let mut kargs = [0u8; 28];
        kargs[0..8].copy_from_slice(&sa.va_addr.to_le_bytes());
        kargs[8..16].copy_from_slice(&sb.va_addr.to_le_bytes());
        kargs[16..24].copy_from_slice(&di_va.to_le_bytes());
        kargs[24..28].copy_from_slice(&(n as u32).to_le_bytes());
        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..28]);

        let nwg = (n as u32 + 255) / 256;
        if !dev.dispatch_enqueue("glu_bwd", args, [nwg, 1, 1], [256, 1, 1]) { return false; }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        let src = unsafe { std::slice::from_raw_parts(ybuf.cpu_ptr as *const f32, 2 * n) };
        d_input[..2 * n].copy_from_slice(src);
        true
    }

    /// Affine LayerNorm backward (single WG, n <= 256).
    pub fn ln_backward(
        &mut self, dev: &mut HsaDevice,
        d_out: &[f32], normalized: &[f32], gamma: &[f32],
        d_gamma: &mut [f32], d_beta: &mut [f32], d_input: &mut [f32],
        inv_std: f32,
    ) -> bool {
        let n = d_out.len();
        if n > 256 { return false; }
        let bytes = n * 4;

        // Need 6 GPU buffers: d_out, normalized, gamma, d_gamma, d_beta, d_input
        // Use scratch_a for d_out, scratch_b for normalized, x for gamma,
        // y for d_input, chain_args[1..2] for d_gamma/d_beta
        if !self.ensure_scratch_a(bytes, dev) { return false; }
        if !self.ensure_scratch_b(bytes, dev) { return false; }
        if !self.ensure_x(bytes, dev) { return false; }
        if !self.ensure_y(bytes, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }
        if !self.ensure_chain_args(1, dev) { return false; }
        if !self.ensure_chain_args(2, dev) { return false; }

        let sa = self.scratch_a.as_ref().unwrap();
        sa.write_f32(0, d_out);
        let sb = self.scratch_b.as_ref().unwrap();
        sb.write_f32(0, normalized);
        let xbuf = self.x_buf.as_ref().unwrap();
        xbuf.write_f32(0, gamma);

        // d_gamma and d_beta need to be uploaded (they accumulate)
        let dg_buf = self.chain_args[1].as_ref().unwrap();
        dg_buf.write_f32(0, d_gamma);
        let db_buf = self.chain_args[2].as_ref().unwrap();
        db_buf.write_f32(0, d_beta);

        let ybuf = self.y_buf.as_ref().unwrap();
        let di_va = ybuf.va_addr;

        // kernargs: d_out, normalized, gamma, d_gamma, d_beta, d_input, inv_std, N
        let mut kargs = [0u8; 56];
        kargs[0..8].copy_from_slice(&sa.va_addr.to_le_bytes());
        kargs[8..16].copy_from_slice(&sb.va_addr.to_le_bytes());
        kargs[16..24].copy_from_slice(&xbuf.va_addr.to_le_bytes());
        kargs[24..32].copy_from_slice(&dg_buf.va_addr.to_le_bytes());
        kargs[32..40].copy_from_slice(&db_buf.va_addr.to_le_bytes());
        kargs[40..48].copy_from_slice(&di_va.to_le_bytes());
        kargs[48..52].copy_from_slice(&inv_std.to_le_bytes());
        kargs[52..56].copy_from_slice(&(n as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..56]);

        // Single WG, 256 threads
        if !dev.dispatch_enqueue("ln_bwd", args, [1, 1, 1], [256, 1, 1]) { return false; }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        // Read back d_input, d_gamma, d_beta
        let src_di = unsafe { std::slice::from_raw_parts(ybuf.cpu_ptr as *const f32, n) };
        d_input[..n].copy_from_slice(src_di);
        let src_dg = unsafe { std::slice::from_raw_parts(dg_buf.cpu_ptr as *const f32, n) };
        d_gamma[..n].copy_from_slice(src_dg);
        let src_db = unsafe { std::slice::from_raw_parts(db_buf.cpu_ptr as *const f32, n) };
        d_beta[..n].copy_from_slice(src_db);
        true
    }

    /// Per-neuron GLU backward with strided layout.
    pub fn per_neuron_glu_backward(
        &mut self, dev: &mut HsaDevice,
        d_out: &[f32], cached_input: &[f32], d_input: &mut [f32],
        n_neurons: usize, out_per: usize,
    ) -> bool {
        let half = out_per / 2;
        let total_out = n_neurons * half;
        let total_in = n_neurons * out_per;
        let dout_bytes = total_out * 4;
        let input_bytes = total_in * 4;

        if !self.ensure_scratch_a(dout_bytes, dev) { return false; }
        if !self.ensure_scratch_b(input_bytes, dev) { return false; }
        if !self.ensure_y(input_bytes, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }

        let sa = self.scratch_a.as_ref().unwrap();
        sa.write_f32(0, d_out);
        let sb = self.scratch_b.as_ref().unwrap();
        sb.write_f32(0, cached_input);
        let ybuf = self.y_buf.as_ref().unwrap();

        let mut kargs = [0u8; 32];
        kargs[0..8].copy_from_slice(&sa.va_addr.to_le_bytes());
        kargs[8..16].copy_from_slice(&sb.va_addr.to_le_bytes());
        kargs[16..24].copy_from_slice(&ybuf.va_addr.to_le_bytes());
        kargs[24..28].copy_from_slice(&(n_neurons as u32).to_le_bytes());
        kargs[28..32].copy_from_slice(&(out_per as u32).to_le_bytes());
        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..32]);

        let nwg = (total_out as u32 + 255) / 256;
        if !dev.dispatch_enqueue("per_neuron_glu_bwd", args, [nwg, 1, 1], [256, 1, 1]) {
            return false;
        }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        let src = unsafe { std::slice::from_raw_parts(ybuf.cpu_ptr as *const f32, total_in) };
        d_input[..total_in].copy_from_slice(src);
        true
    }

    pub fn outer_product(
        &mut self, dev: &mut HsaDevice,
        d_weight: &mut [f32], d_out: &[f32], input: &[f32],
        out_dim: usize, in_dim: usize,
    ) -> bool {
        let total = out_dim * in_dim;
        let dw_bytes = total * 4;
        let dout_bytes = out_dim * 4;
        let in_bytes = in_dim * 4;

        // Need 3 buffers: dW (read+write), d_out (read), input (read)
        if !self.ensure_scratch_a(dw_bytes, dev) { return false; }
        if !self.ensure_x(dout_bytes.max(in_bytes), dev) { return false; }
        if !self.ensure_y(in_bytes.max(dout_bytes), dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }

        let sa = self.scratch_a.as_ref().unwrap();
        sa.write_f32(0, d_weight);
        let dw_va = sa.va_addr;

        let xbuf = self.x_buf.as_ref().unwrap();
        xbuf.write_f32(0, d_out);
        let dout_va = xbuf.va_addr;

        let ybuf = self.y_buf.as_ref().unwrap();
        ybuf.write_f32(0, input);
        let in_va = ybuf.va_addr;

        let mut kargs = [0u8; 32];
        kargs[0..8].copy_from_slice(&dw_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&dout_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&in_va.to_le_bytes());
        kargs[24..28].copy_from_slice(&(out_dim as u32).to_le_bytes());
        kargs[28..32].copy_from_slice(&(in_dim as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..32]);

        if self.cached_outer_product.is_none() {
            self.cached_outer_product = dev.resolve_kernel("outer_product_acc");
        }
        let nwg = (total as u32 + 255) / 256;
        match &self.cached_outer_product {
            Some(entry) => dev.dispatch_enqueue_resolved(entry, args, [nwg, 1, 1], [256, 1, 1]),
            None => return false,
        }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        // Read back accumulated dW
        let result = unsafe {
            std::slice::from_raw_parts(sa.cpu_ptr as *const f32, total)
        };
        d_weight[..total].copy_from_slice(result);
        true
    }

    /// SuperLinear backward: d_weight accumulation + d_input computation.
    /// Two dispatches: bwd_dw (one thread per dW element) + bwd_dx (one thread per dX element).
    pub fn superlinear_backward(
        &mut self, dev: &mut HsaDevice,
        weights: &[f32], d_out: &[f32], input: &[f32],
        d_weights: &mut [f32], d_input: &mut [f32],
        n_neurons: usize, out_per: usize, in_per: usize,
    ) -> bool {
        let total_w = n_neurons * out_per * in_per;
        let total_out = n_neurons * out_per;
        let total_in = n_neurons * in_per;

        // We need buffers for: W, dW, d_out, input, dX
        if !self.ensure_scratch_a(total_w * 4, dev) { return false; }
        if !self.ensure_scratch_b(total_in * 4, dev) { return false; }
        if !self.ensure_x(total_out.max(total_in) * 4, dev) { return false; }
        if !self.ensure_y(total_w.max(total_in) * 4, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }
        if !self.ensure_chain_args(1, dev) { return false; }

        // Upload: dW → scratch_a, d_out → x_buf, input → y_buf, W → scratch_b
        let sa = self.scratch_a.as_ref().unwrap();
        sa.write_f32(0, d_weights);
        let dw_va = sa.va_addr;

        let xbuf = self.x_buf.as_ref().unwrap();
        xbuf.write_f32(0, d_out);
        let dout_va = xbuf.va_addr;

        let ybuf = self.y_buf.as_ref().unwrap();
        ybuf.write_f32(0, input);
        let input_va = ybuf.va_addr;

        let sb = self.scratch_b.as_ref().unwrap();
        sb.write_f32(0, weights);
        let w_va = sb.va_addr;

        // Allocate dX output in a separate temporary
        // Re-use y_buf after dw dispatch (input won't be needed after dw)
        // Actually we need input for BOTH dispatches. Use chain_args for dX output.
        // Need a proper dX buffer — let's re-use x_buf after first dispatch.

        // Dispatch 1: bwd_dw
        // kernargs: W(ptr), dW(ptr), d_out(ptr), input(ptr), dX(ptr), N, O, K
        let mut kargs = [0u8; 52];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());       // W (unused by dw)
        kargs[8..16].copy_from_slice(&dw_va.to_le_bytes());     // dW
        kargs[16..24].copy_from_slice(&dout_va.to_le_bytes());  // d_out
        kargs[24..32].copy_from_slice(&input_va.to_le_bytes()); // input
        kargs[32..40].copy_from_slice(&0u64.to_le_bytes());     // dX (unused by dw)
        kargs[40..44].copy_from_slice(&(n_neurons as u32).to_le_bytes());
        kargs[44..48].copy_from_slice(&(out_per as u32).to_le_bytes());
        kargs[48..52].copy_from_slice(&(in_per as u32).to_le_bytes());

        let args0 = self.chain_args[0].as_ref().unwrap();
        args0.write(0, &kargs[..52]);

        if self.cached_sl_bwd_dw.is_none() {
            self.cached_sl_bwd_dw = dev.resolve_kernel("superlinear_bwd_dw");
        }
        let nwg_dw = (total_w as u32 + 255) / 256;
        match &self.cached_sl_bwd_dw {
            Some(entry) => dev.dispatch_enqueue_resolved(entry, args0, [nwg_dw, 1, 1], [256, 1, 1]),
            None => return false,
        }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        // Read back dW
        let dw_result = unsafe {
            std::slice::from_raw_parts(sa.cpu_ptr as *const f32, total_w)
        };
        d_weights[..total_w].copy_from_slice(dw_result);

        // Dispatch 2: bwd_dx — re-use x_buf for dX output
        // Need to re-upload d_out (x_buf was used for d_out, now we want dX there)
        // Use scratch_a for dX output instead
        sa.write_f32(0, &vec![0.0f32; total_in]); // zero dX
        let dx_va = sa.va_addr;

        // Re-upload d_out to y_buf (input is no longer needed)
        ybuf.write_f32(0, d_out);
        let dout_va2 = ybuf.va_addr;

        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());       // W
        kargs[8..16].copy_from_slice(&0u64.to_le_bytes());      // dW (unused by dx)
        kargs[16..24].copy_from_slice(&dout_va2.to_le_bytes()); // d_out
        kargs[24..32].copy_from_slice(&0u64.to_le_bytes());     // input (unused by dx)
        kargs[32..40].copy_from_slice(&dx_va.to_le_bytes());    // dX

        let args1 = self.chain_args[1].as_ref().unwrap();
        args1.write(0, &kargs[..52]);

        if self.cached_sl_bwd_dx.is_none() {
            self.cached_sl_bwd_dx = dev.resolve_kernel("superlinear_bwd_dx");
        }
        let nwg_dx = (total_in as u32 + 255) / 256;
        match &self.cached_sl_bwd_dx {
            Some(entry) => dev.dispatch_enqueue_resolved(entry, args1, [nwg_dx, 1, 1], [256, 1, 1]),
            None => return false,
        }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        // Read back dX
        let dx_result = unsafe {
            std::slice::from_raw_parts(sa.cpu_ptr as *const f32, total_in)
        };
        d_input[..total_in].copy_from_slice(dx_result);

        true
    }

    /// SGD update: w[i] -= lr_scale * grad[i]; grad[i] = 0
    pub fn sgd_update(
        &mut self, dev: &mut HsaDevice,
        weights: &mut [f32], grads: &mut [f32], lr_scale: f32,
    ) -> bool {
        let n = weights.len();
        if n != grads.len() { return false; }
        let bytes = n * 4;

        if !self.ensure_scratch_a(bytes, dev) { return false; }
        if !self.ensure_x(bytes, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }

        let sa = self.scratch_a.as_ref().unwrap();
        sa.write_f32(0, weights);
        let w_va = sa.va_addr;

        let xbuf = self.x_buf.as_ref().unwrap();
        xbuf.write_f32(0, grads);
        let grad_va = xbuf.va_addr;

        let mut kargs = [0u8; 24];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&grad_va.to_le_bytes());
        kargs[16..20].copy_from_slice(&lr_scale.to_le_bytes());
        kargs[20..24].copy_from_slice(&(n as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..24]);

        if self.cached_sgd_update.is_none() {
            self.cached_sgd_update = dev.resolve_kernel("sgd_update");
        }
        let nwg = (n as u32 + 255) / 256;
        match &self.cached_sgd_update {
            Some(entry) => dev.dispatch_enqueue_resolved(entry, args, [nwg, 1, 1], [256, 1, 1]),
            None => return false,
        }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        // Read back updated weights
        let result = unsafe {
            std::slice::from_raw_parts(sa.cpu_ptr as *const f32, n)
        };
        weights[..n].copy_from_slice(result);
        // Grads are zeroed on GPU — reflect in CPU
        grads.fill(0.0);
        true
    }

    /// AdamW optimizer: update weights, moments, and zero grads on GPU.
    /// All four buffers (w, grad, m, v) are uploaded, kernel runs, results read back.
    pub fn adamw(
        &mut self, dev: &mut HsaDevice,
        weights: &mut [f32], grads: &mut [f32],
        m: &mut [f32], v: &mut [f32],
        lr: f32, beta1: f32, beta2: f32, eps: f32,
        weight_decay: f32, bc1_inv: f32, bc2_inv: f32,
    ) -> bool {
        let n = weights.len();
        if n != grads.len() || n != m.len() || n != v.len() { return false; }
        let bytes = n * 4;

        // scratch_a = weights, x_buf = grads, scratch_b = m, y_buf = v
        if !self.ensure_scratch_a(bytes, dev) { return false; }
        if !self.ensure_x(bytes, dev) { return false; }
        if !self.ensure_scratch_b(bytes, dev) { return false; }
        if !self.ensure_y(bytes, dev) { return false; }
        if !self.ensure_chain_args(0, dev) { return false; }

        let sa = self.scratch_a.as_ref().unwrap();
        sa.write_f32(0, weights);
        let w_va = sa.va_addr;

        let xbuf = self.x_buf.as_ref().unwrap();
        xbuf.write_f32(0, grads);
        let grad_va = xbuf.va_addr;

        let sb = self.scratch_b.as_ref().unwrap();
        sb.write_f32(0, m);
        let m_va = sb.va_addr;

        let ybuf = self.y_buf.as_ref().unwrap();
        ybuf.write_f32(0, v);
        let v_va = ybuf.va_addr;

        // kernarg layout: 4 pointers (32 bytes) + 8 scalars (32 bytes) = 64 bytes
        let mut kargs = [0u8; 64];
        kargs[0..8].copy_from_slice(&w_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&grad_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&m_va.to_le_bytes());
        kargs[24..32].copy_from_slice(&v_va.to_le_bytes());
        kargs[32..36].copy_from_slice(&(n as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&lr.to_le_bytes());
        kargs[40..44].copy_from_slice(&beta1.to_le_bytes());
        kargs[44..48].copy_from_slice(&beta2.to_le_bytes());
        kargs[48..52].copy_from_slice(&eps.to_le_bytes());
        kargs[52..56].copy_from_slice(&weight_decay.to_le_bytes());
        kargs[56..60].copy_from_slice(&bc1_inv.to_le_bytes());
        kargs[60..64].copy_from_slice(&bc2_inv.to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..64]);

        if self.cached_adamw.is_none() {
            self.cached_adamw = dev.resolve_kernel("adamw");
        }
        let nwg = (n as u32 + 255) / 256;
        match &self.cached_adamw {
            Some(entry) => dev.dispatch_enqueue_resolved(entry, args, [nwg, 1, 1], [256, 1, 1]),
            None => return false,
        }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return false; }

        // Read back updated weights, m, v
        let w_result = unsafe {
            std::slice::from_raw_parts(sa.cpu_ptr as *const f32, n)
        };
        weights[..n].copy_from_slice(w_result);

        let m_result = unsafe {
            std::slice::from_raw_parts(sb.cpu_ptr as *const f32, n)
        };
        m[..n].copy_from_slice(m_result);

        let v_result = unsafe {
            std::slice::from_raw_parts(ybuf.cpu_ptr as *const f32, n)
        };
        v[..n].copy_from_slice(v_result);

        // Grads zeroed on GPU
        grads.fill(0.0);
        true
    }

    /// Reduce L2 norm: returns sqrt(sum(x[i]^2)).
    /// GPU computes partial sums (one per workgroup), CPU finishes reduction.
    pub fn reduce_l2(&mut self, dev: &mut HsaDevice, data: &[f32]) -> Option<f32> {
        let n = data.len();
        if n == 0 { return Some(0.0); }

        let n_wg = ((n + 255) / 256) as u32;
        let data_bytes = n * 4;
        let partial_bytes = n_wg as usize * 4;

        // x_buf: input data, y_buf: partial sums output
        if !self.ensure_x(data_bytes, dev) { return None; }
        if !self.ensure_y(partial_bytes, dev) { return None; }
        if !self.ensure_chain_args(0, dev) { return None; }

        let xbuf = self.x_buf.as_ref().unwrap();
        xbuf.write_f32(0, data);
        let x_va = xbuf.va_addr;

        let ybuf = self.y_buf.as_ref().unwrap();
        let partial_va = ybuf.va_addr;

        // kernarg layout: x_ptr(u64) + partial_sums_ptr(u64) + N(u32) = 20 bytes
        let mut kargs = [0u8; 24];
        kargs[0..8].copy_from_slice(&x_va.to_le_bytes());
        kargs[8..16].copy_from_slice(&partial_va.to_le_bytes());
        kargs[16..20].copy_from_slice(&(n as u32).to_le_bytes());

        let args = self.chain_args[0].as_ref().unwrap();
        args.write(0, &kargs[..24]);

        if self.cached_reduce_l2.is_none() {
            self.cached_reduce_l2 = dev.resolve_kernel("reduce_l2_sq");
        }
        match &self.cached_reduce_l2 {
            Some(entry) => dev.dispatch_enqueue_resolved(entry, args, [n_wg, 1, 1], [256, 1, 1]),
            None => return None,
        }
        dev.queue.cache_wb();
        if !dev.submit_wait(5000) { return None; }

        // CPU pass 2: sum partial results
        let partials = unsafe {
            std::slice::from_raw_parts(ybuf.cpu_ptr as *const f32, n_wg as usize)
        };
        let total_sq: f32 = partials.iter().sum();
        Some(total_sq.sqrt())
    }
}
