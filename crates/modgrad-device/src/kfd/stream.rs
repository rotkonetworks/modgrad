//! GPU compute engine with two modes:
//!
//! **Stream mode** (--gpu): Upload weights per-call through ping-pong buffers.
//!   Best for PCIe x16 where transfer overlaps compute.
//!
//! **VRAM mode** (--vram): Upload weights once, keep in VRAM permanently.
//!   Best for PCIe x4 where transfer dominates. Re-upload only after
//!   optimizer step (one bulk transfer vs thousands of per-op transfers).
//!
//! Both modes share the same dispatch_kernel path. The only difference
//! is whether weights are uploaded (stream) or looked up (vram).

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
    fn dispatch_kernel(
        &mut self, dev: &mut HsaDevice,
        kernel: &str,
        weight_data: &[f32], bias_data: &[f32],
        x_data: &[f32],
        extra_args: &[u32],
        nwg: u32,
        out_count: usize,
    ) -> Option<Vec<f32>> {
        // Prepare weights (cached in VRAM mode, uploaded in stream mode)
        let (w_va, bias_va) = self.prepare_weights(dev, weight_data, bias_data)?;

        // Prepare input/output/args
        self.ensure_x(x_data.len() * 4, dev);
        self.ensure_y(out_count * 4, dev);
        self.ensure_args(dev);

        let xbuf = self.x_buf.as_ref()?;
        xbuf.write_f32(0, x_data);

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

        // Dispatch + wait
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

        // Flip ping-pong in stream mode only
        if !self.vram_mode { self.active = 1 - self.active; }
        Some(y)
    }

    /// y = W @ x + b
    pub fn matvec(
        &mut self, dev: &mut HsaDevice,
        weight: &[f32], bias: &[f32], x: &[f32],
        out_dim: usize, in_dim: usize,
    ) -> Option<Vec<f32>> {
        let nwg = ((out_dim as u32) + 255) / 256;
        self.dispatch_kernel(dev, "matvec", weight, bias, x,
            &[out_dim as u32, in_dim as u32], nwg, out_dim)
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
}
