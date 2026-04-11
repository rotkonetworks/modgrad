//! MegaTrain-style streaming engine: double-buffered GPU dispatch.
//!
//! Parameters live in CPU RAM. GPU has two ping-pong weight buffers.
//! While GPU computes with buffer A, CPU uploads next layer into buffer B.
//! On next call, roles swap — zero stall time.
//!
//! Key invariant: GPU memory bounded by 2 × max_layer_size + x + y + args.

use super::memory::GpuBuffer;
use super::HsaDevice;

/// Double-buffered streaming engine.
pub struct StreamEngine {
    /// Two weight buffers (A/B ping-pong).
    w_buf: [Option<GpuBuffer>; 2],
    w_cap: [usize; 2],
    /// Which buffer holds the CURRENT dispatch's weights (0 or 1).
    /// The other buffer is free for prefetching.
    active: usize,
    /// Input/output/args — shared across dispatches.
    x_buf: Option<GpuBuffer>,
    x_cap: usize,
    y_buf: Option<GpuBuffer>,
    y_cap: usize,
    args_buf: Option<GpuBuffer>,
}

impl StreamEngine {
    pub fn new() -> Self {
        Self {
            w_buf: [None, None],
            w_cap: [0, 0],
            active: 0,
            x_buf: None, x_cap: 0,
            y_buf: None, y_cap: 0,
            args_buf: None,
        }
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

    /// Upload weight+bias to the active buffer, dispatch kernel, read result.
    /// Flips active buffer after dispatch for ping-pong on next call.
    fn dispatch_kernel(
        &mut self, dev: &mut HsaDevice,
        kernel: &str,
        weight_data: &[f32], bias_data: &[f32],
        x_data: &[f32],
        extra_args: &[u32],  // kernel-specific u32 args after the 4 pointers
        nwg: u32,
        out_count: usize,
    ) -> Option<Vec<f32>> {
        let w_bytes = weight_data.len() * 4;
        let idx = self.active;

        // Ensure all buffers before borrowing any
        self.ensure_w(idx, w_bytes + bias_data.len() * 4, dev);
        self.ensure_x(x_data.len() * 4, dev);
        self.ensure_y(out_count * 4, dev);
        self.ensure_args(dev);

        // Upload weights + bias
        let wbuf = self.w_buf[idx].as_ref()?;
        wbuf.write_f32(0, weight_data);
        wbuf.write_f32(w_bytes, bias_data);

        // Upload input
        let xbuf = self.x_buf.as_ref()?;
        xbuf.write_f32(0, x_data);

        // Zero output
        let ybuf = self.y_buf.as_ref()?;
        unsafe { std::ptr::write_bytes(ybuf.cpu_ptr, 0, out_count * 4); }

        // Build kernargs: 4 pointers + extra u32s
        let args = self.args_buf.as_ref()?;
        let bias_va = wbuf.va_addr + w_bytes as u64;
        let mut kargs = [0u8; 64];
        kargs[0..8].copy_from_slice(&wbuf.va_addr.to_le_bytes());
        kargs[8..16].copy_from_slice(&bias_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&xbuf.va_addr.to_le_bytes());
        kargs[24..32].copy_from_slice(&ybuf.va_addr.to_le_bytes());
        for (i, &v) in extra_args.iter().enumerate() {
            let off = 32 + i * 4;
            kargs[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let total_bytes = 32 + extra_args.len() * 4;
        args.write(0, &kargs[..total_bytes]);

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

        self.active = 1 - self.active;
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
