//! MegaTrain-style streaming engine for layer-wise GPU compute.
//!
//! Parameters live in CPU RAM. GPU has two ping-pong buffers.
//! For each layer: stream weights in → compute → stream gradients out.
//! Double-buffered: while GPU computes layer i, prefetch layer i+1.
//!
//! This avoids the concurrency bugs of the resident weight cache
//! (rapid sequential dispatches) by serializing to one layer at a time.
//!
//! Key invariant: GPU memory is bounded by 2 × max_layer_size,
//! regardless of total model size.

use super::memory::GpuBuffer;
use super::HsaDevice;

/// Two ping-pong buffers for weight streaming.
/// While buffer A is used for compute, buffer B is being filled.
pub struct StreamEngine {
    /// Two weight buffers (ping-pong)
    buf: [Option<GpuBuffer>; 2],
    buf_cap: [usize; 2],
    /// Which buffer is currently active (0 or 1)
    active: usize,
    /// Output buffer (for y = Wx + b result)
    y_buf: Option<GpuBuffer>,
    y_cap: usize,
    /// Input buffer (for x vector)
    x_buf: Option<GpuBuffer>,
    x_cap: usize,
    /// Kernargs buffer
    args_buf: Option<GpuBuffer>,
}

impl StreamEngine {
    pub fn new() -> Self {
        Self {
            buf: [None, None],
            buf_cap: [0, 0],
            active: 0,
            y_buf: None, y_cap: 0,
            x_buf: None, x_cap: 0,
            args_buf: None,
        }
    }

    /// Stream-in weights, compute y = Wx + b, return y.
    ///
    /// Weights are uploaded to the active buffer, compute runs,
    /// then the active buffer flips for the next call.
    /// This gives the GPU time to finish while we upload the next layer.
    pub fn matvec(
        &mut self,
        dev: &mut HsaDevice,
        weight: &[f32], bias: &[f32],
        x: &[f32],
        out_dim: usize, in_dim: usize,
    ) -> Option<Vec<f32>> {
        let w_bytes = weight.len() * 4;
        let b_bytes = bias.len() * 4;
        let total_wb = w_bytes + b_bytes;
        let idx = self.active;

        // Ensure buffer is large enough
        if self.buf_cap[idx] < total_wb {
            let cap = ((total_wb + 4095) & !4095) as u64;
            self.buf[idx] = dev.alloc.alloc_vram(cap).ok();
            self.buf_cap[idx] = cap as usize;
        }
        let wbuf = self.buf[idx].as_ref()?;

        // Upload weight + bias into the active buffer contiguously
        wbuf.write_f32(0, weight);
        wbuf.write_f32(w_bytes, bias);

        // Ensure x buffer
        let x_bytes = x.len() * 4;
        if self.x_cap < x_bytes {
            let cap = ((x_bytes + 4095) & !4095).max(4096) as u64;
            self.x_buf = dev.alloc.alloc_vram(cap).ok();
            self.x_cap = cap as usize;
        }
        let xbuf = self.x_buf.as_ref()?;
        xbuf.write_f32(0, x);

        // Ensure y buffer
        let y_bytes = out_dim * 4;
        if self.y_cap < y_bytes {
            let cap = ((y_bytes + 4095) & !4095).max(4096) as u64;
            self.y_buf = dev.alloc_output(cap as usize).ok();
            self.y_cap = cap as usize;
        }
        let ybuf = self.y_buf.as_ref()?;
        unsafe { std::ptr::write_bytes(ybuf.cpu_ptr, 0, y_bytes); }

        // Ensure args buffer
        if self.args_buf.is_none() {
            self.args_buf = dev.alloc.alloc_vram(4096).ok();
        }
        let args = self.args_buf.as_ref()?;

        // Build kernargs: W ptr, bias ptr, x ptr, y ptr, out_dim, in_dim
        let bias_va = wbuf.va_addr + w_bytes as u64;
        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&wbuf.va_addr.to_le_bytes());
        kargs[8..16].copy_from_slice(&bias_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&xbuf.va_addr.to_le_bytes());
        kargs[24..32].copy_from_slice(&ybuf.va_addr.to_le_bytes());
        kargs[32..36].copy_from_slice(&(out_dim as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&(in_dim as u32).to_le_bytes());
        args.write(0, &kargs[..40]);

        // Dispatch
        let nwg = ((out_dim as u32) + 255) / 256;
        if !dev.dispatch_enqueue("matvec", args, [nwg, 1, 1], [256, 1, 1]) {
            return None;
        }
        if !dev.submit_wait(2000) {
            return None;
        }

        // Read back
        let y_slice = unsafe {
            std::slice::from_raw_parts(ybuf.cpu_ptr as *const f32, out_dim)
        };
        let mut y = vec![0.0f32; out_dim];
        y.copy_from_slice(y_slice);

        // Flip buffer for next call
        self.active = 1 - self.active;

        Some(y)
    }

    /// SuperLinear forward: batched per-neuron matvec.
    /// Y[n*O+o] = B[n*O+o] + sum_k W[n*O*K+o*K+k] * X[n*K+k]
    /// Streams all weights+bias+input to GPU, dispatches superlinear_fwd kernel.
    pub fn superlinear(
        &mut self,
        dev: &mut HsaDevice,
        weights: &[f32], biases: &[f32], trace: &[f32],
        n_neurons: usize, out_per: usize, in_per: usize,
    ) -> Option<Vec<f32>> {
        let w_bytes = weights.len() * 4;
        let b_bytes = biases.len() * 4;
        let x_bytes = trace.len() * 4;
        let total_upload = w_bytes + b_bytes;
        let idx = self.active;

        // Weight + bias into one contiguous buffer
        if self.buf_cap[idx] < total_upload {
            let cap = ((total_upload + 4095) & !4095) as u64;
            self.buf[idx] = dev.alloc.alloc_vram(cap).ok();
            self.buf_cap[idx] = cap as usize;
        }
        let wbuf = self.buf[idx].as_ref()?;
        wbuf.write_f32(0, weights);
        wbuf.write_f32(w_bytes, biases);

        // Input trace
        if self.x_cap < x_bytes {
            let cap = ((x_bytes + 4095) & !4095).max(4096) as u64;
            self.x_buf = dev.alloc.alloc_vram(cap).ok();
            self.x_cap = cap as usize;
        }
        let xbuf = self.x_buf.as_ref()?;
        xbuf.write_f32(0, trace);

        // Output
        let y_count = n_neurons * out_per;
        let y_bytes = y_count * 4;
        if self.y_cap < y_bytes {
            let cap = ((y_bytes + 4095) & !4095).max(4096) as u64;
            self.y_buf = dev.alloc_output(cap as usize).ok();
            self.y_cap = cap as usize;
        }
        let ybuf = self.y_buf.as_ref()?;
        unsafe { std::ptr::write_bytes(ybuf.cpu_ptr, 0, y_bytes); }

        // Args
        if self.args_buf.is_none() {
            self.args_buf = dev.alloc.alloc_vram(4096).ok();
        }
        let args = self.args_buf.as_ref()?;

        let bias_va = wbuf.va_addr + w_bytes as u64;
        let mut kargs = [0u8; 48];
        kargs[0..8].copy_from_slice(&wbuf.va_addr.to_le_bytes());
        kargs[8..16].copy_from_slice(&bias_va.to_le_bytes());
        kargs[16..24].copy_from_slice(&xbuf.va_addr.to_le_bytes());
        kargs[24..32].copy_from_slice(&ybuf.va_addr.to_le_bytes());
        kargs[32..36].copy_from_slice(&(n_neurons as u32).to_le_bytes());
        kargs[36..40].copy_from_slice(&(out_per as u32).to_le_bytes());
        kargs[40..44].copy_from_slice(&(in_per as u32).to_le_bytes());
        args.write(0, &kargs[..44]);

        // Dispatch: 1 workitem per output element
        let total = (n_neurons * out_per) as u32;
        let nwg = (total + 255) / 256;
        if !dev.dispatch_enqueue("superlinear_fwd", args, [nwg, 1, 1], [256, 1, 1]) {
            return None;
        }
        if !dev.submit_wait(2000) {
            return None;
        }

        let y_slice = unsafe {
            std::slice::from_raw_parts(ybuf.cpu_ptr as *const f32, y_count)
        };
        let mut y = vec![0.0f32; y_count];
        y.copy_from_slice(y_slice);

        self.active = 1 - self.active;
        Some(y)
    }

    /// Stream-in transposed weights, compute d_input = W^T @ d_out.
    /// Same streaming pattern as matvec but with transposed weight.
    pub fn matvec_t(
        &mut self,
        dev: &mut HsaDevice,
        weight: &[f32],  // original W [out_dim × in_dim]
        d_out: &[f32],
        out_dim: usize, in_dim: usize,
    ) -> Option<Vec<f32>> {
        // Transpose on CPU (cheap relative to GPU compute)
        let mut w_t = vec![0.0f32; in_dim * out_dim];
        for i in 0..out_dim {
            for j in 0..in_dim {
                w_t[j * out_dim + i] = weight[i * in_dim + j];
            }
        }
        let bias_zero = vec![0.0f32; in_dim];

        // Dispatch as matvec(W^T, d_out) → d_input
        // W^T is [in_dim × out_dim], so output dim = in_dim
        self.matvec(dev, &w_t, &bias_zero, d_out, in_dim, out_dim)
    }
}
