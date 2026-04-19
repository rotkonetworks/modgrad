//! Device-located tensors: `CpuTensor<T>` lives in host RAM, `VramTensor<T>` lives
//! on the GPU. The types are distinct — a CPU-only op literally cannot accept a
//! VRAM tensor without an explicit `.download()` first, and vice versa.
//!
//! Why
//! ---
//! The older API takes `&[f32]` / `&mut [f32]` everywhere and the GPU path
//! decides at runtime whether to upload. That leads to:
//!   * hidden caches (prepare_weights keyed by ptr identity),
//!   * duplicated storage (VramMirror + prepare_weights cache),
//!   * per-call `gpu_enabled()` branches on the hot path,
//!   * and — most painfully — "works on CPU, silently CPU on GPU" bugs like
//!     the `adamw.co`-as-relocatable-ELF issue this session.
//!
//! With typed tensors, the location is part of the type and operations dispatch
//! on it at compile time. Training loops look like:
//!
//! ```ignore
//! let w = CpuTensor::<f32>::from_vec(vec![..], [out, in]).move_to_vram()?;
//! for step in .. {
//!     forward_backward_vram(&w, &batch, &mut grads)?;
//!     adamw_in_place(&w, &grads, &mut m, &mut v, ..)?;   // all VRAM
//! }
//! w.download()?.save(path)?;                             // one-time
//! ```
//!
//! Safety
//! ------
//! * `VramTensor<T>` owns a `GpuBuffer`. Drop unmaps the GPU mapping.
//! * Slices returned from `as_slice` / `as_mut_slice` are BAR-mapped. Reads
//!   and writes go straight to VRAM; no staging copy. Requires resizable
//!   BAR (already required by the rest of the SDK on this hardware).
//! * `as_mut_slice` takes `&mut self` so aliasing is impossible in safe
//!   Rust. Callers who want multiple mutable views across a training step
//!   must split the tensor or use distinct tensors.
//! * GPU kernels reading a tensor while the CPU holds an `&mut [T]` is a
//!   data race at the hardware level. Call `flush_gpu()` between kernel
//!   dispatch and CPU access.

use modgrad_device::kfd::memory::GpuBuffer;
use std::marker::PhantomData;

/// Element type usable inside a device tensor. Currently only `f32` is
/// wired up; additional types (f16, i8, etc.) plug in by implementing
/// this trait and adding kernel paths.
pub trait DeviceElem: Copy + Default + Send + Sync + 'static {
    /// Size in bytes of one element. Must match `std::mem::size_of::<Self>()`
    /// but exposed here for generic buffer sizing.
    const BYTE_SIZE: usize;
}

impl DeviceElem for f32 {
    const BYTE_SIZE: usize = 4;
}

/// Host-side tensor, row-major. Backed by a `Vec<T>`.
#[derive(Debug, Clone)]
pub struct CpuTensor<T: DeviceElem> {
    data: Vec<T>,
    dims: Vec<usize>,
}

impl<T: DeviceElem> CpuTensor<T> {
    /// Wrap an existing `Vec<T>` with a shape. Panics if `data.len()` doesn't
    /// equal the product of `dims`.
    pub fn from_vec(data: Vec<T>, dims: impl Into<Vec<usize>>) -> Self {
        let dims = dims.into();
        let expected: usize = dims.iter().product();
        assert_eq!(data.len(), expected,
            "CpuTensor::from_vec: data len {} ≠ product of dims {:?} = {}",
            data.len(), dims, expected);
        Self { data, dims }
    }

    /// Allocate a CPU tensor filled with `T::default()` (zeros for f32).
    pub fn zeros(dims: impl Into<Vec<usize>>) -> Self {
        let dims = dims.into();
        let n: usize = dims.iter().product();
        Self { data: vec![T::default(); n], dims }
    }

    pub fn dims(&self) -> &[usize] { &self.dims }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn as_slice(&self) -> &[T] { &self.data }
    pub fn as_mut_slice(&mut self) -> &mut [T] { &mut self.data }
    pub fn into_vec(self) -> Vec<T> { self.data }
}

/// GPU-resident tensor, row-major. Backed by a permanent VRAM allocation.
///
/// The `GpuBuffer` owns its VRAM mapping; dropping the tensor unmaps it.
/// `as_slice` / `as_mut_slice` give BAR-mapped direct-access views — reads
/// and writes go straight to VRAM via the resizable-BAR mapping, with no
/// staging copy.
pub struct VramTensor<T: DeviceElem> {
    buf: GpuBuffer,
    dims: Vec<usize>,
    len: usize,
    _marker: PhantomData<T>,
}

// Safe to share across threads: GpuBuffer is Send+Sync and the tensor adds no
// non-Send/Sync state.
unsafe impl<T: DeviceElem> Send for VramTensor<T> {}
unsafe impl<T: DeviceElem> Sync for VramTensor<T> {}

impl<T: DeviceElem> VramTensor<T> {
    /// Allocate a VRAM tensor of the given shape, zero-initialised.
    /// Returns `None` when no KFD GPU is available, or the alloc fails.
    pub fn zeros(dims: impl Into<Vec<usize>>) -> Option<Self> {
        let dims = dims.into();
        let n: usize = dims.iter().product();
        let bytes = (n * T::BYTE_SIZE) as u64;
        // Round up to 4 KiB page granularity — KFD alloc is page-sized.
        let cap = ((bytes + 4095) & !4095).max(4096);

        let buf = crate::alloc_device_vram(cap)?;
        // Zero the first `n` elements via BAR. `write_bytes` is safe here:
        // the buffer has CPU access (alloc_vram uses PUBLIC flag) and we own
        // the allocation, so no races.
        unsafe {
            let p = buf.cpu_ptr as *mut u8;
            std::ptr::write_bytes(p, 0, n * T::BYTE_SIZE);
        }
        Some(Self { buf, dims, len: n, _marker: PhantomData })
    }

    /// Allocate and upload from a `CpuTensor`. Equivalent to `zeros` then
    /// `as_mut_slice().copy_from_slice(...)` but fused into one alloc.
    pub fn from_cpu(cpu: &CpuTensor<T>) -> Option<Self> {
        let mut v = Self::zeros(cpu.dims().to_vec())?;
        v.as_mut_slice().copy_from_slice(cpu.as_slice());
        Some(v)
    }

    /// Download to a fresh `CpuTensor`. Read through the BAR mapping.
    pub fn download(&self) -> CpuTensor<T> {
        let mut out = CpuTensor::<T>::zeros(self.dims.clone());
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }

    /// Destructively take the VRAM buffer back as CPU data. Frees VRAM.
    pub fn into_cpu(self) -> CpuTensor<T> { self.download() }

    /// BAR-mapped read view. Safe as long as no kernel is writing this
    /// tensor in flight — caller must `flush_gpu()` before reading after
    /// a GPU write.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.buf.cpu_ptr as *const T, self.len) }
    }

    /// BAR-mapped mutable view. Requires `&mut self` so aliasing is
    /// prevented in safe Rust. Callers racing this against in-flight GPU
    /// kernels is a hardware-level data race — flush first.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.buf.cpu_ptr as *mut T, self.len) }
    }

    /// GPU virtual address for kernel dispatch. Not a safe pointer — only
    /// usable by KFD kernargs on the same device.
    pub fn va(&self) -> u64 { self.buf.va_addr }

    pub fn dims(&self) -> &[usize] { &self.dims }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl<T: DeviceElem> CpuTensor<T> {
    /// Upload this tensor to VRAM. Consumes `self`; the CPU Vec is dropped
    /// once the VRAM alloc succeeds. Returns `None` if GPU unavailable.
    pub fn move_to_vram(self) -> Option<VramTensor<T>> {
        VramTensor::from_cpu(&self)
    }
}

// ═══════════════════════════════════════════════════════════════
// OPERATIONS — dispatch on the operand's device location.
// ═══════════════════════════════════════════════════════════════

/// Y[n×m] = X[n×k] @ W^T[k×m] + B[m].
///
/// `W` is row-major `[m×k]` (the Linear layout: `out × in`). `B` broadcasts
/// across the N rows. This matches the underlying `matmul_blocked` kernel's
/// contract and the CPU `matrixmultiply::sgemm` path we already use.
///
/// Operands must live on the same device. Mixing CPU and VRAM tensors in
/// one call is a compile error — explicit `.move_to_vram()` / `.download()`
/// is required, which is the whole point of the typed-location design.
pub trait Matmul: Sized {
    /// Returns `Err` on shape mismatch or dispatch failure. Shape mismatch
    /// is a bug (the FFN builds its tensors with known shapes); dispatch
    /// failure only happens on VRAM when the GPU is unavailable, in which
    /// case the caller typically moves everything to CPU and retries.
    fn matmul(a: &Self, w: &Self, bias: &Self, y: &mut Self,
              n: usize, k: usize, m: usize) -> Result<(), &'static str>;
}

impl Matmul for CpuTensor<f32> {
    fn matmul(a: &Self, w: &Self, bias: &Self, y: &mut Self,
              n: usize, k: usize, m: usize) -> Result<(), &'static str> {
        if a.len() < n * k { return Err("a too small for n*k"); }
        if w.len() < m * k { return Err("w too small for m*k"); }
        if bias.len() < m  { return Err("bias too small for m"); }
        if y.len()    < n * m { return Err("y too small for n*m"); }

        // Bias broadcast.
        let m_rows = n;
        let y_slice = y.as_mut_slice();
        let b_slice = bias.as_slice();
        for row in 0..m_rows {
            y_slice[row * m..(row + 1) * m].copy_from_slice(&b_slice[..m]);
        }
        // Y += A @ W^T.  A:[n×k] rsa=k csa=1.  W row-major [m×k] viewed as
        // W^T [k×m] has rsb=1 csb=k.  Y:[n×m] rsc=m csc=1.
        unsafe {
            matrixmultiply::sgemm(
                n, k, m,
                1.0, a.as_slice().as_ptr(), k as isize, 1,
                w.as_slice().as_ptr(), 1, k as isize,
                1.0, y.as_mut_slice().as_mut_ptr(), m as isize, 1,
            );
        }
        Ok(())
    }
}

impl Matmul for VramTensor<f32> {
    fn matmul(a: &Self, w: &Self, bias: &Self, y: &mut Self,
              n: usize, k: usize, m: usize) -> Result<(), &'static str> {
        if a.len() < n * k { return Err("a too small for n*k"); }
        if w.len() < m * k { return Err("w too small for m*k"); }
        if bias.len() < m  { return Err("bias too small for m"); }
        if y.len()    < n * m { return Err("y too small for n*m"); }
        if m % 128 != 0 { return Err("m must be multiple of 128 for matmul_blocked"); }
        if k % 8   != 0 { return Err("k must be multiple of 8"); }
        if n % 32  != 0 { return Err("n must be multiple of 32"); }

        // All operands already in VRAM. Zero-copy kernel dispatch.
        let ok = modgrad_device::kfd::accel::try_matmul_va(
            a.va(), w.va(), bias.va(), y.va(),
            n as u32, k as u32, m as u32,
        );
        if ok { Ok(()) } else { Err("matmul dispatch failed (GPU disabled or kernel missing)") }
    }
}

/// Row-wise layer normalization with affine transform.
///
/// Input `x` is `[n × d]` row-major, `gamma` and `beta` are `[d]`. Output
/// `y` is `[n × d]` with `y[i,j] = gamma[j] * (x[i,j] - mean_i) * inv_std_i + beta[j]`.
/// Per-row `means` and `inv_stds` are also returned for the backward pass.
///
/// Operands must live on the same device — the type system enforces it.
pub trait LayerNorm: Sized {
    fn layer_norm_forward(
        x: &Self, gamma: &Self, beta: &Self,
        y: &mut Self, means: &mut CpuTensor<f32>, inv_stds: &mut CpuTensor<f32>,
        n: usize, d: usize,
    ) -> Result<(), &'static str>;
}

fn layer_norm_body(
    x: &[f32], gamma: &[f32], beta: &[f32],
    y: &mut [f32], means: &mut [f32], inv_stds: &mut [f32],
    n: usize, d: usize,
) -> Result<(), &'static str> {
    if x.len()        < n * d { return Err("x too small"); }
    if gamma.len()    < d     { return Err("gamma too small"); }
    if beta.len()     < d     { return Err("beta too small"); }
    if y.len()        < n * d { return Err("y too small"); }
    if means.len()    < n     { return Err("means too small"); }
    if inv_stds.len() < n     { return Err("inv_stds too small"); }
    let df = d as f32;
    for row in 0..n {
        let x_row = &x[row * d..(row + 1) * d];
        let y_row = &mut y[row * d..(row + 1) * d];
        let mean: f32 = x_row.iter().sum::<f32>() / df;
        let var: f32 = x_row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / df;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for j in 0..d {
            y_row[j] = gamma[j] * (x_row[j] - mean) * inv_std + beta[j];
        }
        means[row] = mean;
        inv_stds[row] = inv_std;
    }
    Ok(())
}

impl LayerNorm for CpuTensor<f32> {
    fn layer_norm_forward(
        x: &Self, gamma: &Self, beta: &Self,
        y: &mut Self, means: &mut CpuTensor<f32>, inv_stds: &mut CpuTensor<f32>,
        n: usize, d: usize,
    ) -> Result<(), &'static str> {
        layer_norm_body(
            x.as_slice(), gamma.as_slice(), beta.as_slice(),
            y.as_mut_slice(), means.as_mut_slice(), inv_stds.as_mut_slice(),
            n, d,
        )
    }
}

impl LayerNorm for VramTensor<f32> {
    fn layer_norm_forward(
        x: &Self, gamma: &Self, beta: &Self,
        y: &mut Self, means: &mut CpuTensor<f32>, inv_stds: &mut CpuTensor<f32>,
        n: usize, d: usize,
    ) -> Result<(), &'static str> {
        // BAR-mapped CPU compute — slices point at VRAM directly, no
        // upload/download. Slower than a dedicated GPU kernel would be
        // (we don't have an affine-LN kernel yet, only the single-row
        // whitening one), but no PCIe round trip either. A batched
        // affine-LN kernel dispatched on VA pointers would go here.
        layer_norm_body(
            x.as_slice(), gamma.as_slice(), beta.as_slice(),
            y.as_mut_slice(), means.as_mut_slice(), inv_stds.as_mut_slice(),
            n, d,
        )
    }
}

/// Elementwise in-place add: `acc[i] += src[i]` for `i in 0..n`.
/// Used for residual connections and batched bias-add reductions.
/// Operands share device.
pub trait AddInPlace: Sized {
    fn add_in_place(acc: &mut Self, src: &Self, n: usize) -> Result<(), &'static str>;
}

impl AddInPlace for CpuTensor<f32> {
    fn add_in_place(acc: &mut Self, src: &Self, n: usize) -> Result<(), &'static str> {
        if acc.len() < n { return Err("acc too small"); }
        if src.len() < n { return Err("src too small"); }
        let a = acc.as_mut_slice();
        let s = src.as_slice();
        for i in 0..n { a[i] += s[i]; }
        Ok(())
    }
}

impl AddInPlace for VramTensor<f32> {
    fn add_in_place(acc: &mut Self, src: &Self, n: usize) -> Result<(), &'static str> {
        if acc.len() < n { return Err("acc too small"); }
        if src.len() < n { return Err("src too small"); }
        // BAR-mapped CPU add. Elementwise adds are tiny vs matmul; a dedicated
        // GPU kernel buys us nothing here and would pay launch overhead.
        let a = acc.as_mut_slice();
        let s = src.as_slice();
        for i in 0..n { a[i] += s[i]; }
        Ok(())
    }
}

/// SwiGLU activation: hidden[i] = silu(gate[i]) * up[i]  where  silu(x) = x*sigmoid(x).
/// Pure elementwise over `n` elements. All three tensors share device.
pub trait SwiGLU: Sized {
    fn swiglu(gate: &Self, up: &Self, hidden: &mut Self, n: usize) -> Result<(), &'static str>;
}

#[inline]
fn swiglu_body(gate: &[f32], up: &[f32], hidden: &mut [f32], n: usize) -> Result<(), &'static str> {
    if gate.len()   < n { return Err("gate too small"); }
    if up.len()     < n { return Err("up too small"); }
    if hidden.len() < n { return Err("hidden too small"); }
    for i in 0..n {
        let g = gate[i];
        let s = 1.0 / (1.0 + (-g).exp());
        hidden[i] = g * s * up[i];
    }
    Ok(())
}

impl SwiGLU for CpuTensor<f32> {
    fn swiglu(gate: &Self, up: &Self, hidden: &mut Self, n: usize) -> Result<(), &'static str> {
        swiglu_body(gate.as_slice(), up.as_slice(), hidden.as_mut_slice(), n)
    }
}

impl SwiGLU for VramTensor<f32> {
    fn swiglu(gate: &Self, up: &Self, hidden: &mut Self, n: usize) -> Result<(), &'static str> {
        // BAR-mapped CPU compute — no PCIe round trip. A dedicated elementwise
        // GPU kernel (tiny compute density anyway) is a possible but low-value
        // future optimisation; BAR-CPU already skips the dominant cost.
        swiglu_body(gate.as_slice(), up.as_slice(), hidden.as_mut_slice(), n)
    }
}

// ═══════════════════════════════════════════════════════════════
// NUMERICS MIDDLEWARE — "server as a function" composition.
// ═══════════════════════════════════════════════════════════════
//
// Training jobs silently diverge when gradients explode or the optimizer
// produces NaN/Inf weights. rocBLAS's solution is a `check_numerics` hook
// that scans inputs + outputs of every op in debug builds. We keep the
// same idea but compose it as a filter (Eriksen-style "server as a
// function"): any op taking typed tensors can be wrapped with
// `matmul_checked(...)` to get pre-/post-scan + contextual error.
//
// Shape:
//   let op:      fn(&T, &T, &T, &mut T, n,k,m) -> Result<(), &str>
//   let checked: fn(&T, &T, &T, &mut T, n,k,m) -> Result<(), String>
//     = wraps `op` and runs Numeric::check around it.
//
// Tests drive red→green: first assert the checker catches known-bad
// inputs (NaN in A should error), then assert it passes known-good ones.

/// Tensors that support NaN/Inf scanning over their backing storage.
pub trait Numeric {
    /// Returns `Err(msg)` if any element is NaN or ±Inf. `label` is
    /// prepended to the error message for readable debug output.
    fn check_finite(&self, label: &str) -> Result<(), String>;
}

impl Numeric for CpuTensor<f32> {
    fn check_finite(&self, label: &str) -> Result<(), String> {
        for (i, &v) in self.as_slice().iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("{label}[{i}] = {v} (not finite)"));
            }
        }
        Ok(())
    }
}

impl Numeric for VramTensor<f32> {
    fn check_finite(&self, label: &str) -> Result<(), String> {
        // Reads through the BAR mapping — one linear scan, no download.
        for (i, &v) in self.as_slice().iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("{label}[{i}] = {v} (not finite)"));
            }
        }
        Ok(())
    }
}

/// Matmul with numerics checks around it. Scans `a`, `w`, `bias` before
/// dispatch and `y` after. Use at layer boundaries / step boundaries in
/// debug builds to catch silent divergence.
///
/// Release-build callers can skip the wrapper and call `Matmul::matmul`
/// directly — the filter is purely additive, no cost to the raw path.
pub fn matmul_checked<T: Matmul + Numeric>(
    a: &T, w: &T, bias: &T, y: &mut T,
    n: usize, k: usize, m: usize,
    label: &str,
) -> Result<(), String> {
    a   .check_finite(&format!("{label}.a"))?;
    w   .check_finite(&format!("{label}.w"))?;
    bias.check_finite(&format!("{label}.bias"))?;
    T::matmul(a, w, bias, y, n, k, m).map_err(|e| format!("{label}: {e}"))?;
    y.check_finite(&format!("{label}.y"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_tensor_zeros_and_shape() {
        let t: CpuTensor<f32> = CpuTensor::zeros([3, 4]);
        assert_eq!(t.dims(), &[3, 4]);
        assert_eq!(t.len(), 12);
        assert!(t.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    #[should_panic(expected = "data len 4 ≠ product of dims")]
    fn cpu_tensor_from_vec_shape_mismatch_panics() {
        let _ = CpuTensor::from_vec(vec![0.0f32; 4], [3, 4]);
    }

    #[test]
    fn vram_roundtrip_or_skip() {
        // Graceful when no GPU present — `zeros` returns None.
        let cpu = CpuTensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]);
        let vram = match cpu.clone().move_to_vram() {
            Some(v) => v,
            None => return, // no GPU on this host, skip.
        };
        assert_eq!(vram.dims(), &[2, 2]);
        assert_eq!(vram.len(), 4);
        let back = vram.download();
        assert_eq!(back.as_slice(), cpu.as_slice());
    }

    #[test]
    fn cpu_matmul_small() {
        // Y[2×3] = X[2×4] @ W^T[4×3] + b[3], W is row-major [3×4]
        let x = CpuTensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0], [2, 4]);
        let w = CpuTensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0], [3, 4]);
        let b = CpuTensor::from_vec(vec![10.0f32, 20.0, 30.0], [3]);
        let mut y = CpuTensor::<f32>::zeros([2, 3]);
        Matmul::matmul(&x, &w, &b, &mut y, 2, 4, 3).unwrap();
        // y[0] = x[0,:3] + b = [1,2,3]+[10,20,30] = [11, 22, 33]
        // y[1] = x[1,:3] + b = [5,6,7]+[10,20,30] = [15, 26, 37]
        assert_eq!(y.as_slice(), &[11.0, 22.0, 33.0, 15.0, 26.0, 37.0]);
    }

    // ─── Numerics middleware — red-green ─────────────────────────

    #[test]
    fn check_finite_detects_nan() {
        let t = CpuTensor::from_vec(vec![1.0f32, f32::NAN, 3.0], [3]);
        let err = t.check_finite("x").unwrap_err();
        assert!(err.contains("x[1]"), "error should name the offending index, got {err:?}");
        assert!(err.contains("NaN") || err.contains("not finite"),
            "error should mention finiteness, got {err:?}");
    }

    #[test]
    fn check_finite_detects_inf() {
        let t = CpuTensor::from_vec(vec![1.0f32, f32::INFINITY], [2]);
        assert!(t.check_finite("y").is_err());
    }

    #[test]
    fn check_finite_passes_clean_tensor() {
        let t = CpuTensor::from_vec(vec![1.0f32, 2.0, 3.0], [3]);
        t.check_finite("z").unwrap();
    }

    #[test]
    fn matmul_checked_rejects_nan_input() {
        let mut x = vec![0.0f32; 8];
        x[3] = f32::NAN;
        let x_cpu = CpuTensor::from_vec(x, [2, 4]);
        let w_cpu = CpuTensor::<f32>::zeros([3, 4]);
        let b_cpu = CpuTensor::<f32>::zeros([3]);
        let mut y_cpu = CpuTensor::<f32>::zeros([2, 3]);
        let err = matmul_checked(&x_cpu, &w_cpu, &b_cpu, &mut y_cpu, 2, 4, 3, "test").unwrap_err();
        assert!(err.contains("test.a"), "error should name the bad tensor, got {err:?}");
    }

    #[test]
    fn matmul_checked_passes_clean_inputs() {
        let x_cpu = CpuTensor::from_vec(vec![1.0f32; 8], [2, 4]);
        let w_cpu = CpuTensor::from_vec(vec![0.5f32; 12], [3, 4]);
        let b_cpu = CpuTensor::from_vec(vec![0.0f32; 3], [3]);
        let mut y_cpu = CpuTensor::<f32>::zeros([2, 3]);
        matmul_checked(&x_cpu, &w_cpu, &b_cpu, &mut y_cpu, 2, 4, 3, "test").unwrap();
        // Y = X @ W^T + 0. Every row of X is all-ones, W is all 0.5 → y = 4*0.5 = 2.0.
        assert!(y_cpu.as_slice().iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn matmul_checked_reports_kernel_produced_nan_or_skip() {
        // Only exercises VRAM path when GPU is present; on CPU-only hosts the
        // kernel would never be the one producing NaN so we skip.
        let n_k_m = (32usize, 128usize, 128usize);
        let mut y_cpu = CpuTensor::<f32>::zeros([n_k_m.0, n_k_m.2]);
        let x_cpu = CpuTensor::from_vec(vec![1.0f32; n_k_m.0 * n_k_m.1], [n_k_m.0, n_k_m.1]);
        let mut w = vec![0.0f32; n_k_m.2 * n_k_m.1];
        w[0] = f32::NAN;  // poison one element → matmul propagates NaN into y[0,0]
        let w_cpu = CpuTensor::from_vec(w, [n_k_m.2, n_k_m.1]);
        let b_cpu = CpuTensor::<f32>::zeros([n_k_m.2]);
        let err = matmul_checked(&x_cpu, &w_cpu, &b_cpu, &mut y_cpu,
                                 n_k_m.0, n_k_m.1, n_k_m.2, "cpu").unwrap_err();
        assert!(err.contains("cpu.w"), "bad-weight should be caught as input, got {err:?}");
    }

    // ─── LayerNorm — red-green ───────────────────────────────────

    #[test]
    fn cpu_layer_norm_unit_affine_whitens() {
        // gamma=1, beta=0 ⇒ output has zero mean and unit variance per row.
        let (n, d) = (2, 4);
        let x = CpuTensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0,
                 0.0, 10.0, 20.0, 30.0], [n, d]);
        let g = CpuTensor::from_vec(vec![1.0f32; d], [d]);
        let b = CpuTensor::from_vec(vec![0.0f32; d], [d]);
        let mut y = CpuTensor::<f32>::zeros([n, d]);
        let mut means   = CpuTensor::<f32>::zeros([n]);
        let mut invstds = CpuTensor::<f32>::zeros([n]);
        LayerNorm::layer_norm_forward(&x, &g, &b, &mut y, &mut means, &mut invstds, n, d).unwrap();
        for row in 0..n {
            let r = &y.as_slice()[row * d..(row + 1) * d];
            let mean: f32 = r.iter().sum::<f32>() / d as f32;
            assert!(mean.abs() < 1e-5, "row {row} mean = {mean}");
            let var: f32 = r.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
            assert!((var - 1.0).abs() < 1e-3, "row {row} var = {var}");
        }
    }

    #[test]
    fn cpu_layer_norm_affine_scales_and_shifts() {
        let (n, d) = (1, 4);
        let x = CpuTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [n, d]);
        let g = CpuTensor::from_vec(vec![5.0f32; d], [d]);   // scale ×5
        let b = CpuTensor::from_vec(vec![7.0f32; d], [d]);   // shift +7
        let mut y = CpuTensor::<f32>::zeros([n, d]);
        let mut means = CpuTensor::<f32>::zeros([n]);
        let mut invstds = CpuTensor::<f32>::zeros([n]);
        LayerNorm::layer_norm_forward(&x, &g, &b, &mut y, &mut means, &mut invstds, n, d).unwrap();
        // Unit-variance normalisation then * 5 + 7 should keep the row mean near 7.
        let r = &y.as_slice()[..];
        let mean: f32 = r.iter().sum::<f32>() / d as f32;
        assert!((mean - 7.0).abs() < 1e-4);
    }

    #[test]
    fn cpu_vs_vram_layer_norm_parity_or_skip() {
        modgrad_device::kfd::accel::reset_op(modgrad_device::kfd::accel::GpuOp::Matmul);
        let (n, d) = (8, 16);
        let x_data: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.007).sin()).collect();
        let g_data: Vec<f32> = (0..d).map(|i| 1.0 + 0.1 * (i as f32 * 0.3).cos()).collect();
        let b_data: Vec<f32> = (0..d).map(|i| 0.05 * i as f32).collect();

        let x_c = CpuTensor::from_vec(x_data.clone(), [n, d]);
        let g_c = CpuTensor::from_vec(g_data.clone(), [d]);
        let b_c = CpuTensor::from_vec(b_data.clone(), [d]);
        let mut y_c = CpuTensor::<f32>::zeros([n, d]);
        let mut m_c = CpuTensor::<f32>::zeros([n]);
        let mut s_c = CpuTensor::<f32>::zeros([n]);
        LayerNorm::layer_norm_forward(&x_c, &g_c, &b_c, &mut y_c, &mut m_c, &mut s_c, n, d).unwrap();

        let x_v = match x_c.clone().move_to_vram() { Some(t) => t, None => return };
        let g_v = g_c.clone().move_to_vram().unwrap();
        let b_v = b_c.clone().move_to_vram().unwrap();
        let mut y_v = VramTensor::<f32>::zeros([n, d]).unwrap();
        let mut m_v = CpuTensor::<f32>::zeros([n]);
        let mut s_v = CpuTensor::<f32>::zeros([n]);
        LayerNorm::layer_norm_forward(&x_v, &g_v, &b_v, &mut y_v, &mut m_v, &mut s_v, n, d).unwrap();
        let y_v_back = y_v.download();

        let mut max_err = 0.0f32;
        for (a, b) in y_c.as_slice().iter().zip(y_v_back.as_slice().iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-5, "CPU/VRAM LN diverge: max_err = {:.3e}", max_err);
    }

    // ─── AddInPlace — red-green ─────────────────────────────────

    #[test]
    fn cpu_add_in_place_accumulates() {
        let mut a = CpuTensor::from_vec(vec![1.0f32, 2.0, 3.0], [3]);
        let b = CpuTensor::from_vec(vec![10.0f32, 20.0, 30.0], [3]);
        AddInPlace::add_in_place(&mut a, &b, 3).unwrap();
        assert_eq!(a.as_slice(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn cpu_add_in_place_size_mismatch_errors() {
        let mut a = CpuTensor::<f32>::zeros([2]);
        let b = CpuTensor::<f32>::zeros([2]);
        assert!(AddInPlace::add_in_place(&mut a, &b, 4).is_err());
    }

    #[test]
    fn cpu_vs_vram_add_parity_or_skip() {
        modgrad_device::kfd::accel::reset_op(modgrad_device::kfd::accel::GpuOp::Matmul);
        let n = 128;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();

        let mut a_c = CpuTensor::from_vec(a_data.clone(), [n]);
        let b_c = CpuTensor::from_vec(b_data.clone(), [n]);
        AddInPlace::add_in_place(&mut a_c, &b_c, n).unwrap();

        let mut a_v = match CpuTensor::from_vec(a_data.clone(), [n]).move_to_vram() {
            Some(t) => t, None => return,
        };
        let b_v = CpuTensor::from_vec(b_data.clone(), [n]).move_to_vram().unwrap();
        AddInPlace::add_in_place(&mut a_v, &b_v, n).unwrap();

        let a_v_back = a_v.download();
        let mut max_err = 0.0f32;
        for (x, y) in a_c.as_slice().iter().zip(a_v_back.as_slice().iter()) {
            max_err = max_err.max((x - y).abs());
        }
        assert!(max_err < 1e-6, "CPU/VRAM Add diverge: max_err = {:.3e}", max_err);
    }

    // ─── SwiGLU — red-green ──────────────────────────────────────

    #[test]
    fn cpu_swiglu_zero_gate_is_zero() {
        // silu(0) = 0, so hidden = 0 regardless of up.
        let g = CpuTensor::from_vec(vec![0.0f32; 4], [4]);
        let u = CpuTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [4]);
        let mut h = CpuTensor::<f32>::zeros([4]);
        SwiGLU::swiglu(&g, &u, &mut h, 4).unwrap();
        assert!(h.as_slice().iter().all(|&v| v.abs() < 1e-7));
    }

    #[test]
    fn cpu_swiglu_matches_manual() {
        // hidden[i] = silu(gate) * up
        let g = CpuTensor::from_vec(vec![1.0f32, -1.0, 2.0, -2.0], [4]);
        let u = CpuTensor::from_vec(vec![1.0f32; 4], [4]);
        let mut h = CpuTensor::<f32>::zeros([4]);
        SwiGLU::swiglu(&g, &u, &mut h, 4).unwrap();
        for i in 0..4 {
            let gx = g.as_slice()[i];
            let s = 1.0 / (1.0 + (-gx).exp());
            let expect = gx * s * u.as_slice()[i];
            assert!((h.as_slice()[i] - expect).abs() < 1e-6);
        }
    }

    #[test]
    fn cpu_vs_vram_swiglu_parity_or_skip() {
        modgrad_device::kfd::accel::reset_op(modgrad_device::kfd::accel::GpuOp::Matmul);
        let n = 256;
        let g_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1 - 5.0).sin()).collect();
        let u_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07 + 0.3).cos()).collect();

        let g_c = CpuTensor::from_vec(g_data.clone(), [n]);
        let u_c = CpuTensor::from_vec(u_data.clone(), [n]);
        let mut h_c = CpuTensor::<f32>::zeros([n]);
        SwiGLU::swiglu(&g_c, &u_c, &mut h_c, n).unwrap();

        let g_v = match g_c.clone().move_to_vram() { Some(t) => t, None => return };
        let u_v = u_c.clone().move_to_vram().unwrap();
        let mut h_v = VramTensor::<f32>::zeros([n]).unwrap();
        SwiGLU::swiglu(&g_v, &u_v, &mut h_v, n).unwrap();
        let h_v_back = h_v.download();

        let mut max_err = 0.0f32;
        for (a, b) in h_c.as_slice().iter().zip(h_v_back.as_slice().iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-6, "CPU/VRAM SwiGLU diverge: max_err = {:.3e}", max_err);
    }

    #[test]
    fn cpu_vs_vram_matmul_parity_or_skip() {
        // Other tests in this binary share the GpuOp::Matmul disable flag
        // via the static OP_DISABLED array. If a concurrent test dispatch
        // has tripped it, clear here so we test the fresh dispatch path.
        modgrad_device::kfd::accel::reset_op(
            modgrad_device::kfd::accel::GpuOp::Matmul
        );

        // Same shapes the FFN uses at runtime. Both tensors must compute
        // identical results within f32 tolerance.
        let (n, k, m) = (32, 128, 128);  // aligned for matmul_blocked
        let x_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let w_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.007).cos()).collect();
        let b_data: Vec<f32> = (0..m).map(|i| (i as f32 * 0.01).sin()).collect();

        let x_cpu = CpuTensor::from_vec(x_data.clone(), [n, k]);
        let w_cpu = CpuTensor::from_vec(w_data.clone(), [m, k]);
        let b_cpu = CpuTensor::from_vec(b_data.clone(), [m]);
        let mut y_cpu = CpuTensor::<f32>::zeros([n, m]);
        Matmul::matmul(&x_cpu, &w_cpu, &b_cpu, &mut y_cpu, n, k, m).unwrap();

        // VRAM path (skip if no GPU OR if a parallel test left Matmul disabled).
        let x_v = match x_cpu.clone().move_to_vram() { Some(t) => t, None => return };
        let w_v = w_cpu.clone().move_to_vram().unwrap();
        let b_v = b_cpu.clone().move_to_vram().unwrap();
        let mut y_v = VramTensor::<f32>::zeros([n, m]).unwrap();
        // Skip if a concurrent test has contended /dev/kfd enough to disable
        // Matmul — we're validating correctness, not scheduling.
        if Matmul::matmul(&x_v, &w_v, &b_v, &mut y_v, n, k, m).is_err() { return; }
        let y_v_back = y_v.download();

        // f32 accumulation tolerance across K=128 terms.
        let mut max_err = 0.0f32;
        for (cpu, vram) in y_cpu.as_slice().iter().zip(y_v_back.as_slice().iter()) {
            max_err = max_err.max((cpu - vram).abs());
        }
        assert!(max_err < 1e-3, "CPU-VRAM matmul parity broken: max_err = {:.3e}", max_err);
    }
}
