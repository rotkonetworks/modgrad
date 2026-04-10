//! kfd gpu driver integration tests.
//!
//! run: cargo test --release --test kfd_gpu -- --nocapture
//!
//! these tests require /dev/kfd (amd gpu with kfd driver).
//! they share a single HsaDevice to avoid acquire_vm conflicts.
//! each test validates correctness against a cpu reference.

use modgrad::kfd::{self, HsaDevice};
use modgrad::kfd::dispatch::KernArgs;
use std::sync::OnceLock;

/// single shared gpu device for all tests.
/// tests run sequentially (--test-threads=1 enforced by kfd constraint).
static GPU: OnceLock<Option<std::sync::Mutex<HsaDevice>>> = OnceLock::new();

fn gpu() -> Option<std::sync::MutexGuard<'static, HsaDevice>> {
    GPU.get_or_init(|| {
        if !kfd::is_available() { return None; }
        HsaDevice::open().ok().map(std::sync::Mutex::new)
    }).as_ref().map(|m| m.lock().unwrap())
}

// ─── store kernel ───────────────────────────────────────────

#[test]
fn store42_writes_correct_values() {
    let mut dev = match gpu() { Some(d) => d, None => return };

    let n = 32u32;
    let y = dev.alloc.alloc_userptr_public((n as u64) * 4).unwrap();
    // write sentinel — must be overwritten
    y.write_f32(0, &vec![-1.0f32; n as usize]);

    let mut args = KernArgs::new();
    args.push_ptr(&y);
    let args_buf = args.upload(&dev.alloc).unwrap();

    assert!(dev.dispatch_kernel("test_store", &args_buf, [n, 1, 1], [n, 1, 1]),
        "dispatch timed out");

    let result = y.read_f32(0, n as usize);
    for i in 0..n as usize {
        assert_eq!(result[i], 42.0, "y[{i}] = {} (sentinel was -1.0)", result[i]);
    }
}

// ─── matvec kernel ──────────────────────────────────────────

/// dispatch matvec and return output vector.
fn gpu_matvec(dev: &mut HsaDevice, w: &[f32], b: &[f32], x: &[f32],
              out_dim: u32, in_dim: u32) -> Vec<f32> {
    let w_buf = dev.upload_f32(w).unwrap();
    let b_buf = dev.upload_f32(b).unwrap();
    let x_buf = dev.upload_f32(x).unwrap();
    let y_buf = dev.alloc.alloc_userptr_public(((out_dim as u64) * 4 + 4095) & !4095).unwrap();

    let mut args = KernArgs::new();
    args.push_ptr(&w_buf);
    args.push_ptr(&b_buf);
    args.push_ptr(&x_buf);
    args.push_ptr(&y_buf);
    args.push_u32(out_dim);
    args.push_u32(in_dim);
    let args_buf = args.upload(&dev.alloc).unwrap();

    let block = out_dim.min(256);
    assert!(dev.dispatch_kernel("matvec", &args_buf, [out_dim, 1, 1], [block, 1, 1]),
        "matvec dispatch timed out ({}x{})", out_dim, in_dim);

    y_buf.read_f32(0, out_dim as usize)
}

/// cpu reference: y = W*x + b
fn cpu_matvec(w: &[f32], b: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let mut sum = b[row];
        for col in 0..in_dim {
            sum += w[row * in_dim + col] * x[col];
        }
        y[row] = sum;
    }
    y
}

fn assert_close(gpu: &[f32], cpu: &[f32], tol: f32, label: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch {} vs {}", gpu.len(), cpu.len());
    for i in 0..gpu.len() {
        let err = (gpu[i] - cpu[i]).abs();
        assert!(err < tol, "{label}: y[{i}] gpu={} cpu={} err={err}", gpu[i], cpu[i]);
    }
}

#[test]
fn matvec_identity_4x4() {
    let mut dev = match gpu() { Some(d) => d, None => return };
    let n = 4;
    let mut w = vec![0.0f32; n * n];
    for i in 0..n { w[i * n + i] = 1.0; }
    let b = vec![0.5; n];
    let x = vec![1.0, 2.0, 3.0, 4.0];

    let gpu_y = gpu_matvec(&mut dev, &w, &b, &x, n as u32, n as u32);
    let cpu_y = cpu_matvec(&w, &b, &x, n, n);
    assert_close(&gpu_y, &cpu_y, 1e-5, "identity_4x4");
}

#[test]
fn matvec_dense_2x3() {
    let mut dev = match gpu() { Some(d) => d, None => return };
    let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
    let b = vec![0.1, 0.2];
    let x = vec![1.0, 1.0, 1.0];

    let gpu_y = gpu_matvec(&mut dev, &w, &b, &x, 2, 3);
    let cpu_y = cpu_matvec(&w, &b, &x, 2, 3);
    assert_close(&gpu_y, &cpu_y, 1e-4, "dense_2x3");
}

#[test]
fn matvec_32x32_vs_cpu() {
    let mut dev = match gpu() { Some(d) => d, None => return };
    let n = 32usize;

    // deterministic pseudo-random weights
    let w: Vec<f32> = (0..n*n).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 10) as f32 * 0.1).collect();
    let x: Vec<f32> = (0..n).map(|i| ((i * 13 + 5) % 100) as f32 / 100.0).collect();

    let gpu_y = gpu_matvec(&mut dev, &w, &b, &x, n as u32, n as u32);
    let cpu_y = cpu_matvec(&w, &b, &x, n, n);
    assert_close(&gpu_y, &cpu_y, 1e-3, "random_32x32");
}

#[test]
fn matvec_nonsquare_8x16() {
    let mut dev = match gpu() { Some(d) => d, None => return };
    let out = 8usize;
    let inp = 16usize;

    let w = vec![1.0f32; out * inp]; // all ones
    let b = vec![0.0; out];
    let x = vec![1.0; inp]; // all ones

    let gpu_y = gpu_matvec(&mut dev, &w, &b, &x, out as u32, inp as u32);
    let cpu_y = cpu_matvec(&w, &b, &x, out, inp);
    assert_close(&gpu_y, &cpu_y, 1e-3, "nonsquare_8x16");

    // each row should sum to 16.0 (16 ones)
    for i in 0..out {
        assert!((gpu_y[i] - 16.0).abs() < 1e-3, "y[{i}] = {}", gpu_y[i]);
    }
}

#[test]
fn matvec_scalar_1x1() {
    let mut dev = match gpu() { Some(d) => d, None => return };
    let gpu_y = gpu_matvec(&mut dev, &[3.0], &[0.5], &[2.0], 1, 1);
    assert!((gpu_y[0] - 6.5).abs() < 1e-5, "1x1: {} expected 6.5", gpu_y[0]);
}

// ─── async future ───────────────────────────────────────────

#[test]
fn dispatch_async_returns_future() {
    use modgrad::kfd::compute::GpuFuture;

    let mut dev = match gpu() { Some(d) => d, None => return };

    let y = dev.alloc.alloc_userptr_public(4096).unwrap();
    let mut args = KernArgs::new();
    args.push_ptr(&y);
    let args_buf = args.upload(&dev.alloc).unwrap();

    let future = dev.dispatch_async("test_store", &args_buf, [4, 1, 1], [4, 1, 1]);
    assert!(future.is_some(), "dispatch_async returned None");

    let future = future.unwrap();
    assert!(!future.poll() || true, "poll should not panic"); // may already be done
    let elapsed = future.wait(1_000_000);
    assert!(elapsed.is_some(), "future timed out after 1s");

    let result = y.read_f32(0, 4);
    assert_eq!(result, vec![42.0; 4]);
}

#[test]
fn dispatch_nonexistent_kernel_returns_none() {
    let mut dev = match gpu() { Some(d) => d, None => return };
    let y = dev.alloc.alloc_userptr_public(4096).unwrap();
    let mut args = KernArgs::new();
    args.push_ptr(&y);
    let args_buf = args.upload(&dev.alloc).unwrap();

    let future = dev.dispatch_async("nonexistent_kernel", &args_buf, [1, 1, 1], [1, 1, 1]);
    assert!(future.is_none(), "should return None for unknown kernel");
}
