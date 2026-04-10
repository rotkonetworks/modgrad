//! compute dispatch primitives.
//!
//! GpuFuture: poll gpu signal for async completion.
//! ComputeState: what the ctm feels about its compute (interoception).
//!
//! the ctm doesn't know about gpus. it just calls the harness
//! and feels whether compute is fast or slow.

use super::memory::GpuBuffer;
use std::time::Instant;

// ─── future ─────────────────────────────────────────────────

/// async gpu completion. poll or block.
pub struct GpuFuture {
    signal_ptr: *const u32,
    expected: u32,
    submitted_at: Instant,
}

unsafe impl Send for GpuFuture {}

impl GpuFuture {
    pub fn new(signal: &GpuBuffer, expected: u32) -> Self {
        GpuFuture {
            signal_ptr: signal.cpu_ptr as *const u32,
            expected,
            submitted_at: Instant::now(),
        }
    }

    /// true if gpu finished.
    pub fn poll(&self) -> bool {
        let val = unsafe { self.signal_ptr.read_volatile() };
        val >= self.expected
    }

    /// block until done. returns elapsed, or None on timeout.
    pub fn wait(&self, timeout_us: u64) -> Option<std::time::Duration> {
        let start = Instant::now();
        loop {
            if self.poll() {
                return Some(self.submitted_at.elapsed());
            }
            if start.elapsed().as_micros() as u64 > timeout_us {
                return None;
            }
            std::hint::spin_loop();
        }
    }
}

// ─── interoception ──────────────────────────────────────────

/// what the ctm feels about its compute.
/// read-only from organism side. harness updates it.
#[derive(Debug, Clone, Default)]
pub struct ComputeState {
    /// rolling average dispatch latency (microseconds)
    pub latency_us: f32,
    /// true when gpu dispatch is in flight
    pub gpu_busy: bool,
    /// total dispatches
    pub dispatch_count: u64,
    /// faults (should be 0)
    pub fault_count: u32,
}
