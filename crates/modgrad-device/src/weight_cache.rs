//! GPU-resident weight cache: weights uploaded once, kept in VRAM.
//!
//! Keyed by WeightId (stable across optimizer steps, unlike pointer identity).
//! Generation counter drives coherence: CPU writes bump gen, GPU re-uploads
//! when stale. No per-call memcpy — dispatch is just "run kernel".
//!
//! Backend-agnostic: works with KFD (AMD), CUDA (NVIDIA), Vulkan, or CPU-only.

use crate::kfd::memory::GpuBuffer;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Stable identity for a weight matrix's GPU cache entry.
/// Re-exported from modgrad-compute for convenience.
pub type WeightId = u64;

/// GPU-resident copy of one Linear layer's parameters.
pub struct DeviceEntry {
    /// Last CPU generation that was uploaded.
    pub uploaded_gen: u64,
    /// W [out_dim × in_dim] row-major in VRAM.
    pub weight: GpuBuffer,
    /// W^T [in_dim × out_dim] for backward dx.
    pub weight_t: GpuBuffer,
    /// Bias [out_dim].
    pub bias: GpuBuffer,
    /// Zero bias [in_dim] for backward dispatch.
    pub bias_zero: GpuBuffer,
    pub out_dim: usize,
    pub in_dim: usize,
}

/// The global GPU weight registry.
pub struct DeviceWeightCache {
    entries: HashMap<u64, DeviceEntry>,
    /// Reusable scratch buffers (avoid alloc per dispatch).
    pub scratch_x: Option<GpuBuffer>,
    pub scratch_x_cap: usize,
    pub scratch_y: Option<GpuBuffer>,
    pub scratch_y_cap: usize,
    pub scratch_args: Option<GpuBuffer>,
}

impl DeviceWeightCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            scratch_x: None, scratch_x_cap: 0,
            scratch_y: None, scratch_y_cap: 0,
            scratch_args: None,
        }
    }

    /// Ensure GPU has current weights. Returns true if entry exists and is current.
    pub fn ensure_uploaded(
        &mut self,
        id: u64, cpu_gen: u64,
        weight: &[f32], bias: &[f32],
        out_dim: usize, in_dim: usize,
    ) -> bool {
        if let Some(entry) = self.entries.get(&id) {
            if entry.uploaded_gen >= cpu_gen {
                return true; // GPU copy is current
            }
        }

        // Need to upload (first time or stale)
        let dev = match crate::kfd::accel::get_device() {
            Some(d) => d,
            None => return false,
        };

        let w_buf = match dev.upload_f32(weight) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let b_buf = match dev.upload_f32(bias) {
            Ok(b) => b,
            Err(_) => return false,
        };

        // Transpose for backward
        let mut w_t_data = vec![0.0f32; in_dim * out_dim];
        for i in 0..out_dim {
            for j in 0..in_dim {
                w_t_data[j * out_dim + i] = weight[i * in_dim + j];
            }
        }
        let w_t_buf = match dev.upload_f32(&w_t_data) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let b_zero = match dev.upload_f32(&vec![0.0f32; in_dim]) {
            Ok(b) => b,
            Err(_) => return false,
        };

        self.entries.insert(id, DeviceEntry {
            uploaded_gen: cpu_gen,
            weight: w_buf,
            weight_t: w_t_buf,
            bias: b_buf,
            bias_zero: b_zero,
            out_dim, in_dim,
        });
        true
    }

    /// Get GPU buffers for a weight. None if not uploaded.
    pub fn get(&self, id: u64) -> Option<&DeviceEntry> {
        self.entries.get(&id)
    }

    /// Remove entry (on weight drop or invalidation).
    pub fn remove(&mut self, id: u64) {
        self.entries.remove(&id);
    }

    /// Invalidate all entries (e.g., after optimizer step that mutated CPU weights).
    pub fn invalidate_all(&mut self) {
        self.entries.clear();
    }
}

/// Global singleton.
static CACHE: OnceLock<Mutex<DeviceWeightCache>> = OnceLock::new();

pub fn weight_cache() -> &'static Mutex<DeviceWeightCache> {
    CACHE.get_or_init(|| Mutex::new(DeviceWeightCache::new()))
}
