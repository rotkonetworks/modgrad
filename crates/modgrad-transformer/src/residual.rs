//! Per-layer residual stream with x0 shortcuts and mid-layer backout.
//!
//! Each layer has:
//!   resid_lambda: scales the block output before adding to residual
//!   x0_lambda: scales the original input (x0) and adds to residual
//!
//! resid_lambda linearly interpolated from resid_start (layer 0) to resid_end (last layer).
//! x0_lambda linearly interpolated from x0_start to x0_end.
//!
//! Mid-layer backout: at layer n/2, cache the hidden state. After the final layer,
//! subtract backout_lambda * cached_midpoint from the output.

use super::config::ResidualConfig;
use super::dims::*;

/// Pre-computed per-layer residual lambdas.
pub struct ResidualLambdas {
    /// Per-layer resid scale `[num_layers]`.
    pub resid: Vec<f32>,
    /// Per-layer x0 scale `[num_layers]`.
    pub x0: Vec<f32>,
    /// Backout lambda.
    pub backout_lambda: f32,
    /// Midpoint layer index.
    pub midpoint: usize,
}

impl ResidualLambdas {
    /// Compute lambdas from config.
    pub fn from_config(config: &ResidualConfig, num_layers: NumLayers) -> Self {
        let n = num_layers.get();
        let mut resid = Vec::with_capacity(n);
        let mut x0 = Vec::with_capacity(n);

        for i in 0..n {
            let frac = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
            resid.push(config.resid_start + frac * (config.resid_end - config.resid_start));
            x0.push(config.x0_start + frac * (config.x0_end - config.x0_start));
        }

        Self {
            resid,
            x0,
            backout_lambda: config.backout_lambda,
            midpoint: n / 2,
        }
    }

    /// Apply residual connection after a sub-layer (attention or MLP).
    ///
    /// `hidden`: current hidden state — modified in-place.
    /// `block_output`: output of the sub-layer.
    /// `x0`: original input to the entire model (for shortcut).
    /// `layer`: which layer we're in.
    #[inline]
    pub fn apply(
        &self,
        hidden: &mut [f32],
        block_output: &[f32],
        x0: &[f32],
        layer: LayerIdx,
    ) {
        let li = layer.get();
        let r = self.resid[li];
        let x = self.x0[li];
        for i in 0..hidden.len() {
            hidden[i] = hidden[i] + r * block_output[i] + x * x0[i];
        }
    }
}

/// Cross-block context threaded through the forward pass.
/// Solves the "blocks are independent filters but backout needs mid-layer state" problem.
///
/// Supports both single-token (decode) and multi-token (prefill) modes.
/// In prefill mode, `x0` is `[seq_len * model_dim]` and midpoint_hidden
/// caches the first token's hidden (representative).
pub struct ForwardCtx {
    /// Cached hidden state at the midpoint layer (for backout).
    /// Single token: `[model_dim]`. Prefill: `[model_dim]` (first token).
    pub midpoint_hidden: Vec<f32>,
    /// Whether midpoint has been cached this pass.
    pub midpoint_cached: bool,
    /// Original input x0 (for x0 lambda shortcut).
    /// Decode: `[model_dim]`. Prefill: `[seq_len * model_dim]`.
    pub x0: Vec<f32>,
    model_dim: usize,
}

impl ForwardCtx {
    pub fn new(model_dim: usize) -> Self {
        Self {
            midpoint_hidden: vec![0.0; model_dim],
            midpoint_cached: false,
            x0: vec![0.0; model_dim],
            model_dim,
        }
    }

    /// Set x0 for single-token decode.
    pub fn set_x0_one(&mut self, hidden: &[f32]) {
        self.x0.resize(self.model_dim, 0.0);
        self.x0.copy_from_slice(hidden);
    }

    /// Set x0 for batched prefill: `[seq_len * model_dim]`.
    pub fn set_x0_batch(&mut self, hidden: &[f32], seq_len: usize) {
        let total = seq_len * self.model_dim;
        self.x0.resize(total, 0.0);
        self.x0[..total].copy_from_slice(&hidden[..total]);
    }

    /// Cache the current hidden state if this is the midpoint layer.
    pub fn maybe_cache_midpoint(&mut self, hidden: &[f32], layer: LayerIdx, midpoint: usize) {
        if layer.get() == midpoint && !self.midpoint_cached {
            // Only cache first model_dim elements (first token or single token)
            self.midpoint_hidden.copy_from_slice(&hidden[..self.model_dim]);
            self.midpoint_cached = true;
        }
    }

    /// Apply backout subtraction to the final hidden state.
    pub fn apply_backout(&self, hidden: &mut [f32], backout_lambda: f32) {
        if self.midpoint_cached {
            for i in 0..self.model_dim.min(hidden.len()) {
                hidden[i] -= backout_lambda * self.midpoint_hidden[i];
            }
        }
    }

    /// Apply backout to all tokens in a batch.
    pub fn apply_backout_batch(&self, hidden: &mut [f32], seq_len: usize, backout_lambda: f32) {
        if self.midpoint_cached {
            let md = self.model_dim;
            for t in 0..seq_len {
                for i in 0..md {
                    hidden[t * md + i] -= backout_lambda * self.midpoint_hidden[i];
                }
            }
        }
    }

    /// Reset for a new forward pass.
    pub fn reset(&mut self) {
        self.midpoint_cached = false;
    }
}
