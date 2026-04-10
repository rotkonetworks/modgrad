//! Weight offloading: CPU RAM ↔ VRAM streaming.
//!
//! Enables models larger than VRAM by keeping weights in system memory (GTT)
//! and streaming per-layer to VRAM for compute. Double-buffered: while GPU
//! processes layer N from VRAM slot A, CPU copies layer N+1 into VRAM slot B.
//!
//! Layout:
//!   - GpuWeightSlot: a VRAM buffer sized for the largest single-layer weights
//!   - WeightOffloader: manages GTT storage + two VRAM slots
//!   - GpuTransformerBackend: implements TransformerOps with GPU dispatch


/// Memory budget for the offloader.
#[derive(Debug, Clone)]
pub struct OffloadBudget {
    /// Total VRAM available for weight slots (bytes).
    pub vram_budget: u64,
    /// Maximum size of a single layer's weights (bytes).
    pub max_layer_bytes: u64,
    /// Number of layers.
    pub num_layers: usize,
    /// Whether to pin weights in VRAM (no offload, if they fit).
    pub pin_if_fits: bool,
}

impl OffloadBudget {
    /// Compute the budget from model config and device specs.
    pub fn from_config(
        config: &super::config::GptConfig,
        vram_available: u64,
    ) -> Self {
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let mlp_dim = config.mlp_dim.get();

        // Per-layer weight sizes (in f32 elements):
        //   wq: md*md, wk: kv_dim*md, wv: kv_dim*md, wo: md*md
        //   mlp_fc: mlp_dim*md, mlp_proj: md*mlp_dim
        let attn_elems = 2 * md * md + 2 * kv_dim * md;
        let mlp_elems = 2 * mlp_dim * md;
        let max_layer_elems = attn_elems + mlp_elems;
        let max_layer_bytes = (max_layer_elems * 4) as u64;

        // Total model weight size
        let vocab = config.vocab_size.get();
        let embed_bytes = (2 * vocab * md * 4) as u64; // embed + lm_head
        let total_bytes = embed_bytes + max_layer_bytes * config.num_layers.get() as u64;

        // Reserve 20% of VRAM for activations/KV cache
        let usable_vram = (vram_available as f64 * 0.8) as u64;

        Self {
            vram_budget: usable_vram,
            max_layer_bytes,
            num_layers: config.num_layers.get(),
            pin_if_fits: total_bytes <= usable_vram,
        }
    }

    /// How many layers can be pinned in VRAM simultaneously?
    pub fn pinnable_layers(&self) -> usize {
        if self.max_layer_bytes == 0 { return self.num_layers; }
        (self.vram_budget / self.max_layer_bytes) as usize
    }

    /// Strategy summary.
    pub fn strategy(&self) -> OffloadStrategy {
        if self.pin_if_fits {
            OffloadStrategy::AllPinned
        } else {
            let pinnable = self.pinnable_layers();
            if pinnable >= 2 {
                OffloadStrategy::DoubleBuffered { pinnable }
            } else {
                OffloadStrategy::SingleBuffered
            }
        }
    }
}

/// How weights are managed between RAM and VRAM.
#[derive(Debug, Clone)]
pub enum OffloadStrategy {
    /// All weights fit in VRAM — no streaming needed.
    AllPinned,
    /// Double-buffered: prefetch next layer while computing current.
    DoubleBuffered { pinnable: usize },
    /// Single buffer: copy one layer at a time (slowest).
    SingleBuffered,
}

/// Per-layer weight data stored in CPU RAM.
/// Ready to be memcpy'd to a VRAM slot.
pub struct CpuLayerWeights {
    /// Concatenated weight data for this layer: [wq|wk|wv|wo|fc|proj]
    pub data: Vec<f32>,
    /// Byte offsets within `data` for each weight tensor.
    pub offsets: LayerOffsets,
}

/// Byte offsets into the concatenated layer weight buffer.
#[derive(Debug, Clone, Copy)]
pub struct LayerOffsets {
    pub wq: usize,
    pub wk: usize,
    pub wv: usize,
    pub wo: usize,
    pub mlp_fc: usize,
    pub mlp_proj: usize,
    /// Total number of f32 elements.
    pub total_elems: usize,
}

impl LayerOffsets {
    pub fn compute(
        model_dim: usize,
        kv_dim: usize,
        mlp_dim: usize,
    ) -> Self {
        let wq = 0;
        let wk = model_dim * model_dim;
        let wv = wk + kv_dim * model_dim;
        let wo = wv + kv_dim * model_dim;
        let mlp_fc = wo + model_dim * model_dim;
        let mlp_proj = mlp_fc + mlp_dim * model_dim;
        let total = mlp_proj + model_dim * mlp_dim;

        Self {
            wq, wk, wv, wo, mlp_fc, mlp_proj,
            total_elems: total,
        }
    }
}

/// Weight offloader: manages layer weights in CPU RAM with VRAM streaming.
pub struct WeightOffloader {
    /// Per-layer weights in CPU RAM.
    pub layers: Vec<CpuLayerWeights>,
    /// Embedding weights (always in CPU RAM, small enough for direct use).
    pub embed: Vec<f32>,
    /// LM head weights.
    pub lm_head: Vec<f32>,
    /// Offsets for slicing into concatenated layer data.
    pub offsets: LayerOffsets,
    /// Budget/strategy.
    pub budget: OffloadBudget,
}

impl WeightOffloader {
    /// Build from GptWeights — moves weight data into the offloader.
    pub fn from_weights(
        weights: super::weights::GptWeights,
        config: &super::config::GptConfig,
        vram_available: u64,
    ) -> Self {
        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let mlp_dim = config.mlp_dim.get();
        let offsets = LayerOffsets::compute(md, kv_dim, mlp_dim);

        let budget = OffloadBudget::from_config(config, vram_available);

        let layers: Vec<CpuLayerWeights> = weights.blocks.into_iter().map(|bw| {
            // Concatenate all layer weights into one contiguous buffer
            let mut data = Vec::with_capacity(offsets.total_elems);
            data.extend_from_slice(&bw.wq);
            data.extend_from_slice(&bw.wk);
            data.extend_from_slice(&bw.wv);
            data.extend_from_slice(&bw.wo);
            data.extend_from_slice(&bw.mlp_fc);
            data.extend_from_slice(&bw.mlp_proj);
            debug_assert_eq!(data.len(), offsets.total_elems);

            CpuLayerWeights { data, offsets }
        }).collect();

        Self {
            layers,
            embed: weights.token_embed,
            lm_head: weights.lm_head,
            offsets,
            budget,
        }
    }

    /// Get weight slice for a given layer and tensor.
    pub fn layer_wq(&self, layer: usize) -> &[f32] {
        let o = &self.offsets;
        &self.layers[layer].data[o.wq..o.wk]
    }

    pub fn layer_wk(&self, layer: usize) -> &[f32] {
        let o = &self.offsets;
        &self.layers[layer].data[o.wk..o.wv]
    }

    pub fn layer_wv(&self, layer: usize) -> &[f32] {
        let o = &self.offsets;
        &self.layers[layer].data[o.wv..o.wo]
    }

    pub fn layer_wo(&self, layer: usize) -> &[f32] {
        let o = &self.offsets;
        &self.layers[layer].data[o.wo..o.mlp_fc]
    }

    pub fn layer_mlp_fc(&self, layer: usize) -> &[f32] {
        let o = &self.offsets;
        &self.layers[layer].data[o.mlp_fc..o.mlp_proj]
    }

    pub fn layer_mlp_proj(&self, layer: usize) -> &[f32] {
        let o = &self.offsets;
        &self.layers[layer].data[o.mlp_proj..o.total_elems]
    }

    /// Total model weight size in bytes.
    pub fn total_bytes(&self) -> u64 {
        let layer_bytes: u64 = self.layers.iter()
            .map(|l| (l.data.len() * 4) as u64).sum();
        let embed_bytes = (self.embed.len() + self.lm_head.len()) as u64 * 4;
        layer_bytes + embed_bytes
    }
}

// ─── VRAM double-buffer ─────────────────────────────────────

/// Double-buffer for streaming weights CPU RAM → VRAM.
///
/// Two VRAM slots alternate: while GPU reads from slot A (current layer),
/// CPU writes next layer's weights into slot B. On the next layer,
/// roles swap.
///
/// This is a logical abstraction — actual GpuBuffer allocation happens
/// via HsaDevice. This struct tracks which slot is "active" (GPU reading)
/// and which is "staging" (CPU writing).
pub struct DoubleBuffer {
    /// Slot index currently being read by GPU (0 or 1).
    active: usize,
    /// Layer index currently in slot 0 (None if empty).
    slot_layer: [Option<usize>; 2],
    /// Size of each slot in f32 elements.
    slot_size: usize,
    /// CPU-side shadow buffers (for systems without real GPU).
    /// On real GPU: these would be `GpuBuffer` handles.
    slots: [Vec<f32>; 2],
}

impl DoubleBuffer {
    /// Create a double-buffer with slots sized for the largest layer.
    pub fn new(max_layer_elems: usize) -> Self {
        Self {
            active: 0,
            slot_layer: [None, None],
            slot_size: max_layer_elems,
            slots: [vec![0.0; max_layer_elems], vec![0.0; max_layer_elems]],
        }
    }

    /// Stage a layer into the non-active slot.
    /// Returns the staging slot index.
    pub fn stage(&mut self, layer_idx: usize, data: &[f32]) -> usize {
        let staging = 1 - self.active;
        let n = data.len().min(self.slot_size);
        self.slots[staging][..n].copy_from_slice(&data[..n]);
        self.slot_layer[staging] = Some(layer_idx);
        staging
    }

    /// Swap active and staging slots. Call after GPU finishes current layer.
    pub fn swap(&mut self) {
        self.active = 1 - self.active;
    }

    /// Get the active slot's data.
    pub fn active_data(&self) -> &[f32] {
        &self.slots[self.active]
    }

    /// Which layer is in the active slot?
    pub fn active_layer(&self) -> Option<usize> {
        self.slot_layer[self.active]
    }

    /// Get a slice from the active slot using layer offsets.
    pub fn active_slice(&self, start: usize, end: usize) -> &[f32] {
        &self.slots[self.active][start..end]
    }
}

/// Orchestrates weight streaming for a full forward/backward pass.
///
/// Usage:
/// ```ignore
/// let mut stream = LayerStream::new(&offloader);
/// for layer_idx in 0..num_layers {
///     stream.prefetch_next(layer_idx + 1, &offloader); // start staging next
///     let weights = stream.current_weights();           // GPU reads current
///     // ... compute layer ...
///     stream.advance();                                  // swap slots
/// }
/// ```
pub struct LayerStream {
    pub dbuf: DoubleBuffer,
}

impl LayerStream {
    pub fn new(offloader: &WeightOffloader) -> Self {
        let mut dbuf = DoubleBuffer::new(offloader.offsets.total_elems);
        // Pre-load layer 0 into active slot
        if !offloader.layers.is_empty() {
            let n = offloader.layers[0].data.len().min(dbuf.slot_size);
            dbuf.slots[0][..n].copy_from_slice(&offloader.layers[0].data[..n]);
            dbuf.slot_layer[0] = Some(0);
        }
        Self { dbuf }
    }

    /// Prefetch the next layer into the staging slot.
    pub fn prefetch_next(&mut self, next_layer: usize, offloader: &WeightOffloader) {
        if next_layer < offloader.layers.len() {
            self.dbuf.stage(next_layer, &offloader.layers[next_layer].data);
        }
    }

    /// Get current layer's weight data.
    pub fn current_data(&self) -> &[f32] {
        self.dbuf.active_data()
    }

    /// Get a weight tensor slice from the current layer.
    pub fn slice(&self, start: usize, end: usize) -> &[f32] {
        self.dbuf.active_slice(start, end)
    }

    /// Advance to the next layer (swap buffers).
    pub fn advance(&mut self) {
        self.dbuf.swap();
    }
}
