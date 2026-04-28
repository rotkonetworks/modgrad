//! Frozen-cerebellum primitives: the [`Modality`] tag, [`CerebellumCache`]
//! container, and the [`FrozenCerebellum`] trait.
//!
//! ## Why these live here, in `modgrad-traits`
//!
//! These three types are cross-crate: `modgrad-data` produces `Modality`
//! tags, `modgrad-ctm` consumes them through `CerebellumCache`, and
//! `modgrad-blt` (and other LLM adapters such as `qwen_cerebellum`)
//! implements `FrozenCerebellum`. Putting the canonical definitions in
//! the leaf trait crate breaks two would-be cycles:
//!
//! - `modgrad-blt → isis-runtime → modgrad-codec → modgrad-ctm` if the
//!   trait lived in `modgrad-ctm` (the BLT adapter would have to depend
//!   on `modgrad-ctm`, but the codec already does — cycle).
//! - `modgrad-ctm → modgrad-data → modgrad-codec → modgrad-ctm` if
//!   `Modality` lived in `modgrad-data` and `modgrad-ctm` re-imported it
//!   (data already depends on codec which depends on ctm — cycle).
//!
//! `modgrad-traits` is the leaf everyone can depend on, and these are
//! pure-stdlib data types + a trait — no GPU dispatch, no projection, no
//! device code. The projection layer (`CerebProjection`), the random-
//! expansion impl, and the modality-pool helpers stay in `modgrad-ctm`
//! because they call `modgrad-device` for matvec/outer-product dispatch.

// ─── Modality tag ───────────────────────────────────────────

/// Modality tag for a token's cache row. Canonical definition —
/// `modgrad-data::Modality` and `modgrad-ctm::cerebellum::Modality` are
/// re-exports of this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Byte,
    Delimiter,
    ImageVq,
    AudioVq,
    Timestamp,
    Action,
    Other,
}

// ─── Cache ──────────────────────────────────────────────────

/// Cached hidden states from ALL layers of a frozen cerebellum.
/// Computed once per context window, read per-token during training.
pub struct CerebellumCache {
    /// All layer hidden states, contiguous.
    /// Layout: [layer][position][hidden_dim]
    /// Flat: layer * (n_positions * hidden_dim) + position * hidden_dim
    pub hidden_states: Vec<f32>,
    /// Dimension of each hidden state vector.
    pub hidden_dim: usize,
    /// Number of valid positions.
    pub n_positions: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// Optional per-position modality tag. Length must equal
    /// `n_positions` when `Some`. `None` means modality is unknown /
    /// not tracked (the pre-multimodal default; existing callers do
    /// not need updating beyond initialising this field to `None`).
    pub modalities: Option<Vec<Modality>>,
}

impl CerebellumCache {
    /// Get hidden state at a specific layer and position.
    pub fn at(&self, layer: usize, position: usize) -> &[f32] {
        if layer < self.n_layers && position < self.n_positions {
            let start = layer * (self.n_positions * self.hidden_dim)
                + position * self.hidden_dim;
            &self.hidden_states[start..start + self.hidden_dim]
        } else {
            &[]
        }
    }

    /// Get the weighted combination across all layers at a position.
    /// `layer_weights` should be softmaxed (sum to 1.0).
    pub fn blend_layers(&self, position: usize, layer_weights: &[f32]) -> Vec<f32> {
        let d = self.hidden_dim;
        let mut out = vec![0.0f32; d];
        let n = self.n_layers.min(layer_weights.len());
        for l in 0..n {
            let h = self.at(l, position);
            if h.is_empty() { continue; }
            let w = layer_weights[l];
            for i in 0..d {
                out[i] += w * h[i];
            }
        }
        out
    }

    /// Same as blend_layers but writes into pre-allocated buffer.
    pub fn blend_layers_into(&self, position: usize, layer_weights: &[f32], out: &mut [f32]) {
        let d = self.hidden_dim;
        out[..d].fill(0.0);
        let n = self.n_layers.min(layer_weights.len());
        for l in 0..n {
            let h = self.at(l, position);
            if h.is_empty() { continue; }
            let w = layer_weights[l];
            for i in 0..d {
                out[i] += w * h[i];
            }
        }
    }

    /// Empty cache (no cerebellum active).
    pub fn empty() -> Self {
        Self {
            hidden_states: Vec::new(),
            hidden_dim: 0,
            n_positions: 0,
            n_layers: 0,
            modalities: None,
        }
    }

    pub fn is_empty(&self) -> bool { self.n_positions == 0 }

    /// Single-layer cache (backward compat with old encode_context).
    pub fn single_layer(hidden_states: Vec<f32>, hidden_dim: usize, n_positions: usize) -> Self {
        Self {
            hidden_states,
            hidden_dim,
            n_positions,
            n_layers: 1,
            modalities: None,
        }
    }
}

// ─── Trait ──────────────────────────────────────────────────

/// A frozen forward model used as cerebellum region.
///
/// LLM implementations should override `encode_context_layers()` to return
/// hidden states from ALL layers. The learned layer weights (in CerebProjection)
/// blend these into a single representation per position.
pub trait FrozenCerebellum: Send {
    /// Dimension of the model's output (hidden states per position).
    fn hidden_dim(&self) -> usize;

    /// Number of layers in the model.
    fn n_layers(&self) -> usize { 1 }

    /// Encode a full context window, returning hidden states from ALL layers.
    /// Default: calls encode_context() for single-layer models.
    fn encode_context_layers(&mut self, token_ids: &[i64]) -> CerebellumCache {
        // Default: single layer (backward compat)
        let cache = self.encode_context(token_ids);
        cache
    }

    /// Encode a full context window (single layer output).
    /// Override this for simple models; override encode_context_layers for LLMs.
    fn encode_context(&mut self, token_ids: &[i64]) -> CerebellumCache {
        let d = self.hidden_dim();
        let n = token_ids.len();
        let mut hidden_states = vec![0.0f32; n * d];
        for (i, &tid) in token_ids.iter().enumerate() {
            let input = vec![tid as f32 / 128.0; d];
            let h = self.forward(&input);
            let copy_len = d.min(h.len());
            hidden_states[i * d..i * d + copy_len].copy_from_slice(&h[..copy_len]);
        }
        CerebellumCache::single_layer(hidden_states, d, n)
    }

    /// Per-token forward pass (for non-LLM models like RandomExpansion).
    fn forward(&mut self, input: &[f32]) -> Vec<f32>;
}
