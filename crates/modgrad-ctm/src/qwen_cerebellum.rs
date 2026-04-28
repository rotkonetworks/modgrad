//! `QwenCerebellum` — adapter that wraps a device-resident
//! Qwen2.5-class transformer ([`GptModelResident`]) as a
//! [`FrozenCerebellum`].
//!
//! ## What this provides
//!
//! Per-layer hidden-state cache for a context window. The cortex
//! reads these via `cerebellum_at_position` + `CerebProjection` (with
//! its softmaxed `layer_weight_logits`); a swap from Qwen to Llama
//! changes only the construction call, never the trait surface.
//!
//! ## What this explicitly does NOT provide
//!
//! - **No `forward(input: &[f32])`.** Tokens-as-floats is meaningless
//!   for an LLM (the input would have to be an embedding vector or a
//!   token id, not a generic activation). The trait method panics; the
//!   correct entry point is `encode_context_layers`.
//! - **No backward pass.** This struct is FROZEN. The wrapped
//!   `GptModelResident` is consumed by value in `from_resident`, so
//!   callers can't mutate the weights via aliasing. If you need to
//!   fine-tune the LLM, do it through `modgrad-transformer` directly
//!   and rebuild the cerebellum.
//!
//! ## Layout contract
//!
//! `CerebellumCache::hidden_states` is `[n_layers × n_positions ×
//! hidden_dim]` flat row-major:
//!
//! ```text
//! hidden_states[layer * (n_positions * hidden_dim)
//!              + position * hidden_dim
//!              + i]
//! ```
//!
//! This matches `CerebellumCache::at(layer, position)`. The order
//! is layer-major because cortex regions blend across layers at a
//! fixed position much more often than they walk positions at a
//! fixed layer — keeping per-layer slabs contiguous makes the
//! blend a pure stride-1 read.
//!
//! Only available with `--features rocm` (depends on the resident
//! transformer stack in `modgrad-transformer`).

#![cfg(feature = "rocm")]

use modgrad_compute::backend::{GpuVec, ResidencyError};
use modgrad_device::backend::HipBatch;
use modgrad_transformer::kv_cache_resident::KvCacheResident;
use modgrad_transformer::resident::{
    AttentionScratch, GptModelResident, SwigluScratch, TransformerBlockScratch,
};

use crate::cerebellum::{CerebellumCache, FrozenCerebellum};

/// Frozen cerebellum backed by a device-resident transformer.
///
/// Wraps [`GptModelResident`] (the resident stack from
/// `modgrad-transformer`) and exposes its per-layer hidden states for
/// a context window. The transformer weights stay on device for the
/// lifetime of this struct — no PCIe transfers per call.
///
/// **Frozen.** Construction takes the model by value (`from_resident`)
/// so the only way to mutate weights is to drop and rebuild. There is
/// no backward path, no parameter access, no sync method.
///
/// **Single-pass per `encode_context_layers` call.** The KV cache is
/// reset at the start of every call, so each invocation starts from
/// position 0 and walks the full token sequence. Building a longer
/// context is the caller's job (concatenate token ids, call once).
pub struct QwenCerebellum {
    /// Device-resident transformer. Owned by value — no aliasing.
    model: GptModelResident,
    /// Per-call KV cache. Reset at the top of `encode_context_layers`.
    /// Sized for `max_seq_len` tokens at construction time; calls with
    /// longer sequences will panic in debug or write out of bounds in
    /// release. Cap your context length.
    kv_cache: KvCacheResident,
    /// Maximum supported sequence length (= `kv_cache.max_seq_len()`).
    max_seq_len: usize,
}

impl QwenCerebellum {
    /// Wrap a resident transformer as a frozen cerebellum.
    ///
    /// `max_seq_len` is the longest token sequence
    /// `encode_context_layers` will be asked to encode. The KV cache
    /// is allocated once for this size; sequences longer than this
    /// will trip a `debug_assert` in `KvCacheResident::write` or
    /// silently overflow in release.
    ///
    /// Allocates one `KvCacheResident` (`2 × n_layers × n_kv_heads ×
    /// max_seq_len × head_dim × 4` bytes plus a `model_dim × 4`
    /// scratch). For Qwen2.5-0.5B at `max_seq_len = 32`:
    /// `2 × 24 × 2 × 32 × 64 × 4` ≈ 786 KB. Cheap.
    pub fn from_resident(
        model: GptModelResident,
        max_seq_len: usize,
    ) -> Result<Self, ResidencyError> {
        // Pull KV-cache geometry off the first block. `model` always
        // has at least one block (a zero-block transformer would fail
        // earlier at `from_model`); we still defend against it.
        let n_layers = model.num_layers();
        let model_dim = model.model_dim();
        if n_layers == 0 {
            return Err(ResidencyError::WrongVariant {
                expected: "GptModelResident with >=1 block",
                got: "0 blocks",
            });
        }
        let first = &model.blocks[0];
        let n_kv_heads = first.attn.num_kv_heads;
        let head_dim = first.attn.head_dim;

        let kv_cache = KvCacheResident::new(
            n_layers, n_kv_heads, head_dim, max_seq_len, model_dim,
        )?;

        Ok(Self { model, kv_cache, max_seq_len })
    }

    /// Maximum sequence length this cerebellum can encode in one call.
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }
}

impl FrozenCerebellum for QwenCerebellum {
    fn hidden_dim(&self) -> usize { self.model.model_dim() }

    fn n_layers(&self) -> usize { self.model.num_layers() }

    /// Run the resident transformer over `token_ids`, snapshotting
    /// every block's hidden state to host, returning the
    /// `[n_layers × n_positions × hidden_dim]` flat cache.
    ///
    /// **Layout:** layer-major, then position-major:
    /// `out[li * n * d + t * d + i]` is element `i` of layer `li`'s
    /// output at token position `t`. This matches
    /// `CerebellumCache::at(layer, position)`.
    ///
    /// **KV cache:** reset at the start so every call is a fresh
    /// context. The caller does NOT need to reset between calls.
    fn encode_context_layers(&mut self, token_ids: &[i64]) -> CerebellumCache {
        let n = token_ids.len();
        let n_layers = self.model.num_layers();
        let d = self.model.model_dim();

        if n == 0 {
            return CerebellumCache {
                hidden_states: Vec::new(),
                hidden_dim: d,
                n_positions: 0,
                n_layers,
            };
        }
        debug_assert!(
            n <= self.max_seq_len,
            "QwenCerebellum: token sequence ({n}) exceeds max_seq_len ({})",
            self.max_seq_len,
        );

        // Per-call: fresh context window.
        self.kv_cache.reset();

        // Allocate output [n_layers × n × d] flat. ~2.7 MB for
        // Qwen2.5-0.5B at n=32 (24 × 32 × 896 × 4 bytes). Acceptable.
        let mut hidden_states = vec![0.0f32; n_layers * n * d];

        // Allocate scratch once and reuse across the whole call. Same
        // pattern as `GptModelResident::forward` — see resident.rs
        // ~L2410. The block forward writes through `block_scratch`
        // (sublayer outputs, normed buffer); attn/mlp scratch are the
        // attention/MLP intermediates.
        let max_kv = self.kv_cache.n_kv_heads() * self.kv_cache.head_dim();
        let max_seq = self.kv_cache.max_seq_len();
        let n_heads = self.model.blocks[0].attn.num_heads;
        let head_dim = self.kv_cache.head_dim();
        let mlp_dim = self.model.blocks[0].mlp.mlp_dim();

        let mut attn_scratch = AttentionScratch::new(n_heads, head_dim, max_kv, max_seq)
            .expect("QwenCerebellum: alloc attn scratch");
        let mut mlp_scratch = SwigluScratch::new(d, mlp_dim)
            .expect("QwenCerebellum: alloc mlp scratch");
        let mut block_scratch = TransformerBlockScratch::new(d)
            .expect("QwenCerebellum: alloc block scratch");

        let mut hidden_dev = GpuVec::try_hip(d)
            .expect("QwenCerebellum: alloc hidden_dev");
        let mut x0_dev = GpuVec::try_hip(d)
            .expect("QwenCerebellum: alloc x0_dev");

        // Per-token slab to D2H the post-block hidden state into.
        let mut host_slab = vec![0.0f32; d];

        // One HipBatch for the whole encode. `note_dispatch` cadence
        // matches the dispatches we issue: 1 D2D for embed lookup, 1
        // D2D for x0 copy, plus whatever each `block.forward`
        // internally accounts for.
        let batch = HipBatch::new();

        for (t, &tid) in token_ids.iter().enumerate() {
            // Stage 1: embed lookup. embed_dev row `tid` → hidden_dev.
            // D2D copy of `model_dim` floats.
            let embed_off_bytes = tid as usize * d * 4;
            let bytes = d * 4;
            unsafe {
                use std::os::raw::c_void;
                hip_memcpy_d2d(
                    hip_buf_mut(&mut hidden_dev).device_ptr(),
                    (self.model.embed_dev.device_ptr() as *const u8)
                        .add(embed_off_bytes) as *const c_void,
                    bytes,
                ).expect("QwenCerebellum: embed D2D");
            }
            batch.note_dispatch().expect("QwenCerebellum: note embed");
            // Save x0. Smear is omitted (matches `GptModelResident::forward`).
            unsafe {
                hip_memcpy_d2d(
                    hip_buf_mut(&mut x0_dev).device_ptr(),
                    hip_buf(&hidden_dev).device_ptr() as *const std::os::raw::c_void,
                    bytes,
                ).expect("QwenCerebellum: x0 D2D");
            }
            batch.note_dispatch().expect("QwenCerebellum: note x0");

            // Stage 2: walk every block. After block `li` runs,
            // `hidden_dev` holds layer `li`'s output for token `t`.
            // Snapshot it into `hidden_states[li, t, :]`.
            for (li, block) in self.model.blocks.iter().enumerate() {
                block.forward(
                    &batch,
                    &mut hidden_dev, &x0_dev,
                    &mut self.kv_cache, t, &self.model.rope,
                    &mut attn_scratch, &mut mlp_scratch,
                    &mut block_scratch,
                ).expect("QwenCerebellum: block forward");

                // D2H the layer output into the host slab, then copy
                // into the right slot of the flat output buffer.
                hidden_dev.copy_to_host(&mut host_slab);
                let row_off = li * (n * d) + t * d;
                hidden_states[row_off..row_off + d].copy_from_slice(&host_slab);
            }
            // Stage 3 (final norm + lm_head) is intentionally skipped
            // — the cerebellum exposes per-layer pre-final-norm hidden
            // states, not logits. Matches the design in
            // `BRAIN_ARCHITECTURE.md`: cortex blends across layers via
            // `CerebProjection::layer_weights`, then projects to its
            // own `d_model`.
        }
        // Flush the queue once at the end. Per-token flushing would
        // serialise dispatch and lose most of the residency win; one
        // flush per encode is the right cadence.
        batch.flush().expect("QwenCerebellum: flush");

        CerebellumCache {
            hidden_states,
            hidden_dim: d,
            n_positions: n,
            n_layers,
        }
    }

    /// `forward` is meaningless for an LLM cerebellum (tokens are
    /// ids, not generic activation vectors). Use
    /// `encode_context_layers` instead.
    fn forward(&mut self, _input: &[f32]) -> Vec<f32> {
        panic!("QwenCerebellum: use encode_context_layers, not forward")
    }
}

// ─── HIP helpers ────────────────────────────────────────────

/// Local D2D memcpy — same shape as the helpers in
/// `modgrad-transformer/src/resident.rs` and
/// `kv_cache_resident.rs`. Duplicated rather than re-exported so this
/// module is independent of those crates' private helpers.
unsafe fn hip_memcpy_d2d(
    dst: *mut std::os::raw::c_void,
    src: *const std::os::raw::c_void,
    bytes: usize,
) -> Result<(), ResidencyError> {
    use modgrad_device::backend::rocm::ffi;
    const HIP_MEMCPY_DEVICE_TO_DEVICE: std::os::raw::c_int = 3;
    let err = unsafe { ffi::hipMemcpy(dst, src, bytes, HIP_MEMCPY_DEVICE_TO_DEVICE) };
    if err != 0 {
        return Err(ResidencyError::Backend(
            modgrad_device::backend::BackendError::Runtime(format!(
                "hipMemcpy D2D ({bytes} bytes): {}",
                ffi::hip_err_str(err),
            )),
        ));
    }
    Ok(())
}

/// Unwrap `GpuVec::Hip(_)` or panic. Every `GpuVec::try_hip` allocation
/// in this module yields a `Hip` variant — a non-Hip GpuVec here would
/// be a programmer bug, not a runtime condition. Mirrors the `hip_buf`
/// helpers in `modgrad-transformer/src/resident.rs` (which return a
/// `Result` only because they're called from `Result`-typed paths).
#[inline]
fn hip_buf(g: &GpuVec) -> &modgrad_device::backend::HipBuffer {
    match g {
        GpuVec::Hip(b) => b,
        _ => panic!("QwenCerebellum: GpuVec is not Hip variant"),
    }
}

#[inline]
fn hip_buf_mut(g: &mut GpuVec) -> &mut modgrad_device::backend::HipBuffer {
    match g {
        GpuVec::Hip(b) => b,
        _ => panic!("QwenCerebellum: GpuVec is not Hip variant"),
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Construction + encode smoke test. Skipped if HIP runtime is
    /// unavailable or if `MODGRAD_SKIP_HIP_TESTS` is set (CI without
    /// a GPU).
    ///
    /// Uses the same tiny config as `modgrad-transformer`'s resident
    /// tests: 2 layers, model_dim=128, n_heads=4, vocab=256, SwiGLU.
    /// We rebuild the model + swiglu_mlps inline (the helpers in
    /// `modgrad-transformer::resident::tests` are `pub(crate)`, so
    /// reaching them from a downstream crate would require leaking
    /// them — not worth the API surface).
    #[test]
    fn qwen_cerebellum_alloc_and_encode() {
        if std::env::var("MODGRAD_SKIP_HIP_TESTS").is_ok() {
            eprintln!("MODGRAD_SKIP_HIP_TESTS set, skipping");
            return;
        }
        if !modgrad_device::backend::rocm::ffi::runtime_available() {
            eprintln!("hip runtime unavailable, skipping");
            return;
        }

        use modgrad_transformer::attention::{AttentionWeights, CausalSelfAttention};
        use modgrad_transformer::block::TransformerBlock;
        use modgrad_transformer::config::{
            GptConfig, MlpActivation, ResidualConfig, SmearConfig, ValueEmbedConfig,
            WindowPattern, Precision,
        };
        use modgrad_transformer::dims::*;
        use modgrad_transformer::mlp::{Mlp, MlpWeights, SwigluMlp, SwigluWeights};
        use modgrad_transformer::model::GptModel;
        use modgrad_transformer::norm::ScaledRmsNorm;
        use modgrad_transformer::position::fixed::FixedPositioning;
        use modgrad_transformer::residual::ResidualLambdas;
        use modgrad_transformer::rope::RotaryEmbedding;
        use modgrad_transformer::smear::{Inference, Smear, SmearWeights, Training};
        use modgrad_transformer::tensor::Tensor2;

        // Tiny config: 2 layers, model_dim=128, n_heads=4 (no GQA), vocab=256.
        let head_dim = 32usize;
        let n_heads = 4usize;
        let model_dim = head_dim * n_heads;
        let config = GptConfig {
            model_dim: ModelDim::new(model_dim),
            num_heads: NumHeads::new(n_heads),
            num_kv_heads: NumKvHeads::new(n_heads),
            head_dim: HeadDim::new(head_dim),
            num_layers: NumLayers::new(2),
            vocab_size: VocabSize::new(256),
            mlp_dim: MlpDim::new(model_dim * 2),
            max_seq_len: SeqLen::new(16),
            rope_base: 10000.0,
            qk_norm_scale: 1.0,
            window_pattern: WindowPattern::Full,
            mlp_activation: MlpActivation::SwiGlu,
            layer_overrides: Vec::new(),
            tie_embeddings: false,
            logit_cap: 0.0,
            recurrent_steps: 1,
            has_exit_gate: false,
            value_embed: ValueEmbedConfig::default(),
            residual: ResidualConfig {
                resid_start: 1.0, resid_end: 1.0,
                x0_start: 0.0, x0_end: 0.0,
                backout_lambda: 0.0,
            },
            smear: SmearConfig::default(),
            precision: Precision::F32,
            norm_eps: 1e-5,
            use_qk_norm: false,
        };

        let md = config.model_dim.get();
        let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
        let vocab = config.vocab_size.get();
        let mlp_dim = config.mlp_dim.get();

        let mut rng = modgrad_compute::neuron::SimpleRng::new(0xCAFE_BEEF);
        let randn = |rng: &mut modgrad_compute::neuron::SimpleRng, n: usize| -> Vec<f32> {
            (0..n).map(|_| rng.next_normal() * 0.05).collect()
        };

        let token_embed = randn(&mut rng, vocab * md);
        let lm_head = randn(&mut rng, vocab * md);
        let final_norm_scale = vec![1.0f32; md];
        let smear_gate = vec![0.0f32; md * config.smear.gate_channels];

        let mut blocks_host = Vec::with_capacity(config.num_layers.get());
        let mut swiglu_mlps = Vec::with_capacity(config.num_layers.get());
        for li in 0..config.num_layers.get() {
            let attn_w = AttentionWeights {
                wq: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
                wk: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
                wv: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
                wo: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
            };
            let attn = CausalSelfAttention::new(attn_w, &config);

            let swiglu_w = SwigluWeights {
                gate: Tensor2::new(randn(&mut rng, mlp_dim * md), mlp_dim, md).unwrap(),
                up: Tensor2::new(randn(&mut rng, mlp_dim * md), mlp_dim, md).unwrap(),
                down: Tensor2::new(randn(&mut rng, md * mlp_dim), md, mlp_dim).unwrap(),
            };
            let swiglu = SwigluMlp::new(swiglu_w, config.model_dim, config.mlp_dim);
            swiglu_mlps.push(swiglu);

            let placeholder_mlp = Mlp::new(
                MlpWeights {
                    fc: Tensor2::zeros(mlp_dim, md),
                    proj: Tensor2::zeros(md, mlp_dim),
                },
                config.model_dim, config.mlp_dim,
            );
            let layer_idx = LayerIdx::new(li, config.num_layers).unwrap();
            blocks_host.push(TransformerBlock::new(
                attn, placeholder_mlp, None, layer_idx, &config,
            ));
        }

        let model = GptModel {
            embed: Tensor2::new(token_embed, vocab, md).unwrap(),
            lm_head: Tensor2::new(lm_head, vocab, md).unwrap(),
            final_norm: ScaledRmsNorm::new(final_norm_scale, config.norm_eps),
            smear_inference: Smear::<Inference>::new(SmearWeights::new(
                smear_gate.clone(), config.model_dim, &config.smear,
            )),
            smear_training: Smear::<Training>::new(SmearWeights::new(
                smear_gate, config.model_dim, &config.smear,
            )),
            blocks: blocks_host,
            lambdas: ResidualLambdas::from_config(&config.residual, config.num_layers),
            rope: RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_base),
            position: Box::new(FixedPositioning),
            config: config.clone(),
        };

        let resident = GptModelResident::from_model(&model, &swiglu_mlps)
            .expect("upload resident");

        let mut cereb = QwenCerebellum::from_resident(resident, 16)
            .expect("alloc cerebellum");

        assert_eq!(cereb.hidden_dim(), 128);
        assert_eq!(cereb.n_layers(), 2);
        assert_eq!(cereb.max_seq_len(), 16);

        let token_ids: Vec<i64> = (0..8).map(|i| (i * 17) as i64 % 256).collect();
        let cache = cereb.encode_context_layers(&token_ids);

        assert_eq!(cache.n_layers, 2);
        assert_eq!(cache.n_positions, 8);
        assert_eq!(cache.hidden_dim, 128);
        assert_eq!(cache.hidden_states.len(), 2 * 8 * 128);

        // Finite check: NaN/inf would mean a bad dispatch.
        for (i, &v) in cache.hidden_states.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at offset {i}: {v}");
        }

        // Layer 0 token 0 and layer 1 token 0 should differ — every
        // block transforms the hidden state, so successive layers
        // can't be byte-identical at the same position.
        let l0t0 = cache.at(0, 0);
        let l1t0 = cache.at(1, 0);
        let mut differs = false;
        for (a, b) in l0t0.iter().zip(l1t0.iter()) {
            if (a - b).abs() > 1e-6 { differs = true; break; }
        }
        assert!(differs, "layers 0 and 1 produced identical hidden states");
    }
}
