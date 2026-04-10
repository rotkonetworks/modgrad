//! GptModel: full transformer forward pass.
//! TransformerService: wraps GptModel as a Service (drop-in for InferenceService).
//!
//! Forward: embed → smear → (resid_scale → block) * n → backout → norm → lm_head
//!
//! Two paths:
//!   - Prefill: batch-process prompt tokens (fills KV cache)
//!   - Decode:  single-token autoregressive generation

// Service integration types — provided by the main modgrad crate at runtime.
// In standalone mode, TransformerService is not available.
#[cfg(feature = "__service_integration")]
use modgrad_io_types::{LogitMods, Request, Response, Service};
use super::block::TransformerBlock;
use super::config::GptConfig;
use super::dims::*;
use super::kv_cache::*;
use super::norm::ScaledRmsNorm;
use super::ops::TransformerOps;
use super::position::PositionEncoding;
use super::residual::{ForwardCtx, ResidualLambdas};
use super::rope::RotaryEmbedding;
use super::smear::{Inference, Smear, Training};
use super::tensor::Tensor2;

/// The full GPT model.
pub struct GptModel {
    /// Token embedding table: [vocab_size, model_dim].
    pub embed: Tensor2<VocabSize, ModelDim>,
    /// LM head (output projection): [vocab_size, model_dim].
    pub lm_head: Tensor2<VocabSize, ModelDim>,
    /// Final RMS norm with learned scale.
    pub final_norm: ScaledRmsNorm,
    /// Smear (previous-token mixing) in inference mode.
    pub smear_inference: Smear<Inference>,
    /// Smear in training mode (for prefill sequence processing).
    pub smear_training: Smear<Training>,
    /// Transformer blocks.
    pub blocks: Vec<TransformerBlock>,
    /// Residual lambdas.
    pub lambdas: ResidualLambdas,
    /// RoPE embeddings.
    pub rope: RotaryEmbedding,
    /// Position encoding strategy.
    pub position: Box<dyn PositionEncoding>,
    /// Config (kept for runtime queries).
    pub config: GptConfig,
}

impl GptModel {
    /// Prefill: process all prompt tokens in batch.
    ///
    /// Takes an Empty cache, fills K/V for all layers, returns Prefilled cache.
    /// Returns the last token's logits (needed for the first decode step).
    pub fn forward_prefill(
        &self,
        token_ids: &[i64],
        mut cache: KvCache<Empty>,
        ctx: &mut ForwardCtx,
        backend: &dyn TransformerOps,
    ) -> (Vec<f32>, KvCache<Prefilled>) {
        let md = self.config.model_dim.get();
        let vocab = self.config.vocab_size.get();
        let seq_len = token_ids.len();

        // Embed all tokens: [seq_len, model_dim]
        let mut hidden = vec![0.0f32; seq_len * md];
        for (t, &tid) in token_ids.iter().enumerate() {
            let row = self.embed.row(tid as usize);
            hidden[t * md..(t + 1) * md].copy_from_slice(row);
        }

        // Smear (training mode — processes whole sequence)
        self.smear_training.forward(&mut hidden, seq_len);

        // Save x0 for residual shortcuts
        ctx.set_x0_batch(&hidden, seq_len);
        ctx.midpoint_cached = false;

        // Token IDs as usize for VE lookup
        let tids: Vec<usize> = token_ids.iter().map(|&t| t as usize).collect();

        // Run through all blocks
        for block in &self.blocks {
            let window = self.config.window_size(block.layer_idx);
            block.forward_prefill(
                &mut hidden,
                &mut cache,
                &self.rope,
                ctx,
                &self.lambdas,
                seq_len,
                &tids,
                window,
                backend,
            );
        }

        // Mid-layer backout (applied to all tokens)
        ctx.apply_backout_batch(&mut hidden, seq_len, self.lambdas.backout_lambda);

        // Store last token's embedding for smear in decode mode
        cache.set_prev_embedding(&hidden[(seq_len - 1) * md..seq_len * md]);

        // Compute logits for last token only
        let last_hidden = &hidden[(seq_len - 1) * md..seq_len * md];
        let mut normed = vec![0.0f32; md];
        self.final_norm.forward(last_hidden, &mut normed);

        let mut logits = vec![0.0f32; vocab];
        backend.matvec_nobias(
            self.lm_head.as_slice(), &normed, &mut logits,
            vocab, md,
        );

        let prefilled = cache.prefill(seq_len);
        (logits, prefilled)
    }

    /// Decode: forward pass for a single token.
    ///
    /// Returns logits `[vocab_size]`.
    pub fn forward_one(
        &self,
        token_id: usize,
        cache: &mut KvCache<Decoding>,
        ctx: &mut ForwardCtx,
        backend: &dyn TransformerOps,
    ) -> Vec<f32> {
        let md = self.config.model_dim.get();
        let vocab = self.config.vocab_size.get();

        // Token embedding
        let mut hidden = self.embed.row(token_id).to_vec();

        // Smear: mix with previous token
        self.smear_inference.forward_one(&mut hidden, &cache.prev_embedding);

        // Save x0 for residual shortcuts
        ctx.set_x0_one(&hidden);
        ctx.midpoint_cached = false;

        let position = cache.seq_len();

        // Run through all blocks
        for block in &self.blocks {
            let window = self.config.window_size(block.layer_idx);
            block.forward_one(
                &mut hidden,
                cache,
                &self.rope,
                ctx,
                &self.lambdas,
                position,
                token_id,
                window,
                backend,
            );
        }

        // Mid-layer backout
        ctx.apply_backout(&mut hidden, self.lambdas.backout_lambda);

        // Update prev_embedding for next token's smear
        cache.set_prev_embedding(&hidden);
        cache.advance();

        // Final norm
        let mut normed = vec![0.0f32; md];
        self.final_norm.forward(&hidden, &mut normed);

        // LM head: logits = lm_head @ normed
        let mut logits = vec![0.0f32; vocab];
        backend.matvec_nobias(
            self.lm_head.as_slice(), &normed, &mut logits,
            vocab, md,
        );

        logits
    }
}

