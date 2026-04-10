//! TransformerBlock: a single layer implementing Filter.
//!
//! Forward: norm → attn (+VE) → residual → norm → mlp → residual
//!
//! Each block is composable via the Filter trait, enabling per-layer
//! telemetry, interception, or replacement.
//!
//! Two paths: `forward_prefill` (batch prompt) and `forward_one` (single token decode).

use super::attention::CausalSelfAttention;
use super::config::GptConfig;
use super::dims::*;
use super::kv_cache::*;
use super::mlp::Mlp;
use super::norm::RmsNorm;
use super::ops::TransformerOps;
use super::residual::{ForwardCtx, ResidualLambdas};
use super::rope::RotaryEmbedding;
use super::value_embed::ValueEmbedding;

/// A single transformer block.
pub struct TransformerBlock {
    pub attn: CausalSelfAttention,
    pub mlp: Mlp,
    pub attn_norm: RmsNorm,
    pub mlp_norm: RmsNorm,
    pub value_embed: Option<ValueEmbedding>,
    pub layer_idx: LayerIdx,
    model_dim: usize,
    kv_dim: usize,
}

impl TransformerBlock {
    pub fn new(
        attn: CausalSelfAttention,
        mlp: Mlp,
        value_embed: Option<ValueEmbedding>,
        layer_idx: LayerIdx,
        config: &GptConfig,
    ) -> Self {
        Self {
            attn,
            mlp,
            attn_norm: RmsNorm::new(config.norm_eps),
            mlp_norm: RmsNorm::new(config.norm_eps),
            value_embed,
            layer_idx,
            model_dim: config.model_dim.get(),
            kv_dim: config.num_kv_heads.get() * config.head_dim.get(),
        }
    }

    /// Prefill forward pass for a full prompt sequence.
    ///
    /// `hidden`: `[seq_len, model_dim]` row-major — modified in-place.
    /// `cache`: KV cache (Empty state — layer K/V written directly).
    /// `rope`: rotary embeddings.
    /// `ctx`: cross-block context (x0 is per-token, midpoint cached at midpoint layer).
    /// `lambdas`: per-layer residual lambdas.
    /// `token_ids`: token IDs for value embedding lookup.
    /// `window`: optional sliding window size.
    /// `backend`: compute backend.
    pub fn forward_prefill(
        &self,
        hidden: &mut [f32],
        cache: &mut KvCache<Empty>,
        rope: &RotaryEmbedding,
        ctx: &mut ForwardCtx,
        lambdas: &ResidualLambdas,
        seq_len: usize,
        token_ids: &[usize],
        window: Option<usize>,
        backend: &dyn TransformerOps,
    ) {
        let li = self.layer_idx.get();
        let md = self.model_dim;
        let kv_dim = self.kv_dim;

        // Cache midpoint using first token's hidden (representative for the layer)
        ctx.maybe_cache_midpoint(&hidden[..md], self.layer_idx, lambdas.midpoint);

        // ─── Attention sub-layer ────────────────────────────
        // Norm all tokens
        let mut normed = vec![0.0f32; seq_len * md];
        for t in 0..seq_len {
            self.attn_norm.forward(
                &hidden[t * md..(t + 1) * md],
                &mut normed[t * md..(t + 1) * md],
            );
        }

        // Run prefill attention — populates K/V in k_out/v_out
        let mut k_out = vec![0.0f32; seq_len * kv_dim];
        let mut v_out = vec![0.0f32; seq_len * kv_dim];
        let mut attn_out = vec![0.0f32; seq_len * md];

        self.attn.forward_prefill(
            &normed, rope, 0, seq_len, window, backend,
            &mut k_out, &mut v_out, &mut attn_out,
        );

        // Apply value embeddings to V before writing to cache
        if let Some(ref ve) = self.value_embed {
            for t in 0..seq_len {
                ve.apply(
                    &mut v_out[t * kv_dim..(t + 1) * kv_dim],
                    &normed[t * md..(t + 1) * md],
                    token_ids[t],
                );
            }
        }

        // Write K/V to cache
        cache.layer_mut(li).write(&k_out, &v_out, 0, seq_len);

        // Residual connection (attention) — per-token
        for t in 0..seq_len {
            let h = &mut hidden[t * md..(t + 1) * md];
            let a = &attn_out[t * md..(t + 1) * md];
            lambdas.apply(h, a, &ctx.x0[t * md..(t + 1) * md], self.layer_idx);
        }

        // ─── MLP sub-layer ──────────────────────────────────
        for t in 0..seq_len {
            self.mlp_norm.forward(
                &hidden[t * md..(t + 1) * md],
                &mut normed[t * md..(t + 1) * md],
            );
        }

        let mut mlp_out = vec![0.0f32; md];
        for t in 0..seq_len {
            self.mlp.forward(
                &normed[t * md..(t + 1) * md],
                &mut mlp_out,
                backend,
            );
            let h = &mut hidden[t * md..(t + 1) * md];
            lambdas.apply(h, &mlp_out, &ctx.x0[t * md..(t + 1) * md], self.layer_idx);
        }
    }

    /// Forward pass for a single token (decode mode).
    pub fn forward_one(
        &self,
        hidden: &mut [f32],
        cache: &mut KvCache<Decoding>,
        rope: &RotaryEmbedding,
        ctx: &mut ForwardCtx,
        lambdas: &ResidualLambdas,
        position: usize,
        token_id: usize,
        window: Option<usize>,
        backend: &dyn TransformerOps,
    ) {
        let li = self.layer_idx.get();
        let md = self.model_dim;
        let kv_dim = self.kv_dim;

        // Cache midpoint if needed
        ctx.maybe_cache_midpoint(hidden, self.layer_idx, lambdas.midpoint);

        // ─── Attention sub-layer ────────────────────────────
        let mut normed = vec![0.0f32; md];
        self.attn_norm.forward(hidden, &mut normed);

        let mut attn_out = vec![0.0f32; md];
        let mut cur_k = vec![0.0f32; kv_dim];
        let mut cur_v = vec![0.0f32; kv_dim];

        let kv = cache.layer(li);
        let kv_k = kv.k_slice(position).to_vec();
        let kv_v = kv.v_slice(position).to_vec();

        self.attn.forward_one(
            &normed,
            &kv_k, &kv_v,
            &mut cur_k, &mut cur_v,
            rope, position,
            position + 1,
            window,
            backend,
            &mut attn_out,
        );

        // Apply value embedding if this layer has it
        if let Some(ref ve) = self.value_embed {
            ve.apply(&mut cur_v, &normed, token_id);
        }

        // Write K/V to cache
        cache.layer_mut(li).write(&cur_k, &cur_v, position, 1);

        // Residual connection (attention)
        lambdas.apply(hidden, &attn_out, &ctx.x0, self.layer_idx);

        // ─── MLP sub-layer ──────────────────────────────────
        self.mlp_norm.forward(hidden, &mut normed);

        let mut mlp_out = vec![0.0f32; md];
        self.mlp.forward(&normed, &mut mlp_out, backend);

        // Residual connection (MLP)
        lambdas.apply(hidden, &mlp_out, &ctx.x0, self.layer_idx);
    }
}
