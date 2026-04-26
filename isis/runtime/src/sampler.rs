//! Sampler service — generic over `LanguageModel`. Drives the
//! autoregressive decode loop: prefill, then sample-and-step.
//!
//! ## SaaF framing
//!
//! `Sampler` is a service: `(prompt_tokens, model) => generated_tokens`.
//! Filters wrapping a `LanguageModel` (LoRA, quantized residency,
//! streaming) compose freely under it because `Sampler` only depends
//! on the trait, not on `GptModelResident` directly.

#![cfg(feature = "rocm")]

use modgrad_compute::backend::{GpuVec, ResidencyError};
use modgrad_device::backend::HipBatch;
use modgrad_transformer::KvCacheResident;

use crate::language_model::LanguageModel;

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// `0.0` = argmax (greedy). Otherwise softmax(logits / T).
    pub temperature: f32,
    /// Restrict sampling to top-K candidates after temperature scaling.
    pub top_k: Option<usize>,
    pub max_new_tokens: usize,
    /// Token ids that terminate generation (e.g. EOS).
    pub stop_tokens: Vec<i64>,
    /// Seed for the internal xorshift RNG.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            max_new_tokens: 256,
            stop_tokens: Vec::new(),
            seed: 0xdead_beef_cafe_babe,
        }
    }
}

pub struct Sampler {
    pub config: SamplerConfig,
    rng_state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let seed = config.seed.max(1);
        Self { config, rng_state: seed }
    }

    /// Generate tokens autoregressively. Returns only the new tokens
    /// (not the prompt). Per-token callback is invoked with each
    /// emitted token id (useful for streaming output).
    pub fn generate<M, F>(
        &mut self,
        model: &mut M,
        batch: &HipBatch,
        prompt_token_ids: &[i64],
        kv_cache: &mut KvCacheResident,
        mut on_token: F,
    ) -> Result<Vec<i64>, ResidencyError>
    where
        M: LanguageModel,
        F: FnMut(i64),
    {
        let vocab = model.vocab_size();
        let mut logits_dev = GpuVec::try_hip(vocab)?;
        let mut logits_host = vec![0.0f32; vocab];

        if prompt_token_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Prefill — forward each prompt token at its position. The last
        // forward's logits become the first sample-and-step input.
        for (pos, &tok) in prompt_token_ids.iter().enumerate() {
            model.forward_logits(
                batch,
                &[tok],
                &[pos],
                Some(kv_cache),
                &mut logits_dev,
            )?;
        }

        let mut new_tokens: Vec<i64> = Vec::with_capacity(self.config.max_new_tokens);
        let mut next_pos = prompt_token_ids.len();

        for _ in 0..self.config.max_new_tokens {
            logits_dev.copy_to_host(&mut logits_host);
            let next = self.sample(&logits_host);
            new_tokens.push(next);
            on_token(next);
            if self.config.stop_tokens.contains(&next) {
                break;
            }
            model.forward_logits(
                batch,
                &[next],
                &[next_pos],
                Some(kv_cache),
                &mut logits_dev,
            )?;
            next_pos += 1;
        }

        Ok(new_tokens)
    }

    fn sample(&mut self, logits: &[f32]) -> i64 {
        if self.config.temperature <= 0.0 {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            return best as i64;
        }

        let inv_t = 1.0 / self.config.temperature;
        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
            .map(|(i, &v)| (i, v * inv_t))
            .collect();
        if let Some(k) = self.config.top_k {
            indexed.sort_by(|a, b|
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(k.max(1));
        }
        let m = indexed.iter().fold(f32::NEG_INFINITY, |m, &(_, v)| m.max(v));
        let mut sum = 0.0f64;
        for (_, v) in indexed.iter_mut() {
            *v = (*v - m).exp();
            sum += *v as f64;
        }
        let r = self.next_uniform() * sum as f32;
        let mut acc = 0.0f32;
        for (i, p) in &indexed {
            acc += *p;
            if acc >= r {
                return *i as i64;
            }
        }
        indexed.last().map(|(i, _)| *i as i64).unwrap_or(0)
    }

    fn next_uniform(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        ((x >> 40) as f32) / (1u32 << 24) as f32
    }
}
