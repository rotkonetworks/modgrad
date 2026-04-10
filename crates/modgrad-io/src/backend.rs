//! Backend trait: abstraction over inference engines (ONNX, GGUF, etc).
//!
//! Any model that can produce hidden states and logits can be an isis backend.

use crate::episode::normalize;
use crate::memory::MemoryBank;
use crate::types::*;

pub type BoxErr = Box<dyn std::error::Error>;

/// Simple tokenizer wrapper around HuggingFace tokenizers.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn load(path: &str) -> Result<Self, BoxErr> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("tokenizer error: {}", e))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Vec<i64> {
        let enc = self.inner.encode(text, false).unwrap();
        enc.get_ids().iter().map(|&id| id as i64).collect()
    }

    pub fn decode(&self, ids: &[i64]) -> String {
        let u: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner.decode(&u, true).unwrap_or_default()
    }

    pub fn decode_i64(&self, ids: &[i64]) -> String {
        self.decode(ids)
    }
}

/// The minimal interface isis needs from a language model.
pub trait Backend {
    /// Get the normalized hidden state at the last token position (memory key).
    fn get_key(&mut self, token_ids: &[i64]) -> Result<Vec<f32>, BoxErr>;

    /// Full forward pass: token_ids → logits for each position.
    fn forward(&mut self, token_ids: &[i64]) -> Result<Vec<Vec<f32>>, BoxErr>;

    /// Run backbone only: token_ids → (hidden_states, full_hidden_states) per position.
    fn run_backbone(&mut self, token_ids: &[i64]) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), BoxErr>;

    /// Run lm_head: hidden_states → logits.
    fn run_lm_head(&mut self, full_hidden: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, BoxErr>;

    /// Hidden dimension of the backbone.
    fn hidden_dim(&self) -> usize;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
}

/// Teach a fact using any backend. Computes keys, surprise, logit biases.
pub fn teach(
    backend: &mut dyn Backend,
    bank: &mut MemoryBank,
    tokenizer: &Tokenizer,
    prompt: &str,
    answer: &str,
    alter: &str,
    importance: f32,
) -> Result<(), BoxErr> {
    let prompt_ids = tokenizer.encode(prompt);
    let prompt_key = backend.get_key(&prompt_ids)?;

    let full_text = format!("{} {}", prompt, answer);
    let full_ids = tokenizer.encode(&full_text);
    let (hidden, full) = backend.run_backbone(&full_ids)?;
    let logits = backend.run_lm_head(&full)?;

    let answer_ids = tokenizer.encode(answer);
    let prompt_len = prompt_ids.len();

    // Surprise
    let mut surprise_sum = 0.0f32;
    let mut surprise_n = 0;
    for (i, &aid) in answer_ids.iter().enumerate() {
        let pos = prompt_len + i - 1;
        if pos < logits.len() {
            let max_l: f32 = logits[pos]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits[pos].iter().map(|&l| (l - max_l).exp()).sum();
            let log_prob = (logits[pos][aid as usize] - max_l) - exp_sum.ln();
            surprise_sum -= log_prob;
            surprise_n += 1;
        }
    }
    let surprise = if surprise_n > 0 {
        (surprise_sum / surprise_n as f32 / 3.0).clamp(0.5, 2.0)
    } else {
        1.0
    };

    // Content keys
    let full_tokens: Vec<String> =
        full_ids.iter().map(|&id| tokenizer.decode(&[id])).collect();
    let mut content_keys = Vec::new();
    for (pos, tok) in full_tokens.iter().enumerate() {
        if MemoryBank::is_skip_token(tok) || pos >= hidden.len() {
            continue;
        }
        let mut k = hidden[pos].clone();
        normalize(&mut k);
        content_keys.push((k, tok.clone(), pos as i32));
    }

    // Logit biases
    let effective_strength = 50.0 * surprise * importance;
    let last_logits = &logits[prompt_len - 1];
    let top_token = last_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0);

    let mut logit_biases = Vec::new();
    for &tid in &answer_ids {
        let mut suppress = Vec::new();
        if top_token != tid as u32 {
            suppress.push((top_token, -effective_strength * 0.3));
        }
        logit_biases.push(LogitBias {
            token_id: tid as u32,
            token: tokenizer.decode(&[tid]),
            strength: effective_strength,
            suppress,
        });
    }

    bank.teach(
        prompt,
        answer,
        alter,
        content_keys,
        prompt_key,
        logit_biases,
        importance,
        surprise,
    );

    Ok(())
}
