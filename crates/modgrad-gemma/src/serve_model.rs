//! `GemmaLm` — wires Gemma-4 into the model-agnostic `modgrad-serve` server by
//! implementing [`LanguageModel`]. Everything Gemma-specific lives here: the
//! tokenizer, the `<|turn>` chat template, and the `<|channel>` harmony parsing
//! that separates reasoning from the visible answer. The server crate does the
//! HTTP / Anthropic / MCP framing and never sees any of this.

#![cfg(all(feature = "rocm", modgrad_hipcc_kernels))]

use std::io::Cursor;
use std::time::Instant;
use modgrad_device::kfd::gguf::GgufFile;
use modgrad_serve::{LanguageModel, ChatMessage, Role, Generation};
use tokenizers::Tokenizer;
use crate::rocm_gemma::RocmGemma;

/// A lazily-loadable Gemma-4 model behind the server's `LanguageModel` trait.
/// The GGUF bytes + parsed header are kept resident so `unload`/reload only
/// touches VRAM, never the disk.
pub struct GemmaLm {
    gguf: GgufFile,
    bytes: Vec<u8>,
    tok: Tokenizer,
    max_seq: usize,
    bos: u32,
    stop: Vec<u32>,
    name: String,
    model: Option<RocmGemma>,
}

impl GemmaLm {
    /// Read the GGUF + tokenizer from disk (model NOT loaded into VRAM yet —
    /// call `ensure_loaded`, or let the first `generate` do it).
    pub fn open(gguf_path: &str, tok_path: &str, max_seq: usize) -> Result<Self, String> {
        let tok = Tokenizer::from_file(tok_path).map_err(|e| format!("tokenizer: {e}"))?;
        let bytes = std::fs::read(gguf_path).map_err(|e| format!("read gguf: {e}"))?;
        let gguf = GgufFile::parse(&mut Cursor::new(&bytes)).map_err(|e| format!("parse gguf: {e}"))?;
        Ok(Self {
            gguf, bytes, tok, max_seq,
            bos: 2,               // <bos>
            stop: vec![1, 106],   // <eos>, <turn|>
            name: "gemma-4-12b".into(),
            model: None,
        })
    }
}

impl LanguageModel for GemmaLm {
    fn name(&self) -> &str { &self.name }
    fn context_len(&self) -> usize { self.max_seq }

    fn ensure_loaded(&mut self) -> Result<(), String> {
        if self.model.is_none() {
            eprintln!("[gemma] loading model into VRAM ...");
            self.model = Some(RocmGemma::load(&self.gguf, &self.bytes, self.max_seq)?);
        }
        Ok(())
    }
    fn unload(&mut self) { self.model = None; } // Drop frees the HipBuffers => VRAM released
    fn is_loaded(&self) -> bool { self.model.is_some() }

    fn generate(&mut self, messages: &[ChatMessage], max_tokens: usize, wrap: bool,
                on_delta: &mut dyn FnMut(&str)) -> Result<Generation, String> {
        self.ensure_loaded()?;

        // Build the prompt: harmony chat template, or the last user turn raw.
        let prompt = if wrap {
            render_harmony(messages)
        } else {
            messages.iter().rev().find(|m| m.role == Role::User)
                .map(|m| m.content.clone()).unwrap_or_default()
        };
        let enc = self.tok.encode(prompt, false).map_err(|e| format!("encode: {e}"))?;
        let mut ids = vec![self.bos];
        ids.extend_from_slice(enc.get_ids());
        // Front-truncate (keep recent context) to fit the window — Claude Code
        // prompts are large; degrade gracefully instead of erroring.
        let budget = self.max_seq.saturating_sub(max_tokens).saturating_sub(8);
        if ids.len() > budget && budget > 1 {
            let cut = ids.len() - budget;
            ids = std::iter::once(self.bos).chain(ids[cut + 1..].iter().copied()).collect();
        }
        if ids.len() >= self.max_seq { return Err("prompt longer than context window".into()); }
        let prompt_tokens = ids.len();

        // Disjoint field borrows: model (mut) + tok/stop (shared) for the closure.
        let tok = &self.tok;
        let stop = &self.stop;
        let m = self.model.as_mut().ok_or("model not loaded")?;
        m.reset();
        let t = Instant::now();
        let mut acc: Vec<u32> = Vec::new();
        let mut last = String::new();
        let res = m.generate_stream(&ids, max_tokens, stop, |id| {
            acc.push(id);
            // Decode WITH specials so we can see the harmony channel boundary, then
            // stream only the ANSWER (suppressing the "thought" channel header).
            let raw = tok.decode(&acc, false).unwrap_or_default();
            let answer = stream_answer(&raw);
            let cp = common_prefix_len(&last, &answer);
            if answer.len() > cp { on_delta(&answer[cp..]); }
            last = answer;
            true
        });
        let secs = t.elapsed().as_secs_f64();
        res?;
        let raw = tok.decode(&acc, false).unwrap_or_default();
        let (thinking, answer) = split_harmony(&raw);
        eprintln!("[gemma] {prompt_tokens} prompt_tok -> {} tok in {secs:.2}s ({:.1} tok/s)",
            acc.len(), acc.len() as f64 / secs.max(1e-6));
        Ok(Generation {
            answer, thinking, prompt_tokens, output_tokens: acc.len(),
            tok_per_s: ((acc.len() as f64 / secs.max(1e-6)) * 100.0).round() / 100.0,
        })
    }
}

/// Render system + chat turns into Gemma's harmony format, ending open at the
/// model turn so generation continues the assistant reply.
fn render_harmony(messages: &[ChatMessage]) -> String {
    let mut p = String::new();
    for m in messages {
        let role = match m.role { Role::System => "system", Role::User => "user", Role::Assistant => "model" };
        p.push_str(&format!("<|turn>{role}\n{}<turn|>\n", m.content));
    }
    p.push_str("<|turn>model\n");
    p
}

/// The answer-so-far for streaming: everything after the first `<channel|>` (with
/// markers stripped), suppressing the "thought" channel header. Mirrors the
/// behaviour the Anthropic shim relied on so clients see a clean reply.
fn stream_answer(raw: &str) -> String {
    if let Some(p) = raw.find("<channel|>") {
        strip_markers(&raw[p + "<channel|>".len()..])
    } else if raw.contains("<|channel>") {
        String::new() // still inside the channel header — emit nothing yet
    } else {
        strip_markers(raw)
    }
}

/// Strip harmony/turn special-token markers from a string.
fn strip_markers(s: &str) -> String {
    let mut o = s.to_string();
    for m in ["<turn|>", "<|turn>", "<bos>", "<eos>", "<|channel>", "<channel|>", "<pad>", "<|think|>"] {
        o = o.replace(m, "");
    }
    o.trim_start().to_string()
}

/// Split a completed harmony generation into (thinking, answer). Channels look
/// like `<|channel>NAME<channel|>CONTENT`; the "thought"/"think" channel is
/// reasoning, anything else is the answer. No channels => all answer.
fn split_harmony(raw: &str) -> (String, String) {
    fn clean(s: &str) -> String {
        let mut o = s.to_string();
        for m in ["<turn|>", "<|turn>", "<bos>", "<eos>", "<|channel>", "<channel|>", "<pad>"] {
            o = o.replace(m, "");
        }
        o.trim().to_string()
    }
    if !raw.contains("<|channel>") {
        return (String::new(), clean(raw));
    }
    let mut thinking = String::new();
    let mut answer = String::new();
    for seg in raw.split("<|channel>").skip(1) {
        let (name, content) = seg.split_once("<channel|>").unwrap_or(("", seg));
        let content = clean(content);
        if content.is_empty() { continue; }
        let dst = if name.trim_start().starts_with("thought") || name.trim_start().starts_with("think") {
            &mut thinking
        } else { &mut answer };
        if !dst.is_empty() { dst.push('\n'); }
        dst.push_str(&content);
    }
    if answer.is_empty() { std::mem::swap(&mut answer, &mut thinking); }
    (thinking, answer)
}

/// Byte length of the longest char-aligned common prefix of `a` and `b`.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut n = 0;
    for (ai, bc) in a.char_indices().zip(b.chars()) {
        if ai.1 == bc { n = ai.0 + ai.1.len_utf8(); } else { break; }
    }
    n
}
