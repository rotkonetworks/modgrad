//! Minimal Llama-3 BPE tokenizer driven by the GGUF tokenizer metadata.
//!
//! Loads `tokenizer.ggml.tokens` (vocab) + `tokenizer.ggml.merges` (BPE
//! merge ranks) from the GGUF and provides `encode(text) -> Vec<u32>`.
//!
//! Scope: enough to tokenise plain English ASCII text into Llama-3 token IDs.
//! Skips the full tiktoken-style pre-tokenization regex (which needs unicode
//! categories) — uses a simple split that produces correct output for
//! `"tara: Hello world"`-class prompts. If you feed it CJK / emoji / unusual
//! punctuation, results may differ from the reference HF tokenizer.
//!
//! Encoding pipeline per pre-token:
//!   1. UTF-8 bytes → GPT-2 byte-to-unicode mapping (so every byte maps to a
//!      printable unicode char that exists in the vocab)
//!   2. Greedy BPE: at each step, merge the adjacent pair with the lowest
//!      merge rank, until no merges apply
//!   3. Look up each final symbol in the vocab → u32 token id
//!
//! No bos/eos prepended — caller controls that (Orpheus prompts need very
//! specific special-token wrapping).

use std::collections::HashMap;

use super::gguf::{GgufFile, MetaValue};

pub struct LlamaBpeTokenizer {
    /// `vocab[id] = token_string` after byte-to-unicode mapping.
    pub vocab: Vec<String>,
    /// `vocab_lookup[token_string] = id`
    vocab_lookup: HashMap<String, u32>,
    /// `merges[(a, b)] = rank`; lower rank = earlier merge.
    merges: HashMap<(String, String), u32>,
    /// 256-entry GPT-2 byte→char table.
    byte_to_char: [char; 256],
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl LlamaBpeTokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let tokens = match gguf.meta("tokenizer.ggml.tokens") {
            Some(MetaValue::Array(arr)) => arr,
            _ => return Err("missing tokenizer.ggml.tokens".into()),
        };
        let merges = match gguf.meta("tokenizer.ggml.merges") {
            Some(MetaValue::Array(arr)) => arr,
            _ => return Err("missing tokenizer.ggml.merges".into()),
        };

        let vocab: Vec<String> = tokens.iter()
            .map(|v| match v {
                MetaValue::Str(s) => s.clone(),
                _ => String::new(),
            })
            .collect();

        let vocab_lookup: HashMap<String, u32> = vocab.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        let mut merge_map = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            if let MetaValue::Str(s) = m {
                // Merge format is "a b" — split on the FIRST space.
                if let Some(sp) = s.find(' ') {
                    let a = s[..sp].to_string();
                    let b = s[sp+1..].to_string();
                    merge_map.insert((a, b), rank as u32);
                }
            }
        }

        let bos_token_id = gguf.meta("tokenizer.ggml.bos_token_id").and_then(|v| v.as_u32());
        let eos_token_id = gguf.meta("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u32());

        Ok(Self {
            vocab,
            vocab_lookup,
            merges: merge_map,
            byte_to_char: build_byte_to_char(),
            bos_token_id,
            eos_token_id,
        })
    }

    /// Look up a token by exact string match (used for special tokens like
    /// `<|begin_of_text|>` or `<custom_token_N>`).
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.vocab_lookup.get(s).copied()
    }

    /// Encode a plain text string to a sequence of token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for pre in pre_tokenize(text) {
            let symbols = self.bytes_to_symbols(pre.as_bytes());
            let merged = self.bpe_merge(symbols);
            for sym in merged {
                if let Some(&id) = self.vocab_lookup.get(&sym) {
                    out.push(id);
                } else {
                    // Token missing from vocab — should be extremely rare for
                    // ASCII input given the 156k vocab covers every byte.
                    // Emit per-character fallback so we don't silently drop.
                    for ch in sym.chars() {
                        let key = ch.to_string();
                        if let Some(&id) = self.vocab_lookup.get(&key) {
                            out.push(id);
                        }
                    }
                }
            }
        }
        out
    }

    /// Map raw UTF-8 bytes to the GPT-2 byte-encoded characters used as
    /// vocab symbols, returning each as its own single-char `String` so
    /// the BPE merge loop can manipulate them.
    fn bytes_to_symbols(&self, bytes: &[u8]) -> Vec<String> {
        bytes.iter()
            .map(|&b| self.byte_to_char[b as usize].to_string())
            .collect()
    }

    /// Greedy BPE: while there exists an adjacent pair with a merge rank
    /// in the merge table, merge the pair with the LOWEST rank. Repeat.
    fn bpe_merge(&self, mut symbols: Vec<String>) -> Vec<String> {
        if symbols.len() < 2 { return symbols; }
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len() - 1 {
                let key = (symbols[i].clone(), symbols[i+1].clone());
                if let Some(&rank) = self.merges.get(&key) {
                    if best.map_or(true, |(_, br)| rank < br) {
                        best = Some((i, rank));
                    }
                }
            }
            match best {
                Some((i, _)) => {
                    let merged = format!("{}{}", symbols[i], symbols[i+1]);
                    symbols[i] = merged;
                    symbols.remove(i + 1);
                }
                None => break,
            }
        }
        symbols
    }
}

/// Simple pre-tokenization: split on whitespace but ATTACH the leading
/// space to the following token (Llama-3 convention — most non-first
/// tokens start with " " in the vocab). Punctuation stays attached to
/// the previous word, which matches the dominant Llama-3 behaviour for
/// our use case (English sentences with colons / commas).
///
/// This is intentionally simpler than the full tiktoken regex. Good for
/// `"tara: Hello world"`-style inputs; CJK / emoji not guaranteed.
fn pre_tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_whitespace() {
            // Flush current token, then start a new one beginning with ' '.
            if !buf.is_empty() { out.push(std::mem::take(&mut buf)); }
            buf.push(' ');
            // Collapse runs of whitespace into a single ' '.
            while let Some(&peek) = chars.peek() {
                if peek.is_whitespace() { chars.next(); } else { break; }
            }
        } else {
            buf.push(c);
        }
    }
    if !buf.is_empty() { out.push(buf); }
    out
}

/// GPT-2 byte-to-unicode map. Every byte 0..=255 → a unique printable char.
/// Printable ASCII (and Latin-1 supplement) maps to itself; control chars
/// and non-printable bytes get mapped to characters starting at U+0100.
fn build_byte_to_char() -> [char; 256] {
    let mut bs: Vec<u32> = Vec::with_capacity(256);
    bs.extend(33..=126);        // ! .. ~
    bs.extend(161..=172);       // ¡ .. ¬
    bs.extend(174..=255);       // ® .. ÿ
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..=255u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut sorted: Vec<(u32, u32)> = bs.into_iter().zip(cs.into_iter()).collect();
    sorted.sort_by_key(|&(b, _)| b);
    let mut table = ['?'; 256];
    for (i, (_, c)) in sorted.into_iter().enumerate() {
        table[i] = char::from_u32(c).unwrap_or('?');
    }
    table
}
