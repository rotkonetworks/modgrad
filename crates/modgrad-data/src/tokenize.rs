//! isis::tokenize — unified tokenization for all modalities.
//!
//! The type system enforces correct modality handling:
//!   Code::Text(b)   — a byte value, cannot be confused with image/audio
//!   Code::Image(i)  — a VQ-VAE codebook index
//!   Code::Audio(i)  — an audio codec codebook index
//!
//! Tokenizers implement a common trait. Embeddings know how to look up
//! each variant correctly. Wrong modality mixing is impossible by construction.

use modgrad_codec::ngram_hash::NgramHashEmbeddings;
use modgrad_codec::vqvae::VqVae;
use modgrad_compute::neuron::SimpleRng;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
// CODES — type-safe discrete tokens
// ═══════════════════════════════════════════════════════════════

/// A discrete code from any modality.
///
/// Named variants for well-known modalities (type-safe, ergonomic).
/// `Extension` variant for runtime-extensible modalities (no recompilation).
/// Adding touch, radar, protein sequences, or brain-computer interface
/// codes only requires registering a new modality ID — not rebuilding the SDK.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Code {
    /// A single bit (0 or 1).
    Bit(bool),
    /// A raw byte (0..255).
    Text(u8),
    /// A VQ-VAE image codebook index.
    Image(u16),
    /// Audio semantic token — WHAT is said.
    AudioSemantic(u16),
    /// Audio acoustic token — HOW it sounds.
    AudioAcoustic(u16),
    /// Extension: (modality_id, value). Runtime-extensible.
    /// Register new modalities via CodeLayout::register_extension().
    Extension(u16, u32),
}

/// Well-known modality IDs (for Extension codes and layout registration).
pub mod modality {
    pub const BIT: u16 = 0;
    pub const TEXT: u16 = 1;
    pub const IMAGE: u16 = 2;
    pub const AUDIO_SEMANTIC: u16 = 3;
    pub const AUDIO_ACOUSTIC: u16 = 4;
    // Extension IDs start at 256 to avoid collision with builtins.
    pub const EXTENSION_BASE: u16 = 256;
}

impl Code {
    /// Map to a global index for a unified embedding table.
    pub fn global_index(&self, layout: &CodeLayout) -> usize {
        match self {
            Code::Bit(b) => *b as usize,
            Code::Text(b) => layout.text_offset + *b as usize,
            Code::Image(i) => layout.image_offset + *i as usize,
            Code::AudioSemantic(i) => layout.audio_sem_offset + *i as usize,
            Code::AudioAcoustic(i) => layout.audio_aco_offset + *i as usize,
            Code::Extension(mod_id, val) => {
                layout.extension_offset(*mod_id)
                    .expect("unregistered extension modality") + *val as usize
            }
        }
    }

    /// Is this a code the brain should attend to for understanding?
    /// Acoustic tokens are for reconstruction, not comprehension.
    pub fn is_semantic(&self) -> bool {
        !matches!(self, Code::AudioAcoustic(_))
    }

    /// The modality ID of this code.
    pub fn modality_id(&self) -> u16 {
        match self {
            Code::Bit(_) => modality::BIT,
            Code::Text(_) => modality::TEXT,
            Code::Image(_) => modality::IMAGE,
            Code::AudioSemantic(_) => modality::AUDIO_SEMANTIC,
            Code::AudioAcoustic(_) => modality::AUDIO_ACOUSTIC,
            Code::Extension(id, _) => *id,
        }
    }
}

/// Code range layout for the unified embedding table.
///
/// Global index layout:
///   [0, 2)                                        — bit codes
///   [2, 258)                                      — text bytes
///   [258, 258+image)                              — image VQ codes
///   [258+image, 258+image+audio_sem)              — audio semantic (VQ)
///   [258+image+audio_sem, total)                  — audio acoustic (FSQ)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLayout {
    pub bit_offset: usize,
    pub bit_size: usize,
    pub text_offset: usize,
    pub text_size: usize,
    pub image_offset: usize,
    pub image_size: usize,
    pub audio_sem_offset: usize,
    pub audio_sem_size: usize,   // e.g., 8192 (VQ codebook for semantic)
    pub audio_aco_offset: usize,
    pub audio_aco_size: usize,
    /// Extension modalities: (modality_id, offset, vocab_size).
    pub extensions: Vec<(u16, usize, usize)>,
}

impl CodeLayout {
    /// Full multimodal layout.
    pub fn new(n_image_codes: usize, n_audio_sem: usize, n_audio_aco: usize) -> Self {
        let audio_sem_offset = 258 + n_image_codes;
        Self {
            bit_offset: 0, bit_size: 2,
            text_offset: 2, text_size: 256,
            image_offset: 258, image_size: n_image_codes,
            audio_sem_offset, audio_sem_size: n_audio_sem,
            audio_aco_offset: audio_sem_offset + n_audio_sem,
            audio_aco_size: n_audio_aco,
            extensions: Vec::new(),
        }
    }

    /// Text-only layout.
    pub fn text_only() -> Self { Self::new(0, 0, 0) }

    /// Bits-only layout.
    pub fn bits_only() -> Self {
        let mut l = Self::new(0, 0, 0);
        l.text_size = 0; l.text_offset = 2;
        l.image_offset = 2; l.audio_sem_offset = 2; l.audio_aco_offset = 2;
        l
    }

    /// Register a new extension modality. Returns the assigned offset.
    /// Call this before creating the EmbedTable.
    pub fn register_extension(&mut self, modality_id: u16, vocab_size: usize) -> usize {
        let offset = self.total_codes();
        self.extensions.push((modality_id, offset, vocab_size));
        offset
    }

    /// Get the offset for an extension modality.
    pub fn extension_offset(&self, modality_id: u16) -> Option<usize> {
        self.extensions.iter()
            .find(|(id, _, _)| *id == modality_id)
            .map(|(_, offset, _)| *offset)
    }

    /// Total vocabulary size across all modalities including extensions.
    pub fn total_codes(&self) -> usize {
        let base = self.bit_size + self.text_size + self.image_size
            + self.audio_sem_size + self.audio_aco_size;
        let ext: usize = self.extensions.iter().map(|(_, _, sz)| sz).sum();
        base + ext
    }
}

// ═══════════════════════════════════════════════════════════════
// TOKENIZER TRAIT — composable service interface
// ═══════════════════════════════════════════════════════════════

/// A tokenizer converts raw modality data into discrete codes.
/// Pure function: no hidden state mutation.
pub trait Tokenizer {
    /// Raw input type (bytes, pixels, waveform).
    type Input: ?Sized;
    /// Encode raw input into codes.
    fn encode(&self, input: &Self::Input) -> Vec<Code>;
}

// ─── Bit tokenizer ─────────────────────────────────────────

/// Decompose bytes into individual bits (MSB first).
/// A byte 0b01000001 ('A') becomes [false, true, false, false, false, false, false, true].
/// The model can discover byte boundaries from entropy patterns.
pub struct BitTokenizer;

impl Tokenizer for BitTokenizer {
    type Input = [u8];
    fn encode(&self, input: &[u8]) -> Vec<Code> {
        let mut codes = Vec::with_capacity(input.len() * 8);
        for &byte in input {
            for bit in (0..8).rev() {
                codes.push(Code::Bit((byte >> bit) & 1 == 1));
            }
        }
        codes
    }
}

// ─── Byte tokenizer ────────────────────────────────────────

/// Trivial tokenizer: bytes are already codes.
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    type Input = [u8];
    fn encode(&self, input: &[u8]) -> Vec<Code> {
        input.iter().map(|&b| Code::Text(b)).collect()
    }
}

// ─── VQ-VAE image tokenizer ───────────────────────────────

/// Image tokenizer backed by a VQ-VAE.
/// Input: flat pixel array [3 × H × W] as f32.
pub struct ImageTokenizer<'a> {
    pub vqvae: &'a VqVae,
}

impl<'a> Tokenizer for ImageTokenizer<'a> {
    type Input = [f32];
    fn encode(&self, pixels: &[f32]) -> Vec<Code> {
        let indices = self.vqvae.tokenize(pixels);
        indices.into_iter().map(|i| Code::Image(i as u16)).collect()
    }
}

// ─── Audio tokenizer ──────────────────────────────────────

/// Audio tokenizer backed by AudioCodec (WavTokenizer-style).
/// Input: mono waveform samples as &[f32].
/// Output: AudioSemantic codes (single codebook, 75 codes/sec at 24kHz).
pub struct AudioTokenizer<'a> {
    pub codec: &'a modgrad_codec::audio_codec::AudioCodec,
}

impl<'a> Tokenizer for AudioTokenizer<'a> {
    type Input = [f32];
    fn encode(&self, waveform: &[f32]) -> Vec<Code> {
        let codes = self.codec.tokenize(waveform);
        codes.into_iter().map(|i| Code::AudioSemantic(i as u16)).collect()
    }
}

// ═══════════════════════════════════════════════════════════════
// EMBEDDINGS — codes → dense vectors
// ═══════════════════════════════════════════════════════════════

/// Unified embedding table for all modalities.
/// Separate concern from tokenization — this is a weight matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedTable {
    /// Flat table: [total_codes × embed_dim].
    pub weight: Vec<f32>,
    pub embed_dim: usize,
    pub layout: CodeLayout,
}

impl EmbedTable {
    pub fn new(embed_dim: usize, layout: CodeLayout) -> Self {
        let total = layout.total_codes();
        let mut rng = SimpleRng::new(total as u64 * embed_dim as u64);
        let scale = 1.0 / (embed_dim as f32).sqrt();
        let weight: Vec<f32> = (0..total * embed_dim)
            .map(|_| rng.next_normal() * scale)
            .collect();
        Self { weight, embed_dim, layout }
    }

    /// Look up embedding for a single code.
    pub fn embed_one(&self, code: Code) -> &[f32] {
        let d = self.embed_dim;
        let idx = code.global_index(&self.layout);
        &self.weight[idx * d..(idx + 1) * d]
    }

    /// Embed a sequence of codes into flat [n × embed_dim] tokens.
    pub fn embed(&self, codes: &[Code]) -> Vec<f32> {
        let d = self.embed_dim;
        let mut tokens = Vec::with_capacity(codes.len() * d);
        for &code in codes {
            tokens.extend_from_slice(self.embed_one(code));
        }
        tokens
    }

    /// Total parameters.
    pub fn n_params(&self) -> usize { self.weight.len() }

    /// Expand the embedding table for a newly registered modality.
    /// Existing embeddings are preserved. New rows are initialized randomly.
    /// Call this after CodeLayout::register_extension().
    pub fn expand(&mut self, additional_codes: usize) {
        let d = self.embed_dim;
        let mut rng = SimpleRng::new((self.weight.len() + additional_codes) as u64);
        let scale = 1.0 / (d as f32).sqrt();
        self.weight.reserve(additional_codes * d);
        for _ in 0..additional_codes * d {
            self.weight.push(rng.next_normal() * scale);
        }
    }

    /// Shrink by removing the last N codes (e.g., when removing a modality).
    /// Only safe if those codes are at the end of the layout.
    pub fn shrink(&mut self, codes_to_remove: usize) {
        let d = self.embed_dim;
        let remove_floats = codes_to_remove * d;
        if remove_floats <= self.weight.len() {
            self.weight.truncate(self.weight.len() - remove_floats);
        }
    }

    /// Current vocabulary size (total codes this table can embed).
    pub fn vocab_size(&self) -> usize {
        self.weight.len() / self.embed_dim
    }
}

/// Embedding function: combines embed table + optional n-gram hash.
/// This is the transform, separate from the weight data.
pub fn embed_with_ngram(
    codes: &[Code],
    table: &EmbedTable,
    ngram: Option<&NgramHashEmbeddings>,
) -> Vec<f32> {
    let d = table.embed_dim;
    match ngram {
        Some(nge) => {
            // For text codes, use n-gram augmentation.
            // For non-text, use plain lookup.
            let mut tokens = Vec::with_capacity(codes.len() * d);
            // Collect raw bytes for n-gram context
            let bytes: Vec<u8> = codes.iter().map(|c| match c {
                Code::Text(b) => *b,
                _ => 0, // non-text codes don't contribute to n-gram context
            }).collect();

            for (i, &code) in codes.iter().enumerate() {
                let mut emb = table.embed_one(code).to_vec();
                if matches!(code, Code::Text(_)) {
                    nge.augment(&mut emb, &bytes, i);
                    let norm = 1.0 / (nge.tables.len() + 1) as f32;
                    for v in &mut emb { *v *= norm; }
                }
                tokens.extend_from_slice(&emb);
            }
            tokens
        }
        None => table.embed(codes),
    }
}

/// Convenience: interleave code sequences from different modalities.
pub fn interleave(sequences: &[&[Code]]) -> Vec<Code> {
    let total: usize = sequences.iter().map(|s| s.len()).sum();
    let mut result = Vec::with_capacity(total);
    for seq in sequences {
        result.extend_from_slice(seq);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_type_safety() {
        let bit = Code::Bit(true);
        let text = Code::Text(b'a');
        let image = Code::Image(42);
        let audio_sem = Code::AudioSemantic(100);
        let audio_aco = Code::AudioAcoustic(15);

        let layout = CodeLayout::new(4096, 8192, 21);

        assert_eq!(bit.global_index(&layout), 1);
        assert_eq!(text.global_index(&layout), 2 + 97);
        assert_eq!(image.global_index(&layout), 258 + 42);
        assert_eq!(audio_sem.global_index(&layout), 258 + 4096 + 100);
        assert_eq!(audio_aco.global_index(&layout), 258 + 4096 + 8192 + 15);

        // Pattern matching enforces modality awareness
        match text {
            Code::Text(b) => assert_eq!(b, b'a'),
            _ => panic!("should be text"),
        }
    }

    #[test]
    fn bit_tokenizer() {
        let tok = BitTokenizer;
        let codes = tok.encode(&[0b01000001]); // 'A'
        assert_eq!(codes.len(), 8);
        assert_eq!(codes[0], Code::Bit(false)); // MSB = 0
        assert_eq!(codes[1], Code::Bit(true));  // bit 6 = 1
        assert_eq!(codes[7], Code::Bit(true));  // LSB = 1
        // Reconstruct: 01000001 = 65 = 'A'
    }

    #[test]
    fn bit_roundtrip() {
        let tok = BitTokenizer;
        let original = b"Hi";
        let bits = tok.encode(original);
        assert_eq!(bits.len(), 16); // 2 bytes × 8 bits

        // Reconstruct bytes from bits
        let mut reconstructed = Vec::new();
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &code) in chunk.iter().enumerate() {
                if let Code::Bit(true) = code {
                    byte |= 1 << (7 - i);
                }
            }
            reconstructed.push(byte);
        }
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn byte_tokenizer() {
        let tok = ByteTokenizer;
        let codes = tok.encode(b"hello");
        assert_eq!(codes.len(), 5);
        assert_eq!(codes[0], Code::Text(b'h'));
        assert_eq!(codes[4], Code::Text(b'o'));
    }

    #[test]
    fn embed_table_lookup() {
        let layout = CodeLayout::new(256, 0, 0);
        let table = EmbedTable::new(8, layout);
        let code = Code::Text(42);
        let emb = table.embed_one(code);
        assert_eq!(emb.len(), 8);
        // Same code gives same embedding
        assert_eq!(emb, table.embed_one(code));
    }

    #[test]
    fn dynamic_embed_expand() {
        let layout = CodeLayout::new(0, 0, 0); // text-only: 258 codes
        let mut table = EmbedTable::new(8, layout.clone());
        let initial_size = table.vocab_size();
        assert_eq!(initial_size, 258); // 2 bits + 256 text

        // Expand for 100 new codes (e.g., a new modality)
        let old_text_embed = table.embed_one(Code::Text(b'a')).to_vec();
        table.expand(100);
        assert_eq!(table.vocab_size(), 358);

        // Existing embeddings unchanged
        assert_eq!(table.embed_one(Code::Text(b'a')), old_text_embed.as_slice());
    }

    #[test]
    fn extension_modality() {
        let mut layout = CodeLayout::new(4096, 8192, 21);
        let touch_id = modality::EXTENSION_BASE;
        let offset = layout.register_extension(touch_id, 1024);

        let code = Code::Extension(touch_id, 42);
        assert_eq!(code.global_index(&layout), offset + 42);
        assert_eq!(code.modality_id(), touch_id);

        // Total codes includes extension
        assert!(layout.total_codes() > 258 + 4096 + 8192 + 21);
    }

    #[test]
    fn interleave_preserves_modality() {
        let text = vec![Code::Text(b'h'), Code::Text(b'i')];
        let image = vec![Code::Image(1), Code::Image(2), Code::Image(3)];
        let mixed = interleave(&[&text, &image]);
        assert_eq!(mixed.len(), 5);
        assert!(matches!(mixed[0], Code::Text(b'h')));
        assert!(matches!(mixed[2], Code::Image(1)));
    }
}
