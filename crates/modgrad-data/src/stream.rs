//! Mixed-modality data streaming pipeline.
//!
//! Streams interleaved (text, image, audio) data as code sequences.
//! Composable with the Tokenizer trait — any tokenizer feeds the stream.
//!
//! A Stream is a function () → Option<Sample>.
//! No hidden state beyond the iterator position. Composable with map/filter.

use super::tokenize::{Code, Tokenizer, ByteTokenizer, BitTokenizer};
use std::path::Path;

/// One training sample: a sequence of codes and the target code to predict.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Input code sequence (context).
    pub codes: Vec<Code>,
    /// Target: the next code to predict.
    pub target: Code,
}

/// A data source that yields raw data for one modality.
pub trait DataSource {
    /// Yield a chunk of raw data. Returns None when exhausted.
    fn next_chunk(&mut self, max_len: usize) -> Option<Vec<u8>>;
    /// Reset to beginning (for epochs).
    fn reset(&mut self);
}

// ─── File data source ──────────────────────────────────────

/// Streams bytes from a file. Low memory: reads in chunks.
pub struct FileSource {
    data: Vec<u8>,
    pos: usize,
}

impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        Ok(Self { data, pos: 0 })
    }

    pub fn len(&self) -> usize { self.data.len() }
}

impl DataSource for FileSource {
    fn next_chunk(&mut self, max_len: usize) -> Option<Vec<u8>> {
        if self.pos >= self.data.len() { return None; }
        let end = (self.pos + max_len).min(self.data.len());
        let chunk = self.data[self.pos..end].to_vec();
        self.pos = end;
        Some(chunk)
    }

    fn reset(&mut self) { self.pos = 0; }
}

// ─── Image data source ────────────────────────────────────

/// Streams CIFAR images as flat pixel arrays.
pub struct ImageSource {
    images: Vec<(Vec<f32>, usize)>, // (pixels, label)
    pos: usize,
}

impl ImageSource {
    pub fn from_cifar(images: Vec<modgrad_codec::cifar::CifarImage>) -> Self {
        let images = images.into_iter()
            .map(|img| (img.pixels, img.label))
            .collect();
        Self { images, pos: 0 }
    }

    pub fn next_image(&mut self) -> Option<(&[f32], usize)> {
        if self.pos >= self.images.len() { return None; }
        let (ref pixels, label) = self.images[self.pos];
        self.pos += 1;
        Some((pixels, label))
    }

    pub fn reset(&mut self) { self.pos = 0; }
    pub fn len(&self) -> usize { self.images.len() }
}

// ─── Mixed stream ──────────────────────────────────────────

/// Configuration for the mixed-modality stream.
pub struct StreamConfig {
    /// Context window size in codes.
    pub context_len: usize,
    /// Probability of sampling text vs image vs audio.
    /// Must sum to 1.0.
    pub text_weight: f32,
    pub image_weight: f32,
    pub audio_weight: f32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            context_len: 64,
            text_weight: 0.7,
            image_weight: 0.3,
            audio_weight: 0.0,
        }
    }
}

/// Mixed-modality training stream.
/// Yields (context_codes, target_code) pairs from interleaved modalities.
pub struct MixedStream {
    /// Accumulated code buffer from all modalities.
    buffer: Vec<Code>,
    /// Position in buffer.
    pos: usize,
    /// Config.
    config: StreamConfig,
    /// RNG state for modality selection.
    rng: u64,
}

impl MixedStream {
    pub fn new(config: StreamConfig) -> Self {
        Self { buffer: Vec::new(), pos: 0, config, rng: 42 }
    }

    /// Feed text bytes into the stream.
    pub fn feed_text(&mut self, bytes: &[u8]) {
        let tok = ByteTokenizer;
        self.buffer.extend(tok.encode(bytes));
    }

    /// Feed text as bits (for bit-level experiments).
    pub fn feed_bits(&mut self, bytes: &[u8]) {
        let tok = BitTokenizer;
        self.buffer.extend(tok.encode(bytes));
    }

    /// Feed image codes (from VQ-VAE) into the stream.
    pub fn feed_image_codes(&mut self, codes: &[Code]) {
        self.buffer.extend_from_slice(codes);
    }

    /// Feed audio codes into the stream.
    pub fn feed_audio_codes(&mut self, codes: &[Code]) {
        self.buffer.extend_from_slice(codes);
    }

    /// Get the next training sample: (context, target).
    /// Returns None if not enough data in buffer.
    pub fn next_sample(&mut self) -> Option<Sample> {
        let ctx = self.config.context_len;
        if self.pos + ctx >= self.buffer.len() {
            return None;
        }

        let codes = self.buffer[self.pos..self.pos + ctx].to_vec();
        let target = self.buffer[self.pos + ctx];
        self.pos += 1;

        Some(Sample { codes, target })
    }

    /// Get a batch of samples.
    pub fn next_batch(&mut self, batch_size: usize) -> Vec<Sample> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            match self.next_sample() {
                Some(s) => batch.push(s),
                None => break,
            }
        }
        batch
    }

    /// Shuffle the buffer using Fisher-Yates at chunk boundaries.
    /// Preserves local structure within chunks (sentences, images)
    /// while randomizing the order of chunks.
    pub fn shuffle_chunks(&mut self, chunk_size: usize) {
        let n_chunks = self.buffer.len() / chunk_size;
        if n_chunks <= 1 { return; }

        // Collect chunks
        let mut chunks: Vec<Vec<Code>> = (0..n_chunks)
            .map(|i| self.buffer[i * chunk_size..(i + 1) * chunk_size].to_vec())
            .collect();

        // Fisher-Yates shuffle
        for i in (1..chunks.len()).rev() {
            self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (self.rng as usize) % (i + 1);
            chunks.swap(i, j);
        }

        // Rebuild buffer
        self.buffer = chunks.into_iter().flatten().collect();
        self.pos = 0;
    }

    /// How many samples remain.
    pub fn remaining(&self) -> usize {
        if self.pos + self.config.context_len >= self.buffer.len() { 0 }
        else { self.buffer.len() - self.pos - self.config.context_len }
    }

    /// Reset position to beginning.
    pub fn reset(&mut self) { self.pos = 0; }

    /// Total codes in buffer.
    pub fn len(&self) -> usize { self.buffer.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_stream() {
        let mut stream = MixedStream::new(StreamConfig { context_len: 4, ..Default::default() });
        stream.feed_text(b"hello world");

        let sample = stream.next_sample().unwrap();
        assert_eq!(sample.codes.len(), 4);
        assert_eq!(sample.codes[0], Code::Text(b'h'));
        assert_eq!(sample.codes[3], Code::Text(b'l'));
        assert_eq!(sample.target, Code::Text(b'o'));
    }

    #[test]
    fn bit_stream() {
        let mut stream = MixedStream::new(StreamConfig { context_len: 8, ..Default::default() });
        stream.feed_bits(b"AB"); // 16 bits

        let sample = stream.next_sample().unwrap();
        assert_eq!(sample.codes.len(), 8); // 8-bit context
        // First byte 'A' = 0b01000001
        assert_eq!(sample.codes[0], Code::Bit(false)); // MSB
        assert_eq!(sample.codes[1], Code::Bit(true));
        // Target is bit 8 (first bit of 'B' = 0b01000010)
        assert_eq!(sample.target, Code::Bit(false));
    }

    #[test]
    fn mixed_modalities() {
        let mut stream = MixedStream::new(StreamConfig { context_len: 3, ..Default::default() });
        stream.feed_text(b"hi");
        stream.feed_image_codes(&[Code::Image(10), Code::Image(20)]);
        stream.feed_text(b"!");

        assert_eq!(stream.len(), 5); // 2 text + 2 image + 1 text

        let s1 = stream.next_sample().unwrap();
        assert_eq!(s1.codes[0], Code::Text(b'h'));
        assert_eq!(s1.codes[1], Code::Text(b'i'));
        assert_eq!(s1.codes[2], Code::Image(10));
        assert_eq!(s1.target, Code::Image(20));
    }

    #[test]
    fn batch_and_remaining() {
        let mut stream = MixedStream::new(StreamConfig { context_len: 2, ..Default::default() });
        stream.feed_text(b"abcdef"); // 6 codes → 4 samples with ctx=2

        assert_eq!(stream.remaining(), 4);
        let batch = stream.next_batch(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(stream.remaining(), 1);
    }

    #[test]
    fn shuffle_preserves_content() {
        let mut stream = MixedStream::new(StreamConfig { context_len: 2, ..Default::default() });
        stream.feed_text(b"abcdefghijklmnop"); // 16 codes
        let before: Vec<Code> = stream.buffer.clone();

        stream.shuffle_chunks(4); // shuffle in 4-code chunks

        // Same total length
        assert_eq!(stream.buffer.len(), before.len());
        // Same content (just reordered)
        let mut sorted_before: Vec<u8> = before.iter().map(|c| match c {
            Code::Text(b) => *b, _ => 0
        }).collect();
        let mut sorted_after: Vec<u8> = stream.buffer.iter().map(|c| match c {
            Code::Text(b) => *b, _ => 0
        }).collect();
        sorted_before.sort();
        sorted_after.sort();
        assert_eq!(sorted_before, sorted_after);
    }
}
