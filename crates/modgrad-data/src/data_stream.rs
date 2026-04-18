//! Lazy data streaming: read from disk without loading everything into memory.
//!
//! DataStream trait: Iterator<Item = Sample>. Implementations read lazily,
//! holding at most one buffer in memory. Composable: map, filter, interleave.
//!
//! Replaces MixedStream's Vec<Code> buffer for large-scale training.

use super::tokenize::Code;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

/// One training sample from the stream.
#[derive(Debug, Clone)]
pub struct StreamSample {
    /// Context codes.
    pub context: Vec<Code>,
    /// Target code to predict.
    pub target: Code,
}

/// A lazy data stream that yields samples without loading all data into memory.
pub trait DataStream {
    /// Get the next sample. Returns None when the current epoch is done.
    fn next_sample(&mut self) -> Option<StreamSample>;
    /// Reset to the beginning for the next epoch.
    fn reset(&mut self);
    /// Approximate total samples (may be unknown for some sources).
    fn len_hint(&self) -> Option<usize> { None }
}

// ─── Byte file stream ──────────────────────────────────────

/// Streams bytes from a file using a sliding window.
/// Reads in chunks, never holds more than chunk_size + context_len in memory.
pub struct ByteFileStream {
    path: PathBuf,
    reader: Option<BufReader<std::fs::File>>,
    buffer: Vec<u8>,
    buf_pos: usize,
    context_len: usize,
    chunk_size: usize,
    file_len: u64,
}

impl ByteFileStream {
    pub fn open(path: impl AsRef<Path>, context_len: usize) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file_len = std::fs::metadata(&path)?.len();
        let chunk_size = 64 * 1024; // 64KB chunks
        let mut s = Self {
            path, reader: None, buffer: Vec::new(),
            buf_pos: 0, context_len, chunk_size, file_len,
        };
        s.open_reader()?;
        s.fill_buffer()?;
        Ok(s)
    }

    fn open_reader(&mut self) -> io::Result<()> {
        let file = std::fs::File::open(&self.path)?;
        self.reader = Some(BufReader::with_capacity(self.chunk_size, file));
        Ok(())
    }

    fn fill_buffer(&mut self) -> io::Result<bool> {
        let reader = self.reader.as_mut().ok_or_else(||
            io::Error::new(io::ErrorKind::Other, "reader not open"))?;

        // Keep the last context_len bytes for continuity
        if self.buffer.len() > self.context_len && self.buf_pos > 0 {
            let keep_from = self.buf_pos.saturating_sub(self.context_len);
            self.buffer.drain(..keep_from);
            self.buf_pos -= keep_from;
        }

        let mut chunk = vec![0u8; self.chunk_size];
        let n = reader.read(&mut chunk)?;
        if n == 0 { return Ok(false); }
        chunk.truncate(n);
        self.buffer.extend_from_slice(&chunk);
        Ok(true)
    }
}

impl DataStream for ByteFileStream {
    fn next_sample(&mut self) -> Option<StreamSample> {
        let ctx = self.context_len;

        // Need at least context_len + 1 bytes from buf_pos
        while self.buf_pos + ctx >= self.buffer.len() {
            match self.fill_buffer() {
                Ok(true) => {},
                _ => return None, // EOF or error
            }
        }

        let context: Vec<Code> = self.buffer[self.buf_pos..self.buf_pos + ctx]
            .iter().map(|&b| Code::Text(b)).collect();
        let target = Code::Text(self.buffer[self.buf_pos + ctx]);
        self.buf_pos += 1;

        Some(StreamSample { context, target })
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.buf_pos = 0;
        let _ = self.open_reader();
        let _ = self.fill_buffer();
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.file_len.saturating_sub(self.context_len as u64) as usize)
    }
}

// ─── Interleaved stream ────────────────────────────────────

/// Round-robin across multiple data streams.
/// Each call to next_sample() draws from the next stream in rotation.
pub struct InterleavedStream {
    streams: Vec<Box<dyn DataStream>>,
    current: usize,
}

impl InterleavedStream {
    pub fn new(streams: Vec<Box<dyn DataStream>>) -> Self {
        Self { streams, current: 0 }
    }
}

impl DataStream for InterleavedStream {
    fn next_sample(&mut self) -> Option<StreamSample> {
        if self.streams.is_empty() { return None; }

        // Try each stream in rotation until one yields a sample
        for _ in 0..self.streams.len() {
            let idx = self.current % self.streams.len();
            self.current += 1;
            if let Some(sample) = self.streams[idx].next_sample() {
                return Some(sample);
            }
        }
        None // all streams exhausted
    }

    fn reset(&mut self) {
        for s in &mut self.streams { s.reset(); }
        self.current = 0;
    }
}

// ─── Weighted stream ───────────────────────────────────────

/// Sample from multiple streams with configurable weights.
/// E.g., 70% text + 20% image + 10% audio.
pub struct WeightedStream {
    streams: Vec<Box<dyn DataStream>>,
    #[allow(dead_code)] // raw weights kept alongside cumulative for debugging/introspection
    weights: Vec<f32>,
    cumulative: Vec<f32>,
    rng: u64,
}

impl WeightedStream {
    pub fn new(streams: Vec<Box<dyn DataStream>>, weights: Vec<f32>) -> Self {
        assert_eq!(streams.len(), weights.len());
        let total: f32 = weights.iter().sum();
        let norm: Vec<f32> = weights.iter().map(|w| w / total).collect();
        let mut cumulative = Vec::with_capacity(norm.len());
        let mut acc = 0.0;
        for w in &norm {
            acc += w;
            cumulative.push(acc);
        }
        Self { streams, weights: norm, cumulative, rng: 42 }
    }

    fn pick_stream(&mut self) -> usize {
        self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = (self.rng >> 48) as f32 / 65536.0;
        self.cumulative.iter().position(|&c| r < c).unwrap_or(self.streams.len() - 1)
    }
}

impl DataStream for WeightedStream {
    fn next_sample(&mut self) -> Option<StreamSample> {
        // Try the weighted pick first, then fall back to others
        for _ in 0..self.streams.len() * 2 {
            let idx = self.pick_stream();
            if let Some(sample) = self.streams[idx].next_sample() {
                return Some(sample);
            }
        }
        None
    }

    fn reset(&mut self) {
        for s in &mut self.streams { s.reset(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_stream_reads_lazily() {
        // Create a temp file
        let path = "/tmp/isis_stream_test.txt";
        std::fs::write(path, b"hello world, this is a test of lazy streaming!")
            .unwrap();

        let mut stream = ByteFileStream::open(path, 4).unwrap();

        // First sample: context = "hell", target = 'o'
        let s = stream.next_sample().unwrap();
        assert_eq!(s.context.len(), 4);
        assert_eq!(s.context[0], Code::Text(b'h'));
        assert_eq!(s.target, Code::Text(b'o'));

        // Read more samples
        let mut count = 1;
        while stream.next_sample().is_some() { count += 1; }
        assert!(count > 10, "should yield many samples, got {}", count);

        // Reset and read again
        stream.reset();
        let s2 = stream.next_sample().unwrap();
        assert_eq!(s2.context[0], Code::Text(b'h')); // back to start

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn weighted_stream_respects_weights() {
        let path1 = "/tmp/isis_ws_test1.txt";
        let path2 = "/tmp/isis_ws_test2.txt";
        std::fs::write(path1, &vec![b'a'; 500]).unwrap();
        std::fs::write(path2, &vec![b'b'; 500]).unwrap();

        let s1 = Box::new(ByteFileStream::open(path1, 2).unwrap()) as Box<dyn DataStream>;
        let s2 = Box::new(ByteFileStream::open(path2, 2).unwrap()) as Box<dyn DataStream>;

        let mut stream = WeightedStream::new(vec![s1, s2], vec![0.8, 0.2]);

        let mut a_count = 0;
        let mut b_count = 0;
        for _ in 0..100 {
            if let Some(s) = stream.next_sample() {
                match s.target {
                    Code::Text(b'a') => a_count += 1,
                    Code::Text(b'b') => b_count += 1,
                    _ => {}
                }
            }
        }
        // Should be roughly 80/20
        assert!(a_count > b_count, "a={} should be > b={}", a_count, b_count);

        std::fs::remove_file(path1).ok();
        std::fs::remove_file(path2).ok();
    }
}
