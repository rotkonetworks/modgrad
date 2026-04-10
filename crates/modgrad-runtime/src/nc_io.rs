//! Real-time I/O for the Neural Computer.
//!
//! Converts between physical signals and the NC's token stream:
//!   mic     → AudioCodec.encode → audio tokens
//!   camera  → VqVae.encode      → image tokens
//!   speaker ← AudioCodec.decode ← audio tokens
//!   display ← VqVae.decode      ← image tokens
//!
//! Each I/O source runs in its own thread, producing/consuming tokens
//! through a channel. The NC loop reads from all input channels,
//! processes tokens, and writes to output channels.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use modgrad_codec::audio_codec::AudioCodec;
use modgrad_codec::vqvae::VqVae;

use super::regional::*;

// ── Token-level I/O events ───────────────────────────────

/// An event entering the NC from the outside world.
pub enum NcInput {
    /// Typed text from keyboard.
    Text(String),
    /// Audio tokens from microphone (one chunk = ~1 second).
    Audio(Vec<usize>),
    /// Image tokens from camera (one frame).
    Image(Vec<usize>),
    /// Raw action (already tokenized).
    Action(Vec<usize>),
    /// Shutdown signal.
    Quit,
}

/// An event leaving the NC to the outside world.
pub enum NcOutput {
    /// Text bytes to display.
    Text(String),
    /// Audio codes to decode and play.
    AudioCodes(Vec<usize>),
    /// Image codes to decode and display.
    ImageCodes(Vec<usize>),
    /// Action tokens (for logging/replay).
    Action(Vec<usize>),
}

// ── Audio I/O ────────────────────────────────────────────

/// Reads raw audio from a file or device, encodes to tokens.
/// Sends NcInput::Audio to the NC at ~1 second intervals.
pub fn audio_input_thread(
    path: &str,
    tx: mpsc::Sender<NcInput>,
) -> thread::JoinHandle<()> {
    let path = path.to_string();
    thread::spawn(move || {
        let codec = AudioCodec::new_24khz();
        let sample_rate = 24000;
        let chunk_samples = sample_rate; // 1 second chunks

        // Read WAV file (or in future: open ALSA/PulseAudio device)
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  audio input: failed to read {path}: {e}");
                return;
            }
        };

        // Parse WAV: skip 44-byte header, 16-bit PCM mono
        let samples: Vec<f32> = if data.len() > 44 && &data[..4] == b"RIFF" {
            data[44..].chunks_exact(2).map(|c| {
                i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0
            }).collect()
        } else {
            // Raw f32
            data.chunks_exact(4).map(|c| {
                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
            }).collect()
        };

        eprintln!("  audio input: {:.1}s of audio from {path}",
            samples.len() as f32 / sample_rate as f32);

        // Stream chunks at real-time pace
        let start = Instant::now();
        for (i, chunk) in samples.chunks(chunk_samples).enumerate() {
            if chunk.len() < 320 { break; } // too short for codec

            let codes = codec.tokenize(chunk);
            let tokens = audio_codes_to_tokens(&codes);

            // Wait until real-time position
            let target = Duration::from_secs_f32(i as f32);
            let elapsed = start.elapsed();
            if target > elapsed {
                thread::sleep(target - elapsed);
            }

            if tx.send(NcInput::Audio(tokens)).is_err() { break; }
        }
    })
}

/// Decodes audio tokens and writes to a WAV file (or in future: speaker).
pub struct AudioOutput {
    codec: AudioCodec,
    samples: Vec<f32>,
    path: String,
}

impl AudioOutput {
    pub fn new(path: &str) -> Self {
        Self {
            codec: AudioCodec::new_24khz(),
            samples: Vec::new(),
            path: path.to_string(),
        }
    }

    /// Decode audio codes and accumulate samples.
    pub fn write(&mut self, codes: &[usize]) {
        // Convert from unified token space back to codec indices
        let codec_codes: Vec<usize> = codes.iter()
            .filter_map(|&t| {
                if t >= TOKEN_AUD_OFFSET && t < TOKEN_AUD_OFFSET + TOKEN_AUD_CODES {
                    Some(t - TOKEN_AUD_OFFSET)
                } else {
                    None
                }
            }).collect();

        if codec_codes.is_empty() { return; }

        // Decode codes → codebook vectors → waveform
        let vectors = self.codec.vq.decode_indices(&codec_codes);
        // Reshape to channel-first for decoder
        let n_frames = codec_codes.len();
        let d = self.codec.code_dim;
        let mut chan_first = vec![0.0f32; d * n_frames];
        for c in 0..d {
            for t in 0..n_frames {
                chan_first[c * n_frames + t] = vectors[t * d + c];
            }
        }
        let decoded = self.codec.decode(&chan_first, n_frames);
        self.samples.extend_from_slice(&decoded);
    }

    /// Flush accumulated audio to WAV file.
    pub fn flush(&self) -> std::io::Result<()> {
        if self.samples.is_empty() { return Ok(()); }
        write_wav(&self.path, &self.samples, 24000)
    }
}

/// Write 16-bit PCM WAV file.
fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    use std::io::Write;
    let n = samples.len();
    let data_size = (n * 2) as u32;
    let file_size = 36 + data_size;

    let mut f = std::fs::File::create(path)?;
    f.write_all(b"RIFF")?;
    f.write_all(&file_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;     // chunk size
    f.write_all(&1u16.to_le_bytes())?;      // PCM
    f.write_all(&1u16.to_le_bytes())?;      // mono
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?;      // block align
    f.write_all(&16u16.to_le_bytes())?;     // bits per sample
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for &s in samples {
        let i = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&i.to_le_bytes())?;
    }
    Ok(())
}

// ── Camera I/O ───────────────────────────────────────────

/// Reads frames from a directory (or in future: V4L2 device),
/// encodes to image tokens, sends to NC.
pub fn camera_input_thread(
    path: &str,
    fps: f32,
    tx: mpsc::Sender<NcInput>,
) -> thread::JoinHandle<()> {
    let path = path.to_string();
    thread::spawn(move || {
        let vae = VqVae::new(4096, 64);
        let interval = Duration::from_secs_f32(1.0 / fps);

        // Read frame files sorted by name
        let mut paths: Vec<_> = std::fs::read_dir(&path)
            .into_iter().flatten().flatten()
            .filter(|e| {
                e.path().extension()
                    .map(|x| x == "bin" || x == "raw" || x == "ppm")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();
        paths.sort();

        eprintln!("  camera input: {} frames from {path} at {fps}fps", paths.len());

        let start = Instant::now();
        for (i, frame_path) in paths.iter().enumerate() {
            if let Ok(data) = std::fs::read(frame_path) {
                if data.len() >= 3072 {
                    let pixels: Vec<f32> = data[..3072].iter()
                        .map(|&b| b as f32 / 255.0).collect();
                    let codes = vae.tokenize(&pixels);

                    // Add timestamp
                    let t = i as f32 / fps;
                    let mut tokens = vec![timestamp_token(t)];
                    tokens.extend(image_codes_to_tokens(&codes));

                    // Real-time pacing
                    let target = interval * i as u32;
                    let elapsed = start.elapsed();
                    if target > elapsed {
                        thread::sleep(target - elapsed);
                    }

                    if tx.send(NcInput::Image(tokens)).is_err() { break; }
                }
            }
        }
    })
}

/// Decodes image tokens and writes frames to a directory.
pub struct ImageOutput {
    vae: VqVae,
    dir: String,
    frame_count: usize,
}

impl ImageOutput {
    pub fn new(dir: &str) -> Self {
        std::fs::create_dir_all(dir).ok();
        Self {
            vae: VqVae::new(4096, 64),
            dir: dir.to_string(),
            frame_count: 0,
        }
    }

    /// Decode image codes and write a frame.
    pub fn write(&mut self, codes: &[usize]) {
        let codec_codes: Vec<usize> = codes.iter()
            .filter_map(|&t| {
                if t >= TOKEN_IMG_OFFSET && t < TOKEN_IMG_OFFSET + TOKEN_IMG_CODES {
                    Some(t - TOKEN_IMG_OFFSET)
                } else {
                    None
                }
            }).collect();

        if codec_codes.is_empty() { return; }

        let pixels = self.vae.detokenize(&codec_codes);
        // Write as raw RGB (3072 bytes = 3×32×32)
        let path = format!("{}/frame_{:06}.bin", self.dir, self.frame_count);
        let bytes: Vec<u8> = pixels.iter()
            .map(|&p| (p.clamp(0.0, 1.0) * 255.0) as u8).collect();
        std::fs::write(&path, &bytes).ok();
        self.frame_count += 1;
    }
}

// ── NC streaming loop ────────────────────────────────────

/// Configuration for the NC streaming runtime.
pub struct NcStreamConfig {
    pub temperature: f32,
    pub max_response: usize,
    /// Path to write generated audio (None = discard).
    pub audio_out: Option<String>,
    /// Directory to write generated frames (None = discard).
    pub image_out: Option<String>,
}

impl Default for NcStreamConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            max_response: 256,
            audio_out: None,
            image_out: None,
        }
    }
}

/// Run the NC as a streaming processor.
///
/// Reads from `rx` (inputs from mic, camera, keyboard),
/// processes through the CTM, routes outputs to the appropriate sinks.
pub fn nc_stream_loop(
    nc: &mut NeuralComputer,
    rx: mpsc::Receiver<NcInput>,
    config: NcStreamConfig,
) {
    let mut audio_out = config.audio_out.as_ref().map(|p| AudioOutput::new(p));
    let mut image_out = config.image_out.as_ref().map(|p| ImageOutput::new(p));

    eprintln!("  NC stream loop started");

    loop {
        let input = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(input) => input,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };

        match input {
            NcInput::Quit => break,

            NcInput::Text(text) => {
                let response = nc.chat(&text, config.max_response, config.temperature);
                if !response.is_empty() {
                    println!("{response}");
                }
            }

            NcInput::Audio(tokens) => {
                // Feed audio tokens, let the model respond
                let response = nc.act(&tokens, config.max_response, config.temperature);
                route_output(&response, &mut audio_out, &mut image_out);
            }

            NcInput::Image(tokens) => {
                let response = nc.act(&tokens, config.max_response, config.temperature);
                route_output(&response, &mut audio_out, &mut image_out);
            }

            NcInput::Action(tokens) => {
                let response = nc.act(&tokens, config.max_response, config.temperature);
                route_output(&response, &mut audio_out, &mut image_out);
            }
        }
    }

    // Flush outputs
    if let Some(ref ao) = audio_out {
        if let Err(e) = ao.flush() {
            eprintln!("  audio output flush failed: {e}");
        } else if let Some(ref path) = config.audio_out {
            eprintln!("  audio saved to {path}");
        }
    }
    eprintln!("  NC stream loop stopped");
}

/// Route NC output tokens to the appropriate sinks.
fn route_output(
    tokens: &[usize],
    audio_out: &mut Option<AudioOutput>,
    image_out: &mut Option<ImageOutput>,
) {
    let mut text_buf = Vec::new();
    let mut audio_buf = Vec::new();
    let mut image_buf = Vec::new();
    let mut in_audio = false;
    let mut in_image = false;

    for &t in tokens {
        match t {
            TOKEN_AUD_START => { in_audio = true; }
            TOKEN_AUD_END => {
                in_audio = false;
                if let Some(ao) = audio_out {
                    ao.write(&audio_buf);
                }
                audio_buf.clear();
            }
            TOKEN_IMG_START => { in_image = true; }
            TOKEN_IMG_END => {
                in_image = false;
                if let Some(io) = image_out {
                    io.write(&image_buf);
                }
                image_buf.clear();
            }
            _ if in_audio => { audio_buf.push(t); }
            _ if in_image => { image_buf.push(t); }
            0..=255 => { text_buf.push(t as u8); }
            _ => {} // timestamps, actions, etc — logged but not routed
        }
    }

    if !text_buf.is_empty() {
        print!("{}", String::from_utf8_lossy(&text_buf));
    }
}
