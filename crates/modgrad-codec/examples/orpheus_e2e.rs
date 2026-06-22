//! orpheus_e2e — text → Orpheus LM (modgrad) → SNAC decode (modgrad) → WAV.
//!
//! The full bidirectional-codec-ready pipeline running 100% rust on
//! local ROCm. No Python, no transformers.js, no third-party runtime.
//!
//! Layout:
//!   1. Load Orpheus 3B GGUF, build LlmConfig::llama3(), upload to VRAM
//!   2. Construct Orpheus prompt: [128259] + body + [128009, 128260, 128261, 128257]
//!   3. forward_token loop, sampling argmax, stop on EOT or max_tokens
//!   4. Filter audio tokens (id ≥ 128256), parse into 7-token frames
//!   5. Reshape frames into 3 SNAC codebooks per canopyai/decoder.py recipe
//!   6. Load SnacDecoder24k, decode codes → samples
//!   7. Write 24 kHz mono WAV to disk
//!
//! Run:
//!   cargo run --release -p modgrad-codec --example orpheus_e2e -- \
//!     [--gen 252]   # number of audio tokens; multiples of 7 are full frames
//!     [--out  out.wav]
//!     [--gguf /path/to/orpheus-3b.gguf]
//!     [--snac /path/to/snac24k.safetensors]

use std::io::{Cursor, Write};
use std::time::Instant;

use modgrad_codec::snac::SnacDecoder24k;
use modgrad_device::kfd::HsaDevice;
use modgrad_device::kfd::dispatch_queue::GpuQueue;
use modgrad_device::kfd::gguf::GgufFile;
use modgrad_device::kfd::llm::{
    LlmCache, LlmConfig, LlmScratch, LlmWeights, forward_token,
};

const AUDIO_TOKEN_BASE: u32 = 128256;
const TOK_EOT_A: u32 = 128009;
const TOK_EOT_B: u32 = 128001;

fn main() {
    let mut args: Vec<String> = std::env::args().collect();

    fn opt(args: &mut Vec<String>, name: &str) -> Option<String> {
        let i = args.iter().position(|a| a == name)?;
        let v = args.get(i + 1).cloned();
        if v.is_some() { args.drain(i..=i + 1); } else { args.remove(i); }
        v
    }
    let n_gen: usize = opt(&mut args, "--gen").and_then(|s| s.parse().ok()).unwrap_or(252);
    let out_path = opt(&mut args, "--out").unwrap_or_else(|| "/tmp/orpheus_e2e.wav".into());
    let gguf_path = opt(&mut args, "--gguf").unwrap_or_else(|| {
        "/steam/rotko/modgrad/models/orpheus-3b-q4km/orpheus-3b-0.1-ft.Q4_K_M.gguf".into()
    });
    let snac_path = opt(&mut args, "--snac").unwrap_or_else(|| {
        "/steam/rotko/modgrad/models/snac24k.safetensors".into()
    });

    // ── 1. Load Orpheus ──────────────────────────────────────────
    eprintln!("[1/7] mmap {gguf_path}");
    let gguf_bytes = std::fs::read(&gguf_path).expect("read gguf");
    let gguf = GgufFile::parse(&mut Cursor::new(&gguf_bytes)).expect("parse gguf");
    let mut cfg = LlmConfig::llama3();
    cfg.populate_from_gguf(&gguf).expect("populate cfg");
    eprintln!("  {}L d={} ff={} vocab={}", cfg.n_layers, cfg.d_model, cfg.d_ff, cfg.vocab_size);

    eprintln!("[2/7] upload weights to VRAM");
    let dev = HsaDevice::open().expect("open /dev/kfd");
    let mut queue = GpuQueue::new();
    let weights = LlmWeights::load_from_gguf(&gguf, &gguf_bytes, &dev, &mut queue)
        .expect("load weights");

    // ── 2. Build Orpheus prompt ──────────────────────────────────
    let body: Vec<u32> = vec![
        128000,        // <|begin_of_text|>
        83, 5169, 25,  // "tara:" approximation
        24748,         // " hi"
    ];
    let mut prompt: Vec<u32> = vec![128259];  // <|orpheus_start|>
    prompt.extend_from_slice(&body);
    prompt.extend_from_slice(&[128009, 128260, 128261, 128257]);  // <|end_of_text|><|orpheus_end_*|>

    let max_seq = prompt.len() + n_gen + 8;
    eprintln!("[3/7] alloc KV cache (max_seq={max_seq})");
    let mut cache = LlmCache::allocate(&cfg, &dev, &mut queue, max_seq).expect("alloc cache");
    let mut scratch = LlmScratch::allocate(&cfg, &dev, &mut queue).expect("alloc scratch");

    eprintln!("[4/7] forward prompt ({} tokens)", prompt.len());
    let t_prompt = Instant::now();
    let mut dev = dev;
    let mut logits = Vec::new();
    for &tok in &prompt {
        logits = forward_token(&cfg, &weights, &mut cache, &mut scratch,
                               &mut dev, &mut queue, tok);
    }
    eprintln!("  prompt processed in {:.1}s", t_prompt.elapsed().as_secs_f32());

    // ── 3. Generate audio tokens ─────────────────────────────────
    eprintln!("[5/7] generate {n_gen} tokens");
    let t_gen = Instant::now();
    let mut audio_ids: Vec<u32> = Vec::with_capacity(n_gen);
    for step in 0..n_gen {
        let next = argmax(&logits) as u32;
        if next == TOK_EOT_A || next == TOK_EOT_B {
            eprintln!("  eot at step {step}");
            break;
        }
        if next >= AUDIO_TOKEN_BASE {
            audio_ids.push(next);
        } else {
            eprintln!("  non-audio token at step {step}: {next} — stopping");
            break;
        }
        if step + 1 < n_gen {
            logits = forward_token(&cfg, &weights, &mut cache, &mut scratch,
                                   &mut dev, &mut queue, next);
        }
    }
    eprintln!("  generated {} audio tokens in {:.1}s",
        audio_ids.len(), t_gen.elapsed().as_secs_f32());

    // Drop any trailing partial frame: we need a multiple of 7.
    let frame_count = audio_ids.len() / 7;
    audio_ids.truncate(frame_count * 7);
    eprintln!("  {frame_count} full 7-token frames");

    if frame_count == 0 {
        eprintln!("no full frames — nothing to decode");
        std::process::exit(2);
    }

    // ── 4. Parse audio tokens → 3 SNAC codebooks (canopy decoder.py recipe) ──
    // Per-position offset within frame: code = (id - 128256) - (pos%7)*4096.
    // Distribution: codes_0 = frame[0] (1 per frame, finest stride 4 → covers 4 ticks)
    //               codes_1 = frame[1] + frame[4] (2 per frame, stride 2)
    //               codes_2 = frame[2] + frame[3] + frame[5] + frame[6] (4 per frame, stride 1)
    let mut codes_0 = Vec::with_capacity(frame_count);
    let mut codes_1 = Vec::with_capacity(frame_count * 2);
    let mut codes_2 = Vec::with_capacity(frame_count * 4);
    for f in 0..frame_count {
        let base = f * 7;
        let raw = |i: usize| -> i32 {
            let id = audio_ids[base + i] as i32;
            id - AUDIO_TOKEN_BASE as i32 - (i as i32 % 7) * 4096
        };
        let push_or_zero = |dst: &mut Vec<u32>, v: i32| {
            if (0..4096).contains(&v) {
                dst.push(v as u32);
            } else {
                // Out-of-range codes shouldn't happen for a correctly-trained
                // model. Clamp to keep SNAC's range check happy.
                dst.push(v.clamp(0, 4095) as u32);
            }
        };
        push_or_zero(&mut codes_0, raw(0));
        push_or_zero(&mut codes_1, raw(1));
        push_or_zero(&mut codes_2, raw(2));
        push_or_zero(&mut codes_2, raw(3));
        push_or_zero(&mut codes_1, raw(4));
        push_or_zero(&mut codes_2, raw(5));
        push_or_zero(&mut codes_2, raw(6));
    }
    eprintln!("  codebooks: cb0={} cb1={} cb2={}", codes_0.len(), codes_1.len(), codes_2.len());

    // ── 5. SNAC decode ───────────────────────────────────────────
    eprintln!("[6/7] SNAC decode");
    let t_snac = Instant::now();
    let snac = SnacDecoder24k::load_from_safetensors(&snac_path).expect("load snac");
    let samples = snac.decode_seeded(&[codes_0, codes_1, codes_2], 1);
    eprintln!("  {} samples ({:.2}s @ 24kHz) in {:.2}s",
        samples.len(), samples.len() as f32 / 24000.0,
        t_snac.elapsed().as_secs_f32());

    // ── 6. Write WAV ─────────────────────────────────────────────
    eprintln!("[7/7] write {out_path}");
    write_wav_24khz_mono(&out_path, &samples).expect("write wav");
    eprintln!("\ndone — listen: mpv {out_path}");
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0; let mut bv = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > bv { best = i; bv = v; }
    }
    best
}

/// Minimal PCM 16-bit WAV writer, mono, 24 kHz.
fn write_wav_24khz_mono(path: &str, samples: &[f32]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let n = samples.len();
    let data_bytes = (n * 2) as u32;
    let header_size: u32 = 36;
    let total_minus_8 = header_size + data_bytes;

    f.write_all(b"RIFF")?;
    f.write_all(&total_minus_8.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    // fmt chunk
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;      // PCM
    f.write_all(&1u16.to_le_bytes())?;      // mono
    f.write_all(&24000u32.to_le_bytes())?;  // sample rate
    f.write_all(&48000u32.to_le_bytes())?;  // byte rate (sr * 2)
    f.write_all(&2u16.to_le_bytes())?;      // block align
    f.write_all(&16u16.to_le_bytes())?;     // bits per sample
    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_bytes.to_le_bytes())?;
    for &s in samples {
        let s16 = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&s16.to_le_bytes())?;
    }
    Ok(())
}
