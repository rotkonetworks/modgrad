//! snac_ab_modgrad — decode a fixed set of codes through modgrad's SNAC
//! port and dump the waveform to a binary file so a Python script can
//! load PyTorch SNAC, run the same codes, and bit-compare both outputs.
//!
//! Run: cargo run --release -p modgrad-codec --example snac_ab_modgrad
//!
//! Outputs:
//!   /tmp/snac_ab_modgrad.wav       — audible
//!   /tmp/snac_ab_modgrad.f32       — raw f32 samples for byte-comparison

use std::io::Write;
use modgrad_codec::snac::SnacDecoder24k;

fn main() {
    let snac = SnacDecoder24k::load_from_safetensors(
        "/steam/rotko/modgrad/models/snac24k.safetensors"
    ).expect("load SNAC");

    // Fixed, reproducible test codes. 4 frames of audio:
    //   codes_0 has 4 entries (1 per frame)
    //   codes_1 has 8 entries (2 per frame)
    //   codes_2 has 16 entries (4 per frame)
    // Hand-picked values, each within [0, 4095], that should produce a
    // deterministic decoder output we can compare bit-by-bit to PyTorch.
    let codes_0: Vec<u32> = vec![100, 250, 500, 1000];
    let codes_1: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let codes_2: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    let samples = snac.decode_seeded(&[codes_0.clone(), codes_1.clone(), codes_2.clone()], 1);
    println!("modgrad output: {} samples, max|x|={:.4}",
        samples.len(),
        samples.iter().map(|x| x.abs()).fold(0.0_f32, f32::max));

    // Dump raw f32 for byte comparison
    let mut raw = std::fs::File::create("/tmp/snac_ab_modgrad.f32").unwrap();
    for &s in &samples {
        raw.write_all(&s.to_le_bytes()).unwrap();
    }

    // Also write a WAV for listening
    let mut wav = std::fs::File::create("/tmp/snac_ab_modgrad.wav").unwrap();
    let data_bytes = (samples.len() * 2) as u32;
    wav.write_all(b"RIFF").unwrap();
    wav.write_all(&(36 + data_bytes).to_le_bytes()).unwrap();
    wav.write_all(b"WAVE").unwrap();
    wav.write_all(b"fmt ").unwrap();
    wav.write_all(&16u32.to_le_bytes()).unwrap();
    wav.write_all(&1u16.to_le_bytes()).unwrap();
    wav.write_all(&1u16.to_le_bytes()).unwrap();
    wav.write_all(&24000u32.to_le_bytes()).unwrap();
    wav.write_all(&48000u32.to_le_bytes()).unwrap();
    wav.write_all(&2u16.to_le_bytes()).unwrap();
    wav.write_all(&16u16.to_le_bytes()).unwrap();
    wav.write_all(b"data").unwrap();
    wav.write_all(&data_bytes.to_le_bytes()).unwrap();
    for &s in &samples {
        let s16 = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        wav.write_all(&s16.to_le_bytes()).unwrap();
    }

    println!("wrote /tmp/snac_ab_modgrad.wav + .f32");
    println!("\nNow run on bkk07 to get PyTorch reference:");
    println!("  ssh bkk07 'cd /root/norpheus && venv/bin/python snac_ab_pytorch.py'");
    println!("Then scp the result back and diff:");
    println!("  scp bkk07:/root/norpheus/snac_ab_pytorch.f32 /tmp/");
    println!("  cmp /tmp/snac_ab_modgrad.f32 /tmp/snac_ab_pytorch.f32  # exact match");
    println!("  # or audible:  mpv /tmp/snac_ab_modgrad.wav  vs  /tmp/snac_ab_pytorch.wav");
}
