//! snac_smoke — load the SNAC 24 kHz decoder from safetensors and decode
//! a tiny synthetic input. Smoke test to validate the loader doesn't
//! panic and produces a plausibly-shaped waveform. Bit-exact comparison
//! against the PyTorch ONNX oracle comes next, after we feed it real
//! Orpheus-generated codes.
//!
//! Run:
//!   cargo run --release -p modgrad-codec --example snac_smoke \
//!     [-- /path/to/snac24k.safetensors]

use std::time::Instant;

use modgrad_codec::snac::SnacDecoder24k;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/steam/rotko/modgrad/models/snac24k.safetensors".into()
    });

    eprintln!("[1/3] loading {path}");
    let t0 = Instant::now();
    let dec = SnacDecoder24k::load_from_safetensors(&path)
        .expect("load SNAC weights");
    eprintln!("  loaded in {:.2}s", t0.elapsed().as_secs_f32());

    // Synthetic codes — same shape as one frame's worth of Orpheus output
    // ought to produce: 4 finest-stride frames (so codebook 2 has 4 codes,
    // codebook 1 has 2, codebook 0 has 1).
    let t_finest = 4usize;
    let codes_0: Vec<u32> = (0..t_finest / 4).map(|i| (100 + i) as u32 % 4096).collect();
    let codes_1: Vec<u32> = (0..t_finest / 2).map(|i| (200 + i) as u32 % 4096).collect();
    let codes_2: Vec<u32> = (0..t_finest    ).map(|i| (300 + i) as u32 % 4096).collect();

    eprintln!("[2/3] decoding (codebooks: {} + {} + {} codes → t_finest={})",
        codes_0.len(), codes_1.len(), codes_2.len(), t_finest);
    let t1 = Instant::now();
    let wav = dec.decode_seeded(&[codes_0, codes_1, codes_2], 1);
    eprintln!("  decoded {} samples ({:.3}s @ 24kHz) in {:.2}s",
        wav.len(), wav.len() as f32 / 24000.0, t1.elapsed().as_secs_f32());

    let max_abs = wav.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let nan_count = wav.iter().filter(|x| x.is_nan()).count();
    let inf_count = wav.iter().filter(|x| x.is_infinite()).count();
    eprintln!("[3/3] sanity check:");
    eprintln!("  samples       = {}", wav.len());
    eprintln!("  max |sample|  = {:.4}", max_abs);
    eprintln!("  NaN count     = {}", nan_count);
    eprintln!("  inf count     = {}", inf_count);

    assert!(nan_count == 0 && inf_count == 0, "non-finite outputs");
    assert!(max_abs <= 1.0 + 1e-3, "tanh should keep |x| ≤ 1");
}
