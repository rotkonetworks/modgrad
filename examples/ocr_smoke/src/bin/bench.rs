//! Head-to-head benchmark: CTM (loaded weights) vs PP-OCRv5 on the
//! *same* synthetic test images.
//!
//! Procedure:
//!   1. Render N test images via the same renderer the training used.
//!   2. Write each as an 8-bit grayscale PNG to `--out-dir`, plus a
//!      `manifest.json` mapping filename → ground-truth string.
//!   3. Run the loaded CTM weights over the same images, decode greedily,
//!      and compute byte-level CER → printed to stdout.
//!   4. The companion `scripts/bench-ppocr.mjs` reads the same dir,
//!      runs ppu-paddle-ocr on each PNG, computes CER, and reports it.
//!      Run that separately; this binary only handles the CTM side and
//!      the dataset export.
//!
//! Usage:
//!   ocr_smoke_bench --weights ctm.bin --out-dir /tmp/ocr_test --n 100

use ab_glyph::FontRef;
use ocr_smoke::eval::cer;
use ocr_smoke::render::{
    classes_to_string, render_line_augmented, Lcg, LINE_H,
};
use ocr_smoke::train::{greedy_decode_predictions, line_to_tokens};

use modgrad_ctm::weights::{CtmState, CtmWeights};
use modgrad_ctm::forward::{ctm_forward, CtmInput};

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

const DEFAULT_FONT: &str = "/usr/share/fonts/TTF/DejaVuSansMono.ttf";
const VOCAB_ALNUM: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const VOCAB_DIGITS: &[u8] = b"0123456789";
const VOCAB_LOWER: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

struct Cfg {
    weights: String,
    font_path: String,
    fonts: Vec<String>,
    out_dir: String,
    n: usize,
    seed: u64,
    min_chars: usize,
    max_chars: usize,
    font_px: f32,
    vocab: String,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            weights: String::new(),
            font_path: DEFAULT_FONT.into(),
            fonts: Vec::new(),
            out_dir: "/tmp/ocr_bench".into(),
            n: 100,
            seed: 42,
            min_chars: 1,
            max_chars: 1,
            font_px: 24.0,
            vocab: "alnum".into(),
        }
    }
}

fn parse_cfg() -> Cfg {
    let mut cfg = Cfg::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--weights" => cfg.weights = args.next().expect("--weights needs a path"),
            "--out-dir" => cfg.out_dir = args.next().expect("--out-dir needs a path"),
            "--n" => cfg.n = args.next().unwrap().parse().unwrap(),
            "--seed" => cfg.seed = args.next().unwrap().parse().unwrap(),
            "--font" => cfg.font_path = args.next().unwrap(),
            "--fonts" => {
                let raw = args.next().expect("--fonts needs a comma-separated list");
                cfg.fonts = raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            }
            "--px" => cfg.font_px = args.next().unwrap().parse().unwrap(),
            "--min-chars" => cfg.min_chars = args.next().unwrap().parse().unwrap(),
            "--max-chars" => cfg.max_chars = args.next().unwrap().parse().unwrap(),
            "--vocab" => cfg.vocab = args.next().unwrap(),
            "--help" | "-h" => {
                eprintln!("ocr_smoke_bench --weights PATH --out-dir DIR --n N [--seed S]");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(2); }
        }
    }
    cfg
}

fn pick_vocab(name: &str) -> &'static [u8] {
    match name {
        "digits" => VOCAB_DIGITS,
        "lower"  => VOCAB_LOWER,
        _ => VOCAB_ALNUM,
    }
}

fn sample_string(rng: &mut Lcg, vocab: &[u8], min_len: usize, max_len: usize) -> String {
    let lo = min_len as i32;
    let hi = max_len as i32;
    let len = rng.range(lo..=hi) as usize;
    let max_idx = (vocab.len() as i32) - 1;
    let mut s = String::with_capacity(len);
    for _ in 0..len {
        let idx = rng.range(0..=max_idx) as usize;
        s.push(vocab[idx] as char);
    }
    s
}

fn save_png_gray(path: &Path, w: usize, h: usize, pixels: &[f32]) -> std::io::Result<()> {
    // Map [-1, 1] (renderer convention: ink ≈ +1, bg ≈ -1) to 8-bit.
    // OCR engines expect dark ink on light background — so we invert
    // by sending bg→255 and ink→0. This matches scanned-document
    // convention.
    let mut bytes: Vec<u8> = Vec::with_capacity(w * h);
    for &v in pixels {
        let bg_score = (1.0 - v) * 0.5;  // -1→1, +1→0
        let scaled = (bg_score * 255.0).clamp(0.0, 255.0).round() as u8;
        bytes.push(scaled);
    }
    let file = fs::File::create(path)?;
    let bw = BufWriter::new(file);
    let mut encoder = png::Encoder::new(bw, w as u32, h as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| std::io::Error::other(e))?;
    writer.write_image_data(&bytes).map_err(|e| std::io::Error::other(e))?;
    Ok(())
}

fn main() {
    let cfg = parse_cfg();

    eprintln!("ocr_smoke_bench — head-to-head test set + CTM eval");
    eprintln!("  weights: {}", if cfg.weights.is_empty() { "(none — dataset export only)" } else { &cfg.weights });
    eprintln!("  out_dir: {}", cfg.out_dir);
    eprintln!("  n: {}  seed: {}  chars: {}..={}  vocab: {}",
        cfg.n, cfg.seed, cfg.min_chars, cfg.max_chars, cfg.vocab);

    // Load the font pool. --fonts wins if provided; otherwise --font.
    let font_paths: Vec<String> = if !cfg.fonts.is_empty() {
        cfg.fonts.clone()
    } else {
        vec![cfg.font_path.clone()]
    };
    let font_blobs: Vec<Vec<u8>> = font_paths.iter().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("failed to read font {}: {}", p, e);
            std::process::exit(1);
        })
    }).collect();
    let fonts: Vec<FontRef> = font_blobs.iter().map(|b| FontRef::try_from_slice(b).expect("parse font")).collect();
    eprintln!("  fonts: {}", fonts.len());
    for p in &font_paths { eprintln!("    {}", p); }

    // Load weights if a path was given. With "" / "none" / missing
    // file, we still dump the dataset — useful for measuring just the
    // PP-OCRv5 baseline before a checkpoint is ready.
    let w_opt: Option<CtmWeights> = if cfg.weights.is_empty() || cfg.weights == "none" {
        None
    } else {
        match CtmWeights::load(&cfg.weights) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("weights load failed ({e}); proceeding with dataset-only export");
                None
            }
        }
    };

    fs::create_dir_all(&cfg.out_dir).unwrap();
    let mut manifest_path = std::path::PathBuf::from(&cfg.out_dir);
    manifest_path.push("manifest.json");
    let mut manifest = fs::File::create(&manifest_path).unwrap();
    writeln!(manifest, "[").unwrap();

    let mut rng = Lcg::new(cfg.seed);
    let vocab = pick_vocab(&cfg.vocab);
    let max_font_idx = (fonts.len() as i32) - 1;
    let mut pairs = Vec::with_capacity(cfg.n);

    for i in 0..cfg.n {
        let text = sample_string(&mut rng, vocab, cfg.min_chars, cfg.max_chars);
        let font_idx = rng.range(0..=max_font_idx) as usize;
        let line = render_line_augmented(&fonts[font_idx], &text, cfg.font_px, &mut rng, 0.04, 1);
        let img_name = format!("{:04}.png", i);
        let mut img_path = std::path::PathBuf::from(&cfg.out_dir);
        img_path.push(&img_name);
        save_png_gray(&img_path, line.w, line.h, &line.pixels).unwrap();

        // Manifest entry. Plain JSON, one object per line.
        let comma = if i + 1 < cfg.n { "," } else { "" };
        writeln!(
            manifest,
            r#"  {{"file": {:?}, "text": {:?}, "w": {}, "h": {}}}{}"#,
            img_name, text, line.w, line.h, comma
        ).unwrap();

        // CTM inference — only if weights were loaded.
        if let Some(w) = w_opt.as_ref() {
            let (tokens, n_tokens, token_dim) = line_to_tokens(&line);
            let mut state = CtmState::new(w);
            let out = ctm_forward(
                w, &mut state,
                CtmInput::Raw { obs: &tokens, n_tokens, raw_dim: token_dim },
            );
            let decoded = greedy_decode_predictions(&out.predictions);
            let predicted = classes_to_string(&decoded);
            pairs.push((text, predicted));
        } else {
            pairs.push((text, String::new()));
        }
    }
    writeln!(manifest, "]").unwrap();

    eprintln!();
    eprintln!("dataset: {} images, {} (manifest)", cfg.n, manifest_path.display());

    if w_opt.is_some() {
        let our_cer = cer(&pairs);
        let exact_match = pairs.iter().filter(|(t, p)| t == p).count();
        eprintln!();
        eprintln!("CTM result on {} images:", cfg.n);
        eprintln!("  byte-level CER: {:.4}", our_cer);
        eprintln!("  exact-match accuracy: {}/{} = {:.2}%",
            exact_match, cfg.n, 100.0 * exact_match as f32 / cfg.n as f32);
        eprintln!();
        eprintln!("Sample CTM predictions:");
        for (target, predicted) in pairs.iter().take(10) {
            let mark = if target == predicted { "✓" } else { " " };
            eprintln!("  {} {:?} -> {:?}", mark, target, predicted);
        }
    } else {
        eprintln!("CTM eval skipped (no weights loaded).");
    }
    eprintln!();
    eprintln!("Next: run PP-OCRv5 on the same images via:");
    eprintln!("  node /home/alice/rotko/ocr.rotko.net/scripts/bench-ppocr.mjs {}", cfg.out_dir);
}
