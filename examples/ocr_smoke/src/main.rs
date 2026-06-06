//! ocr_smoke — renderer preview binary.
//!
//! Phase 0 step 1: prove the synthetic line renderer works end-to-end.
//! Renders a sample line and prints an ascii preview. The CTM + CTC
//! training loop lives in a separate step; this binary just smoke-
//! tests the data path.
//!
//! Usage:
//!   ocr_smoke "Hello, World!"
//!   ocr_smoke --font /usr/share/fonts/TTF/DejaVuSansMono.ttf "abc 123"

use ab_glyph::FontRef;
use ocr_smoke::render::{render_line, string_to_classes, ALPHABET_SIZE};

const DEFAULT_FONT: &str = "/usr/share/fonts/TTF/DejaVuSansMono.ttf";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut font_path = String::from(DEFAULT_FONT);
    let mut text: Option<String> = None;
    let mut font_px: f32 = 32.0;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--font" => { font_path = args[i + 1].clone(); i += 2; }
            "--px" => { font_px = args[i + 1].parse().unwrap_or(32.0); i += 2; }
            "--help" | "-h" => { print_usage(); return; }
            _ => { text = Some(args[i].clone()); i += 1; }
        }
    }
    let text = text.unwrap_or_else(|| "the quick brown fox".to_string());

    let font_bytes = std::fs::read(&font_path).unwrap_or_else(|e| {
        eprintln!("failed to read font {font_path}: {e}");
        std::process::exit(1);
    });
    let font = FontRef::try_from_slice(&font_bytes).unwrap_or_else(|e| {
        eprintln!("failed to parse font: {e}");
        std::process::exit(1);
    });

    let line = render_line(&font, &text, font_px);
    let classes = string_to_classes(&text).unwrap_or_else(|e| {
        eprintln!("non-ascii input: {e}");
        std::process::exit(1);
    });

    eprintln!("text     : {text:?}");
    eprintln!("size     : {}x{} ({} pixels)", line.w, line.h, line.pixels.len());
    eprintln!("alphabet : {ALPHABET_SIZE} (1 blank + 95 printable ascii)");
    eprintln!("classes  : {} ({:?}...)", classes.len(),
        &classes[..classes.len().min(8)]);
    eprintln!();
    print!("{}", line.ascii_preview());
}

fn print_usage() {
    eprintln!("ocr_smoke — render an ascii line, preview as ascii art\n");
    eprintln!("USAGE:");
    eprintln!("  ocr_smoke [OPTIONS] TEXT\n");
    eprintln!("OPTIONS:");
    eprintln!("  --font PATH   TTF font (default: {DEFAULT_FONT})");
    eprintln!("  --px N        Rasterization size in pixels (default: 32)");
}
