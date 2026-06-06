//! Synthetic line renderer for OCR training.
//!
//! Takes an ascii string + a TTF font, produces a `[H × W]` grayscale
//! image normalized to `[-1, 1]`. Defaults follow PaddleX line-recognition
//! conventions: height 48 px, normalize `(x/255 - 0.5)/0.5`.
//!
//! Reference: `paddlex/inference/models/text_recognition/processors.py`
//! (`OCRReisizeNormImg`).

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};

/// Line height in pixels. Matches PaddleX `rec_image_shape[1] = 48`.
pub const LINE_H: usize = 48;

/// Inverse pixel: rendered as 1.0 (black ink) before normalization.
/// Background stays 0.0 (white).
const INK: f32 = 255.0;

/// Printable ascii range: 0x20 (space) through 0x7E (tilde) = 95 chars.
pub const ALPHA_LEN: usize = 95;

/// CTC class count = 1 blank + 95 ascii. Blank is index 0.
pub const ALPHABET_SIZE: usize = 1 + ALPHA_LEN;

/// Map an ascii byte to its class index (1..=95). Returns `None` for
/// bytes outside the printable range. Class 0 is reserved for blank.
#[inline]
pub fn ascii_to_class(c: u8) -> Option<usize> {
    if (0x20..=0x7E).contains(&c) { Some((c - 0x20) as usize + 1) } else { None }
}

/// Inverse: class index → ascii byte. Class 0 (blank) → `None`.
#[inline]
pub fn class_to_ascii(k: usize) -> Option<u8> {
    if (1..=ALPHA_LEN).contains(&k) { Some((k - 1) as u8 + 0x20) } else { None }
}

/// Convert a `usize` class sequence (e.g. CTC decode output) into a
/// String. Non-printable / blank classes are silently dropped — they
/// can't appear in well-formed greedy-CTC output anyway.
pub fn classes_to_string(classes: &[usize]) -> String {
    classes.iter()
        .filter_map(|&k| class_to_ascii(k))
        .map(|b| b as char)
        .collect()
}

/// Convert a `&str` into a class-index sequence for use as a CTC target.
/// Returns `Err` on any byte outside printable ascii.
pub fn string_to_classes(s: &str) -> Result<Vec<usize>, String> {
    s.bytes().map(|b| ascii_to_class(b).ok_or_else(|| format!("non-ascii byte {b:#x}"))).collect()
}

// ─── Renderer ─────────────────────────────────────────────────

/// One rendered line.
pub struct RenderedLine {
    /// Flat row-major `[H × W]` grayscale, normalized to `[-1, 1]`.
    /// Background ~= -1.0, ink ~= +1.0 (the convention is inverted vs
    /// PaddleX — they keep white background near 0; we invert so the
    /// signal is positive where the model should pay attention).
    pub pixels: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

impl RenderedLine {
    pub fn h_w(&self) -> (usize, usize) { (self.h, self.w) }

    /// ASCII preview — `#` for ink (> 0.3), `.` for background.
    /// Useful as a sanity-check during development.
    pub fn ascii_preview(&self) -> String {
        let mut out = String::with_capacity((self.w + 1) * self.h);
        for y in 0..self.h {
            for x in 0..self.w {
                let v = self.pixels[y * self.w + x];
                out.push(if v > 0.3 { '#' } else { '.' });
            }
            out.push('\n');
        }
        out
    }
}

/// Render a string to a normalized grayscale buffer.
///
/// Font is rasterized at `font_px` (default ~32 looks good at H=48).
/// Width is set automatically to fit the string with a small margin.
pub fn render_line(font: &FontRef<'_>, text: &str, font_px: f32) -> RenderedLine {
    let h = LINE_H;
    let scaled = font.as_scaled(PxScale::from(font_px));
    let ascent = scaled.ascent();
    let baseline_y = ((h as f32 + ascent) * 0.5).round() as i32;

    // Pass 1: measure total width.
    let mut total_w = 4.0f32; // left margin
    for c in text.chars() {
        let g = scaled.scaled_glyph(c);
        total_w += scaled.h_advance(g.id);
    }
    total_w += 4.0; // right margin
    let w = total_w.ceil() as usize;
    let mut buf = vec![0.0f32; h * w];

    // Pass 2: rasterize glyph by glyph at the baseline.
    let mut x_cursor = 4.0f32;
    for c in text.chars() {
        let mut g = scaled.scaled_glyph(c);
        let h_advance = scaled.h_advance(g.id);
        g.position = ab_glyph::point(x_cursor, baseline_y as f32);
        if let Some(outline) = scaled.outline_glyph(g) {
            let bb = outline.px_bounds();
            outline.draw(|px, py, alpha| {
                let dst_x = bb.min.x as i32 + px as i32;
                let dst_y = bb.min.y as i32 + py as i32;
                if dst_x >= 0 && dst_y >= 0 && (dst_x as usize) < w && (dst_y as usize) < h {
                    let idx = dst_y as usize * w + dst_x as usize;
                    let new = (alpha * INK).max(buf[idx] * INK) / INK;
                    buf[idx] = new;
                }
            });
        }
        x_cursor += h_advance;
    }

    // Normalize: alpha in [0,1] → mapped to [-1, +1]. Ink (alpha=1) → +1.
    for v in buf.iter_mut() { *v = *v * 2.0 - 1.0; }
    RenderedLine { pixels: buf, h, w }
}

/// Render with mild noise + ±jitter_px translation along x. RNG is
/// seeded externally so smoke tests are reproducible.
pub fn render_line_augmented(
    font: &FontRef<'_>,
    text: &str,
    font_px: f32,
    rng: &mut Lcg,
    noise: f32,
    jitter_px: i32,
) -> RenderedLine {
    let mut line = render_line(font, text, font_px);
    let jx = if jitter_px > 0 { rng.range(-jitter_px..=jitter_px) } else { 0 };
    if jx != 0 {
        let mut shifted = vec![-1.0f32; line.pixels.len()];
        for y in 0..line.h {
            for x in 0..line.w {
                let sx = x as i32 + jx;
                if sx >= 0 && (sx as usize) < line.w {
                    shifted[y * line.w + x] = line.pixels[y * line.w + sx as usize];
                }
            }
        }
        line.pixels = shifted;
    }
    if noise > 0.0 {
        for v in line.pixels.iter_mut() {
            *v = (*v + (rng.uniform() * 2.0 - 1.0) * noise).clamp(-1.0, 1.0);
        }
    }
    line
}

// ─── Tiny LCG for reproducible noise / jitter ─────────────────

pub struct Lcg(pub u64);
impl Lcg {
    pub fn new(seed: u64) -> Self { Self(seed.wrapping_mul(6364136223846793005) | 1) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn range(&mut self, r: std::ops::RangeInclusive<i32>) -> i32 {
        let lo = *r.start();
        let hi = *r.end();
        let span = (hi - lo + 1) as u64;
        lo + (self.next_u64() % span) as i32
    }
}

// ─── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_FONT: &str = "/usr/share/fonts/TTF/DejaVuSansMono.ttf";

    fn load_font() -> Option<Vec<u8>> { std::fs::read(TEST_FONT).ok() }

    #[test]
    fn alphabet_roundtrip() {
        for b in 0x20u8..=0x7Eu8 {
            let k = ascii_to_class(b).unwrap();
            assert_eq!(class_to_ascii(k), Some(b));
        }
        assert_eq!(ascii_to_class(0x1F), None);
        assert_eq!(ascii_to_class(0x7F), None);
        assert_eq!(class_to_ascii(0), None);
        assert_eq!(class_to_ascii(ALPHABET_SIZE), None);
    }

    #[test]
    fn alphabet_size_matches_alpha_plus_blank() {
        assert_eq!(ALPHABET_SIZE, 96);
        assert_eq!(ALPHA_LEN, 95);
    }

    #[test]
    fn string_to_classes_roundtrip() {
        let s = "Hello, World!";
        let cls = string_to_classes(s).unwrap();
        assert_eq!(classes_to_string(&cls), s);
        assert!(string_to_classes("héllo").is_err());
    }

    #[test]
    fn render_emits_some_ink() {
        let Some(font_bytes) = load_font() else {
            eprintln!("skipping: {TEST_FONT} not present");
            return;
        };
        let font = FontRef::try_from_slice(&font_bytes).unwrap();
        let line = render_line(&font, "Hello", 32.0);
        assert_eq!(line.h, LINE_H);
        assert!(line.w > 20, "width {} too small", line.w);
        let n_ink: usize = line.pixels.iter().filter(|&&v| v > 0.3).count();
        let total = line.pixels.len();
        assert!(n_ink > 50, "only {n_ink} ink pixels of {total}");
        assert!(n_ink < total / 2, "too much ink ({n_ink}/{total}) — render inverted?");
    }

    #[test]
    fn render_width_scales_with_text_length() {
        let Some(font_bytes) = load_font() else { return; };
        let font = FontRef::try_from_slice(&font_bytes).unwrap();
        let short = render_line(&font, "ab", 32.0);
        let long = render_line(&font, "abcdefghij", 32.0);
        assert!(long.w > short.w * 2, "short {} long {}", short.w, long.w);
    }

    #[test]
    fn augmented_render_stays_in_range() {
        let Some(font_bytes) = load_font() else { return; };
        let font = FontRef::try_from_slice(&font_bytes).unwrap();
        let mut rng = Lcg::new(42);
        let line = render_line_augmented(&font, "Test", 32.0, &mut rng, 0.1, 2);
        for &v in &line.pixels {
            assert!(v >= -1.0 && v <= 1.0, "value {v} out of range");
        }
    }
}
