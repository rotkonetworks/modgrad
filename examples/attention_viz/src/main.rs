//! Visualise V4-CTM per-tick attention over real images.
//!
//! Sakana's CTM paper claim: attention saccades from feature to feature
//! across ticks, mimicking human eye-movement traces. To check our
//! retina against that:
//!
//!   1. Load a natural image at 32×32 from imagenet10 .feat data.
//!   2. Run `VisualCortex::cifar_ln()` (Gabor V1 + per-token LN) →
//!      V4 spatial tokens.
//!   3. Run `V4Ctm::forward_with_attn_trace` → per-tick × per-head
//!      softmax weights over the 16 V4 tokens (4×4 grid).
//!   4. Save: original image PPM + per-(tick, head) heatmap PPMs.
//!
//! Heatmaps are upsampled nearest-neighbour from the V4 4×4 grid back
//! to 32×32 and overlaid on the input as red intensity. PPMs are
//! human-viewable in any image tool; no third-party deps.
//!
//! Output: `/tmp/attention_viz/` directory with files
//!   `input.ppm`
//!   `attn_t{T}_h{H}.ppm`     — overlay for tick T head H
//!   `attn_t{T}_mean.ppm`     — mean across heads at tick T
//!   `summary.txt`            — top-attended tokens per tick
//!
//! CLI: `attention_viz [image_index]` (default 0).
//!
//! Compares against random retina via `MODGRAD_VARIANT=random` for
//! the "before LN-fix" baseline showing degenerate attention.

use modgrad_codec::cifar::{load_feat, CifarImage};
use modgrad_codec::retina::VisualCortex;
use std::fs;
use std::io::Write;

const IMAGE_PATH: &str = "/tmp/retina_imagenet10_eval.feat";
const OUT_DIR: &str = "/tmp/attention_viz";
const IMG_H: usize = 32;
const IMG_W: usize = 32;

fn build_cortex() -> (VisualCortex, &'static str) {
    let variant = std::env::var("MODGRAD_VARIANT").unwrap_or_else(|_| "cifar_ln".into());
    match variant.as_str() {
        "cifar"        => (VisualCortex::cifar(),               "cifar (no LN, expected degenerate)"),
        "cifar_ln"     => (VisualCortex::cifar_ln(),            "cifar_ln (Gabor V1 + LN)"),
        "dog_only_ln"  => (VisualCortex::cifar_retina_only_ln(IMG_H, IMG_W),
                            "dog_only_ln (DoG retina + random V1/V2/V4 + LN)"),
        "random"       => (VisualCortex::random(IMG_H, IMG_W),  "random (no priors, no LN)"),
        other => panic!("unknown variant '{other}', want cifar | cifar_ln | dog_only_ln | random"),
    }
}

fn write_ppm(path: &str, rgb: &[u8], h: usize, w: usize) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    write!(f, "P6\n{} {}\n255\n", w, h)?;
    f.write_all(rgb)?;
    Ok(())
}

/// CHW [0,1] → packed RGB u8 row-major (`h * w * 3`).
fn chw_to_rgb_u8(pixels: &[f32], h: usize, w: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = pixels[c * h * w + y * w + x].clamp(0.0, 1.0);
                out[(y * w + x) * 3 + c] = (v * 255.0) as u8;
            }
        }
    }
    out
}

/// Overlay an h_tok×w_tok heatmap (in [0,1]) on a base image,
/// upsampled to (img_h, img_w) by nearest-neighbour. Heatmap is
/// added as additive red-channel intensity (alpha 0.6 of attention,
/// 0.4 of base).
fn overlay_attention(
    base: &[u8],
    attn: &[f32],
    h_tok: usize, w_tok: usize,
    img_h: usize, img_w: usize,
) -> Vec<u8> {
    let max = attn.iter().cloned().fold(f32::NEG_INFINITY, f32::max).max(1e-8);
    let mut out = base.to_vec();
    for y in 0..img_h {
        let ty = y * h_tok / img_h;
        for x in 0..img_w {
            let tx = x * w_tok / img_w;
            let a = (attn[ty * w_tok + tx] / max).clamp(0.0, 1.0);
            let i = (y * img_w + x) * 3;
            // Blend: keep some of the original, push red toward 255.
            out[i]     = (out[i]     as f32 * (1.0 - 0.6 * a) + 255.0 * (0.6 * a)) as u8;
            out[i + 1] = (out[i + 1] as f32 * (1.0 - 0.4 * a)) as u8;
            out[i + 2] = (out[i + 2] as f32 * (1.0 - 0.4 * a)) as u8;
        }
    }
    out
}

fn main() -> std::io::Result<()> {
    let img_idx: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok()).unwrap_or(0);

    fs::create_dir_all(OUT_DIR)?;
    let (mut cortex, mut label_owned) = {
        let (c, l) = build_cortex();
        (c, l.to_string())
    };
    // MODGRAD_V4CTM_PATH=/path/to/saved.bin: replace V4Ctm.weights with
    // the supervised-trained classifier weights from `v4ctm_classifier`.
    // Lets us compare random-init vs trained attention traces on the
    // same image, isolating the effect of training on attention quality.
    if let Ok(path) = std::env::var("MODGRAD_V4CTM_PATH") {
        eprintln!("loading trained V4Ctm weights from {path}");
        let trained = modgrad_ctm::weights::CtmWeights::load(&path)
            .expect("load trained V4Ctm");
        cortex.v4_ctm.weights = trained;
        label_owned = format!("{label_owned} + trained V4Ctm ({path})");
    }
    let label = label_owned.as_str();
    eprintln!("variant: {label}");
    eprintln!("loading {IMAGE_PATH} → image[{img_idx}]");
    let images = load_feat(IMAGE_PATH).expect("load .feat");
    let img: &CifarImage = &images[img_idx];
    eprintln!("  pixels.len={}, label={}", img.pixels.len(), img.label);

    // Save the input.
    let base_rgb = chw_to_rgb_u8(&img.pixels, IMG_H, IMG_W);
    write_ppm(&format!("{OUT_DIR}/input.ppm"), &base_rgb, IMG_H, IMG_W)?;

    // Forward through the visual hierarchy → V4 tokens.
    let scales = cortex.spatial_tokens_multiscale(&img.pixels);
    let (v4_tok, n_tokens, token_dim) = (&scales[2].0, scales[2].1, scales[2].2);
    let h_tok = (n_tokens as f32).sqrt() as usize;
    let w_tok = h_tok;
    assert_eq!(h_tok * w_tok, n_tokens,
        "expected square token grid; got n_tokens={n_tokens}");
    eprintln!("V4 grid: {h_tok}×{w_tok}={n_tokens} tokens × {token_dim} dim");

    // V4-CTM with attention trace.
    let (sync_out, trace) = cortex.v4_ctm.forward_with_attn_trace(v4_tok, n_tokens, token_dim);
    let n_ticks = trace.len();
    let n_heads = if n_ticks > 0 { trace[0].len() } else { 0 };
    eprintln!("trace: {n_ticks} ticks × {n_heads} heads × {n_tokens} tokens");
    eprintln!("sync_out norm: {:.3}", sync_out.iter().map(|x| x * x).sum::<f32>().sqrt());

    // Per-tick × per-head overlay PPMs + a tick-mean overlay.
    let mut summary = String::new();
    summary.push_str(&format!("variant: {label}\n"));
    summary.push_str(&format!("image idx: {img_idx}, label: {}\n", img.label));
    summary.push_str(&format!("V4 grid: {h_tok}×{w_tok}={n_tokens} tokens, {n_heads} heads × {n_ticks} ticks\n\n"));

    for t in 0..n_ticks {
        let mut tick_mean = vec![0.0f32; n_tokens];
        for h in 0..n_heads {
            let attn = &trace[t][h];
            let overlay = overlay_attention(&base_rgb, attn, h_tok, w_tok, IMG_H, IMG_W);
            write_ppm(&format!("{OUT_DIR}/attn_t{t}_h{h}.ppm"), &overlay, IMG_H, IMG_W)?;
            for i in 0..n_tokens { tick_mean[i] += attn[i]; }
        }
        let inv_h = 1.0 / n_heads as f32;
        for v in &mut tick_mean { *v *= inv_h; }
        let overlay = overlay_attention(&base_rgb, &tick_mean, h_tok, w_tok, IMG_H, IMG_W);
        write_ppm(&format!("{OUT_DIR}/attn_t{t}_mean.ppm"), &overlay, IMG_H, IMG_W)?;

        // Top-3 attended positions per head, plus the tick-mean top-3.
        let top3 = |w: &[f32]| -> Vec<(usize, usize, f32)> {
            let mut idx: Vec<usize> = (0..w.len()).collect();
            idx.sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap());
            idx.iter().take(3).map(|&i| (i / w_tok, i % w_tok, w[i])).collect()
        };
        summary.push_str(&format!("tick {t}\n"));
        summary.push_str(&format!("  mean top-3:"));
        for (y, x, v) in top3(&tick_mean) {
            summary.push_str(&format!(" ({y},{x})={v:.3}"));
        }
        summary.push('\n');
        for h in 0..n_heads {
            summary.push_str(&format!("  head {h} top-3:"));
            for (y, x, v) in top3(&trace[t][h]) {
                summary.push_str(&format!(" ({y},{x})={v:.3}"));
            }
            summary.push('\n');
        }
        summary.push('\n');
    }

    // Saccade metric: how much does the tick-mean argmax move tick→tick?
    // Distance in V4 grid units; compares against random-walk expectation.
    let argmax_yx = |w: &[f32]| -> (usize, usize) {
        let i = (0..w.len()).max_by(|&a, &b| w[a].partial_cmp(&w[b]).unwrap()).unwrap();
        (i / w_tok, i % w_tok)
    };
    let mut means: Vec<Vec<f32>> = (0..n_ticks).map(|t| {
        let mut m = vec![0.0f32; n_tokens];
        for h in 0..n_heads { for i in 0..n_tokens { m[i] += trace[t][h][i]; } }
        let inv = 1.0 / n_heads as f32;
        for v in &mut m { *v *= inv; }
        m
    }).collect();
    let mut total_move = 0.0f32;
    for t in 1..n_ticks {
        let (y0, x0) = argmax_yx(&means[t - 1]);
        let (y1, x1) = argmax_yx(&means[t]);
        let dy = y1 as f32 - y0 as f32;
        let dx = x1 as f32 - x0 as f32;
        total_move += (dy * dy + dx * dx).sqrt();
    }
    let _ = &mut means;
    let mean_step = if n_ticks >= 2 { total_move / (n_ticks - 1) as f32 } else { 0.0 };
    summary.push_str(&format!("\nsaccade: argmax moves between adjacent ticks total={:.2} mean={:.2} (grid units)\n",
        total_move, mean_step));
    summary.push_str(&format!("  baseline (random walk on {h_tok}×{w_tok} grid): expected ~{:.2}/step\n",
        ((h_tok * h_tok + w_tok * w_tok) as f32).sqrt() / 3.0));

    fs::write(format!("{OUT_DIR}/summary.txt"), &summary)?;
    print!("{summary}");
    eprintln!("\nwrote {} PPMs + summary.txt to {}", n_ticks * (n_heads + 1) + 1, OUT_DIR);
    Ok(())
}
