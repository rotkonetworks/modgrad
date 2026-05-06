//! Self-contained SVG chart writer for arena eval traces — no plotting
//! deps, just string building. Lays out a single 1000×600 figure with
//! the mid price on a left axis, per-agent cumulative reward on a
//! right axis (one polyline per agent), and the book skew shaded as
//! a translucent band along the bottom.

use std::fs::File;
use std::io::{self, BufWriter, Write};

const W: usize = 1000;
const H: usize = 600;
const PAD_L: usize = 70;
const PAD_R: usize = 80;
const PAD_T: usize = 40;
const PAD_B: usize = 60;

const PALETTE: &[&str] = &[
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#9a6324",
];

pub struct ArenaTrace {
    pub mid: Vec<f64>,
    pub skew: Vec<f64>,
    pub agent_labels: Vec<String>,
    /// `cum_reward[agent_idx][block]`.
    pub cum_reward: Vec<Vec<f64>>,
}

pub fn write_svg(path: &str, t: &ArenaTrace) -> io::Result<()> {
    let n = t.mid.len();
    if n == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty trace"));
    }

    let (mid_min, mid_max) = bounds(&t.mid);
    let (rew_min, rew_max) = t.cum_reward.iter()
        .flat_map(|s| s.iter().copied())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(x), hi.max(x)));
    let (skew_min, skew_max) = bounds(&t.skew);
    let skew_abs = skew_min.abs().max(skew_max.abs()).max(1e-3);

    let plot_w = W - PAD_L - PAD_R;
    let plot_h = H - PAD_T - PAD_B;
    let x_at = |i: usize| PAD_L as f64 + (i as f64) * (plot_w as f64) / ((n - 1).max(1) as f64);
    let y_mid = |v: f64| {
        let t = (v - mid_min) / (mid_max - mid_min).max(1e-12);
        (PAD_T + plot_h) as f64 - t * (plot_h as f64)
    };
    let y_rew = |v: f64| {
        let t = (v - rew_min) / (rew_max - rew_min).max(1e-12);
        (PAD_T + plot_h) as f64 - t * (plot_h as f64)
    };
    let y_skew_band = |v: f64| {
        // Map skew [−amp, +amp] into a 60-px band along the bottom of
        // the plot — purely for visual context, not on the same axis.
        let band_h = 60.0;
        let band_top = (PAD_T + plot_h) as f64 - band_h;
        let centre = band_top + band_h / 2.0;
        centre - (v / skew_abs) * (band_h / 2.0)
    };

    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    writeln!(w, r##"<?xml version="1.0" encoding="UTF-8"?>"##)?;
    writeln!(w, r##"<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" font-size="12">"##)?;
    writeln!(w, r##"<rect width="{W}" height="{H}" fill="#fafafa"/>"##)?;

    writeln!(w, r##"<rect x="{}" y="{}" width="{}" height="{}" fill="#fff" stroke="#bbb"/>"##,
        PAD_L, PAD_T, plot_w, plot_h)?;

    writeln!(w, r##"<text x="{}" y="20" font-size="14" font-weight="bold">Penumbra arena — mid + per-agent cumulative reward + book skew</text>"##, PAD_L)?;
    writeln!(w, r##"<text x="{}" y="{}" text-anchor="middle">block</text>"##, W / 2, H - 10)?;
    writeln!(w, r##"<text x="20" y="{}" transform="rotate(-90 20 {})" text-anchor="middle">mid (UM/USDC)</text>"##, H / 2, H / 2)?;
    writeln!(w, r##"<text x="{}" y="{}" transform="rotate(-90 {} {})" text-anchor="middle">cumulative reward</text>"##, W - 20, H / 2, W - 20, H / 2)?;

    let n_xticks = 5usize;
    for k in 0..=n_xticks {
        let bi = (k * (n - 1)) / n_xticks;
        let x = x_at(bi);
        writeln!(w, r##"<line x1="{x:.1}" y1="{}" x2="{x:.1}" y2="{}" stroke="#eee"/>"##, PAD_T, PAD_T + plot_h)?;
        writeln!(w, r##"<text x="{x:.1}" y="{}" text-anchor="middle">{bi}</text>"##, PAD_T + plot_h + 16)?;
    }

    let n_yticks = 4usize;
    for k in 0..=n_yticks {
        let v = mid_min + (mid_max - mid_min) * (k as f64) / (n_yticks as f64);
        let y = y_mid(v);
        writeln!(w, r##"<line x1="{}" y1="{y:.1}" x2="{}" y2="{y:.1}" stroke="#eee"/>"##, PAD_L, PAD_L + plot_w)?;
        writeln!(w, r##"<text x="{}" y="{:.1}" text-anchor="end">{v:.5}</text>"##, PAD_L - 8, y + 4.0)?;
        let v_r = rew_min + (rew_max - rew_min) * (k as f64) / (n_yticks as f64);
        writeln!(w, r##"<text x="{}" y="{:.1}" text-anchor="start" fill="#666">{v_r:+.2}</text>"##, PAD_L + plot_w + 8, y + 4.0)?;
    }

    let band_top = (PAD_T + plot_h) as f64 - 60.0;
    let band_bot = (PAD_T + plot_h) as f64;
    writeln!(w, r##"<rect x="{}" y="{:.1}" width="{}" height="60" fill="#fff5e0" opacity="0.6"/>"##, PAD_L, band_top, plot_w)?;
    let mut skew_path = String::from("M ");
    for (i, &s) in t.skew.iter().enumerate() {
        let x = x_at(i);
        let y = y_skew_band(s);
        skew_path.push_str(&format!("{x:.1},{y:.1} "));
        if i == 0 { skew_path.push_str("L "); }
    }
    writeln!(w, r##"<path d="{skew_path}" fill="none" stroke="#d18a00" stroke-width="1.2" opacity="0.8"/>"##)?;
    writeln!(w, r##"<line x1="{}" y1="{:.1}" x2="{}" y2="{:.1}" stroke="#d18a00" stroke-dasharray="2 3" opacity="0.5"/>"##,
        PAD_L, (band_top + band_bot) / 2.0, PAD_L + plot_w, (band_top + band_bot) / 2.0)?;
    writeln!(w, r##"<text x="{}" y="{:.1}" font-size="10" fill="#a05d00">book skew (±{:.2})</text>"##,
        PAD_L + 6, band_top + 12.0, skew_abs)?;

    let mut mid_path = String::from("M ");
    for (i, &m) in t.mid.iter().enumerate() {
        let x = x_at(i);
        let y = y_mid(m);
        mid_path.push_str(&format!("{x:.1},{y:.1} "));
        if i == 0 { mid_path.push_str("L "); }
    }
    writeln!(w, r##"<path d="{mid_path}" fill="none" stroke="#222" stroke-width="2"/>"##)?;

    for (ai, series) in t.cum_reward.iter().enumerate() {
        let color = PALETTE[ai % PALETTE.len()];
        let mut p = String::from("M ");
        for (i, &v) in series.iter().enumerate() {
            let x = x_at(i);
            let y = y_rew(v);
            p.push_str(&format!("{x:.1},{y:.1} "));
            if i == 0 { p.push_str("L "); }
        }
        writeln!(w, r##"<path d="{p}" fill="none" stroke="{color}" stroke-width="1.6" opacity="0.85"/>"##)?;
    }

    let lx = PAD_L + 12;
    let ly0 = PAD_T + 12;
    writeln!(w, r##"<rect x="{lx}" y="{ly0}" width="170" height="{}" fill="#fff" stroke="#ccc" opacity="0.9"/>"##,
        20 + 16 * (t.agent_labels.len() + 1))?;
    writeln!(w, r##"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#222" stroke-width="2"/>"##,
        lx + 8, ly0 + 16, lx + 28, ly0 + 16)?;
    writeln!(w, r##"<text x="{}" y="{}">mid</text>"##, lx + 36, ly0 + 20)?;
    for (ai, label) in t.agent_labels.iter().enumerate() {
        let color = PALETTE[ai % PALETTE.len()];
        let yy = ly0 + 32 + 16 * ai;
        writeln!(w, r##"<line x1="{}" y1="{yy}" x2="{}" y2="{yy}" stroke="{color}" stroke-width="1.6"/>"##,
            lx + 8, lx + 28)?;
        writeln!(w, r##"<text x="{}" y="{}">{label}</text>"##, lx + 36, yy + 4)?;
    }

    writeln!(w, "</svg>")?;
    Ok(())
}

fn bounds(xs: &[f64]) -> (f64, f64) {
    xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY),
        |(lo, hi), &x| (lo.min(x), hi.max(x)))
}
