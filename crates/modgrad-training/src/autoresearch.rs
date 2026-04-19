//! Autoresearch summary printer.
//!
//! Karpathy's `autoresearch` repo (github.com/karpathy/autoresearch) defines
//! a grep-parseable end-of-run summary block that the driving agent reads to
//! decide keep/revert. This helper prints the same block so a modgrad-driven
//! run can plug into the same workflow unchanged.
//!
//! The format is the contract — field names and order match the reference
//! implementation exactly. Agents do:
//!
//! ```text
//! grep "^val_bpb:\|^peak_vram_mb:" run.log
//! ```
//!
//! Missing fields are set to `0` / `0.0` rather than omitted, to keep the
//! grep patterns stable.

use std::io::{self, Write};

/// Fields the autoresearch contract requires. Field names here mirror the
/// line labels the agent grep-parses.
#[derive(Debug, Clone)]
pub struct AutoresearchSummary {
    /// Validation bits-per-byte. Lower is better. This is the ground truth.
    pub val_bpb: f32,
    /// Seconds spent in the training loop proper (excluding startup/eval).
    pub training_seconds: f32,
    /// Total wall-clock seconds for the whole run.
    pub total_seconds: f32,
    /// Peak GPU memory in MiB. `0.0` if unmeasured (CPU-only runs).
    pub peak_vram_mb: f32,
    /// Steady-state MFU percent. `0.0` if not computed.
    pub mfu_percent: f32,
    /// Total training tokens, in millions.
    pub total_tokens_m: f32,
    /// Total training steps.
    pub num_steps: u64,
    /// Model parameter count, in millions.
    pub num_params_m: f32,
}

impl AutoresearchSummary {
    /// Print to the given writer in the exact autoresearch format.
    ///
    /// Callers pass `io::stderr().lock()` for the default autoresearch
    /// workflow; tests pass a `Vec<u8>` and assert on the captured bytes.
    pub fn write_to<W: Write>(&self, mut out: W) -> io::Result<()> {
        writeln!(out, "---")?;
        writeln!(out, "val_bpb:          {:.6}", self.val_bpb)?;
        writeln!(out, "training_seconds: {:.1}", self.training_seconds)?;
        writeln!(out, "total_seconds:    {:.1}", self.total_seconds)?;
        writeln!(out, "peak_vram_mb:     {:.1}", self.peak_vram_mb)?;
        writeln!(out, "mfu_percent:      {:.2}", self.mfu_percent)?;
        writeln!(out, "total_tokens_m:   {:.1}", self.total_tokens_m)?;
        writeln!(out, "num_steps:        {}", self.num_steps)?;
        writeln!(out, "num_params_m:     {:.1}", self.num_params_m)?;
        Ok(())
    }

    /// Convenience: print to stderr. Match autoresearch's default — stderr
    /// is where its `train.py` prints to, so `grep` over `run.log` catches
    /// it when the agent redirects with `2>&1`.
    pub fn print(&self) {
        let _ = self.write_to(io::stderr().lock());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> AutoresearchSummary {
        AutoresearchSummary {
            val_bpb: 0.997900,
            training_seconds: 300.1,
            total_seconds: 325.9,
            peak_vram_mb: 45060.2,
            mfu_percent: 39.80,
            total_tokens_m: 499.6,
            num_steps: 953,
            num_params_m: 50.3,
        }
    }

    #[test]
    fn format_matches_autoresearch_contract() {
        // Byte-for-byte equality with the reference block. If this breaks,
        // the driving agent's grep silently stops finding results.
        let mut out = Vec::new();
        sample().write_to(&mut out).unwrap();
        let got = String::from_utf8(out).unwrap();
        let expected = "\
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_m:   499.6
num_steps:        953
num_params_m:     50.3
";
        assert_eq!(got, expected);
    }

    #[test]
    fn grep_patterns_find_expected_fields() {
        // The agent runs `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
        // Verify those line prefixes are present and unambiguous.
        let mut out = Vec::new();
        sample().write_to(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        let bpb_line = s.lines().find(|l| l.starts_with("val_bpb:"))
            .expect("val_bpb line must exist for agent grep");
        let vram_line = s.lines().find(|l| l.starts_with("peak_vram_mb:"))
            .expect("peak_vram_mb line must exist for agent grep");
        assert!(bpb_line.contains("0.997900"));
        assert!(vram_line.contains("45060.2"));
    }

    #[test]
    fn zero_defaults_preserve_field_presence() {
        // Regression: a CPU run has no VRAM; the field must still print
        // as `peak_vram_mb: 0.0` so the agent's grep finds *something*.
        let s = AutoresearchSummary {
            val_bpb: 2.5, training_seconds: 10.0, total_seconds: 12.0,
            peak_vram_mb: 0.0, mfu_percent: 0.0, total_tokens_m: 1.0,
            num_steps: 100, num_params_m: 5.0,
        };
        let mut out = Vec::new();
        s.write_to(&mut out).unwrap();
        let got = String::from_utf8(out).unwrap();
        assert!(got.contains("peak_vram_mb:     0.0"));
        assert!(got.contains("mfu_percent:      0.00"));
    }
}
