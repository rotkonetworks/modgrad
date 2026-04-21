//! `substrate-analyze` — summarise a `--substrate-log` CSV.
//!
//! Consumes the per-step telemetry produced by either
//! `examples/watch.rs --format=csv` or `mazes --substrate-log=PATH`
//! and emits summary statistics plus phase markers. Intended for
//! grepping through a finished run on the command line; zero
//! external dependencies; hand-written CSV parser that accepts
//! the exact schema modgrad emits and nothing else.
//!
//! Schemas supported:
//!   mazes:  step,loss,acc,freq_ratio,mean_freq_khz,max_temp_c,throttle_delta,governor,epp
//!   watch:  (wider row, first field is t_ns instead of step)
//!
//! Discrimination is by the header row: if the first column is
//! `step` we treat it as the mazes schema, else we assume the
//! watch-tool csv shape.
//!
//! Output is plain text, one stat per line, stable key=value so
//! it scripts easily:
//!
//!   $ substrate-analyze /tmp/mazes.csv
//!   rows=200
//!   freq_ratio_mean=0.518
//!   freq_ratio_min=0.445
//!   freq_ratio_max=0.631
//!   max_temp_c_mean=91.4
//!   throttle_events=0
//!   freq_ratio_trend_per_step=0.000284
//!   loss_decrease_pct=39.2
//!   loss_freq_corr=-0.12
//!
//! The last two are the *interesting* ones — they quantify
//! whether workload progress and substrate state move together.
//!
//! ## Design
//!
//! - Streaming: one line at a time, never allocates more than one
//!   row's worth. Can chew a 1M-step log in constant memory.
//! - Running statistics via Welford's algorithm (numerically
//!   stable mean/variance, avoids catastrophic cancellation when
//!   computing correlations on long runs).
//! - Errors in a single row are logged to stderr and that row is
//!   skipped — a malformed middle row does not invalidate the
//!   whole analysis. Empty files produce `rows=0` and no further
//!   stats, not a crash.

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::process::ExitCode;

// ─────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────

const USAGE: &str = r"usage: substrate-analyze <csv> [options]

options:
  --format=key-value|json    output format (default key-value)
  --ignore-parse-errors      skip malformed rows rather than warn (quiet mode)
  -h, --help
";

#[derive(Debug, Clone, Copy)]
enum OutputFormat {
    KeyValue,
    Json,
}

struct Args {
    path: String,
    format: OutputFormat,
    ignore_parse_errors: bool,
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut path: Option<String> = None;
    let mut format = OutputFormat::KeyValue;
    let mut ignore_parse_errors = false;
    for raw in &argv[1..] {
        let (flag, val) = raw
            .split_once('=')
            .map_or((raw.as_str(), None), |(k, v)| (k, Some(v)));
        match (flag, val) {
            ("-h" | "--help", _) => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            ("--format", Some("key-value")) => format = OutputFormat::KeyValue,
            ("--format", Some("json")) => format = OutputFormat::Json,
            ("--format", Some(other)) => {
                return Err(format!("--format: unknown `{other}`"));
            }
            ("--ignore-parse-errors", None) => ignore_parse_errors = true,
            _ if !raw.starts_with("--") && path.is_none() => {
                path = Some(raw.clone());
            }
            _ => return Err(format!("unknown flag `{raw}` (try --help)")),
        }
    }
    match path {
        Some(p) => Ok(Args { path: p, format, ignore_parse_errors }),
        None => Err("missing csv path".to_owned()),
    }
}

// ─────────────────────────────────────────────────────────────
// Row schema — minimal struct that carries what we actually
// compute on. Governor / EPP / driver are carried as the *first
// observed* and *most recent* values so we can flag config
// changes mid-run.
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct MazesRow {
    step: u64,
    loss: f64,
    acc: f64,
    freq_ratio: f64,
    max_temp_c: f64,
    throttle_delta: u64,
}

fn parse_mazes_row(line: &str) -> Result<MazesRow, String> {
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() < 7 {
        return Err(format!("expected >=7 fields, got {}", fields.len()));
    }
    let parse_u64 = |idx: usize, name: &str| -> Result<u64, String> {
        fields[idx]
            .trim()
            .parse::<u64>()
            .map_err(|e| format!("{name}: {e}"))
    };
    let parse_f64 = |idx: usize, name: &str| -> Result<f64, String> {
        let raw = fields[idx].trim();
        if raw.eq_ignore_ascii_case("nan") {
            return Ok(f64::NAN);
        }
        raw.parse::<f64>().map_err(|e| format!("{name}: {e}"))
    };
    Ok(MazesRow {
        step: parse_u64(0, "step")?,
        loss: parse_f64(1, "loss")?,
        acc: parse_f64(2, "acc")?,
        freq_ratio: parse_f64(3, "freq_ratio")?,
        max_temp_c: parse_f64(5, "max_temp_c")?,
        throttle_delta: parse_u64(6, "throttle_delta")?,
    })
}

// ─────────────────────────────────────────────────────────────
// Welford running statistics — numerically stable mean/var/covar
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default)]
struct Running {
    n: u64,
    mean: f64,
    /// Sum of squared deviations from the running mean (M2).
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl Running {
    fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        let delta = x - self.mean;
        #[allow(clippy::cast_precision_loss)]
        let n = self.n as f64;
        self.mean += delta / n;
        let delta2 = x - self.mean;
        self.sum_sq += delta * delta2;
        if self.n == 1 {
            self.min = x;
            self.max = x;
        } else {
            if x < self.min { self.min = x; }
            if x > self.max { self.max = x; }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn variance(&self) -> f64 {
        if self.n < 2 { return 0.0; }
        self.sum_sq / (self.n as f64 - 1.0)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Covariance {
    n: u64,
    mean_x: f64,
    mean_y: f64,
    sum_co: f64,
}

impl Covariance {
    fn push(&mut self, x: f64, y: f64) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        self.n += 1;
        #[allow(clippy::cast_precision_loss)]
        let n = self.n as f64;
        let dx = x - self.mean_x;
        self.mean_x += dx / n;
        let dy = y - self.mean_y;
        self.mean_y += dy / n;
        self.sum_co += dx * (y - self.mean_y);
    }

    #[allow(clippy::cast_precision_loss)]
    fn correlation(&self, var_x: f64, var_y: f64) -> f64 {
        if self.n < 2 || var_x <= 0.0 || var_y <= 0.0 {
            return f64::NAN;
        }
        let cov = self.sum_co / (self.n as f64 - 1.0);
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

// ─────────────────────────────────────────────────────────────
// Main analyser
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct Analysis {
    rows: u64,
    freq_ratio: Running,
    temp_c: Running,
    loss: Running,
    acc: Running,
    throttle_events: u64,
    first_loss: Option<f64>,
    last_loss: Option<f64>,
    /// Linear regression of freq_ratio on step index — slope per
    /// step. Tells us whether the CPU is getting progressively
    /// more or less throttled during the run.
    step_vs_freq: Covariance,
    step_var: Running,
    freq_var: Running,
    /// Correlation of loss and freq_ratio — does workload
    /// progress track substrate state?
    loss_vs_freq: Covariance,
    loss_var: Running,
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("substrate-analyze: {e}");
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    let file = match File::open(&args.path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("substrate-analyze: cannot open {}: {e}", args.path);
            return ExitCode::from(1);
        }
    };
    let reader = BufReader::new(file);

    let mut analysis = Analysis::default();
    let mut is_first = true;
    for (lineno, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!(
                    "substrate-analyze: read error at line {}: {e}",
                    lineno + 1
                );
                return ExitCode::from(1);
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        if is_first {
            // Header. We only support the mazes schema right now.
            if !line.starts_with("step,") {
                eprintln!(
                    "substrate-analyze: unsupported CSV schema (expected header starting with `step,`): got `{}`",
                    line.chars().take(80).collect::<String>()
                );
                return ExitCode::from(1);
            }
            is_first = false;
            continue;
        }
        match parse_mazes_row(&line) {
            Ok(row) => {
                ingest_row(&mut analysis, &row);
            }
            Err(e) => {
                if !args.ignore_parse_errors {
                    eprintln!(
                        "substrate-analyze: line {} parse error: {e}",
                        lineno + 1
                    );
                }
                // Continue — one bad row shouldn't kill the analysis.
            }
        }
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let write_res = match args.format {
        OutputFormat::KeyValue => emit_keyvalue(&mut out, &analysis),
        OutputFormat::Json => emit_json(&mut out, &analysis),
    };
    if let Err(e) = write_res {
        if e.kind() == io::ErrorKind::BrokenPipe {
            return ExitCode::SUCCESS;
        }
        eprintln!("substrate-analyze: write: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

#[allow(clippy::cast_precision_loss)]
fn ingest_row(a: &mut Analysis, row: &MazesRow) {
    a.rows += 1;
    a.freq_ratio.push(row.freq_ratio);
    a.temp_c.push(row.max_temp_c);
    a.loss.push(row.loss);
    a.acc.push(row.acc);
    if row.throttle_delta > 0 {
        a.throttle_events += 1;
    }
    if a.first_loss.is_none() && row.loss.is_finite() {
        a.first_loss = Some(row.loss);
    }
    if row.loss.is_finite() {
        a.last_loss = Some(row.loss);
    }
    let step_f = row.step as f64;
    a.step_vs_freq.push(step_f, row.freq_ratio);
    a.step_var.push(step_f);
    a.freq_var.push(row.freq_ratio);
    a.loss_vs_freq.push(row.loss, row.freq_ratio);
    a.loss_var.push(row.loss);
}

fn emit_keyvalue<W: Write>(w: &mut W, a: &Analysis) -> io::Result<()> {
    writeln!(w, "rows={}", a.rows)?;
    if a.rows == 0 {
        return Ok(());
    }
    writeln!(w, "freq_ratio_mean={:.3}", a.freq_ratio.mean)?;
    writeln!(w, "freq_ratio_min={:.3}", a.freq_ratio.min)?;
    writeln!(w, "freq_ratio_max={:.3}", a.freq_ratio.max)?;
    writeln!(w, "freq_ratio_stddev={:.4}", a.freq_ratio.variance().sqrt())?;
    writeln!(w, "max_temp_c_mean={:.1}", a.temp_c.mean)?;
    writeln!(w, "max_temp_c_min={:.1}", a.temp_c.min)?;
    writeln!(w, "max_temp_c_max={:.1}", a.temp_c.max)?;
    writeln!(w, "loss_mean={:.4}", a.loss.mean)?;
    writeln!(w, "acc_mean={:.4}", a.acc.mean)?;
    writeln!(w, "throttle_events={}", a.throttle_events)?;
    if let (Some(first), Some(last)) = (a.first_loss, a.last_loss) {
        let pct = if first.abs() > 1e-9 {
            (first - last) / first * 100.0
        } else {
            0.0
        };
        writeln!(w, "loss_decrease_pct={pct:.1}")?;
    }
    // Slope of freq_ratio vs step via covariance / variance
    let step_var = a.step_var.variance();
    if step_var > 0.0 {
        let cov = if a.step_vs_freq.n >= 2 {
            #[allow(clippy::cast_precision_loss)]
            let d = a.step_vs_freq.n as f64 - 1.0;
            a.step_vs_freq.sum_co / d
        } else {
            0.0
        };
        let slope = cov / step_var;
        writeln!(w, "freq_ratio_trend_per_step={slope:.6}")?;
    }
    // Correlation of loss and freq_ratio
    let var_loss = a.loss_var.variance();
    let var_freq = a.freq_var.variance();
    let corr = a.loss_vs_freq.correlation(var_loss, var_freq);
    writeln!(w, "loss_freq_corr={corr:.3}")?;
    Ok(())
}

fn emit_json<W: Write>(w: &mut W, a: &Analysis) -> io::Result<()> {
    let step_var = a.step_var.variance();
    let slope = if step_var > 0.0 && a.step_vs_freq.n >= 2 {
        #[allow(clippy::cast_precision_loss)]
        let d = a.step_vs_freq.n as f64 - 1.0;
        (a.step_vs_freq.sum_co / d) / step_var
    } else {
        f64::NAN
    };
    let corr = a
        .loss_vs_freq
        .correlation(a.loss_var.variance(), a.freq_var.variance());
    let loss_pct = match (a.first_loss, a.last_loss) {
        (Some(f), Some(l)) if f.abs() > 1e-9 => (f - l) / f * 100.0,
        _ => f64::NAN,
    };
    write!(w, "{{")?;
    write!(w, "\"rows\":{}", a.rows)?;
    if a.rows > 0 {
        write!(w, ",\"freq_ratio_mean\":{:.3}", a.freq_ratio.mean)?;
        write!(w, ",\"freq_ratio_min\":{:.3}", a.freq_ratio.min)?;
        write!(w, ",\"freq_ratio_max\":{:.3}", a.freq_ratio.max)?;
        write!(w, ",\"max_temp_c_mean\":{:.1}", a.temp_c.mean)?;
        write!(w, ",\"throttle_events\":{}", a.throttle_events)?;
        write!(w, ",\"loss_decrease_pct\":{loss_pct:.2}")?;
        write!(w, ",\"freq_ratio_trend_per_step\":{slope:.6}")?;
        write!(w, ",\"loss_freq_corr\":{corr:.3}")?;
    }
    writeln!(w, "}}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_one_mazes_row() {
        let line = "42,1.234,0.567,0.520,3500000,91.0,0,performance,performance";
        let row = parse_mazes_row(line).unwrap();
        assert_eq!(row.step, 42);
        assert!((row.loss - 1.234).abs() < 1e-6);
        assert!((row.freq_ratio - 0.520).abs() < 1e-6);
        assert!((row.max_temp_c - 91.0).abs() < 1e-6);
        assert_eq!(row.throttle_delta, 0);
    }

    #[test]
    fn parse_rejects_short_row() {
        let line = "42,1.234";
        assert!(parse_mazes_row(line).is_err());
    }

    #[test]
    fn parse_tolerates_nan_fields() {
        // mazes writes `NaN` when no prior history exists.
        let line = "1,NaN,NaN,0.500,3000000,90.0,0,performance,performance";
        let row = parse_mazes_row(line).unwrap();
        assert!(row.loss.is_nan());
        assert!(row.acc.is_nan());
        assert!(!row.freq_ratio.is_nan());
    }

    #[test]
    fn running_stats_match_batched() {
        let mut r = Running::default();
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        for v in vals {
            r.push(v);
        }
        assert!((r.mean - 3.0).abs() < 1e-12);
        // Sample variance is 2.5 for [1..5].
        assert!((r.variance() - 2.5).abs() < 1e-12);
        assert!((r.min - 1.0).abs() < 1e-12);
        assert!((r.max - 5.0).abs() < 1e-12);
    }

    #[test]
    fn covariance_is_stable_on_shifted_inputs() {
        // cov(x,y) must equal cov(x+K, y+K) for any K — Welford
        // has this property; the naïve formula doesn't.
        let mut a = Covariance::default();
        let mut b = Covariance::default();
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 4.0, 5.0, 4.0, 5.0];
        let shift = 1e9;
        for i in 0..xs.len() {
            a.push(xs[i], ys[i]);
            b.push(xs[i] + shift, ys[i] + shift);
        }
        // Compute covariances at the same N.
        #[allow(clippy::cast_precision_loss)]
        let div = (a.n as f64 - 1.0).max(1.0);
        let cov_a = a.sum_co / div;
        let cov_b = b.sum_co / div;
        assert!((cov_a - cov_b).abs() < 1e-6, "cov_a={cov_a} cov_b={cov_b}");
    }
}
