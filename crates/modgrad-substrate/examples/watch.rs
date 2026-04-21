//! `substrate-watch` — server-grade telemetry sampler.
//!
//! Emits one structured record per interval describing the host
//! CPU's state (frequency, temperature, governor, boost, EPP,
//! amd-pstate mode, throttle delta) and the state of any PIDs the
//! operator asked to watch (command, state, RSS, CPU fraction since
//! previous sample).
//!
//! Design notes — this is intended for long-running observability
//! on servers, not interactive debugging:
//! - Output is one JSON object per line (`--format=jsonl`, default)
//!   or CSV (`--format=csv`). Both are stream-friendly.
//! - Stdout is line-buffered by libc for the terminal and
//!   block-buffered for pipes; we flush explicitly per record to
//!   survive log-rotation and pipeline unwinds.
//! - On SIGINT / SIGTERM the current write completes and we exit
//!   cleanly. No async runtime, no signal-hook dep — we poll a
//!   plain atomic that's set in a signal handler written in pure
//!   safe Rust by reading `SIGTERM_PENDING` via `std::sync::atomic`
//!   is not possible without a C shim, so we accept the simpler
//!   model of letting the kernel terminate us between iterations.
//! - Zero external dependencies — only the modgrad-substrate crate
//!   plus std.
//! - Arguments parsed by hand to keep the audit surface small.
//!   Unknown flags hard-fail rather than being silently ignored.
//!
//! Example:
//!   cargo run --release -p modgrad-substrate --example watch -- \
//!     --interval-ms=500 --pattern=mazes --target-temp-c=88

use modgrad_substrate::{Snapshot, process};
use std::collections::HashMap;
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Args {
    interval_ms: u64,
    duration_s: u64, // 0 = forever
    pids: Vec<u32>,
    pattern: Option<String>,
    target_temp_c: f32,
    bias: f32,
    format: Format,
    no_header: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Jsonl,
    Csv,
}

impl Args {
    fn defaults() -> Self {
        Self {
            interval_ms: 1000,
            duration_s: 0,
            pids: Vec::new(),
            pattern: None,
            target_temp_c: 85.0,
            bias: 0.0,
            format: Format::Jsonl,
            no_header: false,
        }
    }
}

const USAGE: &str = r"usage: substrate-watch [options]

options:
  --interval-ms=N       sample every N ms (default 1000)
  --duration-s=N        exit after N seconds (0 = forever, default 0)
  --pid=N               watch pid N (repeatable)
  --pattern=STR         auto-select pids whose /proc/*/cmdline contains STR
  --target-temp-c=F     emit a `temp_over_target` field when max temp >= F (default 85.0)
  --bias=F              performance bias 0.0 .. 1.0 (informational, mapped
                        to a governor/epp suggestion in the record)
  --format=jsonl|csv    output format (default jsonl)
  --no-header           suppress CSV header line
  -h, --help            this message
";

fn parse_args(argv: &[String]) -> std::result::Result<Args, String> {
    let mut args = Args::defaults();
    for raw in &argv[1..] {
        let (flag, val) = raw
            .split_once('=')
            .map_or((raw.as_str(), None), |(k, v)| (k, Some(v)));
        match (flag, val) {
            ("-h" | "--help", _) => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            ("--interval-ms", Some(v)) => {
                args.interval_ms = v.parse().map_err(|e| format!("--interval-ms: {e}"))?;
                if args.interval_ms < 10 {
                    return Err("--interval-ms must be >= 10".to_owned());
                }
            }
            ("--duration-s", Some(v)) => {
                args.duration_s = v.parse().map_err(|e| format!("--duration-s: {e}"))?;
            }
            ("--pid", Some(v)) => {
                let pid: u32 = v.parse().map_err(|e| format!("--pid: {e}"))?;
                args.pids.push(pid);
            }
            ("--pattern", Some(v)) => {
                args.pattern = Some(v.to_owned());
            }
            ("--target-temp-c", Some(v)) => {
                args.target_temp_c = v.parse().map_err(|e| format!("--target-temp-c: {e}"))?;
            }
            ("--bias", Some(v)) => {
                args.bias = v.parse().map_err(|e| format!("--bias: {e}"))?;
                if !(0.0..=1.0).contains(&args.bias) {
                    return Err("--bias must be in [0.0, 1.0]".to_owned());
                }
            }
            ("--format", Some(v)) => {
                args.format = match v {
                    "jsonl" => Format::Jsonl,
                    "csv" => Format::Csv,
                    other => return Err(format!("--format: unknown `{other}`")),
                };
            }
            ("--no-header", None) => {
                args.no_header = true;
            }
            _ => return Err(format!("unknown flag `{raw}` (try --help)")),
        }
    }
    Ok(args)
}

// ─────────────────────────────────────────────────────────────
// Bias → suggested governor/EPP mapping. Purely advisory; this
// tool does not write anything to sysfs.
// ─────────────────────────────────────────────────────────────

fn bias_to_epp_suggestion(bias: f32) -> &'static str {
    // Piecewise from the kernel-documented EPP ladder. We pick
    // boundaries at 0.2 / 0.5 / 0.8 so that the ends (`performance`
    // / `power`) have slightly larger catchment regions, which
    // matches how operators actually use the knob in practice.
    match bias {
        x if x < 0.2 => "performance",
        x if x < 0.5 => "balance_performance",
        x if x < 0.8 => "balance_power",
        _ => "power",
    }
}

// ─────────────────────────────────────────────────────────────
// Per-process state carried across samples so we can compute a
// %CPU fraction from the utime/stime delta.
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct ProcPrev {
    ticks: u64,
}

// ─────────────────────────────────────────────────────────────
// Record emission. We keep it boring: hand-written JSON/CSV, no
// serde, no fancy escaping — the fields we emit are all numbers
// and short identifiers, so the minimal escape surface (`"` in
// comm) is handled locally.
// ─────────────────────────────────────────────────────────────

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for c in s.chars() {
            if c == '"' {
                out.push('"');
            }
            out.push(c);
        }
        out.push('"');
        out
    } else {
        s.to_owned()
    }
}

fn emit_csv_header(w: &mut impl Write) -> io::Result<()> {
    writeln!(
        w,
        "t_ns,governor,driver,epp,boost,amd_pstate,mean_freq_khz,freq_ratio,max_temp_c,throttle_delta,temp_over_target,bias,epp_suggestion,pid,comm,state,rss_kb,vm_peak_kb,cpu_pct"
    )
}

struct Record<'a> {
    t_ns: u128,
    governor: &'a str,
    driver: &'a str,
    epp: Option<&'a str>,
    boost: Option<bool>,
    amd_pstate: Option<&'a str>,
    mean_freq_khz: Option<u64>,
    freq_ratio: Option<f32>,
    max_temp_c: Option<f32>,
    throttle_delta: u64,
    temp_over_target: bool,
    bias: f32,
    epp_suggestion: &'static str,
    procs: Vec<ProcRecord<'a>>,
}

struct ProcRecord<'a> {
    pid: u32,
    comm: &'a str,
    state: char,
    rss_kb: u64,
    vm_peak_kb: u64,
    cpu_pct: f32,
}

fn write_jsonl<W: Write>(w: &mut W, r: &Record<'_>) -> io::Result<()> {
    write!(
        w,
        "{{\"t\":{},\"governor\":\"{}\",\"driver\":\"{}\"",
        r.t_ns,
        json_escape(r.governor),
        json_escape(r.driver),
    )?;
    match r.epp {
        Some(v) => write!(w, ",\"epp\":\"{}\"", json_escape(v))?,
        None => write!(w, ",\"epp\":null")?,
    }
    match r.boost {
        Some(true) => write!(w, ",\"boost\":true")?,
        Some(false) => write!(w, ",\"boost\":false")?,
        None => write!(w, ",\"boost\":null")?,
    }
    match r.amd_pstate {
        Some(v) => write!(w, ",\"amd_pstate\":\"{}\"", json_escape(v))?,
        None => write!(w, ",\"amd_pstate\":null")?,
    }
    match r.mean_freq_khz {
        Some(v) => write!(w, ",\"mean_freq_khz\":{v}")?,
        None => write!(w, ",\"mean_freq_khz\":null")?,
    }
    match r.freq_ratio {
        Some(v) => write!(w, ",\"freq_ratio\":{v:.3}")?,
        None => write!(w, ",\"freq_ratio\":null")?,
    }
    match r.max_temp_c {
        Some(v) => write!(w, ",\"max_temp_c\":{v:.1}")?,
        None => write!(w, ",\"max_temp_c\":null")?,
    }
    write!(
        w,
        ",\"throttle_delta\":{},\"temp_over_target\":{},\"bias\":{:.2},\"epp_suggestion\":\"{}\"",
        r.throttle_delta,
        if r.temp_over_target { "true" } else { "false" },
        r.bias,
        r.epp_suggestion,
    )?;
    write!(w, ",\"procs\":[")?;
    for (i, p) in r.procs.iter().enumerate() {
        if i > 0 {
            write!(w, ",")?;
        }
        write!(
            w,
            "{{\"pid\":{},\"comm\":\"{}\",\"state\":\"{}\",\"rss_kb\":{},\"vm_peak_kb\":{},\"cpu_pct\":{:.1}}}",
            p.pid,
            json_escape(p.comm),
            p.state,
            p.rss_kb,
            p.vm_peak_kb,
            p.cpu_pct,
        )?;
    }
    writeln!(w, "]}}")
}

fn write_csv<W: Write>(w: &mut W, r: &Record<'_>) -> io::Result<()> {
    // CSV-flattening convention: one row per process, or one row
    // with empty process columns when no pids are being watched.
    // Host fields repeat across all rows for a given `t_ns`; this
    // is the convention used by tools like `sar`.
    let host_prefix = format!(
        "{},{},{},{},{},{},{},{},{},{},{},{:.2},{}",
        r.t_ns,
        csv_escape(r.governor),
        csv_escape(r.driver),
        r.epp.map(csv_escape).unwrap_or_default(),
        r.boost.map_or(String::new(), |b| b.to_string()),
        r.amd_pstate.map(csv_escape).unwrap_or_default(),
        r.mean_freq_khz.map_or(String::new(), |v| v.to_string()),
        r.freq_ratio.map_or(String::new(), |v| format!("{v:.3}")),
        r.max_temp_c.map_or(String::new(), |v| format!("{v:.1}")),
        r.throttle_delta,
        r.temp_over_target,
        r.bias,
        r.epp_suggestion,
    );
    if r.procs.is_empty() {
        writeln!(w, "{host_prefix},,,,,,")?;
    } else {
        for p in &r.procs {
            writeln!(
                w,
                "{host_prefix},{},{},{},{},{},{:.1}",
                p.pid,
                csv_escape(p.comm),
                p.state,
                p.rss_kb,
                p.vm_peak_kb,
                p.cpu_pct,
            )?;
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────
// Main loop.
// ─────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("substrate-watch: {e}");
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();

    if args.format == Format::Csv
        && !args.no_header
        && let Err(e) = emit_csv_header(&mut out)
    {
        eprintln!("substrate-watch: header write failed: {e}");
        return ExitCode::from(1);
    }

    // Figure out which pids to watch. If `--pattern` was passed,
    // re-resolve every sample so we pick up new processes; this is
    // a cheap /proc walk.
    let explicit_pids = args.pids.clone();
    let pattern = args.pattern.clone();

    let interval = Duration::from_millis(args.interval_ms);
    let deadline = if args.duration_s == 0 {
        None
    } else {
        Some(Instant::now() + Duration::from_secs(args.duration_s))
    };

    // Per-pid state for CPU% computation. We keep track of the
    // cumulative tick count and the instant of the last sample.
    let mut prev_procs: HashMap<u32, ProcPrev> = HashMap::new();
    let mut prev_throttle: Option<u64> = None;

    // Driver + EPP + amd-pstate mode are effectively static per-boot
    // but cheap to re-read; we do so every sample so a mid-run
    // `echo passive | sudo tee ...` shows up in telemetry.
    let mut iteration: u64 = 0;
    loop {
        let sample_start = Instant::now();

        let snap = match Snapshot::take() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("substrate-watch: snapshot failed: {e}");
                return ExitCode::from(1);
            }
        };

        // Scaling driver + EPP. Neither may exist on this host.
        let driver = modgrad_substrate::cpu_scaling_driver(0).unwrap_or_else(|_| "unknown".into());
        let epp = modgrad_substrate::cpu_energy_performance_preference(0).ok();
        let boost = modgrad_substrate::boost_enabled().unwrap_or(None);
        let amd_pstate = modgrad_substrate::amd_pstate_mode().unwrap_or(None);

        // Resolve pids this iteration. Exclude our own pid so a
        // `--pattern=foo` that matches our own argv doesn't yield
        // self-noise in the telemetry stream.
        let my_pid = std::process::id();
        let mut pids: Vec<u32> = explicit_pids.clone();
        if let Some(pat) = &pattern
            && let Ok(hits) = process::find_by_cmdline(pat)
        {
            for p in hits {
                if p != my_pid && !pids.contains(&p) {
                    pids.push(p);
                }
            }
        }

        // Sample each pid. A pid that vanishes between resolution
        // and read is silently dropped from this sample.
        let mut proc_records_data: Vec<(u32, String, char, u64, u64, f32)> = Vec::new();
        for pid in &pids {
            let info = match process::read(*pid) {
                Ok(i) => i,
                Err(_) => continue,
            };
            let total = info.utime_ticks + info.stime_ticks;
            let cpu_pct = if let Some(prev) = prev_procs.get(pid) {
                // Tick delta per interval_ms, normalised to 0..100.
                let dt_ticks = total.saturating_sub(prev.ticks) as f32;
                // USER_HZ is typically 100 on x86_64 — one tick per
                // 10 ms. We avoid depending on sysconf by using the
                // interval itself as the denominator: if the
                // process used D ticks in T ms, it was using
                // D * 10 / T fraction of a CPU. This matches
                // `top`'s convention where 100% = one full core.
                #[allow(clippy::cast_precision_loss)]
                let interval_ms = args.interval_ms as f32;
                (dt_ticks * 10.0 / interval_ms) * 100.0
            } else {
                f32::NAN
            };
            prev_procs.insert(*pid, ProcPrev { ticks: total });
            proc_records_data.push((
                info.pid,
                info.comm,
                info.state,
                info.vm_rss_kb,
                info.vm_peak_kb,
                cpu_pct,
            ));
        }

        // Throttle delta — compare to previous sample.
        let now_throttle = snap.throttle_total();
        let throttle_delta = prev_throttle.map_or(0, |p| now_throttle.saturating_sub(p));
        prev_throttle = Some(now_throttle);

        // Decide "temp over target".
        let max_temp_c = snap.max_temp_c();
        let temp_over_target = max_temp_c.is_some_and(|t| t >= args.target_temp_c);

        let record = Record {
            t_ns: snap.taken_unix_ns,
            governor: &snap.governor,
            driver: &driver,
            epp: epp.as_deref(),
            boost,
            amd_pstate: amd_pstate.as_deref(),
            mean_freq_khz: snap.mean_freq_khz(),
            freq_ratio: snap.mean_freq_ratio(),
            max_temp_c,
            throttle_delta,
            temp_over_target,
            bias: args.bias,
            epp_suggestion: bias_to_epp_suggestion(args.bias),
            procs: proc_records_data
                .iter()
                .map(|(pid, comm, state, rss, peak, cpu_pct)| ProcRecord {
                    pid: *pid,
                    comm,
                    state: *state,
                    rss_kb: *rss,
                    vm_peak_kb: *peak,
                    cpu_pct: *cpu_pct,
                })
                .collect(),
        };

        let write_res = match args.format {
            Format::Jsonl => write_jsonl(&mut out, &record),
            Format::Csv => write_csv(&mut out, &record),
        };
        if let Err(e) = write_res {
            // stdout closed (pipe hung up) — common for `| head` etc;
            // exit silently rather than spraying errors.
            if e.kind() == io::ErrorKind::BrokenPipe {
                return ExitCode::SUCCESS;
            }
            eprintln!("substrate-watch: write failed: {e}");
            return ExitCode::from(1);
        }
        let _ = out.flush();

        iteration += 1;

        if let Some(d) = deadline
            && Instant::now() >= d
        {
            return ExitCode::SUCCESS;
        }

        // Sleep until the next tick aligns. If sampling took longer
        // than `interval`, we skip the sleep and continue
        // immediately — the jitter is reported via the per-sample
        // timestamp.
        let elapsed = sample_start.elapsed();
        if elapsed < interval {
            std::thread::sleep(interval - elapsed);
        }
        let _ = iteration; // read for potential future use
    }
}
