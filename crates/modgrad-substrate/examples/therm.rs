//! `substrate-therm` — target-temperature CPU frequency governor.
//!
//! Maintains a per-policy `scaling_max_freq` ceiling with the goal
//! of keeping CPU temperature at or below a target. Runs as a
//! userspace daemon, needs write access to
//! `/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq` (root or
//! an explicit DAC grant).
//!
//! Why this works on amd-pstate-epp (active mode) too: the generic
//! `scaling_max_freq` ceiling is honored even when the driver is
//! managing P-states internally. The SMU cannot boost past the
//! ceiling we write, so clock caps turn into thermal caps on any
//! workload that's heat-limited.
//!
//! ## Design
//!
//! - One control tick every `--interval-ms`. Reads the hottest CPU
//!   temp via the modgrad-substrate hwmon reader.
//! - Bang-bang with hysteresis: above `target + hyst` → lower cap;
//!   below `target - hyst` → raise cap. Between the bands: hold.
//!   Simple, robust, no tuning required. A learned controller can
//!   slot into the same write-side later — this tool is the
//!   reference baseline to beat.
//! - SIGINT / SIGTERM → restore every CPU's original
//!   `scaling_max_freq` before exit. Kept uses minimal unsafe for
//!   `libc::signal` (one sighandler registered at startup) and no
//!   inter-thread state beyond a single `AtomicBool`.
//! - Every write is logged to stderr so operators can tell what
//!   the daemon did. `--dry-run` computes but writes nothing,
//!   useful for tuning thresholds before giving the daemon root.
//! - Writes are *idempotent* — if the target cap equals the
//!   current cap, no `write` syscall is issued. Avoids spamming
//!   sysfs when the controller is at steady state.
//!
//! ## Example
//!
//! Keep CPU at ≤80 °C, adjust every 100 ms, 100 MHz steps:
//!
//!   sudo ./target/release/examples/therm \
//!       --target-temp-c=80 --interval-ms=100 --step-khz=100000
//!
//! ## Not in scope
//!
//! - PPT / TDC adjustment (would need ryzen_smu, out-of-tree
//!   kernel module; orthogonal to this tool).
//! - Per-core policy (this writes every online CPU the same way;
//!   amd-pstate shares policy across cores anyway).
//! - Boost on/off toggling (orthogonal knob, operator can
//!   pre-configure).

#![allow(unsafe_code)] // localised to libc::signal registration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

use modgrad_substrate::{Snapshot, online_cpus};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────
// Signals
// ─────────────────────────────────────────────────────────────

static STOPPING: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_sig: libc::c_int) {
    STOPPING.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    // `libc::signal` is the simplest SIGINT/SIGTERM hook and is
    // sufficient for a single-threaded control daemon. The handler
    // does exactly one atomic store and nothing else — trivially
    // async-signal-safe.
    //
    // SAFETY: `on_signal` has `extern "C" fn(c_int)` — matches the
    // `sighandler_t` type libc::signal expects. Called once at
    // startup; the handler function is a zero-size pointer with
    // static lifetime.
    unsafe {
        libc::signal(
            libc::SIGINT,
            on_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            on_signal as *const () as libc::sighandler_t,
        );
    }
}

// ─────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Args {
    target_temp_c: f32,
    hyst_c: f32,
    interval_ms: u64,
    step_khz: u64,
    min_freq_khz: Option<u64>,
    max_freq_khz: Option<u64>,
    dry_run: bool,
    verbose: bool,
}

const USAGE: &str = r"usage: substrate-therm [options]

options:
  --target-temp-c=F  target °C (default 80.0)
  --hyst-c=F         hysteresis in °C (default 2.0)
  --interval-ms=N    control tick period, default 100
  --step-khz=N       adjustment step in kHz, default 100000 (100 MHz)
  --min-freq-khz=N   floor on the scaling_max_freq we'll set;
                     default = hardware cpuinfo_min_freq
  --max-freq-khz=N   ceiling we'll allow back up to;
                     default = hardware cpuinfo_max_freq
  --dry-run          compute the target cap but do not write sysfs
  --verbose          print every non-idempotent cap change to stderr
  -h, --help
";

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut args = Args {
        target_temp_c: 80.0,
        hyst_c: 2.0,
        interval_ms: 100,
        step_khz: 100_000,
        min_freq_khz: None,
        max_freq_khz: None,
        dry_run: false,
        verbose: false,
    };
    for raw in &argv[1..] {
        let (flag, val) = raw
            .split_once('=')
            .map_or((raw.as_str(), None), |(k, v)| (k, Some(v)));
        match (flag, val) {
            ("-h" | "--help", _) => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            ("--target-temp-c", Some(v)) => {
                args.target_temp_c = v.parse().map_err(|e| format!("--target-temp-c: {e}"))?;
            }
            ("--hyst-c", Some(v)) => {
                args.hyst_c = v.parse().map_err(|e| format!("--hyst-c: {e}"))?;
                if args.hyst_c < 0.0 {
                    return Err("--hyst-c must be >= 0".to_owned());
                }
            }
            ("--interval-ms", Some(v)) => {
                args.interval_ms = v.parse().map_err(|e| format!("--interval-ms: {e}"))?;
                if args.interval_ms < 10 {
                    return Err("--interval-ms must be >= 10".to_owned());
                }
            }
            ("--step-khz", Some(v)) => {
                args.step_khz = v.parse().map_err(|e| format!("--step-khz: {e}"))?;
                if args.step_khz == 0 {
                    return Err("--step-khz must be > 0".to_owned());
                }
            }
            ("--min-freq-khz", Some(v)) => {
                args.min_freq_khz = Some(v.parse().map_err(|e| format!("--min-freq-khz: {e}"))?);
            }
            ("--max-freq-khz", Some(v)) => {
                args.max_freq_khz = Some(v.parse().map_err(|e| format!("--max-freq-khz: {e}"))?);
            }
            ("--dry-run", None) => args.dry_run = true,
            ("--verbose", None) => args.verbose = true,
            _ => return Err(format!("unknown flag `{raw}` (try --help)")),
        }
    }
    Ok(args)
}

// ─────────────────────────────────────────────────────────────
// Sysfs writers
// ─────────────────────────────────────────────────────────────

fn scaling_max_path(cpu: u32) -> PathBuf {
    PathBuf::from(format!(
        "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq"
    ))
}

fn cpuinfo_max_path(cpu: u32) -> PathBuf {
    PathBuf::from(format!(
        "/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"
    ))
}

fn cpuinfo_min_path(cpu: u32) -> PathBuf {
    PathBuf::from(format!(
        "/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_min_freq"
    ))
}

fn read_u64_path(p: &PathBuf) -> io::Result<u64> {
    let s = fs::read_to_string(p)?;
    s.trim()
        .parse::<u64>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn write_scaling_max(cpu: u32, khz: u64) -> io::Result<()> {
    fs::write(scaling_max_path(cpu), format!("{khz}"))
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("substrate-therm: {e}");
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    install_signal_handlers();

    let cpus = match online_cpus() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("substrate-therm: online_cpus: {e}");
            return ExitCode::from(1);
        }
    };
    if cpus.is_empty() {
        eprintln!("substrate-therm: no online cpus");
        return ExitCode::from(1);
    }

    // Baseline values we snapshot at startup so we can put the
    // system back the way we found it on clean shutdown. Kept per
    // CPU even though amd-pstate shares policy — keeps us correct
    // on any driver.
    let mut original: Vec<(u32, u64)> = Vec::with_capacity(cpus.len());
    for cpu in &cpus {
        match read_u64_path(&scaling_max_path(*cpu)) {
            Ok(v) => original.push((*cpu, v)),
            Err(e) => {
                eprintln!(
                    "substrate-therm: could not read cpu{cpu} scaling_max_freq: {e}",
                    cpu = cpu,
                );
                return ExitCode::from(1);
            }
        }
    }

    // Resolve the floor / ceiling against hardware. We read cpu0
    // only — amd-pstate policy is shared, and any sane host has
    // uniform cpuinfo_{min,max}_freq across cores.
    let hw_max = match read_u64_path(&cpuinfo_max_path(cpus[0])) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("substrate-therm: cpuinfo_max_freq: {e}");
            return ExitCode::from(1);
        }
    };
    let hw_min = match read_u64_path(&cpuinfo_min_path(cpus[0])) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("substrate-therm: cpuinfo_min_freq: {e}");
            return ExitCode::from(1);
        }
    };
    let floor = args.min_freq_khz.unwrap_or(hw_min);
    let ceiling = args.max_freq_khz.unwrap_or(hw_max);
    if floor >= ceiling {
        eprintln!("substrate-therm: min-freq-khz ({floor}) >= max-freq-khz ({ceiling})");
        return ExitCode::from(2);
    }

    // Sanity-check writability up front unless dry-run. Saves
    // running for an hour before finding out we couldn't write.
    if !args.dry_run {
        let probe = original[0].1;
        if let Err(e) = write_scaling_max(cpus[0], probe) {
            eprintln!(
                "substrate-therm: probe write to cpu0 scaling_max_freq failed: {e} (need root?)"
            );
            return ExitCode::from(13); // EACCES
        }
    }

    eprintln!(
        "substrate-therm: target={}°C hyst={}°C step={} kHz floor={} kHz ceiling={} kHz cpus={} {}",
        args.target_temp_c,
        args.hyst_c,
        args.step_khz,
        floor,
        ceiling,
        cpus.len(),
        if args.dry_run { "(dry-run)" } else { "" }
    );

    let interval = Duration::from_millis(args.interval_ms);
    // Track the cap we last wrote so we can emit write syscalls
    // only when the target actually changes — avoids hammering
    // sysfs every tick in steady state.
    let mut last_written_cap: u64 = original[0].1;

    let exit_code = control_loop(
        &args,
        &cpus,
        floor,
        ceiling,
        interval,
        &mut last_written_cap,
    );

    // Always restore on the way out. Skipped on --dry-run because
    // we never actually wrote anything.
    if !args.dry_run {
        for (cpu, orig) in &original {
            if let Err(e) = write_scaling_max(*cpu, *orig) {
                eprintln!("substrate-therm: restore cpu{cpu}: {e}");
            }
        }
        eprintln!("substrate-therm: restored original scaling_max_freq across {} cpus", original.len());
    }
    exit_code
}

fn control_loop(
    args: &Args,
    cpus: &[u32],
    floor: u64,
    ceiling: u64,
    interval: Duration,
    last_written_cap: &mut u64,
) -> ExitCode {
    let stderr = io::stderr();
    while !STOPPING.load(Ordering::SeqCst) {
        let tick_start = Instant::now();

        let snap = match Snapshot::take() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("substrate-therm: snapshot failed: {e}");
                return ExitCode::from(1);
            }
        };
        let Some(temp_c) = snap.max_temp_c() else {
            eprintln!(
                "substrate-therm: no CPU temp sensor readable — refusing to act. Check /sys/class/hwmon."
            );
            return ExitCode::from(1);
        };

        let high = args.target_temp_c + args.hyst_c;
        let low = args.target_temp_c - args.hyst_c;

        let mut target_cap = *last_written_cap;
        if temp_c > high {
            target_cap = last_written_cap.saturating_sub(args.step_khz);
        } else if temp_c < low {
            target_cap = last_written_cap.saturating_add(args.step_khz);
        }
        target_cap = target_cap.clamp(floor, ceiling);

        if target_cap != *last_written_cap {
            if args.verbose || args.dry_run {
                let mut w = stderr.lock();
                let _ = writeln!(
                    w,
                    "[therm] temp={temp_c:.1}°C cap {prev}→{new} kHz ({delta:+} kHz) {dr}",
                    prev = *last_written_cap,
                    new = target_cap,
                    delta = target_cap as i64 - *last_written_cap as i64,
                    dr = if args.dry_run { "(dry)" } else { "" },
                );
            }
            if !args.dry_run {
                for cpu in cpus {
                    if let Err(e) = write_scaling_max(*cpu, target_cap) {
                        eprintln!("substrate-therm: write cpu{cpu}: {e}");
                        return ExitCode::from(1);
                    }
                }
            }
            *last_written_cap = target_cap;
        }

        // Sleep until next tick. The loop body itself is cheap
        // (sysfs reads + at most one small write), so the period
        // approximates `interval` closely.
        let elapsed = tick_start.elapsed();
        if elapsed < interval {
            std::thread::sleep(interval - elapsed);
        }
    }
    eprintln!("substrate-therm: stop signal received");
    ExitCode::SUCCESS
}
