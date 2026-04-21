//! Substrate observability — read-only sensor snapshots of the host
//! CPU the modgrad training loop is orchestrating on.
//!
//! ## Why this crate exists
//!
//! modgrad training at realistic shapes is CPU-orchestration-bound:
//! a single training step issues tens of thousands of GPU dispatches,
//! each a stack of host-side driver calls. When the CPU frequency
//! governor clock-caps in response to temperature, the whole pipeline
//! serialises at the lower clock and the GPU goes idle waiting.
//!
//! So "was this step throttled" is a first-class piece of context
//! the training loop may want. This crate exposes the kernel-side
//! truth via sysfs, no root required.
//!
//! ## What this crate is NOT
//!
//! This is a read-only observability surface. Writing to governors,
//! setting scaling_setspeed, or touching RAPL caps are all explicitly
//! out of scope for v1. Adding a policy layer would require root /
//! CAP_SYS_ADMIN and a much stronger correctness story — to be done
//! as `modgrad-substrate-policy` when there is a concrete caller.
//!
//! ## Design
//!
//! - Every sysfs read returns `Result<T, SubstrateError>`. No
//!   `.unwrap()`, no `.expect()`, no `panic!()` on kernel interface
//!   surfaces — a missing file or a reparsed value surfaces as a
//!   structured error, never a crash.
//! - Paths are explicit constants. Nothing glob-walks past a handful
//!   of well-known sysfs locations. Minimise attack surface.
//! - No external dependencies. Only `std`. Keeps the build fast and
//!   the audit surface small.
//! - Linux-only. Non-Linux targets get `Unsupported` errors, not
//!   undefined-behaviour fallbacks.
//! - Numerical parsing uses `u64::from_str` with explicit error
//!   conversion. No lenient "try harder" parsing — if a sysfs value
//!   doesn't match the expected shape, callers should know.

#![deny(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::similar_names)] // `lo_s`/`hi_s` etc are intentional
#![allow(clippy::doc_markdown)]  // "NOT"/"CAP_SYS_ADMIN" in prose

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

// ─────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────

/// Every fallible call in this crate returns a `SubstrateError` on
/// the failure path. Deliberately non-`Clone` because it carries
/// `io::Error`.
#[derive(Debug)]
pub enum SubstrateError {
    /// `read`, `open`, `metadata` failed at the named path.
    Io { path: PathBuf, source: io::Error },
    /// A sysfs file existed but its contents did not parse to the
    /// shape the kernel ABI documents (u64, label string, CPU-list,
    /// etc). Includes the raw bytes so callers can log.
    Parse { path: PathBuf, raw: String, reason: String },
    /// The sysfs entry we expected is not present on this host or
    /// not exposed to the current user. Distinct from `Io` so
    /// callers can decide whether to degrade gracefully.
    Missing { path: PathBuf },
    /// We are running on a platform where the Linux sysfs layout
    /// this crate expects does not apply.
    Unsupported,
}

impl fmt::Display for SubstrateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "io error reading {}: {source}", path.display())
            }
            Self::Parse { path, raw, reason } => {
                write!(
                    f,
                    "parse error at {}: {reason} (raw: {raw:?})",
                    path.display()
                )
            }
            Self::Missing { path } => {
                write!(f, "sensor path does not exist: {}", path.display())
            }
            Self::Unsupported => write!(f, "substrate sensors not supported on this platform"),
        }
    }
}

impl std::error::Error for SubstrateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, SubstrateError>;

// ─────────────────────────────────────────────────────────────
// Low-level IO helpers — the only code in this crate that touches
// the filesystem. Both functions classify NotFound as `Missing`
// rather than bundling it inside `Io`, so callers can match on it.
// ─────────────────────────────────────────────────────────────

fn read_trimmed(path: &Path) -> Result<String> {
    match fs::read_to_string(path) {
        Ok(s) => Ok(s.trim().to_owned()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            Err(SubstrateError::Missing { path: path.to_owned() })
        }
        Err(err) => Err(SubstrateError::Io { path: path.to_owned(), source: err }),
    }
}

fn read_u64(path: &Path) -> Result<u64> {
    let raw = read_trimmed(path)?;
    raw.parse::<u64>().map_err(|err| SubstrateError::Parse {
        path: path.to_owned(),
        raw: raw.clone(),
        reason: format!("not a u64: {err}"),
    })
}

// ─────────────────────────────────────────────────────────────
// CPU topology + per-core observability
// ─────────────────────────────────────────────────────────────

const CPU_ONLINE: &str = "/sys/devices/system/cpu/online";

/// Parse the canonical Linux CPU-list format: comma-separated ranges
/// like `0-3,6,8-11`. Returns `Err(Parse)` on anything malformed
/// rather than silently dropping bad tokens.
fn parse_cpu_list(raw: &str, path: &Path) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for segment in raw.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        if let Some((lo_s, hi_s)) = segment.split_once('-') {
            let lo: u32 = lo_s.trim().parse().map_err(|err| SubstrateError::Parse {
                path: path.to_owned(),
                raw: raw.to_owned(),
                reason: format!("bad range lo `{lo_s}`: {err}"),
            })?;
            let hi: u32 = hi_s.trim().parse().map_err(|err| SubstrateError::Parse {
                path: path.to_owned(),
                raw: raw.to_owned(),
                reason: format!("bad range hi `{hi_s}`: {err}"),
            })?;
            if hi < lo {
                return Err(SubstrateError::Parse {
                    path: path.to_owned(),
                    raw: raw.to_owned(),
                    reason: format!("inverted range `{segment}`"),
                });
            }
            for cpu in lo..=hi {
                out.push(cpu);
            }
        } else {
            let cpu: u32 = segment.parse().map_err(|err| SubstrateError::Parse {
                path: path.to_owned(),
                raw: raw.to_owned(),
                reason: format!("bad single `{segment}`: {err}"),
            })?;
            out.push(cpu);
        }
    }
    Ok(out)
}

/// List the CPU indices Linux considers online right now. Reads
/// `/sys/devices/system/cpu/online` exactly once.
pub fn online_cpus() -> Result<Vec<u32>> {
    #[cfg(not(target_os = "linux"))]
    {
        return Err(SubstrateError::Unsupported);
    }
    #[cfg(target_os = "linux")]
    {
        let path = Path::new(CPU_ONLINE);
        let raw = read_trimmed(path)?;
        parse_cpu_list(&raw, path)
    }
}

fn cpu_attr_path(cpu: u32, tail: &str) -> PathBuf {
    PathBuf::from(format!("/sys/devices/system/cpu/cpu{cpu}/{tail}"))
}

/// Current clock of `cpu` in kHz. Source:
/// `cpufreq/scaling_cur_freq`.
pub fn cpu_cur_freq_khz(cpu: u32) -> Result<u64> {
    read_u64(&cpu_attr_path(cpu, "cpufreq/scaling_cur_freq"))
}

/// Policy ceiling for `cpu` in kHz. Source:
/// `cpufreq/scaling_max_freq`. The *governor* will not let the
/// clock rise past this. See [`cpu_hw_max_freq_khz`] for the hard
/// hardware ceiling.
pub fn cpu_max_freq_khz(cpu: u32) -> Result<u64> {
    read_u64(&cpu_attr_path(cpu, "cpufreq/scaling_max_freq"))
}

/// Hardware max for `cpu` in kHz. Source: `cpufreq/cpuinfo_max_freq`.
/// This is the silicon's actual boost ceiling — `scaling_max_freq`
/// is the (possibly lower) policy cap the user or BIOS imposes on
/// top.
///
/// The gap between `cpu_cur_freq_khz` and `cpu_hw_max_freq_khz` is
/// what "the CPU isn't running as fast as it could" actually means.
/// AMD SMU boost management can suppress clocks on a hot Ryzen
/// without ever incrementing `core_throttle_count`, so reading this
/// ratio is the most reliable indicator of boost-limit throttling.
pub fn cpu_hw_max_freq_khz(cpu: u32) -> Result<u64> {
    read_u64(&cpu_attr_path(cpu, "cpufreq/cpuinfo_max_freq"))
}

/// Kernel-reported monotonic counter of thermal throttle events on
/// this core. Take the delta across a training step to detect a
/// throttle window.
pub fn cpu_throttle_count(cpu: u32) -> Result<u64> {
    read_u64(&cpu_attr_path(cpu, "thermal_throttle/core_throttle_count"))
}

/// Active governor name for `cpu` (e.g. `"ondemand"`,
/// `"performance"`, `"schedutil"`).
pub fn cpu_governor(cpu: u32) -> Result<String> {
    read_trimmed(&cpu_attr_path(cpu, "cpufreq/scaling_governor"))
}

// ─────────────────────────────────────────────────────────────
// hwmon temperature sensors — only the CPU-relevant drivers
// ─────────────────────────────────────────────────────────────

const HWMON_ROOT: &str = "/sys/class/hwmon";

/// Driver names whose temp sensors we attribute to the CPU. Kept
/// allow-listed rather than regex-sniffed: a new driver should be
/// explicitly reviewed before this crate reports its numbers to the
/// training loop.
const CPU_TEMP_DRIVERS: &[&str] = &["k10temp", "coretemp", "zenpower", "zenergy"];

/// A single `tempN_input` file plus its human label (`tempN_label`
/// when present, else `<driver>/tempN`).
#[derive(Debug, Clone)]
pub struct TempSensor {
    pub driver: String,
    pub label: String,
    pub path: PathBuf,
}

impl TempSensor {
    /// Read the sensor. Sysfs reports millidegrees Celsius; we
    /// convert to float °C on the way out.
    pub fn celsius(&self) -> Result<f32> {
        let millideg = read_u64(&self.path)?;
        #[allow(clippy::cast_precision_loss)]
        Ok(millideg as f32 / 1000.0)
    }
}

/// Enumerate CPU-side temperature sensors under `/sys/class/hwmon`.
/// Never panics on a hwmon entry that is partly malformed — it is
/// simply skipped. The function returns an empty vector on a host
/// with no CPU-driver hwmons, not an error.
pub fn cpu_temp_sensors() -> Result<Vec<TempSensor>> {
    #[cfg(not(target_os = "linux"))]
    {
        return Err(SubstrateError::Unsupported);
    }
    #[cfg(target_os = "linux")]
    {
        // Enumerate temp1_input .. tempMAX_TEMPS_input per hwmon;
        // fixed upper bound so a malformed hwmon can never wedge us.
        const MAX_TEMPS: u32 = 64;

        let root = Path::new(HWMON_ROOT);
        let entries = match fs::read_dir(root) {
            Ok(e) => e,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                return Err(SubstrateError::Missing { path: root.to_owned() });
            }
            Err(err) => {
                return Err(SubstrateError::Io { path: root.to_owned(), source: err });
            }
        };

        let mut out = Vec::new();
        for entry in entries.flatten() {
            let hwmon_dir = entry.path();
            let Ok(driver) = read_trimmed(&hwmon_dir.join("name")) else {
                // missing or unreadable `name` — skip, don't fail
                continue;
            };
            if !CPU_TEMP_DRIVERS.contains(&driver.as_str()) {
                continue;
            }
            for i in 1..=MAX_TEMPS {
                let input = hwmon_dir.join(format!("temp{i}_input"));
                if !input.exists() {
                    // Canonical convention: absent temp<i>_input
                    // means no temp at this index. Stop scanning.
                    break;
                }
                let label = read_trimmed(&hwmon_dir.join(format!("temp{i}_label")))
                    .unwrap_or_else(|_| format!("{driver}/temp{i}"));
                out.push(TempSensor {
                    driver: driver.clone(),
                    label,
                    path: input,
                });
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────
// Snapshot — one consistent sample of everything above. Cheap to
// take (a handful of small file reads), designed to be called at
// training-step boundaries.
// ─────────────────────────────────────────────────────────────

/// One instant's worth of host observability. Constructed by
/// [`Snapshot::take`]; all fields are plain data so it's easy to
/// diff two snapshots or log one.
///
/// `governor` is read from cpu0 only — in practice all cpus on a
/// modern Linux share a governor policy, and reading N files to
/// confirm that would be paranoia without a payoff.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub taken_unix_ns: u128,
    pub cpu_freqs_khz: Vec<u64>,
    /// Hardware boost ceiling per core (`cpuinfo_max_freq`). Used
    /// as the denominator for boost-throttle ratio checks.
    pub cpu_hw_max_freqs_khz: Vec<u64>,
    pub cpu_throttle_counts: Vec<u64>,
    pub cpu_temps_c: Vec<(String, f32)>,
    pub governor: String,
}

impl Snapshot {
    /// Collect one sample. Individual sensor failures degrade
    /// gracefully: a failed per-core read contributes `0` and a
    /// failed temp read contributes `NaN`, but the overall call
    /// only returns `Err` if the top-level enumeration (online CPUs,
    /// hwmon root) fails.
    ///
    /// The rationale: the loss of one core's thermal counter is
    /// noise, but the loss of the whole sysfs interface is a
    /// substrate-level surprise the caller should handle explicitly.
    pub fn take() -> Result<Self> {
        let taken_unix_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        let cpus = online_cpus()?;

        let mut cpu_freqs_khz = Vec::with_capacity(cpus.len());
        let mut cpu_hw_max_freqs_khz = Vec::with_capacity(cpus.len());
        let mut cpu_throttle_counts = Vec::with_capacity(cpus.len());
        for cpu in &cpus {
            cpu_freqs_khz.push(cpu_cur_freq_khz(*cpu).unwrap_or(0));
            cpu_hw_max_freqs_khz.push(cpu_hw_max_freq_khz(*cpu).unwrap_or(0));
            cpu_throttle_counts.push(cpu_throttle_count(*cpu).unwrap_or(0));
        }

        let sensors = cpu_temp_sensors()?;
        let mut cpu_temps_c = Vec::with_capacity(sensors.len());
        for s in &sensors {
            let c = s.celsius().unwrap_or(f32::NAN);
            cpu_temps_c.push((s.label.clone(), c));
        }

        let governor = match cpus.first() {
            Some(cpu0) => cpu_governor(*cpu0).unwrap_or_else(|_| String::from("unknown")),
            None => String::from("no-cpu"),
        };

        Ok(Self {
            taken_unix_ns,
            cpu_freqs_khz,
            cpu_hw_max_freqs_khz,
            cpu_throttle_counts,
            cpu_temps_c,
            governor,
        })
    }

    /// Ratio of current clock to hardware max, averaged over all
    /// cores where both values are non-zero. Returns a number in
    /// `[0.0, 1.0]` (possibly slightly above 1.0 with XFR-style
    /// boost) that is **the single most useful indicator of boost
    /// throttling on AMD Ryzen** — PPT / TDC / temperature limits
    /// all show up here before they ever touch `core_throttle_count`.
    ///
    /// `None` when no core produced a usable pair.
    #[must_use]
    pub fn mean_freq_ratio(&self) -> Option<f32> {
        let pairs: Vec<(u64, u64)> = self
            .cpu_freqs_khz
            .iter()
            .zip(self.cpu_hw_max_freqs_khz.iter())
            .filter(|(cur, hw_max)| **cur > 0 && **hw_max > 0)
            .map(|(c, m)| (*c, *m))
            .collect();
        if pairs.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let ratios: f32 = pairs
            .iter()
            .map(|(c, m)| *c as f32 / *m as f32)
            .sum::<f32>()
            / pairs.len() as f32;
        Some(ratios)
    }

    /// Mean of `cpu_freqs_khz`, ignoring zero entries. `None` if
    /// every entry is zero (host with no readable cpufreq).
    #[must_use]
    pub fn mean_freq_khz(&self) -> Option<u64> {
        let nonzero: Vec<u64> = self.cpu_freqs_khz.iter().copied().filter(|f| *f > 0).collect();
        if nonzero.is_empty() {
            return None;
        }
        let sum: u128 = nonzero.iter().copied().map(u128::from).sum();
        #[allow(clippy::cast_possible_truncation)]
        Some((sum / nonzero.len() as u128) as u64)
    }

    /// Hottest CPU temperature sampled. `None` if no sensors
    /// produced a finite reading.
    #[must_use]
    pub fn max_temp_c(&self) -> Option<f32> {
        self.cpu_temps_c
            .iter()
            .map(|(_, t)| *t)
            .filter(|t| t.is_finite())
            .fold(None::<f32>, |acc, t| Some(acc.map_or(t, |a| a.max(t))))
    }

    /// Sum of throttle counters across all sampled cores. Per-core
    /// counts are monotonic, so `after.throttle_total() - before.throttle_total()`
    /// is the number of throttle events during the interval.
    #[must_use]
    pub fn throttle_total(&self) -> u64 {
        self.cpu_throttle_counts.iter().copied().sum()
    }
}

// ─────────────────────────────────────────────────────────────
// Tests. These are intentionally minimal — sysfs is a moving target
// per host, so unit tests target only the pure parsers.
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_list_single() {
        let p = Path::new("/dummy");
        assert_eq!(parse_cpu_list("0", p).unwrap(), vec![0]);
        assert_eq!(parse_cpu_list("7", p).unwrap(), vec![7]);
    }

    #[test]
    fn parse_cpu_list_range() {
        let p = Path::new("/dummy");
        assert_eq!(parse_cpu_list("0-3", p).unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn parse_cpu_list_mixed() {
        let p = Path::new("/dummy");
        assert_eq!(
            parse_cpu_list("0-3,6,8-10", p).unwrap(),
            vec![0, 1, 2, 3, 6, 8, 9, 10]
        );
    }

    #[test]
    fn parse_cpu_list_rejects_inverted_range() {
        let p = Path::new("/dummy");
        let err = parse_cpu_list("5-2", p).unwrap_err();
        match err {
            SubstrateError::Parse { reason, .. } => {
                assert!(reason.contains("inverted"), "unexpected reason: {reason}");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn parse_cpu_list_rejects_garbage() {
        let p = Path::new("/dummy");
        assert!(parse_cpu_list("0-x", p).is_err());
        assert!(parse_cpu_list("abc", p).is_err());
    }

    #[test]
    fn parse_cpu_list_tolerates_empty_segments() {
        // kernel rarely emits these, but `"0,"` shouldn't fail.
        let p = Path::new("/dummy");
        assert_eq!(parse_cpu_list("0,", p).unwrap(), vec![0]);
    }

    /// Best-effort: take a snapshot on whatever host runs `cargo
    /// test`. If sysfs isn't present (non-Linux, inside a sandbox
    /// that hides it), skip rather than fail — this is integration
    /// surface, not a unit.
    #[test]
    #[cfg(target_os = "linux")]
    fn snapshot_on_this_host() {
        match Snapshot::take() {
            Ok(s) => {
                assert!(
                    !s.cpu_freqs_khz.is_empty(),
                    "expected at least one online cpu"
                );
                assert_eq!(s.cpu_freqs_khz.len(), s.cpu_throttle_counts.len());
                // governor is either a known string or "unknown" — just sanity-check
                // it's not empty.
                assert!(!s.governor.is_empty());
            }
            Err(SubstrateError::Missing { .. }) | Err(SubstrateError::Unsupported) => {
                // Fine — no sysfs on this host / sandboxed build.
            }
            Err(other) => panic!("unexpected snapshot error: {other}"),
        }
    }
}
