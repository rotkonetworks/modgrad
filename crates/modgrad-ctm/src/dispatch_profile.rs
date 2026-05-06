//! Per-dispatch-type timing for the CTM forward path.
//!
//! Phase 0 instrumentation for the brain-on-GPU plan: count and time
//! each matvec/matmul dispatch site in `regional_forward` /
//! `ctm_forward_with_kv` so we know which buckets dominate before
//! investing in fusion (Phase 1) or block-diagonal batching (Phase 2).
//!
//! Activated by env var `MODGRAD_PROFILE_DISPATCH=1`. Off by default —
//! when off, every `Guard` constructor is a single atomic-load and the
//! Drop is a no-op, so production paths pay essentially nothing.
//!
//! # Usage
//!
//! ```ignore
//! use modgrad_ctm::dispatch_profile::{Guard, DispatchKind};
//! let _g = Guard::new(DispatchKind::KvProj);
//! kv_proj.forward(...);
//! drop(_g); // records elapsed
//! ```
//!
//! At the end of a benchmarked run call [`dump`] to print the table.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::Instant;

#[derive(Copy, Clone, Debug)]
pub enum DispatchKind {
    // Outer (regional_forward) buckets
    ObsProj      = 0,
    ConnSynapse  = 1,
    OutputProj   = 2,
    OuterExitGate = 3,
    Router       = 4,

    // Inner (ctm_forward_with_kv) buckets
    KvProj       = 5,
    QProj        = 6,
    MhaIn        = 7,
    MhaOut       = 8,
    Synapse      = 9,
    NlmS1        = 10,
    NlmS2        = 11,
    OutProjRegion = 12,
    ExitGateRegion = 13,

    // Backward-pass buckets — sub-sections of the regional backward.
    BwdOutputProj   = 14, // output_proj backward (host registry: outer_product_acc + matvec_t)
    BwdGlobalSync   = 15, // global_sync_backward (host scalar)
    BwdInnerCtm     = 16, // per-region inner CTM backward (incl. U-Net)
    BwdConnSynapse  = 17, // connection synapse backward (host scalar triple loop)
    BwdAccumGrads   = 18, // copying per-region backward result into RegionalGradients

    // Inner backward sub-buckets — what's hot inside per-region CTM backward.
    BwdInnerSyncOut    = 19,
    BwdInnerNlm        = 20,
    BwdInnerUnet       = 21, // Path B host re-cache + unet_backward
    BwdInnerMha        = 22,
    BwdInnerQProj      = 23,
    BwdInnerSyncAction = 24,
    BwdInnerKvProj     = 25,
}

const N_KINDS: usize = 26;
const NAMES: [&str; N_KINDS] = [
    "obs_proj", "conn_synapse", "output_proj", "outer_exit_gate", "router",
    "kv_proj", "q_proj", "mha_in", "mha_out", "synapse",
    "nlm_s1", "nlm_s2", "out_proj_region", "exit_gate_region",
    "BWD_output_proj", "BWD_global_sync", "BWD_inner_ctm",
    "BWD_conn_synapse", "BWD_accum_grads",
    "BWDi_sync_out", "BWDi_nlm", "BWDi_unet",
    "BWDi_mha", "BWDi_q_proj", "BWDi_sync_action", "BWDi_kv_proj",
];

// 0 = uninitialised, 1 = enabled, 2 = disabled.
static ENABLED: AtomicU8 = AtomicU8::new(0);

// Per-kind: (count, total_ns). Flat array of 2N atomics. We need
// 2 × 26 = 52 entries; use the const-block init pattern (Rust 1.79+)
// because AtomicU64 doesn't implement Copy.
static COUNTERS: [AtomicU64; N_KINDS * 2] = [const { AtomicU64::new(0) }; N_KINDS * 2];

#[inline]
pub fn enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var_os("MODGRAD_PROFILE_DISPATCH").is_some();
            ENABLED.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

/// RAII guard. Construction starts the timer (when enabled); Drop
/// records the elapsed time into the matching kind's accumulator.
pub struct Guard {
    kind: DispatchKind,
    start: Option<Instant>,
}

impl Guard {
    #[inline]
    pub fn new(kind: DispatchKind) -> Self {
        Self {
            kind,
            start: if enabled() { Some(Instant::now()) } else { None },
        }
    }
}

impl Drop for Guard {
    #[inline]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let ns = start.elapsed().as_nanos() as u64;
            let i = self.kind as usize;
            COUNTERS[i * 2].fetch_add(1, Ordering::Relaxed);
            COUNTERS[i * 2 + 1].fetch_add(ns, Ordering::Relaxed);
        }
    }
}

/// Print the table of (kind, count, total_ms, avg_µs) to stderr,
/// sorted by total time descending. Prints nothing if profiling is
/// disabled or no dispatches were recorded.
pub fn dump() {
    if !enabled() { return; }
    let mut rows: Vec<(usize, u64, u64)> = (0..N_KINDS)
        .map(|i| (
            i,
            COUNTERS[i * 2].load(Ordering::Relaxed),
            COUNTERS[i * 2 + 1].load(Ordering::Relaxed),
        ))
        .filter(|(_, c, _)| *c > 0)
        .collect();
    if rows.is_empty() {
        eprintln!("[dispatch_profile] no dispatches recorded");
        return;
    }
    rows.sort_by(|a, b| b.2.cmp(&a.2));
    let total_ns: u64 = rows.iter().map(|r| r.2).sum();
    eprintln!("[dispatch_profile] total recorded time: {:.3} ms across {} kinds",
              total_ns as f64 / 1e6, rows.len());
    eprintln!("  {:<18}  {:>7}  {:>10}  {:>10}  {:>5}",
              "kind", "count", "total_ms", "avg_µs", "%");
    for (i, count, ns) in rows {
        let total_ms = ns as f64 / 1e6;
        let avg_us = ns as f64 / count.max(1) as f64 / 1000.0;
        let pct = 100.0 * ns as f64 / total_ns.max(1) as f64;
        eprintln!("  {:<18}  {:>7}  {:>10.3}  {:>10.2}  {:>4.1}%",
                  NAMES[i], count, total_ms, avg_us, pct);
    }
}

/// Zero all accumulators (useful between benchmark phases).
pub fn reset() {
    for c in COUNTERS.iter() { c.store(0, Ordering::Relaxed); }
}
