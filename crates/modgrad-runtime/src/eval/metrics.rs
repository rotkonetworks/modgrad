//! Metrics: structured telemetry for monitoring organism state.
//!
//! Collects all observable signals into a single snapshot that can be:
//! - Printed as human-readable text
//! - Serialized as JSON for dashboards
//! - Exported as Prometheus exposition format
//!
//! Every forward pass, sleep cycle, and state change produces metrics.

use serde::Serialize;

/// Complete snapshot of organism state at a point in time.
#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    pub timestamp: f64,

    // Training
    pub step: u64,
    pub tokens_seen: u64,
    pub sleep_cycles: u64,
    pub loss: f32,
    pub alignment: f32,
    pub tokens_per_sec: f32,

    // Homeostasis
    pub pressure: f32,
    pub output_quality: f32,
    pub activation_pressure: f32,
    pub divergence_pressure: f32,
    pub drift_pressure: f32,
    pub buffer_pressure: f32,
    pub emotional_pressure: f32,
    pub surprise_ema: f32,

    // Emotional health
    pub health_score: f32,
    pub fear_ratio: f32,
    pub negative_ratio: f32,
    pub avoidance_generalization: f32,
    pub total_memories: u32,
    pub fear_memories: u32,

    // CTM
    pub ctm_confidence: f32,
    pub dead_neurons: u32,
    pub total_neurons: u32,

    // Angeris bounds
    pub synapse_gaps: Vec<(String, f32)>,

    // Sensory
    pub sensory_recovery_pct: f32,

    // Diagnoses (empty = healthy)
    pub diagnoses: Vec<String>,
}

impl Snapshot {
    pub fn now() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }
}

/// Prometheus exposition format.
impl Snapshot {
    pub fn to_prometheus(&self) -> String {
        let mut out = String::new();
        let ts = (self.timestamp * 1000.0) as u64; // ms

        macro_rules! gauge {
            ($name:expr, $val:expr, $help:expr) => {
                out.push_str(&format!("# HELP isis_{} {}\n# TYPE isis_{} gauge\nisis_{} {} {}\n",
                    $name, $help, $name, $name, $val, ts));
            };
        }

        gauge!("step", self.step, "Training step");
        gauge!("tokens_seen", self.tokens_seen, "Total tokens processed");
        gauge!("sleep_cycles", self.sleep_cycles, "Sleep cycles completed");
        gauge!("loss", self.loss, "Current training loss");
        gauge!("alignment", self.alignment, "Parent-child alignment score");
        gauge!("tokens_per_sec", self.tokens_per_sec, "Processing throughput");

        gauge!("pressure", self.pressure, "Sleep pressure (0=rested, 1=must sleep)");
        gauge!("output_quality", self.output_quality, "Output quality estimate");
        gauge!("activation_pressure", self.activation_pressure, "Neural activation pressure");
        gauge!("divergence_pressure", self.divergence_pressure, "Sync divergence pressure");
        gauge!("drift_pressure", self.drift_pressure, "Hebbian drift pressure");
        gauge!("buffer_pressure", self.buffer_pressure, "Buffer fill pressure");
        gauge!("emotional_pressure", self.emotional_pressure, "Emotional processing pressure");
        gauge!("surprise_ema", self.surprise_ema, "Surprise moving average");

        gauge!("health_score", self.health_score, "Emotional health (0=pathological, 1=healthy)");
        gauge!("fear_ratio", self.fear_ratio, "Fraction of fear-valenced memories");
        gauge!("negative_ratio", self.negative_ratio, "Fraction of negative/fear memories");
        gauge!("avoidance_generalization", self.avoidance_generalization, "Avoidance pattern clustering");
        gauge!("total_memories", self.total_memories, "Total episodic memories");
        gauge!("fear_memories", self.fear_memories, "Fear-valenced memories");

        gauge!("ctm_confidence", self.ctm_confidence, "CTM sync convergence confidence");
        gauge!("dead_neurons", self.dead_neurons, "Dead neurons across all regions");
        gauge!("total_neurons", self.total_neurons, "Total neurons");

        gauge!("sensory_recovery_pct", self.sensory_recovery_pct, "Percent of parent hidden linearly recoverable");

        for (name, gap) in &self.synapse_gaps {
            out.push_str(&format!("isis_synapse_gap{{synapse=\"{}\"}} {} {}\n", name, gap, ts));
        }

        for (i, d) in self.diagnoses.iter().enumerate() {
            out.push_str(&format!("isis_diagnosis{{index=\"{}\",diagnosis=\"{}\"}} 1 {}\n", i, d, ts));
        }

        out
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Metrics collector. Call push() after each step, flush() to write.
pub struct Collector {
    snapshots: Vec<Snapshot>,
    /// Where to write: file path, or None for stderr only.
    output_path: Option<String>,
    /// Write every N snapshots.
    flush_every: usize,
}

impl Collector {
    pub fn new(output_path: Option<String>, flush_every: usize) -> Self {
        Self { snapshots: Vec::new(), output_path, flush_every }
    }

    pub fn push(&mut self, snapshot: Snapshot) {
        self.snapshots.push(snapshot);
        if self.snapshots.len() >= self.flush_every {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if self.snapshots.is_empty() { return; }

        if let Some(ref path) = self.output_path {
            if path.ends_with(".jsonl") {
                // JSON Lines: one JSON object per line
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true).append(true).open(path)
                {
                    for s in &self.snapshots {
                        writeln!(f, "{}", s.to_json()).ok();
                    }
                }
            } else if path.ends_with(".prom") {
                // Prometheus: overwrite with latest snapshot
                if let Some(last) = self.snapshots.last() {
                    std::fs::write(path, last.to_prometheus()).ok();
                }
            }
        }

        self.snapshots.clear();
    }

    pub fn latest(&self) -> Option<&Snapshot> {
        self.snapshots.last()
    }
}
