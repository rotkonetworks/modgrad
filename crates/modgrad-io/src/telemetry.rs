//! Zero-overhead neural telemetry with configurable detail levels.
//!
//! Three detail tiers:
//!   Summary  — 8 region magnitudes + 6 signals + loss (~68 bytes/tick)
//!              Scales to any network size. Always affordable.
//!   Regional — Summary + per-neuron activations for selected regions
//!              User picks which regions to inspect in detail.
//!   Full     — All per-neuron activations for all regions
//!              High bandwidth but complete. For offline analysis.
//!
//! The debugger reconstructs derived quantities (sync pairs, deltas,
//! spike detection) from the raw activations at display time.
//! The stream only carries non-reproducible state.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::sync::{Arc, Mutex};

// ─── Detail Level ──────────────────────────────────────────

/// How much data to stream per tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetailLevel {
    /// Region magnitudes + signals only. ~68 bytes/tick. Scales to any size.
    Summary,
    /// Summary + full activations for selected regions.
    Regional,
    /// All per-neuron activations for all regions. Full fidelity.
    Full,
}

impl Default for DetailLevel {
    fn default() -> Self { DetailLevel::Summary }
}

// ─── Manifest ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: u32,
    pub regions: Vec<RegionSchema>,
    pub signals: Vec<SignalSchema>,
    pub connections: Vec<ConnectionSchema>,
    pub extras: Vec<ExtraSchema>,
    pub detail_level: DetailLevel,
    /// Floats per tick record (depends on detail level).
    pub record_floats: usize,
    /// Sync pair indices (so debugger can recompute sync from activations).
    pub sync_left: Vec<usize>,
    pub sync_right: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionSchema {
    pub id: String,
    pub neurons: usize,
    /// Offset of this region's activations in the Full record.
    pub offset: usize,
    pub color: String,
    pub position: [f32; 3],
    /// Whether this region's per-neuron data is streamed (Regional mode).
    #[serde(default)]
    pub stream_neurons: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalSchema {
    pub id: String,
    pub label: String,
    pub range: [f32; 2],
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionSchema {
    pub from: String,
    pub to: String,
    pub synapse: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraSchema {
    pub id: String,
    pub kind: String,
    pub dims: usize,
    pub offset: usize,
}

impl Manifest {
    /// Build manifest for the 8-region CTM at given detail level.
    pub fn for_ctm(
        region_sizes: &[(&str, usize)],
        n_signals: usize,
        sync_left: Vec<usize>,
        sync_right: Vec<usize>,
        detail_level: DetailLevel,
    ) -> Self {
        let colors = ["#4488ff", "#44ff88", "#ff8844", "#ff4488",
                       "#8844ff", "#88ff44", "#ff88ff", "#ffff44"];
        let positions = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
            [0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [2.0, -1.0, 0.0], [3.0, -1.0, 0.0],
        ];

        let mut regions = Vec::new();
        let mut offset = 0;
        for (i, &(name, n)) in region_sizes.iter().enumerate() {
            regions.push(RegionSchema {
                id: name.to_string(),
                neurons: n,
                offset,
                color: colors.get(i).unwrap_or(&"#888888").to_string(),
                position: *positions.get(i).unwrap_or(&[0.0, 0.0, 0.0]),
                stream_neurons: matches!(detail_level, DetailLevel::Full),
            });
            offset += n;
        }
        let total_neurons: usize = region_sizes.iter().map(|r| r.1).sum();
        let n_regions = region_sizes.len();

        let signals = vec![
            SignalSchema { id: "sync_scale".into(), label: "Dopamine".into(), range: [0.0, 1.0], color: "#ff0000".into() },
            SignalSchema { id: "gate".into(), label: "Serotonin".into(), range: [0.0, 1.0], color: "#00ff00".into() },
            SignalSchema { id: "arousal".into(), label: "NE".into(), range: [0.0, 3.0], color: "#0000ff".into() },
            SignalSchema { id: "precision".into(), label: "ACh".into(), range: [0.0, 1.0], color: "#ffff00".into() },
            SignalSchema { id: "curiosity".into(), label: "Curiosity".into(), range: [0.0, 1.0], color: "#00ffff".into() },
            SignalSchema { id: "anxiety".into(), label: "Anxiety".into(), range: [0.0, 1.0], color: "#ff00ff".into() },
        ];

        let connections = vec![
            ConnectionSchema { from: "motor".into(), to: "input".into(), synapse: "syn_motor_input".into() },
            ConnectionSchema { from: "input".into(), to: "attention".into(), synapse: "syn_input_attn".into() },
            ConnectionSchema { from: "attention".into(), to: "output".into(), synapse: "syn_attn_output".into() },
            ConnectionSchema { from: "output".into(), to: "motor".into(), synapse: "syn_output_motor".into() },
            ConnectionSchema { from: "motor".into(), to: "cerebellum".into(), synapse: "syn_cerebellum".into() },
            ConnectionSchema { from: "output".into(), to: "basal_ganglia".into(), synapse: "syn_basal_ganglia".into() },
            ConnectionSchema { from: "insula".into(), to: "attention".into(), synapse: "syn_insula".into() },
            ConnectionSchema { from: "input".into(), to: "hippocampus".into(), synapse: "syn_hippocampus".into() },
        ];

        // Record size depends on detail level:
        // Header: tick(1) + step(1) + loss(1) = 3
        // Summary: region_magnitudes(n_regions) + signals(n_signals) = n_regions + n_signals
        // Full: all activations(total_neurons) + signals(n_signals)
        let record_floats = match detail_level {
            DetailLevel::Summary => 3 + n_regions + n_signals,
            DetailLevel::Regional => {
                // Summary + selected region neurons (marked stream_neurons=true)
                let streamed: usize = regions.iter()
                    .filter(|r| r.stream_neurons)
                    .map(|r| r.neurons)
                    .sum();
                3 + n_regions + n_signals + streamed
            }
            DetailLevel::Full => 3 + total_neurons + n_signals,
        };

        Self {
            name: "isis-8region".into(),
            version: 2,
            regions,
            signals,
            connections,
            extras: Vec::new(),
            detail_level,
            record_floats,
            sync_left,
            sync_right,
        }
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| format!("{e}"))?;
        std::fs::write(path, json).map_err(|e| format!("{e}"))
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("{e}"))?;
        serde_json::from_str(&json).map_err(|e| format!("{e}"))
    }
}

// ─── Telemetry Recorder ────────────────────────────────────

impl std::fmt::Debug for Telemetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Telemetry")
            .field("detail", &self.manifest.detail_level)
            .field("capacity", &self.capacity)
            .field("total_ticks", &self.total_ticks)
            .finish()
    }
}

impl Clone for Telemetry {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            record_size: self.record_size,
            write_pos: self.write_pos,
            capacity: self.capacity,
            total_ticks: self.total_ticks,
            output: None,
            manifest: self.manifest.clone(),
        }
    }
}

pub struct Telemetry {
    buffer: Vec<f32>,
    record_size: usize,
    write_pos: usize,
    capacity: usize,
    total_ticks: u64,
    output: Option<Arc<Mutex<std::fs::File>>>,
    pub manifest: Manifest,
}

impl Telemetry {
    pub fn new(manifest: Manifest, capacity: usize) -> Self {
        let record_size = manifest.record_floats;
        Self {
            buffer: vec![0.0f32; record_size * capacity],
            record_size,
            write_pos: 0,
            capacity,
            total_ticks: 0,
            output: None,
            manifest,
        }
    }

    pub fn start_streaming(&mut self, path: &str) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| format!("{e}"))?;
        self.output = Some(Arc::new(Mutex::new(file)));
        let manifest_path = path.replace(".bin", ".manifest.json");
        self.manifest.save(&manifest_path)?;
        Ok(())
    }

    /// Record a Summary-level tick: region magnitudes + signals.
    /// Use this for low-bandwidth streaming.
    pub fn record_summary(
        &mut self,
        tick: u32,
        step: u64,
        loss: f32,
        region_magnitudes: &[f32],
        signals: &[f32],
    ) {
        let mut record = Vec::with_capacity(self.record_size);
        record.push(tick as f32);
        record.push(step as f32);
        record.push(loss);
        record.extend_from_slice(region_magnitudes);
        record.extend_from_slice(signals);
        while record.len() < self.record_size { record.push(0.0); }
        record.truncate(self.record_size);
        self.write_record(&record);
    }

    /// Record a Full tick: all per-neuron activations + signals.
    pub fn record_full(
        &mut self,
        tick: u32,
        step: u64,
        loss: f32,
        activations: &[f32],
        signals: &[f32],
    ) {
        let mut record = Vec::with_capacity(self.record_size);
        record.push(tick as f32);
        record.push(step as f32);
        record.push(loss);
        record.extend_from_slice(activations);
        record.extend_from_slice(signals);
        while record.len() < self.record_size { record.push(0.0); }
        record.truncate(self.record_size);
        self.write_record(&record);
    }

    /// Raw record write (for backward compat).
    #[inline]
    pub fn record_tick(&mut self, data: &[f32]) {
        self.write_record(data);
    }

    #[inline]
    fn write_record(&mut self, data: &[f32]) {
        if data.len() > self.record_size { return; }

        let offset = self.write_pos * self.record_size;
        let len = data.len().min(self.record_size);
        self.buffer[offset..offset + len].copy_from_slice(&data[..len]);
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.total_ticks += 1;

        if let Some(ref file) = self.output {
            if let Ok(mut f) = file.try_lock() {
                // Pad to record_size for consistent file layout
                let mut padded = vec![0.0f32; self.record_size];
                padded[..len].copy_from_slice(&data[..len]);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        padded.as_ptr() as *const u8,
                        self.record_size * 4,
                    )
                };
                f.write_all(bytes).ok();
            }
        }
    }

    /// Build record from components (backward compat helper).
    pub fn build_record(
        &self,
        tick: u32, step: u64, loss: f32,
        activations: &[f32], signals: &[f32],
        _sync: &[f32], _extras: &[f32],
    ) -> Vec<f32> {
        match self.manifest.detail_level {
            DetailLevel::Summary => {
                // Compute region magnitudes from activations
                let mut record = Vec::with_capacity(self.record_size);
                record.push(tick as f32);
                record.push(step as f32);
                record.push(loss);
                for region in &self.manifest.regions {
                    let start = region.offset;
                    let end = (start + region.neurons).min(activations.len());
                    if start < activations.len() {
                        let mag: f32 = activations[start..end].iter()
                            .map(|x| x * x).sum::<f32>().sqrt();
                        record.push(mag);
                    } else {
                        record.push(0.0);
                    }
                }
                record.extend_from_slice(signals);
                while record.len() < self.record_size { record.push(0.0); }
                record.truncate(self.record_size);
                record
            }
            DetailLevel::Full | DetailLevel::Regional => {
                let mut record = Vec::with_capacity(self.record_size);
                record.push(tick as f32);
                record.push(step as f32);
                record.push(loss);
                record.extend_from_slice(activations);
                record.extend_from_slice(signals);
                while record.len() < self.record_size { record.push(0.0); }
                record.truncate(self.record_size);
                record
            }
        }
    }

    pub fn recent_ticks(&self, n: usize) -> Vec<&[f32]> {
        let n = n.min(self.capacity).min(self.total_ticks as usize);
        let mut ticks = Vec::with_capacity(n);
        for i in 0..n {
            let pos = if self.write_pos >= n {
                self.write_pos - n + i
            } else {
                (self.capacity + self.write_pos - n + i) % self.capacity
            };
            let offset = pos * self.record_size;
            ticks.push(&self.buffer[offset..offset + self.record_size]);
        }
        ticks
    }

    pub fn total_ticks(&self) -> u64 { self.total_ticks }

    /// Change detail level at runtime. Adjusts record size.
    /// Called when debugger requests more/less detail.
    /// The stream file will have mixed record sizes after this point —
    /// the manifest at file start reflects the INITIAL level.
    /// For clean transitions, flush and start a new stream file.
    pub fn set_detail_level(&mut self, level: DetailLevel) {
        self.manifest.detail_level = level;
        let n_regions = self.manifest.regions.len();
        let n_signals = self.manifest.signals.len();
        let total_neurons: usize = self.manifest.regions.iter().map(|r| r.neurons).sum();

        self.manifest.record_floats = match level {
            DetailLevel::Summary => 3 + n_regions + n_signals,
            DetailLevel::Regional => {
                let streamed: usize = self.manifest.regions.iter()
                    .filter(|r| r.stream_neurons)
                    .map(|r| r.neurons).sum();
                3 + n_regions + n_signals + streamed
            }
            DetailLevel::Full => 3 + total_neurons + n_signals,
        };
        self.record_size = self.manifest.record_floats;
        // Reallocate buffer for new record size
        self.buffer = vec![0.0f32; self.record_size * self.capacity];
        self.write_pos = 0;
    }

    /// Toggle extra metric collection for a specific region.
    pub fn toggle_extra(&mut self, region_id: &str, enable: bool) {
        self.stream_region(region_id, enable);
    }

    /// Enable per-neuron streaming for a specific region (Regional mode).
    pub fn stream_region(&mut self, region_id: &str, enable: bool) {
        for r in &mut self.manifest.regions {
            if r.id == region_id {
                r.stream_neurons = enable;
            }
        }
        // Recalculate record size if in Regional mode
        if self.manifest.detail_level == DetailLevel::Regional {
            self.set_detail_level(DetailLevel::Regional);
        }
    }
}
