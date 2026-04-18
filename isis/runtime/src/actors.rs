//! Actor-based regional CTM: each region runs in its own thread.
//!
//! Each region is a loop: read sources → tick → publish output.
//! No global synchronization barrier. Regions run at their own clock rate.
//! Fast regions (cerebellum) tick 3x per cortical tick.
//!
//! Communication: shared `Arc<Mutex<Vec<f32>>>` per region output.
//! Writers hold the lock only to swap in a new Vec (microseconds).
//! Readers clone the latest state at the start of their tick.
//!
//! For training: use the synchronous `regional_train_step`.
//! Actors are for inference and daemon mode only.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use modgrad_compute::neuron::Linear;
use modgrad_ctm::weights::{CtmWeights, CtmState};
use modgrad_ctm::forward::ctm_forward;

use super::regional::{RegionalConfig, RegionalWeights};

// ─── Published state ───────────────────────────────────────

/// One region's published output. Lock-free reads via brief Mutex.
#[derive(Clone)]
struct Slot {
    activated: Vec<f32>,
    tick: u64,
}

impl Slot {
    fn new(dim: usize) -> Self {
        Self { activated: vec![0.0; dim], tick: 0 }
    }
}

// ─── Region actor ──────────────────────────────────────────

/// Handle to a running region actor.
struct RegionHandle {
    thread: Option<JoinHandle<()>>,
    /// Region's published output — readers clone this.
    output: Arc<Mutex<Slot>>,
}

/// Configuration for one region actor.
struct RegionActorConfig {
    /// Which region index this is.
    index: usize,
    /// How many CTM ticks per actor step.
    ticks_per_step: usize,
    /// Source region indices for connection synapse.
    source_indices: Vec<usize>,
    /// Whether this region receives the external observation.
    receives_observation: bool,
}

// ─── Brain (the actor system) ──────────────────────────────

/// The actor-based brain. Each region runs in its own thread.
/// Call `set_observation` to feed input, `read_sync` to read output.
pub struct ActorBrain {
    /// Handles to region threads.
    regions: Vec<RegionHandle>,
    /// Shared observation input (set by caller, read by input-receiving regions).
    observation: Arc<Mutex<Vec<f32>>>,
    /// Stop signal.
    running: Arc<AtomicBool>,
    /// Global sync output — computed by collator.
    global_sync: Arc<Mutex<Vec<f32>>>,
    /// Global tick counter.
    global_tick: Arc<AtomicU64>,
    /// Collator thread.
    collator: Option<JoinHandle<()>>,
    /// Config for reading.
    #[allow(dead_code)] // retained so ActorBrain can serve config queries later
    config: RegionalConfig,
}

impl ActorBrain {
    /// Spawn all region actors and the global sync collator.
    pub fn spawn(w: &RegionalWeights) -> Self {
        let cfg = &w.config;
        let n_regions = cfg.regions.len();
        let running = Arc::new(AtomicBool::new(true));
        let observation = Arc::new(Mutex::new(vec![0.0f32; cfg.raw_obs_dim]));

        // Create output slots for each region
        let slots: Vec<Arc<Mutex<Slot>>> = cfg.regions.iter()
            .map(|r| Arc::new(Mutex::new(Slot::new(r.d_model))))
            .collect();

        // Build per-region actor configs from connection graph
        let mut actor_configs: Vec<RegionActorConfig> = (0..n_regions)
            .map(|i| RegionActorConfig {
                index: i,
                ticks_per_step: 1,
                source_indices: Vec::new(),
                receives_observation: false,
            })
            .collect();

        // Subcortical regions tick faster
        for (i, name) in cfg.region_names.iter().enumerate() {
            if name == "cerebellum" || name == "basal_ganglia" {
                actor_configs[i].ticks_per_step = 3;
            }
        }

        // Wire connections
        for conn in &cfg.connections {
            actor_configs[conn.to].source_indices = conn.from.clone();
            actor_configs[conn.to].receives_observation = conn.receives_observation;
        }

        // Spawn region actors
        let mut handles = Vec::with_capacity(n_regions);
        for ac in &actor_configs {
            let r = ac.index;
            let weights = Arc::new(w.regions[r].clone());
            let synapse = if !ac.source_indices.is_empty() {
                Some(w.connection_synapses[cfg.connections.iter()
                    .position(|c| c.to == r).unwrap()].clone())
            } else {
                None
            };
            let obs_proj = w.obs_proj.clone();
            let source_slots: Vec<Arc<Mutex<Slot>>> = ac.source_indices.iter()
                .map(|&i| Arc::clone(&slots[i])).collect();
            let output_slot = Arc::clone(&slots[r]);
            let obs = Arc::clone(&observation);
            let stop = Arc::clone(&running);
            let ticks = ac.ticks_per_step;
            let receives_obs = ac.receives_observation;
            let raw_obs_dim = cfg.raw_obs_dim;

            let thread = thread::Builder::new()
                .name(format!("region-{}", cfg.region_names[r]))
                .spawn(move || {
                    region_loop(
                        &weights, source_slots, synapse, obs_proj,
                        output_slot, obs, stop, ticks, receives_obs, raw_obs_dim,
                    );
                })
                .expect("spawn region thread");

            handles.push(RegionHandle {
                thread: Some(thread),
                output: Arc::clone(&slots[r]),
            });
        }

        // Global sync collator
        let sync_slots: Vec<Arc<Mutex<Slot>>> = slots.iter().map(Arc::clone).collect();
        let global_sync = Arc::new(Mutex::new(vec![0.0f32; cfg.n_global_sync]));
        let gs = Arc::clone(&global_sync);
        let stop = Arc::clone(&running);
        let global_tick = Arc::new(AtomicU64::new(0));
        let gt = Arc::clone(&global_tick);
        let sync_left = w.global_sync_left.clone();
        let sync_right = w.global_sync_right.clone();
        let decay = w.global_decay.clone();
        let n_sync = cfg.n_global_sync;

        let collator = thread::Builder::new()
            .name("collator".into())
            .spawn(move || {
                let mut alpha = vec![0.0f32; n_sync];
                let mut beta = vec![1.0f32; n_sync];

                while stop.load(Ordering::Relaxed) {
                    // Read all region outputs
                    let mut all_act = Vec::new();
                    for slot in &sync_slots {
                        let s = slot.lock().unwrap();
                        all_act.extend_from_slice(&s.activated);
                    }

                    // Update sync accumulators
                    for i in 0..n_sync {
                        let l = sync_left[i];
                        let r = sync_right[i];
                        if l < all_act.len() && r < all_act.len() {
                            let pw = all_act[l] * all_act[r];
                            let d = (-decay[i].clamp(0.0, 15.0)).exp();
                            alpha[i] = d * alpha[i] + pw;
                            beta[i] = d * beta[i] + 1.0;
                        }
                    }

                    let sync: Vec<f32> = (0..n_sync)
                        .map(|i| alpha[i] / beta[i].sqrt().max(1e-8))
                        .collect();
                    *gs.lock().unwrap() = sync;
                    gt.fetch_add(1, Ordering::Relaxed);

                    // Collator runs at roughly cortical tick rate
                    thread::sleep(std::time::Duration::from_micros(100));
                }
            })
            .expect("spawn collator");

        Self {
            regions: handles,
            observation,
            running,
            global_sync,
            global_tick,
            collator: Some(collator),
            config: cfg.clone(),
        }
    }

    /// Feed a new observation into the brain.
    pub fn set_observation(&self, obs: &[f32]) {
        *self.observation.lock().unwrap() = obs.to_vec();
    }

    /// Read the current global sync signal.
    pub fn read_sync(&self) -> Vec<f32> {
        self.global_sync.lock().unwrap().clone()
    }

    /// Read a specific region's latest activated state.
    pub fn read_region(&self, index: usize) -> Vec<f32> {
        self.regions[index].output.lock().unwrap().activated.clone()
    }

    /// Current global tick count.
    pub fn tick_count(&self) -> u64 {
        self.global_tick.load(Ordering::Relaxed)
    }

    /// Shut down all actors.
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        for h in &mut self.regions {
            if let Some(t) = h.thread.take() {
                t.join().ok();
            }
        }
        if let Some(t) = self.collator.take() {
            t.join().ok();
        }
    }
}

impl Drop for ActorBrain {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ─── Region loop ───────────────────────────────────────────

fn region_loop(
    weights: &CtmWeights,
    sources: Vec<Arc<Mutex<Slot>>>,
    synapse: Option<Linear>,
    obs_proj: Linear,
    output: Arc<Mutex<Slot>>,
    observation: Arc<Mutex<Vec<f32>>>,
    running: Arc<AtomicBool>,
    ticks_per_step: usize,
    receives_observation: bool,
    _raw_obs_dim: usize,
) {
    let mut state = CtmState::new(weights);
    let d_input = weights.config.d_input;
    let mut local_tick = 0u64;

    while running.load(Ordering::Relaxed) {
        // Build observation from sources
        let obs = if let Some(ref syn) = synapse {
            let mut src = Vec::new();
            for slot in &sources {
                let s = slot.lock().unwrap();
                src.extend_from_slice(&s.activated);
            }
            if receives_observation {
                let ext = observation.lock().unwrap();
                src.extend_from_slice(&ext);
            }
            syn.forward(&src)
        } else {
            let ext = observation.lock().unwrap();
            obs_proj.forward(&ext)
        };

        // Run internal CTM ticks
        for _ in 0..ticks_per_step {
            ctm_forward(weights, &mut state, &obs, 1, d_input);
        }
        local_tick += ticks_per_step as u64;

        // Publish (brief lock — just pointer swap)
        {
            let mut slot = output.lock().unwrap();
            slot.activated.clear();
            slot.activated.extend_from_slice(&state.activated);
            slot.tick = local_tick;
        }

        // Yield to other threads. Fast regions (ticks_per_step=3) naturally
        // complete faster and loop more often — no sleep needed.
        thread::yield_now();
    }
}
