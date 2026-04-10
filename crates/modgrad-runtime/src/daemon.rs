//! Daemon: the organism as a living service.
//!
//! The organism runs continuously. It's not spawned for tasks and killed after.
//! It thinks when idle, sleeps when tired, responds when spoken to.
//!
//! Communication via Unix domain socket. Clients connect, send JSON commands,
//! receive JSON responses. Multiple clients can connect simultaneously.
//!
//! Lifecycle:
//!   isis daemon start → boots, enters idle loop
//!   isis talk "hello" → sends message, gets response
//!   isis teach "the" → "cat" → correction applied live
//!   isis health → vital signs
//!   isis sleep → manual sleep trigger
//!   Ctrl+C → checkpoint + graceful shutdown
//!
//! The organism is never killed. It sleeps and wakes.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use crate::organism::Organism;
use crate::tuning::TuningRegistry;
use modgrad_persist::vocab::Vocab;
use crate::homeostasis::Homeostasis;

const DEFAULT_PORT: u16 = 4747; // "ISIS" on phone keypad (I=4, S=7)

/// Command from client to daemon.
#[derive(Debug, Deserialize)]
#[serde(tag = "cmd")]
pub enum Command {
    /// Generate next word(s) after prompt.
    #[serde(rename = "talk")]
    Talk { prompt: String, max_words: Option<usize> },

    /// Teach: correct the organism's output.
    #[serde(rename = "teach")]
    Teach { prompt: Vec<String>, target: String },

    /// Get health/vital signs.
    #[serde(rename = "health")]
    Health,

    /// Trigger sleep manually.
    #[serde(rename = "sleep")]
    Sleep,

    /// Get generation sample.
    #[serde(rename = "sample")]
    Sample { prompt: String },

    /// Get full status.
    #[serde(rename = "status")]
    Status,

    /// Get a tuning parameter value.
    #[serde(rename = "get-tune")]
    GetTune,

    /// Set a tuning parameter (JSON patch applied to TuningConfig).
    #[serde(rename = "tune")]
    Tune { patch: serde_json::Value },

    /// Write default tuning config to file.
    #[serde(rename = "tune-init")]
    TuneInit { path: String },

    /// Shutdown gracefully.
    #[serde(rename = "shutdown")]
    Shutdown,
}

/// Response from daemon to client.
#[derive(Debug, Serialize)]
pub struct Response {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Response {
    fn ok(text: &str) -> Self {
        Self { ok: true, text: Some(text.into()), data: None, error: None }
    }
    fn data(data: serde_json::Value) -> Self {
        Self { ok: true, text: None, data: Some(data), error: None }
    }
    fn err(msg: &str) -> Self {
        Self { ok: false, text: None, data: None, error: Some(msg.into()) }
    }
}

/// The living organism daemon.
pub struct Daemon {
    pub organism: Organism,
    pub vocab: Vocab,
    pub homeostasis: Homeostasis,
    pub tuning: TuningRegistry,
    pub checkpoint_path: String,
    pub alive: bool,
    pub idle_steps: u64,
}

impl Daemon {
    pub fn new(organism: Organism, vocab: Vocab, checkpoint_path: String) -> Self {
        // Look for tuning.json next to checkpoint
        let tuning_path = checkpoint_path.replace(".bin", ".tuning.json");
        let tuning = TuningRegistry::from_file(&tuning_path);
        Self {
            organism,
            vocab,
            homeostasis: Homeostasis::default(),
            tuning,
            checkpoint_path,
            alive: true,
            idle_steps: 0,
        }
    }

    /// Handle a single command.
    pub fn handle(&mut self, cmd: Command) -> Response {
        match cmd {
            Command::Talk { prompt, max_words } => {
                let n = max_words.unwrap_or(5);
                let ids = self.vocab.encode(&prompt);
                let (_, _syncs) = self.organism.forward_inner(&ids, false);

                // Generate N words
                let mut generated = Vec::new();
                let mut current_ids = ids;
                for _ in 0..n {
                    let (_, syncs) = self.organism.forward_inner(&current_ids, false);
                    let sync = syncs.last().cloned().unwrap_or_default();
                    let logits = self.organism.output_proj.forward(&sync);
                    let predicted = logits.iter().enumerate()
                        .take(self.vocab.size())
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let word = self.vocab.get_word(predicted).to_string();
                    generated.push(word.clone());
                    current_ids.push(predicted);
                }

                // Update homeostasis — interaction increases pressure
                let surprise = 1.0; // simple estimate
                self.homeostasis.tick_from_ctm(1.0, true, surprise);

                Response::ok(&format!("{} {}", prompt, generated.join(" ")))
            }

            Command::Teach { prompt, target } => {
                let prompt_ids: Vec<usize> = prompt.iter()
                    .map(|w| *self.vocab.word_to_id.get(w).unwrap_or(&0))
                    .collect();
                let target_id = *self.vocab.word_to_id.get(&target).unwrap_or(&0);

                if target_id == 0 {
                    return Response::err(&format!("unknown word: {target}"));
                }

                // Get sync for prompt
                let (_, syncs) = self.organism.forward_inner(&prompt_ids, false);
                if let Some(sync) = syncs.last() {
                    // Strong correction (neuvola imprint)
                    let out = &mut self.organism.output_proj;
                    let in_d = out.in_dim;
                    let sync_norm: f32 = sync.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                    for k in 0..in_d.min(sync.len()) {
                        let n = sync[k] / sync_norm;
                        let tw = self.tuning.config.basal_ganglia.teach_toward.val();
                        out.weight[target_id * in_d + k] = (1.0 - tw) * out.weight[target_id * in_d + k] + tw * n;
                    }
                    out.bias[target_id] += self.tuning.config.basal_ganglia.teach_bias.val();
                }

                Response::ok(&format!("taught: {} → {}", prompt.join(" "), target))
            }

            Command::Health => {
                let _health = crate::autonomic::diagnose(&modgrad_io::memory::MemoryBank::default());
                Response::data(serde_json::json!({
                    "pressure": self.homeostasis.pressure,
                    "zone": format!("{:?}", self.homeostasis.zone()),
                    "output_quality": self.homeostasis.output_quality,
                    "dopamine": self.organism.neuromod.dopamine,
                    "serotonin": self.organism.neuromod.serotonin,
                    "norepinephrine": self.organism.neuromod.norepinephrine,
                    "tokens_seen": self.organism.tokens_seen,
                    "sleep_cycles": self.organism.sleep_cycles,
                    "biggest_pressure": self.homeostasis.biggest_pressure_source(),
                    "self_report": self.homeostasis.self_report(),
                }))
            }

            Command::Sleep => {
                self.organism.sleep();
                self.homeostasis.on_sleep(1.0);
                Response::ok(&format!("slept (cycle #{})", self.organism.sleep_cycles))
            }

            Command::Sample { prompt } => {
                let ids = self.vocab.encode(&prompt);
                let mut words = Vec::new();
                let mut current_ids = ids;
                for _ in 0..10 {
                    let (_, syncs) = self.organism.forward_inner(&current_ids, false);
                    let sync = syncs.last().cloned().unwrap_or_default();
                    let logits = self.organism.output_proj.forward(&sync);
                    let predicted = logits.iter().enumerate()
                        .take(self.vocab.size())
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    words.push(self.vocab.get_word(predicted).to_string());
                    current_ids.push(predicted);
                }
                Response::ok(&format!("{} {}", prompt, words.join(" ")))
            }

            Command::Status => {
                Response::data(serde_json::json!({
                    "alive": self.alive,
                    "tokens_seen": self.organism.tokens_seen,
                    "sleep_cycles": self.organism.sleep_cycles,
                    "vocab_size": self.vocab.size(),
                    "params": self.organism.param_count(),
                    "idle_steps": self.idle_steps,
                    "homeostasis": self.homeostasis.self_report(),
                }))
            }

            Command::GetTune => {
                match serde_json::to_value(&self.tuning.config) {
                    Ok(v) => Response::data(v),
                    Err(e) => Response::err(&format!("serialize: {e}")),
                }
            }

            Command::Tune { patch } => {
                // Merge patch into current config JSON, then deserialize back.
                // This allows partial updates: {"noise": {"base_amplitude": {"value": 0.2}}}
                let mut current = serde_json::to_value(&self.tuning.config)
                    .unwrap_or_default();
                json_merge(&mut current, &patch);
                match serde_json::from_value::<crate::tuning::TuningConfig>(current) {
                    Ok(new_config) => {
                        let diffs = new_config.diff_from_default();
                        self.tuning.config = new_config;
                        self.tuning.save().ok();
                        Response::ok(&format!("tuned ({} params differ from default)", diffs.len()))
                    }
                    Err(e) => Response::err(&format!("invalid patch: {e}")),
                }
            }

            Command::TuneInit { path } => {
                // Path sanitization: reject traversal and absolute paths.
                // The daemon should only write within its working directory.
                if path.contains("..") || path.starts_with('/') {
                    return Response::err("path traversal not allowed");
                }
                match crate::tuning::TuningRegistry::write_defaults(&path) {
                    Ok(()) => Response::ok(&format!("wrote defaults to {path}")),
                    Err(e) => Response::err(&e),
                }
            }

            Command::Shutdown => {
                self.organism.save(&self.checkpoint_path).ok();
                self.tuning.save().ok();
                self.alive = false;
                Response::ok("shutting down (checkpoint + tuning saved)")
            }
        }
    }

    /// Run the daemon: listen for connections, process commands.
    pub fn run(&mut self, port: u16) {
        let addr = format!("127.0.0.1:{port}");
        let listener = TcpListener::bind(&addr).expect("can't bind");
        listener.set_nonblocking(true).ok();

        eprintln!("isis daemon listening on {addr}");
        eprintln!("  {} params, {} tokens seen, {} sleep cycles",
            self.organism.param_count(), self.organism.tokens_seen, self.organism.sleep_cycles);

        // Ctrl+C handler
        let alive = Arc::new(Mutex::new(true));
        let alive_clone = alive.clone();
        ctrlc::set_handler(move || {
            *alive_clone.lock().unwrap() = false;
        }).ok();

        while self.alive && *alive.lock().unwrap() {
            // Check for connections
            match listener.accept() {
                Ok((stream, _)) => {
                    self.handle_connection(stream);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No connection — do idle work

                    // Hot-reload tuning config if file changed
                    if self.idle_steps % 10 == 0 { // every ~1s
                        self.tuning.check_reload();
                    }

                    // Check sleep pressure
                    if self.homeostasis.must_sleep() {
                        eprintln!("  [autonomic] forced sleep (pressure={:.2})",
                            self.homeostasis.pressure);
                        self.organism.sleep();
                        self.homeostasis.on_sleep(1.0);
                    } else if self.homeostasis.should_sleep() {
                        // Voluntary sleep in idle time
                        self.organism.sleep();
                        self.homeostasis.on_sleep(0.8); // lighter idle sleep
                    }

                    self.idle_steps += 1;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(e) => {
                    eprintln!("  accept error: {e}");
                }
            }
        }

        // Graceful shutdown
        eprintln!("\nShutting down...");
        self.organism.save(&self.checkpoint_path).ok();
        self.save_tuning();
        eprintln!("Checkpoint saved to {}", self.checkpoint_path);
    }

    /// Save tuning config alongside checkpoint on shutdown.
    fn save_tuning(&self) {
        self.tuning.save().ok();
    }

    fn handle_connection(&mut self, stream: TcpStream) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut writer = stream;

        let mut line = String::new();
        if reader.read_line(&mut line).is_ok() && !line.is_empty() {
            let response = match serde_json::from_str::<Command>(&line) {
                Ok(cmd) => self.handle(cmd),
                Err(e) => Response::err(&format!("invalid command: {e}")),
            };

            let json = serde_json::to_string(&response).unwrap_or_default();
            writeln!(writer, "{json}").ok();
        }
    }
}

/// RFC 7396 JSON Merge Patch: recursively merge `patch` into `target`.
fn json_merge(target: &mut serde_json::Value, patch: &serde_json::Value) {
    if let serde_json::Value::Object(pm) = patch {
        if let serde_json::Value::Object(tm) = target {
            for (k, pv) in pm {
                let tv = tm.entry(k.clone()).or_insert(serde_json::Value::Null);
                json_merge(tv, pv);
            }
            return;
        }
    }
    *target = patch.clone();
}
