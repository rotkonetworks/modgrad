//! TCP debug socket for the Neural Computer.
//!
//! The NC listens on a port. Debuggers connect and can:
//!   - Subscribe to the token stream (every token in/out)
//!   - Read region activations, sync state, NLM traces
//!   - Inject tokens (text, actions, media)
//!   - Pause/resume/single-step the NC
//!   - Read model metadata (region names, param counts, etc.)
//!
//! Protocol: length-prefixed bincode over TCP.
//!   [u32 LE length] [bincode payload]
//!
//! Designed for one debugger at a time. Multiple connections are
//! accepted but share the same NC state — no isolation.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;
use serde::{Deserialize, Serialize};

use super::regional::*;

/// Default debug port.
pub const DEFAULT_PORT: u16 = 4747;

// ── Wire protocol ────────────────────────────────────────

/// Message from debugger → NC.
#[derive(Debug, Serialize, Deserialize)]
pub enum DebugRequest {
    /// Get model metadata (region names, params, vocab size).
    GetMeta,
    /// Get current state snapshot (activations, sync values per region).
    GetState,
    /// Get NLM trace for a specific region.
    GetTrace { region: usize },
    /// Get recent token history.
    GetHistory { last_n: usize },
    /// Inject tokens into the NC (as if typed/received).
    Inject { tokens: Vec<usize> },
    /// Inject text.
    InjectText { text: String },
    /// Pause the NC (stop processing inputs).
    Pause,
    /// Resume.
    Resume,
    /// Single-step: process one token and pause.
    Step { token: usize },
    /// Subscribe to live token stream.
    Subscribe,
    /// Unsubscribe from live token stream.
    Unsubscribe,
    /// Ping (keepalive).
    Ping,
}

/// Message from NC → debugger.
#[derive(Debug, Serialize, Deserialize)]
pub enum DebugResponse {
    /// Model metadata.
    Meta {
        region_names: Vec<String>,
        region_params: Vec<usize>,
        region_d_model: Vec<usize>,
        region_memory: Vec<usize>,
        n_connections: usize,
        vocab_size: usize,
        total_params: usize,
        n_global_sync: usize,
    },
    /// Current state snapshot.
    State {
        /// Per-region: activated neuron values.
        region_activations: Vec<Vec<f32>>,
        /// Per-region: output from last tick.
        region_outputs: Vec<Vec<f32>>,
        /// Global sync alpha/beta.
        global_sync: Vec<f32>,
        /// Total tokens processed.
        history_len: usize,
        /// Outer exit gate: per-tick lambda values from last forward.
        exit_lambdas: Vec<f32>,
        /// How many outer ticks actually ran on last forward.
        ticks_used: usize,
    },
    /// NLM trace for one region: [d_model × memory_length].
    Trace {
        region: usize,
        d_model: usize,
        memory_length: usize,
        trace: Vec<f32>,
    },
    /// Recent token history.
    History { tokens: Vec<usize> },
    /// A token was processed (live stream event).
    TokenEvent {
        /// The token that was processed.
        token: usize,
        /// Whether it was input (true) or generated output (false).
        is_input: bool,
        /// Output logits (top 10 only, to save bandwidth).
        top_logits: Vec<(usize, f32)>,
    },
    /// Ack for pause/resume/step/inject.
    Ok,
    /// Pong.
    Pong,
    /// Error.
    Error { msg: String },
}

// ── Wire encoding ────────────────────────────────────────

fn send_msg(stream: &mut TcpStream, msg: &DebugResponse) -> std::io::Result<()> {
    let data = bincode::serialize(msg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let len = data.len() as u32;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(&data)?;
    stream.flush()
}

fn recv_msg(stream: &mut TcpStream) -> std::io::Result<DebugRequest> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 10_000_000 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "message too large"));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    bincode::deserialize(&buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ── Debug server ─────────────────────────────────────────

/// Events the debug server sends to the NC main loop.
pub enum DebugEvent {
    Inject(Vec<usize>),
    Pause,
    Resume,
    Step(usize),
}

/// Handle for the NC to notify the debug server of token events.
pub struct DebugHandle {
    /// Send token events to connected debuggers.
    event_tx: mpsc::Sender<DebugResponse>,
    /// Receive control events from debuggers.
    pub control_rx: mpsc::Receiver<DebugEvent>,
}

impl DebugHandle {
    /// Notify debugger of a token being processed.
    pub fn on_token(&self, token: usize, is_input: bool, logits: &[f32]) {
        // Top 10 logits
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(10);

        self.event_tx.send(DebugResponse::TokenEvent {
            token,
            is_input,
            top_logits: indexed,
        }).ok();
    }

    /// Check for control events (non-blocking).
    pub fn poll_control(&self) -> Option<DebugEvent> {
        self.control_rx.try_recv().ok()
    }
}

/// Start the debug server on a background thread.
/// Returns a handle for the NC to communicate with connected debuggers.
pub fn start_debug_server(
    port: u16,
    nc: std::sync::Arc<std::sync::Mutex<NcDebugView>>,
) -> DebugHandle {
    let (event_tx, event_rx) = mpsc::channel::<DebugResponse>();
    let (control_tx, control_rx) = mpsc::channel::<DebugEvent>();

    let event_tx_clone = event_tx.clone();

    thread::spawn(move || {
        let listener = match TcpListener::bind(format!("127.0.0.1:{port}")) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("  debug server: failed to bind port {port}: {e}");
                return;
            }
        };
        eprintln!("  debug server: listening on 127.0.0.1:{port}");

        for conn in listener.incoming() {
            let mut stream = match conn {
                Ok(s) => s,
                Err(_) => continue,
            };
            stream.set_nodelay(true).ok();
            eprintln!("  debug server: client connected");

            let nc = nc.clone();
            let control_tx = control_tx.clone();
            let event_rx_proxy = event_rx; // single consumer — last client wins
            // For simplicity: handle one client at a time in this thread.
            // A production server would spawn per-client threads.
            handle_client(&mut stream, &nc, &control_tx, &event_tx_clone);
            eprintln!("  debug server: client disconnected");
            // Re-create event channel for next client
            break; // For now: one client lifetime = server lifetime
        }
    });

    DebugHandle { event_tx, control_rx }
}

/// Snapshot of NC state that the debug server can read without blocking the NC.
pub struct NcDebugView {
    pub region_names: Vec<String>,
    pub region_params: Vec<usize>,
    pub region_d_model: Vec<usize>,
    pub region_memory: Vec<usize>,
    pub n_connections: usize,
    pub vocab_size: usize,
    pub total_params: usize,
    pub n_global_sync: usize,
    pub region_activations: Vec<Vec<f32>>,
    pub region_outputs: Vec<Vec<f32>>,
    pub global_sync: Vec<f32>,
    pub history: Vec<usize>,
    pub traces: Vec<Vec<f32>>, // per-region NLM traces
    /// Outer exit gate lambdas from last forward pass.
    pub exit_lambdas: Vec<f32>,
    /// How many outer ticks ran on last forward.
    pub ticks_used: usize,
}

impl NcDebugView {
    pub fn from_nc(nc: &NeuralComputer) -> Self {
        let cfg = &nc.weights.config;
        Self {
            region_names: cfg.region_names.clone(),
            region_params: nc.weights.regions.iter().map(|r| r.n_params()).collect(),
            region_d_model: cfg.regions.iter().map(|r| r.d_model).collect(),
            region_memory: cfg.regions.iter().map(|r| r.memory_length).collect(),
            n_connections: cfg.connections.len(),
            vocab_size: cfg.out_dims,
            total_params: nc.weights.n_params(),
            n_global_sync: cfg.n_global_sync,
            region_activations: nc.state.region_states.iter()
                .map(|s| s.activated.clone()).collect(),
            region_outputs: nc.state.region_outputs.clone(),
            global_sync: (0..cfg.n_global_sync)
                .map(|i| nc.state.global_alpha[i] / nc.state.global_beta[i].sqrt().max(1e-8))
                .collect(),
            history: nc.history.clone(),
            traces: nc.state.region_states.iter()
                .map(|s| s.trace.clone()).collect(),
            exit_lambdas: nc.last_exit_lambdas.clone(),
            ticks_used: nc.last_ticks_used,
        }
    }
}

fn handle_client(
    stream: &mut TcpStream,
    nc: &std::sync::Arc<std::sync::Mutex<NcDebugView>>,
    control_tx: &mpsc::Sender<DebugEvent>,
    _event_tx: &mpsc::Sender<DebugResponse>,
) {
    loop {
        let req = match recv_msg(stream) {
            Ok(r) => r,
            Err(_) => break,
        };

        let view = nc.lock().unwrap();
        let resp = match req {
            DebugRequest::Ping => DebugResponse::Pong,

            DebugRequest::GetMeta => DebugResponse::Meta {
                region_names: view.region_names.clone(),
                region_params: view.region_params.clone(),
                region_d_model: view.region_d_model.clone(),
                region_memory: view.region_memory.clone(),
                n_connections: view.n_connections,
                vocab_size: view.vocab_size,
                total_params: view.total_params,
                n_global_sync: view.n_global_sync,
            },

            DebugRequest::GetState => DebugResponse::State {
                region_activations: view.region_activations.clone(),
                region_outputs: view.region_outputs.clone(),
                global_sync: view.global_sync.clone(),
                history_len: view.history.len(),
                exit_lambdas: view.exit_lambdas.clone(),
                ticks_used: view.ticks_used,
            },

            DebugRequest::GetTrace { region } => {
                if region < view.traces.len() {
                    DebugResponse::Trace {
                        region,
                        d_model: view.region_d_model[region],
                        memory_length: view.region_memory[region],
                        trace: view.traces[region].clone(),
                    }
                } else {
                    DebugResponse::Error { msg: format!("invalid region {region}") }
                }
            }

            DebugRequest::GetHistory { last_n } => {
                let start = view.history.len().saturating_sub(last_n);
                DebugResponse::History { tokens: view.history[start..].to_vec() }
            }

            DebugRequest::Inject { tokens } => {
                drop(view); // release lock before sending control
                control_tx.send(DebugEvent::Inject(tokens)).ok();
                DebugResponse::Ok
            }

            DebugRequest::InjectText { text } => {
                let tokens = text_to_tokens(text.as_bytes());
                drop(view);
                control_tx.send(DebugEvent::Inject(tokens)).ok();
                DebugResponse::Ok
            }

            DebugRequest::Pause => {
                drop(view);
                control_tx.send(DebugEvent::Pause).ok();
                DebugResponse::Ok
            }

            DebugRequest::Resume => {
                drop(view);
                control_tx.send(DebugEvent::Resume).ok();
                DebugResponse::Ok
            }

            DebugRequest::Step { token } => {
                drop(view);
                control_tx.send(DebugEvent::Step(token)).ok();
                DebugResponse::Ok
            }

            DebugRequest::Subscribe | DebugRequest::Unsubscribe => {
                DebugResponse::Ok // TODO: streaming subscription
            }
        };

        if send_msg(stream, &resp).is_err() { break; }
    }
}
