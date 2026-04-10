//! Unix socket control channel for debugger ↔ host communication.
//!
//! Protocol: newline-delimited JSON over `stream.sock`.
//! Host listens, debugger connects. One connection at a time.
//!
//! Commands (debugger → host):
//!   {"cmd":"set_detail","level":"Full"}
//!   {"cmd":"stream_region","id":"attention","enable":true}
//!   {"cmd":"get_manifest"}
//!   {"cmd":"ping"}
//!
//! Responses (host → debugger):
//!   {"ok":true}
//!   {"ok":true,"manifest":{...}}
//!   {"ok":false,"error":"unknown command"}

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::telemetry::{DetailLevel, Telemetry};

// ─── Protocol types ──────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd")]
pub enum Command {
    #[serde(rename = "set_detail")]
    SetDetail { level: DetailLevel },
    #[serde(rename = "stream_region")]
    StreamRegion { id: String, enable: bool },
    #[serde(rename = "toggle_extra")]
    ToggleExtra { id: String, enable: bool },
    #[serde(rename = "get_manifest")]
    GetManifest,
    #[serde(rename = "ping")]
    Ping,
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest: Option<serde_json::Value>,
}

impl Response {
    fn ok() -> Self {
        Self { ok: true, error: None, manifest: None }
    }
    fn err(msg: impl Into<String>) -> Self {
        Self { ok: false, error: Some(msg.into()), manifest: None }
    }
    fn with_manifest(manifest: serde_json::Value) -> Self {
        Self { ok: true, error: None, manifest: Some(manifest) }
    }
}

// ─── Control Server ──────────────────────────────────────

/// Shared handle to telemetry that the control thread can mutate.
pub type TelemetryHandle = Arc<Mutex<Telemetry>>;

/// Runs the control socket listener in a background thread.
/// Returns the socket path so the debugger knows where to connect.
///
/// The socket is placed next to the stream file:
///   `foo.stream.bin` → `foo.stream.sock`
pub fn start_control_server(
    stream_path: &str,
    telemetry: TelemetryHandle,
) -> Result<PathBuf, String> {
    let sock_path = PathBuf::from(stream_path.replace(".stream.bin", ".stream.sock"));

    // Remove stale socket if it exists
    if sock_path.exists() {
        std::fs::remove_file(&sock_path).ok();
    }

    let listener = UnixListener::bind(&sock_path)
        .map_err(|e| format!("bind {}: {e}", sock_path.display()))?;

    // Non-blocking accept so we can check for shutdown, but each
    // connection is handled in blocking mode for simplicity.
    listener.set_nonblocking(false)
        .map_err(|e| format!("set_nonblocking: {e}"))?;

    let sock_path_clone = sock_path.clone();
    let telem = telemetry;

    std::thread::Builder::new()
        .name("ctrl-sock".into())
        .spawn(move || {
            control_loop(&listener, &sock_path_clone, &telem);
        })
        .map_err(|e| format!("spawn control thread: {e}"))?;

    Ok(sock_path)
}

fn control_loop(listener: &UnixListener, sock_path: &Path, telem: &TelemetryHandle) {
    loop {
        match listener.accept() {
            Ok((stream, _addr)) => {
                eprintln!("[ctrl] debugger connected");
                if let Err(e) = handle_connection(stream, telem) {
                    eprintln!("[ctrl] connection error: {e}");
                }
                eprintln!("[ctrl] debugger disconnected");
            }
            Err(e) => {
                // Socket was removed (shutdown) or other fatal error
                if !sock_path.exists() {
                    break;
                }
                eprintln!("[ctrl] accept error: {e}");
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }
}

fn handle_connection(
    stream: std::os::unix::net::UnixStream,
    telem: &TelemetryHandle,
) -> Result<(), String> {
    let reader = BufReader::new(stream.try_clone().map_err(|e| format!("{e}"))?);
    let mut writer = stream;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("read: {e}"))?;
        let line = line.trim();
        if line.is_empty() { continue; }

        let resp = match serde_json::from_str::<Command>(line) {
            Ok(cmd) => execute(cmd, telem),
            Err(e) => Response::err(format!("parse: {e}")),
        };

        let mut json = serde_json::to_string(&resp).unwrap_or_else(|_| r#"{"ok":false}"#.into());
        json.push('\n');
        writer.write_all(json.as_bytes()).map_err(|e| format!("write: {e}"))?;
        writer.flush().map_err(|e| format!("flush: {e}"))?;
    }

    Ok(())
}

fn execute(cmd: Command, telem: &TelemetryHandle) -> Response {
    match cmd {
        Command::Ping => Response::ok(),

        Command::SetDetail { level } => {
            match telem.lock() {
                Ok(mut t) => {
                    t.set_detail_level(level);
                    eprintln!("[ctrl] detail → {level:?}");
                    Response::ok()
                }
                Err(e) => Response::err(format!("lock: {e}")),
            }
        }

        Command::StreamRegion { id, enable } => {
            match telem.lock() {
                Ok(mut t) => {
                    t.stream_region(&id, enable);
                    eprintln!("[ctrl] stream_region {id} = {enable}");
                    Response::ok()
                }
                Err(e) => Response::err(format!("lock: {e}")),
            }
        }

        Command::ToggleExtra { id, enable } => {
            match telem.lock() {
                Ok(mut t) => {
                    t.toggle_extra(&id, enable);
                    eprintln!("[ctrl] extra {id} = {enable}");
                    Response::ok()
                }
                Err(e) => Response::err(format!("lock: {e}")),
            }
        }

        Command::GetManifest => {
            match telem.lock() {
                Ok(t) => {
                    match serde_json::to_value(&t.manifest) {
                        Ok(v) => Response::with_manifest(v),
                        Err(e) => Response::err(format!("serialize: {e}")),
                    }
                }
                Err(e) => Response::err(format!("lock: {e}")),
            }
        }
    }
}

/// Cleanup: remove the socket file on shutdown.
pub fn remove_socket(stream_path: &str) {
    let sock_path = stream_path.replace(".stream.bin", ".stream.sock");
    std::fs::remove_file(&sock_path).ok();
}
