//! isis daemon runtime — shared between the `isis daemon` subcommand and
//! the dedicated `isisd` binary.
//!
//! Lifecycle:
//!   1. Load or create `RegionalWeights` from a checkpoint path.
//!   2. Wrap in `NeuralComputer`.
//!   3. Start the `nc_socket` debug/control server on `port`.
//!   4. Poll for injected events; respond; loop until SIGINT.
//!   5. Save checkpoint on graceful shutdown.
//!
//! This is intentionally thin — it's the wire from "ran a process" to
//! "brain is serving requests." Structured logging via `tracing` makes
//! the event flow visible without changing runtime semantics.

use modgrad_ctm::graph::{NeuralComputer, RegionalConfig, RegionalWeights};
use isis_runtime::nc_socket;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{error, info, warn};

/// Launch the daemon. Blocks until SIGINT, then saves and returns.
///
/// `checkpoint` is the filesystem path used for both load (at startup
/// if the file exists) and save (at shutdown). `port` is the TCP port
/// served by `nc_socket::start_debug_server` — the wire protocol that
/// external clients (isis send / modgrad-debugger) speak.
pub fn run(checkpoint: &str, port: u16) {
    let w = load_or_create(checkpoint);
    w.print_summary();

    let mut nc = NeuralComputer::new(w);

    let view = nc_socket::NcDebugView::from_nc(&nc);
    let view = std::sync::Arc::new(std::sync::Mutex::new(view));
    let handle = nc_socket::start_debug_server(port, view.clone());

    info!(port = port, "isisd ready — accepting control-protocol clients");
    info!(
        "connect with: modgrad-debugger 127.0.0.1:{port}; \
         send text: isis send \"...\" --addr 127.0.0.1:{port}"
    );

    let running = Arc::new(AtomicBool::new(true));
    install_sigint_handler(running.clone());

    while running.load(Ordering::SeqCst) {
        if let Some(event) = handle.poll_control() {
            match event {
                nc_socket::DebugEvent::Inject(tokens) => {
                    let n_in = tokens.len();
                    let response = nc.act(&tokens, 256, 0.8);
                    if let Ok(mut v) = view.try_lock() {
                        *v = nc_socket::NcDebugView::from_nc(&nc);
                    }
                    // Byte-token output goes to stdout as text (the
                    // classic NC interface); `tracing` gets a summary.
                    for &t in &response {
                        if t < 256 {
                            print!("{}", t as u8 as char);
                        }
                    }
                    io::stdout().flush().ok();
                    tracing::debug!(
                        tokens_in = n_in,
                        tokens_out = response.len(),
                        "inject handled"
                    );
                }
                nc_socket::DebugEvent::Pause | nc_socket::DebugEvent::Resume => {}
                nc_socket::DebugEvent::Step(token) => {
                    nc.step(token);
                    if let Ok(mut v) = view.try_lock() {
                        *v = nc_socket::NcDebugView::from_nc(&nc);
                    }
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    info!("shutting down — saving checkpoint");
    if let Err(e) = nc.weights.save(checkpoint) {
        error!(error = %e, "checkpoint save failed");
    } else {
        info!(path = %checkpoint, "checkpoint saved");
    }
}

fn load_or_create(path: &str) -> RegionalWeights {
    if std::path::Path::new(path).exists() {
        info!(path = %path, "loading checkpoint");
        RegionalWeights::load(path).unwrap_or_else(|e| {
            error!(error = %e, "checkpoint load failed");
            std::process::exit(1);
        })
    } else {
        warn!(path = %path, "no checkpoint found — creating fresh 8-region model");
        let cfg = RegionalConfig::eight_region(32, 256, 2);
        RegionalWeights::new(cfg)
    }
}

fn install_sigint_handler(flag: Arc<AtomicBool>) {
    let f = flag.clone();
    ctrlc::set_handler(move || {
        info!("SIGINT received");
        f.store(false, Ordering::SeqCst);
    })
    .ok();
}

/// Install a default tracing subscriber reading from `RUST_LOG` (with
/// `info` as the fallback filter). Call once at process start.
pub fn init_logging() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}
