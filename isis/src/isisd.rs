//! isisd — dedicated daemon binary for the isis neural-computer runtime.
//!
//! Usage:
//!   isisd [--checkpoint PATH] [--port 4747]
//!   isisd --help
//!
//! Runs until SIGINT, saves checkpoint on exit. Logs to stderr via
//! `tracing`; control filter with `RUST_LOG=debug isisd ...`.
//!
//! For one-off invocations of the same daemon from the main isis CLI,
//! see `isis daemon <checkpoint> --port <n>` — both entry points share
//! `isis::daemon::run`.

use clap::Parser;

#[derive(Parser)]
#[command(name = "isisd", version, about = "isis neural-computer daemon")]
struct Cli {
    /// Checkpoint path — loaded at startup if present, saved at shutdown.
    #[arg(long, default_value = "model.bin")]
    checkpoint: String,

    /// TCP port for the NC control/debug protocol.
    #[arg(long, default_value = "4747")]
    port: u16,
}

fn main() {
    let cli = Cli::parse();
    isis::daemon::init_logging();
    isis::daemon::run(&cli.checkpoint, cli.port);
}
