//! isisd — isis daemon: the organism as a living service.
//!
//! Boots an organism and keeps it running. It thinks when idle,
//! sleeps when tired, responds when spoken to.
//!
//! Usage:
//!   isisd [checkpoint] [--port PORT]

use modgrad::tabula_rasa::{Organism, Dna};
use modgrad::vocab::Vocab;
use modgrad::daemon::Daemon;

use clap::Parser;

#[derive(Parser)]
#[command(name = "isisd", version, about = "isis daemon — living organism service")]
struct Args {
    /// Checkpoint path
    #[arg(default_value = "organism.bin")]
    checkpoint: String,
    /// Port to listen on
    #[arg(short, long, default_value = "4747")]
    port: u16,
}

fn main() {
    modgrad::gpu::init_global();
    let args = Args::parse();

    // Build vocab
    let text = std::fs::read_to_string("train_climbmix.txt")
        .unwrap_or_else(|_| "the cat sat on the mat".repeat(100));
    let vocab = Vocab::from_text(&text, 52);

    // Load or create organism
    let org = if std::path::Path::new(&args.checkpoint).exists() {
        eprintln!("Loading organism from {}...", args.checkpoint);
        Organism::load(&args.checkpoint).expect("failed to load")
    } else {
        eprintln!("Creating new organism...");
        let mut dna = Dna::small();
        dna.vocab_size = vocab.size();
        Organism::new(dna)
    };

    let mut daemon = Daemon::new(org, vocab, args.checkpoint.into());
    daemon.run(args.port);
}
