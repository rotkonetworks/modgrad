//! minictm — nanoGPT but for CTMs.
//!
//! Minimal example of the modgrad SDK. Two modes:
//!   --curriculum  : capability-gated training (earn advancement by passing tests)
//!   --data FILE   : raw text training (just loop over bytes)
//!   --chat        : interactive generation from checkpoint
//!
//! Usage:
//!   minictm --curriculum                              # staged learning with test games
//!   minictm --data train.txt --steps 5000             # raw text
//!   minictm --data train.txt --steps 5000 --chat      # train then chat
//!   minictm -c model.bin --chat                       # just chat

use modgrad_ctm::graph::*;
use modgrad_ctm::curriculum;
use modgrad_runtime::challenges::{self, byte_curriculum};
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut data_path: Option<String> = None;
    let mut steps = 2000usize;
    let mut embed_dim = 32usize;
    let mut regions = 4usize;
    let mut ticks = 2usize;
    let mut lr = 3e-3f32;
    let mut context = 32usize;
    let mut chat = false;
    let mut use_curriculum = false;
    let mut test_only = false;
    let mut checkpoint: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => { data_path = Some(args[i+1].clone()); i += 2; }
            "--steps" => { steps = args[i+1].parse().unwrap(); i += 2; }
            "--embed" => { embed_dim = args[i+1].parse().unwrap(); i += 2; }
            "--regions" => { regions = args[i+1].parse().unwrap(); i += 2; }
            "--ticks" => { ticks = args[i+1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i+1].parse().unwrap(); i += 2; }
            "--context" => { context = args[i+1].parse().unwrap(); i += 2; }
            "--chat" => { chat = true; i += 1; }
            "--curriculum" => { use_curriculum = true; i += 1; }
            "--test" => { test_only = true; i += 1; }
            "--checkpoint" | "-c" => { checkpoint = Some(args[i+1].clone()); i += 2; }
            "--help" | "-h" => { print_help(); return; }
            _ => { i += 1; }
        }
    }

    // Load or create model
    let save_path = checkpoint.clone().unwrap_or_else(|| "minictm.bin".to_string());
    let mut w = if std::path::Path::new(&save_path).exists() {
        eprintln!("Loading {save_path}...");
        RegionalWeights::load(&save_path).expect("failed to load")
    } else {
        create_model(embed_dim, regions, ticks)
    };

    let opt_path = save_path.replace(".bin", ".opt.bin");
    let mut opt = if std::path::Path::new(&opt_path).exists() {
        RegionalAdamW::load(&opt_path).unwrap_or_else(|_| RegionalAdamW::new(&w))
    } else {
        RegionalAdamW::new(&w).with_lr(lr).with_clip(5.0)
    };

    w.print_summary();

    // Test only — run challenges and report
    if test_only {
        run_tests(&w);
        return;
    }

    // Chat only
    if chat && data_path.is_none() && !use_curriculum {
        run_chat(&w);
        return;
    }

    // ── Training ──────────────────────────────────────────

    if use_curriculum {
        let stages = byte_curriculum();

        // Per-stage training data — each stage gets its own focused data
        // + real text mixed into later stages
        let real_text = data_path.as_ref().and_then(|path| {
            let mut text = Vec::new();
            std::fs::File::open(path).ok()?.read_to_end(&mut text).ok()?;
            eprintln!("+ {:.1}MB real text from {path}", text.len() as f64 / 1e6);
            Some(text)
        });

        let mut stage_data: Vec<Vec<u8>> = vec![
            generate_byte_class_data(100_000),
            generate_bigram_data(100_000),
            generate_word_data(200_000),
            generate_coherent_data(300_000),
        ];

        // Mix real text into later stages (word completion + coherent)
        if let Some(ref text) = real_text {
            let limit = text.len().min(1_000_000);
            stage_data[2].extend_from_slice(&text[..limit]);
            stage_data[3].extend_from_slice(&text[..text.len().min(2_000_000)]);
        }

        for (i, d) in stage_data.iter().enumerate() {
            eprintln!("  stage {i} ({}): {:.1}KB", stages[i].name, d.len() as f64 / 1024.0);
        }
        eprintln!();

        let reached = curriculum::run_curriculum(
            &mut w, &mut opt, &stage_data, &stages, context,
            &mut |msg| eprintln!("{msg}"),
        );

        eprintln!("\nReached stage {reached}/{}", stages.len());

        // Save
        w.save(&save_path).expect("save failed");
        opt.save(&opt_path).expect("opt save failed");
        eprintln!("Saved to {save_path}");

        // Final test report
        run_tests(&w);

    } else {
        // Raw text training
        let path = data_path.unwrap_or_else(|| {
            eprintln!("No --data or --curriculum specified. Use --help.");
            std::process::exit(1);
        });
        let mut text = Vec::new();
        std::fs::File::open(&path)
            .unwrap_or_else(|e| { eprintln!("Can't open {path}: {e}"); std::process::exit(1); })
            .read_to_end(&mut text).unwrap();
        eprintln!("data: {path} ({:.1}MB)", text.len() as f64 / 1e6);

        let mut grads = RegionalGradients::zeros(&w);
        let mut losses = Vec::new();

        for step in 0..steps {
            let offset = (step * context) % (text.len().saturating_sub(context + 1)).max(1);
            let chunk = &text[offset..offset + context + 1];

            grads.zero();
            let mut step_loss = 0.0f32;
            let mut correct = 0usize;

            for pos in 0..context {
                let token = chunk[pos] as usize;
                let target = chunk[pos + 1] as usize;
                let (loss, pred) = regional_train_token(&w, &mut grads, token, target);
                step_loss += loss;
                if pred == target { correct += 1; }
            }

            opt.step(&mut w, &grads);
            step_loss /= context as f32;
            losses.push(step_loss);

            if step % 200 == 0 || step == steps - 1 {
                let avg = if losses.len() >= 50 {
                    losses[losses.len()-50..].iter().sum::<f32>() / 50.0
                } else {
                    losses.iter().sum::<f32>() / losses.len() as f32
                };
                eprintln!("step {step:5}: loss={step_loss:.3} avg50={avg:.3} acc={correct}/{context}");
            }
        }

        w.save(&save_path).expect("save failed");
        opt.save(&opt_path).expect("opt save failed");
        eprintln!("Saved to {save_path}");
    }

    // Generation test
    eprintln!("\n--- Generation ---");
    for prompt in &["the ", "hello ", "in the "] {
        let mut nc = NeuralComputer::new(w.clone());
        let out = nc.chat(prompt, 60, 0.7);
        eprintln!("  \"{prompt}\" → \"{out}\"");
    }

    if chat {
        run_chat(&w);
    }
}

fn create_model(embed_dim: usize, regions: usize, ticks: usize) -> RegionalWeights {
    let cfg = if regions <= 4 {
        RegionalConfig::four_region(embed_dim, 256, ticks)
    } else {
        RegionalConfig::eight_region(embed_dim, 256, ticks)
    };
    RegionalWeights::new(cfg)
}

fn run_chat(w: &RegionalWeights) {
    let mut nc = NeuralComputer::new(w.clone());
    eprintln!("Chat mode. Type text, ctrl-D to quit.\n");
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        eprint!("> ");
        std::io::Write::flush(&mut std::io::stderr()).ok();
        line.clear();
        if stdin.read_line(&mut line).unwrap_or(0) == 0 { break; }
        let response = nc.chat(line.trim(), 100, 0.7);
        println!("{response}");
    }
}

fn run_tests(w: &RegionalWeights) {
    eprintln!("\n=== Capability Tests ===");

    let mut nc = NeuralComputer::new(w.clone());
    let r1 = challenges::challenge_byte_classes(&mut nc);
    eprintln!("  {}", r1.summary);

    let mut nc = NeuralComputer::new(w.clone());
    let r2 = challenges::challenge_bigrams(&mut nc);
    eprintln!("  {}", r2.summary);

    let mut nc = NeuralComputer::new(w.clone());
    let r3 = challenges::challenge_word_completion(&mut nc);
    eprintln!("  {}", r3.summary);

    let mut nc = NeuralComputer::new(w.clone());
    let r4 = challenges::challenge_coherent_generation(&mut nc);
    eprintln!("  {}", r4.summary);

    let total = r1.score + r2.score + r3.score + r4.score;
    eprintln!("  ────────────────────────");
    eprintln!("  Total: {:.0}% across 4 challenges", total / 4.0 * 100.0);
}

// ── Training data generators (inline for standalone example) ──

fn generate_byte_class_data(n: usize) -> Vec<u8> {
    cycle(&[
        b"Hello World This Is Text " as &[u8],
        b"age 25, score 100, year 2024 ",
        b"yes. no. wait! really? ok. ",
        b"line one.\nline two.\nline three.\n",
        b"Status: OK. Code: 200.\n",
    ], n)
}

fn generate_bigram_data(n: usize) -> Vec<u8> {
    cycle(&[
        b"the then them there these through " as &[u8],
        b"and another ancient android ",
        b"thing being ring sing string ",
        b"our out outer ours around ",
        b"better after under over ever ",
        b"at that cat bat mat hat ",
        b"to the to them to this ",
        b"is it is there is then ",
        b"on one stone honor alone ",
        b"enter entire went bent sent ",
    ], n)
}

fn generate_word_data(n: usize) -> Vec<u8> {
    cycle(&[
        b"the cat sat on the mat. the dog ran to the park. " as &[u8],
        b"it is a good day. she is not here. he is very tall. ",
        b"and then and after and before and again. ",
        b"of the best of a kind of my own. ",
        b"to the store to a friend to be or not to be. ",
        b"in the morning in a house in my room. ",
        b"she said hello. he said goodbye. they said nothing. ",
        b"the old man sat by the fire. the young girl read a book. ",
    ], n)
}

fn generate_coherent_data(n: usize) -> Vec<u8> {
    cycle(&[
        b"the cat sat on the mat. it was a warm day. the sun was bright. " as &[u8],
        b"the house was old and grey. it had a red door and two windows. ",
        b"hello said the boy. how are you asked the girl. i am fine. ",
        b"water is wet. fire is hot. ice is cold. the sky is blue. ",
        b"she went to the store. she bought bread and milk. then she went home. ",
        b"the bird sang a song. the flowers were in bloom. it was morning. ",
        b"the children played in the park. they ran and laughed. it was a good day. ",
        b"the moon rose over the hill. the stars came out. the night was still. ",
    ], n)
}

fn cycle(patterns: &[&[u8]], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while out.len() < n {
        let p = patterns[i % patterns.len()];
        let rem = n - out.len();
        out.extend_from_slice(if rem >= p.len() { p } else { &p[..rem] });
        i += 1;
    }
    out
}

fn print_help() {
    eprintln!("minictm — nanoGPT but for CTMs");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  minictm --curriculum                     # staged learning with test games");
    eprintln!("  minictm --data train.txt --steps 5000    # raw text training");
    eprintln!("  minictm -c model.bin --chat              # chat with trained model");
    eprintln!("  minictm -c model.bin --test              # run capability tests");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --curriculum       Use capability-gated staged training");
    eprintln!("  --data FILE        Train on raw text file");
    eprintln!("  --steps N          Training steps (default: 2000)");
    eprintln!("  --embed N          Embedding dimension (default: 32)");
    eprintln!("  --regions N        Number of CTM regions (default: 4)");
    eprintln!("  --ticks N          Inner ticks per region (default: 2)");
    eprintln!("  --lr F             Learning rate (default: 0.003)");
    eprintln!("  --context N        Context length (default: 32)");
    eprintln!("  --chat             Enter chat mode after training");
    eprintln!("  --test             Run capability tests on checkpoint");
    eprintln!("  -c, --checkpoint   Checkpoint path (default: minictm.bin)");
}
