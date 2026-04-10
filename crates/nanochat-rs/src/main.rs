//! nanochat — train a CTM on text, then chat with it.
//!
//! Minimal example of the modgrad SDK. No curriculum, no multimodal —
//! just raw text → regional CTM → next byte prediction.
//!
//! Usage:
//!   nanochat --data train.txt --steps 5000
//!   nanochat --data train.txt --steps 5000 --chat

use modgrad_runtime::regional::*;
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut data_path = "train_climbmix_5m.txt".to_string();
    let mut steps = 2000usize;
    let mut embed_dim = 32usize;
    let mut regions = 4usize;
    let mut ticks = 2usize;
    let mut lr = 3e-3f32;
    let mut context = 32usize;
    let mut chat = false;
    let mut checkpoint: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => { data_path = args[i+1].clone(); i += 2; }
            "--steps" => { steps = args[i+1].parse().unwrap(); i += 2; }
            "--embed" => { embed_dim = args[i+1].parse().unwrap(); i += 2; }
            "--regions" => { regions = args[i+1].parse().unwrap(); i += 2; }
            "--ticks" => { ticks = args[i+1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i+1].parse().unwrap(); i += 2; }
            "--context" => { context = args[i+1].parse().unwrap(); i += 2; }
            "--chat" => { chat = true; i += 1; }
            "--checkpoint" | "-c" => { checkpoint = Some(args[i+1].clone()); i += 2; }
            _ => { i += 1; }
        }
    }

    // Load or create model
    let mut w = if let Some(ref path) = checkpoint {
        if std::path::Path::new(path).exists() {
            eprintln!("Loading {path}...");
            RegionalWeights::load(path).expect("failed to load")
        } else {
            create_model(embed_dim, regions, ticks)
        }
    } else {
        create_model(embed_dim, regions, ticks)
    };
    let opt_path = checkpoint.as_deref().unwrap_or("nanochat.bin")
        .replace(".bin", ".opt.bin");
    let mut opt = if std::path::Path::new(&opt_path).exists() {
        RegionalAdamW::load(&opt_path).unwrap_or_else(|_| RegionalAdamW::new(&w))
    } else {
        RegionalAdamW::new(&w).with_lr(lr).with_clip(5.0)
    };

    w.print_summary();

    if chat {
        run_chat(&w);
        return;
    }

    // Load data
    let mut text = Vec::new();
    std::fs::File::open(&data_path)
        .unwrap_or_else(|e| { eprintln!("Can't open {data_path}: {e}"); std::process::exit(1); })
        .read_to_end(&mut text).unwrap();
    eprintln!("data: {data_path} ({:.1}MB)", text.len() as f64 / 1e6);

    // Train
    let mut grads = RegionalGradients::zeros(&w);
    let mut losses = Vec::new();

    for step in 0..steps {
        let offset = (step * context) % (text.len() - context - 1);
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

        if step % 100 == 0 || step == steps - 1 {
            let avg = if losses.len() >= 50 {
                losses[losses.len()-50..].iter().sum::<f32>() / 50.0
            } else {
                losses.iter().sum::<f32>() / losses.len() as f32
            };
            eprintln!("step {step:5}: loss={step_loss:.3} avg50={avg:.3} acc={correct}/{context}");
        }
    }

    // Save
    let save_path = checkpoint.as_deref().unwrap_or("nanochat.bin");
    w.save(save_path).expect("save failed");
    opt.save(&opt_path).expect("opt save failed");
    eprintln!("Saved to {save_path}");

    // Generate
    eprintln!("\n--- Generation ---");
    for prompt in &["the ", "hello ", "in the "] {
        let mut nc = NeuralComputer::new(w.clone());
        let out = nc.chat(prompt, 60, 0.7);
        eprintln!("  \"{prompt}\" → \"{out}\"");
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
