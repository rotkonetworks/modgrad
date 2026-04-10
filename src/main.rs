use modgrad::backend::{self, Backend, Tokenizer};
use modgrad::ctm;
use modgrad::memory::MemoryBank;
use modgrad::filter::BrainPipeline;
#[cfg(feature = "onnx")]
use modgrad::inference::OnnxBackend;
use modgrad::quantize::KeyFormat;

use clap::{Parser, Subcommand};
use std::io::{self, Write, BufRead};

/// ISIS — Intermediate System to Intermediate System
///
/// Neuromorphic computation: 8-region brain with gradient-free learning.
/// organism (guest) / host architecture.
#[derive(Parser)]
#[command(name = "isis", version, about, allow_external_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    // ─── Organism lifecycle ────────────────────────────
    /// Train regional CTM on staged byte curriculum (BPTT + AdamW)
    Train {
        /// Checkpoint path (size inferred from filename: tiny/small/medium/large)
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        /// External curriculum JSON path
        #[arg(short, long)]
        curriculum: Option<String>,
        /// Enable multimodal training (text + image + audio)
        #[arg(long)]
        multimodal: bool,
        /// Path to image directory (CIFAR-10 binary or directory of raw files)
        #[arg(long)]
        images: Option<String>,
        /// Path to audio directory (WAV files)
        #[arg(long)]
        audio: Option<String>,
        /// Path to video directory (frames + audio.wav per subdirectory)
        #[arg(long)]
        video: Option<String>,
        /// Video sampling FPS (default: 2)
        #[arg(long, default_value = "2.0")]
        video_fps: f32,
        /// Debug socket port (debugger connects for live training inspection)
        #[arg(long)]
        debug_port: Option<u16>,
    },
    /// Generate text from a trained organism
    Generate {
        /// Checkpoint path
        checkpoint: String,
        /// Prompt text
        #[arg(default_value = "the ")]
        prompt: String,
        /// Maximum tokens to generate
        #[arg(short, long, default_value = "100")]
        max_tokens: usize,
    },
    /// Run organism as a living daemon service
    Daemon {
        /// Checkpoint path
        #[arg(default_value = "organism.bin")]
        checkpoint: String,
        /// Port to listen on
        #[arg(short, long, default_value = "4747")]
        port: u16,
    },

    // ─── Neural Computer ──────────────────────────────
    /// Interactive neural computer mode (NC)
    Nc {
        /// Model checkpoint path
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        /// Temperature for sampling (0 = argmax)
        #[arg(short, long, default_value = "0.8")]
        temperature: f32,
        /// Maximum response tokens per turn
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Audio input: WAV file or device path
        #[arg(long)]
        audio: Option<String>,
        /// Camera input: directory of frame files
        #[arg(long)]
        camera: Option<String>,
        /// Camera FPS
        #[arg(long, default_value = "2.0")]
        camera_fps: f32,
        /// Audio output: path to write generated audio
        #[arg(long)]
        audio_out: Option<String>,
        /// Image output: directory to write generated frames
        #[arg(long)]
        image_out: Option<String>,
        /// Debug socket port (debugger connects here for live inspection)
        #[arg(long)]
        debug_port: Option<u16>,
    },

    // ─── Evaluation ────────────────────────────────────
    /// Run evaluation suites
    Eval {
        #[command(subcommand)]
        suite: EvalCommands,
    },

    // ─── Tuning ────────────────────────────────────────
    /// Parameter governance
    Tune {
        #[command(subcommand)]
        action: TuneCommands,
    },

    // ─── Device ────────────────────────────────────────
    /// Show available compute devices and region assignments
    Devices,

    // ─── Diagnostics ───────────────────────────────────
    /// Run diagnostics on a trained organism
    Diag {
        #[command(subcommand)]
        kind: DiagCommands,
    },

    // ─── Memory bank (legacy) ──────────────────────────
    /// Create empty memory bank
    New {
        #[arg(default_value = "isis.json")]
        path: String,
    },
    /// Show memory bank statistics
    Stats {
        #[arg(default_value = "isis.json")]
        path: String,
    },
    /// Convert between formats
    Convert {
        input: String,
        output: String,
        #[arg(default_value = "f32")]
        format: String,
    },
    /// Send command to running daemon
    Send {
        /// JSON command string
        json: String,
    },
    /// Interactive chat with memory
    Chat {
        #[arg(default_value = "models")]
        model_dir: String,
        #[arg(default_value = "isis.json")]
        bank: String,
    },
}

#[derive(Subcommand)]
enum EvalCommands {
    /// Architecture validation (hand-crafted features, fast)
    Arch,
    /// Extended architecture tests (11 tasks, 3 tiers)
    Extended {
        #[arg(short, long, default_value = "32")]
        d_model: usize,
    },
    /// Honest organism intelligence challenges (raw bytes)
    Challenge {
        /// DNA config: tiny, small, medium, medplus, large
        #[arg(short, long, default_value = "small")]
        dna: String,
    },
    /// Tick × dimension scaling analysis
    Scaling,
    /// QEC surface code decoding
    Qec {
        /// Weights save/load path
        #[arg(default_value = "datasets/qec_brain.json")]
        weights: String,
        /// Data directory
        #[arg(short, long, default_value = "datasets")]
        data_dir: String,
        /// CTM iterations
        #[arg(short, long, default_value = "5")]
        iterations: usize,
    },
    /// CTM ablation study
    Bench,
    /// CIFAR-10 retina → classify
    CifarRetina {
        #[arg(default_value = "cifar10_train_pixels.feat")]
        train: String,
        #[arg(default_value = "cifar10_test_pixels.feat")]
        test: String,
    },
    /// CIFAR-10 vision pipeline
    CifarVision {
        #[arg(default_value = "cifar10_train_pixels.feat")]
        train: String,
        #[arg(default_value = "cifar10_test_pixels.feat")]
        test: String,
    },
    /// CIFAR-10 direct features
    Cifar {
        #[arg(default_value = "cifar10_train.feat")]
        train: String,
        #[arg(default_value = "cifar10_test.feat")]
        test: String,
    },
}

#[derive(Subcommand)]
enum TuneCommands {
    /// Write default tuning config to file
    Init {
        #[arg(default_value = "tuning.json")]
        path: String,
    },
    /// Show params that differ from defaults
    Show {
        #[arg(default_value = "tuning.json")]
        path: String,
    },
}

#[derive(Subcommand)]
enum DiagCommands {
    /// Sync diversity across prompts
    Sync {
        #[arg(default_value = "organism.bin")]
        checkpoint: String,
    },
    /// Full pipeline introspection
    Deep {
        #[arg(default_value = "organism.bin")]
        checkpoint: String,
    },
    /// Angeris bound on output projection
    Angeris {
        #[arg(default_value = "organism.bin")]
        checkpoint: String,
    },
    /// Information flow at every connection point
    Pipeline {
        #[arg(default_value = "organism.bin")]
        checkpoint: String,
    },
}

fn main() {
    // Initialize GPU if available
    modgrad::gpu::init_global();

    let cli = Cli::parse();

    if let Some(cmd) = cli.command {
        match cmd {
            Commands::Train { checkpoint, curriculum, multimodal, images, audio, video, video_fps, debug_port } => {
                develop_staged(&checkpoint, curriculum.as_deref(), multimodal,
                    images.as_deref(), audio.as_deref(), video.as_deref(), video_fps, debug_port);
            }
            Commands::Generate { checkpoint, prompt, max_tokens } => {
                let mut org = modgrad::tabula_rasa::Organism::load(&checkpoint)
                    .expect("failed to load organism");
                let output = org.generate(prompt.as_bytes(), max_tokens);
                let text = String::from_utf8_lossy(&output);
                eprintln!("Prompt: {prompt:?}");
                eprintln!("Output: {text:?}");
            }
            Commands::Daemon { checkpoint, port } => {
                run_daemon(&checkpoint, port);
            }
            Commands::Nc { checkpoint, temperature, max_tokens, audio, camera, camera_fps, audio_out, image_out, debug_port } => {
                run_nc(&checkpoint, temperature, max_tokens,
                    audio.as_deref(), camera.as_deref(), camera_fps,
                    audio_out.as_deref(), image_out.as_deref(), debug_port);
            }
            Commands::Eval { suite } => match suite {
                EvalCommands::Arch => run_task_suite(),
                EvalCommands::Extended { d_model } => modgrad::tasks::run_extended_suite(d_model),
                EvalCommands::Challenge { dna } => modgrad::challenge::run_all_challenges(&dna),
                EvalCommands::Scaling => tick_analysis(),
                EvalCommands::Qec { weights, data_dir, iterations } =>
                    run_qec(&weights, &data_dir, iterations),
                EvalCommands::Bench => run_ctm_benchmark(),
                EvalCommands::CifarRetina { train, test } =>
                    train_retina_then_classify(&train, &test),
                EvalCommands::CifarVision { train, test } =>
                    train_cifar_vision(&train, &test),
                EvalCommands::Cifar { train, test } =>
                    train_cifar(&train, &test),
            },
            Commands::Tune { action } => match action {
                TuneCommands::Init { path } => {
                    match modgrad::tuning::TuningRegistry::write_defaults(&path) {
                        Ok(()) => eprintln!("Wrote default tuning config to {path}"),
                        Err(e) => eprintln!("Error: {e}"),
                    }
                }
                TuneCommands::Show { path } => {
                    let reg = modgrad::tuning::TuningRegistry::from_file(&path);
                    let diffs = reg.config.diff_from_default();
                    if diffs.is_empty() {
                        eprintln!("All parameters at defaults.");
                    } else {
                        eprintln!("{} params differ from default:", diffs.len());
                        for d in &diffs { eprintln!("  {d}"); }
                    }
                }
            },
            Commands::Devices => {
                let mesh = modgrad::device::DeviceMesh::auto();
                eprintln!("{}", mesh.summary());
            }
            Commands::Diag { kind } => match kind {
                DiagCommands::Sync { checkpoint } => sync_diagnostic(&checkpoint),
                DiagCommands::Deep { checkpoint } => deep_diagnostic(&checkpoint),
                DiagCommands::Angeris { checkpoint } => angeris_bound(&checkpoint),
                DiagCommands::Pipeline { checkpoint } => angeris_pipeline(&checkpoint),
            },
            Commands::New { path } => {
                let bank = MemoryBank::default();
                bank.write(&path).expect("failed to save");
                println!("Created new memory bank: {path}");
            }
            Commands::Stats { path } => {
                match MemoryBank::open(&path) {
                    Ok(bank) => print_stats(&bank, &path),
                    Err(e) => eprintln!("Error: {e}"),
                }
            }
            Commands::Convert { input, output, format } => {
                let fmt = match format.as_str() {
                    "f16" => KeyFormat::F16,
                    "i8" => KeyFormat::I8,
                    _ => KeyFormat::F32,
                };
                match MemoryBank::open(&input) {
                    Ok(bank) => {
                        if output.ends_with(".fb") {
                            bank.save_fb(&output, fmt).expect("failed to save");
                        } else {
                            bank.save(&output).expect("failed to save");
                        }
                        eprintln!("Converted {input} → {output}");
                    }
                    Err(e) => eprintln!("Error: {e}"),
                }
            }
            Commands::Send { json } => send_command(&json),
            Commands::Chat { model_dir, bank } => {
                chat(&model_dir, &bank);
            }
        }
    } else {
        Cli::parse_from(["isis", "--help"]);
    }
}

/// Derive default bank filename from model directory.
/// "models/qwen2.5-0.5b" → "qwen2.5-0.5b.isis.json"
/// "models" → "isis.json"
fn default_bank_name(model_dir: &str) -> String {
    let dir_name = std::path::Path::new(model_dir)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if dir_name == "models" || dir_name.is_empty() {
        "isis.json".into()
    } else {
        format!("{dir_name}.isis.json")
    }
}

fn print_stats(bank: &MemoryBank, path: &str) {
    let n_episodes: usize = bank.alters.iter().map(|a| a.episodes.len()).sum();
    let n_keys: usize = bank.alters.iter()
        .flat_map(|a| a.episodes.iter())
        .map(|e| e.keys.len()).sum();
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let fmt = if path.ends_with(".fb") { "flatbuf" } else { "json" };
    println!("{path} ({fmt}, {file_size} bytes)");
    println!("Model: {}", bank.model_id);
    println!("Alters: {}  Episodes: {}  Keys: {}  Rules: {}  Avoidances: {}",
             bank.alters.len(), n_episodes, n_keys,
             bank.rules.len(), bank.avoidances.len());
    for alter in &bank.alters {
        println!("  [{}] {} episodes", alter.name, alter.episodes.len());
        for ep in &alter.episodes {
            println!("    str={:.1} recalls={} \"{}\" → \"{}\"",
                     ep.strength, ep.recall_count,
                     &ep.prompt[..ep.prompt.len().min(35)], ep.answer);
        }
    }
}

fn chat(model_dir: &str, bank_path: &str) {
    // Load models
    let backbone_path = format!("{model_dir}/backbone.onnx");
    let lm_head_path = format!("{model_dir}/lm_head.onnx");
    let tokenizer_path = format!("{model_dir}/tokenizer/tokenizer.json");

    eprint!("Loading backbone...");
    io::stderr().flush().ok();
    let mut inference = match OnnxBackend::load(&backbone_path, &lm_head_path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("\nFailed to load models: {e}");
            eprintln!("Run: python3 export_onnx.py  (to create models/)");
            return;
        }
    };
    eprintln!(" done");

    eprint!("Loading tokenizer...");
    let tokenizer = match Tokenizer::load(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("\nFailed to load tokenizer: {e}");
            return;
        }
    };
    eprintln!(" done");

    // Load or create memory bank
    let bank = if std::path::Path::new(bank_path).exists() {
        let b = MemoryBank::open(bank_path).unwrap_or_default();
        let n: usize = b.alters.iter().map(|a| a.episodes.len()).sum();
        eprintln!("Loaded {n} memories from {bank_path}");
        eprintln!("Model: {}", b.model_id);
        b
    } else {
        eprintln!("New memory bank (no existing memories)");
        MemoryBank::default()
    };


    let mut pipeline = BrainPipeline::new(bank);

    eprintln!();
    eprintln!("Commands: teach <prompt>|<answer>  teach! (critical)  avoid  rule  save  stats  quit");
    eprintln!();

    let stdin = io::stdin();
    loop {
        print!(">>> ");
        io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() { continue; }

        if line == "quit" || line == "q" { break; }

        if line == "save" {
            pipeline.bank.write(bank_path).unwrap();
            let n: usize = pipeline.bank.alters.iter().map(|a| a.episodes.len()).sum();
            println!("  Saved {n} memories to {bank_path}");
            continue;
        }

        if line == "stats" {
            print_stats(&pipeline.bank, bank_path);
            continue;
        }

        if line.starts_with("rule ") {
            let instruction = &line[5..];
            pipeline.bank.add_rule(instruction, 1.0, "");
            println!("  Rule added: \"{instruction}\"");
            continue;
        }

        if line.starts_with("avoid ") {
            let parts: Vec<&str> = line[6..].splitn(2, '|').collect();
            if parts.len() < 2 {
                println!("  Usage: avoid <pattern> | <reason>");
                continue;
            }
            let pattern = parts[0].trim();
            let reason = parts[1].trim();
            let ids = tokenizer.encode(pattern);
            match inference.get_key(&ids) {
                Ok(key) => {
                    let suppress: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
                    pipeline.bank.add_avoidance(pattern, reason, key, suppress, 10.0);
                    println!("  Avoidance added: \"{pattern}\" ({reason})");
                }
                Err(e) => println!("  Error: {e}"),
            }
            continue;
        }

        let (is_teach, is_critical) = if line.starts_with("teach! ") {
            (true, true)
        } else if line.starts_with("teach ") {
            (true, false)
        } else {
            (false, false)
        };

        if is_teach {
            let rest = if is_critical { &line[7..] } else { &line[6..] };
            let parts: Vec<&str> = rest.splitn(2, '|').collect();
            if parts.len() < 2 {
                println!("  Usage: teach <prompt> | <answer>");
                continue;
            }
            let prompt = parts[0].trim();
            let answer = parts[1].trim();
            let importance = if is_critical { 3.0 } else { 1.0 };

            match backend::teach(&mut inference, &mut pipeline.bank, &tokenizer, prompt, answer, "default", importance) {
                Ok(()) => {
                    let ep = pipeline.bank.alters.last().unwrap().episodes.last().unwrap();
                    println!("  Stored: \"{}\" → \"{}\" (str={:.1})",
                             prompt, answer, ep.strength);
                }
                Err(e) => println!("  Error: {e}"),
            }
            continue;
        }

        // Default: generate (ask)
        let prompt = line;
        let ids = tokenizer.encode(prompt);
        match inference.get_key(&ids) {
            Ok(key) => {
                let response = pipeline.generate(
                    prompt, key, ids, 50,
                    |token_ids: &[i64]| -> Vec<Vec<f32>> {
                        inference.forward(token_ids).unwrap_or_default()
                    },
                );

                let output_ids = &response.token_ids[tokenizer.encode(prompt).len()..];
                let text = tokenizer.decode_i64(output_ids);

                if response.meta.gate > 0.0 {
                    let alter = response.meta.matched_alter.as_deref().unwrap_or("?");
                    print!("  [{alter} sim={:.2}] ", response.meta.match_similarity);
                } else if response.meta.avoided {
                    let reason = response.meta.avoidance_reason.as_deref().unwrap_or("?");
                    print!("  [BLOCKED: {reason}] ");
                } else {
                    print!("  ");
                }
                println!("{prompt}{text}");
            }
            Err(e) => println!("  Error computing hidden state: {e}"),
        }
    }

    // Auto-save on exit
    pipeline.bank.write(bank_path).ok();
    let n: usize = pipeline.bank.alters.iter().map(|a| a.episodes.len()).sum();
    eprintln!("Saved {n} memories to {bank_path}");
}
/// Run organism as a living daemon.
fn run_daemon(save_path: &str, port: u16) {
    use modgrad::tabula_rasa::{Organism, Dna};
    use modgrad::vocab::Vocab;
    use modgrad::daemon::Daemon;

    // Build vocab
    let text = std::fs::read_to_string("train_climbmix.txt")
        .unwrap_or_else(|_| "the cat sat on the mat".repeat(100));
    let vocab = Vocab::from_text(&text, 52);

    // Load or create organism
    let org = if std::path::Path::new(save_path).exists() {
        eprintln!("Loading organism from {save_path}...");
        Organism::load(save_path).expect("failed to load")
    } else {
        eprintln!("Creating new organism...");
        let mut dna = Dna::small();
        dna.vocab_size = vocab.size();
        Organism::new(dna)
    };

    let mut daemon = Daemon::new(org, vocab, save_path.into());

    // TODO: landlock/seccomp sandbox here — after loading model, before accepting connections
    // Similar to llamafile's pledge() after model load.

    daemon.run(port);
}

/// Send a command to the running daemon.
fn send_command(json: &str) {
    use std::io::{BufRead, BufReader, Write};
    match std::net::TcpStream::connect("127.0.0.1:7377") {
        Ok(mut stream) => {
            writeln!(stream, "{json}").ok();
            let mut reader = BufReader::new(stream);
            let mut response = String::new();
            reader.read_line(&mut response).ok();
            println!("{response}");
        }
        Err(_) => {
            eprintln!("Can't connect to daemon. Run: isis daemon [checkpoint]");
        }
    }
}

/// Staged development: train regional CTM through byte-first curriculum phases.
/// Uses BPTT + AdamW. No Hebbian, no sleep, no biological learning rules.
/// Generate synthetic paired multimodal training sequences.
///
/// These teach the model cross-modal associations:
///   text → image:  "a picture of a cat " <img> [VQ codes] </img>
///   text → audio:  "hello world " <aud> [audio codes] </aud>
///   image → text:  <img> [codes] </img> " this is a cat"
///   text → action: "click the center " <act> left_click 0.5 0.5 </act>
///   audio → text:  <aud> [codes] </aud> " the sound was speech"
fn generate_multimodal_pairs() -> Vec<Vec<usize>> {
    use modgrad_runtime::regional::*;
    use modgrad_codec::vqvae::VqVae;
    use modgrad_codec::audio_codec::AudioCodec;
    use modgrad_compute::neuron::SimpleRng;

    let vae = VqVae::new(4096, 64);
    let audio_codec = AudioCodec::new(4096, 64, 24000);
    let mut rng = SimpleRng::new(1337);
    let mut pairs = Vec::new();

    // ── Text → Image pairs (synthetic "images" with CIFAR-10 class names) ──
    let class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"];

    for (label, name) in class_names.iter().enumerate() {
        // Generate synthetic pixel patterns per class (deterministic from label)
        for variant in 0..20 {
            let mut pixels = vec![0.0f32; 3072];
            let seed = (label * 100 + variant) as f32;
            for i in 0..3072 {
                pixels[i] = ((seed + i as f32 * 0.1).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
            }
            let codes = vae.tokenize(&pixels);
            let img_tokens = image_codes_to_tokens(&codes);

            // "a picture of a cat " <img> [codes] </img>
            let mut seq = text_to_tokens(format!("a picture of a {name} ").as_bytes());
            seq.extend(&img_tokens);
            pairs.push(seq);

            // <img> [codes] </img> " this is a cat"
            let mut seq = img_tokens.clone();
            seq.extend(text_to_tokens(format!(" this is a {name}").as_bytes()));
            pairs.push(seq);

            // Describe and show: "here is a cat: " <img> ... </img> " it has fur"
            let descriptions = [
                format!("{name} in a photo"),
                format!("image of {name}"),
                format!("you can see a {name} here"),
            ];
            let desc = &descriptions[variant % descriptions.len()];
            let mut seq = text_to_tokens(desc.as_bytes());
            seq.push(b' ' as usize);
            seq.extend(&img_tokens);
            pairs.push(seq);
        }
    }

    // ── Text → Audio pairs (synthetic waveforms) ──
    let audio_descriptions = [
        ("a sine tone", 440.0),
        ("a low hum", 120.0),
        ("a high pitch", 2000.0),
        ("a beep", 800.0),
        ("silence", 0.0),
    ];

    for (desc, freq) in &audio_descriptions {
        // Generate 0.5 seconds of synthetic audio at 24kHz
        let n_samples = 12000;
        let waveform: Vec<f32> = (0..n_samples).map(|i| {
            if *freq > 0.0 {
                (2.0 * std::f32::consts::PI * freq * i as f32 / 24000.0).sin() * 0.5
            } else {
                0.0
            }
        }).collect();

        let codes = audio_codec.tokenize(&waveform);
        let aud_tokens = audio_codes_to_tokens(&codes);

        // "a sine tone " <aud> [codes] </aud>
        let mut seq = text_to_tokens(format!("{desc} ").as_bytes());
        seq.extend(&aud_tokens);
        pairs.push(seq);

        // <aud> [codes] </aud> " that was a sine tone"
        let mut seq = aud_tokens.clone();
        seq.extend(text_to_tokens(format!(" that was {desc}").as_bytes()));
        pairs.push(seq);
    }

    // ── Text → Action pairs (GUI interaction patterns) ──
    let action_patterns: Vec<(&str, Vec<usize>)> = vec![
        ("click the center of the screen",
            action_click(0.5, 0.5)),
        ("click the top left corner",
            action_click(0.0, 0.0)),
        ("click the bottom right",
            action_click(1.0, 1.0)),
        ("move the mouse to the middle",
            action_mouse_move(0.5, 0.5)),
        ("type hello world",
            action_type_text("hello world")),
        ("press enter",
            action_key(ACT_KEY_ENTER)),
        ("press escape",
            action_key(ACT_KEY_ESCAPE)),
        ("press tab",
            action_key(ACT_KEY_TAB)),
        ("scroll up",
            action_key(ACT_SCROLL_UP)),
        ("scroll down",
            action_key(ACT_SCROLL_DOWN)),
        ("press ctrl c",
            action_modified_key(ACT_KEY_CTRL, b'c')),
        ("press ctrl v to paste",
            action_modified_key(ACT_KEY_CTRL, b'v')),
        ("open a new tab",
            action_modified_key(ACT_KEY_CTRL, b't')),
        ("close the window",
            action_modified_key(ACT_KEY_ALT, b'F')),  // alt+F4 approximation
        ("navigate up",
            action_key(ACT_KEY_UP)),
        ("navigate down",
            action_key(ACT_KEY_DOWN)),
    ];

    for (desc, action) in &action_patterns {
        // "click the center " <act> left_click 0.5 0.5 </act>
        let mut seq = text_to_tokens(format!("{desc} ").as_bytes());
        seq.extend(action);
        pairs.push(seq);

        // Variant: "please {action}" for instruction-following
        let mut seq = text_to_tokens(format!("please {desc} ").as_bytes());
        seq.extend(action);
        pairs.push(seq);
    }

    // ── Conversation patterns (text ↔ text with multimodal context) ──
    let conversations = [
        "what do you see? a cat in the image",
        "describe the sound. it is a high pitched tone",
        "what should i click? click the button in the center",
        "how does it look? the image shows a red car",
        "repeat after me: hello. hello",
    ];
    for conv in &conversations {
        for repeat in 0..5 {
            let mut seq = text_to_tokens(conv.as_bytes());
            // Add slight variation with a trailing space count
            for _ in 0..repeat { seq.push(b' ' as usize); }
            pairs.push(seq);
        }
    }

    // ── Multimodal chains (text → image → text → action) ──
    for (label, name) in class_names.iter().enumerate() {
        let mut pixels = vec![0.0f32; 3072];
        for i in 0..3072 { pixels[i] = ((label as f32 + i as f32 * 0.07).sin() * 0.5 + 0.5).clamp(0.0, 1.0); }
        let codes = vae.tokenize(&pixels);
        let img_tokens = image_codes_to_tokens(&codes);

        // "show me a {name}" → <img> → "now click on it" → <act>
        let mut seq = text_to_tokens(format!("show me a {name} ").as_bytes());
        seq.extend(&img_tokens);
        seq.extend(text_to_tokens(b" now click on it "));
        seq.extend(&action_click(0.5, 0.5));
        pairs.push(seq);
    }

    eprintln!("  Generated {} multimodal paired sequences", pairs.len());
    pairs
}

fn develop_staged(
    save_path: &str,
    curriculum_path: Option<&str>,
    multimodal: bool,
    images_path: Option<&str>,
    audio_path: Option<&str>,
    video_path: Option<&str>,
    video_fps: f32,
    debug_port: Option<u16>,
) {
    use modgrad_runtime::regional::*;
    use modgrad_runtime::curriculum;

    // Model size from filename
    let (embed_dim, n_regions, ticks, context_len) = if save_path.contains("large") {
        (128, 8, 4, 128)
    } else if save_path.contains("medium") {
        (64, 8, 3, 64)
    } else if save_path.contains("tiny") {
        (16, 4, 2, 16)
    } else {
        (32, 8, 2, 32) // small (default)
    };
    let vocab_size = if multimodal { VOCAB_MULTIMODAL } else { VOCAB_TEXT };

    // Load or create model
    let mut w = if std::path::Path::new(save_path).exists() {
        eprintln!("Loading model from {save_path}...");
        RegionalWeights::load(save_path).expect("failed to load model")
    } else {
        let cfg = match (n_regions, multimodal) {
            (4, true) => RegionalConfig::four_region_multimodal(embed_dim, ticks),
            (4, false) => RegionalConfig::four_region(embed_dim, vocab_size, ticks),
            (_, true) => RegionalConfig::eight_region_multimodal(embed_dim, ticks),
            (_, false) => RegionalConfig::eight_region(embed_dim, vocab_size, ticks),
        };
        RegionalWeights::new(cfg)
    };
    w.print_summary();

    // Load or create AdamW optimizer state
    let opt_path = save_path.replace(".bin", ".opt.bin").replace(".json", ".opt.bin");
    let mut opt = if std::path::Path::new(&opt_path).exists() {
        RegionalAdamW::load(&opt_path).unwrap_or_else(|_| RegionalAdamW::new(&w))
    } else {
        RegionalAdamW::new(&w)
            .with_lr(3e-3)
            .with_wd(0.001)
            .with_clip(5.0)
    };
    eprintln!("  AdamW: lr={}, wd={}, clip={}, step={}",
        opt.lr, opt.weight_decay, opt.grad_clip, opt.step);

    // Ctrl+C handler for clean save
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nCtrl+C — saving and exiting...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    }).ok();

    // Debug server — lets the debugger watch training live
    let debug_nc: Option<(
        std::sync::Arc<std::sync::Mutex<modgrad_runtime::nc_socket::NcDebugView>>,
    )> = if let Some(port) = debug_port {
        use modgrad_runtime::nc_socket;
        // Create a temporary NC view from the weights for the debug server
        let nc_tmp = NeuralComputer::new(w.clone());
        let view = nc_socket::NcDebugView::from_nc(&nc_tmp);
        let view = std::sync::Arc::new(std::sync::Mutex::new(view));
        let _handle = nc_socket::start_debug_server(port, view.clone());
        Some((view,))
    } else {
        None
    };

    // Helper: update debug view during training
    let update_train_debug = |w: &RegionalWeights, step: usize, loss: f32,
                               debug: &Option<(std::sync::Arc<std::sync::Mutex<modgrad_runtime::nc_socket::NcDebugView>>,)>| {
        if let Some((view,)) = debug {
            if step % 10 == 0 { // update every 10 steps to avoid overhead
                if let Ok(mut guard) = view.try_lock() {
                    // Update region params (weights may have changed)
                    guard.region_params = w.regions.iter().map(|r| r.n_params()).collect();
                    guard.total_params = w.n_params();
                    // Store step count in history for the debugger to read
                    guard.history = vec![step];
                }
            }
        }
    };

    // Load external curriculum if provided
    let external_data = curriculum_path.and_then(|path| {
        eprintln!("Loading external curriculum from {path}...");
        match curriculum::load_external(path) {
            Ok(data) => {
                eprintln!("  {} items loaded", data.len());
                Some(data)
            }
            Err(e) => {
                eprintln!("  Warning: {e}, using built-in only");
                None
            }
        }
    });

    let max_steps_per_phase = 2000;
    let reps = 5;

    // Try to load real-world text for later phases
    let real_text = {
        let candidates = [
            "train_climbmix_5m.txt", "train_climbmix.txt",
            "train_large.txt", "train_stories.txt",
        ];
        let mut text = Vec::new();
        for c in &candidates {
            let path = format!("{}/{c}", std::path::Path::new(save_path).parent()
                .unwrap_or(std::path::Path::new(".")).display());
            if let Ok(data) = std::fs::read(&path) {
                eprintln!("Loaded real text: {path} ({} bytes)", data.len());
                text = data;
                break;
            }
        }
        text
    };

    // Load multimodal data if requested
    let image_tokens: Vec<Vec<usize>> = if multimodal {
        if let Some(img_dir) = images_path {
            load_image_tokens(img_dir)
        } else {
            eprintln!("  No --images path, generating synthetic image tokens");
            // Synthetic: random VQ codes for testing the pipeline
            (0..100).map(|i| {
                let mut codes = vec![TOKEN_IMG_START];
                for j in 0..64 { codes.push(TOKEN_IMG_OFFSET + (i * 64 + j) % TOKEN_IMG_CODES); }
                codes.push(TOKEN_IMG_END);
                codes
            }).collect()
        }
    } else {
        Vec::new()
    };

    let audio_tokens: Vec<Vec<usize>> = if multimodal {
        if let Some(aud_dir) = audio_path {
            load_audio_tokens(aud_dir)
        } else {
            eprintln!("  No --audio path, generating synthetic audio tokens");
            (0..50).map(|i| {
                let mut codes = vec![TOKEN_AUD_START];
                for j in 0..75 { codes.push(TOKEN_AUD_OFFSET + (i * 75 + j) % TOKEN_AUD_CODES); }
                codes.push(TOKEN_AUD_END);
                codes
            }).collect()
        }
    } else {
        Vec::new()
    };

    // Load video data: each video → one token sequence with timestamps
    let video_tokens: Vec<Vec<usize>> = if multimodal {
        if let Some(vid_path) = video_path {
            load_video_tokens(vid_path, video_fps)
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    // Generate synthetic paired training data (text↔image, text↔audio, text↔action)
    let paired_data: Vec<Vec<usize>> = if multimodal {
        generate_multimodal_pairs()
    } else {
        Vec::new()
    };

    if multimodal {
        eprintln!("  Multimodal: {} images, {} audio, {} video, {} paired sequences",
            image_tokens.len(), audio_tokens.len(), video_tokens.len(), paired_data.len());
    }

    let mut total_tokens = opt.step as u64;

    for phase in 0..curriculum::NUM_PHASES {
        if !running.load(std::sync::atomic::Ordering::SeqCst) { break; }

        let (phase_name, mut phase_data) = curriculum::generate(phase, reps);

        if let Some(ref ext) = external_data {
            for (p, data) in ext {
                if *p == phase {
                    phase_data.extend_from_slice(data);
                }
            }
        }

        // Mix in real text from phase 0 — the model needs real English
        if !real_text.is_empty() {
            let limit = match phase {
                0..=2 => (100_000).min(real_text.len()),   // 100KB early
                3..=4 => (500_000).min(real_text.len()),   // 500KB mid
                _ => (2_000_000).min(real_text.len()),      // 2MB late
            };
            phase_data.extend_from_slice(&real_text[..limit]);
            eprintln!("  + {limit} bytes of real text mixed in");
        }

        eprintln!("\n============================================================");
        eprintln!("=== PHASE {phase}: {phase_name} ({} bytes) ===", phase_data.len());
        eprintln!("  Mastery threshold: {:.1}, streak needed: {}",
            curriculum::MASTERY_THRESHOLDS[phase], curriculum::MASTERY_STREAK);

        // Build training sequences as token indices
        let mut sequences: Vec<Vec<usize>> = phase_data.chunks(context_len)
            .map(|c| c.iter().map(|&b| b as usize).collect())
            .collect();

        // In later phases with multimodal, mix in paired data + raw media.
        // Gradual introduction: phase 5 = 10% paired, phase 6 = 25%, phase 7 = 50%.
        // This prevents loss spikes from suddenly seeing unknown token ranges.
        if multimodal && phase >= 5 {
            let n_text = sequences.len();
            let ratio = match phase {
                5 => 10,  // 1 multimodal per 10 text sequences
                6 => 4,   // 1 per 4
                _ => 2,   // 1 per 2
            };
            let n_pairs = (n_text / ratio).min(paired_data.len());
            for i in 0..n_pairs {
                sequences.push(paired_data[i].clone());
            }

            // Raw media sequences only in final phases
            if phase >= 6 {
                let n_img = image_tokens.len();
                let n_aud = audio_tokens.len();
                let n_vid = video_tokens.len();
                let n_media = n_img + n_aud + n_vid;
                for i in 0..(n_text / 4).min(n_media.max(1)) {
                    if i < n_img {
                        sequences.push(image_tokens[i % n_img].clone());
                    } else if i < n_img + n_aud {
                        sequences.push(audio_tokens[(i - n_img) % n_aud.max(1)].clone());
                    } else if n_vid > 0 {
                        sequences.push(video_tokens[(i - n_img - n_aud) % n_vid].clone());
                    }
                }
            }
        }

        if sequences.is_empty() { continue; }

        let mut mastery_streak = 0;
        let mut step = 0;
        let mut phase_losses = Vec::new();
        let threshold = curriculum::MASTERY_THRESHOLDS[phase];

        let mut grads = RegionalGradients::zeros(&w); // allocate once, reuse
        while step < max_steps_per_phase {
            if !running.load(std::sync::atomic::Ordering::SeqCst) { break; }

            let chunk_idx = step % sequences.len();
            let seq = &sequences[chunk_idx];
            if seq.len() < 2 { step += 1; continue; }

            grads.zero(); // reuse buffer — no allocation
            let mut chunk_loss = 0.0f32;
            let mut chunk_correct = 0usize;
            let n_tokens = seq.len() - 1;

            for pos in 0..n_tokens {
                let token = seq[pos];
                let target = seq[pos + 1];
                let (loss, pred) = regional_train_token(&w, &mut grads, token, target);
                chunk_loss += loss;
                if pred == target { chunk_correct += 1; }
            }

            // AdamW step (no gradient averaging — AdamW adapts to gradient scale)
            opt.step(&mut w, &grads);

            chunk_loss /= n_tokens as f32;
            phase_losses.push(chunk_loss);
            total_tokens += n_tokens as u64;
            update_train_debug(&w, step, chunk_loss, &debug_nc);

            if chunk_loss < threshold {
                mastery_streak += 1;
            } else {
                mastery_streak = 0;
            }

            if step % 50 == 0 || mastery_streak >= curriculum::MASTERY_STREAK {
                let recent: f32 = if phase_losses.len() >= 10 {
                    phase_losses[phase_losses.len()-10..].iter().sum::<f32>() / 10.0
                } else {
                    phase_losses.iter().sum::<f32>() / phase_losses.len() as f32
                };
                eprintln!("  step {step:4}: loss={chunk_loss:.3} avg10={recent:.3} acc={chunk_correct}/{n_tokens} streak={mastery_streak}/{}",
                    curriculum::MASTERY_STREAK);
            }

            if mastery_streak >= curriculum::MASTERY_STREAK {
                eprintln!("  MASTERED phase {phase} ({phase_name}) at step {step}!");
                break;
            }

            step += 1;
        }

        if mastery_streak < curriculum::MASTERY_STREAK {
            let avg: f32 = phase_losses.iter().sum::<f32>() / phase_losses.len().max(1) as f32;
            eprintln!("  Phase {phase} not mastered after {max_steps_per_phase} steps (avg loss={avg:.3})");
        }

        // Save checkpoint after each phase
        w.save(save_path).expect("failed to save model");
        opt.save(&opt_path).expect("failed to save optimizer");
        eprintln!("  Checkpoint saved to {save_path} ({total_tokens} tokens trained)");
    }

    // Final generation test
    eprintln!("\n=== Generation test ===");
    for prompt in &[b"the " as &[u8], b"0x" as &[u8], b"fn " as &[u8], b"ssh " as &[u8]] {
        let mut generated = prompt.to_vec();
        let mut state = RegionalState::new(&w);
        // Feed prompt
        for &b in prompt.iter() {
            let obs = w.embed(b as usize).to_vec();
            let _ = regional_forward(&w, &mut state, &obs);
        }
        // Generate
        for _ in 0..30 {
            let obs = w.embed(*generated.last().unwrap() as usize).to_vec();
            let out = regional_forward(&w, &mut state, &obs);
            if let Some(logits) = out.predictions.last() {
                let next = logits.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as u8).unwrap_or(b' ');
                generated.push(next);
            }
        }
        let prompt_str = std::str::from_utf8(prompt).unwrap_or("?");
        let output_str = String::from_utf8_lossy(&generated[prompt.len()..]);
        eprintln!("  \"{prompt_str}\" -> \"{output_str}\"");
    }

    w.save(save_path).expect("failed to save");
    opt.save(&opt_path).expect("failed to save optimizer");
    let size = std::fs::metadata(save_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("\nFinal save to {save_path} ({size} bytes, {total_tokens} tokens)");
}

/// Sync diversity diagnostic: measure how different sync patterns are across prompts.
fn sync_diagnostic(org_path: &str) {
    use modgrad::tabula_rasa::{Organism, Dna};

    let mut org = if std::path::Path::new(org_path).exists() {
        Organism::load(org_path).expect("failed to load organism")
    } else {
        eprintln!("No organism at {org_path}, creating fresh one for diagnostic");
        Organism::new(Dna::small())
    };

    // Test with single-byte tokens (byte-level vocab) and short sequences
    let test_prompts: Vec<(&str, Vec<usize>)> = vec![
        ("'t','h','e'", vec![b't' as usize, b'h' as usize, b'e' as usize]),
        ("'c','a','t'", vec![b'c' as usize, b'a' as usize, b't' as usize]),
        ("'d','o','g'", vec![b'd' as usize, b'o' as usize, b'g' as usize]),
        ("'r','u','n'", vec![b'r' as usize, b'u' as usize, b'n' as usize]),
        ("'b','i','g'", vec![b'b' as usize, b'i' as usize, b'g' as usize]),
        ("'r','e','d'", vec![b'r' as usize, b'e' as usize, b'd' as usize]),
        ("'a','n','d'", vec![b'a' as usize, b'n' as usize, b'd' as usize]),
        ("'b','u','t'", vec![b'b' as usize, b'u' as usize, b't' as usize]),
        // Also test single bytes to see baseline diversity
        ("byte 'a'", vec![b'a' as usize]),
        ("byte 'z'", vec![b'z' as usize]),
        ("byte '0'", vec![b'0' as usize]),
        ("byte ' '", vec![b' ' as usize]),
    ];

    let total_neurons: usize = org.dna.ctm.input_layer.n_neurons
        + org.dna.ctm.attention_layer.n_neurons
        + org.dna.ctm.output_layer.n_neurons
        + org.dna.ctm.motor_layer.n_neurons;

    eprintln!("=== Sync Diversity Diagnostic ===");
    eprintln!("Organism: {} params, {} total neurons, {} ticks",
        org.param_count(), total_neurons, org.dna.ctm.iterations);
    eprintln!("Sync dim: {} (output), {} (action)",
        org.dna.ctm.n_sync_out, org.dna.ctm.n_sync_action);
    eprintln!("Testing {} prompts...\n", test_prompts.len());

    // Collect sync for each prompt
    let mut syncs: Vec<(String, Vec<f32>)> = Vec::new();
    for (name, ids) in &test_prompts {
        let (_, sync_vecs) = org.forward_inner(ids, false);
        if let Some(sync) = sync_vecs.last() {
            syncs.push((name.to_string(), sync.clone()));
        }
    }

    if syncs.len() < 2 {
        eprintln!("Not enough words encoded. Vocab too small?");
        return;
    }

    // Compute pairwise cosine similarities
    let mut similarities: Vec<f32> = Vec::new();
    let mut max_sim = f32::MIN;
    let mut min_sim = f32::MAX;
    let mut max_pair = (String::new(), String::new());
    let mut min_pair = (String::new(), String::new());

    eprintln!("Pairwise cosine similarities:");
    for i in 0..syncs.len() {
        for j in (i + 1)..syncs.len() {
            let cos = cosine_sim(&syncs[i].1, &syncs[j].1);
            similarities.push(cos);
            if cos > max_sim {
                max_sim = cos;
                max_pair = (syncs[i].0.clone(), syncs[j].0.clone());
            }
            if cos < min_sim {
                min_sim = cos;
                min_pair = (syncs[i].0.clone(), syncs[j].0.clone());
            }
        }
    }

    let mean_sim: f32 = similarities.iter().sum::<f32>() / similarities.len() as f32;
    let var: f32 = similarities.iter().map(|s| (s - mean_sim).powi(2)).sum::<f32>() / similarities.len() as f32;
    let std_sim = var.sqrt();

    eprintln!("  Mean:  {mean_sim:.4}");
    eprintln!("  Std:   {std_sim:.4}");
    eprintln!("  Min:   {min_sim:.4} ({} vs {})", min_pair.0, min_pair.1);
    eprintln!("  Max:   {max_sim:.4} ({} vs {})", max_pair.0, max_pair.1);

    // Sync vector stats
    let sync_norms: Vec<f32> = syncs.iter()
        .map(|(_, s)| s.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();
    let mean_norm: f32 = sync_norms.iter().sum::<f32>() / sync_norms.len() as f32;
    let max_norm: f32 = sync_norms.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_norm: f32 = sync_norms.iter().fold(f32::MAX, |a, &b| a.min(b));

    eprintln!("\nSync vector norms:");
    eprintln!("  Mean: {mean_norm:.4}, Min: {min_norm:.4}, Max: {max_norm:.4}");

    // Sparsity: fraction of near-zero elements
    let sparsity: f32 = syncs.iter()
        .map(|(_, s)| s.iter().filter(|&&x| x.abs() < 0.01).count() as f32 / s.len() as f32)
        .sum::<f32>() / syncs.len() as f32;
    eprintln!("  Sparsity (|x|<0.01): {:.1}%", sparsity * 100.0);

    // Active dimensions: how many sync dimensions vary across prompts
    let n_dims = syncs[0].1.len();
    let mut dim_vars = vec![0.0f32; n_dims];
    for d in 0..n_dims {
        let vals: Vec<f32> = syncs.iter().map(|(_, s)| s[d]).collect();
        let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
        dim_vars[d] = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    }
    let active_dims = dim_vars.iter().filter(|&&v| v > 0.001).count();
    eprintln!("  Active dimensions (var>0.001): {active_dims}/{n_dims}");

    // Verdict
    eprintln!("\n=== VERDICT ===");
    if mean_sim > 0.95 {
        eprintln!("COLLAPSED: mean cosine {mean_sim:.4} > 0.95 — all syncs nearly identical");
        eprintln!("  The output projection CANNOT distinguish between words.");
    } else if mean_sim > 0.8 {
        eprintln!("LOW DIVERSITY: mean cosine {mean_sim:.4} — syncs too similar");
        eprintln!("  Output projection will struggle with fine discrimination.");
    } else if mean_sim > 0.5 {
        eprintln!("MODERATE: mean cosine {mean_sim:.4} — some diversity present");
        eprintln!("  Output projection can learn coarse categories but not all words.");
    } else {
        eprintln!("GOOD: mean cosine {mean_sim:.4} — syncs are diverse");
        eprintln!("  Output projection has enough signal to discriminate words.");
    }

    // Print top-5 most similar and most different pairs
    let mut pairs: Vec<(f32, String, String)> = Vec::new();
    let mut idx = 0;
    for i in 0..syncs.len() {
        for j in (i + 1)..syncs.len() {
            pairs.push((similarities[idx], syncs[i].0.clone(), syncs[j].0.clone()));
            idx += 1;
        }
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    eprintln!("\nMost similar pairs:");
    for (sim, a, b) in pairs.iter().take(5) {
        eprintln!("  {sim:.4}  {a} / {b}");
    }
    eprintln!("\nMost different pairs:");
    for (sim, a, b) in pairs.iter().rev().take(5) {
        eprintln!("  {sim:.4}  {a} / {b}");
    }
}

/// Tick × dimension analysis: does scale unlock tick usefulness?
/// Tests if ticks become valuable at larger neuron counts.
fn tick_analysis() {
    use modgrad::ctm::{Ctm, CtmConfig, LayerConfig};
    use modgrad::tasks;

    eprintln!("=== TICK × DIMENSION ANALYSIS ===");
    eprintln!("Do ticks become useful at larger neuron counts?\n");

    let neuron_counts = [16, 32, 64, 128];
    let tick_counts = [1, 2, 4, 8, 16];

    // Test on maze (most likely to benefit from deliberation)
    eprintln!("--- 5x5 Maze: Angeris-optimal accuracy ---");
    eprintln!("  neurons |  1 tick |  2 tick |  4 tick |  8 tick | 16 tick");

    for &n in &neuron_counts {
        let mut row = format!("  {:>6} |", n * 4);

        for &k in &tick_counts {
            let base = LayerConfig {
                n_neurons: n, memory_length: 4, nlm_depth: 1,
                hebbian_lr: 0.01, inhibitory_fraction: 0.2,
                ..Default::default()
            };
            let d_input = n.max(32); // at least 32 for maze encoding
            let sync_dim = n; // sync_dim = neurons per region

            let cfg = CtmConfig {
                iterations: k,
                d_model: n * 4, d_input,
                heads: 2, n_sync_out: sync_dim, n_sync_action: sync_dim / 2,
                synapse_depth: 1, out_dims: sync_dim,
                global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,                input_layer: LayerConfig { receives_broadcast: true, ..base.clone() },
                attention_layer: LayerConfig { receives_broadcast: false, ..base.clone() },
                output_layer: LayerConfig { receives_broadcast: true, ..base.clone() },
                motor_layer: base.clone(),
        cerebellum_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        basal_ganglia_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        insula_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        hippocampus_layer: LayerConfig { n_neurons: 16, receives_broadcast: true, ..Default::default() },
                ..CtmConfig::default()
            };

            let mut ctm = Ctm::new(cfg.clone());
            ctm.enable_hebbian();
            let examples = tasks::maze_examples(5, cfg.d_input, 1000);
            let (_, _, _, opt_acc) = tasks::train_ctm_on_task(&mut ctm, &examples, 20, 100);
            row.push_str(&format!(" {:5.1}% |", opt_acc * 100.0));
        }
        eprintln!("{row}");
    }

    // Also test parity at different scales
    eprintln!("\n--- 5-bit Parity: Angeris-optimal accuracy ---");
    eprintln!("  neurons |  1 tick |  2 tick |  4 tick |  8 tick | 16 tick");

    for &n in &neuron_counts {
        let mut row = format!("  {:>6} |", n * 4);

        for &k in &tick_counts {
            let base = LayerConfig {
                n_neurons: n, memory_length: 4, nlm_depth: 1,
                hebbian_lr: 0.01, inhibitory_fraction: 0.2,
                ..Default::default()
            };
            let d_input = n.max(16);
            let sync_dim = n;

            let cfg = CtmConfig {
                iterations: k,
                d_model: n * 4, d_input,
                heads: 2, n_sync_out: sync_dim, n_sync_action: sync_dim / 2,
                synapse_depth: 1, out_dims: sync_dim,
                global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,                input_layer: LayerConfig { receives_broadcast: true, ..base.clone() },
                attention_layer: LayerConfig { receives_broadcast: false, ..base.clone() },
                output_layer: LayerConfig { receives_broadcast: true, ..base.clone() },
                motor_layer: base.clone(),
        cerebellum_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        basal_ganglia_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        insula_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        hippocampus_layer: LayerConfig { n_neurons: 16, receives_broadcast: true, ..Default::default() },
                ..CtmConfig::default()
            };

            let mut ctm = Ctm::new(cfg.clone());
            ctm.enable_hebbian();
            let examples = tasks::parity_examples(5, cfg.d_input);
            let (_, _, _, opt_acc) = tasks::train_ctm_on_task(&mut ctm, &examples, 20, 100);
            row.push_str(&format!(" {:5.1}% |", opt_acc * 100.0));
        }
        eprintln!("{row}");
    }

    eprintln!("\n=== END ANALYSIS ===");
}

/// Train visual retina with Hebbian sparse coding, then classify CIFAR-10.
/// Phase 1: Train V1→V2→V4 on raw pixels (unsupervised, no labels).
/// Phase 2: Run trained retina + CTM on CIFAR-10 (supervised readout).
fn train_retina_then_classify(train_pixels_path: &str, test_pixels_path: &str) {
    use modgrad::ctm::{Ctm, CtmConfig, LayerConfig};
    use modgrad::retina::{VisualRetina, Retina};
    use modgrad::tasks;

    eprintln!("=== TRAIN RETINA + CLASSIFY ===");
    eprintln!("Phase 1: Hebbian sparse coding (unsupervised)");
    eprintln!("Phase 2: CTM classification (supervised readout)\n");

    // Load pixels
    let load_pixels = |path: &str| -> Vec<(Vec<f32>, usize)> {
        let data = std::fs::read(path).expect(&format!("failed to read {path}"));
        assert_eq!(&data[0..4], b"FEAT");
        let n = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        eprintln!("  Loading {path}: {n} images");
        let mut samples = Vec::with_capacity(n);
        let mut offset = 16;
        for _ in 0..n {
            let label = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            let pixels: Vec<f32> = (0..dim).map(|i| {
                f32::from_le_bytes(data[offset + i*4..offset + (i+1)*4].try_into().unwrap())
            }).collect();
            offset += dim * 4;
            samples.push((pixels, label));
        }
        samples
    };

    let train_data = load_pixels(train_pixels_path);
    let test_data = load_pixels(test_pixels_path);

    // Phase 1: Train retina (unsupervised)
    eprintln!("\n--- Phase 1: Training visual retina (Hebbian sparse coding) ---");
    let mut retina = VisualRetina::cifar();
    eprintln!("  Retina: {} params", retina.param_count());

    // Use a subset for faster training
    let n_train_images = 10000;
    let image_refs: Vec<&[f32]> = train_data.iter()
        .take(n_train_images)
        .map(|(p, _)| p.as_slice())
        .collect();

    eprintln!("  Training on {} images, 3 epochs...\n", n_train_images);
    let _errors = retina.train_hebbian(&image_refs, 3, 0.01);

    // Phase 2: Process all images through trained retina
    eprintln!("\n--- Phase 2: Processing images through trained retina ---");
    let obs_dim = retina.d_output();

    let train_obs: Vec<(Vec<f32>, usize)> = train_data.iter().enumerate().map(|(i, (pixels, label))| {
        if i % 10000 == 0 { eprintln!("    {i}/{}", train_data.len()); }
        (retina.observe(pixels), *label)
    }).collect();

    let test_obs: Vec<(Vec<f32>, usize)> = test_data.iter().map(|(pixels, label)| {
        (retina.observe(pixels), *label)
    }).collect();

    // Phase 3: CTM classification
    eprintln!("\n--- Phase 3: CTM classification ---");
    let cfg = CtmConfig {
        iterations: 4, d_model: 512, d_input: obs_dim,
        heads: 4, n_sync_out: 128, n_sync_action: 64,
        synapse_depth: 1, out_dims: 128,
        global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,        input_layer: LayerConfig {
            n_neurons: 128, memory_length: 4, nlm_depth: 1,
            hebbian_lr: 0.01, inhibitory_fraction: 0.2,
            receives_broadcast: true, ..Default::default()
        },
        attention_layer: LayerConfig {
            n_neurons: 128, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.005, inhibitory_fraction: 0.2, ..Default::default()
        },
        output_layer: LayerConfig {
            n_neurons: 128, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.003, inhibitory_fraction: 0.2,
            receives_broadcast: true, ..Default::default()
        },
        motor_layer: LayerConfig {
            n_neurons: 128, memory_length: 4, nlm_depth: 1,
            hebbian_lr: 0.003, inhibitory_fraction: 0.2, ..Default::default()
        },
        cerebellum_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        basal_ganglia_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        insula_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        hippocampus_layer: LayerConfig { n_neurons: 16, receives_broadcast: true, ..Default::default() },
        ..CtmConfig::default()
    };

    let train_examples: Vec<tasks::Example> = train_obs.iter().map(|(obs, label)| {
        tasks::Example { input: obs.clone(), target: *label, n_classes: 10 }
    }).collect();
    let test_examples: Vec<tasks::Example> = test_obs.iter().map(|(obs, label)| {
        tasks::Example { input: obs.clone(), target: *label, n_classes: 10 }
    }).collect();

    let mut ctm = Ctm::new(cfg);
    ctm.enable_hebbian();

    let (_, train_acc, _, opt_acc) = tasks::train_ctm_on_task(&mut ctm, &train_examples, 10, 1);

    // Comparison
    eprintln!("\n=== RESULTS ===");
    eprintln!("  Random retina (no training):  28.0% test");
    eprintln!("  ResNet18 features:            50.3% test");
    eprintln!("  Hebbian retina (trained):     computing...");
    eprintln!("  Train acc: {:.1}%, Angeris optimal: {:.1}%", train_acc * 100.0, opt_acc * 100.0);

    // Test with LS-optimal weights
    let sync_dim = 128;
    let n_classes = 10;
    let class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"];

    let mut train_syncs: Vec<(Vec<f32>, usize)> = Vec::new();
    for ex in train_examples.iter().take(10000) {
        let mut state = ctm.init_state();
        let (_, sync) = ctm.forward(&ex.input, &mut state, false);
        train_syncs.push((sync, ex.target));
    }

    let xs: Vec<&[f32]> = train_syncs.iter().map(|(s, _)| s.as_slice()).collect();
    let mut xtx = vec![0.0f32; sync_dim * sync_dim];
    modgrad::linalg::accumulate_xtx(&mut xtx, &xs, sync_dim);
    for i in 0..sync_dim { xtx[i * sync_dim + i] += 1e-4; }

    if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
        let mut opt_w = vec![0.0f32; n_classes * sync_dim];
        for c in 0..n_classes {
            let mut xty = vec![0.0f32; sync_dim];
            for (sync, target) in &train_syncs {
                if *target == c { for k in 0..sync_dim { xty[k] += sync[k]; } }
            }
            let z = modgrad::linalg::forward_solve(&l, &xty, sync_dim);
            let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
            for k in 0..sync_dim { opt_w[c * sync_dim + k] = w[k]; }
        }

        let mut test_correct = 0;
        let mut confusion = vec![vec![0u32; n_classes]; n_classes];
        for ex in &test_examples {
            let mut state = ctm.init_state();
            let (_, sync) = ctm.forward(&ex.input, &mut state, false);
            let mut logits = vec![0.0f32; n_classes];
            for c in 0..n_classes {
                logits[c] = (0..sync_dim.min(sync.len()))
                    .map(|k| opt_w[c * sync_dim + k] * sync[k]).sum();
            }
            let pred = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == ex.target { test_correct += 1; }
            confusion[ex.target][pred] += 1;
        }

        let test_acc = test_correct as f32 / test_examples.len() as f32;
        eprintln!("  Hebbian retina test: {:.1}% ({}/{})", test_acc * 100.0,
            test_correct, test_examples.len());

        eprintln!("\n  Per-class:");
        for c in 0..n_classes {
            let total: u32 = confusion[c].iter().sum();
            let correct = confusion[c][c];
            let acc = if total > 0 { correct as f32 / total as f32 * 100.0 } else { 0.0 };
            eprintln!("    {:12}: {:5.1}%", class_names[c], acc);
        }
    }

    eprintln!("\n=== DONE ===");
}

/// Train CTM on CIFAR-10 with tabula rasa vision.
/// Raw pixels → V1/V2/V4 conv layers (our retina) → CTM → classification.
/// No ResNet. No pre-training. The organism learns to see from scratch.
fn train_cifar_vision(train_path: &str, test_path: &str) {
    use modgrad::ctm::{Ctm, CtmConfig, LayerConfig};
    use modgrad::retina::{VisualRetina, Retina};
    use modgrad::tasks;

    eprintln!("=== TABULA RASA VISION: Learning to see from raw pixels ===\n");

    // Load raw pixels
    let load_pixels = |path: &str| -> Vec<(Vec<f32>, usize)> {
        let data = std::fs::read(path).expect(&format!("failed to read {path}"));
        assert_eq!(&data[0..4], b"FEAT");
        let n = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        eprintln!("  Loading {path}: {n} images, {dim} pixel values");
        let mut samples = Vec::with_capacity(n);
        let mut offset = 16;
        for _ in 0..n {
            let label = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            let pixels: Vec<f32> = (0..dim).map(|i| {
                f32::from_le_bytes(data[offset + i*4..offset + (i+1)*4].try_into().unwrap())
            }).collect();
            offset += dim * 4;
            samples.push((pixels, label));
        }
        samples
    };

    let train_data = load_pixels(train_path);
    let test_data = load_pixels(test_path);

    // Create visual retina (V1→V2→V4)
    let mut retina = VisualRetina::cifar();
    let obs_dim = retina.d_output(); // 128
    eprintln!("  Visual retina: V1(3→32) → V2(32→64) → V4(64→128) → pool → {}d",
        obs_dim);
    eprintln!("  Retina params: {}", retina.param_count());

    // Process all images through retina to get observation vectors
    eprintln!("\n  Processing images through visual retina...");
    let train_obs: Vec<(Vec<f32>, usize)> = train_data.iter().enumerate().map(|(i, (pixels, label))| {
        if i % 10000 == 0 { eprintln!("    {i}/{}", train_data.len()); }
        (retina.observe(pixels), *label)
    }).collect();

    let test_obs: Vec<(Vec<f32>, usize)> = test_data.iter().map(|(pixels, label)| {
        (retina.observe(pixels), *label)
    }).collect();

    eprintln!("  Train observations: {}, Test: {}", train_obs.len(), test_obs.len());

    // Build CTM for 128-dim observations
    let cfg = CtmConfig {
        iterations: 4, d_model: 512, d_input: obs_dim,
        heads: 4, n_sync_out: 128, n_sync_action: 64,
        synapse_depth: 1, out_dims: 128,
        global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,        input_layer: LayerConfig {
            n_neurons: 128, memory_length: 4, nlm_depth: 1,
            hebbian_lr: 0.01, inhibitory_fraction: 0.2,
            receives_broadcast: true,
            ..Default::default()
        },
        attention_layer: LayerConfig {
            n_neurons: 128, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.005, inhibitory_fraction: 0.2,
            ..Default::default()
        },
        output_layer: LayerConfig {
            n_neurons: 128, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.003, inhibitory_fraction: 0.2,
            receives_broadcast: true,
            ..Default::default()
        },
        motor_layer: LayerConfig {
            n_neurons: 128, memory_length: 4, nlm_depth: 1,
            hebbian_lr: 0.003, inhibitory_fraction: 0.2,
            ..Default::default()
        },
        cerebellum_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        basal_ganglia_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        insula_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        hippocampus_layer: LayerConfig { n_neurons: 16, receives_broadcast: true, ..Default::default() },
        ..CtmConfig::default()
    };

    eprintln!("  CTM: {} neurons, {} ticks, {} sync dims\n",
        128*4, cfg.iterations, cfg.n_sync_out);

    let train_examples: Vec<tasks::Example> = train_obs.iter().map(|(obs, label)| {
        tasks::Example { input: obs.clone(), target: *label, n_classes: 10 }
    }).collect();

    let test_examples: Vec<tasks::Example> = test_obs.iter().map(|(obs, label)| {
        tasks::Example { input: obs.clone(), target: *label, n_classes: 10 }
    }).collect();

    let mut ctm = Ctm::new(cfg);
    ctm.enable_hebbian();

    let (_, train_acc, _, opt_acc) = tasks::train_ctm_on_task(
        &mut ctm, &train_examples, 10, 1);

    eprintln!("\n=== RESULTS (tabula rasa vision) ===");
    eprintln!("  Train accuracy:    {:.1}%", train_acc * 100.0);
    eprintln!("  Angeris optimal:   {:.1}%", opt_acc * 100.0);
    eprintln!("  Random baseline:   10.0%");

    // Test evaluation with LS-optimal weights
    let sync_dim = 128;
    let n_classes = 10;
    let class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"];

    let mut train_syncs: Vec<(Vec<f32>, usize)> = Vec::new();
    for ex in train_examples.iter().take(10000) {
        let mut state = ctm.init_state();
        let (_, sync) = ctm.forward(&ex.input, &mut state, false);
        train_syncs.push((sync, ex.target));
    }

    let xs: Vec<&[f32]> = train_syncs.iter().map(|(s, _)| s.as_slice()).collect();
    let mut xtx = vec![0.0f32; sync_dim * sync_dim];
    modgrad::linalg::accumulate_xtx(&mut xtx, &xs, sync_dim);
    for i in 0..sync_dim { xtx[i * sync_dim + i] += 1e-4; }

    if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
        let mut opt_w = vec![0.0f32; n_classes * sync_dim];
        for c in 0..n_classes {
            let mut xty = vec![0.0f32; sync_dim];
            for (sync, target) in &train_syncs {
                if *target == c { for k in 0..sync_dim { xty[k] += sync[k]; } }
            }
            let z = modgrad::linalg::forward_solve(&l, &xty, sync_dim);
            let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
            for k in 0..sync_dim { opt_w[c * sync_dim + k] = w[k]; }
        }

        let mut test_correct = 0;
        let mut confusion = vec![vec![0u32; n_classes]; n_classes];
        for ex in &test_examples {
            let mut state = ctm.init_state();
            let (_, sync) = ctm.forward(&ex.input, &mut state, false);
            let mut logits = vec![0.0f32; n_classes];
            for c in 0..n_classes {
                logits[c] = (0..sync_dim.min(sync.len()))
                    .map(|k| opt_w[c * sync_dim + k] * sync[k]).sum();
            }
            let pred = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == ex.target { test_correct += 1; }
            confusion[ex.target][pred] += 1;
        }

        let test_acc = test_correct as f32 / test_examples.len() as f32;
        eprintln!("\n  Test accuracy:     {:.1}% ({}/{})", test_acc * 100.0,
            test_correct, test_examples.len());

        eprintln!("\n  Per-class accuracy:");
        for c in 0..n_classes {
            let total: u32 = confusion[c].iter().sum();
            let correct = confusion[c][c];
            let acc = if total > 0 { correct as f32 / total as f32 * 100.0 } else { 0.0 };
            eprintln!("    {:12}: {:5.1}% ({}/{})", class_names[c], acc, correct, total);
        }
    }

    eprintln!("\n=== DONE ===");
}

/// Train CTM on CIFAR-10 using pre-extracted ResNet18 features.
/// The ResNet sees, the CTM thinks.
fn train_cifar(train_path: &str, test_path: &str) {
    use modgrad::ctm::{Ctm, CtmConfig, LayerConfig};
    use modgrad::tasks;

    eprintln!("=== CIFAR-10: Can the CTM recognize cats and dogs? ===\n");

    // Load features
    let load_features = |path: &str| -> Vec<(Vec<f32>, usize)> {
        let data = std::fs::read(path).expect(&format!("failed to read {path}"));
        let magic = &data[0..4];
        assert_eq!(magic, b"FEAT", "not a feature file");
        let n = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let _n_classes = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        eprintln!("  Loading {path}: {n} samples, {dim} features");

        let mut samples = Vec::with_capacity(n);
        let mut offset = 16;
        for _ in 0..n {
            let label = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            let features: Vec<f32> = (0..dim).map(|i| {
                f32::from_le_bytes(data[offset + i*4..offset + (i+1)*4].try_into().unwrap())
            }).collect();
            offset += dim * 4;
            samples.push((features, label));
        }
        samples
    };

    let train_data = load_features(train_path);
    let test_data = load_features(test_path);

    let feature_dim = train_data[0].0.len(); // 512
    let n_classes = 10;

    let class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"];

    // Build CTM sized for 512-dim input
    let cfg = CtmConfig {
        iterations: 4,  // our finding: few ticks, wide layers
        d_model: 512, d_input: feature_dim,
        heads: 4, n_sync_out: 256, n_sync_action: 128,
        synapse_depth: 1, out_dims: 256,
        global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,        input_layer: LayerConfig {
            n_neurons: 128, memory_length: 4, nlm_depth: 1,
            hebbian_lr: 0.01, inhibitory_fraction: 0.2,
            receives_broadcast: true,
            ..Default::default()
        },
        attention_layer: LayerConfig {
            n_neurons: 128, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.005, inhibitory_fraction: 0.2,
            ..Default::default()
        },
        output_layer: LayerConfig {
            n_neurons: 128, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.003, inhibitory_fraction: 0.2,
            receives_broadcast: true,
            ..Default::default()
        },
        motor_layer: LayerConfig {
            n_neurons: 128, memory_length: 4, nlm_depth: 1,
            hebbian_lr: 0.003, inhibitory_fraction: 0.2,
            ..Default::default()
        },
        cerebellum_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        basal_ganglia_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        insula_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        hippocampus_layer: LayerConfig { n_neurons: 16, receives_broadcast: true, ..Default::default() },
        ..CtmConfig::default()
    };

    eprintln!("  CTM: {} neurons, {} ticks, {} sync dims, d_input={}",
        128*4, cfg.iterations, cfg.n_sync_out, feature_dim);

    // Convert to task examples
    let train_examples: Vec<tasks::Example> = train_data.iter().map(|(features, label)| {
        tasks::Example { input: features.clone(), target: *label, n_classes }
    }).collect();

    let test_examples: Vec<tasks::Example> = test_data.iter().map(|(features, label)| {
        tasks::Example { input: features.clone(), target: *label, n_classes }
    }).collect();

    eprintln!("  Train: {} examples, Test: {} examples\n", train_examples.len(), test_examples.len());

    // Train
    let mut ctm = Ctm::new(cfg);
    ctm.enable_hebbian();

    let (_final_loss, final_acc, _opt_loss, opt_acc) = tasks::train_ctm_on_task(
        &mut ctm, &train_examples, 20, 1);

    eprintln!("\n=== RESULTS ===");
    eprintln!("  Train accuracy:    {:.1}%", final_acc * 100.0);
    eprintln!("  Angeris optimal:   {:.1}%", opt_acc * 100.0);
    eprintln!("  Random baseline:   10.0%");

    // Test set evaluation
    eprintln!("\n  Evaluating on test set...");
    let sync_dim = 256;
    // We need the weights from training... let me use the Angeris optimal weights on test
    // Actually, let's just run the test examples through the trained CTM
    // The train_ctm_on_task returns final trained weights implicitly
    // For now, let's evaluate using a fresh LS solve on train, then test

    // Collect train syncs for LS
    let mut train_syncs: Vec<(Vec<f32>, usize)> = Vec::new();
    for ex in train_examples.iter().take(10000) { // subsample for speed
        let mut state = ctm.init_state();
        let (_, sync) = ctm.forward(&ex.input, &mut state, false);
        train_syncs.push((sync, ex.target));
    }

    // LS solve
    let xs: Vec<&[f32]> = train_syncs.iter().map(|(s, _)| s.as_slice()).collect();
    let mut xtx = vec![0.0f32; sync_dim * sync_dim];
    modgrad::linalg::accumulate_xtx(&mut xtx, &xs, sync_dim);
    for i in 0..sync_dim { xtx[i * sync_dim + i] += 1e-4; }

    if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
        let mut opt_w = vec![0.0f32; n_classes * sync_dim];
        for c in 0..n_classes {
            let mut xty = vec![0.0f32; sync_dim];
            for (sync, target) in &train_syncs {
                if *target == c { for k in 0..sync_dim { xty[k] += sync[k]; } }
            }
            let z = modgrad::linalg::forward_solve(&l, &xty, sync_dim);
            let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
            for k in 0..sync_dim { opt_w[c * sync_dim + k] = w[k]; }
        }

        // Evaluate on test set
        let mut test_correct = 0;
        let mut confusion = vec![vec![0u32; n_classes]; n_classes]; // [true][pred]
        for ex in &test_examples {
            let mut state = ctm.init_state();
            let (_, sync) = ctm.forward(&ex.input, &mut state, false);
            let mut logits = vec![0.0f32; n_classes];
            for c in 0..n_classes {
                logits[c] = (0..sync_dim.min(sync.len()))
                    .map(|k| opt_w[c * sync_dim + k] * sync[k]).sum();
            }
            let pred = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == ex.target { test_correct += 1; }
            confusion[ex.target][pred] += 1;
        }

        let test_acc = test_correct as f32 / test_examples.len() as f32;
        eprintln!("  Test accuracy:     {:.1}% ({}/{})", test_acc * 100.0,
            test_correct, test_examples.len());

        // Per-class accuracy
        eprintln!("\n  Per-class accuracy:");
        for c in 0..n_classes {
            let total: u32 = confusion[c].iter().sum();
            let correct = confusion[c][c];
            let acc = if total > 0 { correct as f32 / total as f32 * 100.0 } else { 0.0 };
            eprintln!("    {:12}: {:5.1}% ({}/{})", class_names[c], acc, correct, total);
        }
    }

    eprintln!("\n=== DONE ===");
}

/// Run the task suite: XOR → parity → majority → maze.
/// Each task tests a specific cognitive capability.
/// Reports accuracy + Angeris bound (optimal vs achieved).
fn run_task_suite() {
    use modgrad::ctm::{Ctm, CtmConfig, LayerConfig};
    use modgrad::tasks;

    eprintln!("=== TASK SUITE ===");
    eprintln!("Testing cognitive capabilities of the CTM architecture.\n");

    // Build a small CTM for tasks (not the full organism — just the CTM core)
    let cfg = CtmConfig {
        iterations: 8, d_model: 128, d_input: 32,
        heads: 2, n_sync_out: 32, n_sync_action: 16,
        synapse_depth: 1, out_dims: 32,
        global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 32,        input_layer: LayerConfig {
            n_neurons: 32, memory_length: 4, nlm_depth: 1,
            hebbian_lr: 0.01, inhibitory_fraction: 0.2,
            receives_broadcast: true,
            ..Default::default()
        },
        attention_layer: LayerConfig {
            n_neurons: 32, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.005, inhibitory_fraction: 0.2,
            ..Default::default()
        },
        output_layer: LayerConfig {
            n_neurons: 32, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.003, inhibitory_fraction: 0.2,
            receives_broadcast: true,
            ..Default::default()
        },
        motor_layer: LayerConfig {
            n_neurons: 32, memory_length: 4, nlm_depth: 1,
            hebbian_lr: 0.003, inhibitory_fraction: 0.2,
            ..Default::default()
        },
        cerebellum_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        basal_ganglia_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        insula_layer: LayerConfig { n_neurons: 16, receives_broadcast: false, ..Default::default() },
        hippocampus_layer: LayerConfig { n_neurons: 16, receives_broadcast: true, ..Default::default() },
        ..CtmConfig::default()
    };

    // ─── XOR ────────────────────────────────────────────────
    eprintln!("--- Task 1: XOR (nonlinear computation) ---");
    eprintln!("  Can the CTM compute XOR? Requires >=1 nonlinear step.");
    eprintln!("  A linear model achieves 50% (random chance).\n");
    let mut ctm = Ctm::new(cfg.clone());
    ctm.enable_hebbian();
    let xor_data = tasks::xor_examples(cfg.d_input);
    let (loss, acc, opt_loss, opt_acc) = tasks::train_ctm_on_task(&mut ctm, &xor_data, 50, 10);
    eprintln!("\n  Final:    loss={loss:.4} acc={:.1}%", acc * 100.0);
    eprintln!("  Angeris:  loss={opt_loss:.4} acc={:.1}% (LS-optimal)", opt_acc * 100.0);
    eprintln!("  Random:   loss={:.4} acc=50.0%", (2.0f32).ln());
    if opt_acc > 0.9 {
        eprintln!("  PASS: CTM representations can solve XOR!");
    } else if opt_acc > 0.6 {
        eprintln!("  PARTIAL: some nonlinear signal, but not enough");
    } else {
        eprintln!("  FAIL: CTM acts linearly — representations can't solve XOR");
    }

    // ─── 3-bit Parity ───────────────────────────────────────
    eprintln!("\n--- Task 2: 3-bit Parity (chained nonlinearity) ---");
    eprintln!("  Can the CTM chain multiple nonlinear steps?\n");
    let mut ctm = Ctm::new(cfg.clone());
    ctm.enable_hebbian();
    let parity_data = tasks::parity_examples(3, cfg.d_input);
    let (loss, acc, opt_loss, opt_acc) = tasks::train_ctm_on_task(&mut ctm, &parity_data, 50, 10);
    eprintln!("\n  Final:    loss={loss:.4} acc={:.1}%", acc * 100.0);
    eprintln!("  Angeris:  loss={opt_loss:.4} acc={:.1}%", opt_acc * 100.0);
    if opt_acc > 0.8 {
        eprintln!("  PASS: CTM can chain nonlinear operations!");
    } else {
        eprintln!("  FAIL: insufficient chained nonlinearity");
    }

    // ─── Majority Vote ──────────────────────────────────────
    eprintln!("\n--- Task 3: Majority Vote (feature extraction) ---");
    eprintln!("  Can the CTM count? A linear model CAN solve this.\n");
    let mut ctm = Ctm::new(cfg.clone());
    ctm.enable_hebbian();
    let maj_data = tasks::majority_examples(8, cfg.d_input);
    let (loss, acc, opt_loss, opt_acc) = tasks::train_ctm_on_task(&mut ctm, &maj_data, 50, 10);
    eprintln!("\n  Final:    loss={loss:.4} acc={:.1}%", acc * 100.0);
    eprintln!("  Angeris:  loss={opt_loss:.4} acc={:.1}%", opt_acc * 100.0);
    if opt_acc > 0.9 {
        eprintln!("  PASS: CTM extracts majority feature");
    } else {
        eprintln!("  FAIL: can't even count");
    }

    // ─── Maze ───────────────────────────────────────────────
    eprintln!("\n--- Task 4: 5x5 Maze (spatial planning) ---");
    eprintln!("  Can the CTM plan a path? Tests multi-tick deliberation.\n");
    let mut ctm = Ctm::new(cfg.clone());
    ctm.enable_hebbian();
    let maze_data = tasks::maze_examples(5, cfg.d_input, 2000);
    let (loss, acc, opt_loss, opt_acc) = tasks::train_ctm_on_task(&mut ctm, &maze_data, 50, 10);
    eprintln!("\n  Final:    loss={loss:.4} acc={:.1}%", acc * 100.0);
    eprintln!("  Angeris:  loss={opt_loss:.4} acc={:.1}%", opt_acc * 100.0);
    eprintln!("  Random:   loss={:.4} acc=25.0%", (4.0f32).ln());
    if opt_acc > 0.5 {
        eprintln!("  PASS: CTM can do spatial planning!");
    } else if opt_acc > 0.35 {
        eprintln!("  PARTIAL: some spatial reasoning");
    } else {
        eprintln!("  FAIL: no spatial reasoning detected");
    }

    eprintln!("\n=== TASK SUITE COMPLETE ===");
}

/// Angeris pipeline: measure information flow at EVERY connection point.
///
/// For each (X, Y) pair in the pipeline, compute:
///   W_opt = argmin ||Y - WX||²
///   residual = ||Y - W_opt X||² / ||Y||²
///   recoverable = 1 - residual (fraction linearly extractable)
///
/// This is "gradient-free gradient analysis" — finds bottlenecks without backprop.
/// Chain the bounds to see exactly where information dies in the pipeline.
fn angeris_pipeline(org_path: &str) {
    use modgrad::tabula_rasa::{Organism, Dna};
    use modgrad::curriculum;

    let mut org = if std::path::Path::new(org_path).exists() {
        Organism::load(org_path).expect("failed to load")
    } else {
        eprintln!("No organism at {org_path}, creating fresh one");
        Organism::new(Dna::small())
    };

    eprintln!("=== ANGERIS PIPELINE ANALYSIS ===");
    eprintln!("Measures linear recoverability at every connection point.\n");

    // Generate test data
    let mut data = Vec::new();
    for phase in 0..4 {
        let (_, d) = curriculum::generate(phase, 1);
        data.extend_from_slice(&d);
    }
    let chunk_size = org.dna.context_len;
    let chunks: Vec<&[u8]> = data.chunks(chunk_size).take(100).collect();

    // Collect traces at every layer for many byte inputs
    let vocab = org.dna.vocab_size;
    let _embed_dim = org.dna.embed_dim;
    let _sync_dim = org.dna.ctm.n_sync_out;

    // Storage: (embedding, observation, input_act, attn_act, output_act, motor_act, sync, target)
    struct LayerTrace {
        embedding: Vec<f32>,
        observation: Vec<f32>,
        input_act: Vec<f32>,
        attn_act: Vec<f32>,
        output_act: Vec<f32>,
        motor_act: Vec<f32>,
        sync: Vec<f32>,
        target_onehot: Vec<f32>,
    }

    eprintln!("Collecting layer traces...");
    let mut traces: Vec<LayerTrace> = Vec::new();

    for chunk in &chunks {
        let token_ids: Vec<usize> = chunk.iter().map(|&b| b as usize).collect();
        if token_ids.len() < 2 { continue; }

        for i in 0..token_ids.len() - 1 {
            let tid = token_ids[i];
            let target = token_ids[i + 1];
            if target >= vocab { continue; }

            // Embedding
            let embedding = org.embed(tid);
            // Observation (sensory forward)
            let observation = org.sensory_forward(&embedding, false);

            // CTM forward for one token
            let mut state = org.ctm.init_state();
            state.neuromod = org.neuromod.clone();
            let (_preds, sync_vec) = org.ctm.forward(&observation, &mut state, true);

            // Get last tick's activations from tick traces
            if let Some(trace) = state.tick_traces.last() {
                let mut target_onehot = vec![0.0f32; vocab];
                target_onehot[target] = 1.0;

                traces.push(LayerTrace {
                    embedding,
                    observation,
                    input_act: trace.input_activations.clone(),
                    attn_act: trace.attention_activations.clone(),
                    output_act: trace.output_activations.clone(),
                    motor_act: trace.motor_activations.clone(),
                    sync: sync_vec,
                    target_onehot,
                });
            }

            if traces.len() >= 5000 { break; }
        }
        if traces.len() >= 5000 { break; }
    }

    let n = traces.len();
    eprintln!("  {} traces collected\n", n);
    if n < 50 { eprintln!("Too few traces"); return; }

    // Helper: compute Angeris bound between two layers
    let compute_bound = |name: &str, get_x: &dyn Fn(&LayerTrace) -> &[f32], get_y: &dyn Fn(&LayerTrace) -> &[f32]| {
        let xs: Vec<&[f32]> = traces.iter().map(|t| get_x(t)).collect();
        let ys: Vec<&[f32]> = traces.iter().map(|t| get_y(t)).collect();
        let in_dim = xs[0].len();
        let out_dim = ys[0].len();

        if let Some(w_opt) = modgrad::linalg::least_squares(&xs, &ys, in_dim, out_dim, 1e-4) {
            let x_refs: Vec<&[f32]> = xs.iter().copied().collect();
            let y_refs: Vec<&[f32]> = ys.iter().copied().collect();
            let (residual_frac, _) = modgrad::linalg::angeris_residual(&x_refs, &y_refs, &w_opt, in_dim, out_dim);
            let recoverable = (1.0 - residual_frac) * 100.0;
            eprintln!("  {name:40} [{in_dim:>4} → {out_dim:>4}]  {recoverable:5.1}% recoverable  (residual {:.1}%)",
                residual_frac * 100.0);
            recoverable
        } else {
            eprintln!("  {name:40} [{in_dim:>4} → {out_dim:>4}]  FAILED (singular)");
            0.0
        }
    };

    eprintln!("--- Information flow (Angeris bounds) ---\n");
    eprintln!("  Each row: what % of the output is linearly predictable from the input?");
    eprintln!("  100% = perfect linear map. 0% = no linear relationship.\n");

    // 1. Embedding → target (can the embedding predict the next byte at all?)
    let _b1 = compute_bound(
        "embedding → target",
        &|t| &t.embedding, &|t| &t.target_onehot);

    // 2. Observation → target (does sensory processing help?)
    let _b2 = compute_bound(
        "observation → target",
        &|t| &t.observation, &|t| &t.target_onehot);

    // 3. Embedding → observation (how much info does sensory preserve?)
    let _b3 = compute_bound(
        "embedding → observation",
        &|t| &t.embedding, &|t| &t.observation);

    // 4. Observation → input_act (first synapse: how much gets through?)
    let _b4 = compute_bound(
        "observation → input_activations",
        &|t| &t.observation, &|t| &t.input_act);

    // 5. Input → attention (cross-region flow)
    let _b5 = compute_bound(
        "input_act → attention_act",
        &|t| &t.input_act, &|t| &t.attn_act);

    // 6. Attention → output
    let _b6 = compute_bound(
        "attention_act → output_act",
        &|t| &t.attn_act, &|t| &t.output_act);

    // 7. Output → motor
    let _b7 = compute_bound(
        "output_act → motor_act",
        &|t| &t.output_act, &|t| &t.motor_act);

    // 8. Output+motor → sync (the sync accumulator)
    let _b8 = compute_bound(
        "output_act → sync",
        &|t| &t.output_act, &|t| &t.sync);

    // 9. Motor → sync
    let _b9 = compute_bound(
        "motor_act → sync",
        &|t| &t.motor_act, &|t| &t.sync);

    // 10. Sync → target (the final readout — same as angeris bound)
    let _b10 = compute_bound(
        "sync → target",
        &|t| &t.sync, &|t| &t.target_onehot);

    // 11. Each region → target directly (which region carries most info?)
    eprintln!("\n--- Per-region predictive power ---\n");
    let _r1 = compute_bound(
        "input_act → target",
        &|t| &t.input_act, &|t| &t.target_onehot);
    let _r2 = compute_bound(
        "attention_act → target",
        &|t| &t.attn_act, &|t| &t.target_onehot);
    let _r3 = compute_bound(
        "output_act → target",
        &|t| &t.output_act, &|t| &t.target_onehot);
    let _r4 = compute_bound(
        "motor_act → target",
        &|t| &t.motor_act, &|t| &t.target_onehot);

    // 12. Combined: all activations → target
    eprintln!("\n--- Combined representations ---\n");
    // Concatenate all activations
    let all_acts: Vec<Vec<f32>> = traces.iter().map(|t| {
        let mut v = Vec::new();
        v.extend_from_slice(&t.input_act);
        v.extend_from_slice(&t.attn_act);
        v.extend_from_slice(&t.output_act);
        v.extend_from_slice(&t.motor_act);
        v
    }).collect();
    let all_refs: Vec<&[f32]> = all_acts.iter().map(|v| v.as_slice()).collect();
    let target_refs: Vec<&[f32]> = traces.iter().map(|t| t.target_onehot.as_slice()).collect();
    let all_dim = all_acts[0].len();
    let target_dim = traces[0].target_onehot.len();
    if let Some(w) = modgrad::linalg::least_squares(&all_refs, &target_refs, all_dim, target_dim, 1e-4) {
        let (res, _) = modgrad::linalg::angeris_residual(&all_refs, &target_refs, &w, all_dim, target_dim);
        eprintln!("  {:40} [{:>4} → {:>4}]  {:5.1}% recoverable",
            "ALL activations → target", all_dim, target_dim, (1.0 - res) * 100.0);
    }

    // Also: embedding directly → target (bypass everything)
    let _direct = compute_bound(
        "embedding (direct) → target",
        &|t| &t.embedding, &|t| &t.target_onehot);

    eprintln!("\n=== END PIPELINE ANALYSIS ===");
}

/// Angeris bound: what's the BEST loss a linear readout could achieve
/// given the CTM's sync patterns? This separates representation quality
/// from training algorithm quality.
///
/// If optimal_loss ≈ current_loss → CTM doesn't produce useful features
/// If optimal_loss << current_loss → features are there, training can't find weights
fn angeris_bound(org_path: &str) {
    use modgrad::tabula_rasa::{Organism, Dna};
    use modgrad::curriculum;

    let mut org = if std::path::Path::new(org_path).exists() {
        Organism::load(org_path).expect("failed to load organism")
    } else {
        eprintln!("No organism at {org_path}, creating fresh one");
        Organism::new(Dna::small())
    };

    let total_neurons: usize = org.dna.ctm.input_layer.n_neurons
        + org.dna.ctm.attention_layer.n_neurons
        + org.dna.ctm.output_layer.n_neurons
        + org.dna.ctm.motor_layer.n_neurons;

    eprintln!("=== ANGERIS BOUND ANALYSIS ===");
    eprintln!("Organism: {} params, {} neurons, {} sync dims",
        org.param_count(), total_neurons, org.dna.ctm.n_sync_out);
    eprintln!("Tokens seen: {}, Sleep cycles: {}", org.tokens_seen, org.sleep_cycles);

    // Generate training data from multiple curriculum phases
    let mut all_data = Vec::new();
    for phase in 0..4 {
        let (name, data) = curriculum::generate(phase, 1);
        eprintln!("  Phase {phase} ({name}): {} bytes", data.len());
        all_data.extend_from_slice(&data);
    }

    // Collect (sync, target) pairs by running forward
    let chunk_size = org.dna.context_len;
    let chunks: Vec<&[u8]> = all_data.chunks(chunk_size).take(200).collect();
    let vocab = org.dna.vocab_size;
    let sync_dim = org.dna.ctm.n_sync_out;

    eprintln!("\nCollecting sync-target pairs from {} chunks...", chunks.len());
    let mut sync_target_pairs: Vec<(Vec<f32>, usize)> = Vec::new();

    for chunk in &chunks {
        let token_ids: Vec<usize> = chunk.iter().map(|&b| b as usize).collect();
        if token_ids.len() < 2 { continue; }

        let (_logits, syncs) = org.forward_inner(&token_ids, false);

        // Each position i predicts token i+1
        for i in 0..syncs.len().min(token_ids.len() - 1) {
            let target = token_ids[i + 1];
            if target < vocab {
                sync_target_pairs.push((syncs[i].clone(), target));
            }
        }
    }

    let n_pairs = sync_target_pairs.len();
    eprintln!("  Collected {} sync-target pairs", n_pairs);
    if n_pairs < 100 {
        eprintln!("  Too few pairs for meaningful analysis");
        return;
    }

    // ─── CURRENT LOSS ─────────────────────────────────────
    eprintln!("\n--- Current output projection loss ---");
    let mut current_total_loss = 0.0f32;
    for (sync, target) in &sync_target_pairs {
        let logits = org.output_proj.forward(sync);
        let max_l: f32 = logits.iter().take(vocab).fold(f32::MIN, |a, &b| a.max(b));
        let log_sum_exp: f32 = logits.iter().take(vocab).map(|&x| (x - max_l).exp()).sum::<f32>().ln() + max_l;
        let loss = log_sum_exp - logits[*target];
        current_total_loss += loss;
    }
    let current_avg_loss = current_total_loss / n_pairs as f32;
    eprintln!("  Current avg CE loss: {current_avg_loss:.4}");
    eprintln!("  Random baseline: {:.4}", (vocab as f32).ln());

    // ─── OPTIMAL (LS) OUTPUT PROJECTION ───────────────────
    eprintln!("\n--- Computing LS-optimal output projection ---");

    // Build X (syncs) and Y (one-hot targets) matrices
    let xs: Vec<&[f32]> = sync_target_pairs.iter().map(|(s, _)| s.as_slice()).collect();

    // Build XTX once
    let mut xtx = vec![0.0f32; sync_dim * sync_dim];
    modgrad::linalg::accumulate_xtx(&mut xtx, &xs, sync_dim);
    // Regularization
    for i in 0..sync_dim {
        xtx[i * sync_dim + i] += 1e-4;
    }

    // Cholesky
    let l = match modgrad::linalg::cholesky(&xtx, sync_dim) {
        Some(l) => l,
        None => {
            eprintln!("  Cholesky failed (singular XTX). Syncs are degenerate.");
            return;
        }
    };

    // Solve for each token column: W_opt[token, :] = (XTX)^-1 XTY[token]
    let mut optimal_weights = vec![0.0f32; vocab * sync_dim];
    let optimal_bias = vec![0.0f32; vocab];
    let mut token_counts = vec![0u32; vocab];

    for (_sync, target) in &sync_target_pairs {
        token_counts[*target] += 1;
    }

    for tid in 0..vocab {
        if token_counts[tid] < 3 { continue; }

        // XTY for this token
        let mut xty = vec![0.0f32; sync_dim];
        for (sync, target) in &sync_target_pairs {
            if *target == tid {
                for k in 0..sync_dim {
                    xty[k] += sync[k];
                }
            }
        }

        let z = modgrad::linalg::forward_solve(&l, &xty, sync_dim);
        let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);

        for k in 0..sync_dim {
            optimal_weights[tid * sync_dim + k] = w[k];
        }
    }

    // ─── OPTIMAL LOSS ─────────────────────────────────────
    eprintln!("  Computing loss with LS-optimal weights...");
    let mut optimal_total_loss = 0.0f32;
    for (sync, target) in &sync_target_pairs {
        // Manual matmul: logits = W_opt @ sync + bias
        let mut logits = vec![0.0f32; vocab];
        for tid in 0..vocab {
            let row_start = tid * sync_dim;
            let mut dot = optimal_bias[tid];
            for k in 0..sync_dim {
                dot += optimal_weights[row_start + k] * sync[k];
            }
            logits[tid] = dot;
        }

        let max_l: f32 = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
        let log_sum_exp: f32 = logits.iter().map(|&x| (x - max_l).exp()).sum::<f32>().ln() + max_l;
        let loss = log_sum_exp - logits[*target];
        optimal_total_loss += loss;
    }
    let optimal_avg_loss = optimal_total_loss / n_pairs as f32;

    // ─── ANALYSIS ─────────────────────────────────────────
    let random_loss = (vocab as f32).ln();
    let current_gap = current_avg_loss - optimal_avg_loss;
    let optimal_improvement = (1.0 - optimal_avg_loss / random_loss) * 100.0;
    let current_improvement = (1.0 - current_avg_loss / random_loss) * 100.0;

    eprintln!("\n=== RESULTS ===");
    eprintln!("  Random baseline:  {random_loss:.4}");
    eprintln!("  Current loss:     {current_avg_loss:.4}  ({current_improvement:.2}% below random)");
    eprintln!("  LS-optimal loss:  {optimal_avg_loss:.4}  ({optimal_improvement:.2}% below random)");
    eprintln!("  Gap (current - optimal): {current_gap:.4}");
    eprintln!();

    if optimal_avg_loss > random_loss * 0.95 {
        eprintln!("  VERDICT: CTM representations are NOT informative.");
        eprintln!("  Even the optimal linear readout can't beat random by much.");
        eprintln!("  The bottleneck is the CTM / embedding / architecture.");
    } else if current_gap > 0.5 {
        eprintln!("  VERDICT: CTM representations ARE informative!");
        eprintln!("  The LS-optimal readout achieves {optimal_avg_loss:.3} but current weights give {current_avg_loss:.3}.");
        eprintln!("  The training algorithm is failing to find the right output weights.");
        eprintln!("  Fix: use the LS-optimal weights directly (replace output_proj).");
    } else {
        eprintln!("  VERDICT: Training is close to optimal.");
        eprintln!("  Current weights are within {current_gap:.4} of the best possible linear readout.");
    }

    // ─── PER-BYTE ANALYSIS ────────────────────────────────
    eprintln!("\n--- Per-byte optimal probability (top 20) ---");
    // For each byte, what's the optimal probability the readout assigns?
    let mut byte_optimal_probs: Vec<(usize, f32)> = Vec::new();
    for tid in 0..vocab {
        if token_counts[tid] < 5 { continue; }
        // Average optimal logit for this token when it IS the target
        let mut total_prob = 0.0f32;
        let mut count = 0;
        for (sync, target) in &sync_target_pairs {
            if *target == tid {
                let mut logits = vec![0.0f32; vocab];
                for t in 0..vocab {
                    let row = t * sync_dim;
                    logits[t] = (0..sync_dim).map(|k| optimal_weights[row + k] * sync[k]).sum();
                }
                let max_l: f32 = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
                let exps: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
                let sum: f32 = exps.iter().sum();
                total_prob += exps[tid] / sum;
                count += 1;
            }
        }
        if count > 0 {
            byte_optimal_probs.push((tid, total_prob / count as f32));
        }
    }
    byte_optimal_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (tid, prob) in byte_optimal_probs.iter().take(20) {
        let ch = if *tid >= 32 && *tid < 127 { format!("'{}'", *tid as u8 as char) } else { format!("0x{:02x}", tid) };
        let count = token_counts[*tid];
        let random_prob = 1.0 / vocab as f32;
        eprintln!("  byte {:3} ({ch:>5}): optimal_prob={prob:.4} ({}x random), n={count}",
            tid, (prob / random_prob) as u32);
    }

    // ─── SYNC RANK ANALYSIS ──────────────────────────────
    eprintln!("\n--- Sync matrix rank ---");
    // How many independent directions do the syncs span?
    // Compute eigenvalues of XTX (already have it)
    let mut diag = vec![0.0f32; sync_dim];
    for i in 0..sync_dim {
        diag[i] = xtx[i * sync_dim + i];
    }
    diag.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let total_var: f32 = diag.iter().sum();
    let mut cumulative = 0.0f32;
    let mut effective_rank = 0;
    for (i, &d) in diag.iter().enumerate() {
        cumulative += d;
        if cumulative > total_var * 0.95 {
            effective_rank = i + 1;
            break;
        }
    }
    if effective_rank == 0 { effective_rank = sync_dim; }
    eprintln!("  XTX diagonal (top 10): {:?}",
        diag.iter().take(10).map(|d| format!("{d:.2}")).collect::<Vec<_>>());
    eprintln!("  Effective rank (95% variance): {effective_rank}/{sync_dim}");
    if effective_rank < 10 {
        eprintln!("  WARNING: Very low rank — syncs span only {effective_rank} dimensions.");
        eprintln!("  The CTM is not using its full sync capacity.");
    }
}

/// Deep diagnostic: see EVERYTHING inside the organism for different inputs.
fn deep_diagnostic(org_path: &str) {
    use modgrad::tabula_rasa::{Organism, Dna};

    let mut org = if std::path::Path::new(org_path).exists() {
        Organism::load(org_path).expect("failed to load organism")
    } else {
        eprintln!("No organism at {org_path}, creating fresh one");
        Organism::new(Dna::small())
    };

    let total_neurons: usize = org.dna.ctm.input_layer.n_neurons
        + org.dna.ctm.attention_layer.n_neurons
        + org.dna.ctm.output_layer.n_neurons
        + org.dna.ctm.motor_layer.n_neurons;

    eprintln!("=== DEEP DIAGNOSTIC ===");
    eprintln!("Organism: {} params, {} neurons, {} ticks, {} sync dims",
        org.param_count(), total_neurons,
        org.dna.ctm.iterations, org.dna.ctm.n_sync_out);
    eprintln!("Tokens seen: {}, Sleep cycles: {}", org.tokens_seen, org.sleep_cycles);

    // ─── 1. EMBEDDING SPACE ─────────────────────────────────
    eprintln!("\n--- 1. EMBEDDING SPACE ---");
    let d = org.dna.embed_dim;
    let v = org.dna.vocab_size;

    // Sample pairwise cosine between a few byte embeddings
    let test_bytes: Vec<usize> = vec![
        b'a' as usize, b'b' as usize, b'z' as usize,
        b'0' as usize, b'1' as usize, b'9' as usize,
        b' ' as usize, b'.' as usize, b'\n' as usize,
        b'A' as usize,
    ];
    let test_names: Vec<&str> = vec!["'a'", "'b'", "'z'", "'0'", "'1'", "'9'", "' '", "'.'", "'\\n'", "'A'"];

    // Get embeddings
    let embeds: Vec<Vec<f32>> = test_bytes.iter().map(|&b| {
        let off = b.min(v - 1) * d;
        org.embeddings[off..off + d].to_vec()
    }).collect();

    // Norms
    let norms: Vec<f32> = embeds.iter().map(|e| e.iter().map(|x| x*x).sum::<f32>().sqrt()).collect();
    eprintln!("  Embedding norms: {:?}", norms.iter().map(|n| format!("{n:.3}")).collect::<Vec<_>>());

    // Pairwise cosine matrix (condensed)
    eprintln!("  Pairwise cosine (sample):");
    for i in 0..test_bytes.len().min(6) {
        let mut row = String::new();
        for j in 0..test_bytes.len().min(6) {
            let c = cosine_sim(&embeds[i], &embeds[j]);
            row.push_str(&format!("{c:6.3} "));
        }
        eprintln!("    {} {}", test_names[i], row);
    }

    // Mean pairwise cosine across ALL 256 embeddings
    let mut all_cos = Vec::new();
    for i in (0..v).step_by(8) {
        for j in (i+8..v).step_by(8) {
            let ei_off = i * d;
            let ej_off = j * d;
            let ei = &org.embeddings[ei_off..ei_off + d];
            let ej = &org.embeddings[ej_off..ej_off + d];
            all_cos.push(cosine_sim(ei, ej));
        }
    }
    let mean_embed_cos: f32 = all_cos.iter().sum::<f32>() / all_cos.len() as f32;
    eprintln!("  Mean embed cosine (sampled): {mean_embed_cos:.4}");

    // ─── 2. SENSORY OUTPUT ──────────────────────────────────
    eprintln!("\n--- 2. SENSORY MLP OUTPUT ---");
    let obs: Vec<Vec<f32>> = test_bytes.iter().map(|&b| {
        let emb = org.embed(b);
        org.sensory_forward(&emb, false)
    }).collect();

    let obs_norms: Vec<f32> = obs.iter().map(|o| o.iter().map(|x| x*x).sum::<f32>().sqrt()).collect();
    eprintln!("  Observation norms: {:?}", obs_norms.iter().map(|n| format!("{n:.3}")).collect::<Vec<_>>());

    // Pairwise
    let mut obs_cos = Vec::new();
    for i in 0..obs.len() {
        for j in (i+1)..obs.len() {
            obs_cos.push(cosine_sim(&obs[i], &obs[j]));
        }
    }
    let mean_obs_cos: f32 = obs_cos.iter().sum::<f32>() / obs_cos.len().max(1) as f32;
    eprintln!("  Mean observation cosine: {mean_obs_cos:.4}");

    // ─── 3. PER-REGION ACTIVATIONS ──────────────────────────
    eprintln!("\n--- 3. CTM ACTIVATIONS (single byte, all ticks) ---");

    // Run CTM on a single byte and capture tick traces
    for &(name, byte_id) in &[("byte 'a'", b'a' as usize), ("byte '0'", b'0' as usize), ("byte ' '", b' ' as usize)] {
        let emb = org.embed(byte_id);
        let observation = org.sensory_forward(&emb, false);
        let mut state = org.ctm.init_state();
        state.neuromod = org.neuromod.clone();

        let (_preds, _sync) = org.ctm.forward(&observation, &mut state, true);

        eprintln!("\n  {name} (byte {byte_id}):");
        for trace in &state.tick_traces {
            let in_active = trace.input_activations.iter().filter(|x| x.abs() > 0.1).count();
            let attn_active = trace.attention_activations.iter().filter(|x| x.abs() > 0.1).count();
            let out_active = trace.output_activations.iter().filter(|x| x.abs() > 0.1).count();
            let motor_active = trace.motor_activations.iter().filter(|x| x.abs() > 0.1).count();
            let sync_norm: f32 = trace.sync_out.iter().map(|x| x*x).sum::<f32>().sqrt();
            let in_max: f32 = trace.input_activations.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let in_mean: f32 = trace.input_activations.iter().map(|x| x.abs()).sum::<f32>()
                / trace.input_activations.len() as f32;

            eprintln!("    tick {}: active(in={in_active} attn={attn_active} out={out_active} motor={motor_active}) \
                sync_norm={sync_norm:.4} in_max={in_max:.3} in_mean={in_mean:.3} \
                DA={:.3} 5HT={:.3} NE={:.3} motor_decided={}",
                trace.tick,
                trace.modulation.get(crate::ctm::MOD_SYNC_SCALE).copied().unwrap_or(1.0),
                trace.modulation.get(crate::ctm::MOD_GATE).copied().unwrap_or(0.5),
                trace.modulation.get(crate::ctm::MOD_AROUSAL).copied().unwrap_or(0.5),
                trace.motor_decided);
        }
    }

    // ─── 4. SYNC COMPARISON ACROSS BYTES ────────────────────
    eprintln!("\n--- 4. SYNC PATTERNS ACROSS BYTES ---");
    let mut byte_syncs: Vec<(usize, Vec<f32>)> = Vec::new();
    for &b in &test_bytes {
        let (_, syncs) = org.forward_inner(&[b], false);
        if let Some(s) = syncs.last() {
            byte_syncs.push((b, s.clone()));
        }
    }
    for (i, (bi, si)) in byte_syncs.iter().enumerate() {
        let norm: f32 = si.iter().map(|x| x*x).sum::<f32>().sqrt();
        let nonzero = si.iter().filter(|x| x.abs() > 0.01).count();
        let max_val: f32 = si.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!("  byte {:3} ({}): norm={:.4} nonzero={}/{} max={:.4}",
            bi, test_names[i], norm, nonzero, si.len(), max_val);
    }

    // ─── 5. OUTPUT DISTRIBUTION ─────────────────────────────
    eprintln!("\n--- 5. OUTPUT DISTRIBUTION ---");
    for &(name, byte_id) in &[("'a'", b'a' as usize), ("'0'", b'0' as usize), ("' '", b' ' as usize)] {
        let (logits, _) = org.forward_inner(&[byte_id], false);
        if let Some(logit) = logits.last() {
            // softmax
            let max_l: f32 = logit.iter().take(v).fold(f32::MIN, |a, &b| a.max(b));
            let exps: Vec<f32> = logit.iter().take(v).map(|&x| (x - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

            let max_prob = probs.iter().fold(0.0f32, |a, &b| a.max(b));
            let max_idx = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
            let entropy: f32 = -probs.iter().filter(|&&p| p > 1e-10).map(|&p| p * p.ln()).sum::<f32>();
            let max_entropy = (v as f32).ln();

            // Top 5 predictions
            let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            eprintln!("  Input {name}: entropy={entropy:.3}/{max_entropy:.3} ({:.1}% of max), top_prob={max_prob:.4} (byte {}='{}')",
                entropy / max_entropy * 100.0,
                max_idx,
                if max_idx >= 32 && max_idx < 127 { max_idx as u8 as char } else { '?' });
            let top5: Vec<String> = indexed.iter().take(5).map(|(i, p)| {
                let ch = if *i >= 32 && *i < 127 { format!("'{}'", *i as u8 as char) } else { format!("0x{:02x}", i) };
                format!("{ch}={p:.4}")
            }).collect();
            eprintln!("    top5: {}", top5.join(", "));
        }
    }

    // ─── 6. OUTPUT PROJECTION WEIGHT STRUCTURE ──────────────
    eprintln!("\n--- 6. OUTPUT PROJECTION WEIGHTS ---");
    let out_in = org.output_proj.in_dim;
    let out_out = org.output_proj.out_dim;
    eprintln!("  Shape: {out_out} x {out_in} (vocab x sync_dim)");

    let w = &org.output_proj.weight;
    let w_mean: f32 = w.iter().sum::<f32>() / w.len() as f32;
    let w_std: f32 = (w.iter().map(|x| (x - w_mean).powi(2)).sum::<f32>() / w.len() as f32).sqrt();
    let w_max: f32 = w.iter().fold(f32::MIN, |a, &b| a.max(b));
    let w_min: f32 = w.iter().fold(f32::MAX, |a, &b| a.min(b));
    let w_zeros = w.iter().filter(|&&x| x.abs() < 1e-6).count();
    eprintln!("  Stats: mean={w_mean:.6} std={w_std:.6} min={w_min:.4} max={w_max:.4} zeros={w_zeros}/{}",
        w.len());

    // Per-row (per-token) norms
    let mut row_norms: Vec<f32> = (0..out_out.min(v)).map(|r| {
        let start = r * out_in;
        let end = start + out_in;
        w[start..end].iter().map(|x| x*x).sum::<f32>().sqrt()
    }).collect();
    row_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_norm = row_norms[row_norms.len() / 2];
    let min_norm = row_norms[0];
    let max_norm = row_norms[row_norms.len() - 1];
    eprintln!("  Row norms: min={min_norm:.4} median={median_norm:.4} max={max_norm:.4}");

    // Check bias
    let b_mean: f32 = org.output_proj.bias.iter().sum::<f32>() / org.output_proj.bias.len() as f32;
    let b_std: f32 = (org.output_proj.bias.iter().map(|x| (x - b_mean).powi(2)).sum::<f32>()
        / org.output_proj.bias.len() as f32).sqrt();
    eprintln!("  Bias: mean={b_mean:.6} std={b_std:.6}");

    // ─── 7. HEBBIAN STATE ───────────────────────────────────
    eprintln!("\n--- 7. HEBBIAN STATISTICS ---");
    let hebb_names = ["input", "attention", "output", "motor"];
    let hebbs = [&org.ctm.hebb_input, &org.ctm.hebb_attention, &org.ctm.hebb_output, &org.ctm.hebb_motor];
    for (name, h) in hebb_names.iter().zip(hebbs.iter()) {
        let mean_bl: f32 = h.baseline_mean.iter().sum::<f32>() / h.baseline_mean.len().max(1) as f32;
        let std_bl: f32 = (h.baseline_mean.iter().map(|x| (x - mean_bl).powi(2)).sum::<f32>()
            / h.baseline_mean.len().max(1) as f32).sqrt();
        let mean_var: f32 = h.baseline_var.iter().sum::<f32>() / h.baseline_var.len().max(1) as f32;
        eprintln!("  {name}: baseline_mean mean={mean_bl:.4} std={std_bl:.4}, baseline_var mean={mean_var:.4}, lr={:.4}",
            h.lr);
    }

    eprintln!("\n=== END DEEP DIAGNOSTIC ===");
}

/// Load images from a directory, tokenize with VQ-VAE, return token sequences.
fn load_image_tokens(path: &str) -> Vec<Vec<usize>> {
    use modgrad_runtime::regional::*;
    use modgrad_codec::vqvae::VqVae;

    let vae = VqVae::new(4096, 64);
    let mut result = Vec::new();

    // Try CIFAR-10 binary format first
    if let Ok(data) = std::fs::read(path) {
        if data.len() > 3073 {
            // CIFAR-10 binary: each record = 1 label + 3072 pixels (32×32×3)
            let n_images = data.len() / 3073;
            eprintln!("  Loading {n_images} CIFAR-10 images from {path}");
            for i in 0..n_images.min(1000) {
                let offset = i * 3073 + 1; // skip label byte
                let pixels: Vec<f32> = data[offset..offset + 3072]
                    .iter().map(|&b| b as f32 / 255.0).collect();
                let codes = vae.tokenize(&pixels);
                result.push(image_codes_to_tokens(&codes));
            }
            return result;
        }
    }

    // Try directory of raw pixel files
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().map(|e| e == "bin" || e == "raw").unwrap_or(false) {
                if let Ok(data) = std::fs::read(&p) {
                    if data.len() >= 3072 {
                        let pixels: Vec<f32> = data[..3072]
                            .iter().map(|&b| b as f32 / 255.0).collect();
                        let codes = VqVae::new(4096, 64).tokenize(&pixels);
                        result.push(image_codes_to_tokens(&codes));
                    }
                }
            }
        }
    }

    eprintln!("  Loaded {} image token sequences from {path}", result.len());
    result
}

/// Load audio WAV files, tokenize with audio codec, return token sequences.
fn load_audio_tokens(path: &str) -> Vec<Vec<usize>> {
    use modgrad_runtime::regional::*;
    use modgrad_codec::audio_codec::AudioCodec;

    let codec = AudioCodec::new_24khz();
    let mut result = Vec::new();

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().map(|e| e == "wav" || e == "raw").unwrap_or(false) {
                if let Ok(data) = std::fs::read(&p) {
                    // Simple WAV parser: skip 44-byte header, assume 16-bit PCM mono
                    let samples: Vec<f32> = if data.len() > 44 && &data[..4] == b"RIFF" {
                        data[44..].chunks_exact(2).map(|c| {
                            i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0
                        }).collect()
                    } else {
                        // Raw f32 samples
                        data.chunks_exact(4).map(|c| {
                            f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                        }).collect()
                    };

                    if samples.len() > 320 {
                        let codes = codec.tokenize(&samples);
                        // Split long audio into chunks of ~2 seconds (150 codes)
                        for chunk in codes.chunks(150) {
                            result.push(audio_codes_to_tokens(chunk));
                        }
                    }
                }
            }
        }
    }

    eprintln!("  Loaded {} audio token sequences from {path}", result.len());
    result
}

/// Interactive Neural Computer mode.
/// The model IS the running computer — computation, memory, and I/O
/// unified in the CTM's latent runtime state.
fn run_nc(
    checkpoint: &str, temperature: f32, max_tokens: usize,
    audio_in: Option<&str>, camera_in: Option<&str>, camera_fps: f32,
    audio_out: Option<&str>, image_out: Option<&str>,
    debug_port: Option<u16>,
) {
    use modgrad_runtime::regional::*;
    use modgrad_runtime::nc_io;

    let nc = if std::path::Path::new(checkpoint).exists() {
        eprintln!("Loading neural computer from {checkpoint}...");
        NeuralComputer::load(checkpoint).expect("failed to load")
    } else {
        eprintln!("No checkpoint at {checkpoint}, creating fresh NC...");
        let cfg = RegionalConfig::eight_region_multimodal(32, 2);
        let w = RegionalWeights::new(cfg);
        NeuralComputer::new(w)
    };
    let mut nc = nc;
    nc.weights.print_summary();

    // Start debug server if requested
    let debug_view: Option<std::sync::Arc<std::sync::Mutex<modgrad_runtime::nc_socket::NcDebugView>>> =
        if let Some(port) = debug_port {
            use modgrad_runtime::nc_socket;
            let view = nc_socket::NcDebugView::from_nc(&nc);
            let view = std::sync::Arc::new(std::sync::Mutex::new(view));
            let _handle = nc_socket::start_debug_server(port, view.clone());
            Some(view)
        } else {
            None
        };

    // Helper: update debug view after NC state changes
    let update_debug = |nc: &NeuralComputer, view: &Option<std::sync::Arc<std::sync::Mutex<modgrad_runtime::nc_socket::NcDebugView>>>| {
        if let Some(v) = view {
            if let Ok(mut guard) = v.try_lock() {
                *guard = modgrad_runtime::nc_socket::NcDebugView::from_nc(nc);
            }
        }
    };

    // If audio or camera inputs provided, run in streaming mode
    if audio_in.is_some() || camera_in.is_some() {
        let (tx, rx) = std::sync::mpsc::channel();

        // Spawn I/O threads
        let mut handles = Vec::new();
        if let Some(path) = audio_in {
            eprintln!("  Audio input: {path}");
            handles.push(nc_io::audio_input_thread(path, tx.clone()));
        }
        if let Some(path) = camera_in {
            eprintln!("  Camera input: {path} at {camera_fps}fps");
            handles.push(nc_io::camera_input_thread(path, camera_fps, tx.clone()));
        }

        // Keyboard input thread (so text still works alongside audio/camera)
        let tx_kb = tx.clone();
        handles.push(std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                if stdin.read_line(&mut line).unwrap_or(0) == 0 { break; }
                let input = line.trim().to_string();
                if input == "/quit" || input == "/q" {
                    tx_kb.send(nc_io::NcInput::Quit).ok();
                    break;
                }
                if tx_kb.send(nc_io::NcInput::Text(input)).is_err() { break; }
            }
        }));

        drop(tx); // close our copy so rx disconnects when all senders done

        let config = nc_io::NcStreamConfig {
            temperature,
            max_response: max_tokens,
            audio_out: audio_out.map(|s| s.to_string()),
            image_out: image_out.map(|s| s.to_string()),
        };

        nc_io::nc_stream_loop(&mut nc, rx, config);

        for h in handles { h.join().ok(); }
        eprintln!("NC shutdown.");
        return;
    }

    eprintln!("Neural Computer ready. Type text, or commands:");
    eprintln!("  /click <x> <y>     — mouse click at normalized coords");
    eprintln!("  /move <x> <y>      — mouse move");
    eprintln!("  /key <name>        — special key (enter, tab, esc, up, down, left, right)");
    eprintln!("  /ctrl <char>       — ctrl+key combo");
    eprintln!("  /state             — show NC state summary");
    eprintln!("  /save <path>       — save checkpoint");
    eprintln!("  /quit              — exit");
    eprintln!();

    let stdin = std::io::stdin();
    let mut line = String::new();

    loop {
        eprint!("nc> ");
        std::io::Write::flush(&mut std::io::stderr()).ok();
        line.clear();
        if stdin.read_line(&mut line).unwrap_or(0) == 0 { break; }
        let input = line.trim();
        if input.is_empty() { continue; }

        if input.starts_with('/') {
            let parts: Vec<&str> = input.splitn(3, ' ').collect();
            match parts[0] {
                "/click" if parts.len() >= 3 => {
                    let x: f32 = parts[1].parse().unwrap_or(0.5);
                    let y: f32 = parts[2].parse().unwrap_or(0.5);
                    let action = action_click(x, y);
                    let response = nc.act(&action, max_tokens, temperature);
                    print_nc_response(&response);
                }
                "/move" if parts.len() >= 3 => {
                    let x: f32 = parts[1].parse().unwrap_or(0.5);
                    let y: f32 = parts[2].parse().unwrap_or(0.5);
                    let action = action_mouse_move(x, y);
                    let response = nc.act(&action, max_tokens, temperature);
                    print_nc_response(&response);
                }
                "/key" if parts.len() >= 2 => {
                    let key = match parts[1] {
                        "enter" => ACT_KEY_ENTER,
                        "backspace" | "bs" => ACT_KEY_BACKSPACE,
                        "tab" => ACT_KEY_TAB,
                        "esc" | "escape" => ACT_KEY_ESCAPE,
                        "up" => ACT_KEY_UP,
                        "down" => ACT_KEY_DOWN,
                        "left" => ACT_KEY_LEFT,
                        "right" => ACT_KEY_RIGHT,
                        _ => {
                            eprintln!("  unknown key: {}", parts[1]);
                            continue;
                        }
                    };
                    let action = action_key(key);
                    let response = nc.act(&action, max_tokens, temperature);
                    print_nc_response(&response);
                }
                "/ctrl" if parts.len() >= 2 => {
                    let ch = parts[1].as_bytes().first().copied().unwrap_or(b'c');
                    let action = action_modified_key(ACT_KEY_CTRL, ch);
                    let response = nc.act(&action, max_tokens, temperature);
                    print_nc_response(&response);
                }
                "/state" => {
                    eprintln!("  history: {} tokens", nc.history.len());
                    eprintln!("  regions: {}", nc.weights.config.regions.len());
                    eprintln!("  params: {}", nc.weights.n_params());
                    // Show last 20 tokens
                    let tail: Vec<usize> = nc.history.iter().rev().take(20).copied().collect();
                    eprintln!("  last tokens: {:?}", tail.into_iter().rev().collect::<Vec<_>>());
                }
                "/save" if parts.len() >= 2 => {
                    match nc.weights.save(parts[1]) {
                        Ok(_) => eprintln!("  saved to {}", parts[1]),
                        Err(e) => eprintln!("  save failed: {e}"),
                    }
                }
                "/quit" | "/exit" | "/q" => break,
                _ => eprintln!("  unknown command: {input}"),
            }
        } else {
            // Text input — chat mode
            let response = nc.chat(input, max_tokens, temperature);
            if response.is_empty() {
                eprintln!("  (no text response)");
            } else {
                println!("{response}");
            }
        }
        // Update debug view after every interaction
        update_debug(&nc, &debug_view);
    }
    eprintln!("NC shutdown.");
}

fn print_nc_response(tokens: &[usize]) {
    use modgrad_runtime::regional::*;
    // Decode response tokens into human-readable form
    let mut text_buf = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i];
        match t {
            0..=255 => text_buf.push(t as u8),
            TOKEN_IMG_START => { flush_text(&mut text_buf); eprint!("[img:"); }
            TOKEN_IMG_END => eprint!("]"),
            TOKEN_AUD_START => { flush_text(&mut text_buf); eprint!("[aud:"); }
            TOKEN_AUD_END => eprint!("]"),
            TOKEN_VID_START => { flush_text(&mut text_buf); eprint!("[vid:"); }
            TOKEN_VID_END => eprint!("]"),
            ACT_START => { flush_text(&mut text_buf); eprint!("[act:"); }
            ACT_END => eprint!("]"),
            t if t >= TOKEN_TS_OFFSET && t < TOKEN_TS_OFFSET + TOKEN_TS_COUNT => {
                flush_text(&mut text_buf);
                let secs = (t - TOKEN_TS_OFFSET) as f32 * 0.5;
                eprint!("<{secs:.1}s>");
            }
            t if t >= TOKEN_IMG_OFFSET && t < TOKEN_IMG_OFFSET + TOKEN_IMG_CODES => {
                eprint!("{}", (t - TOKEN_IMG_OFFSET));
                if i + 1 < tokens.len() && tokens[i+1] >= TOKEN_IMG_OFFSET
                    && tokens[i+1] < TOKEN_IMG_OFFSET + TOKEN_IMG_CODES { eprint!(","); }
            }
            t if t >= TOKEN_COORD_OFFSET && t < TOKEN_COORD_OFFSET + TOKEN_COORD_COUNT => {
                let v = (t - TOKEN_COORD_OFFSET) as f32 / 255.0;
                eprint!("{v:.2}");
            }
            _ => eprint!("?{t}"),
        }
        i += 1;
    }
    flush_text(&mut text_buf);
    eprintln!();
}

fn flush_text(buf: &mut Vec<u8>) {
    if !buf.is_empty() {
        eprint!("{}", String::from_utf8_lossy(buf));
        buf.clear();
    }
}

/// Load videos from a directory of subdirectories.
/// Each subdirectory = one video, containing frame files + optional audio.wav.
fn load_video_tokens(path: &str, fps: f32) -> Vec<Vec<usize>> {
    use modgrad_runtime::regional::*;

    let mut result = Vec::new();

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                let dir = entry.path().display().to_string();
                let (frames, audio) = extract_video_frames(&dir, fps);
                if !frames.is_empty() {
                    let tokens = video_to_tokens(&frames, &audio);
                    // Split long videos into chunks that fit context
                    for chunk in tokens.chunks(512) {
                        result.push(chunk.to_vec());
                    }
                }
            }
        }
    }

    eprintln!("  Loaded {} video token sequences from {path}", result.len());
    result
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { return 0.0; }
    dot / (na * nb)
}

// ─── QEC Benchmark ─────────────────────────────────────────

fn run_qec(weights_path: &str, data_dir: &str, iterations: usize) {
    use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, CtmTickState, Linear, LayerConfig, forward_split};
    use modgrad::tabula_rasa::Dna;
    use modgrad::gpu_accel::{GpuWeightCache, forward_split_batched_gpu_nlm};
    use modgrad::kfd;
    use rayon::prelude::*;
    use std::sync::Arc;
    use std::time::Instant;

    let train_path = format!("{data_dir}/surface_train.jsonl");
    let test_path = format!("{data_dir}/surface_test.jsonl");

    if !std::path::Path::new(&train_path).exists() {
        // Fallback to data/qec/ for backwards compatibility
        let alt_train = format!("data/qec/surface_train.jsonl");
        if std::path::Path::new(&alt_train).exists() {
            return run_qec(weights_path, "data/qec", iterations);
        }
        eprintln!("ERROR: {train_path} not found. Run: python3 benchmarks/run_qec_benchmark.py");
        return;
    }

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let train = parse(&train_path);
    let test = parse(&test_path);
    let syn_dim = train[0].0.len();
    let n_train = train.len();
    let n_test = test.len();

    // Load or create weights
    let mut weights = if std::path::Path::new(weights_path).exists() {
        eprintln!("Loading weights from {weights_path}...");
        let data = std::fs::read_to_string(weights_path).unwrap();
        serde_json::from_str::<CtmWeights>(&data).unwrap()
    } else {
        eprintln!("Creating QEC brain (syn_dim={syn_dim})...");
        let dna = Dna::default();
        let ctm = Ctm::new(CtmConfig {
            iterations: 8, d_input: syn_dim, n_sync_out: 128,
            input_layer: LayerConfig { n_neurons: 128, ..dna.ctm.input_layer },
            attention_layer: LayerConfig { n_neurons: 128, ..dna.ctm.attention_layer },
            output_layer: LayerConfig { n_neurons: 128, ..dna.ctm.output_layer },
            motor_layer: LayerConfig { n_neurons: 128, ..dna.ctm.motor_layer },
            ..CtmConfig::default()
        });
        let (w, _) = ctm.into_split();
        w
    };

    let total_neurons = weights.config.input_layer.n_neurons
        + weights.config.attention_layer.n_neurons
        + weights.config.output_layer.n_neurons
        + weights.config.motor_layer.n_neurons
        + weights.config.cerebellum_layer.n_neurons
        + weights.config.basal_ganglia_layer.n_neurons
        + weights.config.insula_layer.n_neurons
        + weights.config.hippocampus_layer.n_neurons;
    let sync_dim = weights.config.n_sync_out;
    let proprio = vec![0.0f32; syn_dim];

    eprintln!("  {} neurons, {} sync dims, {} train, {} test",
        total_neurons, sync_dim, n_train, n_test);

    // Enable learning: init eligibility traces + deltas
    weights.config = CtmConfig {
        neuromod: modgrad::ctm::NeuromodConfig { hebb_syn_lr: 0.00005, ..weights.config.neuromod.clone() },
        ..weights.config.clone()
    };
    let mut learning_weights = weights;

    // Pre-training: brain learns from experience.
    // Each sample: forward → predict → check → inject reward as dopamine → weights update.
    // This is the complete biological learning loop.
    {
        use modgrad::ctm::MOD_SYNC_SCALE;
        let learn_n = n_train.min(3000);
        eprintln!("  Learning phase: {} samples, reward-driven...", learn_n);
        let t0 = Instant::now();
        let mut session = CtmSession::new(&learning_weights.config);
        session.init_syn_deltas(&learning_weights);
        session.hebbian_enabled = true;

        // Simple running readout (online, updated each sample)
        let sync_dim = learning_weights.config.n_sync_out;
        let mut readout_w = vec![0.0f32; 2 * sync_dim];
        let mut readout_b = vec![0.0f32; 2];
        let lr = 0.01;
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut last_reward = 1.0f32; // neutral start

        for (i, (syn, label)) in train.iter().take(learn_n).enumerate() {
            if i == 0 { eprintln!("    starting first sample..."); }
            let mut tick_state = learning_weights.init_tick_state();
            tick_state.modulation[MOD_SYNC_SCALE] = last_reward;

            let (_, sync, _) = forward_split(
                &learning_weights, &mut session, &mut tick_state, syn, &proprio, false);
            if i == 0 { eprintln!("    first sample done"); }

            // Simple linear prediction
            let mut logits = [readout_b[0], readout_b[1]];
            for k in 0..sync_dim.min(sync.len()) {
                logits[0] += readout_w[k] * sync[k];
                logits[1] += readout_w[sync_dim + k] * sync[k];
            }
            let pred = if logits[0] > logits[1] { 0 } else { 1 };
            let got_right = pred == *label;
            if got_right { correct += 1; }
            total += 1;

            // Inject reward as dopamine for the NEXT sample's three-factor update.
            // Correct → dopamine spike (> da_gate → learning fires)
            // Wrong → dopamine dip (< da_gate → no learning, or negative learning)
            // This is the TD-error: reward - expected_reward
            // Store reward for NEXT forward pass (this tick_state is about to be dropped)
            last_reward = if got_right { 1.5 } else { 0.5 }; // above/below da_gate=1.2

            // Update online readout (simple perceptron rule)
            if !got_right {
                for k in 0..sync_dim.min(sync.len()) {
                    readout_w[*label * sync_dim + k] += lr * sync[k];
                    readout_w[pred * sync_dim + k] -= lr * sync[k];
                }
                readout_b[*label] += lr;
                readout_b[pred] -= lr;
            }

            // Apply weight changes periodically
            if (i + 1) % 100 == 0 {
                session.apply_syn_deltas(&mut learning_weights);
                if (i + 1) % 1000 == 0 {
                    eprintln!("    sample {}: acc={:.1}%", i + 1,
                        correct as f32 / total as f32 * 100.0);
                }
            }
        }
        session.apply_syn_deltas(&mut learning_weights);
        eprintln!("  Learning done: {:.1}% ({}/{}) in {:.1}s",
            correct as f32 / total as f32 * 100.0, correct, total,
            t0.elapsed().as_secs_f64());
    }

    let weights = Arc::new(learning_weights);

    // Try GPU acceleration via KFD
    let mut hsa_device = if kfd::is_available() {
        match kfd::HsaDevice::open() {
            Ok(dev) => {
                eprintln!("  GPU: KFD available, using batched GPU synapse forward");
                Some(dev)
            }
            Err(e) => { eprintln!("  GPU: KFD open failed: {e}, falling back to CPU"); None }
        }
    } else {
        None
    };
    let gpu_cache = hsa_device.as_ref().and_then(|dev| {
        GpuWeightCache::new(dev, &weights).ok()
    });
    let nlm_cache = hsa_device.as_ref().and_then(|dev| {
        modgrad::gpu_accel::GpuNlmCache::new(dev, &weights).ok()
    });

    // Random projection: activations → 256-dim features (bypasses SyncAccumulator bottleneck)
    let act_dim = weights.init_tick_state().activations.len();
    let proj_dim = 256.min(act_dim);
    let proj: Vec<f32> = {
        use modgrad::ctm::SimpleRng;
        let mut rng = SimpleRng::new(31337);
        let scale = (1.0 / proj_dim as f32).sqrt();
        (0..proj_dim * act_dim).map(|_| rng.next_normal() * scale).collect()
    };
    eprintln!("  activations: {} dims → {} projected features", act_dim, proj_dim);

    for iter in 0..iterations {
        let t0 = Instant::now();

        // Feature collection: GPU with NLM acceleration
        let use_gpu = false; // disabled: learning modifies weights but GPU cache has stale copy
        let features: Vec<(Vec<f32>, usize)> = if use_gpu {
            let dev = hsa_device.as_mut().unwrap();
            let gpu = gpu_cache.as_ref().unwrap();
            let nlm = nlm_cache.as_ref().unwrap();
            let batch_size = 32;
            let mut all_features = Vec::with_capacity(n_train);

            for chunk_start in (0..n_train).step_by(batch_size) {
                let chunk_end = (chunk_start + batch_size).min(n_train);
                let chunk = &train[chunk_start..chunk_end];
                let b = chunk.len();

                let mut sessions: Vec<CtmSession> = (0..b)
                    .map(|_| CtmSession::new(&weights.config)).collect();
                let mut tick_states: Vec<CtmTickState> = (0..b)
                    .map(|_| weights.init_tick_state()).collect();

                let obs: Vec<&[f32]> = chunk.iter().map(|(s, _)| s.as_slice()).collect();
                let pros: Vec<&[f32]> = (0..b).map(|_| proprio.as_slice()).collect();

                let results = forward_split_batched_gpu_nlm(
                    dev, gpu, nlm, &weights,
                    &mut sessions, &mut tick_states,
                    &obs, &pros, false,
                );

                // Use activations as features (bypass sync bottleneck)
                for (i, (_sync, _tick_syncs, _signals)) in results.into_iter().enumerate() {
                    let act = &tick_states[i].activations;
                    let features: Vec<f32> = (0..proj_dim).map(|j| {
                        let row = &proj[j * act_dim..j * act_dim + act_dim.min(act.len())];
                        row.iter().zip(act.iter()).map(|(p, a)| p * a).sum::<f32>()
                    }).collect();
                    all_features.push((features, chunk[i].1));
                }
            }
            all_features
        } else {
            // CPU path: rayon across samples
            // Use raw activations → random projection as features
            train.par_iter()
                .map(|(syn, label)| {
                    let mut session = CtmSession::new(&weights.config);
                    let mut tick_state = weights.init_tick_state();
                    let (_, _sync, _) = forward_split(
                        &weights, &mut session, &mut tick_state, syn, &proprio, false);
                    // Random projection: features = proj @ activations
                    let act = &tick_state.activations;
                    let features: Vec<f32> = (0..proj_dim).map(|i| {
                        let row = &proj[i * act_dim..(i + 1) * act_dim];
                        row.iter().zip(act.iter()).map(|(p, a)| p * a).sum::<f32>()
                    }).collect();
                    (features, *label)
                }).collect()
        };

        let _collect_time = t0.elapsed();

        // LS readout — feature dim adapts to multi-tick if GPU path used
        let feat_dim = features[0].0.len();
        let mut readout = Linear::new(feat_dim, 2);
        let mut xtx = vec![0.0f32; feat_dim * feat_dim];
        let mut xty = vec![0.0f32; feat_dim * 2];
        for (f, l) in &features {
            for r in 0..feat_dim { for c in 0..feat_dim { xtx[r*feat_dim+c] += f[r]*f[c]; }}
            for r in 0..feat_dim { xty[r*2+l] += f[r]; }
        }
        // Ridge regression with lambda sweep
        let best_lambda = {
            let lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1];
            let val_split = n_train * 4 / 5;
            let mut best_acc = 0.0f32;
            let mut best_l = 1e-4f32;
            for &lam in &lambdas {
                let mut xtx_reg = xtx.clone();
                for i in 0..feat_dim { xtx_reg[i*feat_dim+i] += lam; }
                if let Some(l) = modgrad::linalg::cholesky(&xtx_reg, feat_dim) {
                    let mut test_readout = Linear::new(feat_dim, 2);
                    for cls in 0..2 {
                        let rhs: Vec<f32> = (0..feat_dim).map(|r| xty[r*2+cls]).collect();
                        let z = modgrad::linalg::forward_solve(&l, &rhs, feat_dim);
                        let w = modgrad::linalg::backward_solve(&l, &z, feat_dim);
                        for r in 0..feat_dim { test_readout.weight[cls*test_readout.in_dim+r] = w[r]; }
                    }
                    // Quick validation on last 20% of training features
                    let correct: usize = features[val_split..].iter()
                        .map(|(f, label)| {
                            let logits = test_readout.forward(f);
                            if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { 1 } else { 0 }
                        }).sum();
                    let acc = correct as f32 / (n_train - val_split) as f32;
                    if acc > best_acc { best_acc = acc; best_l = lam; }
                }
            }
            best_l
        };
        for i in 0..feat_dim { xtx[i*feat_dim+i] += best_lambda; }
        if let Some(l) = modgrad::linalg::cholesky(&xtx, feat_dim) {
            for cls in 0..2 {
                let rhs: Vec<f32> = (0..feat_dim).map(|r| xty[r*2+cls]).collect();
                let z = modgrad::linalg::forward_solve(&l, &rhs, feat_dim);
                let w = modgrad::linalg::backward_solve(&l, &z, feat_dim);
                for r in 0..feat_dim { readout.weight[cls*readout.in_dim+r] = w[r]; }
            }
        }

        // Parallel test
        let correct: usize = if use_gpu {
            let dev = hsa_device.as_mut().unwrap();
            let gpu = gpu_cache.as_ref().unwrap();
            let nlm = nlm_cache.as_ref().unwrap();
            let mut test_correct = 0usize;
            let batch_size = 32;
            for chunk_start in (0..n_test).step_by(batch_size) {
                let chunk_end = (chunk_start + batch_size).min(n_test);
                let chunk = &test[chunk_start..chunk_end];
                let b = chunk.len();

                let mut sessions: Vec<CtmSession> = (0..b)
                    .map(|_| CtmSession::new(&weights.config)).collect();
                let mut tick_states: Vec<CtmTickState> = (0..b)
                    .map(|_| weights.init_tick_state()).collect();

                let obs: Vec<&[f32]> = chunk.iter().map(|(s, _)| s.as_slice()).collect();
                let pros: Vec<&[f32]> = (0..b).map(|_| proprio.as_slice()).collect();

                let results = forward_split_batched_gpu_nlm(
                    dev, gpu, nlm, &weights,
                    &mut sessions, &mut tick_states,
                    &obs, &pros, false,
                );

                // Use activations as features
                for (i, (_sync, _tick_syncs, _signals)) in results.into_iter().enumerate() {
                    let act = &tick_states[i].activations;
                    let features: Vec<f32> = (0..proj_dim).map(|j| {
                        let row = &proj[j * act_dim..j * act_dim + act_dim.min(act.len())];
                        row.iter().zip(act.iter()).map(|(p, a)| p * a).sum::<f32>()
                    }).collect();
                    let logits = readout.forward(&features);
                    if (if logits[0] > logits[1] { 0 } else { 1 }) == chunk[i].1 {
                        test_correct += 1;
                    }
                }
            }
            test_correct
        } else {
            test.par_iter()
                .map(|(syn, label)| {
                    let mut session = CtmSession::new(&weights.config);
                    let mut tick_state = weights.init_tick_state();
                    let (_, _sync, _) = forward_split(
                        &weights, &mut session, &mut tick_state, syn, &proprio, false);
                    let act = &tick_state.activations;
                    let features: Vec<f32> = (0..proj_dim).map(|i| {
                        let row = &proj[i * act_dim..(i + 1) * act_dim];
                        row.iter().zip(act.iter()).map(|(p, a)| p * a).sum::<f32>()
                    }).collect();
                    let logits = readout.forward(&features);
                    if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { 1 } else { 0 }
                }).sum()
        };

        let acc = correct as f32 / n_test as f32;
        let elapsed = t0.elapsed();
        let throughput = (n_train + n_test) as f64 / elapsed.as_secs_f64();

        eprintln!("  [{}/{}] acc={:.1}% ({:.0} samples/sec, {:.1}s)",
            iter + 1, iterations, acc * 100.0, throughput, elapsed.as_secs_f64());
    }

    // Save weights
    let w = Arc::try_unwrap(weights).unwrap_or_else(|arc| (*arc).clone());
    let data = serde_json::to_string(&w).unwrap();
    std::fs::write(weights_path, &data).unwrap();
    let size_mb = data.len() as f64 / 1e6;
    eprintln!("\nSaved weights to {weights_path} ({size_mb:.1} MB)");
}

/// CTM benchmark suite: compare isis against original CTM paper results.
fn run_ctm_benchmark() {
    use modgrad::ctm::{Ctm, CtmConfig, LayerConfig};
    use modgrad::tabula_rasa::Dna;
    use modgrad::tasks;
    use std::time::Instant;

    eprintln!("═══════════════════════════════════════════════════");
    eprintln!("  isis CTM Ablation Study");
    eprintln!("  Finding what works for learning");
    eprintln!("═══════════════════════════════════════════════════\n");

    // Fast ablation on 3-bit parity (800 examples, 2 classes)
    // Test: Hebbian ON/OFF × ticks × neurons × readout type
    let configs: Vec<(&str, usize, usize, bool, usize)> = vec![
        // (name, neurons_per_region, ticks, hebbian, epochs)
        ("64n/4t/no_hebb",   64,  4, false, 30),
        ("64n/4t/hebb",      64,  4, true,  30),
        ("64n/12t/no_hebb",  64, 12, false, 30),
        ("64n/12t/hebb",     64, 12, true,  30),
        ("128n/12t/no_hebb", 128, 12, false, 30),
        ("128n/12t/hebb",    128, 12, true,  30),
        ("256n/12t/no_hebb", 256, 12, false, 20),
        ("256n/12t/hebb",    256, 12, true,  20),
    ];

    eprintln!("  {:>20} {:>8} {:>8} {:>8} {:>6}",
        "config", "linear", "mlp", "angeris", "time");
    eprintln!("  {}", "-".repeat(55));

    for (name, neurons, ticks, hebb, epochs) in &configs {
        let dna = Dna::default();
        let cfg = CtmConfig {
            iterations: *ticks,
            input_layer: LayerConfig { n_neurons: *neurons, ..dna.ctm.input_layer },
            attention_layer: LayerConfig { n_neurons: *neurons, ..dna.ctm.attention_layer },
            output_layer: LayerConfig { n_neurons: *neurons, ..dna.ctm.output_layer },
            motor_layer: LayerConfig { n_neurons: *neurons, ..dna.ctm.motor_layer },
            ..dna.ctm
        };
        let t0 = Instant::now();

        // Linear + Angeris
        let mut ctm = Ctm::new(cfg.clone());
        if *hebb { ctm.enable_hebbian(); }
        let data = tasks::parity_examples(3, cfg.d_input);
        let (_, linear_acc, _, angeris_acc) = tasks::train_ctm_on_task(&mut ctm, &data, *epochs, 100);

        // MLP
        let mut ctm2 = Ctm::new(cfg.clone());
        if *hebb { ctm2.enable_hebbian(); }
        let (_, mlp_acc) = tasks::train_ctm_mlp(&mut ctm2, &data, *epochs);

        eprintln!("  {:>20} {:>7.1}% {:>7.1}% {:>7.1}% {:>5.1}s",
            name, linear_acc * 100.0, mlp_acc * 100.0, angeris_acc * 100.0,
            t0.elapsed().as_secs_f64());
    }

    // Now test on harder tasks with the best config
    eprintln!("\n  Best config applied to harder tasks:\n");
    let dna = Dna::default();
    let cfg = CtmConfig {
        iterations: 12,
        input_layer: LayerConfig { n_neurons: 128, ..dna.ctm.input_layer },
        attention_layer: LayerConfig { n_neurons: 128, ..dna.ctm.attention_layer },
        output_layer: LayerConfig { n_neurons: 128, ..dna.ctm.output_layer },
        motor_layer: LayerConfig { n_neurons: 128, ..dna.ctm.motor_layer },
        ..dna.ctm
    };

    // Harder tasks with appropriately-sized configs
    let harder_tasks: Vec<(&str, CtmConfig, Vec<tasks::Example>)> = vec![
        ("8-bit parity", cfg.clone(), tasks::parity_examples(8, cfg.d_input)),
        ("XOR", cfg.clone(), tasks::xor_examples(cfg.d_input)),
        ("majority-8", cfg.clone(), tasks::majority_examples(8, cfg.d_input)),
        ("5x5 maze", cfg.clone(), tasks::maze_examples(5, cfg.d_input, 1000)),
        // 7x7 maze needs d_input >= 53
        ("7x7 maze", CtmConfig { d_input: 64, ..cfg.clone() },
            tasks::maze_examples(7, 64, 1000)),
        // 15x15 maze needs d_input >= 229
        ("15x15 maze", CtmConfig { d_input: 256, iterations: 16, ..cfg.clone() },
            tasks::maze_examples(15, 256, 500)),
    ];

    eprintln!("  {:>15} {:>8} {:>8} {:>8} {:>6}",
        "task", "linear", "mlp", "angeris", "time");
    eprintln!("  {}", "-".repeat(50));

    for (name, task_cfg, data) in &harder_tasks {
        let t0 = Instant::now();
        let mut ctm = Ctm::new(task_cfg.clone());
        ctm.enable_hebbian();
        let (_, linear_acc, _, angeris_acc) = tasks::train_ctm_on_task(&mut ctm, data, 20, 100);
        let mut ctm2 = Ctm::new(task_cfg.clone());
        ctm2.enable_hebbian();
        let (_, mlp_acc) = tasks::train_ctm_mlp(&mut ctm2, data, 20);

        eprintln!("  {:>15} {:>7.1}% {:>7.1}% {:>7.1}% {:>5.1}s",
            name, linear_acc * 100.0, mlp_acc * 100.0, angeris_acc * 100.0,
            t0.elapsed().as_secs_f64());
    }

    // Use default config for fast iteration (small brain)
    let dna = Dna::default();
    let cfg = CtmConfig {
        iterations: 12,
        ..dna.ctm
    };
    let total_neurons: usize = cfg.input_layer.n_neurons + cfg.attention_layer.n_neurons
        + cfg.output_layer.n_neurons + cfg.motor_layer.n_neurons
        + cfg.cerebellum_layer.n_neurons + cfg.basal_ganglia_layer.n_neurons
        + cfg.insula_layer.n_neurons + cfg.hippocampus_layer.n_neurons;
    eprintln!("  Config: {} neurons, {} ticks, {} sync_out, d_input={}\n",
        total_neurons, cfg.iterations, cfg.n_sync_out, cfg.d_input);

    let mut results: Vec<(&str, f32, f32, f32, &str)> = Vec::new();

    // ── 3-bit Parity ──
    {
        let t0 = Instant::now();
        eprintln!("--- 3-bit Parity ---");
        let mut ctm = Ctm::new(cfg.clone());
        ctm.enable_hebbian();
        let data = tasks::parity_examples(3, cfg.d_input);
        let (_, linear_acc, _, _) = tasks::train_ctm_on_task(&mut ctm, &data, 50, 25);
        let mut ctm2 = Ctm::new(cfg.clone());
        let (_, mlp_acc) = tasks::train_ctm_mlp(&mut ctm2, &data, 50);
        let best = linear_acc.max(mlp_acc);
        eprintln!("  linear={:.1}%  mlp={:.1}%  best={:.1}%  ({:.1}s)\n",
            linear_acc * 100.0, mlp_acc * 100.0, best * 100.0, t0.elapsed().as_secs_f64());
        results.push(("3-bit parity", best, 1.0, t0.elapsed().as_secs_f64() as f32, "N/A (paper uses 64-bit)"));
    }

    // ── 8-bit Parity ──
    {
        let t0 = Instant::now();
        eprintln!("--- 8-bit Parity ---");
        let mut ctm = Ctm::new(cfg.clone());
        ctm.enable_hebbian();
        let data = tasks::parity_examples(8, cfg.d_input);
        let (_, linear_acc, _, _) = tasks::train_ctm_on_task(&mut ctm, &data, 50, 25);
        let mut ctm2 = Ctm::new(cfg.clone());
        let (_, mlp_acc) = tasks::train_ctm_mlp(&mut ctm2, &data, 50);
        let best = linear_acc.max(mlp_acc);
        eprintln!("  linear={:.1}%  mlp={:.1}%  best={:.1}%  ({:.1}s)\n",
            linear_acc * 100.0, mlp_acc * 100.0, best * 100.0, t0.elapsed().as_secs_f64());
        results.push(("8-bit parity", best, 1.0, t0.elapsed().as_secs_f64() as f32, "N/A"));
    }

    // ── 16-bit Parity ──
    {
        let t0 = Instant::now();
        eprintln!("--- 16-bit Parity ---");
        let mut ctm = Ctm::new(cfg.clone());
        let data = tasks::parity_examples_large(16, cfg.d_input, 5000);
        let (_, linear_acc, _, _) = tasks::train_ctm_on_task(&mut ctm, &data, 30, 15);
        let mut ctm2 = Ctm::new(cfg.clone());
        let (_, mlp_acc) = tasks::train_ctm_mlp(&mut ctm2, &data, 30);
        let best = linear_acc.max(mlp_acc);
        eprintln!("  linear={:.1}%  mlp={:.1}%  best={:.1}%  ({:.1}s)\n",
            linear_acc * 100.0, mlp_acc * 100.0, best * 100.0, t0.elapsed().as_secs_f64());
        results.push(("16-bit parity", best, 1.0, t0.elapsed().as_secs_f64() as f32, "N/A"));
    }

    // ── 64-bit Parity (CTM paper benchmark) ──
    {
        let t0 = Instant::now();
        eprintln!("--- 64-bit Parity (CTM paper benchmark) ---");
        let cfg64 = CtmConfig { d_input: 64, ..cfg.clone() };
        let mut ctm = Ctm::new(cfg64.clone());
        let data = tasks::parity_examples_large(64, 64, 10000);
        let (_, linear_acc, _, _) = tasks::train_ctm_on_task(&mut ctm, &data, 20, 10);
        let mut ctm2 = Ctm::new(cfg64.clone());
        let (_, mlp_acc) = tasks::train_ctm_mlp(&mut ctm2, &data, 20);
        let best = linear_acc.max(mlp_acc);
        eprintln!("  linear={:.1}%  mlp={:.1}%  best={:.1}%  ({:.1}s)\n",
            linear_acc * 100.0, mlp_acc * 100.0, best * 100.0, t0.elapsed().as_secs_f64());
        results.push(("64-bit parity", best, 1.0, t0.elapsed().as_secs_f64() as f32, "~100% (original CTM, 75 ticks)"));
    }

    // ── XOR ──
    {
        let t0 = Instant::now();
        eprintln!("--- XOR ---");
        let mut ctm = Ctm::new(cfg.clone());
        ctm.enable_hebbian();
        let data = tasks::xor_examples(cfg.d_input);
        let (_, linear_acc, _, _) = tasks::train_ctm_on_task(&mut ctm, &data, 50, 25);
        let mut ctm2 = Ctm::new(cfg.clone());
        let (_, mlp_acc) = tasks::train_ctm_mlp(&mut ctm2, &data, 50);
        let best = linear_acc.max(mlp_acc);
        eprintln!("  linear={:.1}%  mlp={:.1}%  best={:.1}%  ({:.1}s)\n",
            linear_acc * 100.0, mlp_acc * 100.0, best * 100.0, t0.elapsed().as_secs_f64());
        results.push(("XOR", best, 1.0, t0.elapsed().as_secs_f64() as f32, "trivial for original CTM"));
    }

    // ── Majority Vote ──
    {
        let t0 = Instant::now();
        eprintln!("--- Majority Vote (8-bit) ---");
        let mut ctm = Ctm::new(cfg.clone());
        ctm.enable_hebbian();
        let data = tasks::majority_examples(8, cfg.d_input);
        let (_, linear_acc, _, _) = tasks::train_ctm_on_task(&mut ctm, &data, 50, 25);
        let mut ctm2 = Ctm::new(cfg.clone());
        let (_, mlp_acc) = tasks::train_ctm_mlp(&mut ctm2, &data, 50);
        let best = linear_acc.max(mlp_acc);
        eprintln!("  linear={:.1}%  mlp={:.1}%  best={:.1}%  ({:.1}s)\n",
            linear_acc * 100.0, mlp_acc * 100.0, best * 100.0, t0.elapsed().as_secs_f64());
        results.push(("majority-8", best, 1.0, t0.elapsed().as_secs_f64() as f32, "trivial for original CTM"));
    }

    // ── 5x5 Maze ──
    {
        let t0 = Instant::now();
        eprintln!("--- 5x5 Maze ---");
        let mut ctm = Ctm::new(cfg.clone());
        ctm.enable_hebbian();
        let data = tasks::maze_examples(5, cfg.d_input, 2000);
        let (_, linear_acc, _, _) = tasks::train_ctm_on_task(&mut ctm, &data, 30, 15);
        let mut ctm2 = Ctm::new(cfg.clone());
        let (_, mlp_acc) = tasks::train_ctm_mlp(&mut ctm2, &data, 30);
        let best = linear_acc.max(mlp_acc);
        eprintln!("  linear={:.1}%  mlp={:.1}%  best={:.1}%  ({:.1}s)\n",
            linear_acc * 100.0, mlp_acc * 100.0, best * 100.0, t0.elapsed().as_secs_f64());
        results.push(("5x5 maze", best, 0.9, t0.elapsed().as_secs_f64() as f32, "~90%+ on 39x39 (original CTM)"));
    }

    // ── Summary ──
    eprintln!("\n═══════════════════════════════════════════════════");
    eprintln!("  {:>20} {:>8} {:>8} {:>6}  {}", "Task", "isis", "CTM", "time", "CTM reference");
    eprintln!("  {}", "-".repeat(75));
    for (name, isis_acc, _ctm_ref, secs, ctm_note) in &results {
        eprintln!("  {:>20} {:>7.1}% {:>8} {:>5.0}s  {}",
            name, isis_acc * 100.0, ctm_note.split(' ').next().unwrap_or("?"), secs, ctm_note);
    }
    eprintln!("  {}", "-".repeat(75));
    eprintln!("═══════════════════════════════════════════════════\n");
}
