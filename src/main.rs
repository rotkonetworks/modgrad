//! isis — 8-region hierarchical CTM runtime built on the modgrad SDK.
//!
//! Commands:
//!   isis train model.bin [--multimodal] [--debug-port 4747]
//!   isis nc model.bin [--audio mic.wav] [--camera frames/] [--debug-port 4747]

use modgrad_ctm::graph::*;
use modgrad_runtime::nc_socket;
use modgrad_runtime::nc_io;
use modgrad_runtime::curriculum;

use clap::{Parser, Subcommand};
use std::io::{self, Write, BufRead};

#[derive(Parser)]
#[command(name = "isis", version, about = "8-region hierarchical CTM — a neural computer")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train on staged byte curriculum (BPTT + AdamW)
    Train {
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        #[arg(short, long)]
        curriculum: Option<String>,
        #[arg(long)]
        multimodal: bool,
        #[arg(long)]
        images: Option<String>,
        #[arg(long)]
        audio: Option<String>,
        #[arg(long)]
        video: Option<String>,
        #[arg(long, default_value = "2.0")]
        video_fps: f32,
        /// Enable GPU dispatch (KFD/CUDA/Vulkan) for linear ops.
        #[arg(long)]
        gpu: bool,
        #[arg(long)]
        debug_port: Option<u16>,
    },
    /// Interactive neural computer
    Nc {
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        #[arg(short, long, default_value = "0.8")]
        temperature: f32,
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        #[arg(long)]
        audio: Option<String>,
        #[arg(long)]
        camera: Option<String>,
        #[arg(long, default_value = "2.0")]
        camera_fps: f32,
        #[arg(long)]
        audio_out: Option<String>,
        #[arg(long)]
        image_out: Option<String>,
        #[arg(long)]
        debug_port: Option<u16>,
    },
    /// Generate text from a trained model
    Generate {
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        /// Prompt text
        #[arg(default_value = "the ")]
        prompt: String,
        #[arg(short, long, default_value = "200")]
        max_tokens: usize,
        #[arg(short, long, default_value = "0.8")]
        temperature: f32,
    },
    /// Run as a headless daemon (NC service on TCP port)
    Daemon {
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        #[arg(short, long, default_value = "4747")]
        port: u16,
    },
    /// Send a command to a running daemon
    Send {
        /// Text to inject
        text: String,
        #[arg(long, default_value = "127.0.0.1:4747")]
        addr: String,
    },
    /// Learn from raw data — no curriculum, no phases, just tokens
    Learn {
        #[arg(default_value = "model.bin")]
        checkpoint: String,
        /// Directory, file(s), or .jsonl with token pairs to learn from.
        #[arg(required = true)]
        data: Vec<String>,
        #[arg(long, default_value = "32")]
        context: usize,
        /// Vocabulary size. 256 = raw bytes, 8192 = VQGAN visual tokens.
        #[arg(long, default_value = "256")]
        vocab: usize,
        /// Enable GPU dispatch (KFD/CUDA/Vulkan) for linear ops.
        #[arg(long)]
        gpu: bool,
        /// Medium model (~55M params, d_model=256). CPU+GPU balanced.
        #[arg(long)]
        medium: bool,
        /// Large model (~81M params, d_model=512). Full GPU.
        #[arg(long)]
        large: bool,
        /// Billion-scale (~1B params, d_model=1024). Needs ~19GB RAM.
        #[arg(long)]
        billion: bool,
        #[arg(long)]
        debug_port: Option<u16>,
    },
    /// Show available compute devices
    Devices,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train { checkpoint, curriculum, multimodal, images, audio, video, video_fps, gpu, debug_port } => {
            if gpu { modgrad_compute::neuron::enable_gpu(); }
            develop_staged(&checkpoint, curriculum.as_deref(), multimodal,
                images.as_deref(), audio.as_deref(), video.as_deref(), video_fps, debug_port);
        }
        Commands::Nc { checkpoint, temperature, max_tokens, audio, camera, camera_fps, audio_out, image_out, debug_port } => {
            run_nc(&checkpoint, temperature, max_tokens,
                audio.as_deref(), camera.as_deref(), camera_fps,
                audio_out.as_deref(), image_out.as_deref(), debug_port);
        }
        Commands::Generate { checkpoint, prompt, max_tokens, temperature } => {
            run_generate(&checkpoint, &prompt, max_tokens, temperature);
        }
        Commands::Learn { checkpoint, data, context, vocab, gpu, medium, large, billion, debug_port } => {
            if gpu { modgrad_compute::neuron::enable_gpu(); }
            learn(&checkpoint, &data, context, vocab, medium, large, billion, debug_port);
        }
        Commands::Daemon { checkpoint, port } => {
            run_daemon(&checkpoint, port);
        }
        Commands::Send { text, addr } => {
            send_command(&text, &addr);
        }
        Commands::Devices => {
            show_devices();
        }
    }
}

// ─── Generate ─────────────────────────────────────────────

fn run_generate(checkpoint: &str, prompt: &str, max_tokens: usize, temperature: f32) {
    let w = RegionalWeights::load(checkpoint)
        .unwrap_or_else(|e| { eprintln!("Failed to load {checkpoint}: {e}"); std::process::exit(1); });
    w.print_summary();

    let mut nc = NeuralComputer::new(w);
    let response = nc.chat(prompt, max_tokens, temperature);
    print!("{prompt}{response}");
    println!();
}

// ─── Daemon ───────────────────────────────────────────────

fn run_daemon(checkpoint: &str, port: u16) {
    let w = if std::path::Path::new(checkpoint).exists() {
        eprintln!("Loading {checkpoint}...");
        RegionalWeights::load(checkpoint)
            .unwrap_or_else(|e| { eprintln!("Failed: {e}"); std::process::exit(1); })
    } else {
        eprintln!("No checkpoint at {checkpoint}, creating fresh 8-region model...");
        let cfg = RegionalConfig::eight_region(32, 256, 2);
        RegionalWeights::new(cfg)
    };
    w.print_summary();

    let mut nc = NeuralComputer::new(w);

    // Start debug server (this IS the daemon — accepts commands via the debug protocol)
    let view = nc_socket::NcDebugView::from_nc(&nc);
    let view = std::sync::Arc::new(std::sync::Mutex::new(view));
    let handle = nc_socket::start_debug_server(port, view.clone());

    eprintln!("Daemon running on port {port}. Ctrl+C to stop.");
    eprintln!("Connect with: modgrad-debugger 127.0.0.1:{port}");
    eprintln!("Or send text: isis send \"hello world\" --addr 127.0.0.1:{port}");

    // Block forever, updating state when debug clients inject tokens
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nShutting down...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    }).ok();

    while running.load(std::sync::atomic::Ordering::SeqCst) {
        // Check for injected tokens from debug clients
        if let Some(event) = handle.poll_control() {
            match event {
                nc_socket::DebugEvent::Inject(tokens) => {
                    let response = nc.act(&tokens, 256, 0.8);
                    // Update view for debugger
                    if let Ok(mut v) = view.try_lock() {
                        *v = nc_socket::NcDebugView::from_nc(&nc);
                    }
                    // Print response as text
                    for &t in &response {
                        if t < 256 { print!("{}", t as u8 as char); }
                    }
                    io::stdout().flush().ok();
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

    // Save on exit
    if let Err(e) = nc.weights.save(checkpoint) {
        eprintln!("Save failed: {e}");
    } else {
        eprintln!("Saved to {checkpoint}");
    }
}

// ─── Send ─────────────────────────────────────────────────

fn send_command(text: &str, addr: &str) {
    use std::io::{Read, Write as IoWrite};
    use std::net::TcpStream;

    let mut stream = match TcpStream::connect(addr) {
        Ok(s) => s,
        Err(e) => { eprintln!("Can't connect to {addr}: {e}"); std::process::exit(1); }
    };
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).ok();

    // Send InjectText request via debug protocol
    let req = nc_socket::DebugRequest::InjectText { text: text.to_string() };
    let data = bincode::serialize(&req).expect("serialize failed");
    let len = data.len() as u32;
    stream.write_all(&len.to_le_bytes()).ok();
    stream.write_all(&data).ok();
    stream.flush().ok();

    // Read response
    let mut len_buf = [0u8; 4];
    if stream.read_exact(&mut len_buf).is_ok() {
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        if stream.read_exact(&mut buf).is_ok() {
            if let Ok(resp) = bincode::deserialize::<nc_socket::DebugResponse>(&buf) {
                match resp {
                    nc_socket::DebugResponse::Ok => eprintln!("Sent."),
                    nc_socket::DebugResponse::Error { msg } => eprintln!("Error: {msg}"),
                    _ => eprintln!("Response: {resp:?}"),
                }
            }
        }
    }
}

// ─── Devices ──────────────────────────────────────────────

fn show_devices() {
    eprintln!("Compute devices:");
    eprintln!("  CPU: available (rayon, {} threads)", rayon::current_num_threads());

    #[cfg(feature = "cuda")]
    eprintln!("  CUDA: enabled");
    #[cfg(not(feature = "cuda"))]
    eprintln!("  CUDA: disabled (build with --features cuda)");

    // Check KFD (AMD GPU)
    if std::path::Path::new("/dev/kfd").exists() {
        eprintln!("  AMD KFD: /dev/kfd present");
    } else {
        eprintln!("  AMD KFD: not available");
    }
}
/// Extract a JSON array of integers from a line by key name.
/// Minimal parser — avoids pulling in serde_json for a simple pattern.
fn extract_json_array(line: &str, key: &str) -> Option<Vec<usize>> {
    let pattern = format!("\"{}\":", key);
    let start = line.find(&pattern)?;
    let after_key = &line[start + pattern.len()..];
    let bracket_start = after_key.find('[')?;
    let bracket_end = after_key.find(']')?;
    let inner = &after_key[bracket_start + 1..bracket_end];
    let tokens: Vec<usize> = inner.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if tokens.is_empty() { None } else { Some(tokens) }
}

fn generate_multimodal_pairs() -> Vec<Vec<usize>> {
    // graph types imported at crate level
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
    // graph types imported at crate level
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

/// Learn from raw bytes. No curriculum, no phases, no graduation.
/// Reads every file in the given paths as a byte stream and predicts next bytes.
/// The model's exit gate, regional specialization, and sync dynamics
/// self-organize around whatever structure exists in the data.
fn learn(
    save_path: &str,
    data_paths: &[String],
    context_len: usize,
    vocab: usize,
    medium: bool,
    large: bool,
    billion: bool,
    debug_port: Option<u16>,
) {
    // Gather all data as token sequences.
    // Two modes:
    //   - .jsonl files: read {input_tokens, output_tokens} pairs, concatenate into sequences
    //   - everything else: read as raw bytes (each byte is a token 0-255)
    let mut all_tokens: Vec<usize> = Vec::new();

    for path in data_paths {
        let p = std::path::Path::new(path);

        if p.extension().map_or(false, |e| e == "jsonl") {
            // JSONL: each line has input_tokens + output_tokens
            if let Ok(text) = std::fs::read_to_string(p) {
                let mut n_samples = 0usize;
                for line in text.lines() {
                    if line.is_empty() { continue; }
                    // Minimal JSON parsing — extract token arrays
                    if let (Some(input), Some(output)) = (
                        extract_json_array(line, "input_tokens"),
                        extract_json_array(line, "output_tokens"),
                    ) {
                        all_tokens.extend_from_slice(&input);
                        all_tokens.extend_from_slice(&output);
                        n_samples += 1;
                    }
                }
                eprintln!("  + {path} ({n_samples} token pairs)");
            }
        } else if p.is_dir() {
            if let Ok(entries) = std::fs::read_dir(p) {
                let mut files: Vec<_> = entries.filter_map(|e| e.ok())
                    .filter(|e| e.path().is_file())
                    .collect();
                files.sort_by_key(|e| e.path());
                for entry in files {
                    let ep = entry.path();
                    if ep.extension().map_or(false, |e| e == "jsonl") {
                        // Recurse into JSONL files in directory
                        if let Ok(text) = std::fs::read_to_string(&ep) {
                            let mut n = 0usize;
                            for line in text.lines() {
                                if line.is_empty() { continue; }
                                if let (Some(input), Some(output)) = (
                                    extract_json_array(line, "input_tokens"),
                                    extract_json_array(line, "output_tokens"),
                                ) {
                                    all_tokens.extend_from_slice(&input);
                                    all_tokens.extend_from_slice(&output);
                                    n += 1;
                                }
                            }
                            eprintln!("  + {} ({n} token pairs)", ep.display());
                        }
                    } else if let Ok(data) = std::fs::read(&ep) {
                        eprintln!("  + {} ({} bytes)", ep.display(), data.len());
                        for &b in &data { all_tokens.push(b as usize); }
                    }
                }
            }
        } else if let Ok(data) = std::fs::read(path) {
            eprintln!("  + {path} ({} bytes)", data.len());
            for &b in &data { all_tokens.push(b as usize); }
        }
    }

    if all_tokens.is_empty() {
        eprintln!("No data found. Provide files, directories, or .jsonl token pairs.");
        return;
    }

    eprintln!("Data: {} tokens, vocab: {}", all_tokens.len(), vocab);

    // Model size from filename (same convention as develop_staged)
    let (embed_dim, n_regions, ticks) = if save_path.contains("large") {
        (128, 8, 4)
    } else if save_path.contains("medium") {
        (64, 8, 3)
    } else if save_path.contains("tiny") {
        (16, 4, 2)
    } else {
        (32, 8, 2)
    };

    // Load or create model
    let mut w = if std::path::Path::new(save_path).exists() {
        eprintln!("Loading {save_path}...");
        RegionalWeights::load(save_path).expect("failed to load")
    } else {
        let cfg = if billion {
            eprintln!("Creating billion-scale model (d_model=1024, ~1B params)...");
            RegionalConfig::eight_region_billion(embed_dim, vocab, ticks)
        } else if large {
            eprintln!("Creating large model (d_model=512, ~81M params)...");
            RegionalConfig::eight_region_large(embed_dim, vocab, ticks)
        } else if medium {
            eprintln!("Creating medium model (d_model=256, ~55M params)...");
            RegionalConfig::eight_region_medium(embed_dim, vocab, ticks)
        } else if n_regions <= 4 {
            RegionalConfig::four_region(embed_dim, vocab, ticks)
        } else {
            RegionalConfig::eight_region(embed_dim, vocab, ticks)
        };
        RegionalWeights::new(cfg)
    };
    w.print_summary();

    let opt_path = save_path.replace(".bin", ".opt.bin");
    let mut opt = if std::path::Path::new(&opt_path).exists() {
        RegionalAdamW::load(&opt_path).unwrap_or_else(|_| RegionalAdamW::new(&w))
    } else {
        {
            // Scale lr and grad clip with model size
            let (lr, clip) = if w.n_params() > 50_000_000 { (3e-4, 1.0) }
                             else if w.n_params() > 10_000_000 { (1e-3, 2.0) }
                             else { (3e-3, 5.0) };
            RegionalAdamW::new(&w).with_lr(lr).with_wd(0.001).with_clip(clip)
        }
    };

    // Ctrl+C → save and exit
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nSaving...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    }).ok();

    // Debug server
    let debug_nc: Option<std::sync::Arc<std::sync::Mutex<modgrad_runtime::nc_socket::NcDebugView>>> =
        if let Some(port) = debug_port {
            let nc_tmp = NeuralComputer::new(w.clone());
            let view = nc_socket::NcDebugView::from_nc(&nc_tmp);
            let view = std::sync::Arc::new(std::sync::Mutex::new(view));
            let _handle = nc_socket::start_debug_server(port, view.clone());
            eprintln!("Debugger on port {port}");
            Some(view)
        } else {
            None
        };

    // ── Learn ──
    let mut grads = RegionalGradients::zeros(&w);
    let mut total_tokens = opt.step as u64;
    let mut step = 0u64;
    let mut loss_sum = 0.0f32;
    let mut correct_sum = 0usize;
    let mut tokens_since_report = 0usize;
    let n_data = all_tokens.len();
    let mut offset = 0usize;

    eprintln!("\nLearning... (Ctrl+C to save and stop)\n");

    while running.load(std::sync::atomic::Ordering::SeqCst) {
        // Next chunk from the token stream (wraps around)
        let end = (offset + context_len + 1).min(n_data);
        if end - offset < 2 {
            offset = 0; // wrap
            continue;
        }
        let chunk = &all_tokens[offset..end];

        grads.zero();
        let mut chunk_loss = 0.0f32;
        let mut chunk_correct = 0usize;
        let n = chunk.len() - 1;

        for pos in 0..n {
            let (loss, pred) = regional_train_token(&w, &mut grads, chunk[pos], chunk[pos + 1]);
            chunk_loss += loss;
            if pred == chunk[pos + 1] { chunk_correct += 1; }
        }

        opt.step(&mut w, &grads);

        // ── Dream phase: free-running rollout with regret correction ──
        // Like embryonic spontaneous neural activity — tests circuit coherence.
        // TODO: re-enable once GPU path is stable
        if false && step % 20 == 0 && n_data > context_len * 2 {
            let seed_pos = ((step * 7919) as usize) % (n_data - 9);
            let ground_truth = &all_tokens[seed_pos + 1..seed_pos + 9];
            let mut dream_grads = RegionalGradients::zeros(&w);
            let _dream_loss = dream_step(
                &w, &mut dream_grads, all_tokens[seed_pos],
                ground_truth, 8, 0.3,
            );
            dream_grads.apply(&mut w, opt.lr * 0.3, 1.0);
        }

        loss_sum += chunk_loss / n as f32;
        correct_sum += chunk_correct;
        tokens_since_report += n;
        total_tokens += n as u64;
        step += 1;
        offset += context_len; // advance through data

        // Update debug view: run inference with router disabled to avoid NaN
        if let Some(ref view) = debug_nc {
            if step % 100 == 0 {
                if let Ok(mut guard) = view.try_lock() {
                    guard.history = vec![step as usize];
                    // Temporarily disable router for stable inference snapshot
                    let saved_router = w.router.take();
                    let mut ss = RegionalState::new(&w);
                    let obs = w.embed(chunk[n.saturating_sub(1)]);
                    let snap = regional_forward(&w, &mut ss, obs);
                    w.router = saved_router;
                    guard.region_activations = snap.region_activations;
                    guard.global_sync = snap.global_sync;
                    guard.exit_lambdas = snap.exit_lambdas;
                    guard.ticks_used = snap.ticks_used;
                }
            }
        }

        // Report
        if step % 100 == 0 {
            let avg_loss = loss_sum / 100.0;
            let avg_acc = correct_sum as f32 / tokens_since_report.max(1) as f32;
            let progress = (offset as f64 / n_data as f64 * 100.0).min(100.0);
            eprintln!("step {step:6} | loss {avg_loss:.3} | acc {avg_acc:.1}% | {total_tokens} tokens | data {progress:.0}%",
                avg_acc = avg_acc * 100.0);
            loss_sum = 0.0;
            correct_sum = 0;
            tokens_since_report = 0;
        }

        // Save periodically
        if step % 5000 == 0 {
            w.save(save_path).expect("save failed");
            opt.save(&opt_path).expect("opt save failed");
            eprintln!("  [saved]");
        }

        // Wrap around when we've seen all data
        if offset >= n_data {
            offset = 0;
            eprintln!("  --- epoch complete ---");
        }
    }

    // Final save
    w.save(save_path).expect("save failed");
    opt.save(&opt_path).expect("opt save failed");
    eprintln!("\nSaved to {save_path} ({total_tokens} tokens learned)");
}

/// Sync diversity diagnostic: measure how different sync patterns are across prompts.
fn load_image_tokens(path: &str) -> Vec<Vec<usize>> {
    // graph types imported at crate level
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
    // graph types imported at crate level
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
    // graph types imported at crate level
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
    // graph types imported at crate level
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
fn load_video_tokens(path: &str, _fps: f32) -> Vec<Vec<usize>> {
    let mut result = Vec::new();

    // Each subdirectory = one video with frame files + optional audio.wav
    // TODO: implement with modgrad_codec when needed
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                eprintln!("  video loading not yet implemented for {}", entry.path().display());
            }
        }
    }

    eprintln!("  Loaded {} video token sequences from {path}", result.len());
    result
}

