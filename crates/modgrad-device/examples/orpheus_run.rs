//! orpheus_run — drive Orpheus 3B through the unified `llm` module and
//! verify it produces audio tokens with the proper prompt format.
//!
//! Orpheus's prompt template (per canopyai/Orpheus-TTS engine_class.py
//! "larger" branch):
//!
//!   [128259] + tokenize("voice: text") + [128009, 128260, 128261, 128257]
//!
//! After feeding that, the model should autoregressively emit tokens in
//! the audio range (id ≥ 128256), which decode to SNAC codebook entries
//! via id - 128256 = (codebook_index * 4096) + code.
//!
//! Run:
//!   cargo run --release -p modgrad-device --example orpheus_run --
//!     [--gen N]      # generate N tokens after the prompt (default 32)
//!     [model.gguf]   # default: /steam/rotko/modgrad/models/orpheus-3b-q4km/...
//!
//! Hardcoded test prompt for now: tokenize("tara: hello world") is
//! approximated as a few plausible BPE ids — the goal here is the
//! format-verification, not lexically perfect output.

use std::io::Cursor;

use modgrad_device::kfd::HsaDevice;
use modgrad_device::kfd::dispatch_queue::GpuQueue;
use modgrad_device::kfd::gguf::GgufFile;
use modgrad_device::kfd::llm::{
    LlmCache, LlmConfig, LlmScratch, LlmWeights, forward_token,
};

const AUDIO_TOKEN_BASE: u32 = 128256;  // vocab[128256..156928] = 7 codebooks × 4096 audio tokens

fn main() {
    let mut args: Vec<String> = std::env::args().collect();

    // crude flag parse
    let mut n_gen = 32usize;
    if let Some(i) = args.iter().position(|a| a == "--gen") {
        n_gen = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(32);
        args.drain(i..=i + 1);
    }
    let path = args.get(1).cloned().unwrap_or_else(|| {
        "/steam/rotko/modgrad/models/orpheus-3b-q4km/orpheus-3b-0.1-ft.Q4_K_M.gguf".into()
    });

    // Orpheus "larger" prompt template:
    //   start [128259] + body + end [128009, 128260, 128261, 128257]
    // For body we use a tiny stand-in BPE sequence ("tara: hi" approximated).
    // Lexical content doesn't matter for format verification — we want to see
    // whether the model emits audio tokens (id ≥ 128256) after this prompt.
    let body: Vec<u32> = vec![
        128000,      // <|begin_of_text|> (often part of llama-3 prompts)
        83, 5169, 25, // "tara:" approximated (3 BPE chunks)
        24748,        // " hi"
    ];
    let mut prompt: Vec<u32> = vec![128259];
    prompt.extend_from_slice(&body);
    prompt.extend_from_slice(&[128009, 128260, 128261, 128257]);

    eprintln!("[1/5] mmap {path}");
    let file = std::fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&mut Cursor::new(&file)).expect("parse gguf");

    eprintln!("[2/5] build LlmConfig (llama3 preset)");
    let mut cfg = LlmConfig::llama3();
    cfg.populate_from_gguf(&gguf).expect("populate cfg");
    eprintln!(
        "  {}L d={} ff={} h={}/{}kv head={} vocab={} rope_base={}",
        cfg.n_layers, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv_heads,
        cfg.head_dim, cfg.vocab_size, cfg.rope_base
    );

    eprintln!("[3/5] open GPU + upload weights");
    let dev = HsaDevice::open().expect("open /dev/kfd");
    let mut queue = GpuQueue::new();
    let weights = LlmWeights::load_from_gguf(&gguf, &file, &dev, &mut queue)
        .expect("load weights");

    eprintln!("[4/5] allocate KV cache + scratch (max_seq={})",
        prompt.len() + n_gen + 8);
    let max_seq = prompt.len() + n_gen + 8;
    let mut cache = LlmCache::allocate(&cfg, &dev, &mut queue, max_seq).expect("alloc cache");
    let mut scratch = LlmScratch::allocate(&cfg, &dev, &mut queue).expect("alloc scratch");

    eprintln!("[5/5] prompt of {} tokens, then generate {}", prompt.len(), n_gen);
    let mut dev = dev;
    let mut logits = Vec::new();
    let t_prompt = std::time::Instant::now();
    for &tok in &prompt {
        logits = forward_token(&cfg, &weights, &mut cache, &mut scratch,
                               &mut dev, &mut queue, tok);
    }
    eprintln!("  prompt processed in {:.1}s", t_prompt.elapsed().as_secs_f32());

    let mut audio_token_count = 0usize;
    let mut last_audio_codebooks: Vec<(usize, u32)> = Vec::new();
    let mut generated_ids = Vec::with_capacity(n_gen);

    for step in 0..n_gen {
        // argmax sampling
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v { best = i; best_v = v; }
        }
        let next = best as u32;
        generated_ids.push(next);

        let kind = classify(next);
        if let TokenKind::Audio { codebook, code } = kind {
            audio_token_count += 1;
            last_audio_codebooks.push((codebook, code));
        }

        eprintln!("  [{step:3}] tok={next:6} {} logit={:.2}",
            kind.label(), best_v);

        // Stop early if we hit an obvious end marker
        if next == 128009 || next == 128001 {
            eprintln!("  end-of-text token at step {step}, stopping");
            break;
        }

        logits = forward_token(&cfg, &weights, &mut cache, &mut scratch,
                               &mut dev, &mut queue, next);
    }

    eprintln!("\n=== summary ===");
    eprintln!("  generated  : {} tokens", generated_ids.len());
    eprintln!("  audio tokens: {}/{} ({:.0}%)",
        audio_token_count, generated_ids.len(),
        100.0 * audio_token_count as f32 / generated_ids.len().max(1) as f32);
    if !last_audio_codebooks.is_empty() {
        eprintln!("  codebook distribution (audio tokens only):");
        let mut by_cb = [0usize; 7];
        for (cb, _) in &last_audio_codebooks {
            if *cb < 7 { by_cb[*cb] += 1; }
        }
        for (i, c) in by_cb.iter().enumerate() {
            eprintln!("    cb{i}: {c}");
        }
    }
}

enum TokenKind {
    BeginOfText,           // 128000
    EndOfText,             // 128001 / 128009
    OrpheusStart,          // 128259 (audio block start)
    OrpheusEnd { id: u32 },// 128260, 128261, 128257 (audio block end markers)
    Audio { codebook: usize, code: u32 },
    Regular(u32),
}

impl TokenKind {
    fn label(&self) -> String {
        match self {
            TokenKind::BeginOfText            => "[<|begin_of_text|>]".into(),
            TokenKind::EndOfText              => "[<|end_of_text|>]".into(),
            TokenKind::OrpheusStart           => "[<|orpheus_start|>]".into(),
            TokenKind::OrpheusEnd { id }      => format!("[<|orpheus_end_{id}|>]"),
            TokenKind::Audio { codebook, code } => format!("[audio cb{codebook}={code}]"),
            TokenKind::Regular(_)             => "(text)".into(),
        }
    }
}

fn classify(tok: u32) -> TokenKind {
    match tok {
        128000 => TokenKind::BeginOfText,
        128001 | 128009 => TokenKind::EndOfText,
        128259 => TokenKind::OrpheusStart,
        128260 | 128261 | 128257 => TokenKind::OrpheusEnd { id: tok },
        t if t >= AUDIO_TOKEN_BASE => {
            let local = (t - AUDIO_TOKEN_BASE) as usize;
            TokenKind::Audio { codebook: local / 4096, code: (local % 4096) as u32 }
        }
        _ => TokenKind::Regular(tok),
    }
}
