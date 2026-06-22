//! sonotxt — rocm tts CLI binary.
//!
//! Self-contained: takes text on the command line, downloads + caches
//! the Orpheus 3B GGUF and SNAC 24kHz weights on first run, runs the
//! whole pipeline locally on your AMD GPU (or CPU fallback), and writes
//! a 24 kHz mono WAV.

use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Instant;

use modgrad_codec::snac::SnacDecoder24k;
use modgrad_device::kfd::HsaDevice;
use modgrad_device::kfd::dispatch_queue::GpuQueue;
use modgrad_device::kfd::gguf::GgufFile;
use modgrad_device::kfd::llm::{
    LlmCache, LlmConfig, LlmScratch, LlmWeights, forward_token, print_and_reset_profile,
};
use modgrad_device::kfd::tokenizer::LlamaBpeTokenizer;

// ─── Model registry ─────────────────────────────────────────────────────────

const ORPHEUS_GGUF_URL: &str =
    "https://huggingface.co/DeviantYC/orpheus-3b-Q4_K_M-GGUF/resolve/main/orpheus-3b-0.1-ft.Q4_K_M.gguf";
const ORPHEUS_GGUF_NAME: &str = "orpheus-3b-0.1-ft.Q4_K_M.gguf";
const ORPHEUS_GGUF_BYTES: u64 = 2_086_595_904;  // ~2.09 GB sanity-check

// SNAC weights: we don't fetch raw `hubertsiuzdak/snac_24khz` directly —
// they're in PyTorch parametrised form and need our converter run on
// them first. Until we mirror a pre-converted safetensors at a stable
// URL, fail with an actionable error pointing at scripts/snac_export.py.
const SNAC_FILE_NAME: &str = "snac24k.safetensors";
const SNAC_MIRROR_HINT: &str = "\
  Run the converter once and place the output here:
    python3 crates/modgrad-codec/scripts/snac_export.py ~/.cache/sonotxt
  Or download a pre-converted copy from:
    https://huggingface.co/rotkonetworks/snac24k-modgrad   (TODO: publish)
";

// Voice list comes from Orpheus's training set + a "default" alias.
const KNOWN_VOICES: &[&str] = &["tara", "leah", "leo", "jess", "mia", "julia", "zac", "zoe"];

/// Vocab index of `<custom_token_0>` (first SNAC-token in the Orpheus vocab
/// extension). Empirically for DeviantYC/orpheus-3b-Q4_K_M-GGUF, `<custom_token_N>`
/// at vocab index 128256+N maps directly to SNAC code N at the corresponding
/// frame position — NO `-10` shift, despite canopyai/decoder.py applying one
/// for their 1B model. A/B test: shifting by -10 produced complete silence;
/// no shift produces audible speech-shaped audio.
const AUDIO_TOKEN_BASE: u32 = 128256;
const TOK_EOT_A: u32 = 128009;
const TOK_EOT_B: u32 = 128001;
/// Audio-end marker emitted by Orpheus when it finishes a generation.
/// From `canopyai/Orpheus-TTS::engine_class.py::generate_tokens_sync`
/// `stop_token_ids = [49158]`. Matches their vLLM sampling config.
const TOK_AUDIO_EOS: u32 = 49158;

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    text: String,
    voice: String,
    output: PathBuf,
    gen_tokens: usize,
    cache_dir: PathBuf,
    gguf_override: Option<PathBuf>,
    snac_override: Option<PathBuf>,
}

fn default_cache_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache/sonotxt")
    } else {
        PathBuf::from("/tmp/sonotxt-cache")
    }
}

fn print_help_and_exit() -> ! {
    eprintln!("\
sonotxt — rocm tts (Orpheus 3B + SNAC, modgrad runtime)

USAGE:
  sonotxt [OPTIONS] <text>

OPTIONS:
  -v, --voice <name>      voice id (default: tara). --list-voices for all
  -o, --out   <path>      output WAV path (default: out.wav)
      --gen   <n>         max audio tokens to generate (default: 252 ≈ 1.5s)
      --cache <dir>       model cache dir (default: ~/.cache/sonotxt)
      --gguf  <path>      override Orpheus GGUF location
      --snac  <path>      override SNAC safetensors location
      --list-voices       print known voices and exit
      --download          fetch any missing model files and exit
  -h, --help              this message

EXAMPLES:
  sonotxt \"Hello world\" -o hello.wav
  sonotxt -v leo --gen 504 \"A longer sentence\" -o longer.wav

First run downloads ~2 GB to the cache; subsequent runs are instant.
");
    std::process::exit(0);
}

fn take(argv: &mut Vec<String>, needle: &str) -> Option<String> {
    let pos = argv.iter().position(|a| a == needle)?;
    let v = argv.get(pos + 1).cloned();
    if v.is_some() { argv.drain(pos..=pos + 1); } else { argv.remove(pos); }
    v
}
fn flag(argv: &mut Vec<String>, needle: &str) -> bool {
    if let Some(pos) = argv.iter().position(|a| a == needle) {
        argv.remove(pos); true
    } else { false }
}

fn parse_args() -> Args {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();

    if flag(&mut argv, "-h") || flag(&mut argv, "--help") { print_help_and_exit(); }
    if flag(&mut argv, "--list-voices") {
        println!("known voices: {}", KNOWN_VOICES.join(", "));
        std::process::exit(0);
    }
    let download_only = flag(&mut argv, "--download");

    let voice  = take(&mut argv, "-v").or_else(|| take(&mut argv, "--voice"))
        .unwrap_or_else(|| "tara".into());
    let output = PathBuf::from(take(&mut argv, "-o").or_else(|| take(&mut argv, "--out"))
        .unwrap_or_else(|| "out.wav".into()));
    let gen_tokens: usize = take(&mut argv, "--gen").and_then(|s| s.parse().ok()).unwrap_or(1200);
    let cache_dir = take(&mut argv, "--cache").map(PathBuf::from).unwrap_or_else(default_cache_dir);
    let gguf_override = take(&mut argv, "--gguf").map(PathBuf::from);
    let snac_override = take(&mut argv, "--snac").map(PathBuf::from);

    let text = if download_only {
        String::new()
    } else if argv.is_empty() {
        eprintln!("error: missing <text>. Use --help for usage.");
        std::process::exit(2);
    } else {
        argv.join(" ")
    };

    if download_only {
        ensure_models(&cache_dir, gguf_override.as_deref(), snac_override.as_deref())
            .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1); });
        eprintln!("done — models cached at {}", cache_dir.display());
        std::process::exit(0);
    }

    Args { text, voice, output, gen_tokens, cache_dir, gguf_override, snac_override }
}

// ─── Model cache management ─────────────────────────────────────────────────

fn ensure_models(
    cache_dir: &std::path::Path,
    gguf_override: Option<&std::path::Path>,
    snac_override: Option<&std::path::Path>,
) -> Result<(PathBuf, PathBuf), String> {
    std::fs::create_dir_all(cache_dir).map_err(|e| format!("mkdir {}: {e}", cache_dir.display()))?;

    let gguf_path = if let Some(p) = gguf_override {
        p.to_path_buf()
    } else {
        let p = cache_dir.join(ORPHEUS_GGUF_NAME);
        if !p.exists() {
            download_file(ORPHEUS_GGUF_URL, &p, Some(ORPHEUS_GGUF_BYTES))?;
        }
        p
    };
    if !gguf_path.exists() {
        return Err(format!("missing GGUF: {}", gguf_path.display()));
    }

    let snac_path = if let Some(p) = snac_override {
        p.to_path_buf()
    } else {
        let p = cache_dir.join(SNAC_FILE_NAME);
        if !p.exists() {
            return Err(format!(
                "missing SNAC weights at {}.\n{}",
                p.display(), SNAC_MIRROR_HINT
            ));
        }
        p
    };
    if !snac_path.exists() {
        return Err(format!("missing SNAC weights: {}", snac_path.display()));
    }

    Ok((gguf_path, snac_path))
}

/// Plain curl-shells the download with `-C -` (resume on partial) and
/// `--retry`. Avoids adding a hard `reqwest`/`ureq` dep just for this.
fn download_file(url: &str, dest: &std::path::Path, expect_bytes: Option<u64>) -> Result<(), String> {
    eprintln!("downloading {} → {}", url, dest.display());
    if let Some(b) = expect_bytes {
        eprintln!("  expected size: {:.2} GB", b as f64 / 1e9);
    }
    let status = std::process::Command::new("curl")
        .args(["-L", "-C", "-", "--retry", "30", "--retry-delay", "5",
               "--retry-all-errors", "--fail", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .map_err(|e| format!("spawn curl: {e}"))?;
    if !status.success() {
        return Err(format!("curl exited {status}"));
    }
    if let Some(want) = expect_bytes {
        let got = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
        if got != want {
            return Err(format!("size mismatch: got {got}, want {want}"));
        }
    }
    Ok(())
}

// ─── Pipeline ───────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    if !KNOWN_VOICES.contains(&args.voice.as_str()) {
        eprintln!("warning: unknown voice '{}', proceeding anyway", args.voice);
    }

    let (gguf_path, snac_path) = ensure_models(
        &args.cache_dir,
        args.gguf_override.as_deref(),
        args.snac_override.as_deref(),
    ).unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1); });

    // ── 1. Load Orpheus ──────────────────────────────────────────
    eprintln!("[1/5] load {}", gguf_path.display());
    let gguf_bytes = std::fs::read(&gguf_path).expect("read gguf");
    let gguf = GgufFile::parse(&mut Cursor::new(&gguf_bytes)).expect("parse gguf");
    let mut cfg = LlmConfig::llama3();
    cfg.populate_from_gguf(&gguf).expect("populate cfg");
    // Orpheus 3B is based on Llama-3.2 which uses Llama-3.1 RoPE scaling.
    // Without this, RoPE phase is wrong past position ~8K (and produces
    // subtle drift even at low positions through frequency-band aliasing).
    cfg.rope_scaling = Some(modgrad_device::kfd::llm::RopeScaling {
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 8192,
    });

    let dev = HsaDevice::open().expect("open /dev/kfd (need ROCm + AMD GPU)");
    let mut queue = GpuQueue::new();

    // SONOTXT_SAFETENSORS=path1[,path2,...] uses lossless FP16 weights instead
    // of the quantized GGUF. Required for clean output — Q4_K_M drift
    // compounds enough over 28 layers to break frame-position tracking.
    let weights = if let Ok(st_paths) = std::env::var("SONOTXT_SAFETENSORS") {
        let paths: Vec<std::path::PathBuf> = st_paths.split(',').map(std::path::PathBuf::from).collect();
        eprintln!("[1b/5] loading safetensors from {} shard(s)...", paths.len());
        modgrad_device::kfd::safetensors_loader::load_from_safetensors(&paths)
            .expect("safetensors load")
    } else {
        LlmWeights::load_from_gguf(&gguf, &gguf_bytes, &dev, &mut queue)
            .expect("upload weights")
    };

    // ── 2. Build Orpheus prompt ──────────────────────────────────
    // Per canopyai/Orpheus-TTS::engine_class.py::_format_prompt — note that
    // HF tokenizer.encode() AUTO-PREPENDS BOS (128000), so the actual prompt
    // sent to vLLM is:
    //   [128259, 128000] + bpe(body) + [128009, 128260, 128261, 128257]
    //
    // We MUST include the BOS or the model is off-distribution and emits
    // truncated/garbled audio (heard "halo" instead of "hello world" without it).
    //
    // Token roles:
    //   128259 = <custom_token_3>  "start of human"
    //   128000 = <|begin_of_text|> BOS  (HF auto-prepend)
    //   128009 = <|eot_id|>        end of text
    //   128260 = <custom_token_4>  end of human
    //   128261 = <custom_token_5>  start of assistant
    //   128257 = <custom_token_1>  begin audio
    let tokenizer = LlamaBpeTokenizer::from_gguf(&gguf).expect("load BPE tokenizer");
    // SONOTXT_PROMPT overrides the body — useful for Maya1 which uses
    // `<description="...">` instead of `voice: text`.
    let body = std::env::var("SONOTXT_PROMPT").unwrap_or_else(|_|
        format!("{}: {}", args.voice, args.text));
    let body_ids = tokenizer.encode(&body);
    let mut prompt: Vec<u32> = Vec::with_capacity(body_ids.len() + 6);
    prompt.push(128259);
    prompt.push(128000);  // BOS — HF auto-prepended in canopy's pipeline
    prompt.extend_from_slice(&body_ids);
    prompt.extend_from_slice(&[128009, 128260, 128261, 128257]);
    eprintln!("  prompt body: {body:?}  →  {} BPE tokens  →  {} total", body_ids.len(), prompt.len());

    // DEBUG: env var MODGRAD_DUMP_LOGITS=/path writes the post-prompt logits
    // for layer-by-layer A/B against llama.cpp's reference.
    let dump_path = std::env::var("MODGRAD_DUMP_LOGITS").ok();

    let max_seq = prompt.len() + args.gen_tokens + 8;
    let mut cache = LlmCache::allocate(&cfg, &dev, &mut queue, max_seq).expect("alloc cache");
    let mut scratch = LlmScratch::allocate(&cfg, &dev, &mut queue).expect("alloc scratch");

    eprintln!("[2/5] prompt ({} tokens)", prompt.len());
    let t_prompt = Instant::now();
    let mut dev = dev;
    let mut logits = Vec::new();
    for &tok in &prompt {
        logits = forward_token(&cfg, &weights, &mut cache, &mut scratch,
                               &mut dev, &mut queue, tok);
    }
    eprintln!("  {:.1}s", t_prompt.elapsed().as_secs_f32());

    // DEBUG dump for A/B against llama.cpp reference. The IMPORTANT test:
    // force-feed the SAME pre-recorded token sequence into both engines and
    // diff the per-step logits. Only this isolates per-step bugs from
    // divergent autoregressive paths.
    //
    // Set MODGRAD_FORCE_FEED=token1,token2,token3 to override greedy.
    if let Some(path_prefix) = &dump_path {
        let n_steps: usize = std::env::var("MODGRAD_DUMP_STEPS").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(5);
        let force_feed: Option<Vec<u32>> = std::env::var("MODGRAD_FORCE_FEED").ok()
            .map(|s| s.split(',').filter_map(|t| t.parse().ok()).collect());

        let mut cur_logits = logits.clone();
        for step in 0..n_steps {
            let path = format!("{path_prefix}.step{step}.bin");
            let bytes: Vec<u8> = cur_logits.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            std::fs::write(&path, &bytes).expect("write logits dump");
            let min = cur_logits.iter().copied().fold(f32::INFINITY, f32::min);
            let max = cur_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = cur_logits.iter().sum::<f32>() / cur_logits.len() as f32;
            let var: f32 = cur_logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / cur_logits.len() as f32;
            let mut idx: Vec<usize> = (0..cur_logits.len()).collect();
            idx.sort_unstable_by(|&a, &b| cur_logits[b].partial_cmp(&cur_logits[a]).unwrap_or(std::cmp::Ordering::Equal));
            let top1 = idx[0] as u32;
            // Which token to feed for the NEXT step
            let next_tok = match &force_feed {
                Some(seq) if step < seq.len() => seq[step],
                _ => top1,
            };
            eprintln!("  STEP {step}: top1={top1} logit={:.3} | stats min={min:.3} max={max:.3} mean={mean:.4} std={:.3} | next_feed={next_tok}",
                cur_logits[idx[0]], var.sqrt());
            if step + 1 < n_steps {
                cur_logits = forward_token(&cfg, &weights, &mut cache, &mut scratch,
                                          &mut dev, &mut queue, next_tok);
            }
        }
        eprintln!("  done — dumped {n_steps} steps");
        std::process::exit(0);
    }

    // ── 3. Generate audio tokens ─────────────────────────────────
    //
    // Sampling matches canopyai/Orpheus-TTS engine_class.py:
    //   temperature=0.6, top_p=0.8, repetition_penalty=1.3, stop=[49158]
    // Argmax pathologically emitted ~85% out-of-range tokens; sampling +
    // repetition_penalty brings drop rate down to canopy's ~10-20% range.
    //
    // Filtering: canopyai/decoder.py::tokens_decoder — accept only tokens
    // that decode to a code in [0, 4095] for the current intra-frame
    // position. Drop the rest without advancing position.
    // Sampler overrides for debugging — useful for isolating drop-rate from
    // sampling vs LM-distribution issues.
    //   SONOTXT_TEMP, SONOTXT_TOP_P, SONOTXT_REP_PENALTY
    let temperature  = std::env::var("SONOTXT_TEMP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.6_f32);
    let top_p        = std::env::var("SONOTXT_TOP_P").ok().and_then(|s| s.parse().ok()).unwrap_or(0.8_f32);
    let rep_penalty  = std::env::var("SONOTXT_REP_PENALTY").ok().and_then(|s| s.parse().ok()).unwrap_or(1.3_f32);
    let mut rng_state: u64 = 0xdeadbeef_cafef00d;
    eprintln!("[3/5] generate up to {} tokens (temp={temperature} top_p={top_p} rep_penalty={rep_penalty}, canopy-filtered)",
        args.gen_tokens);
    let t_gen = Instant::now();
    let mut valid_codes: Vec<u32> = Vec::with_capacity(args.gen_tokens);
    let mut pos: usize = 0;
    let mut last_next: u32 = 0;
    let mut emitted = 0usize;
    let mut dropped = 0usize;
    // History for repetition penalty — every emitted token regardless of
    // whether it was a valid audio code.
    let mut history: Vec<u32> = Vec::with_capacity(args.gen_tokens);

    // Position-aware constrained sampling: at each step `pos`, only sample
    // from the audio-token bucket [AUDIO_BASE + 10 + (pos%7)*4096, AUDIO_BASE + 10 + (pos%7+1)*4096).
    // The +10 accounts for canopy's first 10 control tokens before audio range.
    //
    // Why: modgrad's LM is slightly numerically noisier than llama.cpp's
    // (Q-dequant + AVX-512 reductions accumulate ~5e-3 error per layer × 28
    // layers). Over autoregressive generation this compounds and confuses the
    // LM's internal frame-position tracking after ~4 tokens — drop rate ~90%
    // without this constraint, even at near-greedy sampling.
    //
    // The constraint preserves the autoregressive distribution within each
    // bucket: the LM still chooses among tokens it would naturally favor,
    // it just can't accidentally drift into the wrong position phase. This
    // is similar to canopy's filtering-after-the-fact but applied PRE-sample
    // so the LM stays in sync.
    //
    // Stop tokens (49158, 128001, 128009) remain sample-able regardless of pos.
    let constrained: bool = std::env::var("SONOTXT_UNCONSTRAINED").is_err();
    let stop_tokens: &[u32] = &[TOK_AUDIO_EOS, TOK_EOT_A, TOK_EOT_B];

    for step in 0..args.gen_tokens {
        // +10 to skip control tokens that prefix the actual audio range.
        let bucket_start = (AUDIO_TOKEN_BASE as usize) + 10 + (pos % 7) * 4096;
        let bucket_end   = bucket_start + 4096;
        let next = if constrained {
            sample_top_p_constrained(
                &logits, temperature, top_p, &history, rep_penalty, &mut rng_state,
                bucket_start, bucket_end, stop_tokens,
            ) as u32
        } else {
            sample_top_p(&logits, temperature, top_p, &history, rep_penalty, &mut rng_state) as u32
        };
        last_next = next;
        history.push(next);
        // Canopy stop tokens: 49158 (audio EOS) and the Llama-3 EOTs.
        if next == TOK_EOT_A || next == TOK_EOT_B || next == TOK_AUDIO_EOS {
            eprintln!("  stop token {next} at step {step}");
            break;
        }

        if next >= AUDIO_TOKEN_BASE {
            // Canopy reference formula: code = (vocab_id - AUDIO_TOKEN_BASE) - 10 - (pos%7)*4096
            // Verified against PyTorch reference pipeline 2026-06-18.
            // (Earlier "no shift" experiment was confused by a separate bug.)
            let raw = (next as i64) - (AUDIO_TOKEN_BASE as i64) - 10
                    - ((pos as i64 % 7) * 4096);
            if (0..4096).contains(&raw) {
                valid_codes.push(raw as u32);
                pos += 1;
                emitted += 1;
            } else {
                dropped += 1;
            }
        } else {
            dropped += 1;
        }

        if step + 1 < args.gen_tokens {
            logits = forward_token(&cfg, &weights, &mut cache, &mut scratch,
                                   &mut dev, &mut queue, next);
        }
    }
    eprintln!("  emitted {} valid codes ({} dropped, {} EOT={}) in {:.1}s",
        emitted, dropped,
        if last_next == TOK_EOT_A || last_next == TOK_EOT_B { 1 } else { 0 },
        last_next, t_gen.elapsed().as_secs_f32());

    print_and_reset_profile();
    let frame_count = valid_codes.len() / 7;
    valid_codes.truncate(frame_count * 7);
    eprintln!("  raw codes (first 21): {:?}", &valid_codes[..valid_codes.len().min(21)]);
    eprintln!("  code stats: min={} max={} mean={}",
        valid_codes.iter().copied().min().unwrap_or(0),
        valid_codes.iter().copied().max().unwrap_or(0),
        valid_codes.iter().copied().sum::<u32>() / (valid_codes.len() as u32).max(1));
    if frame_count == 0 {
        eprintln!("error: no complete 7-token frames after filtering — try --gen with a larger value");
        std::process::exit(3);
    }
    eprintln!("  {frame_count} full frames");

    // ── 4. Codes → 3 SNAC codebooks (canopyai recipe) ────────────
    let (codes_0, codes_1, codes_2) = valid_codes_to_codebooks(&valid_codes);

    // ── 5. SNAC decode + WAV write ──────────────────────────────
    eprintln!("[4/5] SNAC decode ({} frames)", frame_count);
    let t_snac = Instant::now();
    let snac = SnacDecoder24k::load_from_safetensors(&snac_path).expect("load snac");
    let samples = snac.decode_seeded(&[codes_0, codes_1, codes_2], 1);
    eprintln!("  {} samples ({:.2}s @ 24kHz) in {:.2}s",
        samples.len(), samples.len() as f32 / 24000.0, t_snac.elapsed().as_secs_f32());

    eprintln!("[5/5] write {}", args.output.display());
    write_wav_24khz_mono(&args.output, &samples)
        .unwrap_or_else(|e| { eprintln!("error writing WAV: {e}"); std::process::exit(1); });

    eprintln!("done — {}", args.output.display());
}

#[allow(dead_code)]
fn argmax(xs: &[f32]) -> usize {
    let mut best = 0; let mut bv = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > bv { best = i; bv = v; }
    }
    best
}

/// splitmix64 → uniform f32 in [0, 1). Deterministic for a given seed.
fn rng_next_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);
    // 24 bits of mantissa for f32
    ((z >> 40) as f32) / (1u64 << 24) as f32
}

/// Temperature + nucleus (top-p) sampling, with optional HF/vLLM-style
/// repetition_penalty (canopy uses 1.3). Matches Orpheus's training-time
/// `SamplingParams(temperature=0.6, top_p=0.8, repetition_penalty=1.3)`.
///
/// Argmax pushes the model far off its training distribution and ~85% of
/// emitted tokens fall outside the per-position vocab range — sampling
/// brings drop rate to roughly canopy's level.
/// Like `sample_top_p` but masks out all tokens OUTSIDE the audio bucket
/// `[bucket_start, bucket_end)` (plus allow `stop_tokens` through). Forces the
/// LM to stay in position phase during autoregressive generation.
fn sample_top_p_constrained(
    logits: &[f32], temperature: f32, top_p: f32,
    history: &[u32], rep_penalty: f32,
    rng_state: &mut u64,
    bucket_start: usize, bucket_end: usize,
    stop_tokens: &[u32],
) -> usize {
    let inv_t = 1.0 / temperature.max(1e-6);
    let mut work: Vec<f32> = logits.iter().enumerate()
        .map(|(i, &l)| {
            let in_bucket = i >= bucket_start && i < bucket_end;
            let is_stop = stop_tokens.contains(&(i as u32));
            if in_bucket || is_stop { l * inv_t } else { f32::NEG_INFINITY }
        })
        .collect();

    // Repetition penalty (only meaningful within the live buckets).
    if rep_penalty > 1.0 {
        for &tok in history {
            let i = tok as usize;
            if i < work.len() && work[i].is_finite() {
                let v = work[i];
                work[i] = if v > 0.0 { v / rep_penalty } else { v * rep_penalty };
            }
        }
    }

    let max = work.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // No survivors — fall back to argmax of raw logits in the bucket
        return (bucket_start..bucket_end).max_by(|&a, &b|
            logits[a].partial_cmp(&logits[b]).unwrap_or(std::cmp::Ordering::Equal)
        ).unwrap_or(bucket_start);
    }
    let mut probs: Vec<f32> = work.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return bucket_start;
    }
    for p in probs.iter_mut() { *p /= sum; }

    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b|
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    );
    let mut cum = 0.0_f32;
    let mut cutoff = idx.len();
    for (i, &j) in idx.iter().enumerate() {
        cum += probs[j];
        if cum >= top_p { cutoff = i + 1; break; }
    }
    let mass: f32 = idx[..cutoff].iter().map(|&j| probs[j]).sum();
    let r = rng_next_f32(rng_state) * mass;
    let mut acc = 0.0_f32;
    for &j in &idx[..cutoff] {
        acc += probs[j];
        if r <= acc { return j; }
    }
    idx[0]
}

fn sample_top_p(
    logits: &[f32], temperature: f32, top_p: f32,
    history: &[u32], rep_penalty: f32,
    rng_state: &mut u64,
) -> usize {
    // Vocab is 156,940 — full softmax + sort was ~10ms/token at f32.
    // Top-K shortcut: a top-p set with p=0.8 NEVER spans more than a few
    // hundred tokens for a coherent LM. Find the top K via partial-sort
    // (O(n) select_nth_unstable), softmax JUST those K, then top-p over them.
    const TOP_K: usize = 512;
    let inv_t = 1.0 / temperature.max(1e-6);

    // Build `work[i] = logit_i / T`, then patch in rep_penalty for seen tokens.
    let mut work: Vec<f32> = logits.iter().map(|&l| l * inv_t).collect();
    if rep_penalty > 1.0 {
        let rp = rep_penalty;
        for &tok in history {
            let i = tok as usize;
            if i < work.len() {
                let v = work[i];
                work[i] = if v > 0.0 { v / rp } else { v * rp };
            }
        }
    }

    // Top-K via partial sort. O(n) instead of O(n log n) full sort.
    let n = work.len();
    let k = TOP_K.min(n);
    let mut idx: Vec<usize> = (0..n).collect();
    let kth = k.saturating_sub(1).min(n - 1);
    idx.select_nth_unstable_by(kth, |&a, &b| {
        // Descending by logit.
        work[b].partial_cmp(&work[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    // Now idx[..k] holds the top K logits (in arbitrary order).
    let mut top: Vec<(usize, f32)> = idx[..k].iter().map(|&i| (i, work[i])).collect();
    // Sort the top K descending — only K=512 elements, ~5µs.
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Numerically stable softmax over the K survivors only.
    let max = top[0].1;
    let mut probs: Vec<(usize, f32)> = top.iter().map(|&(i, l)| (i, (l - max).exp())).collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return probs[0].0;
    }
    for (_, p) in probs.iter_mut() { *p /= sum; }

    // Top-p prefix — small list, just iterate.
    let mut cum = 0.0_f32;
    let mut cutoff = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cum += p;
        if cum >= top_p { cutoff = i + 1; break; }
    }
    let kept = &probs[..cutoff];
    let mass: f32 = kept.iter().map(|(_, p)| p).sum();
    let r = rng_next_f32(rng_state) * mass;
    let mut acc = 0.0_f32;
    for &(i, p) in kept {
        acc += p;
        if r <= acc { return i; }
    }
    kept.last().map(|(i, _)| *i).unwrap_or(0)
}

/// `valid_codes` is already validated [0, 4095] per intra-frame position
/// (the parser above guarantees it). This just reshapes the flat sequence
/// into SNAC's three codebooks per canopyai's recipe.
fn valid_codes_to_codebooks(valid_codes: &[u32]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let frames = valid_codes.len() / 7;
    let mut c0 = Vec::with_capacity(frames);
    let mut c1 = Vec::with_capacity(frames * 2);
    let mut c2 = Vec::with_capacity(frames * 4);
    for f in 0..frames {
        let b = f * 7;
        c0.push(valid_codes[b]);
        c1.push(valid_codes[b + 1]);
        c2.push(valid_codes[b + 2]);
        c2.push(valid_codes[b + 3]);
        c1.push(valid_codes[b + 4]);
        c2.push(valid_codes[b + 5]);
        c2.push(valid_codes[b + 6]);
    }
    (c0, c1, c2)
}

fn write_wav_24khz_mono(path: &std::path::Path, samples: &[f32]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let data_bytes = (samples.len() * 2) as u32;
    let total_minus_8 = 36 + data_bytes;

    f.write_all(b"RIFF")?;
    f.write_all(&total_minus_8.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&24000u32.to_le_bytes())?;
    f.write_all(&48000u32.to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_bytes.to_le_bytes())?;
    for &s in samples {
        let s16 = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&s16.to_le_bytes())?;
    }
    Ok(())
}
