//! gemma_serve — local Gemma-4-12B inference server on ROCm.
//!
//!   cargo run --release -p modgrad-gemma --features rocm --example gemma_serve
//!
//! Serves three protocols on one port (see `modgrad-serve`):
//!   GET  /health                       -> "ok"
//!   POST /generate {prompt,max_tokens} -> {text,thinking,...}  (SSE if stream:true)
//!   POST /v1/messages                  -> Anthropic Messages API (run Claude Code
//!                                          ON the local model via ANTHROPIC_BASE_URL)
//!   POST /mcp                          -> MCP tools: gemma_generate/status/load/unload
//!
//! The model loads/unloads on demand (gemma_unload frees ~6.5 GiB so you can game;
//! the next request reloads it). Env: GEMMA_GGUF, GEMMA_TOKENIZER, GEMMA_ADDR
//! (default 127.0.0.1:8080), GEMMA_MAX_SEQ (512), GEMMA_LAZY=1 to start unloaded.

#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn main() {
    use modgrad_gemma::serve_model::GemmaLm;
    use modgrad_serve::{serve, ServeConfig, LanguageModel};

    let gguf = std::env::var("GEMMA_GGUF").unwrap_or_else(|_|
        "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf".into());
    let tok = std::env::var("GEMMA_TOKENIZER").unwrap_or_else(|_|
        "/steam/rotko/models/gemma-4-12b/tokenizer.json".into());
    let addr = std::env::var("GEMMA_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let max_seq: usize = std::env::var("GEMMA_MAX_SEQ").ok().and_then(|s| s.parse().ok()).unwrap_or(512);

    eprintln!("[gemma_serve] tokenizer {tok}");
    eprintln!("[gemma_serve] reading gguf {gguf}");
    let mut model = GemmaLm::open(&gguf, &tok, max_seq).expect("open gemma");

    if std::env::var("GEMMA_LAZY").is_err() {
        model.ensure_loaded().expect("load model");
    } else {
        eprintln!("[gemma_serve] lazy mode — VRAM free until first request");
    }

    let cfg = ServeConfig { addr, tool_prefix: "gemma".into(), server_name: "gemma-local".into() };
    serve(model, cfg).expect("serve");
}

#[cfg(not(all(feature = "rocm", modgrad_hipcc_kernels)))]
fn main() { eprintln!("build with --features rocm"); }
