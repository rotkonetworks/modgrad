//! gemma_serve — persistent local inference server for Gemma-4-12B on ROCm,
//! with a built-in MCP endpoint so Claude Code can call it as a native tool.
//!
//!   cargo run --release -p modgrad-device --features rocm --example gemma_serve
//!
//! HTTP (REST):
//!   GET  /health                      -> "ok"
//!   POST /generate {prompt,max_tokens} -> {text,thinking,tokens,tok_per_s}
//! MCP (JSON-RPC 2.0 over HTTP, for `claude mcp add --transport http`):
//!   POST /mcp   tools: gemma_generate, gemma_status, gemma_load, gemma_unload
//!
//! The model is loadable/unloadable on demand: `gemma_unload` frees the ~6.5 GiB
//! of VRAM (so you can game) WITHOUT stopping the server; the next generate (or
//! `gemma_load`) reloads it (~1-3 s). Env: GEMMA_GGUF, GEMMA_TOKENIZER,
//! GEMMA_ADDR (default 127.0.0.1:8080), GEMMA_MAX_SEQ (512), GEMMA_LAZY=1 to
//! start WITHOUT loading the model (VRAM free until first use).

#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn main() {
    use std::io::{Read, Write, BufRead, BufReader};
    use std::io::Cursor;
    use std::net::{TcpListener, TcpStream};
    use std::time::Instant;
    use modgrad_device::kfd::gguf::GgufFile;
    use modgrad_device::rocm_gemma::RocmGemma;
    use tokenizers::Tokenizer;

    let gguf_path = std::env::var("GEMMA_GGUF").unwrap_or_else(|_|
        "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf".into());
    let tok_path = std::env::var("GEMMA_TOKENIZER").unwrap_or_else(|_|
        "/steam/rotko/models/gemma-4-12b/tokenizer.json".into());
    let addr = std::env::var("GEMMA_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let max_seq: usize = std::env::var("GEMMA_MAX_SEQ").ok().and_then(|s| s.parse().ok()).unwrap_or(512);

    eprintln!("[gemma_serve] tokenizer {tok_path}");
    let tok = Tokenizer::from_file(&tok_path).expect("load tokenizer.json");
    eprintln!("[gemma_serve] reading gguf {gguf_path}");
    let file = std::fs::read(&gguf_path).expect("read gguf");
    let g = GgufFile::parse(&mut Cursor::new(&file)).expect("parse gguf");
    let bos = 2u32;
    let stop = [1u32, 106u32]; // <eos>, <turn|>

    // Lazy-loadable model: None = unloaded (no VRAM). load() reads from the
    // in-memory gguf, so unload/reload is cheap and does not re-read the disk.
    let mut model: Option<RocmGemma> = None;
    if std::env::var("GEMMA_LAZY").is_err() {
        eprintln!("[gemma_serve] loading model into VRAM ...");
        model = Some(RocmGemma::load(&g, &file, max_seq).expect("load model"));
    } else {
        eprintln!("[gemma_serve] lazy mode — model NOT loaded (VRAM free until first request)");
    }

    let listener = TcpListener::bind(&addr).expect("bind");
    eprintln!("[gemma_serve] READY — http://{addr}  (REST /generate, MCP /mcp, GET /health)");

    for stream in listener.incoming() {
        let mut stream = match stream { Ok(s) => s, Err(_) => continue };
        let (method, path, body) = match read_request(&mut stream) { Some(r) => r, None => continue };
        // Strip any query string (Claude Code calls `/v1/messages?beta=true`).
        let path = path.split('?').next().unwrap_or("").to_string();
        eprintln!("[req] {method} {path} ({}B)", body.len());

        if method == "GET" && path == "/health" {
            respond(&mut stream, "200 OK", "text/plain", "ok");
        } else if method == "POST" && path == "/mcp" {
            let (code, out) = handle_mcp(&body, &mut model, &g, &file, max_seq, &tok, bos, &stop);
            if code == "202" {
                let _ = write!(stream, "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            } else {
                respond(&mut stream, "200 OK", "application/json", &out);
            }
        } else if method == "POST" && path == "/generate" {
            let v: serde_json::Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(e) => { respond(&mut stream, "400 Bad Request", "text/plain", &format!("bad json: {e}")); continue; }
            };
            let prompt = v.get("prompt").and_then(|p| p.as_str()).unwrap_or("");
            let max_tokens = v.get("max_tokens").and_then(|m| m.as_u64()).unwrap_or(256) as usize;
            let wrap = v.get("raw").and_then(|r| r.as_bool()) != Some(true); // raw=true => no chat wrap
            if v.get("stream").and_then(|s| s.as_bool()) == Some(true) {
                // SSE streaming. Pre-header errors (load/encode) -> plain 500;
                // mid-stream errors are emitted as SSE inside stream_generate.
                if let Err(e) = stream_generate(&mut stream, &mut model, &g, &file, max_seq, &tok, bos, &stop, prompt, max_tokens, wrap) {
                    respond(&mut stream, "500 Internal Server Error", "text/plain", &e);
                }
                continue;
            }
            match gen_answer(&mut model, &g, &file, max_seq, &tok, bos, &stop, prompt, max_tokens, wrap) {
                Ok(r) => {
                    let resp = serde_json::json!({
                        "text": r.answer, "thinking": r.thinking, "tokens": r.tokens,
                        "prompt_tokens": r.prompt_tokens, "tok_per_s": r.tok_per_s });
                    respond(&mut stream, "200 OK", "application/json", &resp.to_string());
                }
                Err(e) => respond(&mut stream, "500 Internal Server Error", "text/plain", &e),
            }
        } else if method == "POST" && path == "/v1/messages" {
            // Anthropic Messages API — lets Claude Code run ON the local model
            // (set ANTHROPIC_BASE_URL to this server). handle_messages writes the
            // full HTTP/SSE response; errors before any write -> plain 500.
            if let Err(e) = handle_messages(&mut stream, &body, &mut model, &g, &file, max_seq, &tok, bos, &stop) {
                respond(&mut stream, "500 Internal Server Error", "text/plain", &e);
            }
        } else {
            respond(&mut stream, "404 Not Found", "text/plain", "POST /generate | POST /v1/messages | POST /mcp | GET /health");
        }
    }

    // ── helpers (closures capturing nothing; defined as inner fns below) ──
    fn read_request(stream: &mut TcpStream) -> Option<(String, String, Vec<u8>)> {
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 { return None; }
        let mut parts = line.trim_end().splitn(3, ' ');
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();
        let mut content_len = 0usize;
        loop {
            let mut h = String::new();
            if reader.read_line(&mut h).unwrap_or(0) == 0 { break; }
            let h = h.trim_end();
            if h.is_empty() { break; }
            if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                content_len = v.trim().parse().unwrap_or(0);
            }
        }
        let mut body = vec![0u8; content_len];
        if content_len > 0 && reader.read_exact(&mut body).is_err() { return None; }
        Some((method, path, body))
    }

    fn respond(stream: &mut TcpStream, code: &str, ctype: &str, body: &str) {
        let _ = write!(stream,
            "HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len());
    }
}

#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
struct GenResult { thinking: String, answer: String, tokens: usize, prompt_tokens: usize, tok_per_s: f64 }

/// Ensure the model is resident in VRAM (lazy load), returning a mutable handle.
#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn ensure_loaded<'a>(model: &'a mut Option<modgrad_device::rocm_gemma::RocmGemma>,
    g: &modgrad_device::kfd::gguf::GgufFile, file: &[u8], max_seq: usize)
    -> Result<&'a mut modgrad_device::rocm_gemma::RocmGemma, String> {
    if model.is_none() {
        eprintln!("[gemma_serve] (re)loading model into VRAM ...");
        *model = Some(modgrad_device::rocm_gemma::RocmGemma::load(g, file, max_seq)?);
    }
    Ok(model.as_mut().unwrap())
}

/// Byte length of the longest char-aligned common prefix of a and b.
#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut n = 0;
    for (ai, bc) in a.char_indices().zip(b.chars()) {
        if ai.1 == bc { n = ai.0 + ai.1.len_utf8(); } else { break; }
    }
    n
}

/// Stream a completion as SSE: writes `data: {"token":"..."}` per token, then
/// `data: {"done":true,...}`. Decodes the full sequence each step and emits the
/// new suffix (handles multi-token UTF-8 / BPE re-segmentation).
#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
#[allow(clippy::too_many_arguments)]
fn stream_generate(out: &mut std::net::TcpStream,
    model: &mut Option<modgrad_device::rocm_gemma::RocmGemma>,
    g: &modgrad_device::kfd::gguf::GgufFile, file: &[u8], max_seq: usize,
    tok: &tokenizers::Tokenizer, bos: u32, stop: &[u32],
    prompt: &str, max_tokens: usize, wrap_chat: bool) -> Result<(), String> {
    use std::io::Write;
    use std::time::Instant;
    if prompt.is_empty() { return Err("missing prompt".into()); }
    let m = ensure_loaded(model, g, file, max_seq)?;
    let formatted = if wrap_chat { format!("<|turn>user\n{prompt}<turn|>\n<|turn>model\n") }
                    else { prompt.to_string() };
    let enc = tok.encode(formatted, false).map_err(|e| format!("encode: {e}"))?;
    let mut ids = vec![bos];
    ids.extend_from_slice(enc.get_ids());
    if ids.len() >= max_seq { return Err("prompt longer than GEMMA_MAX_SEQ".into()); }
    m.reset();
    write!(out, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")
        .map_err(|e| e.to_string())?;
    out.flush().ok();
    let t = Instant::now();
    let mut acc: Vec<u32> = Vec::new();
    let mut last = String::new();
    let res = m.generate_stream(&ids, max_tokens, stop, |id| {
        acc.push(id);
        let text = tok.decode(&acc, true).unwrap_or_default();
        let cp = common_prefix_len(&last, &text);
        if text.len() > cp {
            let _ = write!(out, "data: {}\n\n", serde_json::json!({"token": &text[cp..]}));
            let _ = out.flush();
        }
        last = text;
        true
    });
    let secs = t.elapsed().as_secs_f64();
    if let Err(e) = res {
        let _ = write!(out, "data: {}\n\n", serde_json::json!({"error": e}));
    }
    let _ = write!(out, "data: {}\n\n", serde_json::json!({"done": true, "tokens": acc.len(),
        "tok_per_s": ((acc.len() as f64 / secs.max(1e-6)) * 100.0).round() / 100.0}));
    let _ = out.flush();
    Ok(()) // headers already sent; never send a second HTTP response
}

/// Ensure the model is loaded (lazy), then tokenize + generate + split channels.
#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
#[allow(clippy::too_many_arguments)]
fn gen_answer(model: &mut Option<modgrad_device::rocm_gemma::RocmGemma>,
              g: &modgrad_device::kfd::gguf::GgufFile, file: &[u8], max_seq: usize,
              tok: &tokenizers::Tokenizer, bos: u32, stop: &[u32],
              prompt: &str, max_tokens: usize, wrap_chat: bool) -> Result<GenResult, String> {
    use modgrad_device::rocm_gemma::RocmGemma;
    use std::time::Instant;
    if prompt.is_empty() { return Err("missing prompt".into()); }
    let _ = RocmGemma::load; // keep import referenced
    let m = ensure_loaded(model, g, file, max_seq)?;
    let formatted = if wrap_chat {
        format!("<|turn>user\n{prompt}<turn|>\n<|turn>model\n")
    } else { prompt.to_string() };
    let enc = tok.encode(formatted, false).map_err(|e| format!("encode: {e}"))?;
    let mut ids = vec![bos];
    ids.extend_from_slice(enc.get_ids());
    if ids.len() >= max_seq { return Err("prompt longer than GEMMA_MAX_SEQ".into()); }
    m.reset();
    let t = Instant::now();
    let out = m.generate(&ids, max_tokens, stop)?;
    let secs = t.elapsed().as_secs_f64();
    let raw = tok.decode(&out, false).unwrap_or_default();
    let (thinking, answer) = split_harmony(&raw);
    eprintln!("[gemma_serve] gen {} prompt_tok -> {} tok in {:.2}s ({:.1} tok/s)",
        ids.len(), out.len(), secs, out.len() as f64 / secs.max(1e-6));
    Ok(GenResult { thinking, answer, tokens: out.len(), prompt_tokens: ids.len(),
        tok_per_s: ((out.len() as f64 / secs.max(1e-6)) * 100.0).round() / 100.0 })
}

/// Handle one MCP JSON-RPC request. Returns ("200"|"202", json-body).
#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
#[allow(clippy::too_many_arguments)]
fn handle_mcp(body: &[u8], model: &mut Option<modgrad_device::rocm_gemma::RocmGemma>,
              g: &modgrad_device::kfd::gguf::GgufFile, file: &[u8], max_seq: usize,
              tok: &tokenizers::Tokenizer, bos: u32, stop: &[u32]) -> (&'static str, String) {
    use serde_json::{json, Value};
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return ("200", json!({"jsonrpc":"2.0","id":null,
            "error":{"code":-32700,"message":format!("parse error: {e}")}}).to_string()),
    };
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

    // Notifications carry no id and expect no response body.
    if method.starts_with("notifications/") { return ("202", String::new()); }

    let ok = |v: Value| ("200", json!({"jsonrpc":"2.0","id":id.clone(),"result":v}).to_string());
    let tool_text = |s: String, is_err: bool| json!({"content":[{"type":"text","text":s}],"isError":is_err});

    match method {
        "initialize" => {
            let pv = req.get("params").and_then(|p| p.get("protocolVersion"))
                .and_then(|v| v.as_str()).unwrap_or("2024-11-05").to_string();
            ok(json!({"protocolVersion": pv, "capabilities": {"tools": {}},
                      "serverInfo": {"name": "gemma-local", "version": "0.1.0"}}))
        }
        "ping" => ok(json!({})),
        "tools/list" => ok(json!({"tools": [
            {"name": "gemma_generate",
             "description": "Generate text with the local Gemma-4-12B model (runs on this machine's GPU). Pass a user question/instruction; returns the model's answer.",
             "inputSchema": {"type":"object","properties":{
                "prompt":{"type":"string","description":"The user message / question / instruction"},
                "max_tokens":{"type":"integer","description":"Max tokens to generate (default 256)"}},
                "required":["prompt"]}},
            {"name": "gemma_status",
             "description": "Report local model status: whether it is loaded into VRAM, context length.",
             "inputSchema": {"type":"object","properties":{}}},
            {"name": "gemma_load",
             "description": "Load the model into GPU VRAM (~6.5 GiB). Call before heavy use.",
             "inputSchema": {"type":"object","properties":{}}},
            {"name": "gemma_unload",
             "description": "Unload the model and FREE VRAM (e.g. to play a game). The server keeps running; the next gemma_generate reloads it.",
             "inputSchema": {"type":"object","properties":{}}}
        ]})),
        "tools/call" => {
            let params = req.get("params");
            let name = params.and_then(|p| p.get("name")).and_then(|n| n.as_str()).unwrap_or("");
            let args = params.and_then(|p| p.get("arguments")).cloned().unwrap_or(json!({}));
            match name {
                "gemma_generate" => {
                    let prompt = args.get("prompt").and_then(|p| p.as_str()).unwrap_or("");
                    let max_tokens = args.get("max_tokens").and_then(|m| m.as_u64()).unwrap_or(256) as usize;
                    match gen_answer(model, g, file, max_seq, tok, bos, stop, prompt, max_tokens, true) {
                        Ok(r) => {
                            let text = if r.thinking.is_empty() { r.answer }
                                       else { format!("{}\n\n[reasoning: {}]", r.answer, r.thinking) };
                            ok(tool_text(text, false))
                        }
                        Err(e) => ok(tool_text(format!("error: {e}"), true)),
                    }
                }
                "gemma_load" => {
                    if model.is_none() {
                        match modgrad_device::rocm_gemma::RocmGemma::load(g, file, max_seq) {
                            Ok(m) => { *model = Some(m); ok(tool_text("model loaded into VRAM".into(), false)) }
                            Err(e) => ok(tool_text(format!("load failed: {e}"), true)),
                        }
                    } else { ok(tool_text("already loaded".into(), false)) }
                }
                "gemma_unload" => {
                    let was = model.is_some();
                    *model = None; // Drop frees all HipBuffers => VRAM released
                    ok(tool_text(if was { "model unloaded — VRAM freed".into() }
                                 else { "already unloaded".into() }, false))
                }
                "gemma_status" => {
                    let loaded = model.is_some();
                    ok(tool_text(format!("gemma-4-12b (Q4_K_S) | loaded={} | ctx={} | endpoint=this server",
                        loaded, max_seq), false))
                }
                other => ok(tool_text(format!("unknown tool: {other}"), true)),
            }
        }
        other => ("200", json!({"jsonrpc":"2.0","id":id,
            "error":{"code":-32601,"message":format!("method not found: {other}")}}).to_string()),
    }
}

/// Flatten an Anthropic content value (string, or array of blocks) to plain text.
#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn extract_text(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter_map(|b| {
            match b.get("type").and_then(|t| t.as_str()) {
                Some("text") => b.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()),
                Some("tool_result") => b.get("content").map(extract_text),
                _ => None, // tool_use / images / etc. ignored in the text-only shim
            }
        }).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

/// Anthropic Messages API shim (`POST /v1/messages`). Lets Claude Code run ON the
/// local model: set ANTHROPIC_BASE_URL to this server. Streams tokens in Anthropic
/// SSE event format. TEXT-ONLY (no tool_use translation yet) — so the model can
/// chat in the cc UI but cannot drive tools.
#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
#[allow(clippy::too_many_arguments)]
fn handle_messages(out: &mut std::net::TcpStream, body: &[u8],
    model: &mut Option<modgrad_device::rocm_gemma::RocmGemma>,
    g: &modgrad_device::kfd::gguf::GgufFile, file: &[u8], max_seq: usize,
    tok: &tokenizers::Tokenizer, bos: u32, stop: &[u32]) -> Result<(), String> {
    use std::io::Write;
    use serde_json::{json, Value};
    let req: Value = serde_json::from_slice(body).map_err(|e| format!("bad json: {e}"))?;
    let want_stream = req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let max_tokens = req.get("max_tokens").and_then(|m| m.as_u64()).unwrap_or(512).min(4096) as usize;
    let model_name = req.get("model").and_then(|m| m.as_str()).unwrap_or("gemma-4-12b-local").to_string();

    // Anthropic system+messages -> Gemma harmony prompt.
    let mut prompt = String::new();
    if let Some(sys) = req.get("system") {
        let s = extract_text(sys);
        if !s.is_empty() { prompt.push_str(&format!("<|turn>system\n{s}<turn|>\n")); }
    }
    for msg in req.get("messages").and_then(|m| m.as_array()).into_iter().flatten() {
        let role = match msg.get("role").and_then(|r| r.as_str()) { Some("assistant") => "model", _ => "user" };
        let content = extract_text(msg.get("content").unwrap_or(&Value::Null));
        prompt.push_str(&format!("<|turn>{role}\n{content}<turn|>\n"));
    }
    prompt.push_str("<|turn>model\n");

    let m = ensure_loaded(model, g, file, max_seq)?;
    let enc = tok.encode(prompt, false).map_err(|e| format!("encode: {e}"))?;
    let mut ids = vec![bos];
    ids.extend_from_slice(enc.get_ids());
    // Truncate from the FRONT (keep recent context) to fit max_seq — Claude Code
    // prompts are large; this keeps it functional (degraded) instead of erroring.
    let budget = max_seq.saturating_sub(max_tokens).saturating_sub(8);
    if ids.len() > budget && budget > 1 {
        let cut = ids.len() - budget;
        ids = std::iter::once(bos).chain(ids[cut + 1..].iter().copied()).collect();
    }
    let input_tokens = ids.len();
    m.reset();

    if !want_stream {
        let out_ids = m.generate(&ids, max_tokens, stop)?;
        let raw = tok.decode(&out_ids, false).unwrap_or_default();
        let (_think, answer) = split_harmony(&raw);
        let resp = json!({"id":"msg_local","type":"message","role":"assistant",
            "model": model_name, "content":[{"type":"text","text":answer}],
            "stop_reason":"end_turn","stop_sequence":null,
            "usage":{"input_tokens":input_tokens,"output_tokens":out_ids.len()}}).to_string();
        write!(out, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp}", resp.len())
            .map_err(|e| e.to_string())?;
        return Ok(());
    }

    // ── SSE streaming in Anthropic event format ──
    write!(out, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")
        .map_err(|e| e.to_string())?;
    let ev = |out: &mut std::net::TcpStream, name: &str, data: Value| {
        let _ = write!(out, "event: {name}\ndata: {data}\n\n");
        let _ = out.flush();
    };
    ev(out, "message_start", json!({"type":"message_start","message":{"id":"msg_local","type":"message",
        "role":"assistant","model":model_name,"content":[],"stop_reason":null,"stop_sequence":null,
        "usage":{"input_tokens":input_tokens,"output_tokens":0}}}));
    ev(out, "content_block_start", json!({"type":"content_block_start","index":0,
        "content_block":{"type":"text","text":""}}));
    let mut acc: Vec<u32> = Vec::new();
    let mut last = String::new();
    let _ = m.generate_stream(&ids, max_tokens, stop, |id| {
        acc.push(id);
        // Decode WITH specials so we can find the harmony channel boundary, then
        // stream only the ANSWER (text after the first <channel|>), suppressing the
        // "thought" channel header so clients see a clean reply, not "thought\n...".
        let raw = tok.decode(&acc, false).unwrap_or_default();
        let answer = if let Some(p) = raw.find("<channel|>") {
            strip_markers(&raw[p + "<channel|>".len()..])
        } else if raw.contains("<|channel>") {
            String::new() // still inside the channel header — emit nothing yet
        } else {
            strip_markers(&raw) // no channel markers — direct answer
        };
        let cp = common_prefix_len(&last, &answer);
        if answer.len() > cp {
            let _ = write!(out, "event: content_block_delta\ndata: {}\n\n",
                json!({"type":"content_block_delta","index":0,
                       "delta":{"type":"text_delta","text":&answer[cp..]}}));
            let _ = out.flush();
        }
        last = answer;
        true
    });
    ev(out, "content_block_stop", json!({"type":"content_block_stop","index":0}));
    ev(out, "message_delta", json!({"type":"message_delta",
        "delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":acc.len()}}));
    ev(out, "message_stop", json!({"type":"message_stop"}));
    Ok(())
}

/// Strip harmony/turn special-token markers from a string (for streaming, where
/// we can't run the full split but still want clean text).
#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn strip_markers(s: &str) -> String {
    let mut o = s.to_string();
    for m in ["<turn|>", "<|turn>", "<bos>", "<eos>", "<|channel>", "<channel|>", "<pad>", "<|think|>"] {
        o = o.replace(m, "");
    }
    o.trim_start().to_string()
}

/// Split a harmony-channel completion into (thinking, answer). Channels look like
/// `<|channel>NAME<channel|>CONTENT`. The "thought" channel is reasoning; the
/// "final" (or any non-thought) channel is the answer. No channels => all answer.
#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn split_harmony(raw: &str) -> (String, String) {
    fn clean(s: &str) -> String {
        let mut o = s.to_string();
        for m in ["<turn|>", "<|turn>", "<bos>", "<eos>", "<|channel>", "<channel|>", "<pad>"] {
            o = o.replace(m, "");
        }
        o.trim().to_string()
    }
    if !raw.contains("<|channel>") {
        return (String::new(), clean(raw));
    }
    let mut thinking = String::new();
    let mut answer = String::new();
    for seg in raw.split("<|channel>").skip(1) {
        let (name, content) = seg.split_once("<channel|>").unwrap_or(("", seg));
        let content = clean(content);
        if content.is_empty() { continue; }
        let dst = if name.trim_start().starts_with("thought") || name.trim_start().starts_with("think") {
            &mut thinking
        } else { &mut answer };
        if !dst.is_empty() { dst.push('\n'); }
        dst.push_str(&content);
    }
    if answer.is_empty() { std::mem::swap(&mut answer, &mut thinking); }
    (thinking, answer)
}

#[cfg(not(all(feature = "rocm", modgrad_hipcc_kernels)))]
fn main() { eprintln!("build with --features rocm"); }
