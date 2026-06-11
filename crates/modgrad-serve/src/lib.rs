//! modgrad-serve — a model-agnostic local inference server.
//!
//! Exposes any [`LanguageModel`] over three protocols on one HTTP port:
//!
//!   GET  /health                        -> "ok"
//!   POST /generate {prompt,max_tokens}  -> {text,thinking,...}  (SSE if stream:true)
//!   POST /v1/messages                   -> Anthropic Messages API, so Claude Code
//!                                           can run ON the local model (SSE if stream)
//!   POST /mcp                           -> MCP JSON-RPC 2.0 tools:
//!                                           {prefix}_generate / _status / _load / _unload
//!
//! The server knows nothing about any specific model — tokenization, chat
//! templating, and reasoning-channel parsing all live behind the trait. To serve
//! a model, implement [`LanguageModel`] and call [`serve`].

use std::io::{Write, BufRead, BufReader, Read};
use std::net::{TcpListener, TcpStream};
use serde_json::{json, Value};

/// Chat role, as handed to a model's template.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Role { System, User, Assistant }

/// One chat message.
pub struct ChatMessage { pub role: Role, pub content: String }

/// The result of one generation. `answer` is the visible reply; `thinking` is any
/// hidden reasoning the model separated out (may be empty).
pub struct Generation {
    pub answer: String,
    pub thinking: String,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub tok_per_s: f64,
}

/// Everything the server needs from a model. Tokenization, the chat template, and
/// reasoning-channel parsing are the implementor's concern.
pub trait LanguageModel {
    /// Human-facing id (Anthropic `model` field default, MCP status).
    fn name(&self) -> &str;
    /// Max context length in tokens (reported by MCP status).
    fn context_len(&self) -> usize;

    /// Bring the model into a ready state (e.g. load weights into VRAM). Default
    /// suits an always-resident model.
    fn ensure_loaded(&mut self) -> Result<(), String> { Ok(()) }
    /// Release resources (e.g. free VRAM). The next `generate`/`ensure_loaded`
    /// must transparently reload.
    fn unload(&mut self) {}
    fn is_loaded(&self) -> bool { true }

    /// Generate an assistant reply to `messages`. Streams *answer* text deltas via
    /// `on_delta` (reasoning is suppressed from the stream but captured in the
    /// returned [`Generation`]). `wrap == false` => treat the last user message as
    /// a raw prompt and skip the chat template.
    fn generate(&mut self, messages: &[ChatMessage], max_tokens: usize, wrap: bool,
                on_delta: &mut dyn FnMut(&str)) -> Result<Generation, String>;
}

/// Server knobs. `tool_prefix` names the MCP tools (`{prefix}_generate`, …) so an
/// existing `claude mcp add` config keeps working across models.
pub struct ServeConfig {
    pub addr: String,
    pub tool_prefix: String,
    pub server_name: String,
}
impl Default for ServeConfig {
    fn default() -> Self {
        Self { addr: "127.0.0.1:8080".into(), tool_prefix: "model".into(),
               server_name: "modgrad-local".into() }
    }
}

/// Bind `cfg.addr` and serve `model` until the listener dies. Single-threaded: one
/// request at a time (the model is a single GPU resource), which is exactly what a
/// local one-user server wants.
pub fn serve<M: LanguageModel>(mut model: M, cfg: ServeConfig) -> std::io::Result<()> {
    let listener = TcpListener::bind(&cfg.addr)?;
    eprintln!("[modgrad-serve] READY — http://{}  (REST /generate, Anthropic /v1/messages, MCP /mcp, GET /health)", cfg.addr);
    for stream in listener.incoming() {
        let mut stream = match stream { Ok(s) => s, Err(_) => continue };
        let (method, path, body) = match read_request(&mut stream) { Some(r) => r, None => continue };
        // Strip any query string (Claude Code calls `/v1/messages?beta=true`).
        let path = path.split('?').next().unwrap_or("").to_string();
        eprintln!("[req] {method} {path} ({}B)", body.len());
        route(&mut stream, &method, &path, &body, &mut model, &cfg);
    }
    Ok(())
}

fn route<M: LanguageModel>(stream: &mut TcpStream, method: &str, path: &str,
                           body: &[u8], model: &mut M, cfg: &ServeConfig) {
    match (method, path) {
        ("GET", "/health") => respond(stream, "200 OK", "text/plain", "ok"),
        ("POST", "/generate") => handle_generate(stream, body, model),
        ("POST", "/v1/messages") => {
            if let Err(e) = handle_messages(stream, body, model) {
                respond(stream, "500 Internal Server Error", "text/plain", &e);
            }
        }
        ("POST", "/mcp") => {
            let (code, out) = handle_mcp(body, model, cfg);
            if code == "202" {
                let _ = write!(stream, "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            } else {
                respond(stream, "200 OK", "application/json", &out);
            }
        }
        _ => respond(stream, "404 Not Found", "text/plain",
                     "POST /generate | POST /v1/messages | POST /mcp | GET /health"),
    }
}

// ── REST /generate ────────────────────────────────────────────────────────────
fn handle_generate<M: LanguageModel>(stream: &mut TcpStream, body: &[u8], model: &mut M) {
    let v: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => { respond(stream, "400 Bad Request", "text/plain", &format!("bad json: {e}")); return; }
    };
    let prompt = v.get("prompt").and_then(|p| p.as_str()).unwrap_or("").to_string();
    let max_tokens = v.get("max_tokens").and_then(|m| m.as_u64()).unwrap_or(256) as usize;
    let wrap = v.get("raw").and_then(|r| r.as_bool()) != Some(true); // raw=true => no chat wrap
    let want_stream = v.get("stream").and_then(|s| s.as_bool()) == Some(true);
    if prompt.is_empty() { respond(stream, "400 Bad Request", "text/plain", "missing prompt"); return; }
    let messages = [ChatMessage { role: Role::User, content: prompt }];

    if want_stream {
        // Catch load errors before committing to a 200 SSE response.
        if let Err(e) = model.ensure_loaded() { respond(stream, "500 Internal Server Error", "text/plain", &e); return; }
        if write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n").is_err() { return; }
        let _ = stream.flush();
        let res = {
            let mut sink = |chunk: &str| {
                let _ = write!(stream, "data: {}\n\n", json!({"token": chunk}));
                let _ = stream.flush();
            };
            model.generate(&messages, max_tokens, wrap, &mut sink)
        };
        match res {
            Ok(g) => { let _ = write!(stream, "data: {}\n\n", json!({"done": true, "tokens": g.output_tokens, "tok_per_s": g.tok_per_s})); }
            Err(e) => { let _ = write!(stream, "data: {}\n\n", json!({"error": e})); }
        }
        let _ = stream.flush();
        return;
    }

    match model.generate(&messages, max_tokens, wrap, &mut |_| {}) {
        Ok(g) => {
            let resp = json!({"text": g.answer, "thinking": g.thinking, "tokens": g.output_tokens,
                              "prompt_tokens": g.prompt_tokens, "tok_per_s": g.tok_per_s}).to_string();
            respond(stream, "200 OK", "application/json", &resp);
        }
        Err(e) => respond(stream, "500 Internal Server Error", "text/plain", &e),
    }
}

// ── Anthropic Messages API (POST /v1/messages) ─────────────────────────────────
fn handle_messages<M: LanguageModel>(out: &mut TcpStream, body: &[u8], model: &mut M) -> Result<(), String> {
    let req: Value = serde_json::from_slice(body).map_err(|e| format!("bad json: {e}"))?;
    let want_stream = req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let max_tokens = req.get("max_tokens").and_then(|m| m.as_u64()).unwrap_or(512).min(4096) as usize;
    let model_name = req.get("model").and_then(|m| m.as_str()).unwrap_or(model.name()).to_string();

    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = req.get("system") {
        let s = extract_text(sys);
        if !s.is_empty() { messages.push(ChatMessage { role: Role::System, content: s }); }
    }
    for msg in req.get("messages").and_then(|m| m.as_array()).into_iter().flatten() {
        let role = match msg.get("role").and_then(|r| r.as_str()) {
            Some("assistant") => Role::Assistant, _ => Role::User };
        let content = extract_text(msg.get("content").unwrap_or(&Value::Null));
        messages.push(ChatMessage { role, content });
    }

    if !want_stream {
        let g = model.generate(&messages, max_tokens, true, &mut |_| {})?;
        let resp = json!({"id":"msg_local","type":"message","role":"assistant","model": model_name,
            "content":[{"type":"text","text": g.answer}], "stop_reason":"end_turn","stop_sequence":null,
            "usage":{"input_tokens": g.prompt_tokens, "output_tokens": g.output_tokens}}).to_string();
        write!(out, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp}", resp.len())
            .map_err(|e| e.to_string())?;
        return Ok(());
    }

    // Streaming: load before headers so load failure is a proper 500, not an SSE error.
    model.ensure_loaded()?;
    write!(out, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")
        .map_err(|e| e.to_string())?;
    let send = |out: &mut TcpStream, name: &str, data: Value| {
        let _ = write!(out, "event: {name}\ndata: {data}\n\n");
        let _ = out.flush();
    };
    send(out, "message_start", json!({"type":"message_start","message":{"id":"msg_local","type":"message",
        "role":"assistant","model": model_name,"content":[],"stop_reason":null,"stop_sequence":null,
        "usage":{"input_tokens":0,"output_tokens":0}}}));
    send(out, "content_block_start", json!({"type":"content_block_start","index":0,
        "content_block":{"type":"text","text":""}}));
    let res = {
        let mut sink = |chunk: &str| {
            let _ = write!(out, "event: content_block_delta\ndata: {}\n\n",
                json!({"type":"content_block_delta","index":0,
                       "delta":{"type":"text_delta","text": chunk}}));
            let _ = out.flush();
        };
        model.generate(&messages, max_tokens, true, &mut sink)
    };
    let out_tokens = res.as_ref().map(|g| g.output_tokens).unwrap_or(0);
    send(out, "content_block_stop", json!({"type":"content_block_stop","index":0}));
    send(out, "message_delta", json!({"type":"message_delta",
        "delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens": out_tokens}}));
    send(out, "message_stop", json!({"type":"message_stop"}));
    Ok(())
}

/// Flatten an Anthropic content value (string, or array of blocks) to plain text.
fn extract_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter_map(|b| match b.get("type").and_then(|t| t.as_str()) {
            Some("text") => b.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()),
            Some("tool_result") => b.get("content").map(extract_text),
            _ => None, // tool_use / images ignored in the text-only shim
        }).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

// ── MCP (JSON-RPC 2.0 over HTTP) ───────────────────────────────────────────────
fn handle_mcp<M: LanguageModel>(body: &[u8], model: &mut M, cfg: &ServeConfig) -> (&'static str, String) {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return ("200", json!({"jsonrpc":"2.0","id":null,
            "error":{"code":-32700,"message":format!("parse error: {e}")}}).to_string()),
    };
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    if method.starts_with("notifications/") { return ("202", String::new()); }

    let ok = |v: Value| ("200", json!({"jsonrpc":"2.0","id":id.clone(),"result":v}).to_string());
    let tool_text = |s: String, is_err: bool| json!({"content":[{"type":"text","text":s}],"isError":is_err});
    let p = &cfg.tool_prefix;
    let (t_gen, t_status, t_load, t_unload) =
        (format!("{p}_generate"), format!("{p}_status"), format!("{p}_load"), format!("{p}_unload"));

    match method {
        "initialize" => {
            let pv = req.get("params").and_then(|p| p.get("protocolVersion"))
                .and_then(|v| v.as_str()).unwrap_or("2024-11-05").to_string();
            ok(json!({"protocolVersion": pv, "capabilities": {"tools": {}},
                      "serverInfo": {"name": cfg.server_name, "version": "0.1.0"}}))
        }
        "ping" => ok(json!({})),
        "tools/list" => ok(json!({"tools": [
            {"name": t_gen,
             "description": format!("Generate text with the local model ({}), running on this machine's GPU. Pass a user question/instruction; returns the answer.", model.name()),
             "inputSchema": {"type":"object","properties":{
                "prompt":{"type":"string","description":"The user message / question / instruction"},
                "max_tokens":{"type":"integer","description":"Max tokens to generate (default 256)"}},
                "required":["prompt"]}},
            {"name": t_status, "description": "Report local model status: loaded into VRAM? context length.",
             "inputSchema": {"type":"object","properties":{}}},
            {"name": t_load, "description": "Load the model into GPU VRAM. Call before heavy use.",
             "inputSchema": {"type":"object","properties":{}}},
            {"name": t_unload, "description": "Unload the model and FREE VRAM (e.g. to play a game). The server keeps running; the next generate reloads it.",
             "inputSchema": {"type":"object","properties":{}}}
        ]})),
        "tools/call" => {
            let params = req.get("params");
            let name = params.and_then(|p| p.get("name")).and_then(|n| n.as_str()).unwrap_or("");
            let args = params.and_then(|p| p.get("arguments")).cloned().unwrap_or(json!({}));
            if name == t_gen {
                let prompt = args.get("prompt").and_then(|p| p.as_str()).unwrap_or("").to_string();
                let max_tokens = args.get("max_tokens").and_then(|m| m.as_u64()).unwrap_or(256) as usize;
                let messages = [ChatMessage { role: Role::User, content: prompt }];
                match model.generate(&messages, max_tokens, true, &mut |_| {}) {
                    Ok(g) => {
                        let text = if g.thinking.is_empty() { g.answer }
                                   else { format!("{}\n\n[reasoning: {}]", g.answer, g.thinking) };
                        ok(tool_text(text, false))
                    }
                    Err(e) => ok(tool_text(format!("error: {e}"), true)),
                }
            } else if name == t_load {
                if model.is_loaded() { ok(tool_text("already loaded".into(), false)) }
                else { match model.ensure_loaded() {
                    Ok(()) => ok(tool_text("model loaded into VRAM".into(), false)),
                    Err(e) => ok(tool_text(format!("load failed: {e}"), true)),
                }}
            } else if name == t_unload {
                let was = model.is_loaded();
                model.unload();
                ok(tool_text(if was { "model unloaded — VRAM freed".into() } else { "already unloaded".into() }, false))
            } else if name == t_status {
                ok(tool_text(format!("{} | loaded={} | ctx={} | endpoint=this server",
                    model.name(), model.is_loaded(), model.context_len()), false))
            } else {
                ok(tool_text(format!("unknown tool: {name}"), true))
            }
        }
        other => ("200", json!({"jsonrpc":"2.0","id":id,
            "error":{"code":-32601,"message":format!("method not found: {other}")}}).to_string()),
    }
}

// ── tiny HTTP/1.1 plumbing ─────────────────────────────────────────────────────
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
