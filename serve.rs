/// isis serve — GGUF inference with OpenAI-compatible HTTP API.
///
/// Loads a quantized model via candle, serves /v1/chat/completions.
/// Single-threaded, sequential requests. No dependencies beyond std + candle.
///
/// Usage: isis serve model.gguf [--port 8080]

#[cfg(feature = "gguf")]
mod server {
    use candle_core::{Device, Tensor};
    use candle_core::quantized::gguf_file;
    use candle_transformers::models::quantized_qwen2 as qqwen;
    use std::collections::HashMap;
    use std::io::{Read, Write, BufRead, BufReader};
    use std::net::TcpListener;

    /// Loaded model + vocab for inference.
    pub struct InferenceEngine {
        model: qqwen::ModelWeights,
        device: Device,
        vocab: HashMap<usize, String>,
        vocab_size: usize,
        // Special token IDs (Qwen2.5)
        im_start: u32,
        im_end: u32,
        eos: u32,
    }

    impl InferenceEngine {
        pub fn load(path: &str) -> Self {
            eprintln!("Loading {}...", path);
            let device = Device::Cpu;
            let mut file = std::fs::File::open(path).unwrap();
            let model_data = gguf_file::Content::read(&mut file).unwrap();

            let arch = model_data.metadata.get("general.architecture")
                .and_then(|v| match v { gguf_file::Value::String(s) => Some(s.as_str()), _ => None })
                .unwrap_or("unknown");
            eprintln!("  architecture: {}", arch);

            let model = qqwen::ModelWeights::from_gguf(model_data, &mut file, &device).unwrap();
            let vocab = crate::load_vocab(path);
            let vocab_size = vocab.len();

            // Find special tokens
            let find_tok = |name: &str| -> u32 {
                vocab.iter().find(|(_, v)| v.as_str() == name).map(|(k, _)| *k as u32).unwrap_or(0)
            };

            let im_start = find_tok("<|im_start|>");
            let im_end = find_tok("<|im_end|>");
            let eos = find_tok("<|endoftext|>");
            eprintln!("  vocab: {} tokens, im_start={}, im_end={}, eos={}", vocab_size, im_start, im_end, eos);
            eprintln!("Model loaded.");

            Self { model, device, vocab, vocab_size, im_start, im_end, eos }
        }

        /// Encode a ChatML conversation into token IDs.
        /// Uses vocab lookup for known words, falls back to byte tokens.
        fn encode_chat(&self, messages: &[Message]) -> Vec<u32> {
            let mut tokens = Vec::new();
            for msg in messages {
                tokens.push(self.im_start);
                // Tokenize role
                self.encode_text(&msg.role, &mut tokens);
                tokens.push(198); // \n
                self.encode_text(&msg.content, &mut tokens);
                tokens.push(self.im_end);
                tokens.push(198); // \n
            }
            // Start assistant turn
            tokens.push(self.im_start);
            self.encode_text("assistant", &mut tokens);
            tokens.push(198); // \n
            tokens
        }

        /// Simple greedy tokenizer: try longest match from vocab.
        fn encode_text(&self, text: &str, tokens: &mut Vec<u32>) {
            // Build reverse lookup if not cached (first call is slow)
            let text = text.as_bytes();
            let mut pos = 0;
            while pos < text.len() {
                // Try progressively shorter matches
                let mut best_len = 0;
                let mut best_tok = 0u32;
                let max_try = 32.min(text.len() - pos);
                for len in (1..=max_try).rev() {
                    let candidate = &text[pos..pos + len];
                    if let Ok(s) = std::str::from_utf8(candidate) {
                        // Check with Ġ prefix (space) and without
                        let with_space = format!("Ġ{}", s);
                        let found = self.vocab.iter().find(|(_, v)| {
                            v.as_str() == s || v.as_str() == with_space
                        });
                        if let Some((&id, v)) = found {
                            // Prefer the space-prefixed version if we're after a space boundary
                            best_len = len;
                            best_tok = id as u32;
                            break;
                        }
                    }
                }
                if best_len > 0 {
                    tokens.push(best_tok);
                    pos += best_len;
                } else {
                    // Fallback: single byte
                    tokens.push(text[pos] as u32);
                    pos += 1;
                }
            }
        }

        /// Generate tokens from a prompt.
        fn generate(&mut self, prompt_tokens: &[u32], max_tokens: usize, temperature: f32) -> Vec<u32> {
            // Prefill
            let input = Tensor::new(prompt_tokens, &self.device).unwrap()
                .unsqueeze(0).unwrap();
            let logits = self.model.forward(&input, 0).unwrap();
            let mut last_logits = self.extract_logits(&logits);

            let mut output_tokens = Vec::new();
            for _ in 0..max_tokens {
                let next = if temperature < 0.01 {
                    // Greedy
                    last_logits.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as u32).unwrap_or(0)
                } else {
                    // Temperature sampling
                    self.sample_with_temperature(&last_logits, temperature)
                };

                if next == self.im_end || next == self.eos { break; }
                output_tokens.push(next);

                // Next step
                let pos = prompt_tokens.len() + output_tokens.len() - 1;
                let input = Tensor::new(&[next][..], &self.device).unwrap()
                    .unsqueeze(0).unwrap();
                let logits = self.model.forward(&input, pos).unwrap();
                last_logits = self.extract_logits(&logits);
            }
            output_tokens
        }

        fn extract_logits(&self, logits: &Tensor) -> Vec<f32> {
            if logits.dims().len() == 2 {
                logits.squeeze(0).unwrap().to_vec1().unwrap()
            } else {
                logits.to_vec1().unwrap()
            }
        }

        fn sample_with_temperature(&self, logits: &[f32], temp: f32) -> u32 {
            // Apply temperature
            let scaled: Vec<f32> = logits.iter().map(|&v| v / temp).collect();
            // Softmax
            let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scaled.iter().map(|&v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
            // Sample
            let mut r: f32 = rand_f32();
            for (i, &p) in probs.iter().enumerate() {
                r -= p;
                if r <= 0.0 { return i as u32; }
            }
            probs.len() as u32 - 1
        }

        /// Decode token IDs to text.
        fn decode(&self, tokens: &[u32]) -> String {
            let mut text = String::new();
            for &tok in tokens {
                if let Some(s) = self.vocab.get(&(tok as usize)) {
                    // Qwen2.5 BPE byte-level encoding
                    let decoded = s.replace("Ċ", "\n").replace("Ġ", " ").replace("ĉ", "\t");
                    text.push_str(&decoded);
                }
            }
            text
        }
    }

    #[derive(Debug)]
    struct Message {
        role: String,
        content: String,
    }

    /// Simple JSON parser for chat completions request.
    fn parse_chat_request(body: &str) -> (Vec<Message>, usize, f32, bool) {
        let mut messages = Vec::new();
        let mut max_tokens = 256usize;
        let mut temperature = 0.7f32;
        let mut stream = false;

        // Crude JSON parsing — find "messages" array
        if let Some(start) = body.find("\"messages\"") {
            if let Some(arr_start) = body[start..].find('[') {
                let arr = &body[start + arr_start..];
                // Find each message object
                let mut pos = 1;
                while let Some(obj_start) = arr[pos..].find('{') {
                    let obj_pos = pos + obj_start;
                    if let Some(obj_end) = arr[obj_pos..].find('}') {
                        let obj = &arr[obj_pos..obj_pos + obj_end + 1];
                        let role = extract_json_string(obj, "role").unwrap_or_default();
                        let content = extract_json_string(obj, "content").unwrap_or_default();
                        if !role.is_empty() {
                            messages.push(Message { role, content });
                        }
                        pos = obj_pos + obj_end + 1;
                    } else { break; }
                }
            }
        }

        if let Some(v) = extract_json_number(body, "max_tokens") { max_tokens = v as usize; }
        if let Some(v) = extract_json_float(body, "temperature") { temperature = v; }
        if body.contains("\"stream\":true") || body.contains("\"stream\": true") { stream = true; }

        (messages, max_tokens, temperature, stream)
    }

    fn extract_json_string(json: &str, key: &str) -> Option<String> {
        let needle = format!("\"{}\"", key);
        let pos = json.find(&needle)? + needle.len();
        let rest = &json[pos..];
        let colon = rest.find(':')?;
        let after = rest[colon + 1..].trim_start();
        if after.starts_with('"') {
            let end = after[1..].find('"')?;
            Some(after[1..1 + end].to_string())
        } else { None }
    }

    fn extract_json_number(json: &str, key: &str) -> Option<f64> {
        let needle = format!("\"{}\"", key);
        let pos = json.find(&needle)? + needle.len();
        let rest = &json[pos..];
        let colon = rest.find(':')?;
        let after = rest[colon + 1..].trim_start();
        let end = after.find(|c: char| !c.is_numeric() && c != '.').unwrap_or(after.len());
        after[..end].parse().ok()
    }

    fn extract_json_float(json: &str, key: &str) -> Option<f32> {
        extract_json_number(json, key).map(|v| v as f32)
    }

    fn rand_f32() -> f32 {
        use std::time::SystemTime;
        let t = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
        let seed = t.as_nanos() as u64;
        ((seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 33) as f32)
            / (1u64 << 31) as f32
    }

    pub fn run(model_path: &str, port: u16) {
        let mut engine = InferenceEngine::load(model_path);

        let addr = format!("0.0.0.0:{}", port);
        let listener = TcpListener::bind(&addr).unwrap();
        eprintln!("Serving on http://{}:{}", "localhost", port);
        eprintln!("  POST /v1/chat/completions");
        eprintln!("  Compatible with OpenAI API clients.");

        for stream in listener.incoming() {
            let mut stream = match stream { Ok(s) => s, Err(_) => continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());

            // Read HTTP request
            let mut first_line = String::new();
            if reader.read_line(&mut first_line).is_err() { continue; }

            let mut headers = HashMap::new();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() { break; }
                let line = line.trim().to_string();
                if line.is_empty() { break; }
                if let Some((k, v)) = line.split_once(": ") {
                    headers.insert(k.to_lowercase(), v.to_string());
                }
            }

            let content_length: usize = headers.get("content-length")
                .and_then(|v| v.parse().ok()).unwrap_or(0);

            let mut body = vec![0u8; content_length];
            if content_length > 0 {
                let _ = reader.read_exact(&mut body);
            }
            let body = String::from_utf8_lossy(&body).to_string();

            // Route
            let path = first_line.split_whitespace().nth(1).unwrap_or("/");
            let method = first_line.split_whitespace().next().unwrap_or("GET");

            let response = if method == "GET" && path == "/v1/models" {
                // Model list
                let json = r#"{"object":"list","data":[{"id":"qwen2.5-0.5b","object":"model","owned_by":"modgrad"}]}"#;
                format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}", json.len(), json)
            } else if method == "POST" && path == "/v1/chat/completions" {
                let (messages, max_tokens, temperature, _stream) = parse_chat_request(&body);

                if messages.is_empty() {
                    let err = r#"{"error":{"message":"No messages provided","type":"invalid_request_error"}}"#;
                    format!("HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", err.len(), err)
                } else {
                    let start = std::time::Instant::now();
                    let prompt = engine.encode_chat(&messages);
                    let output_tokens = engine.generate(&prompt, max_tokens, temperature);
                    let text = engine.decode(&output_tokens);
                    let elapsed = start.elapsed();
                    let tok_s = output_tokens.len() as f64 / elapsed.as_secs_f64();

                    eprintln!("  {} tokens in {:.0}ms ({:.1} tok/s): {}",
                        output_tokens.len(), elapsed.as_millis(), tok_s,
                        text.chars().take(60).collect::<String>());

                    // Escape text for JSON
                    let escaped = text.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");

                    let json = format!(
                        r#"{{"id":"chatcmpl-1","object":"chat.completion","model":"qwen2.5-0.5b","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
                        escaped, prompt.len(), output_tokens.len(), prompt.len() + output_tokens.len()
                    );
                    format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}", json.len(), json)
                }
            } else if method == "OPTIONS" {
                "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n".to_string()
            } else {
                let err = r#"{"error":"not found"}"#;
                format!("HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", err.len(), err)
            };

            let _ = stream.write_all(response.as_bytes());
        }
    }
}

#[cfg(feature = "gguf")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str())
        .unwrap_or("/steam/llm/qwen2-small.gguf");
    let port: u16 = args.iter().position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    server::run(model_path, port);
}

#[cfg(feature = "gguf")]
fn load_vocab(path: &str) -> std::collections::HashMap<usize, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap();
    let mut buf4 = [0u8; 4]; let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf4).unwrap();
    f.read_exact(&mut buf4).unwrap();
    f.read_exact(&mut buf8).unwrap();
    f.read_exact(&mut buf8).unwrap();
    let nk = u64::from_le_bytes(buf8) as usize;

    fn read_str(f: &mut std::fs::File) -> String {
        let mut buf8 = [0u8; 8];
        f.read_exact(&mut buf8).unwrap();
        let n = u64::from_le_bytes(buf8) as usize;
        let mut s = vec![0u8; n];
        f.read_exact(&mut s).unwrap();
        String::from_utf8_lossy(&s).to_string()
    }
    fn skip_val(f: &mut std::fs::File, t: u32) {
        let sizes: std::collections::HashMap<u32, usize> = [(0,1),(1,1),(2,2),(3,2),(4,4),(5,4),(6,4),(7,1),(10,8),(12,8)].into();
        if t == 8 { read_str(f); }
        else if t == 9 {
            let mut b4 = [0u8; 4]; let mut b8 = [0u8; 8];
            f.read_exact(&mut b4).unwrap();
            let at = u32::from_le_bytes(b4);
            f.read_exact(&mut b8).unwrap();
            let al = u64::from_le_bytes(b8) as usize;
            for _ in 0..al { skip_val(f, at); }
        } else if let Some(&sz) = sizes.get(&t) {
            let mut buf = vec![0u8; sz]; f.read_exact(&mut buf).unwrap();
        } else { let mut buf = [0u8; 4]; f.read_exact(&mut buf).unwrap(); }
    }

    let mut vocab = std::collections::HashMap::new();
    for _ in 0..nk {
        let key = read_str(&mut f);
        let mut b4 = [0u8; 4];
        f.read_exact(&mut b4).unwrap();
        let vt = u32::from_le_bytes(b4);
        if key == "tokenizer.ggml.tokens" {
            f.read_exact(&mut b4).unwrap();
            f.read_exact(&mut buf8).unwrap();
            let al = u64::from_le_bytes(buf8) as usize;
            for i in 0..al {
                let tok = read_str(&mut f);
                vocab.insert(i, tok);
            }
        } else {
            skip_val(&mut f, vt);
        }
    }
    vocab
}

#[cfg(not(feature = "gguf"))]
fn main() {
    eprintln!("Build with: cargo build --release --features gguf -p modgrad --bin serve");
}
