/// isis serve — load a GGUF model and generate text.
/// First milestone: CLI generation. Next: HTTP + MCP.

use modgrad_device::kfd::{HsaDevice, dispatch_queue::GpuQueue, gguf::GgufFile, inference::Gemma4Model};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str())
        .unwrap_or("/steam/llm/gemma-4-E4B-it.Q4_K_M.gguf");
    let prompt = args.get(2).map(|s| s.as_str())
        .unwrap_or("Hello, I am");
    let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);

    eprintln!("Loading {}...", model_path);

    // Parse GGUF
    let mut f = std::fs::File::open(model_path).unwrap();
    let gguf = GgufFile::parse(&mut f).unwrap();
    drop(f);

    // mmap the file
    let file = std::fs::File::open(model_path).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            file_len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            std::os::unix::io::AsRawFd::as_raw_fd(&file),
            0,
        )
    };
    if mmap == libc::MAP_FAILED {
        eprintln!("mmap failed");
        return;
    }
    let file_data = unsafe { std::slice::from_raw_parts(mmap as *const u8, file_len) };

    // Open GPU
    let mut dev = match HsaDevice::open() {
        Ok(d) => d,
        Err(e) => { eprintln!("No GPU: {}", e); return; }
    };
    let mut queue = GpuQueue::new();

    // Load model
    let max_seq = 512; // reduced for 8GB VRAM with varying KV dims
    let mut model = match Gemma4Model::load(&gguf, file_data, &dev, &mut queue, max_seq) {
        Ok(m) => m,
        Err(e) => { eprintln!("Failed to load model: {}", e); return; }
    };

    eprintln!("Model loaded. Generating...\n");

    // Hardcoded token IDs for testing (proper tokenizer TODO)
    // BOS=2, ▁Hello=26352, ▁Hi=18428, ▁H=640
    let prompt_tokens: Vec<u32> = if prompt == "Hello" {
        vec![2, 26352]
    } else if prompt == "Hi" {
        vec![2, 18428]
    } else {
        // Fallback: BOS + each byte shifted into the token range
        // Gemma4 tokens 236700+ are single bytes
        let mut toks = vec![2u32]; // BOS
        for &b in prompt.as_bytes() {
            toks.push(236700 + b as u32); // approximate byte token range
        }
        toks
    };
    eprintln!("Tokens: {:?}", prompt_tokens);

    // Prefill: process all prompt tokens
    let mut last_logits = vec![0.0f32; 0];
    let start = std::time::Instant::now();
    for &tok in &prompt_tokens {
        last_logits = model.forward_token(tok, &mut dev, &mut queue);
    }
    let prefill_time = start.elapsed();
    eprintln!("Prefill: {} tokens in {:.1}ms ({:.1} tok/s)",
        prompt_tokens.len(),
        prefill_time.as_millis(),
        prompt_tokens.len() as f64 / prefill_time.as_secs_f64());

    // Print prompt
    print!("{}", prompt);
    std::io::stdout().flush().ok();

    // Generate
    let gen_start = std::time::Instant::now();
    for i in 0..max_tokens {
        if last_logits.is_empty() { break; }

        // Greedy sampling (argmax, NaN-safe)
        let next_token = last_logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        // Debug: raw logit stats (no softcapping)
        if i == 0 {
            let mut sorted: Vec<(usize, f32)> = last_logits.iter().cloned().enumerate().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("  raw logit stats: min={:.0} max={:.0} mean={:.0}",
                last_logits.iter().cloned().fold(f32::INFINITY, f32::min),
                last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                last_logits.iter().sum::<f32>() / last_logits.len() as f32);
            eprintln!("  top 10: {:?}", sorted[..10].iter().map(|(i,v)| (i, *v as i64)).collect::<Vec<_>>());
            eprintln!("  spread: top-10th = {:.0}", sorted[0].1 - sorted[9].1);
        }

        // Print token ID (no tokenizer yet)
        eprint!(" tok={}", next_token);
        std::io::stdout().flush().ok();

        // EOS check (token 0 or 1 typically)
        if next_token <= 1 { break; }

        last_logits = model.forward_token(next_token, &mut dev, &mut queue);
    }

    let gen_time = gen_start.elapsed();
    let gen_tokens = max_tokens.min(model.kv_len - prompt_tokens.len());
    eprintln!("\n\nGeneration: {} tokens in {:.1}ms ({:.1} tok/s)",
        gen_tokens,
        gen_time.as_millis(),
        gen_tokens as f64 / gen_time.as_secs_f64());

    // Cleanup
    unsafe { libc::munmap(mmap, file_len); }
}
