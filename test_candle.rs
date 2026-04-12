/// Test candle's quantized model with the Qwen2.5-0.5B GGUF.
/// This bypasses our custom forward pass entirely — uses candle's
/// proven quantized_llama implementation.

#[cfg(feature = "gguf")]
fn load_vocab(path: &str) -> std::collections::HashMap<usize, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap();
    let mut buf4 = [0u8; 4]; let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf4).unwrap(); // magic
    f.read_exact(&mut buf4).unwrap(); // version
    f.read_exact(&mut buf8).unwrap(); // n_tensors
    f.read_exact(&mut buf8).unwrap(); // n_kv
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
            f.read_exact(&mut b4).unwrap(); // array type
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

#[cfg(feature = "gguf")]
fn main() {
    use candle_core::{Device, Tensor};
    use candle_core::quantized::gguf_file;
    use candle_transformers::models::quantized_qwen2 as qqwen;

    let path = "/steam/llm/qwen2-small.gguf";
    eprintln!("Loading {}...", path);

    let device = Device::Cpu;
    let mut file = std::fs::File::open(path).unwrap();
    let model_data = gguf_file::Content::read(&mut file).unwrap();

    // Print architecture
    for (key, val) in &model_data.metadata {
        if key.contains("arch") || key.contains("block_count") || key.contains("embedding") {
            eprintln!("  {} = {:?}", key, val);
        }
    }

    let mut model = qqwen::ModelWeights::from_gguf(model_data, &mut file, &device).unwrap();
    eprintln!("Model loaded.");

    // Qwen2.5 ChatML: <|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n
    let prompt_tokens: Vec<u32> = vec![151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198];
    let input = Tensor::new(prompt_tokens.as_slice(), &device).unwrap()
        .unsqueeze(0).unwrap(); // [1, seq_len]

    let logits = model.forward(&input, 0).unwrap();
    eprintln!("Logits shape: {:?}", logits.dims());
    // Handle both [vocab] and [1, vocab] shapes
    let logits_vec: Vec<f32> = if logits.dims().len() == 2 {
        logits.squeeze(0).unwrap().to_vec1().unwrap()
    } else {
        logits.to_vec1().unwrap()
    };

    // Argmax
    let (top_idx, top_val) = logits_vec.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!("Logits: len={} min={:.2} max={:.2}", logits_vec.len(),
        logits_vec.iter().cloned().fold(f32::INFINITY, f32::min),
        logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    eprintln!("Top token: {} (logit={:.2})", top_idx, top_val);

    // Load vocab for decoding
    let vocab = load_vocab("/steam/llm/qwen2-small.gguf");

    // Generate tokens
    let mut all_tokens = prompt_tokens.clone();
    let gen_start = std::time::Instant::now();
    let mut n_gen = 0usize;
    eprint!("Output: ");
    for _ in 0..50 {
        let input = Tensor::new(&all_tokens[all_tokens.len()-1..], &device).unwrap()
            .unsqueeze(0).unwrap();
        let logits = model.forward(&input, all_tokens.len() - 1).unwrap();
        let logits_vec: Vec<f32> = if logits.dims().len() == 2 {
            logits.squeeze(0).unwrap().to_vec1().unwrap()
        } else {
            logits.to_vec1().unwrap()
        };
        let next = logits_vec.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32).unwrap();

        // EOS check
        if next == 151645 || next == 151643 { break; } // <|im_end|> or <|endoftext|>

        all_tokens.push(next);
        n_gen += 1;
        // Decode and print
        if let Some(tok_str) = vocab.get(&(next as usize)) {
            let s = tok_str.replace("Ċ", "\n").replace("Ġ", " ");
            eprint!("{}", s);
        } else {
            eprint!("[{}]", next);
        }
    }
    let elapsed = gen_start.elapsed();
    eprintln!();
    eprintln!("{} tokens in {:.1}ms ({:.1} tok/s)",
        n_gen, elapsed.as_secs_f64() * 1000.0,
        n_gen as f64 / elapsed.as_secs_f64());
}

#[cfg(not(feature = "gguf"))]
fn main() {
    eprintln!("Build with --features gguf");
}
