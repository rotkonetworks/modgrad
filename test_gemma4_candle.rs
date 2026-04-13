/// Gemma4 inference via candle (safetensors, proven implementation).
/// This bypasses our custom forward pass entirely.

#[cfg(feature = "gguf")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::gemma4::text::TextModel;
    use candle_transformers::models::gemma4::config::Gemma4TextConfig;

    let cache_dir = "/steam/llm/hf_cache/models--google--gemma-4-E4B-it/snapshots";
    let snapshot = std::fs::read_dir(cache_dir)?
        .filter_map(|e| e.ok())
        .next()
        .ok_or("no snapshot found")?;
    let snap_path = snapshot.path();

    let config_path = snap_path.join("config.json");
    let tokenizer_path = snap_path.join("tokenizer.json");
    let model_path = snap_path.join("model.safetensors");

    eprintln!("Loading config from {:?}", config_path);
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let text_config: Gemma4TextConfig = if let Some(tc) = raw.get("text_config") {
        serde_json::from_value(tc.clone())?
    } else {
        serde_json::from_value(raw)?
    };

    eprintln!("Loading tokenizer...");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("tokenizer: {}", e))?;

    eprintln!("Loading model weights from {:?}...", model_path);
    let device = Device::Cpu;
    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&model_path], dtype, &device)? };
    let mut model = TextModel::new(&text_config, vb)?;
    eprintln!("Model loaded.");

    // Tokenize
    let prompt = "Hello";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{}", e))?;
    let tokens = encoding.get_ids();
    eprintln!("Prompt tokens: {:?}", tokens);

    // Forward
    let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = logits.to_vec1()?;

    eprintln!("Logits: len={} min={:.2} max={:.2}",
        logits_vec.len(),
        logits_vec.iter().cloned().fold(f32::INFINITY, f32::min),
        logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Top 5
    let mut sorted: Vec<(usize, f32)> = logits_vec.iter().cloned().enumerate().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("Top 5: {:?}", &sorted[..5]);

    // Decode top token
    let top_id = sorted[0].0 as u32;
    if let Some(decoded) = tokenizer.decode(&[top_id], false).ok() {
        eprintln!("Top token: {} = '{}'", top_id, decoded);
    }

    // Generate
    let mut all_tokens: Vec<u32> = tokens.to_vec();
    let start = std::time::Instant::now();
    for i in 0..20 {
        let input = Tensor::new(&all_tokens[all_tokens.len()-1..], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, all_tokens.len() - 1)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        let next = logits_vec.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32).unwrap();
        all_tokens.push(next);
        if let Some(decoded) = tokenizer.decode(&[next], false).ok() {
            eprint!("{}", decoded);
        }
    }
    let elapsed = start.elapsed();
    eprintln!("\n20 tokens in {:.1}ms ({:.1} tok/s)",
        elapsed.as_millis(), 20.0 / elapsed.as_secs_f64());

    Ok(())
}

#[cfg(not(feature = "gguf"))]
fn main() {
    eprintln!("Build with: cargo build --release --features gguf");
}
