/// Test candle's quantized model with the Qwen2.5-0.5B GGUF.
/// This bypasses our custom forward pass entirely — uses candle's
/// proven quantized_llama implementation.

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

    // BOS token for Qwen2.5
    let tokens = vec![151643u32, 9707]; // <|im_start|> + "Hello" approximation
    // Actually let's just use simple tokens
    let input = Tensor::new(&[151643u32][..], &device).unwrap()
        .unsqueeze(0).unwrap(); // [1, 1] — batch=1, seq_len=1

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

    // Generate a few tokens
    let mut all_tokens = vec![151643u32];
    for i in 0..20 {
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
        all_tokens.push(next);
        eprint!(" {}", next);
    }
    eprintln!("\n\nGenerated tokens: {:?}", &all_tokens[1..]);
}

#[cfg(not(feature = "gguf"))]
fn main() {
    eprintln!("Build with --features gguf");
}
