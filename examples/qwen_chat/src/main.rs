//! Path A demo: Qwen2.5-0.5B inference on our resident GPU runtime.
//!
//! Pipeline:
//!   1. `modgrad_io::qwen2::load_qwen2_5_0_5b` → `GptModelResident`
//!   2. `modgrad_io::tokenizer::HfTokenizer` (BPE)
//!   3. `isis_runtime::sampler::Sampler` (argmax / temp / top-k)
//!
//! Run:
//!   cargo run --release --features rocm -p qwen_chat -- \
//!       --prompt "Once upon a time" --max-tokens 50

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("qwen_chat: built without `--features rocm`. Rebuild with:");
    eprintln!("  cargo run --release --features rocm -p qwen_chat -- --prompt \"...\"");
    std::process::exit(0);
}

#[cfg(feature = "rocm")]
fn main() {
    use std::io::Write;
    use std::time::Instant;

    use clap::Parser;
    use modgrad_device::backend::HipBatch;
    use modgrad_device::backend::rocm::ffi::runtime_available;
    use modgrad_io::qwen2::{load_qwen2_5_0_5b, QWEN2_5_0_5B_NUM_KV_HEADS, QWEN2_5_0_5B_HEAD_DIM};
    use modgrad_io::tokenizer::HfTokenizer;
    use modgrad_transformer::kv_cache_resident::KvCacheResident;
    use isis_runtime::sampler::{Sampler, SamplerConfig};

    #[derive(Parser, Debug)]
    #[command(name = "qwen_chat")]
    struct Args {
        #[arg(long, default_value = "/steam/llm/hf_cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors")]
        model: String,

        #[arg(long, default_value = "/steam/llm/hf_cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/tokenizer.json")]
        tokenizer: String,

        #[arg(long, default_value = "Once upon a time")]
        prompt: String,

        #[arg(long, default_value_t = 64)]
        max_tokens: usize,

        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        #[arg(long)]
        top_k: Option<usize>,

        #[arg(long, default_value_t = 2048)]
        max_seq: usize,
    }

    let args = Args::parse();

    if !runtime_available() {
        eprintln!("qwen_chat: HIP runtime unavailable (no GPU?). Skipping.");
        std::process::exit(0);
    }

    eprintln!("qwen_chat: loading tokenizer from {}", args.tokenizer);
    let tokenizer = HfTokenizer::from_file(&args.tokenizer)
        .expect("load tokenizer");

    eprintln!("qwen_chat: loading model from {}", args.model);
    let load_start = Instant::now();
    let mut model = load_qwen2_5_0_5b(&args.model, args.max_seq)
        .expect("load qwen2.5-0.5b");
    eprintln!("qwen_chat: model loaded in {:.2}s",
        load_start.elapsed().as_secs_f32());

    let mut kv_cache = KvCacheResident::new(
        model.num_layers(),
        QWEN2_5_0_5B_NUM_KV_HEADS,
        QWEN2_5_0_5B_HEAD_DIM,
        args.max_seq,
        model.model_dim(),
    ).expect("alloc kv cache");

    let prompt_ids: Vec<i64> = tokenizer.encode(&args.prompt)
        .expect("encode prompt")
        .into_iter()
        .map(|id| id as i64)
        .collect();

    eprintln!("qwen_chat: prompt = {:?} ({} tokens)", args.prompt, prompt_ids.len());
    eprintln!("qwen_chat: prompt token ids = {:?}", prompt_ids);

    let stop_tokens = vec![151643i64, 151645i64];

    let mut sampler = Sampler::new(SamplerConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        max_new_tokens: args.max_tokens,
        stop_tokens,
        seed: 0xc0ffee_face_b00c,
    });

    let batch = HipBatch::new();

    eprintln!();
    print!("{}", args.prompt);
    let _ = std::io::stdout().flush();

    let gen_start = Instant::now();
    let mut emitted: Vec<u32> = Vec::with_capacity(args.max_tokens);
    let new_tokens = sampler.generate(
        &mut model,
        &batch,
        &prompt_ids,
        &mut kv_cache,
        |tok| {
            emitted.push(tok as u32);
            if let Ok(piece) = tokenizer.decode(&[tok as u32]) {
                print!("{piece}");
                let _ = std::io::stdout().flush();
            }
        },
    ).expect("sampler generate");

    let gen_ms = gen_start.elapsed().as_millis();
    let n = new_tokens.len();
    println!();
    eprintln!();
    eprintln!("qwen_chat: generated {n} tokens in {gen_ms} ms ({:.2} tok/s)",
        n as f64 * 1000.0 / gen_ms.max(1) as f64);
}
