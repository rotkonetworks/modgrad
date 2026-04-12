/// Test Gemma4 inference with quantized vec_dot.
use modgrad_device::kfd::{HsaDevice, dispatch_queue::GpuQueue, gguf::GgufFile, inference::Gemma4Model};

fn main() {
    let path = "/steam/llm/gemma-4-E4B-it.Q4_K_M.gguf";
    eprintln!("Loading {}...", path);

    let mut f = std::fs::File::open(path).unwrap();
    let gguf = GgufFile::parse(&mut f).unwrap();
    drop(f);

    let file = std::fs::File::open(path).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = unsafe {
        libc::mmap(std::ptr::null_mut(), file_len, libc::PROT_READ, libc::MAP_PRIVATE,
            std::os::unix::io::AsRawFd::as_raw_fd(&file), 0)
    };
    let file_data = unsafe { std::slice::from_raw_parts(mmap as *const u8, file_len) };

    let mut dev = HsaDevice::open().unwrap();
    let mut queue = GpuQueue::new();

    let mut model = Gemma4Model::load(&gguf, file_data, &dev, &mut queue, 512).unwrap();

    // Token 26352 = ▁Hello
    eprintln!("Forward pass with token 26352 (▁Hello)...");
    let logits = model.forward_token(26352, &mut dev, &mut queue);

    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let nan = logits.iter().filter(|v| v.is_nan()).count();
    eprintln!("Logits: len={} min={:.2} max={:.2} nan={}", logits.len(), min, max, nan);

    // Top 10
    let mut sorted: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("Top 10: {:?}", sorted[..10].iter().map(|(i,v)| (*i, format!("{:.2}", v))).collect::<Vec<_>>());

    unsafe { libc::munmap(mmap, file_len); }
}
