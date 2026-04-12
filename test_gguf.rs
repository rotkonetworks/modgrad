/// Test GGUF parser on real model file + print architecture summary.
use modgrad_device::kfd::gguf::GgufFile;

fn main() {
    let path = "/steam/llm/gemma-4-E4B-it.Q4_K_M.gguf";
    eprintln!("Parsing {}...", path);

    let mut f = std::fs::File::open(path).unwrap();
    let gguf = GgufFile::parse(&mut f).unwrap();

    let arch = gguf.architecture().unwrap_or("unknown");
    println!("Architecture: {}", arch);
    println!("Tensors: {}", gguf.tensors.len());
    println!("Data offset: 0x{:x}\n", gguf.data_offset);

    // Print all architecture-related metadata
    let prefix = format!("{}.", arch);
    let mut keys: Vec<_> = gguf.metadata.keys()
        .filter(|k| k.starts_with(&prefix) || k.starts_with("general."))
        .cloned().collect();
    keys.sort();
    println!("Metadata:");
    for key in &keys {
        let val = gguf.meta(key).unwrap();
        match val {
            modgrad_device::kfd::gguf::MetaValue::Str(s) => println!("  {} = \"{}\"", key, s),
            modgrad_device::kfd::gguf::MetaValue::U32(v) => println!("  {} = {}", key, v),
            modgrad_device::kfd::gguf::MetaValue::I32(v) => println!("  {} = {}", key, v),
            modgrad_device::kfd::gguf::MetaValue::F32(v) => println!("  {} = {}", key, v),
            modgrad_device::kfd::gguf::MetaValue::Bool(v) => println!("  {} = {}", key, v),
            modgrad_device::kfd::gguf::MetaValue::Array(a) => println!("  {} = [{} items]", key, a.len()),
            _ => println!("  {} = {:?}", key, val),
        }
    }

    // Print first 30 tensors with sizes
    println!("\nTensors (first 30):");
    let total_bytes: usize = gguf.tensor_list.iter()
        .map(|name| gguf.tensors[name].data_bytes())
        .sum();
    for (i, name) in gguf.tensor_list.iter().enumerate().take(30) {
        let t = &gguf.tensors[name];
        println!("  {:3}. {:50} {:?} {:?} ({:.2} MB)",
            i, name, t.dims, t.dtype, t.data_bytes() as f64 / 1e6);
    }
    println!("  ...");
    println!("\nTotal tensor data: {:.1} GB", total_bytes as f64 / 1e9);

    // Count layers
    let n_layers = gguf.tensor_list.iter()
        .filter(|n| n.starts_with("blk."))
        .map(|n| n.split('.').nth(1).unwrap_or("0").parse::<usize>().unwrap_or(0))
        .max().map(|m| m + 1).unwrap_or(0);
    println!("Layers: {}", n_layers);

    // Weight shapes per layer (layer 0)
    println!("\nLayer 0 weights:");
    for name in &gguf.tensor_list {
        if name.starts_with("blk.0.") {
            let t = &gguf.tensors[name];
            let short = name.strip_prefix("blk.0.").unwrap();
            println!("  {:30} {:?} {:?} {:.2} MB", short, t.dims, t.dtype, t.data_bytes() as f64 / 1e6);
        }
    }
}
