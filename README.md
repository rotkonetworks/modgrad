# modgrad

**modular gradient SDK for building general intelligence**

composable rust crates for building brains. you pick the architecture — we provide the primitives: continuous thought machines with full BPTT, graph composition, multimodal codecs, bio-inspired learning, GPU dispatch, and a live 3D debugger.

## build your own brain

```toml
# your Cargo.toml
[dependencies]
modgrad-ctm = { git = "https://github.com/rotkonetworks/modgrad" }
modgrad-compute = { git = "https://github.com/rotkonetworks/modgrad" }
modgrad-training = { git = "https://github.com/rotkonetworks/modgrad" }
```

```rust
use modgrad_ctm::graph::*;

// define your own topology — any number of regions, any connections
let cfg = CtmGraphConfig::new(vec![
    CtmConfig::region("sensory", 128, 8, 4),   // 128 neurons, memory=8, ticks=4
    CtmConfig::region("reasoning", 256, 16, 8), // bigger, deeper memory, more ticks
    CtmConfig::region("motor", 64, 4, 2),       // small, fast
], vec![
    Connection { from: vec![2], to: 0, receives_observation: true },
    Connection { from: vec![0], to: 1, receives_observation: false },
    Connection { from: vec![1], to: 2, receives_observation: false },
], 256); // vocab size

let mut w = RegionalWeights::new(cfg);
let mut opt = RegionalAdamW::new(&w).with_lr(3e-4);

// train on your data
let mut grads = RegionalGradients::zeros(&w);
for (token, target) in your_data {
    let (loss, _pred) = regional_train_token(&w, &mut grads, token, target);
    opt.step(&mut w, &grads);
    grads.zero();
}

// run as a neural computer
let mut nc = NeuralComputer::new(w);
let response = nc.chat("hello", 100, 0.8);
```

no frameworks. no config files. just rust functions you compose however you want.

## SDK crates

the building blocks — use any of them independently:

| crate | what it gives you |
|-------|-------------------|
| **modgrad-ctm** | single CTM (NLM traces, sync, MHA, U-Net synapse, full BPTT) + graph composition (N CTMs in a directed graph, embedding table, AdamW, NeuralComputer) |
| **modgrad-compute** | `Linear`, ops, tensor, GPU batched dispatch |
| **modgrad-codec** | VQ-VAE (images → 4096 codes), AudioCodec (speech → 75 codes/sec), FSQ, n-gram hash |
| **modgrad-data** | type-safe multimodal tokenization, mixed-modality streaming, lazy data loading |
| **modgrad-device** | CPU / CUDA / AMD KFD backend abstraction |
| **modgrad-training** | AdamW, Adam, SGD optimizers + warmup/cosine schedulers |
| **modgrad-persist** | bincode/JSON save/load, quantization (f32/f16/i8) |
| **modgrad-transformer** | transformer blocks, MHA, RoPE, KV cache (for hybrid CTM+transformer models) |
| **modgrad-traits** | core traits (`Brain`, `TokenInput`) |
| **modgrad-io** | telemetry streaming, flatbuffers, ONNX/GGUF backends |

### bio-inspired modules (in modgrad-ctm)

optional, toggleable — use them as auxiliary signals or ignore them:

| module | what |
|--------|------|
| `bio::cerebellar` | delta rule forward model + dopamine dynamics |
| `bio::three_factor` | REINFORCE with Titans-style eligibility traces |
| `bio::neuromod` | dopamine / serotonin / norepinephrine state machine |
| `bio::salience` | RPE × motor conflict → learning rate gate |
| `bio::homeostasis` | self-monitoring: sleep pressure, zone detection |
| `bio::consolidation` | SPSA spindle-ripple offline weight optimization |
| `memory::hippocampus` | content-addressable episodic memory (cosine retrieval) |
| `memory::replay` | prioritized experience buffer (surprise-gated) |
| `memory::sleep` | offline least-squares weight consolidation |

### multimodal token space

unified vocabulary — one model, all modalities:

```
  0..255        bytes (text)
  256..263      delimiters (<img> </img> <aud> </aud> <vid> </vid>)
  264..4359     image VQ codes (4096)
  4360..8455    audio VQ codes (4096)
  8456..8855    timestamps (0.5s resolution)
  8856..9133    action tokens (mouse, keyboard, coordinates)
```

## isis

our runtime built on the SDK. 8 brain regions, multimodal, neural computer mode. you don't need isis to use modgrad — it's just one composition.

```bash
# train
isis train model.bin
isis train model.bin --multimodal --images cifar.bin --audio clips/

# interactive neural computer
isis nc model.bin
isis nc model.bin --audio mic.wav --camera frames/ --debug-port 4747

# generate
isis generate model.bin --prompt "the cat "

# run as a service
isis daemon model.bin --port 4747
isis send "hello world" --addr 127.0.0.1:4747

# show devices
isis devices
```

### isis brain regions

| region | neurons | memory | role |
|--------|---------|--------|------|
| input | 64 | 4 | perception + motor feedback |
| attention | 64 | 8 | gating, routing |
| output | 64 | 16 | evidence accumulation |
| motor | 64 | 4 | action selection |
| cerebellum | 8 | 4 | forward model |
| basal ganglia | 8 | 8 | value estimation |
| insula | 8 | 4 | interoception |
| hippocampus | 8 | 16 | episodic binding |

## minictm

nanoGPT but for CTMs. minimal example — uses the SDK directly, no isis:

```bash
cargo run -p minictm --release -- --data train.txt --steps 5000
cargo run -p minictm --release -- --data train.txt --steps 5000 --chat
```

## debugger

live 3D brain visualizer. connects to any running modgrad model via TCP:

```bash
# connect to isis or any NC with --debug-port
modgrad-debugger 127.0.0.1:4747
```

- 3D neuron particles colored by region, sized by activation
- token stream color-coded by modality (text/image/audio/action)
- NLM trace heatmaps per region
- global sync visualization
- command center: pause/resume/step, inject tokens, inspect state

## building

```bash
cargo build --release                      # CPU
cargo build --release --features cuda      # NVIDIA GPU
cargo test --release                       # 23+ tests
```

requires rust 2024 edition.

## references

- sakana AI CTM (arxiv 2505.05522) — continuous thought machine
- qwen3-VL (2025) — text timestamps for video
- meta neural computers (2026) — the model as the running computer
- chameleon (meta) — unified discrete token space for multimodal generation

## license

MIT
