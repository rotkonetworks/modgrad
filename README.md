# modgrad

**modular gradient SDK for building general intelligence**

a hierarchical continuous thought machine trained with backpropagation through time. 8 brain regions, each a full CTM, connected in a directed graph. multimodal: text, images, audio, video, actions — unified in one token space, one model, one loss function.

## architecture

```
observation (any modality)
  → embedding table (9134 tokens)
  → inter-region synapse projections
  → 8 CTM regions (parallel, different tick rates):
      input ←→ attention ←→ output ←→ motor    (cortical loop)
      cerebellum, basal_ganglia, insula, hippocampus  (subcortical)
  → global sync accumulator (spans all regions)
  → output projection → next token prediction
  → BPTT through all ticks, all regions, all connections
  → AdamW optimizer
```

each region runs the full sakana CTM algorithm:
- multihead attention over observation
- synapse U-Net (residual skip connections)
- NLM trace memory (shift register, variable depth per region)
- sync accumulator readout (random neuron pairing)
- variable-depth thinking (ticks with early exit on certainty)

## token space

```
  0..255        bytes (text, raw data)
  256..263      delimiters (<img> </img> <aud> </aud> <vid> </vid>)
  264..4359     image VQ codes (4096, from VQ-VAE encoder)
  4360..8455    audio VQ codes (4096, from WavTokenizer-style codec)
  8456..8855    timestamps (0.00s..199.50s at 0.5s resolution)
  8856..9133    action tokens (mouse/keyboard events + coordinates)
```

one model learns all modalities through the same next-token prediction loss.

## neural computer

the model is the computer. computation, memory, and I/O unified in the CTM's latent state.

```bash
# interactive mode
isis nc model.bin
isis nc model.bin --audio mic.wav --camera frames/ --audio-out speak.wav

# commands
nc> the cat sat on        # text in → text out
nc> /click 0.5 0.3        # mouse action
nc> /key enter             # keyboard action
nc> /ctrl c                # modifier combo
```

## training

```bash
# text-only (learns English byte prediction)
isis train model.bin

# multimodal (text + synthetic image/audio/action pairs)
isis train model.bin --multimodal

# with real data
isis train model.bin --multimodal --images cifar.bin --audio clips/ --video vids/

# watch training live
isis train model.bin --debug-port 4747
modgrad-debugger 127.0.0.1:4747
```

training uses BPTT through the full CTM tick loop + AdamW. no hebbian, no sleep consolidation, no biological learning rules — pure gradient descent. optional auxiliary losses (cerebellar prediction, hippocampal contrastive, BG temporal difference) can be toggled for experimentation.

## debugger

live 3D brain visualization over TCP:

- neuron particles colored by region, sized by activation
- token stream color-coded by modality
- NLM trace heatmaps per region
- global sync bar chart
- command center: pause/resume/step, inject tokens, inspect state
- auto-reconnect, works with any modgrad model

```bash
modgrad-debugger 127.0.0.1:4747
```

## workspace

14 crates:

| crate | what |
|-------|------|
| `modgrad-traits` | core traits (`Brain`, `TokenInput`) |
| `modgrad-compute` | `Linear`, ops, tensor, GPU dispatch |
| `modgrad-ctm` | continuous thought machine with full BPTT |
| `modgrad-codec` | VQ-VAE (images), AudioCodec (speech), FSQ |
| `modgrad-data` | tokenization, mixed-modality streaming |
| `modgrad-device` | CPU / CUDA / AMD KFD backends |
| `modgrad-io` | telemetry, flatbuffers, ONNX/GGUF |
| `modgrad-persist` | bincode/JSON serialization, quantization |
| `modgrad-training` | AdamW, Adam, SGD, schedulers |
| `modgrad-transformer` | transformer blocks, attention, RoPE |
| `modgrad-runtime` | isis: 8-region hierarchical CTM, NC, debug socket |
| `nanochat-rs` | lightweight chat inference |
| `debugger` | live 3D brain visualizer (egui + TCP) |
| `remu` | RDNA3 GPU emulator for kernel testing |

## building

```bash
cargo build --release                      # CPU only
cargo build --release --features cuda      # with NVIDIA GPU
cargo test --release                       # run tests
```

requires rust 2024 edition.

## brain regions

| region | d_model | memory | role |
|--------|---------|--------|------|
| input | 64 | 4 | raw observation + motor feedback |
| attention | 64 | 8 | gating, routing |
| output | 64 | 16 | evidence accumulation |
| motor | 64 | 4 | action selection |
| cerebellum | 8 | 4 | forward model (prediction error) |
| basal ganglia | 8 | 8 | value estimation (critic) |
| insula | 8 | 4 | interoception |
| hippocampus | 8 | 16 | episodic binding (long memory) |

## references

- sakana AI CTM (arxiv 2505.05522) — continuous thought machine
- qwen3-VL — text timestamps for video, interleaved MRoPE
- meta neural computers (2026) — the model as the running computer
- chameleon (meta) — unified discrete token space for multimodal generation

## license

MIT
