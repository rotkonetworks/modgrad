# sonotxt

**rocm tts** — Orpheus 3B + SNAC text-to-speech, 100% rust, runs on your AMD GPU.

No Python, no llama.cpp, no transformers.js. One static binary,
two model files (~2 GB total, downloaded once and cached).

## install

```bash
cargo install --git https://github.com/rotkonetworks/modgrad sonotxt
```

## use

```bash
# Generate audio from text. First run downloads the model (~2 GB).
sonotxt "Hello world" -o hello.wav

# Pick a voice (see --list-voices)
sonotxt -v tara "Hello world" -o hello.wav

# Limit how many audio tokens are emitted (84 ≈ 0.5s of audio)
sonotxt -v tara --gen 252 "A longer line of text" -o longer.wav
```

## requirements

- **AMD GPU + ROCm** for fast inference (~30 ms/token on RX 7600M XT
  with `--features rocm`). Without it falls back to CPU, ~5 s/token.
- About 2.5 GB of disk for the cached Orpheus 3B Q4_K_M GGUF + SNAC
  decoder safetensors. Cached at `~/.cache/sonotxt/`.

## what's inside

`sonotxt` chains two models, both run via the [modgrad](https://github.com/rotkonetworks/modgrad)
SDK on local hardware:

1. **Orpheus 3B Q4_K_M** (Llama-3 backbone, 28 layers, 156940 vocab
   including 28 672 SNAC audio tokens) emits audio tokens from text.
2. **SNAC 24 kHz decoder** turns those audio tokens into a 24 kHz mono
   waveform.

Architectures are described as `LlmConfig` flags and `SnacDecoder24k`
data — adding another LLM (Qwen, Mistral, future Canopy Nano) is a
config preset, not a fork.
