#!/usr/bin/env python3
"""Extract Qwen2.5-0.5B's input embeddings for the 10 ImageNet class
names used by cifar10_probe. Tokenises each name, mean-pools the
per-token input embeddings, writes a flat binary [10 × model_dim f32]
to /tmp/qwen10_class_embed.bin.

Used by cifar10_probe's `qwen-clip` mode: train a 128→model_dim linear
projection from V4 features to Qwen embedding space, predict class by
nearest-neighbour cosine in that space.

Doesn't run Qwen — just reads the embedding table tensor from the
safetensors file directly. ~600 MB → 9 KB output.
"""

import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

SNAPSHOT = "/steam/llm/hf_cache/models--Qwen--Qwen2.5-0.5B/snapshots"
TOK_PATH = next(Path(SNAPSHOT).glob("*/tokenizer.json"))
ST_PATH  = next(Path(SNAPSHOT).glob("*/model.safetensors"))

CLASS_NAMES = [
    "tench", "cock", "flamingo", "golden retriever", "tabby cat",
    "zebra", "indian elephant", "convertible", "keyboard", "piggy bank",
]

OUT = "/tmp/qwen10_class_embed.bin"


def main() -> int:
    print(f"loading tokenizer from {TOK_PATH.parent}")
    tok = AutoTokenizer.from_pretrained(str(TOK_PATH.parent))

    print(f"opening safetensors {ST_PATH.name}")
    with safe_open(str(ST_PATH), framework="pt") as f:
        embed_t = f.get_tensor("model.embed_tokens.weight")
    embed = embed_t.float().numpy()  # [vocab, model_dim]
    vocab, dim = embed.shape
    print(f"  embed: vocab={vocab}, dim={dim}")

    out = np.zeros((len(CLASS_NAMES), dim), dtype=np.float32)
    for i, name in enumerate(CLASS_NAMES):
        ids = tok(name, add_special_tokens=False)["input_ids"]
        if not ids:
            print(f"  WARN: empty tokenisation for '{name}'", file=sys.stderr)
            continue
        vecs = embed[ids]                        # [n_tok, dim]
        out[i] = vecs.mean(axis=0)               # mean-pool
        print(f"  {i:>2} '{name}'  → {len(ids)} tok(s)  ids={ids}")

    with open(OUT, "wb") as f:
        f.write(struct.pack("<II", len(CLASS_NAMES), dim))
        f.write(out.tobytes())
    print(f"wrote {len(CLASS_NAMES)} class embeddings of dim {dim} to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
