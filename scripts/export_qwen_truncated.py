#!/usr/bin/env python3
"""Export truncated Qwen2.5-0.5B — use model's own forward, intercept early layers."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import os

MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUT_DIR = "/steam/llm/qwen2.5-0.5b-onnx"
N_LAYERS = 8  # out of 24

print(f"Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32,
    cache_dir="/steam/llm/hf_cache",
)
model.eval()
hidden_dim = model.config.hidden_size
n_total = len(model.model.layers)
print(f"Full model: {n_total} layers, hidden_dim={hidden_dim}")

# Delete layers beyond N_LAYERS
model.model.layers = model.model.layers[:N_LAYERS]
model.config.num_hidden_layers = N_LAYERS
print(f"Truncated to {len(model.model.layers)} layers")

# Verify with model's own forward
dummy_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
with torch.no_grad():
    out = model(dummy_ids, output_hidden_states=True)
    # hidden_states[0] = embeddings, [1] = after layer 0, ..., [N_LAYERS] = after layer N_LAYERS-1
    # After identity layers, the remaining "hidden states" are just passthrough
    h = out.hidden_states[-1]  # output of last layer (now layer N_LAYERS-1)
    print(f"Layer {N_LAYERS} output shape: {h.shape}")
    print(f"First values: {h[0,0,:5]}")

# Now we need a wrapper that outputs this specific hidden state
class TruncatedModel(nn.Module):
    def __init__(self, model, target_layer):
        super().__init__()
        self.model = model.model  # Qwen2Model
        self.target_layer = target_layer

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, position_ids):
        out = self.model(input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, output_hidden_states=True)
        return out.hidden_states[-1]  # last layer output

truncated = TruncatedModel(model, N_LAYERS)  # target_layer unused now

# Test
dummy_mask = torch.ones(1, 4, dtype=torch.long)
dummy_pos = torch.arange(4, dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    h2 = truncated(dummy_ids, dummy_mask, dummy_pos)
    print(f"Truncated output shape: {h2.shape}")

# Export
out_path = os.path.join(OUT_DIR, f"backbone_{N_LAYERS}L.onnx")
print(f"Exporting {N_LAYERS}-layer backbone to {out_path}...")

torch.onnx.export(
    truncated, (dummy_ids, dummy_mask, dummy_pos),
    out_path,
    input_names=["input_ids", "attention_mask", "position_ids"],
    output_names=["hidden_states"],
    dynamic_axes={
        "input_ids": {1: "seq"},
        "attention_mask": {1: "seq"},
        "position_ids": {1: "seq"},
        "hidden_states": {1: "seq"},
    },
    opset_version=17,
    dynamo=False,
)

size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"Done: {out_path} ({size_mb:.1f} MB)")

# Verify with onnxruntime
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession(out_path)
ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
mask = np.ones_like(ids)
pos = np.arange(8, dtype=np.int64).reshape(1, -1)

import time
t0 = time.time()
for _ in range(10):
    result = sess.run(None, {'input_ids': ids, 'attention_mask': mask, 'position_ids': pos})
t1 = time.time()
print(f"  Verified: shape {result[0].shape}")
print(f"  Speed: {10/(t1-t0):.1f} inferences/sec (8 tokens each)")

# Compare with full model speed
full_sess = ort.InferenceSession(os.path.join(OUT_DIR, "backbone.onnx"))
t0 = time.time()
for _ in range(10):
    result2 = full_sess.run(None, {'input_ids': ids, 'attention_mask': mask, 'position_ids': pos})
t1 = time.time()
print(f"  Full model: {10/(t1-t0):.1f} inferences/sec (8 tokens each)")
