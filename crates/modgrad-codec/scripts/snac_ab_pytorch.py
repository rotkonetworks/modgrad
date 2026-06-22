#!/usr/bin/env python3
"""snac_ab_pytorch.py — decode the SAME test codes as snac_ab_modgrad through
PyTorch SNAC. Dumps raw f32 + WAV for bit-level comparison vs modgrad output.

Run on bkk07:
  cd /root/norpheus && venv/bin/python snac_ab_pytorch.py
"""
import struct
import sys
import wave

import numpy as np
import torch
from snac import SNAC

# Same codes as snac_ab_modgrad.rs
codes_0 = [100, 250, 500, 1000]
codes_1 = [10, 20, 30, 40, 50, 60, 70, 80]
codes_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

print("loading SNAC...", flush=True)
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

print("decoding...", flush=True)
codes = [
    torch.tensor([codes_0], dtype=torch.int32),
    torch.tensor([codes_1], dtype=torch.int32),
    torch.tensor([codes_2], dtype=torch.int32),
]
with torch.inference_mode():
    audio = model.decode(codes)

# audio shape [1, 1, T]
samples = audio[0, 0].cpu().numpy().astype(np.float32)
print(f"PyTorch output: {len(samples)} samples, max|x|={np.max(np.abs(samples)):.4f}")

# Dump raw f32
with open("/root/norpheus/snac_ab_pytorch.f32", "wb") as f:
    f.write(samples.tobytes())

# Dump WAV
clamped = np.clip(samples, -1.0, 1.0)
pcm16 = (clamped * 32767).astype(np.int16)
with wave.open("/root/norpheus/snac_ab_pytorch.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(24000)
    w.writeframes(pcm16.tobytes())

print("wrote /root/norpheus/snac_ab_pytorch.wav + .f32")
