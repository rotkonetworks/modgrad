#!/usr/bin/env python3
"""Pre-extract natural image patches for the Hebbian sanity check.

Reads tiny-imagenet parquet, converts to grayscale 64x64 f32 in [0,1],
writes a tiny binary: u32 n_images, u32 h, u32 w, then raw f32 pixels.
"""
import struct
import sys
from io import BytesIO
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

import glob
PARQUETS = sorted(glob.glob(
    "/steam/hf-datasets/datasets--ILSVRC--imagenet-1k/snapshots/*/data/train-*.parquet"
))[:8]   # 8 shards × ~8-9k images each ≈ 65-70k images
OUT     = "/tmp/hebbian_sanity_images.bin"
N       = 50000
H = W   = 64

def main():
    print(f"sources: {len(PARQUETS)} parquet shards")
    out = np.zeros((N, H, W), dtype=np.float32)
    ok = 0
    for p_idx, path in enumerate(PARQUETS):
        if ok >= N:
            break
        tbl = pq.read_table(path, columns=["image"])
        col = tbl.column("image")
        print(f"  shard {p_idx}: {len(col)} rows  (have {ok}/{N})")
        for i in range(len(col)):
            if ok >= N:
                break
            cell = col[i].as_py()
            if isinstance(cell, dict) and "bytes" in cell:
                data = cell["bytes"]
            else:
                data = cell
            try:
                im = Image.open(BytesIO(data)).convert("L").resize((W, H), Image.BILINEAR)
                out[ok] = np.asarray(im, dtype=np.float32) / 255.0
                ok += 1
            except Exception as e:
                if ok < 10:
                    print(f"skip {p_idx}:{i}: {e}", file=sys.stderr)

    out = out[:ok]
    print(f"decoded {ok} images → {out.shape}, mean={out.mean():.3f}, std={out.std():.3f}")

    with open(OUT, "wb") as f:
        f.write(struct.pack("<III", ok, H, W))
        out.tofile(f)
    print(f"wrote {OUT} ({ok*H*W*4 + 12} bytes)")

if __name__ == "__main__":
    main()
