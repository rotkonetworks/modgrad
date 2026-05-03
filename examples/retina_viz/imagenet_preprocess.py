#!/usr/bin/env python3
"""Extract a few RGB ImageNet samples at 224x224 for retina_viz.

Reads from /steam/hf-datasets/datasets--ILSVRC--imagenet-1k/, decodes
the first N RGB images, resizes to 224x224, and writes a binary:

  u32 n   — number of images
  u32 h   — 224
  u32 w   — 224
  then n * 3 * h * w f32 RGB pixels in [0, 1], channel-first (CHW).

CHW layout matches VisualCortex's expected `pixels` slice (Conv2d
expects [n × in_ch × h × w] contiguous).
"""

import struct
import sys
import glob
from io import BytesIO

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

PARQUETS = sorted(glob.glob(
    "/steam/hf-datasets/datasets--ILSVRC--imagenet-1k/snapshots/*/data/train-*.parquet"
))[:1]

OUT = "/tmp/retina_imagenet.bin"
N = 8
H = W = 224

def main() -> int:
    if not PARQUETS:
        print("no parquets found; check /steam/hf-datasets path", file=sys.stderr)
        return 1
    print(f"source: {PARQUETS[0]}")
    out = np.zeros((N, 3, H, W), dtype=np.float32)
    ok = 0
    tbl = pq.read_table(PARQUETS[0], columns=["image"])
    col = tbl.column("image")
    for i in range(len(col)):
        if ok >= N:
            break
        cell = col[i].as_py()
        data = cell["bytes"] if isinstance(cell, dict) and "bytes" in cell else cell
        try:
            im = Image.open(BytesIO(data)).convert("RGB").resize((W, H), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0  # H, W, C
            out[ok] = arr.transpose(2, 0, 1)                # C, H, W
            ok += 1
        except Exception as e:
            print(f"skip row {i}: {e}", file=sys.stderr)
    out = out[:ok]
    with open(OUT, "wb") as f:
        f.write(struct.pack("<III", ok, H, W))
        f.write(out.tobytes())
    print(f"wrote {ok} images to {OUT} ({out.nbytes / 1e6:.1f} MB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
