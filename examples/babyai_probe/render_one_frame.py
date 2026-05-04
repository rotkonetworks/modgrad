#!/usr/bin/env python3
"""Render N BabyAI frames as 56×56 RGB and save in retina_viz's `--imagenet`
binary format.

Layout (matches `examples/retina_viz/src/main.rs::load_imagenet_bin`):
   header: u32 n, u32 h, u32 w
   payload: n × 3 × h × w f32 in [0, 1], CHW order
"""

import struct
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401  registers BabyAI envs
from minigrid.wrappers import RGBImgPartialObsWrapper


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/babyai_frames.bin")
    level = sys.argv[3] if len(sys.argv) > 3 else "BabyAI-PutNextLocal-v0"

    env = gym.make(level)
    env = RGBImgPartialObsWrapper(env, tile_size=8)

    frames = []
    for s in range(n_frames):
        obs, _ = env.reset(seed=42 + s)
        img = obs["image"]  # [56, 56, 3] uint8 HWC
        frames.append(img)
        if s == 0:
            print(f"frame[0] shape={img.shape}  mission={obs['mission']!r}")

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [N, H, W, 3]
    n, ih, iw, _ = arr.shape
    chw = arr.transpose(0, 3, 1, 2).copy()  # [N, 3, H, W]

    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", n, ih, iw))
        f.write(chw.tobytes())
    print(f"wrote {out_path}: n={n}  h={ih}  w={iw}  ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
