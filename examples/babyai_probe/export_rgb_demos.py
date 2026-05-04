#!/usr/bin/env python3
"""Export BabyAI bot demonstrations with RGB frames (NOT symbolic codes).

Runs the rule-based BabyAIBot expert on a chosen level, records each
step's egocentric RGB view (via `RGBImgPartialObsWrapper`), the
expert action, and the mission's Qwen token IDs.

Output binary layout (little-endian):

  header (32 bytes):
    magic            4B   b"BAYR"
    version          u32  = 1
    n_steps          u32
    n_missions       u32
    img_h            u32   (56 with tile_size=8)
    img_w            u32   (56)
    mission_table_off u32
    steps_off        u32

  mission table (variable-length records, packed):
    for each mission:
      n_tok          u32
      token_ids      [u32; n_tok]

  steps array (fixed-size records):
    mission_id       u32
    rgb              [u8; 3*h*w]   row-major, CHW order
    action           u8
                     → 4 + 3*h*w + 1  bytes per step
                       (= 4 + 9408 + 1 = 9413 with 56×56)

uint8 RGB (not f32) keeps the file ~4× smaller. The Rust loader
divides by 255.0 on read to land in [0, 1] which is what the
VisualCortex pipeline expects.

# One-time setup
#   git clone https://github.com/Farama-Foundation/Minigrid.git extra/Minigrid
#   python3 -m venv extra/venv
#   extra/venv/bin/pip install -e ./extra/Minigrid transformers Pillow
#   extra/venv/bin/python examples/babyai_probe/export_rgb_demos.py
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  registers BabyAI envs
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper
from minigrid.utils.baby_ai_bot import BabyAIBot
from transformers import AutoTokenizer

QWEN_SNAPSHOT = "/steam/llm/hf_cache/models--Qwen--Qwen2.5-0.5B/snapshots"
TILE_SIZE = 8  # 7×7 grid × 8 = 56×56 ego view


def find_qwen_tokenizer_dir() -> Path:
    snap = next(Path(QWEN_SNAPSHOT).glob("*/tokenizer.json"))
    return snap.parent


def collect_episode(env, bot, max_steps):
    """Run one bot-driven episode. Returns list of (rgb_uint8, action) and final reward.

    Pre: caller has already reset env and constructed bot from the same env.unwrapped.
    """
    # Get the current obs via the wrapped env (renders RGB on demand).
    rendered = env.observation(env.unwrapped.gen_obs())
    steps = []
    action = None
    for _ in range(max_steps):
        try:
            action = bot.replan(action_taken=action)
        except Exception:
            return None
        if action is None:
            return None
        steps.append((
            np.asarray(rendered["image"], dtype=np.uint8).copy(),  # [H, W, 3]
            int(action),
        ))
        rendered, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            return steps, float(reward)
    return steps, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="BabyAI-PutNextLocal-v0")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0xBAB1A1)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--out", default="/tmp/babyai_rgb_demos.bin")
    args = ap.parse_args()

    tok_dir = find_qwen_tokenizer_dir()
    print(f"loading Qwen tokenizer from {tok_dir}")
    tok = AutoTokenizer.from_pretrained(str(tok_dir))

    print(f"creating env {args.level} with RGBImgPartialObsWrapper(tile_size={TILE_SIZE})")
    env = gym.make(args.level)
    env = RGBImgPartialObsWrapper(env, tile_size=TILE_SIZE)

    rng = np.random.default_rng(args.seed)
    mission_to_id: dict[str, int] = {}
    mission_token_ids: list[list[int]] = []
    all_steps: list[tuple[int, np.ndarray, int]] = []
    img_h, img_w = None, None

    successful = 0
    skipped = 0
    rewards: list[float] = []
    for ep in range(args.episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        env.reset(seed=seed)
        bot = BabyAIBot(env.unwrapped)
        result = collect_episode(env, bot, args.max_steps)
        if result is None:
            skipped += 1
            continue
        steps, reward = result
        if reward <= 0:
            skipped += 1
            continue

        successful += 1
        rewards.append(reward)
        mission = env.unwrapped.mission
        if mission not in mission_to_id:
            ids = tok(mission, add_special_tokens=False)["input_ids"]
            if not ids:
                skipped += 1
                continue
            mission_to_id[mission] = len(mission_token_ids)
            mission_token_ids.append(ids)
        mid = mission_to_id[mission]

        for rgb, action in steps:
            if img_h is None:
                img_h, img_w = rgb.shape[0], rgb.shape[1]
            assert rgb.shape == (img_h, img_w, 3)
            all_steps.append((mid, rgb, action))

        if (ep + 1) % 100 == 0:
            print(f"  ep {ep + 1}/{args.episodes}  ok={successful}  skip={skipped}  "
                  f"missions={len(mission_token_ids)}  steps={len(all_steps)}")

    print(f"done: {successful} ok, {skipped} skipped, "
          f"{len(mission_token_ids)} missions, {len(all_steps)} steps")
    if rewards:
        print(f"  mean reward: {np.mean(rewards):.3f}")

    if not all_steps:
        print("no steps", file=sys.stderr)
        return 1

    # Show 3 mission examples
    inv = {v: k for k, v in mission_to_id.items()}
    for sid in sorted(mission_to_id.values())[:3]:
        toks = mission_token_ids[sid]
        print(f"  mission[{sid}] '{inv[sid]}'  → {len(toks)} tok  ids={toks[:6]}...")

    # ── pack binary ────────────────────────────────────────────
    img_bytes = 3 * img_h * img_w
    step_record = 4 + img_bytes + 1
    n_missions = len(mission_token_ids)
    n_steps = len(all_steps)

    mt_bytes = bytearray()
    for ids in mission_token_ids:
        mt_bytes += struct.pack("<I", len(ids))
        mt_bytes += np.asarray(ids, dtype=np.uint32).tobytes()
    HEADER = 8 + 7 * 4  # 36 bytes (magic 8 + 7 u32s)
    # Actually let me recompute: magic(4) + version(4) + n_steps(4) + n_missions(4)
    # + img_h(4) + img_w(4) + mt_off(4) + steps_off(4) = 32 bytes
    HEADER = 32
    mt_off = HEADER
    steps_off = HEADER + len(mt_bytes)
    total = steps_off + n_steps * step_record

    print(f"writing {total/1e6:.1f} MB to {args.out} "
          f"(header=32, mt={len(mt_bytes)}, steps={n_steps * step_record})")

    with open(args.out, "wb") as f:
        f.write(b"BAYR")
        f.write(struct.pack("<IIIIIII",
                            1,                # version
                            n_steps,
                            n_missions,
                            img_h, img_w,
                            mt_off,
                            steps_off))
        f.write(bytes(mt_bytes))
        # Pack steps as one big buffer
        buf = bytearray(n_steps * step_record)
        for i, (mid, rgb, action) in enumerate(all_steps):
            base = i * step_record
            buf[base:base + 4] = struct.pack("<I", mid)
            # CHW: [3, H, W]
            chw = rgb.transpose(2, 0, 1).copy()
            buf[base + 4:base + 4 + img_bytes] = chw.tobytes()
            buf[base + 4 + img_bytes] = action
        f.write(bytes(buf))
    print(f"ok ({Path(args.out).stat().st_size} bytes on disk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
