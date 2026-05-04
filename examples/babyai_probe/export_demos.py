#!/usr/bin/env python3
"""Export BabyAI bot demonstrations to a flat binary file.

Plays N episodes of a BabyAI level using the rule-based BabyAIBot
expert (ships with `minigrid`). For each step it records:

  - mission token IDs (Qwen2.5 tokenizer, looked up once per mission)
  - 7x7x3 symbolic observation (uint8: type, color, state)
  - agent direction (0..3)
  - expert action (0..6)

Output binary layout (all little-endian):

  header:
    magic           4B   b"BABY"
    version         u32  = 1
    n_steps         u32
    n_missions      u32
    mission_table_off u32  byte offset of mission table
    steps_off       u32  byte offset of steps array

  mission table (variable length, packed):
    for each of n_missions:
      n_tok         u32
      token_ids     [u32; n_tok]

  steps array (fixed-size records, n_steps total):
    mission_id      u32   index into mission table
    direction       u8    0..3
    obs             [u8; 147]  flat 7x7x3 row-major (h, w, c)
    action          u8    0..6
                          → 153 bytes per step

The Rust loader reads header → mission table → steps array.
Variable-length missions are deduplicated so the steps array stays
fixed-stride for fast random access.

Why use Qwen's tokenizer here (Python-side) rather than ship raw
strings: we already use `transformers.AutoTokenizer` for cifar10_probe
and `qwen_class_embeddings.py`. Token IDs at the SDK boundary keep the
Rust side free of tokenizer machinery (modgrad has no Qwen tokenizer
crate today).

# One-time setup
#   git clone https://github.com/Farama-Foundation/Minigrid.git extra/Minigrid
#   python3 -m venv extra/venv
#   extra/venv/bin/pip install -e ./extra/Minigrid transformers
#   extra/venv/bin/python examples/babyai_probe/export_demos.py
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  registers BabyAI envs
import numpy as np
from minigrid.utils.baby_ai_bot import BabyAIBot
from transformers import AutoTokenizer

QWEN_SNAPSHOT = "/steam/llm/hf_cache/models--Qwen--Qwen2.5-0.5B/snapshots"


def find_qwen_tokenizer_dir() -> Path:
    snap = next(Path(QWEN_SNAPSHOT).glob("*/tokenizer.json"))
    return snap.parent


def collect_episode(env, bot: BabyAIBot, max_steps: int):
    """Run one bot-driven episode. Returns list of (obs, dir, action) and final reward.

    Pre: caller has already reset `env` and constructed `bot` from the
    same `env.unwrapped`. We do NOT reset here — that would desync the
    bot's plan from the env state.

    Returns None if the bot fails to plan an action (rare; we skip the episode).
    """
    obs = {
        "image": env.unwrapped.gen_obs()["image"],
        "direction": env.unwrapped.agent_dir,
    }
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
            np.asarray(obs["image"], dtype=np.uint8).copy(),
            int(obs["direction"]),
            int(action),
        ))
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            return steps, float(reward)
    return steps, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="BabyAI-GoToLocal-v0",
                    help="MiniGrid env id (default: BabyAI-GoToLocal-v0)")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0xBAB1A1)
    ap.add_argument("--max-steps", type=int, default=64,
                    help="cap per-episode step count (env may be larger)")
    ap.add_argument("--out", default="/tmp/babyai_demos.bin")
    args = ap.parse_args()

    tok_dir = find_qwen_tokenizer_dir()
    print(f"loading Qwen tokenizer from {tok_dir}")
    tok = AutoTokenizer.from_pretrained(str(tok_dir))

    print(f"creating env {args.level}")
    env = gym.make(args.level)

    rng = np.random.default_rng(args.seed)
    mission_to_id: dict[str, int] = {}
    mission_token_ids: list[list[int]] = []
    all_steps: list[tuple[int, int, np.ndarray, int]] = []  # (mid, dir, obs, action)

    successful = 0
    skipped = 0
    rewards = []
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
        # Mission string is stable across the episode; pull from env.
        mission = env.unwrapped.mission
        if mission not in mission_to_id:
            ids = tok(mission, add_special_tokens=False)["input_ids"]
            if not ids:
                skipped += 1
                continue
            mission_to_id[mission] = len(mission_token_ids)
            mission_token_ids.append(ids)
        mid = mission_to_id[mission]
        for obs, direction, action in steps:
            all_steps.append((mid, direction, obs, action))
        if (ep + 1) % 200 == 0:
            print(f"  ep {ep + 1}/{args.episodes}  ok={successful}  skip={skipped}  "
                  f"missions={len(mission_token_ids)}  steps={len(all_steps)}")

    print(f"done: {successful} successful, {skipped} skipped, "
          f"{len(mission_token_ids)} unique missions, {len(all_steps)} steps")
    if rewards:
        print(f"  mean reward: {np.mean(rewards):.3f}  min {min(rewards):.3f}  max {max(rewards):.3f}")

    if not all_steps:
        print("no steps collected — aborting", file=sys.stderr)
        return 1

    # Show a couple of mission examples
    inv = {v: k for k, v in mission_to_id.items()}
    sample_ids = sorted(mission_to_id.values())[:3]
    for sid in sample_ids:
        toks = mission_token_ids[sid]
        print(f"  mission[{sid}] '{inv[sid]}'  → {len(toks)} tok  ids={toks}")

    # ── pack binary ───────────────────────────────────────────
    OBS_BYTES = 7 * 7 * 3
    STEP_RECORD = 4 + 1 + OBS_BYTES + 1  # 153
    n_missions = len(mission_token_ids)
    n_steps = len(all_steps)

    # mission table size
    mt_size = 4  # n_missions u32 — actually we put n_missions in header, table is just records
    mt_bytes = bytearray()
    for ids in mission_token_ids:
        mt_bytes += struct.pack("<I", len(ids))
        mt_bytes += np.asarray(ids, dtype=np.uint32).tobytes()
    HEADER = 4 + 4 + 4 + 4 + 4 + 4  # 24 bytes
    mission_table_off = HEADER
    steps_off = HEADER + len(mt_bytes)
    total = steps_off + n_steps * STEP_RECORD

    print(f"writing {total} bytes to {args.out} "
          f"(header={HEADER}, mission_table={len(mt_bytes)}, steps={n_steps * STEP_RECORD})")
    out = open(args.out, "wb")
    out.write(b"BABY")
    out.write(struct.pack("<IIIII",
                          1,                # version
                          n_steps,
                          n_missions,
                          mission_table_off,
                          steps_off))
    out.write(bytes(mt_bytes))
    # Steps: pack as one big buffer for speed.
    step_buf = bytearray(n_steps * STEP_RECORD)
    for i, (mid, direction, obs, action) in enumerate(all_steps):
        base = i * STEP_RECORD
        step_buf[base:base + 4] = struct.pack("<I", mid)
        step_buf[base + 4] = direction
        step_buf[base + 5:base + 5 + OBS_BYTES] = obs.tobytes()
        step_buf[base + 5 + OBS_BYTES] = action
    out.write(bytes(step_buf))
    out.close()
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
