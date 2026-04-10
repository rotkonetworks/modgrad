#!/usr/bin/env python3
"""
Build a massive epimem .fb file for the parent model.

1. Reads curriculum JSONL (cypherpunk Q&A pairs)
2. Runs each through the backbone to extract hidden states + logit biases
3. Saves as episodic memory bank (JSON now, .fb later)

Usage:
    python curriculum/build_epimem.py --model models --curriculum curriculum/cypherpunk_sft.jsonl --output parent_memory.json
"""

import argparse
import json
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


def load_curriculum(path: str) -> list:
    """Load Q&A pairs from JSONL."""
    pairs = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            text = entry["text"]
            # Parse "User: ...\nAssistant: ..."
            if "User: " in text and "\nAssistant: " in text:
                parts = text.split("\nAssistant: ", 1)
                prompt = parts[0].replace("User: ", "", 1)
                answer = parts[1]
                pairs.append({
                    "prompt": prompt,
                    "answer": answer,
                    "topic": entry.get("topic", "unknown"),
                })
    return pairs


def build_memory_bank(backbone, pairs: list, verbose: bool = True) -> list:
    """Process all pairs through backbone, extract hidden states + logit biases."""
    episodes = []
    total = len(pairs)

    for i, pair in enumerate(pairs):
        prompt = pair["prompt"]
        answer = pair["answer"]

        # Extract key (hidden state of prompt)
        prompt_ids = backbone.encode(prompt)
        key = backbone.get_hidden(prompt_ids)
        key_norm = key / (np.linalg.norm(key) + 1e-8)

        # Extract logit biases (per-position)
        answer_ids = backbone.encode(answer)
        full_ids = prompt_ids + answer_ids
        full_logits = backbone.get_logits(full_ids)

        logit_biases = []
        for j, tid in enumerate(answer_ids):
            pos = len(prompt_ids) - 1 + j
            if pos < len(full_logits):
                logits_at_pos = full_logits[pos]
                target_logit = float(logits_at_pos[tid])
                max_logit = float(np.max(logits_at_pos))
                boost = max(max_logit - target_logit + 5.0, 5.0)
                logit_biases.append([int(tid), float(boost)])

        episodes.append({
            "prompt": prompt,
            "answer": answer,
            "topic": pair["topic"],
            "key": key_norm.tolist(),
            "logit_biases": logit_biases,
            "strength": 1.0,
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] processed", file=sys.stderr)

    return episodes


def save_memory_bank(episodes: list, path: str):
    """Save as JSON (FlatBuffer conversion later)."""
    with open(path, "w") as f:
        json.dump(episodes, f)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"Saved {len(episodes)} episodes to {path} ({size_mb:.1f} MB)")


def verify_recall(backbone, episodes: list, n_test: int = 5):
    """Quick verification: can we recall random episodes?"""
    import random
    test_eps = random.sample(episodes, min(n_test, len(episodes)))

    correct = 0
    for ep in test_eps:
        query_ids = backbone.encode(ep["prompt"])
        query_key = backbone.get_hidden(query_ids)
        query_norm = query_key / (np.linalg.norm(query_key) + 1e-8)

        # Find best match
        best_sim = -1
        best_ep = None
        for stored in episodes:
            sim = float(np.dot(query_norm, np.array(stored["key"])))
            if sim > best_sim:
                best_sim = sim
                best_ep = stored

        if best_ep and best_ep["prompt"] == ep["prompt"]:
            correct += 1
            status = "OK"
        else:
            status = "FAIL"

        print(f"  [{status}] \"{ep['prompt'][:50]}...\" sim={best_sim:.3f}")

    print(f"  Recall: {correct}/{len(test_eps)}")


def main():
    parser = argparse.ArgumentParser(description="Build epimem bank from curriculum")
    parser.add_argument("--model", default="models", help="Path to ONNX model dir (or 'hf' for HuggingFace)")
    parser.add_argument("--curriculum", nargs="+", required=True, help="JSONL curriculum files")
    parser.add_argument("--output", default="parent_memory.json", help="Output memory bank path")
    parser.add_argument("--verify", action="store_true", help="Run recall verification after building")
    args = parser.parse_args()

    # Load backbone
    if args.model == "hf":
        from epimem import TransformersBackbone
        backbone = TransformersBackbone("Qwen/Qwen2.5-0.5B")
    else:
        from epimem import OnnxBackbone
        backbone = OnnxBackbone(args.model)

    # Load all curriculum files
    all_pairs = []
    for path in args.curriculum:
        pairs = load_curriculum(path)
        all_pairs.extend(pairs)
        print(f"Loaded {len(pairs)} pairs from {path}")

    print(f"\nTotal: {len(all_pairs)} Q&A pairs")
    print(f"Building memory bank...")

    t0 = time.time()
    episodes = build_memory_bank(backbone, all_pairs)
    elapsed = time.time() - t0

    print(f"Built {len(episodes)} episodes in {elapsed:.1f}s ({len(episodes)/elapsed:.1f} eps/sec)")

    save_memory_bank(episodes, args.output)

    if args.verify:
        print("\nVerification:")
        verify_recall(backbone, episodes)


if __name__ == "__main__":
    main()
