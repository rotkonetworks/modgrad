#!/usr/bin/env python3
"""Generate synthetic QEC syndromes for CTM decoder testing.

Repetition code with bit-flip noise:
- n data qubits in a line
- n-1 syndrome qubits between them
- Each data qubit flips with probability p
- Syndrome bit = XOR of adjacent data qubits (detects boundary of error)
- Label = parity of errors (0 = even errors, 1 = odd errors)

This is the simplest QEC task. If the CTM can learn this,
it can learn surface codes.

Output: JSON lines, each line = {"syndrome": [...], "label": int}
"""

import json
import random
import sys

def generate_sample(n_data: int, p: float) -> dict:
    """Generate one syndrome + label pair."""
    # Random bit-flip errors
    errors = [1 if random.random() < p else 0 for _ in range(n_data)]

    # Syndrome: XOR of adjacent data qubits
    syndrome = []
    for i in range(n_data - 1):
        syndrome.append(errors[i] ^ errors[i + 1])

    # Label: parity of total errors (0 = correctable, 1 = logical error)
    label = sum(errors) % 2

    return {
        "syndrome": syndrome,
        "label": label,
        "errors": errors,  # for debugging
    }


def generate_dataset(n_data: int, p: float, n_samples: int, path: str):
    """Generate a dataset of syndrome/label pairs."""
    with open(path, "w") as f:
        for _ in range(n_samples):
            sample = generate_sample(n_data, p)
            json.dump({"syndrome": sample["syndrome"], "label": sample["label"]}, f)
            f.write("\n")

    # Stats
    with open(path) as f:
        lines = f.readlines()
    labels = [json.loads(l)["label"] for l in lines]
    n_pos = sum(labels)
    print(f"Generated {len(lines)} samples: {n_pos} positive ({100*n_pos/len(lines):.1f}%), "
          f"{len(lines)-n_pos} negative ({100*(len(lines)-n_pos)/len(lines):.1f}%)")
    print(f"  n_data={n_data}, p={p}, syndrome_dim={n_data-1}")
    print(f"  Saved to {path}")


if __name__ == "__main__":
    # d=5 repetition code at p=0.05 (matches the paper's setting)
    n_data = 5
    p = 0.05
    n_train = 10000
    n_test = 2000

    generate_dataset(n_data, p, n_train, "data/qec/train.jsonl")
    generate_dataset(n_data, p, n_test, "data/qec/test.jsonl")

    # Also generate harder versions
    generate_dataset(11, 0.05, 10000, "data/qec/train_d11.jsonl")
    generate_dataset(11, 0.05, 2000, "data/qec/test_d11.jsonl")
    generate_dataset(21, 0.05, 10000, "data/qec/train_d21.jsonl")
    generate_dataset(21, 0.05, 2000, "data/qec/test_d21.jsonl")
