#!/usr/bin/env python3
"""Convert CIFAR-10 native (cifar-10-batches-py format) to .feat binary
that `modgrad_codec::cifar::load_feat` reads.

Format the loader expects:
    "FEAT" [4 bytes]
    u32 n_samples
    u32 n_features  (= 3*32*32 = 3072)
    u32 n_classes   (= 10)
    then n_samples × (3072 f32 pixels CHW in [0,1] + 1 f32 label)

CIFAR-10 batches-py format:
    pickle dict per file with 'data' (uint8 N×3072) and 'labels' (list[int])
    'data' is R-then-G-then-B, row-major within each channel — matches CHW.
"""

import pickle
import struct
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path("/steam/rotko/continuous-thought-machines/data/cifar-10-batches-py")
OUT_TRAIN = Path("/tmp/cifar10_native_train.feat")
OUT_EVAL = Path("/tmp/cifar10_native_eval.feat")
N_FEATURES = 3 * 32 * 32
N_CLASSES = 10


def load_batch(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"]                  # uint8 [N × 3072], CHW row-major
    labels = d[b"labels"]
    assert data.dtype == np.uint8 and data.shape[1] == N_FEATURES
    assert len(labels) == data.shape[0]
    # uint8 [0,255] → f32 [0,1]
    return data.astype(np.float32) / 255.0, np.array(labels, dtype=np.int32)


def write_feat(path: Path, pixels: np.ndarray, labels: np.ndarray):
    n = pixels.shape[0]
    with open(path, "wb") as f:
        f.write(b"FEAT")
        f.write(struct.pack("<III", n, N_FEATURES, N_CLASSES))
        # Interleaved: per sample, 3072 f32 pixels then 1 f32 label.
        # Easiest: stack column-wise and write rows.
        for i in range(n):
            f.write(pixels[i].tobytes())
            f.write(struct.pack("<f", float(labels[i])))


def main():
    if not SRC_DIR.exists():
        print(f"ERROR: source dir not found: {SRC_DIR}")
        sys.exit(1)

    # Train: data_batch_1..5
    train_pix, train_lab = [], []
    for i in range(1, 6):
        p = SRC_DIR / f"data_batch_{i}"
        pixels, labels = load_batch(p)
        train_pix.append(pixels)
        train_lab.append(labels)
    train_pix = np.concatenate(train_pix, axis=0)
    train_lab = np.concatenate(train_lab, axis=0)
    print(f"train: {train_pix.shape[0]} samples")
    write_feat(OUT_TRAIN, train_pix, train_lab)
    print(f"  → {OUT_TRAIN}")

    # Eval: test_batch
    test_pix, test_lab = load_batch(SRC_DIR / "test_batch")
    print(f"eval:  {test_pix.shape[0]} samples")
    write_feat(OUT_EVAL, test_pix, test_lab)
    print(f"  → {OUT_EVAL}")

    # Sanity: per-class counts
    print("\nclass distribution:")
    print("  train:", np.bincount(train_lab, minlength=N_CLASSES).tolist())
    print("  eval: ", np.bincount(test_lab, minlength=N_CLASSES).tolist())


if __name__ == "__main__":
    main()
