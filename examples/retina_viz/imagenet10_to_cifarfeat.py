#!/usr/bin/env python3
"""Extract 10-class ImageNet subset at 32×32, write as a .feat binary
that `modgrad_codec::cifar::load_feat` already understands.

Format the loader expects:
    "FEAT" [4 bytes]
    u32 n_samples
    u32 n_features  (= 3*32*32 = 3072)
    u32 n_classes   (= 10)
    then n_samples × (3072 f32 pixels CHW in [0,1] + 1 f32 label remapped 0..9)

Picks 10 visually distinct ImageNet classes (random list, balanced ish):
  0 tench, 7 cock, 130 flamingo, 207 golden retriever, 281 tabby cat,
  340 zebra, 385 Indian elephant, 511 convertible, 614 keyboard, 906 piggy bank.

Scans parquet shards until each picked class has at least N_TRAIN+N_EVAL
samples. Writes train.feat and eval.feat.
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
))

CLASSES = [0, 7, 130, 207, 281, 340, 385, 511, 614, 906]
CLASS_NAMES = ["tench", "cock", "flamingo", "golden_retriever", "tabby_cat",
               "zebra", "indian_elephant", "convertible", "keyboard", "piggy_bank"]

N_TRAIN_PER_CLASS = 100
N_EVAL_PER_CLASS = 20
H = W = 32

OUT_TRAIN = "/tmp/retina_imagenet10_train.feat"
OUT_EVAL  = "/tmp/retina_imagenet10_eval.feat"


def write_feat(path, pixels, labels):
    """pixels: float32 [n, 3, h, w]  labels: int [n]"""
    n = pixels.shape[0]
    n_features = pixels.shape[1] * pixels.shape[2] * pixels.shape[3]
    with open(path, "wb") as f:
        f.write(b"FEAT")
        f.write(struct.pack("<III", n, n_features, len(CLASSES)))
        for i in range(n):
            f.write(pixels[i].astype(np.float32).tobytes())
            f.write(struct.pack("<f", float(labels[i])))
    print(f"wrote {n} samples to {path} ({n_features} features, {len(CLASSES)} classes)")


def main() -> int:
    if not PARQUETS:
        print("no parquets found", file=sys.stderr); return 1
    print(f"scanning {len(PARQUETS)} parquet shards for classes {CLASSES}")

    by_class = {c: [] for c in CLASSES}
    needed = N_TRAIN_PER_CLASS + N_EVAL_PER_CLASS

    for shard_idx, path in enumerate(PARQUETS):
        if all(len(by_class[c]) >= needed for c in CLASSES):
            break
        tbl = pq.read_table(path, columns=["image", "label"])
        labels = tbl.column("label").to_pylist()
        images = tbl.column("image")
        for i, lab in enumerate(labels):
            if lab not in by_class:
                continue
            if len(by_class[lab]) >= needed:
                continue
            cell = images[i].as_py()
            data = cell["bytes"] if isinstance(cell, dict) and "bytes" in cell else cell
            try:
                im = Image.open(BytesIO(data)).convert("RGB").resize((W, H), Image.BILINEAR)
                arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,C
                by_class[lab].append(arr.transpose(2, 0, 1))    # C,H,W
            except Exception as e:
                print(f"skip shard {shard_idx} row {i}: {e}", file=sys.stderr)
        progress = {CLASS_NAMES[CLASSES.index(c)]: len(by_class[c]) for c in CLASSES}
        print(f"  shard {shard_idx}: {progress}")

    short = [c for c in CLASSES if len(by_class[c]) < needed]
    if short:
        print(f"WARNING: classes {short} have fewer than {needed} samples", file=sys.stderr)

    train_pix, train_lab, eval_pix, eval_lab = [], [], [], []
    for new_label, c in enumerate(CLASSES):
        imgs = by_class[c][:needed]
        train_imgs = imgs[:N_TRAIN_PER_CLASS]
        eval_imgs = imgs[N_TRAIN_PER_CLASS:needed]
        train_pix.extend(train_imgs); train_lab.extend([new_label] * len(train_imgs))
        eval_pix.extend(eval_imgs);   eval_lab.extend([new_label] * len(eval_imgs))

    train_pix = np.stack(train_pix).astype(np.float32)
    eval_pix  = np.stack(eval_pix).astype(np.float32)
    write_feat(OUT_TRAIN, train_pix, train_lab)
    write_feat(OUT_EVAL,  eval_pix,  eval_lab)
    return 0


if __name__ == "__main__":
    sys.exit(main())
