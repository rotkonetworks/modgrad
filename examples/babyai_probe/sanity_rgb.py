#!/usr/bin/env python3
"""Sklearn sanity check on the RGB BAYR demos.

If sklearn LR/MLP can learn (action) from raw 56×56×3 pixels, the data
has signal. If it also collapses near floor, something's wrong with
the data export. If it learns but our Rust brain doesn't, the brain
training pipeline is the bottleneck — not the cortex, not the data.
"""

from __future__ import annotations

import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def load_bayr(path: str):
    buf = Path(path).read_bytes()
    assert buf[:4] == b"BAYR"
    version, n_steps, n_missions, ih, iw, mt_off, steps_off = struct.unpack("<IIIIIII", buf[4:32])
    img_bytes = 3 * ih * iw
    step_record = 4 + img_bytes + 1
    print(f"  n_steps={n_steps} n_missions={n_missions} img={ih}×{iw}")

    mids = np.zeros(n_steps, dtype=np.int64)
    rgb  = np.zeros((n_steps, 3 * ih * iw), dtype=np.float32)
    acts = np.zeros(n_steps, dtype=np.int64)
    for i in range(n_steps):
        base = steps_off + i * step_record
        mids[i] = struct.unpack("<I", buf[base:base + 4])[0]
        rgb[i] = np.frombuffer(buf[base + 4:base + 4 + img_bytes], dtype=np.uint8).astype(np.float32) / 255.0
        acts[i] = buf[base + 4 + img_bytes]
    return rgb, mids, acts, ih, iw


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/babyai_rgb_demos.bin"
    print(f"loading {path}")
    rgb, mids, acts, ih, iw = load_bayr(path)
    n = len(acts)
    print(f"  shape={rgb.shape}  unique missions={len(set(mids.tolist()))}")
    print(f"  action histogram: {dict(Counter(acts.tolist()))}")
    floor = (acts == 2).mean()
    print(f"  always-Forward floor: {floor:.3f}")

    rng = np.random.default_rng(0x5EED)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    tr, ev = idx[:split], idx[split:]

    print(f"\n  train/eval = {len(tr)}/{len(ev)}  feat_dim={rgb.shape[1]}")

    def fit_eval(name, model, X):
        model.fit(X[tr], acts[tr])
        a_tr = accuracy_score(acts[tr], model.predict(X[tr]))
        a_ev = accuracy_score(acts[ev], model.predict(X[ev]))
        print(f"  {name:<40} train={a_tr*100:5.1f}%  eval={a_ev*100:5.1f}%")
        return a_ev

    print("\n— Logistic regression (lbfgs, C=1.0) on raw RGB pixels —")
    e_lr = fit_eval("LR(raw RGB)",
                    LogisticRegression(max_iter=200, n_jobs=-1), rgb)

    print("\n— MLP (Adam, 64 hidden, ReLU) on raw RGB pixels —")
    e_mlp = fit_eval("MLP(raw RGB, 64h)",
                     MLPClassifier(hidden_layer_sizes=(64,), solver="adam",
                                   max_iter=50, random_state=42, batch_size=64),
                     rgb)

    print("\n— Verdict —")
    print(f"  Floor (always-F)     : {floor*100:5.1f}%")
    print(f"  LR  on raw RGB       : {e_lr*100:5.1f}%   (Δ floor: {(e_lr-floor)*100:+.1f} pp)")
    print(f"  MLP on raw RGB       : {e_mlp*100:5.1f}%   (Δ floor: {(e_mlp-floor)*100:+.1f} pp)")
    print()
    if e_mlp > floor + 5:
        print(f"  → Data has signal. If our Rust brain doesn't beat floor, brain training is the bottleneck.")
    else:
        print(f"  → Data near floor. Investigate the export pipeline before more brain tuning.")


if __name__ == "__main__":
    main()
