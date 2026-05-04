#!/usr/bin/env python3
"""Hdevalence-style sanity check: does the data have signal?

Reads the binary file written by `export_demos.py` and trains two
boring sklearn baselines:

  A1  Logistic regression on  obs one-hot       (no mission)
  A2  Logistic regression on (obs + mission_id one-hot)
  B1  MLP (32 hidden, Adam)  on  obs one-hot
  B2  MLP (32 hidden, Adam)  on (obs + mission_id one-hot)

If A2/B2 don't beat the always-Forward floor (61.1%), the data is
broken — bot demos are uninformative or our encoding is lossy. No
amount of brain tuning will help.

If A2/B2 hit 80%+, the data has signal and the brain pipeline is the
bottleneck — we know to fix the optimizer, not the architecture.

Test split: random 80/20 step-level (matches the Rust probe).
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

OBS_BYTES = 7 * 7 * 3  # 147
STEP_RECORD = 4 + 1 + OBS_BYTES + 1  # 153

N_TYPES, N_COLORS, N_STATES = 12, 6, 3
OBS_TOKEN_DIM = N_TYPES + N_COLORS + N_STATES  # 21
N_TOKENS = 49
N_MISSIONS_MAX = 512  # PutNextLocal has ~300 unique missions


def load(path: str):
    buf = Path(path).read_bytes()
    assert buf[:4] == b"BABY"
    version, n_steps, n_missions, mt_off, steps_off = struct.unpack("<IIIII", buf[4:24])
    assert version == 1
    o = mt_off
    missions = []
    for _ in range(n_missions):
        n_tok, = struct.unpack("<I", buf[o:o + 4]); o += 4
        ids = np.frombuffer(buf[o:o + n_tok * 4], dtype=np.uint32); o += n_tok * 4
        missions.append(ids)
    assert o == steps_off

    mids = np.zeros(n_steps, dtype=np.int64)
    obs  = np.zeros((n_steps, OBS_BYTES), dtype=np.uint8)
    acts = np.zeros(n_steps, dtype=np.int64)
    for i in range(n_steps):
        base = steps_off + i * STEP_RECORD
        mids[i] = struct.unpack("<I", buf[base:base + 4])[0]
        obs[i]  = np.frombuffer(buf[base + 5:base + 5 + OBS_BYTES], dtype=np.uint8)
        acts[i] = buf[base + 5 + OBS_BYTES]
    return obs, mids, acts


def one_hot_obs(obs):
    """obs: [N, 147] uint8 → [N, 49 * 21] one-hot float32."""
    n = len(obs)
    out = np.zeros((n, N_TOKENS * OBS_TOKEN_DIM), dtype=np.float32)
    for cell in range(N_TOKENS):
        t = obs[:, cell * 3 + 0].clip(max=N_TYPES - 1).astype(np.int32)
        c = obs[:, cell * 3 + 1].clip(max=N_COLORS - 1).astype(np.int32)
        s = obs[:, cell * 3 + 2].clip(max=N_STATES - 1).astype(np.int32)
        base = cell * OBS_TOKEN_DIM
        out[np.arange(n), base + t]                            = 1.0
        out[np.arange(n), base + N_TYPES + c]                  = 1.0
        out[np.arange(n), base + N_TYPES + N_COLORS + s]       = 1.0
    return out


def one_hot_mission(mids, n_missions=N_MISSIONS_MAX):
    out = np.zeros((len(mids), n_missions), dtype=np.float32)
    valid = mids < n_missions
    out[np.arange(len(mids))[valid], mids[valid]] = 1.0
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/babyai_demos.bin"
    print(f"loading {path}")
    obs, mids, acts = load(path)
    n = len(acts)
    print(f"  {n} steps  {len(set(mids.tolist()))} unique missions")
    print(f"  action histogram: {dict(Counter(acts.tolist()))}")

    floor = (acts == 2).mean()
    print(f"  always-Forward floor: {floor:.3f}")

    # Random 80/20 split
    rng = np.random.default_rng(0x5EED)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    tr, ev = idx[:split], idx[split:]

    X_obs = one_hot_obs(obs)
    X_mid = one_hot_mission(mids)
    X_full = np.hstack([X_obs, X_mid])

    print(f"\n  X_obs.shape={X_obs.shape}  X_full.shape={X_full.shape}")
    print(f"  train/eval = {len(tr)}/{len(ev)}")

    def fit_eval(name, model, X):
        model.fit(X[tr], acts[tr])
        a_tr = accuracy_score(acts[tr], model.predict(X[tr]))
        a_ev = accuracy_score(acts[ev], model.predict(X[ev]))
        print(f"  {name:<40} train={a_tr*100:5.1f}%  eval={a_ev*100:5.1f}%")
        return a_ev

    print("\n— Logistic regression (sklearn, lbfgs, C=1.0) —")
    lr_obs   = fit_eval("A1 LR (obs only)",
                        LogisticRegression(max_iter=200, n_jobs=-1), X_obs)
    lr_full  = fit_eval("A2 LR (obs + mission_id)",
                        LogisticRegression(max_iter=200, n_jobs=-1), X_full)

    print("\n— MLP (sklearn, Adam, 32 hidden, ReLU) —")
    mlp_obs  = fit_eval("B1 MLP (obs only)",
                        MLPClassifier(hidden_layer_sizes=(32,), solver="adam",
                                      max_iter=50, random_state=42, batch_size=64),
                        X_obs)
    mlp_full = fit_eval("B2 MLP (obs + mission_id)",
                        MLPClassifier(hidden_layer_sizes=(32,), solver="adam",
                                      max_iter=50, random_state=42, batch_size=64),
                        X_full)

    print("\n— Verdict —")
    print(f"  Tier-0 floor                      : {floor*100:5.1f}%")
    print(f"  A1 obs-only LR                    : {lr_obs*100:5.1f}%   (Δ vs floor: {(lr_obs-floor)*100:+.1f} pp)")
    print(f"  A2 +mission LR                    : {lr_full*100:5.1f}%   (Δ vs A1: {(lr_full-lr_obs)*100:+.1f} pp)")
    print(f"  B1 obs-only MLP                   : {mlp_obs*100:5.1f}%   (Δ vs floor: {(mlp_obs-floor)*100:+.1f} pp)")
    print(f"  B2 +mission MLP                   : {mlp_full*100:5.1f}%   (Δ vs B1: {(mlp_full-mlp_obs)*100:+.1f} pp)")
    print()
    print("  Interpretation:")
    print("    If B2 ≈ floor → data has no learnable signal; export is broken.")
    print("    If B2 ≫ floor → data learns; brain training is the bottleneck.")


if __name__ == "__main__":
    main()
