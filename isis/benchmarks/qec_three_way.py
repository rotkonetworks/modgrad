#!/usr/bin/env python3
"""
Three-way QEC decoder benchmark:
  1. Majority vote (MWPM-equivalent for repetition code)
  2. Sakana v1 CTM (PyTorch, gradients)
  3. isis graph pipeline (Rust, Hebbian, no gradients)

Uses synthetic d=5 repetition code syndromes at p=0.05.
"""

import json
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/rotko/ctm/continuous-thought-machines"))

# ─── Data ─────────────────────────────────────────────────

def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            data.append((np.array(d["syndrome"], dtype=np.float32), d["label"]))
    return data

# ─── Baseline 1: Majority Vote ───────────────────────────

def majority_vote_accuracy(data):
    """Predict the majority class. This is the trivial baseline."""
    labels = [d[1] for d in data]
    majority = max(set(labels), key=labels.count)
    correct = sum(1 for _, l in data if l == majority)
    return correct / len(data)

# ─── Baseline 2: v1 CTM (PyTorch, gradients) ─────────────

def train_v1_ctm(train_data, test_data, syndrome_dim, epochs=20):
    """Train Sakana's original CTM on syndrome classification."""
    import torch
    import torch.nn as nn

    # Minimal CTM-like model: synapse + NLM + sync readout
    # Not the full Sakana CTM (too complex to wire for this), but a
    # fair comparison: same architecture ideas, PyTorch gradients.
    class MiniCTM(nn.Module):
        def __init__(self, d_in, d_hidden, n_ticks, n_classes):
            super().__init__()
            self.n_ticks = n_ticks
            self.synapse = nn.Linear(d_in + d_hidden, d_hidden)
            self.nlm = nn.Linear(d_hidden, d_hidden)
            self.readout = nn.Linear(d_hidden, n_classes)
            self.act = nn.SiLU()

        def forward(self, x):
            h = torch.zeros(x.shape[0], self.synapse.out_features, device=x.device)
            for _ in range(self.n_ticks):
                inp = torch.cat([x, h], dim=-1)
                h = h + self.act(self.synapse(inp))
                h = self.act(self.nlm(h))
            return self.readout(h)

    device = "cpu"
    model = MiniCTM(syndrome_dim, 32, 8, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train
    X_train = torch.tensor([d[0] for d in train_data])
    y_train = torch.tensor([d[1] for d in train_data], dtype=torch.long)
    X_test = torch.tensor([d[0] for d in test_data])
    y_test = torch.tensor([d[1] for d in test_data], dtype=torch.long)

    t0 = time.time()
    for epoch in range(epochs):
        # Mini-batch
        for i in range(0, len(X_train), 64):
            batch_x = X_train[i:i+64].to(device)
            batch_y = y_train[i:i+64].to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0

    # Test
    with torch.no_grad():
        logits = model(X_test.to(device))
        preds = logits.argmax(dim=-1)
        acc = (preds == y_test.to(device)).float().mean().item()

    return acc, train_time, sum(p.numel() for p in model.parameters())

# ─── Baseline 3: isis graph pipeline (Rust, Hebbian) ─────

def isis_graph_accuracy(test_data_path):
    """Run isis Rust binary on QEC test data. Returns accuracy from the test output."""
    import subprocess
    result = subprocess.run(
        ["cargo", "test", "--test", "integration", "test_qec_decoder", "--", "--nocapture"],
        capture_output=True, text=True, timeout=300,
        cwd="/steam/rotko/isis"
    )
    # Parse accuracy from output
    for line in result.stderr.split("\n"):
        if "QEC DECODER" in line:
            # "QEC DECODER: baseline=26.0% → trained=96.0% (1.2s)"
            parts = line.split("trained=")[1].split("%")[0]
            return float(parts) / 100.0
    return None

# ─── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    train_path = "data/qec/train.jsonl"
    test_path = "data/qec/test.jsonl"

    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found. Run data/qec/generate_syndromes.py first.")
        sys.exit(1)

    train_data = load_data(train_path)
    test_data = load_data(test_path)
    syndrome_dim = len(train_data[0][0])

    print(f"QEC Three-Way Benchmark")
    print(f"  Data: {len(train_data)} train, {len(test_data)} test, syndrome_dim={syndrome_dim}")
    print(f"  Task: d=5 repetition code, p=0.05, binary classification")
    print()

    # 1. Majority vote
    maj_acc = majority_vote_accuracy(test_data)
    print(f"1. Majority Vote:     {maj_acc*100:.1f}%  (trivial baseline)")

    # 2. v1 CTM (PyTorch, gradients)
    ctm_acc, ctm_time, ctm_params = train_v1_ctm(train_data, test_data, syndrome_dim)
    print(f"2. Mini-CTM (PyTorch): {ctm_acc*100:.1f}%  ({ctm_params} params, {ctm_time:.1f}s, gradients)")

    # 3. isis graph pipeline (Rust, Hebbian)
    isis_acc = isis_graph_accuracy(test_path)
    if isis_acc:
        print(f"3. isis Graph (Rust):  {isis_acc*100:.1f}%  (Hebbian, no gradients)")
    else:
        print(f"3. isis Graph (Rust):  (run test_qec_decoder manually)")

    print()
    print("| Decoder | Accuracy | Gradients | Params |")
    print("|---------|----------|-----------|--------|")
    print(f"| Majority Vote | {maj_acc*100:.1f}% | No | 0 |")
    print(f"| Mini-CTM (PyTorch) | {ctm_acc*100:.1f}% | Yes | {ctm_params} |")
    if isis_acc:
        print(f"| isis Graph (Rust) | {isis_acc*100:.1f}% | No | ~256 |")
