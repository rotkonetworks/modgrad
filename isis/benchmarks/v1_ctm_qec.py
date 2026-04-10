#!/usr/bin/env python3
"""Benchmark v1 CTM (PyTorch, gradients) on surface code QEC."""
import json, time, torch, torch.nn as nn, numpy as np

def load(path):
    data = []
    with open(path) as f:
        for l in f:
            d = json.loads(l)
            data.append((d["syndrome"], d["label"]))
    return data

class MiniCTM(nn.Module):
    """Minimal CTM: synapse + recurrent ticks + readout. PyTorch gradients."""
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
            h = h + self.act(self.synapse(torch.cat([x, h], -1)))
            h = self.act(self.nlm(h))
        return self.readout(h)

train = load("data/qec/surface_train.jsonl")
test = load("data/qec/surface_test.jsonl")
syn_dim = len(train[0][0])

X_train = torch.tensor([d[0] for d in train], dtype=torch.float32)
y_train = torch.tensor([d[1] for d in train], dtype=torch.long)
X_test = torch.tensor([d[0] for d in test], dtype=torch.float32)
y_test = torch.tensor([d[1] for d in test], dtype=torch.long)

results = []
for d_hidden in [32, 128, 256]:
    for ticks in [1, 4, 12]:
        model = MiniCTM(syn_dim, d_hidden, ticks, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        params = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        for epoch in range(20):
            for i in range(0, len(X_train), 64):
                logits = model(X_train[i:i+64])
                loss = criterion(logits, y_train[i:i+64])
                opt.zero_grad(); loss.backward(); opt.step()
        train_time = time.time() - t0

        with torch.no_grad():
            preds = model(X_test).argmax(-1)
            acc = (preds == y_test).float().mean().item()

        print(f"  v1 CTM d={d_hidden:3d} ticks={ticks:2d}: {acc*100:.1f}%  ({params} params, {train_time:.1f}s)")
        results.append((d_hidden, ticks, acc, params, train_time))

print(f"\nBest: {max(results, key=lambda x: x[2])}")
