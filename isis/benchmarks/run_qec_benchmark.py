#!/usr/bin/env python3
"""
QEC Benchmark: fusion-blossom MWPM vs isis graph pipeline.
d=5 code capacity planar code, p=0.05.
"""
import fusion_blossom as fb
import numpy as np
import json
import time
import random

def run_benchmark(d, p, n_samples, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    code = fb.CodeCapacityPlanarCode(d=d, p=p, max_half_weight=500)
    init = code.get_initializer()
    edges = init.weighted_edges  # list of (v1, v2, weight)
    n_vertices = init.vertex_num
    n_edges = len(edges)
    virtual = set(init.virtual_vertices)

    samples = []
    mwpm_correct = 0

    for _ in range(n_samples):
        # Random errors on edges
        error_edges = []
        for e_idx in range(n_edges):
            if random.random() < p:
                error_edges.append(e_idx)

        # Syndrome: each non-virtual vertex flips if odd adjacent errors
        syndrome_bits = [0] * n_vertices
        for e_idx in error_edges:
            v1, v2, _ = edges[e_idx]
            syndrome_bits[v1] ^= 1
            syndrome_bits[v2] ^= 1

        defects = [i for i in range(n_vertices) if syndrome_bits[i] == 1 and i not in virtual]

        # Label: parity of errors (simplified)
        label = len(error_edges) % 2

        # MWPM decode
        solver = fb.SolverSerial(init)
        syndrome = fb.SyndromePattern(defects)
        solver.solve(syndrome)
        subgraph = solver.subgraph()
        mwpm_parity = len(subgraph) % 2
        if mwpm_parity == label:
            mwpm_correct += 1
        solver.clear()

        # Only use non-virtual syndrome bits as features
        features = [float(syndrome_bits[i]) for i in range(n_vertices) if i not in virtual]
        samples.append({"syndrome": features, "label": label})

    return samples, mwpm_correct / n_samples

if __name__ == "__main__":
    d = 5
    p = 0.05

    print(f"QEC Benchmark: d={d} code capacity planar code, p={p}")

    t0 = time.time()
    train, _ = run_benchmark(d, p, 5000, seed=42)
    test, mwpm_acc = run_benchmark(d, p, 1000, seed=123)
    elapsed = time.time() - t0

    syndrome_dim = len(train[0]["syndrome"])
    pos_rate = sum(s["label"] for s in test) / len(test)

    print(f"Generated in {elapsed:.1f}s")
    print(f"Syndrome dim: {syndrome_dim}")
    print(f"Class balance: {pos_rate*100:.1f}% positive")

    # Save for isis
    with open("data/qec/surface_train.jsonl", "w") as f:
        for s in train:
            json.dump(s, f); f.write("\n")
    with open("data/qec/surface_test.jsonl", "w") as f:
        for s in test:
            json.dump(s, f); f.write("\n")

    majority_acc = max(pos_rate, 1 - pos_rate)

    print(f"\n{'='*50}")
    print(f"| Decoder          | Accuracy |")
    print(f"|------------------|----------|")
    print(f"| Majority vote    | {majority_acc*100:.1f}%    |")
    print(f"| MWPM (fusion)    | {mwpm_acc*100:.1f}%    |")
    print(f"| isis (Hebbian)   | ???      |")
    print(f"\nRun: cargo test --test integration test_qec_decoder -- --nocapture")
    print(f"(update test to use data/qec/surface_{{train,test}}.jsonl)")
