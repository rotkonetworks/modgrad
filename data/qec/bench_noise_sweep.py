#!/usr/bin/env python3
"""Sweep noise rates to find where MWPM starts failing.
This shows the regime where Hyperion's 4.8x improvement matters.
"""

import stim
import numpy as np
import pymatching
import time

def bench(distance, noise, rounds, n_shots=100_000):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds, distance=distance,
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(shots=n_shots, separate_observables=True)

    t0 = time.time()
    preds = matcher.decode_batch(det)
    dt = time.time() - t0

    correct = (preds == obs).all(axis=1).sum()
    acc = correct / n_shots
    ler = 1.0 - acc
    trivial = 1.0 - obs[:, 0].mean()

    return acc, ler, trivial, dt

if __name__ == "__main__":
    n_shots = 100_000

    print("=== MWPM accuracy vs noise rate (circuit-level depolarizing) ===\n")
    print(f"{'p':>8} {'d=3':>10} {'d=5':>10} {'d=7':>10} {'d=11':>10}   {'trivial_d5':>10}")
    print("-" * 70)

    for p in [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01]:
        row = f"{p:8.4f}"
        triv5 = ""
        for d in [3, 5, 7, 11]:
            try:
                acc, ler, trivial, dt = bench(d, p, rounds=d, n_shots=n_shots)
                row += f" {acc*100:9.2f}%"
                if d == 5:
                    triv5 = f" {trivial*100:9.2f}%"
            except Exception as e:
                row += f" {'err':>9}"
        row += triv5
        print(row)

    print("\n=== Logical error rate (what Hyperion paper reports) ===\n")
    print(f"{'p':>8} {'d=3':>12} {'d=5':>12} {'d=7':>12} {'d=11':>12}")
    print("-" * 60)

    for p in [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01]:
        row = f"{p:8.4f}"
        for d in [3, 5, 7, 11]:
            try:
                acc, ler, _, _ = bench(d, p, rounds=d, n_shots=n_shots)
                if ler == 0:
                    row += f" {'<1e-5':>11}"
                else:
                    row += f" {ler:11.2e}"
            except:
                row += f" {'err':>11}"
        print(row)

    print("\nHyperion claims 4.8x lower logical error rate than MWPM at d=11 depolarizing.")
    print("Look at d=11 column to see where that matters.")
