#!/usr/bin/env python3
"""Benchmark MWPM (blossom) decoder on stim surface code data.

This gives us the proper baseline to compare our brain against.
"""

import stim
import numpy as np
import pymatching
import time

def bench_mwpm(distance: int, noise: float, rounds: int, n_shots: int = 100_000):
    """Run MWPM decoder on surface code and report accuracy."""

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise,
    )

    # Build detector error model for MWPM
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    # Sample
    sampler = circuit.compile_detector_sampler()
    det_samples, obs_samples = sampler.sample(
        shots=n_shots, separate_observables=True
    )

    # Decode
    t0 = time.time()
    predictions = matcher.decode_batch(det_samples)
    dt = time.time() - t0

    # Accuracy
    correct = (predictions == obs_samples).all(axis=1).sum()
    acc = correct / n_shots
    logical_error_rate = 1.0 - acc

    n_det = det_samples.shape[1]
    print(f"  d={distance:2d}  rounds={rounds:2d}  detectors={n_det:5d}  "
          f"acc={acc*100:.2f}%  logical_err={logical_error_rate:.4e}  "
          f"time={dt:.2f}s ({dt/n_shots*1e6:.1f}µs/shot)")

    return acc, logical_error_rate


if __name__ == "__main__":
    noise = 0.001
    n_shots = 100_000

    print(f"=== MWPM (pymatching/blossom) on surface code ===")
    print(f"    noise={noise}, shots={n_shots}, circuit-level depolarizing\n")

    # Also show trivial baseline (always predict 0)
    print("--- Trivial baseline (always predict 0) ---")
    for d in [3, 5, 7, 11]:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=d, distance=d,
            after_clifford_depolarization=noise,
            after_reset_flip_probability=noise,
            before_measure_flip_probability=noise,
            before_round_data_depolarization=noise,
        )
        sampler = circuit.compile_detector_sampler()
        _, obs = sampler.sample(shots=n_shots, separate_observables=True)
        trivial_acc = 1.0 - obs[:, 0].mean()
        print(f"  d={d:2d}  trivial_acc={trivial_acc*100:.2f}% (just predict 0)")

    print("\n--- MWPM (blossom) decoder ---")
    results = {}
    for d in [3, 5, 7, 11]:
        acc, ler = bench_mwpm(d, noise, rounds=d, n_shots=n_shots)
        results[d] = (acc, ler)

    print(f"\n--- Summary ---")
    print(f"{'d':>4} {'detectors':>10} {'trivial':>10} {'MWPM':>10} {'gap':>10}")
    for d in [3, 5, 7, 11]:
        n_det = (d*d - 1) * d  # approximate
        acc = results[d][0]
        # recompute trivial
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=d, distance=d,
            after_clifford_depolarization=noise,
            after_reset_flip_probability=noise,
            before_measure_flip_probability=noise,
            before_round_data_depolarization=noise,
        )
        _, obs = circuit.compile_detector_sampler().sample(shots=10000, separate_observables=True)
        trivial = 1.0 - obs[:, 0].mean()
        print(f"{d:4d} {'-':>10} {trivial*100:9.2f}% {acc*100:9.2f}% {(acc-trivial)*100:+9.2f}pp")
