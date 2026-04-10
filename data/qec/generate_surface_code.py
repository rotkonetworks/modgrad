#!/usr/bin/env python3
"""Generate real surface code QEC syndromes using stim.

Circuit-level noise on rotated surface code — this is what
real quantum computers actually deal with.

Output: JSONL with {"syndrome": [...], "label": int}
  syndrome = detector outcomes (binary)
  label = logical observable (0 or 1)
"""

import stim
import json
import numpy as np
import sys

def generate_surface_code_data(
    distance: int,
    noise: float,
    rounds: int,
    n_samples: int,
    path: str,
    noise_model: str = "depolarizing",
):
    """Generate syndrome data from a real surface code circuit."""

    # Build the circuit
    if noise_model == "depolarizing":
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=distance,
            after_clifford_depolarization=noise,
            after_reset_flip_probability=noise,
            before_measure_flip_probability=noise,
            before_round_data_depolarization=noise,
        )
    elif noise_model == "bitflip":
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=distance,
            before_round_data_depolarization=0,  # no depol
            after_clifford_depolarization=0,
            after_reset_flip_probability=noise,
            before_measure_flip_probability=noise,
        )
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    # Sample
    sampler = circuit.compile_detector_sampler()
    det_samples, obs_samples = sampler.sample(
        shots=n_samples, separate_observables=True
    )

    # det_samples: [n_samples, n_detectors] bool
    # obs_samples: [n_samples, n_observables] bool
    n_det = det_samples.shape[1]
    n_obs = obs_samples.shape[1]

    print(f"  distance={distance}, rounds={rounds}, noise={noise} ({noise_model})")
    print(f"  detectors={n_det}, observables={n_obs}")
    print(f"  samples={n_samples}")

    # Write JSONL
    with open(path, "w") as f:
        for i in range(n_samples):
            syndrome = det_samples[i].astype(int).tolist()
            label = int(obs_samples[i, 0])  # first logical observable
            json.dump({"syndrome": syndrome, "label": label}, f)
            f.write("\n")

    # Stats
    labels = obs_samples[:, 0].astype(int)
    n_pos = labels.sum()
    n_defects_mean = det_samples.sum(axis=1).mean()
    print(f"  label balance: {n_pos}/{n_samples} ({100*n_pos/n_samples:.1f}% positive)")
    print(f"  mean defects per shot: {n_defects_mean:.1f}")
    print(f"  saved to {path}")
    return n_det


if __name__ == "__main__":
    outdir = "data/qec"

    # ── Distance 3: tiny, fast ──
    print("\n=== d=3 surface code (circuit-level depolarizing) ===")
    n_det = generate_surface_code_data(
        distance=3, noise=0.001, rounds=3,
        n_samples=50000, path=f"{outdir}/stim_d3_train.jsonl",
    )
    generate_surface_code_data(
        distance=3, noise=0.001, rounds=3,
        n_samples=10000, path=f"{outdir}/stim_d3_test.jsonl",
    )

    # ── Distance 5: standard benchmark ──
    print("\n=== d=5 surface code (circuit-level depolarizing) ===")
    generate_surface_code_data(
        distance=5, noise=0.001, rounds=5,
        n_samples=50000, path=f"{outdir}/stim_d5_train.jsonl",
    )
    generate_surface_code_data(
        distance=5, noise=0.001, rounds=5,
        n_samples=10000, path=f"{outdir}/stim_d5_test.jsonl",
    )

    # ── Distance 7: harder ──
    print("\n=== d=7 surface code (circuit-level depolarizing) ===")
    generate_surface_code_data(
        distance=7, noise=0.001, rounds=7,
        n_samples=50000, path=f"{outdir}/stim_d7_train.jsonl",
    )
    generate_surface_code_data(
        distance=7, noise=0.001, rounds=7,
        n_samples=10000, path=f"{outdir}/stim_d7_test.jsonl",
    )

    # ── Distance 11: Hyperion benchmark ──
    print("\n=== d=11 surface code (circuit-level depolarizing) ===")
    generate_surface_code_data(
        distance=11, noise=0.001, rounds=11,
        n_samples=50000, path=f"{outdir}/stim_d11_train.jsonl",
    )
    generate_surface_code_data(
        distance=11, noise=0.001, rounds=11,
        n_samples=10000, path=f"{outdir}/stim_d11_test.jsonl",
    )

    print("\nDone. Syndrome dimensions per distance:")
    for d in [3, 5, 7, 11]:
        import json as j
        with open(f"{outdir}/stim_d{d}_test.jsonl") as f:
            sample = j.loads(f.readline())
            print(f"  d={d}: {len(sample['syndrome'])} detectors")
