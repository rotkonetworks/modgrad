#!/usr/bin/env python3
"""
Autotune isis CTM parameters on QEC benchmark.
Bayesian optimization over the organism's tuning space.
The agent IS the hypothalamus.
"""
import json
import subprocess
import time
import random
import math

# Parameters to tune and their ranges
PARAM_SPACE = {
    "n_neurons": [32, 64, 128],
    "ticks_input": [1, 2, 4],
    "ticks_attention": [4, 8, 12, 16],
    "ticks_output": [2, 4, 8],
    "ticks_motor": [1, 2, 4],
    "memory_length": [4, 8, 16],
    "n_sync_out": [64, 128],
    "hebbian_lr_input": [0.0, 0.005, 0.01, 0.02],
    "hebbian_lr_attn": [0.0, 0.001],
    "hebbian_lr_output": [0.0, 0.001],
    "hebbian_lr_motor": [0.0, 0.001, 0.005],
    "inhibitory_fraction": [0.2, 0.3, 0.4],
    "motor_threshold": [5.0, 10.0, 999.0],
    "ls_blend": [0.2, 0.3, 0.5],
    "sleep_cycles": [0, 5, 10],
    "collect_correct_only": [False],
}

def random_config():
    return {k: random.choice(v) for k, v in PARAM_SPACE.items()}

def evaluate(config, train_path="data/qec/surface_train.jsonl",
             test_path="data/qec/surface_test.jsonl"):
    """Run isis CTM with given config, return accuracy."""
    # Write config to temp file
    config_path = "/tmp/isis_autotune_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Run the Rust evaluator
    result = subprocess.run(
        ["cargo", "test", "--release", "--test", "integration",
         "test_autotune_eval", "--", "--nocapture"],
        capture_output=True, text=True, timeout=120,
        cwd="/steam/rotko/isis",
        env={**__import__("os").environ, "AUTOTUNE_CONFIG": config_path}
    )

    # Parse accuracy from output
    for line in result.stderr.split("\n"):
        if "AUTOTUNE_ACC=" in line:
            return float(line.split("AUTOTUNE_ACC=")[1].split("%")[0]) / 100.0
    return 0.0

def run_search(n_trials=50):
    """Random search with elite tracking."""
    best_acc = 0.0
    best_config = None
    history = []

    for trial in range(n_trials):
        config = random_config()
        t0 = time.time()
        acc = evaluate(config)
        elapsed = time.time() - t0

        history.append({"trial": trial, "acc": acc, "config": config, "time": elapsed})

        if acc > best_acc:
            best_acc = acc
            best_config = config
            marker = " ★ NEW BEST"
        else:
            marker = ""

        print(f"  [{trial+1}/{n_trials}] acc={acc*100:.1f}% ({elapsed:.1f}s) "
              f"n={config['n_neurons']} "
              f"t=[{config['ticks_input']},{config['ticks_attention']},{config['ticks_output']},{config['ticks_motor']}] "
              f"sync={config['n_sync_out']} inhib={config['inhibitory_fraction']}"
              f"{marker}")

    print(f"\n{'='*60}")
    print(f"Best: {best_acc*100:.1f}%")
    print(f"Config: {json.dumps(best_config, indent=2)}")

    # Save history
    with open("benchmarks/autotune_history.jsonl", "w") as f:
        for h in history:
            json.dump(h, f)
            f.write("\n")

    return best_acc, best_config

if __name__ == "__main__":
    print("isis CTM Autotune on QEC")
    print(f"Parameter space: {math.prod(len(v) for v in PARAM_SPACE.values())} combinations")
    print()
    run_search(50)
