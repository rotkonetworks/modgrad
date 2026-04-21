# GPU Benchmarks

First measured numbers for the modgrad SDK's GPU backends on a single
gfx1102 laptop iGPU (AMD Radeon RX 7600M XT, ROCm 7). All runs use the
mazes example — `cargo run --release -p mazes --features rocm ...` —
against a CPU baseline via `MODGRAD_BACKEND=cpu`.

Training is deterministic given `--seed`, so `rocm_exit`/`cpu_exit`
eval numbers are expected to match bit-for-bit when the GPU path is
correct. Any divergence indicates a silent numerical bug (this is how
we found the ROCm VRAM-cache invalidation issue — see commit 7f17f42).

## Configurations measured

### Small: `--size 11 --ticks 8 --steps 500 --batch 4 --d-model 128 --route-len 10`

| Backend | Wall time | Per-step acc | First-step acc | Notes |
|---|---:|---:|---:|---|
| CPU   | 140 s  | 32.0 % (OOD) | 50.0 % (OOD) | rayon multi-threaded on 16-core host |
| ROCm  | 144 s (**-3 %**) | 32.0 % (OOD) | 50.0 % (OOD) | Bit-identical eval; slower by dispatch overhead |

At this size, the matvec shapes (`out_dim ≤ 128`) are too small for
hipblas to amortise per-call cost. CPU beats GPU by a hair.

### Big: `--size 21 --ticks 16 --steps 300 --batch 4 --d-model 256 --route-len 20`

| Backend | Wall time | Per-step acc | First-step acc | Notes |
|---|---:|---:|---:|---|
| CPU   | 657 s  | 25.4 % (OOD) | 51.5 % (OOD) | |
| ROCm  | **603 s** (**+8.3 %**) | 25.4 % (OOD) | 51.5 % (OOD) | First measured GPU speedup in this SDK |

At this size, the crossover happens. `out_dim=256` matmuls are large
enough that GPU compute dominates, and ROCm pulls ahead. Eval
numbers remain bit-identical to CPU — correctness preserved.

### Extra-large: `--size 21 --ticks 16 --steps 200 --batch 4 --d-model 384 --route-len 20`

| Backend | Wall time | Per-step acc | First-step acc | Notes |
|---|---:|---:|---:|---|
| CPU                         | 465 s  | 25.8 % (OOD) | 45.0 % (OOD) | |
| ROCm, no cache, no MatvecT  | 413 s (+11 %) | 25.8 % (OOD) | 45.0 % (OOD) | commit 012b923 |
| ROCm, cache, no MatvecT     | **393 s** (**+15 %**) | 25.8 % (OOD) | 45.0 % (OOD) | commit 9923418 — current winner |
| ROCm, no cache, + MatvecT   | 1091 s (−135 %) | 25.8 % (OOD) | 45.0 % (OOD) | commit b8c7a2b (reverted) — perf cliff |
| ROCm, cache, + MatvecT      | 1048 s (−125 %) | 25.8 % (OOD) | 45.0 % (OOD) | cache saves only 43 s of 655 s MatvecT cost |

Biggest win so far. Larger matmuls let hipBLAS compute dominate dispatch
overhead, and eval stays bit-identical to CPU across every row.

The table is the experimental arc. Two hypotheses tested:

1. **MatvecT on ROCm**: first pass routed backward `W^T @ d_out` via
   `hipblasSgemv` with `OP_N` (same weight buffer as forward Matvec,
   one transpose-free trick away). Expected a second win; measured
   2.3× slower than CPU at d_model=384. Reverted in 012b923.
2. **Weight cache would rescue MatvecT**: hypothesis was that the
   weight re-upload per dispatch (disabled in 7f17f42 for correctness)
   was MatvecT's bottleneck, and the fingerprint-keyed cache in
   9923418 would let forward+backward share one upload. Re-ran with
   cache+MatvecT: 1048 s. Cache saved 43 s; MatvecT still cost 655 s.

**Real bottleneck: per-dispatch overhead, not weight upload.** Each
MatvecT dispatch does 3 × `hipMalloc` + 2 × `hipMemcpy` + 1 ×
`hipblasSgemv` + 1 × `hipDeviceSynchronize` + 1 × `hipMemcpy` back +
3 × `hipFree`. At ~20–40 MatvecT dispatches per training step × 200
steps × several hundred µs per dispatch, the fixed cost dominates
whether or not the weight buffer is cached.

Paths that could actually land MatvecT on ROCm:
  - Batch multiple backward matvecs into a single `hipblasSgemmStridedBatched`
    call (one dispatch amortised across many layers).
  - Keep activations resident on GPU across forward/backward — today
    every matvec downloads to host immediately. A GPU-side activation
    buffer ring would eliminate the ping-pong.
  - Fuse Matvec + MatvecT + OuterProductAcc into one backward kernel
    (the entire Linear backward becomes one dispatch).

None of these are small; each is its own project. For now, forward
Matvec on GPU + everything else on CPU is Pareto-optimal on this
hardware/benchmark.

## What's registered

`cargo build --features rocm` produces a binary where `hipBLAS`-backed
`Matvec`/`MatmulNN` dispatches at `out_dim >= 64` and `in_dim >= 64`
(see `crates/modgrad-device/src/backend/rocm.rs::supports`). Smaller
shapes fall through to CPU. This shape gate is what keeps the small
configuration's GPU overhead bounded — ROCm isn't claiming ops that
don't benefit.

`cargo build --features kfd` produces a binary where the KFD
hand-kernel path is available, but it is **not registered by default**.
`MODGRAD_ENABLE_KFD=1` is required in the environment — the compute
ring has been observed to wedge on non-trivial training dispatches
(`put=4194313 read=9`, exact same values every process). Root cause
uninvestigated. See `crates/modgrad-device/src/backend/kfd.rs::try_new`.

## Next measurements we'd like

- `--d-model 512` (probably bigger ROCm win)
- Language-model-sized activations (10× bigger matmuls)
- ROCm vs CUDA parity on an NVIDIA host (cudarc backend is plumbed
  but we haven't had access to validate)
- KFD post-ring-hang-fix, whenever the debug lands

## Reproducing

```
cargo build --release --features rocm -p mazes
# Small:
time MODGRAD_BACKEND=cpu ./target/release/mazes --size 11 --ticks 8 \
    --steps 500 --batch 4 --d-model 128 --route-len 10 --seed 7 --ood-size 21
time ./target/release/mazes --size 11 --ticks 8 \
    --steps 500 --batch 4 --d-model 128 --route-len 10 --seed 7 --ood-size 21
# Big:
time MODGRAD_BACKEND=cpu ./target/release/mazes --size 21 --ticks 16 \
    --steps 300 --batch 4 --d-model 256 --route-len 20 --seed 42 --ood-size 31
time ./target/release/mazes --size 21 --ticks 16 \
    --steps 300 --batch 4 --d-model 256 --route-len 20 --seed 42 --ood-size 31
# Extra-large:
time MODGRAD_BACKEND=cpu ./target/release/mazes --size 21 --ticks 16 \
    --steps 200 --batch 4 --d-model 384 --route-len 20 --lr 3e-4 --seed 42 --ood-size 31
time ./target/release/mazes --size 21 --ticks 16 \
    --steps 200 --batch 4 --d-model 384 --route-len 20 --lr 3e-4 --seed 42 --ood-size 31
```
