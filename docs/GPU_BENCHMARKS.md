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
| CPU   | 465 s  | 25.8 % (OOD) | 45.0 % (OOD) | |
| ROCm  | **413 s** (**+11 %**) | 25.8 % (OOD) | 45.0 % (OOD) | Forward matvec only; MatvecT reverted |

Biggest win so far. Larger matmuls let hipBLAS compute dominate dispatch
overhead, and eval stays bit-identical to CPU.

This number came after a detour worth recording. A first pass added
`MatvecT` (backward-pass `W^T @ x` via `hipblasSgemv` with `OP_N`) to
the ROCm supports() gate, expecting another win. At d_model=384 the
combined Matvec+MatvecT path measured **2.3× slower than CPU** (1091 s).
Reverting MatvecT alone cut the time to 413 s — the win above.

Why MatvecT hurt: the weight VRAM cache is currently disabled (see
commit 7f17f42 for the correctness rationale), so every GPU matvec
dispatch re-uploads its weight buffer via `hipMemcpyHtoD`. Adding
MatvecT doubles the uploads per training step (forward Matvec +
backward MatvecT), and at d_model=384 each upload moves 576 KB
(vs 256 KB at d_model=256). The extra uploads ate the compute win.
d_model=256 happened to sit on the favourable side of that break-even,
which is why the earlier MatvecT benchmark passed without flagging it.

Re-enabling the VRAM cache would flip this — both Matvec and MatvecT
amortise their upload across many steps. Two paths, documented in
`rocm.rs::cached_weight_ptr`:
  - Version-counter cache keys (robust to in-place weight mutation)
  - Every training loop calls `invalidate_caches()` after the optimizer
    step (fragile but quick; isis already does this, modgrad-ctm
    training loops do not)

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
