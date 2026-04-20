# Building modgrad

## Default build

```
cargo build --release
```

Root crate defaults to `features = ["rocm", "cuda"]`. Requires:

- ROCm at `/opt/rocm` (link-time: `libamdhip64`, `libhipblas`)
- A CUDA install discoverable at runtime (cudarc uses
  `fallback-dynamic-loading`, no nvcc needed at build time for the
  default `cuda` feature)

Without both, the default build fails at link time. Use per-crate or
per-feature invocations below on leaner boxes.

## ROCm-less build

The library crate has zero default features, so:

```
cargo build --release -p modgrad-device --no-default-features
cargo build --release -p modgrad-compute --no-default-features
```

The root crate (`modgrad`) needs features dropped explicitly:

```
cargo build --release --no-default-features
```

## CUDA-less build

`modgrad-device`'s `cuda` feature uses `cudarc` with
`fallback-dynamic-loading` — no nvcc required, just a CUDA runtime if
one is present at run time. Safe to enable on boxes without an NVIDIA
toolchain; the dlopen failure is caught and treated as "no CUDA backend".

The heavier `cuda-kernels` feature (candle-kernels PTX) **does** need
`nvcc` at build time. Don't enable on a host without it.

## `--features kfd`

```
cargo build --release --features kfd
```

Build-time: no special deps (the `kfd` module is pure-Rust FFI to
`/dev/kfd` ioctls).

Runtime: requires `/dev/kfd` to exist and gfx1102 (RDNA3 Navi 33) to
be the detected device. Other GPUs: `KfdBackend::try_new()` returns
`None` and the backend declines registration.

## Test invocations

```
# Default backends (ROCm + CUDA both active):
cargo test --release

# KFD on top (exercises hand-written gfx1102 kernels):
cargo test --release --features kfd

# Library crates only, no GPU deps at all:
cargo test --release -p modgrad-device --no-default-features
cargo test --release -p modgrad-ctm --no-default-features
```

Hardware-gated tests emit `[test] SKIP — no gfx1102 detected` and
return early when KFD is unavailable, so CI without hardware does not
produce silent greens. The proptest regression corpus in
`crates/modgrad-device/tests/backend_proptest.proptest-regressions`
gets re-run under `--features kfd` whenever real hardware is present.

## Known gotcha

`cargo test --workspace` fails on boxes without ROCm because workspace
feature unification forces `modgrad-device/rocm` active. Workarounds:

- `cargo test --workspace --no-default-features` (skips ROCm + CUDA
  at the root, but isis/runtime etc. may still pull things)
- Per-crate invocations with explicit `--no-default-features` on the
  binary crates.

The real fix is `resolver = "2"` + per-crate test runs in CI; that's
tracked separately.
