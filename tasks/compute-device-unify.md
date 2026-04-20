# `modgrad-compute` × `modgrad-device` — layering audit + staged unification

**Design doc + task breakdown.**
Owner: @hdevalence (lead) · Reviewers: @redshiftzero (correctness) + @koute (perf) + @micay (defensive)

Working document — edit in place. Git log is the change trail.

## Motivation

Two backend systems exist side by side today:

| system | where | purpose |
|---|---|---|
| `modgrad_device::backend::Backend` | `crates/modgrad-device/src/backend/` | stateless op dispatch on host slices. Op-enum-driven. Plugin model. |
| `modgrad_compute::backend::ComputeBackend` | `crates/modgrad-compute/src/backend.rs` | stateful device context. VRAM lifecycle, arena reset, flush, fused synapse_forward. |

These are **not redundant** today — `ComputeBackend` expresses things the `Op` enum doesn't:

1. **Fused synapse_forward** — matvec → GLU → SiLU → layer_norm in one KFD dispatch. `Op::Matvec + Op::GluFwd + Op::SiluFwdInplace + Op::LayerNormFwd` would be four dispatches on the new path today.
2. **Device-resident buffers** — `alloc_f32(n) -> GpuVec` returns VRAM-backed memory; callers then do multiple ops without host round-trips. `Op`-level dispatch always takes `&[f32]` and re-uploads per call (ROCm's `matvec_f32` re-allocates 3×hipMalloc every dispatch — this is the problem).
3. **Arena allocator** — `arena_reset()` reclaims temporary VRAM between training steps.
4. **Flush / submit-wait batching** — `flush()` lets GPU backends queue dispatches and submit them as one batch.

Overlap is real:

- Both dispatch `matvec`, `glu`, `silu`, `layer_norm`, `trace_shift`, `superlinear`, `sync_update` against the same KFD / CUDA / CPU targets.
- `modgrad-compute` has ~32 direct `modgrad_device::kfd::*` imports bypassing the plugin model.
- `ComputeBackend` has hardcoded `StreamGpuBackend` + `VramGpuBackend` + `CudaBackend` variants — the same pattern `BackendRegistry` generalized.

## The structural diagnosis

`ComputeBackend` is `Backend` + **lifecycle** + **fusion**.

If we add fused ops to the `Op` enum and a device-resident buffer type to `modgrad-device`, `ComputeBackend` collapses to a thin lifecycle wrapper over `BackendRegistry`. The parallel system is the legacy of `ComputeBackend` predating the plugin model. Ready to fold in now.

## Committed layering (the one diagram)

```
caller (modgrad-ctm, modgrad-ffn, isis)
  │
  ▼
modgrad_device::backend::ops::*    ← functions: ops::matvec, ops::synapse_forward, …
  │
  ▼
modgrad_device::backend::registry()
  │
  ▼
Backend trait impls (CpuBackend, KfdBackend, RocmBackend, …)
```

Plus, for stateful/lifecycle workloads:

```
caller
  │
  ▼
ComputeCtx                         ← owns: DeviceBuffer alloc, arena_reset, flush
  ├── calls ops::* for every dispatch
  └── manages backend-affine DeviceBuffers
```

**Rule**: nothing below `ops::` calls `ops::`. `ComputeBackend` impls are **NOT** registered in `BackendRegistry`. The layers are strict and directional.

## Committed policy: DeviceBuffer is backend-affine

A `DeviceBuffer` is owned by **exactly one** backend. Cross-backend dispatch requires explicit host round-trip by the caller — no implicit download. This is enforced at the type level:

```rust
pub trait Backend {
    /// Associated buffer type — opaque handle into THIS backend's memory.
    type Buffer: DeviceBuffer;

    fn alloc_buffer(&self, n_f32: usize) -> Result<Self::Buffer, BackendError>;
    // …
}

pub trait DeviceBuffer: Send + Sync + 'static {
    fn backend_name(&self) -> &'static str;
    fn len(&self) -> usize;
    fn copy_from_host(&mut self, src: &[f32]) -> Result<(), BackendError>;
    fn copy_to_host(&self, dst: &mut [f32]) -> Result<(), BackendError>;
}
```

Ops that take buffers are parameterised over `B: Backend` so the compiler refuses to hand a KFD buffer to ROCm's dispatcher. If a caller wants cross-backend, they write the `copy_to_host` → `copy_from_host` sequence themselves — loudly, by hand.

Registry-level `dispatch(&mut Op)` stays as-is for `&[f32]`-slice ops; resident-buffer dispatch goes through `ComputeCtx::<B>` which is monomorphic on the chosen backend.

## Committed policy: no deprecation window

modgrad is not semver-stable and has zero external consumers. Stage 6 is a single atomic migration commit, not a deprecation cycle. Internal consumers (`modgrad-ctm`, `modgrad-ffn`, `isis`) migrate in the same commit that removes the old surface. Anything slower is fiction.

## Committed stance: global mutable state stays (for now)

`neuron::enable_gpu()` + `set_backend(...)` + `arena_reset()` are process-global mutable state. This document **preserves the invariant** — migration does not introduce new globals, but does not fix the four existing ones. Holding a `GpuVec` across an `arena_reset()` call is UB that this refactor does not address.

Fixing the globals is a separate, larger effort (thread `&ComputeCtx` through every caller). Explicit out-of-scope.

## Non-goals

- Deleting `modgrad_compute::neuron` (Linear, SuperLinear, SimpleRng). Layer primitives, orthogonal to backends. Stay.
- Full graph capture / kernel fusion compiler. Fused variants are hand-picked hot paths, not synthesized.
- Fixing the four process globals named above.
- Cross-backend DeviceBuffer compatibility.

## Staged plan

Each stage is its own reviewable commit. Tree compiles + tests pass on default and `--features kfd` at every stage boundary.

### Stage 1 — fused ops to the Op enum

Add:

```rust
Op::SynapseForward {
    weight: &'a [f32], bias: &'a [f32], x: &'a [f32],
    out: &'a mut [f32],
    out_dim: usize, in_dim: usize,
}
Op::LayerNormInplace {
    x: &'a mut [f32],
    n_rows: usize, n_cols: usize,
}
```

**No `scratch` field.** CPU composes (allocates its own vec), KFD dispatches the fused kernel (uses its own preallocated kernel slot). The `out_dim * 2` scratch detail is a KFD implementation concern and does not leak across the Op boundary.

`ops::synapse_forward(..)` + `ops::layer_norm_inplace(..)` façades added.

#### Tolerance bar (committed)

- **Single-dispatch**: `max_abs_err ≤ 2e-5`, `max_rel_err ≤ 1e-4` between CPU composed and KFD fused.
- **Long-horizon**: `ffn_trains` existing integration test must converge to within ±1% of its current final loss after 500 steps under the fused KFD path. Run on gfx1102 before merge.

#### Test matrix (committed)

```
prop_synapse_forward_parity         # shape-scan, CPU vs KFD single dispatch
prop_layer_norm_inplace_parity      # shape-scan, CPU vs KFD
ffn_trains_fused_kfd                # long-horizon, feature-gated on kfd + gfx1102
```

`ffn_trains_fused_kfd` emits `eprintln!("[test] SKIP — no gfx1102 detected")` and returns early when the KFD runtime is absent, so CI without hardware doesn't get a silent green.

#### Stage 1 failure-mode decision criterion

If single-dispatch tolerance passes but long-horizon fails:
- → tighten tolerance, investigate accumulated drift, one kernel-debug session to fix.

If single-dispatch tolerance fails at any proptest-reachable shape:
- → decline KFD fusion via `supports()` permanently, accept the 4-dispatch cost, land Stage 1 with CPU-only fused impl. **Do not hold the chain.** Future kernel fix re-enables KFD fusion as a standalone commit.

Explicit: we do not block on "fix the KFD kernel" — that's an indefinite timeline.

### Stage 2 — DeviceBuffer trait + backend-affine allocation

Introduce the `DeviceBuffer` trait per committed policy above. Each `Backend` impl adds `type Buffer` and `alloc_buffer`. CPU's Buffer is `Vec<f32>` wrapped in a newtype; KFD's wraps `GpuBuffer`; ROCm's wraps `HipBuffer`.

Add a `ComputeCtx<B: Backend>` struct in `modgrad-device` that owns a reference to one backend and exposes `alloc_buffer`, `arena_reset` (no-op except on backends that have one), `flush` (ditto). Monomorphic — no dyn.

#### Tests
- Alloc/dealloc each backend's buffer type (default + `--features kfd`)
- Roundtrip `copy_from_host` + `copy_to_host` equals identity for every backend
- Compile-fail test: assigning a KFD `Buffer` to a `ComputeCtx<CpuBackend>` must fail to compile

### Stage 3 — `StreamGpuBackend` delegates to `ops::`

Rewrite `StreamGpuBackend::matvec/glu/silu/…` to call `ops::*`. Delete direct `modgrad_device::kfd::accel::*` imports from `backend.rs`. `StreamGpuBackend` remains a `ComputeBackend` impl (stateful), but internally is now a thin facade over `ops::*`.

Regression watch: before/after, measure dispatch overhead on the existing `bench_matvec` example. Acceptable delta: ≤5% per dispatch. More than that means the ops:: façade is adding indirection we didn't expect; investigate before merge.

### Stage 4 — `VramGpuBackend` → `DeviceBuffer`

`VramGpuBackend` allocates via `alloc_device_vram` today. Switch to `DeviceBuffer` via `ComputeCtx<KfdBackend>`. `modgrad_compute::alloc_device_vram` becomes a `#[doc(hidden)]` shim marked for removal in Stage 6.

### Stage 5 — `BatchedOptimizer` lives at the `ops::` layer (not device)

Move `VramMirror` into `modgrad-device` under `#[cfg(feature = "kfd")]` — it's the KFD-specific ioctl wrapper, belongs at device. But the *orchestration* trait `BatchedOptimizer` (batching N weight updates into one submit) lives at `ops::` — it's a dispatch policy, not a device concern:

```rust
// in modgrad_device::backend::ops
pub trait BatchedOptimizer {
    fn step_batch<'a>(&mut self, slots: impl Iterator<Item = AdamWArgs<'a>>);
    fn flush(&mut self);
}
```

KFD's impl uses `VramMirror` + `try_adamw_vram_batch`; CPU's impl iterates one-by-one. `modgrad-compute::optimizer_state` becomes a thin caller of this trait.

### Stage 6 — atomic migration commit

Delete every `ComputeBackend` dispatch method (keep lifecycle-only, or remove the trait entirely if all users migrated). Delete every duplicated kernel path in `modgrad-compute/src/backend.rs`. Same commit updates `modgrad-ctm`, `modgrad-ffn`, `isis` call sites.

No `#[deprecated]` window, no grace period. Either the chain is ready or it isn't; we don't ship half-migrated state.

## Risk notes

- **Stage 1 is load-bearing.** See failure-mode criterion above — refactor chain proceeds even if KFD fusion fails (CPU-only fused impl is still a win for the non-KFD backends that inherit it automatically).
- **ROCm / CUDA fused kernels don't exist yet.** Stage 1's CPU impl is the reference; `supports()` declines fused variants on ROCm/CUDA/Vulkan, registry falls through to CPU composition. Follow-up commits add ROCm/CUDA fused kernels as they're written, one at a time.
- **CI without gfx1102** never exercises the KFD fused path. The `eprintln!(SKIP)` emission makes this visible in test logs. Merge gate: at least one `--features kfd` run on real gfx1102 hardware per PR.
- **Globals**: treated as invariant; see "Committed stance" above.

## Definition of done

- `ComputeBackend` trait and its `StreamGpuBackend` / `VramGpuBackend`
  impls retain direct `modgrad_device::kfd::accel::*` call sites
  (~13 across `modgrad-compute/src/{backend.rs, tensor_device.rs,
  optimizer_state.rs, lib.rs}`). These are **not** feature-gated —
  the `kfd` Rust module compiles unconditionally (modgrad-compute
  depends on its types), and the VRAM-lifecycle path does not yet
  have an ops-layer equivalent. Accepted technical debt pending
  `Op::SyncUpdateFwd` alignment + `DeviceBuffer` adoption in
  `VramGpuBackend`. Tracked as the remaining straggler — not a
  regression, a known follow-up.
- `ComputeBackend` trait reduced toward lifecycle-only; dispatch
  methods that already have an `ops::` equivalent delegate there.
  Full removal of the trait depends on the VRAM shim landing above.
- No `&[f32]`-slice op implementation exists in two places. (VRAM
  lifecycle dispatches are the exception — see straggler bullet.)
- Parity + proptest harness covers every fused variant; long-horizon
  convergence test wired.
- `cargo test --workspace` passes on default + `--features kfd`;
  gfx1102 run logged in PR description.

## Not in this plan

- Rocm buffer pool (separate perf commit, orthogonal).
- CUDA NVRTC wiring for SuperLinear (separate backend-feature commit).
- Vulkan quantized kernels (separate backend-scope commit).
- KFD small-shape kernel bugs (separate kernel-debug commits, filed per variant).
- Fixing the four process-global mutable-state knobs.
