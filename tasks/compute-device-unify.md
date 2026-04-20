# `modgrad-compute` × `modgrad-device` — layering audit + staged unification

**Design doc + task breakdown.**
Owner: @hdevalence (lead) · Reviewers: @redshiftzero (correctness) + @koute (perf)

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

But the overlap is real too:

- Both dispatch `matvec`, `glu`, `silu`, `layer_norm`, `trace_shift`, `superlinear`, `sync_update` against the same KFD / CUDA / CPU targets.
- `modgrad-compute` has ~32 direct `modgrad_device::kfd::*` imports bypassing the plugin model.
- `ComputeBackend` has hardcoded `StreamGpuBackend` + `VramGpuBackend` + `CudaBackend` variants — the same pattern `BackendRegistry` generalized.

## The structural diagnosis

`ComputeBackend` is `Backend` + **lifecycle** + **fusion**.

If we add fused ops to the `Op` enum and a device-resident buffer type to `modgrad-device`, `ComputeBackend` collapses to a thin wrapper over `BackendRegistry` with a lifecycle protocol.

The current parallel system is the legacy of ComputeBackend predating the plugin model. That's fine; it wasn't wrong when written. It's just ready to fold in now.

## Non-goals

- Deleting `modgrad_compute::neuron` (Linear, SuperLinear, SimpleRng). Those are **layer** primitives, orthogonal to backends. Stay.
- Full graph capture / kernel fusion compiler. Fused variants are hand-picked hot paths, not synthesized.
- Breaking public API. Any `pub use modgrad_compute::*` path must keep working through a deprecation window.

## Staged plan

Each stage is its own reviewable commit. The chain is ordered so that at every point, the tree compiles and tests pass on both default and `--features kfd`.

### Stage 1 — fused ops to the Op enum

Add these variants:

- `Op::SynapseForward { weight, bias, x, out, scratch, out_dim, in_dim }` — fused matvec → GLU → SiLU → layer_norm. CPU impl composes the four existing ops; KFD impl dispatches the single fused kernel.
- `Op::LayerNormInplace { x, n_rows, n_cols }` — KFD has an inplace LN kernel that the current `Op::LayerNormFwd { out, cache }` can't express. Matches the pattern we already set with `Op::SiluFwdInplace`.

Also add `ops::synapse_forward(..)` and `ops::layer_norm_inplace(..)` façades.

Tests: new `prop_synapse_forward_parity` proptest; CPU composed vs KFD fused must agree within tolerance.

### Stage 2 — device-resident buffer type in modgrad-device

Introduce a cross-backend `DeviceBuffer` type that owns VRAM on the backend where it was allocated:

```rust
pub trait DeviceBuffer: Send + Sync {
    fn len(&self) -> usize;
    fn as_device_slice(&self) -> DeviceSlice<'_>;
    fn copy_from_host(&mut self, src: &[f32]);
    fn copy_to_host(&self, dst: &mut [f32]);
}
```

Backends implement `fn alloc_buffer(&self, n_f32: usize) -> Result<Box<dyn DeviceBuffer>>`. KFD's impl wraps the existing `GpuBuffer`; ROCm's wraps `HipBuffer`; CPU's is just `Vec<f32>`.

Ops take `&dyn DeviceBuffer` or `&[f32]` via a thin `DeviceSliceRef` enum so callers can pass either. (This is the ergonomic-vs-purity trade; pick the enum for now.)

Tests: buffer allocation, host↔device roundtrip, op-against-buffer parity.

### Stage 3 — migrate `ComputeBackend::StreamGpuBackend` onto BackendRegistry

Rewrite `StreamGpuBackend` to dispatch through `registry()` using the fused ops from Stage 1. Delete the direct `modgrad_device::kfd::accel::*` imports from `backend.rs`. The `ComputeBackend` trait stays (it's the lifecycle surface), but its KFD impl now has zero direct-kfd references — all dispatch flows through the registry.

Expected diff: `backend.rs` shrinks by ~300 LOC; `optimizer_state.rs` and `tensor_device.rs` still hold direct kfd usage for VRAM-specific flows, handled in stages 4-5.

### Stage 4 — migrate `VramGpuBackend` to DeviceBuffer

`VramGpuBackend` allocates via `alloc_device_vram` today. Switch to `DeviceBuffer` trait. `modgrad_compute::alloc_device_vram` becomes a thin shim that eventually gets removed.

### Stage 5 — migrate `OptimizerState` / `VramMirror`

`optimizer_state.rs` has the last deep kfd hook: `VramMirror` + `try_adamw_vram_batch`. Move `VramMirror` into modgrad-device under a feature-gated submodule, expose a `BatchedOptimizer` trait at the `modgrad-device` layer, implement it for KFD. `modgrad-compute` just calls the trait.

### Stage 6 — delete deprecated surface

Anything in `modgrad_compute` that duplicates modgrad-device gets marked `#[deprecated]` for one release, then removed.

## Risk notes

- **Stage 1 is load-bearing for everything after.** If `SynapseForward` parity fails we discover real KFD kernel bugs (possibly) or fusion-tolerance issues. Proptest first, commit second.
- **ROCm / CUDA fused kernels don't exist yet.** Stage 1's CPU impl is the reference; KFD supports the single fused variant. ROCm/CUDA decline via `supports()` and fall back to the CPU composition.
- **`modgrad_compute::neuron::enable_gpu` + `set_backend`** are process-global mutable state (still). Staging this out of existence is beyond the scope of this document — treat the global switch as an invariant until migration is complete.

## Definition of done

- Zero `modgrad_device::kfd::*` imports in `modgrad-compute/src/*` except via `cfg(feature = "kfd")` in the explicit VRAM lifecycle shim.
- `ComputeBackend` trait either (a) reduced to lifecycle-only (no dispatch methods), or (b) removed entirely in favor of `BackendRegistry + DeviceBuffer`.
- No op implementation exists in two places. Fused variants live on `Op`; decomposed variants live on `Op`; nothing in `modgrad-compute::backend` duplicates dispatch logic.
- Parity + proptest harness covers every fused variant.
- `cargo test --workspace --exclude modgrad-compute` passes unchanged; `-p modgrad-compute` passes at every stage.

## Not in this plan

- Rocm buffer pool (separate perf commit, orthogonal).
- CUDA NVRTC wiring for SuperLinear (separate backend-feature commit).
- Vulkan quantized kernels (separate backend-scope commit).
- KFD small-shape kernel bugs (separate kernel-debug commits, filed per variant).
