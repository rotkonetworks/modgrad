# ROCm Support + `Op` Enum Backend Refactor

**Design doc + task breakdown.**
Owner: @hdevalence (lead) · Reviewers: @redshiftzero (correctness) + @cronokirby (bindings)

## Motivation

modgrad has 4 incoherent GPU paths: hand-written KFD kernels (RDNA3 gfx1102 only),
a 274-line stub of CUDA via cudarc, a partial Vulkan backend via `ash`, and no
ROCm path at all. Adding any new backend today means re-implementing every op
from scratch — tech debt scales linearly with backends × ops.

JAX's PJRT/StableHLO separation solves this: **ops are data, backends are plugins**.
We adapt the structural lesson without taking on a full XLA/MLIR dependency.

## Goals (in priority order)

1. Bring up ROCm (HIP runtime + rocBLAS) on any AMD GPU supported by ROCm ≥ 6.x
2. Preserve the hand-written KFD fast path on gfx1102 — zero perf regression
3. Make adding future backends (Apple Metal, Intel oneAPI, etc.) a one-day plugin task
4. Keep all existing modgrad-ctm / modgrad-ffn callers working throughout

## Non-goals

- Full JAX/MLIR IR (years of work, not needed for the CTM-class models we care about)
- `grad`/`jit`/`vmap` composition as transformations over ops (possible follow-up once backend is stable)
- Quantized kernel parity across backends (Q4_K stays KFD-only in phase 1)

---

## Phase 0 — Design freeze (1–2 days)

### Task 0.1: Design the `Op` enum
**Owner:** @hdevalence · **Reviewer:** @redshiftzero · **Size:** 1 day

Define the finite op set every backend must speak. Granularity rule: one variant
per *logical* op. Size-tuned kernels (`matmul_blocked` vs `matmul_small`) are
dispatched *within* a backend, not exposed as separate variants.

Proposed variants (13 logical ops, 22 forward/backward pairs):

| Op | Notes |
|---|---|
| `Matmul { a, b, out, m, k, n }` | covers `matmul_blocked`, `matmul_small`, `matmul_dbg` internally |
| `Matvec { w, x, bias, out, quant: QuantKind }` | QuantKind covers f32, q4k, tiled variants |
| `MatvecT { w, x, bias, out }` | transposed matvec (backward of Linear) |
| `OuterProductAcc { a, b, accum }` | accumulates outer product into existing buffer |
| `LayerNormFwd { x, gamma, beta, out, cache }` | |
| `LayerNormBwd { d_out, cache, d_x, d_gamma, d_beta }` | |
| `LnSiluFwd { x, gamma, beta, out, cache }` | fused LN+SiLU |
| `SiluFwd { x, out }` | |
| `SiluBwd { d_out, x, d_x }` | |
| `GluFwd { x, out }` | x is split in half internally |
| `GluBwd { d_out, x, d_x }` | |
| `PerNeuronGluBwd { ... }` | CTM-specific, sized per-neuron |
| `AdamW { w, g, m, v, lr, b1, b2, eps, wd, step }` | whole optimizer step |
| `SgdUpdate { w, g, lr }` | |
| `ReduceL2Sq { x, out }` | scalar reduction |
| `SuperLinearFwd { w, b, t_bias, x, out, cache }` | CTM neuron-level MLP |
| `SuperLinearBwdDw { d_out, x, d_w }` | |
| `SuperLinearBwdDx { d_out, w, d_x }` | |
| `SyncUpdateFwd { h, pairs, decay, out, state }` | CTM pair synchronization |
| `SyncBackwardScatter { d_sync, pairs, d_h }` | |
| `TraceShiftFwd { trace, new_val, shifted }` | memory rotation |

**Acceptance:**
- [ ] Enum compiles with `#[derive(Debug)]`
- [ ] Every existing KFD kernel maps to exactly one `Op` variant (write the mapping in a comment, no code yet)
- [ ] No lifetime foot-guns (all refs are `&` or `&mut`, no owned `Vec` inside variants)
- [ ] Quant variants use enum attribute (`QuantKind`), not separate op variants

**Deliverable:** `crates/modgrad-device/src/backend/op.rs` with the enum + `impl Debug`. No `dispatch()` logic yet. Reviewed by @redshiftzero for correctness and edge cases.

---

### Task 0.2: Design the `Backend` trait
**Owner:** @hdevalence · **Reviewer:** @redshiftzero · **Size:** 0.5 day

```rust
pub trait Backend: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn device_info(&self) -> DeviceInfo;
    fn supports(&self, op: &Op) -> bool;
    fn dispatch(&self, op: &mut Op) -> Result<(), BackendError>;
}
```

**Acceptance:**
- [ ] Trait compiles
- [ ] `DeviceInfo { name, kind: DeviceKind, total_mem, arch: Option<String> }` defined
- [ ] `BackendError` enum covers: `Unsupported`, `Runtime(String)`, `OutOfMemory`, `DeviceLost`
- [ ] Document whether `dispatch` is blocking or async (recommend: blocking, backends queue internally)

**Deliverable:** `crates/modgrad-device/src/backend/mod.rs` with trait + error types.

---

## Phase 1 — Reference implementations (3 days)

### Task 1.1: CPU backend
**Owner:** @hdevalence · **Reviewer:** @redshiftzero · **Size:** 1.5 days · **Depends:** 0.1, 0.2

Always-works reference. Ground truth for later numerical regression tests.

**Acceptance:**
- [ ] Every `Op` variant handled (even if naive implementation via rayon)
- [ ] `supports()` returns true for all ops
- [ ] Matches existing CPU math in `modgrad-compute::ops` to f32 bit-precision

---

### Task 1.2: `BackendRegistry`
**Owner:** @hdevalence · **Reviewer:** @redshiftzero · **Size:** 0.5 day · **Depends:** 0.1, 0.2

Plain-old-data registry with dispatch loop:

```rust
pub struct BackendRegistry { backends: Vec<Box<dyn Backend>> }

impl BackendRegistry {
    pub fn detect() -> Self { /* probe KFD, CUDA, ROCm, Vulkan, CPU */ }
    pub fn dispatch(&self, op: &mut Op) -> Result<&'static str, BackendError> {
        for b in &self.backends {
            if b.supports(op) { b.dispatch(op)?; return Ok(b.name()); }
        }
        Err(BackendError::Unsupported)
    }
}
```

**Acceptance:**
- [ ] Fallback chain works: first-fit wins, errors propagate
- [ ] `detect()` orders backends by preference: KFD (if gfx1102) > ROCm > CUDA > Vulkan > CPU
- [ ] Environment override: `MODGRAD_BACKEND=cpu` forces CPU for debugging

---

### Task 1.3: Numerical regression harness
**Owner:** @redshiftzero · **Reviewer:** @hdevalence · **Size:** 1 day · **Depends:** 1.1

Golden tests that run every `Op` on the CPU backend and every other registered
backend, assert outputs match within tolerance. Runs in CI.

**Acceptance:**
- [ ] Test file `crates/modgrad-device/tests/backend_parity.rs`
- [ ] Covers all 22 op variants with small fixed inputs
- [ ] Tolerance: `1e-5` absolute / `1e-4` relative for f32 ops
- [ ] Skips backends not registered on the test host (no hard-fail when CUDA absent)
- [ ] Runs in < 30s on CPU-only host

---

## Phase 2 — Port existing backends (4 days)

### Task 2.1: Port KFD to `Backend` trait
**Owner:** @hdevalence · **Reviewer:** @redshiftzero · **Size:** 2 days · **Depends:** 1.1, 1.2, 1.3

The heaviest port — 25 existing kernels. Zero-regression requirement.

**Acceptance:**
- [ ] Existing kernel dispatch logic wrapped in `impl Backend for KfdBackend`
- [ ] `supports()` returns true only when `hsa_device_arch == "gfx1102"` AND the kernel exists for this op
- [ ] `backend_parity` test passes on gfx1102 hardware
- [ ] Maze benchmark (`mazes --brain --size 21 --steps 500`) produces **bit-identical** outputs to pre-refactor code (fixed seed)
- [ ] No perf regression > 2% on the 500-step maze run (measured via `time`)

---

### Task 2.2: Port CUDA to trait
**Owner:** @hdevalence · **Reviewer:** @cronokirby · **Size:** 0.5 day · **Depends:** 1.1

Already minimal (only `matvec`). Easy port. Everything else falls through via `supports()` returning false.

**Acceptance:**
- [ ] `impl Backend for CudaBackend` compiles under `--features cuda`
- [ ] `supports(Op::Matvec{..})` returns true, everything else false
- [ ] `backend_parity` passes on CUDA host (won't run in CI unless we add one)

---

### Task 2.3: Port Vulkan to trait
**Owner:** @hdevalence · **Reviewer:** @cronokirby · **Size:** 0.5 day · **Depends:** 1.1

Existing `gpu.rs` covers matvec + superlinear. Port with same pattern.

**Acceptance:**
- [ ] `impl Backend for VulkanBackend` compiles under `--features vulkan`
- [ ] `supports()` honest about what the existing shaders cover
- [ ] Doesn't regress on whatever hardware currently uses Vulkan

---

## Phase 3 — ROCm backend (3 days)

### Task 3.1: HIP runtime + device probe
**Owner:** @cronokirby · **Reviewer:** @hdevalence · **Size:** 1 day · **Depends:** 1.2

FFI to HIP runtime via `cubecl-hip-sys` (562k downloads, maintained by CubeCL team).

**Acceptance:**
- [ ] `cubecl-hip-sys = { optional = true }` in `modgrad-device/Cargo.toml` under `[features] rocm`
- [ ] `RocmDevice { ctx, ordinal, arch, total_mem }` with `new(ordinal)` working
- [ ] `init_global()` enumerates HIP devices
- [ ] Correctly detects gfx1102 in its arch string (so dispatch can defer to KFD)
- [ ] Works on host with ROCm 6.x or 7.x installed

---

### Task 3.2: rocBLAS matmul + matvec
**Owner:** @cronokirby · **Reviewer:** @hdevalence · **Size:** 1 day · **Depends:** 3.1, 0.1

Link `librocblas.so`. Implement `Op::Matmul` via `rocblas_sgemm` and `Op::Matvec` via `rocblas_sgemv`.

**Acceptance:**
- [ ] `impl Backend for RocmBackend` implements Matmul + Matvec
- [ ] `supports()` returns true for those two ops
- [ ] `backend_parity` test passes (requires gfx1102 box with both KFD and ROCm available — run on @hdevalence's RX 7600)

---

### Task 3.3: MIOpen norm + activation
**Owner:** @cronokirby · **Reviewer:** @hdevalence · **Size:** 1 day · **Depends:** 3.1, 0.1

Link `libmiopen.so`. Implement `LayerNormFwd/Bwd`, `SiluFwd/Bwd`, `GluFwd/Bwd` via MIOpen ops.

**Acceptance:**
- [ ] Above ops added to `supports()` + dispatch
- [ ] `backend_parity` still passes

---

## Phase 4 — Caller migration (3 days)

### Task 4.1: Port `modgrad-ctm::train.rs` to dispatch
**Owner:** @eowigley · **Reviewer:** @hdevalence · **Size:** 1.5 days · **Depends:** 2.1, 2.2, 2.3

Every direct kernel call in `crates/modgrad-ctm/src/train.rs` becomes
`registry.dispatch(&mut Op::X { .. })`. Keep the old functions as deprecated
wrappers for one minor version to let downstream callers migrate.

**Acceptance:**
- [ ] Zero direct kernel calls remain in `modgrad-ctm`
- [ ] Maze benchmark produces bit-identical outputs at seed=42 pre/post
- [ ] `cargo test -p modgrad-ctm` passes
- [ ] `cargo bench` on the existing maze path shows < 2% regression

---

### Task 4.2: Port `modgrad-ffn` to dispatch
**Owner:** @eowigley · **Reviewer:** @hdevalence · **Size:** 0.5 day · **Depends:** 4.1

Same pattern. Smaller surface.

**Acceptance:**
- [ ] `cargo test -p modgrad-ffn` passes
- [ ] No direct kernel calls remain

---

### Task 4.3: Dispatch preference tuning + benchmark re-run
**Owner:** @redshiftzero · **Reviewer:** @hdevalence · **Size:** 1 day · **Depends:** 4.1, 4.2, 3.3

Verify at runtime that gfx1102 picks KFD for every op KFD supports, and ROCm
for everything else. Re-run the 3-seed brain-vs-single benchmark from the
README; verify numbers match within 1σ of pre-refactor results.

**Acceptance:**
- [ ] Log output on gfx1102 shows `backend=kfd` for all CTM hot-path ops
- [ ] Log output on non-gfx1102 AMD (if obtainable) shows `backend=rocm`
- [ ] README numbers updated if material change

---

## Phase 5 — Documentation (0.5 day)

### Task 5.1: Update README backend section
**Owner:** @hdevalence · **Reviewer:** @redshiftzero · **Size:** 0.5 day · **Depends:** 4.3

Replace the current "hand-written RDNA3 kernels" line with the layered story:
KFD fast-path for gfx1102 as reference arch; ROCm portable path for any AMD;
CUDA for NVIDIA; Vulkan cross-vendor fallback; CPU always.

**Acceptance:**
- [ ] README reflects actual shipping backends
- [ ] `docs/backends.md` written (optional — new file explaining the `Op`/`Backend` architecture for external contributors)

---

## Timeline

**2 weeks focused.** Phases 0–2 can overlap by 1 day. Phases 3 and 4.1 are partially
parallel. Phase 4.3 is the gate — no claim of "ships with ROCm support" until that's green.

| Week | Days | Deliverables |
|---|---|---|
| 1 | Mon–Tue | Phase 0 (design) |
| 1 | Wed–Fri | Phase 1 (reference + harness) |
| 2 | Mon–Tue | Phase 2 (existing backends port) |
| 2 | Wed–Thu | Phase 3 (ROCm) |
| 2 | Fri | Phase 4 (migration) + Phase 5 (docs) |

## Risk register

| Risk | Mitigation |
|---|---|
| `cubecl-hip-sys` API regresses in a new release | Pin to exact version in Cargo.toml; bump deliberately |
| KFD port introduces numerical drift | Task 1.3 harness is the test; gate Phase 2 on it passing |
| Dispatch overhead slows hot path | Inline-hot path in BackendRegistry; only check first-matching backend in a static per-op cache |
| ROCm tested only on gfx1102 | Document as "tested on gfx1102; other ROCm archs are best-effort in phase 1; phase 2 adds CI on MI-series" — explicit scope |
| Caller migration is invasive | Deprecation shim: keep old function names, route internally to dispatch |

## Out of scope for this refactor

- ROCm Q4_K matvec (stays KFD-only until demand)
- Per-device autotuner (backend picks op variant internally)
- Multi-device sharding (follow-up)
- Dynamic code generation (Triton-style) — architectural door left open, not walked through

---

## Reviewer checklist (for each PR)

- [ ] Code compiles under relevant feature flags
- [ ] `backend_parity` test passes on available backends
- [ ] No new `unwrap()` on paths reachable from user code (use `?` + `BackendError`)
- [ ] All public types have at least a one-line doc
- [ ] No dangling `TODO` without an assignee
- [ ] Cargo.toml pins new dep versions exactly
