# KFD kernel bugs — proptest-surfaced regressions gated in `supports()`

Each row: Op variant, proptest seed (if captured), shape that fails, Δ
magnitude, hypothesis, re-enable criterion.

Seeds below cite `crates/modgrad-device/tests/backend_proptest.proptest-regressions`.
Gate source: `crates/modgrad-device/src/backend/kfd.rs::supports()`
(lines ~85–160 at the time of writing).

---

## Matvec (small shapes)

- **Shape**: out_dim=6, in_dim=9 (and other sub-32 combos)
- **Seed**: `2019355988255635502` (shrinks to out_dim=6, in_dim=9)
- **Δ**: unbounded at small shapes; production probes pass at
  256×64, 1024×64, 4096×64, 8448×64.
- **Hypothesis**: KFD matvec kernel assumes wavefront-aligned workload
  and in-dim ≥ 32; below that the tail path diverges.
- **Re-enable criterion**: kernel audit + tail fix; until then gate is
  `out_dim >= 32 && in_dim >= 32`.

## MatvecT (small shapes)

- **Shape**: same regime as Matvec
- **Seed**: not independently captured; shares the Matvec regression class.
- **Δ**: unbounded at small shapes.
- **Hypothesis**: shared tail-path issue with `Matvec`; transpose variant
  re-uses the same dispatcher template.
- **Re-enable criterion**: fix with Matvec; gate is
  `out_dim >= 32 && in_dim >= 32`.

## MatmulNN (alignment)

- **Shape**: any `(m, k, n)` where `n % 32 != 0`, `k % 8 != 0`, or `m % 128 != 0`
- **Seed**: seed not captured in regressions file (hard precondition, caught
  deterministically at boot by parity cases).
- **Δ**: kernel errors out / garbage output on mis-aligned shapes.
- **Hypothesis**: hard alignment preconditions baked into the tiled matmul
  ISA — not a numerical drift, an unconditional kernel requirement.
- **Re-enable criterion**: not a bug per se; gate `n % 32 == 0 && k % 8 == 0
  && m % 128 == 0` is the correct expression of the kernel contract.

## OuterProductAcc

- **Shape**: m=20, n=13 (non-power-of-2 n)
- **Seed**: `0` (shrinks to m=20, n=13)
- **Δ**: unbounded — "many mid-sized shapes"
- **Hypothesis**: float-div row/col approximation fails when n isn't a
  power of two.
- **Re-enable criterion**: fix kernel's row/col calculation, or widen
  `supports()` to drop the `n.is_power_of_two()` requirement. Currently
  gated as `m >= 16 && n.is_power_of_two() && n >= 32`.

## TraceRotateInplace

- **Shape**: (d_model=3, memory_length=27); also (d_model=10, memory_length=32)
- **Seeds**:
  - `11108682193298569517` → d_model=3, memory_length=27
  - `15920992692462394518` → d_model=10, memory_length=32
- **Δ**: divergence at non-wavefront-aligned totals.
- **Hypothesis**: tail handling when `d_model * memory_length % 32 != 0`.
- **Re-enable criterion**: kernel tail fix; gate is
  `d_model * memory_length >= 32 && (d_model * memory_length) % 32 == 0`.
  Production shapes (32×8, 512×64) clear this.

## SuperLinearFwd

- **Shape**: d_model=1, memory_length=2, out_per=1; also d_model=9,
  memory_length=4, out_per=3
- **Seeds**:
  - `8413577157799323814` → d_model=1, memory_length=2, out_per=1
  - `7659530290609023813` → d_model=9, memory_length=4, out_per=3
  - `7774585889669485863` → d_model=32, memory_length=8, out_per=4
- **Δ**: proptest divergence at small d_model / out_per combos.
- **Hypothesis**: boot-probe shapes clear; small-combo failures likely
  per-wavefront reduction drift.
- **Re-enable criterion**: gate is
  `d_model >= 16 && out_per >= 2 && memory_length >= 8`.

## Elementwise tail handling (Silu / Glu / AdamW / SgdUpdate)

- **Shape**: non-wavefront-aligned lengths (e.g. len=59, len=96, len=224,
  half=64, half=98)
- **Seeds**:
  - `5032973736253505565` → len=59, step=16
  - `11705766536134547691` → half=98
  - `13594252235897356635` → half=64
  - `4438135088035341265` → len=224
  - `7286245980069229257` → len=96
- **Δ**: divergence at non-wavefront-aligned sizes.
- **Hypothesis**: kernels assume `len % 32 == 0` (GLU: `len % 64 == 0`)
  and don't execute a tail wavefront correctly.
- **Re-enable criterion**: fix tail-wavefront dispatch; gates are
  `len >= 32 && len % 32 == 0` for Silu/AdamW/SgdUpdate and
  `len >= 64 && len % 64 == 0` for GluFwd.

## SynapseForward (fused) — Stage 1 newcomer

- **Shape**: out_dim=32, in_dim=96
- **Seed**: `12390942540437353913`
- **Δ**: |Δ|≈1.5e-4 (rel 1.5e-4), against committed tolerance max_abs=2e-5,
  max_rel=1e-4.
- **Hypothesis**: fused KFD synapse_forward accumulates drift past f32
  tolerance at small output shapes — parallel reduction + approximate
  division stack.
- **Re-enable criterion**: land CPU-only fused impl now (per plan), KFD
  fusion unlocked by a future kernel-fix commit. Currently
  `Op::SynapseForward { .. } => false`.

## LayerNormInplace — Stage 1 newcomer

- **Shape**: n_cols=96
- **Seed**: `556123943721261328`
- **Δ**: |Δ|≈3.96e-4 (rel 3.96e-4), against committed tolerance max_abs=2e-5,
  max_rel=1e-4.
- **Hypothesis**: single-WG `layer_norm_fwd` kernel's parallel reduction
  pushes drift past f32 tolerance at small n_cols.
- **Re-enable criterion**: kernel fix or tolerance renegotiation; currently
  `Op::LayerNormInplace { .. } => false`.

## Deferred — require Op/kernel alignment first (not numeric bugs)

- **LayerNormFwd**: KFD's inplace kernel doesn't emit a `cache` tensor
  required by the Op variant's backward handoff. Blocked on kernel
  surface change.
- **SyncUpdateFwd**: KFD's kernel expects `alpha/beta/phases/dopamine`
  arguments that the Op struct doesn't carry. Blocked on Op enum extension.
- **SuperLinearBwd{Dw,Dx}**: arg-shape mismatch between Op variant and
  KFD's existing dispatcher. Blocked on dispatcher rework.
- **Seeds**: N/A — these never reach proptest, they're rejected at the
  Op match arm before dispatch.
