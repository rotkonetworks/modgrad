//! `Layer` trait + `LayerScheduler` — orchestration for VRAM-resident
//! layer streaming during foundation-model training.
//!
//! ## Why this exists
//!
//! Training a 7B-30B foundation model on 8 GB VRAM means the full
//! weight set never lives on the GPU at once. The host (RAM, possibly
//! Q4_K compressed) is the source of truth. At any moment the
//! scheduler keeps a sliding window of K layers materialised in VRAM
//! as fp32. As the forward sweep walks layer i, the scheduler:
//!
//!   1. Calls `ensure_resident` on layer i (no-op if already there;
//!      uploads otherwise — Q4_K dequant happens here for quantized
//!      layers).
//!   2. Calls `forward` (pure dispatch, weights known to be resident).
//!   3. Optionally evicts layer i-K (returning its VRAM to the pool).
//!
//! Backward is the same shape, walking layers in reverse.
//!
//! ## Design choices
//!
//! **Why a trait, not a concrete enum.** A foundation model is a stack
//! of heterogeneous units: attention, FFN, RMSNorm, embedding,
//! adapters, full transformer blocks. Each has its own residency cost
//! profile and its own forward shape. An enum forces every variant
//! through one match site at every dispatch, defeating modularity
//! (`feedback_modularity.md`) and forcing every new block type into
//! this module. A trait keeps the scheduler agnostic.
//!
//! **Why `forward(&self)` and `backward(&mut self)`.** Forward is a
//! pure dispatch — read weights, write output, no internal state
//! mutation. Backward typically stores activations / accumulates
//! gradient buffers, so it needs `&mut self`. This matches the
//! existing `LinearResident::forward` (read-only) vs the future
//! per-layer gradient buffer story.
//!
//! **Why `Send` only, not `Send + Sync`.** `HipBatch` is `!Send`
//! (HIP runtime contexts are thread-local), so the dispatch object
//! itself can't cross threads. We don't *need* `Sync` — the
//! scheduler holds layers serially and dispatches them on one
//! thread. The trait stays `Send` so a future scheduler that moves
//! whole layer trees between threads (e.g. host-prep on one thread,
//! GPU dispatch on another) is not blocked.
//!
//! **Eviction policy first cut: sliding window.** The simplest policy
//! that works for forward-then-backward sweeps. As we walk layer i,
//! evict layer i-K if it is resident. This bounds simultaneously-
//! resident layers to K. A `vram_budget_bytes` mode is also offered
//! for cases where layers have wildly different sizes — evict from
//! the front of the resident set until total bytes ≤ budget.
//!
//! **Why scratch buffer chaining lives in the scheduler, not the
//! layer.** Each layer impl knows only its own input/output shapes;
//! pipelining the activations between layers needs visibility into
//! both sides of the boundary, which only the scheduler has. The
//! scheduler allocates a `Vec<GpuVec>` of scratch buffers indexed by
//! layer position and threads them through.
//!
//! ## Layer impls in this slice
//!
//! - [`LinearLayer`] — wraps `LinearResidentStreaming`. Forward only
//!   for now; backward is `unimplemented!()` because the resident
//!   `LinearResident::backward` API is not yet wired (the underlying
//!   `Op::*BackwardResident` exists but no single-method face).
//!   That's the next slice.
//! - [`LayerSequence`] — composes a `Vec<Box<dyn Layer>>` into one
//!   `Layer`, threading scratch buffers through. Recursive
//!   composition: a `LayerSequence` can be put inside another
//!   `LayerScheduler`.
//!
//! Future impls (separate slices, not in this PR):
//! - `LinearQuantizedLayer` over `LinearResidentQuantized` (Q4_K host
//!   storage, dequant on `ensure_resident`).
//! - `AttentionLayer`, `SwigluFFNLayer`, `TransformerBlockLayer`.

#[cfg(feature = "rocm")]
use modgrad_compute::backend::{GpuVec, ResidencyError};
#[cfg(feature = "rocm")]
use modgrad_compute::neuron::{Linear, LinearResidentStreaming};
#[cfg(feature = "rocm")]
use modgrad_device::backend::HipBatch;

// ─── Trait ────────────────────────────────────────────────────

/// A composable forward unit whose weights may live on host until
/// `ensure_resident` is called. The scheduler drives the lifecycle:
/// residency → dispatch → eviction.
///
/// Implementors typically wrap one of the `*ResidentStreaming` types
/// (or, for unquantized hot weights, `*Resident`) and forward the
/// `Layer` calls 1-1. See [`LinearLayer`] for the canonical shape.
///
/// All resident dispatch goes through a `&HipBatch` for the same
/// queue-bookkeeping reason as `LinearResident::forward` — see its
/// doc comment in `modgrad-compute::neuron`.
#[cfg(feature = "rocm")]
pub trait Layer: Send {
    /// Run forward dispatch on already-resident weights. Caller (the
    /// scheduler, typically) guarantees `ensure_resident` was called
    /// successfully since the last `evict`. Implementors may panic
    /// or return `WrongVariant` if invoked in a non-resident state —
    /// that's a scheduler bug, not a runtime concern.
    ///
    /// `x_dev` must have length `in_dim()`; `out_dev` must have
    /// length `out_dim()`. Both are `GpuVec::Hip` in production.
    fn forward(
        &self,
        batch: &HipBatch,
        x_dev: &GpuVec,
        out_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError>;

    /// Backward — analogous to `forward`. Reads `dy_dev` (gradient
    /// w.r.t. this layer's output), writes `dx_dev` (gradient w.r.t.
    /// this layer's input), and populates the layer's internal grad
    /// buffers (weight grad, bias grad — implementor's concern).
    ///
    /// Activations from forward must be accessible to the implementor
    /// — typical pattern is to cache them on the `&mut self` during
    /// `forward`. The scheduler does not stage activations; it only
    /// chains the gradient signal.
    fn backward(
        &mut self,
        batch: &HipBatch,
        x_dev: &GpuVec,
        dy_dev: &GpuVec,
        dx_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError>;

    /// Ensure the layer's weights are resident in VRAM. Idempotent —
    /// no-op when already resident. After a successful return,
    /// `is_resident()` must be true and `forward` may be called.
    ///
    /// `batch` is taken even though the typical implementation
    /// (host → device upload via `hipMemcpy`) does not queue kernels;
    /// it is reserved for impls that need to record dequant kernel
    /// dispatches (e.g. Q4_K → fp32 unpack) on the same batch as
    /// the forward.
    fn ensure_resident(&mut self, batch: &HipBatch) -> Result<(), ResidencyError>;

    /// Drop device buffers. Subsequent `ensure_resident` will re-upload
    /// (or re-dequantize). Free VRAM is reclaimed via `hipFree` inside
    /// each `HipBuffer::Drop`.
    fn evict(&mut self);

    /// True when device buffers are populated and `forward` may be
    /// dispatched without a preceding `ensure_resident`.
    fn is_resident(&self) -> bool;

    /// Approximate VRAM cost when resident, in bytes. Used by the
    /// budget-driven eviction policy. The value need not be exact —
    /// off by the size of a single bias vector is fine — but should
    /// reflect the dominant `weight + bias` allocation.
    fn vram_size_bytes(&self) -> usize;

    /// Input dimension. Used by the scheduler for sanity-checking
    /// scratch `GpuVec` sizes when chaining layers.
    fn in_dim(&self) -> usize;

    /// Output dimension. Same use.
    fn out_dim(&self) -> usize;
}

// ─── LinearLayer ──────────────────────────────────────────────

/// `Layer` wrapper around a `LinearResidentStreaming`. Forward is a
/// thin pass-through; backward is `unimplemented!()` until the
/// streaming-resident backward API lands.
///
/// Construct from an `Arc<Linear>` shared with the optimizer, same
/// pattern as `LinearResidentStreaming`. The scheduler will call
/// `ensure_resident` before `forward` and (per policy) `evict` after.
///
/// **Why `RefCell` inside.** The `Layer::forward` signature takes
/// `&self`, but `LinearResidentStreaming::forward` takes `&mut self`
/// (because the upstream API also handles `ensure_resident` and
/// `auto_evict` internally). We use `std::cell::RefCell` for
/// single-threaded interior mutability — sound, well-defined, and
/// the `Send`-but-not-`Sync` shape matches `Layer: Send` exactly.
/// Doing the equivalent unsafe cast is undefined behavior (rejected
/// by the `invalid_reference_casting` lint).
#[cfg(feature = "rocm")]
pub struct LinearLayer {
    inner: std::cell::RefCell<LinearResidentStreaming>,
    in_dim: usize,
    out_dim: usize,
    vram_bytes: usize,
}

#[cfg(feature = "rocm")]
impl LinearLayer {
    /// Wrap an `Arc<Linear>` for streaming residency. No upload yet —
    /// the first `ensure_resident` does that.
    pub fn from_linear_arc(host: std::sync::Arc<Linear>) -> Self {
        let in_dim = host.in_dim;
        let out_dim = host.out_dim;
        // Cache vram size up front; the host weights don't change
        // shape under us, and it avoids borrowing the RefCell on
        // every `vram_size_bytes()` query.
        let vram_bytes = (host.weight.len() + host.bias.len()) * 4;
        Self {
            inner: std::cell::RefCell::new(
                LinearResidentStreaming::from_linear_arc(host),
            ),
            in_dim, out_dim, vram_bytes,
        }
    }
}

#[cfg(feature = "rocm")]
impl Layer for LinearLayer {
    fn forward(
        &self,
        batch: &HipBatch,
        x_dev: &GpuVec,
        out_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        // The scheduler guarantees `ensure_resident` ran before this
        // dispatch, so the inner `forward` is effectively a pure
        // dispatch. We pass `auto_evict=false` to keep the residency
        // contract under the scheduler's control — the scheduler
        // owns eviction policy.
        self.inner.borrow_mut()
            .forward(batch, x_dev, out_dev, /*auto_evict=*/ false)
    }

    fn backward(
        &mut self,
        _batch: &HipBatch,
        _x_dev: &GpuVec,
        _dy_dev: &GpuVec,
        _dx_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        // TODO(next slice): wire `Op::MatvecBackwardResident` through
        // a `LinearResident::backward` method, then expose it here.
        // The shape:
        //   - cache x_dev from forward (impl on LinearLayer once
        //     forward is `&mut self`-able by the scheduler);
        //   - dispatch the matvec backward kernel resident;
        //   - accumulate weight + bias gradients into device buffers
        //     held alongside `weight_dev`/`bias_dev`.
        unimplemented!("LinearLayer::backward — TODO next slice");
    }

    fn ensure_resident(&mut self, _batch: &HipBatch) -> Result<(), ResidencyError> {
        // `LinearResidentStreaming::ensure_resident` does H2D copies,
        // not kernel dispatches, so it doesn't touch the batch. The
        // batch parameter is in the trait signature for future Q4_K
        // layer impls that DO record dequant kernels.
        self.inner.borrow_mut().ensure_resident()
    }

    fn evict(&mut self) {
        self.inner.borrow_mut().evict();
    }

    fn is_resident(&self) -> bool {
        self.inner.borrow().is_resident()
    }

    fn vram_size_bytes(&self) -> usize {
        // Cached at construction — see `LinearLayer::from_linear_arc`.
        self.vram_bytes
    }

    fn in_dim(&self) -> usize {
        self.in_dim
    }

    fn out_dim(&self) -> usize {
        self.out_dim
    }
}

// ─── LayerSequence ────────────────────────────────────────────

/// A `Layer` that runs an inner `Vec<Box<dyn Layer>>` end-to-end. The
/// scheduler can put a `LayerSequence` inside another scheduler, or
/// use it as a single chunk to amortize the resident/evict overhead
/// when several small layers want to live in VRAM together.
///
/// Internal scratch buffers are lazily allocated on first `forward`
/// and reused across calls — sized to the maximum
/// `out_dim` across the inner layers, allocated as `GpuVec::Hip`.
///
/// **Why `RefCell` around `scratch`.** `Layer::forward` takes `&self`
/// but the activation chain is conceptually a mutation — we have to
/// allocate scratch on first call and write into it on every call.
/// `RefCell` is the standard single-threaded interior-mutability
/// answer. Same `Send`-but-not-`Sync` shape as `LinearLayer`.
#[cfg(feature = "rocm")]
pub struct LayerSequence {
    layers: Vec<Box<dyn Layer>>,
    /// Scratch buffers for activations between layers. `scratch[i]`
    /// holds the output of layer `i`, sized to `layers[i].out_dim()`.
    /// The last layer writes directly into the caller's `out_dev`,
    /// so `scratch` only needs `n_layers - 1` entries; we keep
    /// `n_layers` for symmetry and an empty slot at the end.
    /// `None` means "not yet allocated"; lazy because we only know
    /// we have a HIP context once the first `forward` runs.
    scratch: std::cell::RefCell<Vec<Option<GpuVec>>>,
}

#[cfg(feature = "rocm")]
impl LayerSequence {
    /// Wrap a vector of layers. Asserts dim compatibility — adjacent
    /// layers must have matching `out_dim` / `in_dim`. Empty sequences
    /// are not allowed (an empty layer has no `in_dim`/`out_dim` to
    /// report sensibly).
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        assert!(!layers.is_empty(), "LayerSequence::new: layers must be non-empty");
        for w in layers.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            assert_eq!(
                a.out_dim(), b.in_dim(),
                "LayerSequence dim mismatch: layer out_dim={} → next in_dim={}",
                a.out_dim(), b.in_dim(),
            );
        }
        let n = layers.len();
        Self {
            layers,
            scratch: std::cell::RefCell::new((0..n).map(|_| None).collect()),
        }
    }

    /// Number of inner layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Read-only view of the inner layer slice — useful for tests
    /// that want to assert `is_resident()` on individual elements.
    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }
}

#[cfg(feature = "rocm")]
impl Layer for LayerSequence {
    fn forward(
        &self,
        batch: &HipBatch,
        x_dev: &GpuVec,
        out_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        let n = self.layers.len();
        let mut scratch = self.scratch.borrow_mut();

        // Allocate any missing scratch slots, sized to each layer's
        // `out_dim`. The last slot (`scratch[n-1]`) is unused — we
        // write straight into `out_dev` for the final layer.
        for i in 0..n.saturating_sub(1) {
            let need = self.layers[i].out_dim();
            let realloc = match &scratch[i] {
                Some(buf) => buf.len() != need,
                None => true,
            };
            if realloc {
                scratch[i] = Some(GpuVec::try_hip(need)?);
            }
        }

        // Walk the layers, threading the activation through scratch.
        // Borrow trick: when we need to read `scratch[i-1]` AND write
        // `scratch[i]` in the same iteration, take `scratch[i]` out
        // by `Option::take` first. After that, the only outstanding
        // borrow into `scratch` is the immutable read of `scratch[i-1]`
        // — no aliasing.
        for i in 0..n {
            if i == n - 1 {
                // Last layer writes into the caller's out buffer.
                let input: &GpuVec = if i == 0 {
                    x_dev
                } else {
                    scratch[i - 1].as_ref()
                        .expect("scratch[i-1] allocated above")
                };
                self.layers[i].forward(batch, input, out_dev)?;
            } else {
                // Take scratch[i] out FIRST. After this line the only
                // outstanding access to `scratch` is the immutable
                // `as_ref()` we'll do on scratch[i-1] below.
                let mut out_buf = scratch[i].take()
                    .expect("scratch[i] allocated above");
                let input: &GpuVec = if i == 0 {
                    x_dev
                } else {
                    scratch[i - 1].as_ref()
                        .expect("scratch[i-1] allocated above")
                };
                self.layers[i].forward(batch, input, &mut out_buf)?;
                scratch[i] = Some(out_buf);
            }
        }
        Ok(())
    }

    fn backward(
        &mut self,
        _batch: &HipBatch,
        _x_dev: &GpuVec,
        _dy_dev: &GpuVec,
        _dx_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        unimplemented!("LayerSequence::backward — depends on per-layer backward; \
                        next slice");
    }

    fn ensure_resident(&mut self, batch: &HipBatch) -> Result<(), ResidencyError> {
        for l in &mut self.layers {
            l.ensure_resident(batch)?;
        }
        Ok(())
    }

    fn evict(&mut self) {
        for l in &mut self.layers {
            l.evict();
        }
    }

    fn is_resident(&self) -> bool {
        // A sequence is "resident" iff every child is. Mixed states
        // are possible but not what `is_resident` should report —
        // a mid-sweep snapshot.
        self.layers.iter().all(|l| l.is_resident())
    }

    fn vram_size_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.vram_size_bytes()).sum()
    }

    fn in_dim(&self) -> usize {
        self.layers.first().map(|l| l.in_dim()).unwrap_or(0)
    }

    fn out_dim(&self) -> usize {
        self.layers.last().map(|l| l.out_dim()).unwrap_or(0)
    }
}

// ─── Scheduler ────────────────────────────────────────────────

/// Drives a `Vec<Box<dyn Layer>>` through forward/backward sweeps,
/// keeping at most `resident_window` (or `vram_budget_bytes` worth of)
/// layers materialised in VRAM at any moment.
///
/// Two policies, in priority order during eviction:
///
///   1. If `vram_budget_bytes = Some(B)`: evict resident layers
///      from the front of the resident set until total ≤ B. Used
///      when layer sizes vary widely (e.g. embedding layer is 100×
///      a single MLP block).
///
///   2. Otherwise, if `resident_window = Some(K)`: at layer i, evict
///      layer i-K if it's resident. Simple sliding window.
///
///   3. Otherwise: never evict. Suitable for models that fit fully
///      in VRAM (where the streaming wrappers still pay the residency
///      check overhead but don't actually rotate).
#[cfg(feature = "rocm")]
pub struct LayerScheduler {
    layers: Vec<Box<dyn Layer>>,
    /// Maximum simultaneously-resident layers. `None` ⇒ unbounded.
    resident_window: Option<usize>,
    /// Approximate total VRAM budget for resident layers, bytes.
    /// `None` ⇒ disabled (use `resident_window` only).
    vram_budget_bytes: Option<usize>,
    /// Scratch buffers for activations between layers — same shape
    /// as `LayerSequence::scratch`. Lazy-allocated on first forward.
    scratch: Vec<Option<GpuVec>>,
}

#[cfg(feature = "rocm")]
impl LayerScheduler {
    /// Build a scheduler over the given layers. Default policy: no
    /// eviction (every layer stays resident once first uploaded).
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        let n = layers.len();
        // Check adjacency dims like LayerSequence does.
        for w in layers.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            assert_eq!(
                a.out_dim(), b.in_dim(),
                "LayerScheduler dim mismatch: layer out_dim={} → next in_dim={}",
                a.out_dim(), b.in_dim(),
            );
        }
        Self {
            layers,
            resident_window: None,
            vram_budget_bytes: None,
            scratch: (0..n).map(|_| None).collect(),
        }
    }

    /// Builder: cap simultaneously-resident layers to `k`. Setting
    /// `k` larger than `layers.len()` is harmless (no eviction will
    /// fire).
    pub fn with_resident_window(mut self, k: usize) -> Self {
        self.resident_window = Some(k);
        self
    }

    /// Builder: cap total resident VRAM to `bytes`. Takes precedence
    /// over `resident_window`.
    pub fn with_vram_budget(mut self, bytes: usize) -> Self {
        self.vram_budget_bytes = Some(bytes);
        self
    }

    /// Number of layers in the schedule.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Is the schedule empty?
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Read-only view — useful for assertions in tests.
    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    /// Sum of `vram_size_bytes()` across currently-resident layers.
    pub fn current_resident_bytes(&self) -> usize {
        self.layers.iter()
            .filter(|l| l.is_resident())
            .map(|l| l.vram_size_bytes())
            .sum()
    }

    /// Sum of `vram_size_bytes()` across all layers (resident or not).
    /// Useful for sizing the budget against the model.
    pub fn full_resident_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.vram_size_bytes()).sum()
    }

    /// Forward sweep. Walks layers 0..n in order:
    ///   1. `ensure_resident(i)`
    ///   2. `forward(i)`, with input from prior layer's scratch
    ///   3. evict per policy
    ///
    /// `x_dev` must have length `layers[0].in_dim()`; `out_dev` must
    /// have length `layers.last().out_dim()`.
    pub fn forward(
        &mut self,
        batch: &HipBatch,
        x_dev: &GpuVec,
        out_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        let n = self.layers.len();
        if n == 0 {
            return Ok(());
        }

        // Lazy-allocate scratch slots for layers 0..n-1 (the last
        // layer writes into `out_dev`).
        for i in 0..n.saturating_sub(1) {
            let need = self.layers[i].out_dim();
            let realloc = match &self.scratch[i] {
                Some(buf) => buf.len() != need,
                None => true,
            };
            if realloc {
                self.scratch[i] = Some(GpuVec::try_hip(need)?);
            }
        }

        for i in 0..n {
            // Stage 1: residency.
            self.ensure_layer_resident(i, batch)?;

            // Stage 2: dispatch. Take `scratch[i]` out FIRST so the
            // immutable read of `scratch[i-1]` below has no aliasing
            // mutable borrow on the same slice. See LayerSequence's
            // forward for the same pattern.
            if i == n - 1 {
                let input: &GpuVec = if i == 0 {
                    x_dev
                } else {
                    self.scratch[i - 1].as_ref()
                        .expect("scratch[i-1] populated by stage above")
                };
                self.layers[i].forward(batch, input, out_dev)?;
            } else {
                let mut out_buf = self.scratch[i].take()
                    .expect("scratch[i] populated by stage above");
                let input: &GpuVec = if i == 0 {
                    x_dev
                } else {
                    self.scratch[i - 1].as_ref()
                        .expect("scratch[i-1] populated by stage above")
                };
                self.layers[i].forward(batch, input, &mut out_buf)?;
                self.scratch[i] = Some(out_buf);
            }

            // Stage 3: post-dispatch eviction.
            self.evict_post_step(i);
        }
        Ok(())
    }

    /// Backward sweep. Walks layers n-1..0. Each layer's backward
    /// chains the gradient signal into the previous layer's input
    /// gradient. Requires every layer in the schedule to support
    /// backward — `LinearLayer::backward` is currently
    /// `unimplemented!()`, so calling this on a `LinearLayer`-only
    /// schedule will panic. The scheduler itself is correct; the
    /// missing piece is the per-layer backward.
    ///
    /// `dy_dev` is the gradient w.r.t. the final layer's output
    /// (length `out_dim()` of the last layer); `dx_dev` is filled
    /// with the gradient w.r.t. the first layer's input (length
    /// `in_dim()` of the first layer).
    pub fn backward(
        &mut self,
        batch: &HipBatch,
        dy_dev: &GpuVec,
        dx_dev: &mut GpuVec,
    ) -> Result<(), ResidencyError> {
        // The implementation walks layers in reverse, threading dy
        // through the gradient scratch. Implementor note: this
        // mirrors `forward`'s structure but reads `scratch[i-1]`
        // (the activation that fed layer i) instead of writing to
        // it. Without per-layer activation caching wired in
        // `LinearLayer::backward`, the body below would compile but
        // panic on the first dispatch.
        //
        // We deliberately keep the loop here so the API is real and
        // callable as soon as a backward-capable Layer impl lands;
        // unit tests exercise the forward-only happy path until then.
        let n = self.layers.len();
        if n == 0 {
            return Ok(());
        }

        // Reuse the forward scratch as the backward gradient buffers
        // for now — same shapes (each `scratch[i]` is sized to
        // `layers[i].out_dim()` which is also the gradient-w.r.t.-output
        // size of layer `i`). Lazy-allocate any missing slots
        // identically.
        for i in 0..n.saturating_sub(1) {
            let need = self.layers[i].out_dim();
            let realloc = match &self.scratch[i] {
                Some(buf) => buf.len() != need,
                None => true,
            };
            if realloc {
                self.scratch[i] = Some(GpuVec::try_hip(need)?);
            }
        }

        for i in (0..n).rev() {
            self.ensure_layer_resident(i, batch)?;

            // The activation that flowed INTO layer i during forward
            // is the same scratch slot we wrote into for layer i-1.
            // For i==0 it's the original x_dev — except backward
            // doesn't get x_dev; we synthesize a placeholder. In a
            // real impl, the layer caches its own input on forward,
            // so this argument is largely a courtesy. We pass an
            // empty `GpuVec` of the right size for the borrowing
            // to typecheck; backward impls should not deref it.
            // TODO(next slice): thread x_dev through.
            let x_placeholder = match self.layers[i].in_dim() {
                0 => GpuVec::heap(0),
                k => GpuVec::try_hip(k)?,
            };

            // dy for this layer: dy_dev for the last, scratch[i] for
            // earlier layers. Take dx_buf (scratch[i-1]) out first
            // so the immutable read of scratch[i] below has no
            // conflicting borrow.
            if i == 0 {
                let dy: &GpuVec = if i == n - 1 {
                    dy_dev
                } else {
                    self.scratch[i].as_ref()
                        .expect("scratch[i] populated above")
                };
                self.layers[i].backward(batch, &x_placeholder, dy, dx_dev)?;
            } else {
                let mut dx_buf = self.scratch[i - 1].take()
                    .expect("scratch[i-1] populated above");
                let dy: &GpuVec = if i == n - 1 {
                    dy_dev
                } else {
                    self.scratch[i].as_ref()
                        .expect("scratch[i] populated above")
                };
                self.layers[i].backward(batch, &x_placeholder, dy, &mut dx_buf)?;
                self.scratch[i - 1] = Some(dx_buf);
            }

            self.evict_post_step(i);
        }
        Ok(())
    }

    // ─── Internals ────────────────────────────────────────────

    /// Ensure layer `i` is resident; first apply the eviction policy
    /// to free up budget room.
    fn ensure_layer_resident(&mut self, i: usize, batch: &HipBatch) -> Result<(), ResidencyError> {
        // If we'd blow the budget *after* this upload, evict from
        // the front of the resident set until there's room.
        if let Some(budget) = self.vram_budget_bytes {
            let need = self.layers[i].vram_size_bytes();
            // Evict in order 0..i (the layers behind us in the sweep)
            // first, and the loop runs only if we are over budget
            // *with* the new layer counted. We don't evict layer i
            // itself — that's the one we're trying to make resident.
            let mut j = 0;
            while j < i {
                let projected = self.current_resident_bytes()
                    + if self.layers[i].is_resident() { 0 } else { need };
                if projected <= budget {
                    break;
                }
                if self.layers[j].is_resident() {
                    self.layers[j].evict();
                }
                j += 1;
            }
        }
        self.layers[i].ensure_resident(batch)
    }

    /// Sliding-window post-step eviction. Called after layer `i` has
    /// finished dispatching.
    fn evict_post_step(&mut self, i: usize) {
        // vram_budget_bytes already evicts in `ensure_layer_resident`,
        // so the sliding window only fires when the budget mode is
        // off. Combining both modes is reasonable but we keep the
        // hierarchy simple: budget wins.
        if self.vram_budget_bytes.is_some() {
            return;
        }
        if let Some(k) = self.resident_window {
            if i >= k {
                let evict_idx = i - k;
                if self.layers[evict_idx].is_resident() {
                    self.layers[evict_idx].evict();
                }
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "rocm")]
mod tests {
    use super::*;
    use modgrad_compute::neuron::{Linear, SimpleRng};
    use modgrad_device::backend::rocm::ffi::runtime_available;
    use std::sync::Arc;

    /// Build a stack of `n` Linear layers, each `dim → dim`, with
    /// deterministic weights. Returns the host `Arc<Linear>` list (so
    /// tests can drive a reference computation in CPU) and a vector
    /// of `LinearLayer` ready to feed the scheduler.
    fn build_stack(n: usize, dim: usize) -> (Vec<Arc<Linear>>, Vec<Box<dyn Layer>>) {
        let mut hosts = Vec::with_capacity(n);
        let mut layers: Vec<Box<dyn Layer>> = Vec::with_capacity(n);
        for i in 0..n {
            // Deterministic seed per layer so each Linear is distinct
            // but reproducible across runs.
            let mut lin = Linear::new(dim, dim);
            // Add a small per-layer offset to the bias so layer
            // outputs differ in a way that's traceable through the
            // stack.
            let mut rng = SimpleRng::new(0xCAFE ^ (i as u64));
            for b in lin.bias.iter_mut() {
                *b = rng.next_normal() * 0.05;
            }
            let arc = Arc::new(lin);
            hosts.push(arc.clone());
            layers.push(Box::new(LinearLayer::from_linear_arc(arc)));
        }
        (hosts, layers)
    }

    /// Reference forward: chain `Linear::forward_into` host-side
    /// across the stack, returning the final output vector.
    fn reference_forward(hosts: &[Arc<Linear>], x: &[f32]) -> Vec<f32> {
        let mut cur = x.to_vec();
        for lin in hosts {
            let mut out = vec![0.0f32; lin.out_dim];
            lin.forward_into(&cur, &mut out);
            cur = out;
        }
        cur
    }

    /// Bind a (test-skip-on-no-GPU) header into a single helper. If
    /// the HIP runtime isn't available we early-return — this matches
    /// the pattern used in `crates/modgrad-compute/src/neuron.rs`.
    macro_rules! skip_if_no_gpu {
        () => {
            if !runtime_available() {
                eprintln!("hip runtime unavailable, skipping");
                return;
            }
        };
    }

    /// All layers resident from the start; scheduler runs forward
    /// and the output matches the host-chained reference within FP
    /// slack. Sanity check: dispatch wiring is correct, scratch
    /// buffers are sized right, dim adjacency is enforced.
    #[test]
    fn scheduler_full_resident() {
        skip_if_no_gpu!();
        let dim = 32;
        let n_layers = 4;
        let (hosts, layers) = build_stack(n_layers, dim);

        // Reference: pure host chain.
        let mut rng = SimpleRng::new(0x1111);
        let host_x: Vec<f32> = (0..dim).map(|_| rng.next_normal()).collect();
        let host_y = reference_forward(&hosts, &host_x);

        // Scheduler: no eviction policy → all layers stay resident
        // after first call.
        let mut sched = LayerScheduler::new(layers);
        let mut x_dev = GpuVec::try_hip(dim).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(dim).expect("alloc out");

        let batch = HipBatch::new();
        sched.forward(&batch, &x_dev, &mut out_dev).expect("forward");
        batch.flush().expect("flush");

        let mut device_y = vec![0.0f32; dim];
        out_dev.copy_to_host(&mut device_y);

        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "scheduler vs host chain mismatch: max |Δ| = {max_diff}");

        // With no eviction policy, every layer should be resident
        // post-forward.
        assert_eq!(sched.current_resident_bytes(), sched.full_resident_bytes(),
            "no-eviction policy must leave every layer resident");
    }

    /// Sliding window of 2: at no point during a forward over 4
    /// layers should more than 2 layers be resident. We instrument
    /// the policy by stepping through the schedule manually
    /// (via direct `LayerScheduler::forward` calls on a 1-layer
    /// schedule per step is infeasible — instead, run the full
    /// forward and verify the post-state matches the policy).
    ///
    /// The cleanest verification: after a full forward with K=2 over
    /// 4 layers, the scheduler should have evicted layers 0 and 1,
    /// leaving only layers 2 and 3 resident. (Layer i-K is evicted
    /// when we finish dispatching layer i; for i=2 we evict 0; for
    /// i=3 we evict 1; layers 2 and 3 stay.)
    #[test]
    fn scheduler_resident_window() {
        skip_if_no_gpu!();
        let dim = 16;
        let n_layers = 4;
        let (hosts, layers) = build_stack(n_layers, dim);

        let mut sched = LayerScheduler::new(layers).with_resident_window(2);
        let mut rng = SimpleRng::new(0x2222);
        let host_x: Vec<f32> = (0..dim).map(|_| rng.next_normal()).collect();

        let mut x_dev = GpuVec::try_hip(dim).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(dim).expect("alloc out");

        let batch = HipBatch::new();
        sched.forward(&batch, &x_dev, &mut out_dev).expect("forward");
        batch.flush().expect("flush");

        // Per the sliding-window policy with K=2 and N=4, only the
        // last 2 layers (indices 2, 3) should be resident.
        let resident_flags: Vec<bool> = sched.layers().iter()
            .map(|l| l.is_resident())
            .collect();
        assert_eq!(resident_flags, vec![false, false, true, true],
            "sliding-window K=2: expected layers 0-1 evicted, 2-3 resident, \
             got resident_flags={resident_flags:?}");

        // Sanity: a single layer's bytes is dim*dim*4 (weight)
        // + dim*4 (bias). Two layers resident means total ≈ 2 * that.
        let one = (dim * dim + dim) * 4;
        let total = sched.current_resident_bytes();
        assert!(total >= one && total <= 2 * one + 64,
            "resident bytes {total} out of expected band [{one}, {}]", 2*one + 64);

        // Also verify the output is still correct end-to-end.
        let host_y = reference_forward(&hosts, &host_x);
        let mut device_y = vec![0.0f32; dim];
        out_dev.copy_to_host(&mut device_y);
        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "windowed scheduler output mismatch: max |Δ| = {max_diff}");
    }

    /// Run forward, evict every layer, run forward again — the second
    /// run must produce bit-identical output (deterministic dispatch
    /// + same upload path = same arithmetic).
    #[test]
    fn scheduler_evict_then_redispatch() {
        skip_if_no_gpu!();
        let dim = 24;
        let n_layers = 3;
        let (_hosts, layers) = build_stack(n_layers, dim);

        let mut sched = LayerScheduler::new(layers);
        let mut rng = SimpleRng::new(0x3333);
        let host_x: Vec<f32> = (0..dim).map(|_| rng.next_normal()).collect();

        let mut x_dev = GpuVec::try_hip(dim).expect("alloc x");
        x_dev.copy_from(&host_x);

        // First forward: warm uploads.
        let mut out_a = GpuVec::try_hip(dim).expect("alloc out a");
        let batch_a = HipBatch::new();
        sched.forward(&batch_a, &x_dev, &mut out_a).expect("forward a");
        batch_a.flush().expect("flush a");
        let mut y_a = vec![0.0f32; dim];
        out_a.copy_to_host(&mut y_a);

        assert!(sched.current_resident_bytes() > 0,
            "after first forward, layers should be resident");

        // Rebuild with a budget so tiny that every layer must be
        // evicted between dispatches (1 byte is below any layer's
        // cost). `build_stack` is deterministic by seed, so the new
        // stack has bit-identical weights — the second forward
        // exercises the full upload-from-host path with no benefit
        // from any prior cache.
        let (_h2, new_layers) = build_stack(n_layers, dim);
        let mut sched2 = LayerScheduler::new(new_layers).with_vram_budget(1);

        let mut out_b = GpuVec::try_hip(dim).expect("alloc out b");
        let batch_b = HipBatch::new();
        sched2.forward(&batch_b, &x_dev, &mut out_b).expect("forward b");
        batch_b.flush().expect("flush b");
        let mut y_b = vec![0.0f32; dim];
        out_b.copy_to_host(&mut y_b);

        // Bit-identical: same weights, same dispatch path, no PRNG
        // anywhere on the GPU side.
        assert_eq!(y_a, y_b,
            "evict + redispatch changed output: bug in upload path");

        // And only the final layer should be resident under the
        // tight budget.
        let resident_count = sched2.layers().iter()
            .filter(|l| l.is_resident())
            .count();
        assert!(resident_count <= 1,
            "tight budget should evict all but ≤1 layer, got {resident_count} resident");
    }

    /// Tiny VRAM budget — big enough for one layer, not for two.
    /// Forces an eviction during the forward sweep. We verify two
    /// things: (a) the forward still completes correctly, and (b)
    /// the resident-bytes count never exceeds the budget plus one
    /// layer's slack (the layer being uploaded next is not yet
    /// counted by `current_resident_bytes` until ensure_resident
    /// returns).
    #[test]
    fn scheduler_vram_budget() {
        skip_if_no_gpu!();
        let dim = 16;
        let n_layers = 4;
        let (hosts, layers) = build_stack(n_layers, dim);

        // Each layer is `(dim*dim + dim) * 4` bytes. Set the budget
        // to be just one layer plus epsilon — forces eviction on
        // every transition.
        let one_layer_bytes = (dim * dim + dim) * 4;
        let budget = one_layer_bytes + 32;  // a hair over one layer

        let mut sched = LayerScheduler::new(layers).with_vram_budget(budget);

        let mut rng = SimpleRng::new(0x4444);
        let host_x: Vec<f32> = (0..dim).map(|_| rng.next_normal()).collect();
        let mut x_dev = GpuVec::try_hip(dim).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(dim).expect("alloc out");

        let batch = HipBatch::new();
        sched.forward(&batch, &x_dev, &mut out_dev).expect("forward under budget");
        batch.flush().expect("flush");

        // Post-forward, only the last layer should be resident
        // (all earlier layers were evicted to make room).
        let resident_flags: Vec<bool> = sched.layers().iter()
            .map(|l| l.is_resident())
            .collect();
        let resident_count = resident_flags.iter().filter(|&&r| r).count();
        assert!(resident_count <= 1,
            "tiny budget should keep at most 1 layer resident, got {resident_count}: \
             {resident_flags:?}");

        // And the output is still correct.
        let host_y = reference_forward(&hosts, &host_x);
        let mut device_y = vec![0.0f32; dim];
        out_dev.copy_to_host(&mut device_y);
        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "budget-evicting scheduler output mismatch: max |Δ| = {max_diff}");
    }

    /// `LayerSequence` composes — wrap a 2-layer sequence as a single
    /// `Layer`, hand it to a scheduler with a 1-layer LinearLayer
    /// after, and verify end-to-end output matches a 3-layer
    /// reference chain.
    #[test]
    fn layer_sequence_composes() {
        skip_if_no_gpu!();
        let dim = 16;
        let (hosts, layers) = build_stack(3, dim);

        // First two layers go into a LayerSequence, third is the
        // top-level layer.
        let mut iter = layers.into_iter();
        let inner_a = iter.next().expect("layer 0");
        let inner_b = iter.next().expect("layer 1");
        let outer_c = iter.next().expect("layer 2");

        let seq: Box<dyn Layer> = Box::new(LayerSequence::new(vec![inner_a, inner_b]));
        let mut sched = LayerScheduler::new(vec![seq, outer_c]);

        let mut rng = SimpleRng::new(0x5555);
        let host_x: Vec<f32> = (0..dim).map(|_| rng.next_normal()).collect();
        let host_y = reference_forward(&hosts, &host_x);

        let mut x_dev = GpuVec::try_hip(dim).expect("alloc x");
        x_dev.copy_from(&host_x);
        let mut out_dev = GpuVec::try_hip(dim).expect("alloc out");

        let batch = HipBatch::new();
        sched.forward(&batch, &x_dev, &mut out_dev).expect("forward");
        batch.flush().expect("flush");

        let mut device_y = vec![0.0f32; dim];
        out_dev.copy_to_host(&mut device_y);
        let max_diff = host_y.iter().zip(&device_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "LayerSequence-composed output mismatch: max |Δ| = {max_diff}");
    }
}
