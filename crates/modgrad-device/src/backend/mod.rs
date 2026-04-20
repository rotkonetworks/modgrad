//! Backend abstraction: ops as data, backends as plugins.
//!
//! Architectural inspiration: JAX's PJRT plugin model. Each backend
//! implements the `Backend` trait once and becomes a routable target
//! for every `Op` variant it supports. Callers build an `Op` value and
//! dispatch through a `BackendRegistry`, which tries registered
//! backends in preference order until one accepts.
//!
//! This is the stable boundary between modgrad-ctm / modgrad-ffn
//! callers and the physical hardware. Adding a new hardware target
//! means writing one `impl Backend for NewBackend` block — no per-op
//! wrapper duplication across backends.
//!
//! Adding a new logical op requires adding a variant to [`Op`] and
//! extending each backend's `supports` + `dispatch` — that's a
//! coordinated change, reviewed as a unit.

pub mod op;
pub mod ops;
pub mod buffer;
pub mod cpu;
pub mod kfd;
pub mod cuda_be;
pub mod vulkan;
pub mod rocm;

pub use buffer::{DeviceBuffer, HostBuffer};
// `BufferBackend` + `ComputeCtx` are defined below in this file.
// Re-export them explicitly so call sites can write
//   `use modgrad_device::backend::{ComputeCtx, BufferBackend};`
// without digging into the module layout.
pub use cpu::CpuBackend;
pub use kfd::KfdBackend;
pub use cuda_be::CudaBackend;
pub use vulkan::VulkanBackend;
pub use rocm::RocmBackend;
pub use op::{AdamWArgs, Op, QuantKind, SyncBackwardScatterArgs};

/// Kind of physical device a backend runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// CPU cores, rayon-parallelized.
    Cpu,
    /// AMD GPU programmed via direct KFD ioctl (RDNA3 only for now).
    Kfd,
    /// AMD GPU programmed via HIP runtime + rocBLAS/MIOpen (any ROCm arch).
    Rocm,
    /// NVIDIA GPU programmed via CUDA driver + cuBLAS.
    Cuda,
    /// Cross-vendor GPU via Vulkan compute shaders.
    Vulkan,
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Kfd => "kfd",
            DeviceKind::Rocm => "rocm",
            DeviceKind::Cuda => "cuda",
            DeviceKind::Vulkan => "vulkan",
        };
        f.write_str(s)
    }
}

/// Runtime-discovered info about a backend's target device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    pub name: String,
    pub total_mem_bytes: u64,
    /// Architecture string where meaningful (e.g. "gfx1102" for KFD/ROCm,
    /// "sm_90" for CUDA). `None` for devices without an arch concept (CPU).
    pub arch: Option<String>,
}

/// Errors a backend can return from `dispatch`.
#[derive(Debug)]
pub enum BackendError {
    /// This backend's `supports()` returned true but dispatch failed
    /// — programmer error or a race after a prior `supports()` check.
    /// Registry may fall through to the next backend.
    Unsupported { op: &'static str, backend: &'static str },
    /// Device returned an error during the operation (allocation
    /// failure, kernel crash, driver error). Message is implementation-
    /// defined.
    Runtime(String),
    /// Device ran out of memory during the operation.
    OutOfMemory,
    /// Device lost (hang, crash, unplug). Registry should mark the
    /// backend dead and fall through.
    DeviceLost,
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::Unsupported { op, backend } =>
                write!(f, "backend '{backend}' does not support op '{op}'"),
            BackendError::Runtime(msg) => write!(f, "runtime error: {msg}"),
            BackendError::OutOfMemory => f.write_str("out of device memory"),
            BackendError::DeviceLost => f.write_str("device lost"),
        }
    }
}

impl std::error::Error for BackendError {}

/// One hardware target. Implementations are expected to be `Send + Sync`
/// so a shared registry can be wrapped in `Arc` and dispatched from any
/// thread. Ops are dispatched one-at-a-time; higher-level batching
/// (kernel fusion, graph capture) is a follow-up concern.
///
/// `Backend` is deliberately dyn-compatible so the registry can stash
/// `Box<dyn Backend>`. Device-resident buffer allocation lives on the
/// sibling trait [`BufferBackend`] instead — adding an associated type
/// there doesn't compromise this trait's dyn usage.
pub trait Backend: Send + Sync + 'static {
    /// Short human-readable name (e.g. "kfd", "rocm", "cuda", "cpu").
    /// Used in telemetry, test names, and error messages.
    fn name(&self) -> &'static str;

    /// Describe the device this backend will run on.
    fn device_info(&self) -> DeviceInfo;

    /// Can this backend run this op, given the tensor shapes and
    /// attributes (quantization, etc.)? Must be cheap — called on every
    /// dispatch as the first check. A conservative `false` is always
    /// correct (registry falls through); a falsely optimistic `true`
    /// will surface as `Unsupported` errors from `dispatch()`.
    fn supports(&self, op: &Op) -> bool;

    /// Execute the op. Blocking: when this returns `Ok`, the output
    /// buffers are valid for the CPU to read. Internal queueing and
    /// host↔device transfers are the backend's responsibility.
    ///
    /// Callers must not rely on backend identity across calls — a
    /// registry may route sequential ops to different backends
    /// depending on `supports()`.
    fn dispatch(&self, op: &mut Op) -> Result<(), BackendError>;

    /// Invalidate any internal weight cache the backend holds.
    /// Called by callers after optimizer steps to ensure subsequent
    /// forward dispatches re-upload fresh weights. Default: no-op.
    fn invalidate_cache(&self) {}
}

/// A backend that can also allocate device-resident buffers.
///
/// This sibling of [`Backend`] carries the associated [`Buffer`] type
/// used by [`ComputeCtx`]. Splitting the allocation surface into its
/// own trait keeps `Backend` dyn-compatible (no associated type at the
/// dispatch layer) while still letting us enforce backend-affine
/// buffers at compile time through `ComputeCtx<B: BufferBackend>`.
///
/// Every concrete backend in this crate implements `BufferBackend`.
/// CPU / CUDA / Vulkan use [`HostBuffer`]; KFD and ROCm define their
/// own device-resident Buffer types.
///
/// [`Buffer`]: BufferBackend::Buffer
pub trait BufferBackend: Backend {
    /// Opaque buffer handle this backend owns. Enforces backend-affine
    /// memory: a `KfdBackend::Buffer` cannot be handed to a
    /// `ComputeCtx<RocmBackend>`, by construction of the type system.
    ///
    /// Associated type defaults are still unstable in Rust, so every
    /// `impl BufferBackend` must declare `type Buffer` explicitly. Most
    /// pick `HostBuffer`; KFD and ROCm define their own.
    type Buffer: DeviceBuffer;

    /// Allocate a buffer of `n_f32` elements on this backend's device.
    ///
    /// The returned handle is backend-affine: only this backend (or a
    /// `ComputeCtx<Self>` wrapping it) can dispatch against it.
    /// `HostBuffer`-using backends hit a plain `Vec<f32>` allocation;
    /// KFD + ROCm hit their device allocators.
    fn alloc_buffer(&self, n_f32: usize) -> Result<Self::Buffer, BackendError>;
}

/// Stateful handle around a single backend.
///
/// `ComputeCtx<B>` is the entry point for workflows that need device-
/// resident buffers (vs. the stateless slice-based `registry::dispatch`).
/// It exposes lifecycle methods — `alloc_buffer`, `arena_reset`,
/// `flush` — that don't belong on the `Backend` trait itself because
/// they're not universally meaningful.
///
/// **Monomorphic by design**: `ComputeCtx<KfdBackend>` and
/// `ComputeCtx<RocmBackend>` are distinct types. Cross-backend mixing
/// cannot silently happen — the type system refuses to compile it.
///
/// The default `arena_reset` / `flush` are no-ops; backends that have
/// meaningful implementations override via an inherent impl on
/// `ComputeCtx<SpecificBackend>` (see the KFD override in `kfd.rs`).
///
/// # Compile-time enforcement of backend affinity
///
/// The associated `BufferBackend::Buffer` type is what prevents cross-
/// backend mixing. The KFD feature gate is intentional — without it,
/// `KfdBuffer` is still the type KFD's `alloc_buffer` returns, but the
/// backend is never constructible so the scenario is moot. The doctest
/// below runs under `--features kfd`.
///
/// ```compile_fail
/// # #[cfg(feature = "kfd")]
/// # fn _demo() {
/// use modgrad_device::backend::{ComputeCtx, CpuBackend, KfdBackend};
/// let cpu: &'static CpuBackend = Box::leak(Box::new(CpuBackend::new()));
/// let cpu_ctx: ComputeCtx<CpuBackend> = ComputeCtx::new(cpu);
/// // This binding asks for a KFD buffer from a CPU context — the
/// // compiler must refuse, because `CpuBackend::Buffer == HostBuffer`,
/// // not `KfdBuffer`.
/// let _buf: <KfdBackend as modgrad_device::backend::BufferBackend>::Buffer
///     = cpu_ctx.alloc_buffer(8).unwrap();
/// # }
/// ```
pub struct ComputeCtx<B: BufferBackend + 'static> {
    backend: &'static B,
}

impl<B: BufferBackend + 'static> ComputeCtx<B> {
    /// Wrap a long-lived backend reference. The `'static` bound matches
    /// how backends are actually stored (`OnceLock` singletons, leaked
    /// `Box` for tests, or `&'static` from a `Box::leak` in the registry).
    pub fn new(backend: &'static B) -> Self {
        Self { backend }
    }

    /// Borrow the underlying backend — useful when the caller still needs
    /// to run a slice-based `Op` through the trait's `dispatch` hook.
    pub fn backend(&self) -> &'static B { self.backend }

    /// Short-circuit to the backend's `name()`. Primarily for diagnostics.
    pub fn backend_name(&self) -> &'static str { self.backend.name() }

    /// Allocate a buffer of `n_f32` elements. The return type is
    /// `B::Buffer`, enforcing backend affinity: the compiler refuses to
    /// let a `ComputeCtx<CpuBackend>::alloc_buffer` return a `KfdBuffer`.
    pub fn alloc_buffer(&self, n_f32: usize) -> Result<B::Buffer, BackendError> {
        self.backend.alloc_buffer(n_f32)
    }
}

// Arena and flush hooks live on per-backend inherent impls rather than
// the generic `impl<B: BufferBackend> ComputeCtx<B>` block above.
// Reason: Rust doesn't let an inherent impl on `ComputeCtx<KfdBackend>`
// shadow a same-named method on the generic impl. The cleanest way to
// keep `ctx.arena_reset()` valid for every backend is to provide the
// method on each concrete context.
//
// Cost: each backend's module must spell out a no-op `arena_reset` +
// `flush` if it doesn't have real ones. The upside is that when a
// backend grows a real arena or an async submit path, only one file
// changes — no risk of accidentally inheriting a stale default. See
// `cpu.rs`, `kfd.rs`, `rocm.rs`, `cuda_be.rs`, `vulkan.rs` for the
// per-backend impls.

/// Preference-ordered set of backends. First whose `supports()` returns
/// true wins. `detect()` instantiates whatever's available on this host.
pub struct BackendRegistry {
    backends: Vec<Box<dyn Backend>>,
}

impl BackendRegistry {
    /// Empty registry. Primarily for tests — most callers want `detect`.
    pub fn new() -> Self {
        Self { backends: Vec::new() }
    }

    /// Probe the host for available backends and build a registry
    /// ordered by preference:
    ///
    /// 1. KFD (if gfx1102 present — our hand-written fast path)
    /// 2. ROCm (any AMD GPU with ROCm runtime)
    /// 3. CUDA (NVIDIA)
    /// 4. Vulkan (cross-vendor)
    /// 5. CPU (always registered last, guaranteed to handle every op)
    ///
    /// The `MODGRAD_BACKEND` environment variable forces the registry
    /// to contain only the named backend (e.g. `MODGRAD_BACKEND=cpu`
    /// for debugging). Useful for isolating backend-specific bugs.
    ///
    /// TODO(phase-2): wire KFD, ROCm, CUDA, Vulkan backends as they are
    /// ported from their existing modules. CPU is always registered as
    /// the final fallback.
    pub fn detect() -> Self {
        // Honor MODGRAD_BACKEND=cpu (or any future single-backend
        // override) for deterministic debugging.
        if let Ok(forced) = std::env::var("MODGRAD_BACKEND") {
            let mut reg = Self::new();
            if forced.eq_ignore_ascii_case("cpu") {
                reg.register(Box::new(CpuBackend::new()));
            }
            // Unknown forced value → empty registry; dispatch will error
            // loudly, which is what we want for an invalid override.
            return reg;
        }

        let mut reg = Self::new();
        // General-purpose default: probe ROCm first (portable AMD) and
        // CUDA second (NVIDIA), with Vulkan as cross-vendor fallback.
        //
        // KFD (gfx1102 hand-written kernels) is opt-in via the `kfd`
        // feature. When enabled it registers FIRST so its supports()
        // takes priority over ROCm on the same hardware. Note: the
        // HSA runtime (used by hipblas) and KFD's direct device access
        // can't coexist — the first one to open the device wins.
        // Probing KFD first ensures the fast path gets the handle.
        if let Some(kfd) = KfdBackend::try_new() {
            reg.register(Box::new(kfd));
        }
        if let Some(rocm) = RocmBackend::try_new() {
            reg.register(Box::new(rocm));
        }
        if let Some(cuda) = CudaBackend::try_new() {
            reg.register(Box::new(cuda));
        }
        if let Some(vk) = VulkanBackend::try_new() {
            reg.register(Box::new(vk));
        }
        reg.register(Box::new(CpuBackend::new()));
        reg
    }

    /// Add a backend to the bottom of the preference order (lowest priority).
    pub fn register(&mut self, backend: Box<dyn Backend>) {
        self.backends.push(backend);
    }

    /// Add a backend at the top of the preference order (highest priority).
    /// Used when a fast-path backend (e.g. KFD) should be preferred over
    /// a general-purpose one (e.g. ROCm) on the same hardware.
    pub fn register_preferred(&mut self, backend: Box<dyn Backend>) {
        self.backends.insert(0, backend);
    }

    /// Try each backend in preference order; return the name of the
    /// backend that handled the op, or an error if none did.
    pub fn dispatch(&self, op: &mut Op) -> Result<&'static str, BackendError> {
        for b in &self.backends {
            if b.supports(op) {
                b.dispatch(op)?;
                return Ok(b.name());
            }
        }
        Err(BackendError::Unsupported {
            op: op.name(),
            backend: "<none>",
        })
    }

    /// Iterate registered backends in preference order. Primarily for
    /// parity tests that need to dispatch the same op on every backend.
    pub fn iter(&self) -> impl Iterator<Item = &dyn Backend> {
        self.backends.iter().map(|b| b.as_ref())
    }

    /// Number of registered backends.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Invalidate all backend weight caches. Call after a batched
    /// optimizer step that mutated weights outside of a registered
    /// `Op::AdamW` / `Op::SgdUpdate` dispatch.
    pub fn invalidate_caches(&self) {
        for b in &self.backends { b.invalidate_cache(); }
    }

    /// Whether this registry has any backends registered.
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::detect()
    }
}

/// Process-wide shared `BackendRegistry`. Initialized on first call via
/// `detect()`. Subsequent calls return the same instance with zero
/// locking overhead.
///
/// Callers in modgrad-ctm / modgrad-ffn should dispatch through this
/// rather than constructing their own registry per-call — probes
/// (hipGetDeviceCount, accel::available, etc.) are cheap individually
/// but add up over a training loop.
///
/// The registry is immutable once created; to override for debugging,
/// set `MODGRAD_BACKEND=cpu` in the environment *before* the first
/// call. Changing it at runtime is intentionally not supported.
pub fn registry() -> &'static BackendRegistry {
    use std::sync::OnceLock;
    static REGISTRY: OnceLock<BackendRegistry> = OnceLock::new();
    REGISTRY.get_or_init(BackendRegistry::detect)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Struct-test backend that accepts nothing; verifies the dispatch
    /// fall-through behavior when no backend supports an op.
    struct NullBackend;
    impl Backend for NullBackend {
        fn name(&self) -> &'static str { "null" }
        fn device_info(&self) -> DeviceInfo {
            DeviceInfo {
                kind: DeviceKind::Cpu,
                name: "null".into(),
                total_mem_bytes: 0,
                arch: None,
            }
        }
        fn supports(&self, _: &Op) -> bool { false }
        fn dispatch(&self, _: &mut Op) -> Result<(), BackendError> {
            Err(BackendError::Unsupported { op: "n/a", backend: "null" })
        }
    }
    impl BufferBackend for NullBackend {
        type Buffer = HostBuffer;
        fn alloc_buffer(&self, n: usize) -> Result<HostBuffer, BackendError> {
            Ok(HostBuffer::new(n))
        }
    }

    #[test]
    fn empty_registry_returns_unsupported() {
        let reg = BackendRegistry::new();
        let a = vec![0.0f32; 4];
        let b = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 2];
        let mut op = Op::MatmulNN {
            a: &a, b: &b, out: &mut out, bias: None,
            m: 1, k: 2, n: 2,
        };
        let err = reg.dispatch(&mut op).unwrap_err();
        match err {
            BackendError::Unsupported { .. } => {}
            _ => panic!("expected Unsupported, got {err:?}"),
        }
    }

    #[test]
    fn null_backend_falls_through() {
        let mut reg = BackendRegistry::new();
        reg.register(Box::new(NullBackend));
        let x = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 1];
        let mut op = Op::ReduceL2Sq { x: &x, out: &mut out };
        assert!(matches!(
            reg.dispatch(&mut op),
            Err(BackendError::Unsupported { .. })
        ));
    }

    #[test]
    fn global_registry_is_stable_across_calls() {
        // Same pointer both times → OnceLock is working.
        let a = super::registry() as *const _;
        let b = super::registry() as *const _;
        assert_eq!(a, b);
        // Non-empty because CPU backend always registers.
        assert!(!super::registry().is_empty());
    }

    #[test]
    fn compute_ctx_alloc_roundtrip() {
        // Leak a CpuBackend for 'static — matches how registry-owned
        // backends actually live. ComputeCtx::alloc_buffer must return a
        // HostBuffer whose copy_* round-trips.
        let be: &'static CpuBackend = Box::leak(Box::new(CpuBackend::new()));
        let ctx: ComputeCtx<CpuBackend> = ComputeCtx::new(be);
        assert_eq!(ctx.backend_name(), "cpu");
        let mut buf = ctx.alloc_buffer(4).unwrap();
        buf.copy_from_host(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut out = vec![0.0f32; 4];
        buf.copy_to_host(&mut out).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
        // Default arena_reset / flush are no-ops — just asserting they
        // compile + are callable on a CpuBackend context.
        ctx.arena_reset();
        ctx.flush();
    }

    #[test]
    fn op_name_covers_every_variant() {
        // Compile-time smoke test that every variant has a name.
        // Not exhaustive over runtime state — just makes sure
        // `Op::name` has a match arm per variant.
        let x = [0.0f32];
        let mut out = [0.0f32];
        assert_eq!(Op::SiluFwd { x: &x, out: &mut out }.name(), "silu_fwd");
    }
}
