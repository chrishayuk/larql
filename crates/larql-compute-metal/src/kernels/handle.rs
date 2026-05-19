//! `KernelHandle` — bundled pipeline state, dispatch geometry, and
//! kernel name. See `super` module docs for context.

use metal::{ComputePipelineState, Device, Library};

use super::TiledKernel;

/// A compiled shader pipeline plus the per-TG geometry the dispatcher
/// must use to drive it correctly.
///
/// Every dispatch site reads `state` for `set_compute_pipeline_state`
/// and `rows_per_tg`/`threads_per_tg` for `dispatch_thread_groups`.
/// Geometry travels with the pipeline; bumping a shader = swap the
/// type parameter at the [`from_kernel`](Self::from_kernel) call site.
///
/// `Clone` is cheap — `ComputePipelineState` is a wrapper around a
/// ref-counted Objective-C object, and the geometry constants are
/// plain `u64`. Cloning is only used for runtime kernel selection
/// (e.g., `LARQL_Q6K_8SG=0` opt-out to the 4sg variant).
#[derive(Clone)]
pub struct KernelHandle {
    /// The underlying pipeline state. Use this for
    /// `enc.set_compute_pipeline_state(&handle.state)`.
    pub state: ComputePipelineState,
    /// Output rows the kernel covers per threadgroup. Dispatchers
    /// compute `num_tgs = num_rows.div_ceil(rows_per_tg)`.
    pub rows_per_tg: u64,
    /// Threads per threadgroup the kernel expects. Constructor
    /// guarantees this fits within the pipeline's
    /// `maxTotalThreadsPerThreadgroup` cap.
    pub threads_per_tg: u64,
    /// Metal kernel function name (for diagnostics only).
    pub kernel_name: &'static str,
}

impl KernelHandle {
    /// Build a handle from a shader module that exposes its kernel
    /// name + geometry via the [`TiledKernel`] trait. This is the
    /// preferred constructor — the caller writes the shader-module
    /// path once and all three constants travel with it.
    ///
    /// ```ignore
    /// matvec: KernelHandle::from_kernel::<shaders::q4_matvec_v4::Kernel>(
    ///     &device, &library,
    /// )?,
    /// ```
    pub fn from_kernel<K: TiledKernel>(device: &Device, library: &Library) -> Option<Self> {
        Self::compile(
            device,
            library,
            K::KERNEL_NAME,
            K::ROWS_PER_TG,
            K::THREADS_PER_TG,
        )
    }

    /// Lower-level constructor used by [`from_kernel`](Self::from_kernel).
    /// Prefer that path — it forces the shader module to own its own
    /// name + geometry instead of hand-typing them at the call site.
    fn compile(
        device: &Device,
        library: &Library,
        kernel_name: &'static str,
        rows_per_tg: u64,
        threads_per_tg: u64,
    ) -> Option<Self> {
        let f = library.get_function(kernel_name, None).ok()?;
        let state = device.new_compute_pipeline_state_with_function(&f).ok()?;
        let cap = state.max_total_threads_per_threadgroup();
        if cap < threads_per_tg {
            eprintln!(
                "[metal] kernel `{kernel_name}`: pipeline cap {cap} < requested \
                 threads_per_tg {threads_per_tg}. Metal would silently dispatch \
                 only {cap} threads/TG → fewer simdgroups → rows dropped. \
                 Either lower threads_per_tg, or reduce the kernel's per-thread \
                 register / threadgroup-memory pressure to raise the cap."
            );
            return None;
        }
        Some(Self {
            state,
            rows_per_tg,
            threads_per_tg,
            kernel_name,
        })
    }
}
