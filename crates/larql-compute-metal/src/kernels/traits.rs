//! `TiledKernel` — marker trait that lets a shader module own its own
//! kernel name + dispatch geometry as compile-time constants.
//!
//! The shader source already lives in `shaders/<name>.rs`. Adding a
//! `pub struct Kernel; impl TiledKernel for Kernel { … }` block to
//! that file co-locates name + geometry + source. Binding the
//! pipeline becomes a one-line call to
//! [`KernelHandle::from_kernel::<…::Kernel>(device, library)`](super::KernelHandle::from_kernel).
//! Bumping a shader (e.g. `q4_matvec_v4` → `_v6`) = change the type
//! parameter at the binding site. No magic strings at the binding
//! site, no chance of geometry drifting from the kernel.

/// A flat-dispatch compute kernel driven by `dispatch_threads` or
/// `dispatch_thread_groups` with fixed geometry. Implemented by a
/// marker struct inside each shader module. Lets `MetalBackend::new()`
/// read the kernel name from a compile-time constant rather than a
/// raw string literal that would drift silently on rename.
///
/// Binding pattern:
/// ```ignore
/// let pl = get_shader_pipeline::<shaders::qk_norm::QkNormKernel>(&device, &library)?;
/// ```
pub trait ShaderKernel {
    /// Metal kernel function name as it appears in `kernel void <name>(…)`.
    const KERNEL_NAME: &'static str;
}

/// Convenience: look up `T::KERNEL_NAME` in `library` and create a pipeline.
/// Returns `None` if the function isn't found or pipeline creation fails.
pub fn get_shader_pipeline<T: ShaderKernel>(
    device: &metal::Device,
    library: &metal::Library,
) -> Option<metal::ComputePipelineState> {
    let f = library.get_function(T::KERNEL_NAME, None).ok()?;
    device.new_compute_pipeline_state_with_function(&f).ok()
}

/// A simdgroup-tiled compute kernel that needs `dispatch_thread_groups`
/// geometry to drive correctly. Implemented by a marker `Kernel` type
/// inside each tiled-shader module.
///
/// Flat-dispatch kernels (one thread per output element, driven by
/// `dispatch_threads`) don't need geometry and shouldn't implement
/// this trait — they're plain `ComputePipelineState`s. Use
/// [`ShaderKernel`] + [`get_shader_pipeline`] for those.
pub trait TiledKernel {
    /// Metal kernel function name as it appears in
    /// `kernel void <name>(…)` in the shader source.
    const KERNEL_NAME: &'static str;
    /// Output rows the kernel covers per threadgroup.
    const ROWS_PER_TG: u64;
    /// Threads per threadgroup the kernel is sized for.
    const THREADS_PER_TG: u64;
}
