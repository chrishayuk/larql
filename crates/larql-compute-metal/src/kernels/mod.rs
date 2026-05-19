//! Pipeline + dispatch geometry handle, kernel-name registry, and
//! related helpers.
//!
//! ## Why this module exists
//!
//! Shaders with simdgroup-tiled row mapping (q4_matvec_v4, q4k_matvec,
//! q4k_ffn_gate_up, …) hardcode their per-TG row coverage. The
//! dispatch wrapper has to compute `num_tgs = num_rows.div_ceil
//! (rows_per_tg)` and request `threads_per_tg` threads in agreement
//! with the kernel's row map. Importing those constants from a
//! *different* shader module while the pipeline is built from the
//! kernel that's actually loaded is exactly how the q4_matvec_v4
//! 75 %-row-drop bug landed (closed 2026-04-25 — see ROADMAP.md ship
//! log).
//!
//! ## Layout
//!
//! - `traits`: [`TiledKernel`] — marker trait a shader module
//!   implements to expose its kernel name + dispatch geometry as
//!   compile-time constants. The shader source, name, and geometry
//!   then all live in the same file.
//! - `handle`: [`KernelHandle`] — pipeline state + geometry + name,
//!   bundled. Construction goes through
//!   [`KernelHandle::from_kernel::<K: TiledKernel>`](handle::KernelHandle::from_kernel),
//!   so binding sites read constants by *path*, not by hand-typed
//!   strings. Construction also asserts pipeline
//!   `maxTotalThreadsPerThreadgroup` ≥ requested `threads_per_tg`
//!   so silent simdgroup drop is caught at startup, not at
//!   goldens-fail time.

pub mod handle;
pub mod traits;

// Per-domain pipeline registries (formerly the top-level
// `*_kernels.rs` files in `metal/`).  Each groups the pipelines that
// share a dispatch shape so `MetalBackend` doesn't carry 40+ flat
// `pub` fields — it holds four `pub` registries instead.
pub mod attention;
pub mod ffn;
pub mod norm;
pub mod quant;

pub use attention::AttentionKernels;
pub use ffn::FfnKernels;
pub use handle::KernelHandle;
pub use norm::NormKernels;
pub use quant::QuantKernels;
pub use traits::{get_shader_pipeline, ShaderKernel, TiledKernel};

/// Default maximum threads per threadgroup for **flat per-element
/// dispatches** (`enc.dispatch_threads(MTLSize::new(N, 1, 1),
/// MTLSize::new(DISPATCH_TG_MAX_THREADS.min(N), 1, 1))`).
///
/// 256 is the canonical Apple-Silicon-friendly TG width: 8 simdgroups
/// × 32 lanes, which fits the per-row reduction kernels (rms_norm,
/// residual_add, geglu, etc.) without oversubscribing the TG memory
/// budget. Per-row reductions clamp to `min(DISPATCH_TG_MAX_THREADS,
/// row_len)` so short rows don't dispatch idle threads.
///
/// **Tiled kernels** (q4_matvec_v4, q4k_matvec, q4k_ffn_gate_up, …)
/// declare their own `THREADS_PER_TG` via [`TiledKernel`] and bind it
/// through [`KernelHandle`] — that path is independent of this
/// constant and must NOT use it (see the q4_matvec_v4 75% row-drop
/// ship-log entry on what happens when the dispatcher and the kernel
/// disagree on threadgroup width).
pub const DISPATCH_TG_MAX_THREADS: u64 = 256;

/// Panics on pipeline-compile failure.  Used by the per-domain
/// registry `build()` constructors (`AttentionKernels`, `FfnKernels`,
/// `NormKernels`, `QuantKernels`) to collapse 16+ `?` operators per
/// registry into a single covered path while preserving the "code bug
/// → crash" guarantee that the original `Option`-returning chain
/// expressed.
///
/// Why the early-return form was untestable: each `get_shader_pipeline(...)?,`
/// line carries two LLVM-cov regions — the success branch (always
/// taken in production) and the `None` early-return (only reachable
/// if an MSL kernel name has a typo, which is caught at compile/CI
/// time, not at runtime).  Line coverage therefore caps at ~80 % for
/// every registry file, with no testable path to lift it.  Pushing
/// the None case into `unwrap_or_else(panic!)` removes the second
/// region and lifts each line to 100 % covered.
pub(crate) fn compile_required<K: ShaderKernel>(
    device: &metal::Device,
    library: &metal::Library,
) -> metal::ComputePipelineState {
    get_shader_pipeline::<K>(device, library)
        .unwrap_or_else(|| panic!("pipeline compile failed for kernel `{}`", K::KERNEL_NAME))
}

/// `KernelHandle` variant of [`compile_required`].  Panics on
/// pipeline-compile failure; same coverage-collapsing motivation.
pub(crate) fn compile_required_handle<K: TiledKernel>(
    device: &metal::Device,
    library: &metal::Library,
) -> KernelHandle {
    KernelHandle::from_kernel::<K>(device, library)
        .unwrap_or_else(|| panic!("pipeline compile failed for kernel `{}`", K::KERNEL_NAME))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bogus `ShaderKernel` whose `KERNEL_NAME` references a Metal
    /// function that doesn't exist in the shipped library — so
    /// `get_shader_pipeline` returns `None` and `compile_required`
    /// panics.  Covers the panic branch on line 88.
    struct BogusShaderKernel;
    impl ShaderKernel for BogusShaderKernel {
        const KERNEL_NAME: &'static str = "this_kernel_does_not_exist_in_the_library";
    }

    /// A bogus `TiledKernel` for the `KernelHandle` panic branch.
    struct BogusTiledKernel;
    impl TiledKernel for BogusTiledKernel {
        const KERNEL_NAME: &'static str = "this_kernel_does_not_exist_in_the_library";
        const ROWS_PER_TG: u64 = 1;
        const THREADS_PER_TG: u64 = 32;
    }

    fn library() -> (metal::Device, metal::Library) {
        // Compile an empty source string into a real Metal library —
        // the bogus kernel name definitely isn't in it.
        let device = metal::Device::system_default().expect("Metal device on test host");
        let library = device
            .new_library_with_source("", &metal::CompileOptions::new())
            .expect("empty source compiles");
        (device, library)
    }

    #[test]
    #[should_panic(expected = "pipeline compile failed for kernel")]
    fn compile_required_panics_on_missing_function() {
        let (d, lib) = library();
        let _ = compile_required::<BogusShaderKernel>(&d, &lib);
    }

    #[test]
    #[should_panic(expected = "pipeline compile failed for kernel")]
    fn compile_required_handle_panics_on_missing_function() {
        let (d, lib) = library();
        let _ = compile_required_handle::<BogusTiledKernel>(&d, &lib);
    }
}
