//! # larql-compute
//!
//! [`ComputeBackend`] trait + CPU implementation + pipeline-shape
//! types.  GPU backends ship as sibling crates that implement the
//! same trait — see [`larql-compute-metal`](https://docs.rs/larql-compute-metal)
//! for the Apple-silicon impl.
//!
//! ## Trait split
//!
//! `ComputeBackend` is the umbrella trait every caller takes as
//! `&dyn ComputeBackend`. It supertraits three narrower traits, each
//! in its own module:
//!
//! - [`MatMul`] — f32 / f16 matmul, gemv, batch matmul
//! - [`QuantMatVec`] — unified `quant_matvec` + per-format pre-quantised helpers
//! - [`DecodeBackend`] — KV-cached decode + prefill + MoE hook
//! - umbrella `ComputeBackend` — `name`, `device_info`, [`Capability`] probe
//!
//! `use larql_compute::prelude::*;` brings every sub-trait in scope at once.
//!
//! ## Backends
//!
//! | Backend | Crate                  | Operations |
//! |---------|------------------------|------------|
//! | CPU     | `larql-compute`         | BLAS f32, C kernel Q4 (ARM vdotq_s32), vector ops |
//! | Metal   | `larql-compute-metal`   | Tiled f32, simdgroup Q4, multi-layer pipeline |
//! | Vulkan  | `larql-compute-vulkan` (planned) | — |
//! | CUDA    | `larql-compute-cuda` (planned)   | — |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use larql_compute::prelude::*;
//! use larql_compute::{default_backend, QuantFormat};
//!
//! // `default_backend()` always returns CPU.  Hosts that want GPU
//! // depend on the relevant backend crate and compose a fallback chain
//! // explicitly — see `larql-compute-metal::metal_backend()`.
//! let backend = default_backend();
//! println!("Using: {} ({})", backend.name(), backend.device_info());
//!
//! // Branch on capability instead of probing for `Option::None`:
//! if backend.supports(Capability::F32Gemv) {
//!     // Specialised LM-head gemv is available on this backend.
//! }
//! ```
//!
//! ## Adding a quant format
//!
//! Adding e.g. FP4 = one [`QuantFormat`] variant + one match arm in
//! [`QuantMatVec::quant_matvec`]'s default impl + one CPU kernel +
//! one shader per GPU-backend crate.  The shader-side wiring is
//! local to each backend crate, so a new format doesn't require
//! touching every consumer.

#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "macos",
    target_os = "windows"
))]
extern crate blas_src;

pub mod backend;
pub mod cpu;
pub mod options;
pub mod pipeline;

// ── Re-exports: pipeline types ──

pub use pipeline::{
    Activation, AttentionSpec, AttentionWeights, FfnSpec, FfnType, FfnWeights, FullPipelineLayer,
    LayerNorms, LayerWeights, MoeDownPaddingPolicy, MoeExpertScalePolicy, MoeInputSource,
    MoeLayerWeights, MoePostExpertNormPolicy, MoeRouterNormPolicy, MoeRoutingPolicy, MoeSpec,
    MoeTopKWeightPolicy, MoeWeightLayout, NormType, PositionEncodingType, QuantFormat, QuantWeight,
    RemoteFfnSpec, RMSNORM_EPSILON_DEFAULT, ROPE_BASE_DEFAULT, ROPE_BASE_GLOBAL,
};

// ── Re-exports: backend ──

pub use backend::{
    dot_proj_gpu, matmul_gpu, Capability, ComputeBackend, DecodeBackend, MatMul, MatMulOp,
    QuantMatVec,
};

/// Bring every backend sub-trait into scope at once.
///
/// Most test/bench/example code calls methods like `matmul_transb` or
/// `q4_matvec` directly on a concrete backend value, which Rust
/// resolves through the sub-trait that defines the method.
/// `use larql_compute::prelude::*;` saves listing them one by one.
pub mod prelude {
    pub use crate::backend::{
        Capability, ComputeBackend, DecodeBackend, MatMul, MatMulOp, QuantMatVec,
    };
}

pub use cpu::ops::linalg::{cholesky, cholesky_inverse, cholesky_solve, ridge_decomposition_solve};
pub use cpu::ops::moe::{quantize_x_to_q8k, Q8KActivation};
pub use cpu::ops::vector::{cosine, dot, norm};
pub use cpu::CpuBackend;

/// Build a CPU backend.  Always returns a usable backend (BLAS on
/// macOS via Accelerate, OpenBLAS on Linux/Windows).
///
/// Callers that want GPU compose an explicit fallback chain via the
/// relevant backend crate:
///
/// ```rust,no_run
/// # #[cfg(target_os = "macos")] {
/// use larql_compute::{cpu_backend, ComputeBackend};
///
/// let backend: Box<dyn ComputeBackend> =
///     larql_compute_metal::metal_backend()
///         .map(|m| Box::new(m) as Box<dyn ComputeBackend>)
///         .unwrap_or_else(|| cpu_backend());
/// # }
/// ```
///
/// [`default_backend`] is a synonym kept for backwards compatibility
/// with the pre-split callers.
pub fn cpu_backend() -> Box<dyn ComputeBackend> {
    Box::new(cpu::CpuBackend)
}

/// Build the default backend.  Returns CPU.
///
/// Before the GPU-backend extraction, this function auto-detected
/// Metal and fell back to CPU.  After the split, GPU selection is the
/// caller's responsibility — see [`cpu_backend`] for the explicit
/// fallback pattern.  Kept as an alias of [`cpu_backend`] so existing
/// CPU-only callers don't need to change.
pub fn default_backend() -> Box<dyn ComputeBackend> {
    cpu_backend()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_exposes_cpu_backend_capabilities() {
        let backend = cpu_backend();

        assert!(backend.name().starts_with("cpu"));
        assert!(!backend.device_info().is_empty());
        assert!(backend.supports(Capability::QuantMatVec));
    }

    #[test]
    fn default_backend_is_usable_through_prelude_traits() {
        fn assert_compute_backend<T: prelude::ComputeBackend + ?Sized>(backend: &T) {
            assert!(backend.supports(prelude::Capability::QuantMatVec));
        }

        let backend = default_backend();
        assert_compute_backend(backend.as_ref());
    }

    /// After the GPU-backend extraction, `default_backend()` and
    /// `cpu_backend()` are synonyms — both return CPU.  Pin that
    /// invariant so a future reintroduction of GPU auto-selection in
    /// this crate is caught.
    #[test]
    fn default_backend_returns_cpu_after_metal_extraction() {
        let default = default_backend();
        let cpu = cpu_backend();
        assert_eq!(default.name(), cpu.name());
    }
}
