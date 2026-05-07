//! # larql-compute
//!
//! Hardware-accelerated compute backends for LARQL.
//!
//! Provides the [`ComputeBackend`] trait that abstracts all hardware-specific
//! matrix operations. Every LARQL crate (inference, vindex) uses this trait —
//! the caller never knows whether the operation runs on CPU or GPU.
//!
//! ## Trait split
//!
//! `ComputeBackend` is the umbrella trait every caller takes as
//! `&dyn ComputeBackend`. It supertraits four narrower traits, each in
//! its own module:
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
//! | Backend | Feature | Operations |
//! |---------|---------|------------|
//! | CPU | (always) | BLAS f32, C kernel Q4 (ARM vdotq_s32), vector ops |
//! | Metal | `metal` | Tiled f32, simdgroup Q4, multi-layer pipeline |
//! | CUDA | `cuda` | (Phase-1 stub — feature-gated scaffold; cuBLAS f32, Q4 matvec, fused attention land in follow-up changes — see openspec/changes/cuda-and-rotorquant-kv/) |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use larql_compute::prelude::*;
//! use larql_compute::{default_backend, QuantFormat};
//!
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
//! [`QuantMatVec::quant_matvec`]'s default impl + one CPU kernel + one
//! Metal shader. The Metal shader gets a `Kernel` marker (impl
//! `metal::kernel::TiledKernel`) so its name + dispatch geometry travel
//! with it via [`metal::kernel::KernelHandle`] — no parallel
//! `shaders::*::ROWS_PER_TG` imports that could drift from the pipeline.
//!
//! ## Feature flags
//!
//! - `metal`: Metal GPU backend (macOS only). Adds optimised Q4 shaders,
//!   multi-layer pipeline, zero-copy mmap buffers.
//! - `cuda`: CUDA GPU backend (Linux + NVIDIA). Currently a feature-flag
//!   stub — `default_backend()` will return a `CudaBackend` and
//!   `Capability::Cuda` is advertised, but actual kernels are
//!   `unimplemented!()` until the follow-up changes
//!   (`cuda-f32-baseline`, `cuda-q4-matvec`, `cuda-fused-attention`)
//!   ship. See `openspec/changes/cuda-and-rotorquant-kv/`.

extern crate blas_src;

pub mod backend;
pub mod cpu;
pub mod pipeline;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

// ── Re-exports: pipeline types ──

pub use pipeline::{
    Activation, FfnType, FullPipelineLayer, MoeLayerWeights, NormType, QuantFormat, QuantWeight,
};

// ── Re-exports: backend ──

pub use backend::{
    dot_proj_gpu, matmul_gpu, Capability, ComputeBackend, DecodeBackend, MatMul, MatMulOp,
    QuantMatVec,
};

/// Bring every backend sub-trait into scope at once.
///
/// Most test/bench/example code calls methods like `matmul_transb` or
/// `q4_matvec` directly on a concrete `CpuBackend` / `MetalBackend`,
/// which Rust resolves through the sub-trait that defines the method.
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

/// Read and clear the per-stage timings stored after the most recent
/// Metal decode step. Returns `None` when `LARQL_PROFILE_SPLIT` is unset
/// or no step has run yet. Used by the generate loop to accumulate
/// gate+up / act+down averages into `StageTimings`.
#[cfg(feature = "metal")]
pub use metal::take_last_split_timings as metal_take_last_split_timings;
#[cfg(feature = "metal")]
pub use metal::{MetalBackend, MoeScratch};

/// Re-export of the metal-rs `Buffer` type so downstream crates (e.g.
/// `larql-server`) can hold cached `(gate_up, down)` Metal buffer pairs
/// without taking a direct dependency on the `metal` crate.
#[cfg(feature = "metal")]
pub use ::metal::Buffer as MetalBuffer;

/// Create the best available backend.
///
/// Precedence: **CUDA → Metal → CPU**, gated by feature flags and runtime
/// availability. The `LARQL_BACKEND` env var (`cpu`, `metal`, `cuda`)
/// overrides the auto-selection; an unrecognised value logs a warning
/// and proceeds with auto-selection.
///
/// - With `--features cuda` on Linux + working CUDA driver → CUDA stub
///   (real kernels land in follow-ups).
/// - With `--features metal` on macOS → Metal GPU + hybrid dispatch
///   calibration.
/// - Otherwise → CPU (Accelerate BLAS on macOS, OpenBLAS on Linux).
///
/// # Example
/// ```rust,no_run
/// let backend = larql_compute::default_backend();
/// println!("{} ({})", backend.name(), backend.device_info());
/// ```
pub fn default_backend() -> Box<dyn ComputeBackend> {
    // Honour LARQL_BACKEND override before any auto-selection.
    if let Ok(forced) = std::env::var("LARQL_BACKEND") {
        match forced.as_str() {
            "cpu" => return Box::new(cpu::CpuBackend),
            #[cfg(feature = "metal")]
            "metal" => {
                if let Some(m) = metal::MetalBackend::new() {
                    m.calibrate();
                    return Box::new(m);
                }
                eprintln!("[compute] LARQL_BACKEND=metal but Metal not available; falling back");
            }
            #[cfg(feature = "cuda")]
            "cuda" => match cuda::CudaBackend::new() {
                Ok(c) => return Box::new(c),
                Err(e) => eprintln!(
                    "[compute] LARQL_BACKEND=cuda but CUDA init failed: {e}; falling back"
                ),
            },
            other => {
                eprintln!("[compute] LARQL_BACKEND={other:?} not recognised; auto-selecting");
            }
        }
    }

    #[cfg(feature = "cuda")]
    {
        match cuda::CudaBackend::new() {
            Ok(c) => return Box::new(c),
            Err(e) => eprintln!("[compute] CUDA not available ({e}); trying next backend"),
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Some(m) = metal::MetalBackend::new() {
            m.calibrate();
            return Box::new(m);
        }
        eprintln!("[compute] Metal not available, falling back to CPU");
    }

    Box::new(cpu::CpuBackend)
}

/// Force CPU-only backend. No GPU, no calibration overhead.
///
/// Use when you want deterministic CPU execution or to benchmark
/// CPU vs GPU paths.
pub fn cpu_backend() -> Box<dyn ComputeBackend> {
    Box::new(cpu::CpuBackend)
}
