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
//! | CUDA | `cuda` | Linux/NVIDIA f32 GEMV via cuBLAS, CPU fallback for other ops |
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
//! - `cuda`: Linux/NVIDIA CUDA backend. Accelerates f32 GEMV via cuBLAS;
//!   other operations fall back to CPU while CUDA coverage expands.

extern crate blas_src;

pub mod backend;
pub mod cpu;
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub mod cuda;
pub mod pipeline;

#[cfg(feature = "metal")]
pub mod metal;

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
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use cuda::CudaBackend;

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
/// With `--features cuda` on Linux: tries CUDA first, falls back to CPU.
/// With `--features metal`: tries Metal GPU first, auto-calibrates the
/// FLOP threshold for hybrid CPU/GPU dispatch, falls back to CPU.
/// Without GPU features: returns CPU (Accelerate BLAS on macOS, OpenBLAS on Linux).
///
/// # Example
/// ```rust,no_run
/// let backend = larql_compute::default_backend();
/// println!("{} ({})", backend.name(), backend.device_info());
/// ```
pub fn default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    {
        if let Some(cuda) = cuda::CudaBackend::new() {
            return Box::new(cuda);
        }
        eprintln!("[compute] CUDA not available, falling back to CPU");
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

#[cfg(all(test, feature = "cuda"))]
mod cuda_backend_tests {
    use super::*;
    use ndarray::Array2;

    fn maybe_cuda() -> Option<CudaBackend> {
        let cuda = CudaBackend::new();
        if cuda.is_none() {
            eprintln!("CUDA driver/device unavailable; skipping CUDA smoke test");
        }
        cuda
    }

    #[test]
    fn default_backend_prefers_cuda_when_available() {
        if maybe_cuda().is_none() {
            return;
        }

        let backend = default_backend();
        assert!(backend.supports(Capability::F32Gemv));
        assert!(backend.name().contains("cuda"), "got {}", backend.name());
    }

    #[test]
    fn cuda_backend_f32_gemv_matches_cpu() {
        let Some(cuda) = maybe_cuda() else {
            return;
        };

        let n = 64;
        let k = 96;
        let w = Array2::from_shape_fn((n, k), |(row, col)| {
            ((row * 17 + col * 31) % 23) as f32 / 11.0 - 1.0
        });
        let x: Vec<f32> = (0..k).map(|i| (i % 19) as f32 / 9.0 - 1.0).collect();

        let got = cuda
            .f32_gemv_force(w.view(), &x)
            .expect("CUDA f32_gemv_force");
        let want = CpuBackend.f32_gemv_force(w.view(), &x).unwrap_or_else(|| {
            let out = CpuBackend.matmul_transb(
                Array2::from_shape_vec((1, k), x.clone()).unwrap().view(),
                w.view(),
            );
            out.row(0).to_vec()
        });

        assert_eq!(got.len(), want.len());
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-3, "idx={idx} got={g} want={w}");
        }
    }
}
