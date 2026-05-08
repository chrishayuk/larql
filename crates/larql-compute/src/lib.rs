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
pub use cuda::{CudaBackend, CudaResidentF32Matrix};

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

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < tol, "idx={idx} got={g} want={w}");
        }
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
        let out = CpuBackend.matmul_transb(
            Array2::from_shape_vec((1, k), x.clone()).unwrap().view(),
            w.view(),
        );
        let want = out.row(0).to_vec();
        assert_close(&got, &want, 1e-3);
    }

    #[test]
    fn cuda_backend_f32_gemv_topk1_matches_cpu_argmax() {
        let Some(cuda) = maybe_cuda() else {
            return;
        };

        let n = 64;
        let k = 96;
        let w = Array2::from_shape_fn((n, k), |(row, col)| {
            ((row * 13 + col * 7) % 29) as f32 / 13.0 - 0.7
        });
        let x: Vec<f32> = (0..k).map(|i| (i % 17) as f32 / 8.0 - 1.0).collect();
        let out = CpuBackend.matmul_transb(
            Array2::from_shape_vec((1, k), x.clone()).unwrap().view(),
            w.view(),
        );
        let scores = out.row(0).to_vec();
        let (want_idx, want_score) = scores
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .map(|(idx, score)| (idx as u32, score))
            .min_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)))
            .unwrap();

        let got = cuda
            .f32_gemv_topk1(w.view(), &x)
            .expect("CUDA f32_gemv_topk1");
        assert_eq!(got.0, want_idx);
        assert!((got.1 - want_score).abs() < 1e-3);
    }

    #[test]
    fn cuda_backend_resident_f32_gemv_reuses_device_matrix_matches_cpu() {
        let Some(cuda) = maybe_cuda() else {
            return;
        };

        let n = 32;
        let k = 48;
        let w = Array2::from_shape_fn((n, k), |(row, col)| {
            ((row * 11 + col * 5) % 31) as f32 / 15.0 - 1.0
        });
        let x1: Vec<f32> = (0..k).map(|i| (i % 13) as f32 / 6.0 - 1.0).collect();
        let x2: Vec<f32> = (0..k).map(|i| (i % 17) as f32 / 8.0 - 0.5).collect();

        let resident = cuda
            .resident_f32_matrix(w.view())
            .expect("copy f32 matrix once to device");
        assert_eq!(resident.rows(), n);
        assert_eq!(resident.cols(), k);

        let got1 = resident.gemv(&cuda, &x1).expect("resident gemv #1");
        let got2 = resident.gemv(&cuda, &x2).expect("resident gemv #2");
        let want1 = CpuBackend.matmul_transb(
            Array2::from_shape_vec((1, k), x1.clone()).unwrap().view(),
            w.view(),
        );
        let want2 = CpuBackend.matmul_transb(
            Array2::from_shape_vec((1, k), x2.clone()).unwrap().view(),
            w.view(),
        );

        assert_close(&got1, &want1.row(0).to_vec(), 1e-3);
        assert_close(&got2, &want2.row(0).to_vec(), 1e-3);
    }

    #[test]
    fn cuda_backend_q4k_matvec_and_stride32_match_cpu() {
        let Some(cuda) = maybe_cuda() else {
            return;
        };

        let rows = 16;
        let hidden = 256;
        let weights: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i * 19 + 5) % 37) as f32 / 18.0 - 1.0)
            .collect();
        let q4k = crate::cpu::ops::q4_common::quantize_q4_k(&weights);
        let x: Vec<f32> = (0..hidden).map(|i| (i % 23) as f32 / 11.0 - 1.0).collect();

        let want = CpuBackend.q4k_matvec(&q4k, &x, rows, hidden).unwrap();
        let got = cuda
            .q4k_matvec(&q4k, &x, rows, hidden)
            .expect("CUDA q4k_matvec");
        assert_close(&got, &want, 1e-3);

        let got_stride = cuda
            .q4k_matvec_stride32(&q4k, &x, rows, hidden)
            .expect("CUDA q4k_matvec_stride32");
        assert_close(&got_stride, &want, 1e-3);
    }

    #[test]
    fn cuda_backend_q4_topk_and_pair_batch_match_cpu() {
        let Some(cuda) = maybe_cuda() else {
            return;
        };

        let rows = 32;
        let hidden = 64;
        let weights: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i * 11 + 3) % 41) as f32 / 20.0 - 1.0)
            .collect();
        let q4 = crate::cpu::ops::q4_common::quantize_q4_0(&weights);
        let x: Vec<f32> = (0..hidden).map(|i| (i % 13) as f32 / 6.0 - 1.0).collect();
        let (q8_x, q8_scales) = crate::cpu::ops::q4_common::quantize_to_q8(&x);

        let scores = CpuBackend
            .q4_matvec(&q4, &q8_x, &q8_scales, rows, hidden)
            .unwrap();
        let top1 = cuda
            .q4_matvec_topk1(&q4, &q8_x, &q8_scales, rows, hidden)
            .expect("CUDA q4_matvec_topk1");
        let want_top1 = scores
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(idx, score)| (idx as u32, score))
            .unwrap();
        assert_eq!(top1.0, want_top1.0);
        assert!((top1.1 - want_top1.1).abs() < 1e-3);

        let topk = cuda
            .q4_matvec_topk(&q4, &q8_x, &q8_scales, rows, hidden, 4)
            .expect("CUDA q4_matvec_topk");
        assert_eq!(topk.len(), 4);

        let (gate, up) = cuda
            .q4_matvec_pair_batch(&q4, &q4, &x, 1, rows, hidden)
            .expect("CUDA q4_matvec_pair_batch");
        assert_eq!(gate.len(), 1);
        assert_eq!(up.len(), 1);
        assert_close(&gate[0], &scores, 1e-3);
        assert_close(&up[0], &scores, 1e-3);
    }
}
