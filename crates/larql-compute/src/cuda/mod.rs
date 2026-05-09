//! # CUDA backend
//!
//! CUDA support for the [`cuda-and-rotorquant-kv`][change] OpenSpec
//! change. The backend compiles behind `--features cuda`, registers
//! `Capability::Cuda`, and returns from `default_backend()` on Linux
//! when a CUDA driver is reachable.
//!
//! The current implementation covers cuBLAS f32 GEMM/GEMV, a
//! correctness-first Q4/Q6 matvec path, and low-level fused attention
//! helpers. Higher-level decode pipeline integration continues to land
//! through focused follow-up changes.
//!
//! [change]: ../../../../openspec/changes/cuda-and-rotorquant-kv/

pub mod attn;
mod backend;
mod cache;
mod decode;
mod dequant;
mod driver;
mod elem;
mod error;
mod matmul;
#[cfg(feature = "cuda-oxide")]
mod oxide_kernels;
pub mod q4k_batched;
mod q4k_direct;
mod q4k_mmvq;
mod q6k_mmvq;
mod quant_matvec;
pub mod sampling;
mod scratch;

pub use backend::CudaBackend;
pub use error::CudaInitError;
