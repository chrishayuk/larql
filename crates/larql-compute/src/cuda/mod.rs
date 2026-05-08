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
mod error;
mod matmul;
mod quant_matvec;

pub use backend::CudaBackend;
pub use error::CudaInitError;
