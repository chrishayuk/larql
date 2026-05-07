//! # CUDA backend (scaffolding)
//!
//! Phase 1 of the [`cuda-and-rotorquant-kv`][change] OpenSpec change: a
//! feature-flag stub that compiles, registers `Capability::Cuda`, and
//! returns from `default_backend()` on Linux when the `cuda` feature is
//! enabled and a CUDA driver is reachable.
//!
//! Real kernels land in follow-up changes (`cuda-f32-baseline`,
//! `cuda-q4-matvec`, `cuda-fused-attention`); every dispatch path here
//! either returns `None` (default trait impls already do this for the
//! optional methods) or `unimplemented!()` for the required ones.
//! Construction is the only thing that actually does work today: it
//! checks that a CUDA device is reachable so callers get a clear error
//! instead of a silent fallback.
//!
//! [change]: ../../../../openspec/changes/cuda-and-rotorquant-kv/

mod backend;
mod cache;
mod dequant;
mod driver;
mod error;
mod matmul;

pub use backend::CudaBackend;
pub use error::CudaInitError;
