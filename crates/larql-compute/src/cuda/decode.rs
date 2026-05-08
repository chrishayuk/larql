//! CUDA decode backend integration.
//!
//! Fused decode kernels live in `attn.rs` today as low-level helpers. The
//! `DecodeBackend` trait implementation intentionally keeps the default
//! fallback behavior until the full layer pipeline can consume those helpers.

use crate::backend::DecodeBackend;

use super::backend::CudaBackend;

impl DecodeBackend for CudaBackend {}
