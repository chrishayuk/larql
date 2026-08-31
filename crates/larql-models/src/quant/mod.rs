//! Quantization and dequantization — data format encoding/decoding.
//!
//! Supports:
//! - **half**: f16/bf16 ↔ f32 conversion
//! - **ggml**: GGML block quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
//! - **mxfp4**: Microscaling 4-bit floats with e8m0 scales (GPT-OSS/OpenAI)
//!
//! This module handles data format encoding/decoding only.
//! Compute operations (matvec, vecmat, GPU shaders) are in `larql-compute`.

pub mod fp4;
pub mod fp4_block;
pub mod fp8;
pub mod ggml;
pub mod half;
pub mod mxfp4;
pub mod nvfp4;
pub mod nvfp4_ggml;
