//! Metal pipeline stages — per-stage, format-aware Metal dispatches.
//!
//! Each stage is a pure free function that takes a `ComputeCommandEncoder`
//! plus the pipelines, buffers, and per-layer metadata it needs. The
//! callers (`ops::full_pipeline::dispatch_full_pipeline` for prefill and
//! `MetalBackend::decode_token` for per-token decode) compose these
//! stages into the per-layer orchestration they need.
//!
//! This split isolates the format-dispatch logic (Q4_K / Q4_KF / Q6_K /
//! Q4_0 / Q8_0) that used to be inlined across both files, and gives the
//! golden-value tests one place to aim at when a shader/layout change
//! moves a stage's output.

/// Byte length of a 4-byte scalar (`u32`/`f32`) passed by value through
/// `set_bytes`. Metal's binding API takes a raw length, so this names
/// the one size every scalar slot in this crate uses.
pub(crate) const SCALAR_BYTES: u64 = 4;

pub mod attention;
pub mod bias_add;
pub mod ffn;
mod ffn_down;
pub mod input_norm;
pub mod layer_scalar;
pub mod o_proj;
pub mod qk_norm;
pub mod qkv_proj;
pub mod quant_matvec;
pub mod residual;
pub mod rope;
pub mod rope_freq;
pub mod sinks;
