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

pub mod attention;
pub mod ffn;
pub mod input_norm;
pub mod layer_scalar;
pub mod o_proj;
pub mod qk_norm;
pub mod qkv_proj;
pub mod quant_matvec;
pub mod residual;
pub mod rope;
