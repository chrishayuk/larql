//! Stage A gates (V3-G5b-2): the plan executor against the
//! checkpoint-driven production forward — layer by layer.
//!
//! The two sides share **nothing but the fixture's weight values**: the
//! oracle loads the HF checkpoint through `larql-models` and runs
//! `larql-compute`'s production layers (BLAS, hooks); the executor reads
//! the encoded container through the closure-verified operand path and
//! computes with its own naive loops. Agreement is therefore a claim
//! about *semantics* — plan interpretation, operand binding, norm
//! placement, RoPE convention, residual order — not shared arithmetic.

mod accounting;
mod attention_kv_parity;
mod attn_res_2a_decode;
mod backend_rows;
mod bf16_gemv_bench;
mod bf16_residency;
mod bf16_zlib_execution;
mod compact_consumption;
mod continuation;
mod controls;
mod coverage_backend_decode;
mod coverage_device;
mod coverage_experts_production;
mod decode;
mod device;
mod draft_slice;
mod gated_delta_parity;
mod gated_delta_tiny;
mod hybrid_traversal;
mod hyper_connection;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod kda_metal;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod kda_native_parity;
mod kda_parity;
mod kda_parity_real;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod kda_q8_real;
mod kda_state;
mod kimi_kda_layer_real;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod kimi_layer_metal;
mod kimi_mla_layer_real;
mod kimi_moe_block;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod kimi_moe_metal;
mod kimi_moe_real;
mod kimi_router;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod kimi_two_layer;
mod mamba2_exec;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod mla_metal;
mod mla_parity;
mod mla_state;
mod mrope_parity;
mod nvfp4_decode;
mod nvfp4_projection;
mod output_gate_fused;
mod plan_fixtures;
mod projection_bench;
mod realization;
// Each module carries its OWN cfg: inserting a bare `mod` line above a
// gated one hands the attribute to the newcomer and silently un-gates
// the original — that exact capture broke six CI jobs on PR #346.
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod q2a_decode_bench;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod q2a_teacher_forced;
mod qw36c_layer0;
mod stack_dispatch_refusal;
mod stack_parity;
mod stack_real;
mod token2_real;
mod token_real;
mod token_tiny;
mod wave19_hc_batch;
mod wave19_hc_decode;
mod wave19_hc_substrate;
#[rustfmt::skip]
mod qw2_tiny_fixture;
mod gated_delta_refusal;
mod gemma4;
mod gemma4_refusals;
mod generate_baseline;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod generate_metal;
mod generate_real;
mod golden;
mod kernels;
mod kimi_per_expert_prepared;
mod kquant_projection;
mod kquant_projection_real;
mod kv;
mod llama3_rope;
mod observe;
mod overrides;
mod parity;
mod partial_residency;
mod recurrence_shape;
mod replay_capture;
mod requirements;
mod residency;
mod residency_budget;
mod residency_census;
mod routed;
mod seam;
mod shared_projection;
mod sinks_bias;
mod smoke;
mod streaming;
mod timing;

// The fixture writers and geometry moved to the public
// `format::vindex3::fixtures` module (so sibling crates' integration
// tests can encode the same containers these gates certify). The
// re-exports keep every test file's `super::*` imports stable, with
// the dense geometry under its historical short names.
pub(super) use crate::format::vindex3::fixtures::{
    dense_f32_model, lcg_values, norm_values, ShardBuilder, DENSE_HEAD_DIM as HEAD_DIM,
    DENSE_HIDDEN as HIDDEN, DENSE_INTERMEDIATE as INTERMEDIATE, DENSE_LAYERS as LAYERS,
    DENSE_Q_HEADS as Q_HEADS, DENSE_VOCAB as VOCAB,
};
mod sigmoid_router;
mod step_many;

/// `step_many`'s gates run on the same encoded hybrid stack the
/// traversal gates use — one fixture, so the two cannot drift.
mod hybrid_traversal_fixture {
    pub(super) use super::hybrid_traversal::hybrid;
}
