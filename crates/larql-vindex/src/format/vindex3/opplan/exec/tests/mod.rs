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

mod attention_kv_parity;
mod bf16_gemv_bench;
mod bf16_residency;
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
mod mrope_parity;
mod output_gate_fused;
mod plan_fixtures;
mod projection_bench;
mod qw36c_layer0;
#[rustfmt::skip]
mod qw2_tiny_fixture;
mod gated_delta_refusal;
mod gemma4;
mod gemma4_refusals;
mod golden;
mod kernels;
mod kv;
mod observe;
mod overrides;
mod parity;
mod recurrence_shape;
mod residency;
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
