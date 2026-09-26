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
mod attested_fidelity;
mod attn_res_2a_decode;
mod attn_res_2b_batch;
mod attn_res_2b_controls;
mod attn_res_substrate;
mod backend_rows;
mod bf16_gemv_bench;
mod bf16_residency;
mod bf16_zlib_execution;
mod carrier_entry;
mod carrier_write;
mod carrier_write_real;
mod codec_owned_weight;
mod compact_consumption;
mod composed_floor;
mod continuation;
mod continuation_handoff;
mod continuation_identity;
mod continuation_registry;
mod controls;
mod coverage_backend_decode;
mod coverage_device;
mod coverage_experts_production;
mod decode;
mod dense_ffn_placement;
mod device;
mod device_gate_refusal;
mod draft_slice;
mod external_embedding;
mod f32_planes_execution;
mod ffn_down_input_admission;
mod fp8_carriage;
mod gated_delta_parity;
mod gated_delta_tiny;
mod head_replay;
mod history_range;
mod hybrid_traversal;
mod hyper_connection;
mod intervene;
mod intervene_heads;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod kda_metal;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod kda_native_parity;
mod kda_parity;
mod kda_parity_full_rank_gate;
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
mod lowering_identity;
mod lowering_pin;
mod lowering_registry;
mod mamba2_exec;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod mla_metal;
mod mla_parity;
mod mla_parity_output_gate;
mod mla_parity_q_lora;
mod mla_state;
mod mrope_parity;
mod nvfp4_decode;
mod nvfp4_projection;
mod output_gate_fused;
mod placement_seams;
mod plan_fixtures;
mod projection_bench;
mod provenance;
mod realization;
mod vq8_shared_execution;
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
mod head_observation;
mod head_observation_gates;
mod kernels;
mod kimi_per_expert_prepared;
mod kquant_projection;
mod kquant_projection_real;
mod kv;
mod lens;
mod linear_rope;
mod llama3_rope;
mod observe;
mod observe_stats;
mod overrides;
mod parity;
mod partial_residency;
mod payload_prefix;
mod recurrence_shape;
mod reference_mrope;
mod reference_refusal_arms;
mod replay_capture;
mod requirements;
mod residency;
mod residency_budget;
mod residency_census;
mod routed;
mod seam;
mod selected_output_head;
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
mod latent_moe_execution;
mod latent_moe_parity;
mod one_shot_continuation;
mod prefetch;
mod sigmoid_router;
mod stages_and_routing;
mod step_many;

/// `step_many`'s gates run on the same encoded hybrid stack the
/// traversal gates use — one fixture, so the two cannot drift.
mod hybrid_traversal_fixture {
    pub(super) use super::hybrid_traversal::hybrid;
}

/// `row/v1` selected for `plan` — the explicit continuation a test names
/// now that the executor has no default (CONTINUATION-PLUGIN-1, C3).
pub(crate) fn row_continuation(
    plan: &super::super::ComponentOpPlan,
) -> super::continuation_registry::SelectedContinuation {
    let mut registry = super::continuation_registry::ContinuationRegistry::new();
    registry.register(Box::new(super::kv::RowFactory)).unwrap();
    let geometry = super::continuation::plan_continuation_geometry(plan).unwrap();
    registry
        .select(
            &super::kv::RowKvState::identity(),
            &super::continuation_authority::ContinuationConfig::empty(),
            &geometry,
        )
        .unwrap()
}
