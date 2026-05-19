//! CPU and backend forward paths driven by Q4_K / Q6_K vindexes.
//!
//! The normal CPU path reads attention Q/K/V/O and FFN gate/up/down from
//! `weights.tensors` as f32 matrices. For Q4/Q6 vindexes those tensors are
//! materialized one layer at a time, then removed before the next layer. This
//! module keeps that layer-scoped tensor lifetime in one place while exposing
//! focused entry points for hidden-state forward, generation, hooks,
//! interventions, remote FFN, Metal decode, and per-layer FFN serving.

mod cached;
mod dequant;
mod generation;
mod hidden;
mod hooks;
mod interventions;
mod metal;
mod remote_ffn;
mod tensors;
mod walk_ffn;

pub use cached::{
    attention_decode_step_native, ffn_decode_step_native, fused_decode_step, fused_prefill,
    predict_kquant_decode_step, predict_kquant_decode_step_direct, predict_kquant_prefill,
    supports_cached_decode, supports_direct_matvec_decode, CachedTimings, CpuKvCache,
};

pub(crate) use generation::generate_kquant_cpu_constrained_streaming_sampled_with_eos;
pub use generation::{
    generate_kquant_cpu, generate_kquant_cpu_constrained,
    generate_kquant_cpu_constrained_streaming, generate_kquant_cpu_constrained_streaming_sampled,
    generate_kquant_cpu_remote, is_end_of_turn, predict_kquant,
};
pub use hidden::predict_kquant_hidden;
pub use hooks::predict_kquant_hidden_hooked;
pub use interventions::{
    predict_kquant_hidden_with_mapped_head_residual_delta,
    predict_kquant_hidden_with_mapped_pre_o_head,
    predict_kquant_hidden_with_original_head_residual_delta,
    predict_kquant_hidden_with_replaced_head_residual_delta,
    predict_kquant_hidden_with_replaced_pre_o_head,
    predict_kquant_hidden_with_subtracted_pre_o_heads,
    predict_kquant_hidden_with_zeroed_pre_o_heads,
};
pub use metal::{
    predict_kquant_metal, predict_kquant_metal_capture_pre_wo, predict_kquant_metal_hidden,
    predict_kquant_metal_with_replaced_head_residual_delta,
};
pub use remote_ffn::{predict_kquant_hidden_with_ffn, predict_kquant_with_ffn};
pub use tensors::{insert_q4k_layer_tensors, remove_layer_tensors};
pub use walk_ffn::{kquant_ffn_forward_layer, kquant_ffn_forward_layer_q8k};
