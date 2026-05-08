//! CPU and backend forward paths driven by Q4_K / Q6_K vindexes.
//!
//! The normal CPU path reads attention Q/K/V/O and FFN gate/up/down from
//! `weights.tensors` as f32 matrices. For Q4/Q6 vindexes those tensors are
//! materialized one layer at a time, then removed before the next layer. This
//! module keeps that layer-scoped tensor lifetime in one place while exposing
//! focused entry points for hidden-state forward, generation, hooks,
//! interventions, remote FFN, Metal decode, and per-layer FFN serving.

mod dequant;
mod generation;
mod hidden;
mod hooks;
mod interventions;
mod metal;
mod remote_ffn;
mod tensors;
mod walk_ffn;

pub use generation::{
    generate_q4k_cpu, generate_q4k_cpu_constrained, generate_q4k_cpu_constrained_streaming,
    generate_q4k_cpu_constrained_streaming_sampled, generate_q4k_cpu_remote, is_end_of_turn,
    predict_q4k,
};
pub use hidden::{predict_q4k_hidden, prefill_q4k_from_embeddings};
pub use hooks::predict_q4k_hidden_hooked;
pub use interventions::{
    predict_q4k_hidden_with_mapped_head_residual_delta, predict_q4k_hidden_with_mapped_pre_o_head,
    predict_q4k_hidden_with_original_head_residual_delta,
    predict_q4k_hidden_with_replaced_head_residual_delta,
    predict_q4k_hidden_with_replaced_pre_o_head, predict_q4k_hidden_with_subtracted_pre_o_heads,
    predict_q4k_hidden_with_zeroed_pre_o_heads,
};
pub use metal::predict_q4k_metal;
pub use remote_ffn::{predict_q4k_hidden_with_ffn, predict_q4k_with_ffn};
pub use tensors::{insert_q4k_layer_tensors, remove_layer_tensors};
pub use walk_ffn::{q4k_ffn_forward_layer, q4k_ffn_forward_layer_q8k};
