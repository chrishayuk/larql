//! Vindex integration — WalkFfn for inference.
//!
//! The build pipeline, weight IO, clustering, and format handling
//! now live in `larql-vindex`. This module provides only WalkFfn
//! (the FFN backend that uses vindex KNN for feature selection).

pub mod l1_cache;
mod loader;
mod q4k_forward;
mod walk_config;
mod walk_ffn;

pub use l1_cache::FfnL1Cache;
pub use loader::open_inference_vindex;
pub use q4k_forward::{
    generate_q4k_cpu, generate_q4k_cpu_constrained, generate_q4k_cpu_constrained_streaming,
    generate_q4k_cpu_constrained_streaming_sampled, generate_q4k_cpu_remote,
    insert_q4k_layer_tensors, is_end_of_turn, predict_q4k, predict_q4k_full_vocab_probs,
    predict_q4k_hidden, predict_q4k_hidden_hooked, predict_q4k_hidden_with_ffn,
    predict_q4k_hidden_with_mapped_head_residual_delta, predict_q4k_hidden_with_mapped_pre_o_head,
    predict_q4k_hidden_with_original_head_residual_delta,
    predict_q4k_hidden_with_replaced_head_residual_delta,
    predict_q4k_hidden_with_replaced_pre_o_head, predict_q4k_hidden_with_subtracted_pre_o_heads,
    predict_q4k_hidden_with_zeroed_pre_o_heads, predict_q4k_metal, predict_q4k_with_ffn,
    prefill_q4k_from_embeddings, q4k_ffn_forward_layer, q4k_ffn_forward_layer_q8k,
    remove_layer_tensors,
};
pub use walk_config::WalkFfnConfig;
pub use walk_ffn::WalkFfn;
