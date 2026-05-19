//! Vindex integration — WalkFfn for inference.
//!
//! The build pipeline, weight IO, clustering, and format handling
//! now live in `larql-vindex`. This module provides only WalkFfn
//! (the FFN backend that uses vindex KNN for feature selection).

pub mod dequant;
mod kquant_forward;
pub mod l1_cache;
mod loader;
mod walk_config;
mod walk_ffn;

pub use dequant::ensure_attn_tensors_dequantised;
pub(crate) use kquant_forward::generate_kquant_cpu_constrained_streaming_sampled_with_eos;
pub use kquant_forward::{
    attention_decode_step_native, ffn_decode_step_native, fused_decode_step, fused_prefill,
    generate_kquant_cpu, generate_kquant_cpu_constrained,
    generate_kquant_cpu_constrained_streaming, generate_kquant_cpu_constrained_streaming_sampled,
    generate_kquant_cpu_remote, insert_q4k_layer_tensors, is_end_of_turn, kquant_ffn_forward_layer,
    kquant_ffn_forward_layer_q8k, predict_kquant, predict_kquant_decode_step,
    predict_kquant_decode_step_direct, predict_kquant_hidden, predict_kquant_hidden_hooked,
    predict_kquant_hidden_with_ffn, predict_kquant_hidden_with_mapped_head_residual_delta,
    predict_kquant_hidden_with_mapped_pre_o_head,
    predict_kquant_hidden_with_original_head_residual_delta,
    predict_kquant_hidden_with_replaced_head_residual_delta,
    predict_kquant_hidden_with_replaced_pre_o_head,
    predict_kquant_hidden_with_subtracted_pre_o_heads,
    predict_kquant_hidden_with_zeroed_pre_o_heads, predict_kquant_metal,
    predict_kquant_metal_capture_pre_wo, predict_kquant_metal_hidden,
    predict_kquant_metal_with_replaced_head_residual_delta, predict_kquant_prefill,
    predict_kquant_with_ffn, remove_layer_tensors, supports_cached_decode,
    supports_direct_matvec_decode, CachedTimings, CpuKvCache,
};
pub use l1_cache::FfnL1Cache;
pub use loader::{open_inference_vindex, ENV_VINDEX_PATH};
pub use walk_config::WalkFfnConfig;
pub use walk_ffn::WalkFfn;
