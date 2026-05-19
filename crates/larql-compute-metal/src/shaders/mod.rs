//! Metal shader registry — one file per shader, compiled as one library.
//!
//! Each shader module exports a `SHADER` constant with the MSL source.
//! `all_shaders()` concatenates them with the common header for compilation.

pub mod common;
pub mod sgemm;
pub mod sgemm_transb;
// Q4_0 matvec: only `q4_matvec_v4` ships. Earlier variants
// (q4_matvec, _v2, _v3, _v5) were experiments kept around for ad-hoc
// benchmarks; deleted 2026-04-25 because every shader compiled into
// the library is reachable by `library.get_function(name)` and was a
// pipeline-selection hazard (see ROADMAP P0b / q4_matvec_v4 ship-log).
// If a future variant lands, add its file here AND a `Kernel` marker
// implementing `metal::kernel::TiledKernel` so the binding site reads
// it by *path*, not by hand-typed string.
pub mod activation;
pub mod attn_fused;
pub mod causal_attention;
pub mod f16_gemv;
pub mod f32_gemv;
pub mod fused_attention;
pub mod fused_ops;
pub mod geglu;
pub mod graph_walk_knn;
pub mod kv_append_attend_fused;
pub mod kv_attention;
pub mod layer_norm;
pub mod per_layer_embed;
pub mod post_attn_residual_norm_store;
pub mod post_ffn_norm_residual_add;
pub mod q4_f32_matvec;
pub mod q4_matvec_v4;
pub mod q4_sparse_matvec;
pub mod q4_vecmat;
pub mod q4k_ffn_gate_up;
pub mod q4k_ffn_gate_up_8sg;
pub mod q4k_ffn_gate_up_coop;
pub mod q4k_ffn_gate_up_f16acc;
pub mod q4k_geglu_down;
pub mod q4k_matmul;
pub mod q4k_matvec;
pub mod q4k_matvec_8sg;
pub mod q4k_matvec_stride32;
pub mod q4k_q6k_qkv_proj;
pub mod q4k_qkv_proj;
pub mod q4kf_ffn_gate_up;
pub mod q4kf_qkv_proj;
pub mod q6k_geglu_down;
pub mod q6k_geglu_gelu_tanh_down_cached;
pub mod q6k_matvec;
pub mod q6k_matvec_8sg;
pub mod q8_attn_proj;
pub mod q8_matvec;
pub mod qk_norm;
pub mod qk_norm_rope_fused;
pub mod quantize_q8;
pub mod residual_inject;
pub mod rope;
pub mod turboquant_decode;
pub mod turboquant_encode;
pub mod v_norm;

/// Concatenate all shaders into one MSL source string for compilation.
pub fn all_shaders() -> String {
    let mut src = String::with_capacity(32768);
    src.push_str(common::HEADER);
    // f32 matmul
    src.push_str(sgemm::SHADER);
    src.push_str(sgemm_transb::SHADER);
    src.push_str(f32_gemv::SHADER);
    // Templated MSL: substitutes `MAX_SIMDGROUPS_PER_TG` (argmax) and
    // `K_TOPK` / `PARTIAL_TG_SZ` / `MAX_SIMDGROUPS_PER_TG` (topk) from
    // the Rust constants of the same name so the shaders can't drift
    // from their dispatchers.
    src.push_str(&f32_gemv::argmax_shader_source());
    src.push_str(&f32_gemv::topk_shader_source());
    src.push_str(f16_gemv::SHADER);
    // Q4 dense matvec
    src.push_str(q4_matvec_v4::SHADER);
    // Q4 other
    src.push_str(q4_vecmat::SHADER);
    src.push_str(q4_f32_matvec::SHADER);
    src.push_str(q4_sparse_matvec::SHADER);
    // Q8
    src.push_str(q8_matvec::SHADER);
    // Element-wise
    src.push_str(geglu::SHADER);
    src.push_str(quantize_q8::SHADER);
    src.push_str(residual_inject::SHADER);
    // Attention
    src.push_str(causal_attention::SHADER);
    src.push_str(kv_attention::SHADER);
    src.push_str(kv_append_attend_fused::SHADER);
    src.push_str(attn_fused::SHADER);
    src.push_str(rope::SHADER);
    src.push_str(fused_attention::SHADER);
    src.push_str(fused_ops::SHADER);
    src.push_str(q8_attn_proj::SHADER);
    src.push_str(q4k_matvec::SHADER);
    src.push_str(q4k_matvec_8sg::SHADER);
    src.push_str(q4k_matvec_stride32::SHADER);
    src.push_str(q4k_matmul::SHADER);
    src.push_str(q4k_qkv_proj::SHADER);
    src.push_str(q4k_q6k_qkv_proj::SHADER);
    src.push_str(q4kf_qkv_proj::SHADER);
    src.push_str(q4k_ffn_gate_up::SHADER);
    src.push_str(q4k_ffn_gate_up_f16acc::SHADER);
    src.push_str(q4k_ffn_gate_up_8sg::SHADER);
    src.push_str(q4k_ffn_gate_up_coop::SHADER);
    src.push_str(q4k_q6k_qkv_proj::NORMED_SHADER);
    src.push_str(q4k_geglu_down::SHADER);
    src.push_str(q4kf_ffn_gate_up::SHADER);
    src.push_str(q6k_geglu_down::SHADER);
    src.push_str(q6k_geglu_gelu_tanh_down_cached::SHADER);
    src.push_str(q6k_matvec::SHADER);
    src.push_str(q6k_matvec_8sg::SHADER);
    // Standalone activations (non-gated FFN)
    src.push_str(activation::SHADER);
    // LayerNorm (StarCoder2, GPT-2)
    src.push_str(layer_norm::SHADER);
    // V-norm (parameter-free, Gemma 4)
    src.push_str(v_norm::SHADER);
    // QK-norm (learned-weight per-head RMS, Gemma 3/4)
    src.push_str(qk_norm::SHADER);
    src.push_str(qk_norm_rope_fused::SHADER);
    src.push_str(post_attn_residual_norm_store::SHADER);
    src.push_str(post_ffn_norm_residual_add::SHADER);
    // Per-Layer Embeddings (Gemma 4 E2B)
    src.push_str(per_layer_embed::SHADER);
    // TurboQuant (KV cache compression)
    src.push_str(turboquant_encode::SHADER);
    src.push_str(turboquant_decode::SHADER);
    // Graph walk KNN
    src.push_str(graph_walk_knn::SHADER);
    src
}
