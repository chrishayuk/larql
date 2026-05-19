//! Honest production pipeline — real computation, no over-caching.
//!
//! - L0-12: cached (template-fixed, proven at 0.999 cosine — legitimate)
//! - L13-33: interleaved attention (CPU BLAS) + FFN (GPU Q4 via compute crate)
//! - Logits: GPU Q4 matvec against lm_head (1ms)
//!
//! Every entity-dependent layer is computed. No approximate residuals.
//! This is what `larql bench` and the streaming-demo runner use.

use super::super::{CachedLayerGraph, LayerGraph};
use crate::layer_graph::logits::finalize_logits;
use crate::layer_graph::prefill::prefill_kv_cache_cpu;
use crate::model::ModelWeights;
use larql_compute::prelude::*;

/// Honest production pipeline: real computation, no over-caching.
///
/// - L0-12: cached (template-fixed, proven at 0.999 cosine — legitimate)
/// - L13-33: interleaved attention (CPU BLAS) + FFN (GPU Q4 via compute crate)
/// - Logits: GPU Q4 matvec against lm_head (1ms)
///
/// Every entity-dependent layer is computed. No approximate residuals.
#[allow(clippy::too_many_arguments)]
pub fn predict_honest(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
) -> crate::forward::PredictResult {
    let norm_offset = weights.arch.norm_weight_offset();

    // Pass 0: cached layers (legitimate — template-fixed)
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    for layer in 0..layer_range.start {
        if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
            h = output.residual;
        }
    }

    // GPU pipeline: decode (seq=1) uses decode_token/full_pipeline_q4,
    // prefill (seq>1) uses prefill_kquant for GPU-accelerated multi-position inference.
    let seq_len = h.shape()[0];
    let used_gpu = if backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
        let gate_index: &dyn larql_vindex::GateIndex = index;
        // Prefer Q4_K FFN (Ollama-compatible) over Q4_0
        let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_kquant_mmap_ref() {
            (Some(mmap), true)
        } else {
            (gate_index.interleaved_q4_mmap_ref(), false)
        };
        let has_q4k = index.attn_kquant_layer_data(layer_range.start).is_some();
        let has_q8 = index.attn_q8_layer_data(layer_range.start).is_some();

        if let Some(q4_ffn_mmap) = q4_ffn {
            let intermediate = gate_index.num_features(layer_range.start);
            let hidden = weights.hidden_size;
            if intermediate > 0 && (has_q4k || has_q8) {
                let ffn_format = if ffn_is_q4k {
                    larql_compute::QuantFormat::Q4_K
                } else {
                    larql_compute::QuantFormat::Q4_0
                };
                let q4_ffn_per_matrix = ffn_format
                    .packed_matrix_bytes(intermediate, hidden)
                    .expect("Q4 interleaved FFN format must have packed geometry");
                // q4_ffn_per_layer computed inside build_pipeline_layers
                let arch = &*weights.arch;

                let layers = crate::layer_graph::pipeline_layer::build_pipeline_layers(
                    weights,
                    index,
                    layer_range.clone(),
                    q4_ffn_mmap,
                    q4_ffn_per_matrix,
                    ffn_format,
                );

                let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
                let qk_norm = arch.attn_q_norm_key(layer_range.start).is_some();

                if seq_len == 1 {
                    // Decode path (seq=1): try KV-cached decode first, then full_pipeline
                    let x: Vec<f32> = h.row(0).to_vec();

                    if let Some(result) = backend.decode_token(&layers, &x, hidden, intermediate) {
                        let mut row = h.row_mut(0);
                        for j in 0..hidden {
                            row[j] = result[j];
                        }
                        return finalize_logits(
                            weights,
                            tokenizer,
                            &h,
                            top_k,
                            index,
                            backend,
                            norm_offset,
                        );
                    }

                    if let Some(result) = backend.full_pipeline_q4(
                        &layers,
                        &x,
                        hidden,
                        intermediate,
                        1,
                        qk_norm,
                        softcap,
                    ) {
                        let mut row = h.row_mut(0);
                        for j in 0..hidden {
                            row[j] = result[j];
                        }
                        true
                    } else {
                        false
                    }
                } else if !arch.has_post_norms() {
                    // Prefill path (seq>1): GPU Q4 pipeline for pre-norm models (Llama, Mistral)
                    // Post-norm models (Gemma3) fall through to CPU — prefill.rs post-norm
                    // handling needs further work (see ADR-009).
                    let x: Vec<f32> = h.as_slice().unwrap_or(&[]).to_vec();

                    if let Some(result) = backend.prefill_kquant(
                        &layers,
                        &x,
                        hidden,
                        intermediate,
                        seq_len,
                        qk_norm,
                        softcap,
                    ) {
                        // Copy result back to h matrix (all positions)
                        for s in 0..seq_len {
                            let mut row = h.row_mut(s);
                            for j in 0..hidden {
                                row[j] = result[s * hidden + j];
                            }
                        }

                        // Populate KV cache via CPU for subsequent decode
                        // (lightweight: just QKV projection + RoPE, no FFN)
                        prefill_kv_cache_cpu(weights, token_ids, index, backend, &layer_range);

                        true
                    } else {
                        false
                    }
                } else {
                    // Post-norm models (Gemma3): CPU prefill (correct) → GPU logits (fast)
                    // CPU handles post-norms correctly. Use CPU hidden state, GPU for logits only.
                    // KV cache populated for future decode_token calls (token generation).
                    backend.reset_kv_cache();

                    let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
                    let mut h_cpu = h.clone();
                    for (rel_idx, abs_layer) in layer_range.clone().enumerate() {
                        let (h_post_attn, k_rope, v) =
                            crate::attention::gpu::run_attention_with_kv_backend(
                                weights,
                                &h_cpu,
                                abs_layer,
                                Some(backend),
                            )
                            .unwrap();

                        if backend.has_kv_cache() {
                            let k_flat = k_rope.as_slice().unwrap_or(&[]);
                            let v_flat = v.as_slice().unwrap_or(&[]);
                            backend.populate_kv_layer(
                                rel_idx,
                                k_flat,
                                v_flat,
                                seq_len,
                                weights.arch.num_kv_heads_for_layer(abs_layer),
                                weights.arch.head_dim_for_layer(abs_layer),
                            );
                        }

                        let (h_out, _) = crate::forward::run_ffn(
                            weights,
                            &h_post_attn,
                            abs_layer,
                            &walk_ffn,
                            false,
                        );
                        h_cpu = h_out;
                    }

                    // Use correct CPU hidden state, finalize with GPU logits
                    h = h_cpu;
                    return finalize_logits(
                        weights,
                        tokenizer,
                        &h,
                        top_k,
                        index,
                        backend,
                        norm_offset,
                    );
                }
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };

    // CPU fallback: interleaved attention + FFN (for prefill or when GPU not available)
    if !used_gpu {
        let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
        for layer in layer_range {
            let (h_post_attn, _, _) =
                crate::attention::run_attention_block_gpu(weights, &h, layer, false, None).unwrap();
            let (h_out, _) =
                crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h = h_out;
        }
    }

    finalize_logits(weights, tokenizer, &h, top_k, index, backend, norm_offset)
}
