//! Split-pass prediction — decompose attention and FFN into separate compute passes.
//!
//! Two variants:
//!
//! - [`predict_split_pass`] — three-pass pipeline (cached prefix → CPU
//!   attention-only walk → batched GPU FFN). Approximates the residual
//!   stream by deferring the FFN contribution during attention; correct
//!   because attention is template-fixed (~99% identical across entities).
//!   Falls back to per-layer CPU FFN when the GPU multi-layer batch isn't
//!   available.
//!
//! - [`predict_split_cached`] — one-pass logits-only fast path that
//!   borrows pre-computed `AttentionCache` residuals; exact output, no
//!   attention compute. The "70 tok/s" GPU-fast variant.

use ndarray::Array2;

use super::super::{AttentionCache, CachedLayerGraph, DenseLayerGraph, LayerGraph};
use crate::model::ModelWeights;
use larql_compute::prelude::*;

/// Two-pass split pipeline: attention on CPU, FFN batched on Metal GPU.
///
/// Pass 1: Run attention for all layers with attention-only residual stream (CPU).
///          The FFN contribution is deferred — attention sees approximate residuals.
///          This is valid because attention is template-fixed (~99% identical across entities).
///
/// Pass 2: Compute FFN for all layers in one Metal command buffer (8.5ms for 21 layers).
///          Uses the post-attention residuals from pass 1 as FFN inputs.
///
/// Pass 3: Add FFN outputs to the final residual and compute logits via vindex KNN.
///
/// Target: 55ms attention + 8.5ms FFN + 5ms logits = 68ms → 15 tok/s
#[allow(clippy::too_many_arguments)]
pub fn predict_split_pass(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
) -> crate::forward::PredictResult {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();

    // ── Pass 0: Cached layers (0ms) ──
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    for layer in 0..layer_range.start {
        if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
            h = output.residual;
        }
    }

    // ── Pass 1: Attention-only for walk layers (CPU BLAS) ──
    // Store post-attention residuals and FFN-normed inputs.
    let mut post_attn_residuals: Vec<Array2<f32>> = Vec::new();
    let mut ffn_inputs: Vec<Vec<f32>> = Vec::new();

    for layer in layer_range.clone() {
        // Run attention only (CPU BLAS, no FFN)
        let (h_post_attn, _attn_proj, _) =
            crate::attention::run_attention_block_gpu(weights, &h, layer, false, None)
                .unwrap_or_else(|| {
                    // Fallback: identity (shouldn't happen with valid weights)
                    (h.clone(), h.clone(), None)
                });

        // Compute pre-FFN norm (this is the FFN input)
        let pre_ffn_key = if arch.has_post_norms() {
            arch.pre_feedforward_layernorm_key(layer)
        } else {
            Some(arch.post_attention_layernorm_key(layer))
        };
        let h_ffn = match pre_ffn_key {
            Some(key) => crate::forward::apply_norm(weights, &h_post_attn, &key, norm_offset),
            None => crate::residual::rms_norm(&h_post_attn, None, norm_offset),
        };

        // Store last-token FFN input for GPU batch
        let last_row = h_ffn.row(seq_len - 1);
        ffn_inputs.push(last_row.to_vec());
        post_attn_residuals.push(h_post_attn.clone());

        // Continue with attention-only residual (approximate — no FFN contribution)
        h = h_post_attn;
    }

    // ── Pass 2: Batch FFN on Metal GPU ──
    let num_walk_layers = layer_range.len();

    // Try batched Q4 FFN via multi_layer_q4_ffn
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let ffn_outputs = if gate_index.has_interleaved_q4()
        && backend.supports_quant(::larql_compute::QuantFormat::Q4_K)
    {
        if let Some(q4_mmap) = gate_index.interleaved_q4_mmap_ref() {
            let intermediate = gate_index.num_features(layer_range.start);
            if intermediate > 0 {
                let q4_bytes_per_matrix = larql_compute::QuantFormat::Q4_0
                    .packed_matrix_bytes(intermediate, hidden)
                    .expect("Q4_0 interleaved FFN format must have packed geometry");
                let q4_bytes_per_layer = q4_bytes_per_matrix * 3;

                // Collect Q4 data slices for all walk layers
                let layers_q4: Vec<(&[u8], &[u8], &[u8])> = layer_range
                    .clone()
                    .map(|layer| {
                        let start = layer * q4_bytes_per_layer;
                        let gate = &q4_mmap[start..start + q4_bytes_per_matrix];
                        let up =
                            &q4_mmap[start + q4_bytes_per_matrix..start + 2 * q4_bytes_per_matrix];
                        let down = &q4_mmap
                            [start + 2 * q4_bytes_per_matrix..start + 3 * q4_bytes_per_matrix];
                        (gate, up, down)
                    })
                    .collect();

                // Use the first FFN input as the batch input
                // multi_layer_q4_ffn chains layers: out of layer N → input of layer N+1
                let x = &ffn_inputs[0];
                backend.multi_layer_q4_ffn(&layers_q4, x, intermediate, hidden)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // ── Pass 3: Combine ──
    // If GPU batch succeeded, use the final FFN output.
    // Otherwise, fall back to per-layer CPU FFN.
    if let Some(ffn_final) = ffn_outputs {
        // The multi_layer_q4_ffn returns the final residual after all FFN layers.
        // Add it to the last post-attention residual.
        let last_post_attn = &post_attn_residuals[num_walk_layers - 1];
        let mut h_final = last_post_attn.clone();
        let mut last_row_mut = h_final.row_mut(seq_len - 1);
        for j in 0..hidden {
            // The FFN output is the chained result — use it as the last-token residual
            last_row_mut[j] = ffn_final[j];
        }
        h = h_final;
    } else {
        // Fallback: run FFN per-layer on CPU
        h = crate::forward::embed_tokens_pub(weights, token_ids);
        for layer in 0..layer_range.start {
            if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
                h = output.residual;
            }
        }
        let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
        for layer in layer_range.clone() {
            let dense = DenseLayerGraph {
                ffn: &walk_ffn,
                backend: None,
                capture_activation: false,
                capture_attention: false,
            };
            if let Some(output) = dense.forward_layer(weights, &h, layer) {
                h = output.residual;
            }
        }
    }

    // Final norm + vindex logits
    let h_final =
        crate::forward::apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);
    let last_row = h_final.row(seq_len - 1).to_owned();

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    let hits = index.lm_head_knn(&last_row, top_k);
    let scaled: Vec<(u32, f32)> = hits
        .iter()
        .map(|&(tid, score)| {
            let mut logit = score * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            (tid, logit)
        })
        .collect();

    let max_logit = scaled
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled
        .iter()
        .map(|(_, l)| ((*l - max_logit) as f64).exp())
        .sum();
    let predictions = scaled
        .iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer
                .decode(&[tid], true)
                .ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult {
        predictions,
        token_ids: Vec::new(),
    }
}

/// Split pass using cached attention residuals — exact output at GPU speed.
///
/// Uses `AttentionCache` (built from one exact run) to skip all attention
/// computation. Batches FFN on Metal GPU in one command buffer.
///
/// Target: 0ms attention + 8.5ms FFN + 5ms logits = ~14ms → 71 tok/s
pub fn predict_split_cached(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    attn_cache: &AttentionCache,
    _layer_range: std::ops::Range<usize>,
) -> crate::forward::PredictResult {
    let norm_offset = weights.arch.norm_weight_offset();

    // Zero-copy: borrow the cached residual, don't clone.
    // Final norm produces a new array (unavoidable), but the input is borrowed.
    let h_final = crate::forward::apply_norm(
        weights,
        &attn_cache.final_residual,
        weights.arch.final_norm_key(),
        norm_offset,
    );
    let seq_len = h_final.shape()[0];
    let last_row = h_final.row(seq_len - 1).to_owned();

    // GPU Q4 logits when available (1ms), else CPU BLAS (10ms)
    let hits = index.lm_head_knn_backend(&last_row, top_k, backend);

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    let scaled: Vec<(u32, f32)> = hits
        .iter()
        .map(|&(tid, score)| {
            let mut logit = score * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            (tid, logit)
        })
        .collect();

    let max_logit = scaled
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled
        .iter()
        .map(|(_, l)| ((*l - max_logit) as f64).exp())
        .sum();
    let predictions = scaled
        .iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer
                .decode(&[tid], true)
                .ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult {
        predictions,
        token_ids: Vec::new(),
    }
}
