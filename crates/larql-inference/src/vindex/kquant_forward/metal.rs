use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;
use tokenizers::Tokenizer;

use crate::forward::PredictResult;

const MIN_KV_CACHE_SEQ: usize = 64;

/// End-to-end predict on a Q4_K vindex driven by a Metal (or any Q4-capable)
/// `ComputeBackend`.
pub fn predict_kquant_metal(
    weights: &ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
) -> PredictResult {
    use crate::layer_graph::pipeline_layer::{build_arch_params, resolve_attn_weights};
    use larql_compute::QuantFormat;

    let arch = &*weights.arch;
    let num_layers = weights.num_layers;

    let layers: Vec<_> = (0..num_layers)
        .map(|layer| {
            let (wq, wk, wv, wo) =
                resolve_attn_weights(index, layer).expect("attn Q4K slices missing for layer");
            let [(gate_bytes, gate_fmt), (up_bytes, up_fmt), (down_bytes, down_fmt)] = index
                .interleaved_kquant_layer_data(layer)
                .expect("ffn Q4K slices missing for layer");
            fn to_format(s: &str) -> QuantFormat {
                match s {
                    "Q4_K" => QuantFormat::Q4_K,
                    "Q6_K" => QuantFormat::Q6_K,
                    other => panic!(
                        "kquant_forward: registry tag {other:?} has no compute::QuantFormat mapping"
                    ),
                }
            }
            let gate = larql_compute::QuantWeight {
                data: gate_bytes,
                scales: None,
                format: to_format(gate_fmt),
            };
            let up = larql_compute::QuantWeight {
                data: up_bytes,
                scales: None,
                format: to_format(up_fmt),
            };
            let down = larql_compute::QuantWeight {
                data: down_bytes,
                scales: None,
                format: to_format(down_fmt),
            };
            build_arch_params(weights, layer, wq, wk, wv, wo, gate, up, down)
        })
        .collect();

    let max_seq = token_ids.len().max(MIN_KV_CACHE_SEQ);
    let shapes: Vec<(usize, usize)> = layers
        .iter()
        .map(|l| (l.num_kv_heads, l.head_dim))
        .collect();
    backend.preallocate_kv_cache_per_layer(&shapes, max_seq);
    backend.reset_kv_cache();

    let hidden = weights.hidden_size;
    let embed = &weights.embed;
    let embed_scale = arch.embed_scale();

    let mut h_vec: Vec<f32> = Vec::with_capacity(hidden);
    for &tok in token_ids {
        let row = embed.row(tok as usize);
        let x: Vec<f32> = row.iter().map(|v| v * embed_scale).collect();

        let out = backend
            .decode_token(&layers, &x, hidden, weights.intermediate_size)
            .expect("backend doesn't support decode_token - need Metal with Q4 kernels");
        h_vec = out;
    }

    let h_last = ndarray::Array2::from_shape_vec((1, hidden), h_vec).expect("residual shape");
    crate::forward::predict::logits_to_predictions_pub(weights, &h_last, tokenizer, top_k, 1.0)
}

/// Metal-accelerated head-replacement forward pass via `full_pipeline_q4_with_head_replacement`.
///
/// Uses the same KV-cache + per-position RoPE path as `prefill_kquant`, so all
/// seq_len positions have correct positional encodings. The intervention hooks
/// in `dispatch_full_pipeline` zero head `target_head` at `target_layer` and
/// inject `replacement_delta` in its place.
pub fn predict_kquant_metal_with_replaced_head_residual_delta(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
    target_layer: usize,
    target_head: usize,
    replacement_delta: &Array2<f32>,
) -> Option<Array2<f32>> {
    use crate::layer_graph::pipeline_layer::{build_arch_params, resolve_attn_weights};
    use larql_compute::QuantFormat;

    let arch = &*weights.arch;
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    let embed_scale = arch.embed_scale();
    let seq_len = token_ids.len();
    if seq_len == 0 {
        return None;
    }

    let layers: Vec<_> = (0..num_layers)
        .map(|layer| {
            let (wq, wk, wv, wo) =
                resolve_attn_weights(index, layer).expect("attn Q4K slices missing");
            let [(gate_bytes, gate_fmt), (up_bytes, up_fmt), (down_bytes, down_fmt)] = index
                .interleaved_kquant_layer_data(layer)
                .expect("ffn Q4K slices missing");
            fn to_fmt(s: &str) -> QuantFormat {
                match s {
                    "Q4_K" => QuantFormat::Q4_K,
                    "Q6_K" => QuantFormat::Q6_K,
                    o => panic!("unknown quant format {o:?}"),
                }
            }
            build_arch_params(
                weights,
                layer,
                wq,
                wk,
                wv,
                wo,
                larql_compute::QuantWeight {
                    data: gate_bytes,
                    scales: None,
                    format: to_fmt(gate_fmt),
                },
                larql_compute::QuantWeight {
                    data: up_bytes,
                    scales: None,
                    format: to_fmt(up_fmt),
                },
                larql_compute::QuantWeight {
                    data: down_bytes,
                    scales: None,
                    format: to_fmt(down_fmt),
                },
            )
        })
        .collect();

    // All token embeddings concatenated: [seq_len × hidden].
    let mut x_all = Vec::with_capacity(seq_len * hidden);
    for &tok in token_ids {
        x_all.extend(
            weights
                .embed
                .row(tok as usize)
                .iter()
                .map(|v| v * embed_scale),
        );
    }

    // Flat replacement delta: [seq_len × hidden].
    let delta_flat = replacement_delta.as_slice()?.to_vec();

    let result = backend.full_pipeline_q4_with_head_replacement(
        &layers,
        &x_all,
        hidden,
        weights.intermediate_size,
        seq_len,
        arch.attn_q_norm_key(0).is_some(),
        arch.attn_logit_softcapping().unwrap_or(0.0),
        target_layer,
        target_head,
        &delta_flat,
    )?;

    Array2::from_shape_vec((seq_len, hidden), result).ok()
}

/// Metal-accelerated baseline forward pass — full seq_len, no intervention.
///
/// Returns the hidden state for ALL positions `[seq_len × hidden]`. Uses
/// `full_pipeline_q4_with_head_replacement` with an identity delta (zero) so
/// the same per-position RoPE path that the intervention pass uses is applied,
/// ensuring bit-identical residuals between the baseline and program passes.
///
/// Returns `None` if the backend does not support the path.
pub fn predict_kquant_metal_hidden(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
) -> Option<Array2<f32>> {
    use crate::layer_graph::pipeline_layer::{build_arch_params, resolve_attn_weights};
    use larql_compute::QuantFormat;

    let arch = &*weights.arch;
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    let embed_scale = arch.embed_scale();
    let seq_len = token_ids.len();
    if seq_len == 0 {
        return None;
    }

    let layers: Vec<_> = (0..num_layers)
        .map(|layer| {
            let (wq, wk, wv, wo) =
                resolve_attn_weights(index, layer).expect("attn Q4K slices missing");
            let [(gate_bytes, gate_fmt), (up_bytes, up_fmt), (down_bytes, down_fmt)] = index
                .interleaved_kquant_layer_data(layer)
                .expect("ffn Q4K slices missing");
            fn to_fmt(s: &str) -> QuantFormat {
                match s {
                    "Q4_K" => QuantFormat::Q4_K,
                    "Q6_K" => QuantFormat::Q6_K,
                    o => panic!("unknown quant format {o:?}"),
                }
            }
            build_arch_params(
                weights,
                layer,
                wq,
                wk,
                wv,
                wo,
                larql_compute::QuantWeight {
                    data: gate_bytes,
                    scales: None,
                    format: to_fmt(gate_fmt),
                },
                larql_compute::QuantWeight {
                    data: up_bytes,
                    scales: None,
                    format: to_fmt(up_fmt),
                },
                larql_compute::QuantWeight {
                    data: down_bytes,
                    scales: None,
                    format: to_fmt(down_fmt),
                },
            )
        })
        .collect();

    let mut x_all = Vec::with_capacity(seq_len * hidden);
    for &tok in token_ids {
        x_all.extend(
            weights
                .embed
                .row(tok as usize)
                .iter()
                .map(|v| v * embed_scale),
        );
    }

    // Use full_pipeline_q4 with per-position RoPE but no intervention.
    let result = backend.full_pipeline_q4_with_head_replacement(
        &layers,
        &x_all,
        hidden,
        weights.intermediate_size,
        seq_len,
        arch.attn_q_norm_key(0).is_some(),
        arch.attn_logit_softcapping().unwrap_or(0.0),
        // Setting target_layer to num_layers means the hooks never fire
        // (the condition `l == target_layer` is never true in the layer loop).
        num_layers,
        0,
        &[], // empty delta — hooks will never fire so length doesn't matter
    )?;

    Array2::from_shape_vec((seq_len, hidden), result).ok()
}

/// Capture the target head's pre-W_O output at `target_layer` using only GPU
/// layers 0..=target_layer. Returns `[seq_len × head_dim]` f32.
///
/// For L0H6 (target_layer=0): only runs 1/34 GPU layers vs the full CPU pass.
/// The caller uses the captured data to compute oracle PQ codes on CPU.
pub fn predict_kquant_metal_capture_pre_wo(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
    target_layer: usize,
    target_head: usize,
) -> Option<Vec<f32>> {
    use crate::layer_graph::pipeline_layer::{build_arch_params, resolve_attn_weights};
    use larql_compute::QuantFormat;

    let arch = &*weights.arch;
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    let embed_scale = arch.embed_scale();
    let seq_len = token_ids.len();
    if seq_len == 0 {
        return None;
    }

    // Only need layers 0..=target_layer for the capture.
    let layers: Vec<_> = (0..num_layers)
        .map(|layer| {
            let (wq, wk, wv, wo) =
                resolve_attn_weights(index, layer).expect("attn Q4K slices missing");
            let [(gate_bytes, gate_fmt), (up_bytes, up_fmt), (down_bytes, down_fmt)] = index
                .interleaved_kquant_layer_data(layer)
                .expect("ffn Q4K slices missing");
            fn to_fmt(s: &str) -> QuantFormat {
                match s {
                    "Q4_K" => QuantFormat::Q4_K,
                    "Q6_K" => QuantFormat::Q6_K,
                    o => panic!("unknown quant format {o:?}"),
                }
            }
            build_arch_params(
                weights,
                layer,
                wq,
                wk,
                wv,
                wo,
                larql_compute::QuantWeight {
                    data: gate_bytes,
                    scales: None,
                    format: to_fmt(gate_fmt),
                },
                larql_compute::QuantWeight {
                    data: up_bytes,
                    scales: None,
                    format: to_fmt(up_fmt),
                },
                larql_compute::QuantWeight {
                    data: down_bytes,
                    scales: None,
                    format: to_fmt(down_fmt),
                },
            )
        })
        .collect();

    let mut x_all = Vec::with_capacity(seq_len * hidden);
    for &tok in token_ids {
        x_all.extend(
            weights
                .embed
                .row(tok as usize)
                .iter()
                .map(|v| v * embed_scale),
        );
    }

    backend.full_pipeline_kquant_capture_pre_wo(
        &layers,
        &x_all,
        hidden,
        weights.intermediate_size,
        seq_len,
        arch.attn_q_norm_key(0).is_some(),
        arch.attn_logit_softcapping().unwrap_or(0.0),
        target_layer,
        target_head,
    )
}
