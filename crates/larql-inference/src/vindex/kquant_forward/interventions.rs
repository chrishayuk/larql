use std::collections::HashMap;

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::attention::SharedKV;
use crate::forward::embed_tokens_pub;
use crate::forward::ple::precompute_per_layer_inputs;
use crate::forward::{
    run_layer_with_ffn, run_layer_with_mapped_head_residual_delta,
    run_layer_with_mapped_pre_o_head, run_layer_with_original_head_residual_delta,
    run_layer_with_replaced_head_residual_delta, run_layer_with_replaced_pre_o_head,
    run_layer_with_subtracted_pre_o_heads, run_layer_with_zeroed_pre_o_heads,
};

use super::tensors::{insert_q4k_layer_tensors, remove_layer_tensors};

#[allow(clippy::type_complexity)]
fn predict_kquant_hidden_with_target_layer_step<F>(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
    mut run_target_layer: F,
    label: &str,
) -> Result<Array2<f32>, String>
where
    F: FnMut(
        &ModelWeights,
        &Array2<f32>,
        usize,
        &dyn crate::ffn::FfnBackend,
        Option<&Array2<f32>>,
        Option<&SharedKV>,
    ) -> Result<Option<(Array2<f32>, Option<SharedKV>)>, String>,
{
    if weights.arch.is_hybrid_moe() {
        return Err(format!(
            "{label} currently supports dense FFN vindexes only"
        ));
    }
    if target_layer >= weights.num_layers {
        return Err(format!(
            "target_layer {target_layer} out of range for {} layers",
            weights.num_layers
        ));
    }

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..weights.num_layers {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        let ffn_backend = crate::ffn::WeightFfn { weights };

        let step = if layer == target_layer {
            run_target_layer(
                weights,
                &h,
                layer,
                &ffn_backend,
                ple_inputs.get(layer),
                shared_kv,
            )?
        } else {
            run_layer_with_ffn(
                weights,
                &h,
                layer,
                &ffn_backend,
                false,
                ple_inputs.get(layer),
                shared_kv,
            )
            .map(|(h_new, _, kv_out)| (h_new, kv_out))
        };

        let Some((h_new, kv_out)) = step else {
            remove_layer_tensors(weights, inserted);
            return Err(format!("{label} failed at layer {layer}"));
        };
        h = h_new;
        if let Some(kv) = kv_out {
            kv_cache.insert(layer, kv);
        }
        remove_layer_tensors(weights, inserted);
    }

    Ok(h)
}

/// Compute final hidden states on a Q4_K/Q6_K vindex while mapping one
/// pre-W_O head at `target_layer`.
pub fn predict_kquant_hidden_with_mapped_pre_o_head<F>(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
    target_head: usize,
    mut map_head: F,
) -> Result<Array2<f32>, String>
where
    F: FnMut(&Array2<f32>) -> Result<Array2<f32>, String>,
{
    predict_kquant_hidden_with_target_layer_step(
        weights,
        token_ids,
        index,
        target_layer,
        |weights, h, layer, ffn_backend, ple_input, shared_kv| {
            let mut mapper_error = None;
            run_layer_with_mapped_pre_o_head(
                weights,
                h,
                layer,
                ffn_backend,
                target_head,
                ple_input,
                shared_kv,
                |original_head| match map_head(original_head) {
                    Ok(replacement) => Some(replacement),
                    Err(err) => {
                        mapper_error = Some(err);
                        None
                    }
                },
            )
            .ok_or_else(|| mapper_error.unwrap_or_else(|| "pre-W_O mapper returned None".into()))
            .map(Some)
        },
        "Q4K pre-W_O mapped forward",
    )
}

/// Compute final hidden states while replacing one pre-W_O head with a fixed
/// `(seq_len, head_dim)` matrix at `target_layer`.
pub fn predict_kquant_hidden_with_replaced_pre_o_head(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
    target_head: usize,
    replacement: &Array2<f32>,
) -> Result<Array2<f32>, String> {
    predict_kquant_hidden_with_target_layer_step(
        weights,
        token_ids,
        index,
        target_layer,
        |weights, h, layer, ffn_backend, ple_input, shared_kv| {
            Ok(run_layer_with_replaced_pre_o_head(
                weights,
                h,
                layer,
                ffn_backend,
                target_head,
                replacement,
                ple_input,
                shared_kv,
            ))
        },
        "Q4K pre-W_O replacement forward",
    )
}

/// Compute final hidden states while zeroing selected pre-W_O heads at one
/// target layer.
pub fn predict_kquant_hidden_with_zeroed_pre_o_heads(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
    heads: &[usize],
) -> Result<Array2<f32>, String> {
    predict_kquant_hidden_with_target_layer_step(
        weights,
        token_ids,
        index,
        target_layer,
        |weights, h, layer, ffn_backend, ple_input, shared_kv| {
            Ok(run_layer_with_zeroed_pre_o_heads(
                weights,
                h,
                layer,
                ffn_backend,
                heads,
                ple_input,
                shared_kv,
            ))
        },
        "Q4K pre-W_O zero forward",
    )
}

/// Compute final hidden states while subtracting selected pre-W_O heads at one
/// target layer after W_O projection.
pub fn predict_kquant_hidden_with_subtracted_pre_o_heads(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
    heads: &[usize],
) -> Result<Array2<f32>, String> {
    predict_kquant_hidden_with_target_layer_step(
        weights,
        token_ids,
        index,
        target_layer,
        |weights, h, layer, ffn_backend, ple_input, shared_kv| {
            Ok(run_layer_with_subtracted_pre_o_heads(
                weights,
                h,
                layer,
                ffn_backend,
                heads,
                ple_input,
                shared_kv,
            ))
        },
        "Q4K pre-W_O subtract forward",
    )
}

/// Compute final hidden states while replacing one attention head's residual
/// contribution at one target layer.
pub fn predict_kquant_hidden_with_replaced_head_residual_delta(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
    target_head: usize,
    replacement_delta: &Array2<f32>,
) -> Result<Array2<f32>, String> {
    predict_kquant_hidden_with_target_layer_step(
        weights,
        token_ids,
        index,
        target_layer,
        |weights, h, layer, ffn_backend, ple_input, shared_kv| {
            Ok(run_layer_with_replaced_head_residual_delta(
                weights,
                h,
                layer,
                ffn_backend,
                target_head,
                replacement_delta,
                ple_input,
                shared_kv,
            ))
        },
        "Q4K residual-delta replacement forward",
    )
}

/// Compute final hidden states while mapping one original pre-W_O head to a
/// residual-space replacement delta at `target_layer`.
pub fn predict_kquant_hidden_with_mapped_head_residual_delta<F>(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
    target_head: usize,
    mut map_head_delta: F,
) -> Result<Array2<f32>, String>
where
    F: FnMut(&Array2<f32>) -> Result<Array2<f32>, String>,
{
    predict_kquant_hidden_with_target_layer_step(
        weights,
        token_ids,
        index,
        target_layer,
        |weights, h, layer, ffn_backend, ple_input, shared_kv| {
            let mut mapper_error = None;
            run_layer_with_mapped_head_residual_delta(
                weights,
                h,
                layer,
                ffn_backend,
                target_head,
                ple_input,
                shared_kv,
                |original_head| match map_head_delta(original_head) {
                    Ok(replacement) => Some(replacement),
                    Err(err) => {
                        mapper_error = Some(err);
                        None
                    }
                },
            )
            .ok_or_else(|| {
                mapper_error.unwrap_or_else(|| "residual-delta mapper returned None".into())
            })
            .map(Some)
        },
        "Q4K residual-delta mapped forward",
    )
}

/// Compute final hidden states while replacing one head's residual contribution
/// with its original `pre_W_O @ W_O_head` delta at `target_layer`.
pub fn predict_kquant_hidden_with_original_head_residual_delta(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
    target_head: usize,
) -> Result<Array2<f32>, String> {
    predict_kquant_hidden_with_target_layer_step(
        weights,
        token_ids,
        index,
        target_layer,
        |weights, h, layer, ffn_backend, ple_input, shared_kv| {
            Ok(run_layer_with_original_head_residual_delta(
                weights,
                h,
                layer,
                ffn_backend,
                target_head,
                ple_input,
                shared_kv,
            ))
        },
        "Q4K original residual-delta forward",
    )
}
