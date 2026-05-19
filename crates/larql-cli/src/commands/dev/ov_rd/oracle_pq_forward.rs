use std::collections::HashMap;

use larql_inference::attention::{
    run_attention_block_with_pre_o_and_all_attention_weights,
    run_attention_block_with_pre_o_and_reduced_qk_attention_weights, SharedKV,
};
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_ffn, run_layer_with_ffn};
use larql_inference::{hidden_to_raw_logits, WeightFfn};
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};

use super::address::top_feature_ids_from_activation_row;
use super::basis::{RoundtripPatchMetrics, WoRoundtripBasis, ZPcaBasis};
use super::pq::{ModeDTable, PqCodebook};
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::stats::StaticHeadMeans;
use super::types::HeadId;

pub(super) fn forward_q4k_oracle_pq_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    codebook: &PqCodebook,
    stratum: &str,
) -> Result<(Array2<f32>, RoundtripPatchMetrics, Vec<Vec<usize>>), Box<dyn std::error::Error>> {
    let mut metrics = None;
    let mut oracle_codes = Vec::new();

    let h = larql_inference::vindex::predict_kquant_hidden_with_mapped_pre_o_head(
        weights,
        token_ids,
        index,
        head.layer,
        head.head,
        |original_head| {
            let mut replacement = Vec::with_capacity(original_head.len());
            let mut pre_sq = 0.0;
            let mut visible_sq = 0.0;
            let mut count = 0usize;
            for pos in 0..original_head.nrows() {
                let row = original_head.row(pos);
                let values = row
                    .as_slice()
                    .ok_or("pre-W_O head row was not contiguous during PQ")?;
                let base = means.positions.get(pos).unwrap_or(&means.global);
                let residual = values
                    .iter()
                    .zip(base.iter())
                    .map(|(&yi, &bi)| yi - bi)
                    .collect::<Vec<_>>();
                let z = basis.residual_to_z(&residual);
                let coords = pca_basis.coordinates_with_rank(&z, codebook.config.k);
                let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                let quantized_coords = codebook.quantize_from_indices_for_stratum(&codes, stratum);
                oracle_codes.push(codes);
                let z_projected = pca_basis.reconstruct_from_coordinates(&quantized_coords);
                let residual_projected = basis.z_to_residual(&z_projected);
                let projected = residual_projected
                    .into_iter()
                    .zip(base.iter())
                    .map(|(ri, &bi)| ri + bi)
                    .collect::<Vec<_>>();
                for (&original, &recon) in values.iter().zip(projected.iter()) {
                    let delta = original as f64 - recon as f64;
                    pre_sq += delta * delta;
                }
                let delta = values
                    .iter()
                    .zip(projected.iter())
                    .map(|(&original, &recon)| original as f64 - recon as f64)
                    .collect::<Vec<_>>();
                visible_sq += basis.visible_sq_norm(&delta);
                count += 1;
                replacement.extend_from_slice(&projected);
            }
            metrics = Some(RoundtripPatchMetrics {
                pre_wo_l2: (pre_sq / count.max(1) as f64).sqrt(),
                wo_visible_l2: (visible_sq / count.max(1) as f64).sqrt(),
            });
            Array2::from_shape_vec((original_head.nrows(), original_head.ncols()), replacement)
                .map_err(|err| err.to_string())
        },
    )?;

    Ok((
        h,
        metrics.ok_or("oracle PQ did not visit target layer")?,
        oracle_codes,
    ))
}

pub(super) fn forward_q4k_oracle_pq_mode_d_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    codebook: &PqCodebook,
    mode_d_table: &ModeDTable,
    stratum: &str,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let hidden_size = weights.hidden_size;
    larql_inference::vindex::predict_kquant_hidden_with_mapped_head_residual_delta(
        weights,
        token_ids,
        index,
        head.layer,
        head.head,
        |original_head| {
            let mut replacement_delta = Vec::with_capacity(original_head.nrows() * hidden_size);
            for pos in 0..original_head.nrows() {
                let row = original_head.row(pos);
                let values = row
                    .as_slice()
                    .ok_or("pre-W_O head row was not contiguous during Mode D PQ")?;
                let base = means.positions.get(pos).unwrap_or(&means.global);
                let residual = values
                    .iter()
                    .zip(base.iter())
                    .map(|(&yi, &bi)| yi - bi)
                    .collect::<Vec<_>>();
                let z = basis.residual_to_z(&residual);
                let coords = pca_basis.coordinates_with_rank(&z, codebook.config.k);
                let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                let delta =
                    mode_d_table.delta_for_position_codes_with_stratum(pos, &codes, stratum);
                replacement_delta.extend_from_slice(&delta);
            }
            Array2::from_shape_vec((original_head.nrows(), hidden_size), replacement_delta)
                .map_err(|err| err.to_string())
        },
    )
    .map_err(Into::into)
}

pub(super) fn forward_q4k_predicted_address_mode_d_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    mode_d_table: &ModeDTable,
    predicted_codes_by_position: &[Vec<usize>],
    stratum: &str,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut replacement_delta = Vec::with_capacity(token_ids.len() * weights.hidden_size);
    for pos in 0..token_ids.len() {
        let codes = predicted_codes_by_position
            .get(pos)
            .ok_or("missing predicted address for sequence position")?;
        let delta = mode_d_table.delta_for_position_codes_with_stratum(pos, codes, stratum);
        replacement_delta.extend_from_slice(&delta);
    }
    let replacement_delta =
        Array2::from_shape_vec((token_ids.len(), weights.hidden_size), replacement_delta)?;
    larql_inference::vindex::predict_kquant_hidden_with_replaced_head_residual_delta(
        weights,
        token_ids,
        index,
        head.layer,
        head.head,
        &replacement_delta,
    )
    .map_err(Into::into)
}

pub(super) fn capture_layer_input_hidden(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..target_layer {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let step = {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let ffn = WeightFfn { weights };
            run_layer_with_ffn(
                weights,
                &h,
                layer,
                &ffn,
                false,
                ple_inputs.get(layer),
                shared_kv,
            )
            .map(|(h_new, _, kv_out)| (h_new, kv_out))
        };
        if let Some((h_new, kv_out)) = step {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        } else {
            remove_layer_tensors(weights, inserted);
            return Err(format!("layer {layer} returned no output").into());
        }
        remove_layer_tensors(weights, inserted);
    }

    Ok(h)
}

pub(super) fn capture_prev_ffn_feature_keys(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
    feature_top_k: usize,
) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let mut prev_features_by_pos = vec![Vec::<usize>::new(); token_ids.len()];
    if target_layer == 0 || feature_top_k == 0 {
        return Ok(prev_features_by_pos);
    }

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..target_layer {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let step = {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let ffn = WeightFfn { weights };
            run_layer_with_ffn(
                weights,
                &h,
                layer,
                &ffn,
                layer + 1 == target_layer,
                ple_inputs.get(layer),
                shared_kv,
            )
            .map(|(h_new, activation, kv_out)| (h_new, activation, kv_out))
        };
        if let Some((h_new, activation, kv_out)) = step {
            if let Some(activation) = activation {
                prev_features_by_pos = activation
                    .rows()
                    .into_iter()
                    .map(|row| top_feature_ids_from_activation_row(row, feature_top_k))
                    .collect();
            }
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        } else {
            remove_layer_tensors(weights, inserted);
            return Err(format!("layer {layer} returned no output").into());
        }
        remove_layer_tensors(weights, inserted);
    }

    Ok(prev_features_by_pos)
}

pub(super) fn capture_ffn_first_feature_keys(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    target_layer: usize,
    feature_top_k: usize,
) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..=target_layer {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        if layer == target_layer {
            let ffn = WeightFfn { weights };
            let (_, activation) = run_ffn(weights, &h, layer, &ffn, feature_top_k > 0);
            remove_layer_tensors(weights, inserted);
            if let Some(activation) = activation {
                return Ok(activation
                    .rows()
                    .into_iter()
                    .map(|row| top_feature_ids_from_activation_row(row, feature_top_k))
                    .collect());
            }
            return Ok(vec![Vec::<usize>::new(); token_ids.len()]);
        }

        let step = {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let ffn = WeightFfn { weights };
            run_layer_with_ffn(
                weights,
                &h,
                layer,
                &ffn,
                false,
                ple_inputs.get(layer),
                shared_kv,
            )
            .map(|(h_new, _, kv_out)| (h_new, kv_out))
        };
        if let Some((h_new, kv_out)) = step {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        } else {
            remove_layer_tensors(weights, inserted);
            return Err(format!("layer {layer} returned no output").into());
        }
        remove_layer_tensors(weights, inserted);
    }

    Err(format!("target layer {target_layer} was not reached").into())
}

pub(super) fn capture_attention_relation_rows(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..=head.layer {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        if layer == head.layer {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let (_, _, all_weights) = run_attention_block_with_pre_o_and_all_attention_weights(
                weights, &h, layer, shared_kv,
            )
            .ok_or_else(|| {
                format!(
                    "all-position attention capture failed at L{}H{}",
                    head.layer, head.head
                )
            })?;
            remove_layer_tensors(weights, inserted);
            return all_weights.heads.get(head.head).cloned().ok_or_else(|| {
                format!("attention capture missing L{}H{}", head.layer, head.head).into()
            });
        }

        let step = {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let ffn = WeightFfn { weights };
            run_layer_with_ffn(
                weights,
                &h,
                layer,
                &ffn,
                false,
                ple_inputs.get(layer),
                shared_kv,
            )
            .map(|(h_new, _, kv_out)| (h_new, kv_out))
        };
        if let Some((h_new, kv_out)) = step {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        } else {
            remove_layer_tensors(weights, inserted);
            return Err(format!("layer {layer} returned no output").into());
        }
        remove_layer_tensors(weights, inserted);
    }

    Err(format!("target layer {} was not reached", head.layer).into())
}

pub(super) fn capture_reduced_qk_attention_rows(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    qk_rank: usize,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..=head.layer {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        if layer == head.layer {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let (_, _, all_weights) =
                run_attention_block_with_pre_o_and_reduced_qk_attention_weights(
                    weights, &h, layer, shared_kv, qk_rank,
                )
                .ok_or_else(|| {
                    format!(
                        "reduced-QK attention capture failed at L{}H{} rank {}",
                        head.layer, head.head, qk_rank
                    )
                })?;
            remove_layer_tensors(weights, inserted);
            return all_weights.heads.get(head.head).cloned().ok_or_else(|| {
                format!(
                    "reduced-QK attention capture missing L{}H{}",
                    head.layer, head.head
                )
                .into()
            });
        }

        let step = {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let ffn = WeightFfn { weights };
            run_layer_with_ffn(
                weights,
                &h,
                layer,
                &ffn,
                false,
                ple_inputs.get(layer),
                shared_kv,
            )
            .map(|(h_new, _, kv_out)| (h_new, kv_out))
        };
        if let Some((h_new, kv_out)) = step {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        } else {
            remove_layer_tensors(weights, inserted);
            return Err(format!("layer {layer} returned no output").into());
        }
        remove_layer_tensors(weights, inserted);
    }

    Err(format!("target layer {} was not reached", head.layer).into())
}

pub(super) fn final_logits(weights: &larql_inference::ModelWeights, h: &Array2<f32>) -> Vec<f32> {
    let last = h.nrows().saturating_sub(1);
    let h_last = h.slice(s![last..last + 1, ..]).to_owned();
    hidden_to_raw_logits(weights, &h_last)
}
