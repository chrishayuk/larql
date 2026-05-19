use super::config::GridRuntimeConfig;
use super::setup::{build_grid_pipeline_setup, reset_and_preallocate_grid_kv, RemotePatch};
use super::GridGenerateResult;
use crate::ffn::moe_remote::RemoteMoeError;
use crate::ffn::{FfnBackend, LayerShardedBackend};
use crate::forward::apply_norm;
use crate::layer_graph::generate::detok::Detokenizer;
use crate::layer_graph::generate::eos::EosConfig;
use crate::layer_graph::generate::policy::{
    build_special_suppress_set_with_policy, pick_next_filtered_with_policy,
};
use crate::residual::rms_norm;
use larql_compute::cpu::ops::q4k_q8k_dot::{quantize_x_to_q8k, Q8KActivation};
use larql_compute::prelude::*;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

/// Autoregressive generation with Metal GPU attention and remote dense FFN.
///
/// For dense models (not MoE) where the entire FFN should be offloaded to a
/// remote server (`--ffn URL`). Metal handles attention on the local GPU;
/// every layer's FFN is a round trip to `remote` via `LayerShardedBackend::forward`.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_remote_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    remote: &LayerShardedBackend,
    eos: &EosConfig,
) -> Result<GridGenerateResult, RemoteMoeError> {
    let runtime = GridRuntimeConfig::from_env();
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let setup = build_grid_pipeline_setup(weights, index, RemotePatch::Ffn)?;
    let layers = setup.layers;
    let hidden = setup.hidden;
    let intermediate = setup.intermediate;

    reset_and_preallocate_grid_kv(weights, backend);

    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(&prompt_ids);

    let suppress = build_special_suppress_set_with_policy(tokenizer, eos, &runtime.token_policy);

    for &tok_id in &prompt_ids {
        let tok_embed = crate::forward::embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            let h_normed = apply_norm_for_ffn(weights, h_post_attn, layer);
            let x = ndarray::Array2::from_shape_vec((1, hidden), h_normed)
                .expect("shape must match hidden");
            let raw_out = remote.forward(layer, &x).row(0).to_vec();
            apply_post_ffn_norm(weights, &raw_out, layer)
        };

        let h = backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut moe_fn);
        last_hidden_vec = h.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode_token_with_moe returned None during prefill".into())
        })?;
    }
    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();
    let mut ffn_rtt_ms = Vec::new();

    let prefill_h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let h_norm0 = apply_norm(weights, &prefill_h_arr, arch.final_norm_key(), norm_offset);
    let last0 = h_norm0.row(0).to_owned();
    let first_id = pick_next_filtered_with_policy(
        index,
        weights,
        &last0,
        backend,
        &suppress,
        tokenizer,
        &runtime.token_policy,
    );

    let first_tok = detok.push(first_id);
    let first_is_eos = eos.is_eos_with_tokenizer(first_id, &first_tok, tokenizer);
    tokens.push(first_tok);
    current_ids.push(first_id);
    if first_is_eos || tokens.len() >= max_tokens {
        return Ok(GridGenerateResult {
            tokens,
            decode_ms: vec![0.0],
            ffn_rtt_ms: Vec::new(),
        });
    }

    for _step in 0..max_tokens.saturating_sub(1) {
        let t0 = std::time::Instant::now();
        let next_input_id = *current_ids.last().unwrap();

        let tok_embed = crate::forward::embed_tokens_pub(weights, &[next_input_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();





        let step_ffn_cell = std::cell::Cell::new(0.0f64);
        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            let t_ffn = std::time::Instant::now();
            let x = ndarray::Array2::from_shape_vec((1, hidden), h_post_attn.to_vec())
                .expect("shape must match hidden");
            let result = remote.forward(layer, &x).row(0).to_vec();
            step_ffn_cell.set(step_ffn_cell.get() + t_ffn.elapsed().as_secs_f64() * 1000.0);
            result
        };









        let h_vec = backend
            .decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut moe_fn)
            .ok_or_else(|| {
                RemoteMoeError::BadResponse("decode_token_with_moe returned None".into())
            })?;


        last_hidden_vec = h_vec;
        let kv_after = backend.kv_cache_len();
        eprintln!("[debug] after decode step: kv_len={kv_after}");
        break;
        ffn_rtt_ms.push(step_ffn_cell.get());



        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let last_hidden = h_normed.row(0).to_owned();

        let next_id = pick_next_filtered_with_policy(

            index,
            weights,
            &last_hidden,
            backend,
            &suppress,
            tokenizer,
            &runtime.token_policy,
        );

        let token_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        decode_ms.push(token_wall_ms);

        let tok_str = detok.push(next_id);
        let is_eos = eos.is_eos_with_tokenizer(next_id, &tok_str, tokenizer);
        tokens.push(tok_str);
        current_ids.push(next_id);
        if is_eos {
            break;
        }
    }

    Ok(GridGenerateResult {
        tokens,
        decode_ms,
        ffn_rtt_ms,
    })
}

fn apply_norm_for_ffn(weights: &ModelWeights, h_post_attn: &[f32], layer: usize) -> Vec<f32> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();

    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };

    let h = ndarray::Array2::from_shape_vec((1, h_post_attn.len()), h_post_attn.to_vec())
        .expect("apply_norm_for_ffn: shape error");

    let normed = match pre_ffn_key {
        Some(ref key) => apply_norm(weights, &h, key, norm_offset),
        None => rms_norm(&h, None, norm_offset),
    };
    normed.row(0).to_vec()
}

fn apply_post_ffn_norm(weights: &ModelWeights, ffn_out: &[f32], layer: usize) -> Vec<f32> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let key = arch.post_feedforward_layernorm_key(layer);
    let h = ndarray::Array2::from_shape_vec((1, ffn_out.len()), ffn_out.to_vec())
        .expect("apply_post_ffn_norm: shape error");
    let normed = match key {
        Some(ref k) => apply_norm(weights, &h, k, norm_offset),
        None => {
            if arch.has_post_norms() {
                rms_norm(&h, None, norm_offset)
            } else {
                return ffn_out.to_vec();
            }
        }
    };
    normed.row(0).to_vec()
}

fn dispatch_ffn_with_q8k_fallback(
    remote: &LayerShardedBackend,
    weights: &ModelWeights,
    h_capture: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let hidden = h_capture.first().map(|v| v.len()).unwrap_or(0);
    if hidden == 0 || !hidden.is_multiple_of(crate::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS) {
        return remote.forward_predispatch_all(h_capture);
    }

    let q8k_all: Vec<Q8KActivation> = h_capture
        .iter()
        .enumerate()
        .map(|(layer, h)| {
            let h_ffn = apply_norm_for_ffn(weights, h, layer);
            quantize_x_to_q8k(&h_ffn)
        })
        .collect();

    let results = remote.forward_predispatch_all_q8k(&q8k_all);
    let any_zero_result = results.iter().any(|v| v.iter().all(|&x| x == 0.0));
    if any_zero_result {
        remote.forward_predispatch_all(h_capture)
    } else {
        results
    }
}

/// Batch pre-dispatch variant of [`generate_with_remote_ffn`].
#[allow(clippy::too_many_arguments)]
pub fn generate_with_remote_ffn_batch(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    index: &VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
    remote: &LayerShardedBackend,
    eos: &EosConfig,
    predispatch_iters: usize,
) -> Result<GridGenerateResult, RemoteMoeError> {
    let runtime = GridRuntimeConfig::from_env();
    let predispatch_iters = predispatch_iters.max(1);
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let setup = build_grid_pipeline_setup(weights, index, RemotePatch::Ffn)?;
    let layers = setup.layers;
    let hidden = setup.hidden;
    let intermediate = setup.intermediate;
    let num_layers = setup.num_layers;
    reset_and_preallocate_grid_kv(weights, backend);

    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(&prompt_ids);

    let suppress = build_special_suppress_set_with_policy(tokenizer, eos, &runtime.token_policy);

    for &tok_id in &prompt_ids {
        let tok_embed = crate::forward::embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();
        let kv_len = backend.kv_cache_len();

        let mut h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        {
            let h_cap = &mut h_capture;
            let mut cap_fn = |layer: usize, h: &[f32]| -> Vec<f32> {
                if h_cap.len() == layer {
                    h_cap.push(h.to_vec());
                }
                vec![0.0f32; hidden]
            };
            backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut cap_fn);
        }
        backend.truncate_kv_cache(kv_len);

        let mut h2_final: Option<Vec<f32>> = None;
        for iter in 0..predispatch_iters {
            let is_final = iter + 1 == predispatch_iters;
            let h2 = dispatch_ffn_with_q8k_fallback(remote, weights, &h_capture);

            if !is_final {
                let mut new_cap: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
                let h2r = &h2;
                let nc = &mut new_cap;
                let mut fn_apply = |l: usize, h: &[f32]| -> Vec<f32> {
                    if nc.len() == l {
                        nc.push(h.to_vec());
                    }
                    h2r.get(l).cloned().unwrap_or_else(|| vec![0.0f32; hidden])
                };
                backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut fn_apply);
                backend.truncate_kv_cache(kv_len);
                h_capture = new_cap;
            } else {
                let h2r = &h2;
                let mut fn_final = |l: usize, _: &[f32]| -> Vec<f32> {
                    h2r.get(l).cloned().unwrap_or_else(|| vec![0.0f32; hidden])
                };
                h2_final = backend.decode_token_with_moe(
                    &layers,
                    &x_tok,
                    hidden,
                    intermediate,
                    &mut fn_final,
                );
            }
        }
        last_hidden_vec = h2_final.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode returned None during prefill".into())
        })?;
    }

    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();
    let prefill_h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let h_norm0 = apply_norm(weights, &prefill_h_arr, arch.final_norm_key(), norm_offset);
    let first_id = pick_next_filtered_with_policy(
        index,
        weights,
        &h_norm0.row(0).to_owned(),
        backend,
        &suppress,
        tokenizer,
        &runtime.token_policy,
    );
    let first_tok = detok.push(first_id);
    let first_is_eos = eos.is_eos_with_tokenizer(first_id, &first_tok, tokenizer);
    tokens.push(first_tok);
    current_ids.push(first_id);
    if first_is_eos || tokens.len() >= max_tokens {
        return Ok(GridGenerateResult {
            tokens,
            decode_ms: vec![0.0],
            ffn_rtt_ms: Vec::new(),
        });
    }

    let mut ffn_rtt_ms: Vec<f64> = Vec::new();



    let kv_before = backend.kv_cache_len();
    eprintln!("[debug] decode start: kv_len={kv_before}");
    for _step in 0..max_tokens.saturating_sub(1) {
        let t0 = std::time::Instant::now();
        let next_input_id = *current_ids.last().unwrap();


        let tok_embed = crate::forward::embed_tokens_pub(weights, &[next_input_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();
        let kv_len = backend.kv_cache_len();

        let mut h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        {
            let h_cap = &mut h_capture;
            let mut cap_fn = |layer: usize, h: &[f32]| -> Vec<f32> {
                if h_cap.len() == layer {
                    h_cap.push(h.to_vec());
                }
                vec![0.0f32; hidden]
            };
            backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut cap_fn);
        }
        backend.truncate_kv_cache(kv_len);

        let mut h_out_opt: Option<Vec<f32>> = None;
        let mut step_ffn_ms = 0.0f64;

        for iter in 0..predispatch_iters {
            let is_final = iter + 1 == predispatch_iters;
            let t_ffn = std::time::Instant::now();
            let h2 = dispatch_ffn_with_q8k_fallback(remote, weights, &h_capture);
            step_ffn_ms += t_ffn.elapsed().as_secs_f64() * 1000.0;

            if !is_final {
                let h2r = &h2;
                let mut new_h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
                let new_h = &mut new_h_capture;
                let mut fn_apply = |l: usize, h: &[f32]| -> Vec<f32> {
                    if new_h.len() == l {
                        new_h.push(h.to_vec());
                    }
                    h2r.get(l).cloned().unwrap_or_else(|| vec![0.0f32; hidden])
                };
                backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut fn_apply);
                backend.truncate_kv_cache(kv_len);
                h_capture = new_h_capture;
            } else {
                let h2r = &h2;
                let mut fn_final = |l: usize, _: &[f32]| -> Vec<f32> {
                    h2r.get(l).cloned().unwrap_or_else(|| vec![0.0f32; hidden])
                };
                h_out_opt = backend.decode_token_with_moe(
                    &layers,
                    &x_tok,
                    hidden,
                    intermediate,
                    &mut fn_final,
                );
            }
        }

        let h_vec = h_out_opt.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode_token_with_moe returned None".into())
        })?;

        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_vec)
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let last_hidden = h_normed.row(0).to_owned();

        let next_id = pick_next_filtered_with_policy(
            index,
            weights,
            &last_hidden,
            backend,
            &suppress,
            tokenizer,
            &runtime.token_policy,
        );

        let token_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        decode_ms.push(token_wall_ms);
        ffn_rtt_ms.push(step_ffn_ms);

        let tok_str = detok.push(next_id);
        let is_eos = eos.is_eos_with_tokenizer(next_id, &tok_str, tokenizer);
        tokens.push(tok_str);
        current_ids.push(next_id);
        if is_eos {
            break;
        }
    }

    Ok(GridGenerateResult {
        tokens,
        decode_ms,
        ffn_rtt_ms,
    })
}
