use std::collections::HashMap;

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::attention::SharedKV;
use crate::forward::embed_tokens_pub;
use crate::forward::ple::precompute_per_layer_inputs;
use crate::forward::run_layer_with_ffn;

use super::tensors::{insert_q4k_layer_tensors, remove_layer_tensors};

/// Like [`predict_q4k_hidden`] but takes pre-embedded inputs and
/// returns the per-layer K/V cache alongside the final hidden state.
/// Used by the attention-service-routes prefill handler so it can
/// stash K/V into the session's `KvCache` without re-tokenising.
///
/// This variant SKIPS Per-Layer-Embeddings (Gemma 4 E2B). If the
/// model declares `has_per_layer_embeddings()`, the caller MUST
/// route through the token_ids-aware
/// [`predict_q4k_hidden`] path instead — otherwise the residual
/// stream loses the per-layer per-position contribution and the
/// output is gibberish.
///
/// Returns `(final_h, kv)` where:
/// - `final_h` is `[seq_len, hidden]` post-last-layer.
/// - `kv[layer]` is the layer's `(k_post_rope, v)` pair when the
///   layer participated, or `None` for skipped/KV-shared layers.
pub fn prefill_q4k_from_embeddings(
    weights: &mut ModelWeights,
    h0: Array2<f32>,
    index: &VectorIndex,
    moe_remote: Option<&crate::ffn::RemoteMoeBackend>,
) -> (Array2<f32>, Vec<Option<SharedKV>>) {
    let num_layers = weights.num_layers;
    let mut h = h0;
    let mut kvs: Vec<Option<SharedKV>> = vec![None; num_layers];
    let mut shared_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..num_layers {
        let inserted =
            insert_q4k_layer_tensors(weights, index, layer).unwrap_or_else(|err| panic!("{err}"));

        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| shared_cache.get(&src));
        let is_moe_layer = weights.arch.is_hybrid_moe();
        let ffn_backend = crate::ffn::WeightFfn { weights };

        if is_moe_layer {
            if let Some((h_new, kv_out)) = run_moe_layer_cpu(
                weights,
                &h,
                layer,
                &ffn_backend,
                None, // PLE skipped for embedding-input variant
                shared_kv,
                moe_remote,
            ) {
                h = h_new;
                if let Some(kv) = kv_out.clone() {
                    shared_cache.insert(layer, kv.clone());
                    kvs[layer] = Some(kv);
                }
            }
        } else if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            &ffn_backend,
            false,
            None, // PLE skipped
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out.clone() {
                shared_cache.insert(layer, kv.clone());
                kvs[layer] = Some(kv);
            }
        }

        remove_layer_tensors(weights, inserted);
    }

    (h, kvs)
}

/// Compute the final hidden state for `token_ids` against a Q4_K/Q6_K
/// vindex, dequantising attn + FFN one layer at a time. Returns the
/// `[seq_len, hidden]` array; caller owns the lm_head step.
pub fn predict_q4k_hidden(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    moe_remote: Option<&crate::ffn::RemoteMoeBackend>,
) -> Array2<f32> {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens_pub(weights, token_ids);

    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();
    let dump_dir = std::env::var("LARQL_CPU_DUMP_LAYERS").ok();
    if let Some(ref dir) = dump_dir {
        let slice = h.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/cpu_h_embed.f32"), &bytes);
    }

    for layer in 0..num_layers {
        let inserted =
            insert_q4k_layer_tensors(weights, index, layer).unwrap_or_else(|err| panic!("{err}"));

        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        let is_moe_layer = weights.arch.is_hybrid_moe();
        let ffn_backend = crate::ffn::WeightFfn { weights };
        if is_moe_layer {
            if let Some((h_new, kv_out)) = run_moe_layer_cpu(
                weights,
                &h,
                layer,
                &ffn_backend,
                ple_inputs.get(layer),
                shared_kv,
                moe_remote,
            ) {
                h = h_new;
                if let Some(kv) = kv_out {
                    kv_cache.insert(layer, kv);
                }
            }
        } else if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            &ffn_backend,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }

        remove_layer_tensors(weights, inserted);

        if let Some(ref dir) = dump_dir {
            let slice = h.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let path = format!("{dir}/cpu_layer_{layer:02}.f32");
            if let Err(e) = std::fs::write(&path, &bytes) {
                eprintln!("[dump] failed to write {path}: {e}");
            }
        }
    }

    h
}

/// Build `MoeRouterWeights` for a single layer from the model's vector store.
fn build_moe_router_weights<'a>(
    weights: &'a larql_models::ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Option<crate::ffn::MoeRouterWeights<'a>> {
    let router_key = arch.moe_router_key(layer)?;
    let router_proj = weights.vectors.get(&router_key)?.as_slice();
    let sl = |k: Option<String>| -> &'a [f32] {
        k.and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    };
    Some(crate::ffn::MoeRouterWeights {
        router_proj,
        router_scale: sl(arch.moe_router_scale_key(layer)),
        router_per_expert_scale: sl(arch.moe_router_per_expert_scale_key(layer)),
        router_norm: sl(arch.moe_router_norm_key(layer)),
        router_norm_parameter_free: arch.moe_router_norm_parameter_free(),
        router_input_scalar: arch.moe_router_input_scalar().unwrap_or(1.0),
        pre_experts_norm: sl(arch.moe_pre_experts_norm_key(layer)),
        post_experts_norm: sl(arch.moe_post_experts_norm_key(layer)),
        num_experts: arch.num_experts(),
        top_k: arch.num_experts_per_token(),
    })
}

/// CPU forward for one hybrid-MoE layer (Gemma 4 26B A4B).
fn run_moe_layer_cpu(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn crate::ffn::FfnBackend,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
    moe_remote: Option<&crate::ffn::RemoteMoeBackend>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let hidden = h.ncols();

    let (h_post_attn, kv_out) = if let Some(shared) = shared_kv {
        let (h_pa, _, _) =
            crate::attention::run_attention_block_shared(weights, h, layer, false, Some(shared))?;
        (h_pa, None)
    } else {
        let (h_pa, _, _, k_rope, v_final) =
            crate::attention::run_attention_block_with_kv_out(weights, h, layer, false, None)?;
        (h_pa, Some((k_rope, v_final)))
    };

    if let Ok(dir) = std::env::var("LARQL_CPU_DUMP_LAYERS") {
        let slice = h_post_attn.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = format!("{dir}/cpu_layer_{layer:02}_h_post_attn.f32");
        let _ = std::fs::write(&path, &bytes);
    }

    let (h_post_ffn_dense, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, ffn, false);
    let h1 = &h_post_ffn_dense - &h_post_attn;

    let seq_len = h_post_attn.nrows();
    let mut h2 = Array2::<f32>::zeros((seq_len, hidden));

    if let Some(remote) = moe_remote {
        if let Some(router) = build_moe_router_weights(weights, arch, layer) {
            match remote.forward_moe_seq(layer, &h_post_attn, &router, norm_offset, eps) {
                Ok(out) => h2 = out,
                Err(e) => eprintln!("[run_moe_layer_cpu] remote dispatch error L{layer}: {e}"),
            }
        }
    } else {
        let moe_weights =
            crate::layer_graph::pipeline_layer::build_moe_weights(weights, arch, layer);
        if let Some(ref moe) = moe_weights {
            for pos in 0..seq_len {
                let row: Vec<f32> = h_post_attn.row(pos).to_vec();
                let moe_out =
                    larql_compute::cpu::ops::moe::cpu_moe_forward(&row, moe, norm_offset, eps);
                for (dst, src) in h2.row_mut(pos).iter_mut().zip(moe_out.iter()) {
                    *dst = *src;
                }
            }
        } else {
            let mut out = h_post_ffn_dense;
            let mut h_ple =
                crate::forward::ple::apply_per_layer_embedding(weights, &out, layer, ple_input);
            crate::forward::layer::apply_layer_scalar(weights, &mut h_ple, layer);
            out = h_ple;
            return Some((out, kv_out));
        }
    }

    let combined = &h1 + &h2;

    let l0_stage_dump = if layer == 0 {
        std::env::var("LARQL_CPU_STAGE_DUMP").ok()
    } else {
        None
    };
    let dump_l0_arr = |name: &str, arr: &Array2<f32>| {
        if let Some(ref dir) = l0_stage_dump {
            let slice = arr.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/cpu_L0_{name}.f32"), &bytes);
        }
    };
    dump_l0_arr("h1_dense_norm1", &h1);
    dump_l0_arr("h2_moe_norm2", &h2);
    dump_l0_arr("combined_h1_plus_h2", &combined);

    let outer_w_vec: Option<&Vec<f32>> = if arch.moe_has_combined_output_norm() {
        arch.moe_post_outer_norm_key(layer)
            .or_else(|| arch.post_feedforward_layernorm_key(layer))
            .and_then(|k| weights.vectors.get(&k))
    } else {
        None
    };

    let seq = combined.nrows();
    let mut out_buf = Array2::<f32>::zeros((seq, hidden));
    for pos in 0..seq {
        let h_post_attn_row = h_post_attn.row(pos);
        let combined_row = combined.row(pos);
        let combined_normed = larql_compute::cpu::ops::outer_combine::outer_post_norm_residual(
            h_post_attn_row.as_slice().expect("contiguous row"),
            combined_row.as_slice().expect("contiguous row"),
            outer_w_vec.map(|v| v.as_slice()),
            norm_offset,
            eps,
        );
        for (dst, src) in out_buf.row_mut(pos).iter_mut().zip(combined_normed.iter()) {
            *dst = *src;
        }
    }
    dump_l0_arr("h_out_pre_layer_scalar", &out_buf);

    let mut h_out =
        crate::forward::ple::apply_per_layer_embedding(weights, &out_buf, layer, ple_input);
    if let Some(scalar_key) = arch.layer_scalar_key(layer) {
        if let Some(scalars) = weights.vectors.get(&scalar_key) {
            if let Some(&scalar) = scalars.first() {
                let flat = h_out.as_slice_mut().expect("contiguous out_buf");
                larql_compute::cpu::ops::outer_combine::apply_layer_scalar_in_place(flat, scalar);
            }
        }
    }

    Some((h_out, kv_out))
}
