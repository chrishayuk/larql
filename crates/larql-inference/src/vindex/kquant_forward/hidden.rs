use std::collections::HashMap;

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::attention::SharedKV;
use crate::forward::embed_tokens_pub;
use crate::forward::ple::precompute_per_layer_inputs;
use crate::forward::run_layer_with_ffn;

use super::tensors::{insert_q4k_layer_tensors, remove_layer_tensors};

/// Compute the final hidden state for `token_ids` against a Q4_K/Q6_K
/// vindex, dequantising attn + FFN one layer at a time. Returns the
/// `[seq_len, hidden]` array; caller owns the lm_head step.
pub fn predict_kquant_hidden(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    moe_remote: Option<&crate::ffn::RemoteMoeBackend>,
) -> Array2<f32> {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens_pub(weights, token_ids);

    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();
    let dump_cfg = crate::forward::dump_config::DumpConfig::get();
    let dump_dir = dump_cfg.layer_dir();
    if let Some(dir) = dump_dir {
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

        if let Some(dir) = dump_dir {
            let slice = h.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let path = crate::forward::dump_config::cpu_layer_path(dir, layer);
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

    if let Some(dir) = crate::forward::dump_config::DumpConfig::get().layer_dir() {
        let slice = h_post_attn.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = crate::forward::dump_config::cpu_layer_h_post_attn_path(dir, layer);
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

    let l0_dump_cfg = crate::forward::dump_config::DumpConfig::get();
    let l0_stage_dump = l0_dump_cfg.stage_dir(layer);
    let dump_l0_arr = |name: &str, arr: &Array2<f32>| {
        if let Some(dir) = l0_stage_dump {
            let slice = arr.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(
                crate::forward::dump_config::cpu_stage_path(dir, name),
                &bytes,
            );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    /// `predict_kquant_hidden` on the Gemma 3-style Q4K fixture — drives
    /// the non-MoE branch (every layer's `run_layer_with_ffn` call) for
    /// the full prompt. The MoE branch is unreachable without a Gemma 4
    /// hybrid-MoE arch fixture; this test covers the rest.
    #[test]
    fn predict_kquant_hidden_returns_shape_and_finite() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let h = predict_kquant_hidden(&mut weights, &[0u32, 1, 2], &index, None);
        assert_eq!(h.shape(), &[3, weights.hidden_size]);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "Q4K hidden state must be finite"
        );
    }

    #[test]
    fn predict_kquant_hidden_single_token() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let h = predict_kquant_hidden(&mut weights, &[5u32], &index, None);
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// `build_moe_router_weights` returns `None` for non-MoE archs
    /// (every standard arch's `moe_router_key` returns None). Drives
    /// the early-return.
    #[test]
    fn build_moe_router_weights_none_for_non_moe_arch() {
        let weights = make_test_q4k_weights();
        let arch = &*weights.arch;
        assert!(build_moe_router_weights(&weights, arch, 0).is_none());
    }

    /// Gemma 4 MoE arch + populated router weights → builder returns Some.
    #[test]
    fn build_moe_router_weights_some_for_gemma4_moe_fixture() {
        use crate::test_utils::{
            make_test_gemma4_moe_weights, GEMMA4_MOE_NUM_EXPERTS, GEMMA4_MOE_TOP_K,
        };
        let weights = make_test_gemma4_moe_weights();
        let arch = &*weights.arch;
        let router = build_moe_router_weights(&weights, arch, 0)
            .expect("Gemma 4 MoE fixture must produce router weights");
        assert_eq!(router.num_experts, GEMMA4_MOE_NUM_EXPERTS);
        assert_eq!(router.top_k, GEMMA4_MOE_TOP_K);
        assert!(!router.router_proj.is_empty());
    }

    /// `predict_kquant_hidden` on the Gemma 4 MoE fixture — drives the
    /// `is_hybrid_moe()` branch via `run_moe_layer_cpu`. Synthetic
    /// weights produce garbage values but the body executes
    /// end-to-end. Requires a Q4K vindex (for `insert_q4k_layer_tensors`)
    /// in addition to the MoE weights.
    #[test]
    fn predict_kquant_hidden_routes_through_moe_branch_on_gemma4_fixture() {
        use crate::test_utils::{make_test_gemma4_moe_weights, make_test_q4k_vindex};
        let mut weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let h = predict_kquant_hidden(&mut weights, &[0u32, 1], &index, None);
        assert_eq!(h.shape(), &[2, weights.hidden_size]);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "Gemma 4 MoE hidden state must be finite"
        );
    }

    /// `predict_kquant_hidden` with a `RemoteMoeBackend` provided drives
    /// the `Some(remote)` branch inside `run_moe_layer_cpu` (lines 156-162).
    /// The backend is disconnected (no shards), so `forward_moe_seq`
    /// returns `Err`, exercising the eprintln-fallback path at line 160
    /// while leaving `h2` at zero — the test asserts shape + finiteness,
    /// not numerical correctness.
    #[test]
    fn predict_kquant_hidden_with_disconnected_remote_moe_backend_falls_back() {
        use crate::ffn::RemoteMoeBackend;
        use crate::test_utils::{make_test_gemma4_moe_weights, make_test_q4k_vindex};
        let mut weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let remote = RemoteMoeBackend::new_disconnected();
        let h = predict_kquant_hidden(&mut weights, &[0u32, 1], &index, Some(&remote));
        assert_eq!(h.shape(), &[2, weights.hidden_size]);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "Gemma 4 MoE hidden state must be finite under remote fallback"
        );
    }

    /// `predict_kquant_hidden` with both `LARQL_CPU_DUMP_LAYERS` and
    /// `LARQL_CPU_STAGE_DUMP` set drives the dump branches inside the
    /// main loop (lines 30-33, 78-84) and inside `run_moe_layer_cpu`
    /// (lines 143-147, 190-194). Serialized via a local mutex because
    /// `DumpConfig::get()` reads process-global env vars on every call.
    #[test]
    fn predict_kquant_hidden_writes_dumps_when_env_vars_set() {
        use std::sync::{Mutex, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _g = LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let layer_dir = tempfile::tempdir().expect("layer dump tempdir");
        let stage_dir = tempfile::tempdir().expect("stage dump tempdir");
        let prev_l = std::env::var("LARQL_CPU_DUMP_LAYERS").ok();
        let prev_s = std::env::var("LARQL_CPU_STAGE_DUMP").ok();
        // SAFETY: held lock serialises env reads/writes for this test.
        unsafe {
            std::env::set_var("LARQL_CPU_DUMP_LAYERS", layer_dir.path());
            std::env::set_var("LARQL_CPU_STAGE_DUMP", stage_dir.path());
        }

        use crate::test_utils::{make_test_gemma4_moe_weights, make_test_q4k_vindex};
        let mut weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let h = predict_kquant_hidden(&mut weights, &[0u32, 1], &index, None);
        assert_eq!(h.shape(), &[2, weights.hidden_size]);

        // Restore env (or remove if not previously set).
        unsafe {
            match prev_l {
                Some(v) => std::env::set_var("LARQL_CPU_DUMP_LAYERS", v),
                None => std::env::remove_var("LARQL_CPU_DUMP_LAYERS"),
            }
            match prev_s {
                Some(v) => std::env::set_var("LARQL_CPU_STAGE_DUMP", v),
                None => std::env::remove_var("LARQL_CPU_STAGE_DUMP"),
            }
        }

        // Embed dump must exist (written at line 33 unconditionally when
        // layer_dir is Some). Per-layer dumps land under cpu_layer_NN.f32.
        assert!(
            layer_dir.path().join("cpu_h_embed.f32").is_file(),
            "embed dump must exist when LARQL_CPU_DUMP_LAYERS set"
        );
    }
}
