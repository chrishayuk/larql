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
/// Analysis-mode forward: a refusing MoE route is recorded, not fatal.
///
/// Kept at this signature because the overwhelming majority of its callers
/// pass `None`, where no refusal is reachable at all. Any caller that *does*
/// pass a backend and produces a continuation, a score, a parity number or a
/// benchmark must use [`predict_kquant_hidden_checked`] with
/// [`crate::ffn::MoeFailurePolicy::Fatal`] instead — under `RecordRefusal` a
/// failed layer contributes zeros and the result is analysis-only.
pub fn predict_kquant_hidden(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    moe: Option<&dyn crate::ffn::MoeExpertBackend>,
) -> Array2<f32> {
    let route = moe.map(crate::ffn::MoeRoute::recording);
    predict_kquant_hidden_checked(weights, token_ids, index, route)
        // `RecordRefusal` never propagates an error out of the block, so this
        // is unreachable rather than swallowed. Stated as an expectation so a
        // future policy change cannot turn it back into a silent zero.
        .expect("RecordRefusal never returns Err")
}

/// Forward pass whose MoE refusals are governed by the caller's policy.
///
/// This is the form every operation that produces a result about the model
/// should use: it returns the refusal rather than folding it into the tensor.
pub fn predict_kquant_hidden_checked(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    moe: Option<crate::ffn::MoeRoute<'_>>,
) -> Result<Array2<f32>, crate::ffn::MoeBackendError> {
    let num_layers = weights.num_layers;
    let mut scratch = larql_models::DequantScratch::new();
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
        let inserted = insert_q4k_layer_tensors(&mut scratch, weights, index, layer)
            .unwrap_or_else(|err| panic!("{err}"));

        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        // Pure MoE (GraniteMoE, OLMoE) as well as hybrid (Gemma 4). Gating on
        // `is_hybrid_moe()` alone sent pure-MoE models down the dense branch,
        // which then panicked on FFN slices the checkpoint never had.
        let is_moe_layer = weights.arch.is_moe() || weights.arch.is_hybrid_moe();
        let view = larql_models::WeightsView::with_scratch(weights, &scratch);
        let ffn_backend = crate::ffn::ViewFfn { view };
        if is_moe_layer {
            if let Some((h_new, kv_out)) = run_moe_layer_cpu(
                view,
                &h,
                layer,
                &ffn_backend,
                ple_inputs.get(layer),
                shared_kv,
                moe,
            )? {
                h = h_new;
                if let Some(kv) = kv_out {
                    kv_cache.insert(layer, kv);
                }
            }
        } else if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            view,
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

        remove_layer_tensors(&mut scratch, inserted);

        if let Some(dir) = dump_dir {
            let slice = h.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let path = crate::forward::dump_config::cpu_layer_path(dir, layer);
            if let Err(e) = std::fs::write(&path, &bytes) {
                eprintln!("[dump] failed to write {path}: {e}");
            }
        }
    }

    Ok(h)
}

/// Build `MoeRouterWeights` for a single layer from the model's vector store.
///
/// `pub` since the DEC routed-experts capture extension (ROADMAP hardening
/// item 17, first bullet): the capture path
/// (`layer_graph::grid::remote_ffn`) constructs router weights from
/// `ModelWeights` exactly the way the server-facing MoE dispatch does, so
/// captured routing bit-matches production routing. Returns `None` when the
/// arch has no MoE router for `layer`.
pub fn build_moe_router_weights<'a>(
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

/// What one MoE layer produced: the new residual, plus any K/V worth caching.
///
/// The outer `Option` is an **absence** — attention had nothing to run for this
/// layer — while the `Result` around it is a **failure**, a refused expert
/// route. Keeping them distinct is the point: collapsing "nothing to do" into
/// "something went wrong" is how a missing operand becomes a zeroed layer.
type MoeLayerOutcome = Option<(Array2<f32>, Option<SharedKV>)>;

/// CPU forward for one hybrid-MoE layer (Gemma 4 26B A4B).
fn run_moe_layer_cpu(
    weights: larql_models::WeightsView,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn crate::ffn::FfnBackend,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
    moe: Option<crate::ffn::MoeRoute<'_>>,
) -> Result<MoeLayerOutcome, crate::ffn::MoeBackendError> {
    // Attention returning `None` means this layer has nothing to run — an
    // absence, not a failure, so it stays `Ok(None)` and is distinct from a
    // refused expert route.
    let (h_post_attn, kv_out) = if let Some(shared) = shared_kv {
        let Some((h_pa, _, _)) =
            crate::attention::run_attention_block_shared(weights, h, layer, false, Some(shared))
        else {
            return Ok(None);
        };
        (h_pa, None)
    } else {
        let Some((h_pa, _, _, k_rope, v_final)) =
            crate::attention::run_attention_block_with_kv_out(weights, h, layer, false, None)
        else {
            return Ok(None);
        };
        (h_pa, Some((k_rope, v_final)))
    };

    let h_out = moe_ffn_block_cpu(
        weights.canonical(),
        &h_post_attn,
        layer,
        ffn,
        ple_input,
        moe,
    )?;
    Ok(Some((h_out, kv_out)))
}

/// CPU MoE FFN block for one hybrid-MoE layer, given the **post-attention**
/// hidden state. Computes the dense FFN contribution (`h1`), the expert
/// contribution (`h2` — via `moe` when a route is bound, else local
/// `cpu_moe_forward`), combines + outer-norms, and applies PLE +
/// layer-scalar. Returns the full layer output (the new residual).
///
/// Factored out of [`run_moe_layer_cpu`] so the KvEngine layer can drive it
/// via `RemoteMoeFfn::forward_moe_full_layer` (CPU remote-MoE with a real KV
/// cache); the attention half stays with the engine. The body is an exact
/// move from `run_moe_layer_cpu` — keep it byte-equivalent for parity.
///
/// `ple_input` is `None` on the engine path (callers must guard out
/// PLE-using architectures — see the larql-kv "MoE-aware KV engines"
/// roadmap item); the full-recompute path passes the precomputed per-layer
/// input so PLE models stay correct there.
pub fn moe_ffn_block_cpu(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    ffn: &dyn crate::ffn::FfnBackend,
    ple_input: Option<&Array2<f32>>,
    moe: Option<crate::ffn::MoeRoute<'_>>,
) -> Result<Array2<f32>, crate::ffn::MoeBackendError> {
    moe_ffn_block_cpu_with_index(weights, h_post_attn, layer, ffn, ple_input, moe, None)
}

/// `LARQL_Q4K_DIRECT_FFN=1` routes the hybrid-MoE *dense slab* through the
/// direct Q4_K/Q6_K matvec (`ffn_decode_step_native`) instead of the
/// f32-resident `run_ffn` — on the 26B-A4B this drops the slab's per-token
/// traffic ~7× (2.14 GB f32 → ~0.3 GB quantised). Decode-only (single-row):
/// prefill stays on the f32 BLAS gemm, where repeated quantised matvec
/// loses (the task-#16 prefill falsification). **Default on**
/// (`LARQL_Q4K_DIRECT_FFN=0` opts out); see [`larql_compute::options`].
fn q4k_direct_ffn_enabled() -> bool {
    larql_compute::options::q4k_direct_ffn_enabled()
}

/// Index-aware variant of [`moe_ffn_block_cpu`]: when `index` is provided
/// (the resident engine path threads it) and `LARQL_Q4K_DIRECT_FFN=1`, the
/// dense `h1` contribution reads quantised gate/up/down bytes directly;
/// otherwise byte-identical to [`moe_ffn_block_cpu`].
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn_block_cpu_with_index(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    ffn: &dyn crate::ffn::FfnBackend,
    ple_input: Option<&Array2<f32>>,
    moe: Option<crate::ffn::MoeRoute<'_>>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Result<Array2<f32>, crate::ffn::MoeBackendError> {
    // This block owns the layer coordinate, and the served CPU route boundary
    // below it does not. Scope it so routing observations are attributable;
    // without this they are refused, not guessed.
    let _route_scope = larql_compute::moe_route_observe::LayerScope::new(layer);
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let hidden = h_post_attn.ncols();

    if let Some(dir) = crate::forward::dump_config::DumpConfig::get().layer_dir() {
        let slice = h_post_attn.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = crate::forward::dump_config::cpu_layer_h_post_attn_path(dir, layer);
        let _ = std::fs::write(&path, &bytes);
    }

    // Hybrid MoE (Gemma 4 26B A4B) runs a dense slab *alongside* the experts
    // and sums both deltas. Pure MoE (GraniteMoE, OLMoE, Mixtral) has no
    // dense slab at all — its FFN *is* the expert block — so the dense leg is
    // skipped and `h1` is identically zero. Calling `run_ffn` there would
    // look up gate/up/down tensors that do not exist in the checkpoint.
    let is_hybrid = arch.is_hybrid_moe();

    let _t_dense = std::time::Instant::now();
    // Dense slab: quantised-direct on the decode step when enabled, with a
    // per-layer fallback to the f32 path (`ffn_decode_step_native` returns
    // `None` on unsupported formats/shapes).
    let h_post_ffn_dense = if is_hybrid {
        index
            .filter(|_| q4k_direct_ffn_enabled() && h_post_attn.nrows() == 1)
            .and_then(|idx| {
                super::cached::ffn_decode_step_native(
                    weights,
                    idx,
                    &larql_compute::CpuBackend,
                    h_post_attn,
                    layer,
                )
            })
            .unwrap_or_else(|| crate::forward::run_ffn(weights, h_post_attn, layer, ffn, false).0)
    } else {
        h_post_attn.clone()
    };
    crate::decode_stages::record_dense(_t_dense.elapsed().as_nanos());
    // Zero for pure MoE by construction (`h_post_ffn_dense == h_post_attn`).
    let h1 = &h_post_ffn_dense - h_post_attn;

    let seq_len = h_post_attn.nrows();
    let mut h2 = Array2::<f32>::zeros((seq_len, hidden));

    if let Some(route) = moe {
        // Every non-default route goes through one call. Which route it is —
        // remote shards, a VINDEX3 bound plan — is the backend's business, and
        // the block loop's only job is to place the contribution.
        let _t_expert = std::time::Instant::now();
        let out = route
            .backend
            .forward_moe_seq(weights, layer, h_post_attn, norm_offset, eps);
        crate::decode_stages::record_expert(_t_expert.elapsed().as_nanos());
        match out {
            Ok(out) => h2 = out,
            // The failure is the caller's to interpret. This branch used to
            // print and continue unconditionally, which left `h2` at zeros —
            // so a broken layer produced a fluent continuation from a model
            // with that layer's entire expert contribution removed, and the
            // only trace was a line on stderr.
            Err(e) => match route.policy {
                crate::ffn::MoeFailurePolicy::Fatal => return Err(e),
                crate::ffn::MoeFailurePolicy::RecordRefusal => {
                    // Analysis-only: `h2` stays zero and the caller has
                    // declared it will report the incompleteness.
                    eprintln!(
                        "[moe_ffn_block_cpu] {} refused L{layer} (analysis mode, \
                         expert contribution omitted): {e}",
                        route.backend.name()
                    );
                }
            },
        }
    } else {
        // Local experts count toward the expert stage too (`LARQL_DECODE_STAGES`)
        // — previously only the remote branch recorded, so in-process MoE
        // decode showed 0 expert time and the split was unusable.
        let _t_expert = std::time::Instant::now();
        let moe_weights =
            crate::layer_graph::pipeline_layer::build_moe_weights(weights, arch, layer);
        if let Some(ref moe) = moe_weights {
            // Within-expert routing probe: tag the layer for the expert calls
            // below. No-op (one relaxed atomic store) unless a schedule is
            // installed via `larql_compute::cpu::ops::moe::set_routing`; layers
            // run sequentially so one store covers the per-position loop.
            larql_compute::cpu::ops::moe::set_current_layer(layer);
            // BW-C whole-expert-skip oracle: same tag, same reason —
            // `add_expert` runs on rayon worker threads, so this needs
            // its own cross-thread atomic rather than the thread-local
            // `moe_route_observe::LayerScope` above. No-op unless armed
            // or observing.
            larql_compute::cpu::ops::moe::expert_override::set_current_layer(layer);
            for pos in 0..seq_len {
                let row: Vec<f32> = h_post_attn.row(pos).to_vec();
                // BW-C1 contribution/residual-norm covariate: the
                // incoming residual's L2 norm, set once per POSITION
                // (unlike the per-layer `set_current_layer` calls above
                // — this varies every position within a layer's loop).
                // No-op cost when not observing (one extra sqrt(sum(v*v))
                // per position); read back inside `expert_override::
                // observe` via `current_residual_norm`, not threaded as
                // a parameter.
                larql_compute::cpu::ops::moe::expert_override::set_current_residual_norm(
                    (row.iter().map(|v| v * v).sum::<f32>()).sqrt(),
                );
                let moe_out =
                    larql_compute::cpu::ops::moe::cpu_moe_forward(&row, moe, norm_offset, eps);
                for (dst, src) in h2.row_mut(pos).iter_mut().zip(moe_out.iter()) {
                    *dst = *src;
                }
            }
            crate::decode_stages::record_expert(_t_expert.elapsed().as_nanos());
        } else if is_hybrid {
            let out = h_post_ffn_dense;
            let mut h_ple =
                crate::forward::ple::apply_per_layer_embedding(weights, &out, layer, ple_input);
            crate::forward::layer::apply_layer_scalar(weights, &mut h_ple, layer);
            return Ok(h_ple);
        } else {
            // Pure MoE with no expert weights would otherwise fall through to
            // `h_post_ffn_dense`, which here *is* `h_post_attn` — an identity
            // FFN silently returning plausible-looking activations for a layer
            // that computed nothing. Fail loudly instead.
            panic!(
                "pure-MoE layer {layer} has no expert weights: the vindex is \
                 missing its per-layer expert store (layers/layer_{layer:02}.weights)"
            );
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

    Ok(h_out)
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
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let h = predict_kquant_hidden(&weights, &[0u32, 1, 2], &index, None);
        assert_eq!(h.shape(), &[3, weights.hidden_size]);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "Q4K hidden state must be finite"
        );
    }

    #[test]
    fn predict_kquant_hidden_single_token() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let h = predict_kquant_hidden(&weights, &[5u32], &index, None);
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
        let weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let h = predict_kquant_hidden(&weights, &[0u32, 1], &index, None);
        assert_eq!(h.shape(), &[2, weights.hidden_size]);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "Gemma 4 MoE hidden state must be finite"
        );
    }

    /// MoE arch but no per-expert weights → `build_moe_weights` returns `None`
    /// and `moe_ffn_block_cpu` takes the **dense-FFN fallback** branch
    /// (`else { … return h_ple }`). Dropping `raw_bytes` removes the packed
    /// expert blobs the BF16 path reads; attention + dense FFN come from
    /// `tensors`/`vectors`/the q4k index, so the forward still completes.
    #[test]
    fn predict_kquant_hidden_moe_dense_fallback_when_no_expert_weights() {
        use crate::test_utils::{make_test_gemma4_moe_weights, make_test_q4k_vindex};
        let mut weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        weights.raw_bytes.clear(); // drop per-expert blobs → build_moe_weights None
        let h = predict_kquant_hidden(&weights, &[0u32, 1], &index, None);
        assert_eq!(h.shape(), &[2, weights.hidden_size]);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "dense-FFN fallback hidden state must be finite"
        );
    }

    /// Within-expert routing (`larql_compute::…::moe::set_routing`) installed
    /// before the forward must flow through the in-process MoE path
    /// (`set_current_layer` → `prune_act` inside the expert kernel) and change
    /// the output vs the dense (no-routing) baseline. Regression guard for the
    /// `walk_ffn_v1_moe_within_expert` probe wiring. The `RoutingReset` drop
    /// guard restores global state even on panic so it can't leak to other
    /// tests (their MoE assertions are finiteness-only and tolerate pruning).
    #[test]
    fn predict_kquant_hidden_within_expert_routing_changes_output() {
        use crate::test_utils::{make_test_gemma4_moe_weights, make_test_q4k_vindex};
        use larql_compute::cpu::ops::moe::{
            set_current_layer, set_routing, ExpertFeatureSelector, WithinExpertRouting,
        };

        struct RoutingReset;
        impl Drop for RoutingReset {
            fn drop(&mut self) {
                set_routing(None);
                set_current_layer(0);
            }
        }

        let weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let nl = weights.num_layers;

        let _reset = RoutingReset;
        set_routing(None);
        let dense = predict_kquant_hidden(&weights, &[0u32, 1], &index, None);

        // Aggressively prune every expert layer's feature set.
        set_routing(Some(WithinExpertRouting {
            frac_per_layer: vec![Some(0.125); nl],
            selector: ExpertFeatureSelector::ActMagnitude,
        }));
        let pruned = predict_kquant_hidden(&weights, &[0u32, 1], &index, None);
        drop(_reset); // restore before asserting

        assert_eq!(pruned.shape(), dense.shape());
        assert!(pruned.iter().all(|v| v.is_finite()));
        let max_abs_diff = dense
            .iter()
            .zip(pruned.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff > 1e-6,
            "within-expert pruning must change the forward output (got max |Δ|={max_abs_diff})"
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
        let weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let remote = RemoteMoeBackend::new_disconnected();
        let h = predict_kquant_hidden(&weights, &[0u32, 1], &index, Some(&remote));
        assert_eq!(h.shape(), &[2, weights.hidden_size]);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "Gemma 4 MoE hidden state must be finite under remote fallback"
        );
    }

    /// `predict_kquant_hidden` with both `LARQL_CPU_DUMP_LAYERS` and
    /// `LARQL_CPU_STAGE_DUMP` set drives the dump branches inside the
    /// main loop (lines 30-33, 78-84) and inside `run_moe_layer_cpu`
    /// (lines 143-147, 190-194). The flags are toggled via the thread-local
    /// override (NOT `std::env::set_var`, which races concurrent `getenv` on
    /// the decode path → SIGSEGV); `DumpConfig::from_env` reads them through
    /// the override-aware `options::env_value` helper, so the override reaches
    /// the producer in this same thread. No serialising mutex needed — the
    /// override is per-thread, so it can't leak into a parallel test.
    #[test]
    fn predict_kquant_hidden_writes_dumps_when_env_vars_set() {
        use larql_compute::forward::dump_config::{ENV_CPU_DUMP_LAYERS, ENV_CPU_STAGE_DUMP};

        /// Clears the thread-local overrides on drop so a panicking assert
        /// can't leak them into a later test on the same worker thread.
        struct DumpEnvGuard;
        impl Drop for DumpEnvGuard {
            fn drop(&mut self) {
                larql_compute::options::clear_fast_path_overrides();
            }
        }

        let layer_dir = tempfile::tempdir().expect("layer dump tempdir");
        let stage_dir = tempfile::tempdir().expect("stage dump tempdir");
        let _guard = DumpEnvGuard;
        larql_compute::options::set_env_override(
            ENV_CPU_DUMP_LAYERS,
            Some(layer_dir.path().to_str().expect("utf-8 tempdir path")),
        );
        larql_compute::options::set_env_override(
            ENV_CPU_STAGE_DUMP,
            Some(stage_dir.path().to_str().expect("utf-8 tempdir path")),
        );

        use crate::test_utils::{make_test_gemma4_moe_weights, make_test_q4k_vindex};
        let weights = make_test_gemma4_moe_weights();
        let index = make_test_q4k_vindex(&weights);
        let h = predict_kquant_hidden(&weights, &[0u32, 1], &index, None);
        assert_eq!(h.shape(), &[2, weights.hidden_size]);

        // Embed dump must exist (written at line 33 unconditionally when
        // layer_dir is Some). Per-layer dumps land under cpu_layer_NN.f32.
        assert!(
            layer_dir.path().join("cpu_h_embed.f32").is_file(),
            "embed dump must exist when LARQL_CPU_DUMP_LAYERS set"
        );
    }
}
