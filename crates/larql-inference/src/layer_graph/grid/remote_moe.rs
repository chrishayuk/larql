use super::config::GridRuntimeConfig;
use super::setup::{build_grid_pipeline_setup, reset_and_preallocate_grid_kv, RemotePatch};
use super::timing::{moe_call_timed, print_run_summary, print_token_breakdown, LayerTiming};
use super::GridGenerateResult;
use crate::ffn::moe_remote::{InflightMoe, MoeRouterWeights, RemoteMoeError};
use crate::ffn::RemoteMoeBackend;
use crate::forward::{apply_norm, embed_tokens_pub};
use crate::layer_graph::generate::detok::Detokenizer;
use crate::layer_graph::generate::eos::EosConfig;
use crate::layer_graph::generate::policy::{
    build_special_suppress_set_with_policy, pick_next_filtered_with_policy,
};
use larql_compute::prelude::*;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

/// Build `MoeRouterWeights` for one layer from the model's vector store.
/// Returns None if the required router projection is absent.
///
/// `LARQL_MOE_TOP_K=<N>` overrides the architecture-default top_k at runtime
/// (clamped to `[1, arch_top_k]`).  Cheap accuracy/speed knob — Gemma 4 ships
/// with top_k=8; testing top_k=4 cuts active experts in half for a roughly
/// 2× server-compute speedup at the cost of some routing fidelity.
fn build_router<'a>(
    weights: &'a ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
    config: &GridRuntimeConfig,
) -> Option<MoeRouterWeights<'a>> {
    let router_proj_key = arch.moe_router_key(layer)?;
    let router_proj = weights.vectors.get(&router_proj_key)?.as_slice();
    let sl = |k: Option<String>| -> &'a [f32] {
        k.and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    };
    let arch_top_k = arch.num_experts_per_token();
    let top_k = config.moe_top_k(arch_top_k);
    Some(MoeRouterWeights {
        router_proj,
        router_scale: sl(arch.moe_router_scale_key(layer)),
        router_per_expert_scale: sl(arch.moe_router_per_expert_scale_key(layer)),
        router_norm: sl(arch.moe_router_norm_key(layer)),
        router_norm_parameter_free: arch.moe_router_norm_parameter_free(),
        router_input_scalar: arch.moe_router_input_scalar().unwrap_or(1.0),
        pre_experts_norm: sl(arch.moe_pre_experts_norm_key(layer)),
        post_experts_norm: sl(arch.moe_post_experts_norm_key(layer)),
        num_experts: arch.num_experts(),
        top_k,
    })
}

/// Greedy autoregressive generation through a remote-expert grid.
///
/// Requires a Metal (or Q4-capable) backend — attention and dense FFN run on
/// the GPU exactly as in the normal `generate()` path.  Expert blocks are
/// dispatched to `remote` instead of running locally.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_remote_moe(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    index: &VectorIndex,
    remote: &RemoteMoeBackend,
    backend: &dyn ComputeBackend,
    eos: &EosConfig,
) -> Result<GridGenerateResult, RemoteMoeError> {
    let runtime = GridRuntimeConfig::from_env();
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let setup = build_grid_pipeline_setup(weights, index, RemotePatch::Moe)?;
    let layers = setup.layers;
    let hidden = setup.hidden;
    let intermediate = setup.intermediate;

    // ── Open gRPC streams (one pair for the entire generation) ───────────────
    //
    // For gRPC shards (`grpc://` URLs), we open one bidirectional stream per
    // shard and reuse it for every layer of every token (prefill + decode).
    // This eliminates the per-layer connection setup cost: each layer pays only
    // the cost of one proto frame exchange on an existing HTTP/2 connection
    // (~0.5ms) instead of ~12ms for a new unary call.
    //
    // For HTTP shards, `open_streams` returns an empty vec and we fall back to
    // `forward_moe` (per-layer HTTP calls, as before).
    let mut streams: Vec<crate::ffn::moe_remote::ShardStream> = if remote.has_grpc_shards() {
        remote.open_streams().unwrap_or_default()
    } else {
        vec![]
    };

    // ── Prefill ───────────────────────────────────────────────────────────────
    //
    // Run one `decode_token_with_moe` per prompt token rather than `prefill_kquant`.
    // `prefill_kquant` does not correctly apply MoE experts for hybrid-MoE post-norm
    // models (Gemma 4 26B-A4B), so the first-token prediction and subsequent KV
    // cache entries are wrong.  Sequential decode builds the KV cache correctly
    // — each token processes with the proper remote expert contribution.
    reset_and_preallocate_grid_kv(weights, backend);

    let skip_moe = runtime.skip_moe;
    let timing_enabled = runtime.timing_enabled;
    let bytes_enabled = crate::ffn::moe_remote::metrics::enabled();
    let run_bytes_before = bytes_enabled.then(crate::ffn::moe_remote::metrics::snapshot);
    let mut transport_tokens = 0usize;
    let mut per_token_timings: Vec<Vec<LayerTiming>> = Vec::new();
    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    // Streaming detokeniser: handles SentencePiece `▁` leading-space prefix
    // and skips special tokens (`<mask>`, `<turn|>`, etc.) so the surface
    // string is the same as HF's `decode(..., skip_special_tokens=true)`.
    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(&prompt_ids);

    // Special-token suppression set: prevents Q4_K-noise-induced picks of
    // `<mask>`, `<|tool>`, `<|channel>`, etc. EOS tokens stay unmasked so
    // the EOS check can still fire when the model legitimately wants to halt.
    let suppress = build_special_suppress_set_with_policy(tokenizer, eos, &runtime.token_policy);

    for (prefill_idx, &tok_id) in prompt_ids.iter().enumerate() {
        let token_bytes_before = bytes_enabled.then(crate::ffn::moe_remote::metrics::snapshot);
        let tok_embed = embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        let mut step_error: Option<RemoteMoeError> = None;
        let mut tok_timings: Vec<LayerTiming> = Vec::new();
        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            if skip_moe {
                return vec![0.0f32; hidden];
            }
            if step_error.is_some() {
                return vec![0.0f32; hidden];
            }
            let router = match build_router(weights, arch, layer, &runtime) {
                Some(r) => r,
                None => return vec![0.0f32; hidden],
            };
            let timing_slot = if timing_enabled {
                Some(&mut tok_timings)
            } else {
                None
            };
            match moe_call_timed(
                remote,
                layer,
                h_post_attn,
                &router,
                &mut streams,
                norm_offset,
                eps,
                timing_slot,
            ) {
                Ok(out) => out,
                Err(e) => {
                    step_error = Some(e);
                    vec![0.0f32; hidden]
                }
            }
        };

        let h = backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut moe_fn);
        if let Some(err) = step_error {
            return Err(err);
        }
        last_hidden_vec = h.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode_token_with_moe returned None during prefill".into())
        })?;
        if timing_enabled {
            print_token_breakdown("prefill", prefill_idx, &tok_timings);
            per_token_timings.push(tok_timings);
        }
        if let Some(before) = token_bytes_before.as_ref() {
            crate::ffn::moe_remote::metrics::print_delta("prefill", prefill_idx, before);
        }
        transport_tokens += 1;
    }

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();

    // First token from the (correct) prefill output.
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
    let debug_ids = runtime.token_policy.debug_token_ids;
    if debug_ids {
        let raw = tokenizer.id_to_token(first_id).unwrap_or_default();
        eprintln!("[tok 0] id={first_id:6} raw={raw:?} delta={first_tok:?}");
    }
    tokens.push(first_tok);
    current_ids.push(first_id);
    if first_is_eos || tokens.len() >= max_tokens {
        if let Some(before) = run_bytes_before.as_ref() {
            crate::ffn::moe_remote::metrics::print_summary("generate", before, transport_tokens);
        }
        return Ok(GridGenerateResult {
            tokens,
            decode_ms: vec![0.0],
            ffn_rtt_ms: Vec::new(),
        });
    }

    for step in 0..max_tokens.saturating_sub(1) {
        let token_bytes_before = bytes_enabled.then(crate::ffn::moe_remote::metrics::snapshot);
        let t0 = std::time::Instant::now();
        let next_input_id = *current_ids.last().unwrap();

        // Embed next token.
        let tok_embed = embed_tokens_pub(weights, &[next_input_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        let mut step_error: Option<RemoteMoeError> = None;
        let mut tok_timings: Vec<LayerTiming> = Vec::new();

        // Two paths:
        //   - streams (gRPC) → split fire/collect so dense FFN overlaps with
        //     the remote MoE round trip.  Reliably ~10% faster on M3 Max
        //     loopback in steady state (re-measured 2026-05-01: 19.5 vs
        //     17.7 tok/s on Gemma 4 26B-A4B with one local gRPC shard,
        //     stable across alternating cooled runs).  The historical
        //     "20 → 4 tok/s catastrophic regression" warning predates the
        //     Metal MoE accuracy fix and the predispatch refactor; under
        //     thermal pressure both paths regress similarly, but
        //     stable-state SPLIT wins.  Set `LARQL_MOE_NO_SPLIT=1` to
        //     force the unary path (e.g., to debug a regression on a new
        //     hardware / driver combo).
        //   - otherwise → existing unary HTTP / synchronous moe_fn (used
        //     for HTTP shards which don't open gRPC streams, plus the
        //     opt-out path above).
        let result = if streams.is_empty() || runtime.split_disabled {
            let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
                if skip_moe {
                    return vec![0.0f32; hidden];
                }
                if step_error.is_some() {
                    return vec![0.0f32; hidden];
                }
                let router = match build_router(weights, arch, layer, &runtime) {
                    Some(r) => r,
                    None => return vec![0.0f32; hidden],
                };
                let timing_slot = if timing_enabled {
                    Some(&mut tok_timings)
                } else {
                    None
                };
                match moe_call_timed(
                    remote,
                    layer,
                    h_post_attn,
                    &router,
                    &mut streams,
                    norm_offset,
                    eps,
                    timing_slot,
                ) {
                    Ok(out) => out,
                    Err(e) => {
                        step_error = Some(e);
                        vec![0.0f32; hidden]
                    }
                }
            };
            backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut moe_fn)
        } else {
            // Split path: shared inflight handle + step_error via RefCell
            // because both closures capture them and can't both have unique
            // mut borrows.  Closures are still called strictly sequentially
            // by the metal decode loop so RefCell never panics in practice.
            use std::cell::RefCell;
            let inflight: RefCell<Option<(InflightMoe, std::time::Instant)>> = RefCell::new(None);
            let step_err_cell: RefCell<Option<RemoteMoeError>> = RefCell::new(None);
            let tok_timings_cell: RefCell<Vec<LayerTiming>> = RefCell::new(Vec::new());

            let mut fire_fn = |layer: usize, h_post_attn: &[f32]| {
                if skip_moe {
                    return;
                }
                if step_err_cell.borrow().is_some() {
                    return;
                }
                let router = match build_router(weights, arch, layer, &runtime) {
                    Some(r) => r,
                    None => return,
                };
                let t_start = std::time::Instant::now();
                match remote.forward_moe_stream_fire(
                    layer,
                    h_post_attn,
                    &router,
                    &streams,
                    norm_offset,
                    eps,
                ) {
                    Ok(inf) => {
                        *inflight.borrow_mut() = Some((inf, t_start));
                    }
                    Err(e) => {
                        *step_err_cell.borrow_mut() = Some(e);
                    }
                }
            };
            let mut collect_fn = |_layer: usize| -> Vec<f32> {
                if skip_moe {
                    return vec![0.0f32; hidden];
                }
                if step_err_cell.borrow().is_some() {
                    return vec![0.0f32; hidden];
                }
                let Some((inf, t_start)) = inflight.borrow_mut().take() else {
                    return vec![0.0f32; hidden];
                };
                match remote.forward_moe_stream_collect_with_timing(&streams, inf) {
                    Ok((h2, per_shard)) => {
                        if timing_enabled {
                            let total_ms = t_start.elapsed().as_secs_f32() * 1000.0;
                            tok_timings_cell.borrow_mut().push(LayerTiming {
                                total_ms,
                                route_fire_ms: 0.0,
                                collect_ms: total_ms,
                                per_shard,
                            });
                        }
                        h2
                    }
                    Err(e) => {
                        *step_err_cell.borrow_mut() = Some(e);
                        vec![0.0f32; hidden]
                    }
                }
            };
            let r = backend.decode_token_with_moe_split(
                &layers,
                &x_tok,
                hidden,
                intermediate,
                &mut fire_fn,
                &mut collect_fn,
            );
            // Propagate any error captured by the closures.
            step_error = step_err_cell.into_inner();
            tok_timings = tok_timings_cell.into_inner();
            r
        };

        if let Some(err) = step_error {
            return Err(err);
        }

        let h_vec = result.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode_token_with_moe returned None".into())
        })?;

        last_hidden_vec = h_vec;

        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let last_hidden = h_normed.row(0).to_owned();
        if runtime.token_policy.debug_token_ids {
            let raw_rms = (last_hidden_vec.iter().map(|v| v * v).sum::<f32>()
                / last_hidden_vec.len() as f32)
                .sqrt();
            let normed_rms =
                (last_hidden.iter().map(|v| v * v).sum::<f32>() / last_hidden.len() as f32).sqrt();
            let max_abs = last_hidden.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            eprintln!(
                "  [step {step}] h_pre_norm_rms={raw_rms:.5} h_normed_rms={normed_rms:.5} max_abs={max_abs:.5}"
            );
        }
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
        if timing_enabled {
            print_token_breakdown("decode", step, &tok_timings);
            let moe_total: f32 = tok_timings.iter().map(|t| t.total_ms).sum();
            let other = (token_wall_ms as f32 - moe_total).max(0.0);
            eprintln!(
                "[moe-timing] decode tok={step} wall={token_wall_ms:.1}ms \
                 moe={moe_total:.1}ms other(gpu+sample)={other:.1}ms"
            );
            per_token_timings.push(tok_timings);
        }
        if let Some(before) = token_bytes_before.as_ref() {
            crate::ffn::moe_remote::metrics::print_delta("decode", step, before);
        }
        transport_tokens += 1;
        let tok_str = detok.push(next_id);
        let is_eos = eos.is_eos_with_tokenizer(next_id, &tok_str, tokenizer);
        if debug_ids {
            let raw = tokenizer.id_to_token(next_id).unwrap_or_default();
            eprintln!(
                "[tok {}] id={next_id:6} raw={raw:?} delta={tok_str:?}",
                step + 1
            );
        }
        tokens.push(tok_str);
        current_ids.push(next_id);
        if is_eos {
            break;
        }
    }

    if timing_enabled {
        print_run_summary("generate", &per_token_timings);
    }
    if let Some(before) = run_bytes_before.as_ref() {
        crate::ffn::moe_remote::metrics::print_summary("generate", before, transport_tokens);
    }

    Ok(GridGenerateResult {
        tokens,
        decode_ms,
        ffn_rtt_ms: Vec::new(),
    })
}

/// Batch pre-dispatch variant of [`generate_with_remote_moe`].
///
/// Each decode step runs two Metal passes:
///   1. **SKIP_MOE pass**: Metal runs attention + dense FFN with zero expert
///      contributions, capturing `h_post_attn` at each of the 30 MoE layers.
///   2. **Batch dispatch**: ONE gRPC `ExpertBatch` call per shard (parallel),
///      carrying all 30 layers' expert inputs.  The server processes all 120
///      expert matmuls concurrently with `join_all(spawn_blocking)`.
///   3. **Apply pass**: Metal runs the same 30 layers, but `moe_fn` now returns
///      the pre-computed h2 instead of calling remote shards per-layer.
///
/// **Trade-off vs streaming**: streaming is exact (each layer's `h_post_attn`
/// includes all previous layers' expert contributions). Batch uses the
/// SKIP_MOE pass `h_post_attn` as an approximation — the error is small for
/// well-trained models and typically produces the same top-1 token.
///
/// **Speed**: streaming makes 30 sequential round-trips per token (each paying
/// ~3.5ms server compute + condvar overhead).  Batch makes ONE round-trip whose
/// server-side cost is max(N_experts / N_cores) × t_expert — much less than
/// 30 × t_expert when the server has enough parallel cores.
/// Two-pass (or more) predispatch decode.
///
/// `predispatch_iters` controls how many remote dispatch + Metal pass cycles
/// are run per token to refine the expert contributions:
///
/// - `1`: one dispatch, two Metal passes (fast, approximate — later layers miss
///   earlier layers' expert contributions in the routing input).
/// - `2`: two dispatches, three Metal passes (slower but much more accurate —
///   the second dispatch sees h_post_attn that already includes the first
///   round's expert outputs, so routing is much closer to ground truth).
///
/// Values above 2 have diminishing returns. 1 is the speed default; 2 is
/// the quality default.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_remote_moe_batch(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    index: &VectorIndex,
    remote: &RemoteMoeBackend,
    backend: &dyn ComputeBackend,
    eos: &EosConfig,
    predispatch_iters: usize,
) -> Result<GridGenerateResult, RemoteMoeError> {
    let runtime = GridRuntimeConfig::from_env();
    let predispatch_iters = predispatch_iters.max(1);
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let setup = build_grid_pipeline_setup(weights, index, RemotePatch::Moe)?;
    let layers = setup.layers;
    let hidden = setup.hidden;
    let intermediate = setup.intermediate;
    let num_layers = setup.num_layers;

    // Prefill: sequential decode_token_with_moe (same as streaming variant).
    reset_and_preallocate_grid_kv(weights, backend);

    let skip_moe = runtime.skip_moe;
    let bytes_enabled = crate::ffn::moe_remote::metrics::enabled();
    let run_bytes_before = bytes_enabled.then(crate::ffn::moe_remote::metrics::snapshot);
    let mut transport_tokens = 0usize;
    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    // Build routers once here so both prefill and decode loops can use them.
    let routers_all: Vec<MoeRouterWeights<'_>> = (0..num_layers)
        .filter_map(|l| build_router(weights, arch, l, &runtime))
        .collect();

    for (prefill_idx, &tok_id) in prompt_ids.iter().enumerate() {
        let token_bytes_before = bytes_enabled.then(crate::ffn::moe_remote::metrics::snapshot);
        let tok_embed = embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();
        let kv_len = backend.kv_cache_len();

        // Pass 0: skip MoE, capture h_post_attn.
        let mut h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        {
            let h_cap = &mut h_capture;
            let mut moe_pass0 = |layer: usize, h: &[f32]| -> Vec<f32> {
                if h_cap.len() == layer {
                    h_cap.push(h.to_vec());
                }
                vec![0.0f32; hidden]
            };
            backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut moe_pass0);
        }
        if !skip_moe {
            backend.truncate_kv_cache(kv_len);
        }

        // Refinement iterations.
        let mut h2_final: Option<Vec<f32>> = None;
        let iters = if skip_moe { 0 } else { predispatch_iters };
        for iter in 0..iters.max(1) {
            let is_final = iter + 1 == iters.max(1);
            let h2_per_layer = if skip_moe || h_capture.is_empty() {
                vec![vec![0.0f32; hidden]; num_layers]
            } else {
                remote
                    .forward_moe_predispatch(&h_capture, &routers_all, norm_offset, eps)
                    .unwrap_or_else(|_| vec![vec![0.0f32; hidden]; num_layers])
            };
            if !is_final {
                let mut new_cap: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
                let h2r = &h2_per_layer;
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
                let h2r = &h2_per_layer;
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
        transport_tokens += 1;
        if let Some(before) = token_bytes_before.as_ref() {
            crate::ffn::moe_remote::metrics::print_delta("prefill", prefill_idx, before);
        }
    }

    // First token from prefill.
    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();
    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(&prompt_ids);
    let suppress = build_special_suppress_set_with_policy(tokenizer, eos, &runtime.token_policy);
    let pfa = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let pfn = apply_norm(weights, &pfa, arch.final_norm_key(), norm_offset);
    let first_id = pick_next_filtered_with_policy(
        index,
        weights,
        &pfn.row(0).to_owned(),
        backend,
        &suppress,
        tokenizer,
        &runtime.token_policy,
    );
    let first_tok = detok.push(first_id);
    let first_eos = eos.is_eos_with_tokenizer(first_id, &first_tok, tokenizer);
    tokens.push(first_tok);
    current_ids.push(first_id);
    if first_eos || tokens.len() >= max_tokens {
        if let Some(before) = run_bytes_before.as_ref() {
            crate::ffn::moe_remote::metrics::print_summary("generate", before, transport_tokens);
        }
        return Ok(GridGenerateResult {
            tokens,
            decode_ms: vec![0.0],
            ffn_rtt_ms: Vec::new(),
        });
    }

    // ── Decode loop ──────────────────────────────────────────────────────────
    //
    // Each token runs (predispatch_iters + 1) Metal passes:
    //
    //   Pass 0  — skip MoE, capture h_post_attn for each MoE layer.
    //             KV is rolled back after this pass (not the final write).
    //
    //   Iter 0..N-1  — dispatch(h_capture) → h2, then apply pass:
    //                  • non-final: capture updated h_capture, roll back KV.
    //                  • final: write KV permanently, produce h_out.
    //
    // Rolling back KV after every non-final pass ensures the KV cache advances
    // by exactly one position per token regardless of iteration count.

    for step in 0..max_tokens.saturating_sub(1) {
        let token_bytes_before = bytes_enabled.then(crate::ffn::moe_remote::metrics::snapshot);
        let t0 = std::time::Instant::now();
        let next_id = *current_ids.last().unwrap();
        let tok_embed = embed_tokens_pub(weights, &[next_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();
        let kv_len = backend.kv_cache_len();

        // ── Pass 0: capture h_post_attn (MoE = zeros) ───────────────────────
        let mut h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        {
            let h_cap = &mut h_capture;
            let mut moe_pass0 = |layer: usize, h: &[f32]| -> Vec<f32> {
                if h_cap.len() == layer {
                    h_cap.push(h.to_vec());
                }
                vec![0.0f32; hidden]
            };
            backend.decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut moe_pass0);
        }
        if !skip_moe {
            // Roll back KV — only the final apply pass should advance it.
            backend.truncate_kv_cache(kv_len);
        }

        if skip_moe {
            // No expert computation; pass 0 was the only pass needed.
            // (KV already advanced correctly.)
            let h_out_skip = backend
                .decode_token_with_moe(
                    &layers,
                    &x_tok,
                    hidden,
                    intermediate,
                    &mut |_layer: usize, _h: &[f32]| vec![0.0f32; hidden],
                )
                .ok_or_else(|| RemoteMoeError::BadResponse("skip_moe pass returned None".into()))?;
            let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out_skip.clone())
                .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
            let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
            let next_tok_id = pick_next_filtered_with_policy(
                index,
                weights,
                &h_normed.row(0).to_owned(),
                backend,
                &suppress,
                tokenizer,
                &runtime.token_policy,
            );
            decode_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            let tok_str = detok.push(next_tok_id);
            let is_eos = eos.is_eos_with_tokenizer(next_tok_id, &tok_str, tokenizer);
            tokens.push(tok_str);
            current_ids.push(next_tok_id);
            if is_eos {
                break;
            }
            continue;
        }

        // ── Refinement iterations ────────────────────────────────────────────
        let mut h_out_opt: Option<Vec<f32>> = None;

        for iter in 0..predispatch_iters {
            let is_final = iter + 1 == predispatch_iters;

            // Dispatch: expert outputs for the current h_capture approximation.
            let h2 = if h_capture.is_empty() {
                vec![vec![0.0f32; hidden]; num_layers]
            } else {
                remote.forward_moe_predispatch(&h_capture, &routers_all, norm_offset, eps)?
            };

            if !is_final {
                // Non-final apply pass: inject h2, capture updated h_post_attn,
                // then roll back KV so only the last pass keeps it.
                let h2_ref = &h2;
                let mut new_h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
                let new_h = &mut new_h_capture;
                let mut moe_apply = |layer: usize, h: &[f32]| -> Vec<f32> {
                    if new_h.len() == layer {
                        new_h.push(h.to_vec());
                    }
                    h2_ref
                        .get(layer)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0f32; hidden])
                };
                backend.decode_token_with_moe(
                    &layers,
                    &x_tok,
                    hidden,
                    intermediate,
                    &mut moe_apply,
                );
                backend.truncate_kv_cache(kv_len);
                h_capture = new_h_capture;
            } else {
                // Final apply pass: inject best-available h2, advance KV permanently.
                let h2_ref = &h2;
                let mut moe_final = |layer: usize, _h: &[f32]| -> Vec<f32> {
                    h2_ref
                        .get(layer)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0f32; hidden])
                };
                h_out_opt = backend.decode_token_with_moe(
                    &layers,
                    &x_tok,
                    hidden,
                    intermediate,
                    &mut moe_final,
                );
            }
        }

        let h_out = h_out_opt
            .ok_or_else(|| RemoteMoeError::BadResponse("predispatch: no output".into()))?;

        // Pick next token.
        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out.clone())
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let next_tok_id = pick_next_filtered_with_policy(
            index,
            weights,
            &h_normed.row(0).to_owned(),
            backend,
            &suppress,
            tokenizer,
            &runtime.token_policy,
        );

        decode_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        transport_tokens += 1;
        if let Some(before) = token_bytes_before.as_ref() {
            crate::ffn::moe_remote::metrics::print_delta("decode", step, before);
        }
        let tok_str = detok.push(next_tok_id);
        let is_eos = eos.is_eos_with_tokenizer(next_tok_id, &tok_str, tokenizer);
        tokens.push(tok_str);
        current_ids.push(next_tok_id);
        if is_eos {
            break;
        }
    }

    if let Some(before) = run_bytes_before.as_ref() {
        crate::ffn::moe_remote::metrics::print_summary("generate", before, transport_tokens);
    }

    Ok(GridGenerateResult {
        tokens,
        decode_ms,
        ffn_rtt_ms: Vec::new(),
    })
}
