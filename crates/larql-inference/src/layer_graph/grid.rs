//! Autoregressive generation via a sharded expert grid.
//!
//! Uses the Metal pipeline for attention + dense FFN (same as normal `generate`),
//! but intercepts the MoE expert block per layer via a callback that dispatches
//! to remote shards over HTTP instead of calling `cpu_moe_forward` locally.
//!
//! The hook: `ComputeBackend::decode_token_with_moe(layers, x, ..., moe_fn)`
//! where `moe_fn(layer, h_post_attn) -> Vec<f32>` calls
//! `RemoteMoeBackend::forward_moe`.
//!
//! # Diagnostics
//!
//! Set `SKIP_MOE=1` to zero out the expert block on every decode step.
//! This isolates whether errors come from remote dispatch vs. dense FFN.

use larql_compute::prelude::*;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

use std::collections::HashSet;

use larql_compute::cpu::ops::q4k_q8k_dot::{quantize_x_to_q8k, Q8KActivation};

use crate::ffn::moe_remote::{InflightMoe, MoeRouterWeights, RemoteMoeError, ShardStream};
use crate::ffn::RemoteMoeBackend;
use crate::ffn::{FfnBackend, LayerShardedBackend};
use crate::forward::{apply_norm, embed_tokens_pub};
use crate::layer_graph::generate::detok::Detokenizer;
use crate::layer_graph::generate::eos::EosConfig;
use crate::layer_graph::generate::lm_head_topk as lm_topk;
use crate::layer_graph::pipeline_layer::{
    attention_geometry_for_arch_layer, build_pipeline_layers, kv_cache_shapes_for_arch,
    patch_pipeline_layers_for_remote_ffn, patch_pipeline_layers_for_remote_moe,
    DEFAULT_GPU_KV_CACHE_MAX_SEQ,
};
use crate::residual::rms_norm;

/// IDs of tokens that should never be picked during text generation.
///
/// Built from the tokenizer's `added_tokens` table (everything marked
/// `special: true`) minus any IDs in the EOS set — those are kept so the
/// EOS check in [`EosConfig`] can fire when the model wants to halt.
///
/// Without this filter, Q4_K quantisation noise occasionally lifts a special
/// token's logit above the intended next-word logit. On Gemma 4 26B-A4B,
/// `<mask>` (id 4) and the channel/turn markers leak into the answer at
/// random positions, producing fragments like "The<mask>capital of France".
fn build_special_suppress_set(tokenizer: &tokenizers::Tokenizer, eos: &EosConfig) -> HashSet<u32> {
    let mut out = HashSet::new();
    // 1. Anything the tokenizer config explicitly marks as a special added
    //    token (`<bos>`, `<mask>`, `<|tool>`, channel/turn markers, etc.).
    for (&id, added) in tokenizer.get_added_tokens_decoder().iter() {
        if added.special && !eos.eos_token_ids.contains(&id) {
            out.insert(id);
        }
    }
    // 2. Vocab-resident structural tokens that aren't flagged `special` but
    //    should never appear in a natural-language answer:
    //      - `<unusedN>` placeholders reserved for future training,
    //      - `[multimodal]` and similar bracketed markers,
    //      - HTML/markdown tags (`<table>`, `<h1>`, `<strong>`, …),
    //
    //    Without this widening, Q4_K quantisation noise on Gemma 4 26B-A4B
    //    occasionally outranks the intended next-word logit with one of
    //    these markers, producing fragments like "The<mask>capital..." or
    //    "The<unused25>...". Suppressing pulls the next-best legitimate
    //    word continuation forward, and the cascade effect through the KV
    //    cache cleans up later positions too (we observed `<0xC2>` →
    //    "랑" sequences disappear once position 1 picks a real word).
    let vocab = tokenizer.get_vocab(true);
    let mut structural_count = 0;
    for (tok, &id) in vocab.iter() {
        if eos.eos_token_ids.contains(&id) || out.contains(&id) {
            continue;
        }
        if is_structural_marker(tok) {
            out.insert(id);
            structural_count += 1;
        }
    }
    if std::env::var("LARQL_DEBUG_TOKEN_IDS").is_ok() {
        eprintln!(
            "[suppress] {} ids ({} from added_tokens.special, {} from structural-marker scan)",
            out.len(),
            out.len() - structural_count,
            structural_count,
        );
        // Dump a sample so we can see what got captured.
        let mut sorted: Vec<u32> = out.iter().copied().collect();
        sorted.sort_unstable();
        let sample: Vec<String> = sorted
            .iter()
            .take(20)
            .map(|id| {
                let raw = tokenizer.id_to_token(*id).unwrap_or_default();
                format!("{id}={raw:?}")
            })
            .collect();
        eprintln!("[suppress] first 20: {}", sample.join(", "));
        // Also explicitly probe id 31 (`<unused25>`) and id 5 (`[multimodal]`).
        for &probe in &[5u32, 31, 4, 168, 184] {
            let raw = tokenizer.id_to_token(probe).unwrap_or_default();
            let in_set = out.contains(&probe);
            let in_vocab = vocab.contains_key(&raw);
            eprintln!(
                "[suppress] probe id={probe} raw={raw:?} in_set={in_set} in_vocab={in_vocab}"
            );
        }
    }
    out
}

/// Returns `true` for vocab strings that look like structural markup or
/// reserved placeholders rather than natural-language tokens. Conservative:
/// only matches strings of the form `<...>`, `</...>`, or `[...]` with
/// non-whitespace bodies. Whitespace tokens (`\n`, `▁`-prefixed,
/// `▁▁▁...`) are intentionally NOT matched — those are legitimate parts
/// of normal text.
fn is_structural_marker(tok: &str) -> bool {
    if tok.is_empty() {
        return false;
    }
    let trimmed = tok.trim();
    if trimmed.len() < 2 {
        return false;
    }
    let bytes = trimmed.as_bytes();
    let first = bytes[0];
    let last = bytes[bytes.len() - 1];
    let bracketed = (first == b'<' && last == b'>') || (first == b'[' && last == b']');
    if !bracketed {
        return false;
    }
    // Body must be non-empty and contain no whitespace (markers are tight
    // tokens; a token like `<some real text>` from natural language would
    // contain a space and shouldn't be suppressed).
    let body = &trimmed[1..trimmed.len() - 1];
    !body.is_empty() && !body.chars().any(char::is_whitespace)
}

/// Pick the top-1 vocabulary id from logits, skipping any id in `suppress`.
///
/// Falls back to the raw argmax when every top candidate is suppressed
/// (degenerate case — should never happen unless `suppress` covers most of
/// the vocab).
///
/// Set `LARQL_DEBUG_TOPK=1` to log the top-5 logit candidates per step;
/// useful when the chosen token is wrong and you want to see whether the
/// right answer was even in the running.
fn pick_next_filtered(
    index: &VectorIndex,
    weights: &ModelWeights,
    h: &ndarray::Array1<f32>,
    backend: &dyn ComputeBackend,
    suppress: &HashSet<u32>,
    tokenizer: &tokenizers::Tokenizer,
) -> u32 {
    let debug_topk = std::env::var("LARQL_DEBUG_TOPK").is_ok();
    if suppress.is_empty() && !debug_topk {
        return lm_topk(index, weights, h, 1, backend)
            .into_iter()
            .next()
            .map(|(id, _)| id)
            .unwrap_or(0);
    }
    // Pull a wider top-K so that when the model's logits put many
    // structural markers at the top (which Q4_K-quantised Gemma 4 26B-A4B
    // does at the first answer position), we still find a real word.
    let candidates = lm_topk(index, weights, h, 256, backend);
    if debug_topk {
        let summary: Vec<String> = candidates
            .iter()
            .take(8)
            .map(|(id, score)| {
                let raw = tokenizer.id_to_token(*id).unwrap_or_default();
                let mark = if suppress.contains(id) { "✗" } else { " " };
                format!("{mark}id={id:6} {score:+.4e} {raw:?}")
            })
            .collect();
        let max_abs = candidates.iter().fold(0.0f32, |a, &(_, s)| a.max(s.abs()));
        let nan_count = candidates.iter().filter(|(_, s)| s.is_nan()).count();
        let zero_count = candidates.iter().filter(|(_, s)| *s == 0.0).count();
        let suppressed_in_top16 = candidates
            .iter()
            .take(16)
            .filter(|(id, _)| suppress.contains(id))
            .count();
        eprintln!(
            "    top8: {}\n    (max|score|={max_abs:.6e}  zeros={zero_count}/{}  nans={nan_count}  suppressed_top16={suppressed_in_top16}/16)",
            summary.join("  |  "),
            candidates.len()
        );
    }
    candidates
        .iter()
        .find(|(id, _)| !suppress.contains(id))
        .or_else(|| candidates.first())
        .map(|(id, _)| *id)
        .unwrap_or(0)
}

// ── Bottleneck diagnostic ────────────────────────────────────────────────────
//
// Activated by `LARQL_MOE_TIMING=1`.  The streaming path swaps
// `forward_moe_stream` for an explicit fire/collect_with_timing pair so we can
// see, for every MoE layer of every decoded token:
//
//   - `total_ms`:        wall-clock time inside the moe_fn closure
//   - `route_fire_ms`:   CPU routing + non-blocking fire
//   - `collect_ms`:      condvar-blocking wait for all shards' h2 frames
//   - per-shard `(wall_ms, server_compute_ms)` so `network_ms` is derivable
//     as `wall_ms − server_compute_ms`
//
// Everything is per-MoE-layer; the GPU side (attention + dense FFN) is timed
// independently by `LARQL_GPU_TIMING=1` in the metal backend.

#[derive(Clone, Debug)]
struct LayerTiming {
    layer: usize,
    total_ms: f32,
    route_fire_ms: f32,
    collect_ms: f32,
    /// One entry per shard: `(wall_collect_ms, server_compute_ms)`.
    per_shard: Vec<(f32, f32)>,
}

/// Sum of per-shard wall times — pre-2026-05-02 this matched `collect_ms`
/// because shards collected sequentially. After the parallel-collect change
/// (`forward_moe_stream_collect_with_timing` uses `std::thread::scope`),
/// `collect_ms ≈ max(per_shard.wall)` not the sum. Kept for diagnostics:
/// `shard_wall_sum / collect_ms` shows the parallel-collect speedup ratio
/// (≥ N for an N-shard topology where the parallelism is fully realised).
fn shard_wall_sum(t: &LayerTiming) -> f32 {
    t.per_shard.iter().map(|(w, _)| *w).sum()
}

/// Max of per-shard wall times — post-2026-05-02 this matches `collect_ms`
/// to within microseconds (parallel collect → bound by the slowest shard).
fn shard_wall_max(t: &LayerTiming) -> f32 {
    t.per_shard.iter().map(|(w, _)| *w).fold(0.0, f32::max)
}

fn shard_compute_max(t: &LayerTiming) -> f32 {
    t.per_shard.iter().map(|(_, c)| *c).fold(0.0, f32::max)
}

fn print_token_breakdown(label: &str, tok_idx: usize, timings: &[LayerTiming]) {
    if timings.is_empty() {
        return;
    }
    let n = timings.len();
    let total: f32 = timings.iter().map(|t| t.total_ms).sum();
    let route: f32 = timings.iter().map(|t| t.route_fire_ms).sum();
    let collect: f32 = timings.iter().map(|t| t.collect_ms).sum();
    let server_max: f32 = timings.iter().map(shard_compute_max).sum();
    let network = (collect - server_max).max(0.0);
    eprintln!(
        "[moe-timing] {label} tok={tok_idx} layers={n} \
         moe_total={total:.1}ms (route+fire={route:.1}ms collect={collect:.1}ms \
         | server_compute≈{server_max:.1}ms network≈{network:.1}ms)"
    );
}

fn print_run_summary(label: &str, per_token: &[Vec<LayerTiming>]) {
    if per_token.is_empty() {
        return;
    }
    let n_tokens = per_token.len();
    let layers_per_tok = per_token.iter().map(|v| v.len()).max().unwrap_or(0);

    // Per-token aggregates.
    let mut tot_total = 0.0f32;
    let mut tot_route = 0.0f32;
    let mut tot_collect = 0.0f32;
    let mut tot_server = 0.0f32;
    for tok in per_token {
        tot_total += tok.iter().map(|t| t.total_ms).sum::<f32>();
        tot_route += tok.iter().map(|t| t.route_fire_ms).sum::<f32>();
        tot_collect += tok.iter().map(|t| t.collect_ms).sum::<f32>();
        tot_server += tok.iter().map(shard_compute_max).sum::<f32>();
    }
    let avg_total = tot_total / n_tokens as f32;
    let avg_route = tot_route / n_tokens as f32;
    let avg_collect = tot_collect / n_tokens as f32;
    let avg_server = tot_server / n_tokens as f32;
    let avg_net = (avg_collect - avg_server).max(0.0);

    eprintln!(
        "[moe-timing] {label} SUMMARY ({n_tokens} tokens, {layers_per_tok} MoE layers/token)"
    );
    eprintln!(
        "[moe-timing]   per-token avg: moe_total={avg_total:.1}ms \
         (route+fire={avg_route:.1}ms collect={avg_collect:.1}ms \
         | server_compute≈{avg_server:.1}ms network≈{avg_net:.1}ms)"
    );
    if layers_per_tok > 0 {
        let avg_per_layer_total = avg_total / layers_per_tok as f32;
        let avg_per_layer_collect = avg_collect / layers_per_tok as f32;
        let avg_per_layer_server = avg_server / layers_per_tok as f32;
        let avg_per_layer_net = (avg_per_layer_collect - avg_per_layer_server).max(0.0);
        eprintln!(
            "[moe-timing]   per-layer avg: total={avg_per_layer_total:.2}ms \
             collect={avg_per_layer_collect:.2}ms \
             (server≈{avg_per_layer_server:.2}ms net≈{avg_per_layer_net:.2}ms)"
        );
    }
    // Bottleneck attribution: collect dominates when remote round-trip dwarfs
    // local routing.  The "X% of MoE time" framing is what the operator wants
    // — it's the actionable lever (move shards closer / use batch mode / …).
    if avg_total > 0.0 {
        let collect_pct = 100.0 * avg_collect / avg_total;
        let server_pct = 100.0 * avg_server / avg_total;
        let net_pct = 100.0 * avg_net / avg_total;
        let route_pct = 100.0 * avg_route / avg_total;
        eprintln!(
            "[moe-timing]   bottleneck: collect={collect_pct:.0}% \
             (of which server≈{server_pct:.0}%, network≈{net_pct:.0}%) \
             route+fire={route_pct:.0}%"
        );
    }
}

/// Inner moe call with optional timing capture.  Returns the h2 vec.  When
/// `timing.is_some()`, splits the call into fire + collect_with_timing so we
/// can record per-shard wall + server-compute breakdown.
fn moe_call_timed(
    remote: &RemoteMoeBackend,
    layer: usize,
    h_post_attn: &[f32],
    router: &MoeRouterWeights<'_>,
    streams: &mut [ShardStream],
    norm_offset: f32,
    eps: f32,
    timing: Option<&mut Vec<LayerTiming>>,
) -> Result<Vec<f32>, RemoteMoeError> {
    if streams.is_empty() {
        return remote.forward_moe(layer, h_post_attn, router, norm_offset, eps);
    }
    let Some(timing) = timing else {
        return remote.forward_moe_stream(layer, h_post_attn, router, streams, norm_offset, eps);
    };
    let t_total = std::time::Instant::now();
    let t_fire = std::time::Instant::now();
    let inflight =
        remote.forward_moe_stream_fire(layer, h_post_attn, router, streams, norm_offset, eps)?;
    let route_fire_ms = t_fire.elapsed().as_secs_f32() * 1000.0;
    let t_collect = std::time::Instant::now();
    let (h2, per_shard) = remote.forward_moe_stream_collect_with_timing(streams, inflight)?;
    let collect_ms = t_collect.elapsed().as_secs_f32() * 1000.0;
    let total_ms = t_total.elapsed().as_secs_f32() * 1000.0;
    timing.push(LayerTiming {
        layer,
        total_ms,
        route_fire_ms,
        collect_ms,
        per_shard,
    });
    Ok(h2)
}

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
) -> Option<MoeRouterWeights<'a>> {
    let router_proj_key = arch.moe_router_key(layer)?;
    let router_proj = weights.vectors.get(&router_proj_key)?.as_slice();
    let sl = |k: Option<String>| -> &'a [f32] {
        k.and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    };
    let arch_top_k = arch.num_experts_per_token();
    let top_k = std::env::var("LARQL_MOE_TOP_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|k| k.clamp(1, arch_top_k))
        .unwrap_or(arch_top_k);
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

#[derive(Debug)]
pub struct GridGenerateResult {
    pub tokens: Vec<String>,
    pub decode_ms: Vec<f64>,
    /// Sum of remote FFN round-trip time per decode step (all layers, streaming path only).
    /// Empty for MoE paths and the batch predispatch path.
    pub ffn_rtt_ms: Vec<f64>,
}

/// Greedy autoregressive generation through a remote-expert grid.
///
/// Requires a Metal (or Q4-capable) backend — attention and dense FFN run on
/// the GPU exactly as in the normal `generate()` path.  Expert blocks are
/// dispatched to `remote` instead of running locally.
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
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    // ── Build pipeline layers (same as generate()) ────────────────────────────
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let q4_ffn = gate_index
        .interleaved_q4k_mmap_ref()
        .or_else(|| gate_index.interleaved_q4_mmap_ref())
        .ok_or_else(|| {
            RemoteMoeError::BadResponse("no interleaved Q4 FFN mmap in vindex".into())
        })?;
    let ffn_is_q4k = gate_index.interleaved_q4k_mmap_ref().is_some();

    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let intermediate = gate_index.num_features(0);
    let q4_ffn_per_matrix = ffn_format
        .packed_matrix_bytes(intermediate, hidden)
        .ok_or_else(|| RemoteMoeError::BadResponse("unsupported interleaved FFN format".into()))?;

    let mut layers = build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn,
        q4_ffn_per_matrix,
        ffn_format,
    );
    // Client-only vindexes (--moe-shards without local expert bytes) have
    // layer.moe = None for every layer, so has_moe = false and moe_fn would
    // never be called.  Inject stubs so the Metal decode knows to dispatch to
    // moe_fn (the remote shard callback) instead of local cpu_moe_forward.
    patch_pipeline_layers_for_remote_moe(&mut layers, weights);

    let attention = attention_geometry_for_arch_layer(weights, 0);

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
    // Run one `decode_token_with_moe` per prompt token rather than `prefill_q4`.
    // `prefill_q4` does not correctly apply MoE experts for hybrid-MoE post-norm
    // models (Gemma 4 26B-A4B), so the first-token prediction and subsequent KV
    // cache entries are wrong.  Sequential decode builds the KV cache correctly
    // — each token processes with the proper remote expert contribution.
    backend.reset_kv_cache();
    {
        let kv_shapes = kv_cache_shapes_for_arch(weights);
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
    }

    let skip_moe = std::env::var("SKIP_MOE").is_ok();
    let timing_enabled = std::env::var("LARQL_MOE_TIMING").is_ok();
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
    let suppress = build_special_suppress_set(tokenizer, eos);

    for (prefill_idx, &tok_id) in prompt_ids.iter().enumerate() {
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
            let router = match build_router(weights, arch, layer) {
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

        let h = backend.decode_token_with_moe(
            &layers,
            &x_tok,
            hidden,
            intermediate,
            attention.q_dim,
            attention.kv_dim,
            attention.num_q_heads,
            attention.num_kv_heads,
            attention.head_dim,
            attention.rope_base,
            &mut moe_fn,
        );
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
    }

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();

    // First token from the (correct) prefill output.
    let prefill_h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let h_norm0 = apply_norm(weights, &prefill_h_arr, arch.final_norm_key(), norm_offset);
    let last0 = h_norm0.row(0).to_owned();
    let first_id = pick_next_filtered(index, weights, &last0, backend, &suppress, tokenizer);

    let first_tok = detok.push(first_id);
    let first_is_eos = eos.is_eos_with_tokenizer(first_id, &first_tok, tokenizer);
    let debug_ids = std::env::var("LARQL_DEBUG_TOKEN_IDS").is_ok();
    if debug_ids {
        let raw = tokenizer.id_to_token(first_id).unwrap_or_default();
        eprintln!("[tok 0] id={first_id:6} raw={raw:?} delta={first_tok:?}");
    }
    tokens.push(first_tok);
    current_ids.push(first_id);
    if first_is_eos || tokens.len() >= max_tokens {
        return Ok(GridGenerateResult {
            tokens,
            decode_ms: vec![0.0],
            ffn_rtt_ms: Vec::new(),
        });
    }

    for step in 0..max_tokens.saturating_sub(1) {
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
        let split_disabled = std::env::var("LARQL_MOE_NO_SPLIT").is_ok();
        let result = if streams.is_empty() || split_disabled {
            let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
                if skip_moe {
                    return vec![0.0f32; hidden];
                }
                if step_error.is_some() {
                    return vec![0.0f32; hidden];
                }
                let router = match build_router(weights, arch, layer) {
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
            backend.decode_token_with_moe(
                &layers,
                &x_tok,
                hidden,
                intermediate,
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
                &mut moe_fn,
            )
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
                let router = match build_router(weights, arch, layer) {
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
            let mut collect_fn = |layer: usize| -> Vec<f32> {
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
                                layer,
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
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
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
        if std::env::var("LARQL_DEBUG_TOKEN_IDS").is_ok() {
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
        let next_id =
            pick_next_filtered(index, weights, &last_hidden, backend, &suppress, tokenizer);

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
    let predispatch_iters = predispatch_iters.max(1);
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    let gate_index: &dyn larql_vindex::GateIndex = index;
    let q4_ffn = gate_index
        .interleaved_q4k_mmap_ref()
        .or_else(|| gate_index.interleaved_q4_mmap_ref())
        .ok_or_else(|| RemoteMoeError::BadResponse("no interleaved Q4 FFN mmap".into()))?;
    let ffn_is_q4k = gate_index.interleaved_q4k_mmap_ref().is_some();
    let intermediate = gate_index.num_features(0);
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format
        .packed_matrix_bytes(intermediate, hidden)
        .ok_or_else(|| RemoteMoeError::BadResponse("unsupported interleaved FFN format".into()))?;
    let mut layers = crate::layer_graph::pipeline_layer::build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn,
        q4_ffn_per_matrix,
        ffn_format,
    );
    patch_pipeline_layers_for_remote_moe(&mut layers, weights);

    let attention = attention_geometry_for_arch_layer(weights, 0);

    // Prefill: sequential decode_token_with_moe (same as streaming variant).
    backend.reset_kv_cache();
    {
        let kv_shapes = kv_cache_shapes_for_arch(weights);
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
    }

    let skip_moe = std::env::var("SKIP_MOE").is_ok();
    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    // Build routers once here so both prefill and decode loops can use them.
    let routers_all: Vec<MoeRouterWeights<'_>> = (0..num_layers)
        .filter_map(|l| build_router(weights, arch, l))
        .collect();

    for &tok_id in &prompt_ids {
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
            backend.decode_token_with_moe(
                &layers,
                &x_tok,
                hidden,
                intermediate,
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
                &mut moe_pass0,
            );
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
                backend.decode_token_with_moe(
                    &layers,
                    &x_tok,
                    hidden,
                    intermediate,
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
                    &mut fn_apply,
                );
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
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
                    &mut fn_final,
                );
            }
        }
        last_hidden_vec = h2_final.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode returned None during prefill".into())
        })?;
    }

    // First token from prefill.
    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();
    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(&prompt_ids);
    let suppress = build_special_suppress_set(tokenizer, eos);
    let pfa = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let pfn = apply_norm(weights, &pfa, arch.final_norm_key(), norm_offset);
    let first_id = pick_next_filtered(
        index,
        weights,
        &pfn.row(0).to_owned(),
        backend,
        &suppress,
        tokenizer,
    );
    let first_tok = detok.push(first_id);
    let first_eos = eos.is_eos_with_tokenizer(first_id, &first_tok, tokenizer);
    tokens.push(first_tok);
    current_ids.push(first_id);
    if first_eos || tokens.len() >= max_tokens {
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

    for _step in 0..max_tokens.saturating_sub(1) {
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
            backend.decode_token_with_moe(
                &layers,
                &x_tok,
                hidden,
                intermediate,
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
                &mut moe_pass0,
            );
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
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
                    &mut |_layer: usize, _h: &[f32]| vec![0.0f32; hidden],
                )
                .ok_or_else(|| RemoteMoeError::BadResponse("skip_moe pass returned None".into()))?;
            let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out_skip.clone())
                .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
            let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
            let next_tok_id = pick_next_filtered(
                index,
                weights,
                &h_normed.row(0).to_owned(),
                backend,
                &suppress,
                tokenizer,
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
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
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
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
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
        let next_tok_id = pick_next_filtered(
            index,
            weights,
            &h_normed.row(0).to_owned(),
            backend,
            &suppress,
            tokenizer,
        );

        decode_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let tok_str = detok.push(next_tok_id);
        let is_eos = eos.is_eos_with_tokenizer(next_tok_id, &tok_str, tokenizer);
        tokens.push(tok_str);
        current_ids.push(next_tok_id);
        if is_eos {
            break;
        }
    }

    Ok(GridGenerateResult {
        tokens,
        decode_ms,
        ffn_rtt_ms: Vec::new(),
    })
}

/// Autoregressive generation with Metal GPU attention and remote dense FFN.
///
/// For dense models (not MoE) where the entire FFN should be offloaded to a
/// remote server (`--ffn URL`). Metal handles attention on the local GPU;
/// every layer's FFN is a round trip to `remote` via `LayerShardedBackend::forward`.
///
/// Analogous to [`generate_with_remote_moe`] but without the local expert block:
/// `new_h = attn_out + remote_ffn_out` (no local FFN component).
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
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    // ── Build pipeline layers ─────────────────────────────────────────────────
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let q4_ffn = gate_index
        .interleaved_q4k_mmap_ref()
        .or_else(|| gate_index.interleaved_q4_mmap_ref())
        .ok_or_else(|| {
            RemoteMoeError::BadResponse("no interleaved Q4 FFN mmap in vindex".into())
        })?;
    let ffn_is_q4k = gate_index.interleaved_q4k_mmap_ref().is_some();
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let intermediate = gate_index.num_features(0);
    let q4_ffn_per_matrix = ffn_format
        .packed_matrix_bytes(intermediate, hidden)
        .ok_or_else(|| RemoteMoeError::BadResponse("unsupported interleaved FFN format".into()))?;

    let mut layers = build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn,
        q4_ffn_per_matrix,
        ffn_format,
    );
    // Mark every layer as remote-FFN so the Metal decode loop skips the
    // local GPU FFN dispatches and routes through the moe_fn callback instead.
    patch_pipeline_layers_for_remote_ffn(&mut layers);

    let attention = attention_geometry_for_arch_layer(weights, 0);

    // ── KV cache setup ────────────────────────────────────────────────────────
    backend.reset_kv_cache();
    {
        let kv_shapes = kv_cache_shapes_for_arch(weights);
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
    }

    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(&prompt_ids);

    let suppress = build_special_suppress_set(tokenizer, eos);

    // ── Prefill ───────────────────────────────────────────────────────────────
    for (prefill_idx, &tok_id) in prompt_ids.iter().enumerate() {
        let tok_embed = crate::forward::embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            let x = ndarray::Array2::from_shape_vec((1, hidden), h_post_attn.to_vec())
                .expect("shape must match hidden");
            remote.forward(layer, &x).row(0).to_vec()
        };

        let h = backend.decode_token_with_moe(
            &layers,
            &x_tok,
            hidden,
            intermediate,
            attention.q_dim,
            attention.kv_dim,
            attention.num_q_heads,
            attention.num_kv_heads,
            attention.head_dim,
            attention.rope_base,
            &mut moe_fn,
        );
        last_hidden_vec = h.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode_token_with_moe returned None during prefill".into())
        })?;
        let _ = prefill_idx; // suppress unused-variable warning
    }

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();
    let mut ffn_rtt_ms = Vec::new();

    // First token from the prefill output.
    let prefill_h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let h_norm0 = apply_norm(weights, &prefill_h_arr, arch.final_norm_key(), norm_offset);
    let last0 = h_norm0.row(0).to_owned();
    let first_id = pick_next_filtered(index, weights, &last0, backend, &suppress, tokenizer);

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

        // Time just the remote round-trips; Cell avoids &mut aliasing with the closure.
        let step_ffn_cell = std::cell::Cell::new(0.0f64);
        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            let t_ffn = std::time::Instant::now();
            // Try Q8K NEON path (avoids gate+up dequant on server; hidden must be
            // a multiple of 256 for Q8K block alignment).
            let result = if hidden.is_multiple_of(256) {
                let h_ffn = apply_norm_for_ffn(weights, h_post_attn, layer);
                let q8k = quantize_x_to_q8k(&h_ffn);
                remote.forward_single_q8k(layer, &q8k).unwrap_or_else(|| {
                    let x = ndarray::Array2::from_shape_vec((1, hidden), h_post_attn.to_vec())
                        .expect("shape must match hidden");
                    remote.forward(layer, &x).row(0).to_vec()
                })
            } else {
                let x = ndarray::Array2::from_shape_vec((1, hidden), h_post_attn.to_vec())
                    .expect("shape must match hidden");
                remote.forward(layer, &x).row(0).to_vec()
            };
            step_ffn_cell.set(step_ffn_cell.get() + t_ffn.elapsed().as_secs_f64() * 1000.0);
            result
        };

        let h_vec = backend
            .decode_token_with_moe(
                &layers,
                &x_tok,
                hidden,
                intermediate,
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
                &mut moe_fn,
            )
            .ok_or_else(|| {
                RemoteMoeError::BadResponse("decode_token_with_moe returned None".into())
            })?;

        last_hidden_vec = h_vec;
        ffn_rtt_ms.push(step_ffn_cell.get());

        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let last_hidden = h_normed.row(0).to_owned();

        let next_id =
            pick_next_filtered(index, weights, &last_hidden, backend, &suppress, tokenizer);

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

/// Apply the FFN input norm to `h_post_attn`, producing the pre-FFN normed
/// activation `h_ffn` that the server would compute internally.
///
/// Mirrors the first step of `run_ffn` in `forward/layer.rs`:
/// - When `arch.has_post_norms()` is true → `pre_feedforward_layernorm_key`
/// - Otherwise → `post_attention_layernorm_key`
///
/// The result is the input to `ffn.forward(layer, &h_ffn)`.  Quantising it
/// to Q8_K and sending it saves `rms_norm` work on the server.
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
        None => {
            let normed_row = rms_norm(&h, None, norm_offset);
            normed_row
        }
    };
    normed.row(0).to_vec()
}

/// Dispatch FFN outputs for all layers, using the Q8K wire format when possible.
///
/// 1. For each layer in `h_capture`, apply the FFN input norm and quantise to Q8_K.
/// 2. Call `remote.forward_predispatch_all_q8k()`.
/// 3. If any output vector is all-zeros (indicating the server returned zeros
///    for a layer it couldn't handle), fall back to `forward_predispatch_all` for
///    the entire batch to keep semantics consistent.
///
/// Returns `Vec<Vec<f32>>` in the same format as `forward_predispatch_all`.
fn dispatch_ffn_with_q8k_fallback(
    remote: &LayerShardedBackend,
    weights: &ModelWeights,
    h_capture: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let hidden = h_capture.first().map(|v| v.len()).unwrap_or(0);
    // Require hidden to be a multiple of 256 (Q8_K block size).
    if hidden == 0 || !hidden.is_multiple_of(256) {
        return remote.forward_predispatch_all(h_capture);
    }

    // Norm + quantise all captured layers.
    let q8k_all: Vec<Q8KActivation> = h_capture
        .iter()
        .enumerate()
        .map(|(layer, h)| {
            let h_ffn = apply_norm_for_ffn(weights, h, layer);
            quantize_x_to_q8k(&h_ffn)
        })
        .collect();

    let results = remote.forward_predispatch_all_q8k(&q8k_all);

    // Check: if all results are zeros for any layer, the Q8K path returned
    // a fallback stub — re-dispatch via f32.
    let any_zero_result = results.iter().any(|v| v.iter().all(|&x| x == 0.0));
    if any_zero_result {
        remote.forward_predispatch_all(h_capture)
    } else {
        results
    }
}

/// Batch pre-dispatch variant of [`generate_with_remote_ffn`].
///
/// Each decode step runs two Metal passes:
///   1. **Capture pass**: Metal runs attention with zero FFN contributions,
///      capturing `h_post_attn` at each layer.  KV is rolled back.
///   2. **Parallel dispatch**: `forward_predispatch_all` fires one HTTP
///      request per layer concurrently.
///   3. **Apply pass**: Metal re-runs with the pre-computed FFN outputs
///      injected via `moe_fn`.  KV advances permanently.
///
/// Repeat for `predispatch_iters` if > 1 to refine the approximation.
///
/// **Trade-off vs streaming**: streaming is exact (each layer's `h_post_attn`
/// includes all previous layers' FFN contributions). Batch uses the capture
/// pass `h_post_attn` as an approximation — the error is small in practice
/// and typically produces the same top-1 token.
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
    let predispatch_iters = predispatch_iters.max(1);
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    let gate_index: &dyn larql_vindex::GateIndex = index;
    let q4_ffn = gate_index
        .interleaved_q4k_mmap_ref()
        .or_else(|| gate_index.interleaved_q4_mmap_ref())
        .ok_or_else(|| {
            RemoteMoeError::BadResponse("no interleaved Q4 FFN mmap in vindex".into())
        })?;
    let ffn_is_q4k = gate_index.interleaved_q4k_mmap_ref().is_some();
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let intermediate = gate_index.num_features(0);
    let q4_ffn_per_matrix = ffn_format
        .packed_matrix_bytes(intermediate, hidden)
        .ok_or_else(|| RemoteMoeError::BadResponse("unsupported interleaved FFN format".into()))?;

    let mut layers = build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn,
        q4_ffn_per_matrix,
        ffn_format,
    );
    patch_pipeline_layers_for_remote_ffn(&mut layers);

    let attention = attention_geometry_for_arch_layer(weights, 0);

    backend.reset_kv_cache();
    {
        let kv_shapes = kv_cache_shapes_for_arch(weights);
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
    }

    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(&prompt_ids);

    let suppress = build_special_suppress_set(tokenizer, eos);

    // ── Prefill: sequential (same as streaming variant) ───────────────────────
    for &tok_id in &prompt_ids {
        let tok_embed = crate::forward::embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();
        let kv_len = backend.kv_cache_len();

        // Pass 0: capture h_post_attn (FFN = zeros).
        let mut h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        {
            let h_cap = &mut h_capture;
            let mut cap_fn = |layer: usize, h: &[f32]| -> Vec<f32> {
                if h_cap.len() == layer {
                    h_cap.push(h.to_vec());
                }
                vec![0.0f32; hidden]
            };
            backend.decode_token_with_moe(
                &layers,
                &x_tok,
                hidden,
                intermediate,
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
                &mut cap_fn,
            );
        }
        backend.truncate_kv_cache(kv_len);

        // Refinement iterations.
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
                backend.decode_token_with_moe(
                    &layers,
                    &x_tok,
                    hidden,
                    intermediate,
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
                    &mut fn_apply,
                );
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
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
                    &mut fn_final,
                );
            }
        }
        last_hidden_vec = h2_final.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode returned None during prefill".into())
        })?;
    }

    // First token from prefill.
    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();
    let prefill_h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let h_norm0 = apply_norm(weights, &prefill_h_arr, arch.final_norm_key(), norm_offset);
    let first_id = pick_next_filtered(
        index,
        weights,
        &h_norm0.row(0).to_owned(),
        backend,
        &suppress,
        tokenizer,
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

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut ffn_rtt_ms: Vec<f64> = Vec::new();
    for _step in 0..max_tokens.saturating_sub(1) {
        let t0 = std::time::Instant::now();
        let next_input_id = *current_ids.last().unwrap();
        let tok_embed = crate::forward::embed_tokens_pub(weights, &[next_input_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();
        let kv_len = backend.kv_cache_len();

        // Pass 0: capture h_post_attn (FFN = zeros), then roll back KV.
        let mut h_capture: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        {
            let h_cap = &mut h_capture;
            let mut cap_fn = |layer: usize, h: &[f32]| -> Vec<f32> {
                if h_cap.len() == layer {
                    h_cap.push(h.to_vec());
                }
                vec![0.0f32; hidden]
            };
            backend.decode_token_with_moe(
                &layers,
                &x_tok,
                hidden,
                intermediate,
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
                &mut cap_fn,
            );
        }
        backend.truncate_kv_cache(kv_len);

        // Refinement iterations.
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
                backend.decode_token_with_moe(
                    &layers,
                    &x_tok,
                    hidden,
                    intermediate,
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
                    &mut fn_apply,
                );
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
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
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

        let next_id =
            pick_next_filtered(index, weights, &last_hidden, backend, &suppress, tokenizer);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::{make_test_tokenizer, make_test_vindex, make_test_weights};
    use crate::ffn::moe_remote::RemoteMoeBackend;
    use larql_compute::CpuBackend;

    // ── generate_with_remote_moe — error path ────────────────────────────────

    #[test]
    fn errors_when_vindex_has_no_q4k_mmap() {
        let weights = make_test_weights();
        let idx = make_test_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);

        // make_test_vindex has no interleaved Q4K or Q4 mmap.
        // The function should fail at the mmap guard, before any GPU or shard call.
        let remote = RemoteMoeBackend::new_disconnected();
        let result = generate_with_remote_moe(
            &weights,
            &tokenizer,
            vec![0u32],
            1,
            &idx,
            &remote,
            &CpuBackend,
            &EosConfig::builtin(),
        );
        match result {
            Err(RemoteMoeError::BadResponse(msg)) => {
                assert!(
                    msg.contains("no interleaved Q4 FFN mmap"),
                    "unexpected error message: {msg}"
                );
            }
            other => panic!("expected BadResponse, got: {other:?}"),
        }
    }
}
