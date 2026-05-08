//! Metal GPU generate paths — fused prefill + KV-cached decode loop.

use super::detok::Detokenizer;
use super::eos::EosConfig;
use super::sampling::{Sampler, SamplingConfig};
use super::types::{GenerateResult, StageTimings};
use crate::layer_graph::pipeline_layer::{
    attention_geometry_for_arch_layer, kv_cache_shapes_for_arch, DEFAULT_GPU_KV_CACHE_MAX_SEQ,
};
use crate::layer_graph::CachedLayerGraph;
use crate::model::ModelWeights;
use larql_compute::prelude::*;

use super::cpu::{
    backend_supports_fused_q4_pipeline, generate_constrained_via_cpu_q4k,
    generate_constrained_via_cpu_q4k_streaming_sampled, generate_via_cpu_q4k,
};
use super::lm_head::{
    backend_lm_head_scores, cpu_lm_head_topk, lm_head_topk, pick_next_token_masked,
    pick_next_token_masked_sampled,
};

/// LM-head top-K size when running greedy decode. Matches the historical
/// behaviour preserved by [`generate`].
const LMHEAD_TOPK_GREEDY: usize = 5;
/// LM-head top-K minimum when sampling. Larger K gives the sampler enough
/// distribution mass to apply temperature / top-p meaningfully without
/// paying for a full-vocab gemv. `cfg.top_k.unwrap_or(0).max(this)` is
/// what actually gets requested.
const LMHEAD_TOPK_SAMPLING_MIN: usize = 64;

fn lmhead_k_for_sampling(cfg: &SamplingConfig) -> usize {
    if cfg.is_greedy() {
        LMHEAD_TOPK_GREEDY
    } else {
        cfg.top_k.unwrap_or(0).max(LMHEAD_TOPK_SAMPLING_MIN)
    }
}

/// Timings and forced tokens from [`stream_forced_full_logits`].
#[derive(Debug, Clone, Default)]
pub struct ForcedLogitsResult {
    /// Tokens returned by the caller and forced into the decode cache.
    pub forced_tokens: Vec<u32>,
    /// Fused prefill time for the seed token.
    pub prefill_ms: f64,
    /// Per forced-token decode-step time. Length is `forced_tokens.len() - 1`
    /// when at least one token was forced.
    pub decode_ms: Vec<f64>,
}

/// Stream full-vocabulary next-token logits while forcing known tokens
/// through the Q4K/Metal KV-cache path.
///
/// This is the Shannon-codec primitive: unlike [`generate_streaming`], this
/// does not sample. At each step the caller receives logits for
/// `p(next_token | context)` and returns the token id to append to the cache.
/// Encode returns the known corpus token; decode returns the arithmetic-decoded
/// token. The implementation reuses the same fused prefill and
/// `decode_token` machinery as generation, so each step extends the KV cache
/// instead of recomputing the full prefix.
#[allow(clippy::too_many_arguments)]
pub fn stream_forced_full_logits<F>(
    weights: &mut ModelWeights,
    first_token: u32,
    target_steps: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    mut on_logits: F,
) -> Result<ForcedLogitsResult, String>
where
    F: FnMut(usize, &[f32]) -> Result<u32, String>,
{
    if target_steps == 0 {
        return Ok(ForcedLogitsResult::default());
    }
    if !backend_supports_fused_q4_pipeline(backend) {
        return Err("forced Shannon logits require a fused Q4 backend; pass --metal".into());
    }
    if weights.arch.has_per_layer_embeddings() {
        return Err("forced Shannon logits do not yet support per-layer embeddings".into());
    }
    if weights.has_per_layer_ffn() {
        return Err("forced Shannon logits do not yet support per-layer expert FFN blobs".into());
    }

    let norm_offset = weights.arch.norm_weight_offset();
    let hidden = weights.hidden_size;
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_mmap_ref() {
        (Some(mmap), true)
    } else {
        (gate_index.interleaved_q4_mmap_ref(), false)
    };
    let has_q4k = index.attn_q4k_layer_data(0).is_some();
    let has_q8 = index.attn_q8_layer_data(0).is_some();
    if !backend.has_q4() || q4_ffn.is_none() || (!has_q4k && !has_q8) {
        return Err(
            "vindex is missing Q4 attention/FFN data required for forced Shannon logits".into(),
        );
    }

    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let intermediate = gate_index.num_features(0);
    let q4_ffn_per_matrix = ffn_format
        .packed_matrix_bytes(intermediate, hidden)
        .ok_or_else(|| "invalid Q4 FFN packed geometry".to_string())?;
    let q4_ffn_mmap = q4_ffn.unwrap();
    let num_layers = weights.num_layers;
    let layers = crate::layer_graph::pipeline_layer::build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn_mmap,
        q4_ffn_per_matrix,
        ffn_format,
    );
    let attention = attention_geometry_for_arch_layer(weights, 0);

    let prefill_start = std::time::Instant::now();
    backend.reset_kv_cache();
    {
        let kv_shapes = kv_cache_shapes_for_arch(weights);
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
    }

    let h_embed = crate::forward::embed_tokens_pub(weights, &[first_token]);
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();
    let softcap_val = weights.arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm_val = weights.arch.attn_q_norm_key(0).is_some();
    let h_vec = backend
        .prefill_q4(
            &layers,
            &x,
            hidden,
            intermediate,
            attention.q_dim,
            attention.kv_dim,
            1,
            attention.num_q_heads,
            attention.num_kv_heads,
            attention.head_dim,
            attention.rope_base,
            qk_norm_val,
            softcap_val,
        )
        .ok_or_else(|| "Q4 prefill failed".to_string())?;
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let mut h_1d = final_norm_row(weights, &h_vec, hidden, norm_offset)?;

    let mut forced_tokens = Vec::with_capacity(target_steps);
    let mut decode_ms = Vec::with_capacity(target_steps.saturating_sub(1));
    for step in 0..target_steps {
        let logits = full_logits_from_vindex(index, weights, &h_1d, backend)?;
        let forced = on_logits(step, &logits)?;
        forced_tokens.push(forced);

        if step + 1 == target_steps {
            break;
        }

        let decode_start = std::time::Instant::now();
        let h_tok = crate::forward::embed_tokens_pub(weights, &[forced]);
        let x_dec: Vec<f32> = h_tok.row(0).to_vec();
        let h_out = backend
            .decode_token(
                &layers,
                &x_dec,
                hidden,
                intermediate,
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
            )
            .ok_or_else(|| format!("Q4 decode failed at forced step {step}"))?;
        h_1d = final_norm_row(weights, &h_out, hidden, norm_offset)?;
        decode_ms.push(decode_start.elapsed().as_secs_f64() * 1000.0);
    }

    Ok(ForcedLogitsResult {
        forced_tokens,
        prefill_ms,
        decode_ms,
    })
}

fn final_norm_row(
    weights: &ModelWeights,
    h_vec: &[f32],
    hidden: usize,
    norm_offset: f32,
) -> Result<ndarray::Array1<f32>, String> {
    if h_vec.len() < hidden {
        return Err(format!(
            "hidden vector too short: got {}, need {}",
            h_vec.len(),
            hidden
        ));
    }
    let start = h_vec.len() - hidden;
    let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_vec[start..].to_vec())
        .map_err(|e| format!("hidden shape error: {e}"))?;
    let h_final =
        crate::forward::apply_norm(weights, &h_arr, weights.arch.final_norm_key(), norm_offset);
    Ok(h_final.row(0).to_owned())
}

fn full_logits_from_vindex(
    index: &larql_vindex::VectorIndex,
    weights: &ModelWeights,
    h_1d: &ndarray::Array1<f32>,
    backend: &dyn ComputeBackend,
) -> Result<Vec<f32>, String> {
    let vocab = index.vocab_size.max(weights.vocab_size);
    if vocab == 0 {
        return Err("vocab size is zero".into());
    }
    // Shannon coding needs encode and decode to rebuild identical frequency
    // tables. Prefer the stable-reduction LM-head route over the fastest
    // production route; tiny low-order logit drift is enough to desync an
    // arithmetic decoder on longer excerpts.
    let hits = index.lm_head_knn_backend_skip_q4k(h_1d, vocab, backend);
    if hits.is_empty() {
        return Err("vindex lm_head returned no scores".into());
    }

    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let softcap = weights.arch.final_logit_softcapping();
    let mut logits = vec![f32::NEG_INFINITY; vocab];
    for (tid, score) in hits {
        let idx = tid as usize;
        if idx >= logits.len() {
            continue;
        }
        let mut logit = score * inv_scale;
        if let Some(cap) = softcap {
            logit = (logit / cap).tanh() * cap;
        }
        logits[idx] = logit;
    }
    Ok(logits)
}

/// Greedy multi-token generation. Thin wrapper over
/// [`generate_with_sampling`] with [`SamplingConfig::greedy`] and
/// [`EosConfig::builtin`] — preserves the historical behaviour of every
/// caller in the crate.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
) -> GenerateResult {
    generate_with_sampling(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        backend,
        cached_layers,
        layer_range,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
    )
}

/// Multi-token generation with explicit sampling and EOS configuration.
/// Identical to [`generate_streaming`] but with no per-token callback.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_sampling(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    sampling: SamplingConfig,
    eos: &EosConfig,
) -> GenerateResult {
    generate_streaming(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        backend,
        cached_layers,
        layer_range,
        sampling,
        eos,
        |_, _, _| {},
    )
}

/// Streaming multi-token generation. Fires `on_token(id, text, prob)` for
/// every generated token as it's produced, including the first (which
/// comes out of prefill).
///
/// Pipeline:
///
/// 1. GPU prefill: `prefill_q4` populates KV cache for all layers.
/// 2. Decode loop: `decode_token` reads from KV cache, generates one token
///    at a time.
/// 3. Logits: vindex lm_head KNN (size depends on sampling config —
///    [`LMHEAD_TOPK_GREEDY`] for greedy, larger for sampling so the
///    distribution has enough mass to apply temperature / top-p).
/// 4. Pick: greedy → argmax of KNN; sampling → temperature + top-k +
///    top-p over the KNN hits via [`Sampler::sample_from_topk`].
/// 5. Surface form via [`Detokenizer`], which preserves HF leading-space
///    semantics by emitting only the cumulative-decode delta.
/// 6. EOS check via `eos.is_eos(tid, &tok_str)`.
///
/// `on_token` is invoked synchronously inside the decode loop. For UI
/// printing pass `|_, text, _| { print!("{text}"); std::io::Write::flush(&mut std::io::stdout()).ok(); }`.
///
/// Returns `(token_string, probability)` per generated token plus timing.
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming<F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    sampling: SamplingConfig,
    eos: &EosConfig,
    mut on_token: F,
) -> GenerateResult
where
    F: FnMut(u32, &str, f64),
{
    // Backends that don't implement the fused Q4 prefill (today: CpuBackend)
    // delegate to the CPU Q4K per-layer dequant path. It mutates `weights.tensors`
    // per layer and needs &mut; this is the sole reason `generate` itself takes
    // &mut. Metal backends pass straight through and never touch the map here.
    //
    // Per-Layer Embeddings (Gemma 4 E2B `hidden_size_per_layer_input`) are
    // also routed to the CPU path: the `per_layer_input_gate` /
    // `per_layer_projection` / `post_per_layer_input_norm` mechanism is
    // implemented in `q4k_forward.rs` but not in the Metal pipeline, so the
    // residual stream would be missing a per-layer per-position contribution
    // on every layer. Without this routing the model produces multilingual
    // gibberish ("ened retainingcB variations 유doucara…"); on CPU the same
    // weights produce coherent reasoning text.
    let needs_per_layer_embed = weights.arch.has_per_layer_embeddings();
    if !backend_supports_fused_q4_pipeline(backend) || needs_per_layer_embed {
        return generate_via_cpu_q4k(weights, tokenizer, token_ids, max_tokens, index);
    }

    let norm_offset = weights.arch.norm_weight_offset();
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let gate_index: &dyn larql_vindex::GateIndex = index;

    // Build layer descriptors
    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_mmap_ref() {
        (Some(mmap), true)
    } else {
        (gate_index.interleaved_q4_mmap_ref(), false)
    };
    let has_q4k = index.attn_q4k_layer_data(layer_range.start).is_some();
    let has_q8 = index.attn_q8_layer_data(layer_range.start).is_some();

    if !backend.has_q4() || q4_ffn.is_none() {
        let r = crate::layer_graph::predict::predict_honest(
            weights,
            tokenizer,
            token_ids,
            5,
            index,
            backend,
            cached_layers,
            layer_range,
        );
        return GenerateResult {
            tokens: r.predictions.into_iter().take(1).collect(),
            prefill_ms: 0.0,
            decode_ms: vec![],
            stage_timings: StageTimings::default(),
        };
    }

    let q4_ffn_mmap = q4_ffn.unwrap();
    let intermediate = gate_index.num_features(layer_range.start);
    if intermediate == 0 || (!has_q4k && !has_q8) {
        let r = crate::layer_graph::predict::predict_honest(
            weights,
            tokenizer,
            token_ids,
            5,
            index,
            backend,
            cached_layers,
            layer_range,
        );
        return GenerateResult {
            tokens: r.predictions.into_iter().take(1).collect(),
            prefill_ms: 0.0,
            decode_ms: vec![],
            stage_timings: StageTimings::default(),
        };
    }

    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format
        .packed_matrix_bytes(intermediate, hidden)
        .expect("Q4 interleaved FFN format must have packed geometry");

    let num_layers = weights.num_layers;
    let layers = crate::layer_graph::pipeline_layer::build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn_mmap,
        q4_ffn_per_matrix,
        ffn_format,
    );

    let attention = attention_geometry_for_arch_layer(weights, layer_range.start);

    // ── Phase 1: GPU prefill ──
    let prefill_start = std::time::Instant::now();
    backend.reset_kv_cache();

    // Pre-allocate per-layer KV cache for models with asymmetric attention geometry
    // (e.g. Gemma 4 26B: sliding layers use 8×256, global layers use 2×512).
    // Without this, the lazy uniform allocation uses the first layer's dims for all layers,
    // causing global layers to read/write off the end of under-sized KV buffers.
    {
        let kv_shapes = kv_cache_shapes_for_arch(weights);
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
    }
    let seq_len = token_ids.len();

    let h_embed = crate::forward::embed_tokens_pub(weights, token_ids);
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();

    let softcap_val = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm_val = arch.attn_q_norm_key(0).is_some();

    // For per-layer Q4K expert format: prefill using token-by-token GPU expert dispatch.
    // The standard prefill_q4 path calls cpu_moe_forward which expects BF16 blobs;
    // that would panic on Q4K expert bytes. Token-by-token is correct and builds the
    // KV cache identically to the batched prefill.
    let h_vec = if weights.has_per_layer_ffn() {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if let Some(metal) = backend
                .as_any()
                .downcast_ref::<larql_compute::metal::MetalBackend>()
            {
                let norm_eps = weights.arch.norm_eps();
                let mut last_h = vec![0.0f32; hidden];
                for pos in 0..seq_len {
                    let x_pos: Vec<f32> = x[pos * hidden..(pos + 1) * hidden].to_vec();
                    last_h = metal
                        .decode_token_q4k_moe(
                            &layers,
                            &x_pos,
                            hidden,
                            intermediate,
                            attention.q_dim,
                            attention.kv_dim,
                            attention.num_q_heads,
                            attention.num_kv_heads,
                            attention.head_dim,
                            attention.rope_base,
                            norm_eps,
                            |layer_idx, expert_idx| {
                                weights.get_layer_entry_bytes(layer_idx, expert_idx)
                            },
                        )
                        .unwrap_or_else(|| vec![0.0f32; hidden]);
                }
                // Return only the last position (same shape as batched prefill output)
                let mut out = vec![0.0f32; seq_len * hidden];
                out[(seq_len - 1) * hidden..].copy_from_slice(&last_h);
                out
            } else {
                return GenerateResult {
                    tokens: Vec::new(),
                    prefill_ms: 0.0,
                    decode_ms: Vec::new(),
                    stage_timings: StageTimings::default(),
                };
            }
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            return GenerateResult {
                tokens: Vec::new(),
                prefill_ms: 0.0,
                decode_ms: Vec::new(),
                stage_timings: StageTimings::default(),
            };
        }
    } else {
        match backend.prefill_q4(
            &layers,
            &x,
            hidden,
            intermediate,
            attention.q_dim,
            attention.kv_dim,
            seq_len,
            attention.num_q_heads,
            attention.num_kv_heads,
            attention.head_dim,
            attention.rope_base,
            qk_norm_val,
            softcap_val,
        ) {
            Some(v) => v,
            None => {
                return GenerateResult {
                    tokens: Vec::new(),
                    prefill_ms: 0.0,
                    decode_ms: Vec::new(),
                    stage_timings: StageTimings::default(),
                }
            }
        }
    };

    let h_metal = ndarray::Array2::from_shape_vec((seq_len, hidden), h_vec.clone())
        .unwrap_or_else(|_| h_embed.clone());

    let compare = std::env::var("LARQL_METAL_COMPARE_CPU").is_ok();

    let h = h_metal;
    let h_1d = {
        let h_final =
            crate::forward::apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);
        h_final.row(seq_len - 1).to_owned()
    };

    // CPU-vs-Metal comparison mode (LARQL_METAL_COMPARE_CPU=1). Runs the
    // known-correct `predict_q4k` CPU path on the same prompt and diffs
    // the top-5 predicted tokens against the Metal path. Purpose: isolate
    // whether wrong-token output is from the compute path or from the
    // lm_head / logits-sampling layer.
    if compare {
        let metal_hits_vindex = index.lm_head_knn_backend(&h_1d, 5, backend);
        let metal_hits_cpu_lm = cpu_lm_head_topk(weights, &h_1d, 5);
        let as_toks = |hits: &[(u32, f32)]| -> Vec<String> {
            hits.iter()
                .map(|(t, _)| {
                    tokenizer
                        .decode(&[*t], true)
                        .unwrap_or_default()
                        .trim()
                        .to_string()
                })
                .collect()
        };
        eprintln!(
            "[compare] metal final h_1d:  len={}  nan={}  inf={}  max_abs={:.3e}",
            h_1d.len(),
            h_1d.iter().filter(|v| v.is_nan()).count(),
            h_1d.iter().filter(|v| v.is_infinite()).count(),
            h_1d.iter()
                .map(|v| v.abs())
                .filter(|v| v.is_finite())
                .fold(0.0f32, f32::max)
        );
        eprintln!(
            "[compare] metal top-5 via vindex-KNN:    {:?}",
            as_toks(&metal_hits_vindex)
        );
        eprintln!(
            "[compare] metal top-5 via CPU lm_head:   {:?}",
            as_toks(&metal_hits_cpu_lm)
        );

        eprintln!("[compare] (run `larql walk --predict` (no --metal) for CPU reference tokens)");
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // Sample first token
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);

    let mut sampler = Sampler::new(sampling);
    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(token_ids);

    // Running list of token ids the model has emitted so far. Fed
    // into the sampler's repetition-penalty path; empty on the first
    // pick (no history yet).
    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);

    let knn_k = lmhead_k_for_sampling(&sampling);
    let first_hits = lm_head_topk(index, weights, &h_1d, knn_k, backend);
    let first_pick = sampler.sample_from_topk_with_history(&first_hits, &generated_ids);
    if let Some(picked_id) = first_pick {
        // Detokenizer.push emits the cumulative-decode delta — handles HF
        // leading-space (`▁`) correctly across SP and BPE tokenizers.
        let tok_str = detok.push(picked_id);
        let score = first_hits
            .iter()
            .find(|(t, _)| *t == picked_id)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let prob = crate::layer_graph::logits::softmax_prob(
            score,
            &first_hits,
            weights.arch.logits_scaling(),
            weights.arch.final_logit_softcapping(),
        );
        on_token(picked_id, &tok_str, prob);
        generated_ids.push(picked_id);
        tokens.push((tok_str, prob));
    }

    // ── Phase 2: GPU decode loop ──
    let mut current_token_id = first_pick.unwrap_or(0);

    // Per-stage decode profiling. Set LARQL_PROFILE_DECODE=1 to log a
    // one-line per-step breakdown of embed / GPU forward / final norm /
    // lm_head / detokenize, plus a summary at the end.
    let profile = std::env::var("LARQL_PROFILE_DECODE").is_ok();
    let profile_split = std::env::var("LARQL_PROFILE_SPLIT").is_ok();
    let mut t_embed = 0.0f64;
    let mut t_gpu = 0.0f64;
    let mut t_gate_up = 0.0f64;
    let mut t_down = 0.0f64;
    let mut t_norm = 0.0f64;
    let mut t_lmhead = 0.0f64;
    let mut t_detok = 0.0f64;

    for _step in 1..max_tokens {
        let decode_start = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let h_tok = crate::forward::embed_tokens_pub(weights, &[current_token_id]);
        let x_dec: Vec<f32> = h_tok.row(0).to_vec();
        let embed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        if profile && _step <= 2 {
            let x_nan = x_dec.iter().filter(|v| v.is_nan()).count();
            let x_max = x_dec
                .iter()
                .map(|v| v.abs())
                .filter(|v| v.is_finite())
                .fold(0.0f32, f32::max);
            eprintln!(
                "[profile] step={} input tok={} x_dec: len={} nan={} max_abs={:.3e}",
                _step,
                current_token_id,
                x_dec.len(),
                x_nan,
                x_max,
            );
        }

        let t1 = std::time::Instant::now();
        let result = if profile_split && _step == 2 {
            // Step 2 is post-JIT warm — run split profiling once and print.
            let (r, _ta, _tgu, _td) = backend.decode_token_split_profile(
                &layers,
                &x_dec,
                hidden,
                intermediate,
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
            );
            r
        } else if {
            let v = weights.has_per_layer_ffn();
            v
        } {
            // Per-layer Q4_K expert format: route on CPU, dispatch expert FFNs on GPU.
            // Eliminates the BF16 dequant + CPU BLAS path and the per-layer commit
            // overhead that was doing nothing useful for MoE experts.
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if let Some(metal) = backend
                .as_any()
                .downcast_ref::<larql_compute::metal::MetalBackend>()
            {
                let norm_eps = weights.arch.norm_eps();
                metal.decode_token_q4k_moe(
                    &layers,
                    &x_dec,
                    hidden,
                    intermediate,
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
                    norm_eps,
                    |layer_idx, expert_idx| weights.get_layer_entry_bytes(layer_idx, expert_idx),
                )
            } else {
                backend.decode_token(
                    &layers,
                    &x_dec,
                    hidden,
                    intermediate,
                    attention.q_dim,
                    attention.kv_dim,
                    attention.num_q_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                    attention.rope_base,
                )
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            backend.decode_token(
                &layers,
                &x_dec,
                hidden,
                intermediate,
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
            )
        } else {
            backend.decode_token(
                &layers,
                &x_dec,
                hidden,
                intermediate,
                attention.q_dim,
                attention.kv_dim,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
                attention.rope_base,
            )
        };
        let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

        if profile && _step <= 2 {
            match &result {
                Some(h) => {
                    let h_nan = h.iter().filter(|v| v.is_nan()).count();
                    let h_max = h
                        .iter()
                        .map(|v| v.abs())
                        .filter(|v| v.is_finite())
                        .fold(0.0f32, f32::max);
                    eprintln!(
                        "[profile] step={} decode_token h_out: len={} nan={} max_abs={:.3e}",
                        _step,
                        h.len(),
                        h_nan,
                        h_max,
                    );
                }
                None => eprintln!("[profile] step={} decode_token returned None", _step),
            }
        }

        if let Some(h_out) = result {
            let t2 = std::time::Instant::now();
            let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out).unwrap();
            let h_final = crate::forward::apply_norm(
                weights,
                &h_arr,
                weights.arch.final_norm_key(),
                norm_offset,
            );
            let h_1d = h_final.row(0).to_owned();
            let norm_ms = t2.elapsed().as_secs_f64() * 1000.0;

            let t3 = std::time::Instant::now();
            let hits = lm_head_topk(index, weights, &h_1d, knn_k, backend);
            let lmhead_ms = t3.elapsed().as_secs_f64() * 1000.0;
            if profile && _step <= 2 {
                let h_nan = h_1d.iter().filter(|v| v.is_nan()).count();
                let h_inf = h_1d.iter().filter(|v| v.is_infinite()).count();
                let h_max = h_1d
                    .iter()
                    .map(|v| v.abs())
                    .filter(|v| v.is_finite())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "[profile] step={} h_1d: len={} nan={} inf={} max_abs={:.3e}  hits.len()={}",
                    _step,
                    h_1d.len(),
                    h_nan,
                    h_inf,
                    h_max,
                    hits.len(),
                );
            }

            let step_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
            decode_ms.push(step_ms);

            if let Some(picked_id) = sampler.sample_from_topk_with_history(&hits, &generated_ids) {
                let t4 = std::time::Instant::now();
                let tok_str = detok.push(picked_id);
                let detok_ms = t4.elapsed().as_secs_f64() * 1000.0;
                let score = hits
                    .iter()
                    .find(|(t, _)| *t == picked_id)
                    .map(|(_, s)| *s)
                    .unwrap_or(0.0);
                let prob = crate::layer_graph::logits::softmax_prob(
                    score,
                    &hits,
                    weights.arch.logits_scaling(),
                    weights.arch.final_logit_softcapping(),
                );
                let is_eos = eos.is_eos_with_tokenizer(picked_id, &tok_str, tokenizer);
                if profile {
                    eprintln!(
                        "[profile] step={} total={:.1}ms  embed={:.2}  gpu={:.1}  norm={:.2}  lm_head={:.1}  detok={:.2}",
                        _step, step_ms, embed_ms, gpu_ms, norm_ms, lmhead_ms, detok_ms,
                    );
                }
                t_embed += embed_ms;
                t_gpu += gpu_ms;
                #[cfg(all(feature = "metal", target_os = "macos"))]
                if profile_split {
                    if let Some(pt) = larql_compute::metal_take_last_split_timings() {
                        t_gate_up += pt.gate_up_ms;
                        t_down += pt.down_ms;
                    }
                }
                t_norm += norm_ms;
                t_lmhead += lmhead_ms;
                t_detok += detok_ms;
                on_token(picked_id, &tok_str, prob);
                tokens.push((tok_str, prob));
                generated_ids.push(picked_id);
                current_token_id = picked_id;
                if is_eos {
                    break;
                }
            } else {
                if profile {
                    eprintln!("[profile] step={} — lm_head returned empty; break", _step);
                }
                break;
            }
        } else {
            // GPU returned None mid-decode. The generate() function routes
            // non-fused-Q4 backends (today: CPU) to a full CPU Q4K path at
            // the top, so this branch can only fire when a GPU backend that
            // passed `backend_supports_fused_q4_pipeline` subsequently fails
            // a single decode step. Treat as early-stop rather than re-run
            // the O(N²) CPU path mid-loop without a kept id list.
            if profile {
                eprintln!(
                    "[profile] step={} — GPU decode returned None; stopping generation",
                    _step
                );
            }
            break;
        }
    }

    if profile && !decode_ms.is_empty() {
        let n = decode_ms.len() as f64;
        eprintln!(
            "[profile] SUMMARY over {} steps: embed={:.2}ms  gpu={:.1}ms  norm={:.2}ms  lm_head={:.1}ms  detok={:.2}ms  total={:.1}ms",
            decode_ms.len(),
            t_embed / n, t_gpu / n, t_norm / n, t_lmhead / n, t_detok / n,
            decode_ms.iter().sum::<f64>() / n,
        );
    }

    // Per-stage totals across all successful steps (not vec-per-step to
    // keep the struct tiny — the `larql bench` harness averages these
    // against `decode_ms.len()`).
    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings {
            embed_ms_total: t_embed,
            gpu_ms_total: t_gpu,
            gate_up_ms_total: t_gate_up,
            down_ms_total: t_down,
            norm_ms_total: t_norm,
            lm_head_ms_total: t_lmhead,
            detok_ms_total: t_detok,
        },
    }
}

/// Constrained variant of [`generate`] for grammar-controlled decoding.
///
/// Differs from `generate` in two places only:
///
///   1. The LM-head step uses a **dense** vocabulary score vector
///      ([`backend_lm_head_scores`]) rather than the sparse vindex KNN.
///      Required because an arbitrary mask can disqualify tokens that
///      would otherwise have fallen outside the top-K.
///   2. After scoring, `mask_fn(generated_ids, &mut logits)` runs and the
///      next token is the masked argmax.
///
/// Per-token cost is slightly higher than unconstrained `generate` (full
/// 2.68 GB tied LM-head gemv vs. KNN over the 5-NN partial), but on Metal
/// it's still ~3-5 ms — acceptable for grammar-constrained dispatch.
///
/// Stops on EOS / common end-of-turn markers or when `max_tokens` is hit.
#[allow(clippy::too_many_arguments)]
pub fn generate_constrained<M>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    mask_fn: M,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    generate_constrained_streaming(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        backend,
        cached_layers,
        layer_range,
        mask_fn,
        |_, _, _| {},
    )
}

/// Streaming variant of [`generate_constrained`] — fires
/// `on_token(id, text, prob)` after each masked-argmax pick so SSE
/// callers can flush JSON / structured-output chunks as they're
/// produced. Greedy under the mask; for sampling under mask see
/// [`generate_constrained_streaming_sampled`].
#[allow(clippy::too_many_arguments)]
pub fn generate_constrained_streaming<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    mask_fn: M,
    on_token: F,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    generate_constrained_streaming_sampled(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        backend,
        cached_layers,
        layer_range,
        mask_fn,
        on_token,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
    )
}

/// Streaming + sampling-aware constrained decode. Drives token
/// selection through the supplied [`SamplingConfig`] (temperature,
/// top_p, top_k, seed, repetition penalties) over the *masked* logits.
/// Pass `SamplingConfig::greedy()` for the existing argmax behaviour
/// (which is what most JSON / tools modes want today).
///
/// `eos` is consulted on top of the built-in end-of-turn detection so
/// the caller can extend the stop set with user-supplied stop strings.
#[allow(clippy::too_many_arguments)]
pub fn generate_constrained_streaming_sampled<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    mut mask_fn: M,
    mut on_token: F,
    sampling: SamplingConfig,
    eos: &EosConfig,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    let _ = eos; // built-in end-of-turn check still primary; eos extension is a follow-up
    let mut sampler = Sampler::new(sampling);
    // Same PLE delegation as `generate_streaming` — the Metal pipeline
    // doesn't implement Gemma 4 E2B's per-layer-input gate.
    let needs_per_layer_embed = weights.arch.has_per_layer_embeddings();
    if !backend_supports_fused_q4_pipeline(backend) || needs_per_layer_embed {
        return generate_constrained_via_cpu_q4k_streaming_sampled(
            weights, tokenizer, token_ids, max_tokens, index, mask_fn, on_token, sampling,
        );
    }

    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let hidden = weights.hidden_size;
    let gate_index: &dyn larql_vindex::GateIndex = index;

    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_mmap_ref() {
        (Some(mmap), true)
    } else {
        (gate_index.interleaved_q4_mmap_ref(), false)
    };
    let has_q4k = index.attn_q4k_layer_data(layer_range.start).is_some();
    let has_q8 = index.attn_q8_layer_data(layer_range.start).is_some();

    // Constrained mode requires the GPU prefill + Q4 path to be available.
    // Fall back to the unconstrained dense single-token predict if it isn't —
    // the mask still applies to that one token via pick_next_token_masked.
    if !backend.has_q4() || q4_ffn.is_none() {
        // Dense single-token prediction with mask.
        let r = crate::layer_graph::predict::predict_honest(
            weights,
            tokenizer,
            token_ids,
            5,
            index,
            backend,
            cached_layers,
            layer_range,
        );
        return GenerateResult {
            tokens: r.predictions.into_iter().take(1).collect(),
            prefill_ms: 0.0,
            decode_ms: vec![],
            stage_timings: StageTimings::default(),
        };
    }
    let q4_ffn_mmap = q4_ffn.unwrap();
    let intermediate = gate_index.num_features(layer_range.start);
    if intermediate == 0 || (!has_q4k && !has_q8) {
        let r = crate::layer_graph::predict::predict_honest(
            weights,
            tokenizer,
            token_ids,
            5,
            index,
            backend,
            cached_layers,
            layer_range,
        );
        return GenerateResult {
            tokens: r.predictions.into_iter().take(1).collect(),
            prefill_ms: 0.0,
            decode_ms: vec![],
            stage_timings: StageTimings::default(),
        };
    }

    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format
        .packed_matrix_bytes(intermediate, hidden)
        .expect("Q4 interleaved FFN format must have packed geometry");

    let num_layers = weights.num_layers;
    let layers = crate::layer_graph::pipeline_layer::build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn_mmap,
        q4_ffn_per_matrix,
        ffn_format,
    );

    let attention = attention_geometry_for_arch_layer(weights, layer_range.start);

    // ── Phase 1: GPU prefill ──
    let prefill_start = std::time::Instant::now();
    backend.reset_kv_cache();
    {
        let kv_shapes = kv_cache_shapes_for_arch(weights);
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
    }
    let seq_len = token_ids.len();
    let h_embed = crate::forward::embed_tokens_pub(weights, token_ids);
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();
    let softcap_val = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm_val = arch.attn_q_norm_key(0).is_some();

    // Constrained-path prefill: CPU-only backends delegate at the top of the
    // function, so `prefill_q4` should succeed. If it returns None, bail out
    // with no tokens rather than taking the removed dense-tensor panic path.
    let h_vec = match backend.prefill_q4(
        &layers,
        &x,
        hidden,
        intermediate,
        attention.q_dim,
        attention.kv_dim,
        seq_len,
        attention.num_q_heads,
        attention.num_kv_heads,
        attention.head_dim,
        attention.rope_base,
        qk_norm_val,
        softcap_val,
    ) {
        Some(v) => v,
        None => {
            return GenerateResult {
                tokens: Vec::new(),
                prefill_ms: 0.0,
                decode_ms: Vec::new(),
                stage_timings: StageTimings::default(),
            };
        }
    };

    let h_metal = ndarray::Array2::from_shape_vec((seq_len, hidden), h_vec.clone())
        .unwrap_or_else(|_| h_embed.clone());
    let h_1d = {
        let h_final = crate::forward::apply_norm(
            weights,
            &h_metal,
            weights.arch.final_norm_key(),
            norm_offset,
        );
        h_final.row(seq_len - 1).to_owned()
    };
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // ── First token: dense LM-head + mask + argmax ──
    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    let first = pick_next_token_masked_sampled(
        weights,
        &h_1d,
        &generated,
        backend,
        &mut mask_fn,
        &mut sampler,
    );
    let mut current_token_id = match first {
        Some((tid, _)) => {
            let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default();
            let is_eos = crate::vindex::is_end_of_turn(tok_str.trim());
            on_token(tid, &tok_str, 1.0);
            tokens.push((tok_str, 1.0));
            generated.push(tid);
            if is_eos {
                return GenerateResult {
                    tokens,
                    prefill_ms,
                    decode_ms,
                    stage_timings: StageTimings::default(),
                };
            }
            tid
        }
        None => {
            return GenerateResult {
                tokens,
                prefill_ms,
                decode_ms,
                stage_timings: StageTimings::default(),
            }
        }
    };

    // ── Phase 2: GPU decode loop ──
    for _step in 1..max_tokens {
        let decode_start = std::time::Instant::now();

        let h_tok = crate::forward::embed_tokens_pub(weights, &[current_token_id]);
        let x_dec: Vec<f32> = h_tok.row(0).to_vec();

        let result = backend.decode_token(
            &layers,
            &x_dec,
            hidden,
            intermediate,
            attention.q_dim,
            attention.kv_dim,
            attention.num_q_heads,
            attention.num_kv_heads,
            attention.head_dim,
            attention.rope_base,
        );

        let h_1d = if let Some(h_out) = result {
            let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out).unwrap();
            let h_final = crate::forward::apply_norm(
                weights,
                &h_arr,
                weights.arch.final_norm_key(),
                norm_offset,
            );
            h_final.row(0).to_owned()
        } else {
            // GPU returned None mid-decode. Stop rather than re-run a long
            // O(N²) CPU Q4K path (CPU-only backends already delegate at the
            // top of the function, so this is reachable only via a GPU fault).
            break;
        };

        let pick = pick_next_token_masked_sampled(
            weights,
            &h_1d,
            &generated,
            backend,
            &mut mask_fn,
            &mut sampler,
        );
        decode_ms.push(decode_start.elapsed().as_secs_f64() * 1000.0);

        match pick {
            Some((tid, _)) => {
                let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default();
                let is_eos = crate::vindex::is_end_of_turn(tok_str.trim());
                on_token(tid, &tok_str, 1.0);
                tokens.push((tok_str, 1.0));
                generated.push(tid);
                current_token_id = tid;
                if is_eos {
                    break;
                }
            }
            None => break,
        }
    }

    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings::default(),
    }
}
