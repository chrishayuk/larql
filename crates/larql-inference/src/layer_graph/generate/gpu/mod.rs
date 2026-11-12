//! Metal GPU generate paths — fused prefill + KV-cached decode loop.
//!
//! Public surface lives here; the heavy lifting is split into:
//! - [`prefill`] — three-branch GPU prefill (PLE / Q4_K MoE / standard)
//! - [`decode_loop`] — per-step embed → forward → norm → lm_head → sample
//! - [`sampling_step`] — sample + detok + EOS-check helper shared by
//!   first-token and decode-loop branches
//! - [`forced_logits`] — Shannon-codec primitive (independent of sampling)

mod decode_loop;
mod forced_logits;
mod prefill;
mod sampling_step;

pub use forced_logits::{stream_forced_full_logits, ForcedLogitsResult};

/// PLE upload callback — invoked once per token before the per-token GPU
/// forward pass on per-layer-embedding models (e.g. Gemma 4 E2B). `None`
/// elsewhere. See [`crate::forward::ple`] for the per-layer-input
/// precompute that feeds this. Aliased to keep clippy's
/// `type_complexity` lint happy at the call sites.
pub(super) type UploadPleFn<'a> = &'a dyn Fn(u32, &[f32]);

use super::cpu::{backend_supports_fused_q4_pipeline, generate_via_cpu_q4k};
use super::detok::Detokenizer;
use super::eos::EosConfig;
use super::gpu_setup::{
    build_gpu_decode_setup, ensure_prompt_fits, reset_and_preallocate_kv_cache,
};
use super::lm_head::{cpu_lm_head_topk, lm_head_topk_with_policy};
use super::policy::GenerationRuntimeConfig;
use super::sampling::{Sampler, SamplingConfig};
use super::types::{GenerateError, GenerateResult, StageTimings};
use crate::layer_graph::CachedLayerGraph;
use crate::model::ModelWeights;
use larql_compute::prelude::*;

use sampling_step::sample_and_emit;

/// LM-head top-K size when running greedy decode.
const LMHEAD_TOPK_GREEDY: usize = 5;
/// LM-head top-K minimum when sampling. Larger K gives the sampler enough
/// distribution mass to apply temperature / top-p meaningfully without
/// paying for a full-vocab gemv.
const LMHEAD_TOPK_SAMPLING_MIN: usize = 64;

fn lmhead_k_for_sampling(cfg: &SamplingConfig) -> usize {
    if cfg.is_greedy() {
        LMHEAD_TOPK_GREEDY
    } else {
        cfg.top_k.unwrap_or(0).max(LMHEAD_TOPK_SAMPLING_MIN)
    }
}

/// Greedy multi-token generation. Thin wrapper over
/// [`generate_with_sampling`] with [`SamplingConfig::greedy`] and
/// [`EosConfig::builtin`].
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

/// [`generate`] with the expert-bank authority supplied by a routed
/// VINDEX3 container — the composed Metal serve path. Everything except
/// the routed banks is identical to [`generate`].
#[allow(clippy::too_many_arguments)]
pub fn generate_routed(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    routed: &crate::ffn::ContainerRoutedBackend,
) -> GenerateResult {
    let _ = cached_layers;
    generate_streaming(
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
        |_, _, _| {},
        Some(routed),
    )
}

/// Fallible variant of [`generate`].
#[allow(clippy::too_many_arguments)]
pub fn try_generate(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
) -> Result<GenerateResult, GenerateError> {
    generate(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        index,
        backend,
        cached_layers,
        layer_range,
    )
    .into_result()
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
        None,
    )
}

/// Fallible variant of [`generate_with_sampling`].
#[allow(clippy::too_many_arguments)]
pub fn try_generate_with_sampling(
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
) -> Result<GenerateResult, GenerateError> {
    generate_with_sampling(
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
    )
    .into_result()
}

/// Streaming multi-token generation. Fires `on_token(id, text, prob)` for
/// every generated token as it's produced, including the first (which
/// comes out of prefill).
///
/// Pipeline:
///
/// 1. GPU prefill: [`prefill::prefill_for_streaming`] populates KV cache.
/// 2. First-token sample: greedy → argmax of KNN; sampling → temperature +
///    top-k + top-p over the KNN hits via [`Sampler::sample_from_topk`].
/// 3. Decode loop: [`decode_loop::run_decode_loop`] runs steps 1..max_tokens.
/// 4. Logits: vindex lm_head KNN (size depends on sampling config —
///    [`LMHEAD_TOPK_GREEDY`] for greedy, larger for sampling so the
///    distribution has enough mass to apply temperature / top-p).
/// 5. Surface form via [`Detokenizer`], which preserves HF leading-space
///    semantics by emitting only the cumulative-decode delta.
/// 6. EOS check via [`EosConfig::is_eos_with_tokenizer`].
///
/// `on_token` is invoked synchronously inside the decode loop. For UI
/// printing pass `|_, text, _| { print!("{text}"); std::io::Write::flush(&mut std::io::stdout()).ok(); }`.
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
    // Expert-bank authority override: routed container if supplied.
    routed: Option<&crate::ffn::ContainerRoutedBackend>,
) -> GenerateResult
where
    F: FnMut(u32, &str, f64),
{
    if max_tokens == 0 {
        return GenerateResult::empty_success();
    }

    // Backends that don't implement the fused Q4 prefill (today: CpuBackend)
    // delegate to the CPU Q4K per-layer dequant path. It mutates `weights.tensors`
    // per layer and needs &mut; this is the sole reason `generate` itself takes
    // &mut. Metal backends pass straight through and never touch the map here.
    //
    // Per-Layer Embeddings (Gemma 4 E2B `hidden_size_per_layer_input`):
    // when `LARQL_METAL_PLE=1`, route through the Metal decode loop with
    // PLE applied per layer (see `metal/decode/encode_ple.rs`).  Otherwise
    // fall back to the CPU `kquant_forward.rs` path — without PLE applied,
    // the residual stream would be missing a per-layer per-position
    // contribution on every layer and the model produces multilingual
    // gibberish; the CPU path produces coherent reasoning text.
    let needs_per_layer_embed = weights.arch.has_per_layer_embeddings();
    let metal_ple_enabled = needs_per_layer_embed
        && larql_compute::options::env_opt_in(larql_compute::options::ENV_METAL_PLE);
    if !backend_supports_fused_q4_pipeline(backend) || (needs_per_layer_embed && !metal_ple_enabled)
    {
        // The CPU fallback runs the spine's banks; falling into it with a
        // routed override would silently measure or serve the wrong bytes.
        if routed.is_some() {
            return GenerateResult::empty_error(GenerateError::missing_weights(
                "the routed expert-bank override requires the fused GPU pipeline; \
                 this backend would silently serve the spine's banks instead",
            ));
        }
        return generate_via_cpu_q4k(
            weights,
            tokenizer,
            token_ids,
            max_tokens,
            index,
            eos,
            sampling,
            &mut on_token,
        );
    }

    // Register each packed weight mmap for zero-copy GPU aliasing — the
    // MoE decode binds per-expert byte offsets into these regions instead
    // of memcpy-staging ~top_k × 22 MB per layer per token. Repeat
    // registration is a cheap no-op (deduped by base address); the CPU
    // backend's default impl does nothing.
    for mmap in weights.packed_mmaps.values() {
        backend.register_weight_region(&mmap[..]);
    }
    // The routed container's banks are weight regions of the same standing:
    // when the expert-bank authority is overridden, the override's bytes are
    // what the per-expert offsets bind into, so they must be registered
    // before the residency set is sealed.
    if let Some(routed) = routed {
        for region in routed.weight_regions() {
            backend.register_weight_region(region);
        }
    }
    // Every region is registered; let the backend prepare them once. Metal
    // uses this to declare an explicit residency set instead of paying
    // per-command-buffer residency bookkeeping on a ~697 MB allocation 24
    // times per token.
    backend.seal_weight_regions();

    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let setup = match build_gpu_decode_setup(weights, index, backend, layer_range, false, routed) {
        Ok(setup) => setup,
        Err(err) => {
            let _ = cached_layers;
            return GenerateResult::empty_error(err);
        }
    };
    let layers = setup.layers;
    let hidden = setup.hidden;
    let intermediate = setup.intermediate;

    // ── Phase 1: GPU prefill ──
    let prefill_start = std::time::Instant::now();
    let seq_len = token_ids.len();
    if let Err(err) = ensure_prompt_fits(seq_len) {
        return GenerateResult::empty_error(err);
    }

    // Pre-allocate per-layer KV cache for models with asymmetric attention geometry
    // (e.g. Gemma 4 26B: sliding layers use 8×256, global layers use 2×512).
    reset_and_preallocate_kv_cache(weights, backend);

    let h_embed = crate::forward::embed_tokens_pub(weights, token_ids);
    let x: Vec<f32> = crate::row_major::row_major(&h_embed).into_owned();

    let softcap_val = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm_val = arch.attn_q_norm_key(0).is_some();

    // PLE setup: probe the backend for `PerLayerEmbeddings` instead of
    // downcasting to a concrete type. The closure dispatches through
    // `ComputeBackend::prepare_ple_inputs`; CPU and any non-PLE backend
    // get the trait's no-op default and the closure stays unused (we
    // gate construction on `use_ple`).
    let use_ple =
        metal_ple_enabled && backend.supports(larql_compute::Capability::PerLayerEmbeddings);
    let ple_dim = if use_ple {
        weights.arch.per_layer_embed_dim()
    } else {
        0
    };
    // Helper closure: precompute the per-layer-input table for one token
    // and hand it to the backend's PLE upload. Mirrors
    // `precompute_per_layer_inputs` for a single position.
    let upload_ple_closure = |token_id: u32, embed_row: &[f32]| {
        let embed_arr = ndarray::Array2::from_shape_vec((1, hidden), embed_row.to_vec())
            .unwrap_or_else(|_| ndarray::Array2::<f32>::zeros((1, hidden)));
        let per_layer_inputs =
            crate::forward::ple::precompute_per_layer_inputs(weights, &embed_arr, &[token_id]);
        let num_layers = weights.num_layers;
        let mut flat: Vec<f32> = Vec::with_capacity(num_layers * ple_dim);
        for layer_arr in &per_layer_inputs {
            for v in layer_arr.row(0).iter() {
                flat.push(*v);
            }
        }
        backend.prepare_ple_inputs(&flat, num_layers, ple_dim);
    };
    let upload_ple: Option<UploadPleFn> = if use_ple {
        Some(&upload_ple_closure)
    } else {
        None
    };

    let h_vec = match prefill::prefill_for_streaming(
        weights,
        backend,
        &layers,
        hidden,
        intermediate,
        token_ids,
        &x,
        qk_norm_val,
        softcap_val,
        upload_ple,
    ) {
        Ok(v) => v,
        Err(err) => return GenerateResult::empty_error(err),
    };

    let h_metal = ndarray::Array2::from_shape_vec((seq_len, hidden), h_vec.clone())
        .unwrap_or_else(|_| h_embed.clone());

    let runtime = GenerationRuntimeConfig::from_env();

    let h = h_metal;
    let h_1d = {
        let h_final =
            crate::forward::apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);
        h_final.row(seq_len - 1).to_owned()
    };

    // CPU-vs-Metal comparison mode (LARQL_METAL_COMPARE_CPU=1). Runs the
    // known-correct `predict_kquant` CPU path on the same prompt and diffs
    // the top-5 predicted tokens against the Metal path.
    if runtime.compare_cpu {
        diag_compare_cpu_topk(tokenizer, weights, index, backend, &h_1d);
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // Sample first token from prefill output.
    let mut tokens: Vec<(String, f64)> =
        Vec::with_capacity(crate::generation_capacity::generation_capacity(max_tokens));
    let mut sampler = Sampler::new(sampling);
    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(token_ids);

    // Running list of token ids the model has emitted so far. Fed into
    // the sampler's repetition-penalty path; empty on the first pick.
    let mut generated_ids: Vec<u32> =
        Vec::with_capacity(crate::generation_capacity::generation_capacity(max_tokens));

    let knn_k = lmhead_k_for_sampling(&sampling);
    let first_hits =
        lm_head_topk_with_policy(index, weights, &h_1d, knn_k, backend, &runtime.lm_head);
    let first_picked = sample_and_emit(
        &mut sampler,
        &mut detok,
        tokenizer,
        weights,
        eos,
        &first_hits,
        &generated_ids,
        &mut on_token,
    );
    let current_token_id = match first_picked {
        Some(picked) => {
            generated_ids.push(picked.id);
            tokens.push((picked.text, picked.prob));
            if picked.is_eos {
                return GenerateResult {
                    tokens,
                    prefill_ms,
                    decode_ms: Vec::new(),
                    stage_timings: StageTimings::default(),
                    error: None,
                };
            }
            picked.id
        }
        None => 0,
    };

    // ── Phase 2: GPU decode loop ──
    let outcome = decode_loop::run_decode_loop(
        weights,
        tokenizer,
        index,
        backend,
        &layers,
        hidden,
        intermediate,
        norm_offset,
        knn_k,
        &runtime,
        &mut sampler,
        &mut detok,
        eos,
        &mut generated_ids,
        current_token_id,
        max_tokens,
        upload_ple,
        &mut on_token,
    );

    tokens.extend(outcome.tokens);

    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms: outcome.decode_ms,
        stage_timings: StageTimings {
            embed_ms_total: outcome.t_embed,
            gpu_ms_total: outcome.t_gpu,
            gpu_only_ms_total: outcome.t_gpu_only,
            attn_ms_total: outcome.t_attn,
            cmd_buffers_total: outcome.n_cmd_buffers,
            cpu_fwd_ms_total: 0.0,
            gate_up_ms_total: outcome.t_gate_up,
            down_ms_total: outcome.t_down,
            norm_ms_total: outcome.t_norm,
            lm_head_ms_total: outcome.t_lmhead,
            detok_ms_total: outcome.t_detok,
            dequant_ms_total: 0.0,
        },
        error: None,
    }
}

/// Fallible variant of [`generate_streaming`].
#[allow(clippy::too_many_arguments)]
pub fn try_generate_streaming<F>(
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
    on_token: F,
) -> Result<GenerateResult, GenerateError>
where
    F: FnMut(u32, &str, f64),
{
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
        on_token,
        None,
    )
    .into_result()
}

/// Diagnostic: print top-5 token comparison between the GPU path's
/// vindex-KNN lm_head and the CPU dequant lm_head. Triggered by
/// `LARQL_METAL_COMPARE_CPU=1` to isolate whether wrong-token output
/// is from the compute path or from lm_head/sampling.
fn diag_compare_cpu_topk(
    tokenizer: &tokenizers::Tokenizer,
    weights: &ModelWeights,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    h_1d: &ndarray::Array1<f32>,
) {
    let metal_hits_vindex = index.lm_head_knn_backend(h_1d, 5, backend);
    let metal_hits_cpu_lm = cpu_lm_head_topk(weights, h_1d, 5);
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

#[cfg(test)]
mod tests;
