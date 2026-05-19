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
        && std::env::var("LARQL_METAL_PLE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    if !backend_supports_fused_q4_pipeline(backend) || (needs_per_layer_embed && !metal_ple_enabled)
    {
        return generate_via_cpu_q4k(weights, tokenizer, token_ids, max_tokens, index, eos);
    }

    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let setup = match build_gpu_decode_setup(weights, index, backend, layer_range, false) {
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
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();

    let softcap_val = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm_val = arch.attn_q_norm_key(0).is_some();

    // PLE setup: downcast the backend to MetalBackend so we can call
    // `prepare_ple_inputs` between decode steps. None unless the env flag
    // is set AND backend is Metal. The `metal_ple` machinery is only
    // meaningful with `--features metal`; without that feature the
    // backend cannot be a `MetalBackend` and the PLE path is unreachable.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let metal_ple = if metal_ple_enabled {
        backend
            .as_any()
            .downcast_ref::<larql_compute_metal::MetalBackend>()
    } else {
        None
    };
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let metal_ple: Option<()> = {
        let _ = metal_ple_enabled;
        None
    };
    // PLE input width is only consumed by the metal-gated `upload_ple`
    // closure below; on non-metal builds the binding is unused. Split
    // the cfg so each path gets the right warning posture without
    // breaking the metal name resolution.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let ple_dim = if metal_ple.is_some() {
        weights.arch.per_layer_embed_dim()
    } else {
        0
    };
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let _ple_dim = if metal_ple.is_some() {
        weights.arch.per_layer_embed_dim()
    } else {
        0
    };
    // Helper closure: precompute the per-layer-input table for one token
    // and upload it onto the Metal backend. Mirrors
    // `precompute_per_layer_inputs` for a single position.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let upload_ple =
        |metal: &larql_compute_metal::MetalBackend, token_id: u32, embed_row: &[f32]| {
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
            metal.prepare_ple_inputs(&flat, num_layers, ple_dim);
        };

    #[cfg(all(feature = "metal", target_os = "macos"))]
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
        metal_ple,
        &upload_ple,
    ) {
        Ok(v) => v,
        Err(err) => return GenerateResult::empty_error(err),
    };
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
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
    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut sampler = Sampler::new(sampling);
    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(token_ids);

    // Running list of token ids the model has emitted so far. Fed into
    // the sampler's repetition-penalty path; empty on the first pick.
    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);

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
    #[cfg(all(feature = "metal", target_os = "macos"))]
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
        metal_ple,
        &upload_ple,
        &mut on_token,
    );
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
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
mod tests {
    //! Tests that exercise the early-return guards reachable with
    //! `CpuBackend`. The full GPU path needs a Q4-supporting backend
    //! AND a Q4_K-loaded vindex, so the deeper branches stay 0%
    //! covered until that fixture infrastructure exists.
    use super::*;
    use crate::test_utils::TestFixtures;

    #[test]
    fn generate_returns_empty_success_when_max_tokens_is_zero() {
        let fx = TestFixtures::build();
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let mut weights = crate::test_utils::make_test_weights();
        let result = generate(
            &mut weights,
            &fx.tokenizer,
            &[0u32, 1],
            /*max_tokens=*/ 0,
            &fx.index,
            &larql_compute::CpuBackend,
            &cached,
            0..2,
        );
        assert!(result.tokens.is_empty());
        assert!(result.error.is_none());
    }

    #[test]
    fn generate_streaming_returns_empty_success_when_max_tokens_is_zero() {
        let fx = TestFixtures::build();
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let mut fired = false;
        let mut weights = crate::test_utils::make_test_weights();
        let result = generate_streaming(
            &mut weights,
            &fx.tokenizer,
            &[0u32],
            0,
            &fx.index,
            &larql_compute::CpuBackend,
            &cached,
            0..2,
            SamplingConfig::greedy(),
            &EosConfig::builtin(),
            |_, _, _| fired = true,
        );
        assert!(result.tokens.is_empty());
        assert!(!fired, "callback must not fire when max_tokens == 0");
    }

    // NOTE: a `generate_with_cpu_backend_falls_back_to_cpu_q4k_path`
    // test would route to `generate_via_cpu_q4k`, which then panics with
    // "attn Q4K slices missing for layer 0" because the synthetic vindex
    // has no Q4K attention data. Adding such a test requires either a
    // synthetic Q4K vindex fixture or a Q4K-tolerant CPU fallback.

    #[test]
    fn try_generate_returns_ok_for_empty_max_tokens() {
        let fx = TestFixtures::build();
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let mut weights = crate::test_utils::make_test_weights();
        let result = try_generate(
            &mut weights,
            &fx.tokenizer,
            &[0u32],
            0,
            &fx.index,
            &larql_compute::CpuBackend,
            &cached,
            0..2,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn try_generate_streaming_passes_through_to_streaming_path() {
        let fx = TestFixtures::build();
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let mut weights = crate::test_utils::make_test_weights();
        let result = try_generate_streaming(
            &mut weights,
            &fx.tokenizer,
            &[0u32],
            0,
            &fx.index,
            &larql_compute::CpuBackend,
            &cached,
            0..2,
            SamplingConfig::greedy(),
            &EosConfig::builtin(),
            |_, _, _| {},
        );
        assert!(result.is_ok());
    }

    #[test]
    fn try_generate_with_sampling_passes_through() {
        let fx = TestFixtures::build();
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let mut weights = crate::test_utils::make_test_weights();
        let result = try_generate_with_sampling(
            &mut weights,
            &fx.tokenizer,
            &[0u32],
            0,
            &fx.index,
            &larql_compute::CpuBackend,
            &cached,
            0..2,
            SamplingConfig::greedy(),
            &EosConfig::builtin(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn lmhead_k_for_sampling_uses_topk_greedy_for_greedy_config() {
        let greedy = SamplingConfig::greedy();
        assert_eq!(lmhead_k_for_sampling(&greedy), LMHEAD_TOPK_GREEDY);
    }

    #[test]
    fn lmhead_k_for_sampling_uses_min_when_sampling_topk_unset() {
        let mut cfg = SamplingConfig::greedy();
        cfg.temperature = 0.7; // non-greedy
        cfg.top_k = None;
        let k = lmhead_k_for_sampling(&cfg);
        assert!(k >= LMHEAD_TOPK_SAMPLING_MIN);
    }

    #[test]
    fn lmhead_k_for_sampling_honours_topk_when_larger_than_min() {
        let mut cfg = SamplingConfig::greedy();
        cfg.temperature = 0.7;
        cfg.top_k = Some(LMHEAD_TOPK_SAMPLING_MIN * 2);
        let k = lmhead_k_for_sampling(&cfg);
        assert_eq!(k, LMHEAD_TOPK_SAMPLING_MIN * 2);
    }

    // ── MockGpuBackend-driven tests (full GPU path body) ─────────────────

    /// `generate` with a Q4-capable mock backend + the on-disk Q4K
    /// fixture exercises the entire GPU dispatch chain end-to-end:
    /// `build_gpu_decode_setup` → `prefill_kquant_prompt` → first-token
    /// sample → decode-loop. With max_tokens=2 the loop iterates once.
    #[test]
    fn generate_with_mock_gpu_backend_runs_through_decode_loop() {
        use crate::test_utils::{
            make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer, MockGpuBackend,
        };
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let backend = MockGpuBackend::new();
        let num_layers = weights.num_layers;
        let result = generate(
            &mut weights,
            &tokenizer,
            &[0u32, 1, 2],
            /*max_tokens=*/ 2,
            &index,
            &backend,
            &cached,
            0..num_layers,
        );
        // Mock returns zeros so the sampler picks some deterministic
        // token — coverage is the goal, not the token id. The pipeline
        // ran to completion if there's no error.
        assert!(
            result.error.is_none(),
            "GPU path with mock backend must succeed: {:?}",
            result.error
        );
    }

    /// `generate_streaming` with the mock backend — same shape as above
    /// but also verifies the streaming callback fires for each emitted
    /// token.
    #[test]
    fn generate_streaming_with_mock_gpu_backend_emits_tokens_via_callback() {
        use crate::test_utils::{
            make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer, MockGpuBackend,
        };
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let backend = MockGpuBackend::new();
        let num_layers = weights.num_layers;
        let mut streamed: Vec<u32> = Vec::new();
        let result = generate_streaming(
            &mut weights,
            &tokenizer,
            &[0u32, 1, 2],
            3,
            &index,
            &backend,
            &cached,
            0..num_layers,
            SamplingConfig::greedy(),
            &EosConfig::builtin(),
            |id, _text, _prob| streamed.push(id),
        );
        assert!(
            result.error.is_none(),
            "streaming GPU path with mock backend must succeed: {:?}",
            result.error
        );
        // Callback fires once per emitted token.
        assert_eq!(streamed.len(), result.tokens.len());
    }

    /// Prompt-too-long error path — the mock backend isn't even
    /// consulted; `ensure_prompt_fits` triggers before any GPU dispatch.
    #[test]
    fn generate_streaming_rejects_oversized_prompt() {
        use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights, MockGpuBackend};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = MockGpuBackend::new();
        let cached = CachedLayerGraph::from_residuals(vec![]);
        // 4097 tokens > DEFAULT_GPU_KV_CACHE_MAX_SEQ.
        let huge_prompt: Vec<u32> = (0..4097u32).collect();
        let num_layers = weights.num_layers;
        let tokenizer = fx_tokenizer(weights.vocab_size);
        let result = generate(
            &mut weights,
            &tokenizer,
            &huge_prompt,
            5,
            &index,
            &backend,
            &cached,
            0..num_layers,
        );
        assert!(result.is_error());
        assert!(matches!(
            result.error,
            Some(GenerateError::PromptTooLong { .. })
        ));
    }

    fn fx_tokenizer(vocab_size: usize) -> tokenizers::Tokenizer {
        crate::test_utils::make_test_tokenizer(vocab_size)
    }

    /// `diag_compare_cpu_topk` is the diagnostic side-channel emitted when
    /// `LARQL_METAL_COMPARE_CPU=1`. It only prints to stderr — no return
    /// value, no side effects on weights/index. Direct call drives the
    /// function body (top-k formatting + stderr writes) so the dead
    /// stage-coverage region clears.
    #[test]
    fn diag_compare_cpu_topk_runs_against_synthetic_weights() {
        use crate::test_utils::TestFixtures;
        let fx = TestFixtures::build();
        let h_1d = ndarray::Array1::<f32>::from_elem(fx.weights.hidden_size, 0.1);
        // No assertions — function returns (). Just confirm no panic and
        // the body executes.
        diag_compare_cpu_topk(
            &fx.tokenizer,
            &fx.weights,
            &fx.index,
            &larql_compute::CpuBackend,
            &h_1d,
        );
    }
}
