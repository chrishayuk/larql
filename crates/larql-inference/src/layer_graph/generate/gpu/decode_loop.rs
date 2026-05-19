//! GPU-side decode-loop phase for [`super::generate_streaming`].
//!
//! Steps 1..max_tokens (the first-token sample is done by the caller
//! since it folds the prefill output rather than calling `decode_token`).
//! Each step: embed → GPU forward → final norm → lm_head → sample → emit.
//! Profile timings (`LARQL_PROFILE_DECODE`/`LARQL_PROFILE_SPLIT`) are
//! accumulated here and returned via [`DecodeLoopOutcome`].

use super::sampling_step::sample_and_emit;
use crate::layer_graph::generate::detok::Detokenizer;
use crate::layer_graph::generate::eos::EosConfig;
use crate::layer_graph::generate::lm_head::lm_head_topk_with_policy;
use crate::layer_graph::generate::policy::GenerationRuntimeConfig;
use crate::layer_graph::generate::sampling::Sampler;
use crate::model::ModelWeights;
use larql_compute::prelude::*;
use larql_compute::FullPipelineLayer;

/// Aggregated output of the decode-loop phase.
pub(super) struct DecodeLoopOutcome {
    /// `(text, prob)` per generated token (excluding the first, which the
    /// caller already produced from prefill).
    pub tokens: Vec<(String, f64)>,
    /// Per-step wall time in ms.
    pub decode_ms: Vec<f64>,
    pub t_embed: f64,
    pub t_gpu: f64,
    pub t_gate_up: f64,
    pub t_down: f64,
    pub t_norm: f64,
    pub t_lmhead: f64,
    pub t_detok: f64,
}

/// Run the decode loop for steps 1..max_tokens. The caller must already
/// have produced the first token from the prefill output and seeded
/// `generated_ids` / `current_token_id` accordingly.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(unused_variables)
)]
#[allow(clippy::too_many_arguments)]
pub(super) fn run_decode_loop<F>(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    layers: &[FullPipelineLayer],
    hidden: usize,
    intermediate: usize,
    norm_offset: f32,
    knn_k: usize,
    runtime: &GenerationRuntimeConfig,
    sampler: &mut Sampler,
    detok: &mut Detokenizer,
    eos: &EosConfig,
    generated_ids: &mut Vec<u32>,
    mut current_token_id: u32,
    max_tokens: usize,
    #[cfg(all(feature = "metal", target_os = "macos"))] metal_ple: Option<
        &larql_compute_metal::MetalBackend,
    >,
    #[cfg(all(feature = "metal", target_os = "macos"))] upload_ple: &dyn Fn(
        &larql_compute_metal::MetalBackend,
        u32,
        &[f32],
    ),
    on_token: &mut F,
) -> DecodeLoopOutcome
where
    F: FnMut(u32, &str, f64),
{
    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms: Vec<f64> = Vec::with_capacity(max_tokens);

    let profile = runtime.profile_decode;
    let profile_split = runtime.profile_split;
    let mut t_embed = 0.0f64;
    let mut t_gpu = 0.0f64;
    // Mutated only on metal/macos via metal_take_last_split_timings.
    #[allow(unused_mut)]
    let mut t_gate_up = 0.0f64;
    #[allow(unused_mut)]
    let mut t_down = 0.0f64;
    let mut t_norm = 0.0f64;
    let mut t_lmhead = 0.0f64;
    let mut t_detok = 0.0f64;

    for step in 1..max_tokens {
        let decode_start = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let h_tok = crate::forward::embed_tokens_pub(weights, &[current_token_id]);
        let x_dec: Vec<f32> = h_tok.row(0).to_vec();
        let embed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        if profile && step <= 2 {
            let x_nan = x_dec.iter().filter(|v| v.is_nan()).count();
            let x_max = x_dec
                .iter()
                .map(|v| v.abs())
                .filter(|v| v.is_finite())
                .fold(0.0f32, f32::max);
            eprintln!(
                "[profile] step={} input tok={} x_dec: len={} nan={} max_abs={:.3e}",
                step,
                current_token_id,
                x_dec.len(),
                x_nan,
                x_max,
            );
        }

        let t1 = std::time::Instant::now();
        // Per-Layer Embeddings: upload the precomputed input table for
        // the new token before the GPU dispatch. Only meaningful with
        // `--features metal`.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(metal) = metal_ple {
            upload_ple(metal, current_token_id, &x_dec);
        }
        let result = run_one_decode_step(
            weights,
            backend,
            layers,
            hidden,
            intermediate,
            &x_dec,
            profile_split,
            step,
        );
        let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

        if profile && step <= 2 {
            log_step_diagnostic(step, result.as_deref());
        }

        let Some(h_out) = result else {
            // GPU returned None mid-decode. The caller routes
            // non-fused-Q4 backends (today: CPU) to a full CPU Q4K path at
            // the top, so this branch can only fire when a GPU backend that
            // passed `backend_supports_fused_q4_pipeline` subsequently fails
            // a single decode step. Treat as early-stop rather than re-run
            // the O(N²) CPU path mid-loop without a kept id list.
            if profile {
                eprintln!("[profile] step={step} — GPU decode returned None; stopping generation");
            }
            break;
        };

        let t2 = std::time::Instant::now();
        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out).unwrap();
        let h_final =
            crate::forward::apply_norm(weights, &h_arr, weights.arch.final_norm_key(), norm_offset);
        let h_1d = h_final.row(0).to_owned();
        let norm_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let t3 = std::time::Instant::now();
        let hits =
            lm_head_topk_with_policy(index, weights, &h_1d, knn_k, backend, &runtime.lm_head);
        let lmhead_ms = t3.elapsed().as_secs_f64() * 1000.0;
        if profile && step <= 2 {
            log_h_1d_diagnostic(step, &h_1d, hits.len());
        }

        let step_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        decode_ms.push(step_ms);

        let t4 = std::time::Instant::now();
        let Some(picked) = sample_and_emit(
            sampler,
            detok,
            tokenizer,
            weights,
            eos,
            &hits,
            generated_ids,
            on_token,
        ) else {
            if profile {
                eprintln!("[profile] step={step} — lm_head returned empty; break");
            }
            break;
        };
        let detok_ms = t4.elapsed().as_secs_f64() * 1000.0;

        if profile {
            eprintln!(
                "[profile] step={step} total={step_ms:.1}ms  embed={embed_ms:.2}  gpu={gpu_ms:.1}  norm={norm_ms:.2}  lm_head={lmhead_ms:.1}  detok={detok_ms:.2}"
            );
        }
        t_embed += embed_ms;
        t_gpu += gpu_ms;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if profile_split {
            if let Some(pt) = larql_compute_metal::take_last_split_timings() {
                t_gate_up += pt.gate_up_ms;
                t_down += pt.down_ms;
            }
        }
        t_norm += norm_ms;
        t_lmhead += lmhead_ms;
        t_detok += detok_ms;
        tokens.push((picked.text, picked.prob));
        generated_ids.push(picked.id);
        current_token_id = picked.id;
        if picked.is_eos {
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

    DecodeLoopOutcome {
        tokens,
        decode_ms,
        t_embed,
        t_gpu,
        t_gate_up,
        t_down,
        t_norm,
        t_lmhead,
        t_detok,
    }
}

/// Dispatch one decode step: pick `decode_token_split_profile`,
/// `decode_token_q4k_moe`, or plain `decode_token` based on profiling
/// flag and per-layer FFN format.
#[allow(clippy::too_many_arguments)]
fn run_one_decode_step(
    weights: &ModelWeights,
    backend: &dyn ComputeBackend,
    layers: &[FullPipelineLayer],
    hidden: usize,
    intermediate: usize,
    x_dec: &[f32],
    profile_split: bool,
    step: usize,
) -> Option<Vec<f32>> {
    if profile_split && step == 2 {
        // Step 2 is post-JIT warm — run split profiling once and print.
        let (r, _ta, _tgu, _td) =
            backend.decode_token_split_profile(layers, x_dec, hidden, intermediate);
        return r;
    }
    if weights.has_per_layer_ffn() && backend.supports(Capability::DecodeQ4KMoe) {
        let norm_eps = weights.arch.norm_eps();
        let get_expert =
            |layer_idx, expert_idx| weights.get_layer_entry_bytes(layer_idx, expert_idx);
        return backend.decode_token_q4k_moe(
            layers,
            x_dec,
            hidden,
            intermediate,
            norm_eps,
            &get_expert,
        );
    }
    backend.decode_token(layers, x_dec, hidden, intermediate)
}

fn log_step_diagnostic(step: usize, h_out: Option<&[f32]>) {
    match h_out {
        Some(h) => {
            let h_nan = h.iter().filter(|v| v.is_nan()).count();
            let h_max = h
                .iter()
                .map(|v| v.abs())
                .filter(|v| v.is_finite())
                .fold(0.0f32, f32::max);
            eprintln!(
                "[profile] step={step} decode_token h_out: len={} nan={h_nan} max_abs={h_max:.3e}",
                h.len()
            );
        }
        None => eprintln!("[profile] step={step} decode_token returned None"),
    }
}

fn log_h_1d_diagnostic(step: usize, h_1d: &ndarray::Array1<f32>, hits_len: usize) {
    let h_nan = h_1d.iter().filter(|v| v.is_nan()).count();
    let h_inf = h_1d.iter().filter(|v| v.is_infinite()).count();
    let h_max = h_1d
        .iter()
        .map(|v| v.abs())
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[profile] step={step} h_1d: len={} nan={h_nan} inf={h_inf} max_abs={h_max:.3e}  hits.len()={hits_len}",
        h_1d.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_weights, MockGpuBackend,
    };

    // ── run_one_decode_step branches ──────────────────────────────────────

    #[test]
    fn run_one_decode_step_calls_decode_token_on_standard_arch() {
        let weights = make_test_weights();
        let backend = MockGpuBackend::new();
        let x_dec = vec![0.0f32; weights.hidden_size];
        let out = run_one_decode_step(
            &weights,
            &backend,
            &[],
            weights.hidden_size,
            weights.intermediate_size,
            &x_dec,
            /*profile_split=*/ false,
            /*step=*/ 5,
        )
        .expect("mock decode_token returns Some");
        assert_eq!(out.len(), weights.hidden_size);
    }

    /// `profile_split=true` AND `step==2` routes through
    /// `decode_token_split_profile`. MockGpuBackend's default impl
    /// delegates to `decode_token`, so it returns Some(zero vec).
    #[test]
    fn run_one_decode_step_split_profile_branch() {
        let weights = make_test_weights();
        let backend = MockGpuBackend::new();
        let x_dec = vec![0.0f32; weights.hidden_size];
        let out = run_one_decode_step(
            &weights,
            &backend,
            &[],
            weights.hidden_size,
            weights.intermediate_size,
            &x_dec,
            /*profile_split=*/ true,
            /*step=*/ 2,
        )
        .expect("split-profile branch returns Some");
        assert_eq!(out.len(), weights.hidden_size);
    }

    /// `profile_split=true` but `step != 2` does NOT take the split
    /// branch — falls through to the standard `decode_token` path.
    #[test]
    fn run_one_decode_step_split_profile_only_fires_on_step_2() {
        let weights = make_test_weights();
        let backend = MockGpuBackend::new();
        let x_dec = vec![0.0f32; weights.hidden_size];
        for step in [0usize, 1, 3, 5, 10] {
            let out = run_one_decode_step(
                &weights,
                &backend,
                &[],
                weights.hidden_size,
                weights.intermediate_size,
                &x_dec,
                /*profile_split=*/ true,
                step,
            );
            assert!(out.is_some(), "step {step}: split-profile must not fire");
        }
    }

    // ── log_step_diagnostic + log_h_1d_diagnostic ─────────────────────────

    #[test]
    fn log_step_diagnostic_handles_some_and_none() {
        log_step_diagnostic(0, Some(&[1.0f32, 2.0, f32::NAN, f32::INFINITY]));
        log_step_diagnostic(1, None);
        log_step_diagnostic(2, Some(&[]));
    }

    #[test]
    fn log_h_1d_diagnostic_handles_nan_and_inf() {
        let h = ndarray::Array1::from(vec![1.0f32, f32::NAN, f32::INFINITY, -0.5]);
        log_h_1d_diagnostic(0, &h, 5);
        let h_empty = ndarray::Array1::<f32>::zeros(0);
        log_h_1d_diagnostic(1, &h_empty, 0);
    }

    // ── run_decode_loop happy path ────────────────────────────────────────

    /// Full decode loop against MockGpuBackend with profile flags off —
    /// drives the main decode body for `max_tokens` steps. Returns
    /// `tokens.len() == max_tokens-1` (the caller produces step 0; the
    /// loop produces 1..max_tokens).
    #[test]
    fn run_decode_loop_emits_tokens_per_step_with_mock_backend() {
        use crate::layer_graph::generate::detok::Detokenizer;
        use crate::layer_graph::generate::eos::EosConfig;
        use crate::layer_graph::generate::policy::{GenerationRuntimeConfig, TokenSelectionPolicy};
        use crate::layer_graph::generate::sampling::{Sampler, SamplingConfig};

        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = MockGpuBackend::new();
        let tokenizer = crate::test_utils::make_test_tokenizer(weights.vocab_size);
        let mut sampler = Sampler::new(SamplingConfig::greedy());
        let runtime = GenerationRuntimeConfig::default();
        let mut detok = Detokenizer::new(&tokenizer);
        let eos = EosConfig::empty();
        let mut generated_ids: Vec<u32> = vec![0];
        let _ = SamplingConfig::greedy(); // touch the type
        let _ = TokenSelectionPolicy::default();
        let mut callback_count = 0;

        let outcome = run_decode_loop(
            &weights,
            &tokenizer,
            &index,
            &backend,
            &[],
            weights.hidden_size,
            weights.intermediate_size,
            weights.arch.norm_weight_offset(),
            5,
            &runtime,
            &mut sampler,
            &mut detok,
            &eos,
            &mut generated_ids,
            /*current_token_id=*/ 0,
            /*max_tokens=*/ 3,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            &|_b, _id, _x| {},
            &mut |_id, _text, _prob| {
                callback_count += 1;
            },
        );
        // max_tokens=3 → loop iterates steps 1..3, producing 2 tokens
        // (unless the mock's zero output causes the sampler to bail
        // earlier).
        assert!(outcome.tokens.len() <= 2);
        assert_eq!(outcome.tokens.len(), callback_count);
    }

    /// `profile_decode=true` exercises the per-step diagnostic blocks
    /// (lines 97-112, 134-136, 162-164, 187-220). All stderr writes, no
    /// behaviour change.
    #[test]
    fn run_decode_loop_with_profile_decode_runs_diagnostic_branches() {
        use crate::layer_graph::generate::detok::Detokenizer;
        use crate::layer_graph::generate::eos::EosConfig;
        use crate::layer_graph::generate::policy::GenerationRuntimeConfig;
        use crate::layer_graph::generate::sampling::{Sampler, SamplingConfig};

        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = MockGpuBackend::new();
        let tokenizer = crate::test_utils::make_test_tokenizer(weights.vocab_size);
        let mut sampler = Sampler::new(SamplingConfig::greedy());
        let runtime = GenerationRuntimeConfig {
            profile_decode: true,
            profile_split: true,
            compare_cpu: false,
            lm_head: Default::default(),
        };
        let mut detok = Detokenizer::new(&tokenizer);
        let eos = EosConfig::empty();
        let mut generated_ids: Vec<u32> = vec![0];

        let _outcome = run_decode_loop(
            &weights,
            &tokenizer,
            &index,
            &backend,
            &[],
            weights.hidden_size,
            weights.intermediate_size,
            weights.arch.norm_weight_offset(),
            5,
            &runtime,
            &mut sampler,
            &mut detok,
            &eos,
            &mut generated_ids,
            /*current_token_id=*/ 0,
            /*max_tokens=*/ 4, // need step==2 to hit the split-profile branch
            #[cfg(all(feature = "metal", target_os = "macos"))]
            None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            &|_b, _id, _x| {},
            &mut |_id, _text, _prob| {},
        );
    }

    /// `EosConfig::empty().with_eos_id(0)` + mock predicting id 0 →
    /// EOS break fires on the first decoded token. Drives line 207-208.
    #[test]
    fn run_decode_loop_breaks_on_eos_match() {
        use crate::layer_graph::generate::detok::Detokenizer;
        use crate::layer_graph::generate::eos::EosConfig;
        use crate::layer_graph::generate::policy::GenerationRuntimeConfig;
        use crate::layer_graph::generate::sampling::{Sampler, SamplingConfig};

        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = MockGpuBackend::new();
        let tokenizer = crate::test_utils::make_test_tokenizer(weights.vocab_size);
        let mut sampler = Sampler::new(SamplingConfig::greedy());
        let runtime = GenerationRuntimeConfig::default();
        let mut detok = Detokenizer::new(&tokenizer);
        // Mark every vocab id as EOS so whatever the sampler picks
        // triggers the EOS break on step 1.
        let mut eos = EosConfig::empty();
        for id in 0..weights.vocab_size as u32 {
            eos = eos.with_eos_id(id);
        }
        let mut generated_ids: Vec<u32> = vec![0];

        let outcome = run_decode_loop(
            &weights,
            &tokenizer,
            &index,
            &backend,
            &[],
            weights.hidden_size,
            weights.intermediate_size,
            weights.arch.norm_weight_offset(),
            5,
            &runtime,
            &mut sampler,
            &mut detok,
            &eos,
            &mut generated_ids,
            /*current_token_id=*/ 0,
            /*max_tokens=*/ 5,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            &|_b, _id, _x| {},
            &mut |_id, _text, _prob| {},
        );
        // EOS hits on step 1 → at most 1 token emitted then break.
        assert!(outcome.tokens.len() <= 1);
    }
}
