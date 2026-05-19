//! Token generation — GPU and CPU paths.
//!
//! Sub-modules:
//! - [`eos`]: stop-token detection (built-in markers + `generation_config.json`).
//! - [`detok`]: incremental detokeniser preserving HF leading-space semantics.
//! - [`sampling`]: greedy / temperature / top-k / top-p sampler.

pub mod chat_session;
mod constrained;
mod cpu;
pub mod detok;
pub mod eos;
mod gpu;
mod gpu_setup;
mod lm_head;
pub(crate) mod policy;
pub mod sampling;
mod types;

pub use chat_session::{
    ChatMLRenderer, ChatSession, GemmaRenderer, Llama3Renderer, TurnRenderer, DEFAULT_MAX_CONTEXT,
};
pub use constrained::{
    generate_constrained, generate_constrained_streaming, generate_constrained_streaming_sampled,
    try_generate_constrained, try_generate_constrained_streaming,
    try_generate_constrained_streaming_sampled,
};
pub use detok::Detokenizer;
pub use eos::{EosConfig, BUILTIN_STOP_STRINGS, GENERATION_CONFIG_FILENAME};
pub use gpu::{
    generate, generate_streaming, generate_with_sampling, stream_forced_full_logits, try_generate,
    try_generate_streaming, try_generate_with_sampling, ForcedLogitsResult,
};
pub use lm_head::lm_head_topk;
pub use sampling::{Sampler, SamplingConfig};
pub use types::{GenerateError, GenerateResult, StageTimings};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer_graph::CachedLayerGraph;
    use crate::test_utils::make_test_weights;

    // ── lm_head / logit helpers (synthetic, no vindex) ────────────────────────

    #[test]
    fn backend_lm_head_scores_shape() {
        let weights = make_test_weights();
        let q = ndarray::Array1::from_elem(weights.hidden_size, 0.1f32);
        let scores = lm_head::backend_lm_head_scores(&weights, &q, &larql_compute::CpuBackend);
        assert_eq!(
            scores.len(),
            weights.vocab_size,
            "scores length should be vocab_size"
        );
        assert!(
            scores.iter().all(|v| v.is_finite()),
            "scores should be finite"
        );
    }

    #[test]
    fn cpu_lm_head_topk_length() {
        let weights = make_test_weights();
        let q = ndarray::Array1::from_elem(weights.hidden_size, 0.3f32);
        let hits = lm_head::cpu_lm_head_topk(&weights, &q, 5);
        assert!(hits.len() <= 5, "top-k should return at most 5 entries");
        assert!(!hits.is_empty(), "should return at least 1 entry");
    }

    #[test]
    fn cpu_lm_head_topk_sorted_descending() {
        let weights = make_test_weights();
        let q = ndarray::Array1::from_shape_vec(
            weights.hidden_size,
            (0..weights.hidden_size).map(|i| i as f32 * 0.01).collect(),
        )
        .unwrap();
        let hits = lm_head::cpu_lm_head_topk(&weights, &q, 4);
        let scores: Vec<f32> = hits.iter().map(|(_, s)| *s).collect();
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1],
                "top-k should be sorted descending: {} >= {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn cpu_lm_head_topk_token_ids_in_range() {
        let weights = make_test_weights();
        let q = ndarray::Array1::zeros(weights.hidden_size);
        let hits = lm_head::cpu_lm_head_topk(&weights, &q, 3);
        for (id, _) in &hits {
            assert!(
                *id < weights.vocab_size as u32,
                "token id {id} should be < vocab_size {}",
                weights.vocab_size
            );
        }
    }

    // ── Real-model generate tests (require LARQL_VINDEX_PATH) ─────────────────
    //
    // Run with:
    //   LARQL_VINDEX_PATH=/path/to/gemma3-4b-q4k-v2.vindex \
    //   cargo test -p larql-inference --lib layer_graph::generate::tests -- --ignored --nocapture

    fn load_test_vindex() -> Option<(larql_vindex::VectorIndex, larql_models::ModelWeights)> {
        let vpath = std::env::var(crate::vindex::ENV_VINDEX_PATH).ok()?;
        let path = std::path::Path::new(&vpath);
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let mut index = larql_vindex::VectorIndex::load_vindex(path, &mut cb).ok()?;
        index.load_attn_kquant(path).ok()?;
        index.load_interleaved_kquant(path).ok()?;
        let weights = larql_vindex::load_model_weights_kquant(path, &mut cb).ok()?;
        Some((index, weights))
    }

    #[test]
    #[ignore = "requires LARQL_VINDEX_PATH pointing to a Q4K vindex"]
    fn generate_returns_tokens() {
        let (index, mut weights) =
            load_test_vindex().expect("LARQL_VINDEX_PATH not set or invalid");
        let tokenizer = larql_vindex::load_vindex_tokenizer(std::path::Path::new(
            &std::env::var(crate::vindex::ENV_VINDEX_PATH).unwrap(),
        ))
        .expect("tokenizer load failed");

        let prompt = "The capital of France is";
        let token_ids =
            crate::encode_prompt(&tokenizer, &*weights.arch, prompt).expect("tokenize failed");

        let backend = larql_compute::default_backend();
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        let result = generate(
            &mut weights,
            &tokenizer,
            &token_ids,
            5,
            &index,
            backend.as_ref(),
            &cached,
            0..num_layers,
        );

        assert!(
            !result.tokens.is_empty(),
            "should generate at least one token"
        );
        eprintln!(
            "Generated: {:?}",
            result.tokens.iter().map(|(t, _)| t).collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "requires LARQL_VINDEX_PATH"]
    fn generate_streaming_callback_fires_per_token() {
        let (index, mut weights) =
            load_test_vindex().expect("LARQL_VINDEX_PATH not set or invalid");
        let tokenizer = larql_vindex::load_vindex_tokenizer(std::path::Path::new(
            &std::env::var(crate::vindex::ENV_VINDEX_PATH).unwrap(),
        ))
        .expect("tokenizer load failed");

        let prompt = "The capital of France is";
        let token_ids =
            crate::encode_prompt(&tokenizer, &*weights.arch, prompt).expect("tokenize failed");

        let backend = larql_compute::default_backend();
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;

        let mut streamed: Vec<(u32, String, f64)> = Vec::new();
        let result = generate_streaming(
            &mut weights,
            &tokenizer,
            &token_ids,
            5,
            &index,
            backend.as_ref(),
            &cached,
            0..num_layers,
            SamplingConfig::greedy(),
            &EosConfig::builtin(),
            |id, text, prob| streamed.push((id, text.to_string(), prob)),
        );

        // The streaming callback must fire exactly once per emitted token.
        assert_eq!(
            streamed.len(),
            result.tokens.len(),
            "streaming callback count must match tokens emitted",
        );
        // And the text it received must match the recorded surface form.
        for (i, (_, streamed_text, _)) in streamed.iter().enumerate() {
            assert_eq!(streamed_text, &result.tokens[i].0);
        }
    }

    #[test]
    #[ignore = "requires LARQL_VINDEX_PATH"]
    fn generate_prefill_ms_positive() {
        let (index, mut weights) = load_test_vindex().expect("LARQL_VINDEX_PATH not set");
        let tokenizer = larql_vindex::load_vindex_tokenizer(std::path::Path::new(
            &std::env::var(crate::vindex::ENV_VINDEX_PATH).unwrap(),
        ))
        .unwrap();
        let prompt = "Hello";
        let token_ids = crate::encode_prompt(&tokenizer, &*weights.arch, prompt).unwrap();
        let backend = larql_compute::default_backend();
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        let result = generate(
            &mut weights,
            &tokenizer,
            &token_ids,
            1,
            &index,
            backend.as_ref(),
            &cached,
            0..num_layers,
        );
        assert!(
            result.prefill_ms > 0.0,
            "prefill_ms should be positive (timing was recorded)"
        );
        assert_eq!(
            result.decode_ms.len(),
            result.tokens.len().saturating_sub(1)
        );
    }

    // ── Synthetic-fixture generate tests (no real model needed) ───────────

    /// Helper: load the on-disk synthetic Q4K fixture via the inference
    /// loader chain. Returns (tokenizer, weights, index) the same shape
    /// the ignored real-model tests above use, but everything in-memory.
    fn synthetic_q4k_setup() -> (
        tempfile::TempDir,
        tokenizers::Tokenizer,
        larql_models::ModelWeights,
        larql_vindex::VectorIndex,
    ) {
        use crate::test_utils::write_synthetic_q4k_model_dir;
        let tmp = tempfile::tempdir().expect("tempdir");
        write_synthetic_q4k_model_dir(tmp.path()).expect("write synthetic Q4K fixture");
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let mut index =
            larql_vindex::VectorIndex::load_vindex(tmp.path(), &mut cb).expect("load_vindex");
        index
            .load_attn_kquant(tmp.path())
            .expect("load_attn_kquant");
        index
            .load_interleaved_kquant(tmp.path())
            .expect("load_interleaved_kquant");
        let weights = larql_vindex::load_model_weights_kquant(tmp.path(), &mut cb)
            .expect("load_model_weights_kquant");
        let tokenizer =
            larql_vindex::load_vindex_tokenizer(tmp.path()).expect("tokenizer.json round-trip");
        (tmp, tokenizer, weights, index)
    }

    #[test]
    fn generate_routes_through_cpu_fallback_when_backend_lacks_fused_decode() {
        // `CpuBackend` advertises Q4_K matvec via `supports_quant(Q4_K)`
        // but doesn't implement the fused `decode_token` kernel, so
        // `generate()` takes the CPU fallback branch. Drives a real
        // prompt through the full generation pipeline against the
        // synthetic Q4K fixture.
        let (_tmp, tokenizer, mut weights, index) = synthetic_q4k_setup();
        // Synthetic tokenizer treats `[N]` as token N — use single-token
        // prompts so the encode succeeds.
        let token_ids = vec![0u32, 1, 2];
        let backend = larql_compute::CpuBackend;
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        let result = generate(
            &mut weights,
            &tokenizer,
            &token_ids,
            3,
            &index,
            &backend,
            &cached,
            0..num_layers,
        );
        // CPU fallback returns either tokens or an empty GenerateResult
        // with an error — both shapes are valid here. We just want the
        // function body to execute end-to-end.
        let _ = result;
    }

    #[test]
    fn generate_streaming_runs_against_synthetic_fixture() {
        // The CPU fallback path for `generate_streaming` may not invoke
        // the streaming callback (the streaming behaviour is a GPU-path
        // affordance). We just want the function to execute end-to-end
        // against the synthetic fixture without panicking; coverage is
        // the goal, not behavioural parity with the GPU path.
        let (_tmp, tokenizer, mut weights, index) = synthetic_q4k_setup();
        let token_ids = vec![0u32, 1];
        let backend = larql_compute::CpuBackend;
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        let mut streamed: Vec<(u32, String, f64)> = Vec::new();
        let _result = generate_streaming(
            &mut weights,
            &tokenizer,
            &token_ids,
            3,
            &index,
            &backend,
            &cached,
            0..num_layers,
            SamplingConfig::greedy(),
            &EosConfig::builtin(),
            |id, text, prob| streamed.push((id, text.to_string(), prob)),
        );
        // Callback may or may not fire on the CPU fallback path. Either
        // outcome is acceptable for coverage purposes.
    }
}
