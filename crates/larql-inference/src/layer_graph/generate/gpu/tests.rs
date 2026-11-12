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
        None,
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
        None,
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

/// A GPU that fails a decode step part-way through a generation is an
/// error on the result, with the tokens produced before it. The public
/// entry point used to return those tokens with no error: a truncated
/// generation reported as a complete one.
#[test]
fn a_gpu_decode_step_that_fails_mid_generation_is_reported() {
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer, MockGpuBackend,
    };
    /// Decode steps the backend completes before it fails.
    const GOOD_STEPS: usize = 2;
    const MAX_TOKENS: usize = 6;

    let run = |backend: MockGpuBackend| {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let num_layers = weights.num_layers;
        generate(
            &mut weights,
            &tokenizer,
            &[0u32, 1, 2],
            MAX_TOKENS,
            &index,
            &backend,
            &cached,
            0..num_layers,
        )
    };

    // The control: the same request on a backend that does not fail.
    let whole = run(MockGpuBackend::new());
    assert!(whole.error.is_none(), "{:?}", whole.error);
    assert_eq!(whole.tokens.len(), MAX_TOKENS);

    let cut_short = run(MockGpuBackend::failing_after(GOOD_STEPS));
    assert!(
        cut_short.tokens.len() < MAX_TOKENS,
        "the failing backend must cut the generation short"
    );
    assert!(
        !cut_short.tokens.is_empty(),
        "the tokens before the failure are kept"
    );
    let error = cut_short
        .error
        .expect("a generation cut short by a failed step must carry an error");
    assert!(error.to_string().contains("decode step"), "{error}");
}

// ── The CPU fallback, through the streaming entry point ─────────────────
//
// `CpuBackend` has no fused Q4 pipeline, so `generate_streaming` hands the
// request to the CPU Q4K path. It used to hand over neither the callback
// nor the sampling settings.

mod cpu_fallback {
    use super::*;
    use crate::test_utils::Q4KTestFixtures;

    const PROMPT: [u32; 3] = [1, 2, 3];
    const MAX_TOKENS: usize = 4;
    /// Hot enough that the argmax is not the only token ever drawn.
    const HOT: f32 = 5.0;
    /// Seeds tried when looking for a draw that differs from greedy.
    const SEEDS_TRIED: u64 = 32;

    /// Stream on the CPU backend. Returns the tokens of the result and the
    /// texts the callback was given.
    fn stream(sampling: SamplingConfig) -> (Vec<String>, Vec<String>) {
        let mut fx = Q4KTestFixtures::build();
        let cached = CachedLayerGraph::from_residuals(vec![]);
        let layers = 0..fx.weights.num_layers;
        let mut streamed = Vec::new();
        let result = generate_streaming(
            &mut fx.weights,
            &fx.tokenizer,
            &PROMPT,
            MAX_TOKENS,
            &fx.index,
            &larql_compute::CpuBackend,
            &cached,
            layers,
            sampling,
            &EosConfig::builtin(),
            |_id, text, _prob| streamed.push(text.to_string()),
            None,
        );
        assert!(result.error.is_none(), "{:?}", result.error);
        let tokens = result.tokens.into_iter().map(|(text, _)| text).collect();
        (tokens, streamed)
    }

    #[test]
    fn streaming_on_the_cpu_backend_reports_each_token() {
        let (tokens, streamed) = stream(SamplingConfig::greedy());

        assert!(!tokens.is_empty());
        assert_eq!(streamed, tokens, "one callback per token, in order");
    }

    #[test]
    fn sampling_on_the_cpu_backend_is_not_decoded_greedily() {
        // Greedy is one sequence. If the sampling settings were dropped,
        // every seed would reproduce it.
        let (greedy, _) = stream(SamplingConfig::greedy());

        let differs = (0..SEEDS_TRIED)
            .any(|seed| stream(SamplingConfig::temperature(HOT).with_seed(seed)).0 != greedy);

        assert!(
            differs,
            "{SEEDS_TRIED} seeds at temperature {HOT} all reproduced the greedy output"
        );
    }
}
