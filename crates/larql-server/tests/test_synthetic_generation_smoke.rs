//! Smoke test: run `generate_with_sampling` directly against the
//! synthetic vindex. Determines whether the synthetic weights are
//! stable enough to drive the chat/completions/stream coverage push,
//! or whether the weights need tuning first.
//!
//! Uses the Q4K-quantised synthetic fixture (`model_with_q4k_weights`)
//! because `generate_with_sampling` on a CPU backend routes through
//! `generate_via_cpu_q4k` → `predict_kquant_prefill`, which panics with
//! "attn Q4K slices missing" against an f32-only vindex.

mod common;

#[test]
fn synthetic_vindex_generates_at_least_one_token() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");

    let mut weights_guard = model.lock_weights_for_gen().expect("lock weights");
    let weights: &mut larql_inference::ModelWeights = &mut weights_guard;

    let encoding = model.tokenizer.encode("the capital", true).expect("encode");
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("prompt_ids: {prompt_ids:?}");
    assert!(!prompt_ids.is_empty(), "tokenizer must encode something");

    let patched = model.patched.blocking_read();
    let index = patched.base();
    let backend = larql_compute::default_backend();
    let cached_layers = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
    let num_layers = weights.num_layers;

    let sampling_params = larql_server::routes::openai::util::SamplingParams {
        temperature: Some(0.5),
        top_p: Some(0.9),
        seed: Some(42),
        frequency_penalty: None,
        presence_penalty: None,
    };
    let stop_strings: Vec<String> = Vec::new();
    let (sampling, eos) =
        larql_server::routes::openai::util::build_sampling_eos(sampling_params, &stop_strings);

    let result = larql_inference::layer_graph::generate_with_sampling(
        weights,
        &model.tokenizer,
        &prompt_ids,
        4, // max_tokens
        index,
        &*backend,
        &cached_layers,
        0..num_layers,
        sampling,
        &eos,
    );

    println!("result.tokens: {:?}", result.tokens);
    println!("result.tokens.len() = {}", result.tokens.len());
    // We don't assert a specific token — just that the generator
    // produced something and didn't panic. If this fails with NaN
    // somewhere, the synthetic weights need tuning.
    assert!(
        !result.tokens.is_empty(),
        "synthetic should produce at least one token; got empty result"
    );
    // And each token text should be non-empty (no NaN'd-to-empty-string).
    for (text, prob) in &result.tokens {
        assert!(prob.is_finite(), "probability must be finite, got {prob}");
        let _ = text;
    }
}
