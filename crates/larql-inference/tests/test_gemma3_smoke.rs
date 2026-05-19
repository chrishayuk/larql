//! Gemma 3 4B regression smoke test — first-token sanity check.
//!
//! Loads a vindex, encodes a fixed prompt, runs greedy single-token
//! generation, asserts the first token decodes to the expected surface.
//! This is the cheapest possible end-to-end regression net for the
//! generate / EOS / detok / sampling stack — a one-token call exercises
//! every component except multi-step decode.
//!
//! The default expected output ("Paris" for "The capital of France is")
//! is the same one already pinned by `test_logits_goldens.rs` for Gemma 3
//! 4B; this test is the generate-path counterpart.
//!
//! ## Run
//!
//! ```bash
//! LARQL_VINDEX_PATH=output/gemma3-4b-q4k-v2.vindex \
//!   cargo test -p larql-inference --test test_gemma3_smoke -- --ignored
//! ```
//!
//! Set `CI_INTEGRATION=1` to drop the `#[ignore]` and require the test
//! to run as part of the integration tier.
//!
//! ## Override
//!
//! - `LARQL_SMOKE_PROMPT` — prompt string. Default: "The capital of France is".
//! - `LARQL_SMOKE_EXPECTED` — expected first-token surface (trimmed match).
//!   Default: "Paris".

use larql_compute::default_backend;
use larql_inference::layer_graph::{generate, CachedLayerGraph};
use larql_inference::open_inference_vindex;

const DEFAULT_PROMPT: &str = "The capital of France is";
const DEFAULT_EXPECTED_FIRST_TOKEN: &str = "Paris";
const ENV_VINDEX_PATH: &str = "LARQL_VINDEX_PATH";
const ENV_PROMPT: &str = "LARQL_SMOKE_PROMPT";
const ENV_EXPECTED: &str = "LARQL_SMOKE_EXPECTED";
const ENV_CI_INTEGRATION: &str = "CI_INTEGRATION";

#[test]
#[ignore = "requires LARQL_VINDEX_PATH; run with --ignored. Set CI_INTEGRATION=1 to fail-loud on missing vindex."]
fn first_token_matches_expected_surface() {
    // CI override: setting CI_INTEGRATION=1 makes this fail-loud rather
    // than silently skipping when the vindex path isn't set. Mirrors the
    // pattern used by test_logits_goldens.rs.
    let strict = std::env::var(ENV_CI_INTEGRATION).is_ok();
    let vindex_path = match std::env::var(ENV_VINDEX_PATH) {
        Ok(p) => p,
        Err(_) if strict => {
            panic!("{ENV_CI_INTEGRATION}=1 set but {ENV_VINDEX_PATH} not — cannot run smoke test")
        }
        Err(_) => return,
    };
    let prompt = std::env::var(ENV_PROMPT).unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
    let expected =
        std::env::var(ENV_EXPECTED).unwrap_or_else(|_| DEFAULT_EXPECTED_FIRST_TOKEN.to_string());

    let path = std::path::Path::new(&vindex_path);
    let index = open_inference_vindex(path).expect("failed to open vindex for inference");
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut weights =
        larql_vindex::load_model_weights_kquant(path, &mut cb).expect("failed to load weights");
    let tokenizer = larql_vindex::load_vindex_tokenizer(path).expect("tokenizer load");

    let token_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &prompt)
        .expect("tokenize failed");
    let backend = default_backend();
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
        !result.tokens.is_empty(),
        "generate must emit at least one token"
    );

    let first = &result.tokens[0].0;
    let trimmed = first.trim();
    assert_eq!(
        trimmed, expected,
        "first generated token mismatch: got {first:?} (trimmed {trimmed:?}), expected {expected:?}",
    );
}
