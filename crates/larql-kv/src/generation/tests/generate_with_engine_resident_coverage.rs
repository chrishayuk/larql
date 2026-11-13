//! generate_with_engine_resident coverage

use super::*;

#[test]
fn resident_happy_path_via_standard_engine() {
    use crate::engines::standard::StandardEngine;
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = NullFfn;
    let mut engine = crate::AnyEngine::Kv(Box::new(StandardEngine::new(None)));
    let mut callbacks = 0usize;
    let out = generate_with_engine_resident(
        &mut engine,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[0u32, 1, 2],
        4,
        |_, _| callbacks += 1,
    );
    assert_eq!(out.len(), callbacks, "callback fires once per token");
    assert!(out.len() <= 4);
}

#[test]
fn resident_zero_max_returns_empty() {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let out = generate_with_engine_resident(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[0u32, 1],
        0,
        |_, _| {},
    );
    assert!(out.is_empty());
}

#[test]
fn resident_empty_prompt_returns_empty() {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let out = generate_with_engine_resident(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[],
        4,
        |_, _| {},
    );
    assert!(out.is_empty());
}

#[test]
fn resident_prefill_failure_returns_empty() {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let mut stub = fresh_stub();
    stub.fail_prefill = true;
    let mut eng = crate::AnyEngine::Kv(Box::new(stub));
    let out = generate_with_engine_resident(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[0u32],
        3,
        |_, _| {},
    );
    assert!(out.is_empty());
}

#[test]
fn resident_max_one_returns_single_token() {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let out = generate_with_engine_resident(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[0u32, 1],
        1,
        |_, _| {},
    );
    assert_eq!(out.len(), 1);
}

#[test]
fn resident_decode_failure_breaks_loop_early() {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let mut stub = fresh_stub();
    stub.fail_decode_after = Some(1);
    let mut eng = crate::AnyEngine::Kv(Box::new(stub));
    let out = generate_with_engine_resident(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[0u32, 1],
        5,
        |_, _| {},
    );
    assert!(
        out.len() <= 2,
        "should break after decode failure, got {} tokens",
        out.len()
    );
}

// ── The checked variant ────────────────────────────────────────────────────

/// Decode steps the stub runs before failing, and the token budget that
/// outlasts it — so the stream ends on the failure, not on the budget.
const DECODES_BEFORE_FAILURE: usize = 1;
const CHECKED_MAX_NEW_TOKENS: usize = 5;

fn checked_run(stub: StubEngine) -> (EngineGeneration, usize) {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let mut eng = crate::AnyEngine::Kv(Box::new(stub));
    let mut callbacks = 0usize;
    let outcome = generate_with_engine_resident_checked(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[0u32, 1],
        CHECKED_MAX_NEW_TOKENS,
        |_, _| callbacks += 1,
    );
    (outcome, callbacks)
}

#[test]
fn checked_reports_a_prefill_failure() {
    let mut stub = fresh_stub();
    stub.fail_prefill = true;
    let (outcome, callbacks) = checked_run(stub);
    assert!(outcome.tokens.is_empty());
    assert_eq!(callbacks, 0);
    assert!(
        matches!(
            outcome.error,
            Some(larql_inference::kv_engine::EngineError::BackendFailure { .. })
        ),
        "{:?}",
        outcome.error
    );
}

#[test]
fn checked_reports_a_decode_failure_and_keeps_what_came_before() {
    let mut stub = fresh_stub();
    stub.fail_decode_after = Some(DECODES_BEFORE_FAILURE);
    let (outcome, callbacks) = checked_run(stub);
    assert_eq!(outcome.tokens.len(), callbacks);
    assert!(outcome.tokens.len() < CHECKED_MAX_NEW_TOKENS);
    // The prefill token, one token per successful decode, then the failure.
    assert_eq!(outcome.tokens.len(), DECODES_BEFORE_FAILURE + 1);
    assert!(
        matches!(
            outcome.error,
            Some(larql_inference::kv_engine::EngineError::BackendFailure { .. })
        ),
        "the decode failure must be returned: {:?}",
        outcome.error
    );
}

#[test]
fn checked_reports_nothing_when_nothing_failed() {
    let (outcome, callbacks) = checked_run(fresh_stub());
    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(outcome.tokens.len(), callbacks);
}

#[test]
fn checked_with_no_budget_or_no_prompt_is_empty_and_clean() {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    for (prompt, max) in [(&[0u32][..], 0usize), (&[][..], CHECKED_MAX_NEW_TOKENS)] {
        let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
        let outcome = generate_with_engine_resident_checked(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &index,
            prompt,
            max,
            |_, _| {},
        );
        assert!(outcome.tokens.is_empty() && outcome.error.is_none());
    }
}
