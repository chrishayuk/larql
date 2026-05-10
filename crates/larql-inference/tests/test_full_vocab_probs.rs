//! Parity + invariant tests for `predict_q4k_full_vocab_probs` —
//! the new full-softmax API used by phase 4b's naive `target_forward`.
//!
//! Gated on `LARQL_FULL_VOCAB_PROBS_VINDEX` (a real LARQL Q4_K vindex
//! directory). When unset, all tests no-op cleanly.
//!
//! Local validation:
//!   LARQL_FULL_VOCAB_PROBS_VINDEX=$(pwd)/output/gemma-3-4b-it-vindex \
//!     cargo test -p larql-inference --test test_full_vocab_probs

use std::env;

fn vindex_path_or_skip() -> Option<std::path::PathBuf> {
    env::var("LARQL_FULL_VOCAB_PROBS_VINDEX")
        .ok()
        .map(std::path::PathBuf::from)
}

fn load(
    path: &std::path::Path,
) -> (
    larql_models::ModelWeights,
    tokenizers::Tokenizer,
    larql_vindex::VectorIndex,
) {
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_q4k(path, &mut callbacks).expect("load weights");
    let tokenizer = larql_inference::load_tokenizer(path).expect("load tokenizer");
    let index = larql_inference::open_inference_vindex(path).expect("load vindex");
    (weights, tokenizer, index)
}

#[test]
fn full_vocab_probs_normalize_to_one() {
    let Some(path) = vindex_path_or_skip() else {
        return;
    };
    let (mut weights, _tok, index) = load(&path);
    let prompt = vec![2u32, 100, 200, 300]; // arbitrary 4-token context
    let probs = larql_inference::predict_q4k_full_vocab_probs(&mut weights, &prompt, &index);
    assert_eq!(
        probs.len(),
        weights.vocab_size,
        "probability vector must have length vocab_size"
    );
    let sum: f64 = probs.iter().map(|&p| p as f64).sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "probabilities must sum to 1.0 within fp32 softmax tolerance, got {sum}"
    );
    for p in &probs {
        assert!(
            *p >= 0.0 && *p <= 1.0 + 1e-6,
            "probability out of [0, 1] range: {p}"
        );
    }
}

#[test]
fn full_vocab_probs_argmax_matches_predict_q4k_top1() {
    let Some(path) = vindex_path_or_skip() else {
        return;
    };
    let (mut weights, tokenizer, index) = load(&path);
    let prompt = vec![2u32, 100, 200, 300];
    let probs = larql_inference::predict_q4k_full_vocab_probs(&mut weights, &prompt, &index);
    let argmax = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .expect("non-empty probs");
    let top1 = larql_inference::predict_q4k(&mut weights, &tokenizer, &prompt, 1, &index);
    let top1_id = *top1.token_ids.first().expect("top1 result");
    assert_eq!(
        argmax, top1_id,
        "full_vocab argmax must match predict_q4k top-1 id"
    );
}

#[test]
fn full_vocab_probs_batched_matches_per_row_calls() {
    // Phase 4c task C.1 contract: full_vocab_probs_batched(h_batch)[k]
    // SHALL bit-equal full_vocab_probs(h_batch.row(k)) for each k.
    // Locks the parity contract phase 4c's GPU implementation must satisfy.
    let Some(path) = vindex_path_or_skip() else {
        return;
    };
    let (mut weights, _tok, index) = load(&path);
    // Get a real hidden state by running a forward pass on a short prompt.
    let prompt = vec![2u32, 100, 200];
    let h_real = larql_inference::vindex::predict_q4k_hidden(&mut weights, &prompt, &index, None);
    // Build a synthetic batch of N hidden states by stacking variants
    // of h_real with small perturbations (so each row is distinct).
    let hidden = h_real.shape()[1];
    let n = 3usize;
    let last_row = h_real.row(h_real.shape()[0] - 1);
    let mut batch = ndarray::Array2::<f32>::zeros((n, hidden));
    for k in 0..n {
        for d in 0..hidden {
            // Perturb each row slightly so they're not identical.
            batch[[k, d]] = last_row[d] + (k as f32) * 1e-3;
        }
    }

    let via_batched = larql_inference::full_vocab_probs_batched(&weights, &batch, 1.0);
    assert_eq!(via_batched.len(), n);

    for k in 0..n {
        let row_2d = batch.slice(ndarray::s![k..k + 1, ..]).to_owned();
        let via_single = larql_inference::full_vocab_probs(&weights, &row_2d, 1.0);
        assert_eq!(via_batched[k].len(), via_single.len(), "row {k} length");
        for (i, (a, b)) in via_batched[k].iter().zip(&via_single).enumerate() {
            assert!(
                (a - b).abs() < 1e-7,
                "row {k} vocab {i}: batched {a} vs single {b}"
            );
        }
    }
}

#[test]
fn full_vocab_probs_top1_probability_matches_predict_q4k() {
    let Some(path) = vindex_path_or_skip() else {
        return;
    };
    let (mut weights, tokenizer, index) = load(&path);
    let prompt = vec![2u32, 100, 200, 300];
    let probs = larql_inference::predict_q4k_full_vocab_probs(&mut weights, &prompt, &index);
    let max_p = probs.iter().fold(0.0_f32, |a, &b| a.max(b));
    let top1 = larql_inference::predict_q4k(&mut weights, &tokenizer, &prompt, 1, &index);
    let top1_p = top1
        .predictions
        .first()
        .map(|(_, p)| *p as f32)
        .expect("top1 prob");
    let diff = (max_p - top1_p).abs();
    assert!(
        diff < 1e-5,
        "top-1 probability mismatch: full_vocab max {max_p} vs predict_q4k {top1_p} (diff {diff})"
    );
}
