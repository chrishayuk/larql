//! Integration test for `larql_inference::vindex::generate_kquant_cpu`.
//!
//! Loads a real Q4_K vindex and runs CPU autoregressive decode for a handful
//! of tokens to prove the helper actually drives the model end-to-end.
//!
//! ## Running
//!
//! Marked `#[ignore]` because it takes ~minutes per token on CPU with a 4B
//! model. Opt in explicitly:
//!
//! ```sh
//! cargo test -p larql-inference --test test_generate_q4k_cpu -- --ignored
//! ```
//!
//! Skip behaviour: when no Q4_K vindex is found in the standard search
//! locations, the test prints where it looked and returns cleanly even
//! under `--ignored`. Override the search with `LARQL_TEST_VINDEX=<path>`.

use std::path::PathBuf;
use std::time::Instant;

use larql_inference::vindex::generate_kquant_cpu;
use larql_vindex::{
    load_model_weights_kquant, load_vindex_config, load_vindex_tokenizer, QuantFormat,
    SilentLoadCallbacks, VectorIndex,
};

/// Search known locations for a Q4_K vindex on disk.
fn find_q4k_vindex() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LARQL_TEST_VINDEX") {
        let path = PathBuf::from(p);
        if path.is_dir() {
            return Some(path);
        }
    }

    let home = std::env::var("HOME").ok()?;
    let candidates = [
        // Workspace output directory (matches `cargo run -p larql-cli build`).
        PathBuf::from("output/gemma3-4b-q4k-v2.vindex"),
        PathBuf::from("output/gemma4-e2b-q4k.vindex"),
        // User cache locations.
        PathBuf::from(&home).join(".cache/larql/local/mistral-7b-v0.1-q4k.vindex"),
        PathBuf::from(&home).join(".cache/larql/local/llama2-7b-q4k.vindex"),
    ];

    for candidate in &candidates {
        if candidate.is_dir() {
            // Verify it's actually Q4_K — non-Q4 vindexes would fail downstream.
            if let Ok(cfg) = load_vindex_config(candidate) {
                if cfg.quant == QuantFormat::Q4K {
                    return Some(candidate.clone());
                }
            }
        }
    }
    None
}

#[test]
#[ignore = "loads a 4B model; ~minutes per token on CPU. Run with --ignored."]
fn generate_q4k_cpu_produces_tokens_against_real_vindex() {
    let Some(vindex_path) = find_q4k_vindex() else {
        eprintln!("skip: no Q4_K vindex found. Set LARQL_TEST_VINDEX=<path> to override.",);
        return;
    };
    eprintln!("vindex: {}", vindex_path.display());

    // ── Load weights + tokenizer + Q4 index ──
    let mut cb = SilentLoadCallbacks;
    let mut weights = load_model_weights_kquant(&vindex_path, &mut cb).expect("load weights");
    let tokenizer = load_vindex_tokenizer(&vindex_path).expect("load tokenizer");
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb).expect("load index");
    index.load_attn_kquant(&vindex_path).expect("load attn Q4K");
    index
        .load_interleaved_kquant(&vindex_path)
        .expect("load FFN Q4K");
    let _ = index.load_lm_head_kquant(&vindex_path);

    // ── Tokenise a tiny prompt ──
    let prompt = "The capital of France is";
    let prompt_ids =
        larql_inference::encode_prompt(&tokenizer, &*weights.arch, prompt).expect("tokenize");
    eprintln!("prompt: {prompt:?} → {} tokens", prompt_ids.len());

    // ── Generate a handful of tokens ──
    let max_tokens = 4;
    let t0 = Instant::now();
    let tokens = generate_kquant_cpu(&mut weights, &tokenizer, &prompt_ids, max_tokens, &index);
    let elapsed = t0.elapsed();

    eprintln!(
        "produced {} tokens in {:.2}s ({:.2}s/tok)",
        tokens.len(),
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / tokens.len().max(1) as f64,
    );
    let text: String = tokens.iter().map(|(t, _)| t.as_str()).collect();
    eprintln!("output: {text:?}");

    assert!(
        !tokens.is_empty(),
        "generate_kquant_cpu produced zero tokens"
    );
    assert!(
        tokens.len() <= max_tokens,
        "generate_kquant_cpu exceeded max_tokens cap"
    );
    // Each entry's id is u32 → just sanity-check we got distinct (text, id) pairs
    // populated. An all-empty-string output would suggest a tokeniser/decode bug.
    assert!(
        tokens.iter().any(|(t, _)| !t.is_empty()),
        "all output tokens were empty strings",
    );
}
