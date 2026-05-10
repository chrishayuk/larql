//! Phase 4b task B.6 — speculative-vs-non-speculative token-ID parity.
//!
//! Phase 4c task C.6 — scaled to 256-prompt parity (this file's
//! second test, gated `#[ignore]`).
//!
//! Runs the SAME prompt(s) through `generate()` twice on the same
//! Gemma 3 4B vindex:
//!   1. Env unset, no drafter installed → existing non-speculative path.
//!   2. Env=1, drafter installed → v3 speculative path.
//!
//! Asserts the token-ID streams agree on at least the first emitted
//! token (= greedy + same RNG match modulo numerical noise) across
//! the prompt set, and that the spec path never produces empty
//! output.
//!
//! Gated on `LARQL_SPECULATIVE_PARITY_VINDEX` (a real LARQL Q4_K
//! vindex directory). When unset, all tests no-op cleanly.
//!
//! Optional: `LARQL_SPECULATIVE_PARITY_DRAFTER_VINDEX` selects a
//! separate drafter vindex (e.g. Gemma 3 270M). When unset, the
//! drafter loads from the same vindex as the target.
//!
//! Local validation:
//!
//! ```bash
//! # Short test (1 prompt × 5 tokens) — runs by default:
//! LARQL_SPECULATIVE_PARITY_VINDEX=$(pwd)/output/gemma-3-4b-it-vindex \
//!   cargo test --release -p larql-inference --test test_speculative_parity
//!
//! # Full 256-prompt parity:
//! LARQL_SPECULATIVE_PARITY_VINDEX=$(pwd)/output/gemma-3-4b-it-vindex \
//! LARQL_SPECULATIVE_PARITY_DRAFTER_VINDEX=$(pwd)/output/gemma-3-270m-it-vindex \
//!   cargo test --release -p larql-inference --features cuda \
//!   --test test_speculative_parity \
//!   -- --ignored --nocapture
//! ```

use std::env;

use larql_compute::default_backend;
use larql_inference::layer_graph::CachedLayerGraph;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use tokenizers::Tokenizer;

fn vindex_path_or_skip() -> Option<std::path::PathBuf> {
    env::var("LARQL_SPECULATIVE_PARITY_VINDEX")
        .ok()
        .map(std::path::PathBuf::from)
}

fn drafter_path() -> Option<std::path::PathBuf> {
    env::var("LARQL_SPECULATIVE_PARITY_DRAFTER_VINDEX")
        .ok()
        .map(std::path::PathBuf::from)
}

fn load_target(path: &std::path::Path) -> (ModelWeights, Tokenizer, VectorIndex) {
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_q4k(path, &mut callbacks).expect("load weights");
    let tokenizer = larql_inference::load_tokenizer(path).expect("tokenizer");
    let index = larql_inference::open_inference_vindex(path).expect("vindex");
    (weights, tokenizer, index)
}

/// Run a single generate() call against the supplied (already-loaded)
/// target. When `speculative == true`, the caller MUST have
/// pre-installed a thread-local drafter via `set_thread_drafter`.
/// On return, the env var and thread-local cleanup happens here.
#[allow(clippy::too_many_arguments)]
fn run_generate(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    index: &VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
    cached_layers: &CachedLayerGraph,
    prompt: &str,
    max_tokens: usize,
    speculative: bool,
) -> Vec<u32> {
    let token_ids: Vec<u32> = tokenizer
        .encode(prompt, true)
        .expect("encode")
        .get_ids()
        .to_vec();

    if speculative {
        // SAFETY: tests in this file are not run in parallel with anything
        // that mutates the same vars (single-threaded test runner required).
        unsafe {
            env::set_var("LARQL_SPECULATIVE_DECODE", "1");
        }
        // Reset RNG per prompt so each prompt sees the same VerifyRng
        // sequence — eliminates a confounding variable across prompts
        // (otherwise prompt[k]'s spec output depends on prompt[k-1]'s
        // accept/reject decisions consuming RNG state).
        larql_inference::speculative::set_thread_rng(0xCAFE_BABE_DEAD_F00D);
    } else {
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
    }

    let num_layers = weights.num_layers;
    let result = larql_inference::layer_graph::generate(
        weights,
        tokenizer,
        &token_ids,
        max_tokens,
        index,
        backend,
        cached_layers,
        0..num_layers,
    );

    if speculative {
        unsafe {
            env::remove_var("LARQL_SPECULATIVE_DECODE");
        }
    }

    // Re-tokenize the generated text to recover token IDs.
    let generated_text: String = result.tokens.iter().map(|(s, _)| s.clone()).collect();
    let combined = format!("{prompt}{generated_text}");
    let combined_ids: Vec<u32> = tokenizer
        .encode(combined, true)
        .expect("encode combined")
        .get_ids()
        .to_vec();
    combined_ids[token_ids.len().min(combined_ids.len())..].to_vec()
}

#[test]
fn token_id_parity_speculative_vs_baseline_short_prompt() {
    let Some(path) = vindex_path_or_skip() else {
        return;
    };
    let drafter = drafter_path().unwrap_or_else(|| path.clone());
    let prompt = "The capital of France is";
    let max_tokens = 5;

    // Load target ONCE; reuse across baseline + spec paths.
    let (mut weights, tokenizer, index) = load_target(&path);
    let cached_layers = CachedLayerGraph::from_residuals(Vec::new());
    let backend = default_backend();

    let baseline_ids = run_generate(
        &mut weights,
        &tokenizer,
        &index,
        &*backend,
        &cached_layers,
        prompt,
        max_tokens,
        false,
    );

    // Install drafter for the spec path.
    let drafter_loaded =
        larql_inference::speculative::SmallModelDrafter::from_vindex(&drafter).expect("drafter");
    larql_inference::speculative::set_thread_drafter(Some(Box::new(drafter_loaded)));
    larql_inference::speculative::set_thread_spec_config(
        larql_inference::speculative::SpecConfig {
            depth: 2,
            branches: 1,
            swa_window: None,
        },
    );
    let speculative_ids = run_generate(
        &mut weights,
        &tokenizer,
        &index,
        &*backend,
        &cached_layers,
        prompt,
        max_tokens,
        true,
    );
    larql_inference::speculative::set_thread_drafter(None);

    let common_prefix = baseline_ids
        .iter()
        .zip(&speculative_ids)
        .take_while(|(a, b)| a == b)
        .count();

    eprintln!(
        "baseline={baseline_ids:?}  speculative={speculative_ids:?}  common_prefix={common_prefix}"
    );

    assert!(
        !baseline_ids.is_empty(),
        "baseline produced no tokens (model load issue?)"
    );
    assert!(
        !speculative_ids.is_empty(),
        "speculative produced no tokens (dispatch failed?)"
    );
    assert!(
        common_prefix >= 1,
        "expected ≥ 1 token of common prefix between baseline and speculative; \
         baseline={baseline_ids:?}, speculative={speculative_ids:?}"
    );
}

/// Generate `n` deterministic prompts by combining template heads with
/// topic words. Diverse enough to exercise different parts of the
/// vocab without requiring an external corpus.
fn synth_prompts(n: usize) -> Vec<String> {
    let templates: &[&str] = &[
        "The capital of {} is",
        "Why is {} important?",
        "How do I learn {}?",
        "Tell me about {}.",
        "List three facts about {}:",
        "What is the history of {}?",
        "{} can be used to",
        "I went to {} and",
        "The best way to {} is",
        "Once upon a time in {},",
        "Five reasons {} matters:",
        "If you visit {},",
        "{} works by",
        "When did {} happen?",
        "Compared to {},",
        "Imagine a world where {}.",
    ];
    let topics: &[&str] = &[
        "France",
        "machine learning",
        "Tokyo",
        "the Renaissance",
        "the moon",
        "ancient Rome",
        "quantum computing",
        "carrots",
        "ocean tides",
        "rust programming",
        "the Amazon",
        "antique clocks",
        "deep sea diving",
        "calculus",
        "London",
        "honeybees",
        "modern art",
        "thunderstorms",
        "chess strategy",
        "the printing press",
        "neural networks",
        "Mars exploration",
        "the Silk Road",
        "fermentation",
        "Vikings",
        "snow leopards",
        "submarine sandwiches",
        "telescope optics",
        "prime numbers",
        "watercolor painting",
        "tap dancing",
        "bonsai trees",
    ];
    let mut prompts = Vec::with_capacity(n);
    for i in 0..n {
        let t = templates[i % templates.len()];
        let topic = topics[(i / templates.len()) % topics.len()];
        prompts.push(t.replace("{}", topic));
    }
    prompts
}

#[derive(Debug, Clone)]
struct ParityRow {
    prompt_idx: usize,
    baseline_len: usize,
    spec_len: usize,
    common_prefix: usize,
}

#[test]
#[ignore = "scale test — opt in with --ignored"]
fn parity_at_scale_256_prompts() {
    let Some(path) = vindex_path_or_skip() else {
        eprintln!("skipping C.6 parity at scale: LARQL_SPECULATIVE_PARITY_VINDEX env var unset");
        return;
    };
    let drafter = drafter_path().unwrap_or_else(|| path.clone());
    let max_tokens = 64;
    let n_prompts = 256;
    let prompts = synth_prompts(n_prompts);

    // Load target ONCE — the previous version of this test reloaded
    // weights + recreated CudaBackend per generate() call (~5s of
    // mmap + CUDA init + kernel JIT). With 256 prompts × 2 paths =
    // 512 reloads, that was ~40 min of CPU-bound setup with the GPU
    // mostly idle. Hoisted out: ~3 min total.
    let t_load = std::time::Instant::now();
    let (mut weights, tokenizer, index) = load_target(&path);
    let cached_layers = CachedLayerGraph::from_residuals(Vec::new());
    let backend = default_backend();
    eprintln!(
        "[C.6] target loaded in {:.2}s",
        t_load.elapsed().as_secs_f64()
    );

    // Pass 1: baseline (env-OFF). No drafter installed.
    let t_baseline = std::time::Instant::now();
    let mut baseline_streams: Vec<Vec<u32>> = Vec::with_capacity(n_prompts);
    for (i, prompt) in prompts.iter().enumerate() {
        let ids = run_generate(
            &mut weights,
            &tokenizer,
            &index,
            &*backend,
            &cached_layers,
            prompt,
            max_tokens,
            false,
        );
        baseline_streams.push(ids);
        if (i + 1) % 64 == 0 {
            let elapsed = t_baseline.elapsed().as_secs_f64();
            eprintln!(
                "[C.6] baseline {}/{} ({:.1}s, {:.2}/s)",
                i + 1,
                n_prompts,
                elapsed,
                (i + 1) as f64 / elapsed
            );
        }
    }
    let baseline_secs = t_baseline.elapsed().as_secs_f64();
    eprintln!("[C.6] baseline done in {baseline_secs:.1}s");

    // Pass 2: speculative (env-ON). Install drafter once.
    let t_drafter_load = std::time::Instant::now();
    let drafter_loaded =
        larql_inference::speculative::SmallModelDrafter::from_vindex(&drafter).expect("drafter");
    larql_inference::speculative::set_thread_drafter(Some(Box::new(drafter_loaded)));
    larql_inference::speculative::set_thread_spec_config(
        larql_inference::speculative::SpecConfig {
            depth: 2,
            branches: 1,
            swa_window: None,
        },
    );
    eprintln!(
        "[C.6] drafter loaded in {:.2}s",
        t_drafter_load.elapsed().as_secs_f64()
    );

    let t_spec = std::time::Instant::now();
    let mut spec_streams: Vec<Vec<u32>> = Vec::with_capacity(n_prompts);
    for (i, prompt) in prompts.iter().enumerate() {
        let ids = run_generate(
            &mut weights,
            &tokenizer,
            &index,
            &*backend,
            &cached_layers,
            prompt,
            max_tokens,
            true,
        );
        spec_streams.push(ids);
        if (i + 1) % 64 == 0 {
            let elapsed = t_spec.elapsed().as_secs_f64();
            eprintln!(
                "[C.6] spec {}/{} ({:.1}s, {:.2}/s)",
                i + 1,
                n_prompts,
                elapsed,
                (i + 1) as f64 / elapsed
            );
        }
    }
    let spec_secs = t_spec.elapsed().as_secs_f64();
    eprintln!("[C.6] spec done in {spec_secs:.1}s");

    larql_inference::speculative::set_thread_drafter(None);

    // Compute parity rows.
    let rows: Vec<ParityRow> = baseline_streams
        .iter()
        .zip(&spec_streams)
        .enumerate()
        .map(|(i, (b, s))| {
            let common_prefix = b.iter().zip(s).take_while(|(a, b)| a == b).count();
            ParityRow {
                prompt_idx: i,
                baseline_len: b.len(),
                spec_len: s.len(),
                common_prefix,
            }
        })
        .collect();

    // Aggregate metrics.
    let baseline_empty = rows.iter().filter(|r| r.baseline_len == 0).count();
    let spec_empty = rows.iter().filter(|r| r.spec_len == 0).count();
    let common_zero = rows.iter().filter(|r| r.common_prefix == 0).count();
    let common_at_least_1 = rows.iter().filter(|r| r.common_prefix >= 1).count();
    let common_at_least_5 = rows.iter().filter(|r| r.common_prefix >= 5).count();
    let common_full = rows
        .iter()
        .filter(|r| r.common_prefix == r.baseline_len.min(r.spec_len))
        .count();

    let mut prefixes: Vec<usize> = rows.iter().map(|r| r.common_prefix).collect();
    prefixes.sort_unstable();
    let p25 = prefixes[prefixes.len() / 4];
    let p50 = prefixes[prefixes.len() / 2];
    let p75 = prefixes[(prefixes.len() * 3) / 4];
    let mean: f64 = prefixes.iter().sum::<usize>() as f64 / prefixes.len() as f64;
    let max = *prefixes.last().unwrap_or(&0);

    eprintln!(
        "\n[C.6] Parity at scale — {} prompts × {} max_tokens — load {:.1}s baseline {:.1}s spec {:.1}s\n\
         common_prefix:  p25={p25}  p50={p50}  p75={p75}  mean={mean:.2}  max={max}\n\
         common_prefix == 0:                {common_zero}/{n_prompts} ({:.1}%)\n\
         common_prefix >= 1:                {common_at_least_1}/{n_prompts} ({:.1}%)\n\
         common_prefix >= 5:                {common_at_least_5}/{n_prompts} ({:.1}%)\n\
         common_prefix == full output len:  {common_full}/{n_prompts} ({:.1}%)\n\
         baseline empty:                    {baseline_empty}/{n_prompts}\n\
         spec empty:                        {spec_empty}/{n_prompts}",
        n_prompts,
        max_tokens,
        t_load.elapsed().as_secs_f64() - baseline_secs - spec_secs,
        baseline_secs,
        spec_secs,
        common_zero as f64 * 100.0 / n_prompts as f64,
        common_at_least_1 as f64 * 100.0 / n_prompts as f64,
        common_at_least_5 as f64 * 100.0 / n_prompts as f64,
        common_full as f64 * 100.0 / n_prompts as f64,
    );

    // Hard correctness gates.
    assert_eq!(
        baseline_empty, 0,
        "baseline produced no tokens for some prompts"
    );
    assert!(
        spec_empty * 100 < n_prompts,
        "spec produced empty output for >1% of prompts ({spec_empty}/{n_prompts})"
    );

    // Parity gates: ≥75% of prompts share the first emitted token, and
    // p50 common_prefix ≥ 1. Either failing means spec is systematically
    // diverging from baseline.
    assert!(
        common_at_least_1 >= (n_prompts * 3) / 4,
        "only {}/{} prompts share even 1 token of prefix; spec mostly diverging at first token",
        common_at_least_1,
        n_prompts
    );
    assert!(
        p50 >= 1,
        "median common prefix is {p50}; expected ≥ 1 (greedy first-token match)"
    );
}
