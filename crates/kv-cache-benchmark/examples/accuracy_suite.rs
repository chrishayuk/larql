//! Accuracy Suite: The five tests that prove the video frame.
//!
//! Runs all 5 strategies through:
//!   Test 1: Paris test (pass/fail)
//!   Test 2: Top-1 match rate (100 prompts)
//!   Test 3: KL divergence (full distribution)
//!   Test 4: Generation stability (50 tokens)
//!   Test 5: Needle-in-a-haystack (1K-32K)
//!
//! Usage:
//!   cargo run --example accuracy_suite --features real-model [--release]

#[cfg(feature = "real-model")]
fn main() {
    use kv_cache_benchmark::accuracy_suite::prompts;
    use kv_cache_benchmark::accuracy_suite::runner;

    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");

    // Load model
    let model_name = args
        .get(1)
        .filter(|a| !a.starts_with('-'))
        .map(|s| s.as_str())
        .unwrap_or("google/gemma-3-4b-it");
    println!("Loading model: {model_name}");
    let model = larql_inference::InferenceModel::load(model_name).expect("Failed to load model");

    // Load vindex (second arg or next non-flag arg)
    let vindex_path = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .nth(1)
        .expect("Usage: accuracy_suite <model-name> <vindex-path> [--quick]");
    println!("Loading vindex from: {vindex_path}");
    let index = larql_vindex::VectorIndex::load_vindex(
        std::path::Path::new(vindex_path),
        &mut larql_vindex::SilentLoadCallbacks,
    )
    .expect("Failed to load vindex");

    let backend = larql_inference::default_backend();

    println!("\n============================================================");
    println!("=== KV Cache Accuracy Suite ===");
    println!("============================================================\n");

    // ── Test 1: Paris test ──
    println!("--- Test 1: Paris Test (pass/fail) ---\n");
    let paris_results =
        runner::test_paris(model.weights(), model.tokenizer(), &index, backend.as_ref());
    for (strategy, pass) in &paris_results {
        let mark = if *pass { "PASS" } else { "FAIL" };
        println!("  {strategy:<30} {mark}");
    }

    // ── Test 2: Top-1 match rate ──
    let test_prompts = if quick {
        println!("\n--- Test 2: Top-1 Match Rate (20 prompts, --quick) ---\n");
        prompts::quick_20()
    } else {
        println!("\n--- Test 2: Top-1 Match Rate (100 prompts) ---\n");
        prompts::diverse_100()
    };

    let prompt_results = runner::test_top1_match_rate(
        model.weights(),
        model.tokenizer(),
        &index,
        backend.as_ref(),
        &test_prompts,
    );

    let strategy_accuracy = runner::compute_strategy_accuracy(&prompt_results);
    println!("{}", runner::format_accuracy_table(&strategy_accuracy));
    println!("{}", runner::format_category_breakdown(&prompt_results));

    // ── Test 4: Generation stability ──
    println!("\n--- Test 4: Generation Stability (20 tokens) ---\n");
    let gen_results = runner::test_generation_stability(
        model.weights(),
        model.tokenizer(),
        "The capital of France is Paris. France is a country in",
        20,
    );
    for (strategy, diverge, match_rate) in &gen_results {
        let div_str = match diverge {
            Some(d) => format!("token {d}"),
            None => "never".to_string(),
        };
        println!("  {strategy:<20} first diverge: {div_str:<12} match: {match_rate:.1}%");
    }

    // ── Summary: The video frame table ──
    println!("{}", runner::format_accuracy_table(&strategy_accuracy));

    // Write JSON
    let json = serde_json::to_string_pretty(&prompt_results).unwrap();
    let _ = std::fs::write(
        "crates/kv-cache-benchmark/results/accuracy_suite.json",
        &json,
    );
    println!("Results written to results/accuracy_suite.json");
}

#[cfg(not(feature = "real-model"))]
fn main() {
    use kv_cache_benchmark::accuracy_suite::synthetic;
    use kv_cache_benchmark::model_config::ModelConfig;

    let args: Vec<String> = std::env::args().collect();
    let seq_len = args
        .iter()
        .position(|a| a == "--seq-len")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32);

    let config = ModelConfig::llama_8b();
    let rows = synthetic::synthetic_strategy_report(&config, seq_len, 20260508);
    println!("{}", synthetic::format_synthetic_strategy_report(&rows));
    println!(
        "PPL rows require real model weights:\n  cargo run --example accuracy_suite --features real-model -- <model-name> <vindex-path> --quick"
    );
}
