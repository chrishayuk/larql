//! Tests for the accuracy suite infrastructure.
//! These test the prompt sets, needle generation, and formatting
//! without needing model weights.

use kv_cache_benchmark::accuracy_suite::synthetic;

#[test]
fn synthetic_strategy_report_includes_rotorquant_rows() {
    let rows = synthetic::synthetic_strategy_report(
        &kv_cache_benchmark::model_config::ModelConfig::llama_8b(),
        2,
        7,
    );
    let names: Vec<&str> = rows.iter().map(|r| r.strategy.as_str()).collect();
    assert!(names.iter().any(|n| n.contains("RotorQuant Iso3")));
    assert!(names.iter().any(|n| n.contains("RotorQuant Planar3")));
    assert!(rows.iter().all(|r| r.decode_tok_s.is_finite()));
}

#[test]
fn synthetic_strategy_report_formats_decode_throughput_and_ppl_placeholder() {
    let rows = synthetic::synthetic_strategy_report(
        &kv_cache_benchmark::model_config::ModelConfig::llama_8b(),
        1,
        11,
    );
    let table = synthetic::format_synthetic_strategy_report(&rows);
    assert!(table.contains("decode tok/s"));
    assert!(table.contains("RotorQuant Iso3"));
    assert!(table.contains("real-only"));
}

#[cfg(feature = "real-model")]
mod with_model {
    use kv_cache_benchmark::accuracy_suite::needle;
    use kv_cache_benchmark::accuracy_suite::prompts;
    use kv_cache_benchmark::accuracy_suite::runner;

    #[test]
    fn test_diverse_100_has_100_prompts() {
        let prompts = prompts::diverse_100();
        assert_eq!(prompts.len(), 100, "Need exactly 100 diverse prompts");
    }

    #[test]
    fn test_diverse_100_all_categories() {
        let prompts = prompts::diverse_100();
        let mut categories: Vec<&str> = prompts.iter().map(|p| p.category).collect();
        categories.sort();
        categories.dedup();

        let expected = vec![
            "arithmetic",
            "code",
            "completion",
            "conversational",
            "factual",
            "geographic",
            "reasoning",
            "scientific",
        ];
        assert_eq!(categories, expected, "Missing categories");
    }

    #[test]
    fn test_diverse_100_balanced_categories() {
        let prompts = prompts::diverse_100();
        let mut categories: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for p in &prompts {
            *categories.entry(p.category).or_default() += 1;
        }
        // Each category should have at least 10 prompts
        for (cat, count) in &categories {
            assert!(
                *count >= 10,
                "Category '{cat}' has {count} prompts, expected >=10"
            );
        }
        // Total should be 100
        let total: usize = categories.values().sum();
        assert_eq!(total, 100, "Total prompts: {total}, expected 100");
    }

    #[test]
    fn test_quick_20_is_subset() {
        let quick = prompts::quick_20();
        let full = prompts::diverse_100();
        assert_eq!(quick.len(), 20);
        for q in &quick {
            assert!(
                full.iter().any(|f| f.text == q.text),
                "Quick prompt '{}' not in full set",
                q.text,
            );
        }
    }

    #[test]
    fn test_paris_test_prompt() {
        let p = prompts::paris_test();
        assert!(p.text.contains("France"));
        assert_eq!(p.expected_contains, "Paris");
    }

    #[test]
    fn test_needle_tests_scaling() {
        let tests = needle::needle_tests();
        assert!(tests.len() >= 5);
        // Context lengths should be increasing
        for w in tests.windows(2) {
            assert!(w[1].context_tokens > w[0].context_tokens);
        }
    }

    #[test]
    fn test_build_haystack_contains_needle() {
        let ctx = needle::build_haystack(1000, "SECRET-CODE-XYZ");
        assert!(ctx.contains("SECRET-CODE-XYZ"));
    }

    #[test]
    fn test_build_haystack_length_reasonable() {
        for target in [512, 4096, 32768] {
            let ctx = needle::build_haystack(target, "NEEDLE");
            // ~4 chars per token, allow 2x variance
            let expected_chars = target * 4;
            assert!(
                ctx.len() > expected_chars / 2,
                "Haystack at {target} tokens too short: {} chars",
                ctx.len(),
            );
        }
    }

    #[test]
    fn test_needle_found_detection() {
        assert!(needle::needle_found("The answer is AURORA-7749", "AURORA"));
        assert!(needle::needle_found("aurora is the code", "AURORA")); // case insensitive
        assert!(!needle::needle_found("The answer is unknown", "AURORA"));
    }

    #[test]
    fn test_multi_needle_tests() {
        let tests = needle::multi_needle_tests();
        assert_eq!(tests.len(), 5);
        for (fact, answer, query) in &tests {
            assert!(!fact.is_empty());
            assert!(!answer.is_empty());
            assert!(query.contains('?'));
        }
    }

    #[test]
    fn test_format_needle_results() {
        let results = vec![
            (
                512,
                vec![
                    ("Standard KV".to_string(), true),
                    ("Markov RS".to_string(), true),
                ],
            ),
            (
                32768,
                vec![
                    ("Standard KV".to_string(), false),
                    ("Markov RS".to_string(), true),
                ],
            ),
        ];
        let table = needle::format_needle_results(&results);
        assert!(table.contains("PASS"));
        assert!(table.contains("FAIL"));
        assert!(table.contains("512 tokens"));
        assert!(table.contains("32768 tokens"));
    }

    #[test]
    fn test_format_accuracy_table() {
        let strategies = vec![
            runner::StrategyAccuracy {
                strategy: "Standard KV".to_string(),
                top1_match_rate: 1.0,
                top1_matches: 100,
                top1_total: 100,
                mean_kl_divergence: 0.0,
                gen_first_diverge: None,
                gen_token_match_rate: 1.0,
                needle_pass_rate: 1.0,
                needle_passes: 5,
                needle_total: 5,
            },
            runner::StrategyAccuracy {
                strategy: "Markov RS".to_string(),
                top1_match_rate: 1.0,
                top1_matches: 100,
                top1_total: 100,
                mean_kl_divergence: 0.0,
                gen_first_diverge: None,
                gen_token_match_rate: 1.0,
                needle_pass_rate: 1.0,
                needle_passes: 5,
                needle_total: 5,
            },
        ];
        let table = runner::format_accuracy_table(&strategies);
        assert!(table.contains("100.0%"));
        assert!(table.contains("Standard KV"));
        assert!(table.contains("Markov RS"));
    }
}
