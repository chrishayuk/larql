//! Needle-in-a-haystack test.
//!
//! Plant a fact at a known position, fill with distractor text,
//! query at the end. Scale from 1K to 32K context.
//!
//! The key finding:
//! "At long context, Standard KV and TurboQuant FAIL because of softmax dilution.
//!  Markov RS PASSES because it routes to the relevant window."

/// A needle-in-haystack test case.
#[derive(Debug, Clone, serde::Serialize)]
pub struct NeedleTest {
    pub context_tokens: usize,
    pub needle_text: &'static str,
    pub needle_answer: &'static str,
    pub query_text: &'static str,
}

/// Standard needle tests at increasing context lengths.
pub fn needle_tests() -> Vec<NeedleTest> {
    let needle = "The secret project code name is AURORA-7749.";
    let answer = "AURORA";
    let query = "What is the secret project code name?";

    vec![
        NeedleTest {
            context_tokens: 512,
            needle_text: needle,
            needle_answer: answer,
            query_text: query,
        },
        NeedleTest {
            context_tokens: 1024,
            needle_text: needle,
            needle_answer: answer,
            query_text: query,
        },
        NeedleTest {
            context_tokens: 2048,
            needle_text: needle,
            needle_answer: answer,
            query_text: query,
        },
        NeedleTest {
            context_tokens: 4096,
            needle_text: needle,
            needle_answer: answer,
            query_text: query,
        },
        NeedleTest {
            context_tokens: 8192,
            needle_text: needle,
            needle_answer: answer,
            query_text: query,
        },
        NeedleTest {
            context_tokens: 16384,
            needle_text: needle,
            needle_answer: answer,
            query_text: query,
        },
        NeedleTest {
            context_tokens: 32768,
            needle_text: needle,
            needle_answer: answer,
            query_text: query,
        },
    ]
}

/// Multi-needle test: 5 facts at different positions in 32K context.
pub fn multi_needle_tests() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "Agent Alpha's code name is FALCON.",
            "FALCON",
            "What is Agent Alpha's code name?",
        ),
        (
            "The launch date is March 15th.",
            "March",
            "What is the launch date?",
        ),
        (
            "Budget allocation is $4.7 million.",
            "4.7",
            "What is the budget allocation?",
        ),
        (
            "The target city is Reykjavik.",
            "Reykjavik",
            "What is the target city?",
        ),
        (
            "Project sponsor is Dr. Kimura.",
            "Kimura",
            "Who is the project sponsor?",
        ),
    ]
}

/// Build a haystack context with needle planted at ~10% position.
pub fn build_haystack(target_tokens: usize, needle: &str) -> String {
    // Filler: ~4 chars per token average
    let filler_sentence =
        "The quick brown fox jumps over the lazy dog near the old oak tree by the river. ";
    let needle_position = target_tokens / 10; // Plant early (~10% in)
    let chars_per_token = 4;

    let mut context = String::new();

    // Filler before needle
    let chars_before = needle_position * chars_per_token;
    while context.len() < chars_before {
        context.push_str(filler_sentence);
    }

    // Plant needle
    context.push_str(needle);
    context.push(' ');

    // Filler after needle
    let total_chars = target_tokens * chars_per_token;
    while context.len() < total_chars {
        context.push_str(filler_sentence);
    }

    context
}

/// Check if a prediction contains the needle answer.
pub fn needle_found(prediction: &str, answer: &str) -> bool {
    prediction.to_lowercase().contains(&answer.to_lowercase())
}

/// Format needle test results.
pub fn format_needle_results(
    results: &[(usize, Vec<(String, bool)>)], // (context_len, [(strategy, found)])
) -> String {
    let mut out = String::new();
    out.push_str("\n=== Needle-in-a-Haystack Results ===\n\n");

    if results.is_empty() {
        return out;
    }

    // Header from first result's strategies
    let strategy_names: Vec<&str> = results[0].1.iter().map(|(s, _)| s.as_str()).collect();
    out.push_str(&format!("{:<15}", "Context"));
    for name in &strategy_names {
        let short = if name.len() > 12 { &name[..12] } else { name };
        out.push_str(&format!(" {:>12}", short));
    }
    out.push('\n');
    out.push_str(&"-".repeat(15 + strategy_names.len() * 13));
    out.push('\n');

    for (ctx_len, strat_results) in results {
        out.push_str(&format!("{:<15}", format!("{} tokens", ctx_len)));
        for (_, found) in strat_results {
            let mark = if *found { "PASS" } else { "FAIL" };
            out.push_str(&format!(" {:>12}", mark));
        }
        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn needle_tests_corpus_is_non_empty() {
        let tests = needle_tests();
        assert!(!tests.is_empty());
        for t in &tests {
            assert!(!t.needle_text.is_empty(), "needle_text missing");
            assert!(!t.needle_answer.is_empty(), "needle_answer missing");
            assert!(!t.query_text.is_empty(), "query_text missing");
            assert!(t.context_tokens > 0, "context_tokens must be > 0");
        }
    }

    #[test]
    fn multi_needle_tests_corpus_is_non_empty() {
        let tests = multi_needle_tests();
        assert!(!tests.is_empty());
    }

    #[test]
    fn build_haystack_plants_needle() {
        let needle = "MAGIC-WORD-XYZ";
        let hay = build_haystack(200, needle);
        assert!(
            hay.contains(needle),
            "needle should be embedded in haystack"
        );
        // Haystack should approximately match the requested length
        // (4 chars per token average).
        assert!(hay.len() >= 200 * 4 / 2);
    }

    #[test]
    fn build_haystack_zero_tokens_returns_needle_only() {
        let needle = "X";
        let hay = build_haystack(0, needle);
        assert!(hay.contains(needle));
    }

    #[test]
    fn needle_found_is_case_insensitive() {
        assert!(needle_found("the answer is PARIS", "paris"));
        assert!(needle_found("Paris is the capital", "PARIS"));
        assert!(!needle_found("no match here", "berlin"));
    }

    #[test]
    fn format_needle_results_empty_input() {
        let s = format_needle_results(&[]);
        assert!(s.contains("Needle-in-a-Haystack"));
    }

    #[test]
    fn format_needle_results_renders_rows() {
        let rows = vec![
            (1024usize, vec![("Standard".to_string(), true)]),
            (4096usize, vec![("Standard".to_string(), false)]),
        ];
        let s = format_needle_results(&rows);
        assert!(s.contains("1024 tokens"));
        assert!(s.contains("4096 tokens"));
        assert!(s.contains("PASS"));
        assert!(s.contains("FAIL"));
    }
}
