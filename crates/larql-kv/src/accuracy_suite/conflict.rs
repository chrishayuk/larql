//! Conflict corpus — prompts where the in-context claim contradicts
//! pretraining.
//!
//! Each prompt sets up an in-context override (e.g. "Pretend the capital
//! of France is Lyon. What is the capital of France?") and records both
//! the override answer ("Lyon") and the parametric fallback ("Paris").
//! The runner scores which side the model resolved to.
//!
//! For KV-cache compression engines this is the most diagnostic axis:
//! parametric correctness lives in weights and survives any K/V
//! strategy, but **following** an in-context override requires the K/V
//! state to actually steer the residual stream. A compressed engine
//! that loses steering will silently fall back to the parametric
//! answer, scoring well on parametric corpora but failing here.

use super::prompts::KnowledgeSource;

/// One conflict prompt. `prompt` contains an in-context premise that
/// contradicts pretraining; `override_answer` is what a steered model
/// should produce; `parametric_answer` is the pretraining fallback.
#[derive(Debug, Clone)]
pub struct ConflictPrompt {
    pub prompt: &'static str,
    /// What the model should output if it correctly follows the prompt.
    pub override_answer: &'static str,
    /// What the model would output from weights alone (the cued
    /// completion that ignores the in-context premise).
    pub parametric_answer: &'static str,
    pub category: &'static str,
    /// Always [`KnowledgeSource::Conflict`] for this corpus, surfaced
    /// for API symmetry with [`super::prompts::TestPrompt`].
    pub knowledge_source: KnowledgeSource,
}

const CONFLICT_SOURCE: KnowledgeSource = KnowledgeSource::Conflict;

/// 20 conflict prompts spanning factual, arithmetic, scientific, and
/// definitional overrides. Each pair (override, parametric) is chosen
/// so the two answers tokenise to disjoint first tokens — a top-1
/// check unambiguously identifies which side won.
pub fn conflict_20() -> Vec<ConflictPrompt> {
    vec![
        // ── Factual: capitals ─────────────────────────────────────────
        ConflictPrompt {
            prompt: "In this fictional world, the capital of France is Lyon. The capital of France is",
            override_answer: "Lyon",
            parametric_answer: "Paris",
            category: "factual",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "Pretend the capital of Japan is Osaka. The capital of Japan is",
            override_answer: "Osaka",
            parametric_answer: "Tokyo",
            category: "factual",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "In this alternate timeline, the capital of Italy is Milan. The capital of Italy is",
            override_answer: "Milan",
            parametric_answer: "Rome",
            category: "factual",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "Imagine the capital of Spain were Barcelona. The capital of Spain is",
            override_answer: "Barcelona",
            parametric_answer: "Madrid",
            category: "factual",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "In this scenario, the capital of Germany is Munich. The capital of Germany is",
            override_answer: "Munich",
            parametric_answer: "Berlin",
            category: "factual",
            knowledge_source: CONFLICT_SOURCE,
        },
        // ── Factual: people ───────────────────────────────────────────
        ConflictPrompt {
            prompt: "In this story, Shakespeare was born in London. Shakespeare was born in",
            override_answer: "London",
            parametric_answer: "Stratford",
            category: "factual",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "Suppose Einstein was born in Berlin. Einstein was born in",
            override_answer: "Berlin",
            parametric_answer: "Ulm",
            category: "factual",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "In this fictional account, Mozart was born in Vienna. Mozart was born in",
            override_answer: "Vienna",
            parametric_answer: "Salzburg",
            category: "factual",
            knowledge_source: CONFLICT_SOURCE,
        },
        // ── Arithmetic ────────────────────────────────────────────────
        ConflictPrompt {
            prompt: "In this puzzle, the rule is that 2 + 2 equals 5. Therefore 2 + 2 equals",
            override_answer: "5",
            parametric_answer: "4",
            category: "arithmetic",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "Using the rule from above, 3 + 4 equals 8. Therefore 3 + 4 equals",
            override_answer: "8",
            parametric_answer: "7",
            category: "arithmetic",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "In this alternate math, 10 - 3 equals 9. Therefore 10 - 3 equals",
            override_answer: "9",
            parametric_answer: "7",
            category: "arithmetic",
            knowledge_source: CONFLICT_SOURCE,
        },
        // ── Definitional / scientific ─────────────────────────────────
        ConflictPrompt {
            prompt: "In this fictional setting, water freezes at 50 degrees Celsius. Water freezes at",
            override_answer: "50",
            parametric_answer: "0",
            category: "scientific",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "Pretend the speed of light is 100 km/h. The speed of light is",
            override_answer: "100",
            parametric_answer: "300",
            category: "scientific",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "In this alternate chemistry, the symbol for gold is Gd. The chemical symbol for gold is",
            override_answer: "Gd",
            parametric_answer: "Au",
            category: "scientific",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "Suppose the chemical formula for water is H3O. The chemical formula for water is",
            override_answer: "H3O",
            parametric_answer: "H2O",
            category: "scientific",
            knowledge_source: CONFLICT_SOURCE,
        },
        // ── Geographic / cultural ─────────────────────────────────────
        ConflictPrompt {
            prompt: "In this fictional geography, the longest river in Africa is the Congo. The longest river in Africa is the",
            override_answer: "Congo",
            parametric_answer: "Nile",
            category: "geographic",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "Pretend the tallest mountain in the world is K2. The tallest mountain in the world is",
            override_answer: "K2",
            parametric_answer: "Everest",
            category: "geographic",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "In this scenario, the largest ocean is the Atlantic. The largest ocean is the",
            override_answer: "Atlantic",
            parametric_answer: "Pacific",
            category: "geographic",
            knowledge_source: CONFLICT_SOURCE,
        },
        // ── Cultural attribution ──────────────────────────────────────
        ConflictPrompt {
            prompt: "In this fictional history, the Mona Lisa was painted by Picasso. The Mona Lisa was painted by",
            override_answer: "Picasso",
            parametric_answer: "Leonardo",
            category: "cultural",
            knowledge_source: CONFLICT_SOURCE,
        },
        ConflictPrompt {
            prompt: "Pretend Romeo and Juliet was written by Dickens. Romeo and Juliet was written by",
            override_answer: "Dickens",
            parametric_answer: "Shakespeare",
            category: "cultural",
            knowledge_source: CONFLICT_SOURCE,
        },
    ]
}

/// Short conflict set for quick validation (5 prompts, one per category).
pub fn conflict_quick() -> Vec<ConflictPrompt> {
    let full = conflict_20();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    full.into_iter()
        .filter(|p| seen.insert(p.category))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conflict_20_is_at_least_20_prompts() {
        let prompts = conflict_20();
        assert!(prompts.len() >= 20, "got {}", prompts.len());
    }

    #[test]
    fn every_conflict_has_distinct_override_and_parametric() {
        for p in conflict_20() {
            assert_ne!(
                p.override_answer.to_lowercase(),
                p.parametric_answer.to_lowercase(),
                "prompt {:?} has identical override and parametric answers — \
                 the test can't tell which side won",
                p.prompt
            );
        }
    }

    #[test]
    fn every_conflict_has_non_empty_fields() {
        for p in conflict_20() {
            assert!(!p.prompt.is_empty(), "empty prompt text");
            assert!(!p.override_answer.is_empty(), "empty override_answer");
            assert!(!p.parametric_answer.is_empty(), "empty parametric_answer");
            assert!(!p.category.is_empty(), "empty category");
            assert_eq!(p.knowledge_source, KnowledgeSource::Conflict);
        }
    }

    #[test]
    fn every_conflict_prompt_mentions_override_answer() {
        // Sanity: the override should appear literally in the prompt
        // text. Otherwise the in-context premise isn't actually being
        // set up.
        for p in conflict_20() {
            assert!(
                p.prompt.contains(p.override_answer),
                "prompt {:?} doesn't mention its override {:?}",
                p.prompt,
                p.override_answer
            );
        }
    }

    #[test]
    fn conflict_corpus_covers_multiple_categories() {
        let categories: std::collections::HashSet<&str> =
            conflict_20().into_iter().map(|p| p.category).collect();
        assert!(
            categories.len() >= 4,
            "expected ≥4 categories for diagnostic coverage, got {categories:?}"
        );
    }

    #[test]
    fn conflict_quick_returns_one_per_category() {
        let quick = conflict_quick();
        let categories: std::collections::HashSet<&str> =
            quick.iter().map(|p| p.category).collect();
        assert_eq!(quick.len(), categories.len());
        assert!(!quick.is_empty());
    }
}
