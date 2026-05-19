//! Prompt sets for accuracy testing.
//!
//! 100 diverse prompts spanning: factual, completion, reasoning,
//! code, arithmetic, scientific, geographic, conversational.
//!
//! Every prompt is tagged with a [`KnowledgeSource`] — `Parametric` for
//! the corpus here (knowledge that lives in the model's weights and is
//! recoverable from a short cue) vs. `InContext` for needle-style
//! corpora that plant a fact in the prompt itself (see
//! [`super::needle`]) and `Conflict` for prompts where the in-context
//! claim contradicts pretraining (see [`super::conflict`]).
//!
//! For KV-cache compression engines, the distinction matters: parametric
//! knowledge survives any K/V strategy (it's in the weights), while
//! in-context knowledge is exactly what's at risk under sliding-window,
//! residual-stream-replacement, quantised K/V, or boundary checkpoints.
//! Reporting one number without splitting hides whichever signal you
//! care about.

/// Origin of the answer a prompt is testing for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeSource {
    /// Answer lives in the model's weights. Recoverable from a short
    /// cue ("The capital of France is" → "Paris"). Used to verify that
    /// a KV-cache strategy doesn't break weight-level correctness.
    Parametric,
    /// Answer is planted in the prompt. The model must attend to the
    /// relevant span and reproduce it. Used to measure whether a
    /// strategy preserves long-context K/V state. See
    /// [`super::needle`].
    InContext,
    /// Prompt's in-context claim contradicts pretraining; the test
    /// scores whether the model follows the prompt or falls back to
    /// weights. See [`super::conflict`].
    Conflict,
}

/// A test prompt with expected top-1 token (or prefix thereof).
#[derive(Debug, Clone)]
pub struct TestPrompt {
    pub text: &'static str,
    pub expected_contains: &'static str,
    pub category: &'static str,
    /// Whether the expected answer is parametric (in weights),
    /// in-context (in the prompt), or a conflict between the two.
    pub knowledge_source: KnowledgeSource,
}

/// The Paris test — single pass/fail sanity check.
pub fn paris_test() -> TestPrompt {
    TestPrompt {
        text: "The capital of France is",
        expected_contains: "Paris",
        category: "factual",
        knowledge_source: KnowledgeSource::Parametric,
    }
}

/// 100 diverse prompts for top-1 match rate testing.
pub fn diverse_100() -> Vec<TestPrompt> {
    vec![
        // Factual: capitals (20)
        TestPrompt {
            text: "The capital of France is",
            expected_contains: "Paris",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Germany is",
            expected_contains: "Berlin",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Japan is",
            expected_contains: "Tokyo",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Italy is",
            expected_contains: "Rome",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Spain is",
            expected_contains: "Madrid",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Brazil is",
            expected_contains: "Bras",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Australia is",
            expected_contains: "Canberra",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Canada is",
            expected_contains: "Ottawa",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Egypt is",
            expected_contains: "Cairo",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of India is",
            expected_contains: "Delhi",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Mexico is",
            expected_contains: "Mexico",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Russia is",
            expected_contains: "Moscow",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of China is",
            expected_contains: "Beijing",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of South Korea is",
            expected_contains: "Seoul",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Turkey is",
            expected_contains: "Ankara",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Thailand is",
            expected_contains: "Bangkok",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Argentina is",
            expected_contains: "Buenos",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Sweden is",
            expected_contains: "Stockholm",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Norway is",
            expected_contains: "Oslo",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The capital of Poland is",
            expected_contains: "Warsaw",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        // Factual: people (10)
        TestPrompt {
            text: "Mozart was born in",
            expected_contains: "Salzburg",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Einstein was born in",
            expected_contains: "Ulm",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Shakespeare was born in",
            expected_contains: "Strat",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The Mona Lisa was painted by",
            expected_contains: "Leonardo",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The theory of relativity was developed by",
            expected_contains: "Einstein",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The first president of the United States was",
            expected_contains: "George",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Apple Inc. was co-founded by Steve",
            expected_contains: "Jobs",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The author of Harry Potter is J.K.",
            expected_contains: "Rowling",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Beethoven's first name was",
            expected_contains: "Ludwig",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Isaac Newton discovered",
            expected_contains: "grav",
            category: "factual",
            knowledge_source: KnowledgeSource::Parametric,
        },
        // Factual: science (10)
        TestPrompt {
            text: "Water freezes at",
            expected_contains: "0",
            category: "scientific",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The chemical symbol for gold is",
            expected_contains: "Au",
            category: "scientific",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The chemical formula for water is",
            expected_contains: "H",
            category: "scientific",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The speed of light is approximately",
            expected_contains: "3",
            category: "scientific",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The largest planet in our solar system is",
            expected_contains: "Jupiter",
            category: "scientific",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "DNA stands for deoxyribonucle",
            expected_contains: "ic",
            category: "scientific",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The atomic number of carbon is",
            expected_contains: "6",
            category: "scientific",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Photosynthesis converts sunlight into",
            expected_contains: "energy",
            category: "scientific",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The boiling point of water is",
            expected_contains: "100",
            category: "scientific",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The nearest star to Earth is the",
            expected_contains: "Sun",
            category: "scientific",
            knowledge_source: KnowledgeSource::Parametric,
        },
        // Factual: geography (10)
        TestPrompt {
            text: "The longest river in Africa is the",
            expected_contains: "Nile",
            category: "geographic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The tallest mountain in the world is",
            expected_contains: "Everest",
            category: "geographic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The largest ocean is the",
            expected_contains: "Pacific",
            category: "geographic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The Amazon River flows through",
            expected_contains: "Brazil",
            category: "geographic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The Sahara Desert is located in",
            expected_contains: "Africa",
            category: "geographic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The Great Wall of China is located in",
            expected_contains: "China",
            category: "geographic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The currency of Japan is the",
            expected_contains: "yen",
            category: "geographic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The currency of the United Kingdom is the",
            expected_contains: "pound",
            category: "geographic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The official language of Brazil is",
            expected_contains: "Portug",
            category: "geographic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The smallest continent is",
            expected_contains: "Australia",
            category: "geographic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        // Completion (10)
        TestPrompt {
            text: "To be or not to be, that is the",
            expected_contains: "question",
            category: "completion",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "I think, therefore I",
            expected_contains: "am",
            category: "completion",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "All that glitters is not",
            expected_contains: "gold",
            category: "completion",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "A journey of a thousand miles begins with a single",
            expected_contains: "step",
            category: "completion",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The early bird catches the",
            expected_contains: "worm",
            category: "completion",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Actions speak louder than",
            expected_contains: "words",
            category: "completion",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Rome was not built in a",
            expected_contains: "day",
            category: "completion",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Knowledge is",
            expected_contains: "power",
            category: "completion",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Practice makes",
            expected_contains: "perfect",
            category: "completion",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Where there is smoke, there is",
            expected_contains: "fire",
            category: "completion",
            knowledge_source: KnowledgeSource::Parametric,
        },
        // Arithmetic (10)
        TestPrompt {
            text: "2 + 2 =",
            expected_contains: "4",
            category: "arithmetic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "10 × 10 =",
            expected_contains: "100",
            category: "arithmetic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "100 / 4 =",
            expected_contains: "25",
            category: "arithmetic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The square root of 144 is",
            expected_contains: "12",
            category: "arithmetic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "15 + 27 =",
            expected_contains: "42",
            category: "arithmetic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "One dozen equals",
            expected_contains: "12",
            category: "arithmetic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "A century is",
            expected_contains: "100",
            category: "arithmetic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "One kilometer equals",
            expected_contains: "1",
            category: "arithmetic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "There are 60 seconds in a",
            expected_contains: "minute",
            category: "arithmetic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "There are 24 hours in a",
            expected_contains: "day",
            category: "arithmetic",
            knowledge_source: KnowledgeSource::Parametric,
        },
        // Code (10)
        TestPrompt {
            text: "In Python, to print 'hello' you write print(",
            expected_contains: "'",
            category: "code",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "In JavaScript, a variable is declared with let, const, or",
            expected_contains: "var",
            category: "code",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "HTML stands for Hyper",
            expected_contains: "Text",
            category: "code",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The HTTP status code for 'Not Found' is",
            expected_contains: "404",
            category: "code",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "In SQL, to select all columns you use SELECT",
            expected_contains: "*",
            category: "code",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Git is a distributed version",
            expected_contains: "control",
            category: "code",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "JSON stands for JavaScript Object",
            expected_contains: "Notation",
            category: "code",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The file extension for Python files is .",
            expected_contains: "py",
            category: "code",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "In CSS, to make text bold you use font-weight:",
            expected_contains: "bold",
            category: "code",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The command to list files in Linux is",
            expected_contains: "ls",
            category: "code",
            knowledge_source: KnowledgeSource::Parametric,
        },
        // Conversational (10)
        TestPrompt {
            text: "How are you today? I'm doing",
            expected_contains: "well",
            category: "conversational",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Thank you very much! You're",
            expected_contains: "welcome",
            category: "conversational",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Good morning! How did you",
            expected_contains: "sleep",
            category: "conversational",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "See you later! Have a great",
            expected_contains: "day",
            category: "conversational",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Happy birthday! How old are",
            expected_contains: "you",
            category: "conversational",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Sorry for the delay. I was",
            expected_contains: "busy",
            category: "conversational",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "What do you think about",
            expected_contains: "the",
            category: "conversational",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Let me know if you need any",
            expected_contains: "help",
            category: "conversational",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "I completely agree with",
            expected_contains: "you",
            category: "conversational",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "That's a really good",
            expected_contains: "point",
            category: "conversational",
            knowledge_source: KnowledgeSource::Parametric,
        },
        // Reasoning (10)
        TestPrompt {
            text: "If it rains, the ground gets",
            expected_contains: "wet",
            category: "reasoning",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The opposite of hot is",
            expected_contains: "cold",
            category: "reasoning",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The color of grass is",
            expected_contains: "green",
            category: "reasoning",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The day after Monday is",
            expected_contains: "Tuesday",
            category: "reasoning",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Ice is the solid form of",
            expected_contains: "water",
            category: "reasoning",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The month after January is",
            expected_contains: "February",
            category: "reasoning",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "Cats are a type of",
            expected_contains: "animal",
            category: "reasoning",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The sun rises in the",
            expected_contains: "east",
            category: "reasoning",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "The plural of child is",
            expected_contains: "children",
            category: "reasoning",
            knowledge_source: KnowledgeSource::Parametric,
        },
        TestPrompt {
            text: "A triangle has three",
            expected_contains: "side",
            category: "reasoning",
            knowledge_source: KnowledgeSource::Parametric,
        },
    ]
}

/// Short prompt set for quick validation (20 prompts).
pub fn quick_20() -> Vec<TestPrompt> {
    diverse_100().into_iter().step_by(5).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paris_prompt_is_well_formed() {
        let p = paris_test();
        assert!(p.text.contains("France"));
        assert_eq!(p.expected_contains, "Paris");
        assert_eq!(p.category, "factual");
    }

    #[test]
    fn diverse_100_is_at_least_100_prompts() {
        let prompts = diverse_100();
        // The "100" in the name is the target; the corpus is allowed to
        // grow but should never shrink below it (the runner caps to N
        // when sampling).
        assert!(
            prompts.len() >= 100,
            "diverse_100 should have ≥100 prompts, got {}",
            prompts.len()
        );
    }

    #[test]
    fn diverse_100_covers_multiple_categories() {
        let prompts = diverse_100();
        let categories: std::collections::HashSet<&str> =
            prompts.iter().map(|p| p.category).collect();
        // The doc claims 8 categories: factual, completion, reasoning,
        // code, arithmetic, scientific, geographic, conversational.
        // Don't pin the exact set (it may evolve), but require >=3.
        assert!(
            categories.len() >= 3,
            "expected diverse category coverage, got {categories:?}"
        );
    }

    #[test]
    fn diverse_100_every_prompt_has_expected_answer() {
        for p in diverse_100() {
            assert!(
                !p.text.is_empty(),
                "category {:?} prompt is empty",
                p.category
            );
            assert!(
                !p.expected_contains.is_empty(),
                "category {:?} prompt {:?} has no expected answer",
                p.category,
                p.text
            );
            assert!(
                !p.category.is_empty(),
                "prompt {:?} has no category",
                p.text
            );
        }
    }

    #[test]
    fn quick_20_is_subset_of_diverse_100() {
        let quick = quick_20();
        let full = diverse_100();
        assert!(quick.len() <= full.len());
        assert!(!quick.is_empty());
        // Every quick prompt should appear in the full set.
        for q in &quick {
            assert!(
                full.iter().any(|f| f.text == q.text),
                "quick_20 prompt {:?} not found in diverse_100",
                q.text
            );
        }
    }
}
