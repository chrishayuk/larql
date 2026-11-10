//! `SELECT * FROM ENTITIES` — distinct entity-like top tokens
//! across the loaded vindex.
//!
//! The "entity-like" filter is heuristic: capitalised, 3+ chars,
//! all alphabetic, not in `STOP_WORDS`. The stop-word list intends
//! to catch sentence-initial English function words and small
//! numerals that look like proper nouns by accident.

use crate::ast::Condition;
use crate::error::LqlError;
use crate::executor::predicate::{EdgePredicate, PredicateScope};
use crate::executor::Session;

use super::format::ENTITIES_DEFAULT_LIMIT;

/// Common English function/closed-class words that get capitalised
/// at sentence starts but aren't named entities. Pulling this out as
/// a `const` makes the heuristic auditable and keeps the verb body
/// focused on the scan.
const STOP_WORDS: &[&str] = &[
    "The", "For", "And", "But", "Not", "This", "That", "With", "From", "Into", "Will", "Can",
    "One", "All", "Any", "Has", "Had", "Was", "Are", "Were", "Been", "His", "Her", "Its", "Our",
    "Who", "How", "Why", "When", "What", "Where", "Which", "Each", "Both", "Some", "Most", "Many",
    "Much", "More", "Such", "Than", "Then", "Also", "Just", "Now", "May", "Per", "Pre", "Pro",
    "Con", "Dis", "Via", "Yet", "Nor", "Should", "Would", "Could", "Did", "Does", "Too", "Very",
    "Instead", "Mon", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "First",
    "Second", "Third", "Fourth", "Fifth", "Sixth", "Forty", "Fifty", "Only", "Over", "Under",
    "After", "Before", "About", "Above", "Below", "Between", "Through",
];

/// Minimum length for a token to qualify as an "entity-like" name.
/// Anything shorter is overwhelmingly punctuation, single letters, or
/// pronouns and would dominate the output.
const MIN_ENTITY_LEN: usize = 3;

/// Decide whether a feature's top token looks like a named entity
/// for purposes of `SELECT * FROM ENTITIES`.
fn looks_like_entity(tok: &str) -> bool {
    if tok.len() < MIN_ENTITY_LEN {
        return false;
    }
    let first = match tok.chars().next() {
        Some(c) => c,
        None => return false,
    };
    if !first.is_ascii_uppercase() {
        return false;
    }
    if !tok.chars().all(|c| c.is_alphabetic()) {
        return false;
    }
    if STOP_WORDS.contains(&tok) {
        return false;
    }
    true
}

impl Session {
    pub(crate) fn exec_select_entities(
        &self,
        conditions: &[Condition],
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let ctx = self.browse()?;

        let predicate = EdgePredicate::parse(conditions, PredicateScope::ENTITIES)?;
        let layer_filter = predicate.layer;
        let entity_filter = predicate.entity;
        let limit = limit.unwrap_or(ENTITIES_DEFAULT_LIMIT) as usize;

        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l]
        } else {
            (0..ctx.num_layers).collect()
        };

        // Aggregate: token → (occurrence count, max c_score across layers).
        let mut entity_counts: std::collections::HashMap<String, (usize, f32)> =
            std::collections::HashMap::new();

        for layer in &scan_layers {
            let nf = ctx.source.num_features(*layer);
            for feat in 0..nf {
                if let Some(meta) = ctx.source.feature_meta(*layer, feat) {
                    let tok = meta.top_token.trim().to_string();
                    if !looks_like_entity(&tok) {
                        continue;
                    }
                    if let Some(ef) = entity_filter {
                        if !tok.to_lowercase().contains(&ef.to_lowercase()) {
                            continue;
                        }
                    }
                    let entry = entity_counts.entry(tok).or_insert((0, 0.0));
                    entry.0 += 1;
                    if meta.c_score > entry.1 {
                        entry.1 = meta.c_score;
                    }
                }
            }
        }

        let mut entities: Vec<(String, usize, f32)> = entity_counts
            .into_iter()
            .map(|(tok, (count, max_score))| (tok, count, max_score))
            .collect();
        // Sort by occurrence count descending — entities that appear at
        // many layers come first as the most "load-bearing".
        entities.sort_by_key(|(_, count, _)| std::cmp::Reverse(*count));
        entities.truncate(limit);

        let mut out = Vec::new();
        out.push(format!(
            "{:<24} {:>10} {:>10}",
            "Entity", "Features", "Max Score"
        ));
        out.push(super::format::banner(48));

        for (tok, count, max_score) in &entities {
            out.push(format!("{:<24} {:>10} {:>10.4}", tok, count, max_score));
        }

        if entities.is_empty() {
            out.push("  (no entities found)".into());
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_like_entity_accepts_capitalised_alphabetic() {
        assert!(looks_like_entity("Paris"));
        assert!(looks_like_entity("Atlantis"));
        assert!(looks_like_entity("Köln"));
    }

    #[test]
    fn looks_like_entity_rejects_short_tokens() {
        assert!(!looks_like_entity("Hi"));
        assert!(!looks_like_entity("X"));
        assert!(!looks_like_entity(""));
    }

    #[test]
    fn looks_like_entity_rejects_lowercase_initial() {
        assert!(!looks_like_entity("paris"));
    }

    #[test]
    fn looks_like_entity_rejects_non_alphabetic() {
        assert!(!looks_like_entity("Paris2"));
        assert!(!looks_like_entity("E=mc"));
    }

    #[test]
    fn looks_like_entity_rejects_stop_words() {
        for word in STOP_WORDS {
            assert!(
                !looks_like_entity(word),
                "stop word should be rejected: {word}"
            );
        }
    }

    #[test]
    fn min_entity_len_is_three() {
        // Pinned for the test cases above; if you change this, audit
        // the rejection tests so they still cover the boundary.
        assert_eq!(MIN_ENTITY_LEN, 3);
    }
}
