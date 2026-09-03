//! **Which rules turned facts into conclusions.**
//!
//! 1c drew a line between an observation and its meaning so that a later
//! contract could reinterpret evidence already held. Persistence forces
//! the same line one level up, and it is easy to miss: six months from
//! now `decide_promotion` legitimately changes, an old snapshot replays,
//! and the answer differs while every stored measurement is intact.
//!
//! **That is not corruption. The decision procedure changed.** A replay
//! that could not say which of the two had happened would make every
//! improvement to the search look like a data problem.
//!
//! ```text
//! observation replay          same facts, CURRENT rules
//!                             → "what would we conclude today?"
//!
//! historical decision replay  same facts, the rules OF THE TIME
//!                             → "what did we conclude, and why?"
//! ```
//!
//! So a snapshot records the semantics its conclusions were originally
//! drawn under, and a replay can compare.
//!
//! # Version identities, not source hashes
//!
//! Each field names a normative version — `"physical-dominance/v1"` —
//! and not a commit or a binary digest. The reasoning is
//! [`super::instrument`]'s, applied to decision procedures rather than
//! to measurement: a refactor that changes no rule must not invalidate a
//! scientific record, and a rule change must be visible even when the
//! diff is one line. The same limit applies too — a declaration can lie,
//! and only construction from the rule's own constant can close that.

use serde::{Deserialize, Serialize};

use super::super::compile::hash_bytes;
use super::surface::{FIELD, SECTION};

/// The canonical-form version every semantics id is computed under.
pub const SEARCH_SEMANTICS_ID_VERSION: &str = "search-semantics-id/v1";

/// **What rules a conclusion was drawn under.**
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SearchSemanticsId(String);

impl SearchSemanticsId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn short(&self) -> &str {
        &self.0[..self.0.len().min(12)]
    }
}

impl std::fmt::Display for SearchSemanticsId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The five decision procedures between a fact and a conclusion.
///
/// Named separately because they change independently and for different
/// reasons — Ruling 1 rewrote pruning without touching evidence
/// interpretation, and R5-F4 rewrote evidence interpretation three times
/// without touching pruning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchSemantics {
    /// How candidate states are proposed.
    pub candidate_generation: String,
    /// What may be pruned before it is measured. Ruling 1's list of
    /// four, and no fifth.
    pub pre_measurement_pruning: String,
    /// How a statistic may be used — the `SearchEvidence` ladder.
    pub evidence_interpretation: String,
    /// How candidates become admitted maps.
    ///
    /// Named `promotion_rule` and not `promotion` on purpose: the
    /// structural check that a stored snapshot carries no CONCLUSION
    /// looks for a key called `promotion`, and a rule identity must not
    /// be mistaken for a decision. The name is the disambiguation, so
    /// the check can stay blunt instead of growing an exemption.
    pub promotion_rule: String,
    /// How bytes are counted.
    pub physical_accounting: String,
}

impl SearchSemantics {
    pub fn new(
        candidate_generation: impl Into<String>,
        pre_measurement_pruning: impl Into<String>,
        evidence_interpretation: impl Into<String>,
        promotion: impl Into<String>,
        physical_accounting: impl Into<String>,
    ) -> Self {
        Self {
            candidate_generation: candidate_generation.into(),
            pre_measurement_pruning: pre_measurement_pruning.into(),
            evidence_interpretation: evidence_interpretation.into(),
            promotion_rule: promotion.into(),
            physical_accounting: physical_accounting.into(),
        }
    }

    /// What these rules ARE.
    pub fn id(&self) -> SearchSemanticsId {
        let input = format!(
            "{SEARCH_SEMANTICS_ID_VERSION}{SECTION}generation={}{FIELD}pruning={}{FIELD}\
             evidence={}{FIELD}promotion={}{FIELD}physical={}",
            self.candidate_generation,
            self.pre_measurement_pruning,
            self.evidence_interpretation,
            self.promotion_rule,
            self.physical_accounting
        );
        SearchSemanticsId(hash_bytes(input.as_bytes()))
    }
}
