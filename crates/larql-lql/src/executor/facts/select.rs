//! The asserted block of `SELECT FROM EDGES` (`docs/fact-id-1.md` §5.2).
//!
//! For `WHERE entity = … AND relation = …` alone, an asserted fact with that
//! key is listed above the discovered rows, which are printed exactly as
//! before under their own heading. With no asserted fact, or any other clause,
//! SELECT's output is unchanged.

use larql_vindex::FactKey;

use crate::ast::Condition;
use crate::executor::Session;

use super::{semantic_pair, AssertedFact};

/// Heading of the asserted block.
const ASSERTED_HEADING: &str = "Asserted facts:";
/// Heading put over the discovered listing when an asserted block precedes it.
const DISCOVERED_HEADING: &str = "Discovered edges:";

impl Session {
    /// The asserted fact a SELECT clause addresses by identity, if any.
    pub(crate) fn selected_fact(&self, conditions: &[Condition]) -> Option<AssertedFact> {
        let (entity, relation) = semantic_pair(conditions).ok().flatten()?;
        let ledger = self.ledger().ok()?;
        let single = |t: &str, id: u32| ledger.is_single(t, id);
        ledger
            .catalogue()
            .live(&FactKey::new(&entity, &relation), &single)
    }
}

/// `discovered` with the asserted block for `fact` above it.
pub(crate) fn with_asserted_block(fact: &AssertedFact, discovered: Vec<String>) -> Vec<String> {
    let mut out = vec![
        ASSERTED_HEADING.to_string(),
        format!(
            "  {:<19} {:<16} {:<16} {:<16} {:<11} {:<13} {:>7}  {}",
            "fact",
            "entity",
            "relation",
            "target",
            "realization",
            "status",
            "version",
            "durability"
        ),
        format!(
            "  {:<19} {:<16} {:<16} {:<16} {:<11} {:<13} {:>7}  {}",
            fact.id,
            fact.entity,
            fact.relation,
            fact.target,
            fact.realization.label(),
            fact.status.label(),
            fact.version,
            fact.durability.label()
        ),
        DISCOVERED_HEADING.to_string(),
    ];
    out.extend(discovered);
    out
}
