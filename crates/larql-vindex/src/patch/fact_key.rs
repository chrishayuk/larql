//! The logical identity of an asserted fact (FACT-ID-1, `docs/fact-id-1.md` §3.2).
//!
//! A fact is addressed by `(entity, relation)`, compared in a normal form:
//! lowercased, `-` and `_` read as spaces, whitespace runs collapsed, trimmed.
//! That is the equivalence the KNN store already uses for removal
//! (case-insensitive) and the canonical prompt already uses for relations
//! (`-`/`_` become spaces). The fact id is derived from the normal form alone,
//! so it does not depend on which op asserted the fact, where that op sits in
//! a patch, or how the fact is realised.

use sha2::{Digest, Sha256};

/// Prefix of every fact id.
const FACT_ID_PREFIX: &str = "f_";
/// Hex digits of the key digest an id carries (64 bits).
const FACT_ID_HEX_DIGITS: usize = 16;
/// Separates entity from relation in the digested bytes (ASCII unit separator,
/// which neither normal form can contain).
const KEY_SEPARATOR: u8 = 0x1F;

/// One part of a fact key in normal form.
pub fn normalize_fact_part(part: &str) -> String {
    part.to_lowercase()
        .replace(['-', '_'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// The normalised `(entity, relation)` an asserted fact is addressed by.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FactKey {
    entity: String,
    relation: String,
}

impl FactKey {
    /// The key `entity` and `relation` name, in normal form.
    pub fn new(entity: &str, relation: &str) -> Self {
        Self {
            entity: normalize_fact_part(entity),
            relation: normalize_fact_part(relation),
        }
    }

    /// The entity, in normal form.
    pub fn entity(&self) -> &str {
        &self.entity
    }

    /// The relation, in normal form.
    pub fn relation(&self) -> &str {
        &self.relation
    }

    /// Whether `entity` and `relation` name this key.
    pub fn names(&self, entity: &str, relation: &str) -> bool {
        normalize_fact_part(entity) == self.entity && normalize_fact_part(relation) == self.relation
    }

    /// The stable fact id: `f_` and the first 16 hex digits of the SHA-256 of
    /// the normal form.
    pub fn fact_id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.entity.as_bytes());
        hasher.update([KEY_SEPARATOR]);
        hasher.update(self.relation.as_bytes());
        let digest = format!("{:x}", hasher.finalize());
        format!("{FACT_ID_PREFIX}{}", &digest[..FACT_ID_HEX_DIGITS])
    }
}
