//! **The vocabulary of moves, declared rather than inferred.**
//!
//! A search over precision maps needs a set of named, reusable edits —
//! `E24`, `K25`, `M26` — and the question of *which* edits exist is not
//! one this module may answer by guessing. R5-F6 is why: neighbourhood 1
//! drew its in-moves from `{H, M23, K24}`, the candidates left
//! unpromoted at iteration 4, and missed `{E20,E22,E23,E24,E25}`
//! entirely. **In an exchange frame every never-admitted action is an
//! in-move, whenever it was last considered.** Two moves worth ~430 MB
//! each were invisible because the vocabulary had been mistaken for the
//! last round's leftovers.
//!
//! So the vocabulary is an input. This module holds it, fixes its order,
//! and turns an applied SET of edits into a map — nothing more.
//!
//! # Why a set, and how it becomes an ordered map
//!
//! [`super::super::map::PrecisionMap`] resolves exceptions in
//! declaration order, first match deciding, so a map is not a set. But a
//! *search state* is: `{E26, M26, K25}` names one map however the round
//! that built it happened to order its edits. The vocabulary's own
//! declaration order supplies the map order, so an applied set has
//! exactly one map and two rounds that reach the same set reach the same
//! bytes — which is what makes 1a's identity contract meaningful over
//! search states rather than only over hand-written maps.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::super::map::{Exception, PrecisionMap};
use crate::error::VindexError;

/// One named, reusable change to a precision map.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MapEdit {
    /// The programme's own name for it — `"E24"`, `"K25"`.
    pub name: String,
    /// What it puts into the map.
    pub exception: Exception,
}

impl MapEdit {
    pub fn new(name: impl Into<String>, exception: Exception) -> Self {
        Self {
            name: name.into(),
            exception,
        }
    }
}

/// **Every move the search may make**, in the order that fixes map
/// order.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ActionVocabulary {
    edits: Vec<MapEdit>,
}

impl ActionVocabulary {
    /// Build a vocabulary, refusing a repeated name.
    ///
    /// Two edits under one name would make an applied set ambiguous and
    /// the map it produces dependent on which was found first.
    pub fn new(edits: impl IntoIterator<Item = MapEdit>) -> Result<Self, VindexError> {
        let edits: Vec<MapEdit> = edits.into_iter().collect();
        let mut seen = BTreeSet::new();
        for edit in &edits {
            if !seen.insert(edit.name.as_str()) {
                return Err(VindexError::Parse(format!(
                    "action `{}` is declared twice — an applied set naming it would resolve to \
                     whichever declaration was found first",
                    edit.name
                )));
            }
        }
        Ok(Self { edits })
    }

    pub fn edits(&self) -> &[MapEdit] {
        &self.edits
    }

    pub fn len(&self) -> usize {
        self.edits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.edits.iter().any(|e| e.name == name)
    }

    /// Every name, in declaration order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.edits.iter().map(|e| e.name.as_str())
    }

    /// **The map an applied set produces.**
    ///
    /// Exceptions are emitted in vocabulary order and prepended to the
    /// base map's own, so an applied edit always outranks a default the
    /// base map declared. Unknown names are refused rather than skipped:
    /// silently dropping one would hand back a map for a state that does
    /// not exist.
    pub fn map_for(
        &self,
        base: &PrecisionMap,
        applied: &BTreeSet<String>,
    ) -> Result<PrecisionMap, VindexError> {
        if let Some(unknown) = applied.iter().find(|n| !self.contains(n)) {
            return Err(VindexError::Parse(format!(
                "action `{unknown}` is not in this vocabulary — a state cannot apply a move the \
                 search does not have"
            )));
        }
        let mut exceptions: Vec<Exception> = self
            .edits
            .iter()
            .filter(|e| applied.contains(&e.name))
            .map(|e| e.exception.clone())
            .collect();
        exceptions.extend(base.exceptions.iter().cloned());
        Ok(PrecisionMap {
            name: base.name.clone(),
            encoding: base.encoding.clone(),
            roles: base.roles.clone(),
            exceptions,
        })
    }
}
