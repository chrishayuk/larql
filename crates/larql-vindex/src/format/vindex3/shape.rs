//! **Which top-level shape a VINDEX3 container has** (ADR-0027).
//!
//! The graph container is the sole normative VINDEX3 3.0 shape. The bank
//! shape — a routed-programme manifest and LYRW segments with no system
//! graph — is legacy input: a conforming reader recognises it and either
//! opens it or refuses it *by name*, and never reports it as a conforming
//! 3.0 container. Every surface that has to tell the two apart asks
//! [`Vindex3Index::shape`], so the discrimination and the words used for
//! it have one owner.
//!
//! The discriminator is the envelope's own declarations — which authority
//! `index.json` names — never a directory sniff (candidate spec §5.5).

use super::index::Vindex3Index;
use crate::error::VindexError;

/// How to turn a legacy bank container into a VINDEX3 3.0 one, and where
/// it remains readable meanwhile. Re-extraction from the source checkpoint
/// is the migration path until a re-encode witness exists (ADR-0027
/// closure criterion 3).
pub const LEGACY_BANK_MIGRATION: &str = "re-extract it from its source checkpoint with \
     `larql vindex3 encode` or `larql extract-index --generation v3`; the bank form stays \
     readable by `larql show`, `larql verify` and `run --routed-from`";

/// A container's top-level shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerShape {
    /// Carries a system graph: the normative VINDEX3 3.0 shape. A
    /// routed-programme manifest beside it is subordinate — the graph is
    /// the semantic authority (candidate spec §5.5).
    Graph,
    /// A routed-programme manifest with no system graph: the legacy bank
    /// shape, produced only by `extract-index --expert-banks-out`.
    LegacyBank,
}

impl ContainerShape {
    /// The shape as a surface names it to a person.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Graph => "graph (normative VINDEX3 3.0)",
            Self::LegacyBank => {
                "legacy bank (moe_manifest + LYRW segments, no system graph) — \
                 not a VINDEX3 3.0 container (ADR-0027)"
            }
        }
    }

    pub fn is_normative(self) -> bool {
        matches!(self, Self::Graph)
    }
}

impl Vindex3Index {
    /// This container's shape, from the authorities `index.json` names.
    ///
    /// An index naming neither a system graph nor a routed-programme
    /// manifest describes no container at all, and is refused rather than
    /// defaulted to either shape.
    pub fn shape(&self) -> Result<ContainerShape, VindexError> {
        match (&self.system_graph, &self.moe_manifest) {
            (Some(_), _) => Ok(ContainerShape::Graph),
            (None, Some(_)) => Ok(ContainerShape::LegacyBank),
            (None, None) => Err(VindexError::Parse(
                "VINDEX3 index.json names neither a system graph nor a routed-programme \
                 manifest; a container carries at least one (candidate spec §5.5)"
                    .into(),
            )),
        }
    }
}

#[cfg(test)]
#[path = "shape_tests.rs"]
mod tests;
