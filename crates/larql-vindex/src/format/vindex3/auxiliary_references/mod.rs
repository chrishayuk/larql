//! **The auxiliary reference table** — which represented object stands for
//! a codec's declared dependency.
//!
//! A codec declares that it NEEDS an auxiliary by name (`codebook`); this
//! table says where that name points, per owner operand. Two rules make it
//! a reference rather than a convention:
//!
//! * it is keyed by `(owner, auxiliary name)`, so ONE TARGET MAY SERVE
//!   MANY OWNERS — a shared codebook is the ordinary case here, not a
//!   special one; and
//! * nothing is derived from a tensor's spelling. A sibling stream
//!   (`<tensor>.<stream>`) is a NAME rule for the streams of one tensor,
//!   and it cannot express sharing: a dependency is an address, and an
//!   address that could be guessed from a name could not be shared.
//!
//! It carries only the ADDRESS of its target. Shape and dtype stay the
//! segment's to state, because a table that repeated them would be a
//! second authority able to disagree with the container it describes.
//!
//! Versioned on its own and optional: every container written before this
//! table existed simply has none, and the absence means *no dependency is
//! declared* — never *the dependencies are somewhere else*.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::VindexError;

pub mod closure;

#[cfg(test)]
mod tests;

/// The table's own schema. Independent of the index's: a container may
/// gain dependencies without its index generation moving.
pub const AUXILIARY_REFERENCES_SCHEMA: u32 = 1;

/// Where an operand lives — and nothing else.
///
/// Deliberately not an `OperandRef`: that carries dtype and shape, which
/// the segment already states. A reference addresses; it does not
/// describe.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OperandAddress {
    pub object: String,
    pub tensor: String,
}

impl OperandAddress {
    pub fn new(object: impl Into<String>, tensor: impl Into<String>) -> Self {
        Self {
            object: object.into(),
            tensor: tensor.into(),
        }
    }

    /// How a refusal spells an address.
    pub fn describe(&self) -> String {
        format!("`{}`/`{}`", self.object, self.tensor)
    }

    fn judge(&self, what: &str) -> Result<(), VindexError> {
        if self.object.trim().is_empty() || self.tensor.trim().is_empty() {
            return Err(VindexError::Parse(format!(
                "auxiliary reference table: {what} {} names an empty object or tensor",
                self.describe()
            )));
        }
        Ok(())
    }
}

/// One declared dependency, as the container stores it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuxiliaryReference {
    pub owner: OperandAddress,
    /// The name the OWNER's codec declared for this dependency. The table
    /// does not judge whether the codec declares it — that needs the
    /// codec, and this file has no registry.
    pub auxiliary: String,
    pub target: OperandAddress,
}

/// The table as the container stores it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuxiliaryReferences {
    /// Always [`AUXILIARY_REFERENCES_SCHEMA`] when this build wrote it.
    pub schema: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<AuxiliaryReference>,
}

impl AuxiliaryReferences {
    pub fn new(references: Vec<AuxiliaryReference>) -> Self {
        Self {
            schema: AUXILIARY_REFERENCES_SCHEMA,
            references,
        }
    }

    /// Judge the stored table into one that can be asked questions.
    ///
    /// Four refusals, each about a row and none about a codec: a schema
    /// this build does not implement, an empty address or name, a target
    /// that is its own owner, and the same `(owner, auxiliary)` declared
    /// twice. The last of those refuses rather than keeping the last entry
    /// written, because "which of the two did the encoder mean" has no
    /// answer, and silently picking one would make a container's meaning
    /// depend on its serialisation order.
    pub fn judge(&self) -> Result<ReferenceTable, VindexError> {
        if self.schema != AUXILIARY_REFERENCES_SCHEMA {
            return Err(VindexError::Parse(format!(
                "auxiliary reference table: schema {} was written by another build; this one \
                 implements {AUXILIARY_REFERENCES_SCHEMA}",
                self.schema
            )));
        }
        let mut by_key: BTreeMap<(OperandAddress, String), OperandAddress> = BTreeMap::new();
        for reference in &self.references {
            reference.owner.judge("owner")?;
            reference.target.judge("target")?;
            if reference.auxiliary.trim().is_empty() {
                return Err(VindexError::Parse(format!(
                    "auxiliary reference table: the reference from {} names an empty auxiliary",
                    reference.owner.describe()
                )));
            }
            if reference.owner == reference.target {
                return Err(VindexError::Parse(format!(
                    "auxiliary reference table: {} declares itself as its own `{}`; an operand \
                     cannot be its own dependency",
                    reference.owner.describe(),
                    reference.auxiliary
                )));
            }
            let key = (reference.owner.clone(), reference.auxiliary.clone());
            if let Some(first) = by_key.get(&key) {
                return Err(VindexError::Parse(format!(
                    "auxiliary reference table: {} declares `{}` twice — {} and {}; a container \
                     that names two targets for one dependency states no dependency",
                    reference.owner.describe(),
                    reference.auxiliary,
                    first.describe(),
                    reference.target.describe(),
                )));
            }
            by_key.insert(key, reference.target.clone());
        }
        Ok(ReferenceTable { by_key })
    }

    /// Read and judge the table `name` names, relative to `root`.
    pub fn read(root: &Path, name: &str) -> Result<ReferenceTable, VindexError> {
        let path = root.join(name);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            VindexError::Parse(format!(
                "auxiliary reference table `{}` is named by the index and cannot be read: {e}",
                path.display()
            ))
        })?;
        let stored: Self = serde_json::from_str(&text).map_err(|e| {
            VindexError::Parse(format!(
                "auxiliary reference table `{}` is not readable as one: {e}",
                path.display()
            ))
        })?;
        stored.judge()
    }
}

/// The table after it has been judged: at most one target per
/// `(owner, auxiliary name)`, and every address non-empty.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReferenceTable {
    by_key: BTreeMap<(OperandAddress, String), OperandAddress>,
}

impl ReferenceTable {
    /// What a container with no table has: no declared dependency.
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }

    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    /// The target `owner` declares for `auxiliary`, if it declares one.
    pub fn target(&self, owner: &OperandAddress, auxiliary: &str) -> Option<&OperandAddress> {
        self.by_key.get(&(owner.clone(), auxiliary.to_string()))
    }

    /// Every dependency `owner` declares, by name, in name order — what a
    /// closure walk reads and what a refusal lists.
    pub fn auxiliaries_of(&self, owner: &OperandAddress) -> Vec<(&str, &OperandAddress)> {
        self.by_key
            .iter()
            .filter(|((declared, _), _)| declared == owner)
            .map(|((_, name), target)| (name.as_str(), target))
            .collect()
    }

    /// Whether anything declares `(object, tensor)` as a dependency.
    ///
    /// What the closure pass asks: a tensor nothing plans and nothing
    /// references has no fate, and one that IS referenced is accounted
    /// for by whoever requires it.
    pub fn is_referenced(&self, object: &str, tensor: &str) -> bool {
        self.by_key
            .values()
            .any(|target| target.object == object && target.tensor == tensor)
    }

    /// Every owner that declares `target` as one of its dependencies —
    /// the sharing question, asked from the other end.
    pub fn owners_of(&self, target: &OperandAddress) -> Vec<&OperandAddress> {
        self.by_key
            .iter()
            .filter(|(_, declared)| *declared == target)
            .map(|((owner, _), _)| owner)
            .collect()
    }

    /// The rows, for a writer that round-trips a table it read.
    pub fn stored(&self) -> AuxiliaryReferences {
        AuxiliaryReferences::new(
            self.by_key
                .iter()
                .map(|((owner, auxiliary), target)| AuxiliaryReference {
                    owner: owner.clone(),
                    auxiliary: auxiliary.clone(),
                    target: target.clone(),
                })
                .collect(),
        )
    }
}
