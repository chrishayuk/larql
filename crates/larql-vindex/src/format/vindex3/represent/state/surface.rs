//! **The tensor surface a map resolves against.**
//!
//! A surface is the container's own answer to *what tensors exist and
//! what is each one for*, in the identity the container itself uses:
//! `(object, tensor)`. That pairing is not a convenience — it is what
//! [`super::super::plan_roles::PlanRoles`] is keyed by and what
//! [`super::super::compile::CompilationLedger`] seals against, so a
//! search state built on anything weaker would be identifying tensors
//! differently from the two systems that materialise them.
//!
//! **Not "sort by tensor name".** A tensor name is unique within an
//! object and nowhere else; `weight` occurs in almost every object a
//! container holds. Ordering by name alone would let the surface's
//! canonical form depend on which object happened to be enumerated
//! first, which is incidental file order — exactly the kind of thing
//! that becomes an identity bug six weeks later.
//!
//! # What a surface entry carries
//!
//! ```text
//! object    the container's object id       identity
//! tensor    the tensor name within it       identity
//! role      what the operator computes with it
//! shape     what the storage layout has to hold
//! ```
//!
//! Role and shape are in the surface rather than derived at resolution
//! time because both are the *container's* judgement and both change
//! what a map resolves to. A tensor reclassified from `decoder-linear`
//! to `unknown` is a different surface even where the resolved encoding
//! is unchanged for both, because the set of maps that would compile it
//! is different. The surface is the model, and the model moved.
//!
//! # Aliases
//!
//! Two objects may resolve to the same bytes — a tied embedding and
//! output head being the obvious case. **They are two surface entries,
//! not one.** A map resolves per `(object, tensor)` and may legitimately
//! decide differently for each, and REPRESENT compiles one pack per
//! object; collapsing them would make identity claim an agreement the
//! compiler does not enforce. Aliasing is a fact about payload bytes,
//! and payload bytes are already covered by the model identity's
//! per-segment hashes.

use serde::{Deserialize, Serialize};

use super::super::compile::hash_bytes;
use super::super::policy::Role;
use crate::error::VindexError;

/// Separates fields inside one canonical record.
///
/// `\u{1}` is not a legal character in a tensor or object name, so no
/// name can forge a field boundary — the same separator
/// [`super::super::compile::CompilationLedger`] keys its seals with.
pub(crate) const FIELD: char = '\u{1}';

/// Separates canonical records from each other.
pub(crate) const RECORD: char = '\u{2}';

/// Separates the top-level sections of a digest input.
pub(crate) const SECTION: char = '\u{3}';

/// One tensor on the surface, identified the way the container
/// identifies it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceTensor {
    /// The container's object id.
    pub object: String,
    /// The tensor's name within that object.
    pub tensor: String,
    /// The role its operator computes with, from the plan where the plan
    /// binds it and from name classification otherwise. Whichever
    /// answered, it is the container's judgement and not this module's.
    pub role: Role,
    pub shape: Vec<usize>,
}

impl SurfaceTensor {
    pub fn new(
        object: impl Into<String>,
        tensor: impl Into<String>,
        role: Role,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            object: object.into(),
            tensor: tensor.into(),
            role,
            shape,
        }
    }

    /// The identity pair. Ordering and lookup use this and nothing else.
    pub fn key(&self) -> (&str, &str) {
        (&self.object, &self.tensor)
    }

    /// This entry's canonical record.
    ///
    /// The shape is rendered element-wise rather than through `Debug` so
    /// the canonical form is a stated format rather than whatever
    /// `{:?}` prints this release.
    pub(crate) fn canonical(&self) -> String {
        let shape = self
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        format!(
            "{}{FIELD}{}{FIELD}{}{FIELD}{shape}",
            self.object,
            self.tensor,
            self.role.name()
        )
    }
}

/// **Every tensor of one model, in a deterministic order.**
///
/// Construction sorts, so a caller may build a surface in whatever order
/// it enumerated the container and still obtain the same identity.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TensorSurface {
    entries: Vec<SurfaceTensor>,
}

impl TensorSurface {
    /// Build a surface, refusing a repeated `(object, tensor)`.
    ///
    /// A container cannot present one tensor twice, so a duplicate is a
    /// bug in whatever built the surface. Deduplicating silently would
    /// let two disagreeing shapes collapse into whichever arrived last
    /// and hand back an identity for a model that does not exist; the
    /// refusal names the pair so the caller can find it.
    pub fn new(entries: impl IntoIterator<Item = SurfaceTensor>) -> Result<Self, VindexError> {
        let mut entries: Vec<SurfaceTensor> = entries.into_iter().collect();
        entries.sort_by(|a, b| a.key().cmp(&b.key()));
        if let Some(w) = entries.windows(2).find(|w| w[0].key() == w[1].key()) {
            return Err(VindexError::Parse(format!(
                "tensor surface names `{}`/`{}` twice — a container presents one tensor \
                 once, so this surface describes no model",
                w[0].object, w[0].tensor
            )));
        }
        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[SurfaceTensor] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look one tensor up by its identity pair.
    pub fn get(&self, object: &str, tensor: &str) -> Option<&SurfaceTensor> {
        self.entries
            .binary_search_by(|e| e.key().cmp(&(object, tensor)))
            .ok()
            .map(|i| &self.entries[i])
    }

    /// The surface's canonical form: every record, in key order.
    pub(crate) fn canonical(&self) -> String {
        self.entries
            .iter()
            .map(SurfaceTensor::canonical)
            .collect::<Vec<_>>()
            .join(&RECORD.to_string())
    }

    /// **What this surface IS.** A digest, so a state identity can name
    /// the surface it was resolved against without carrying it.
    pub fn identity(&self) -> String {
        hash_bytes(self.canonical().as_bytes())
    }
}
