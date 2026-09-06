//! What a codec REQUIRES of other represented objects, and what a
//! container states about the ones it points at.
//!
//! A stream is bytes of this operand; an auxiliary is another operand.
//! The difference is not decoration: a stream can be found from the
//! operand's own name, an auxiliary has to be addressed, and one auxiliary
//! may serve many owners. So a codec DECLARES the names it needs, the
//! container's reference table says where each name points, and the two
//! are held against each other before a payload byte is opened.
//!
//! Four things this declaration is not:
//!
//! * **not extent-independent** — [`RepresentationCodec::required_auxiliaries`]
//!   is asked for a specific extent, so a codec whose deeper extents need
//!   a dependency its base does not can say exactly that;
//! * **not free to rename** — a requirement's name is part of the codec
//!   REVISION's semantics, in the same way its stream names are. Renaming
//!   one changes what a stored container means and needs a revision bump;
//! * **not approximate** — a container provides exactly the declared
//!   names. Missing and undeclared both refuse, before I/O;
//! * **not about lifetime** — a requirement says the decode NEEDS this
//!   object to mean anything. Whether it stays resident, or is touched
//!   while serving, is the selected realization's declaration and has
//!   nothing to do with the operand being an auxiliary.

use super::error::CodecError;
use super::extent::RepresentationExtent;
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// One dependency a codec requires, by the name the container's reference
/// table keys on.
///
/// Deliberately only a name. What the target must LOOK like is judged by
/// the codec itself against the container's metadata
/// ([`RepresentationCodec::validate_auxiliary`]), because only the codec
/// knows what its dependency has to be — a shape rule stated here would
/// be this module guessing on a plugin's behalf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AuxiliarySpec {
    pub name: &'static str,
}

impl AuxiliarySpec {
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }
}

/// What the container authoritatively states about a resolved auxiliary.
///
/// Metadata only, and that is the point: it is built from the segment's
/// tensor table and the registry, so a dependency can be judged — and
/// refused — with no payload opened anywhere in the closure.
///
/// The shape and the label are the CONTAINER's, never the reference
/// table's: a table that restated them could disagree with the bytes it
/// addresses, and then a reader would have two authorities and no rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuxiliaryMetadata {
    /// Object and tensor, as the reference table addressed them.
    pub object: String,
    pub tensor: String,
    /// The stored label the segment records for the target.
    pub label: String,
    /// The shape the segment records.
    pub shape: Vec<usize>,
    /// The identity of the codec that label resolves to, when this build
    /// registers one. `None` is a fact, not a gap: an unregistered label
    /// has no decode, and the closure refuses it as a missing provider
    /// rather than letting the owner judge a stranger.
    pub identity: Option<CodecIdentity>,
}

impl AuxiliaryMetadata {
    /// How a refusal spells the target.
    pub fn describe(&self) -> String {
        format!("`{}`/`{}` ({})", self.object, self.tensor, self.label)
    }

    /// Refuse a target whose shape is not `expected` — the check most
    /// codecs will want, worded once so two plugins refuse the same way.
    pub fn require_shape(
        &self,
        expected: &[usize],
        owner: &str,
        owner_label: &str,
        name: &str,
    ) -> Result<(), CodecError> {
        if self.shape == expected {
            return Ok(());
        }
        Err(CodecError::AuxiliaryGeometry {
            tensor: owner.into(),
            label: owner_label.into(),
            name: name.into(),
            why: format!(
                "{} has shape {:?}; {expected:?} is required",
                self.describe(),
                self.shape
            ),
        })
    }
}

/// Hold what a container declared for one operand against what its codec
/// requires at `extent`: exactly the declared names, no more and no fewer.
///
/// Free-standing rather than a method so the rule has ONE implementation
/// whoever asks — the loader before it reads, the planner before it plans,
/// and a test that wants to ask without a container.
pub fn admit_auxiliary_names(
    required: &[AuxiliarySpec],
    provided: &[&str],
    label: &str,
    tensor: &str,
    extent: RepresentationExtent,
) -> Result<(), CodecError> {
    let names: Vec<String> = required.iter().map(|a| a.name.to_string()).collect();
    for spec in required {
        if !provided.contains(&spec.name) {
            return Err(CodecError::MissingAuxiliary {
                tensor: tensor.into(),
                label: label.into(),
                name: spec.name.into(),
                depth: extent.depth,
                required: names.clone(),
            });
        }
    }
    for given in provided {
        if !required.iter().any(|spec| spec.name == *given) {
            return Err(CodecError::UnexpectedAuxiliary {
                tensor: tensor.into(),
                label: label.into(),
                name: (*given).into(),
                depth: extent.depth,
                required: names.clone(),
            });
        }
    }
    Ok(())
}
