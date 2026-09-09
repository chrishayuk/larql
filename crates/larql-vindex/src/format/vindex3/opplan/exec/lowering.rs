//! **The lowering identity** — what a realization provider IS, as distinct
//! from what it is called.
//!
//! L1 of LOWERING-PLUGIN-1 (`docs/represent/forecasts/lowering-plugin-1.json`).
//! Before this module a [`PlanBackend`](super::backend::PlanBackend) had a
//! name "for diagnostics and parity reports, not dispatched on" and no
//! identity: a prepared image could say which CODEC produced its bytes by
//! family and revision, and which LOWERING qualified its pin only as
//! `Cpu | Device`. This is the identity the codec plane already carries
//! ([`CodecIdentity`](crate::format::vindex3::represent::nvfp4_pack::CodecIdentity)),
//! brought to the other half of the contract.
//!
//! Two fields, on purpose. The **family** names the provider — the code
//! that derives candidates from representation facts and realizes what
//! it pinned. The **revision** says whether that code still means what
//! it meant: bump it when the same pin would compute a different number,
//! never for a faster kernel that computes the same one. Configuration a
//! provider is constructed with — a device's per-class format table, the
//! injected `MatMul` — is not identity; what it changes is already pinned
//! in the realization form.
//!
//! Nothing dispatches on this yet. L2 keys a registry by it, L4 records
//! it in every pin and invalidates preparation by it. L1 only makes it
//! exist and be stated by every provider, which is the smallest thing
//! that can be falsified.

use std::fmt;

use crate::error::VindexError;

/// A lowering provider's semantic identity: family and revision.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct LoweringIdentity {
    /// Which provider. Stable across builds; a registry keys on it.
    pub family: String,
    /// Which meaning. Two revisions of one family may compute different
    /// numbers for the same pin, and a prepared image records which.
    pub revision: u32,
}

impl LoweringIdentity {
    pub fn new(family: impl Into<String>, revision: u32) -> Self {
        Self {
            family: family.into(),
            revision,
        }
    }

    /// Refuse an identity that could not key a registry or name a
    /// refusal: an empty or blank family, one that would not survive a
    /// file name or a flag (anything outside `[A-Za-z0-9_-]`), or a
    /// revision of zero, which is the "unstated" value and never a real
    /// one.
    pub fn validate(&self) -> Result<(), VindexError> {
        if self.family.trim().is_empty() {
            return Err(VindexError::Parse(
                "lowering identity: the family is empty".into(),
            ));
        }
        if let Some(bad) = self
            .family
            .chars()
            .find(|c| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '-'))
        {
            return Err(VindexError::Parse(format!(
                "lowering identity: family `{}` contains `{bad}`; only ASCII letters, digits, \
                 `_` and `-` name a provider",
                self.family
            )));
        }
        if self.revision == 0 {
            return Err(VindexError::Parse(format!(
                "lowering identity: `{}` declares revision 0, which means unstated",
                self.family
            )));
        }
        Ok(())
    }
}

impl fmt::Display for LoweringIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/v{}", self.family, self.revision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_valid_identity_displays_as_family_and_revision() {
        let id = LoweringIdentity::new("cpu-production", 1);
        id.validate().unwrap();
        assert_eq!(id.to_string(), "cpu-production/v1");
        assert_eq!(id, LoweringIdentity::new("cpu-production".to_string(), 1));
        assert_ne!(id, LoweringIdentity::new("cpu-production", 2));
    }

    #[test]
    fn an_invalid_identity_is_refused_naming_what_is_wrong() {
        let empty = LoweringIdentity::new("", 1).validate().unwrap_err();
        assert!(empty.to_string().contains("empty"), "{empty}");
        let blank = LoweringIdentity::new("  ", 1).validate().unwrap_err();
        assert!(blank.to_string().contains("empty"), "{blank}");
        let spaced = LoweringIdentity::new("cpu production", 1)
            .validate()
            .unwrap_err();
        assert!(spaced.to_string().contains("` `"), "{spaced}");
        let slashed = LoweringIdentity::new("cpu/v1", 1).validate().unwrap_err();
        assert!(slashed.to_string().contains("`/`"), "{slashed}");
        let unstated = LoweringIdentity::new("cpu", 0).validate().unwrap_err();
        assert!(unstated.to_string().contains("revision 0"), "{unstated}");
    }
}
