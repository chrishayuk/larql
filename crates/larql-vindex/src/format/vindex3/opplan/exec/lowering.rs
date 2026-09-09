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
//! L2 keys a registry by it — [`LoweringRegistry`], a VALUE a caller
//! carries and asks, never a default anything consults on its own. L4
//! records it in every pin and invalidates preparation by it. L1 only
//! made it exist and be stated by every provider, which was the smallest
//! thing that could be falsified.

use std::fmt;

use super::backend::PlanBackend;
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

/// Why a registry refused.
#[derive(Debug, thiserror::Error)]
pub enum LoweringError {
    /// An identity that could not key a registry (see
    /// [`LoweringIdentity::validate`]).
    #[error("{0}")]
    Invalid(VindexError),

    /// The same family and revision registered twice. Refused rather than
    /// replaced: "which of the two did the caller mean" has no answer, and
    /// a registry whose answer depended on registration order would be a
    /// registry whose meaning depended on the caller's program text.
    #[error("lowering provider `{identity}` is already registered")]
    Duplicate { identity: LoweringIdentity },

    /// No provider under that identity. Names every identity that IS
    /// registered, so the remedy is in the refusal: a family that is
    /// there at another revision is a different provider, and the
    /// refusal says which revisions exist.
    #[error("no lowering provider `{identity}` is registered; registered: {}", registered.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", "))]
    Unregistered {
        identity: LoweringIdentity,
        registered: Vec<LoweringIdentity>,
    },
}

impl From<LoweringError> for VindexError {
    fn from(e: LoweringError) -> Self {
        match e {
            LoweringError::Invalid(inner) => inner,
            other => VindexError::Parse(other.to_string()),
        }
    }
}

/// The lowering providers a caller carries, keyed by identity.
///
/// **A value, not a default.** There is no static registry and no
/// `builtin()` anything consults when a caller says nothing: the codec
/// plane found three hidden built-in registries at execution (rung 3, F8)
/// and one more at open (#461), and each was a place registration could
/// not reach. Here the registry exists only where a caller constructed
/// one, and every path that resolves a provider is handed it explicitly
/// ([`super::prepared::PreparedOperands::load_via`],
/// [`super::execute_plan_via`]). [`Self::shipped`] builds the providers
/// this crate ships as a fresh value each time it is asked.
///
/// One configured INSTANCE per identity. A provider is registered as the
/// instance the caller built — a device provider with the device it was
/// handed and the format table it was given — and a second instance
/// under the same identity is a duplicate, not an alternative. That is
/// the registry answering L1's open question the only way it can: the
/// identity is the provider's, the configuration is the registration's,
/// and a caller that wants two differently configured device providers
/// side by side has two providers to name.
#[derive(Default)]
pub struct LoweringRegistry {
    providers: Vec<Box<dyn PlanBackend + Send>>,
}

impl fmt::Debug for LoweringRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.providers.iter().map(|p| p.identity()))
            .finish()
    }
}

impl LoweringRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a provider under the identity it states, refusing an
    /// identity that could not be keyed and one already taken.
    pub fn register(
        mut self,
        provider: Box<dyn PlanBackend + Send>,
    ) -> Result<Self, LoweringError> {
        let identity = provider.identity();
        identity.validate().map_err(LoweringError::Invalid)?;
        if self.providers.iter().any(|p| p.identity() == identity) {
            return Err(LoweringError::Duplicate { identity });
        }
        self.providers.push(provider);
        Ok(self)
    }

    /// The providers this crate ships and needs no device to build: the
    /// reference oracle and the production CPU executor. A device
    /// provider is registered by whoever holds the device. A fresh value
    /// on every call — two callers asking get two registries, and neither
    /// is the other's default.
    pub fn shipped() -> Self {
        Self::new()
            .register(Box::new(super::reference::ReferenceBackend::new()))
            .and_then(|r| r.register(Box::new(super::production::ProductionBackend::new())))
            .expect("the shipped providers carry distinct, valid identities")
    }

    /// The provider `identity` names, or a refusal naming every provider
    /// that is registered.
    pub fn provider(&self, identity: &LoweringIdentity) -> Result<&dyn PlanBackend, LoweringError> {
        self.providers
            .iter()
            .find(|p| p.identity() == *identity)
            .map(|p| p.as_ref() as &dyn PlanBackend)
            .ok_or_else(|| LoweringError::Unregistered {
                identity: identity.clone(),
                registered: self.identities(),
            })
    }

    /// Every registered provider of `family`, at every revision, in
    /// registration order.
    pub fn family(&self, family: &str) -> Vec<&dyn PlanBackend> {
        self.providers
            .iter()
            .filter(|p| p.identity().family == family)
            .map(|p| p.as_ref() as &dyn PlanBackend)
            .collect()
    }

    /// Registered identities, in registration order — what a refusal lists.
    pub fn identities(&self) -> Vec<LoweringIdentity> {
        self.providers.iter().map(|p| p.identity()).collect()
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}
