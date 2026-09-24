//! **The continuation identity** — what a continuation provider IS, as
//! distinct from the Rust type that happens to implement it.
//!
//! C1 of CONTINUATION-PLUGIN-1
//! (`docs/represent/forecasts/continuation-plugin-1.json`). Before this
//! module a [`ContinuationProvider`](super::kv::ContinuationProvider) was
//! known only by its type, so the one durable handoff that outlives a
//! request (`V3KvHandoff`) carried its authority as a concrete type. This
//! is the identity the codec and lowering planes already carry, brought to
//! the third plane.
//!
//! Two fields, as for a lowering. The **family** names the provider — the
//! code that holds and serves a conversation's state. The **revision**
//! says whether that code still means what it meant: bump it when the
//! same appended history would be served back differently, never for a
//! faster store serving the same bits. Configuration a provider is built
//! with is NOT identity; the forecast makes it a second, separate
//! authority (C4), because one implementation under two configurations is
//! two interpretations of the same stored state.
//!
//! The identity is stated by constants in each provider's own module and
//! read through the provider's inherent `identity()` (`RowKvState::identity`
//! here, `CanonicalKvState::identity` in `larql-kv`), not by a trait method: C1–C5 keep the provider contract at its nine
//! methods (F3), and a registry (C2) states identity per factory anyway.

use std::fmt;

use crate::error::VindexError;

/// Which plane an identity's refusals name.
const PLANE: &str = "continuation";

/// A continuation provider's semantic identity: family and revision.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct ContinuationIdentity {
    /// Which provider. Stable across builds; a registry keys on it.
    pub family: String,
    /// Which meaning. Two revisions of one family may serve a stored
    /// history back differently, and a handoff records which.
    pub revision: u32,
}

impl ContinuationIdentity {
    pub fn new(family: impl Into<String>, revision: u32) -> Self {
        Self {
            family: family.into(),
            revision,
        }
    }

    /// Refuse an identity that could not key a registry or name a
    /// refusal — the rule every provider identity shares.
    pub fn validate(&self) -> Result<(), VindexError> {
        super::provider_identity::validate(PLANE, &self.family, self.revision)
    }
}

impl fmt::Display for ContinuationIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/v{}", self.family, self.revision)
    }
}
