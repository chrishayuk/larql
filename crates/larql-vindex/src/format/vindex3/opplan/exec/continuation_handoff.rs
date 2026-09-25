//! **Continuation state that outlives a call, and the one rule for
//! resuming it.**
//!
//! C4 of CONTINUATION-PLUGIN-1
//! (`docs/represent/forecasts/continuation-plugin-1.json`, boundary in
//! `continuation-plugin-1-notes.json`). A [`ContinuationHandoff`] is a
//! conversation's state sealed with the [`ContinuationAuthority`] that
//! built it. The authority is attached when the state is built
//! ([`SelectedContinuation::begin`](super::continuation_registry::SelectedContinuation::begin))
//! and the fields are private, so no caller can pair state with an
//! authority that did not produce it.
//!
//! **Resume compares; it never recovers.** [`ContinuationHandoff::resume`]
//! returns the recorded state when the resolving selection is the same
//! authority, and otherwise a typed [`ResumeRefusal`] that names both
//! sides. It builds nothing, consults no registry and prefills nothing:
//! a working provider that happens to be registered is not a licence to
//! reinterpret state another provider wrote. A caller that wants to
//! carry on after a refusal starts a fresh conversation itself, and says
//! so.
//!
//! Identity is compared before the configuration digest. The digest
//! covers the identity too (C2), so a changed identity always reports as
//! a provider or revision refusal, never as a configuration mismatch.

use super::continuation_authority::{ConfigDigest, ContinuationAuthority};
use super::continuation_identity::ContinuationIdentity;
use super::continuation_registry::BoxedContinuation;
use super::kv::ContinuationProvider;

/// A conversation's continuation state and the authority that answers
/// for it.
pub struct ContinuationHandoff {
    authority: ContinuationAuthority,
    state: BoxedContinuation,
}

impl std::fmt::Debug for ContinuationHandoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinuationHandoff")
            .field("authority", &format_args!("{}", self.authority))
            .field("position", &self.state.position())
            .finish()
    }
}

/// Why recorded state cannot resume under the resolving selection. Each
/// variant names both sides; none carries the state, which is dropped
/// rather than handed to anyone else.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResumeRefusal {
    /// The provider that wrote the state is not the one this request
    /// resolves: no provider of the recorded family is at hand.
    #[error(
        "continuation state held by `{recorded}` cannot resume: this request resolves \
         `{resolved}`, and the provider that holds the state is absent"
    )]
    ProviderAbsent {
        recorded: ContinuationIdentity,
        resolved: ContinuationIdentity,
    },

    /// Same provider family, different implementation revision.
    #[error(
        "continuation state held by `{recorded}` cannot resume under `{resolved}`: \
         the provider's revision changed"
    )]
    RevisionChanged {
        recorded: ContinuationIdentity,
        resolved: ContinuationIdentity,
    },

    /// Same implementation, asked to hold the state under a different
    /// configuration: a different interpretation of the same bytes.
    #[error(
        "continuation state held by `{identity}` under configuration {recorded} cannot \
         resume under configuration {resolved}: the configuration changed"
    )]
    ConfigurationChanged {
        identity: ContinuationIdentity,
        recorded: ConfigDigest,
        resolved: ConfigDigest,
    },
}

impl ResumeRefusal {
    /// The refusal's kind as a stable token, for records and counters.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::ProviderAbsent { .. } => "provider_absent",
            Self::RevisionChanged { .. } => "revision_changed",
            Self::ConfigurationChanged { .. } => "configuration_changed",
        }
    }

    /// Compare a recorded authority with a resolving one: `None` when they
    /// are the same authority, the refusal otherwise.
    pub fn between(
        recorded: &ContinuationAuthority,
        resolved: &ContinuationAuthority,
    ) -> Option<Self> {
        let (was, now) = (&recorded.identity, &resolved.identity);
        if was.family != now.family {
            return Some(Self::ProviderAbsent {
                recorded: was.clone(),
                resolved: now.clone(),
            });
        }
        if was.revision != now.revision {
            return Some(Self::RevisionChanged {
                recorded: was.clone(),
                resolved: now.clone(),
            });
        }
        if recorded.config_digest != resolved.config_digest {
            return Some(Self::ConfigurationChanged {
                identity: was.clone(),
                recorded: recorded.config_digest.clone(),
                resolved: resolved.config_digest.clone(),
            });
        }
        None
    }
}

impl ContinuationHandoff {
    /// Seal freshly built state with the authority that built it. Only a
    /// selection calls this, at build time.
    pub(super) fn seal(authority: ContinuationAuthority, state: BoxedContinuation) -> Self {
        Self { authority, state }
    }

    /// Who answers for this state.
    pub fn authority(&self) -> &ContinuationAuthority {
        &self.authority
    }

    /// The logical position the state continues from.
    pub fn position(&self) -> usize {
        self.state.position()
    }

    /// The state, for the traversal that drives it.
    pub fn state(&self) -> &(dyn ContinuationProvider + Send) {
        &*self.state
    }

    /// The state, for the traversal that drives it.
    pub fn state_mut(&mut self) -> &mut (dyn ContinuationProvider + Send) {
        &mut *self.state
    }

    /// Resume this state under `resolved`: the handoff back when
    /// `resolved` is the authority that wrote it, the refusal otherwise.
    /// On refusal the state is dropped; nothing is built, selected or
    /// prefilled here.
    pub fn resume(self, resolved: &ContinuationAuthority) -> Result<Self, ResumeRefusal> {
        match ResumeRefusal::between(&self.authority, resolved) {
            None => Ok(self),
            Some(refusal) => Err(refusal),
        }
    }
}
