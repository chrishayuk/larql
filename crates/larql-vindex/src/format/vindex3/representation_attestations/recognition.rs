//! Whose word this build takes, and for which measurement.
//!
//! An attestation that is well-formed, current, and bound to exactly the
//! right bytes is still only a CLAIM. Someone wrote a number into a file.
//! Nothing about the file makes the number true, and a build that reads a
//! radius simply because one is present has replaced a measurement with a
//! rumour.
//!
//! So recognition is a separate gate with a separate answer:
//! [`AttestationStatus::Unrecognised`] is not
//! [`Absent`](AttestationStatus::Absent) and not
//! [`Stale`](AttestationStatus::Stale). The container is fine, the
//! artifact is fine, the attestation describes exactly what is there —
//! and this build does not take that authority's word, or does not
//! implement that method. Those are the reader's limitation, not the
//! artifact's fault, and collapsing them into "no certificate" would tell
//! an operator to re-encode when what they actually need is to configure
//! recognition.
//!
//! ## Nothing is recognised by default
//!
//! [`RecognisedMethods::none`] is the default, and it is the honest one:
//! no encoder in this build produces attestations yet, so this build has
//! qualified no authority and no method. A default that recognised, say,
//! anything calling itself `larql-encoder` would mean any file claiming
//! that name could assert any radius it liked — the plane's whole purpose
//! defeated by its own convenience default.
//!
//! The consequence is deliberate: an artifact full of attestations is
//! unusable until a caller says whose measurements it accepts. That is
//! what "presence is not trust" costs, and it is the correct cost.
//!
//! ## Recognition is exact, including the version
//!
//! `measured-rms@1` being recognised says nothing about `measured-rms@2`.
//! A method version exists precisely because the measurement changed —
//! sampled rows instead of the whole tensor, a different treatment of
//! outliers — and a build that accepted any version of a name it knew
//! would be honouring a measurement it had never seen the definition of.
//! The same rule VQ-1 applied to metric and domain ids, for the same
//! reason.
//!
//! What this module is NOT: signing, key material, a trust chain, or any
//! opinion about whether a recognised authority DESERVES recognition.
//! Recognition here is a declared list. Who belongs on it is a question
//! for whoever configures it.

use std::collections::BTreeSet;

use super::StoredId;

/// The authorities and methods this reader takes as its own.
///
/// Two independent lists, and an attestation must satisfy BOTH: an
/// authority you trust using a method you do not implement is as
/// unusable as a method you implement quoted by someone you do not
/// trust.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecognisedMethods {
    authorities: BTreeSet<String>,
    methods: BTreeSet<(String, u32)>,
}

impl RecognisedMethods {
    /// Recognise nothing — the default, and what a build that has
    /// qualified no measurement should say.
    pub fn none() -> Self {
        Self::default()
    }

    /// Also take this authority's word.
    pub fn with_authority(mut self, authority: impl Into<String>) -> Self {
        self.authorities.insert(authority.into());
        self
    }

    /// Also implement this exact method at this exact version.
    pub fn with_method(mut self, name: impl Into<String>, version: u32) -> Self {
        self.methods.insert((name.into(), version));
        self
    }

    /// Whether this reader takes `authority`'s word at all.
    pub fn trusts(&self, authority: &str) -> bool {
        self.authorities.contains(authority)
    }

    /// Whether this reader implements `method` at that exact version.
    pub fn implements(&self, method: &StoredId) -> bool {
        self.methods
            .contains(&(method.name.clone(), method.version))
    }

    /// Whether this reader recognises nothing at all — the state a
    /// diagnostic wants to distinguish, because "you recognise nothing"
    /// and "you recognise someone else" send an operator to different
    /// places.
    pub fn is_empty(&self) -> bool {
        self.authorities.is_empty() && self.methods.is_empty()
    }

    /// The authorities this reader takes, for a refusal that says what it
    /// WOULD have accepted.
    pub fn authorities(&self) -> impl Iterator<Item = &str> {
        self.authorities.iter().map(String::as_str)
    }

    /// The methods this reader implements, as `name@version`.
    pub fn methods(&self) -> impl Iterator<Item = String> + '_ {
        self.methods
            .iter()
            .map(|(name, version)| format!("{name}@{version}"))
    }
}

/// Why this build will not take an attestation's word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecognitionGap {
    /// The reader recognises nothing at all. Called out separately
    /// because it is almost always a configuration that was never set,
    /// not a judgement about this particular authority.
    NothingRecognised,
    /// A name this reader does not take.
    Authority { given: String },
    /// A measurement this reader does not implement — including a
    /// version bump of one it does.
    Method {
        given: String,
        /// True when the NAME is implemented at some other version,
        /// which is a much more useful thing to be told than "unknown".
        other_versions: Vec<u32>,
    },
}

impl RecognitionGap {
    /// Why, in words, with what this reader would have accepted.
    pub fn describe(&self, recognised: &RecognisedMethods) -> String {
        match self {
            Self::NothingRecognised => "this build recognises no attestation authority or method, \
                                        so no attestation is usable to it"
                .to_string(),
            Self::Authority { given } => {
                let taken: Vec<&str> = recognised.authorities().collect();
                format!("this build does not take `{given}`'s word; it takes {taken:?}")
            }
            Self::Method {
                given,
                other_versions,
            } if !other_versions.is_empty() => format!(
                "this build does not implement `{given}`; it implements that method at versions \
                 {other_versions:?}, and a version exists because the measurement differs"
            ),
            Self::Method { given, .. } => {
                let known: Vec<String> = recognised.methods().collect();
                format!("this build does not implement `{given}`; it implements {known:?}")
            }
        }
    }
}

/// The first reason `method` is not recognised, or `None` if it is.
///
/// Authority before method: being told "I do not take your word" is more
/// fundamental than "I do not implement your measurement", and a reader
/// who is not trusted at all does not need a lecture about versions.
pub(super) fn gap(
    authority: &str,
    method: &StoredId,
    recognised: &RecognisedMethods,
) -> Option<RecognitionGap> {
    if recognised.is_empty() {
        return Some(RecognitionGap::NothingRecognised);
    }
    if !recognised.trusts(authority) {
        return Some(RecognitionGap::Authority {
            given: authority.to_string(),
        });
    }
    if !recognised.implements(method) {
        let other_versions = recognised
            .methods
            .iter()
            .filter(|(name, _)| *name == method.name)
            .map(|(_, version)| *version)
            .collect();
        return Some(RecognitionGap::Method {
            given: format!("{}@{}", method.name, method.version),
            other_versions,
        });
    }
    None
}
