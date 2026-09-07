//! Stage one of two: does this attestation still describe what is
//! actually there?
//!
//! Everything here is answered from METADATA — the tensor table, the
//! codec's declared identity, the reference table — with no payload read,
//! exactly as VQ-1 admits an auxiliary closure. What it cannot answer is
//! whether the BYTES are still the bytes that were measured; the container
//! holds no per-tensor digest, so that check needs the payload and happens
//! at preparation, when it is being read anyway.
//!
//! ## Three answers, and none of them is zero
//!
//! [`AttestationStatus`] keeps apart what a single `Option` would blur:
//!
//! - **Absent** — the container attests nothing here. Ordinary, and true
//!   of every container written before this wave.
//! - **Stale** — something IS attested and it was measured against
//!   something that is no longer what is there. Names the cause.
//! - **Bound** — the tuple holds. Not yet "trusted": recognition is a
//!   separate question, and so is the content digest.
//!
//! A caller that collapses these into "no certificate" loses the ability
//! to say WHY, which is the difference between a container that never
//! promised anything and one whose promise has quietly expired.
//!
//! ## What the container cannot check, and why it is still recorded
//!
//! Two of the six staleness causes are not container-checkable at all.
//! The index says it plainly — *once encoded, the source checkpoint
//! disappears as an authority* — so an artifact holds neither the source
//! tensor it was fitted to nor the recipe that produced it. They are
//! carried as IDENTITY, and they are compared only when a caller who
//! actually has them supplies the expectation: a re-encode pipeline, a
//! qualification harness, an external verifier holding the checkpoint.
//! Recording them uncheckably is not decoration — it is what lets a
//! verifier outside this process do the check this process cannot.

use std::collections::BTreeMap;

use super::{AttestationTable, JudgedAttestation};
use crate::format::vindex3::auxiliary_references::OperandAddress;

/// What the container actually says about an operand, gathered from
/// metadata alone, for an attestation to be checked against.
///
/// Borrowed rather than owned because building one is a read of things
/// that already exist — a tensor table entry, a codec's declared
/// identity — and copying them would invite building one from something
/// other than the container.
#[derive(Debug, Clone)]
pub struct AttestedSubject<'a> {
    pub operand: &'a OperandAddress,
    /// The extent whose attestation is being sought.
    pub extent_depth: u32,
    /// From the codec the stored dtype resolves to, never from the
    /// attestation — the point is to compare two independent readings.
    pub codec_family: &'a str,
    pub codec_revision: u32,
    /// From the container's tensor table.
    pub shape: &'a [usize],
    /// Each declared dependency's TERMINAL identity as the container
    /// currently states it, by the auxiliary name the codec declared.
    pub auxiliary_baselines: BTreeMap<String, String>,
    /// The source digest the CALLER knows, where it knows one. `None`
    /// from any caller holding only the container, which is most of them
    /// — see the module note.
    pub expected_source_digest: Option<&'a str>,
    /// The recipe the caller expects, on the same terms.
    pub expected_recipe: Option<&'a str>,
}

/// Why an attestation no longer describes what is there.
///
/// One variant per cause rather than a string, so a caller can act on the
/// difference — a changed dependency baseline is worth re-attesting, a
/// changed codec revision usually means re-encoding — and so a test can
/// assert WHICH check fired rather than that something did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StalenessCause {
    CodecFamily {
        attested: String,
        found: String,
    },
    CodecRevision {
        attested: u32,
        found: u32,
    },
    Shape {
        attested: Vec<usize>,
        found: Vec<usize>,
    },
    /// The operand depends on a different SET of objects than when it was
    /// measured — one added, one gone, or one renamed.
    DependencySet {
        attested: Vec<String>,
        found: Vec<String>,
    },
    /// The same dependency, a different baseline: the codebook was
    /// replaced. This is the cause the terminal-baseline rule exists to
    /// make visible, and it must NOT fire merely because a plan selected
    /// a shallower extent.
    DependencyBaseline {
        name: String,
        attested: String,
        found: String,
    },
    /// Only ever raised when the caller supplied an expectation.
    SourceDigest {
        attested: String,
        expected: String,
    },
    Recipe {
        attested: String,
        expected: String,
    },
}

impl StalenessCause {
    /// The cause in a sentence, naming both readings, because "stale" on
    /// its own sends nobody anywhere.
    pub fn describe(&self) -> String {
        match self {
            Self::CodecFamily { attested, found } => {
                format!("it was measured on codec family `{attested}` and the container stores `{found}`")
            }
            Self::CodecRevision { attested, found } => format!(
                "it was measured against codec revision {attested} and the container stores \
                 revision {found}; a revision may change what the same bytes mean"
            ),
            Self::Shape { attested, found } => {
                format!("it was measured at shape {attested:?} and the operand is {found:?}")
            }
            Self::DependencySet { attested, found } => format!(
                "it was measured with dependencies {attested:?} and the operand now declares \
                 {found:?}"
            ),
            Self::DependencyBaseline {
                name,
                attested,
                found,
            } => format!(
                "its dependency `{name}` was `{attested}` when the error was measured and is now \
                 `{found}`; the measurement is about the pair, not about the codes alone"
            ),
            Self::SourceDigest { attested, expected } => format!(
                "it was measured against source `{attested}` and the caller expects `{expected}`"
            ),
            Self::Recipe { attested, expected } => {
                format!(
                    "it was produced by recipe `{attested}` and the caller expects `{expected}`"
                )
            }
        }
    }
}

/// What the container has to say about an operand's attested fidelity.
#[derive(Debug, Clone, PartialEq)]
pub enum AttestationStatus<'a> {
    /// Nothing attested for this operand at this extent.
    ///
    /// `elsewhere` carries the depths that ARE attested, so a caller can
    /// say "not at depth 1, though depths 0 and 2 are" rather than
    /// leaving a reader to wonder whether the file was even loaded.
    Absent { elsewhere: Vec<u32> },
    /// Attested, and the attestation is about something else now.
    Stale {
        attestation: &'a JudgedAttestation,
        cause: StalenessCause,
    },
    /// The tuple holds. NOT yet a usable guarantee: the authority still
    /// has to be recognised, and the content digest still has to match.
    Bound(&'a JudgedAttestation),
}

impl AttestationStatus<'_> {
    /// The attestation this status is about, where there is one.
    pub fn attestation(&self) -> Option<&JudgedAttestation> {
        match self {
            Self::Absent { .. } => None,
            Self::Stale { attestation, .. } => Some(attestation),
            Self::Bound(attestation) => Some(attestation),
        }
    }

    /// Why no guarantee is available here — `None` when one is.
    ///
    /// Every branch says something a reader can act on, and none of them
    /// says or implies zero.
    pub fn unavailable_because(&self) -> Option<String> {
        match self {
            Self::Absent { elsewhere } if elsewhere.is_empty() => {
                Some("nothing is attested for it".to_string())
            }
            Self::Absent { elsewhere } => Some(format!(
                "nothing is attested at this extent; the container attests depths {elsewhere:?}"
            )),
            Self::Stale { cause, .. } => {
                Some(format!("its attestation is stale: {}", cause.describe()))
            }
            Self::Bound(_) => None,
        }
    }
}

impl AttestationTable {
    /// Stage one: find the attestation for `subject` and check every part
    /// of its binding that the container can speak to.
    ///
    /// Metadata only. No payload is read, and none needs to be: every
    /// comparison here is between something the attestation states and
    /// something the container already told the caller.
    ///
    /// The content digest is deliberately NOT checked — a [`Bound`] status
    /// means "the tuple holds", never "the bytes are right".
    ///
    /// [`Bound`]: AttestationStatus::Bound
    pub fn status_of<'a>(&'a self, subject: &AttestedSubject<'_>) -> AttestationStatus<'a> {
        let Some(attestation) = self.at(subject.operand, subject.extent_depth) else {
            return AttestationStatus::Absent {
                elsewhere: self.depths(subject.operand),
            };
        };
        match first_difference(attestation, subject) {
            Some(cause) => AttestationStatus::Stale { attestation, cause },
            None => AttestationStatus::Bound(attestation),
        }
    }
}

/// The first way `attestation` and `subject` disagree.
///
/// Ordered cheapest and most fundamental first: a wrong codec makes every
/// later comparison meaningless, so reporting the shape mismatch of an
/// operand stored under another codec entirely would send a reader after
/// the wrong thing.
fn first_difference(
    attestation: &JudgedAttestation,
    subject: &AttestedSubject<'_>,
) -> Option<StalenessCause> {
    let b = &attestation.binding;
    if b.codec_family != subject.codec_family {
        return Some(StalenessCause::CodecFamily {
            attested: b.codec_family.clone(),
            found: subject.codec_family.to_string(),
        });
    }
    if b.codec_revision != subject.codec_revision {
        return Some(StalenessCause::CodecRevision {
            attested: b.codec_revision,
            found: subject.codec_revision,
        });
    }
    if b.shape != subject.shape {
        return Some(StalenessCause::Shape {
            attested: b.shape.clone(),
            found: subject.shape.to_vec(),
        });
    }

    let attested_names: Vec<String> = b.auxiliary_baselines.keys().cloned().collect();
    let found_names: Vec<String> = subject.auxiliary_baselines.keys().cloned().collect();
    if attested_names != found_names {
        return Some(StalenessCause::DependencySet {
            attested: attested_names,
            found: found_names,
        });
    }
    for (name, attested) in &b.auxiliary_baselines {
        let found = &subject.auxiliary_baselines[name];
        if attested != found {
            return Some(StalenessCause::DependencyBaseline {
                name: name.clone(),
                attested: attested.clone(),
                found: found.clone(),
            });
        }
    }

    // Provenance last, and only where the caller brought an expectation:
    // the container cannot supply one.
    if let Some(expected) = subject.expected_source_digest {
        if b.source_digest != expected {
            return Some(StalenessCause::SourceDigest {
                attested: b.source_digest.clone(),
                expected: expected.to_string(),
            });
        }
    }
    if let Some(expected) = subject.expected_recipe {
        if b.recipe != expected {
            return Some(StalenessCause::Recipe {
                attested: b.recipe.clone(),
                expected: expected.to_string(),
            });
        }
    }
    None
}
