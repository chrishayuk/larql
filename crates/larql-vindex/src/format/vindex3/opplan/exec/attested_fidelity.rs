//! B3: an attested radius reaching the floor that gates selection.
//!
//! VQ-1 built a composition algebra and nothing called it: the floor
//! reads `option.certificate.radius`, which is the CODEC's declared bound
//! with no knowledge of any dependency. That was harmless only because no
//! shipped codec both declares a radius and requires an auxiliary.
//! Attestation removes that accident — an attested VQ operand HAS a
//! radius — so wiring composition into selection is part of this wave and
//! not a follow-up. Without it the sidecar is stored evidence nobody
//! relies on, and the silent over-promise VQ-1 exposed stays possible.
//!
//! # Three phases, and the order is the point
//!
//! Step 4 established the boundary: `Bound` means the claim addresses the
//! right representation; only `Verified` means its certificate may
//! influence a decision. Verification needs the payload, so selection has
//! to be phased explicitly:
//!
//! 1. **Metadata admission** ([`admit`]) — derive candidates and reject
//!    absent, stale and unrecognised attestations with NO payload read.
//! 2. **Evidence verification** ([`Admitted::verify`]) — hash only the
//!    subjects admission bound. No executable pin exists yet.
//! 3. **Final selection** ([`VerifiedEvidence::derive`]) — build the
//!    option's certificate from `Verified` claims only, compose parent
//!    with dependency, and let the floor decide.
//!
//! **There is no provisional selection from `Bound` evidence, even if it
//! would be checked later.** A plan built from unverified claims and
//! corrected afterwards would, however briefly, claim a guarantee it had
//! not earned — and "briefly" is not a property a guarantee can have. The
//! types enforce it: [`VerifiedEvidence`] is the only thing [`derive`]
//! accepts, and it is unconstructible except by [`Admitted::verify`].
//!
//! # Time of check, time of use
//!
//! Hashing bytes proves nothing about the bytes a later decode reads
//! unless both come from the same immutable object. Every phase therefore
//! carries the [`SourceStamp`] — the store's process-unique id plus the
//! overlay generation — and [`VerifiedEvidence::ensure_current_for`]
//! refuses a source that has moved since. An overlay edit between
//! verification and decode invalidates the evidence rather than silently
//! carrying a guarantee across the change.
//!
//! [`derive`]: VerifiedEvidence::derive

use std::collections::BTreeMap;

use super::operands::{OperandSource, SourceStamp};
use crate::error::VindexError;
use crate::format::vindex3::auxiliary_references::OperandAddress;
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::{CodecError, FidelityCertificate};
use crate::format::vindex3::representation_attestations::recognition::RecognisedMethods;
use crate::format::vindex3::representation_attestations::tuple::{
    AttestationStatus, AttestedSubject,
};
use crate::format::vindex3::representation_attestations::AttestationTable;

/// An operand at an extent — what an attestation is keyed by.
pub type AttestedKey = (OperandAddress, u32);

/// One attestation that survived metadata admission, and the operand
/// whose bytes will settle it.
///
/// Holds no certificate: a candidate's claim is not readable until it is
/// verified, so there is nothing here for a caller to reach past the
/// phases and use.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub key: AttestedKey,
    /// What must be hashed — the operand the attestation is bound to.
    pub operand: OperandRef,
}

/// Why an attestation did not become a candidate.
///
/// Kept rather than discarded so a plan can say why an operand got no
/// guarantee. "No certificate" is not an explanation, and the three
/// causes send an operator to three different places.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejected {
    pub key: AttestedKey,
    pub why: String,
}

/// Phase one's result: what may be verified, and what was refused
/// without reading a byte.
#[derive(Debug, Clone)]
pub struct Admitted {
    candidates: Vec<Candidate>,
    rejected: Vec<Rejected>,
    stamp: SourceStamp,
}

/// **Phase one.** Which attestations address what is actually there.
///
/// Metadata only. Every refusal here is decided from the tensor table,
/// the codec's declared identity and the reference table — the same
/// admission discipline VQ-1 applies to an auxiliary closure.
pub fn admit(
    table: &AttestationTable,
    subjects: &[(AttestedSubject<'_>, OperandRef)],
    recognised: &RecognisedMethods,
    source: &OperandSource<'_>,
) -> Admitted {
    let mut candidates = Vec::new();
    let mut rejected = Vec::new();
    for (subject, operand) in subjects {
        let key = (subject.operand.clone(), subject.extent_depth);
        match table.status_of(subject, recognised) {
            AttestationStatus::Bound(_) => candidates.push(Candidate {
                key,
                operand: operand.clone(),
            }),
            // Verified cannot occur here: nothing has read a payload.
            other => {
                if let Some(why) = other.unavailable_because(recognised) {
                    rejected.push(Rejected { key, why });
                }
            }
        }
    }
    Admitted {
        candidates,
        rejected,
        stamp: source.stamp(),
    }
}

impl Admitted {
    /// What phase two will hash.
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// What was refused before any payload read, and why.
    pub fn rejected(&self) -> &[Rejected] {
        &self.rejected
    }

    /// **Phase two.** Hash exactly the admitted subjects, and nothing
    /// else.
    ///
    /// The bytes read here are PREPARATION WORK and are reported as such:
    /// verifying evidence is not free, and a ledger that hid it would
    /// price a plan as if guarantees cost nothing. A caller that admits
    /// nothing reads nothing.
    ///
    /// No executable pin exists at this point, by construction — this
    /// returns evidence, not a selection.
    pub fn verify(
        self,
        table: &AttestationTable,
        source: &OperandSource<'_>,
    ) -> Result<VerifiedEvidence, VindexError> {
        if source.stamp() != self.stamp {
            return Err(VindexError::Parse(
                "the operand source changed between attestation admission and verification; \
                 evidence gathered against one generation cannot settle another"
                    .to_string(),
            ));
        }
        let mut verified = BTreeMap::new();
        let mut refused = self.rejected;
        let mut read_to_prepare = 0u64;
        let mut read_by_operand: BTreeMap<OperandAddress, u64> = BTreeMap::new();
        for candidate in &self.candidates {
            // Admission already established that this attestation exists,
            // addresses this operand and is recognised. The only question
            // left is the bytes, so nothing here re-derives a status from
            // a subject — which is also why `verify` takes no second
            // subject list to disagree with the first.
            let Some(attestation) = table.at(&candidate.key.0, candidate.key.1) else {
                refused.push(Rejected {
                    key: candidate.key.clone(),
                    why: "its attestation was withdrawn between admission and verification"
                        .to_string(),
                });
                continue;
            };
            let raw = source.load_raw(&candidate.operand)?;
            // Every candidate hashes its own read. Two attestations over
            // one operand at two depths therefore cost TWO reads, and are
            // counted twice, because `load_raw` opens and copies each
            // time — nothing here shares a materialisation, so nothing
            // here may claim to.
            read_to_prepare += raw.bytes.len() as u64;
            *read_by_operand.entry(candidate.key.0.clone()).or_default() += raw.bytes.len() as u64;
            match attestation.content_mismatch(&raw.bytes) {
                None => {
                    verified.insert(candidate.key.clone(), attestation.claimed().clone());
                }
                Some(cause) => refused.push(Rejected {
                    key: candidate.key.clone(),
                    why: format!("its attestation is stale: {}", cause.describe()),
                }),
            }
        }
        Ok(VerifiedEvidence {
            verified,
            refused,
            stamp: self.stamp,
            read_to_prepare,
            read_by_operand,
        })
    }
}

/// Phase two's result: claims that may influence a decision, and the
/// generation they were settled against.
///
/// Unconstructible outside [`Admitted::verify`], which is what makes
/// "only `Verified` evidence reaches selection" a property of the types
/// rather than a rule someone has to remember.
#[derive(Debug, Clone)]
pub struct VerifiedEvidence {
    verified: BTreeMap<AttestedKey, FidelityCertificate>,
    refused: Vec<Rejected>,
    stamp: SourceStamp,
    read_to_prepare: u64,
    /// Bytes hashed PER OPERAND, so preparation accounting can attribute
    /// the cost to the record that incurred it rather than carrying one
    /// global total nothing can be held against.
    read_by_operand: BTreeMap<OperandAddress, u64>,
}

impl VerifiedEvidence {
    /// Evidence for a plan that admitted nothing — every operand falls
    /// back to what its codec declares.
    pub fn none(source: &OperandSource<'_>) -> Self {
        Self {
            verified: BTreeMap::new(),
            refused: Vec::new(),
            stamp: source.stamp(),
            read_to_prepare: 0,
            read_by_operand: BTreeMap::new(),
        }
    }

    /// Bytes read to settle this evidence. Preparation work, reported
    /// honestly rather than absorbed.
    pub fn read_to_prepare(&self) -> u64 {
        self.read_to_prepare
    }

    /// Bytes hashed to settle THIS operand's attestations, across every
    /// extent depth attested for it. Zero for an operand that attests
    /// nothing, and zero for one whose attestation never reached the
    /// payload — admission refuses absent, stale and unrecognised claims
    /// from metadata, and a refusal there costs no read.
    pub fn verified_bytes_for(&self, operand: &OperandAddress) -> u64 {
        self.read_by_operand.get(operand).copied().unwrap_or(0)
    }

    /// Everything that did not survive, from either phase.
    pub fn refused(&self) -> &[Rejected] {
        &self.refused
    }

    /// How many claims may influence a decision.
    pub fn count(&self) -> usize {
        self.verified.len()
    }

    /// The verified claim for an operand at an extent, if there is one.
    pub fn at(&self, key: &AttestedKey) -> Option<&FidelityCertificate> {
        self.verified.get(key)
    }

    /// Refuse a source that has moved since verification.
    ///
    /// The time-of-check/time-of-use guard: bytes hashed under one
    /// generation say nothing about the bytes a decode reads under
    /// another.
    pub fn ensure_current_for(&self, source: &OperandSource<'_>) -> Result<(), VindexError> {
        if source.stamp() == self.stamp {
            return Ok(());
        }
        Err(VindexError::Parse(
            "attestation evidence was verified against a different operand generation — the \
             overlay changed, or this is another container. Re-verify rather than selecting on \
             a guarantee that was checked against other bytes."
                .to_string(),
        ))
    }

    /// **Phase three.** The certificate a caller may rely on for this
    /// operand at this extent, composed with the dependency extents
    /// ACTUALLY selected.
    ///
    /// `declared` is what the codec states for the extent — used when
    /// nothing is attested, which is every operand in every container
    /// written before this wave. An attested claim REPLACES the declared
    /// one rather than widening it: they are two statements about the
    /// same quantity, one measured on this instance and one derived from
    /// the scheme, and adding them would double-count.
    ///
    /// `dependencies` are the certificates of the auxiliary extents the
    /// plan selected. The attestation binds to each dependency's TERMINAL
    /// baseline, so what a shallower selection costs is composed here and
    /// nowhere else — which is precisely why an attestation does not go
    /// stale when a plan picks a different depth.
    pub fn derive(
        &self,
        key: &AttestedKey,
        declared: Option<&FidelityCertificate>,
        dependencies: &[FidelityCertificate],
        tensor: &str,
        label: &str,
    ) -> Result<Option<FidelityCertificate>, CodecError> {
        let Some(base) = self.at(key).or(declared) else {
            return Ok(None);
        };
        let mut composed = base.clone();
        for dependency in dependencies {
            composed = composed.widened_by(dependency, tensor, label)?;
        }
        Ok(Some(composed))
    }
}
