//! What a REPRESENTED OPERAND INSTANCE achieved, as opposed to what its
//! codec's scheme guarantees.
//!
//! A codec can state a radius when its error is a property of the SCHEME:
//! `F32_PLANES` truncates bit planes, and the bound follows from which
//! planes are read. `VQ8_SHARED` cannot, because a vector quantiser's
//! assignment error is a property of a FITTED codebook against ACTUAL
//! weights — the same codec, the same 256 entries and the same grouping
//! give a different error on every tensor they were fitted to. That is an
//! artifact-INSTANCE fact, and nothing in the codec plane can carry it.
//!
//! So fidelity is not one authority but four, and they compose:
//!
//! | layer | certifies |
//! |---|---|
//! | artifact attestation | the error THIS operand instance introduces at THIS extent |
//! | auxiliary composition | the additional error from the dependency extents SELECTED |
//! | realization qualification | the error a particular kernel or backend introduces |
//! | final guarantee | the composition of whichever apply |
//!
//! This module is the first row. The second is VQ-1's
//! [`composed_certificate`](super::represent::codec::RepresentationCodec::composed_certificate);
//! the third is a later rung and deliberately absent here.
//!
//! ## What an attestation is ABOUT
//!
//! An attestation attaches to the represented operand INSTANCE at an
//! extent — never to a realization, whose numerical error is a different
//! layer with a different authority. Binding it to a realization would
//! let a kernel change silently invalidate a measurement that had nothing
//! to do with the kernel, or worse, fail to.
//!
//! ## Why it binds to the TERMINAL baseline of its dependencies
//!
//! An attested VQ error is measured against a codebook. If it bound to
//! whatever codebook DEPTH a plan happened to select, every parent ×
//! dependency-depth combination would need its own attestation — a
//! combinatorial explosion, and one where an attestation goes stale for a
//! reason that has nothing to do with the measurement. So it binds to the
//! dependency's TERMINAL identity, and the depth actually selected is
//! composed on top at planning time:
//!
//! ```text
//! attested assignment error against the terminal codebook
//!                          +
//!         certificate for the selected codebook extent
//!                          ↓
//!                derived parent certificate
//! ```
//!
//! ## Presence is not trust
//!
//! A well-formed, current attestation bound to exactly the right bytes is
//! still only a claim. Admission must RECOGNISE the authority and the
//! method; one this build does not know leaves the certificate
//! unavailable, named as unrecognised rather than as absent or as stale.
//! That is the same discipline VQ-1 applied to metric and domain ids, and
//! for the same reason: a claim nobody can spell is a claim nobody can
//! refuse.
//!
//! What this module does NOT do: signing, key material, trust chains,
//! provenance. Whether a recognised authority DESERVES recognition is a
//! different plane again.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::auxiliary_references::OperandAddress;
use super::represent::codec::{DomainId, FidelityCertificate, MetricId, SemanticId};
use crate::error::VindexError;

pub mod tuple;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_tuple;

/// The schema this build implements. A table stamped with anything else
/// is refused rather than read optimistically: an attestation is a
/// guarantee, and a guarantee read under the wrong rules is worse than no
/// guarantee at all.
pub const REPRESENTATION_ATTESTATIONS_SCHEMA: u32 = 1;

/// A versioned semantic id as the container stores it.
///
/// The same shape as the codec plane's [`MetricId`] and [`DomainId`],
/// stored rather than declared. Kept as a plain record here — judging it
/// into the codec plane's type is [`StoredAttestation::judge`]'s job, so
/// that a malformed id is a refusal with a container in hand rather than
/// a panic in a deserialiser.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredId {
    pub name: String,
    pub version: u32,
}

impl StoredId {
    pub fn new(name: impl Into<String>, version: u32) -> Self {
        Self {
            name: name.into(),
            version,
        }
    }

    fn describe(&self) -> String {
        format!("{}@{}", self.name, self.version)
    }
}

/// Who measured an attestation, and how.
///
/// Separate from the binding because they answer different questions: the
/// binding says WHAT was measured, this says whose word it is. A build
/// recognises methods, not artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationMethod {
    /// Who states it — an encoder, a qualification harness, a vendor.
    pub authority: String,
    /// How it was measured, versioned, because "relative RMS over the
    /// whole tensor" and "relative RMS over sampled rows" are different
    /// measurements that would otherwise share a name.
    pub method: StoredId,
}

impl AttestationMethod {
    pub fn new(authority: impl Into<String>, method: StoredId) -> Self {
        Self {
            authority: authority.into(),
            method,
        }
    }

    /// How this method reads in a refusal.
    pub fn describe(&self) -> String {
        format!("`{}` by {}", self.method.describe(), self.authority)
    }
}

/// What an attestation is ABOUT — the binding tuple.
///
/// Every field is part of the identity, and a change to any of them makes
/// the attestation STALE: it was measured against something that is no
/// longer what is there. Staleness is unavailability, never zero and
/// never "probably still fine".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationBinding {
    /// The owner operand this measures.
    pub operand: OperandAddress,
    /// The parent extent it was measured at. An operand may carry one
    /// attestation per extent; a measurement at depth 1 says nothing
    /// about depth 0.
    pub extent_depth: u32,
    /// The codec family and revision that produced the bytes. A revision
    /// bump may change what the same bytes MEAN, so it invalidates.
    pub codec_family: String,
    pub codec_revision: u32,
    /// The operand's logical shape, so a re-encode at another shape
    /// cannot inherit a measurement.
    pub shape: Vec<usize>,
    /// Digest of the stored content this measures — the codes
    /// themselves. Checked against the bytes at preparation, not here:
    /// see the two-stage note on [`AttestationTable`].
    pub content_digest: String,
    /// Digest of the SOURCE tensor the error was measured against. Two
    /// artifacts can hold identical codes and have been fitted to
    /// different sources; the error is a statement about the pair.
    pub source_digest: String,
    /// Each dependency's TERMINAL identity, by the auxiliary name the
    /// codec declared. Never the selected depth — see the module note.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub auxiliary_baselines: BTreeMap<String, String>,
    /// The encoder recipe: what was run to produce these bytes. Opaque to
    /// this module, which compares it and never interprets it.
    pub recipe: String,
}

/// One measured claim, as the container stores it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredAttestation {
    pub binding: AttestationBinding,
    pub method: AttestationMethod,
    /// The metric the radius is stated in.
    pub metric: StoredId,
    /// The domain it is certified over.
    pub domain: StoredId,
    /// The measured error, in `metric` over `domain`.
    pub radius: f64,
}

/// The table as the container stores it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepresentationAttestations {
    /// Always [`REPRESENTATION_ATTESTATIONS_SCHEMA`] when this build
    /// wrote it.
    pub schema: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attestations: Vec<StoredAttestation>,
}

impl RepresentationAttestations {
    pub fn new(attestations: Vec<StoredAttestation>) -> Self {
        Self {
            schema: REPRESENTATION_ATTESTATIONS_SCHEMA,
            attestations,
        }
    }

    /// Judge the stored table into one that can be asked questions.
    ///
    /// Every refusal here is STRUCTURAL — about a row, readable with no
    /// container, no registry and no bytes. Whether a row matches the
    /// artifact it claims to describe is a separate question asked later
    /// and with more in hand; see [`AttestationTable`].
    pub fn judge(&self) -> Result<AttestationTable, VindexError> {
        if self.schema != REPRESENTATION_ATTESTATIONS_SCHEMA {
            return Err(VindexError::Parse(format!(
                "representation attestations: schema {} was written by another build; this one \
                 implements {REPRESENTATION_ATTESTATIONS_SCHEMA}",
                self.schema
            )));
        }
        let mut by_key: BTreeMap<(OperandAddress, u32), JudgedAttestation> = BTreeMap::new();
        for stored in &self.attestations {
            let judged = stored.judge()?;
            let key = (judged.binding.operand.clone(), judged.binding.extent_depth);
            if let Some(first) = by_key.get(&key) {
                return Err(VindexError::Parse(format!(
                    "representation attestations: {} at depth {} is attested twice — {} and {}; \
                     two measurements of one operand at one extent state no measurement, because \
                     nothing says which was meant",
                    judged.binding.operand.describe(),
                    judged.binding.extent_depth,
                    first.method.describe(),
                    judged.method.describe(),
                )));
            }
            by_key.insert(key, judged);
        }
        Ok(AttestationTable { by_key })
    }

    /// Read and judge the table `name` names, relative to `root`.
    pub fn read(root: &Path, name: &str) -> Result<AttestationTable, VindexError> {
        let path = root.join(name);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            VindexError::Parse(format!(
                "representation attestations `{}` are named by the index and cannot be read: {e}",
                path.display()
            ))
        })?;
        let stored: Self = serde_json::from_str(&text).map_err(|e| {
            VindexError::Parse(format!(
                "representation attestations `{}` are not readable as such: {e}",
                path.display()
            ))
        })?;
        stored.judge()
    }
}

impl StoredAttestation {
    /// One row, judged: every field present, the ids well-formed, and the
    /// radius a number a certificate will accept.
    fn judge(&self) -> Result<JudgedAttestation, VindexError> {
        let b = &self.binding;
        if b.operand.object.trim().is_empty() || b.operand.tensor.trim().is_empty() {
            return Err(VindexError::Parse(format!(
                "representation attestations: an attestation names an empty object or tensor \
                 ({}); an attestation about no operand is about nothing",
                b.operand.describe()
            )));
        }
        let at = || format!("{} at depth {}", b.operand.describe(), b.extent_depth);
        for (what, value) in [
            ("codec family", &b.codec_family),
            ("content digest", &b.content_digest),
            ("source digest", &b.source_digest),
            ("encoder recipe", &b.recipe),
            ("attesting authority", &self.method.authority),
        ] {
            if value.trim().is_empty() {
                return Err(VindexError::Parse(format!(
                    "representation attestations: {} names an empty {what}; every part of the \
                     binding is part of the identity, so an absent one is not a blank field but \
                     an attestation about nothing in particular",
                    at()
                )));
            }
        }
        if b.shape.is_empty() || b.shape.contains(&0) {
            return Err(VindexError::Parse(format!(
                "representation attestations: {} states shape {:?}; an operand with no elements \
                 has no error to measure",
                at(),
                b.shape
            )));
        }
        for (what, name) in b.auxiliary_baselines.iter() {
            if what.trim().is_empty() || name.trim().is_empty() {
                return Err(VindexError::Parse(format!(
                    "representation attestations: {} states an empty dependency baseline; a \
                     baseline that names nothing cannot be checked against anything",
                    at()
                )));
            }
        }

        // The ids and the radius are the codec plane's vocabulary, not a
        // parallel one — an attestation PRODUCES the same certificate a
        // codec declares, or composition would have two algebras.
        let describe = |e: super::represent::codec::CodecError| {
            VindexError::Parse(format!("representation attestations: {}: {e}", at()))
        };
        let metric =
            MetricId::new(self.metric.name.clone(), self.metric.version).map_err(describe)?;
        let domain =
            DomainId::new(self.domain.name.clone(), self.domain.version).map_err(describe)?;
        // A method id is validated by the SAME rules as a metric id and
        // is not one: "how it was measured" and "what the number means"
        // are different questions, and giving the method a MetricId would
        // invite composing against it.
        let method = SemanticId::new(
            "attestation method",
            self.method.method.name.clone(),
            self.method.method.version,
        )
        .map_err(|e| {
            VindexError::Parse(format!(
                "representation attestations: {}: its method id is malformed: {e}",
                at()
            ))
        })?;
        let certificate =
            FidelityCertificate::new(metric, domain, self.radius).map_err(describe)?;

        Ok(JudgedAttestation {
            binding: self.binding.clone(),
            method: AttestationMethod {
                authority: self.method.authority.clone(),
                method: StoredId::new(method.name(), method.version()),
            },
            certificate,
        })
    }
}

/// One attestation that survived judging.
///
/// It carries a real [`FidelityCertificate`], so from here on an attested
/// claim and a declared claim are the same kind of thing and compose
/// through the same algebra.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgedAttestation {
    pub binding: AttestationBinding,
    pub method: AttestationMethod,
    certificate: FidelityCertificate,
}

impl JudgedAttestation {
    /// What this attestation claims — subject to recognition and to the
    /// binding actually matching, neither of which this type asserts.
    ///
    /// Deliberately not a public field: a caller that reaches past
    /// recognition to the number is exactly the "presence implies trust"
    /// failure this plane exists to prevent, and a method is easier to
    /// find at a review than a struct field is.
    pub fn claimed(&self) -> &FidelityCertificate {
        &self.certificate
    }

    /// How this attestation reads when it is named in a refusal.
    pub fn describe(&self) -> String {
        format!(
            "{} at depth {}, attested {} {}",
            self.binding.operand.describe(),
            self.binding.extent_depth,
            self.certificate.describe(),
            self.method.describe()
        )
    }
}

/// The table after judging: at most one attestation per
/// `(operand, extent depth)`, every id well-formed, every radius a number.
///
/// **Two-stage admission.** What this table can answer is the TUPLE
/// question — is there an attestation for this operand at this extent,
/// does it name this codec family and revision, this shape, this recipe,
/// these dependency baselines — and all of it from metadata, before a
/// byte of payload is read, exactly as the auxiliary closure is admitted.
/// What it cannot answer is whether the CONTENT is still the content that
/// was measured: the container holds no per-tensor digest, so that check
/// needs the bytes and happens at preparation, when they are being read
/// anyway. An attestation that passes the tuple check and fails the digest
/// check is stale, and stale is unavailable.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AttestationTable {
    by_key: BTreeMap<(OperandAddress, u32), JudgedAttestation>,
}

impl AttestationTable {
    /// A container that declares no attestations.
    ///
    /// Every container written before this wave, and every one whose
    /// representations all declare their own radius. Absence is the
    /// ordinary case and means "nothing is attested" — never "nothing
    /// needed attesting".
    pub fn empty() -> Self {
        Self::default()
    }

    /// The attestation for `operand` at `depth`, if the container states
    /// one. `None` is absence, which is distinct from stale and from
    /// unrecognised, and all three are distinct from zero.
    pub fn at(&self, operand: &OperandAddress, depth: u32) -> Option<&JudgedAttestation> {
        self.by_key.get(&(operand.clone(), depth))
    }

    /// Whether anything is attested for `operand` at any extent — the
    /// question a diagnostic asks when `at` returned `None` and the
    /// caller wants to say "at depth 2, though" rather than "not at all".
    pub fn attests(&self, operand: &OperandAddress) -> bool {
        self.by_key.keys().any(|(o, _)| o == operand)
    }

    /// Every extent depth attested for `operand`, ascending.
    pub fn depths(&self, operand: &OperandAddress) -> Vec<u32> {
        self.by_key
            .keys()
            .filter(|(o, _)| o == operand)
            .map(|(_, depth)| *depth)
            .collect()
    }

    /// How many attestations the container states.
    pub fn stated(&self) -> usize {
        self.by_key.len()
    }
}
