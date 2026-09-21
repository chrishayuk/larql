//! **The immutable record an execution yields, and ingestion judges.**

use std::fmt;

use serde::{Deserialize, Serialize};

use super::super::actuate::executor::Observed;
use super::super::actuate::prepare::PreparedExperiment;
use super::super::compile::hash_bytes;
use super::super::measure::outcome::VerifiedFacts;
use super::super::quality::QualityBank;
use super::super::state::key::MeasurementKey;
use super::state_evidence::EstablishedState;

/// **What a run produced, sealed against its own contents.**
///
/// Immutable by construction: every field is private, there is no
/// setter. [`MeasurementArtifact::from_execution`] constructs and seals a
/// record; deserialization restores an untrusted claim that ingestion must
/// validate. A caller cannot mutate an existing instance in place.
///
/// # What it deliberately cannot hold
///
/// No verdict, no rank, no promotion, no admissibility flag. The OPT-6
/// forecast's arm 10 prefers structural impossibility to "ignored": a
/// field that cannot be expressed cannot be ignored by accident, and an
/// executor cannot smuggle a conclusion into facts through a field that
/// does not exist. It is not enough to intend this — the serialized
/// shape is pinned by a test, so widening it later fails rather than
/// passing quietly.
///
/// # What the seal is for, and what it is not
///
/// The seal catches an artifact that changed after it was made — the
/// round trip through bytes being the realistic vector, since the type
/// forbids the in-process one. It is NOT evidence that the observation
/// is true, that the key is the right key, or that the bytes executed
/// established the state claimed. Those are ingestion's questions, and
/// a valid seal is precisely the condition under which they become
/// worth asking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementArtifact {
    /// The experiment this is an observation OF, as the run restated it.
    key: MeasurementKey,
    /// The procedure the run was performed under, BY NAME — and in no
    /// digest, exactly as `MeasurementProtocol::procedure` is in none.
    /// Ingestion resolves it; nothing here defaults it.
    procedure: String,
    /// The container identity the run executed against, so ingestion can
    /// refuse an artifact produced against a container the snapshot no
    /// longer describes (arm 5) rather than discovering it later.
    ///
    /// The SEMANTIC identity — `SourceIdentity::semantic_digest`, what
    /// `RepresentationStateId` reads — and deliberately NOT
    /// `SourceIdentity::artifact`, which is provenance that moves on a
    /// re-export changing no value. Sealing provenance here would make
    /// arm 5 refuse artifacts whose container is semantically the one
    /// authorised, which is a false refusal dressed as rigour.
    source_semantic_digest: String,
    /// Independently read artifact binding captured with the observation.
    candidate_authority_digest: String,
    observation: QualityBank,
    /// Every validity condition the run checked. The executor's own
    /// account of itself: evidence to be re-derived against, never an
    /// authority to be trusted.
    verified: VerifiedFacts,
    /// What the executor did, in its own vocabulary. A provenance line
    /// for a reader, never read as authority.
    execution_note: String,
    seal: String,
}

/// The digest input. A named struct rather than a tuple so the sealed
/// set is legible, and so adding a field to the artifact without
/// sealing it is a visible omission here rather than an invisible one.
#[derive(Serialize)]
struct Sealed<'a> {
    key: &'a MeasurementKey,
    procedure: &'a str,
    source_semantic_digest: &'a str,
    candidate_authority_digest: &'a str,
    observation: &'a QualityBank,
    verified: &'a VerifiedFacts,
    execution_note: &'a str,
}

/// Why an artifact is not usable as the record it claims to be.
///
/// Typed, and its rendered form names the expected and the observed
/// authority — optimizer contract 5, which OPT6-N1 resolved that
/// ingestion receives no exemption from.
#[derive(Debug, Clone, PartialEq)]
pub enum ArtifactRefusal {
    /// The contents do not hash to the seal they carry.
    SealBroken { expected: String, observed: String },
    /// The contents could not be canonicalised to seal at all.
    Unsealable { detail: String },
    /// There is no authorised experiment to bind an observation to.
    NotAuthorised { detail: String },
    /// The run restated a DIFFERENT experiment from the one prepared.
    /// No artifact is produced — not a suspect one, none.
    SubstitutedExperiment { expected: String, observed: String },
}

impl fmt::Display for ArtifactRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SealBroken { expected, observed } => write!(
                f,
                "this artifact does not hash to the seal it carries — it was sealed as \
                 {expected} and its contents now hash to {observed}, so it changed after \
                 it was made and no longer records the run it claims to"
            ),
            Self::NotAuthorised { detail } => write!(
                f,
                "there is no authorised experiment to bind this observation to: {detail} \
                 — an observation without an authorised experiment is not a measurement \
                 of anything, and sealing one would only make it look like a record"
            ),
            Self::SubstitutedExperiment { expected, observed } => write!(
                f,
                "the run restated a different experiment from the one prepared — it was \
                 authorised for {expected} and reports an observation of {observed}, so \
                 an artifact for either would be a substitution"
            ),
            Self::Unsealable { detail } => write!(
                f,
                "this artifact could not be canonicalised to seal: {detail} — an artifact \
                 that cannot be sealed cannot be shown to be unchanged, so it may not be \
                 treated as a record"
            ),
        }
    }
}

impl MeasurementArtifact {
    /// Seal a run's output into an artifact.
    ///
    /// Kept private so the public constructor binds the record to its
    /// prepared request. Deserialized records can still claim any key;
    /// ingestion independently validates those claims before acceptance.
    fn sealing(
        key: MeasurementKey,
        procedure: impl Into<String>,
        source_semantic_digest: impl Into<String>,
        candidate_authority_digest: &str,
        observation: QualityBank,
        verified: VerifiedFacts,
        execution_note: impl Into<String>,
    ) -> Result<Self, ArtifactRefusal> {
        let procedure = procedure.into();
        let source_semantic_digest = source_semantic_digest.into();
        let execution_note = execution_note.into();
        let seal = seal_of(&Sealed {
            key: &key,
            procedure: &procedure,
            source_semantic_digest: &source_semantic_digest,
            candidate_authority_digest,
            observation: &observation,
            verified: &verified,
            execution_note: &execution_note,
        })?;
        Ok(Self {
            key,
            procedure,
            source_semantic_digest,
            candidate_authority_digest: candidate_authority_digest.into(),
            observation,
            verified,
            execution_note,
            seal,
        })
    }

    /// **Bind an observation to the experiment that authorised it.**
    ///
    /// The candidate binding comes from an independent read captured with the run.
    /// Ingestion must repeat that read; this captured evidence is not final acceptance.
    ///
    /// The only public constructor. Every field the artifact carries is
    /// taken from exactly one side, so most of the binding is structural
    /// rather than checked:
    ///
    /// ```text
    /// procedure, source identity   <- the prepared request, ALWAYS
    /// observation, verified facts  <- the Observed, ALWAYS
    /// provenance                   <- the executor's own note
    /// key                          <- checked to agree, then the request's
    /// ```
    ///
    /// A structural source cannot disagree, so it cannot be got wrong; the
    /// key is the one thing both sides state, so it is the one thing worth
    /// checking. When they disagree NO artifact is produced — not a suspect
    /// one, none — because a sealed artifact is exactly the thing that later
    /// looks like a record.
    ///
    /// # What this does NOT make true
    ///
    /// It closes substitution at creation. It does not make a
    /// `MeasurementArtifact` trusted. After this returns, the artifact
    /// still only SAYS "I am the observation produced for experiment K";
    /// whether the authoritative record independently agrees that this
    /// observation constitutes K is ingestion's question, asked against
    /// the snapshot rather than against the artifact's own say-so. Keeping
    /// those apart is why executor restatement and ingestion authority
    /// stay two steps and not one.
    pub fn from_execution(
        prepared: &PreparedExperiment,
        observed: &Observed,
        candidate: &EstablishedState,
    ) -> Result<Self, ArtifactRefusal> {
        let request = prepared
            .request()
            .ok_or_else(|| ArtifactRefusal::NotAuthorised {
                detail: match prepared {
                    PreparedExperiment::Exhausted => {
                        "the optimiser reports nothing left to measure".to_string()
                    }
                    PreparedExperiment::NotSelectable { detail } => detail.clone(),
                    PreparedExperiment::NotPreparable(refusal) => refusal.to_string(),
                    PreparedExperiment::Ready(_) => unreachable!("Ready yields a request"),
                },
            })?;

        if observed.key != *request.key() {
            return Err(ArtifactRefusal::SubstitutedExperiment {
                expected: format!("{:?}", request.key()),
                observed: format!("{:?}", observed.key),
            });
        }

        Self::sealing(
            request.key().clone(),
            request.procedure(),
            request.model().semantic_digest(),
            candidate.candidate_authority_digest(),
            observed.observation.clone(),
            observed.verified.clone(),
            observed.execution_note.clone(),
        )
    }

    /// Does this artifact still hash to the seal it carries?
    ///
    /// Ingestion asks this FIRST. Everything downstream reads these
    /// fields as the run's claims, and a claim that changed in transit
    /// is not the run's.
    pub fn verify_seal(&self) -> Result<(), ArtifactRefusal> {
        let observed = seal_of(&Sealed {
            key: &self.key,
            procedure: &self.procedure,
            source_semantic_digest: &self.source_semantic_digest,
            candidate_authority_digest: &self.candidate_authority_digest,
            observation: &self.observation,
            verified: &self.verified,
            execution_note: &self.execution_note,
        })?;
        if observed == self.seal {
            Ok(())
        } else {
            Err(ArtifactRefusal::SealBroken {
                expected: self.seal.clone(),
                observed,
            })
        }
    }

    pub fn key(&self) -> &MeasurementKey {
        &self.key
    }

    pub fn procedure(&self) -> &str {
        &self.procedure
    }

    pub fn source_semantic_digest(&self) -> &str {
        &self.source_semantic_digest
    }

    pub fn candidate_authority_digest(&self) -> &str {
        &self.candidate_authority_digest
    }

    pub fn observation(&self) -> &QualityBank {
        &self.observation
    }

    pub fn verified(&self) -> &VerifiedFacts {
        &self.verified
    }

    pub fn execution_note(&self) -> &str {
        &self.execution_note
    }

    pub fn seal(&self) -> &str {
        &self.seal
    }
}

fn seal_of(sealed: &Sealed<'_>) -> Result<String, ArtifactRefusal> {
    let bytes = serde_json::to_vec(sealed).map_err(|e| ArtifactRefusal::Unsealable {
        detail: e.to_string(),
    })?;
    Ok(hash_bytes(&bytes))
}

#[cfg(test)]
mod tests;
