//! **What the standing intent's digests stand for.**
//!
//! [`MeasurementIntent`] names the experiment the next run would be as
//! three values, two of which are one-way hashes:
//!
//! ```text
//! bank        EvidenceBankId          a digest of a corpus declaration
//! scale       EvidenceScale           in the clear
//! instrument  InstrumentSemanticsId   a digest of a meaning declaration
//! ```
//!
//! That is exactly right for identity and useless for actuation. A
//! caller holding `MeasurementKey` knows two observations are the same
//! observation and cannot say which corpus to read, how many sequences
//! of it, or what the reading would mean — the digests are one-way, and
//! before this module `EvidenceBank` and [`InstrumentSemantics`] had no
//! production construction site at all. Every one in the tree was a test
//! or a fixture.
//!
//! So the record carries the declarations its own intent stands for.
//! This is 4b's movement one plane over:
//!
//! ```text
//! 4b   next_experiment could not ANSWER          → the record carries the
//!      until the record carried the accounting     accounting authority it
//!      authority it prices from                    prices from
//!
//! 5b   a selected experiment cannot be RUN       → the record carries the
//!      until the record carries the protocol       protocol authority it
//!      authority it measures under                 measures under
//! ```
//!
//! Both are inputs to add, not conclusions to invent.
//!
//! # Self-checking, because a declaration can lie
//!
//! [`super::instrument`] states the limit it could not close: nothing
//! forces a runner's declaration to match what it does, and the
//! mitigation is construction rather than validation. Here BOTH halves
//! are present for the first time — the declaration and the digest the
//! search is actually running under — so the check is possible, and
//! [`MeasurementProtocol::describes`] makes it mandatory before anything
//! is prepared.
//!
//! A record whose protocol digests to a different bank than its intent
//! names is not a record with a cosmetic inconsistency. It is a record
//! that would run one experiment and file the result under another.
//!
//! # Two things called `procedure`, and they are not the same claim
//!
//! ```text
//! InstrumentSemantics::procedure   "q2a-teacher-forced/baseline-vs-overlay"
//!                                  what the reading MEANS; inside the digest
//!
//! MeasurementProtocol::procedure   "teacher-forced-two-arm/v1"
//!                                  what this build must PERFORM; resolved
//!                                  by name at preparation, outside every digest
//! ```
//!
//! They are deliberately separate fields and no cross-check is asserted
//! between them, because no mapping between the two namespaces has been
//! established. Inventing one — "a q2a instrument implies the
//! teacher-forced procedure" — would be a claim about instruments this
//! programme has not made, and it would be invisible the first time it
//! was wrong. When such a mapping is earned it arrives as a registration
//! with its own evidence, not as a string comparison here.

use serde::{Deserialize, Serialize};

use super::super::measurement::EvidenceScale;
use super::candidate::MeasurementIntent;
use super::evidence_bank::{EvidenceBank, EvidenceBankId};
use super::instrument::{InstrumentSemantics, InstrumentSemanticsId};
use super::key::MeasurementKey;

/// **A protocol that describes a different experiment from the one the
/// record searches under.**
///
/// One variant per identity rather than a single "mismatch", because
/// the two are fixed by different actions: a wrong bank means the
/// declaration names another corpus, a wrong instrument means it names
/// another meaning, and a reader told only "mismatch" has to diff two
/// digests to find out which.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolMismatch {
    Bank {
        declared: EvidenceBankId,
        searched: EvidenceBankId,
    },
    Instrument {
        declared: InstrumentSemanticsId,
        searched: InstrumentSemanticsId,
    },
}

impl std::fmt::Display for ProtocolMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bank { declared, searched } => write!(
                f,
                "the declared corpus digests to {} but the search runs under {} — the run \
                 would read one body of data and the observation would be filed under another",
                declared.short(),
                searched.short()
            ),
            Self::Instrument { declared, searched } => write!(
                f,
                "the declared instrument digests to {} but the search runs under {} — the \
                 reading would mean one thing and be recorded as meaning another",
                declared.short(),
                searched.short()
            ),
        }
    }
}

/// **The declarations behind an experiment's identities.**
///
/// Carries no scale. The scale is in the clear on the intent and on
/// every key, and duplicating it here would create a second authority
/// for the one part of the experiment that never needed one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementProtocol {
    /// The corpus, as [`EvidenceBank`] defines it: schema, manifest
    /// digest, and WHICH samples in what order.
    pub bank: EvidenceBank,
    /// What a reading under this protocol means.
    pub instrument: InstrumentSemantics,
    /// The procedure a run is performed under, BY NAME. Resolved at
    /// preparation and refused when this build does not implement it;
    /// nothing here defaults it, and it is in no digest.
    pub procedure: String,
}

impl MeasurementProtocol {
    pub fn new(
        bank: EvidenceBank,
        instrument: InstrumentSemantics,
        procedure: impl Into<String>,
    ) -> Self {
        Self {
            bank,
            instrument,
            procedure: procedure.into(),
        }
    }

    /// Whether these declarations are the ones the intent names.
    pub fn describes(&self, intent: &MeasurementIntent) -> Result<(), ProtocolMismatch> {
        self.check(&intent.bank, &intent.instrument)
    }

    /// Whether these declarations are the ones a KEY names.
    ///
    /// Separate from [`Self::describes`] rather than derived from it:
    /// the selected experiment's key is what will actually be run, and
    /// a policy that ever selects under an intent other than the
    /// standing one must be caught by checking the key itself.
    pub fn describes_key(&self, key: &MeasurementKey) -> Result<(), ProtocolMismatch> {
        self.check(key.bank(), key.instrument())
    }

    fn check(
        &self,
        bank: &EvidenceBankId,
        instrument: &InstrumentSemanticsId,
    ) -> Result<(), ProtocolMismatch> {
        let declared = self.bank.id();
        if &declared != bank {
            return Err(ProtocolMismatch::Bank {
                declared,
                searched: bank.clone(),
            });
        }
        let declared = self.instrument.id();
        if &declared != instrument {
            return Err(ProtocolMismatch::Instrument {
                declared,
                searched: instrument.clone(),
            });
        }
        Ok(())
    }

    /// **The standing intent these declarations stand behind.**
    ///
    /// Derived rather than supplied, so a record built from a protocol
    /// cannot describe one experiment and search under another. The
    /// scale is the caller's because it is the one part of an intent
    /// that is not a declaration — it is a claim about how much evidence
    /// a reading carries, and the same corpus and instrument serve both
    /// scales.
    pub fn intent(&self, scale: EvidenceScale) -> MeasurementIntent {
        MeasurementIntent::new(self.bank.id(), scale, self.instrument.id())
    }

    /// **How many samples a run consumes**, from the declaration rather
    /// than from an environment variable.
    ///
    /// The slice the Q2a harness took with `LARQL_Q2A_SEQUENCES` is the
    /// difference between a diagnostic and an authority run, and
    /// [`EvidenceBank`] exists because a count cannot say WHICH samples.
    /// Reading it here means a record-derived run consumes exactly the
    /// samples its own bank id was computed over.
    pub fn sequences(&self) -> usize {
        self.bank.sample_count()
    }
}

#[cfg(test)]
#[path = "protocol_tests.rs"]
mod tests;
