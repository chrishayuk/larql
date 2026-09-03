//! **`MeasurementKey` — have we already run THIS experiment against
//! THIS physical state?**
//!
//! R5-F3 registered the shape and the correction that makes it work:
//! dedup on a bare map hash would be wrong, because it would forbid the
//! diagnostic → authority escalation the whole ladder depends on.
//!
//! ```text
//! state_id    which physical representation was measured
//! bank        which samples it was measured over
//! scale       whether the reading may be priced against a contract
//! instrument  what the measurement MEANS
//! ```
//!
//! ```text
//! recipes       are transactions
//! states        are what exists
//! measurements  are cached observations, keyed by state + query conditions
//! ```
//!
//! # The join with 1a and 1b
//!
//! The key reads the PHYSICAL id, and that is the whole point of the two
//! identities:
//!
//! ```text
//! RealizationId differs, RepresentationStateId same
//!     action search    → distinguish
//!     measurement      → collapse, reuse the observation
//! ```
//!
//! A protected tensor and a layout-refused one present identical bytes,
//! so a measurement of one IS a measurement of the other, while the moves
//! available from each still differ. Evidence deduplicates more
//! aggressively than search does, and this is where that cashes out.
//!
//! # What is deliberately absent
//!
//! No PASS/FAIL. No promotion status. No contract.
//!
//! The key says *what experiment was performed against what state*; it
//! does not say what the result meant. Classification stays downstream —
//! [`Margin`] and [`Frontier`] price an observation against a gate, and
//! [`decide_promotion`] rules on it — so that a later contract, or a
//! later promotion policy, reinterprets observations already held
//! instead of making them disappear. A key carrying the verdict would
//! turn every re-reading of old evidence into a re-measurement.
//!
//! [`Margin`]: super::super::constraint::Margin
//! [`Frontier`]: super::super::constraint::Frontier
//! [`decide_promotion`]: super::super::decision::decide_promotion

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::super::compile::hash_bytes;
use super::super::measurement::EvidenceScale;
use super::super::quality::QualityBank;
use super::evidence_bank::EvidenceBankId;
use super::identity::RepresentationStateId;
use super::instrument::InstrumentSemanticsId;
use super::surface::{FIELD, SECTION};
use crate::error::VindexError;

/// The canonical-form version every key is computed under.
pub const MEASUREMENT_KEY_VERSION: &str = "measurement-key/v1";

/// **What makes two observations the same observation.**
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementKey {
    state: RepresentationStateId,
    bank: EvidenceBankId,
    scale: EvidenceScale,
    instrument: InstrumentSemanticsId,
    /// The digest of the four above. Ordering is by this, so the key can
    /// index a registry without [`EvidenceScale`] having to grow an
    /// ordering it has no meaning for.
    digest: String,
}

impl MeasurementKey {
    pub fn new(
        state: &RepresentationStateId,
        bank: &EvidenceBankId,
        scale: EvidenceScale,
        instrument: &InstrumentSemanticsId,
    ) -> Self {
        let scale_name = match scale {
            EvidenceScale::Diagnostic => "diagnostic",
            EvidenceScale::Authority => "authority",
        };
        let input = format!(
            "{MEASUREMENT_KEY_VERSION}{SECTION}state={state}{FIELD}bank={bank}{FIELD}\
             scale={scale_name}{FIELD}instrument={instrument}"
        );
        Self {
            state: state.clone(),
            bank: bank.clone(),
            scale,
            instrument: instrument.clone(),
            digest: hash_bytes(input.as_bytes()),
        }
    }

    pub fn state(&self) -> &RepresentationStateId {
        &self.state
    }

    pub fn bank(&self) -> &EvidenceBankId {
        &self.bank
    }

    pub fn scale(&self) -> EvidenceScale {
        self.scale
    }

    pub fn instrument(&self) -> &InstrumentSemanticsId {
        &self.instrument
    }

    pub fn as_str(&self) -> &str {
        &self.digest
    }

    pub fn short(&self) -> &str {
        &self.digest[..self.digest.len().min(12)]
    }
}

impl std::fmt::Display for MeasurementKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.digest)
    }
}

impl PartialOrd for MeasurementKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MeasurementKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.digest.cmp(&other.digest)
    }
}

/// **Observations, keyed by what they observed.**
///
/// Holds [`QualityBank`]s — the readings — and nothing about what they
/// meant. A registry that also stored verdicts would have to be
/// invalidated whenever a contract moved; this one does not.
///
/// Serialises as a LIST of records rather than a map. A key is four
/// identities, not a string, and flattening it to one so it could be a
/// JSON object key would throw away the parts a reader needs to see why
/// two observations are distinct.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(into = "RegistryRecords", try_from = "RegistryRecords")]
pub struct MeasurementRegistry {
    observations: BTreeMap<MeasurementKey, QualityBank>,
}

/// The persisted shape.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RegistryRecords {
    observations: Vec<RegistryRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegistryRecord {
    key: MeasurementKey,
    observation: QualityBank,
}

impl From<MeasurementRegistry> for RegistryRecords {
    fn from(registry: MeasurementRegistry) -> Self {
        Self {
            observations: registry
                .observations
                .into_iter()
                .map(|(key, observation)| RegistryRecord { key, observation })
                .collect(),
        }
    }
}

impl TryFrom<RegistryRecords> for MeasurementRegistry {
    type Error = VindexError;

    /// Refuses a file that names one experiment twice.
    ///
    /// The in-memory registry cannot hold a duplicate, so a stored one
    /// carrying two records under a single key was written by something
    /// that bypassed [`MeasurementRegistry::record`] — and loading it
    /// would silently keep whichever came last.
    fn try_from(records: RegistryRecords) -> Result<Self, Self::Error> {
        let mut observations = BTreeMap::new();
        for record in records.observations {
            let short = record.key.short().to_string();
            if observations
                .insert(record.key, record.observation)
                .is_some()
            {
                return Err(VindexError::Parse(format!(
                    "measurement {short} is recorded twice — a stored registry that names one \
                     experiment twice was not written by `record`, and loading it would keep \
                     whichever came last"
                )));
            }
        }
        Ok(Self { observations })
    }
}

impl MeasurementRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether this exact experiment has already been run.
    pub fn contains(&self, key: &MeasurementKey) -> bool {
        self.observations.contains_key(key)
    }

    pub fn get(&self, key: &MeasurementKey) -> Option<&QualityBank> {
        self.observations.get(key)
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &MeasurementKey> {
        self.observations.keys()
    }

    /// Every observation of one physical state, whatever bank, scale or
    /// instrument produced it.
    pub fn of_state<'a>(
        &'a self,
        state: &'a RepresentationStateId,
    ) -> impl Iterator<Item = (&'a MeasurementKey, &'a QualityBank)> + 'a {
        self.observations
            .iter()
            .filter(move |(k, _)| &k.state == state)
    }

    /// **Record an observation, refusing a contradiction.**
    ///
    /// Re-recording an identical reading is a no-op: that is a control
    /// witness reproducing, which is exactly what a replayed round should
    /// do. A DIFFERENT reading under the same key is not a duplicate — it
    /// says the experiment is not reproducible, and quietly keeping
    /// either one would hide that. The refusal is the finding.
    pub fn record(
        &mut self,
        key: MeasurementKey,
        observation: QualityBank,
    ) -> Result<(), VindexError> {
        match self.observations.get(&key) {
            Some(held) if held == &observation => Ok(()),
            Some(_) => Err(VindexError::Parse(format!(
                "measurement {} is already held with a different reading — the same state, \
                 bank, scale and instrument produced two different observations, so the \
                 experiment is not reproducible and neither reading may silently win",
                key.short()
            ))),
            None => {
                self.observations.insert(key, observation);
                Ok(())
            }
        }
    }
}
