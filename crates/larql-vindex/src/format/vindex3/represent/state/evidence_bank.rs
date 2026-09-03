//! **Which samples an observation was taken over.**
//!
//! Note what this is NOT. [`super::super::quality::QualityBank`] is
//! *"one bank of measurements over a fixed token sequence"* — the
//! OBSERVATION: measured KL, routing evidence, margin distributions. The
//! programme's "evidence bank" is the CORPUS those were taken over — 256
//! teacher-forced sequences of real prose, a directory with a
//! `manifest.json`. One word, two things, and only the second can
//! identify an experiment.
//!
//! Before this module the corpus had no type at all. Its identity was a
//! `sha256` of `manifest.json` computed inline in the Q2a harness and
//! written into the report as `bank_manifest_sha256` — which is the
//! right instinct and not sufficient, because **the harness slices the
//! corpus**. `LARQL_Q2A_SEQUENCES` takes 32 of 256, and a 32-sequence
//! reading and a 256-sequence reading share a manifest digest while
//! being different experiments. That slice is the difference between a
//! diagnostic and an authority run.
//!
//! ```text
//! IN   schema             the bank format the samples are in
//! IN   manifest digest    the corpus's own content hash
//! IN   ordered samples    WHICH sequences, in the order used
//! IN   positions/sample   how many positions each contributes
//!
//! OUT  where it is stored  /tmp has proven ephemeral; a path is where a
//!                          bank was last seen, not what it is
//! OUT  when it was built
//! OUT  what a human called it
//! ```
//!
//! The rule, stated so it can be argued with:
//!
//! > **If changing it can legitimately change the measured result or its
//! > comparability, it must change [`EvidenceBankId`]. If it cannot, it
//! > must not.**
//!
//! The excluded-path rule is the same one
//! [`super::super::compiler::SourceDependency`] already states for
//! containers — *identified by CONTENT, not by path* — and it is here
//! for the same reason: a bank that moved disk is the same bank, and a
//! registry that thought otherwise would re-run twenty minutes of
//! instrument time to learn what it already knew.

use serde::{Deserialize, Serialize};

use super::super::compile::hash_bytes;
use super::surface::{FIELD, RECORD, SECTION};

/// The canonical-form version every bank id is computed under.
pub const EVIDENCE_BANK_ID_VERSION: &str = "evidence-bank-id/v1";

/// **What a corpus IS**, by the samples it presents.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EvidenceBankId(String);

impl EvidenceBankId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn short(&self) -> &str {
        &self.0[..self.0.len().min(12)]
    }
}

impl std::fmt::Display for EvidenceBankId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The corpus an observation was taken over.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceBank {
    /// The bank format these samples are in. A reader that changed how
    /// it interprets the files is measuring something else.
    pub schema: String,
    /// The corpus's own `manifest.json` digest, as the Q2a harness
    /// already records it.
    pub manifest_sha256: String,
    /// **The samples actually used, in the order used.** Not a count: a
    /// different 32 of the same 256 is a different experiment, and a
    /// count cannot say which 32.
    pub samples: Vec<String>,
    /// Positions each sample contributes.
    pub positions_per_sample: u32,
    /// Where the bank was last seen. A hint for FINDING it, never part
    /// of what it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator_hint: Option<String>,
    /// What a human called it. Provenance, never identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl EvidenceBank {
    pub fn new(
        schema: impl Into<String>,
        manifest_sha256: impl Into<String>,
        samples: impl IntoIterator<Item = impl Into<String>>,
        positions_per_sample: u32,
    ) -> Self {
        Self {
            schema: schema.into(),
            manifest_sha256: manifest_sha256.into(),
            samples: samples.into_iter().map(Into::into).collect(),
            positions_per_sample,
            locator_hint: None,
            description: None,
        }
    }

    pub fn found_at(mut self, locator: impl Into<String>) -> Self {
        self.locator_hint = Some(locator.into());
        self
    }

    pub fn described(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// **Positions, computed — never multiplied by hand.**
    ///
    /// `LARQL_Q2A_SEQUENCES=256` is SEQUENCES; the bank is 256 × 32 =
    /// 8192 POSITIONS, and an acceptance check written against the wrong
    /// one of those would have accepted a diagnostic run and rejected the
    /// authority one. It nearly did.
    pub fn positions(&self) -> u64 {
        self.samples.len() as u64 * u64::from(self.positions_per_sample)
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// What this bank IS.
    pub fn id(&self) -> EvidenceBankId {
        let samples = self.samples.join(&RECORD.to_string());
        let input = format!(
            "{EVIDENCE_BANK_ID_VERSION}{SECTION}schema={}{FIELD}manifest={}{FIELD}\
             positions_per_sample={}{SECTION}samples={samples}",
            self.schema, self.manifest_sha256, self.positions_per_sample
        );
        EvidenceBankId(hash_bytes(input.as_bytes()))
    }
}
