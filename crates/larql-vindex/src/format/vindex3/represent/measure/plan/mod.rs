//! **`teacher-forced-two-arm/plan-v1`**: MEASURE-PLAN-1's procedure
//! (`docs/measure-plan-1.md`).
//!
//! One procedure measures any candidate realization of a VINDEX3 container
//! against a reference realization, for any model the plan executes. It
//! carries the Kimi procedure's five claims:
//!
//! ```text
//! measurement
//! + candidate-scope proof       the changed variable exists
//! + physical-attribution proof  each arm bound what it declared
//! + non-candidate identity      what both arms bound is the same bytes
//! + seal/read consistency       every bound representation, and every
//!                               bank payload, matches its seal
//! ```
//!
//! plus the corpus proof and the null arm. The numbers are computable
//! without any of the proofs and would look the same, which is why a
//! violation is a typed refusal ([`PlanInadmissible`]), never a warning. A
//! machine problem is [`PlanExecutionFailure`] instead: nothing was
//! measured, so it is worth retrying. An inadmissible run is not.
//!
//! The Kimi procedure (`teacher-forced-two-arm/v1`) is untouched.

pub mod arm;
pub mod identity;
pub mod metrics;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use arm::{ArmDescription, TeacherForcedArm};
use metrics::Summary;

/// This procedure's name.
pub const PROCEDURE: &str = "teacher-forced-two-arm/plan-v1";

/// Samples the null arm scores twice. The freeze fixes `min(4, sequences)`.
pub const NULL_ARM_SAMPLES: usize = 4;

/// Output files, written into the request's output directory.
pub const REPORT_FILE: &str = "report.json";
pub const POSITIONS_FILE: &str = "positions.jsonl";
pub const RECEIPT_FILE: &str = "receipt.json";

/// UNCERTAINTY-2's sample-size facts for a KL p99, stated in every report
/// so an underpowered reading says so.
pub const P99_BIASED_BELOW_SEQUENCES: usize = 32;
pub const P99_QUARTER_PRECISION_SEQUENCES: usize = 64;

/// What to measure. The arms are built by the caller, which knows the
/// backends; the procedure reads their descriptions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanMeasureRequest {
    /// Token bank directory (`teacher-forced-token-bank/v1`).
    pub bank: PathBuf,
    /// Samples to measure, in bank order from `seq-000`.
    pub sequences: usize,
    /// A name for the run, recorded in the receipt.
    pub label: String,
    /// Directory the report, positions and receipt are written to. It must
    /// not already hold a report.
    pub output: PathBuf,
}

/// Nothing was measured. Worth retrying once the cause is fixed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanExecutionFailure {
    /// The request is not runnable as written.
    RequestRefused { detail: String },
    /// A bank, container or index could not be read.
    ArtifactUnreadable { detail: String },
    /// An arm could not score a sample.
    StepRefused { arm: String, detail: String },
    /// The output could not be written.
    OutputUnwritable { detail: String },
}

/// Numbers may exist, and they are NOT evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanInadmissible {
    /// The reference arm, run twice over the same sample, did not produce
    /// bit-identical logits. Any KL it reports could be device noise.
    NullArmNotZero { sample: usize, position: usize },
    /// The arms bound the same bytes through the same arm: the changed
    /// variable does not exist.
    CandidateCompilesNothing,
    /// A representation both arms bound is not the same bytes in the two
    /// containers.
    ProtectedOperandChanged { representation: String },
    /// An arm bound something other than it declared.
    UnexpectedPhysicalRead { arm: String, detail: String },
    /// Bytes read are not the bytes sealed: a bank payload, the bank's own
    /// manifest, or a bound representation.
    SealMismatch {
        what: String,
        expected: String,
        found: String,
    },
    /// The bank was tokenised for another model.
    CorpusNotForThisModel {
        arm: String,
        bank: String,
        model: String,
    },
    /// The two arms scored a different number of positions, or a different
    /// vocabulary, for the same sample.
    PositionCountMismatch { sample: usize, detail: String },
}

/// Why a run produced no evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanRefusal {
    Execution(PlanExecutionFailure),
    Inadmissible(PlanInadmissible),
}

/// What the run checked, recorded rather than asserted.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PlanVerifiedFacts {
    /// Samples the null arm scored twice and found bit-identical.
    pub null_arm_samples: usize,
    /// The representations only the candidate bound.
    pub changed_representations: Vec<String>,
    /// Whether the two arms are different execution arms.
    pub arm_changed: bool,
    /// Arms whose bound objects were checked against their declaration.
    pub attributed_arms: usize,
    /// Representations bound by both arms and checked byte-identical.
    pub protected_representations: usize,
    /// Bound representations whose bytes were re-hashed against their seal.
    pub sealed_representations: usize,
    /// Bank payloads read against their seal.
    pub bank_samples_read: usize,
    /// Arms whose tokenizer was checked against the bank.
    pub tokenizer_checked_arms: usize,
    /// Positions measured.
    pub positions: u64,
}

impl PlanVerifiedFacts {
    /// Whether every proof an admissible reading needs was carried out.
    /// Explicit, not a count.
    pub fn complete(&self, sequences: usize) -> bool {
        let _ = sequences;
        todo!("MEASURE-PLAN-1 PR 2")
    }
}

/// What a finished run says about itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanReceipt {
    pub procedure: String,
    pub label: String,
    pub bank_id: String,
    pub reference: ArmDescription,
    pub candidate: ArmDescription,
    pub facts: PlanVerifiedFacts,
    pub summary: Summary,
}

/// Run the procedure. On success the report, positions and a receipt are
/// in `request.output`; on refusal a receipt naming the refusal is written
/// there when the directory is writable.
pub fn run(
    request: &PlanMeasureRequest,
    reference: &mut dyn TeacherForcedArm,
    candidate: &mut dyn TeacherForcedArm,
) -> Result<PlanReceipt, PlanRefusal> {
    let _ = (request, reference, candidate);
    todo!("MEASURE-PLAN-1 PR 2")
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
