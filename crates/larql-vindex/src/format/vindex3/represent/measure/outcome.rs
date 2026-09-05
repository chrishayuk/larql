//! **Execution failure is not validity failure.**
//!
//! The teacher-forced runner is not a benchmark with some assertions
//! around it. Reading it, it is five claims at once:
//!
//! ```text
//! measurement
//! + candidate-scope proof
//! + physical-attribution proof
//! + non-candidate identity proof
//! + seal/read consistency proof
//! ```
//!
//! which is why extracting "the part that computes KL" would be
//! dangerous: the numbers are computable without any of the other four,
//! and would look exactly the same.
//!
//! As a `#[cfg(test)]` harness those four were `assert!`, and a test
//! runner turned a violation into a red test. A CALLABLE procedure has
//! no test runner, so each becomes a typed refusal — and the two kinds
//! must not be confused:
//!
//! ```text
//! ExecutionFailure    nothing was measured
//!                     retry, fix the machine, open the file
//!
//! Inadmissible        numbers may exist and are NOT evidence
//!                     the run happened and proved nothing
//! ```
//!
//! An agent told only "it failed" would retry the second class forever.
//! An agent told "the candidate arm read bytes outside its sealed
//! scope" knows the experiment was mis-prepared, not unlucky.
//!
//! # Every variant is a check that exists
//!
//! Nothing here is anticipated. Each one is a condition
//! `q2a_teacher_forced` asserts today, and each variant's doc names
//! what it was. That is the foreign reference: the vocabulary is
//! derived from a harness that has produced real Rung 4/5 verdicts,
//! not designed against an imagined one.

use serde::{Deserialize, Serialize};

/// The run could not be performed. Nothing was measured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionFailure {
    /// No Metal device. On macOS the shader library failed to build;
    /// elsewhere there is no backend at all.
    BackendUnavailable { detail: String },
    /// A named artifact would not open.
    ArtifactUnreadable {
        what: String,
        path: String,
        detail: String,
    },
    /// A layer did not load with zero missing operands.
    ///
    /// `build_stack` panics here: *"layer {i} must load with zero
    /// missing operands"*. A partly-loaded arm is not a cheaper arm, it
    /// is a different model.
    LayerIncomplete { layer: usize, detail: String },
    /// The stack ends on a device layer, so the head must attach.
    HeadDidNotAttach,
    /// A teacher-forced step refused mid-sequence.
    StepRefused {
        sequence: usize,
        position: usize,
        detail: String,
    },
}

/// The run happened and its numbers are not admissible evidence.
///
/// The distinction that matters: every one of these can occur with a
/// complete, plausible-looking set of logits already computed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Inadmissible {
    /// The residency SET would wire ~94 GB of expert bank, past the
    /// wired-collector wall (~45 GB), and the run would degrade into a
    /// measurement of the collector rather than of the representation.
    ///
    /// The harness panics on `LARQL_RESIDENCY_SET` for exactly this.
    ResidencyModeWouldBeMeasured { detail: String },
    /// The corpus was exported for a different model.
    ///
    /// `assert_eq!(g.hidden, bank_hidden, "the bank was exported for
    /// this model")`. Teacher-forced rows of the wrong width are not a
    /// smaller experiment; they are a different one.
    CorpusNotForThisModel {
        corpus_hidden: usize,
        model_hidden: usize,
    },
    /// The overlay compiles nothing, so the two arms are one arm and
    /// the changed variable does not exist.
    CandidateCompilesNothing,
    /// An arm bound a store or encoding it was not supposed to.
    ///
    /// Covers both directions the harness checks per compiled layer:
    /// the BASELINE must be entirely source-backed at BF16 whatever the
    /// candidate's scope, and the CANDIDATE must substitute exactly the
    /// projections the overlay compiled and leave the rest
    /// source-backed.
    UnexpectedPhysicalRead {
        arm: String,
        layer: usize,
        projection: String,
        expected_store: String,
        actual_store: String,
    },
    /// A projection OUTSIDE the candidate's scope was not the same
    /// bytes in both arms.
    ///
    /// The harness compares `region.bytes().as_ptr()` — pointer-
    /// identical, not merely same-named — because the only difference
    /// between the arms must be the compiled one.
    ProtectedOperandChanged { layer: usize, projection: String },
    /// A compiled projection was not identity-addressed over every
    /// expert, or an uncompiled one was not table-addressed.
    ///
    /// A compiled projection must resolve ANY route, including one the
    /// baseline never took; a projection-scoped candidate leaves the
    /// others table-addressed over the source, and that asymmetry
    /// inside one layer is what the binding exists to express.
    AddressingMismatch {
        layer: usize,
        projection: String,
        detail: String,
    },
    /// The bytes the loader read are not the bytes the compiler sealed.
    ///
    /// `verify_reads_match_seals`. Without it, "the candidate" is
    /// whatever is on disk under a path the overlay names.
    SealMismatch { layer: u32, detail: String },
    /// A routed operand carries no seal to check against.
    OperandNotSealed { tensor: String },
    /// The bank does not hold the positions the request asked for.
    ///
    /// `assert_eq!(bank.positions, sequences * positions_per_seq)`. A
    /// gate judging on tail statistics over a different position count
    /// is judging a different experiment (1c).
    PositionCountMismatch { expected: u64, measured: u64 },
    /// The verdict was drawn under a gate other than the requested one.
    ///
    /// The defect 5a-0a closed structurally, kept here because a
    /// receipt must be able to SAY it rather than rely on the code
    /// still being right.
    GateMismatch {
        requested: String,
        evaluated: String,
    },
}

/// Why a measurement produced no admissible reading.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementRefusal {
    Execution(ExecutionFailure),
    Inadmissible(Inadmissible),
}

impl MeasurementRefusal {
    /// Whether anything was measured at all.
    ///
    /// The question an agent needs answered before deciding to retry:
    /// an execution failure may succeed next time, and an inadmissible
    /// run produces the same non-evidence however often it is repeated.
    pub fn nothing_was_measured(&self) -> bool {
        matches!(self, Self::Execution(_))
    }

    /// Whether repeating this run unchanged could produce evidence.
    pub fn worth_retrying(&self) -> bool {
        self.nothing_was_measured()
    }
}

impl std::fmt::Display for MeasurementRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execution(e) => write!(f, "the run could not be performed: {e:?}"),
            Self::Inadmissible(i) => write!(
                f,
                "the run produced numbers that are not admissible evidence: {i:?}"
            ),
        }
    }
}

impl std::error::Error for MeasurementRefusal {}

impl From<ExecutionFailure> for MeasurementRefusal {
    fn from(e: ExecutionFailure) -> Self {
        Self::Execution(e)
    }
}

impl From<Inadmissible> for MeasurementRefusal {
    fn from(i: Inadmissible) -> Self {
        Self::Inadmissible(i)
    }
}

/// **What was checked, recorded rather than asserted.**
///
/// The other half of turning assertions into a contract: a condition
/// that merely fails loudly leaves no trace when it passes, and a
/// receipt that cannot say what it verified is indistinguishable from
/// one that verified nothing.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct VerifiedFacts {
    /// Layers the overlay compiles, as the run observed them.
    pub compiled_layers: Vec<u32>,
    /// Projections the overlay compiles, in the container's spelling.
    pub compiled_projections: Vec<String>,
    /// Compiled layers at which both arms were checked to bind the
    /// stores they claimed.
    pub attribution_checked_layers: Vec<usize>,
    /// Operands whose read bytes hash-matched their seals.
    pub seal_checked_operands: usize,
    /// A routed layer OUTSIDE the compiled set, checked to be
    /// pointer-identical in both arms.
    pub invariant_neighbour_layer: Option<usize>,
    /// Positions actually measured.
    pub positions: u64,
    /// The gate the verdict was drawn under, as evaluated.
    pub gate_evaluated: String,
}

impl VerifiedFacts {
    /// Whether this run checked everything an admissible reading needs.
    ///
    /// Explicit rather than a count: a run that verified four of five
    /// conditions has not verified 80% of anything.
    pub fn complete(&self) -> bool {
        !self.compiled_layers.is_empty()
            && !self.compiled_projections.is_empty()
            && self.attribution_checked_layers.len() == self.compiled_layers.len()
            && self.seal_checked_operands > 0
            && self.invariant_neighbour_layer.is_some()
            && self.positions > 0
            && !self.gate_evaluated.is_empty()
    }
}
