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

//! # The conservation inventory
//!
//! The compiler cannot tell whether a check got WEAKER, so the move of
//! the harness body into a callable procedure is specified here first.
//! Every assertion site in `q2a_teacher_forced`'s test function is
//! listed with where its authority goes, and the review question
//! becomes *where did each one go* rather than *does the new function
//! look equivalent*.
//!
//! ```text
//! harness site   what it asserts                  new authority
//! ────────────────────────────────────────────────────────────────────
//! 372            Metal present                    ExecutionFailure::BackendUnavailable
//! 378-9,384,391  artifacts open and parse         ExecutionFailure::ArtifactUnreadable
//! 416            stores register                  ExecutionFailure::ArtifactUnreadable
//! 294 (helper)   layer loads, zero missing        ExecutionFailure::LayerIncomplete
//! 308 (helper)   head attaches                    ExecutionFailure::HeadDidNotAttach
//! 267 (helper)   sequence file reads, right size  ExecutionFailure::ArtifactUnreadable
//! 338 (helper)   the step does not refuse         ExecutionFailure::StepRefused
//! 433,436        probe layers load                ExecutionFailure::LayerIncomplete
//! 530,533        neighbour probes load            ExecutionFailure::LayerIncomplete
//!
//! 368            LARQL_RESIDENCY_SET unset        Inadmissible::ResidencyModeWouldBeMeasured
//! 386            bank hidden == model hidden      Inadmissible::CorpusNotForThisModel
//! 459            the overlay compiles something   Inadmissible::CandidateCompilesNothing
//! 444,449        baseline is source-backed BF16   Inadmissible::UnexpectedPhysicalRead
//! 476,477        candidate matches its scope      Inadmissible::UnexpectedPhysicalRead
//! 520,521        shared expert from the stack     Inadmissible::UnexpectedPhysicalRead
//! 534            neighbour is source-backed       Inadmissible::UnexpectedPhysicalRead
//! 487            out-of-scope pointer identity    Inadmissible::ProtectedOperandChanged
//! 535            neighbour pointer identity       Inadmissible::ProtectedOperandChanged
//! 506,512        identity vs table addressing     Inadmissible::AddressingMismatch
//! 542            reads match seals                Inadmissible::SealMismatch
//! 741            the routed operand was sealed    STAYS IN THE TEST — it is how the
//!                                                 operand-removal control FINDS a seal to
//!                                                 remove, not a condition of the run. The
//!                                                 audit caught this: the variant the first
//!                                                 inventory assigned here went unused.
//! 571,613        position counts                  Inadmissible::PositionCountMismatch
//! 575,579,583-6  the null arm is exactly zero     Inadmissible::NullArmNotZero
//! (5a-0a)        the gate evaluated is the one
//!                requested                        Inadmissible::GateMismatch
//!
//! 391,459,471    which layers/projections         VerifiedFacts::compiled_layers,
//!                the overlay compiles             VerifiedFacts::compiled_projections
//! 430-521        per-layer attribution done       VerifiedFacts::attribution_checked_layers
//! 542            operands whose reads matched     VerifiedFacts::seal_checked_operands
//! 529            the invariant neighbour          VerifiedFacts::invariant_neighbour_layer
//! 613            positions measured               VerifiedFacts::positions
//! 630            the gate evaluated               VerifiedFacts::gate_evaluated
//!
//! 688-706        a sub-4096 bank cannot pass      STAYS IN THE TEST. A claim about the
//!                                                 GATE, already enforced by
//!                                                 `QualityGate::evaluate`, not a validity
//!                                                 condition of the measurement.
//! 723-750        removing a sealed operand makes  STAYS IN THE TEST. A claim about
//!                `verify_complete` refuse         `verify_complete`, not about this run.
//! ```
//!
//! Two sites are deliberately NOT production authority, and saying so
//! is part of the inventory: a check that moves for tidiness is as lost
//! as one that vanishes.
//!
//! **The inventory has already earned itself.** `NullArmNotZero` is
//! absent from the first pass of this vocabulary, which was written
//! after reading the harness once. Walking it assertion by assertion
//! found it — and it is the most dangerous condition in the set,
//! because a non-deterministic device yields a plausible KL that is
//! entirely artifact while every other check still passes.

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// The bank does not hold the positions the request asked for.
    ///
    /// `assert_eq!(bank.positions, sequences * positions_per_seq)`. A
    /// gate judging on tail statistics over a different position count
    /// is judging a different experiment (1c).
    PositionCountMismatch { expected: u64, measured: u64 },
    /// The baseline arm compared against ITSELF was not exactly zero.
    ///
    /// The determinism control, and the one the first pass of this
    /// vocabulary missed — found by walking the harness assertion by
    /// assertion rather than by reasoning about what it "should"
    /// check. `assert_eq!(null_bank.logits.kl_p99, 0.0, "null arm KL
    /// must be exactly zero")`, plus bit-equal logits and zero flips
    /// and route changes.
    ///
    /// It is the most dangerous condition in the list: a
    /// non-deterministic device produces a plausible KL that is
    /// entirely artifact, and every other check here would still pass.
    /// Metal decode non-determinism has been a real defect on this
    /// stack before.
    NullArmNotZero { statistic: String, observed: f64 },
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
