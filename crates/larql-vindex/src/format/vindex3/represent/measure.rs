//! **What a measurement run is asked to do — as data, not as
//! environment.**
//!
//! Stage 4/4b closed the read path: a stored record derives what
//! experiment should run next. Stage 5 is the authority to CAUSE that
//! experiment, and it has a precondition nothing had checked:
//!
//! > **A prepared experiment must not describe a run more precisely
//! > than the executor can be instructed to perform.**
//!
//! The teacher-forced runner violated it. Its gate was a literal —
//! `kimi_logit_v3()` — while an optimiser record declares
//! `kimi-logit-balanced-v1`; its slice was an environment variable
//! read at the point of use; its label the same. Sealing those fields
//! into a `PreparedExperimentId` would have attested to DECLARED
//! INTENT and not to caused execution, which is the accounting
//! declaration-versus-procedure gap (4b-c) one level up:
//!
//! ```text
//! declared   the experiment's gate, slice, label
//! held       a name beside a run that read a constant
//! missing    an executor that CONSUMES them
//! ```
//!
//! So the request comes first. Every control is a field, resolved by
//! name and refused when unknown; the environment form becomes an
//! ADAPTER that builds one, not a second way to run.
//!
//! ```text
//! environment variables  ─┐
//!                         ├─→ TeacherForcedRequest ─→ one procedure
//! a prepared experiment  ─┘
//! ```
//!
//! One path, so "the env invocation and the direct invocation agree"
//! is true by construction rather than by a comparison that needs a
//! 48 B model to run.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::quality::{gate_by_id, QualityGate};
use crate::error::VindexError;

/// The procedure this request names.
///
/// One variant today. Written as an enum so a second measurement
/// procedure arrives as a visible schema change rather than as a
/// caller's assumption, and so a record naming a procedure this build
/// does not implement is refused rather than run under whichever one
/// happened to be compiled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementProcedure {
    /// Two arms through the real stack over a teacher-forced corpus:
    /// baseline from the source container, candidate from an overlay.
    TeacherForcedTwoArm,
}

/// The name `MeasurementProcedure::TeacherForcedTwoArm` answers to.
pub const TEACHER_FORCED_TWO_ARM: &str = "teacher-forced-two-arm/v1";

impl MeasurementProcedure {
    pub fn name(&self) -> &'static str {
        match self {
            Self::TeacherForcedTwoArm => TEACHER_FORCED_TWO_ARM,
        }
    }

    /// Resolve a declared procedure, refusing what this build cannot
    /// perform. The same discipline as `layout_admission` and
    /// `compiled_bytes`: a record names it, this resolves it, nothing
    /// defaults.
    pub fn by_name(name: &str) -> Result<Self, VindexError> {
        match name {
            TEACHER_FORCED_TWO_ARM => Ok(Self::TeacherForcedTwoArm),
            other => Err(VindexError::Parse(format!(
                "no measurement procedure named `{other}` is implemented by this build — a \
                 run under another procedure would produce a reading nobody asked for"
            ))),
        }
    }
}

/// **Everything a teacher-forced run is instructed with.**
///
/// Paths are where the artifacts are; every other field is a CONTROL,
/// and each one has to change the run or it does not belong here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeacherForcedRequest {
    pub procedure: MeasurementProcedure,
    /// The container the baseline arm is loaded from.
    pub source: PathBuf,
    /// The compiled overlay the candidate arm substitutes.
    pub candidate: PathBuf,
    /// The exported teacher-forced corpus.
    pub quality_bank: PathBuf,
    /// How many of the corpus's sequences to consume.
    ///
    /// Part of the evidence bank's IDENTITY, not a convenience: a
    /// 32-sequence and a 256-sequence reading share a manifest hash
    /// while being different experiments — the difference between a
    /// diagnostic and an authority (1c).
    pub sequences: usize,
    /// The gate the verdict is drawn under, BY NAME.
    ///
    /// Resolved through `gate_by_id`, which refuses an unknown name.
    /// The runner used to hold a literal here, so a record could
    /// declare one gate and the run be judged by another.
    pub gate: String,
    /// Names the run's report, so a sweep's outputs do not overwrite
    /// each other. The one field that is not a control — it changes
    /// where evidence lands and nothing about what is measured.
    pub label: String,
}

/// Environment variables the adapter reads. Named here so the adapter
/// and its test cannot disagree about the spelling.
pub const SOURCE_ENV: &str = "LARQL_KIMI_VINDEX3";
pub const CANDIDATE_ENV: &str = "LARQL_KIMI_Q6_CANDIDATE";
pub const BANK_ENV: &str = "LARQL_KIMI_QUALITY_BANK";
pub const SEQUENCES_ENV: &str = "LARQL_Q2A_SEQUENCES";
pub const LABEL_ENV: &str = "LARQL_Q2A_LABEL";
pub const GATE_ENV: &str = "LARQL_Q2A_GATE";

/// Q2a's slice of the exported bank when none is named: 32 of 256.
pub const DEFAULT_SEQUENCES: usize = 32;
/// The gate the teacher-forced runner has always judged by.
///
/// A default the ADAPTER applies, so the historic command line keeps
/// meaning what it meant. It is not a fallback inside the procedure:
/// the request always carries a name, and an unnamed one never reaches
/// the run.
pub const DEFAULT_GATE: &str = "kimi-logit-v3";
pub const DEFAULT_LABEL: &str = "q2a";

impl TeacherForcedRequest {
    /// Build a request directly — the form a prepared experiment uses.
    pub fn new(
        source: impl Into<PathBuf>,
        candidate: impl Into<PathBuf>,
        quality_bank: impl Into<PathBuf>,
    ) -> Self {
        Self {
            procedure: MeasurementProcedure::TeacherForcedTwoArm,
            source: source.into(),
            candidate: candidate.into(),
            quality_bank: quality_bank.into(),
            sequences: DEFAULT_SEQUENCES,
            gate: DEFAULT_GATE.to_string(),
            label: DEFAULT_LABEL.to_string(),
        }
    }

    pub fn with_sequences(mut self, sequences: usize) -> Self {
        self.sequences = sequences;
        self
    }

    pub fn with_gate(mut self, gate: impl Into<String>) -> Self {
        self.gate = gate.into();
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// **The environment form, as an ADAPTER.**
    ///
    /// `None` when the three artifact paths are not all set, which is
    /// how the in-crate runner has always skipped without a model. It
    /// builds the same request a caller would and runs nothing itself,
    /// so there is one procedure and not two.
    pub fn from_env() -> Option<Self> {
        Self::from_vars(|var| std::env::var(var).ok())
    }

    /// The adapter's whole behaviour, over a lookup rather than the
    /// process environment.
    ///
    /// Separated so it can be tested without `set_var`, which races
    /// against every other test in the binary — the environment is
    /// process-global and the harness runs in parallel. A test that
    /// mutated it would be measuring scheduling.
    pub fn from_vars(lookup: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let (source, candidate, quality_bank) = (
            lookup(SOURCE_ENV)?,
            lookup(CANDIDATE_ENV)?,
            lookup(BANK_ENV)?,
        );
        Some(
            Self::new(source, candidate, quality_bank)
                .with_sequences(
                    lookup(SEQUENCES_ENV)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(DEFAULT_SEQUENCES),
                )
                .with_gate(lookup(GATE_ENV).unwrap_or_else(|| DEFAULT_GATE.into()))
                .with_label(lookup(LABEL_ENV).unwrap_or_else(|| DEFAULT_LABEL.into())),
        )
    }

    /// The gate this request is judged under, or a refusal naming it.
    pub fn resolve_gate(&self) -> Result<QualityGate, VindexError> {
        gate_by_id(&self.gate)
    }

    /// Refuse a request this build cannot perform, before anything is
    /// loaded.
    ///
    /// Cheap checks only, and deliberately so: what remains — that the
    /// bank was exported for this model, that the overlay compiles
    /// something, that both arms bind the bytes they claim — needs the
    /// artifacts open, and belongs to the run rather than to the
    /// request.
    pub fn admit(&self) -> Result<QualityGate, VindexError> {
        MeasurementProcedure::by_name(self.procedure.name())?;
        if self.sequences == 0 {
            return Err(VindexError::Parse(
                "a measurement over zero sequences has no positions, and a gate that judges \
                 on tail statistics would be reading an empty distribution"
                    .into(),
            ));
        }
        for (what, path) in [
            ("source", &self.source),
            ("candidate", &self.candidate),
            ("quality bank", &self.quality_bank),
        ] {
            if !Path::new(path).exists() {
                return Err(VindexError::Parse(format!(
                    "the {what} at `{}` is not there — a run cannot be prepared against an \
                     artifact that does not exist",
                    path.display()
                )));
            }
        }
        self.resolve_gate()
    }
}
