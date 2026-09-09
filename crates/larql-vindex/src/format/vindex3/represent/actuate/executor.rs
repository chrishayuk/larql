//! **Who performs a request, and where the things it names actually
//! are.**
//!
//! Two seams, kept apart because they fail for different reasons and are
//! fixed by different people:
//!
//! ```text
//! ExperimentExecutor   HOW this procedure is performed
//!                      selected by the PROCEDURE the record declares
//!
//! ArtifactLocator      WHERE the identified things are on this machine
//!                      identity in, a path out, and the path is checked
//! ```
//!
//! # Selected by procedure, never by device
//!
//! An executor answers to a name — `teacher-forced-two-arm/v1` — and the
//! registry refuses a name it has no executor for, listing what it does
//! have. That is the same discipline `layout_admission`,
//! `compiled_bytes` and `gate_by_id` already use: a record names it,
//! this resolves it, nothing defaults.
//!
//! Nothing here branches on `Cpu`, `Metal`, a kernel, a lowering target
//! or a store id, and that is a sequencing decision rather than a
//! stylistic one. The representation plane is already pluggable and the
//! lowering plane is not yet equally open; an actuation bridge that
//! learned those names would finish the autonomous optimiser by
//! cementing precisely the lowering authority the codec work is
//! dismantling. A new provider should enter by declaring a procedure it
//! performs.
//!
//! # Location is checked, not trusted
//!
//! `EvidenceBank::locator_hint` is documented as *where a bank was last
//! seen, never part of what it is*, and `SourceDependency` says the same
//! for containers. So a locator answers with a path and
//! [`verify_container`] reads that container's own identity back and
//! compares it with the one the request names. A run against a container
//! that merely sits at the expected path is a run against whatever is at
//! that path.
//!
//! # The executor may not re-aim the experiment either
//!
//! [`Observed`] carries the key it is an observation OF, and
//! [`ExecutorRegistry::execute`] refuses when that is not the key it
//! handed over. The bridge's rule — a request restates an authorisation
//! and cannot author one — would be worth little if the layer below
//! could return an observation of something else and have it accepted.
//!
//! # What an observation is NOT, yet
//!
//! [`Observed`] is what a run saw and what it checked. It is **not**
//! admissible evidence and it carries no verdict: whether it constitutes
//! the measurement its key names, and how it is sealed into
//! `SearchFacts`, is stage 6. Nothing in this module can write to the
//! scientific record, and `Observed` is deliberately thin so that stage 6
//! is free to decide what an artifact must carry rather than inheriting
//! whatever was convenient here.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::super::compiler::read_source_identity;
use super::super::measure::outcome::{MeasurementRefusal, VerifiedFacts};
use super::super::quality::QualityBank;
use super::super::state::identity::RepresentationStateId;
use super::super::state::key::MeasurementKey;
use super::request::MeasurementRequest;
use crate::error::VindexError;

/// Why an identified artifact could not be located.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocatorRefusal {
    /// Nothing this locator knows about holds it.
    NotHeld { what: String, identity: String },
    /// Something is at the path and it is not what was asked for. The
    /// dangerous case, and the reason a path is checked rather than
    /// trusted.
    NotWhatItClaims {
        what: String,
        path: String,
        detail: String,
    },
    /// **The overlay for this state has never been compiled.**
    ///
    /// Actionable rather than merely absent: the request carries the map
    /// that must be presented, so a caller told this knows exactly what
    /// to build. A locator must NOT build it silently — a build is where
    /// a representation could come to differ from the one the key names,
    /// and stage 5b does not own that.
    NotBuilt {
        state: RepresentationStateId,
        map: String,
    },
}

impl std::fmt::Display for LocatorRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotHeld { what, identity } => {
                write!(f, "no {what} with identity {identity} is held here")
            }
            Self::NotWhatItClaims { what, path, detail } => write!(
                f,
                "the {what} at `{path}` is not the one this experiment names: {detail}"
            ),
            Self::NotBuilt { state, map } => write!(
                f,
                "no overlay presenting state {} has been compiled; the request carries the map \
                 it must present (`{map}`)",
                state.short()
            ),
        }
    }
}

/// **Where the things an experiment names currently are.**
///
/// Every method takes the whole request, so an implementation sees the
/// experiment it is answering about rather than a bare identity — which
/// is what lets a future provider that BUILDS an overlay slot in without
/// the signature moving.
pub trait ArtifactLocator {
    /// The container the baseline arm is read from.
    fn container(&self, request: &MeasurementRequest) -> Result<PathBuf, LocatorRefusal>;

    /// The exported corpus the run is taken over.
    fn corpus(&self, request: &MeasurementRequest) -> Result<PathBuf, LocatorRefusal>;

    /// The compiled overlay presenting the candidate state.
    fn candidate(&self, request: &MeasurementRequest) -> Result<PathBuf, LocatorRefusal>;
}

/// **Read a located container's own identity back and compare it.**
///
/// The SEMANTIC identity, not the artifact digest: a re-export that
/// changes no value moves the second and not the first, and refusing a
/// container for having been re-serialised would be the false split
/// `state/identity.rs` removed in v2.
pub fn verify_container(request: &MeasurementRequest, path: &Path) -> Result<(), LocatorRefusal> {
    let found = read_source_identity(path).map_err(|e| LocatorRefusal::NotWhatItClaims {
        what: "container".into(),
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    let expected = request.model().semantic_digest();
    let actual = found.semantic_digest();
    if expected == actual {
        return Ok(());
    }
    Err(LocatorRefusal::NotWhatItClaims {
        what: "container".into(),
        path: path.display().to_string(),
        detail: format!(
            "it identifies as {} and this experiment was priced against {}",
            &actual[..actual.len().min(12)],
            &expected[..expected.len().min(12)]
        ),
    })
}

/// **What a run saw, and what it checked.**
///
/// No verdict. The gate the record declares prices this observation
/// downstream through `ConstraintVector`, exactly as it does for a
/// reading that arrived any other way — 1c's rule that an observation
/// and its meaning stay apart applies to observations this bridge
/// caused as much as to ones an operator filed by hand.
#[derive(Debug, Clone, PartialEq)]
pub struct Observed {
    /// The experiment this is an observation OF, as the executor
    /// restates it. Checked against what was requested.
    pub key: MeasurementKey,
    pub observation: QualityBank,
    /// Every validity condition the run checked.
    pub verified: VerifiedFacts,
    /// What the executor actually did, in its own vocabulary — a
    /// provenance line for a reader, never read as authority.
    pub execution_note: String,
}

/// Why an authorised request produced no observation.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionRefusal {
    /// No executor in this build performs the procedure the record
    /// declares. Names what is implemented, because "unknown procedure"
    /// without the alternatives is a dead end.
    NoSuchProcedure {
        named: String,
        implemented: Vec<String>,
    },
    /// An artifact the run needs could not be located.
    Locator(LocatorRefusal),
    /// The request is well formed and this procedure cannot be
    /// instructed with it — a bank declaring no samples, say. Distinct
    /// from a locator refusal: nothing is missing from the machine, the
    /// experiment as stated cannot be turned into an instruction.
    NotInstructable { procedure: String, detail: String },
    /// The run happened, or could not be performed. Already a typed
    /// distinction — `MeasurementRefusal` separates *nothing was
    /// measured* from *the run proved nothing* — and it is forwarded
    /// whole rather than flattened.
    Measurement(MeasurementRefusal),
    /// **The executor returned an observation of a different
    /// experiment.**
    ///
    /// The seam's own falsifier. A bridge that guarantees a request
    /// cannot misstate its experiment buys nothing if the layer below
    /// may answer about another one.
    ///
    /// Boxed: two keys are four digests apiece, and every refusal here
    /// travels in the error arm of a `Result` whose success arm is a
    /// measurement. The rare case must not widen the common one.
    ObservedAnotherExperiment(Box<Misdirected>),
}

/// The experiment that was asked for, and the one that came back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Misdirected {
    pub requested: MeasurementKey,
    pub observed: MeasurementKey,
}

impl std::fmt::Display for ExecutionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchProcedure { named, implemented } => write!(
                f,
                "no executor in this build performs `{named}`; it performs {}",
                match implemented.is_empty() {
                    true => "nothing".to_string(),
                    false => implemented.join(", "),
                }
            ),
            Self::Locator(r) => write!(f, "{r}"),
            Self::NotInstructable { procedure, detail } => write!(
                f,
                "this experiment cannot be turned into an instruction for `{procedure}`: {detail}"
            ),
            Self::Measurement(r) => write!(f, "{r}"),
            Self::ObservedAnotherExperiment(m) => write!(
                f,
                "the executor was asked for experiment {} and returned an observation of {} — \
                 recording it would credit one representation with another's evidence",
                m.requested.short(),
                m.observed.short()
            ),
        }
    }
}

impl From<LocatorRefusal> for ExecutionRefusal {
    fn from(r: LocatorRefusal) -> Self {
        Self::Locator(r)
    }
}

/// **Something that performs one named procedure.**
///
/// It executes the question it was handed. It does not pick another
/// state, change the bank, change the scale, invent a codec, decide
/// promotion, or decide that a failed run means the candidate failed.
pub trait ExperimentExecutor {
    /// The procedure this executor performs, by name.
    fn procedure(&self) -> &str;

    /// Perform the request, or refuse.
    fn execute(
        &self,
        request: &MeasurementRequest,
        artifacts: &dyn ArtifactLocator,
    ) -> Result<Observed, ExecutionRefusal>;
}

/// **The executors this build has**, resolved by name.
pub struct ExecutorRegistry<'a> {
    executors: Vec<&'a dyn ExperimentExecutor>,
}

impl<'a> ExecutorRegistry<'a> {
    /// Refuses two executors claiming one procedure: which of them ran
    /// would then depend on registration order, and a measurement whose
    /// procedure depends on link order is not a measurement of anything.
    pub fn new(
        executors: impl IntoIterator<Item = &'a dyn ExperimentExecutor>,
    ) -> Result<Self, VindexError> {
        let executors: Vec<&'a dyn ExperimentExecutor> = executors.into_iter().collect();
        let mut seen = BTreeSet::new();
        for executor in &executors {
            if !seen.insert(executor.procedure()) {
                return Err(VindexError::Parse(format!(
                    "two executors both claim to perform `{}` — which one ran would depend on \
                     registration order",
                    executor.procedure()
                )));
            }
        }
        Ok(Self { executors })
    }

    /// Every procedure this build performs, in name order.
    pub fn implemented(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .executors
            .iter()
            .map(|e| e.procedure().to_string())
            .collect();
        names.sort();
        names
    }

    /// The executor for a procedure, or a refusal naming what exists.
    pub fn for_procedure(
        &self,
        named: &str,
    ) -> Result<&'a dyn ExperimentExecutor, ExecutionRefusal> {
        self.executors
            .iter()
            .copied()
            .find(|e| e.procedure() == named)
            .ok_or_else(|| ExecutionRefusal::NoSuchProcedure {
                named: named.to_string(),
                implemented: self.implemented(),
            })
    }

    /// **Perform an authorised request through whichever executor the
    /// record's procedure names.**
    ///
    /// The observation is checked to be an observation of the experiment
    /// that was requested before it is returned.
    pub fn execute(
        &self,
        request: &MeasurementRequest,
        artifacts: &dyn ArtifactLocator,
    ) -> Result<Observed, ExecutionRefusal> {
        let executor = self.for_procedure(request.procedure())?;
        let observed = executor.execute(request, artifacts)?;
        if &observed.key != request.key() {
            return Err(ExecutionRefusal::ObservedAnotherExperiment(Box::new(
                Misdirected {
                    requested: request.key().clone(),
                    observed: observed.key,
                },
            )));
        }
        Ok(observed)
    }
}
