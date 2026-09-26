//! **Readings and gates, by the instrument that produced them**
//! (MEASURE-PLAN-2, `docs/measure-plan-2.md`).
//!
//! Two procedures measure a candidate, and they are different
//! instruments:
//!
//! ```text
//! teacher-forced-two-arm/v1       Kimi   QualityBank   nats over the baseline's top 2048
//! teacher-forced-two-arm/plan-v1  Plan   Aggregate     nats over the full vocabulary
//! ```
//!
//! A reading carries its kind and so does a gate. A gate judges only
//! readings of its own kind ([`GateMismatch`] otherwise), which binds a
//! gate to the instrument whose units its limits are written in.
//!
//! Both enums are serialised untagged, the plan form first. A Kimi
//! reading or gate serialises exactly as its bare struct always has, so
//! every persisted record reads back byte-identically; a plan form
//! requires its `procedure` field, so it can never be mistaken for one.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::measure::outcome::VerifiedFacts;
use super::measure::plan::metrics::{Aggregate, Summary};
use super::measure::plan::{PlanVerifiedFacts, PROCEDURE as PLAN_PROCEDURE};
use super::quality::{gate_by_id, QualityBank, QualityGate};
use super::state::instrument::{InstrumentSemantics, MetricSemantics};
use crate::error::VindexError;

/// Which instrument a reading or gate belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ReadingKind {
    /// `teacher-forced-two-arm/v1`.
    Kimi,
    /// `teacher-forced-two-arm/plan-v1`.
    Plan,
}

impl ReadingKind {
    /// The kind of reading a named procedure produces, if this build
    /// knows the procedure.
    pub fn of_procedure(procedure: &str) -> Option<Self> {
        match procedure {
            super::measure::TEACHER_FORCED_TWO_ARM => Some(Self::Kimi),
            PLAN_PROCEDURE => Some(Self::Plan),
            _ => None,
        }
    }
}

impl std::fmt::Display for ReadingKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Kimi => "teacher-forced-two-arm/v1",
            Self::Plan => PLAN_PROCEDURE,
        })
    }
}

/// The plan procedure's name, as a field that deserialises from exactly
/// that string and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlanProcedure;

impl Serialize for PlanProcedure {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(PLAN_PROCEDURE)
    }
}

impl<'de> Deserialize<'de> for PlanProcedure {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let named = String::deserialize(d)?;
        if named == PLAN_PROCEDURE {
            Ok(Self)
        } else {
            Err(serde::de::Error::custom(format!(
                "procedure `{named}` is not `{PLAN_PROCEDURE}`"
            )))
        }
    }
}

/// **A plan-v1 reading**: the procedure's summary and nothing it did not
/// measure. There is no routing, no top-10 statistic and no covered mass
/// here, because plan-v1 measures none of them; absent is absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanObservation {
    pub procedure: PlanProcedure,
    /// Bank samples measured.
    pub sequences: usize,
    pub positions: u64,
    pub all: Aggregate,
    pub by_category: Vec<(String, Aggregate)>,
    pub by_margin_band: Vec<((f64, f64), Aggregate)>,
}

impl PlanObservation {
    /// The reading a plan-v1 run's summary is, over `sequences` samples.
    pub fn from_summary(summary: &Summary, sequences: usize) -> Self {
        Self {
            procedure: PlanProcedure,
            sequences,
            positions: summary.all.positions as u64,
            all: summary.all.clone(),
            by_category: summary.by_category.clone(),
            by_margin_band: summary.by_margin_band.clone(),
        }
    }
}

/// One stored reading, of whichever instrument took it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Observation {
    Plan(PlanObservation),
    /// Boxed: a Kimi bank is several times a plan reading's size, and a
    /// `Box` serialises transparently, so the persisted form is unchanged.
    Kimi(Box<QualityBank>),
}

impl Observation {
    pub fn kind(&self) -> ReadingKind {
        match self {
            Self::Kimi(_) => ReadingKind::Kimi,
            Self::Plan(_) => ReadingKind::Plan,
        }
    }

    /// Positions the reading covers, whatever its kind.
    pub fn positions(&self) -> u64 {
        match self {
            Self::Kimi(bank) => bank.positions,
            Self::Plan(plan) => plan.positions,
        }
    }

    /// The Kimi bank, for analyses defined only over it.
    pub fn as_kimi(&self) -> Option<&QualityBank> {
        match self {
            Self::Kimi(bank) => Some(bank.as_ref()),
            Self::Plan(_) => None,
        }
    }

    /// The Kimi bank, mutably — for a caller building or deliberately
    /// corrupting one.
    pub fn as_kimi_mut(&mut self) -> Option<&mut QualityBank> {
        match self {
            Self::Kimi(bank) => Some(bank.as_mut()),
            Self::Plan(_) => None,
        }
    }

    pub fn as_plan(&self) -> Option<&PlanObservation> {
        match self {
            Self::Plan(plan) => Some(plan),
            Self::Kimi(_) => None,
        }
    }
}

impl From<QualityBank> for Observation {
    fn from(bank: QualityBank) -> Self {
        Self::Kimi(Box::new(bank))
    }
}

impl From<PlanObservation> for Observation {
    fn from(plan: PlanObservation) -> Self {
        Self::Plan(plan)
    }
}

/// **What a run checked**, of whichever procedure ran it. Untagged, plan
/// form first: Kimi facts serialise exactly as the bare struct always
/// has, and plan facts require fields the Kimi form does not carry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RunFacts {
    Plan(PlanVerifiedFacts),
    Kimi(VerifiedFacts),
}

impl RunFacts {
    pub fn kind(&self) -> ReadingKind {
        match self {
            Self::Kimi(_) => ReadingKind::Kimi,
            Self::Plan(_) => ReadingKind::Plan,
        }
    }

    pub fn as_kimi(&self) -> Option<&VerifiedFacts> {
        match self {
            Self::Kimi(facts) => Some(facts),
            Self::Plan(_) => None,
        }
    }

    pub fn as_kimi_mut(&mut self) -> Option<&mut VerifiedFacts> {
        match self {
            Self::Kimi(facts) => Some(facts),
            Self::Plan(_) => None,
        }
    }

    pub fn as_plan(&self) -> Option<&PlanVerifiedFacts> {
        match self {
            Self::Plan(facts) => Some(facts),
            Self::Kimi(_) => None,
        }
    }
}

impl From<VerifiedFacts> for RunFacts {
    fn from(facts: VerifiedFacts) -> Self {
        Self::Kimi(facts)
    }
}

impl From<PlanVerifiedFacts> for RunFacts {
    fn from(facts: PlanVerifiedFacts) -> Self {
        Self::Plan(facts)
    }
}

/// **A gate over plan-v1 readings.** Limits are in the plan procedure's
/// units: KL in nats over the full vocabulary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanGate {
    pub procedure: PlanProcedure,
    pub id: String,
    /// The metric its limits are written in. A reading from an
    /// instrument declaring anything else is refused before evaluation.
    pub semantics: MetricSemantics,
    pub positions_min: u64,
    pub kl_p99_max: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kl_mean_max: Option<f64>,
    /// Ceiling on `1 − top1_agreement`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_disagreement_max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_nll_mean_max: Option<f64>,
}

/// A contract, of whichever instrument its limits are written for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Gate {
    Plan(PlanGate),
    Kimi(QualityGate),
}

impl Gate {
    pub fn id(&self) -> &str {
        match self {
            Self::Kimi(gate) => &gate.id,
            Self::Plan(gate) => &gate.id,
        }
    }

    pub fn kind(&self) -> ReadingKind {
        match self {
            Self::Kimi(_) => ReadingKind::Kimi,
            Self::Plan(_) => ReadingKind::Plan,
        }
    }

    /// The Kimi gate, for machinery defined only over it.
    pub fn as_kimi(&self) -> Option<&QualityGate> {
        match self {
            Self::Kimi(gate) => Some(gate),
            Self::Plan(_) => None,
        }
    }
}

impl From<QualityGate> for Gate {
    fn from(gate: QualityGate) -> Self {
        Self::Kimi(gate)
    }
}

impl From<PlanGate> for Gate {
    fn from(gate: PlanGate) -> Self {
        Self::Plan(gate)
    }
}

/// A gate asked to judge a reading of another instrument.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("gate `{gate}` judges {gate_kind} readings; this reading is {reading_kind}")]
pub struct GateMismatch {
    pub gate: String,
    pub gate_kind: ReadingKind,
    pub reading_kind: ReadingKind,
}

impl GateMismatch {
    /// `Ok` when `gate` may judge a reading of `reading`'s kind.
    pub fn check(gate: &Gate, reading: ReadingKind) -> Result<(), Self> {
        if gate.kind() == reading {
            Ok(())
        } else {
            Err(Self {
                gate: gate.id().to_string(),
                gate_kind: gate.kind(),
                reading_kind: reading,
            })
        }
    }
}

/// Why a gate may not judge readings from an instrument.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GateBindingRefusal {
    #[error(transparent)]
    Kind(#[from] GateMismatch),
    /// The instrument's declared metric is not the one the gate's
    /// limits are written in; `None` means it declared no structure.
    #[error(
        "gate `{gate}` is written for {gate_semantics:?}; the instrument declares {instrument:?}"
    )]
    Semantics {
        gate: String,
        gate_semantics: MetricSemantics,
        instrument: Option<MetricSemantics>,
    },
    /// The instrument's structured semantics contradict its truncation.
    #[error("instrument is inconsistent: {0}")]
    Instrument(String),
}

/// **Whether `gate` may judge readings of `kind` taken by `instrument`.**
/// Checked before any evaluation. Kimi gates predate structured
/// semantics and bind by kind alone.
pub fn check_binding(
    gate: &Gate,
    kind: ReadingKind,
    instrument: &InstrumentSemantics,
) -> Result<(), GateBindingRefusal> {
    instrument.check().map_err(GateBindingRefusal::Instrument)?;
    GateMismatch::check(gate, kind)?;
    if let Gate::Plan(plan) = gate {
        if instrument.semantics != Some(plan.semantics) {
            return Err(GateBindingRefusal::Semantics {
                gate: plan.id.clone(),
                gate_semantics: plan.semantics,
                instrument: instrument.semantics,
            });
        }
    }
    Ok(())
}

/// A plan gate that exists only under test, so the loop can be proved
/// end to end without a production plan gate. MEASURE-PLAN-2 registers
/// no plan gate id outside tests; slice 3 earns them.
#[cfg(test)]
pub const TEST_PLAN_GATE: &str = "plan-test-only-v1";

#[cfg(test)]
pub fn test_plan_gate() -> PlanGate {
    PlanGate {
        procedure: PlanProcedure,
        id: TEST_PLAN_GATE.into(),
        semantics: MetricSemantics::PLAN_V1,
        positions_min: 1,
        kl_p99_max: 1e-3,
        kl_mean_max: None,
        top1_disagreement_max: None,
        delta_nll_mean_max: None,
    }
}

/// **Resolve a gate id, of either kind.** The Kimi ids are
/// [`gate_by_id`]'s; this build registers no plan gate.
pub fn gate_of_any_kind(id: &str) -> Result<Gate, VindexError> {
    #[cfg(test)]
    if id == TEST_PLAN_GATE {
        return Ok(Gate::Plan(test_plan_gate()));
    }
    gate_by_id(id).map(Gate::Kimi)
}

#[cfg(test)]
#[path = "reading_tests.rs"]
mod tests;
