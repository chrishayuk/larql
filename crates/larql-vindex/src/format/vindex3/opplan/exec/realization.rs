//! **The realization vocabulary** — what a prepared plan resolves, considers,
//! selects and pins for each planned operand, and how it refuses.
//!
//! Rung 3b of the representation/execution contract
//! (`docs/represent/forecasts/rung3-planned-realizations.json`). Before this
//! module the seam between plan and backend was three stored-dtype booleans
//! and a silent fallback: any dtype the policy did not recognise widened to
//! f32 with nothing recorded. Now a backend is handed the planned operation
//! and the REPRESENTATION FACTS the registry declares for the stored dtype,
//! and answers with one [`RealizationId`] chosen from a candidate set it
//! derived from those declarations — or refuses, naming every candidate it
//! considered and why. Nothing here reads a label to decide anything.
//!
//! The forms a realization can take are the executor's own, made explicit:
//! a direct kernel over the stored bytes (declared by the codec), the
//! universal decode to f32 followed by an f32 projection, a decode followed
//! by a lossy re-quantisation (the executor's own compact forms), the packed
//! bank sliced per expert from stored rows, a decoded table gathered per
//! token, or a device backend's own resident form.

use std::fmt;

use super::backend::{MatrixClass, WeightFormat};
use super::cpu::physical::PhysicalProjectionPlan;
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::{
    Acceleration, AccelerationBackend, CodecCapabilities, CodecError, CodecRegistry,
    RepresentationCodec, RequiredAccess, ResidencyProfile,
};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// What the registry declares for one stored dtype — resolved once per
/// operand at preparation, and the only thing a backend is told about the
/// representation.
#[derive(Debug, Clone, PartialEq)]
pub struct RepresentationFacts {
    /// The label the container stores the operand under.
    pub label: String,
    /// The codec's declarations, when the label names a registered codec.
    /// `None` is a fact too: an unregistered label has no decode and no
    /// capabilities, and only a loader with its own dialect can bind it.
    pub registered: Option<RegisteredFacts>,
    /// Whether an overlay edit stands on the operand. An edit is an
    /// f32-space fact with no stored bytes, so no direct realization can
    /// honour it; only decode can.
    pub overlaid: bool,
}

/// A registered codec's declarations, copied out so a backend never holds
/// the codec itself.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredFacts {
    pub identity: CodecIdentity,
    pub capabilities: CodecCapabilities,
    pub accelerations: Vec<Acceleration>,
    pub decode_residency: ResidencyProfile,
}

impl RepresentationFacts {
    /// Resolve `label` through the built-in registry.
    pub fn resolve(label: &str) -> Self {
        Self::resolve_in(CodecRegistry::builtin(), label)
    }

    /// Resolve `label` through `registry` — a scratch registry in a test is
    /// how a codec that is not shipped gets facts.
    pub fn resolve_in(registry: &CodecRegistry, label: &str) -> Self {
        Self {
            label: label.to_string(),
            registered: registry.by_label(label).map(RegisteredFacts::of),
            overlaid: false,
        }
    }

    /// The same facts, with an overlay edit standing on the operand.
    pub fn overlaid(mut self) -> Self {
        self.overlaid = true;
        self
    }

    /// Whether the STORED bytes can be addressed as `required` asks.
    pub fn provides(&self, required: RequiredAccess) -> bool {
        self.registered
            .as_ref()
            .is_some_and(|r| r.capabilities.access.provides(required))
    }

    /// The direct CPU realizations the codec declares — none while an
    /// overlay edit stands on the operand, because there are no stored
    /// bytes for a kernel to run over.
    pub fn direct_cpu_plans(&self) -> Vec<PhysicalProjectionPlan> {
        if self.overlaid {
            return Vec::new();
        }
        self.registered
            .as_ref()
            .map(|r| {
                r.accelerations
                    .iter()
                    .filter(|a| a.backend == AccelerationBackend::Cpu)
                    .map(|a| a.plan)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The residency the codec declares for a direct realization over
    /// `plan`, if it declares one.
    pub fn direct_residency(&self, plan: PhysicalProjectionPlan) -> Option<ResidencyProfile> {
        self.registered.as_ref().and_then(|r| {
            r.accelerations
                .iter()
                .find(|a| a.plan == plan)
                .map(|a| a.residency)
        })
    }

    /// Admit slicing the stored bytes per expert — the packed-bank
    /// realization's requirement, judged BEFORE any byte is read.
    ///
    /// A registered codec must provide row access; an unregistered label
    /// is a dialect the bank loader judges itself (the MXFP4 bank's `U8`
    /// streams), and is admitted here so that judgement stays where it
    /// is until 3d moves it into declarations.
    pub fn admit_row_slicing(&self) -> Result<(), CodecError> {
        match &self.registered {
            Some(r) => r
                .capabilities
                .require(RequiredAccess::RowRandom, &self.label),
            None => Ok(()),
        }
    }
}

impl RegisteredFacts {
    pub fn of(codec: &dyn RepresentationCodec) -> Self {
        Self {
            identity: codec.identity(),
            capabilities: codec.capabilities(),
            accelerations: codec.accelerations(),
            decode_residency: codec.decode_residency(),
        }
    }
}

/// Where a realization runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealizationBackend {
    /// This crate's CPU executor, including its reference transcription.
    Cpu,
    /// A device backend's own resident form.
    Device,
}

/// How a planned operand is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealizationForm {
    /// A kernel the codec declared, over the stored bytes.
    Direct(PhysicalProjectionPlan),
    /// The universal decode to f32, then an f32 projection.
    Decode(PhysicalProjectionPlan),
    /// Decode, then the executor's own lossy re-quantisation.
    Requantise(PhysicalProjectionPlan),
    /// The packed bank, sliced per expert from STORED rows and converted.
    SliceStored { convert: WeightFormat },
    /// The whole table decoded, one row gathered per token.
    DecodedGather,
    /// A device backend's resident form, declared per class by that
    /// backend for its own target.
    DeviceResident(WeightFormat),
}

/// One realization, named so a plan can pin it and a trace can say it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealizationId {
    pub backend: RealizationBackend,
    pub form: RealizationForm,
}

impl RealizationId {
    pub const fn cpu(form: RealizationForm) -> Self {
        Self {
            backend: RealizationBackend::Cpu,
            form,
        }
    }

    /// The representation the loader makes resident for this realization.
    pub fn format(self) -> WeightFormat {
        match self.form {
            RealizationForm::Direct(plan)
            | RealizationForm::Decode(plan)
            | RealizationForm::Requantise(plan) => plan.format(),
            RealizationForm::SliceStored { convert } => convert,
            RealizationForm::DecodedGather => WeightFormat::F32,
            RealizationForm::DeviceResident(format) => format,
        }
    }

    /// The CPU projection plan this realization runs, when it is one.
    pub fn cpu_plan(self) -> Option<PhysicalProjectionPlan> {
        match self.form {
            RealizationForm::Direct(plan)
            | RealizationForm::Decode(plan)
            | RealizationForm::Requantise(plan) => Some(plan),
            _ => None,
        }
    }

    pub fn name(self) -> String {
        let form = match self.form {
            RealizationForm::Direct(plan) => format!("direct/{plan:?}"),
            RealizationForm::Decode(plan) => format!("decode-f32+{plan:?}"),
            RealizationForm::Requantise(plan) => format!("requantise/{plan:?}"),
            RealizationForm::SliceStored { convert } => format!("slice-stored→{convert:?}"),
            RealizationForm::DecodedGather => "decode-f32+gather".to_string(),
            RealizationForm::DeviceResident(format) => format!("device-resident/{format:?}"),
        };
        match self.backend {
            RealizationBackend::Cpu => format!("cpu:{form}"),
            RealizationBackend::Device => format!("device:{form}"),
        }
    }
}

/// What the executor makes resident for `format`, priced from its own
/// block geometry — see [`super::accounting::resident_profile_with`],
/// which is the one definition; this is it under the executor's constants.
pub fn resident_profile(format: WeightFormat) -> ResidencyProfile {
    super::accounting::resident_profile_with(format, super::accounting::BlockGeometry::executor())
}

/// Why a backend chose what it chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionReason {
    /// The codec declares a direct realization and the backend takes it.
    DirectDeclared,
    /// The codec declares no direct realization; decode is the only path.
    NoDirectRealization,
    /// A direct realization exists and the process arm prefers decoding.
    ArmPrefersDecode,
    /// The size policy over a float source chose this resident form.
    SizePolicy,
    /// A packed bank is sliced per expert from stored rows at load.
    BankSlicedAtLoad,
    /// The device backend's class table names its resident form.
    DeviceClassTable,
    /// An embedding table is decoded whole and gathered per token.
    EmbeddingGather,
    /// The reference backend takes the literal transcription, always.
    ReferenceOracle,
    /// An overlay edit stands on the operand; only decode can honour it.
    OverlaidEdit,
}

impl SelectionReason {
    pub const fn name(self) -> &'static str {
        match self {
            Self::DirectDeclared => "direct realization declared by the codec",
            Self::NoDirectRealization => "no direct realization registered",
            Self::ArmPrefersDecode => "the process arm prefers decoding",
            Self::SizePolicy => "size policy over a float source",
            Self::BankSlicedAtLoad => "packed bank sliced per expert at load",
            Self::DeviceClassTable => "device class table",
            Self::EmbeddingGather => "table decoded whole, gathered per token",
            Self::ReferenceOracle => "reference oracle",
            Self::OverlaidEdit => "an overlay edit stands on the operand; only decode honours it",
        }
    }
}

/// The realization a backend pinned for one operand, with everything a
/// trace needs to say about the choice.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub realization: RealizationId,
    /// What the selected realization makes resident, declared — never
    /// measured — so the census can be checked against it.
    pub residency: ResidencyProfile,
    pub reason: SelectionReason,
    /// Every realization the backend considered, the selected one
    /// included. Derived from declarations, never from a label.
    pub candidates: Vec<RealizationId>,
}

/// Why no realization could be selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalKind {
    /// The stored label names no registered codec, so nothing can decode
    /// it and no capability is declared for it.
    UnregisteredRepresentation,
    /// Every candidate needs access the stored representation does not
    /// provide.
    AccessRefused,
    /// The plan executes an operation no realization on this backend can
    /// bind — a planned operand with nowhere to go.
    MissingRealization,
}

impl RefusalKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::UnregisteredRepresentation => "unregistered representation",
            Self::AccessRefused => "access refused",
            Self::MissingRealization => "missing realization",
        }
    }
}

/// A refusal that names the operand, what it asked for, what the
/// representation is, and every candidate with the reason it was rejected.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionRefusal {
    pub operand: OperandRef,
    pub operation: Operation,
    pub representation: String,
    pub requested: RequiredAccess,
    pub kind: RefusalKind,
    pub considered: Vec<(RealizationId, String)>,
}

impl fmt::Display for SelectionRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "operand `{}` ({}, requires {} access) stored as `{}`: {}",
            self.operand.tensor,
            self.operation.name(),
            self.requested.name(),
            self.representation,
            self.kind.name()
        )?;
        if self.considered.is_empty() {
            write!(f, "; no realization to consider")?;
        }
        for (candidate, why) in &self.considered {
            write!(f, "; {} — {why}", candidate.name())?;
        }
        Ok(())
    }
}

/// Every refusal a plan raised, together, so a caller sees the whole
/// problem and not its first symptom.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionRefusals(pub Vec<SelectionRefusal>);

impl fmt::Display for SelectionRefusals {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} planned operand(s) have no admissible realization",
            self.0.len()
        )?;
        for refusal in &self.0 {
            write!(f, "\n  {refusal}")?;
        }
        Ok(())
    }
}

/// One planned operand's pinned realization — the record the prepared
/// plan keeps and the trace reads.
#[derive(Debug, Clone, PartialEq)]
pub struct RealizationRecord {
    pub planned: PlannedOperand,
    pub representation: String,
    /// The identity the stored label resolved to at preparation; `None`
    /// for a label no codec claims.
    pub provider: Option<CodecIdentity>,
    pub selection: Selection,
}

// ── The candidate sets, derived from declarations ─────────────────────

/// The candidates the CPU executor has for a projection-class operand:
/// every direct realization the codec declares, the universal decode
/// (`decode_plan` is the f32 projection that follows it), and — for a
/// source whose stored bytes the executor knows how to re-quantise, which
/// it declares by naming a direct bf16 kernel — the executor's own compact
/// forms. Nothing is added by label.
pub fn cpu_projection_candidates(
    facts: &RepresentationFacts,
    decode_plan: PhysicalProjectionPlan,
    requantise: &[PhysicalProjectionPlan],
) -> Vec<RealizationId> {
    let mut out: Vec<RealizationId> = facts
        .direct_cpu_plans()
        .into_iter()
        .map(|p| RealizationId::cpu(RealizationForm::Direct(p)))
        .collect();
    if facts.registered.is_some() {
        out.push(RealizationId::cpu(RealizationForm::Decode(decode_plan)));
        if facts
            .direct_cpu_plans()
            .contains(&PhysicalProjectionPlan::FusedBf16)
        {
            out.extend(
                requantise
                    .iter()
                    .map(|p| RealizationId::cpu(RealizationForm::Requantise(*p))),
            );
        }
    }
    out
}

/// The selection every backend makes for the operations that are not a
/// projection, given the candidate it offers for them; `None` for the
/// shared expert, which no backend binds through the prepared plan.
pub fn common_selection(
    operand: &PlannedOperand,
    facts: &RepresentationFacts,
    bank_convert: WeightFormat,
) -> Option<Result<Selection, Box<SelectionRefusal>>> {
    let refuse = |kind, considered: Vec<(RealizationId, String)>| {
        Box::new(SelectionRefusal {
            operand: operand.operand.clone(),
            operation: operand.operation,
            representation: facts.label.clone(),
            requested: operand.access,
            kind,
            considered,
        })
    };
    match operand.operation {
        Operation::Embed => {
            let id = RealizationId::cpu(RealizationForm::DecodedGather);
            Some(if facts.registered.is_some() {
                Ok(Selection {
                    realization: id,
                    residency: ResidencyProfile::DECODED_F32,
                    reason: SelectionReason::EmbeddingGather,
                    candidates: vec![id],
                })
            } else {
                Err(refuse(RefusalKind::UnregisteredRepresentation, vec![]))
            })
        }
        Operation::ExpertBankSlice => {
            let id = RealizationId::cpu(RealizationForm::SliceStored {
                convert: bank_convert,
            });
            Some(match facts.admit_row_slicing() {
                Ok(()) => Ok(Selection {
                    realization: id,
                    residency: resident_profile(bank_convert),
                    reason: SelectionReason::BankSlicedAtLoad,
                    candidates: vec![id],
                }),
                Err(e) => Err(refuse(
                    RefusalKind::AccessRefused,
                    vec![(id, e.to_string())],
                )),
            })
        }
        Operation::SharedExpertProject => {
            Some(Err(refuse(RefusalKind::MissingRealization, vec![])))
        }
        Operation::Project(_) | Operation::OutputHead => None,
    }
}

/// The reference backend's answer: the literal f32 transcription, always,
/// for every projection of a registered representation.
pub fn reference_selection(
    operand: &PlannedOperand,
    facts: &RepresentationFacts,
) -> Result<Selection, Box<SelectionRefusal>> {
    if let Some(common) = common_selection(operand, facts, WeightFormat::F32) {
        return common;
    }
    let id = RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::ScalarF32));
    match &facts.registered {
        Some(r) => Ok(Selection {
            realization: id,
            residency: r.decode_residency,
            reason: SelectionReason::ReferenceOracle,
            candidates: vec![id],
        }),
        None => Err(Box::new(SelectionRefusal {
            operand: operand.operand.clone(),
            operation: operand.operation,
            representation: facts.label.clone(),
            requested: operand.access,
            kind: RefusalKind::UnregisteredRepresentation,
            considered: vec![],
        })),
    }
}

/// The class a projection-class operation names; the two non-projection
/// operations never reach a class table.
pub fn class_of(operation: Operation) -> Option<MatrixClass> {
    match operation {
        Operation::Project(class) => Some(class),
        Operation::OutputHead => Some(MatrixClass::OutputHead),
        Operation::Embed | Operation::ExpertBankSlice | Operation::SharedExpertProject => None,
    }
}
