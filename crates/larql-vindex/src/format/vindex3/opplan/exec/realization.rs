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

use serde::Serialize;

use super::backend::WeightFormat;
use super::cpu::physical::PhysicalProjectionPlan;
use crate::format::vindex3::represent::codec::{
    Acceleration, AccelerationBackend, CodecCapabilities, CodecError, CodecRegistry,
    ExtentCertificate, RepresentationCodec, RequiredAccess, ResidencyProfile,
};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

mod refusal;
mod selection;
pub use refusal::*;
pub use selection::*;

/// What the registry declares for one stored dtype — resolved once per
/// operand at preparation, and the only thing a backend is told about the
/// representation.
#[derive(Debug, Clone, PartialEq)]
pub struct RepresentationFacts {
    /// The label the facts were resolved from: the container's stored
    /// label, or the codec the plan declares when the stored label is a
    /// carrier dialect that names none (see [`Self::resolve_declared`]).
    pub label: String,
    /// The codec's declarations, when the label names a registered codec.
    /// `None` is a fact too: an unregistered label has no decode and no
    /// capabilities, and nothing binds it.
    pub registered: Option<RegisteredFacts>,
    /// Whether an overlay edit stands on the operand. An edit is an
    /// f32-space fact with no stored bytes, so no direct realization can
    /// honour it; only decode can.
    pub overlaid: bool,
    /// Whether an NVFP4 request for this operand binds at source
    /// precision — read from `OperandStore::nvfp4_request_binds_at_source`,
    /// the same fact the loader binds by. A backend that pins NVFP4 must
    /// honour it or its pin and the resident bytes disagree.
    pub nvfp4_at_source: bool,
    /// Whether an f16 request for this operand binds its compiled NVFP4
    /// image as stored — read from
    /// `OperandStore::f16_request_binds_compiled_nvfp4`, the same fact the
    /// loader binds by. A backend that pins f16 must honour it for the
    /// same reason.
    pub nvfp4_compiled: bool,
    /// Whether the store forbids manufacturing a representation at load
    /// (`--representation-source stored`) — read from
    /// `OperandStore::representation_source`, the same fact the loader
    /// refuses by. A selector must not offer a requantised form the
    /// loader would then refuse, or quietly run one the caller forbade.
    pub requantise_forbidden: bool,
}

/// A registered codec's declarations, copied out so a backend never holds
/// the codec itself.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredFacts {
    pub identity: CodecIdentity,
    pub capabilities: CodecCapabilities,
    pub accelerations: Vec<Acceleration>,
    pub decode_residency: ResidencyProfile,
    /// Every extent the codec declares, base first. One for a terminal
    /// representation; several for a progressive one, and then a pin has
    /// something to choose between.
    pub extents: Vec<ExtentCertificate>,
}

impl RepresentationFacts {
    /// Resolve `label` through the built-in registry.
    #[cfg(test)]
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
            nvfp4_at_source: false,
            nvfp4_compiled: false,
            requantise_forbidden: false,
        }
    }

    /// The same facts, with an overlay edit standing on the operand.
    pub fn overlaid(mut self) -> Self {
        self.overlaid = true;
        self
    }

    /// The same facts, with the store's answer to whether an NVFP4
    /// request binds this operand at source precision.
    pub fn with_nvfp4_at_source(mut self, at_source: bool) -> Self {
        self.nvfp4_at_source = at_source;
        self
    }

    /// The same facts, with the store's answer to whether an f16 request
    /// binds this operand's compiled NVFP4 image as stored.
    pub fn with_nvfp4_compiled(mut self, compiled: bool) -> Self {
        self.nvfp4_compiled = compiled;
        self
    }

    /// The same facts, with the store's answer to whether it forbids
    /// quantising at load.
    pub fn with_requantise_forbidden(mut self, forbidden: bool) -> Self {
        self.requantise_forbidden = forbidden;
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

    /// The facts for an operand whose container label is `stored` and
    /// whose plan declares `declared`. The STORED representation is
    /// authoritative: the container's label wins whenever it names a
    /// codec, because the bytes say what they are. The plan's declaration
    /// is a legacy default, consulted only for a carrier dialect whose
    /// label names no codec — a packed MXFP4 bank stored as two `U8`
    /// streams. These are not two competing truths; the declaration fills
    /// in where the container says nothing a registry knows. One rule, so
    /// the selector and the bank loader cannot disagree about what a
    /// bank is.
    pub fn resolve_declared(
        registry: &CodecRegistry,
        stored: &str,
        declared: Option<&str>,
    ) -> Self {
        let label = match declared {
            Some(declared) if registry.by_label(stored).is_none() => declared,
            _ => stored,
        };
        Self::resolve_in(registry, label)
    }

    /// Admit slicing the stored bytes per expert — the packed-bank
    /// realization's requirement, judged BEFORE any byte is read. A
    /// registered codec must declare row access. An unregistered label
    /// has no capabilities to judge; registration itself is refused
    /// first, by the same rule as every other operation, so this answers
    /// only for a codec.
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
            extents: codec.extents(),
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
    /// The STORED bytes bound as a mapping of the container's segment —
    /// bound once, never copied or converted — and executed in place in
    /// their stored form. A bank's realization: one physical object
    /// serving every logical expert access, paged in as touched.
    MappedStored {
        format: WeightFormat,
        /// How the selected experts' pages are brought in for a token.
        access: MappedAccess,
    },
    /// A device backend's resident form, declared per class by that
    /// backend for its own target.
    DeviceResident(WeightFormat),
}

/// How a mapped bank's selected experts are brought into memory for one
/// token — an ACCESS realization of the same lossless bytes. The bytes,
/// the mapping and the touch are identical across variants; only the
/// request shape differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize)]
pub enum MappedAccess {
    /// The projection loop faults each page as it reaches it: one page
    /// per fault, serially, in row order.
    #[default]
    Demand,
    /// `madvise(MADV_WILLNEED)` over the selected experts' regions before
    /// the loop; the kernel decides how much it reads ahead.
    Advise,
    /// The selected experts' pages are touched concurrently, ordered by
    /// address, before the loop; every fault is taken in parallel and
    /// the loop then finds resident pages.
    Touch,
}

impl MappedAccess {
    pub const ALL: [MappedAccess; 3] = [
        MappedAccess::Demand,
        MappedAccess::Advise,
        MappedAccess::Touch,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MappedAccess::Demand => "demand",
            MappedAccess::Advise => "advise",
            MappedAccess::Touch => "touch",
        }
    }

    /// The policy named by a flag, or the names it does accept.
    pub fn parse(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|a| a.name() == name)
            .ok_or_else(|| {
                format!(
                    "unknown expert access `{name}`; one of {}",
                    Self::ALL.map(|a| a.name()).join(", ")
                )
            })
    }
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
            RealizationForm::MappedStored { format, .. } => format,
            RealizationForm::DeviceResident(format) => format,
        }
    }

    /// Whether this realization keeps its codec's DEPENDENCIES resident
    /// while serving — the fact a dependency pin's lifetime is set from.
    ///
    /// A realization over the STORED bytes reads what those bytes need on
    /// every token: a direct kernel over FP8 codes multiplies by the scale
    /// grid, a mapped bank executes in place. A realization that decodes
    /// — to f32, to a re-quantised image, to a device's own form — is
    /// finished with the dependency once the image exists.
    pub fn retains_dependencies(self) -> bool {
        match self.form {
            RealizationForm::Direct(_) | RealizationForm::MappedStored { .. } => true,
            RealizationForm::Decode(_)
            | RealizationForm::Requantise(_)
            | RealizationForm::SliceStored { .. }
            | RealizationForm::DecodedGather
            | RealizationForm::DeviceResident(_) => false,
        }
    }

    /// The access realization of a mapped form; every other form is
    /// brought in whole at binding and has none.
    pub fn access(self) -> MappedAccess {
        match self.form {
            RealizationForm::MappedStored { access, .. } => access,
            _ => MappedAccess::Demand,
        }
    }

    /// The same realization under another access policy — only a mapped
    /// form changes; every other form is returned as it is.
    pub fn with_access(self, access: MappedAccess) -> Self {
        match self.form {
            RealizationForm::MappedStored { format, .. } => Self {
                backend: self.backend,
                form: RealizationForm::MappedStored { format, access },
            },
            _ => self,
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
            RealizationForm::MappedStored { format, access } => {
                format!("mapped-stored/{format:?}/{}", access.name())
            }
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
    /// A per-expert bank's matrices bound as a mapping of the stored
    /// bytes, in their stored form — one physical binding per bank.
    BankMappedAsStored,
    /// Re-selected from the operand's candidates so the plan's physical
    /// working set fits the residency budget: a cheaper-resident
    /// realization the backend had considered and not preferred.
    BudgetPolicy,
    /// The device backend's class table names its resident form.
    DeviceClassTable,
    /// An embedding table is decoded whole and gathered per token.
    EmbeddingGather,
    /// A shared branch's scalar gate is one `[1, hidden]` row: decoded to
    /// f32 and applied by the literal dot product, on every backend.
    ScalarBranchGate,
    /// The reference backend takes the literal transcription, always.
    ReferenceOracle,
    /// An overlay edit stands on the operand; only decode can honour it.
    OverlaidEdit,
    /// The backend asked for a compact form the container does not store
    /// — NVFP4 from the class table, or a requantised Q8/Q4 image from
    /// the size policy under `--representation-source stored` — and the
    /// container holds this operand at source precision; it binds at that
    /// precision instead, as the loader does, and manufactures nothing.
    SourcePrecisionHeld,
    /// The class table asked for f16, and the container stores this
    /// operand compiled to NVFP4 with no source bytes; it binds as stored,
    /// as the loader does, and manufactures nothing.
    CompiledPrecisionHeld,
}

impl SelectionReason {
    pub const fn name(self) -> &'static str {
        match self {
            Self::DirectDeclared => "direct realization declared by the codec",
            Self::NoDirectRealization => "no direct realization registered",
            Self::ArmPrefersDecode => "the process arm prefers decoding",
            Self::SizePolicy => "size policy over a float source",
            Self::BankSlicedAtLoad => "packed bank sliced per expert at load",
            Self::BankMappedAsStored => "per-expert bank mapped as stored, bound once",
            Self::BudgetPolicy => "re-selected to fit the residency budget",
            Self::DeviceClassTable => "device class table",
            Self::EmbeddingGather => "table decoded whole, gathered per token",
            Self::ScalarBranchGate => "one-row branch gate decoded to f32, applied by a literal dot",
            Self::ReferenceOracle => "reference oracle",
            Self::OverlaidEdit => "an overlay edit stands on the operand; only decode honours it",
            Self::SourcePrecisionHeld => {
                "the container holds it at source precision; bound at f16, not the class format"
            }
            Self::CompiledPrecisionHeld => {
                "the container stores it compiled to NVFP4 with no source; bound as stored, not the class format"
            }
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
