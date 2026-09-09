//! **The experiment, as something an executor can be handed.**
//!
//! A [`MeasurementKey`] is four identities and no instructions. A
//! `TeacherForcedRequest` is instructions and no identity — 5a-0 made
//! every one of its controls a field, and left open which authority
//! fills them in. This is the join, and it exists under one rule:
//!
//! > **A request may only ever restate an experiment the optimiser
//! > already selected. It may not change one, and it may not author
//! > one.**
//!
//! # Construct-or-refuse
//!
//! The rule is enforced by making it impossible to hold a request that
//! violates it. There is no constructor that takes fields; the only way
//! to obtain one is [`MeasurementRequest::of`], which resolves the
//! candidate map over the record's own surface under the record's own
//! layout policy and refuses unless the result is the physical state the
//! key names.
//!
//! ```text
//! base_map + applied ─map_for─→ PrecisionMap ─resolve─→ RepresentationStateId
//!                                                              ║
//!                                        key.state()  ═════════╝  or refuse
//! ```
//!
//! So a request cannot drift from its key by construction, and
//! [`MeasurementRequest::derived_key`] re-derives the whole four-part
//! key from the request's own contents afterwards — no arguments, the
//! layout policy resolved from the name the request carries. A caller
//! holding a request can therefore check what it is about without
//! consulting the record it came from.
//!
//! # What is NOT here
//!
//! No path, no device, no backend, no kernel, no lowering target, no
//! store id, no batch width. The request says WHAT is to be measured and
//! under what protocol; WHERE the artifacts are is a locator's business
//! ([`super::executor::ArtifactLocator`]) and HOW they are executed is a
//! provider's. That separation is the whole reason this type is worth
//! having: an out-of-tree codec or a second lowering provider must be
//! able to become optimiser-searchable without the optimiser learning
//! its name.
//!
//! Nor a label. `TeacherForcedRequest::label` is documented as the one
//! field that is not a control, so it is derived from the experiment
//! rather than carried — which also means a sweep's outputs are named
//! by what they measured instead of by an environment variable.
//!
//! # Self-contained, deliberately
//!
//! The request clones the model identity, the surface and the map. On a
//! large surface that is not free, and it is the point: a request that
//! had to be read back against the record to be understood would let the
//! record move underneath a queued experiment. One request, one
//! experiment, checkable alone.

use std::collections::BTreeSet;

use super::super::compiler::SourceIdentity;
use super::super::map::PrecisionMap;
use super::super::measurement::EvidenceScale;
use super::super::quality::{gate_by_id, QualityGate};
use super::super::state::evidence_bank::EvidenceBank;
use super::super::state::identity::{RepresentationState, RepresentationStateId};
use super::super::state::instrument::InstrumentSemantics;
use super::super::state::key::MeasurementKey;
use super::super::state::protocol::ProtocolMismatch;
use super::super::state::resolved::{layout_admission, LayoutAdmission};
use super::super::state::snapshot::SearchSnapshot;
use super::super::state::surface::TensorSurface;
use crate::error::VindexError;

/// Why an authorised experiment could not be restated as a request.
///
/// One variant per missing or disagreeing authority, because each is
/// fixed by a different action: three of these are record repairs, two
/// are build limitations, and one — [`Self::StateMismatch`] — is a
/// defect in whatever produced the applied set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestRefusal {
    /// The record names its experiment by digest and does not say what
    /// the digests stand for.
    NoProtocol,
    /// It does say, and says something else.
    Protocol(ProtocolMismatch),
    /// The applied set holds a move this vocabulary does not have, so
    /// no map can be built from it.
    NoSuchMap { detail: String },
    /// The record names a layout policy this build does not implement,
    /// so nothing can be resolved under it.
    UnresolvableLayout { named: String, detail: String },
    /// **The map does not present the state the key names.**
    ///
    /// The falsifier, as a refusal. Whatever produced this applied set
    /// believes it reaches one physical state and it reaches another,
    /// and running it would file an observation of the second under the
    /// name of the first.
    StateMismatch {
        named: RepresentationStateId,
        resolved: RepresentationStateId,
    },
    /// The record is judged by a gate this build cannot resolve.
    UnresolvableGate { named: String, detail: String },
    /// This build has a gate of that name and it is not the record's
    /// gate. A run judged under it would carry a verdict drawn against
    /// thresholds nobody in this record agreed to.
    GateRedefined { named: String },
}

impl std::fmt::Display for RequestRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoProtocol => f.write_str(
                "this record names its corpus and instrument by digest and does not carry the \
                 declarations they stand for, so a run could not be told what to read or what \
                 the reading would mean",
            ),
            Self::Protocol(m) => write!(f, "{m}"),
            Self::NoSuchMap { detail } => write!(f, "no map could be built: {detail}"),
            Self::UnresolvableLayout { named, detail } => write!(
                f,
                "the record was resolved under layout policy `{named}`, which this build does \
                 not implement: {detail}"
            ),
            Self::StateMismatch { named, resolved } => write!(
                f,
                "the map resolves to state {} and the experiment names {} — the run would \
                 measure one representation and record it as another",
                resolved.short(),
                named.short()
            ),
            Self::UnresolvableGate { named, detail } => write!(
                f,
                "the record is judged by gate `{named}`, which this build cannot resolve: \
                 {detail}"
            ),
            Self::GateRedefined { named } => write!(
                f,
                "this build's `{named}` is not the gate this record carries — the thresholds \
                 have moved, and a verdict drawn under the current ones would not be the \
                 verdict this search is asking for"
            ),
        }
    }
}

impl From<ProtocolMismatch> for RequestRefusal {
    fn from(m: ProtocolMismatch) -> Self {
        Self::Protocol(m)
    }
}

/// **One authorised experiment, stated so that something else can
/// perform it.**
///
/// Every field is private. A request is obtained from
/// [`MeasurementRequest::of`] or not at all, because the invariant this
/// type carries is about the RELATIONSHIP between its fields and a
/// literal would let a caller construct one that holds no relationship.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementRequest {
    key: MeasurementKey,
    model: SourceIdentity,
    surface: TensorSurface,
    candidate_map: PrecisionMap,
    layout_admission: String,
    bank: EvidenceBank,
    instrument: InstrumentSemantics,
    scale: EvidenceScale,
    gate: String,
    procedure: String,
}

impl MeasurementRequest {
    /// **Restate an authorised experiment as a request, or refuse.**
    ///
    /// `applied` is the edit set the optimiser's own candidate carries;
    /// it is an input rather than a search, because finding an applied
    /// set that reaches a given state would be choosing a route, and
    /// choosing is what this layer must not do.
    pub fn of(
        snapshot: &SearchSnapshot,
        key: &MeasurementKey,
        applied: &BTreeSet<String>,
    ) -> Result<Self, RequestRefusal> {
        let protocol = snapshot.protocol().ok_or(RequestRefusal::NoProtocol)?;
        protocol.describes_key(key)?;

        let named_layout = &snapshot.semantics().layout_admission;
        let layout =
            layout_admission(named_layout).map_err(|e| RequestRefusal::UnresolvableLayout {
                named: named_layout.clone(),
                detail: e.to_string(),
            })?;

        let space = snapshot.space();
        let candidate_map = space
            .vocabulary
            .map_for(&space.base_map, applied)
            .map_err(|e| RequestRefusal::NoSuchMap {
                detail: e.to_string(),
            })?;

        let resolved = resolve_state(
            snapshot.graph().model(),
            &space.surface,
            &candidate_map,
            layout,
        );
        if &resolved != key.state() {
            return Err(RequestRefusal::StateMismatch {
                named: key.state().clone(),
                resolved,
            });
        }

        let gate = snapshot.gate();
        let implemented = gate_by_id(&gate.id).map_err(|e| RequestRefusal::UnresolvableGate {
            named: gate.id.clone(),
            detail: e.to_string(),
        })?;
        if &implemented != gate {
            return Err(RequestRefusal::GateRedefined {
                named: gate.id.clone(),
            });
        }

        Ok(Self {
            key: key.clone(),
            model: snapshot.graph().model().clone(),
            surface: space.surface.clone(),
            candidate_map,
            layout_admission: named_layout.clone(),
            bank: protocol.bank.clone(),
            instrument: protocol.instrument.clone(),
            scale: key.scale(),
            gate: gate.id.clone(),
            procedure: protocol.procedure.clone(),
        })
    }

    /// The experiment this request performs. Not chosen here.
    pub fn key(&self) -> &MeasurementKey {
        &self.key
    }

    /// The container the baseline arm must be read from, BY IDENTITY.
    pub fn model(&self) -> &SourceIdentity {
        &self.model
    }

    /// The surface the map resolves over.
    pub fn surface(&self) -> &TensorSurface {
        &self.surface
    }

    /// **What the candidate arm must present**, as a policy an encoder
    /// can act on rather than as a digest it cannot invert.
    pub fn candidate_map(&self) -> &PrecisionMap {
        &self.candidate_map
    }

    /// The layout policy the state was resolved under, by name.
    pub fn layout_admission(&self) -> &str {
        &self.layout_admission
    }

    pub fn bank(&self) -> &EvidenceBank {
        &self.bank
    }

    pub fn instrument(&self) -> &InstrumentSemantics {
        &self.instrument
    }

    pub fn scale(&self) -> EvidenceScale {
        self.scale
    }

    /// The gate the verdict is drawn under, by id, from the record —
    /// never a literal, and never an adapter's default.
    pub fn gate(&self) -> &str {
        &self.gate
    }

    /// The procedure that must perform this, by name. Resolved by the
    /// executor registry, which refuses a name this build has no
    /// executor for.
    pub fn procedure(&self) -> &str {
        &self.procedure
    }

    /// **How many samples this run consumes**, from the bank's own
    /// declaration.
    pub fn sequences(&self) -> usize {
        self.bank.sample_count()
    }

    /// **A name for this run's outputs**, derived from the experiment.
    ///
    /// Not a control. It changes where evidence lands and nothing about
    /// what is measured, so it is derived rather than carried — and a
    /// sweep's reports are then named by what they measured.
    pub fn label(&self) -> String {
        format!("exp-{}", self.key.short())
    }

    /// **Re-derive the whole experiment from this request alone.**
    ///
    /// No arguments: the layout policy is resolved from the name the
    /// request carries. A caller that has been handed a request can
    /// check what it is about without the record it came from — compare
    /// this with [`Self::key`] and they agree, or the request is not
    /// measuring what it says.
    ///
    /// There was a `attests_to_its_key` doing that comparison. It had no
    /// production caller and could not fail: [`Self::of`] is the only
    /// constructor, so an in-process request cannot be built that would
    /// refuse its own attestation, and a full mutation pass duly found
    /// its body replaceable by `Ok(())` with nothing to notice
    /// (ACT1-N10). The comparison belongs to whichever transition first
    /// lets a request cross a trust boundary — serde, a queue, a remote
    /// executor — and it should arrive then, with the test that can fail
    /// it.
    pub fn derived_key(&self) -> Result<MeasurementKey, VindexError> {
        let layout = layout_admission(&self.layout_admission)?;
        let state = resolve_state(&self.model, &self.surface, &self.candidate_map, layout);
        Ok(MeasurementKey::new(
            &state,
            &self.bank.id(),
            self.scale,
            &self.instrument.id(),
        ))
    }

    /// The gate this request is judged under, resolved.
    pub fn resolve_gate(&self) -> Result<QualityGate, VindexError> {
        gate_by_id(&self.gate)
    }
}

fn resolve_state(
    model: &SourceIdentity,
    surface: &TensorSurface,
    map: &PrecisionMap,
    layout: &dyn LayoutAdmission,
) -> RepresentationStateId {
    RepresentationState::resolve(model, surface, map, layout)
        .id()
        .clone()
}
