//! **The Rung 5 record, as facts.**
//!
//! One canonical fixture, shared by every test that needs a real
//! search to read: the 1d replay gate, and the stage-4 views that
//! render it. Two copies of these numbers would be two records, and
//! the second one would drift.
//!
//! Rung 5's neighbourhoods 1 and 2, measured at 8192 positions on the
//! selection bank and judged by the frozen `kimi-logit-balanced-v1`:
//!
//! ```text
//! map   logical bytes      auth kl p99   recorded verdict
//! P     13,684,764,800     3.3532e-03    admitted (K25 survives)
//! T1    13,682,673,664     3.6480e-03    REFUSED  kl 3.648e-3 > 3.500e-3
//! S2    13,602,484,352     4.0563e-03    REFUSED
//! S1    13,600,393,216     —             never measured; dominated,
//!                                        and the protocol spends ONE run
//! ```
//!
//! Only `kl_p99`, `min_covered_mass` and the byte figures are recorded
//! values. The other criteria are fixtures chosen to sit well inside
//! their limits, so that what a replay turns on is the recorded
//! evidence and not a number invented here — and so that `binding()`
//! reproducing `KlP99` means something.

use std::collections::{BTreeMap, BTreeSet};

use super::super::compiler::{read_source_identity, SourceIdentity};
use super::super::diagnostic::DiagnosticPolicy;
use super::super::execution_cost::ExecutionCostModel;
use super::super::map::{Exception, PrecisionMap};
use super::super::measure::TEACHER_FORCED_TWO_ARM;
use super::super::measurement::{EvidenceScale, TailSupportPolicy};
use super::super::nvfp4_pack::DTYPE_NVFP4;
use super::super::policy::Role;
use super::super::quality::{
    kimi_logit_balanced_v1, Distribution, LogitEvidence, QualityBank, QualityGate, RoutingEvidence,
};
use super::super::search_evidence::SearchCalibrationRegistry;
use super::accounting::read_source_storage;
use super::resolved::{layout_admission, PACK_LAYOUT_ADMISSION};
use super::*;

// ---------------------------------------------------------------- fixtures

pub fn model() -> SourceIdentity {
    SourceIdentity::synthetic(
        "kimi-linear-48b",
        "aligned-vindex3",
        [("target.decoder_stack".to_string(), "seg-dddd".to_string())],
    )
}

pub fn surface() -> TensorSurface {
    TensorSurface::new(
        ["q_proj", "k_proj", "v_proj", "o_proj"]
            .into_iter()
            .map(|p| {
                SurfaceTensor::new(
                    "target.decoder_stack",
                    format!("0.self_attn.{p}.weight"),
                    Role::DecoderLinear,
                    vec![64, 64],
                )
            }),
    )
    .expect("distinct tensors")
}

pub fn protect(projection: &str) -> Exception {
    Exception {
        projection: Some(projection.into()),
        layers: None,
        encoding: None,
    }
}

pub fn map(exceptions: Vec<Exception>) -> PrecisionMap {
    PrecisionMap {
        name: "m".into(),
        encoding: DTYPE_NVFP4.into(),
        roles: vec!["decoder-linear".into()],
        exceptions,
    }
}

pub fn realization(exceptions: Vec<Exception>, bytes: u64) -> ResolvedState {
    ResolvedState::new(
        RepresentationState::resolve(&model(), &surface(), &map(exceptions), &PackLayoutAdmission),
        LogicalBytes::new(bytes),
    )
}

pub fn p() -> ResolvedState {
    realization(vec![protect("v_proj"), protect("o_proj")], 13_684_764_800)
}
pub fn t1() -> ResolvedState {
    realization(vec![protect("o_proj")], 13_682_673_664)
}
pub fn s2() -> ResolvedState {
    realization(vec![protect("k_proj")], 13_602_484_352)
}
pub fn s1() -> ResolvedState {
    realization(vec![], 13_600_393_216)
}

pub fn dist(p99: f64, max: f64) -> Option<Distribution> {
    Some(Distribution {
        count: 8192,
        min: 0.0,
        p50: 0.0,
        p95: 0.0,
        p99,
        max,
    })
}

/// An 8192-position authority reading. `kl_p99`, `route_flips` and
/// `min_covered_mass` are the recorded values; the rest are fixtures
/// sitting well inside their limits.
pub fn authority_reading(kl_p99: f64, route_flips: u64) -> QualityBank {
    QualityBank {
        activations: None,
        positions: 8192,
        logits: LogitEvidence {
            kl_p50: 0.0,
            kl_p95: 0.0,
            kl_p99,
            max_logit_delta: 0.0,
            top1_flips: 0,
            top10_changes: 0,
        },
        routing: RoutingEvidence {
            route_flips,
            positions_with_route_change: 0,
            layers_with_route_change: 0,
            first_layer_with_route_change: None,
            route_margin: None,
            route_weight_mass_moved: dist(0.113, 0.19),
        },
        min_covered_mass: Some(0.6315),
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: dist(0.065, 0.065),
        top10_rank_displacement: None,
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: dist(0.03, 0.03),
    }
}

pub fn selection_bank() -> EvidenceBank {
    EvidenceBank::new(
        "kimi-teacher-forced/v1",
        "17d59a6b",
        (0..256).map(|i| format!("seq-{i:03}")),
        32,
    )
}

pub fn instrument() -> InstrumentSemantics {
    InstrumentSemantics::new(
        "kl(baseline || candidate)",
        "distribution{min,p50,p95,p99,max}",
        "teacher-forced, all positions",
        "q2a-teacher-forced/baseline-vs-overlay",
    )
    .truncated_to(2048)
}

pub fn semantics() -> SearchSemantics {
    SearchSemantics::new(
        "exchange-1-out-1-in/v1",
        "ruling-1-three-prunes/v1",
        "search-evidence-ladder/v1",
        "decide-promotion-ordinal/v1",
        "physical-prize-first/v1",
        "logical-bytes/v1",
        PACK_LAYOUT_ADMISSION,
    )
}

pub fn key_for(s: &ResolvedState, scale: EvidenceScale) -> MeasurementKey {
    MeasurementKey::new(
        s.physical_id(),
        &selection_bank().id(),
        scale,
        &instrument().id(),
    )
}

/// The space, config and facts a snapshot is built from. Split so the
/// tests below vary one at a time.
pub fn space() -> SearchSpace {
    SearchSpace {
        surface: surface(),
        base_map: map(vec![]),
        vocabulary: ActionVocabulary::default(),
        applied: BTreeSet::new(),
    }
}

pub fn config() -> SearchConfig {
    SearchConfig {
        objective: Objective::MinimiseLogicalBytes,
        gate: kimi_logit_balanced_v1(),
        tail_support: TailSupportPolicy::route_cal_1(),
        calibrations: SearchCalibrationRegistry::default(),
        diagnostic_policy: DiagnosticPolicy::bs2_kimi_v1(),
        semantics: semantics(),
        ranking: RankingSemantics::new(RankingRule::PhysicalPrizeFirst),
        standing_intent: standing_intent(),
        protocol: Some(protocol()),
    }
}

/// The declarations the record's standing intent stands for.
///
/// The bank and instrument are the same values `standing_intent` is
/// built from — one authority, so the record cannot describe a protocol
/// it does not search under.
pub fn protocol() -> MeasurementProtocol {
    MeasurementProtocol::new(selection_bank(), instrument(), TEACHER_FORCED_TWO_ARM)
}

/// The experiment the record's next run would be.
pub fn standing_intent() -> MeasurementIntent {
    MeasurementIntent::new(
        selection_bank().id(),
        EvidenceScale::Authority,
        instrument().id(),
    )
}

pub fn facts(graph: RepresentationStateGraph, measurements: MeasurementRegistry) -> SearchFacts {
    SearchFacts {
        graph,
        measurements,
        byte_ledgers: BTreeMap::new(),
        execution_cost: ExecutionCostModel::new(Vec::new()),
        accounting: None,
    }
}

pub fn snapshot(
    graph: RepresentationStateGraph,
    measurements: MeasurementRegistry,
) -> SearchSnapshot {
    SearchSnapshot::new(space(), config(), facts(graph, measurements))
}

/// The Rung 5 record, as FACTS: which states exist, how they were
/// reached, and what was observed of them. No verdicts.
pub fn rung5_snapshot() -> SearchSnapshot {
    let mut graph = RepresentationStateGraph::new(TransitionPolicy::StrictlyImprovingPhysical, p());
    for (child, action, who) in [
        (
            t1(),
            Action::new("−M26 +K24").removing(["M26"]).adding(["K24"]),
            "rung5/N2",
        ),
        (
            s2(),
            Action::new("−K25 +H").removing(["K25"]).adding(["H"]),
            "rung5/N1",
        ),
        (
            s1(),
            Action::new("−M26 +H").removing(["M26"]).adding(["H"]),
            "rung5/N1",
        ),
    ] {
        graph
            .apply(p().physical_id(), action, child, Provenance::new(who))
            .expect("all three are physically lighter than the parent");
    }

    let mut measurements = MeasurementRegistry::new();
    for (s, kl, flips) in [
        (p(), 3.3532e-3, 1427),
        (t1(), 3.6480e-3, 1570),
        (s2(), 4.0563e-3, 1309),
    ] {
        measurements
            .record(
                key_for(&s, EvidenceScale::Authority),
                authority_reading(kl, flips),
            )
            .expect("record");
    }

    snapshot(graph, measurements)
}

/// Store it, throw the in-memory object away, and read it back. Every
/// assertion below runs off the reloaded facts.
pub fn reloaded() -> SearchSnapshot {
    let json = serde_json::to_string(&rung5_snapshot()).expect("serialize");
    let back: SearchSnapshot = serde_json::from_str(&json).expect("deserialize");
    back.check_schema().expect("schema");
    back
}

/// The Rung 5 record with a **diagnostic** reading added on S1 — the
/// state the protocol never spent an authority run on.
///
/// The reading is P's own numbers, so it clears the contract outright.
/// The whole ladder rests on that not being an admission: a reader who
/// saw S1 admitted here would be reading a short bank as authority,
/// which is the inference R5-F4 and R5-F9 closed.
pub fn rung5_with_diagnostic_on_s1() -> SearchSnapshot {
    let base = rung5_snapshot();
    let mut measurements = base.measurements().clone();
    measurements
        .record(
            key_for(&s1(), EvidenceScale::Diagnostic),
            authority_reading(3.3532e-3, 1427),
        )
        .expect("record");
    snapshot(base.graph().clone(), measurements)
}

// ------------------------------------------------ a record over a real container

/// **A search record over a REAL encoded container.**
///
/// Every fact traces to the container's own sealed authority: the
/// surface is what it stores, the model identity is read from its index,
/// and the accounting facts are read through the segment digests that
/// identity seals. Nothing derived is stored — no bind, no price table,
/// no footprint, no ranking.
///
/// Lives here and not beside one test module for the reason
/// `state/tests/container.rs` already gives about containers: several
/// modules now need "the record that can actually answer", and three
/// builders would be three ideas of what such a record is. The stage-4
/// view tests and the stage-5b actuation tests read the same one.
pub struct PricedRecord {
    container: std::path::PathBuf,
    applied: BTreeSet<String>,
    layout: String,
    accounting: String,
    shape: Vec<usize>,
    accounting_from: Option<std::path::PathBuf>,
    distinct_edits: bool,
    protocol: Option<MeasurementProtocol>,
    intent: Option<MeasurementIntent>,
    gate: QualityGate,
}

impl PricedRecord {
    pub fn new(container: &std::path::Path) -> Self {
        Self {
            container: container.to_path_buf(),
            applied: BTreeSet::new(),
            layout: PACK_LAYOUT_ADMISSION.to_string(),
            accounting: "logical-bytes/v1".to_string(),
            shape: vec![64, 64],
            accounting_from: None,
            distinct_edits: false,
            protocol: None,
            intent: None,
            gate: kimi_logit_balanced_v1(),
        }
    }

    /// The applied set the record's next question is asked FROM.
    pub fn applied(mut self, applied: BTreeSet<String>) -> Self {
        self.applied = applied;
        self
    }

    /// The layout policy the record declares. Whatever is named here is
    /// what state resolution and price-table construction must both
    /// resolve to; nothing may fall back to this build's favourite.
    pub fn layout(mut self, declared: &str) -> Self {
        self.layout = declared.to_string();
        self
    }

    /// The physical accounting procedure the record declares.
    pub fn accounting(mut self, declared: &str) -> Self {
        self.accounting = declared.to_string();
        self
    }

    /// The surface's tensor shape — it decides whether the pack layout
    /// can hold the tensor, which is what makes the declared layout
    /// policy observable.
    pub fn shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = shape;
        self
    }

    /// Read the accounting facts from ANOTHER container.
    pub fn accounting_from(mut self, other: &std::path::Path) -> Self {
        self.accounting_from = Some(other.to_path_buf());
        self
    }

    /// Give the two edits different resolutions, so the policy has more
    /// than one opportunity to order rather than two routes to one.
    pub fn distinct_edits(mut self) -> Self {
        self.distinct_edits = true;
        self
    }

    /// Carry the declarations the standing intent's digests stand for,
    /// AND search under them — the consistent case, and what a real
    /// record must always be.
    pub fn with_protocol(mut self, protocol: MeasurementProtocol) -> Self {
        self.intent = Some(protocol.intent(EvidenceScale::Authority));
        self.protocol = Some(protocol);
        self
    }

    /// Carry declarations while searching under the FIXTURE's intent —
    /// the inconsistent case, which a record must be refused for. It has
    /// no legitimate use outside a test that asserts the refusal, which
    /// is why it is a separate call rather than an argument.
    pub fn with_protocol_only(mut self, protocol: MeasurementProtocol) -> Self {
        self.protocol = Some(protocol);
        self
    }

    /// The behavioural contract the record's conclusions are drawn
    /// against. Supplied so a test can carry a gate whose id this build
    /// implements and whose thresholds have moved — the case a run must
    /// refuse rather than judge under whichever definition it compiled.
    pub fn gate(mut self, gate: QualityGate) -> Self {
        self.gate = gate;
        self
    }

    pub fn build(self) -> SearchSnapshot {
        let model = read_source_identity(&self.container).expect("identity");
        let facts_root = self
            .accounting_from
            .unwrap_or_else(|| self.container.clone());
        let facts_model = read_source_identity(&facts_root).expect("identity");
        let accounting = read_source_storage(&facts_root, &facts_model).expect("storage facts");
        let priced = read_source_storage(&self.container, &model).expect("storage facts");

        // The container's own tensors, at an NVFP4-admissible shape.
        let surface = TensorSurface::new(priced.tensors().map(|(id, _)| {
            SurfaceTensor::new(
                &id.object,
                &id.tensor,
                Role::DecoderLinear,
                self.shape.clone(),
            )
        }))
        .expect("one entry per stored tensor");

        // The role is in the map's domain and a blanket exception
        // protects it, so the base state presents source bytes
        // throughout. Each edit lifts that protection, which is the
        // direction the objective wants.
        let compile_everything = || Exception {
            projection: None,
            layers: None,
            encoding: Some(DTYPE_NVFP4.into()),
        };
        let base_map = PrecisionMap {
            name: "protect-everything".into(),
            encoding: DTYPE_NVFP4.into(),
            roles: vec!["decoder-linear".into()],
            exceptions: vec![Exception {
                projection: None,
                layers: None,
                encoding: None,
            }],
        };
        // Two edits that resolve identically: one physical state reached
        // by two routes, which is the case `MeasurementOpportunity`
        // exists for — two realizations, ONE experiment.
        let second = match self.distinct_edits {
            // Compiles only the q projections, so it resolves to a
            // different physical state and the policy has two
            // opportunities to order rather than two routes to one.
            true => Exception {
                projection: Some("q_proj".into()),
                layers: None,
                encoding: Some(DTYPE_NVFP4.into()),
            },
            false => compile_everything(),
        };
        let vocabulary = ActionVocabulary::new([
            MapEdit::new("compile-all", compile_everything()),
            MapEdit::new("compile-all-by-another-name", second),
        ])
        .expect("distinct names");

        // The root: the base map resolved, priced by the same procedure
        // the record declares. A writer may compute; the record stores
        // only the resulting node.
        let layout = layout_admission(PACK_LAYOUT_ADMISSION).expect("this build implements it");
        let root_state = RepresentationState::resolve(&model, &surface, &base_map, layout);
        let root_bytes = priced
            .bind(&model, &surface)
            .ok()
            .and_then(|bound| {
                SurfaceFootprint::new(
                    &bound,
                    &surface,
                    layout,
                    &PackCompiledBytes,
                    &[DTYPE_NVFP4.to_string()],
                )
                .ok()
                .map(|f| f.logical_bytes(&root_state))
            })
            .unwrap_or_else(|| {
                // A shape the pack cannot hold prices as source
                // throughout, which is what the base map presents anyway.
                LogicalBytes::new(priced.tensors().map(|(_, f)| f.logical_bytes.get()).sum())
            });
        let root = ResolvedState::new(root_state.clone(), root_bytes);

        SearchSnapshot::new(
            SearchSpace {
                surface,
                base_map,
                vocabulary,
                applied: self.applied,
            },
            SearchConfig {
                objective: Objective::MinimiseLogicalBytes,
                gate: self.gate,
                tail_support: TailSupportPolicy::route_cal_1(),
                calibrations: SearchCalibrationRegistry::default(),
                diagnostic_policy: DiagnosticPolicy::bs2_kimi_v1(),
                semantics: SearchSemantics::new(
                    "exchange-1-out-1-in/v1",
                    "ruling-1-three-prunes/v1",
                    "search-evidence-ladder/v1",
                    "kimi-balanced-v1-authority-only/v1",
                    "physical-prize-first/v1",
                    &self.accounting,
                    &self.layout,
                ),
                ranking: RankingSemantics::new(RankingRule::PhysicalPrizeFirst),
                standing_intent: self.intent.unwrap_or_else(standing_intent),
                protocol: self.protocol,
            },
            SearchFacts {
                graph: RepresentationStateGraph::new(
                    TransitionPolicy::StrictlyImprovingPhysical,
                    root,
                ),
                measurements: MeasurementRegistry::default(),
                byte_ledgers: BTreeMap::new(),
                execution_cost: ExecutionCostModel::new(Vec::new()),
                accounting: Some(accounting),
            },
        )
    }
}
