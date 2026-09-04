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

use super::super::compiler::SourceIdentity;
use super::super::diagnostic::DiagnosticPolicy;
use super::super::execution_cost::ExecutionCostModel;
use super::super::map::{Exception, PrecisionMap};
use super::super::measurement::{EvidenceScale, TailSupportPolicy};
use super::super::nvfp4_pack::DTYPE_NVFP4;
use super::super::policy::Role;
use super::super::quality::{
    kimi_logit_balanced_v1, Distribution, LogitEvidence, QualityBank, RoutingEvidence,
};
use super::super::search_evidence::SearchCalibrationRegistry;
use super::resolved::PACK_LAYOUT_ADMISSION;
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
    }
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
