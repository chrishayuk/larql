//! **3b: the whole chain, from stored facts.**
//!
//! ```text
//! snapshot facts
//!     ↓ action space
//!     ↓ assessment
//!     ↓ ranking
//!     ↓ promotion
//! ```
//!
//! with none of the intermediates serialised. 1d closed the contract
//! half of this — margins, binding constraint, admissibility, refusal —
//! and left promotion open because `CandidateAssessment` carries a
//! `ranking_score`, a conclusion, and persisting one to make the test
//! pass would have been the exact cheat the stage forbids. The two
//! FACTS it needed are now in the snapshot: a per-state [`ByteLedger`]
//! and an [`ExecutionCostModel`].
//!
//! # What is real here and what is a fixture
//!
//! The recorded rung-5 numbers this replays against — kl p99, logical
//! bytes, covered mass — are exercised in `snapshot_tests`. This file
//! tests the DERIVATION CHAIN, and the per-token ledgers and the GPU
//! cost observation are fixtures: the record holds no measured GPU time
//! for P, T1 or S2, and inventing one and calling it recorded would be
//! worse than saying so.

use std::collections::{BTreeMap, BTreeSet};

use super::super::byte_ledger::{ByteLedger, ScopeBytes};
use super::super::compiler::SourceIdentity;
use super::super::decision::{decide_promotion, NoPromotableCandidate, PromotionDecision};
use super::super::diagnostic::DiagnosticPolicy;
use super::super::execution_cost::{ExecutionCostModel, ExecutionCostObservation};
use super::super::map::{Exception, PrecisionMap};
use super::super::measurement::{EvidenceScale, TailSupportPolicy};
use super::super::nvfp4_pack::{PackLayout, DTYPE_NVFP4};
use super::super::policy::Role;
use super::super::promotion::PromotionReadiness;
use super::super::quality::{
    kimi_logit_balanced_v1, Distribution, LogitEvidence, QualityBank, RoutingEvidence,
};
use super::super::search_evidence::SearchCalibrationRegistry;
use super::*;

const MODEL: &str = "kimi-linear-48b";

// ---------------------------------------------------------------- fixtures

fn model() -> SourceIdentity {
    SourceIdentity::synthetic(
        MODEL,
        "aligned-vindex3",
        [("target.decoder_stack".to_string(), "seg-dddd".to_string())],
    )
}

fn surface() -> TensorSurface {
    TensorSurface::new(
        [("e_proj", 256usize), ("k_proj", 128), ("m_proj", 64)]
            .into_iter()
            .map(|(p, rows)| {
                SurfaceTensor::new(
                    "target.decoder_stack",
                    format!("0.self_attn.{p}.weight"),
                    Role::DecoderLinear,
                    vec![rows, 64],
                )
            }),
    )
    .expect("distinct tensors")
}

fn compile(projection: &str) -> Exception {
    Exception {
        projection: Some(projection.into()),
        layers: None,
        encoding: Some(DTYPE_NVFP4.into()),
    }
}

fn space() -> SearchSpace {
    SearchSpace {
        surface: surface(),
        base_map: PrecisionMap {
            name: "base".into(),
            encoding: DTYPE_NVFP4.into(),
            roles: vec!["decoder-linear".into()],
            exceptions: vec![Exception {
                projection: None,
                layers: None,
                encoding: None,
            }],
        },
        vocabulary: ActionVocabulary::new([
            MapEdit::new("E24", compile("e_proj")),
            MapEdit::new("K25", compile("k_proj")),
            MapEdit::new("M26", compile("m_proj")),
        ])
        .expect("distinct names"),
    }
}

struct ShapeFootprint {
    surface: TensorSurface,
}

impl Footprint for ShapeFootprint {
    fn logical_bytes(&self, state: &RepresentationState) -> LogicalBytes {
        let total: u64 = state
            .decisions()
            .decisions()
            .iter()
            .map(|d| {
                let t = self
                    .surface
                    .get(&d.object, &d.tensor)
                    .expect("resolved against this surface");
                let elements: usize = t.shape.iter().product();
                if d.encoding.is_compiled() {
                    PackLayout::derive(&t.shape, &t.tensor)
                        .expect("compiled means admitted")
                        .total_len as u64
                } else {
                    elements as u64 * 2
                }
            })
            .sum();
        LogicalBytes::new(total)
    }
}

/// A per-token ledger. **Not** `LogicalBytes`: these are the bytes the
/// decoder READS per token, which is the quantity a throughput
/// prediction is a function of.
fn ledger(name: &str, e: u64, k: u64, m: u64) -> ByteLedger {
    let scope = |scope: &str, family: &str, baseline, candidate| ScopeBytes {
        scope: scope.into(),
        family: family.into(),
        baseline_bytes: baseline,
        candidate_bytes: candidate,
    };
    ByteLedger {
        model: MODEL.into(),
        baseline_representation: "BF16".into(),
        candidate_representation: name.into(),
        scopes: vec![
            scope("e", "attention", 16_384, e),
            scope("k", "attention", 8_192, k),
            scope("m", "attention", 4_096, m),
        ],
    }
}

/// One measured execution observation. A FIXTURE — the record holds no
/// GPU timing for these maps — but shaped like a real one, carrying the
/// machine, device, backend and commit that make a later reader able to
/// tell whether it predates a kernel change.
fn cost_model() -> ExecutionCostModel {
    ExecutionCostModel::new(vec![ExecutionCostObservation {
        id: "fixture-001".into(),
        machine: "fixture".into(),
        device: "fixture-gpu".into(),
        backend: "metal".into(),
        compiler_commit: "0000000".into(),
        model_identity: MODEL.into(),
        baseline_representation: "BF16".into(),
        candidate_representation: "E24".into(),
        families_changed: vec!["attention".into()],
        scopes_changed: 1,
        baseline_bytes_per_token: 28_672,
        candidate_bytes_per_token: 20_480,
        baseline_gpu_ms_per_token: 10.0,
        candidate_gpu_ms_per_token: 8.0,
        fixed_overhead_ms: 1.0,
        benchmark_protocol: "fixture".into(),
        evidence: vec![],
    }])
}

fn bank() -> EvidenceBank {
    EvidenceBank::new("kimi-teacher-forced/v1", "17d59a6b", ["seq-000"], 32)
}

fn instrument() -> InstrumentSemantics {
    InstrumentSemantics::new("kl", "distribution", "teacher-forced", "q2a").truncated_to(2048)
}

fn intent(scale: EvidenceScale) -> MeasurementIntent {
    MeasurementIntent::new(bank().id(), scale, instrument().id())
}

fn reading(kl_p99: f64) -> QualityBank {
    let dist = |v: f64| {
        Some(Distribution {
            count: 8192,
            min: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: v,
            max: v,
        })
    };
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
            route_flips: 0,
            positions_with_route_change: 0,
            layers_with_route_change: 0,
            first_layer_with_route_change: None,
            route_margin: None,
            route_weight_mass_moved: dist(0.113),
        },
        min_covered_mass: Some(0.6315),
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: dist(0.065),
        top10_rank_displacement: None,
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: dist(0.03),
    }
}

fn applied(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| (*n).to_string()).collect()
}

fn footprint() -> ShapeFootprint {
    ShapeFootprint { surface: surface() }
}

fn realize(names: &[&str]) -> ResolvedState {
    let space = space();
    let map = space
        .vocabulary
        .map_for(&space.base_map, &applied(names))
        .expect("known moves");
    let state = RepresentationState::resolve(&model(), &surface(), &map, &PackLayoutAdmission);
    let bytes = footprint().logical_bytes(&state);
    ResolvedState::new(state, bytes)
}

/// The whole factual record: which states exist, how they were reached,
/// what was observed, what each reads per token, and what execution has
/// been measured to cost. No verdict, rank, frontier or promotion.
fn recorded() -> SearchSnapshot {
    let root = realize(&[]);
    let e24 = realize(&["E24"]);

    let mut graph =
        RepresentationStateGraph::new(TransitionPolicy::StrictlyImprovingPhysical, root.clone());
    graph
        .apply(
            root.physical_id(),
            Action::new("+E24").adding(["E24"]),
            e24.clone(),
            Provenance::new("rung5/N3"),
        )
        .expect("lighter");

    let mut measurements = MeasurementRegistry::new();
    measurements
        .record(
            intent(EvidenceScale::Authority).key_for(root.physical_id()),
            reading(0.0),
        )
        .expect("record");
    measurements
        .record(
            intent(EvidenceScale::Authority).key_for(e24.physical_id()),
            reading(1.2e-3),
        )
        .expect("record");

    SearchSnapshot::new(
        space(),
        SearchConfig {
            objective: Objective::MinimiseLogicalBytes,
            gate: kimi_logit_balanced_v1(),
            tail_support: TailSupportPolicy::route_cal_1(),
            calibrations: SearchCalibrationRegistry::route_cal_1(),
            diagnostic_policy: DiagnosticPolicy::bs2_kimi_v1(),
            semantics: SearchSemantics::new(
                "exchange-1-out-1-in/v1",
                "ruling-1-three-prunes/v1",
                "search-evidence-ladder/v1",
                "decide-promotion-ordinal/v1",
                "physical-prize-first/v1",
                "logical-bytes/v1",
            ),
            ranking: RankingSemantics::new(RankingRule::PhysicalPrizeFirst),
        },
        SearchFacts {
            graph,
            measurements,
            byte_ledgers: BTreeMap::from([
                (
                    root.physical_id().clone(),
                    ledger("BF16", 16_384, 8_192, 4_096),
                ),
                (
                    e24.physical_id().clone(),
                    ledger("E24", 4_096, 8_192, 4_096),
                ),
            ]),
            execution_cost: cost_model(),
        },
    )
}

/// Store it, throw the live object away, read it back.
fn reloaded() -> SearchSnapshot {
    let json = serde_json::to_string(&recorded()).expect("serialize");
    let back: SearchSnapshot = serde_json::from_str(&json).expect("deserialize");
    back.check_schema().expect("schema");
    back
}

// ------------------------------------------------------------ the chain

#[test]
fn the_action_space_is_derived_from_the_snapshot_alone() {
    // 1d could not do this: the snapshot had no surface, base map or
    // vocabulary, so nothing could build a `Generator` from it.
    let snap = reloaded();
    let set = snap
        .generator(&PackLayoutAdmission, &footprint())
        .candidates(&applied(&[]), &intent(EvidenceScale::Diagnostic))
        .expect("generate");

    let census = set.census();
    assert_eq!(census.enumerated, 3, "+E24, +K25, +M26");
    assert!(census.conserves(), "{census}");
    assert_eq!(census.eligible, 3);
}

#[test]
fn the_next_experiment_is_derived_end_to_end() {
    let snap = reloaded();
    let selection = snap
        .next_experiment(
            &applied(&[]),
            &intent(EvidenceScale::Diagnostic),
            &PackLayoutAdmission,
            &footprint(),
        )
        .expect("derive");

    match selection {
        Selection::Ranked { chosen, considered } => {
            assert_eq!(considered, 3);
            assert_eq!(chosen.leading().action.label, "+E24", "the biggest prize");
            assert_eq!(chosen.key.scale(), EvidenceScale::Diagnostic);
        }
        other => panic!("expected a ranked selection, got {other:?}"),
    }
}

#[test]
fn promotion_is_derived_from_stored_facts() {
    // The gap 1d left open. `CandidateAssessment` is rebuilt here from
    // the ledgers, the cost model and the two contract standings — never
    // loaded — and `decide_promotion` is the crate's own, unmodified.
    let snap = reloaded();
    let candidates = snap
        .promotion_candidates(EvidenceScale::Authority)
        .expect("the cost model covers this model");

    // The graph holds one edge, and both its ends carry a reading and a
    // ledger. A move with either missing is skipped, not defaulted — a
    // marginal quantity with one end absent is not a smaller number, it
    // is no number.
    assert_eq!(candidates.len(), 1);
    let only = &candidates[0];
    assert_eq!(only.id, "+E24");

    let assessment = &only.promotion.assessment;
    assert_eq!(assessment.scale, EvidenceScale::Authority);
    assert_eq!(
        assessment.bytes_removed_marginal, 12_288,
        "16,384 -> 4,096 per token, computed from two ledgers"
    );
    assert!(assessment.binding_after().is_some());

    let decision = decide_promotion(
        &candidates,
        &snap.config().calibrations,
        snap.tail_support(),
    );
    // What matters is that a decision is REACHED from stored facts. Its
    // content is `decide_promotion`'s to determine, and this test does
    // not second-guess it.
    assert!(
        !matches!(
            decision,
            PromotionDecision::None {
                reason: NoPromotableCandidate::EmptySet
            }
        ),
        "the candidate set reached the comparator: {decision:?}"
    );
}

#[test]
fn a_move_with_one_end_unmeasured_is_skipped_rather_than_defaulted() {
    // A second edge whose child nothing has measured. It is a real move
    // in a real graph and it produces NO promotion candidate — not one
    // carrying zeroes, because a marginal quantity with one end absent
    // is no number at all.
    let mut snap = recorded();
    let root = realize(&[]);
    let k25 = realize(&["K25"]);
    let mut facts = snap.facts().clone();
    facts
        .graph
        .apply(
            root.physical_id(),
            Action::new("+K25").adding(["K25"]),
            k25.clone(),
            Provenance::new("rung5/N3"),
        )
        .expect("lighter");
    // It even has a ledger; what it lacks is a READING.
    facts.byte_ledgers.insert(
        k25.physical_id().clone(),
        ledger("K25", 16_384, 2_048, 4_096),
    );
    snap = SearchSnapshot::new(snap.space().clone(), snap.config().clone(), facts);

    assert_eq!(snap.graph().edge_count(), 2, "two moves were built");
    let candidates = snap
        .promotion_candidates(EvidenceScale::Authority)
        .expect("no cost refusal");
    assert_eq!(candidates.len(), 1, "only the measured move");
    assert_eq!(candidates[0].id, "+E24");
}

#[test]
fn readiness_and_diagnostic_come_from_the_stored_policy_and_bank() {
    let snap = reloaded();
    let candidates = snap
        .promotion_candidates(EvidenceScale::Authority)
        .expect("cost");
    let only = &candidates[0];

    // The diagnostic vector is read through the STORED policy, so a
    // snapshot taken under one policy cannot be replayed under another
    // without saying so.
    assert_eq!(
        only.diagnostic.policy_id,
        DiagnosticPolicy::bs2_kimi_v1().id
    );
    assert!(!only.diagnostic.readings.is_empty());
    assert!(matches!(
        only.promotion.readiness(),
        PromotionReadiness::Priceable
            | PromotionReadiness::ProxySupported
            | PromotionReadiness::Uninformed
    ));

    // No proxies were invented. A proxy is a registered finding about an
    // instrument, and none exists for these statistics.
    assert!(only.promotion.proxies.is_empty());
}

// ------------------------------------------------- still no conclusions

#[test]
fn the_widened_snapshot_still_stores_no_conclusion() {
    // Six new fields went in. Every one is a fact or a rule, and the
    // structural check still holds over the whole document.
    let json = serde_json::to_value(recorded()).expect("serialize");

    const FORBIDDEN: &[&str] = &[
        "admissible",
        "admitted",
        "refused",
        "binding",
        "frontier",
        "promotion",
        "rank",
        "chosen",
        "best",
        "recommendation",
        "failures",
        "passed",
        "adjudication",
        "ranking_score",
        "selection",
        "next_experiment",
    ];

    fn walk(value: &serde_json::Value, path: &str, found: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    // `config.calibrations[].verdict` is a REGISTERED
                    // FINDING about an instrument — ROUTE-CAL-1 saying
                    // how a statistic may be used — not a conclusion
                    // about a candidate. Named, so the check stays blunt
                    // everywhere else.
                    let calibration = path.starts_with(".config.calibrations");
                    if FORBIDDEN.contains(&k.as_str()) || (k == "verdict" && !calibration) {
                        found.push(format!("{path}.{k}"));
                    }
                    walk(v, &format!("{path}.{k}"), found);
                }
            }
            serde_json::Value::Array(items) => {
                for (i, v) in items.iter().enumerate() {
                    walk(v, &format!("{path}[{i}]"), found);
                }
            }
            _ => {}
        }
    }

    let mut found = Vec::new();
    walk(&json, "", &mut found);
    assert!(
        found.is_empty(),
        "conclusions in the stored form: {found:?}"
    );

    // And the new FACTS are all there, so the emptiness is not vacuous.
    assert_eq!(
        json["space"]["vocabulary"]["edits"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert!(json["space"]["surface"].is_object());
    assert!(json["space"]["base_map"].is_object());
    assert_eq!(json["facts"]["byte_ledgers"].as_object().unwrap().len(), 2);
    assert_eq!(
        json["facts"]["execution_cost"]["observations"][0]["machine"],
        "fixture"
    );
    assert!(json["config"]["diagnostic_policy"]["id"].is_string());

    // The cost itself is NOT stored — only the observations it is
    // derived from, and the model's own honesty about its calibration.
    assert!(json["facts"]["execution_cost"]
        .get("gpu_ms_per_token")
        .is_none());
    assert!(json["facts"]["execution_cost"].get("status").is_none());
}

#[test]
fn a_role_survives_an_owning_deserializer() {
    // `TensorSurface` carries `Role`, whose `Deserialize` read `&str`
    // and so demanded a BORROWED string: it worked for `from_str` and
    // failed for `from_value`, `from_reader` and every binary format —
    // with an `invalid type: string, expected a borrowed string` far
    // from the cause. An MCP surface will go through `Value`.
    let value = serde_json::to_value(recorded()).expect("serialize");
    let back: SearchSnapshot = serde_json::from_value(value).expect("owning deserializer");
    assert_eq!(back.space().surface, surface());

    let bytes = serde_json::to_vec(&recorded()).expect("bytes");
    let from_reader: SearchSnapshot =
        serde_json::from_reader(bytes.as_slice()).expect("reader deserializer");
    assert_eq!(from_reader.space().surface, surface());
}
