//! **3: what should we try next — and one answer, always.**
//!
//! Best-first is where implementation accidents start contaminating
//! replay, so the order is specified end to end and tested against
//! permuted input. If two runs over the same facts can disagree, nothing
//! above this layer is reproducible.

use std::collections::{BTreeMap, BTreeSet};

use super::super::compiler::SourceIdentity;
use super::super::map::{Exception, PrecisionMap};
use super::super::measurement::{EvidenceScale, TailSupportPolicy};
use super::super::nvfp4_pack::{PackLayout, DTYPE_NVFP4};
use super::super::policy::Role;
use super::super::quality::{
    kimi_logit_balanced_v1, Criterion, Distribution, LogitEvidence, QualityBank, RoutingEvidence,
};
use super::*;

// ---------------------------------------------------------------- fixtures

fn model() -> SourceIdentity {
    SourceIdentity {
        manifest_hash: "kimi-linear-48b".into(),
        graph_hash: "aligned-vindex3".into(),
        segments: BTreeMap::from([("target.decoder_stack".to_string(), "seg-dddd".to_string())]),
    }
}

/// Three projections of increasing width, so the moves that compile them
/// carry genuinely different physical prizes.
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

fn base_map() -> PrecisionMap {
    PrecisionMap {
        name: "base".into(),
        encoding: DTYPE_NVFP4.into(),
        roles: vec!["decoder-linear".into()],
        exceptions: vec![Exception {
            projection: None,
            layers: None,
            encoding: None,
        }],
    }
}

fn compile(projection: &str) -> Exception {
    Exception {
        projection: Some(projection.into()),
        layers: None,
        encoding: Some(DTYPE_NVFP4.into()),
    }
}

fn vocabulary() -> ActionVocabulary {
    ActionVocabulary::new([
        MapEdit::new("E24", compile("e_proj")),
        MapEdit::new("K25", compile("k_proj")),
        MapEdit::new("M26", compile("m_proj")),
    ])
    .expect("distinct names")
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
                let tensor = self
                    .surface
                    .get(&d.object, &d.tensor)
                    .expect("resolved against this surface");
                let elements: usize = tensor.shape.iter().product();
                if d.encoding.is_compiled() {
                    PackLayout::derive(&tensor.shape, &tensor.tensor)
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

fn bank() -> EvidenceBank {
    EvidenceBank::new("kimi-teacher-forced/v1", "17d59a6b", ["seq-000"], 32)
}

fn instrument() -> InstrumentSemantics {
    InstrumentSemantics::new("kl", "distribution", "teacher-forced", "q2a").truncated_to(2048)
}

fn intent(scale: EvidenceScale) -> MeasurementIntent {
    MeasurementIntent::new(bank().id(), scale, instrument().id())
}

fn semantics() -> RankingSemantics {
    RankingSemantics::new(RankingRule::PhysicalPrizeFirst)
        .because("rung 5 spent the run on the prize: -431,777,920 B over -2,091,136 B")
}

fn applied(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| (*n).to_string()).collect()
}

struct Rig {
    vocabulary: ActionVocabulary,
    base: PrecisionMap,
    surface: TensorSurface,
    footprint: ShapeFootprint,
    measurements: MeasurementRegistry,
    model: SourceIdentity,
}

impl Rig {
    fn new() -> Self {
        Self {
            vocabulary: vocabulary(),
            base: base_map(),
            surface: surface(),
            footprint: ShapeFootprint { surface: surface() },
            measurements: MeasurementRegistry::new(),
            model: model(),
        }
    }

    fn generator(&self) -> Generator<'_> {
        Generator {
            model: &self.model,
            surface: &self.surface,
            base_map: &self.base,
            vocabulary: &self.vocabulary,
            layout: &PackLayoutAdmission,
            footprint: &self.footprint,
            policy: TransitionPolicy::StrictlyImprovingPhysical,
            measurements: &self.measurements,
        }
    }

    fn candidates(&self, from: &[&str], scale: EvidenceScale) -> CandidateSet {
        self.generator()
            .candidates(&applied(from), &intent(scale))
            .expect("generate")
    }
}

// ------------------------------------------------------- one answer, always

#[test]
fn the_order_is_total_and_independent_of_the_order_candidates_arrived_in() {
    let rig = Rig::new();
    let policy = BestFirst::new(semantics());
    let set = rig.candidates(&[], EvidenceScale::Diagnostic);
    let assessed = policy.assess(&set, &NothingMeasured);
    assert_eq!(assessed.len(), 3, "+E24, +K25, +M26");

    let forward = policy.order(assessed.clone());
    let mut reversed = assessed.clone();
    reversed.reverse();
    let backward = policy.order(reversed);
    let mut rotated = assessed;
    rotated.rotate_left(1);
    let rotated = policy.order(rotated);

    let labels =
        |v: &[Assessment]| -> Vec<String> { v.iter().map(|a| a.action.label.clone()).collect() };
    assert_eq!(labels(&forward), labels(&backward));
    assert_eq!(labels(&forward), labels(&rotated));

    // The prize orders them: 256x64 removes most, 64x64 least.
    assert_eq!(labels(&forward), vec!["+E24", "+K25", "+M26"]);
    assert!(forward
        .windows(2)
        .all(|w| w[0].physical_delta <= w[1].physical_delta));
}

#[test]
fn the_tie_break_chain_is_stated_and_reaches_identity_last() {
    let s = semantics();
    assert_eq!(
        s.tie_break_chain(),
        [
            "registered rule",
            "greater physical improvement",
            "canonical child state id",
            "canonical child realization id",
            "canonical action identity",
        ]
    );
    // The chain is part of what the rule IS: changing it changes which
    // experiment runs first, so it must move the id.
    assert!(s.id().as_str().len() == 64);
    assert_eq!(s.rule.name(), "physical-prize-first");
    assert_eq!(format!("{}", s.id()), s.id().as_str());
    assert_eq!(s.id().short().len(), 12);

    // Provenance is not semantics.
    let reworded = RankingSemantics::new(RankingRule::PhysicalPrizeFirst).because("said otherwise");
    assert_ne!(reworded.provenance, s.provenance);
    assert_eq!(reworded.id(), s.id());
    assert!(RANKING_SEMANTICS_ID_VERSION.starts_with("ranking-semantics-id/"));
}

#[test]
fn candidates_that_tie_on_the_prize_still_have_exactly_one_order() {
    // Two moves of identical width: the rule cannot separate them, so
    // the chain must — and must do so the same way every run.
    let surface = TensorSurface::new(["a_proj", "b_proj"].into_iter().map(|p| {
        SurfaceTensor::new(
            "target.decoder_stack",
            format!("0.self_attn.{p}.weight"),
            Role::DecoderLinear,
            vec![64, 64],
        )
    }))
    .expect("distinct");
    let rig = Rig {
        vocabulary: ActionVocabulary::new([
            MapEdit::new("A", compile("a_proj")),
            MapEdit::new("B", compile("b_proj")),
        ])
        .expect("distinct names"),
        surface: surface.clone(),
        footprint: ShapeFootprint { surface },
        ..Rig::new()
    };
    let policy = BestFirst::new(semantics());
    let set = rig.candidates(&[], EvidenceScale::Diagnostic);
    let assessed = policy.assess(&set, &NothingMeasured);
    assert_eq!(assessed.len(), 2);
    assert_eq!(
        assessed[0].physical_delta, assessed[1].physical_delta,
        "a genuine tie on the rule"
    );

    let forward = policy.order(assessed.clone());
    let mut reversed = assessed;
    reversed.reverse();
    let backward = policy.order(reversed);
    assert_eq!(
        forward[0].child_state, backward[0].child_state,
        "the tie-break decides, not arrival order"
    );
    assert!(forward[0].child_state.as_str() < forward[1].child_state.as_str());
}

#[test]
fn two_realizations_of_one_state_are_separated_by_the_realization_tie_break() {
    // The chain's fourth element, reached only when two candidates agree
    // on the rule, on the prize AND on the physical state. That happens
    // exactly in 1b's case: `v_proj` held at source precision by a
    // protection in one map and by a layout refusal in the other. Same
    // bytes, same state, two realizations — and the order must still be
    // decided by identity rather than by arrival.
    let surface = TensorSurface::new([
        SurfaceTensor::new(
            "target.decoder_stack",
            "0.self_attn.q_proj.weight",
            Role::DecoderLinear,
            vec![64, 64],
        ),
        SurfaceTensor::new(
            "target.decoder_stack",
            "0.self_attn.v_proj.weight",
            Role::DecoderLinear,
            vec![64, 24],
        ),
    ])
    .expect("distinct");
    let protecting = PrecisionMap {
        name: "m".into(),
        encoding: DTYPE_NVFP4.into(),
        roles: vec!["decoder-linear".into()],
        exceptions: vec![Exception {
            projection: Some("v_proj".into()),
            layers: None,
            encoding: None,
        }],
    };
    let refusing = PrecisionMap {
        exceptions: vec![],
        ..protecting.clone()
    };
    let resolve = |m: &PrecisionMap| {
        ResolvedState::new(
            RepresentationState::resolve(&model(), &surface, m, &PackLayoutAdmission),
            LogicalBytes::new(4_000),
        )
    };
    let (a, b) = (resolve(&protecting), resolve(&refusing));
    assert_eq!(a.physical_id(), b.physical_id(), "one physical state");
    assert_ne!(a.realization_id(), b.realization_id(), "two realizations");

    let assessment = |r: &ResolvedState| Assessment {
        action: Action::new("+V"),
        parent_state: a.physical_id().clone(),
        child_state: r.physical_id().clone(),
        child_realization: r.realization_id().clone(),
        physical_delta: -4_000,
        child_bytes: r.logical_bytes(),
        intended_key: intent(EvidenceScale::Diagnostic).key_for(r.physical_id()),
        prior_observations: Vec::new(),
        parent_standing: None,
    };
    let policy = BestFirst::new(semantics());
    let forward = policy.order(vec![assessment(&a), assessment(&b)]);
    let backward = policy.order(vec![assessment(&b), assessment(&a)]);
    assert_eq!(forward[0].child_realization, backward[0].child_realization);
    assert!(
        forward[0].child_realization.as_str() < forward[1].child_realization.as_str(),
        "ordered by canonical realization id"
    );
}

// ------------------------------------- search orders states, measurement experiments

#[test]
fn two_routes_to_one_physical_state_are_one_experiment() {
    // The payoff of the whole identity model. From {} the moves `+E24`
    // and `+K25` reach different states; from {E24} the move `+K25` and
    // from {K25} the move `+E24` reach the SAME physical state by
    // different routes. Search keeps both; measurement runs one.
    let rig = Rig::new();
    let policy = BestFirst::new(semantics());

    let from_e = rig.candidates(&["E24"], EvidenceScale::Diagnostic);
    let from_k = rig.candidates(&["K25"], EvidenceScale::Diagnostic);
    let route_a = policy
        .assess(&from_e, &NothingMeasured)
        .into_iter()
        .find(|a| a.action.label == "+K25")
        .expect("E24 then K25");
    let route_b = policy
        .assess(&from_k, &NothingMeasured)
        .into_iter()
        .find(|a| a.action.label == "+E24")
        .expect("K25 then E24");

    assert_eq!(route_a.child_state, route_b.child_state, "one state");
    assert_eq!(route_a.intended_key, route_b.intended_key, "one experiment");
    assert_ne!(route_a.parent_state, route_b.parent_state, "two routes");
    assert_eq!(
        route_a.child_realization, route_b.child_realization,
        "here the routes also agree on the decisions; the stronger case is below"
    );

    let opportunities = policy.opportunities(policy.order(vec![route_a, route_b]));
    assert_eq!(opportunities.len(), 1, "scheduled once, not twice");
    assert_eq!(opportunities[0].routes(), 2);
    assert_eq!(
        opportunities[0].physical_delta(),
        opportunities[0].leading().physical_delta
    );
}

#[test]
fn two_realizations_of_one_state_still_schedule_one_experiment() {
    // **The strong form**, and the one the identity model exists for:
    //
    //     A --action x--> C (realization r1)
    //     B --action y--> C (realization r2)
    //     search:      r1 != r2, both kept
    //     measurement: one MeasurementKey, run once
    //
    // `v_proj` is layout-refused, so `+V` changes the decisions and no
    // bytes. From `{}` the move `+Q` reaches `{Q}` with v at Source;
    // from `{V}` the same move reaches `{V,Q}` with v LayoutRefused.
    // One physical state, two realizations, one experiment.
    let surface = TensorSurface::new([
        SurfaceTensor::new(
            "target.decoder_stack",
            "0.self_attn.q_proj.weight",
            Role::DecoderLinear,
            vec![64, 64],
        ),
        SurfaceTensor::new(
            "target.decoder_stack",
            "0.self_attn.v_proj.weight",
            Role::DecoderLinear,
            vec![64, 24],
        ),
    ])
    .expect("distinct");
    let rig = Rig {
        vocabulary: ActionVocabulary::new([
            MapEdit::new("Q", compile("q_proj")),
            MapEdit::new("V", compile("v_proj")),
        ])
        .expect("distinct names"),
        surface: surface.clone(),
        footprint: ShapeFootprint { surface },
        ..Rig::new()
    };
    let policy = BestFirst::new(semantics());

    let pick = |from: &[&str]| {
        policy
            .assess(
                &rig.candidates(from, EvidenceScale::Diagnostic),
                &NothingMeasured,
            )
            .into_iter()
            .find(|a| a.action.label == "+Q")
            .expect("+Q is eligible from here")
    };
    let plain = pick(&[]);
    let after_v = pick(&["V"]);

    assert_eq!(plain.child_state, after_v.child_state, "one physical state");
    assert_ne!(
        plain.child_realization, after_v.child_realization,
        "two realizations — the action spaces differ"
    );
    assert_eq!(plain.intended_key, after_v.intended_key, "one experiment");

    let opportunities = policy.opportunities(policy.order(vec![plain, after_v]));
    assert_eq!(
        opportunities.len(),
        1,
        "grouping is by experiment, not by realization"
    );
    assert_eq!(opportunities[0].routes(), 2);
    let realizations: BTreeSet<&str> = opportunities[0]
        .candidates
        .iter()
        .map(|c| c.child_realization.as_str())
        .collect();
    assert_eq!(realizations.len(), 2, "both routes survive inside it");
}

#[test]
fn an_observation_of_the_state_reaches_every_route_to_it() {
    // Once the experiment lands, both search realizations inherit it —
    // 1c keys evidence on the physical state while 1b keeps the
    // realizations apart.
    let mut rig = Rig::new();
    let policy = BestFirst::new(semantics());
    let shared = rig
        .generator()
        .realize(&applied(&["E24", "K25"]))
        .expect("realize");
    rig.measurements
        .record(
            intent(EvidenceScale::Diagnostic).key_for(shared.physical_id()),
            observation(1.0e-3),
        )
        .expect("record");

    // Both routes to it are now pruned as already observed, from either
    // parent, with no second run scheduled.
    for parent in [["E24"], ["K25"]] {
        let set = rig.candidates(&parent, EvidenceScale::Diagnostic);
        assert_eq!(set.census().already_observed, 1, "from {parent:?}");
        assert!(policy
            .select(&set, &NothingMeasured)
            .opportunity()
            .map(|o| o.state != *shared.physical_id())
            .unwrap_or(true));
    }
}

// --------------------------------------------------------------- Ruling 3

#[test]
fn ruling_three_exhausted_sole_and_ranked() {
    let rig = Rig::new();
    let policy = BestFirst::new(semantics());

    // > 1 — the registered rule chooses which is measured FIRST.
    let many = policy.select(
        &rig.candidates(&[], EvidenceScale::Diagnostic),
        &NothingMeasured,
    );
    match &many {
        Selection::Ranked { chosen, considered } => {
            assert_eq!(*considered, 3);
            assert_eq!(chosen.leading().action.label, "+E24", "the biggest prize");
        }
        other => panic!("expected a ranked selection, got {other:?}"),
    }
    assert!(!many.is_exhausted());

    // 1 — SELECT it. Nothing to rank, so nothing may veto.
    let one_edit = Rig {
        vocabulary: ActionVocabulary::new([MapEdit::new("M26", compile("m_proj"))])
            .expect("one edit"),
        ..Rig::new()
    };
    let sole = policy.select(
        &one_edit.candidates(&[], EvidenceScale::Diagnostic),
        &NothingMeasured,
    );
    assert!(matches!(sole, Selection::Sole(_)));
    assert_eq!(sole.opportunity().expect("selected").routes(), 1);

    // 0 — the neighbourhood is closed.
    let exhausted = policy.select(
        &Rig {
            vocabulary: ActionVocabulary::default(),
            ..Rig::new()
        }
        .candidates(&[], EvidenceScale::Diagnostic),
        &NothingMeasured,
    );
    assert_eq!(exhausted, Selection::Exhausted);
    assert!(exhausted.is_exhausted());
    assert!(exhausted.opportunity().is_none());
}

// ----------------------------------------- assessment carries ingredients

#[test]
fn an_assessment_holds_ingredients_and_no_score_of_record() {
    let rig = Rig::new();
    let policy = BestFirst::new(semantics());
    let set = rig.candidates(&[], EvidenceScale::Diagnostic);
    let assessed = policy.order(policy.assess(&set, &NothingMeasured));
    let first = &assessed[0];

    // Exact, computed, and the sign convention is not flipped on the way
    // to the comparator.
    assert!(first.physical_delta < 0);
    assert_eq!(first.score(&semantics()).get(), first.physical_delta);
    assert!(first.child_bytes.get() > 0);
    assert_eq!(first.intended_key.scale(), EvidenceScale::Diagnostic);
    assert!(first.prior_observations.is_empty());
    assert!(!first.is_escalation());

    // The parent is unmeasured, and that is `None` rather than a zero.
    assert!(first.parent_standing.is_none());
}

#[test]
fn a_measured_parent_contributes_its_whole_standing_not_a_number() {
    // The binding margin, the headroom and which criterion is scarce are
    // all questions a reader may want; a scalar would answer none.
    let rig = Rig::new();
    let policy = BestFirst::new(semantics());
    let parent = rig.generator().realize(&applied(&[])).expect("realize");

    let mut measurements = MeasurementRegistry::new();
    measurements
        .record(
            intent(EvidenceScale::Authority).key_for(parent.physical_id()),
            observation(3.3532e-3),
        )
        .expect("record");
    let snapshot = SearchSnapshot::new(
        Objective::MinimiseLogicalBytes,
        kimi_logit_balanced_v1(),
        TailSupportPolicy::route_cal_1(),
        SearchSemantics::new("g/v1", "p/v1", "e/v1", "pr/v1", "rank/v1", "b/v1"),
        RepresentationStateGraph::new(TransitionPolicy::StrictlyImprovingPhysical, parent.clone()),
        measurements,
    );

    let set = rig.candidates(&[], EvidenceScale::Diagnostic);
    let assessed = policy.assess(&set, &snapshot);
    let standing = assessed[0]
        .parent_standing
        .as_ref()
        .expect("the parent was measured at authority");
    assert_eq!(standing.gate_id, "kimi-logit-balanced-v1");
    assert_eq!(
        standing.binding().expect("a ceiling scored").criterion,
        Criterion::KlP99
    );
    assert!(standing.admissible());
}

#[test]
fn a_diagnostic_reading_of_the_parent_is_not_its_standing() {
    // Authority and not "whatever was measured": a diagnostic reading
    // prices nothing against the contract, and handing one over as
    // though it did is the inference R5-F4 and R5-F9 closed.
    let rig = Rig::new();
    let policy = BestFirst::new(semantics());
    let parent = rig.generator().realize(&applied(&[])).expect("realize");

    let mut measurements = MeasurementRegistry::new();
    measurements
        .record(
            intent(EvidenceScale::Diagnostic).key_for(parent.physical_id()),
            observation(3.3532e-3),
        )
        .expect("record");
    let snapshot = SearchSnapshot::new(
        Objective::MinimiseLogicalBytes,
        kimi_logit_balanced_v1(),
        TailSupportPolicy::route_cal_1(),
        SearchSemantics::new("g/v1", "p/v1", "e/v1", "pr/v1", "rank/v1", "b/v1"),
        RepresentationStateGraph::new(TransitionPolicy::StrictlyImprovingPhysical, parent),
        measurements,
    );

    let assessed = policy.assess(&rig.candidates(&[], EvidenceScale::Diagnostic), &snapshot);
    assert!(
        assessed[0].parent_standing.is_none(),
        "a diagnostic reading is not a standing against the contract"
    );
}

#[test]
fn the_ranking_rule_is_named_in_the_search_semantics() {
    // A snapshot must be able to distinguish the same observations under
    // best-first-v1 from the same observations under v2: those can
    // legitimately select different experiments without anything
    // measured having changed.
    let v1 = SearchSemantics::new("g/v1", "p/v1", "e/v1", "pr/v1", "rank/v1", "b/v1");
    let v2 = SearchSemantics {
        ranking_rule: "rank/v2".into(),
        ..v1.clone()
    };
    assert_ne!(v1.id(), v2.id());
    assert_eq!(v1.ranking_rule, "rank/v1");
    assert_ne!(
        v1.ranking_rule, v1.promotion_rule,
        "priority and admissibility are different questions"
    );
}

// ---------------------------------------------------------------- helper

fn observation(kl_p99: f64) -> QualityBank {
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
