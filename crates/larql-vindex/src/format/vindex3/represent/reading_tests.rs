//! MEASURE-PLAN-2 PR 1 witnesses: W1–W3, W9, and the adjudication half
//! of W8 (a characterisation-only record judges nothing).

use super::super::constraint::ConstraintVector;
use super::super::measure::plan::metrics::Aggregate;
use super::super::quality::{kimi_logit_balanced_v1, Criterion, Statistic};
use super::super::state::fixtures;
use super::super::state::instrument::{
    Direction, InstrumentSemantics, MetricSemantics, Quantity, Support, Unit,
};
use super::super::state::snapshot::SearchSnapshot;
use super::*;

fn aggregate(positions: usize, kl_p99: f64, delta_nll_mean: Option<f64>) -> Aggregate {
    Aggregate {
        positions,
        kl_mean: kl_p99 / 4.0,
        kl_p50: kl_p99 / 8.0,
        kl_p99,
        kl_max: kl_p99 * 2.0,
        top1_agreement: 0.9,
        top5_overlap_mean: 0.95,
        delta_nll_mean,
        max_abs_delta_mean: 0.5,
        max_abs_delta_p99: 1.5,
    }
}

fn plan_reading(kl_p99: f64) -> PlanObservation {
    PlanObservation {
        procedure: PlanProcedure,
        sequences: 4,
        positions: 128,
        all: aggregate(128, kl_p99, Some(0.01)),
        by_category: vec![("prose".into(), aggregate(128, kl_p99, Some(0.01)))],
        by_margin_band: vec![((0.0, 0.1), aggregate(128, kl_p99, Some(0.01)))],
    }
}

fn plan_gate(semantics: MetricSemantics) -> PlanGate {
    PlanGate {
        semantics,
        ..test_plan_gate()
    }
}

fn plan_instrument() -> InstrumentSemantics {
    InstrumentSemantics::new(
        "kl(reference || candidate), nats",
        "mean, p50, p99, max",
        "teacher-forced, every position",
        PLAN_PROCEDURE,
    )
    .with_semantics(MetricSemantics::PLAN_V1)
    .unwrap()
}

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/measure_plan_2");

// ------------------------------------------------------------------ W1

/// Snapshots written by main's code (before these kinds existed) read
/// back and re-serialise byte for byte.
#[test]
fn w1_records_written_before_the_kinds_round_trip_byte_identically() {
    for (file, pretty) in [
        ("rung5-snapshot.json", true),
        ("reloaded-snapshot.json", false),
    ] {
        let bytes = std::fs::read(format!("{GOLDEN}/{file}")).unwrap();
        let snapshot: SearchSnapshot = serde_json::from_slice(&bytes).unwrap();
        let again = if pretty {
            serde_json::to_vec_pretty(&snapshot).unwrap()
        } else {
            serde_json::to_vec(&snapshot).unwrap()
        };
        assert_eq!(again, bytes, "{file} changed on a round trip");
        assert_eq!(snapshot.gate().map(Gate::kind), Some(ReadingKind::Kimi));
        for key in snapshot.measurements().keys() {
            let reading = snapshot.measurements().get(key).unwrap();
            assert_eq!(reading.kind(), ReadingKind::Kimi, "{file}");
        }
    }
}

#[test]
fn w1_a_kimi_reading_and_gate_serialise_as_their_bare_structs() {
    let bank = fixtures::authority_reading(2.5e-3, 3);
    assert_eq!(
        serde_json::to_vec(&Observation::from(bank.clone())).unwrap(),
        serde_json::to_vec(&bank).unwrap()
    );
    let gate = kimi_logit_balanced_v1();
    assert_eq!(
        serde_json::to_vec(&Gate::from(gate.clone())).unwrap(),
        serde_json::to_vec(&gate).unwrap()
    );
    let read: Observation = serde_json::from_slice(&serde_json::to_vec(&bank).unwrap()).unwrap();
    assert_eq!(read, Observation::from(bank));
    let read: Gate = serde_json::from_slice(&serde_json::to_vec(&gate).unwrap()).unwrap();
    assert_eq!(read, Gate::Kimi(gate));
}

// ------------------------------------------------------------------ W2

#[test]
fn w2_a_gate_judges_only_its_own_instrument() {
    let kimi_gate = Gate::from(kimi_logit_balanced_v1());
    let plan = Observation::from(plan_reading(1e-4));
    let err = ConstraintVector::judge(&kimi_gate, &plan).unwrap_err();
    assert_eq!(err.gate_kind, ReadingKind::Kimi);
    assert_eq!(err.reading_kind, ReadingKind::Plan);

    let plan_gate = Gate::from(test_plan_gate());
    let kimi = Observation::from(fixtures::authority_reading(1e-4, 0));
    assert!(ConstraintVector::judge(&plan_gate, &kimi).is_err());

    // Same kinds judge; the Kimi arm is exactly `of`.
    let bank = fixtures::authority_reading(1e-4, 0);
    assert_eq!(
        ConstraintVector::judge(&kimi_gate, &Observation::from(bank.clone())).unwrap(),
        ConstraintVector::of(&kimi_logit_balanced_v1(), &bank)
    );
    assert!(ConstraintVector::judge(&plan_gate, &plan).is_ok());
}

// ------------------------------------------------------------------ W3

#[test]
fn w3_a_plan_reading_is_only_ever_a_plan_reading() {
    let plan = Observation::from(plan_reading(1e-4));
    let json = serde_json::to_value(&plan).unwrap();
    let back: Observation = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(back, plan);

    let mut stripped = json.clone();
    stripped.as_object_mut().unwrap().remove("procedure");
    assert!(serde_json::from_value::<Observation>(stripped).is_err());

    let mut renamed = json;
    renamed["procedure"] = "teacher-forced-two-arm/v1".into();
    assert!(serde_json::from_value::<Observation>(renamed).is_err());
}

#[test]
fn w3_absent_statistics_are_absent_never_zero() {
    let plan = Observation::from(plan_reading(1e-4));
    for kimi_only in [
        Statistic::KlP99,
        Statistic::Top1Flips,
        Statistic::Top10Changes,
        Statistic::RouteFlips,
        Statistic::CoveredMass,
        Statistic::Top10MassDisplacedP99,
    ] {
        assert_eq!(
            kimi_only.observe_reading(&plan),
            (None, None),
            "{kimi_only}"
        );
    }
    let kimi = Observation::from(fixtures::authority_reading(1e-4, 0));
    for plan_only in [
        Statistic::PlanKlP99,
        Statistic::PlanKlMean,
        Statistic::PlanTop1Disagreement,
        Statistic::PlanDeltaNllMean,
    ] {
        assert_eq!(
            plan_only.observe_reading(&kimi),
            (None, None),
            "{plan_only}"
        );
    }
    // A plan gate reads plan criteria only.
    let margins = ConstraintVector::of_plan(&test_plan_gate(), &plan_reading(1e-4)).margins;
    assert!(margins.iter().all(|m| matches!(
        m.criterion,
        Criterion::PlanKlP99
            | Criterion::PlanKlMean
            | Criterion::PlanTop1Disagreement
            | Criterion::PlanDeltaNllMean
            | Criterion::Positions
    )));
    // A limit on a statistic the run could not produce is unmet.
    let mut gate = test_plan_gate();
    gate.delta_nll_mean_max = Some(1.0);
    let mut reading = plan_reading(1e-4);
    reading.all.delta_nll_mean = None;
    let v = ConstraintVector::of_plan(&gate, &reading);
    let m = v
        .margins
        .iter()
        .find(|m| m.criterion == Criterion::PlanDeltaNllMean)
        .unwrap();
    assert!(!m.satisfied());
    assert!(!v.admissible());
}

// ------------------------------------------------------------------ W8

#[test]
fn w8_a_characterisation_only_record_adjudicates_nothing() {
    let snapshot = fixtures::rung5_snapshot();
    assert!(snapshot.measurements().keys().next().is_some());
    let mut config = snapshot.config().clone();
    config.gate = None;
    let bare = SearchSnapshot::new(snapshot.space().clone(), config, snapshot.facts().clone());
    assert!(bare.gate().is_none());
    for key in bare.measurements().keys() {
        assert!(bare.adjudicate(key).is_none());
        assert!(
            bare.measurements().get(key).is_some(),
            "the reading is held"
        );
    }
    assert!(bare.frontier().iter().all(|e| e.adjudications.is_empty()));
    assert!(bare.admitted().is_empty());
    let bytes = serde_json::to_vec(&bare).unwrap();
    let back: SearchSnapshot = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(back, bare, "a gate-less record replays");
}

// ------------------------------------------------------------------ W9

#[test]
fn w9_a_plan_gate_binds_to_its_metric_semantics() {
    let full = plan_instrument();
    assert!(check_binding(&Gate::from(test_plan_gate()), ReadingKind::Plan, &full).is_ok());

    // A gate written for a top-2048 KL refuses full-vocabulary readings.
    let top_n = Gate::from(plan_gate(MetricSemantics {
        support: Support::TopN(2048),
        ..MetricSemantics::PLAN_V1
    }));
    assert!(matches!(
        check_binding(&top_n, ReadingKind::Plan, &full),
        Err(GateBindingRefusal::Semantics { .. })
    ));
    // An instrument declaring no structure cannot satisfy a plan gate.
    let mut undeclared = full.clone();
    undeclared.semantics = None;
    assert!(matches!(
        check_binding(
            &Gate::from(test_plan_gate()),
            ReadingKind::Plan,
            &undeclared
        ),
        Err(GateBindingRefusal::Semantics {
            instrument: None,
            ..
        })
    ));
    // Kind still binds first.
    assert!(matches!(
        check_binding(&Gate::from(test_plan_gate()), ReadingKind::Kimi, &full),
        Err(GateBindingRefusal::Kind(_))
    ));
}

#[test]
fn w9_an_instrument_whose_support_contradicts_its_truncation_is_refused() {
    let semantics = MetricSemantics {
        quantity: Quantity::KlDivergence,
        unit: Unit::Nats,
        support: Support::TopN(2048),
        direction: Direction::ReferenceToCandidate,
    };
    let untruncated = InstrumentSemantics::new("kl", "p99", "all", PLAN_PROCEDURE);
    assert!(untruncated.clone().with_semantics(semantics).is_err());
    assert!(untruncated
        .clone()
        .truncated_to(2048)
        .with_semantics(semantics)
        .is_ok());
    assert!(untruncated
        .truncated_to(1024)
        .with_semantics(semantics)
        .is_err());
    // Refused at the binding too, whatever the gate.
    let mut inconsistent = plan_instrument();
    inconsistent.truncation = Some(2048);
    assert!(matches!(
        check_binding(
            &Gate::from(test_plan_gate()),
            ReadingKind::Plan,
            &inconsistent
        ),
        Err(GateBindingRefusal::Instrument(_))
    ));
}

#[test]
fn w9_every_existing_kimi_instrument_id_is_unchanged() {
    // The golden record was written by main before `semantics` existed;
    // its standing intent names the fixture instrument's id then.
    let bytes = std::fs::read(format!("{GOLDEN}/rung5-snapshot.json")).unwrap();
    let snapshot: SearchSnapshot = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        snapshot.standing_intent().instrument,
        fixtures::instrument().id()
    );
    // Declaring semantics DOES move an id: the field is in the digest.
    let declared = fixtures::instrument()
        .with_semantics(MetricSemantics {
            support: Support::TopN(2048),
            ..MetricSemantics::PLAN_V1
        })
        .unwrap();
    assert_ne!(declared.id(), fixtures::instrument().id());
}

#[test]
fn w3_a_plan_gate_is_only_ever_a_plan_gate() {
    let gate = Gate::from(test_plan_gate());
    let json = serde_json::to_value(&gate).unwrap();
    let back: Gate = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(back, gate, "a plan gate must not read back as a Kimi one");
    assert_eq!(back.kind(), ReadingKind::Plan);
}
