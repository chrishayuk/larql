//! **1d: the replay gate.**
//!
//! > Given a root realization, the search semantics, the contract and
//! > the recorded measurements — and WITHOUT any recorded ranking,
//! > pruning decision, promotion or frontier — the optimiser shall
//! > reproduce the registered Rung 5 conclusion.
//!
//! The numbers below are the real ones. Rung 5's neighbourhoods 1 and 2,
//! measured at 8192 positions on the selection bank and judged by the
//! frozen `kimi-logit-balanced-v1`:
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
//! their limits, so that what the replay turns on is the recorded
//! evidence and not a number invented here — and so that `binding()`
//! reproducing `KlP99` means something.

use super::super::measurement::{EvidenceScale, TailSupportPolicy};
use super::super::quality::Criterion;
use super::fixtures::*;
use super::*;

// -------------------------------------------------------- the replay gate

#[test]
fn the_rung5_conclusion_is_re_derived_from_stored_facts_alone() {
    let snap = reloaded();

    // P — admitted. The recorded reading clears every criterion.
    let p_verdict = snap
        .adjudicate(&key_for(&p(), EvidenceScale::Authority))
        .expect("P was measured at authority");
    assert!(p_verdict.admissible(), "K25 survives");
    assert!(
        p_verdict.sound(),
        "the floors cleared, so the instrument saw"
    );
    assert!(p_verdict.failures().is_empty());

    // T1 — refused, on kl alone, by 1.48e-4. The ledger's
    // `balanced-v1 FAIL ['kl_p99: 3.648e-3 > 3.500e-3']`.
    let t1_verdict = snap
        .adjudicate(&key_for(&t1(), EvidenceScale::Authority))
        .expect("T1 was measured");
    assert!(!t1_verdict.admissible());
    let failed: Vec<Criterion> = t1_verdict.failures().iter().map(|m| m.criterion).collect();
    assert_eq!(failed, vec![Criterion::KlP99], "kl alone");
    let over = t1_verdict.failures()[0];
    assert!((over.observed.expect("observed") - 3.6480e-3).abs() < 1e-12);
    assert!((over.limit - 3.5e-3).abs() < 1e-12);

    // S2 — refused, and further out than T1.
    let s2_verdict = snap
        .adjudicate(&key_for(&s2(), EvidenceScale::Authority))
        .expect("S2 was measured");
    assert!(!s2_verdict.admissible());
    assert!(
        s2_verdict.constraints().utilisation_of(Criterion::KlP99)
            > t1_verdict.constraints().utilisation_of(Criterion::KlP99),
        "S2 blew through where T1 grazed"
    );

    // KL is the binding constraint on the admitted parent — the fact the
    // whole exchange rung exists because of.
    assert_eq!(
        p_verdict.binding().expect("a ceiling was scored").criterion,
        Criterion::KlP99
    );

    // S1 was never measured. Dominated, and the protocol spends ONE run:
    // a MISS is a fact about the record, not a failure.
    assert!(snap
        .adjudicate(&key_for(&s1(), EvidenceScale::Authority))
        .is_none());
}

#[test]
fn the_admitted_set_and_the_frontier_are_re_derived_cheapest_first() {
    let snap = reloaded();

    let admitted = snap.admitted();
    assert_eq!(admitted.len(), 1, "one map survives");
    assert_eq!(&admitted[0].state, p().physical_id());
    assert_eq!(admitted[0].logical_bytes, LogicalBytes::new(13_684_764_800));

    let frontier = snap.frontier();
    assert_eq!(frontier.len(), 4, "P, T1, S2, S1");
    let refused: Vec<&FrontierEntry> = frontier.iter().filter(|e| e.refused()).collect();
    assert_eq!(refused.len(), 2, "T1 and S2");
    assert!(refused
        .iter()
        .all(|e| !e.admitted() && e.measured_at(EvidenceScale::Authority)));

    // Nothing was measured diagnostically in this record, and the
    // snapshot says so rather than guessing.
    let mut unmeasured = snap.unmeasured_at(EvidenceScale::Diagnostic);
    unmeasured.sort();
    assert_eq!(unmeasured.len(), 4);
    assert_eq!(
        snap.unmeasured_at(EvidenceScale::Authority),
        vec![s1().physical_id().clone()],
        "only S1 lacks an authority reading"
    );
}

#[test]
fn the_objective_orders_the_admitted_set_and_ties_break_deterministically() {
    // Two survivors: `MinimiseLogicalBytes` means the cheaper one leads.
    // The tie-break is the state id and not insertion order, because a
    // scientific record must not depend on which map was recorded first.
    let mut graph = RepresentationStateGraph::new(TransitionPolicy::StrictlyImprovingPhysical, p());
    graph
        .apply(
            p().physical_id(),
            Action::new("−M26 +H"),
            s1(),
            Provenance::new("rung5/N1"),
        )
        .expect("lighter");

    let mut measurements = MeasurementRegistry::new();
    for s in [p(), s1()] {
        measurements
            .record(
                key_for(&s, EvidenceScale::Authority),
                authority_reading(3.3532e-3, 1427),
            )
            .expect("record");
    }
    let snap = snapshot(graph, measurements);

    let admitted = snap.admitted();
    assert_eq!(admitted.len(), 2);
    assert_eq!(
        admitted[0].logical_bytes,
        LogicalBytes::new(13_600_393_216),
        "cheapest first"
    );
    assert_eq!(&admitted[0].state, s1().physical_id());
    assert_eq!(&admitted[1].state, p().physical_id());
}

#[test]
fn an_authority_pass_admits_and_a_diagnostic_pass_does_not() {
    // The ladder rests on this: a diagnostic reading that clears the
    // contract is not an admission.
    let mut measurements = MeasurementRegistry::new();
    measurements
        .record(
            key_for(&p(), EvidenceScale::Diagnostic),
            authority_reading(3.3532e-3, 1427),
        )
        .expect("record");
    let snap = snapshot(
        RepresentationStateGraph::new(TransitionPolicy::StrictlyImprovingPhysical, p()),
        measurements,
    );

    let verdict = snap
        .adjudicate(&key_for(&p(), EvidenceScale::Diagnostic))
        .expect("measured");
    assert!(verdict.admissible(), "it does clear the contract");
    assert!(
        snap.admitted().is_empty(),
        "and it is still not an admission"
    );
}

// ------------------------------------------------- no conclusion is stored

#[test]
fn the_stored_form_carries_no_conclusion() {
    // The anti-cheat. If a verdict, a rank or a frontier were persisted,
    // the replay above would prove serialisation rather than derivation.
    let json = serde_json::to_value(rung5_snapshot()).expect("serialize");

    const FORBIDDEN: &[&str] = &[
        "admissible",
        "admitted",
        "refused",
        "binding",
        "frontier",
        "verdict",
        "promotion",
        "rank",
        "chosen",
        "best",
        "recommendation",
        "failures",
        "passed",
        "adjudication",
    ];

    fn walk(value: &serde_json::Value, path: &str, found: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    if FORBIDDEN.contains(&k.as_str()) {
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

    // And the facts ARE there, so the emptiness above is not vacuous.
    assert_eq!(json["schema"], SNAPSHOT_SCHEMA);
    assert_eq!(json["config"]["objective"], "MinimiseLogicalBytes");
    assert_eq!(json["config"]["gate"]["id"], "kimi-logit-balanced-v1");
    assert_eq!(
        json["facts"]["graph"]["policy"],
        "StrictlyImprovingPhysical"
    );
    assert_eq!(
        json["facts"]["graph"]["nodes"]
            .as_object()
            .expect("nodes")
            .len(),
        4
    );
    assert_eq!(
        json["facts"]["measurements"]["observations"]
            .as_array()
            .expect("observations")
            .len(),
        3
    );
    assert!(json["config"]["semantics"]["promotion_rule"].is_string());
}

// ------------------------------------------------- decision-procedure drift

#[test]
fn the_semantics_a_conclusion_was_drawn_under_travel_with_it() {
    // Six months from now `decide_promotion` legitimately changes and an
    // old snapshot replays differently. That is not corruption — the
    // decision procedure changed — and a replay must be able to say so.
    let snap = reloaded();
    assert_eq!(snap.semantics_id(), semantics().id());

    let new_promotion = SearchSemantics {
        promotion_rule: "decide-promotion-ordinal/v2".into(),
        ..semantics()
    };
    assert_ne!(
        new_promotion.id(),
        semantics().id(),
        "a changed rule is visible"
    );

    // Every field is normative; none of them is a source hash.
    for changed in [
        SearchSemantics {
            candidate_generation: "beam/v1".into(),
            ..semantics()
        },
        SearchSemantics {
            pre_measurement_pruning: "ruling-1-plus-monotonicity/v2".into(),
            ..semantics()
        },
        SearchSemantics {
            evidence_interpretation: "search-evidence-ladder/v2".into(),
            ..semantics()
        },
        SearchSemantics {
            physical_accounting: "bytes-per-token/v1".into(),
            ..semantics()
        },
        SearchSemantics {
            ranking_rule: "beam-width-4/v1".into(),
            ..semantics()
        },
    ] {
        assert_ne!(changed.id(), semantics().id());
    }
    assert!(SEARCH_SEMANTICS_ID_VERSION.starts_with("search-semantics-id/"));
    assert_eq!(semantics().id().short().len(), 12);
    assert_eq!(format!("{}", semantics().id()), semantics().id().as_str());
}

#[test]
fn a_snapshot_written_under_another_schema_is_refused() {
    let mut json = serde_json::to_value(rung5_snapshot()).expect("serialize");
    json["schema"] = serde_json::Value::String("represent-search-snapshot/v0".into());
    let stale: SearchSnapshot = serde_json::from_value(json).expect("still parses");
    let err = stale.check_schema().expect_err("stale schema");
    assert!(format!("{err}").contains("recognisably stale"), "{err}");
    assert_eq!(stale.schema(), "represent-search-snapshot/v0");
}

#[test]
fn a_snapshot_names_the_configuration_its_conclusions_depend_on() {
    let snap = reloaded();
    assert_eq!(snap.objective(), Objective::MinimiseLogicalBytes);
    assert_eq!(snap.gate().id, "kimi-logit-balanced-v1");
    assert_eq!(snap.gate().kl_p99_max, 3.5e-3);
    assert_eq!(
        snap.tail_support().min_tail_observations,
        TailSupportPolicy::route_cal_1().min_tail_observations
    );
    assert_eq!(
        snap.semantics().promotion_rule,
        "decide-promotion-ordinal/v1"
    );
    assert_eq!(snap.graph().len(), 4);
    assert_eq!(snap.measurements().len(), 3);
    assert_eq!(snap.schema(), SNAPSHOT_SCHEMA);

    // A verdict belongs to an EXPERIMENT, so an adjudication names the
    // key it came from.
    let k = key_for(&p(), EvidenceScale::Authority);
    assert_eq!(snap.adjudicate(&k).expect("measured").key(), &k);
}

#[test]
fn a_measurement_of_a_state_this_graph_does_not_hold_is_not_this_frontier() {
    // Registries outlive graphs. An observation of a state some other
    // search built must not appear here as an admitted map.
    let mut measurements = MeasurementRegistry::new();
    measurements
        .record(
            key_for(&s2(), EvidenceScale::Authority),
            authority_reading(1.0e-3, 10),
        )
        .expect("record");
    let snap = snapshot(
        RepresentationStateGraph::new(TransitionPolicy::StrictlyImprovingPhysical, p()),
        measurements,
    );

    assert_eq!(snap.frontier().len(), 1, "only the root is in the graph");
    assert!(
        snap.admitted().is_empty(),
        "a passing reading of a foreign state admits nothing"
    );
}
