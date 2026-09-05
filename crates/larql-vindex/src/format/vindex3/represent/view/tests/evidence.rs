//! **The raw bank beside the verdict**, and the rules for reading both.

use super::super::super::measurement::EvidenceScale;
use super::super::super::state::fixtures;
use super::super::OptimizerView;
use super::{reloaded, view};

#[test]
fn every_observation_carries_the_reading_it_was_drawn_from() {
    let snap = reloaded();
    let report = view(&snap).evidence(None);

    assert_eq!(report.observations.len(), snap.measurements().len());
    assert_eq!(report.observations.len(), 3, "P, T1 and S2");
    for observation in &report.observations {
        assert_eq!(
            Some(&observation.bank),
            snap.measurements().get(&observation.key)
        );
        let adjudication = snap.adjudicate(&observation.key).expect("measured");
        assert_eq!(
            observation.adjudication.admissible,
            adjudication.admissible()
        );
        assert_eq!(observation.adjudication.sound, adjudication.sound());
    }
}

#[test]
fn the_recorded_arms_survive_the_render() {
    let snap = reloaded();
    let report = view(&snap).evidence(None);

    // A verdict read without the arms behind it inherits every
    // assumption of the classifier that produced it. The p99 and the
    // covered mass are the recorded values.
    let kls: Vec<f64> = report
        .observations
        .iter()
        .map(|o| o.bank.logits.kl_p99)
        .collect();
    assert!(kls.contains(&3.3532e-3));
    assert!(kls.contains(&3.6480e-3));
    assert!(kls.contains(&4.0563e-3));

    for observation in &report.observations {
        assert_eq!(observation.bank.positions, 8192);
        assert_eq!(observation.bank.min_covered_mass, Some(0.6315));
    }
}

#[test]
fn narrowing_to_a_state_returns_that_states_experiments_and_no_others() {
    let snap = reloaded();
    let state = snap.graph().root().clone();
    let report = view(&snap).evidence(Some(&state));

    assert_eq!(report.of_state.as_ref(), Some(&state));
    assert_eq!(report.observations.len(), 1);
    assert!(report.observations.iter().all(|o| o.key.state() == &state));
}

#[test]
fn a_state_with_no_reading_returns_an_empty_record_and_not_a_default() {
    let snap = reloaded();
    let unmeasured = snap
        .unmeasured_at(EvidenceScale::Authority)
        .first()
        .expect("S1")
        .clone();
    let report = view(&snap).evidence(Some(&unmeasured));

    assert!(report.observations.is_empty());
    assert_eq!(report.of_state, Some(unmeasured));
    // The rules for reading the record are still there: what is absent
    // is the evidence, not the contract it would be judged by.
    assert_eq!(&report.tail_support, snap.tail_support());
}

#[test]
fn a_diagnostic_reading_is_evidence_and_is_not_an_admission() {
    let snap = fixtures::rung5_with_diagnostic_on_s1();
    let facade = OptimizerView::new(&snap);
    let s1 = fixtures::s1().physical_id().clone();

    let observations = facade.evidence(Some(&s1)).observations;
    assert_eq!(observations.len(), 1);
    assert_eq!(observations[0].key.scale(), EvidenceScale::Diagnostic);
    assert!(
        observations[0].adjudication.admissible,
        "it does clear the contract"
    );

    let standing = facade
        .frontier()
        .states
        .into_iter()
        .find(|s| s.state == s1)
        .expect("in the frontier");
    assert!(!standing.admitted, "and it is still not an admission");
    assert!(!standing.refused);
    assert_eq!(standing.measured_at, vec![EvidenceScale::Diagnostic]);
    assert!(
        facade.current().incumbent.is_none_or(|i| i.state != s1),
        "a diagnostic pass must never make a state the incumbent"
    );
}

#[test]
fn the_calibrations_travel_with_the_observations_they_qualify() {
    let snap = reloaded();
    let report = view(&snap).evidence(None);

    assert_eq!(&report.calibrations, &snap.config().calibrations);
    assert_eq!(&report.diagnostic_policy, &snap.config().diagnostic_policy);
}
