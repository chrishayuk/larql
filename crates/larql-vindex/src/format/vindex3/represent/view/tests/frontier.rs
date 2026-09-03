//! **The frontier, and that it is the optimiser's frontier.**

use super::super::super::measurement::EvidenceScale;
use super::super::super::quality::Criterion;
use super::{reloaded, view};

#[test]
fn every_state_and_every_verdict_is_the_substrates_own() {
    let snap = reloaded();
    let rendered = view(&snap).frontier();
    let substrate = snap.frontier();

    assert_eq!(rendered.states.len(), substrate.len());
    for (rendered, entry) in rendered.states.iter().zip(&substrate) {
        assert_eq!(rendered.state, entry.state);
        assert_eq!(rendered.logical_bytes, entry.logical_bytes);
        assert_eq!(rendered.admitted, entry.admitted());
        assert_eq!(rendered.refused, entry.refused());
        assert_eq!(rendered.adjudications.len(), entry.adjudications.len());
        for (rendered, adjudication) in rendered.adjudications.iter().zip(&entry.adjudications) {
            assert_eq!(&rendered.key, adjudication.key());
            assert_eq!(rendered.admissible, adjudication.admissible());
            assert_eq!(rendered.sound, adjudication.sound());
            assert_eq!(rendered.binding.as_ref(), adjudication.binding());
            assert_eq!(rendered.failures.len(), adjudication.failures().len());
        }
    }
}

#[test]
fn the_rung5_record_survives_the_render() {
    let snap = reloaded();
    let rendered = view(&snap).frontier();

    let standing = |bytes: u64| {
        rendered
            .states
            .iter()
            .find(|s| s.logical_bytes.get() == bytes)
            .unwrap_or_else(|| panic!("no state at {bytes} B"))
    };

    // P — admitted. K25 survives.
    let p = standing(13_684_764_800);
    assert!(p.admitted);
    assert!(!p.refused);
    assert!(p.adjudications[0].failures.is_empty());

    // T1 — refused on kl alone, by 1.48e-4.
    let t1 = standing(13_682_673_664);
    assert!(!t1.admitted);
    assert!(t1.refused);
    let failed: Vec<Criterion> = t1.adjudications[0]
        .failures
        .iter()
        .map(|m| m.criterion)
        .collect();
    assert_eq!(failed, vec![Criterion::KlP99]);
    assert_eq!(t1.adjudications[0].failures[0].observed, Some(3.6480e-3));
    assert_eq!(t1.adjudications[0].failures[0].limit, 3.5e-3);

    // S2 — refused.
    let s2 = standing(13_602_484_352);
    assert!(s2.refused);

    // S1 — never measured. Neither admitted nor refused, which is a
    // third state and not a rounding of one of the other two.
    let s1 = standing(13_600_393_216);
    assert!(!s1.admitted);
    assert!(!s1.refused);
    assert!(s1.adjudications.is_empty());
    assert!(s1.measured_at.is_empty());
}

#[test]
fn the_binding_constraint_is_re_derived_and_not_stored() {
    let snap = reloaded();
    let rendered = view(&snap).frontier();
    let p = rendered
        .states
        .iter()
        .find(|s| s.admitted)
        .expect("P is admitted");

    assert_eq!(
        p.adjudications[0].binding.as_ref().map(|m| m.criterion),
        Some(Criterion::KlP99),
        "the scarce resource, recomputed from the reading and the gate"
    );
}

#[test]
fn admitted_is_ordered_cheapest_first_by_the_objective() {
    let snap = reloaded();
    let rendered = view(&snap).frontier();
    let substrate: Vec<_> = snap.admitted().into_iter().map(|e| e.state).collect();
    let ours: Vec<_> = rendered.admitted.iter().map(|s| s.state.clone()).collect();

    assert_eq!(ours, substrate, "the order is the optimiser's");
    assert!(rendered.admitted.iter().all(|s| s.admitted));
}

#[test]
fn a_diagnostic_reading_appears_as_evidence_and_not_as_an_admission() {
    let snap = reloaded();
    let rendered = view(&snap).frontier();

    // The whole ladder rests on this: the record holds authority
    // readings only, so nothing is admitted on a short bank.
    for state in &rendered.states {
        for adjudication in &state.adjudications {
            assert_eq!(adjudication.key.scale(), EvidenceScale::Authority);
        }
        assert_eq!(
            state.admitted,
            state.adjudications.iter().any(|a| a.admissible)
        );
        assert!(!state.measured_at.contains(&EvidenceScale::Diagnostic));
    }
}
