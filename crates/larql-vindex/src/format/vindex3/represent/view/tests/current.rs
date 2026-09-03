//! **Where the search stands**, and that the incumbent is the
//! optimiser's pick rather than the facade's.

use super::super::super::measurement::EvidenceScale;
use super::super::super::state::fixtures;
use super::super::super::state::MeasurementRegistry;
use super::super::OptimizerView;
use super::{reloaded, view};

#[test]
fn the_graph_shape_is_read_and_not_recounted() {
    let snap = reloaded();
    let current = view(&snap).current();

    assert_eq!(&current.root, snap.graph().root());
    assert_eq!(current.states, snap.graph().len());
    assert_eq!(current.transitions, snap.graph().edge_count());
    assert_eq!(current.acyclic, snap.graph().is_acyclic());

    // The Rung 5 record: P and its three neighbours, one edge each.
    assert_eq!(current.states, 4);
    assert_eq!(current.transitions, 3);
    assert!(current.acyclic);
}

#[test]
fn the_incumbent_is_position_zero_of_the_optimisers_own_ordering() {
    let snap = reloaded();
    let current = view(&snap).current();

    let admitted = snap.admitted();
    assert_eq!(
        current.incumbent.as_ref().map(|s| &s.state),
        admitted.first().map(|e| &e.state)
    );
    assert_eq!(
        current.incumbent.as_ref().map(|s| s.logical_bytes.get()),
        Some(13_684_764_800),
        "P — the only admitted state in the record"
    );
    assert_eq!(current.admitted.len(), 1);
}

/// **Not the record.** Rung 5 deliberately never spent an authority run
/// on S1: it was dominated, and the protocol spends one run. This gives
/// it P's passing numbers anyway, purely so that the admitted set has
/// TWO members and an ordering claim has something to order.
///
/// With one admitted state, `first()` and `last()` are the same element
/// and a test that says "position zero" says nothing at all.
fn counterfactual_two_admitted() -> super::super::super::state::snapshot::SearchSnapshot {
    let base = reloaded();
    let mut measurements = base.measurements().clone();
    measurements
        .record(
            fixtures::key_for(&fixtures::s1(), EvidenceScale::Authority),
            fixtures::authority_reading(3.3532e-3, 1427),
        )
        .expect("record");
    fixtures::snapshot(base.graph().clone(), measurements)
}

#[test]
fn the_incumbent_is_the_cheapest_admitted_and_not_the_last_one_seen() {
    let snap = counterfactual_two_admitted();
    let current = OptimizerView::new(&snap).current();

    assert_eq!(current.admitted.len(), 2);
    assert_eq!(
        current
            .admitted
            .iter()
            .map(|s| s.logical_bytes.get())
            .collect::<Vec<_>>(),
        vec![13_600_393_216, 13_684_764_800],
        "cheapest first — the objective, applied by the optimiser"
    );
    assert_eq!(
        current.incumbent.as_ref().map(|s| s.logical_bytes.get()),
        Some(13_600_393_216),
        "position zero, which is now a different element from the last"
    );
}

#[test]
fn nothing_admitted_is_reported_as_nothing_and_not_as_a_best_guess() {
    // The same record with P's reading withheld: every state now
    // refused or unmeasured. A facade that fell back on "cheapest so
    // far" would name S1, which nobody ever measured.
    let base = reloaded();
    let mut measurements = MeasurementRegistry::new();
    for (state, kl, flips) in [
        (fixtures::t1(), 3.6480e-3, 1570),
        (fixtures::s2(), 4.0563e-3, 1309),
    ] {
        measurements
            .record(
                fixtures::key_for(&state, EvidenceScale::Authority),
                fixtures::authority_reading(kl, flips),
            )
            .expect("record");
    }
    let stripped = fixtures::snapshot(base.graph().clone(), measurements);

    let current = OptimizerView::new(&stripped).current();
    assert!(current.incumbent.is_none());
    assert!(current.admitted.is_empty());
    assert_eq!(current.states, 4, "the states are still all there");
    assert_eq!(
        current.unmeasured[1].states.len(),
        2,
        "P joins S1 in the dark at authority scale"
    );
}

#[test]
fn the_dark_states_are_reported_per_scale_and_left_unordered() {
    let snap = reloaded();
    let current = view(&snap).current();

    assert_eq!(
        current
            .unmeasured
            .iter()
            .map(|g| g.scale)
            .collect::<Vec<_>>(),
        EvidenceScale::ALL.to_vec(),
        "one gap per scale, cheapest evidence first"
    );
    for gap in &current.unmeasured {
        assert_eq!(gap.states, snap.unmeasured_at(gap.scale));
    }

    let diagnostic = &current.unmeasured[0];
    assert_eq!(diagnostic.scale, EvidenceScale::Diagnostic);
    assert_eq!(diagnostic.states.len(), 4, "nothing was measured short");

    let authority = &current.unmeasured[1];
    assert_eq!(authority.states.len(), 1, "S1 alone");
}

#[test]
fn every_scale_is_listed() {
    for scale in EvidenceScale::ALL {
        // Exhaustive on purpose: a new variant fails to compile here
        // rather than silently dropping out of every `measured_at`.
        match scale {
            EvidenceScale::Diagnostic | EvidenceScale::Authority => {}
        }
    }
    assert_eq!(EvidenceScale::ALL.len(), 2);
}
