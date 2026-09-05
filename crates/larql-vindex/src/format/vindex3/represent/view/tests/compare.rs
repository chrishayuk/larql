//! **Two states side by side**, with the physical difference and
//! nothing else.

use super::{reloaded, view};

#[test]
fn the_delta_is_the_difference_between_two_footprints() {
    let snap = reloaded();
    let edge = snap.graph().edges().next().expect("the record has edges");
    let (parent, child) = (edge.parent().clone(), edge.child().clone());
    let comparison = view(&snap).compare(&parent, &child).expect("both held");

    assert_eq!(
        comparison.physical_delta,
        comparison
            .right
            .logical_bytes
            .delta_from(comparison.left.logical_bytes)
    );
    assert_eq!(
        comparison.physical_delta,
        edge.physical_delta(),
        "and it agrees with the delta the graph computed when the move was made"
    );
    assert!(comparison.physical_delta < 0, "the child removes bytes");
}

#[test]
fn the_edge_is_rendered_when_one_exists_and_absent_when_none_does() {
    let snap = reloaded();
    let edge = snap.graph().edges().next().expect("edges");
    let (parent, child) = (edge.parent().clone(), edge.child().clone());

    let forward = view(&snap).compare(&parent, &child).expect("held");
    assert_eq!(forward.transition.as_ref(), Some(edge));

    // The graph is directed and the policy is strictly improving, so
    // there is no edge back. The comparison still stands; only its
    // provenance is gone.
    let backward = view(&snap).compare(&child, &parent).expect("held");
    assert!(backward.transition.is_none());
    assert_eq!(backward.physical_delta, -forward.physical_delta);
}

#[test]
fn two_states_with_no_edge_between_them_still_compare() {
    let snap = reloaded();
    let children: Vec<_> = snap.graph().edges().map(|e| e.child().clone()).collect();
    let comparison = view(&snap)
        .compare(&children[0], &children[1])
        .expect("both held");

    assert!(comparison.transition.is_none(), "siblings, not a move");
    assert_ne!(comparison.physical_delta, 0);
}

#[test]
fn a_state_the_record_does_not_hold_is_named_rather_than_hidden() {
    let snap = reloaded();
    let held = snap.graph().root().clone();
    let absent: super::super::super::state::RepresentationStateId =
        serde_json::from_str(r#""no-such-state""#).expect("an id");

    let refusal = view(&snap)
        .compare(&held, &absent)
        .expect_err("one of them is not held");
    assert_eq!(refusal.states, vec![absent.clone()]);

    let both = view(&snap)
        .compare(&absent, &absent)
        .expect_err("neither is held");
    assert_eq!(
        both.states.len(),
        2,
        "a caller is told which of its two ids was wrong, not that something was"
    );
}

#[test]
fn the_standings_are_the_frontiers_own() {
    let snap = reloaded();
    let edge = snap.graph().edges().next().expect("edges");
    let comparison = view(&snap)
        .compare(edge.parent(), edge.child())
        .expect("held");
    let frontier = view(&snap).frontier();

    let of = |id| {
        frontier
            .states
            .iter()
            .find(|s| &s.state == id)
            .expect("in the frontier")
    };
    assert_eq!(&comparison.left, of(edge.parent()));
    assert_eq!(&comparison.right, of(edge.child()));
}
