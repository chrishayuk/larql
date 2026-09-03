//! **State identity separated from discovery path.**

use super::{reloaded, view};

#[test]
fn a_state_the_graph_does_not_hold_is_a_miss_and_not_an_error() {
    let snap = reloaded();
    let absent = snap
        .frontier()
        .first()
        .expect("the record has states")
        .state
        .clone();
    assert!(view(&snap).explain(&absent).is_some());

    let made_up = serde_json::from_str(r#""not-a-state-this-graph-holds""#).expect("an id");
    assert!(
        view(&snap).explain(&made_up).is_none(),
        "a fact about the record, reported as one"
    );
}

#[test]
fn identity_and_discovery_are_answered_separately() {
    let snap = reloaded();
    let edge = snap.graph().edges().next().expect("the record has edges");
    let child = edge.child().clone();
    let explained = view(&snap).explain(&child).expect("held");

    // Identity: what it IS.
    assert_eq!(explained.state, child);
    assert_eq!(
        explained.logical_bytes,
        snap.graph().node(&child).expect("held").logical_bytes()
    );
    assert_eq!(explained.realizations.len(), 1);

    // Discovery: how it was REACHED — on the incoming edge, never on
    // the node, so one state can carry several explanations.
    assert_eq!(
        explained.reached_by.len(),
        snap.graph().incoming(&child).len()
    );
    assert_eq!(explained.reached_by[0].parent(), edge.parent());
    assert_eq!(explained.reached_by[0].action(), edge.action());
    assert_eq!(
        explained.reached_by[0].physical_delta(),
        edge.physical_delta()
    );
    assert!(
        explained.leads_to.is_empty(),
        "the three children are leaves"
    );
}

#[test]
fn the_root_is_reached_by_nothing_and_leads_to_everything() {
    let snap = reloaded();
    let root = snap.graph().root().clone();
    let explained = view(&snap).explain(&root).expect("held");

    assert!(
        explained.reached_by.is_empty(),
        "no edge arrives at the root, and that is not a missing provenance"
    );
    assert_eq!(explained.leads_to.len(), 3);
    for edge in &explained.leads_to {
        assert_eq!(edge.parent(), &root);
        assert!(
            edge.physical_delta() < 0,
            "every edge strictly reduces bytes under the declared policy"
        );
    }
}

#[test]
fn the_provenance_on_an_edge_is_carried_whole() {
    let snap = reloaded();
    let root = snap.graph().root().clone();
    let explained = view(&snap).explain(&root).expect("held");

    let who: Vec<&str> = explained
        .leads_to
        .iter()
        .flat_map(|e| e.provenance())
        .map(|p| p.by.as_str())
        .collect();
    assert!(who.contains(&"rung5/N1"));
    assert!(who.contains(&"rung5/N2"));
}

#[test]
fn a_state_with_no_ledger_reports_none_rather_than_a_zero() {
    let snap = reloaded();
    let root = snap.graph().root().clone();
    let explained = view(&snap).explain(&root).expect("held");

    // The record holds no per-token ledger for P. A zero here would
    // price a candidate against a decoder that reads nothing.
    assert!(explained.ledger.is_none());
    assert_eq!(explained.ledger, snap.ledger(&root).cloned());
}
