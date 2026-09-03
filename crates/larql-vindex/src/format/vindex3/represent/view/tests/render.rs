//! **The facade as a whole: seven questions, no writers, one answer.**
//!
//! The per-view tests check that each response is the substrate's own.
//! These check the properties of the SURFACE — what it exposes, that it
//! survives the wire, and that asking twice gives the same answer.

use std::collections::BTreeSet;

use super::{reloaded, view};

/// The facade's own source, so the surface is checked against the code
/// rather than against a list someone maintained by hand.
const FACADE: &str = include_str!("../../view.rs");

/// Every public method [`super::super::OptimizerView`] is allowed to
/// have. Seven questions and the constructor.
const ALLOWED: [&str; 8] = [
    "new",
    "describe",
    "current",
    "frontier",
    "explain",
    "compare",
    "evidence",
    "next_experiment",
];

fn facade_methods() -> BTreeSet<String> {
    let body = FACADE
        .split_once("impl<'a> OptimizerView<'a> {")
        .expect("the facade impl block")
        .1;
    body.lines()
        .filter_map(|line| line.trim().strip_prefix("pub fn "))
        .filter_map(|rest| rest.split(['(', '<']).next())
        .map(str::to_string)
        .collect()
}

#[test]
fn the_facade_exposes_seven_questions_and_nothing_else() {
    let methods = facade_methods();
    let allowed: BTreeSet<String> = ALLOWED.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        methods, allowed,
        "the read surface changed; a new method needs a new test, not a new alibi"
    );
}

#[test]
fn no_method_writes_records_promotes_or_accepts() {
    // The tools that must not exist. Not "not yet": the optimiser and
    // the evidence system decide what is true, and an agent gets no
    // vote on what an answer means.
    for forbidden in [
        "record", "apply", "expand", "promote", "accept", "set", "add", "remove", "update",
        "measure", "search",
    ] {
        assert!(
            !facade_methods().iter().any(|m| m.contains(forbidden)),
            "the facade grew a `{forbidden}` method"
        );
    }
}

#[test]
fn the_check_would_notice_a_writer() {
    // The list above proves nothing unless it can fail. This is the
    // same parse against a facade that did grow one.
    let smuggled = "impl<'a> OptimizerView<'a> {\n    pub fn accept_candidate(&self) {}\n}";
    let found: Vec<&str> = smuggled
        .split_once("impl<'a> OptimizerView<'a> {")
        .expect("block")
        .1
        .lines()
        .filter_map(|line| line.trim().strip_prefix("pub fn "))
        .filter_map(|rest| rest.split(['(', '<']).next())
        .collect();
    assert_eq!(found, vec!["accept_candidate"]);
}

#[test]
fn every_response_survives_the_wire() {
    let snap = reloaded();
    let facade = view(&snap);
    let state = snap.graph().root().clone();
    let edge = snap.graph().edges().next().expect("edges");

    // Stage 3b found `Role::deserialize` reading a BORROWED string
    // only: fine under `from_str`, broken under `from_value`. A
    // transport goes through `Value`, so every response is put through
    // both.
    let responses = [
        serde_json::to_value(facade.describe()).expect("describe"),
        serde_json::to_value(facade.current()).expect("current"),
        serde_json::to_value(facade.frontier()).expect("frontier"),
        serde_json::to_value(facade.explain(&state)).expect("explain"),
        serde_json::to_value(facade.compare(edge.parent(), edge.child())).expect("compare"),
        serde_json::to_value(facade.evidence(None)).expect("evidence"),
        serde_json::to_value(facade.next_experiment()).expect("next_experiment"),
    ];
    for response in responses {
        let text = serde_json::to_string(&response).expect("string");
        let back: serde_json::Value = serde_json::from_str(&text).expect("round trip");
        assert_eq!(back, response);
        assert!(!text.is_empty());
    }
}

#[test]
fn asking_twice_gives_the_same_answer() {
    let snap = reloaded();
    let facade = view(&snap);
    let state = snap.graph().root().clone();

    // Replay needs one answer, not a usually-stable one. Map iteration
    // order is the classic way a transport starts disagreeing with
    // itself between two identical requests.
    for _ in 0..4 {
        assert_eq!(
            serde_json::to_string(&facade.frontier()).expect("json"),
            serde_json::to_string(&view(&snap).frontier()).expect("json")
        );
        assert_eq!(
            serde_json::to_string(&facade.evidence(Some(&state))).expect("json"),
            serde_json::to_string(&view(&snap).evidence(Some(&state))).expect("json")
        );
        assert_eq!(
            serde_json::to_string(&facade.describe()).expect("json"),
            serde_json::to_string(&view(&snap).describe()).expect("json")
        );
    }
}

#[test]
fn a_reloaded_record_renders_identically_to_the_one_in_memory() {
    let built = super::super::super::state::fixtures::rung5_snapshot();
    let stored = reloaded();

    // The facade reads facts, so storage must not change what it says.
    assert_eq!(
        serde_json::to_string(&view(&built).frontier()).expect("json"),
        serde_json::to_string(&view(&stored).frontier()).expect("json")
    );
    assert_eq!(
        serde_json::to_string(&view(&built).describe()).expect("json"),
        serde_json::to_string(&view(&stored).describe()).expect("json")
    );
}
