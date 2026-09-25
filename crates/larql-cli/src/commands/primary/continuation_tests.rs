//! The CLI's alias table and selection (CONTINUATION-PLUGIN-1, C3).

use std::path::Path;

use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, hybrid_lllf_f32_model,
};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

use super::*;

fn plan_of(model: fn(&Path), name: &str) -> (tempfile::TempDir, ComponentOpPlan) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(model, checkpoint.path(), container.path(), name);
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .expect("closed");
    (container, plan)
}

#[test]
fn aliases_resolve_to_identities_and_replay_to_the_default() {
    let row = ContinuationIdentity::new("row", 1);
    assert_eq!(resolve_engine(None).unwrap(), row);
    assert_eq!(resolve_engine(Some("row")).unwrap(), row);
    assert_eq!(resolve_engine(Some(REPLAY_ENGINE)).unwrap(), row);
    assert_eq!(
        resolve_engine(Some("standard")).unwrap(),
        ContinuationIdentity::new("canonical", 1)
    );
    let err = resolve_engine(Some("markov-rs")).unwrap_err();
    assert!(
        err.contains("`markov-rs`") && err.contains("row, standard, no-cache"),
        "{err}"
    );
    for known in ["row", "standard", REPLAY_ENGINE] {
        assert!(is_known_engine(known), "{known}");
    }
    assert!(!is_known_engine("turbo-quant"));
    assert_eq!(resolve_engine(Some(DEFAULT_ENGINE)).unwrap(), row);
}

#[test]
fn selection_happens_against_the_plan_and_names_its_authority() {
    let (_c, plan) = plan_of(dense_f32_model, "cli-dense");
    let canonical = select_for(&plan, Some("standard")).unwrap();
    assert_eq!(canonical.authority().identity.to_string(), "canonical/v1");
    let err = select_for(&plan, Some("apollo")).unwrap_err();
    assert!(err.contains("`apollo`"), "{err}");
}

/// A one-shot forward asks the executor whether it needs state: none for
/// a wholly-softmax plan, a fresh provider for a hybrid one.
#[test]
fn one_shot_state_follows_the_executor_s_rule() {
    let (_d, dense) = plan_of(dense_f32_model, "cli-dense");
    assert!(one_shot_state(&dense).unwrap().is_none());
    let (_h, hybrid) = plan_of(hybrid_lllf_f32_model, "cli-hybrid");
    let state = one_shot_state(&hybrid)
        .unwrap()
        .expect("a hybrid needs state");
    assert_eq!(state.position(), 0);
}
