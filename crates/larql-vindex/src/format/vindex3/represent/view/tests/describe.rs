//! **What this search IS**, and that every word of it came from the
//! record rather than from this module.

use super::super::super::state::graph::TransitionPolicy;
use super::super::super::state::snapshot::Objective;
use super::{reloaded, view};

#[test]
fn the_declared_identity_is_the_snapshots_own() {
    let snap = reloaded();
    let described = view(&snap).describe();

    assert_eq!(described.schema, snap.schema());
    assert_eq!(&described.model, snap.graph().model());
    assert_eq!(described.surface_identity, snap.graph().surface_identity());
    assert_eq!(described.surface_tensors, snap.space().surface.len());
    assert_eq!(described.objective, snap.objective());
    assert_eq!(&described.contract, snap.gate());
    assert_eq!(&described.tail_support, snap.tail_support());
    assert_eq!(described.transition_policy, snap.graph().policy());
    assert_eq!(&described.semantics, snap.semantics());
    assert_eq!(described.semantics_id, snap.semantics_id().as_str());
    assert_eq!(&described.vocabulary, &snap.space().vocabulary);
}

#[test]
fn the_frozen_contract_travels_with_every_verdict_it_licenses() {
    let snap = reloaded();
    let described = view(&snap).describe();

    // The gate id is what makes a verdict mean anything a year later:
    // change a threshold and it is a different gate, so a reader who
    // has this string knows exactly what was asked.
    assert_eq!(described.contract.id, snap.gate().id);
    assert_eq!(described.contract.kl_p99_max, 3.5e-3);
    assert_eq!(described.objective, Objective::MinimiseLogicalBytes);
}

#[test]
fn acyclicity_is_reported_as_a_theorem_under_the_declared_policy() {
    let snap = reloaded();
    let described = view(&snap).describe();

    assert_eq!(
        described.transition_policy,
        TransitionPolicy::StrictlyImprovingPhysical
    );
    assert!(described.guarantees_acyclic);
    assert_eq!(
        described.guarantees_acyclic,
        snap.graph().policy().guarantees_acyclic(),
        "the guarantee is the policy's, not this view's"
    );
}

#[test]
fn the_whole_tie_break_chain_is_rendered_and_not_a_prefix_of_it() {
    let snap = reloaded();
    let described = view(&snap).describe();
    let chain = snap.config().ranking.tie_break_chain();

    assert_eq!(described.tie_break_chain, chain);
    assert_eq!(
        described.tie_break_chain.first().map(String::as_str),
        Some("registered rule")
    );
    assert_eq!(
        described.tie_break_chain.last().map(String::as_str),
        Some("canonical action identity"),
        "a truncated chain leaves an answer depending on traversal order"
    );
    assert_eq!(described.ranking_rule, "physical-prize-first");
    assert_eq!(described.ranking_id, snap.config().ranking.id().as_str());
}

#[test]
fn the_six_decision_procedures_are_named_separately() {
    let snap = reloaded();
    let semantics = view(&snap).describe().semantics;

    // Named separately because they change independently: Ruling 1
    // rewrote pruning without touching evidence interpretation.
    assert_eq!(semantics.candidate_generation, "exchange-1-out-1-in/v1");
    assert_eq!(
        semantics.pre_measurement_pruning,
        "ruling-1-three-prunes/v1"
    );
    assert_eq!(
        semantics.evidence_interpretation,
        "search-evidence-ladder/v1"
    );
    assert_eq!(semantics.promotion_rule, "decide-promotion-ordinal/v1");
    assert_eq!(semantics.ranking_rule, "physical-prize-first/v1");
    assert_eq!(semantics.physical_accounting, "logical-bytes/v1");
}
