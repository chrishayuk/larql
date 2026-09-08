//! **Four answers, and each calls for a different action.**
//!
//! The distinction the enum exists for:
//!
//! ```text
//! NotSelectable   the record cannot say WHAT to measure   fix its pricing
//! NotPreparable   it said what, and not HOW               fix its protocol
//! ```
//!
//! Re-reading a container fixes the first and cannot fix the second, so
//! collapsing them would send an operator to the wrong place.

use super::super::super::state::key::MeasurementKey;
use super::super::PreparedExperiment;
use super::{fixtures, glimmer, ready, ready_record, PricedRecord};

/// The whole chain, from a stored record to a request something could
/// perform, with nothing injected at any point.
#[test]
fn a_record_that_prices_and_declares_its_protocol_prepares_a_real_experiment() {
    let dir = glimmer();
    let snapshot = ready_record(dir.path());
    let prepared = ready(&snapshot);

    // The request is FOR the experiment the optimiser selected, and the
    // two derivations are independent: one goes through the ranking
    // policy, the other re-resolves the map the request carries.
    let selected = snapshot
        .next_experiment()
        .expect("the record prices")
        .opportunity()
        .expect("not exhausted")
        .key
        .clone();
    assert_eq!(prepared.request.key(), &selected);
    assert_eq!(
        prepared.request.derived_key().expect("resolves"),
        selected,
        "the request re-derives the experiment the policy chose"
    );

    // The prize is the opportunity's, and it points the way the
    // objective wants.
    assert!(
        prepared.physical_delta < 0,
        "compiling removes bytes: {}",
        prepared.physical_delta
    );
    assert!(prepared.routes >= 1);
    assert!(prepared.considered >= 1);

    // And what it says to measure came from the record.
    assert_eq!(prepared.request.model(), snapshot.graph().model());
    assert_eq!(prepared.request.surface(), &snapshot.space().surface);
    assert_eq!(
        prepared.request.layout_admission(),
        snapshot.semantics().layout_admission
    );
    assert_eq!(prepared.request.bank(), &fixtures::selection_bank());
    assert_eq!(prepared.request.instrument(), &fixtures::instrument());
}

/// A record with no pricing authority cannot say what to measure, and
/// that is NOT the same answer as being unable to prepare it.
#[test]
fn a_record_that_cannot_price_is_not_selectable_rather_than_not_preparable() {
    let snapshot = fixtures::reloaded();
    let answer = PreparedExperiment::of(&snapshot);
    let PreparedExperiment::NotSelectable { detail } = &answer else {
        panic!("the rung-5 record carries no accounting facts: {answer:?}");
    };
    assert!(
        detail.contains("accounting"),
        "the refusal must name the missing authority: {detail}"
    );
    assert!(!answer.is_ready());
    assert!(answer.request().is_none());
}

/// A record that CAN price and names its corpus and instrument only by
/// digest gets the other refusal — the one that says the protocol is
/// missing rather than the prices.
#[test]
fn a_record_that_names_its_protocol_only_by_digest_refuses_and_says_so() {
    let dir = glimmer();
    let snapshot = PricedRecord::new(dir.path()).build();

    // It can still answer the read-path question, which is the point:
    // nothing derived changes when the protocol is absent.
    snapshot
        .next_experiment()
        .expect("pricing authority is intact");

    let answer = PreparedExperiment::of(&snapshot);
    let PreparedExperiment::NotPreparable(refusal) = &answer else {
        panic!("expected a protocol refusal: {answer:?}");
    };
    assert_eq!(
        refusal,
        &super::super::RequestRefusal::NoProtocol,
        "and it names the missing authority rather than the prices"
    );
    let said = refusal.to_string();
    assert!(said.contains("digest"), "{said}");
}

/// Every route to one experiment shares its measurement, and the request
/// is built from the route the policy ranked first — so preparation
/// never overrules the ordering it was handed.
#[test]
fn the_request_follows_the_route_the_policy_ranked_first() {
    let dir = glimmer();
    let snapshot = ready_record(dir.path());
    let prepared = ready(&snapshot);

    let selection = snapshot.next_experiment().expect("prices");
    let opportunity = selection.opportunity().expect("not exhausted");
    assert_eq!(
        prepared.request.candidate_map(),
        &snapshot
            .space()
            .vocabulary
            .map_for(&snapshot.space().base_map, &opportunity.leading().applied)
            .expect("the leading route's map"),
    );
    // Two routes reach this one physical state, and both would run the
    // same experiment — which is why taking the leading one changes
    // nothing about what is measured.
    assert_eq!(prepared.routes, 2);
    for route in &opportunity.candidates {
        assert_eq!(&route.intended_key, prepared.request.key());
    }
}

/// The answer survives being asked twice, and asking does not move the
/// record. Preparation is a derivation like every other one on this
/// record, so a second call must be the same call.
#[test]
fn preparing_twice_gives_the_same_request_and_changes_nothing() {
    let dir = glimmer();
    let snapshot = ready_record(dir.path());
    let once = ready(&snapshot);
    let twice = ready(&snapshot);
    assert_eq!(once, twice);

    // Nothing was recorded: preparing an experiment is not observing one.
    assert!(snapshot.measurements().is_empty());
    let key: &MeasurementKey = once.request.key();
    assert!(
        !snapshot.measurements().contains(key),
        "the bridge must not write to the scientific record"
    );
}
