//! **A request states the experiment it names, or it does not exist.**

use std::collections::BTreeSet;

use super::super::super::measure::{DEFAULT_GATE, DEFAULT_LABEL, DEFAULT_SEQUENCES};
use super::super::super::measurement::EvidenceScale;
use super::super::super::quality::{kimi_logit_balanced_v1, QualityGate};
use super::super::super::state::evidence_bank::EvidenceBank;
use super::super::super::state::key::MeasurementKey;
use super::super::super::state::protocol::{MeasurementProtocol, ProtocolMismatch};
use super::super::MeasurementRequest;
use super::super::RequestRefusal;
use super::{fixtures, glimmer, ready, ready_record, record_declaring, PricedRecord};

/// **P2.** The four-part key is reconstructible from the request alone —
/// no record, no arguments, the layout policy resolved from the name the
/// request carries. A request handed over a queue or read out of a file
/// can be checked for what it is about.
#[test]
fn a_request_re_derives_the_whole_experiment_from_itself() {
    let dir = glimmer();
    let snapshot = ready_record(dir.path());
    let request = ready(&snapshot).request;

    let derived = request.derived_key().expect("the layout policy resolves");
    assert_eq!(&derived, request.key());
    request
        .attests_to_its_key()
        .expect("and the request says so itself");

    // Each part individually, so a digest collision could not carry it.
    assert_eq!(derived.state(), request.key().state());
    assert_eq!(derived.bank(), &request.bank().id());
    assert_eq!(derived.instrument(), &request.instrument().id());
    assert_eq!(derived.scale(), request.scale());
}

/// **The falsifier, as a refusal.** A map that resolves to one state may
/// not be handed out as a request to measure another.
#[test]
fn a_request_for_a_state_the_map_does_not_present_is_refused_naming_both() {
    let dir = glimmer();
    let snapshot = ready_record(dir.path());
    let authorised = ready(&snapshot).request;

    // The applied set that genuinely reaches the authorised state,
    // asked for under a key naming a DIFFERENT physical state. Bank,
    // scale and instrument match the protocol, so the refusal cannot
    // come from the protocol check.
    let elsewhere = MeasurementKey::new(
        fixtures::p().physical_id(),
        &fixtures::selection_bank().id(),
        EvidenceScale::Authority,
        &fixtures::instrument().id(),
    );
    let applied = applied_of(&snapshot);
    let refusal = MeasurementRequest::of(&snapshot, &elsewhere, &applied)
        .expect_err("the map does not present that state");

    let RequestRefusal::StateMismatch { named, resolved } = &refusal else {
        panic!("the state is what disagrees: {refusal:?}");
    };
    assert_eq!(named, fixtures::p().physical_id());
    assert_eq!(resolved, authorised.key().state());
    let said = refusal.to_string();
    assert!(said.contains(named.short()), "{said}");
    assert!(said.contains(resolved.short()), "{said}");
}

/// The same authorisation asked for under another corpus is refused
/// before anything is resolved — a request may not quietly re-aim which
/// body of data an observation is about.
#[test]
fn a_protocol_that_does_not_cover_the_experiment_is_refused() {
    let dir = glimmer();
    let sliced = EvidenceBank::new(
        "kimi-teacher-forced/v1",
        "17d59a6b",
        (0..32).map(|i| format!("seq-{i:03}")),
        32,
    );
    let snapshot = record_declaring(
        dir.path(),
        MeasurementProtocol::new(sliced, fixtures::instrument(), "teacher-forced-two-arm/v1"),
    );
    let refusal = match super::PreparedExperiment::of(&snapshot) {
        super::PreparedExperiment::NotPreparable(r) => r,
        other => panic!("expected a protocol refusal: {other:?}"),
    };
    assert!(
        matches!(
            refusal,
            RequestRefusal::Protocol(ProtocolMismatch::Bank { .. })
        ),
        "{refusal:?}"
    );
}

/// **P4.** Every control comes from the record. The three the historic
/// environment invocation supplied are each observably different here,
/// which is what makes the claim checkable rather than asserted.
#[test]
fn the_controls_come_from_the_record_and_never_from_the_adapters_defaults() {
    let dir = glimmer();
    let snapshot = ready_record(dir.path());
    let request = ready(&snapshot).request;

    // The gate. `measure/mod.rs` names this discrepancy as the reason
    // the request type exists: the runner held a literal while an
    // optimiser record declares something else.
    assert_eq!(request.gate(), "kimi-logit-balanced-v1");
    assert_ne!(request.gate(), DEFAULT_GATE);
    assert_eq!(
        request.resolve_gate().expect("resolves"),
        kimi_logit_balanced_v1()
    );
    assert_eq!(&request.resolve_gate().expect("resolves"), snapshot.gate());

    // The slice. `LARQL_Q2A_SEQUENCES` took 32 of 256 and nothing
    // recorded which 32; the bank declares which, and how many.
    assert_eq!(request.sequences(), 256);
    assert_ne!(request.sequences(), DEFAULT_SEQUENCES);
    assert_eq!(request.bank().positions(), 8192);

    // The label, which is not a control at all — so it is derived from
    // the experiment, and a sweep's outputs are named by what they
    // measured.
    assert_eq!(request.label(), format!("exp-{}", request.key().short()));
    assert_ne!(request.label(), DEFAULT_LABEL);

    // And the procedure, resolved by the executor and never by a device.
    assert_eq!(request.procedure(), "teacher-forced-two-arm/v1");
}

/// A gate whose id this build implements and whose thresholds have moved
/// is refused. Judging under the current definition would draw a verdict
/// against limits nobody in this record agreed to — the `GateMismatch`
/// class, caught before the run rather than reported after it.
#[test]
fn a_record_whose_gate_this_build_has_redefined_is_refused() {
    let dir = glimmer();
    let mut moved: QualityGate = kimi_logit_balanced_v1();
    moved.kl_p99_max *= 2.0;
    let snapshot = PricedRecord::new(dir.path())
        .with_protocol(fixtures::protocol())
        .gate(moved)
        .build();

    let refusal = match super::PreparedExperiment::of(&snapshot) {
        super::PreparedExperiment::NotPreparable(r) => r,
        other => panic!("expected a gate refusal: {other:?}"),
    };
    let RequestRefusal::GateRedefined { named } = &refusal else {
        panic!("the gate is what disagrees: {refusal:?}");
    };
    assert_eq!(named, "kimi-logit-balanced-v1");
}

/// The record-derived path never reaches this refusal — `next_experiment`
/// needs the layout policy to build a price table and fails first — so it
/// is asserted through the door a batch uses, where a caller holds keys
/// rather than a selection.
#[test]
fn a_layout_policy_this_build_does_not_implement_is_refused_by_name() {
    let dir = glimmer();
    let authorised = {
        let snapshot = ready_record(dir.path());
        (
            ready(&snapshot).request.key().clone(),
            applied_of(&snapshot),
        )
    };
    let snapshot = PricedRecord::new(dir.path())
        .with_protocol(fixtures::protocol())
        .layout("no-such-layout-policy/v1")
        .build();

    let refusal = MeasurementRequest::of(&snapshot, &authorised.0, &authorised.1)
        .expect_err("nothing can be resolved under a policy this build lacks");
    let RequestRefusal::UnresolvableLayout { named, .. } = &refusal else {
        panic!("the layout policy is what is missing: {refusal:?}");
    };
    assert_eq!(named, "no-such-layout-policy/v1");
}

/// An applied set naming a move the vocabulary does not have builds no
/// map at all, and says so rather than resolving the base map and
/// silently measuring the wrong thing.
#[test]
fn an_applied_set_the_vocabulary_does_not_have_builds_no_map() {
    let dir = glimmer();
    let snapshot = ready_record(dir.path());
    let key = ready(&snapshot).request.key().clone();

    let refusal = MeasurementRequest::of(
        &snapshot,
        &key,
        &BTreeSet::from(["a-move-nobody-declared".to_string()]),
    )
    .expect_err("the vocabulary does not have it");
    assert!(
        matches!(refusal, RequestRefusal::NoSuchMap { .. }),
        "{refusal:?}"
    );
}

/// The applied set the record's own selection reaches its experiment by.
fn applied_of(snapshot: &super::SearchSnapshot) -> BTreeSet<String> {
    let selection = snapshot.next_experiment().expect("the record prices");
    selection
        .opportunity()
        .expect("not exhausted")
        .leading()
        .applied
        .clone()
}

/// A record judged by a gate this build has never heard of is refused
/// naming the gate, and distinctly from one whose gate has merely moved.
/// A reader told "unresolvable" goes looking for an implementation; one
/// told "redefined" goes looking at a diff.
#[test]
fn a_record_naming_a_gate_this_build_does_not_implement_is_refused() {
    let dir = glimmer();
    let mut unknown = kimi_logit_balanced_v1();
    unknown.id = "a-gate-from-another-programme/v1".into();
    let snapshot = PricedRecord::new(dir.path())
        .with_protocol(fixtures::protocol())
        .gate(unknown)
        .build();

    let refusal = match super::PreparedExperiment::of(&snapshot) {
        super::PreparedExperiment::NotPreparable(r) => r,
        other => panic!("expected a gate refusal: {other:?}"),
    };
    let RequestRefusal::UnresolvableGate { named, detail } = &refusal else {
        panic!("this build cannot resolve it at all: {refusal:?}");
    };
    assert_eq!(named, "a-gate-from-another-programme/v1");
    assert!(
        !detail.is_empty(),
        "the resolver's own message must survive"
    );
}
