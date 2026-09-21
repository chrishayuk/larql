//! **A record may not describe a protocol it does not search under.**
//!
//! The declarations exist so a selected experiment can be RUN. That is
//! only safe if they are the declarations the digests were computed
//! from — otherwise actuation would read one corpus, under one meaning,
//! and file the observation under another.

use super::super::super::measurement::EvidenceScale;
use super::super::fixtures;
use super::*;

fn other_bank() -> EvidenceBank {
    // The same corpus sliced to 32 of its 256 sequences: the difference
    // between a diagnostic and an authority run, and the case the bank
    // id exists to make visible.
    EvidenceBank::new(
        "kimi-teacher-forced/v1",
        "17d59a6b",
        (0..32).map(|i| format!("seq-{i:03}")),
        32,
    )
}

#[test]
fn the_record_s_protocol_describes_the_experiment_it_searches_under() {
    let protocol = fixtures::protocol();
    protocol
        .describes(&fixtures::standing_intent())
        .expect("the fixture's declarations are the ones its intent names");
    protocol
        .describes_key(&fixtures::key_for(&fixtures::p(), EvidenceScale::Authority))
        .expect("and the ones a key built from that intent names");
}

#[test]
fn a_protocol_declaring_another_corpus_is_refused_naming_both_digests() {
    let protocol = MeasurementProtocol::new(
        other_bank(),
        fixtures::instrument(),
        "teacher-forced-two-arm/v1",
    );
    let refusal = protocol
        .describes(&fixtures::standing_intent())
        .expect_err("32 of 256 sequences is a different corpus");

    let ProtocolMismatch::Bank { declared, searched } = &refusal else {
        panic!("the corpus is what differs: {refusal:?}");
    };
    assert_eq!(declared, &other_bank().id());
    assert_eq!(searched, &fixtures::selection_bank().id());
    // Both digests reach the reader, so the refusal can be acted on
    // without diffing two records by hand.
    let said = refusal.to_string();
    assert!(said.contains(declared.short()), "{said}");
    assert!(said.contains(searched.short()), "{said}");
}

#[test]
fn a_protocol_declaring_another_meaning_is_refused_distinctly() {
    // Same corpus, same procedure, a truncation covering a third of the
    // mass instead of most of it — which 1c established is a different
    // measurement and not a repeat.
    let protocol = MeasurementProtocol::new(
        fixtures::selection_bank(),
        fixtures::instrument().truncated_to(128),
        "teacher-forced-two-arm/v1",
    );
    let refusal = protocol
        .describes(&fixtures::standing_intent())
        .expect_err("another truncation is another instrument");
    assert!(
        matches!(refusal, ProtocolMismatch::Instrument { .. }),
        "the meaning is what differs, and it must not be reported as a corpus: {refusal:?}"
    );
}

/// The two checks answer different questions, and the difference is
/// load bearing: `describes` says the record is internally consistent,
/// `describes_key` says THIS experiment is the one the declarations
/// cover. A policy that ever selects under another intent is caught
/// only by the second.
#[test]
fn describing_the_intent_does_not_describe_every_key() {
    let protocol = fixtures::protocol();
    protocol
        .describes(&fixtures::standing_intent())
        .expect("internally consistent");

    let foreign = MeasurementKey::new(
        fixtures::p().physical_id(),
        &other_bank().id(),
        EvidenceScale::Authority,
        &fixtures::instrument().id(),
    );
    let refusal = protocol
        .describes_key(&foreign)
        .expect_err("a key over another corpus is not covered by these declarations");
    assert!(
        matches!(refusal, ProtocolMismatch::Bank { .. }),
        "{refusal:?}"
    );
}

/// `LARQL_Q2A_SEQUENCES` took 32 of 256 and nothing recorded which 32.
/// A record-derived run consumes exactly the samples its own bank id
/// was computed over, so the slice is no longer a control anyone sets.
#[test]
fn sequences_come_from_the_declaration_rather_than_from_a_setting() {
    let protocol = fixtures::protocol();
    assert_eq!(protocol.sequences(), 256);
    assert_eq!(protocol.bank.positions(), 8192);

    let sliced = MeasurementProtocol::new(
        other_bank(),
        fixtures::instrument(),
        "teacher-forced-two-arm/v1",
    );
    assert_eq!(sliced.sequences(), 32);
    assert_eq!(sliced.bank.positions(), 1024);
    // And the two are not the same experiment, which is why the count
    // may be read off the declaration at all.
    assert_ne!(protocol.bank.id(), sliced.bank.id());
}

/// The procedure to perform and the procedure a reading was taken under
/// are two different claims in two different namespaces. Nothing here
/// asserts a mapping between them, and the instrument's own string
/// stays inside its digest where 1c put it.
#[test]
fn the_executable_procedure_is_not_the_instrument_s_procedure() {
    let protocol = fixtures::protocol();
    assert_eq!(protocol.procedure, "teacher-forced-two-arm/v1");
    assert_eq!(
        protocol.instrument.procedure,
        "q2a-teacher-forced/baseline-vs-overlay"
    );

    // Renaming what this build performs cannot move what the reading
    // means; the instrument digest is unchanged.
    let renamed = MeasurementProtocol::new(
        fixtures::selection_bank(),
        fixtures::instrument(),
        "some-other-runner/v1",
    );
    assert_eq!(renamed.instrument.id(), protocol.instrument.id());
    renamed
        .describes(&fixtures::standing_intent())
        .expect("the procedure is in no digest");
}

#[test]
fn a_protocol_survives_json() {
    let protocol = fixtures::protocol();
    let text = serde_json::to_string(&protocol).expect("serialise");
    let back: MeasurementProtocol = serde_json::from_str(&text).expect("reload");
    assert_eq!(back, protocol);
    // The identities it stands for survive with it, which is the only
    // reason storing the declarations helps.
    assert_eq!(back.bank.id(), fixtures::selection_bank().id());
    assert_eq!(back.instrument.id(), fixtures::instrument().id());
}
