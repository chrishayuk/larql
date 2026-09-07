//! Step 2 of ATTESTATION-1: the tuple check, and every staleness cause
//! by name.
//!
//! The forecast (X3) asked for six causes. Two of them — a changed source
//! and a changed recipe — turn out not to be container-checkable at all,
//! because the index says outright that once encoded the source
//! checkpoint disappears as an authority. They are checked here only when
//! a caller who HAS them supplies the expectation, and the arm below
//! proves both halves: refused when expected, silent when not.

use std::collections::BTreeMap;

use super::tuple::{AttestationStatus, AttestedSubject, StalenessCause};
use super::*;

const OBJECT: &str = "block.0";
const TENSOR: &str = "mlp.gate_proj.weight";
const CODEBOOK: &str = "codebook";
const BASELINE: &str = "VQ8_SHARED@1/terminal";
const SOURCE: &str = "sha256:source";
const RECIPE: &str = "kmeans/lloyd@8";

fn address() -> OperandAddress {
    OperandAddress::new(OBJECT, TENSOR)
}

fn stored() -> StoredAttestation {
    StoredAttestation {
        binding: AttestationBinding {
            operand: address(),
            extent_depth: 0,
            codec_family: "VQ8_SHARED".into(),
            codec_revision: 1,
            shape: vec![64, 101],
            content_digest: "sha256:codes".into(),
            source_digest: SOURCE.into(),
            auxiliary_baselines: BTreeMap::from([(CODEBOOK.to_string(), BASELINE.to_string())]),
            recipe: RECIPE.into(),
        },
        method: AttestationMethod::new("larql-encoder", StoredId::new("measured-rms", 1)),
        metric: StoredId::new("relative-rms", 1),
        domain: StoredId::new("finite-normals", 1),
        radius: 0.031,
    }
}

fn table() -> AttestationTable {
    RepresentationAttestations::new(vec![stored()])
        .judge()
        .unwrap()
}

/// What the container currently says — matching the attestation, so each
/// arm below can break exactly one thing.
fn subject<'a>(operand: &'a OperandAddress, shape: &'a [usize]) -> AttestedSubject<'a> {
    AttestedSubject {
        operand,
        extent_depth: 0,
        codec_family: "VQ8_SHARED",
        codec_revision: 1,
        shape,
        auxiliary_baselines: BTreeMap::from([(CODEBOOK.to_string(), BASELINE.to_string())]),
        expected_source_digest: None,
        expected_recipe: None,
    }
}

const SHAPE: [usize; 2] = [64, 101];

/// The arm: everything the container can speak to agrees, so the tuple
/// holds — and `Bound` says only that, not that the bytes are right.
#[test]
fn an_attestation_that_still_describes_the_operand_is_bound() {
    let table = table();
    let operand = address();
    let status = table.status_of(&subject(&operand, &SHAPE));
    assert!(matches!(status, AttestationStatus::Bound(_)));
    assert!(status.unavailable_because().is_none());
    assert_eq!(status.attestation().unwrap().claimed().radius(), 0.031);
}

/// Absence names the depths that ARE attested, so "no guarantee here" is
/// never mistaken for "the file did not load".
#[test]
fn absence_says_where_the_container_does_attest() {
    let table = table();
    let operand = address();

    let mut elsewhere = subject(&operand, &SHAPE);
    elsewhere.extent_depth = 2;
    let status = table.status_of(&elsewhere);
    assert_eq!(status, AttestationStatus::Absent { elsewhere: vec![0] });
    let why = status.unavailable_because().unwrap();
    assert!(why.contains("depths [0]"), "{why}");
    assert!(status.attestation().is_none());

    // An operand nothing attests at all reads differently again.
    let unattested = OperandAddress::new(OBJECT, "mlp.down_proj.weight");
    let status = table.status_of(&subject(&unattested, &SHAPE));
    assert_eq!(status, AttestationStatus::Absent { elsewhere: vec![] });
    let why = status.unavailable_because().unwrap();
    assert!(why.contains("nothing is attested for it"), "{why}");
}

/// Each container-checkable cause fires by name, and each names both
/// readings — the attested one and the found one.
#[test]
fn every_container_checkable_staleness_cause_is_named() {
    let table = table();
    let operand = address();

    let mut family = subject(&operand, &SHAPE);
    family.codec_family = "MXFP4";
    let StalenessCause::CodecFamily { attested, found } = cause(&table, &family) else {
        panic!("expected a family mismatch");
    };
    assert_eq!((attested.as_str(), found.as_str()), ("VQ8_SHARED", "MXFP4"));

    let mut revision = subject(&operand, &SHAPE);
    revision.codec_revision = 2;
    assert_eq!(
        cause(&table, &revision),
        StalenessCause::CodecRevision {
            attested: 1,
            found: 2
        }
    );

    let other_shape = [64, 100];
    let shape = subject(&operand, &other_shape);
    assert_eq!(
        cause(&table, &shape),
        StalenessCause::Shape {
            attested: vec![64, 101],
            found: vec![64, 100]
        }
    );

    let mut replaced = subject(&operand, &SHAPE);
    replaced.auxiliary_baselines =
        BTreeMap::from([(CODEBOOK.to_string(), "VQ8_SHARED@1/another".to_string())]);
    let StalenessCause::DependencyBaseline { name, .. } = cause(&table, &replaced) else {
        panic!("expected a baseline mismatch");
    };
    assert_eq!(name, CODEBOOK);

    let mut gained = subject(&operand, &SHAPE);
    gained
        .auxiliary_baselines
        .insert("palette".to_string(), "x".to_string());
    let StalenessCause::DependencySet { attested, found } = cause(&table, &gained) else {
        panic!("expected a dependency-set mismatch");
    };
    assert_eq!(attested, vec![CODEBOOK.to_string()]);
    assert_eq!(found, vec![CODEBOOK.to_string(), "palette".to_string()]);
}

/// A dependency that DISAPPEARS is a set change too — the arm above only
/// showed one gained, and a missing one takes the other branch of the
/// same comparison.
#[test]
fn a_dependency_that_is_gone_is_a_set_change_not_a_baseline_change() {
    let table = table();
    let operand = address();
    let mut lost = subject(&operand, &SHAPE);
    lost.auxiliary_baselines.clear();
    let StalenessCause::DependencySet { attested, found } = cause(&table, &lost) else {
        panic!("expected a dependency-set mismatch");
    };
    assert_eq!(attested, vec![CODEBOOK.to_string()]);
    assert!(found.is_empty());
}

/// The two causes the container cannot check: refused when a caller
/// brings the expectation, silent when nobody does.
///
/// Both halves matter. The first proves the check exists; the second
/// proves it is not quietly passing because it never runs — which is
/// exactly how an unverifiable field turns into a decorative one.
#[test]
fn source_and_recipe_are_checked_only_when_a_caller_supplies_them() {
    let table = table();
    let operand = address();

    let mut wrong_source = subject(&operand, &SHAPE);
    wrong_source.expected_source_digest = Some("sha256:a-different-checkpoint");
    assert!(matches!(
        cause(&table, &wrong_source),
        StalenessCause::SourceDigest { .. }
    ));

    let mut wrong_recipe = subject(&operand, &SHAPE);
    wrong_recipe.expected_recipe = Some("kmeans/lloyd@16");
    assert!(matches!(
        cause(&table, &wrong_recipe),
        StalenessCause::Recipe { .. }
    ));

    // Supplied and matching: bound, so the check is comparing and not
    // merely rejecting whatever it is handed.
    let mut right = subject(&operand, &SHAPE);
    right.expected_source_digest = Some(SOURCE);
    right.expected_recipe = Some(RECIPE);
    assert!(matches!(
        table.status_of(&right),
        AttestationStatus::Bound(_)
    ));

    // Not supplied: bound, because a container holding neither the source
    // nor the recipe cannot say either is wrong.
    assert!(matches!(
        table.status_of(&subject(&operand, &SHAPE)),
        AttestationStatus::Bound(_)
    ));
}

/// A wrong codec is reported instead of the shape difference it also
/// causes: comparing the shape of an operand stored under another codec
/// entirely would send a reader after the wrong thing.
#[test]
fn the_most_fundamental_difference_is_the_one_reported() {
    let table = table();
    let operand = address();
    let other_shape = [8, 8];
    let mut both_wrong = subject(&operand, &other_shape);
    both_wrong.codec_family = "MXFP4";
    both_wrong.codec_revision = 9;
    both_wrong.auxiliary_baselines.clear();
    assert!(matches!(
        cause(&table, &both_wrong),
        StalenessCause::CodecFamily { .. }
    ));
}

/// Staleness carries the attestation it is about, so a caller can name
/// whose measurement expired rather than only that one did.
#[test]
fn a_stale_status_still_names_the_measurement_and_never_implies_zero() {
    let table = table();
    let operand = address();
    let mut revision = subject(&operand, &SHAPE);
    revision.codec_revision = 4;
    let status = table.status_of(&revision);

    let attestation = status.attestation().expect("stale still has one");
    assert!(attestation.describe().contains("larql-encoder"));
    let why = status.unavailable_because().unwrap();
    assert!(why.contains("stale"), "{why}");
    assert!(why.contains("revision 1"), "{why}");
    assert!(why.contains("revision 4"), "{why}");
    // The number is NOT offered. Asserted against the radius itself
    // rather than against the digit `0`, which would fail the moment
    // someone put the operand address `block.0` in the message — a
    // correct change failing a test is worse than the test not existing.
    assert!(
        !why.contains("0.031"),
        "a staleness reason must not hand over the bound it is refusing: {why}"
    );
}

/// The tuple check reads nothing but what it was handed: no file, no
/// payload, no registry. The property the forecast (X1) states as
/// "metadata only, before I/O" — asserted here at the level where it is
/// decidable, since this function has no store to read from at all.
#[test]
fn the_tuple_check_needs_no_container_at_all() {
    // Constructed entirely in memory, from a table that was never read
    // from disk, against a subject nobody loaded.
    let table = table();
    let operand = address();
    assert!(matches!(
        table.status_of(&subject(&operand, &SHAPE)),
        AttestationStatus::Bound(_)
    ));
}

fn cause(table: &AttestationTable, subject: &AttestedSubject<'_>) -> StalenessCause {
    match table.status_of(subject) {
        AttestationStatus::Stale { cause, .. } => cause,
        other => panic!("expected staleness, got {other:?}"),
    }
}

/// Every cause says what it found AND what was attested, in words.
///
/// The arms above assert which VARIANT fired, which is what a caller acts
/// on. This asserts the sentence a human reads — a refusal whose message
/// names only one side of the comparison sends the reader to look up the
/// other, and six of these were unexercised until this test existed.
#[test]
fn every_cause_names_both_readings_in_its_message() {
    let cases = [
        (
            StalenessCause::CodecFamily {
                attested: "VQ8_SHARED".into(),
                found: "MXFP4".into(),
            },
            vec!["VQ8_SHARED", "MXFP4"],
        ),
        (
            StalenessCause::CodecRevision {
                attested: 1,
                found: 4,
            },
            vec!["revision 1", "revision 4"],
        ),
        (
            StalenessCause::Shape {
                attested: vec![64, 101],
                found: vec![64, 100],
            },
            vec!["[64, 101]", "[64, 100]"],
        ),
        (
            StalenessCause::DependencySet {
                attested: vec![CODEBOOK.into()],
                found: vec!["palette".into()],
            },
            vec![CODEBOOK, "palette"],
        ),
        (
            StalenessCause::DependencyBaseline {
                name: CODEBOOK.into(),
                attested: BASELINE.into(),
                found: "another".into(),
            },
            vec![CODEBOOK, BASELINE, "another"],
        ),
        (
            StalenessCause::SourceDigest {
                attested: SOURCE.into(),
                expected: "sha256:other".into(),
            },
            vec![SOURCE, "sha256:other"],
        ),
        (
            StalenessCause::Recipe {
                attested: RECIPE.into(),
                expected: "kmeans/lloyd@16".into(),
            },
            vec![RECIPE, "kmeans/lloyd@16"],
        ),
    ];
    for (cause, must_name) in cases {
        let described = cause.describe();
        for part in must_name {
            assert!(
                described.contains(part),
                "{cause:?} does not name `{part}`: {described}"
            );
        }
        // And it reads as a sentence about a mismatch, not a label.
        assert!(described.len() > 30, "{described}");
    }
}
