//! Step 3 of ATTESTATION-1: presence is not trust.
//!
//! The forecast's X4 asks for an attestation that is well-formed, current
//! and bound to the right bytes, by an authority or method this build
//! does not recognise, to leave the certificate unavailable — named as
//! unrecognised, not as absent, not as stale — with a discriminating
//! control proving the refusal is the recognition and nothing else.

use std::collections::BTreeMap;

use super::recognition::{RecognisedMethods, RecognitionGap};
use super::tuple::{AttestationStatus, AttestedSubject};
use super::*;

const OBJECT: &str = "block.0";
const TENSOR: &str = "mlp.gate_proj.weight";
const CODEBOOK: &str = "codebook";
const BASELINE: &str = "VQ8_SHARED@1/terminal";
const AUTHORITY: &str = "larql-encoder";
const METHOD: &str = "measured-rms";
const SHAPE: [usize; 2] = [64, 101];

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
            shape: SHAPE.to_vec(),
            content_digest: "sha256:codes".into(),
            source_digest: "sha256:source".into(),
            auxiliary_baselines: BTreeMap::from([(CODEBOOK.to_string(), BASELINE.to_string())]),
            recipe: "kmeans/lloyd@8".into(),
        },
        method: AttestationMethod::new(AUTHORITY, StoredId::new(METHOD, 1)),
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

/// What a reader that has qualified this measurement says.
fn recognising() -> RecognisedMethods {
    RecognisedMethods::none()
        .with_authority(AUTHORITY)
        .with_method(METHOD, 1)
}

/// **The default recognises nothing, and that makes every attestation
/// unusable.**
///
/// The costly half of "presence is not trust": an artifact full of
/// perfectly good attestations is worth nothing to a build that has
/// qualified no authority. Asserted rather than assumed, because a
/// convenience default here would defeat the entire plane — any file
/// claiming a familiar name could assert any radius it liked.
#[test]
fn a_build_that_has_qualified_nothing_recognises_nothing() {
    let nothing = RecognisedMethods::none();
    assert!(nothing.is_empty());
    assert!(!nothing.trusts(AUTHORITY));
    assert!(!nothing.implements(&StoredId::new(METHOD, 1)));
    assert_eq!(RecognisedMethods::default(), nothing);

    let table = table();
    let operand = address();
    let status = table.status_of(&subject(&operand, &SHAPE), &nothing);
    let AttestationStatus::Unrecognised { gap, .. } = &status else {
        panic!("expected unrecognised, got {status:?}");
    };
    assert_eq!(*gap, RecognitionGap::NothingRecognised);
    let why = status.unavailable_because(&nothing).unwrap();
    assert!(why.contains("recognises no attestation authority"), "{why}");
}

/// The discriminating control: the SAME attestation, the SAME artifact,
/// one recognised and one not. The refusal is the recognition and
/// nothing else.
#[test]
fn the_same_attestation_is_usable_or_not_purely_by_recognition() {
    let table = table();
    let operand = address();
    let subject = subject(&operand, &SHAPE);

    let recognised = recognising();
    assert!(matches!(
        table.status_of(&subject, &recognised),
        AttestationStatus::Bound(_)
    ));
    // Recognised, so the only thing left between it and a guarantee is
    // its payload — which this test deliberately does not supply, because
    // it is about recognition and nothing else.
    let why = table
        .status_of(&subject, &recognised)
        .unavailable_because(&recognised)
        .unwrap();
    assert!(why.contains("payload has not been checked"), "{why}");

    let stranger = RecognisedMethods::none()
        .with_authority("someone-else")
        .with_method(METHOD, 1);
    assert!(matches!(
        table.status_of(&subject, &stranger),
        AttestationStatus::Unrecognised { .. }
    ));
}

/// Unrecognised is neither absent nor stale, and the three say different
/// things — the distinction the forecast asks for, asserted as a
/// distinction rather than three separate facts.
#[test]
fn unrecognised_is_not_absent_and_not_stale() {
    let table = table();
    let operand = address();
    let recognised = recognising();
    let stranger = RecognisedMethods::none().with_authority("someone-else");

    // Absent: nothing attested at this depth, whoever is asking.
    let mut elsewhere = subject(&operand, &SHAPE);
    elsewhere.extent_depth = 3;
    let absent = table.status_of(&elsewhere, &recognised);
    assert!(matches!(absent, AttestationStatus::Absent { .. }));
    assert!(absent.attestation().is_none());

    // Stale: the artifact moved, whoever is asking.
    let mut moved = subject(&operand, &SHAPE);
    moved.codec_revision = 9;
    let stale = table.status_of(&moved, &recognised);
    assert!(matches!(stale, AttestationStatus::Stale { .. }));

    // Unrecognised: nothing wrong with the artifact at all.
    let unknown = table.status_of(&subject(&operand, &SHAPE), &stranger);
    assert!(matches!(unknown, AttestationStatus::Unrecognised { .. }));

    // All three refuse, and each says something different.
    let reasons: Vec<String> = [absent, stale, unknown]
        .iter()
        .map(|s| s.unavailable_because(&recognised).unwrap())
        .collect();
    assert_eq!(
        reasons
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        3,
        "three states must not read alike: {reasons:?}"
    );
    // And none of them offers a number.
    for reason in &reasons {
        assert!(!reason.contains("0.031"), "{reason}");
    }
}

/// Recognition is exact, version included: a method version exists
/// because the measurement differs, so honouring `@2` because `@1` is
/// implemented would be honouring a definition nobody read.
///
/// And the refusal is useful about it — being told "you have that method
/// at version 1" is far more actionable than "unknown method".
#[test]
fn a_version_this_build_does_not_implement_is_not_recognised() {
    let mut newer = stored();
    newer.method = AttestationMethod::new(AUTHORITY, StoredId::new(METHOD, 2));
    let table = RepresentationAttestations::new(vec![newer])
        .judge()
        .unwrap();

    let operand = address();
    let recognised = recognising();
    let status = table.status_of(&subject(&operand, &SHAPE), &recognised);
    let AttestationStatus::Unrecognised { gap, .. } = &status else {
        panic!("expected unrecognised, got {status:?}");
    };
    assert_eq!(
        *gap,
        RecognitionGap::Method {
            given: format!("{METHOD}@2"),
            other_versions: vec![1],
        }
    );
    let why = status.unavailable_because(&recognised).unwrap();
    assert!(why.contains("versions [1]"), "{why}");
    assert!(why.contains("the measurement differs"), "{why}");
}

/// Both lists must be satisfied: a trusted authority quoting a method
/// this build does not implement is as unusable as a stranger quoting one
/// it does.
#[test]
fn authority_and_method_are_both_required() {
    let table = table();
    let operand = address();
    let subject = subject(&operand, &SHAPE);

    // Trusted, method not implemented.
    let no_method = RecognisedMethods::none().with_authority(AUTHORITY);
    assert!(matches!(
        table.status_of(&subject, &no_method),
        AttestationStatus::Unrecognised {
            gap: RecognitionGap::Method { .. },
            ..
        }
    ));

    // Method implemented, authority not trusted.
    let no_authority = RecognisedMethods::none()
        .with_authority("other")
        .with_method(METHOD, 1);
    assert!(matches!(
        table.status_of(&subject, &no_authority),
        AttestationStatus::Unrecognised {
            gap: RecognitionGap::Authority { .. },
            ..
        }
    ));

    // Both: bound.
    assert!(matches!(
        table.status_of(&subject, &recognising()),
        AttestationStatus::Bound(_)
    ));
}

/// Authority is reported before method: a reader whose word is not taken
/// at all does not need a lecture about versions.
#[test]
fn an_untrusted_authority_is_reported_before_an_unknown_method() {
    let mut both = stored();
    both.method = AttestationMethod::new("a-stranger", StoredId::new("their-method", 7));
    let table = RepresentationAttestations::new(vec![both]).judge().unwrap();

    let operand = address();
    let recognised = recognising();
    let AttestationStatus::Unrecognised { gap, .. } =
        table.status_of(&subject(&operand, &SHAPE), &recognised)
    else {
        panic!("expected unrecognised");
    };
    assert_eq!(
        gap,
        RecognitionGap::Authority {
            given: "a-stranger".into()
        }
    );
    let why = gap.describe(&recognised);
    assert!(why.contains("a-stranger"), "{why}");
    // It says what it WOULD have taken, so the reader can act.
    assert!(why.contains(AUTHORITY), "{why}");
}

/// **Precedence, pinned deliberately.** When an attestation is both stale
/// and unrecognised, staleness wins.
///
/// Staleness is a fact about the container and holds for every reader;
/// unrecognition is a fact about this build. Reporting recognition first
/// would send an operator to configure trust, after which they would
/// discover the attestation had expired anyway. This is a judgement call,
/// so it is asserted rather than left to whichever check happens to run
/// first.
#[test]
fn staleness_is_reported_before_unrecognition_when_both_hold() {
    let table = table();
    let operand = address();
    let mut moved = subject(&operand, &SHAPE);
    moved.codec_revision = 9;

    let stranger = RecognisedMethods::none().with_authority("someone-else");
    assert!(
        matches!(
            table.status_of(&moved, &stranger),
            AttestationStatus::Stale { .. }
        ),
        "the portable fact is the one to report"
    );
    // And with recognition fixed, the staleness is still there — which is
    // exactly the wasted trip the ordering avoids.
    assert!(matches!(
        table.status_of(&moved, &recognising()),
        AttestationStatus::Stale { .. }
    ));
}

/// A refusal names what this reader would have accepted, so the operator
/// is not left guessing at the shape of the configuration.
#[test]
fn a_recognition_refusal_says_what_it_would_have_taken() {
    let recognised = recognising().with_method("sampled-rms", 3);
    assert_eq!(
        recognised.authorities().collect::<Vec<_>>(),
        vec![AUTHORITY]
    );
    assert_eq!(
        recognised.methods().collect::<Vec<_>>(),
        vec!["measured-rms@1".to_string(), "sampled-rms@3".to_string()]
    );

    let gap = RecognitionGap::Method {
        given: "unheard-of@1".into(),
        other_versions: vec![],
    };
    let why = gap.describe(&recognised);
    assert!(why.contains("measured-rms@1"), "{why}");
    assert!(why.contains("sampled-rms@3"), "{why}");
}
