//! Step 4 of ATTESTATION-1: the bytes.
//!
//! The forecast's X2 asks for a tampered payload whose tuple still
//! matches to be refused at preparation, naming the operand and the
//! digest that did not match — with the control that the SAME operand
//! untampered prepares successfully, so the refusal is the digest and not
//! the path.
//!
//! This is the check that cannot be metadata-only. The container holds no
//! per-tensor digest (B4), so an attestation binding to content is
//! verifiable only against the content.

use std::collections::BTreeMap;

use super::recognition::RecognisedMethods;
use super::tuple::{AttestationStatus, AttestedSubject, StalenessCause};
use super::*;

const OBJECT: &str = "block.0";
const TENSOR: &str = "mlp.gate_proj.weight";
const CODEBOOK: &str = "codebook";
const BASELINE: &str = "VQ8_SHARED@1/terminal";
const AUTHORITY: &str = "larql-encoder";
const METHOD: &str = "measured-rms";
const SHAPE: [usize; 2] = [64, 101];

/// The bytes this operand is attested about.
fn payload() -> Vec<u8> {
    (0..=255u8).cycle().take(64 * 101).collect()
}

fn address() -> OperandAddress {
    OperandAddress::new(OBJECT, TENSOR)
}

fn stored_for(bytes: &[u8]) -> StoredAttestation {
    StoredAttestation {
        binding: AttestationBinding {
            operand: address(),
            extent_depth: 0,
            codec_family: "VQ8_SHARED".into(),
            codec_revision: 1,
            shape: SHAPE.to_vec(),
            content_digest: content_digest(bytes),
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

fn table(bytes: &[u8]) -> AttestationTable {
    RepresentationAttestations::new(vec![stored_for(bytes)])
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

fn recognising() -> RecognisedMethods {
    RecognisedMethods::none()
        .with_authority(AUTHORITY)
        .with_method(METHOD, 1)
}

/// The arm and its control in one test, because the tamper arm means
/// nothing without it: the same operand, the same attestation, the same
/// policy — one byte different.
#[test]
fn a_tampered_payload_is_refused_and_the_same_one_untampered_is_not() {
    let bytes = payload();
    let table = table(&bytes);
    let operand = address();
    let subject = subject(&operand, &SHAPE);

    // Control: the bytes that were measured.
    let verified = table
        .status_of(&subject, &recognising())
        .verified_against(&bytes);
    assert!(matches!(verified, AttestationStatus::Verified(_)));
    assert_eq!(verified.certificate().unwrap().radius(), 0.031);

    // Arm: ONE byte flipped, in the middle, changing no length and no
    // metadata — so every tuple check still passes and only the payload
    // can catch it.
    let mut tampered = bytes.clone();
    tampered[3000] ^= 0x01;
    let status = table
        .status_of(&subject, &recognising())
        .verified_against(&tampered);
    let AttestationStatus::Stale { cause, .. } = &status else {
        panic!("expected staleness, got {status:?}");
    };
    let StalenessCause::ContentDigest { attested, found } = cause else {
        panic!("expected a content mismatch, got {cause:?}");
    };
    assert_eq!(*attested, content_digest(&bytes));
    assert_eq!(*found, content_digest(&tampered));
    assert!(
        status.certificate().is_none(),
        "a tampered payload yields nothing"
    );

    // The refusal names the operand and both digests.
    let why = status.unavailable_because(&recognising()).unwrap();
    assert!(why.contains("re-encoded"), "{why}");
    assert!(why.contains(&content_digest(&bytes)), "{why}");
    assert!(status.attestation().unwrap().describe().contains(TENSOR));
}

/// **Bound is not Verified, and only Verified hands over a number.**
///
/// The single most important property of this step: a status that has
/// passed the tuple and recognition but has NOT seen the payload must not
/// yield a certificate. Otherwise every caller that forgets stage two
/// silently gets an unverified guarantee, which is the plane's whole
/// purpose defeated by an ergonomic accessor.
#[test]
fn a_bound_attestation_hands_over_nothing_until_its_bytes_are_checked() {
    let bytes = payload();
    let table = table(&bytes);
    let operand = address();
    let bound = table.status_of(&subject(&operand, &SHAPE), &recognising());

    assert!(matches!(bound, AttestationStatus::Bound(_)));
    assert!(
        bound.certificate().is_none(),
        "an unchecked payload must not yield a certificate"
    );
    let why = bound.unavailable_because(&recognising()).unwrap();
    assert!(why.contains("payload has not been checked"), "{why}");

    // The claim is reachable for a diagnostic, but only through the
    // attestation itself — never through the status that gates it.
    assert_eq!(bound.attestation().unwrap().claimed().radius(), 0.031);
}

/// Bytes cannot rescue a status that failed an earlier stage: an absent
/// attestation stays absent, a stale one stays stale for its original
/// reason, and an unrecognised one stays unrecognised.
///
/// Without this, `verified_against` would be a back door — hash the right
/// bytes and promote anything.
#[test]
fn checking_bytes_cannot_promote_a_status_that_already_failed() {
    let bytes = payload();
    let table = table(&bytes);
    let operand = address();

    // Absent stays absent, even against exactly the right bytes.
    let mut deeper = subject(&operand, &SHAPE);
    deeper.extent_depth = 4;
    let absent = table
        .status_of(&deeper, &recognising())
        .verified_against(&bytes);
    assert!(matches!(absent, AttestationStatus::Absent { .. }));
    assert!(absent.certificate().is_none());

    // Stale stays stale, and for the ORIGINAL cause — the content check
    // must not overwrite the reason it already had.
    let mut moved = subject(&operand, &SHAPE);
    moved.codec_revision = 9;
    let stale = table
        .status_of(&moved, &recognising())
        .verified_against(&bytes);
    let AttestationStatus::Stale { cause, .. } = &stale else {
        panic!("expected staleness");
    };
    assert!(matches!(cause, StalenessCause::CodecRevision { .. }));
    assert!(stale.certificate().is_none());

    // Unrecognised stays unrecognised.
    let stranger = RecognisedMethods::none().with_authority("someone-else");
    let unknown = table
        .status_of(&subject(&operand, &SHAPE), &stranger)
        .verified_against(&bytes);
    assert!(matches!(unknown, AttestationStatus::Unrecognised { .. }));
    assert!(unknown.certificate().is_none());
}

/// The digest is self-describing about its algorithm, and it actually
/// discriminates — including on the cases a length check would miss.
#[test]
fn the_digest_names_its_algorithm_and_separates_lookalike_payloads() {
    let bytes = payload();
    let digest = content_digest(&bytes);
    assert!(digest.starts_with("sha256:"), "{digest}");
    assert_eq!(digest.len(), "sha256:".len() + 64);
    assert_eq!(digest, content_digest(&bytes), "and it is deterministic");

    // Same length, different content — what a length check cannot see.
    let mut swapped = bytes.clone();
    swapped.swap(0, 1);
    assert_ne!(content_digest(&swapped), digest);

    // A transposition that preserves the multiset of bytes, and the sum.
    let mut reordered = bytes.clone();
    reordered.reverse();
    assert_ne!(content_digest(&reordered), digest);

    // Empty is a digest too, not an absence.
    assert!(content_digest(&[]).starts_with("sha256:"));
    assert_ne!(content_digest(&[]), digest);
}

/// Truncation and extension are caught, which matters because a partial
/// write is a likelier failure in practice than a deliberate flip.
#[test]
fn a_truncated_or_extended_payload_is_refused() {
    let bytes = payload();
    let table = table(&bytes);
    let operand = address();
    let subject = subject(&operand, &SHAPE);

    for broken in [&bytes[..bytes.len() - 1], &bytes[1..]] {
        let status = table
            .status_of(&subject, &recognising())
            .verified_against(broken);
        assert!(
            matches!(
                status,
                AttestationStatus::Stale {
                    cause: StalenessCause::ContentDigest { .. },
                    ..
                }
            ),
            "a short payload must not verify"
        );
    }

    let mut longer = bytes.clone();
    longer.push(0);
    let status = table
        .status_of(&subject, &recognising())
        .verified_against(&longer);
    assert!(matches!(
        status,
        AttestationStatus::Stale {
            cause: StalenessCause::ContentDigest { .. },
            ..
        }
    ));
}
