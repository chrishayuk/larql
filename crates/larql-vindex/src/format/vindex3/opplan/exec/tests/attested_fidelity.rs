//! Step 5 of ATTESTATION-1: B3 — an attested radius reaching the floor.
//!
//! The decisive regression, with the numbers the contract names:
//!
//! ```text
//! attested parent radius: 0.003
//! shallow codebook:       0.003  → composed 0.006 → rejected
//! deeper codebook:        0.001  → composed 0.004 → selected
//! floor:                  0.005
//! ```
//!
//! Neither half alone decides it: 0.003 is inside the floor and so is
//! each codebook bound. Only the COMPOSITION separates them, which is
//! what makes this a witness for B3 rather than for arithmetic.

use std::collections::BTreeMap;

use super::super::accounting::RepresentationFloor;
use super::super::attested_fidelity::{admit, VerifiedEvidence};
use super::super::operands::{OperandSource, OperandStore};
use super::super::realization::ExtentOption;
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::auxiliary_references::OperandAddress;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::{
    DomainId, ExtentCertificate, FidelityCertificate, MetricId, RepresentationExtent,
};
use crate::format::vindex3::representation_attestations::recognition::RecognisedMethods;
use crate::format::vindex3::representation_attestations::tuple::AttestedSubject;
use crate::format::vindex3::representation_attestations::{
    content_digest, AttestationBinding, AttestationMethod, AttestationTable,
    RepresentationAttestations, StoredAttestation, StoredId,
};

const TENSOR: &str = "0.mlp.gate_proj.weight";
const CODEBOOK: &str = "codebook";
const BASELINE: &str = "VQ8_SHARED@1/terminal";
const AUTHORITY: &str = "larql-encoder";
const METHOD: &str = "measured-rms";

/// The contract's numbers, named so the test reads as the contract does.
const ATTESTED_PARENT: f64 = 0.003;
const SHALLOW_CODEBOOK: f64 = 0.003;
const DEEPER_CODEBOOK: f64 = 0.001;
const FLOOR: f64 = 0.005;

fn certificate(radius: f64) -> FidelityCertificate {
    FidelityCertificate::new(MetricId::relative_rms(), DomainId::finite_normals(), radius).unwrap()
}

fn recognising() -> RecognisedMethods {
    RecognisedMethods::none()
        .with_authority(AUTHORITY)
        .with_method(METHOD, 1)
}

/// A container holding one real operand, so verification hashes bytes
/// that actually exist rather than a fixture's idea of them.
struct Built {
    dir: tempfile::TempDir,
    object: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl Built {
    fn container(&self) -> std::path::PathBuf {
        self.dir.path().join("container")
    }

    fn operand(&self) -> OperandRef {
        OperandRef {
            object: self.object.clone(),
            tensor: TENSOR.to_string(),
            dtype: String::new(),
            shape: self.shape.clone(),
        }
    }

    fn address(&self) -> OperandAddress {
        OperandAddress::new(&self.object, TENSOR)
    }

    fn store(&self) -> OperandStore {
        let inspection = inspect_container(&self.container(), false).unwrap();
        OperandStore::open(&self.container(), &inspection).unwrap()
    }

    /// What the container says about the operand, for admission.
    fn subject<'a>(&'a self, baselines: &'a BTreeMap<String, String>) -> AttestedSubject<'a> {
        AttestedSubject {
            operand: Box::leak(Box::new(self.address())),
            extent_depth: 0,
            codec_family: "F32",
            codec_revision: 1,
            shape: &self.shape,
            auxiliary_baselines: baselines.clone(),
            expected_source_digest: None,
            expected_recipe: None,
        }
    }

    fn attestation(&self, radius: f64) -> AttestationTable {
        RepresentationAttestations::new(vec![StoredAttestation {
            binding: AttestationBinding {
                operand: self.address(),
                extent_depth: 0,
                codec_family: "F32".into(),
                codec_revision: 1,
                shape: self.shape.clone(),
                content_digest: content_digest(&self.bytes),
                source_digest: "sha256:source".into(),
                auxiliary_baselines: BTreeMap::from([(CODEBOOK.to_string(), BASELINE.to_string())]),
                recipe: "kmeans/lloyd@8".into(),
            },
            method: AttestationMethod::new(AUTHORITY, StoredId::new(METHOD, 1)),
            metric: StoredId::new("relative-rms", 1),
            domain: StoredId::new("finite-normals", 1),
            radius,
        }])
        .judge()
        .unwrap()
    }
}

fn build() -> Built {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    let container = dir.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "att");

    let index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(container.join(INDEX_JSON)).unwrap())
            .unwrap();
    let (mut object, mut shape, mut bytes) = (String::new(), Vec::new(), Vec::new());
    for entry in index.representations.values() {
        let path = container.join(&entry.segment);
        let (header, payload_start) = read_segment_header(&path).unwrap();
        let Some(tensor) = header.tensors.iter().find(|t| t.name == TENSOR) else {
            continue;
        };
        object = entry.object.clone();
        shape = tensor.shape.clone();
        let file = std::fs::read(&path).unwrap();
        let payload = &file[payload_start as usize..];
        bytes = payload[tensor.offset as usize..(tensor.offset + tensor.len) as usize].to_vec();
    }
    assert!(!bytes.is_empty(), "the fixture must hold {TENSOR}");
    Built {
        dir,
        object,
        shape,
        bytes,
    }
}

fn baselines() -> BTreeMap<String, String> {
    BTreeMap::from([(CODEBOOK.to_string(), BASELINE.to_string())])
}

/// **The decisive regression.** One artifact, one floor, three outcomes.
///
/// If selection never changes or refuses on the composition, ATTESTATION-1
/// has not delivered usable fidelity authority — so this test is the
/// wave's exit condition in miniature.
#[test]
fn a_floor_is_met_or_missed_by_the_composition_and_not_by_either_half() {
    let built = build();
    let store = built.store();
    let source = OperandSource::from(&store);
    let table = built.attestation(ATTESTED_PARENT);
    let baselines = baselines();
    let subjects = [(built.subject(&baselines), built.operand())];

    let evidence = admit(&table, &subjects, &recognising(), &source)
        .verify(&table, &source)
        .expect("verification");
    assert_eq!(evidence.count(), 1, "the parent is attested and verified");

    let key = (built.address(), 0);
    let floor = RepresentationFloor::Within(FLOOR);
    // A depth the options are NOT at, so `admits` cannot wave them
    // through as "terminal" and actually has to judge the radius.
    let terminal = RepresentationExtent::BASE;

    // NEITHER HALF DECIDES IT. The attested parent alone is inside the
    // floor, and so is each codebook bound on its own. Checked at COMPILE
    // time: if someone edits these constants so that one half alone
    // decides the outcome, the build fails rather than the test quietly
    // becoming a witness for arithmetic.
    const { assert!(ATTESTED_PARENT < FLOOR) };
    const { assert!(SHALLOW_CODEBOOK < FLOOR) };
    const { assert!(DEEPER_CODEBOOK < FLOOR) };
    const { assert!(ATTESTED_PARENT + SHALLOW_CODEBOOK > FLOOR) };
    const { assert!(ATTESTED_PARENT + DEEPER_CODEBOOK < FLOOR) };

    // Shallow codebook: 0.003 + 0.003 = 0.006, over the 0.005 floor.
    let shallow = evidence
        .derive(&key, None, &[certificate(SHALLOW_CODEBOOK)], TENSOR, "F32")
        .unwrap()
        .expect("a composed certificate");
    assert!((shallow.radius() - 0.006).abs() < 1e-12, "{shallow:?}");
    assert!(
        !floor.admits(&option(shallow.clone(), 1), terminal),
        "0.006 must not satisfy a 0.005 floor"
    );

    // Deeper codebook: 0.003 + 0.001 = 0.004, inside it.
    let deeper = evidence
        .derive(&key, None, &[certificate(DEEPER_CODEBOOK)], TENSOR, "F32")
        .unwrap()
        .expect("a composed certificate");
    assert!((deeper.radius() - 0.004).abs() < 1e-12, "{deeper:?}");
    assert!(
        floor.admits(&option(deeper, 1), terminal),
        "0.004 must satisfy a 0.005 floor"
    );

    // And the point of the pair: a floor that consulted only the parent
    // would have admitted BOTH, which is the silent over-promise.
    assert!(
        floor.admits(&option(certificate(ATTESTED_PARENT), 1), terminal),
        "the parent alone passes, which is exactly why composing matters"
    );
}

/// An `ExtentOption` at a non-terminal depth carrying `radius`, so the
/// floor judges the bound rather than waving it through as terminal.
fn option(radius: FidelityCertificate, depth: u32) -> ExtentOption {
    ExtentOption {
        certificate: ExtentCertificate {
            extent: RepresentationExtent { depth },
            bits_per_weight: 32.0,
            radius: Some(radius),
        },
        stored_bytes: None,
    }
}

/// **No provisional selection from `Bound` evidence.**
///
/// Admission returns candidates that carry no certificate at all, and
/// `derive` takes only `VerifiedEvidence` — which nothing but `verify`
/// can construct. A plan built on unverified claims and corrected later
/// would, however briefly, assert a guarantee it had not earned, and
/// "briefly" is not a property a guarantee can have.
#[test]
fn an_admitted_candidate_carries_no_claim_a_selection_could_use() {
    let built = build();
    let store = built.store();
    let source = OperandSource::from(&store);
    let table = built.attestation(ATTESTED_PARENT);
    let baselines = baselines();
    let subjects = [(built.subject(&baselines), built.operand())];

    let admitted = admit(&table, &subjects, &recognising(), &source);
    assert_eq!(admitted.candidates().len(), 1);
    assert!(admitted.rejected().is_empty());
    // The candidate names what must be hashed and nothing more — there is
    // no radius on it for a caller to reach.
    assert_eq!(admitted.candidates()[0].operand.tensor, TENSOR);

    // Empty evidence derives from the DECLARED certificate only, never
    // from a candidate.
    let empty = VerifiedEvidence::none(&source);
    assert_eq!(empty.count(), 0);
    assert_eq!(empty.read_to_prepare(), 0);
    let key = (built.address(), 0);
    assert!(empty
        .derive(&key, None, &[], TENSOR, "F32")
        .unwrap()
        .is_none());
    let declared = certificate(0.5);
    assert_eq!(
        empty
            .derive(&key, Some(&declared), &[], TENSOR, "F32")
            .unwrap()
            .unwrap()
            .radius(),
        0.5
    );
}

/// Phase one refuses without reading a byte, and says why.
#[test]
fn admission_refuses_before_any_payload_read() {
    let built = build();
    let store = built.store();
    let source = OperandSource::from(&store);
    let table = built.attestation(ATTESTED_PARENT);
    let baselines = baselines();

    // Unrecognised: the artifact is perfect, this build is not equipped.
    let subjects = [(built.subject(&baselines), built.operand())];
    let before = store.load_count();
    let admitted = admit(&table, &subjects, &RecognisedMethods::none(), &source);
    assert_eq!(store.load_count(), before, "admission read nothing");
    assert!(admitted.candidates().is_empty());
    assert_eq!(admitted.rejected().len(), 1);
    assert!(admitted.rejected()[0].why.contains("unrecognised"));

    // Stale: the same, by a different cause.
    let mut moved = built.subject(&baselines);
    moved.codec_revision = 9;
    let subjects = [(moved, built.operand())];
    let admitted = admit(&table, &subjects, &recognising(), &source);
    assert_eq!(store.load_count(), before, "admission still read nothing");
    assert!(admitted.rejected()[0].why.contains("stale"));

    // And verifying an admission that admitted nothing reads nothing.
    let evidence = admitted.verify(&table, &source).unwrap();
    assert_eq!(evidence.read_to_prepare(), 0);
    assert_eq!(evidence.count(), 0);
}

/// Verification reads are PREPARATION WORK and are reported as such.
///
/// A ledger that hid them would price a plan as though guarantees were
/// free. The number is the operand's stored length, because that is
/// exactly what had to be hashed.
#[test]
fn verification_reads_are_reported_as_preparation_work() {
    let built = build();
    let store = built.store();
    let source = OperandSource::from(&store);
    let table = built.attestation(ATTESTED_PARENT);
    let baselines = baselines();
    let subjects = [(built.subject(&baselines), built.operand())];

    let evidence = admit(&table, &subjects, &recognising(), &source)
        .verify(&table, &source)
        .unwrap();
    assert_eq!(
        evidence.read_to_prepare(),
        built.bytes.len() as u64,
        "the whole operand was hashed, and the ledger says so"
    );
}

/// **Time of check, time of use.** Evidence verified against one
/// generation must not settle another.
///
/// Hashing proves nothing about the bytes a later decode reads unless
/// both come from the same immutable object, so the stamp travels with
/// the evidence and a moved source is refused rather than trusted.
#[test]
fn evidence_does_not_carry_across_a_changed_generation() {
    let built = build();
    let store = built.store();
    let source = OperandSource::from(&store);
    let table = built.attestation(ATTESTED_PARENT);
    let baselines = baselines();
    let subjects = [(built.subject(&baselines), built.operand())];

    let evidence = admit(&table, &subjects, &recognising(), &source)
        .verify(&table, &source)
        .unwrap();
    evidence
        .ensure_current_for(&source)
        .expect("the source has not moved");

    // A DIFFERENT store over the same container is a different generation:
    // same bytes on disk, but not the object the evidence was settled
    // against, and nothing here may assume the file did not change.
    let reopened = built.store();
    let elsewhere = OperandSource::from(&reopened);
    let err = evidence
        .ensure_current_for(&elsewhere)
        .expect_err("a different opened object must be refused")
        .to_string();
    assert!(err.contains("different operand generation"), "{err}");
    assert!(err.contains("Re-verify"), "{err}");
}

/// Composition refuses across metrics here exactly as it does in the
/// codec plane: an attested relative-RMS bound and a dependency stated in
/// other terms do not add.
#[test]
fn an_attested_bound_does_not_compose_with_a_foreign_metric() {
    let built = build();
    let store = built.store();
    let source = OperandSource::from(&store);
    let table = built.attestation(ATTESTED_PARENT);
    let baselines = baselines();
    let subjects = [(built.subject(&baselines), built.operand())];
    let evidence = admit(&table, &subjects, &recognising(), &source)
        .verify(&table, &source)
        .unwrap();

    let foreign = FidelityCertificate::new(
        MetricId::new("max-absolute", 1).unwrap(),
        DomainId::finite_normals(),
        0.001,
    )
    .unwrap();
    let key = (built.address(), 0);
    assert!(
        evidence
            .derive(&key, None, &[foreign], TENSOR, "F32")
            .is_err(),
        "adding a max-absolute bound to an L2 one would manufacture a guarantee"
    );
}

/// An attested claim REPLACES the codec's declared one rather than
/// widening it: they are two statements about the same quantity, one
/// measured on this instance and one derived from the scheme.
#[test]
fn a_measured_claim_replaces_the_declared_one_rather_than_adding_to_it() {
    let built = build();
    let store = built.store();
    let source = OperandSource::from(&store);
    let table = built.attestation(ATTESTED_PARENT);
    let baselines = baselines();
    let subjects = [(built.subject(&baselines), built.operand())];
    let evidence = admit(&table, &subjects, &recognising(), &source)
        .verify(&table, &source)
        .unwrap();

    let key = (built.address(), 0);
    let declared = certificate(0.25);
    let derived = evidence
        .derive(&key, Some(&declared), &[], TENSOR, "F32")
        .unwrap()
        .unwrap();
    assert_eq!(
        derived.radius(),
        ATTESTED_PARENT,
        "the measurement is used, and the scheme's looser bound is not added to it"
    );
}

/// The OTHER time-of-check hole: the source moving between phase one and
/// phase two, before any hashing has happened at all.
///
/// `ensure_current_for` guards check-to-use; this guards admit-to-verify.
/// Both are needed, and the second is easier to miss because nothing has
/// been read yet when it opens.
#[test]
fn admission_and_verification_must_happen_against_one_generation() {
    let built = build();
    let store = built.store();
    let source = OperandSource::from(&store);
    let table = built.attestation(ATTESTED_PARENT);
    let baselines = baselines();
    let subjects = [(built.subject(&baselines), built.operand())];

    let admitted = admit(&table, &subjects, &recognising(), &source);
    assert_eq!(admitted.candidates().len(), 1);

    let reopened = built.store();
    let elsewhere = OperandSource::from(&reopened);
    let err = admitted
        .verify(&table, &elsewhere)
        .expect_err("a moved source must not be verified against")
        .to_string();
    assert!(
        err.contains("changed between attestation admission and verification"),
        "{err}"
    );
}

/// A payload that changes after admission is refused AT VERIFICATION, and
/// the refusal is kept rather than dropped.
///
/// This is the phase-two failure the whole boundary exists for: admission
/// said the claim addresses the right representation, and the bytes then
/// said it does not.
#[test]
fn a_payload_that_fails_verification_is_recorded_not_silently_dropped() {
    let built = build();
    let store = built.store();
    let source = OperandSource::from(&store);
    let baselines = baselines();
    let subjects = [(built.subject(&baselines), built.operand())];

    // An attestation whose content digest is for OTHER bytes: every
    // metadata check passes, so only the payload can catch it.
    let mut wrong = built.attestation(ATTESTED_PARENT);
    wrong = {
        let mut other = built.bytes.clone();
        other[0] ^= 0xFF;
        let digest = content_digest(&other);
        let mut stored = RepresentationAttestations::new(vec![]);
        stored.attestations.push(StoredAttestation {
            binding: AttestationBinding {
                operand: built.address(),
                extent_depth: 0,
                codec_family: "F32".into(),
                codec_revision: 1,
                shape: built.shape.clone(),
                content_digest: digest,
                source_digest: "sha256:source".into(),
                auxiliary_baselines: BTreeMap::from([(CODEBOOK.to_string(), BASELINE.to_string())]),
                recipe: "kmeans/lloyd@8".into(),
            },
            method: AttestationMethod::new(AUTHORITY, StoredId::new(METHOD, 1)),
            metric: StoredId::new("relative-rms", 1),
            domain: StoredId::new("finite-normals", 1),
            radius: ATTESTED_PARENT,
        });
        let _ = wrong;
        stored.judge().unwrap()
    };

    let admitted = admit(&wrong, &subjects, &recognising(), &source);
    assert_eq!(admitted.candidates().len(), 1, "metadata admits it");

    let evidence = admitted.verify(&wrong, &source).unwrap();
    assert_eq!(evidence.count(), 0, "the bytes refuse it");
    assert_eq!(evidence.refused().len(), 1);
    assert!(
        evidence.refused()[0].why.contains("stale"),
        "{:?}",
        evidence.refused()
    );
    assert!(evidence.refused()[0].why.contains("re-encoded"));
    // It still read the payload, and says so.
    assert_eq!(evidence.read_to_prepare(), built.bytes.len() as u64);

    // And no certificate is derivable from it.
    let key = (built.address(), 0);
    assert!(evidence
        .derive(&key, None, &[], TENSOR, "F32")
        .unwrap()
        .is_none());
}

/// An attestation withdrawn between the phases is refused by name rather
/// than silently vanishing from the candidate set.
#[test]
fn an_attestation_withdrawn_between_phases_is_refused_by_name() {
    let built = build();
    let store = built.store();
    let source = OperandSource::from(&store);
    let table = built.attestation(ATTESTED_PARENT);
    let baselines = baselines();
    let subjects = [(built.subject(&baselines), built.operand())];

    let admitted = admit(&table, &subjects, &recognising(), &source);
    assert_eq!(admitted.candidates().len(), 1);

    // Verify against an EMPTY table: the row admission saw is gone.
    let withdrawn = RepresentationAttestations::new(vec![]).judge().unwrap();
    let evidence = admitted.verify(&withdrawn, &source).unwrap();
    assert_eq!(evidence.count(), 0);
    assert_eq!(evidence.refused().len(), 1);
    assert!(
        evidence.refused()[0].why.contains("withdrawn"),
        "{:?}",
        evidence.refused()
    );
    // Nothing was read, because there was nothing left to check.
    assert_eq!(evidence.read_to_prepare(), 0);
}

/// Refusals from phase one survive into the evidence, so one place lists
/// everything that got no guarantee and why.
#[test]
fn phase_one_refusals_are_carried_into_the_evidence() {
    let built = build();
    let store = built.store();
    let source = OperandSource::from(&store);
    let table = built.attestation(ATTESTED_PARENT);
    let baselines = baselines();

    let mut moved = built.subject(&baselines);
    moved.codec_revision = 9;
    let subjects = [(moved, built.operand())];

    let evidence = admit(&table, &subjects, &recognising(), &source)
        .verify(&table, &source)
        .unwrap();
    assert_eq!(evidence.count(), 0);
    assert_eq!(
        evidence.refused().len(),
        1,
        "the metadata refusal is still listed"
    );
    assert!(evidence.refused()[0].why.contains("stale"));
}
