//! Step 1 of ATTESTATION-1: the sidecar, its binding tuple, and every
//! way a row can be malformed BEFORE any container or codec is in hand.
//!
//! What is deliberately NOT here: whether an attestation matches the
//! artifact it claims to describe (the tuple check, step 2), whether its
//! authority is recognised (step 3), and whether its content digest still
//! holds (step 4). Those need a container, a registry and bytes
//! respectively. This file needs none of the three, which is the point —
//! a malformed attestation is refusable on its own terms.

use std::collections::BTreeMap;

use super::*;

const OBJECT: &str = "block.0";
const TENSOR: &str = "mlp.gate_proj.weight";
const CODEBOOK: &str = "codebook";

fn address() -> OperandAddress {
    OperandAddress::new(OBJECT, TENSOR)
}

/// A well-formed attestation: a VQ operand whose assignment error was
/// measured against its terminal codebook.
fn stored() -> StoredAttestation {
    StoredAttestation {
        binding: AttestationBinding {
            operand: address(),
            extent_depth: 0,
            codec_family: "VQ8_SHARED".into(),
            codec_revision: 1,
            shape: vec![64, 101],
            content_digest: "sha256:codes".into(),
            source_digest: "sha256:source".into(),
            auxiliary_baselines: BTreeMap::from([(
                CODEBOOK.to_string(),
                "sha256:terminal-codebook".to_string(),
            )]),
            recipe: "kmeans/lloyd@8".into(),
        },
        method: AttestationMethod::new("larql-encoder", StoredId::new("measured-rms", 1)),
        metric: StoredId::new("relative-rms", 1),
        domain: StoredId::new("finite-normals", 1),
        radius: 0.031,
    }
}

fn table_of(rows: Vec<StoredAttestation>) -> Result<AttestationTable, crate::error::VindexError> {
    RepresentationAttestations::new(rows).judge()
}

/// The arm: a well-formed row judges, and what comes back is the CODEC
/// PLANE's certificate — not a parallel vocabulary that would give
/// composition two algebras.
#[test]
fn an_attestation_judges_into_the_same_certificate_a_codec_declares() {
    let table = table_of(vec![stored()]).unwrap();
    assert_eq!(table.stated(), 1);

    let judged = table.at(&address(), 0).expect("attested at depth 0");
    let certificate = judged.claimed();
    assert_eq!(certificate.radius(), 0.031);
    assert_eq!(*certificate.metric(), MetricId::relative_rms());
    assert_eq!(*certificate.domain(), DomainId::finite_normals());

    // It composes with a declared certificate through the ordinary
    // algebra, which is the whole reason it is this type.
    let dependency =
        FidelityCertificate::new(MetricId::relative_rms(), DomainId::finite_normals(), 0.004)
            .unwrap();
    let composed = certificate
        .widened_by(&dependency, TENSOR, "VQ8_SHARED")
        .unwrap();
    assert!((composed.radius() - 0.035).abs() < 1e-12);
}

/// Absence, staleness and unrecognition are three different answers, and
/// none of them is zero. Step 1 can only witness the first, so it does.
#[test]
fn a_container_that_attests_nothing_says_so_and_does_not_say_zero() {
    let empty = AttestationTable::empty();
    assert_eq!(empty.stated(), 0);
    assert!(empty.at(&address(), 0).is_none());
    assert!(!empty.attests(&address()));
    assert!(empty.depths(&address()).is_empty());

    // And a table with rows still answers "no" for an operand it does not
    // mention, rather than offering the nearest thing it has.
    let table = table_of(vec![stored()]).unwrap();
    let other = OperandAddress::new(OBJECT, "mlp.down_proj.weight");
    assert!(!table.attests(&other));
    assert!(table.at(&other, 0).is_none());
}

/// An attestation is per EXTENT: a measurement at one depth says nothing
/// about another, and the table must not answer for a depth nobody
/// measured.
#[test]
fn a_measurement_at_one_extent_is_not_offered_for_another() {
    let mut deep = stored();
    deep.binding.extent_depth = 2;
    deep.radius = 0.004;
    let table = table_of(vec![stored(), deep]).unwrap();

    assert_eq!(table.depths(&address()), vec![0, 2]);
    assert_eq!(table.at(&address(), 0).unwrap().claimed().radius(), 0.031);
    assert_eq!(table.at(&address(), 2).unwrap().claimed().radius(), 0.004);
    // Depth 1 was never measured. The table holds a tighter bound at 2 and
    // a looser one at 0, and offers NEITHER.
    assert!(table.at(&address(), 1).is_none());
    assert!(table.attests(&address()), "though something is attested");
}

/// Two measurements of one operand at one extent state no measurement:
/// nothing says which was meant, and picking one would make a container's
/// guarantee depend on its serialisation order.
#[test]
fn one_operand_at_one_extent_cannot_be_attested_twice() {
    let mut second = stored();
    second.radius = 0.5;
    second.method = AttestationMethod::new("someone-else", StoredId::new("guessed", 1));

    let err = table_of(vec![stored(), second]).unwrap_err().to_string();
    assert!(err.contains(TENSOR), "{err}");
    assert!(err.contains("attested twice"), "{err}");
    // Both claimants are named, because "which one" is the question the
    // reader now has.
    assert!(err.contains("larql-encoder"), "{err}");
    assert!(err.contains("someone-else"), "{err}");
}

/// A schema another build wrote is refused, naming both versions.
///
/// Sharper here than for a reference table: reading an unknown
/// attestation schema optimistically would mean honouring a guarantee
/// under rules this build does not implement.
#[test]
fn a_schema_another_build_wrote_is_refused_naming_both_versions() {
    let table = RepresentationAttestations {
        schema: REPRESENTATION_ATTESTATIONS_SCHEMA + 7,
        attestations: vec![stored()],
    };
    let err = table.judge().unwrap_err().to_string();
    assert!(
        err.contains(&(REPRESENTATION_ATTESTATIONS_SCHEMA + 7).to_string()),
        "{err}"
    );
    assert!(
        err.contains(&REPRESENTATION_ATTESTATIONS_SCHEMA.to_string()),
        "{err}"
    );
}

/// One way of emptying a field, for the table-driven refusal arm.
type Break = Box<dyn Fn(&mut StoredAttestation)>;

/// Every part of the binding is part of the identity, so an empty one is
/// not a blank field — it is an attestation about nothing in particular.
#[test]
fn every_empty_part_of_the_binding_is_refused_by_name() {
    let cases: Vec<(&str, Break)> = vec![
        (
            "codec family",
            Box::new(|a: &mut StoredAttestation| a.binding.codec_family.clear()),
        ),
        (
            "content digest",
            Box::new(|a: &mut StoredAttestation| a.binding.content_digest.clear()),
        ),
        (
            "source digest",
            Box::new(|a: &mut StoredAttestation| a.binding.source_digest.clear()),
        ),
        (
            "encoder recipe",
            Box::new(|a: &mut StoredAttestation| a.binding.recipe.clear()),
        ),
        (
            "attesting authority",
            Box::new(|a: &mut StoredAttestation| a.method.authority.clear()),
        ),
    ];
    for (what, break_it) in cases {
        let mut row = stored();
        break_it(&mut row);
        let err = table_of(vec![row]).unwrap_err().to_string();
        assert!(err.contains(what), "expected `{what}` to be named: {err}");
        assert!(err.contains(TENSOR), "expected the operand named: {err}");
    }
}

/// An attestation about no operand is about nothing.
#[test]
fn an_attestation_without_an_operand_is_refused() {
    for (object, tensor) in [("", TENSOR), (OBJECT, ""), ("   ", TENSOR)] {
        let mut row = stored();
        row.binding.operand = OperandAddress::new(object, tensor);
        let err = table_of(vec![row]).unwrap_err().to_string();
        assert!(err.contains("empty object or tensor"), "{err}");
    }
}

/// An operand with no elements has no error to measure — and a shape with
/// a zero dimension is the case a length check would miss.
#[test]
fn a_shape_with_nothing_in_it_is_refused() {
    for shape in [vec![], vec![0], vec![64, 0], vec![0, 101]] {
        let mut row = stored();
        row.binding.shape = shape.clone();
        let err = table_of(vec![row]).unwrap_err().to_string();
        assert!(err.contains("no elements"), "{shape:?}: {err}");
    }
}

/// A dependency baseline that names nothing cannot be checked against
/// anything — on either side of the pair.
#[test]
fn an_empty_dependency_baseline_is_refused() {
    for (name, baseline) in [("", "sha256:x"), (CODEBOOK, ""), (CODEBOOK, "  ")] {
        let mut row = stored();
        row.binding.auxiliary_baselines =
            BTreeMap::from([(name.to_string(), baseline.to_string())]);
        let err = table_of(vec![row]).unwrap_err().to_string();
        assert!(
            err.contains("empty dependency baseline"),
            "{name}/{baseline}: {err}"
        );
    }
}

/// A metric, a domain or a method this build never heard of is
/// EXPRESSIBLE — that is VQ-1's rule and it holds here. What is refused is
/// a MALFORMED id, which is a different failure: unknown is a thing the
/// container may say, unspellable is not.
#[test]
fn an_unknown_id_is_expressible_and_a_malformed_one_is_refused() {
    let mut foreign = stored();
    foreign.metric = StoredId::new("someone-elses-metric", 3);
    foreign.domain = StoredId::new("all-reals", 1);
    foreign.method.method = StoredId::new("their-harness", 9);
    let table = table_of(vec![foreign]).unwrap();
    let judged = table.at(&address(), 0).unwrap();
    assert_eq!(
        judged.claimed().metric().to_string(),
        "someone-elses-metric@3"
    );
    assert_eq!(judged.method.method.name, "their-harness");

    // Malformed, by each rule SemanticId enforces.
    for bad in ["", "has space", "has@at", "nonascii-é"] {
        let mut row = stored();
        row.metric = StoredId::new(bad, 1);
        assert!(
            table_of(vec![row]).is_err(),
            "metric `{bad}` should be refused"
        );

        let mut row = stored();
        row.method.method = StoredId::new(bad, 1);
        let err = table_of(vec![row]).unwrap_err().to_string();
        assert!(
            err.contains("method id is malformed"),
            "method `{bad}`: {err}"
        );
    }
}

/// A radius the certificate would not accept is refused where the
/// container is still in hand, not where the number is later used.
#[test]
fn a_radius_that_is_not_a_number_is_refused_at_the_table() {
    for bad in [f64::NAN, f64::INFINITY, -1.0, -0.0001] {
        let mut row = stored();
        row.radius = bad;
        assert!(
            table_of(vec![row]).is_err(),
            "radius {bad} should be refused"
        );
    }
    // And the boundary that IS admissible: an exact representation
    // measured as exact.
    let mut zero = stored();
    zero.radius = 0.0;
    assert_eq!(
        table_of(vec![zero])
            .unwrap()
            .at(&address(), 0)
            .unwrap()
            .claimed()
            .radius(),
        0.0
    );
}

/// The table survives the round trip a container puts it through, and a
/// row written by this build is stamped with this build's schema.
#[test]
fn a_table_round_trips_through_the_form_a_container_stores() {
    let written = RepresentationAttestations::new(vec![stored()]);
    assert_eq!(written.schema, REPRESENTATION_ATTESTATIONS_SCHEMA);

    let text = serde_json::to_string_pretty(&written).unwrap();
    let read: RepresentationAttestations = serde_json::from_str(&text).unwrap();
    assert_eq!(read, written);
    assert_eq!(read.judge().unwrap(), written.judge().unwrap());

    // The optional map is omitted when empty rather than written as `{}`,
    // so a container that attests an operand with no dependencies is not
    // larger than one that could not have had any.
    let mut bare = stored();
    bare.binding.auxiliary_baselines.clear();
    let text = serde_json::to_string(&RepresentationAttestations::new(vec![bare])).unwrap();
    assert!(!text.contains("auxiliary_baselines"), "{text}");
}

/// A table the index names and the container does not hold is a refusal,
/// not an empty table — the same rule the reference table follows, and
/// for a sharper reason: silently treating a missing attestation file as
/// "nothing attested" would turn a packaging mistake into a quiet loss of
/// every guarantee.
#[test]
fn a_table_the_index_names_and_the_container_lacks_is_a_refusal() {
    let dir = tempfile::tempdir().unwrap();
    let err = RepresentationAttestations::read(dir.path(), "representation_attestations.json")
        .unwrap_err()
        .to_string();
    assert!(err.contains("cannot be read"), "{err}");
    assert!(err.contains("named by the index"), "{err}");

    // Present but not a table of this kind: also a refusal, and a
    // different one, because "malformed" and "missing" send a reader to
    // different places.
    let path = dir.path().join("representation_attestations.json");
    std::fs::write(&path, "[1, 2, 3]").unwrap();
    let err = RepresentationAttestations::read(dir.path(), "representation_attestations.json")
        .unwrap_err()
        .to_string();
    assert!(err.contains("not readable as such"), "{err}");

    // And the arm that proves the reader works at all.
    std::fs::write(
        &path,
        serde_json::to_string(&RepresentationAttestations::new(vec![stored()])).unwrap(),
    )
    .unwrap();
    let table =
        RepresentationAttestations::read(dir.path(), "representation_attestations.json").unwrap();
    assert_eq!(table.stated(), 1);
}

/// An attestation names itself completely enough to be argued with: the
/// operand, the extent, the claim and whose word it is.
#[test]
fn an_attestation_describes_itself_for_a_refusal() {
    let table = table_of(vec![stored()]).unwrap();
    let described = table.at(&address(), 0).unwrap().describe();
    assert!(described.contains(TENSOR), "{described}");
    assert!(described.contains("depth 0"), "{described}");
    assert!(described.contains("larql-encoder"), "{described}");
    assert!(described.contains("measured-rms"), "{described}");
}
