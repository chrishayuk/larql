//! The container path rule, and that `index.json` enforces it while parsing.

use std::collections::BTreeMap;
use std::path::Path;

use super::*;
use crate::format::capability::authority::Fidelity;
use crate::format::vindex3::index::{RepresentationEntry, Vindex3Index};
use crate::format::vindex3::variants::{RegionSetVariants, StoredVariant, VariantCatalogue};

/// Names real writers emit (`encode`, `write::segment_path`, the sidecars).
const ACCEPTED: &[&str] = &[
    "index.json",
    "system_graph.json",
    "segments/target.embedding.bin",
    "routed/layer_000",
    "routed/layer_000.q6k",
];

/// Every way a name can leave the root, or mean different files on
/// different hosts.
const REFUSED: &[&str] = &[
    "/etc/passwd",
    "../outside.bin",
    "segments/../../outside.bin",
    "segments/..",
    "./segments/a.bin",
    ".",
    "segments\\..\\..\\outside.bin",
    "segments/a\0.bin",
];

#[test]
fn names_real_writers_emit_are_accepted() {
    for name in ACCEPTED {
        check(name).unwrap_or_else(|e| panic!("`{name}` must be accepted: {e}"));
    }
}

#[test]
fn every_escaping_name_is_refused_by_name() {
    for name in REFUSED {
        match check(name) {
            Err(VindexError::UnsafeContainerPath { name: refused, .. }) => {
                assert_eq!(&refused, name)
            }
            other => panic!("`{name:?}` must be refused as unsafe, got {other:?}"),
        }
    }
}

#[test]
fn each_rule_reports_its_own_reason() {
    let reason = |name: &str| match check(name) {
        Err(VindexError::UnsafeContainerPath { reason, .. }) => reason,
        other => panic!("`{name:?}` must be refused, got {other:?}"),
    };
    assert_eq!(reason("a\0b"), NUL);
    assert_eq!(reason("a\\b"), BACKSLASH);
    assert_eq!(reason("/abs"), NOT_RELATIVE);
    assert_eq!(reason("a/../b"), NOT_RELATIVE);
}

#[test]
fn an_empty_name_is_left_to_the_completeness_checks() {
    // It names the root, which escapes nothing; the layers that require
    // the field refuse it there, naming the field and the entry.
    check("").expect("emptiness is not a containment question");
}

#[test]
fn resolve_joins_only_a_contained_name() {
    let root = Path::new("/container");
    assert_eq!(
        resolve(root, "segments/a.bin").unwrap(),
        root.join("segments/a.bin")
    );
    // An absolute name would REPLACE the root under a bare `join`.
    assert!(resolve(root, "/etc/passwd").is_err());
    assert!(resolve(root, "../a.bin").is_err());
}

// ─── `index.json` enforces the rule while it is parsed ────────────────

fn entry(segment: &str) -> RepresentationEntry {
    RepresentationEntry {
        object: "target.embedding".to_string(),
        encoding: "BF16".to_string(),
        segment: segment.to_string(),
        tensor_count: 1,
        payload_bytes: 0,
        payload_sha256: String::new(),
        segment_sha256: String::new(),
        compiled_from: None,
        codec: None,
        source_representation_digest: None,
        encoder: None,
    }
}

/// A well-formed index naming a file in every path-bearing field.
fn index_json() -> serde_json::Value {
    let mut index = Vindex3Index::new(
        "fixture",
        "direct-moe",
        8,
        1,
        "moe_manifest.json",
        BTreeMap::from([("routed/layer_000".to_string(), 1)]),
    );
    index.system_graph = Some("system_graph.json".to_string());
    index.auxiliary_references = Some("auxiliary_references.json".to_string());
    index.representation_attestations = Some("representation_attestations.json".to_string());
    index.weight_patches = Some("weight_patches.json".to_string());
    index.representations = BTreeMap::from([(
        "target.embedding@BF16".to_string(),
        entry("segments/target.embedding.bin"),
    )]);
    index.variants = VariantCatalogue::new().with_set(
        "gate_up",
        RegionSetVariants::single(
            "q6k",
            StoredVariant::new("routed/layer_000.q6k", Fidelity::SourceEquivalent),
        ),
    );
    serde_json::to_value(index).unwrap()
}

fn parse(value: serde_json::Value) -> Result<Vindex3Index, serde_json::Error> {
    serde_json::from_value(value)
}

#[test]
fn a_well_formed_index_still_parses() {
    parse(index_json()).expect("every name in the fixture is container-relative");
}

#[test]
fn every_path_field_of_the_index_refuses_an_escaping_name() {
    let escape = "../../outside";
    let mut cases: Vec<(&str, serde_json::Value)> = Vec::new();
    for field in [
        "moe_manifest",
        "system_graph",
        "auxiliary_references",
        "representation_attestations",
        "weight_patches",
    ] {
        let mut v = index_json();
        v[field] = escape.into();
        cases.push((field, v));
    }

    let mut v = index_json();
    v["representations"]["target.embedding@BF16"]["segment"] = "/etc/passwd".into();
    cases.push(("representations.*.segment", v));

    let mut v = index_json();
    v["segments"] = serde_json::json!({ escape: 1 });
    cases.push(("segments key", v));

    let mut v = index_json();
    v["variants"]["gate_up"]["variants"]["q6k"]["storage"] = escape.into();
    cases.push(("variants.*.storage", v));

    for (field, value) in cases {
        let err = parse(value).expect_err(field);
        assert!(
            err.to_string().contains("which is refused"),
            "{field}: the refusal must be the container-path rule, got: {err}"
        );
    }
}

// ─── `prepare_write`: what a writer may create ────────────────────────

#[test]
fn a_write_creates_its_missing_parents_inside_the_root() {
    let root = tempfile::tempdir().unwrap();
    let path = prepare_write(root.path(), "segments/deep/a.bin").unwrap();
    assert!(path.parent().unwrap().is_dir());
    assert!(path.starts_with(root.path().canonicalize().unwrap()));
}

#[cfg(unix)]
#[test]
fn a_write_through_a_symlinked_directory_that_leaves_the_root_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(outside.path(), root.path().join("segments")).unwrap();
    match prepare_write(root.path(), "segments/sub/a.bin") {
        Err(VindexError::UnsafeContainerPath { reason, .. }) => {
            assert_eq!(reason, RESOLVES_OUTSIDE)
        }
        other => panic!("must refuse, got {other:?}"),
    }
    assert!(
        std::fs::read_dir(outside.path()).unwrap().next().is_none(),
        "nothing — not even a parent directory — is created outside"
    );
}

#[cfg(unix)]
#[test]
fn a_symlinked_directory_that_stays_inside_the_root_is_allowed() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("real")).unwrap();
    std::os::unix::fs::symlink(root.path().join("real"), root.path().join("segments")).unwrap();
    prepare_write(root.path(), "segments/a.bin").expect("the link resolves inside the root");
}

#[cfg(unix)]
#[test]
fn a_destination_that_is_a_symlink_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let target = outside.path().join("victim");
    std::fs::write(&target, b"untouched").unwrap();
    std::os::unix::fs::symlink(&target, root.path().join("a.bin")).unwrap();
    match prepare_write(root.path(), "a.bin") {
        Err(VindexError::UnsafeContainerPath { reason, .. }) => {
            assert_eq!(reason, DESTINATION_IS_SYMLINK)
        }
        other => panic!("must refuse, got {other:?}"),
    }
    assert_eq!(std::fs::read(&target).unwrap(), b"untouched");
}

#[test]
fn a_write_applies_the_lexical_rule_first() {
    let root = tempfile::tempdir().unwrap();
    assert!(prepare_write(root.path(), "../a.bin").is_err());
}
