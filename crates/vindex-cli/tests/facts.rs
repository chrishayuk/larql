//! The vindex facts, gated on a real encoded container.
//!
//! The invariant under test: every command answers from the artifact
//! alone — the fixture is encoded through the real pipeline, then the
//! facts must reconstruct identity, directory, derived precision, and
//! self-verification from its bytes, with no source checkpoint
//! present. A corrupted byte must fail verify by name.

use larql_vindex::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};
use vindex_cli::{
    describe_facts, inspect_facts, precision_facts, representations_facts, verify_facts,
};

fn container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        dir.path(),
        "vindex-cli-fixture",
    );
    dir
}

#[test]
fn inspect_reconstructs_identity_and_census_from_the_artifact_alone() {
    let dir = container();
    let v = inspect_facts(dir.path()).unwrap();
    assert_eq!(v["model"], "vindex-cli-fixture");
    assert_eq!(v["generation"], 3);
    assert!(v["coherent"].as_bool().unwrap(), "{v}");
    assert_eq!(v["components"][0]["id"], "target");
}

#[test]
fn representations_list_the_physical_directory_with_fidelity() {
    let dir = container();
    let v = representations_facts(dir.path()).unwrap();
    let entries = v["entries"].as_array().unwrap();
    assert!(!entries.is_empty());
    assert!(
        entries
            .iter()
            .all(|e| e["payload_bytes"].as_u64().unwrap() > 0),
        "{v}"
    );
}

#[test]
fn describe_finds_an_object_by_suffix_and_shows_its_tensor_table() {
    let dir = container();
    let v = describe_facts(dir.path(), "decoder_stack", 4).unwrap();
    assert_eq!(v["object"]["id"], "target.decoder_stack");
    let head = &v["directory"][0]["tensor_table_head"];
    assert!(!head.as_array().unwrap().is_empty(), "{v}");
}

#[test]
fn describe_refuses_an_unknown_address_by_naming_the_holdings() {
    let dir = container();
    let err = describe_facts(dir.path(), "no.such.object", 4).unwrap_err();
    assert!(err.contains("the graph holds"), "{err}");
}

#[test]
fn precision_is_derived_from_bytes_over_elements_never_asserted() {
    let dir = container();
    let v = precision_facts(dir.path()).unwrap();
    let eff = v["effective_bits_per_weight"].as_f64().unwrap();
    // The miniature encodes F32 source tensors: 32 bits per weight.
    assert!(
        (eff - 32.0).abs() < 0.5,
        "effective {eff} — expected ~32 for the F32 fixture"
    );
}

#[test]
fn verify_passes_on_the_intact_artifact_and_fails_on_a_flipped_byte() {
    let dir = container();
    let v = verify_facts(dir.path()).unwrap();
    assert!(v["verified"].as_bool().unwrap(), "{v}");

    // Flip one payload byte in one segment: verify must fail, by name.
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("index.json")).unwrap())
            .unwrap();
    let segment = index["representations"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()["segment"]
        .as_str()
        .unwrap()
        .to_string();
    let path = dir.path().join(&segment);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&path, bytes).unwrap();

    let v = verify_facts(dir.path()).unwrap();
    assert!(
        !v["verified"].as_bool().unwrap(),
        "corruption must not verify: {v}"
    );
    assert_eq!(v["failures"], 1, "{v}");
}
