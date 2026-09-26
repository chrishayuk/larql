//! ADR-0027 closure criterion 2: a legacy bank container is recognised by
//! name, on the index and on every graph-path surface, and never reported
//! as a conforming VINDEX3 3.0 container.

use super::*;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::test_support::fixture_a_spec;
use crate::format::vindex3::verify_system::verify_system;
use crate::format::vindex3::write::write_container;
use crate::format::vindex3::Vindex3Index;

/// What any legacy refusal must say: that it is legacy, which decision
/// made it so, and how to leave it.
const LEGACY_MUST_NAME: [&str; 3] = ["legacy bank-shape", "ADR-0027", "larql vindex3 encode"];

fn legacy_bank() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_container(dir.path(), &fixture_a_spec()).unwrap();
    dir
}

fn graph() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        "target",
    );
    container
}

fn index_of(dir: &std::path::Path) -> Vindex3Index {
    let raw = std::fs::read_to_string(dir.join(crate::format::filenames::INDEX_JSON)).unwrap();
    serde_json::from_str(&raw).unwrap()
}

fn assert_names_the_legacy_shape(err: &crate::error::VindexError) {
    assert!(
        matches!(err, crate::error::VindexError::LegacyBankContainer),
        "expected the legacy-bank refusal, got {err:?}"
    );
    let message = err.to_string();
    for phrase in LEGACY_MUST_NAME {
        assert!(
            message.contains(phrase),
            "{phrase:?} missing from: {message}"
        );
    }
}

#[test]
fn the_bank_writer_produces_the_legacy_shape_and_the_encoder_the_normative_one() {
    let bank = index_of(legacy_bank().path()).shape().unwrap();
    assert_eq!(bank, ContainerShape::LegacyBank);
    assert!(!bank.is_normative());
    assert!(bank.describe().contains("not a VINDEX3 3.0 container"));

    let graph = index_of(graph().path()).shape().unwrap();
    assert_eq!(graph, ContainerShape::Graph);
    assert!(graph.is_normative());
}

#[test]
fn a_graph_beside_a_manifest_is_the_graph_shape() {
    // The convergence configuration: the graph is the semantic authority
    // and the manifest describes the programme it locates (§5.5).
    let mut index = index_of(graph().path());
    index.moe_manifest = Some("moe_manifest.json".into());
    assert_eq!(index.shape().unwrap(), ContainerShape::Graph);
}

#[test]
fn an_index_naming_neither_authority_is_refused_rather_than_defaulted() {
    let mut index = index_of(graph().path());
    index.system_graph = None;
    index.moe_manifest = None;
    let err = index.shape().unwrap_err().to_string();
    assert!(err.contains("§5.5"), "{err}");
}

#[test]
fn inspection_refuses_a_legacy_bank_by_name_and_names_the_migration() {
    let bank = legacy_bank();
    for verify_payloads in [false, true] {
        assert_names_the_legacy_shape(
            &inspect_container(bank.path(), verify_payloads).unwrap_err(),
        );
    }
}

#[test]
fn system_verification_refuses_a_legacy_bank_by_name() {
    let bank = legacy_bank();
    assert_names_the_legacy_shape(&verify_system(&[], bank.path()).unwrap_err());
}
