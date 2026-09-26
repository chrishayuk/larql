//! `larql verify` and `larql show` on both VINDEX3 shapes (ADR-0027).
//!
//! A graph container is the normative shape and must verify through the
//! graph path; a legacy bank container stays readable and verifiable under
//! its legacy label. Before shape dispatch, `verify` sent every V3
//! container to the bank reader, which refuses a graph container outright.

use super::*;
use crate::commands::primary::show_cmd::{self, ShowArgs};
use larql_vindex::format::vindex3::encode::SEGMENTS_DIR;
use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::test_support::fixture_a_spec;
use larql_vindex::format::vindex3::write::write_container;

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

fn legacy_bank() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_container(dir.path(), &fixture_a_spec()).unwrap();
    dir
}

fn verify(dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    run(VerifyArgs {
        vindex: dir.to_path_buf(),
    })
}

#[test]
fn a_graph_container_verifies_through_the_graph_path() {
    let dir = graph();
    verify(dir.path()).expect("the normative shape must verify");
}

#[test]
fn a_corrupted_graph_payload_fails_verification() {
    let dir = graph();
    let segment = std::fs::read_dir(dir.path().join(SEGMENTS_DIR))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.is_file())
        .expect("an encoded container has segments");
    let mut bytes = std::fs::read(&segment).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&segment, bytes).unwrap();
    let err = verify(dir.path()).unwrap_err().to_string();
    assert!(err.contains("not coherent"), "{err}");
}

#[test]
fn a_legacy_bank_container_stays_verifiable() {
    let dir = legacy_bank();
    verify(dir.path()).expect("a legacy bank is recognised and read, not refused");
}

#[test]
fn show_describes_both_shapes() {
    for dir in [graph(), legacy_bank()] {
        show_cmd::run(ShowArgs {
            model: dir.path().display().to_string(),
        })
        .expect("show describes either shape");
    }
}
