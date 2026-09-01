//! **What a remote container refuses, and what it counts.**
//!
//! `parity.rs` proves the happy path moves the right bytes. This covers
//! the arms that fire when the thing on the other end is not the
//! container it claimed to be — each one existing because the alternative
//! is a plausible-looking failure much further downstream:
//!
//! ```text
//! index carries no representations   →  a container with no objects
//! representations name no segments   →  an index that describes nothing
//! a segment header claims 2^64 bytes →  a range read that missed
//! a named segment is not served      →  hydration silently short
//! ```
//!
//! Every subject is a REAL encoded container with exactly one thing done
//! to it, so what is under test is the arm and not the fixture.

use serial_test::serial;

use crate::format::filenames::INDEX_JSON;
use crate::format::huggingface::range::test_support::{
    MockRepo, RangeBehaviour, MOCK_REPO, MOCK_REVISION,
};
use crate::format::huggingface::range::HfRangeClient;
use crate::format::vindex3::encode::SEGMENTS_DIR;
use crate::format::vindex3::fixtures::{
    dense_f32_model_with, encode_fixture_container, HeadStorage,
};
use crate::format::vindex3::remote::RemoteContainer;

const COMPONENT: &str = "target";

fn client() -> HfRangeClient {
    HfRangeClient::new(MOCK_REPO, MOCK_REVISION).unwrap()
}

/// A real encoded container, to be served and then damaged one way.
fn subject() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        |dir| dense_f32_model_with(dir, HeadStorage::Separate),
        checkpoint.path(),
        container.path(),
        COMPONENT,
    );
    container
}

fn index_of(root: &std::path::Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(root.join(INDEX_JSON)).unwrap()).unwrap()
}

fn write_index(root: &std::path::Path, index: &serde_json::Value) {
    std::fs::write(root.join(INDEX_JSON), serde_json::to_vec(index).unwrap()).unwrap();
}

/// Open the served container into a fresh root, expecting a refusal.
fn open_err(served: &std::path::Path) -> String {
    let _repo = MockRepo::serve(served, RangeBehaviour::Honour);
    let root = tempfile::tempdir().unwrap();
    RemoteContainer::open(client(), root.path())
        .err()
        .expect("this container should not open")
        .to_string()
}

#[test]
#[serial]
fn an_index_without_representations_is_refused() {
    // `representations` is the only place an object is bound to a
    // segment. An index missing it parses as valid JSON and yields an
    // empty object set, which would present as a container of a model
    // with no weights rather than as a malformed index.
    let container = subject();
    let mut index = index_of(container.path());
    index.as_object_mut().unwrap().remove("representations");
    write_index(container.path(), &index);

    let message = open_err(container.path());
    assert!(
        message.contains("carries no representations"),
        "got: {message}"
    );
}

#[test]
#[serial]
fn an_index_whose_representations_name_no_segments_is_refused() {
    // Entries that carry neither an object nor a segment are skipped
    // rather than refused one by one — a forward-compatible index may
    // carry entries this reader does not understand. But if NONE of them
    // resolves, there is nothing to hydrate and that is a refusal.
    let container = subject();
    let mut index = index_of(container.path());
    let representations = index["representations"].as_object().unwrap().clone();
    let blanked: serde_json::Map<String, serde_json::Value> = representations
        .keys()
        .map(|k| (k.clone(), serde_json::json!({"encoding": "BF16"})))
        .collect();
    index["representations"] = serde_json::Value::Object(blanked);
    write_index(container.path(), &index);

    let message = open_err(container.path());
    assert!(message.contains("names no"), "got: {message}");
}

#[test]
#[serial]
fn a_segment_whose_header_claims_an_absurd_length_is_refused() {
    // The first eight bytes of anything decode to a number. If the range
    // read landed on payload, or on an HTML error page, the "header
    // length" it reports is arbitrary — and the only cheap way to know is
    // that it is impossibly large.
    let container = subject();
    let segment = std::fs::read_dir(container.path().join(SEGMENTS_DIR))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .next()
        .expect("the container has segments");
    let mut damaged = u64::MAX.to_le_bytes().to_vec();
    damaged.extend_from_slice(&[0u8; 128]);
    std::fs::write(&segment, damaged).unwrap();

    let message = open_err(container.path());
    assert!(message.contains("not a segment"), "got: {message}");
}

#[test]
#[serial]
fn a_segment_the_repo_does_not_serve_is_named_in_the_refusal() {
    // The index is the authority for what exists; the repo is the
    // authority for what can be fetched. When they disagree, hydration
    // must stop and say which object it could not get, rather than
    // report a short but successful transfer.
    let container = subject();
    let index = index_of(container.path());
    let segment_path = index["representations"]
        .as_object()
        .unwrap()
        .values()
        .find_map(|entry| entry["segment"].as_str())
        .expect("a representation names a segment")
        .to_string();
    std::fs::remove_file(container.path().join(&segment_path)).unwrap();

    let message = open_err(container.path());
    assert!(
        message.contains(&segment_path) || message.contains("404"),
        "the refusal must name the segment it could not get, got: {message}"
    );
}

#[test]
#[serial]
fn the_two_byte_counters_separate_description_from_payload() {
    // Headers and payload are the two things whose ratio is the whole
    // claim of this rung. One counter for both would make "we fetched
    // 6 MB of description and 6.29 GB of weights" unsayable.
    let container = subject();
    let _repo = MockRepo::serve(container.path(), RangeBehaviour::Honour);
    let root = tempfile::tempdir().unwrap();
    let mut remote = RemoteContainer::open(client(), root.path()).unwrap();

    let described = remote.described_objects().clone();
    assert!(!described.is_empty());
    let after_open = remote.metadata_bytes();
    assert!(
        after_open > 0,
        "opening reads the index, the graph and every segment header"
    );
    assert_eq!(
        remote.payload_bytes(),
        0,
        "opening must not fetch one payload byte"
    );

    let one: std::collections::BTreeSet<String> = described.iter().take(1).cloned().collect();
    remote.allow(one.clone()).unwrap();
    remote.hydrate().unwrap();

    let hydrated: u64 = one
        .iter()
        .map(|object| {
            std::fs::metadata(
                remote
                    .container_root()
                    .join(SEGMENTS_DIR)
                    .join(format!("{object}.bin")),
            )
            .map(|m| m.len())
            .unwrap_or(0)
        })
        .sum();
    assert_eq!(
        remote.payload_bytes(),
        hydrated,
        "the payload counter must equal the bytes that landed on disk"
    );
    assert_eq!(
        remote.metadata_bytes(),
        after_open,
        "hydrating payload must not be counted as description"
    );
}
