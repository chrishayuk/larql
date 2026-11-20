//! Gate B, B6 §3.4, R4: the check a staged VINDEX2 vindex must pass before
//! it is published.

use std::path::Path;

use larql_vindex::format::filenames::KNN_STORE_BIN;

use super::super::atomic::validate_compiled_vindex2;

/// A format version this build reads as VINDEX2.
const VINDEX2_VERSION: u32 = 2;
const PAYLOAD: &str = "gate_vectors.bin";
const PAYLOAD_BYTES: &[u8] = b"gate";

/// A staged vindex with one payload file and an index whose checksum for it
/// is `recorded`.
fn staged(dir: &Path, recorded: &str) {
    std::fs::write(dir.join(PAYLOAD), PAYLOAD_BYTES).unwrap();
    let config = larql_vindex::VindexConfig {
        version: VINDEX2_VERSION,
        checksums: Some(
            [(PAYLOAD.to_string(), recorded.to_string())]
                .into_iter()
                .collect(),
        ),
        ..Default::default()
    };
    larql_vindex::VectorIndex::save_config(&config, dir).unwrap();
}

fn digest() -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(PAYLOAD);
    std::fs::write(&path, PAYLOAD_BYTES).unwrap();
    larql_vindex::format::checksums::sha256_file(&path).unwrap()
}

#[test]
fn a_vindex_whose_checksums_match_passes() {
    let dir = tempfile::tempdir().unwrap();
    staged(dir.path(), &digest());
    validate_compiled_vindex2(dir.path()).unwrap();
}

#[test]
fn a_file_that_does_not_match_its_checksum_is_named() {
    let dir = tempfile::tempdir().unwrap();
    staged(dir.path(), &"0".repeat(64));
    let err = validate_compiled_vindex2(dir.path())
        .unwrap_err()
        .to_string();
    assert!(err.contains("failed validation"), "{err}");
    assert!(err.contains(PAYLOAD), "{err}");
}

#[test]
fn a_vindex_with_no_index_does_not_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let err = validate_compiled_vindex2(dir.path())
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not reopen"), "{err}");
}

#[test]
fn a_knn_store_that_does_not_load_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    staged(dir.path(), &digest());
    std::fs::write(dir.path().join(KNN_STORE_BIN), b"not a knn store").unwrap();
    let err = validate_compiled_vindex2(dir.path())
        .unwrap_err()
        .to_string();
    assert!(err.contains(KNN_STORE_BIN), "{err}");
    assert!(err.contains("does not load"), "{err}");
}
