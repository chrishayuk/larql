//! When a pulled VINDEX3 container's segments are checked against its
//! own index, and when the check is allowed to be skipped.

use std::path::Path;

use super::payloads::{verify_v3_payloads, PendingVerification, VERIFIED_MARKER};
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};
use crate::format::vindex3::index::Vindex3Index;

/// A real encoded container, as a pull would leave it on disk.
fn pulled() -> (tempfile::TempDir, tempfile::TempDir) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "pull-fixture",
    );
    (checkpoint, container)
}

/// Flip one payload byte in place: same length, so only a hash sees it.
fn corrupt_one_segment(root: &Path) -> String {
    let index: Vindex3Index =
        serde_json::from_slice(&std::fs::read(root.join(INDEX_JSON)).unwrap()).unwrap();
    let segment = index
        .representations
        .values()
        .next()
        .expect("the fixture has a representation")
        .segment
        .clone();
    let path = root.join(&segment);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&path, bytes).unwrap();
    segment
}

fn marker(root: &Path) -> Option<String> {
    std::fs::read_to_string(root.join(VERIFIED_MARKER)).ok()
}

#[test]
fn an_intact_pull_verifies_and_records_which_index_it_verified() {
    let (_c, root) = pulled();
    verify_v3_payloads(root.path(), true).expect("an intact container verifies");
    let recorded = marker(root.path()).expect("a verified pull leaves a marker");
    assert_eq!(recorded.len(), 64, "the marker is the index's hex sha256");
}

#[test]
fn a_corrupt_segment_is_refused_by_name_and_leaves_no_marker() {
    let (_c, root) = pulled();
    let segment = corrupt_one_segment(root.path());
    match verify_v3_payloads(root.path(), true) {
        Err(VindexError::PayloadDigestMismatch { mismatches, .. }) => {
            assert!(
                mismatches.iter().any(|m| m.contains(&segment)),
                "the refusal must name `{segment}`: {mismatches:?}"
            );
        }
        other => panic!("a corrupt segment must be refused, got {other:?}"),
    }
    assert!(marker(root.path()).is_none());
}

#[test]
fn a_failed_check_is_not_skipped_on_the_next_cached_pull() {
    // The next pull finds every file in the cache and fetches nothing.
    // Without a marker that is no reason to trust the cache.
    let (_c, root) = pulled();
    corrupt_one_segment(root.path());
    assert!(verify_v3_payloads(root.path(), true).is_err());
    assert!(
        verify_v3_payloads(root.path(), false).is_err(),
        "a cache-only pull of a container that failed must fail again"
    );
}

#[test]
fn a_cached_pull_of_a_verified_index_is_not_rehashed() {
    // The marker is the only reason to skip: prove it is honoured by
    // damaging bytes the skip would not look at.
    let (_c, root) = pulled();
    verify_v3_payloads(root.path(), true).unwrap();
    corrupt_one_segment(root.path());
    verify_v3_payloads(root.path(), false).expect("nothing fetched, same index: skipped");
}

#[test]
fn anything_fetched_invalidates_the_marker() {
    let (_c, root) = pulled();
    verify_v3_payloads(root.path(), true).unwrap();
    corrupt_one_segment(root.path());
    assert!(
        verify_v3_payloads(root.path(), true).is_err(),
        "a pull that fetched bytes must verify them"
    );
    assert!(
        marker(root.path()).is_none(),
        "a failed re-check removes the stale marker"
    );
}

#[test]
fn a_different_index_invalidates_the_marker() {
    let (_c, root) = pulled();
    verify_v3_payloads(root.path(), true).unwrap();
    corrupt_one_segment(root.path());
    // A new revision's index: same files, different bytes.
    let path = root.path().join(INDEX_JSON);
    let mut index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    index["model"] = "pull-fixture-revised".into();
    std::fs::write(&path, serde_json::to_vec(&index).unwrap()).unwrap();
    assert!(
        verify_v3_payloads(root.path(), false).is_err(),
        "the marker names a different index, so it vouches for nothing"
    );
}

#[test]
fn a_named_segment_that_is_absent_is_a_mismatch_not_a_pass() {
    let (_c, root) = pulled();
    let segment = corrupt_one_segment(root.path());
    std::fs::remove_file(root.path().join(&segment)).unwrap();
    match verify_v3_payloads(root.path(), true) {
        Err(VindexError::PayloadDigestMismatch { mismatches, .. }) => {
            assert!(mismatches.iter().any(|m| m.contains("cannot be read")))
        }
        other => panic!("an absent segment must be refused, got {other:?}"),
    }
}

#[test]
fn a_replacement_cut_short_after_a_valid_marker_is_rehashed_next_time() {
    // The sequence the marker must survive: a verified cache, then a pull
    // that starts replacing a segment and never finishes. The marker is
    // invalidated BEFORE the replacement, so the interrupted pull cannot
    // leave it vouching for bytes it never checked — and the next pull,
    // which finds everything cached and fetches nothing, must hash.
    let (_c, root) = pulled();
    verify_v3_payloads(root.path(), true).unwrap();
    assert!(
        marker(root.path()).is_some(),
        "precondition: a valid marker"
    );

    let pending = PendingVerification::begin(root.path()).unwrap();
    assert!(
        marker(root.path()).is_none(),
        "the marker is gone before anything is replaced"
    );
    corrupt_one_segment(root.path()); // the replacement
    drop(pending); // the interruption: `finish` never runs

    assert!(marker(root.path()).is_none());
    assert!(
        verify_v3_payloads(root.path(), false).is_err(),
        "a cache-only pull after an interrupted replacement must verify"
    );
}

#[test]
fn the_marker_is_written_by_rename_and_leaves_no_staging_file() {
    let (_c, root) = pulled();
    verify_v3_payloads(root.path(), true).unwrap();
    let leftovers: Vec<_> = std::fs::read_dir(root.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(VERIFIED_MARKER) && n != VERIFIED_MARKER)
        .collect();
    assert!(leftovers.is_empty(), "staging left behind: {leftovers:?}");
}

#[test]
fn a_skipped_check_re_arms_the_marker() {
    // `begin` always deletes it; a pull that legitimately skips hashing
    // must put it back, or every second pull would re-hash the model.
    let (_c, root) = pulled();
    verify_v3_payloads(root.path(), true).unwrap();
    verify_v3_payloads(root.path(), false).unwrap();
    assert!(marker(root.path()).is_some());
}
