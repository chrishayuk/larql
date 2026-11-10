//! Vindex vs checkpoint vs neither, and the hub-cache short circuit.

use super::*;
use crate::commands::primary::hf_auto::classify::*;

#[test]
fn index_json_makes_a_vindex_whatever_else_is_there() {
    let probe = FakeProbe::carrying(&[INDEX_JSON, CHECKPOINT_CONFIG]);
    assert_eq!(classify(&probe, REPO, "main"), RepoKind::Vindex);
    assert_eq!(probe.asked(), vec![INDEX_JSON.to_string()]);
}

#[test]
fn config_without_index_is_a_checkpoint() {
    let probe = FakeProbe::carrying(&[CHECKPOINT_CONFIG]);
    assert_eq!(classify(&probe, REPO, "main"), RepoKind::Checkpoint);
}

#[test]
fn neither_file_is_unrecognised() {
    let probe = FakeProbe::carrying(&[]);
    assert_eq!(classify(&probe, REPO, "main"), RepoKind::Unrecognised);
}

#[test]
fn a_probe_that_fails_is_unrecognised_not_an_error() {
    assert_eq!(
        classify(&FakeProbe::offline(), REPO, "main"),
        RepoKind::Unrecognised
    );
    /// Answers the first question and fails the second.
    struct HalfProbe;
    impl RepoProbe for HalfProbe {
        fn has_file(&self, _: &str, _: &str, file: &str) -> Result<bool, BoxErr> {
            if file == INDEX_JSON {
                Ok(false)
            } else {
                Err("gated".into())
            }
        }
        fn commit(&self, _: &str, _: &str) -> Result<Option<String>, BoxErr> {
            Ok(None)
        }
    }
    assert_eq!(classify(&HalfProbe, REPO, "main"), RepoKind::Unrecognised);
    assert_eq!(HalfProbe.commit(REPO, "main").unwrap(), None);
}

#[test]
fn hub_cache_vindex_is_recognised_for_its_repo_only() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(!hub_has_vindex(tmp.path(), REPO));
    fake_hub_vindex(tmp.path(), REPO);
    assert!(hub_has_vindex(tmp.path(), REPO));
    assert!(!hub_has_vindex(tmp.path(), "acme/other"));
    assert!(!hub_has_vindex(tmp.path(), "no-slash"));
    // A snapshot without index.json (e.g. a checkpoint hf-hub cached) is not one.
    let bare = tmp.path().join("models--acme--bare/snapshots/x");
    std::fs::create_dir_all(bare).unwrap();
    assert!(!hub_has_vindex(tmp.path(), "acme/bare"));
}
