//! Staging gates.
//!
//! Two properties carry the feature, and both are refusals rather than
//! happy paths:
//!
//! 1. **A stub directory that is not a checkpoint must be refused, not
//!    half-staged.** Admission runs against the stub, so a stub missing
//!    the file admission reads would produce a verdict about nothing.
//! 2. **A range read that did not land on a safetensors file must be
//!    caught at the header, not at the tensor.** The length prefix is
//!    eight arbitrary bytes; if the read landed anywhere else they decode
//!    to a plausible-looking number, and the only cheap check is whether
//!    that number is absurd.
//!
//! Plus resume, asserted so that it cannot pass by doing nothing: the
//! second pass is served a DIFFERENT repo under the same names, so any
//! byte that was re-fetched shows up as a byte that changed.

use std::path::Path;

use serial_test::serial;

use super::super::range::test_support::{MockRepo, RangeBehaviour, MOCK_COMMIT};
use super::super::range::HfRangeClient;
use super::{
    header_cache_dir, resolve_commit, stage_metadata_checkpoint, LENGTH_PREFIX_BYTES,
    SAFETENSORS_INDEX_FILE,
};
const REPO: &str = "larql-test/fixture";
const REVISION: &str = "main";

fn client() -> HfRangeClient {
    HfRangeClient::new(REPO, REVISION).unwrap()
}

/// A safetensors stub: the length prefix and the header it announces.
fn shard_stub(header: &serde_json::Value) -> Vec<u8> {
    let json = serde_json::to_vec(header).unwrap();
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&json);
    out
}

/// A minimal one-tensor header, offsets relative to the payload base.
fn one_tensor_header(name: &str) -> serde_json::Value {
    serde_json::json!({
        name: {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
    })
}

/// A served repo: `config.json`, a shard index, and each shard's stub.
fn served_repo(dir: &Path, shards: &[&str]) {
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&serde_json::json!({"model_type": "llama"})).unwrap(),
    )
    .unwrap();
    let mut weight_map = serde_json::Map::new();
    for (i, shard) in shards.iter().enumerate() {
        weight_map.insert(format!("weight.{i}"), serde_json::Value::from(*shard));
    }
    std::fs::write(
        dir.join(SAFETENSORS_INDEX_FILE),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {"total_size": 16 * shards.len()},
            "weight_map": weight_map,
        }))
        .unwrap(),
    )
    .unwrap();
    for (i, shard) in shards.iter().enumerate() {
        let path = dir.join(shard);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(
            &path,
            shard_stub(&one_tensor_header(&format!("weight.{i}"))),
        )
        .unwrap();
    }
}

#[test]
#[serial]
fn the_cache_is_keyed_by_repo_and_revision_under_larql_home() {
    // Headers staged at one commit and payloads read at another address a
    // different checkpoint with the same offsets — the one failure this
    // path can have that still produces plausible bytes. The revision has
    // to be part of the directory, not a note inside it.
    let home = tempfile::tempdir().unwrap();
    let prev = std::env::var("LARQL_HOME").ok();
    std::env::set_var("LARQL_HOME", home.path());

    let a = header_cache_dir("Qwen/Qwen3-4B", "abc123").unwrap();
    let b = header_cache_dir("Qwen/Qwen3-4B", "def456").unwrap();
    assert_ne!(a, b, "two revisions must not share a stub directory");
    assert!(a.starts_with(home.path()), "LARQL_HOME must be honoured");
    assert!(
        a.ends_with("abc123"),
        "the revision is the leaf, got {}",
        a.display()
    );
    assert!(
        a.to_string_lossy().contains("Qwen--Qwen3-4B"),
        "a repo id is not a path; `/` must be escaped, got {}",
        a.display()
    );

    // An empty override is not an override — otherwise `LARQL_HOME=`
    // in a shell profile would stage headers at the filesystem root.
    std::env::set_var("LARQL_HOME", "");
    let fallback = header_cache_dir("Qwen/Qwen3-4B", "abc123").unwrap();
    assert!(
        fallback.to_string_lossy().contains(".cache"),
        "an empty LARQL_HOME must fall back to the cache dir, got {}",
        fallback.display()
    );

    match prev {
        Some(prev) => std::env::set_var("LARQL_HOME", prev),
        None => std::env::remove_var("LARQL_HOME"),
    }
}

#[test]
#[serial]
fn the_home_fallback_accepts_either_platform_spelling() {
    // Windows sets `USERPROFILE` and not `HOME`, so a fallback reading
    // only `HOME` makes every unconfigured `hf://` command there fail
    // with "HOME is not set" instead of staging anything. Asserted from
    // both directions rather than per-platform, so whichever variable the
    // host happens to define, the one it does NOT define is the one under
    // test.
    let prev_larql = std::env::var("LARQL_HOME").ok();
    let prev_home = std::env::var("HOME").ok();
    let prev_profile = std::env::var("USERPROFILE").ok();
    std::env::remove_var("LARQL_HOME");

    for var in ["HOME", "USERPROFILE"] {
        let other = if var == "HOME" { "USERPROFILE" } else { "HOME" };
        std::env::remove_var(other);
        std::env::set_var(var, "/somewhere");
        let dir = header_cache_dir("Qwen/Qwen3-4B", "abc123")
            .unwrap_or_else(|e| panic!("{var} alone should resolve a cache root: {e}"));
        assert!(
            dir.to_string_lossy().contains("hf-headers"),
            "{var} alone gave {}",
            dir.display()
        );
        std::env::remove_var(var);
    }

    // With neither, the refusal names both spellings rather than one.
    let err = header_cache_dir("Qwen/Qwen3-4B", "abc123")
        .expect_err("no home variable at all is not resolvable");
    let message = err.to_string();
    assert!(
        message.contains("HOME") && message.contains("USERPROFILE"),
        "the refusal should name both, got: {message}"
    );

    for (var, value) in [
        ("LARQL_HOME", prev_larql),
        ("HOME", prev_home),
        ("USERPROFILE", prev_profile),
    ] {
        match value {
            Some(value) => std::env::set_var(var, value),
            None => std::env::remove_var(var),
        }
    }
}

#[test]
#[serial]
fn the_revision_resolves_to_the_commit_the_hub_names() {
    let dir = tempfile::tempdir().unwrap();
    served_repo(dir.path(), &["model.safetensors"]);
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);

    let commit = resolve_commit(&client()).unwrap();
    assert_eq!(
        commit.as_deref(),
        Some(MOCK_COMMIT),
        "the commit is reported, never invented"
    );
}

#[test]
#[serial]
fn a_repo_without_a_config_is_refused_as_not_a_checkpoint() {
    // The stub is what admission reads. Staging a repo with no
    // `config.json` would produce a directory that `build_inventory`
    // cannot identify, and the failure would surface far from its cause.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("README.md"), b"not a model").unwrap();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);

    let staged = tempfile::tempdir().unwrap();
    let err = stage_metadata_checkpoint(&client(), staged.path())
        .expect_err("a repo with no config.json is not a checkpoint");
    assert!(
        err.to_string().contains("not a model checkpoint"),
        "the refusal must say what is missing, got: {err}"
    );
}

#[test]
#[serial]
fn an_index_naming_no_shards_is_refused() {
    // An index with an empty weight_map parses fine and yields nothing to
    // fetch. Without this arm the stub would be written, admission would
    // run against zero tensors, and the model would look empty rather
    // than unreadable.
    let dir = tempfile::tempdir().unwrap();
    served_repo(dir.path(), &["model-00001-of-00001.safetensors"]);
    std::fs::write(
        dir.path().join(SAFETENSORS_INDEX_FILE),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {"total_size": 0},
            "weight_map": {},
        }))
        .unwrap(),
    )
    .unwrap();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);

    let staged = tempfile::tempdir().unwrap();
    let err = stage_metadata_checkpoint(&client(), staged.path())
        .expect_err("an index that names no shards has nothing to stage");
    assert!(err.to_string().contains("names no shards"), "got: {err}");
}

#[test]
#[serial]
fn a_length_prefix_that_is_not_a_header_is_refused_at_the_prefix() {
    // If the range read landed on payload rather than a header, the first
    // eight bytes still decode to a number — just an absurd one. Catching
    // it here is the difference between a named refusal and a JSON parse
    // error two layers down.
    let dir = tempfile::tempdir().unwrap();
    served_repo(dir.path(), &["model.safetensors"]);
    let mut nonsense = u64::MAX.to_le_bytes().to_vec();
    nonsense.extend_from_slice(&[0u8; 64]);
    std::fs::write(dir.path().join("model.safetensors"), &nonsense).unwrap();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);

    let staged = tempfile::tempdir().unwrap();
    let err = stage_metadata_checkpoint(&client(), staged.path())
        .expect_err("a header claiming u64::MAX bytes is not a header");
    let message = err.to_string();
    assert!(
        message.contains("did not land on a safetensors file"),
        "the refusal must name what it concluded, got: {message}"
    );
}

#[test]
#[serial]
fn staging_resumes_from_what_is_already_staged() {
    // Idempotence asserted so that it cannot pass vacuously: stage from
    // one repo, then serve a DIFFERENT repo at the same names and stage
    // again into the same directory. Anything the second pass re-fetched
    // would come back with the second repo's bytes, so byte-identity of
    // the stub is the evidence that the cache-hit path was taken.
    //
    // Note what is NOT asserted: the optional metadata files this repo
    // does not carry are re-probed every pass, because absence is not
    // something the stub can record. Only what was staged is skipped.
    let good = tempfile::tempdir().unwrap();
    served_repo(good.path(), &["model.safetensors"]);
    let staged = tempfile::tempdir().unwrap();

    let first = {
        let _repo = MockRepo::serve(good.path(), RangeBehaviour::Honour);
        stage_metadata_checkpoint(&client(), staged.path()).unwrap()
    };
    assert!(first.stub_bytes > LENGTH_PREFIX_BYTES);
    let staged_stub = std::fs::read(staged.path().join("model.safetensors")).unwrap();
    let staged_config = std::fs::read(staged.path().join("config.json")).unwrap();

    let other = tempfile::tempdir().unwrap();
    served_repo(other.path(), &["model.safetensors"]);
    std::fs::write(
        other.path().join("model.safetensors"),
        shard_stub(&one_tensor_header("a.completely.different.tensor")),
    )
    .unwrap();
    std::fs::write(
        other.path().join("config.json"),
        serde_json::to_vec(&serde_json::json!({"model_type": "not-the-same"})).unwrap(),
    )
    .unwrap();
    let _repo = MockRepo::serve(other.path(), RangeBehaviour::Honour);

    let second = stage_metadata_checkpoint(&client(), staged.path()).unwrap();
    assert_eq!(second.shards, first.shards);
    assert_eq!(second.stub_bytes, first.stub_bytes);
    assert_eq!(
        std::fs::read(staged.path().join("model.safetensors")).unwrap(),
        staged_stub,
        "the shard stub was re-fetched — resume did not take"
    );
    assert_eq!(
        std::fs::read(staged.path().join("config.json")).unwrap(),
        staged_config,
        "the staged metadata was re-fetched — resume did not take"
    );
}

#[test]
#[serial]
fn a_shard_in_a_subdirectory_is_staged_under_the_same_relative_path() {
    // The offsets in a stub are addressed by the shard's repo-relative
    // name, so a repo that keeps its shards in a subdirectory must stage
    // to the same shape — a flattened stub would index correctly and then
    // request the wrong URL.
    let dir = tempfile::tempdir().unwrap();
    served_repo(dir.path(), &["weights/model-00001-of-00001.safetensors"]);
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);

    let staged = tempfile::tempdir().unwrap();
    let out = stage_metadata_checkpoint(&client(), staged.path()).unwrap();
    assert_eq!(out.shards, vec!["weights/model-00001-of-00001.safetensors"]);
    assert!(
        staged
            .path()
            .join("weights/model-00001-of-00001.safetensors")
            .exists(),
        "the stub must mirror the repo's own layout"
    );
    assert_eq!(out.declared_total_size, Some(16));
}
