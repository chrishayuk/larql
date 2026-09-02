//! What an artifact argument means.
//!
//! Three spellings resolve to one thing, and the differences between them
//! are exactly the places a model's identity could be got wrong: the name
//! recorded in the container, the revision the payloads will be read at,
//! and whether anything was staged at all.

use std::path::{Path, PathBuf};

use serial_test::serial;

use crate::format::huggingface::range::test_support::{
    MockRepo, RangeBehaviour, MOCK_COMMIT, MOCK_REPO,
};
use crate::format::vindex3::artifact::{is_remote_spec, resolve, resolve_all};
use crate::format::vindex3::fixtures::miniature_glimmer;

/// A checkpoint directory named after the model it holds.
fn checkpoint(name: &str) -> tempfile::TempDir {
    let parent = tempfile::tempdir().unwrap();
    let dir = parent.path().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    miniature_glimmer(&dir);
    // Leak the parent so the named child outlives this helper.
    let kept = tempfile::TempDir::new().unwrap();
    let moved = kept.path().join(name);
    std::fs::rename(&dir, &moved).unwrap();
    drop(parent);
    kept
}

#[test]
fn a_repo_spec_is_told_from_a_path() {
    // Both ingest verbs branch on this before anything is parsed, so a
    // checkpoint directory whose name merely mentions the hub must not be
    // taken for a repo — it would be staged instead of read.
    assert!(is_remote_spec(Path::new("hf://Qwen/Qwen3-4B")));
    assert!(is_remote_spec(Path::new("hf://ibm-granite/granite@main")));
    assert!(!is_remote_spec(Path::new("./granite-4.2-3b")));
    assert!(!is_remote_spec(Path::new("/models/hf/Qwen3-4B")));
    assert!(!is_remote_spec(Path::new("inventory.json")));
}

#[test]
fn a_local_checkpoint_is_named_for_its_directory() {
    // The name travels into the container and is how every later verb
    // addresses the model, so it is identity, not cosmetics.
    let dir = checkpoint("miniature-glimmer");
    let resolved = resolve(&dir.path().join("miniature-glimmer")).unwrap();

    assert_eq!(resolved.name, "miniature-glimmer");
    assert!(resolved.commit().is_none(), "a local dir has no revision");
    assert!(
        resolved.staging().is_none(),
        "a local dir staged nothing — that is absence, not a zero"
    );
}

#[test]
fn a_saved_inventory_resolves_without_re_inspecting_the_checkpoint() {
    // The `.json` spelling exists so a plan can be re-run against a
    // recorded inventory. It must produce the same inventory the
    // directory did, or the two spellings describe different models.
    let dir = checkpoint("subject");
    let from_dir = resolve(&dir.path().join("subject")).unwrap();

    let saved = dir.path().join("inventory.json");
    std::fs::write(&saved, serde_json::to_string(&from_dir.inventory).unwrap()).unwrap();
    let from_json = resolve(&saved).unwrap();

    assert_eq!(from_json.name, "inventory");
    assert_eq!(
        from_json.inventory.identity.model_type,
        from_dir.inventory.identity.model_type
    );
    assert_eq!(
        from_json.inventory.path, from_dir.inventory.path,
        "the saved spelling must point at the same checkpoint"
    );
}

#[test]
fn a_malformed_inventory_is_refused_by_name() {
    // A `.json` that is not an inventory would otherwise fail somewhere
    // downstream, describing a model nobody wrote.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not-an-inventory.json");
    std::fs::write(&path, b"{\"hello\": true}").unwrap();

    // `is_err`/`unwrap_err` rather than `expect_err`: ResolvedArtifact holds
    // a live HTTP client and deliberately does not derive Debug.
    let err = match resolve(&path) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("a JSON file that is not an inventory must be refused"),
    };
    assert!(
        err.contains("not-an-inventory.json"),
        "the refusal must name the file it could not read: {err}"
    );
}

#[test]
#[serial]
fn a_repo_is_named_for_the_model_and_pinned_to_a_commit() {
    // The owner is dropped from the name — a container records the model,
    // not who published it — and the revision is resolved to a commit
    // BEFORE anything is staged. Headers read at one commit and payloads
    // at another would address a different checkpoint with the same
    // offsets, which is the one failure here that still produces
    // plausible bytes.
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);

    let resolved = resolve(Path::new(&format!("hf://{MOCK_REPO}"))).unwrap();

    assert_eq!(
        resolved.name,
        MOCK_REPO.rsplit('/').next().unwrap(),
        "the owner is not part of the model's name"
    );
    assert_eq!(resolved.commit(), Some(MOCK_COMMIT));
    assert!(
        resolved.unpinned_revision().is_none(),
        "the hub named a commit, so nothing fell back to a moving name"
    );
}

#[test]
#[serial]
fn staging_a_repo_reports_what_it_cost_and_what_it_stands_in_for() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);

    let resolved = resolve(Path::new(&format!("hf://{MOCK_REPO}"))).unwrap();
    let report = resolved.staging().expect("a repo staged something");

    assert!(report.shards >= 1);
    assert!(report.header_bytes > 0, "shard headers were staged");
    assert!(report.metadata_bytes > 0, "config.json at least was staged");
    let payload = *report
        .payload_bytes
        .as_ref()
        .expect("the headers total cleanly");
    assert!(
        payload > report.staged_bytes(),
        "the point is that the payload dwarfs the staging: {payload} vs {}",
        report.staged_bytes()
    );
}

#[test]
#[serial]
fn every_argument_resolves_or_none_does() {
    // `resolve_all` is what the CLIs call. One bad argument must fail the
    // whole call rather than silently encode a container from the
    // artifacts that happened to work.
    let dir = checkpoint("good");
    let good = dir.path().join("good");
    let missing = dir.path().join("no-such-checkpoint");

    assert_eq!(resolve_all(std::slice::from_ref(&good)).unwrap().len(), 1);
    assert!(
        resolve_all(&[good, missing]).is_err(),
        "a partial resolve would encode a partial system"
    );
}

#[test]
fn a_path_with_no_stem_still_gets_a_name() {
    // Degenerate, but the name is not allowed to be empty: it addresses
    // the artifact in every later verb.
    let resolved = resolve(&PathBuf::from("/"));
    assert!(
        resolved.is_err(),
        "root is not a checkpoint, and must fail rather than resolve namelessly"
    );
}

// ── the commit probe: identity without staging ───────────────────────
//
// `resolve_pinned_commit` exists so a caller holding a verdict cache can
// look up an answer before paying for the headers that would produce it.
// What matters here is that it says "no persistent identity" for exactly
// the cases where a cached verdict would be wrong to reuse, and that it
// does so without touching the network.

#[test]
fn a_local_path_has_no_commit_to_pin() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        super::super::resolve_pinned_commit(dir.path()).unwrap(),
        None,
        "a local checkpoint has no persistent identity, so a verdict about it is never cacheable"
    );
}

#[test]
fn a_relative_local_path_is_not_mistaken_for_a_repo() {
    assert_eq!(
        super::super::resolve_pinned_commit(Path::new("./some/checkpoint")).unwrap(),
        None
    );
}

/// One unpinnable artifact among many is the case the caller must not
/// get wrong, so the plural form reports per artifact rather than
/// collapsing to a single answer.
#[test]
fn commits_are_reported_per_artifact_in_order() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let got = super::super::resolve_pinned_commits(&[
        a.path().to_path_buf(),
        b.path().to_path_buf(),
    ])
    .unwrap();
    assert_eq!(got, vec![None, None]);
}

#[test]
fn no_specs_is_no_commits_rather_than_an_error() {
    assert_eq!(
        super::super::resolve_pinned_commits(&[]).unwrap(),
        Vec::<Option<String>>::new()
    );
}

/// A malformed reference is refused by the spec parser, before a client
/// is built — so a bad argument costs nothing, and the probe cannot be
/// used to make this process open connections on request.
#[test]
fn a_malformed_reference_is_refused_before_any_network() {
    let err = super::super::resolve_pinned_commit(Path::new("hf://"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("malformed"), "{err}");
}
