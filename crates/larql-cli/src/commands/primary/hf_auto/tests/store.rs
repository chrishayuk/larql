//! The encoded cache's layout, lookup and atomic publish.

use super::*;
use crate::commands::primary::hf_auto::store::*;

fn store() -> (tempfile::TempDir, EncodedStore) {
    let tmp = tempfile::tempdir().unwrap();
    let store = EncodedStore::new(tmp.path().join(ENCODED_CACHE_SUBDIR));
    (tmp, store)
}

/// A finished encode: a partial with a container, committed.
fn publish(store: &EncodedStore, repo: &str, commit: &str) -> PathBuf {
    let partial = store.begin(repo).unwrap();
    std::fs::create_dir_all(partial.path()).unwrap();
    std::fs::write(partial.path().join(INDEX_JSON), b"{}").unwrap();
    partial
        .commit(&store.snapshot_dir(repo, commit).unwrap())
        .unwrap()
}

#[test]
fn layout_mirrors_the_hub_cache() {
    let (_tmp, store) = store();
    let dir = store.snapshot_dir(REPO, COMMIT).unwrap();
    assert!(dir.ends_with(format!("models--acme--tiny-model/snapshots/{COMMIT}")));
    assert!(store.root().ends_with(ENCODED_CACHE_SUBDIR));
}

#[test]
fn unsafe_repo_ids_and_commits_are_refused() {
    let (_tmp, store) = store();
    for bad in [
        "noslash",
        "a/b/c",
        "../x",
        "a/..",
        "a/.hidden",
        "a--b/c",
        "/x",
        "x/",
    ] {
        assert!(store.repo_dir(bad).is_err(), "{bad} must be refused");
    }
    assert!(store.snapshot_dir(REPO, "../escape").is_err());
    assert!(store.complete(REPO, "..").is_none());
}

#[test]
fn a_committed_encode_is_found_by_commit_and_by_ref() {
    let (_tmp, store) = store();
    assert!(store.lookup(REPO, "main").is_none());
    let path = publish(&store, REPO, COMMIT);
    assert_eq!(store.complete(REPO, COMMIT), Some(path.clone()));
    assert_eq!(store.lookup(REPO, COMMIT), Some(path.clone()));
    // No ref yet: a branch name is not answered offline.
    assert!(store.lookup(REPO, "main").is_none());
    store.record_ref(REPO, "main", COMMIT).unwrap();
    assert_eq!(store.lookup(REPO, "main"), Some(path));
}

#[test]
fn refs_that_cannot_be_files_or_point_nowhere_are_ignored() {
    let (_tmp, store) = store();
    // A slash-bearing revision is not recorded and so never looked up.
    store.record_ref(REPO, "refs/pr/1", COMMIT).unwrap();
    assert!(store.lookup(REPO, "refs/pr/1").is_none());
    // A revision that IS the commit needs no ref.
    store.record_ref(REPO, COMMIT, COMMIT).unwrap();
    assert!(!store.repo_dir(REPO).unwrap().join(REFS_DIR).exists());
    // A ref to a commit that was never published answers nothing.
    store.record_ref(REPO, "main", COMMIT).unwrap();
    assert!(store.lookup(REPO, "main").is_none());
    // A corrupt ref answers nothing either.
    let refs = store.repo_dir(REPO).unwrap().join(REFS_DIR);
    std::fs::write(refs.join("main"), "../../elsewhere").unwrap();
    assert!(store.lookup(REPO, "main").is_none());
}

#[test]
fn a_failed_encode_leaves_nothing_behind() {
    let (_tmp, store) = store();
    let repo_dir = store.repo_dir(REPO).unwrap();
    {
        let partial = store.begin(REPO).unwrap();
        std::fs::create_dir_all(partial.path()).unwrap();
        // Even a container-shaped partial is not an entry until committed.
        std::fs::write(partial.path().join(INDEX_JSON), b"{}").unwrap();
        assert!(store.complete(REPO, COMMIT).is_none());
        assert!(partial
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with(PARTIAL_PREFIX));
    }
    assert!(!repo_dir.exists(), "a refused first run leaves no repo dir");
    // A repo dir that holds anything else is kept.
    std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
    drop(store.begin(REPO).unwrap());
    assert!(repo_dir.join("refs").exists());
}

#[test]
fn a_partial_without_a_container_is_never_published() {
    let (_tmp, store) = store();
    let partial = store.begin(REPO).unwrap();
    std::fs::create_dir_all(partial.path()).unwrap();
    let err = partial
        .commit(&store.snapshot_dir(REPO, COMMIT).unwrap())
        .unwrap_err();
    assert!(err.to_string().contains("nothing cached"), "{err}");
    assert!(store.complete(REPO, COMMIT).is_none());
}

#[test]
fn a_concurrent_winner_is_kept_and_the_loser_discarded() {
    let (_tmp, store) = store();
    let first = publish(&store, REPO, COMMIT);
    std::fs::write(first.join("marker"), b"first").unwrap();
    let second = publish(&store, REPO, COMMIT);
    assert_eq!(first, second);
    assert!(first.join("marker").exists(), "winner must be untouched");
    let partials = std::fs::read_dir(store.repo_dir(REPO).unwrap())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with(PARTIAL_PREFIX))
        .count();
    assert_eq!(partials, 0);
}

/// A pid no process can have (beyond every platform's pid range).
const DEAD_PID: u32 = i32::MAX as u32;

#[test]
fn begin_sweeps_partials_of_exited_processes_only() {
    let (_tmp, store) = store();
    let repo_dir = store.repo_dir(REPO).unwrap();
    let orphan = repo_dir.join(format!("{PARTIAL_PREFIX}{DEAD_PID}-0"));
    let live = repo_dir.join(format!("{PARTIAL_PREFIX}1-0"));
    let foreign = repo_dir.join(format!("{PARTIAL_PREFIX}not-a-pid"));
    for dir in [&orphan, &live, &foreign] {
        std::fs::create_dir_all(dir).unwrap();
    }
    let partial = store.begin(REPO).unwrap();
    assert!(!orphan.exists(), "an exited owner's partial is swept");
    assert!(live.exists(), "pid 1 is alive; its partial is not ours");
    assert!(foreign.exists(), "a partial with no pid is left alone");
    assert!(
        !partial.path().exists(),
        "the encoder creates its own output"
    );
}

#[test]
fn partial_owner_parses_only_our_names() {
    assert_eq!(partial_owner(&format!("{PARTIAL_PREFIX}42-7")), Some(42));
    assert_eq!(partial_owner(&format!("{PARTIAL_PREFIX}42")), None);
    assert_eq!(partial_owner("snapshots"), None);
    assert!(process_alive(std::process::id()));
    assert!(!process_alive(DEAD_PID));
    assert!(!process_alive(u32::MAX));
}

#[test]
fn safe_components() {
    assert!(is_safe_component("Qwen3-0.6B"));
    assert!(is_safe_component(COMMIT));
    for bad in ["", ".", "..", ".x", "a/b", "a\\b", "a--b"] {
        assert!(!is_safe_component(bad), "{bad}");
    }
}

#[test]
fn open_default_lives_under_larql_home() {
    let store = EncodedStore::open_default().unwrap();
    assert!(store.root().ends_with(ENCODED_CACHE_SUBDIR));
}
