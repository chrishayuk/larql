//! `larql pull` as the explicit refresh — the hub's answer is faked, so
//! a commit that moves between two pulls is a fixture, not a wait.

use super::*;
use crate::commands::primary::hf_auto::classify::CHECKPOINT_CONFIG;
use crate::commands::primary::hf_auto::refresh::pull_spec_for_shorthand;
use crate::commands::primary::hf_auto::resolve::*;
use crate::commands::primary::hf_auto::store::EncodedStore;

/// The commit `main` moves to between pulls.
const NEXT_COMMIT: &str = "fedcba9876543210fedcba9876543210fedcba98";

/// The repo spec every test pulls.
const SPEC: &str = "hf://acme/tiny-model";

/// A checkpoint repo whose revisions all resolve to `commit`.
fn hub_at(commit: &str) -> FakeProbe {
    FakeProbe {
        commit: Some(commit.to_string()),
        ..FakeProbe::carrying(&[CHECKPOINT_CONFIG])
    }
}

struct World {
    tmp: tempfile::TempDir,
    store: EncodedStore,
}

fn world() -> World {
    let tmp = tempfile::tempdir().unwrap();
    let store = EncodedStore::new(tmp.path().join("encoded"));
    World { tmp, store }
}

/// One pull: its result, the vindex fetches it made, what it said.
fn pull(
    world: &World,
    probe: &FakeProbe,
    encoder: &FakeEncoder,
    spec: &str,
) -> (Result<Resolution, BoxErr>, Vec<String>, String) {
    let hub = world.tmp.path().join("hub");
    let mut fetched = Vec::new();
    let mut fetch = |s: &str| -> Result<PathBuf, BoxErr> {
        fetched.push(s.to_string());
        Ok(hub.join("fetched"))
    };
    let mut status = Vec::new();
    let result = HfAuto {
        store: &world.store,
        hub: &hub,
        probe,
        encoder,
    }
    .pull(spec, &mut fetch, &mut status);
    (result, fetched, String::from_utf8(status).unwrap())
}

/// `world` with `SPEC` already encoded at [`COMMIT`].
fn encoded_world() -> World {
    let world = world();
    let (result, _, _) = pull(&world, &hub_at(COMMIT), &FakeEncoder::default(), SPEC);
    assert!(matches!(
        result.unwrap(),
        Resolution::Encoded { fresh: true, .. }
    ));
    world
}

#[test]
fn a_repo_with_nothing_encoded_pulls_exactly_as_run_resolves() {
    // A published vindex is still downloaded…
    let world = world();
    let vindex = FakeProbe::carrying(&[INDEX_JSON]);
    let (result, fetched, _) = pull(&world, &vindex, &FakeEncoder::default(), SPEC);
    assert!(matches!(result.unwrap(), Resolution::Vindex(_)));
    assert_eq!(fetched, [SPEC]);
    // …and a checkpoint encoded on its first pull.
    let encoder = FakeEncoder::default();
    let (result, _, status) = pull(&world, &hub_at(COMMIT), &encoder, SPEC);
    assert!(matches!(
        result.unwrap(),
        Resolution::Encoded { fresh: true, .. }
    ));
    assert!(status.contains("first run only"), "{status}");
}

#[test]
fn an_unmoved_revision_is_already_up_to_date_and_encodes_nothing() {
    let world = encoded_world();
    let encoder = FakeEncoder::default();
    let probe = hub_at(COMMIT);
    let (result, fetched, status) = pull(&world, &probe, &encoder, SPEC);
    let Resolution::Encoded { path, fresh } = result.unwrap() else {
        panic!("an encoded entry stays encoded");
    };
    assert!(!fresh);
    assert_eq!(path, world.store.snapshot_dir(REPO, COMMIT).unwrap());
    assert!(encoder.calls().is_empty() && fetched.is_empty());
    assert_eq!(
        status,
        format!("{SPEC}@main is already up to date (commit {COMMIT})\n")
    );
    // The hub WAS asked — that is what a refresh is.
    assert_eq!(probe.asked(), ["<commit>"]);
}

#[test]
fn a_moved_revision_encodes_the_new_commit_repoints_and_prunes_the_old() {
    let world = encoded_world();
    let encoder = FakeEncoder::default();
    let (result, _, status) = pull(&world, &hub_at(NEXT_COMMIT), &encoder, SPEC);
    let Resolution::Encoded { path, fresh } = result.unwrap() else {
        panic!("a moved checkpoint re-encodes");
    };
    assert!(fresh);
    assert_eq!(path, world.store.snapshot_dir(REPO, NEXT_COMMIT).unwrap());
    assert_eq!(encoder.calls()[0].0, format!("hf://{REPO}@{NEXT_COMMIT}"));
    // The ref now answers offline with the new commit…
    assert_eq!(world.store.lookup(REPO, "main"), Some(path));
    // …and the superseded snapshot is gone.
    assert_eq!(world.store.snapshots(REPO), [NEXT_COMMIT]);
    for needle in [
        format!("moved: {COMMIT} → {NEXT_COMMIT}"),
        format!("pruned the superseded snapshot {COMMIT}"),
    ] {
        assert!(status.contains(&needle), "{needle} missing from {status}");
    }
}

#[test]
fn a_snapshot_another_revision_still_names_is_kept() {
    let world = encoded_world();
    world.store.record_ref(REPO, "v1", COMMIT).unwrap();
    let (result, _, status) = pull(&world, &hub_at(NEXT_COMMIT), &FakeEncoder::default(), SPEC);
    result.unwrap();
    assert_eq!(world.store.snapshots(REPO), [COMMIT, NEXT_COMMIT]);
    assert!(
        status.contains("another revision still points at it"),
        "{status}"
    );
}

#[test]
fn a_refused_re_encode_leaves_the_old_entry_runnable() {
    let world = encoded_world();
    let (result, _, _) = pull(&world, &hub_at(NEXT_COMMIT), &FakeEncoder::failing(), SPEC);
    assert!(result.is_err());
    assert_eq!(world.store.snapshots(REPO), [COMMIT]);
    assert_eq!(
        world.store.lookup(REPO, "main"),
        world.store.complete(REPO, COMMIT)
    );
}

#[test]
fn an_unreachable_hub_refuses_the_refresh_and_says_the_cache_still_runs() {
    let world = encoded_world();
    let (result, _, _) = pull(&world, &FakeProbe::offline(), &FakeEncoder::default(), SPEC);
    let err = result.unwrap_err().to_string();
    assert!(
        err.starts_with(&format!("cannot refresh {SPEC}@main")),
        "{err}"
    );
    assert!(
        err.contains(&format!("the cached commit {COMMIT} still runs offline")),
        "{err}"
    );
}

#[test]
fn a_full_commit_revision_is_up_to_date_without_asking_the_hub() {
    let world = encoded_world();
    let probe = FakeProbe::offline();
    let spec = format!("{SPEC}@{COMMIT}");
    let (result, _, status) = pull(&world, &probe, &FakeEncoder::default(), &spec);
    assert!(matches!(
        result.unwrap(),
        Resolution::Encoded { fresh: false, .. }
    ));
    assert!(probe.asked().is_empty());
    assert!(status.contains("already up to date"), "{status}");
}

#[test]
fn a_new_revision_of_an_encoded_repo_is_encoded_without_pruning() {
    let world = encoded_world();
    let encoder = FakeEncoder::default();
    let (result, _, status) = pull(
        &world,
        &hub_at(NEXT_COMMIT),
        &encoder,
        &format!("{SPEC}@v2"),
    );
    result.unwrap();
    assert!(status.contains("is not encoded yet"), "{status}");
    assert!(!status.contains("pruned"), "{status}");
    assert_eq!(world.store.snapshots(REPO), [COMMIT, NEXT_COMMIT]);
}

// ── shorthand → the spec `pull` refreshes ─────────────────────────────

#[test]
fn a_shorthand_names_the_revision_its_entry_tracks() {
    let world = encoded_world();
    let name = shorthand(REPO);
    assert_eq!(
        pull_spec_for_shorthand(&world.store, name).unwrap(),
        Some(SPEC.to_string())
    );
    assert_eq!(
        pull_spec_for_shorthand(&world.store, "other").unwrap(),
        None
    );
}

#[test]
fn a_shorthand_tracking_one_non_default_revision_pulls_that_revision() {
    let world = world();
    let spec = format!("{SPEC}@v1");
    pull(&world, &hub_at(COMMIT), &FakeEncoder::default(), &spec)
        .0
        .unwrap();
    assert_eq!(
        pull_spec_for_shorthand(&world.store, shorthand(REPO)).unwrap(),
        Some(spec)
    );
}

#[test]
fn a_shorthand_encoded_by_commit_pulls_that_commit() {
    let world = world();
    let spec = format!("{SPEC}@{COMMIT}");
    pull(&world, &hub_at(COMMIT), &FakeEncoder::default(), &spec)
        .0
        .unwrap();
    assert!(world.store.refs(REPO).is_empty());
    assert_eq!(
        pull_spec_for_shorthand(&world.store, shorthand(REPO)).unwrap(),
        Some(spec)
    );
}

#[test]
fn an_ambiguous_shorthand_is_refused_naming_the_candidates() {
    let world = encoded_world();
    world.store.record_ref(REPO, "v1", COMMIT).unwrap();
    std::fs::remove_file(world.store.repo_dir(REPO).unwrap().join("refs/main")).unwrap();
    world.store.record_ref(REPO, "v2", COMMIT).unwrap();
    let err = pull_spec_for_shorthand(&world.store, shorthand(REPO))
        .unwrap_err()
        .to_string();
    assert!(err.contains("v1, v2"), "{err}");

    // Two repos sharing a name half.
    let other = "other-org/tiny-model";
    let snap = world.store.snapshot_dir(other, COMMIT).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(snap.join(INDEX_JSON), b"{}").unwrap();
    let err = pull_spec_for_shorthand(&world.store, shorthand(REPO))
        .unwrap_err()
        .to_string();
    assert!(err.contains("several encoded checkpoints"), "{err}");
}

#[test]
fn a_shorthand_with_several_unreferenced_snapshots_names_them() {
    let world = world();
    for commit in [COMMIT, NEXT_COMMIT] {
        let snap = world.store.snapshot_dir(REPO, commit).unwrap();
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join(INDEX_JSON), b"{}").unwrap();
    }
    let err = pull_spec_for_shorthand(&world.store, shorthand(REPO))
        .unwrap_err()
        .to_string();
    assert!(err.contains(COMMIT) && err.contains(NEXT_COMMIT), "{err}");
}
