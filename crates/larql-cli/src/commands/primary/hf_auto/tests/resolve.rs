//! The `hf://` decision end to end, with every collaborator faked.

use super::*;
use crate::commands::primary::hf_auto::classify::CHECKPOINT_CONFIG;
use crate::commands::primary::hf_auto::resolve::*;
use crate::commands::primary::hf_auto::store::EncodedStore;

struct World {
    _tmp: tempfile::TempDir,
    store: EncodedStore,
    hub: PathBuf,
}

fn world() -> World {
    let tmp = tempfile::tempdir().unwrap();
    let store = EncodedStore::new(tmp.path().join("encoded"));
    let hub = tmp.path().join("hub");
    World {
        _tmp: tmp,
        store,
        hub,
    }
}

/// What one resolve did: its result, the vindex fetches it made, and
/// what it told the user.
struct Outcome {
    result: Result<Resolution, BoxErr>,
    fetched: Vec<String>,
    status: String,
}

fn resolve(world: &World, probe: &FakeProbe, encoder: &FakeEncoder, spec: &str) -> Outcome {
    let mut fetched = Vec::new();
    let vindex_dir = world.hub.join("fetched-vindex");
    let mut fetch = |s: &str| -> Result<PathBuf, BoxErr> {
        fetched.push(s.to_string());
        Ok(vindex_dir.clone())
    };
    let mut status = Vec::new();
    let result = HfAuto {
        store: &world.store,
        hub: &world.hub,
        probe,
        encoder,
    }
    .resolve(spec, &mut fetch, &mut status);
    Outcome {
        result,
        fetched,
        status: String::from_utf8(status).unwrap(),
    }
}

#[test]
fn a_checkpoint_is_encoded_once_then_answered_offline() {
    let world = world();
    let encoder = FakeEncoder::default();
    let probe = FakeProbe::carrying(&[CHECKPOINT_CONFIG]);
    let first = resolve(&world, &probe, &encoder, "hf://acme/tiny-model");
    let Resolution::Encoded { path, fresh } = first.result.unwrap() else {
        panic!("a checkpoint must encode");
    };
    assert!(fresh);
    assert_eq!(path, world.store.snapshot_dir(REPO, COMMIT).unwrap());
    assert!(
        first.fetched.is_empty(),
        "a checkpoint is never fetched as a vindex"
    );
    // The encoder was handed the spec pinned to the commit, and a partial dir.
    let calls = encoder.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, format!("hf://{REPO}@{COMMIT}"));
    assert_ne!(
        calls[0].1, path,
        "the encoder writes a partial, not the entry"
    );
    for needle in [
        "is a Hugging Face checkpoint",
        "planning a VINDEX3 encode (first run only)",
        COMMIT,
        "cached at",
        "larql run tiny-model",
    ] {
        assert!(
            first.status.contains(needle),
            "{needle} in {}",
            first.status
        );
    }

    // Second run: no probe, no encode, no output.
    let offline = FakeProbe::offline();
    let second = resolve(&world, &offline, &encoder, "hf://acme/tiny-model");
    assert_eq!(
        second.result.unwrap(),
        Resolution::Encoded {
            path: path.clone(),
            fresh: false
        }
    );
    assert!(
        offline.asked().is_empty(),
        "a cached encode needs no network"
    );
    assert_eq!(encoder.calls().len(), 1);
    assert!(second.status.is_empty());

    // The pinned commit spelling hits the same entry, offline too.
    let pinned = resolve(&world, &offline, &encoder, &format!("hf://{REPO}@{COMMIT}"));
    assert_eq!(pinned.result.unwrap().into_path(), path);
}

#[test]
fn a_new_revision_of_an_encoded_commit_reuses_it() {
    let world = world();
    let encoder = FakeEncoder::default();
    let probe = FakeProbe::carrying(&[CHECKPOINT_CONFIG]);
    resolve(&world, &probe, &encoder, "hf://acme/tiny-model")
        .result
        .unwrap();
    // `v1` resolves to the same commit: no second encode, ref recorded.
    let tagged = resolve(&world, &probe, &encoder, "hf://acme/tiny-model@v1");
    assert_eq!(
        tagged.result.unwrap(),
        Resolution::Encoded {
            path: world.store.snapshot_dir(REPO, COMMIT).unwrap(),
            fresh: false
        }
    );
    assert_eq!(encoder.calls().len(), 1);
    assert!(world.store.lookup(REPO, "v1").is_some());
}

#[test]
fn a_failed_encode_caches_nothing_and_reports_the_error() {
    let world = world();
    let encoder = FakeEncoder::failing();
    let probe = FakeProbe::carrying(&[CHECKPOINT_CONFIG]);
    let out = resolve(&world, &probe, &encoder, "hf://acme/tiny-model");
    assert_eq!(out.result.unwrap_err().to_string(), "encode refused");
    assert!(world.store.complete(REPO, COMMIT).is_none());
    assert!(world.store.lookup(REPO, "main").is_none());
    assert!(
        !world.store.repo_dir(REPO).unwrap().exists(),
        "partial removed"
    );
}

#[test]
fn a_published_vindex_takes_the_vindex_path_unchanged() {
    let world = world();
    let encoder = FakeEncoder::default();
    let probe = FakeProbe::carrying(&[INDEX_JSON, CHECKPOINT_CONFIG]);
    let out = resolve(&world, &probe, &encoder, "hf://acme/tiny-model@rev");
    assert!(matches!(out.result.unwrap(), Resolution::Vindex(_)));
    assert_eq!(out.fetched, vec!["hf://acme/tiny-model@rev".to_string()]);
    assert!(encoder.calls().is_empty());
}

#[test]
fn a_hub_cached_vindex_is_answered_without_a_probe() {
    let world = world();
    fake_hub_vindex(&world.hub, REPO);
    let probe = FakeProbe::offline();
    let out = resolve(
        &world,
        &probe,
        &FakeEncoder::default(),
        "hf://acme/tiny-model",
    );
    assert!(matches!(out.result.unwrap(), Resolution::Vindex(_)));
    assert!(probe.asked().is_empty());
}

#[test]
fn unrecognised_or_unreachable_repos_keep_the_vindex_path_and_its_errors() {
    for probe in [FakeProbe::carrying(&[]), FakeProbe::offline()] {
        let world = world();
        let encoder = FakeEncoder::default();
        let out = resolve(&world, &probe, &encoder, "hf://acme/tiny-model");
        assert!(matches!(out.result.unwrap(), Resolution::Vindex(_)));
        assert_eq!(out.fetched.len(), 1);
        assert!(encoder.calls().is_empty());
    }
}

#[test]
fn an_unpinnable_revision_is_refused_unless_it_is_a_commit() {
    let world = world();
    let encoder = FakeEncoder::default();
    let mut probe = FakeProbe::carrying(&[CHECKPOINT_CONFIG]);
    probe.commit = None;
    let err = resolve(&world, &probe, &encoder, "hf://acme/tiny-model")
        .result
        .unwrap_err()
        .to_string();
    assert!(err.contains("named no commit"), "{err}");
    assert!(encoder.calls().is_empty());

    let by_commit = resolve(&world, &probe, &encoder, &format!("hf://{REPO}@{COMMIT}"));
    assert!(matches!(
        by_commit.result.unwrap(),
        Resolution::Encoded { fresh: true, .. }
    ));
}

#[test]
fn malformed_specs_are_refused_before_anything_runs() {
    let world = world();
    let probe = FakeProbe::offline();
    for spec in ["not-hf", "hf://", "hf://acme/x@"] {
        assert!(resolve(&world, &probe, &FakeEncoder::default(), spec)
            .result
            .is_err());
    }
    // A spec that parses but names no `owner/name` cannot be stored.
    let checkpoint = FakeProbe::carrying(&[CHECKPOINT_CONFIG]);
    let err = resolve(&world, &checkpoint, &FakeEncoder::default(), "hf://lonely")
        .result
        .unwrap_err();
    assert!(err.to_string().contains("owner/name"), "{err}");
}

#[test]
fn resolution_accessors_and_helpers() {
    let p = PathBuf::from("/x");
    assert_eq!(Resolution::Vindex(p.clone()).into_path(), p);
    let e = Resolution::Encoded {
        path: p.clone(),
        fresh: true,
    };
    assert_eq!(e.clone().into_path(), p);
    assert!(is_full_commit(COMMIT));
    assert!(!is_full_commit("main"));
    assert!(!is_full_commit(&"g".repeat(40)));
    assert_eq!(shorthand("acme/tiny"), "tiny");
    assert_eq!(shorthand("tiny"), "tiny");
}
