//! The installed-only resolver: names, `owner/name`, `hf://` and paths
//! resolve against fake caches; a miss never fetches and says how to.

use super::*;
use larql_vindex::format::filenames::INDEX_JSON;

/// A pinned commit id the encoded store accepts as a snapshot key.
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
/// A revision the encoded store has a recorded ref for.
const RECORDED_REVISION: &str = "v1";

/// A snapshot under a hub-layout root for `owner/name`, holding `index.json`.
fn hub_entry(root: &Path, repo: &str) -> PathBuf {
    let (owner, name) = repo.split_once('/').unwrap();
    let dir = root.join(format!("models--{owner}--{name}/snapshots/{COMMIT}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(INDEX_JSON), b"{}").unwrap();
    dir
}

/// A `larql link` registration named `name`.
fn local_entry(root: &Path, name: &str) -> PathBuf {
    let dir = root.join(format!("{name}{}", super::super::LOCAL_LINK_SUFFIX));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(INDEX_JSON), b"{}").unwrap();
    dir
}

fn roots(tmp: &Path) -> CacheRoots {
    CacheRoots {
        hub: tmp.join("hub"),
        local: tmp.join("local"),
        encoded: tmp.join("encoded"),
    }
}

fn resolve(model: &str, roots: &CacheRoots) -> Result<PathBuf, String> {
    resolve_installed_model_in(Path::new(model), roots).map_err(|e| e.to_string())
}

#[test]
fn an_existing_directory_is_returned_unchanged_even_when_a_cached_name_matches() {
    let tmp = tempfile::tempdir().unwrap();
    let roots = roots(tmp.path());
    hub_entry(&roots.hub, "acme/model");
    let shadow = tmp.path().join("model");
    std::fs::create_dir_all(&shadow).unwrap();
    let got = resolve_installed_model_in(&shadow, &roots).unwrap();
    assert_eq!(got, shadow);
}

#[test]
fn an_existing_file_is_returned_unchanged_for_the_verb_to_judge() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("not-a-container.json");
    std::fs::write(&file, b"{}").unwrap();
    assert_eq!(resolve_installed_model(&file).unwrap(), file);
}

#[test]
fn a_plain_name_resolves_to_its_hub_entry() {
    let tmp = tempfile::tempdir().unwrap();
    let roots = roots(tmp.path());
    let snap = hub_entry(&roots.hub, "Qwen/Qwen3-0.6B");
    assert_eq!(resolve("Qwen3-0.6B", &roots).unwrap(), snap);
}

#[test]
fn a_plain_name_resolves_to_an_encoded_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();
    let roots = roots(tmp.path());
    let snap = hub_entry(&roots.encoded, "Qwen/Qwen3-0.6B");
    assert_eq!(resolve("Qwen3-0.6B", &roots).unwrap(), snap);
}

#[test]
fn a_plain_name_resolves_to_a_linked_container() {
    let tmp = tempfile::tempdir().unwrap();
    let roots = roots(tmp.path());
    let dir = local_entry(&roots.local, "my-build");
    assert_eq!(resolve("my-build", &roots).unwrap(), dir);
}

#[test]
fn owner_name_resolves_to_exactly_that_repo() {
    let tmp = tempfile::tempdir().unwrap();
    let roots = roots(tmp.path());
    let wanted = hub_entry(&roots.hub, "acme/model");
    hub_entry(&roots.hub, "other/model");
    assert_eq!(resolve("acme/model", &roots).unwrap(), wanted);
}

#[test]
fn an_ambiguous_shorthand_lists_every_candidate() {
    let tmp = tempfile::tempdir().unwrap();
    let roots = roots(tmp.path());
    hub_entry(&roots.hub, "acme/model");
    hub_entry(&roots.encoded, "other/model");
    let err = resolve("model", &roots).unwrap_err();
    assert!(err.contains("ambiguous"), "{err}");
    assert!(err.contains("acme/model [hf]"), "{err}");
    assert!(err.contains("other/model [encoded]"), "{err}");
}

#[test]
fn hf_uri_resolves_to_a_cached_vindex_at_the_default_revision() {
    let tmp = tempfile::tempdir().unwrap();
    let roots = roots(tmp.path());
    let snap = hub_entry(&roots.hub, "acme/model");
    assert_eq!(resolve("hf://acme/model", &roots).unwrap(), snap);
}

#[test]
fn hf_uri_resolves_a_recorded_revision_from_the_encoded_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let roots = roots(tmp.path());
    let snap = hub_entry(&roots.encoded, "acme/model");
    let store = EncodedStore::new(&roots.encoded);
    store
        .record_ref("acme/model", RECORDED_REVISION, COMMIT)
        .unwrap();
    let spec = format!("hf://acme/model@{RECORDED_REVISION}");
    assert_eq!(resolve(&spec, &roots).unwrap(), snap);
    let pinned = format!("hf://acme/model@{COMMIT}");
    assert_eq!(resolve(&pinned, &roots).unwrap(), snap);
}

#[test]
fn hf_uri_at_an_unknown_revision_is_not_answered_by_another_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let roots = roots(tmp.path());
    hub_entry(&roots.hub, "acme/model");
    let err = resolve("hf://acme/model@some-branch", &roots).unwrap_err();
    assert!(err.contains("never downloads"), "{err}");
}

#[test]
fn a_malformed_hf_uri_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let err = resolve("hf://", &roots(tmp.path())).unwrap_err();
    assert!(err.contains("malformed"), "{err}");
}

#[test]
fn a_missing_name_names_the_commands_that_fetch_it() {
    let tmp = tempfile::tempdir().unwrap();
    let err = resolve("Qwen3-0.6B", &roots(tmp.path())).unwrap_err();
    assert!(err.contains("never downloads"), "{err}");
    assert!(err.contains("larql run hf://<owner>/Qwen3-0.6B"), "{err}");
    assert!(err.contains("larql pull hf://<owner>/Qwen3-0.6B"), "{err}");
    assert!(err.contains("larql list"), "{err}");
}

#[test]
fn a_missing_owner_name_hints_its_own_hf_uri() {
    let tmp = tempfile::tempdir().unwrap();
    let err = resolve("Qwen/Qwen3-0.6B", &roots(tmp.path())).unwrap_err();
    assert!(err.contains("larql run hf://Qwen/Qwen3-0.6B "), "{err}");
    let err = resolve("hf://Qwen/Qwen3-0.6B", &roots(tmp.path())).unwrap_err();
    assert!(err.contains("larql pull hf://Qwen/Qwen3-0.6B "), "{err}");
}

#[test]
fn a_missing_path_passes_through_unchanged_and_is_not_looked_up_as_a_name() {
    let tmp = tempfile::tempdir().unwrap();
    let roots = roots(tmp.path());
    // A cache entry with the path's final segment must not answer it.
    local_entry(&roots.local, "model");
    let absolute = tmp.path().join("absent").join("model");
    for spec in [
        absolute.to_str().unwrap(),
        "./model",
        "../model",
        "~/model",
        r"C:\models\model",
        r"models\model",
        "/absent/model",
    ] {
        assert_eq!(
            resolve(spec, &roots).unwrap(),
            PathBuf::from(spec),
            "{spec}"
        );
    }
}

#[cfg(unix)]
#[test]
fn a_non_utf8_missing_path_passes_through() {
    use std::os::unix::ffi::OsStrExt;
    let tmp = tempfile::tempdir().unwrap();
    let name = Path::new(std::ffi::OsStr::from_bytes(b"model-\xff"));
    assert_eq!(
        resolve_installed_model_in(name, &roots(tmp.path())).unwrap(),
        name
    );
}

#[test]
fn the_default_roots_follow_the_cache_helpers() {
    let roots = CacheRoots::from_env().unwrap();
    assert_eq!(roots.hub, super::super::hf_hub_dir().unwrap());
    assert_eq!(roots.local, super::super::larql_local_dir().unwrap());
}

#[test]
fn the_env_resolver_passes_a_missing_path_through_for_the_verb_to_report() {
    let path = Path::new("./larql-no-such-container");
    assert_eq!(resolve_installed_model(path).unwrap(), path);
}

#[test]
fn the_env_resolver_looks_a_name_up_in_the_default_caches() {
    // A name no cache can hold: the lookup runs and misses with guidance.
    let err = resolve_installed_model(Path::new("larql-test-no-such-model"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("never downloads"), "{err}");
}
