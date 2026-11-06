//! Tests for the shared cache scan (`super`).

use super::*;

/// Build a fake HF hub layout under `root` with the given
/// `owner/name` entries. Each entry gets one snapshot containing
/// `index.json` plus a small `stub.bin` so size calculations have
/// something to report. Uses the `models--` prefix to match the
/// `larql publish` default; for dataset-prefix coverage see
/// [`build_fake_hub_with_prefix`].
fn build_fake_hub(root: &Path, repos: &[&str]) {
    build_fake_hub_with_prefix(root, "models--", repos);
}

fn build_fake_hub_with_prefix(root: &Path, prefix: &str, repos: &[&str]) {
    for repo in repos {
        let (owner, name) = repo.split_once('/').expect("owner/name");
        let dir = root.join(format!("{prefix}{owner}--{name}/snapshots/abc123"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(INDEX_JSON), b"{}").unwrap();
        std::fs::write(dir.join("stub.bin"), vec![0u8; 1024]).unwrap();
    }
}

/// Build a fake local cache under `root` with the given bare names.
/// Each is a `<name>.vindex/` directory with an `index.json`.
fn build_fake_local(root: &Path, names: &[&str]) {
    for name in names {
        let dir = root.join(format!("{name}.vindex"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(INDEX_JSON), b"{}").unwrap();
        std::fs::write(dir.join("stub.bin"), vec![0u8; 512]).unwrap();
    }
}

// ── HF-only scan (legacy single-cache shape) ────────────────────

#[test]
fn scan_returns_empty_for_missing_dir() {
    let root = std::path::PathBuf::from("/definitely/not/a/path");
    let out = scan_hf_hub_at(&root).unwrap();
    assert!(out.is_empty());
}

#[test]
fn scan_finds_cached_vindexes_and_sorts() {
    let tmp = tempfile::tempdir().unwrap();
    build_fake_hub(tmp.path(), &["zebra/last", "acme/first", "beta/mid"]);
    let out = scan_hf_hub_at(tmp.path()).unwrap();
    let repos: Vec<_> = out.iter().map(|c| c.repo.as_str()).collect();
    assert_eq!(repos, vec!["acme/first", "beta/mid", "zebra/last"]);
    assert!(out.iter().all(|c| c.source == CacheSource::HuggingFace));
}

#[test]
fn scan_skips_snapshots_without_index_json() {
    let tmp = tempfile::tempdir().unwrap();
    let bare = tmp.path().join("models--foo--bar/snapshots/deadbeef");
    std::fs::create_dir_all(&bare).unwrap();
    std::fs::write(bare.join("not-a-vindex.txt"), b"hi").unwrap();
    let out = scan_hf_hub_at(tmp.path()).unwrap();
    assert!(
        out.is_empty(),
        "snapshot without index.json should be skipped"
    );
}

#[test]
fn scan_finds_both_models_and_datasets_prefixes() {
    let tmp = tempfile::tempdir().unwrap();
    build_fake_hub_with_prefix(tmp.path(), "models--", &["chrishayuk/gemma-3-4b-it-vindex"]);
    build_fake_hub_with_prefix(tmp.path(), "datasets--", &["legacy/older-vindex"]);
    let out = scan_hf_hub_at(tmp.path()).unwrap();
    let repos: Vec<_> = out.iter().map(|c| c.repo.as_str()).collect();
    assert_eq!(
        repos,
        vec!["chrishayuk/gemma-3-4b-it-vindex", "legacy/older-vindex"]
    );
    assert!(out.iter().all(|c| c.source == CacheSource::HuggingFace));
}

#[test]
fn scan_ignores_non_repo_hub_entries() {
    // hf-hub also drops `.locks/`, `blobs/`, and `version.txt` into
    // the hub root. None of these are repo dirs and must be skipped
    // (they have no `--` and no `snapshots/`).
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join(".locks")).unwrap();
    std::fs::create_dir_all(tmp.path().join("blobs")).unwrap();
    std::fs::write(tmp.path().join("version.txt"), b"1\n").unwrap();
    let out = scan_hf_hub_at(tmp.path()).unwrap();
    assert!(out.is_empty());
}

#[test]
fn scan_records_nonzero_size_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    build_fake_hub(tmp.path(), &["o/one"]);
    let out = scan_hf_hub_at(tmp.path()).unwrap();
    assert_eq!(out.len(), 1);
    assert!(out[0].size_bytes >= 1024);
}

// ── Local cache scan ────────────────────────────────────────────

#[test]
fn scan_local_empty_when_dir_missing() {
    let out = scan_local_at(Path::new("/does/not/exist")).unwrap();
    assert!(out.is_empty());
}

#[test]
fn scan_local_finds_bare_name_entries() {
    let tmp = tempfile::tempdir().unwrap();
    build_fake_local(tmp.path(), &["gemma3-4b-f16", "v10c-tinystories"]);
    let out = scan_local_at(tmp.path()).unwrap();
    assert_eq!(out.len(), 2);
    // sort_by repo; alphabetical → gemma first, v10c second
    assert_eq!(out[0].repo, "gemma3-4b-f16");
    assert_eq!(out[1].repo, "v10c-tinystories");
    assert!(out.iter().all(|c| c.source == CacheSource::Local));
}

#[test]
fn scan_local_skips_non_vindex_dirs() {
    let tmp = tempfile::tempdir().unwrap();
    // no index.json
    std::fs::create_dir_all(tmp.path().join("junk.vindex")).unwrap();
    std::fs::write(tmp.path().join("loose-file.txt"), b"nope").unwrap();
    let out = scan_local_at(tmp.path()).unwrap();
    assert!(out.is_empty());
}

/// Unix-only — Windows directory symlinks require admin or
/// developer-mode and aren't available on the default
/// `windows-latest` runner. The "resolves symlinks" behaviour
/// itself is unix-specific (`std::os::unix::fs::symlink`); on
/// Windows the scan path falls back to walking the directory
/// normally, which is covered by `scan_local_finds_bare_name_entries`.
#[cfg(unix)]
#[test]
fn scan_local_resolves_symlinks() {
    let tmp = tempfile::tempdir().unwrap();
    let local = tmp.path().join("local");
    let target = tmp.path().join("src/my-model.vindex");
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join(INDEX_JSON), b"{}").unwrap();
    std::fs::create_dir_all(&local).unwrap();
    std::os::unix::fs::symlink(&target, local.join("my-model.vindex")).unwrap();
    let out = scan_local_at(&local).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].repo, "my-model");
}

// ── Merged scan ─────────────────────────────────────────────────

#[test]
fn scan_both_merges_and_orders_local_first() {
    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path().join("hub");
    let local = tmp.path().join("local");
    std::fs::create_dir_all(&hub).unwrap();
    std::fs::create_dir_all(&local).unwrap();
    build_fake_hub(&hub, &["chrishayuk/gemma-3-4b-it-vindex"]);
    build_fake_local(&local, &["gemma4-31b-f16"]);
    let out = scan_cached_vindexes_at_all(&hub, &local, &tmp.path().join("encoded")).unwrap();
    assert_eq!(out.len(), 2);
    // Local first.
    assert_eq!(out[0].source, CacheSource::HuggingFace);
    assert_eq!(out[1].source, CacheSource::Local);
}

// ── Shorthand resolution ────────────────────────────────────────

#[test]
fn shorthand_unique_match_returns_snapshot_path() {
    let tmp = tempfile::tempdir().unwrap();
    build_fake_hub(tmp.path(), &["alice/cool-vindex"]);
    let cache = scan_hf_hub_at(tmp.path()).unwrap();
    let path = resolve_shorthand_from("cool-vindex", &cache).unwrap();
    assert!(path.ends_with("snapshots/abc123"));
}

#[test]
fn shorthand_matches_bare_local_name() {
    let tmp = tempfile::tempdir().unwrap();
    build_fake_local(tmp.path(), &["gemma4-31b-f16"]);
    let cache = scan_local_at(tmp.path()).unwrap();
    let path = resolve_shorthand_from("gemma4-31b-f16", &cache).unwrap();
    assert!(path.ends_with("gemma4-31b-f16.vindex"));
}

#[test]
fn shorthand_no_match_mentions_both_registration_paths() {
    let cache: Vec<CachedVindex> = Vec::new();
    let err = resolve_shorthand_from("missing", &cache).unwrap_err();
    let s = err.to_string();
    assert!(s.contains("no cached vindex matches `missing`"));
    assert!(s.contains("larql pull"));
    assert!(s.contains("larql link"));
}

#[test]
fn shorthand_ambiguous_across_hf_and_local_errors_with_sources() {
    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path().join("hub");
    let local = tmp.path().join("local");
    std::fs::create_dir_all(&hub).unwrap();
    std::fs::create_dir_all(&local).unwrap();
    build_fake_hub(&hub, &["someone/gemma4-31b"]);
    build_fake_local(&local, &["gemma4-31b"]);
    let cache = scan_cached_vindexes_at_all(&hub, &local, &tmp.path().join("encoded")).unwrap();
    let err = resolve_shorthand_from("gemma4-31b", &cache).unwrap_err();
    let s = err.to_string();
    assert!(s.contains("ambiguous"));
    assert!(s.contains("[hf]"));
    assert!(s.contains("[local]"));
}

// ── resolve_cached ──────────────────────────────────────────────

#[test]
fn resolve_cached_accepts_owner_slash_name() {
    let tmp = tempfile::tempdir().unwrap();
    build_fake_hub(tmp.path(), &["alice/x", "bob/y"]);
    let cache = scan_hf_hub_at(tmp.path()).unwrap();
    let hit = resolve_cached_from("alice/x", &cache).unwrap();
    assert_eq!(hit.repo, "alice/x");
    assert_eq!(hit.source, CacheSource::HuggingFace);
}

#[test]
fn resolve_cached_strips_hf_scheme() {
    let tmp = tempfile::tempdir().unwrap();
    build_fake_hub(tmp.path(), &["alice/x"]);
    let cache = scan_hf_hub_at(tmp.path()).unwrap();
    let hit = resolve_cached_from("hf://alice/x", &cache).unwrap();
    assert_eq!(hit.repo, "alice/x");
}

#[test]
fn resolve_cached_rejects_uncached_owner_slash_name() {
    let cache: Vec<CachedVindex> = Vec::new();
    let err = resolve_cached_from("not/here", &cache).unwrap_err();
    assert!(err.to_string().contains("not cached: not/here"));
}

#[test]
fn resolve_cached_accepts_hf_shorthand() {
    let tmp = tempfile::tempdir().unwrap();
    build_fake_hub(tmp.path(), &["alice/unique-name"]);
    let cache = scan_hf_hub_at(tmp.path()).unwrap();
    let hit = resolve_cached_from("unique-name", &cache).unwrap();
    assert_eq!(hit.repo, "alice/unique-name");
}

#[test]
fn resolve_cached_accepts_local_shorthand() {
    let tmp = tempfile::tempdir().unwrap();
    build_fake_local(tmp.path(), &["my-extract"]);
    let cache = scan_local_at(tmp.path()).unwrap();
    let hit = resolve_cached_from("my-extract", &cache).unwrap();
    assert_eq!(hit.repo, "my-extract");
    assert_eq!(hit.source, CacheSource::Local);
}

#[test]
fn shorthand_key_strips_owner_slash() {
    assert_eq!(shorthand_key("owner/name"), "name");
    assert_eq!(shorthand_key("just-name"), "just-name");
    assert_eq!(shorthand_key("a/b/c"), "c"); // rsplit_once — last segment wins
}

// ── Sizes ───────────────────────────────────────────────────────────

/// An HF snapshot is symlinks into `blobs/`; its size is the blobs'.
#[cfg(unix)]
#[test]
fn hf_snapshot_size_follows_symlinks_to_blobs() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = tmp.path().join("models--o--linked");
    let blobs = repo.join("blobs");
    let snap = repo.join("snapshots/abc123");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(blobs.join("idx"), b"{}").unwrap();
    std::fs::write(blobs.join("weights"), vec![0u8; 4096]).unwrap();
    std::os::unix::fs::symlink(blobs.join("idx"), snap.join(INDEX_JSON)).unwrap();
    std::os::unix::fs::symlink(blobs.join("weights"), snap.join("weights.bin")).unwrap();
    let out = scan_hf_hub_at(tmp.path()).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].size_bytes, 4096 + 2);
}

/// Directory symlinks are not followed (a cycle must not hang the scan)
/// and a dangling link counts nothing; real subdirectories are walked.
#[cfg(unix)]
#[test]
fn dir_size_walks_real_subdirs_but_not_dir_links_or_dangling_links() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("v");
    std::fs::create_dir_all(root.join("segments")).unwrap();
    std::fs::write(root.join("a.bin"), vec![0u8; 10]).unwrap();
    std::fs::write(root.join("segments/b.bin"), vec![0u8; 20]).unwrap();
    std::os::unix::fs::symlink(&root, root.join("loop")).unwrap();
    std::os::unix::fs::symlink(root.join("missing"), root.join("dangling")).unwrap();
    assert_eq!(dir_size_bytes(&root).unwrap(), 30);
}

// ── Encoded cache (`hf://` checkpoints encoded to VINDEX3) ───────────

use crate::commands::primary::hf_auto::store::EncodedStore;

/// A full-length commit id for encoded-cache fixtures.
const ENCODED_COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

/// Publish a minimal encoded container for `repo` through the store's own
/// partial-then-rename path.
fn build_fake_encoded(store: &EncodedStore, repo: &str) -> PathBuf {
    let partial = store.begin(repo).unwrap();
    std::fs::create_dir_all(partial.path()).unwrap();
    std::fs::write(partial.path().join(INDEX_JSON), b"{}").unwrap();
    partial
        .commit(&store.snapshot_dir(repo, ENCODED_COMMIT).unwrap())
        .unwrap()
}

#[test]
fn encoded_entries_are_scanned_labelled_and_resolved() {
    let tmp = tempfile::tempdir().unwrap();
    let (hub, local) = (tmp.path().join("hub"), tmp.path().join("local"));
    let store = EncodedStore::new(tmp.path().join("encoded"));
    let snap = build_fake_encoded(&store, "Qwen/Qwen3-0.6B");
    // An in-flight encode beside it is invisible.
    let inflight = store.begin("acme/inflight").unwrap();
    std::fs::create_dir_all(inflight.path()).unwrap();
    std::fs::write(inflight.path().join(INDEX_JSON), b"{}").unwrap();
    build_fake_hub(&hub, &["chrishayuk/some-vindex"]);

    let all = scan_cached_vindexes_at_all(&hub, &local, store.root()).unwrap();
    let repos: Vec<_> = all.iter().map(|c| (c.repo.as_str(), c.source)).collect();
    assert_eq!(
        repos,
        vec![
            ("chrishayuk/some-vindex", CacheSource::HuggingFace),
            ("Qwen/Qwen3-0.6B", CacheSource::Encoded),
        ]
    );
    assert_eq!(CacheSource::Encoded.label(), "encoded");

    // Runnable by shorthand and by owner/name; `rm` finds it without network.
    assert_eq!(resolve_shorthand_from("Qwen3-0.6B", &all).unwrap(), snap);
    let hit = resolve_cached_from("hf://Qwen/Qwen3-0.6B", &all).unwrap();
    assert_eq!(hit.source, CacheSource::Encoded);
    // `rm` evicts the repo directory: two levels above the snapshot.
    assert_eq!(
        hit.snapshot.parent().and_then(Path::parent),
        Some(store.repo_dir("Qwen/Qwen3-0.6B").unwrap().as_path())
    );
}

#[test]
fn larql_home_roots_every_larql_cache() {
    let home = larql_home().unwrap();
    assert_eq!(larql_local_dir().unwrap(), home.join("local"));
    assert_eq!(
        EncodedStore::open_default().unwrap().root(),
        home.join("encoded")
    );
}

#[test]
fn shorthand_miss_mentions_first_run_encode() {
    let err = resolve_shorthand_from("missing", &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("larql run hf://owner/missing"), "{err}");
}
