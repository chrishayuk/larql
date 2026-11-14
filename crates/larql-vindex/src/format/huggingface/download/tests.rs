//! Tests for [`super`].
//!
//! Split out of `mod.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

//! Unit tests for the hf_hub-bound functions — pure helpers tested
//! in `helpers.rs`.
//!
//! VINDEX3 generation-aware download-completeness tests live in the
//! sibling `tests_v3.rs`; both files share the mock/env scaffolding in
//! `test_support.rs`.
use super::test_support::{mock_hf_file_resolve, HfTestEnv, NoOpProgress};
use super::*;
use serial_test::serial;

// ─── hf_hub-bound functions: not-an-hf-path early return ────────────
//
// These four functions all share the same `hf://` strip_prefix +
// `@revision` parsing + `Api::new()` setup head. Pin the early-return
// path that fires when the input doesn't start with `hf://`. No HTTP
// mocking needed — the error fires before any network call.

#[test]
fn resolve_hf_vindex_rejects_non_hf_path() {
    let err = resolve_hf_vindex("/local/path").expect_err("must reject local paths");
    assert!(err.to_string().contains("not an hf://"));
}

#[test]
fn resolve_hf_vindex_rejects_https_url() {
    let err = resolve_hf_vindex("https://huggingface.co/owner/repo").expect_err("must reject");
    assert!(err.to_string().contains("not an hf://"));
}

#[test]
fn download_hf_weights_rejects_non_hf_path() {
    let err = download_hf_weights("./relative").expect_err("must reject");
    assert!(err.to_string().contains("not an hf://"));
}

#[test]
fn download_hf_weights_rejects_empty_string() {
    let err = download_hf_weights("").expect_err("must reject empty");
    assert!(err.to_string().contains("not an hf://"));
}

#[test]
fn resolve_hf_vindex_with_progress_rejects_non_hf_path() {
    let err =
        resolve_hf_vindex_with_progress("/tmp/foo", |_| NoOpProgress).expect_err("must reject");
    assert!(err.to_string().contains("not an hf://"));
}

#[test]
fn resolve_hf_model_with_progress_rejects_non_hf_path() {
    let err =
        resolve_hf_model_with_progress("./local-model", |_| NoOpProgress).expect_err("must reject");
    assert!(err.to_string().contains("not an hf://"));
}

// ─── hf_hub-bound: revision parsing covered by error path ──────────
//
// The `@revision` split happens after the `hf://` prefix strip but
// before any network call. The functions then do `Api::new()` which
// (with HF_ENDPOINT pointing at a non-existent server) fails fast.
// That path covers the revision-vs-no-revision branches.

#[test]
#[serial]
fn resolve_hf_vindex_errors_when_both_repo_kinds_404() {
    // mockito returns 404 for every URL → the Model probe (the only
    // entry in HF_PULL_REPO_KINDS now that the dataset fallback is
    // gone) fails → resolve_hf_vindex returns the wrapped
    // "failed to download index.json" error. Exercises: hf:// strip,
    // no-revision branch, Api::new(), full HF_PULL_REPO_KINDS loop.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("GET", mockito::Matcher::Any)
        .with_status(404)
        .create();

    let err = resolve_hf_vindex("hf://owner/repo").expect_err("404 must error");
    assert!(
        err.to_string().contains("failed to download index.json"),
        "got: {err}"
    );
}

#[test]
#[serial]
fn resolve_hf_vindex_errors_with_revision_pinned() {
    // Same as above but with `@v2.0` revision. The split path takes
    // a different `repo` constructor (with_revision) — verify the
    // revision-bearing branch with the same all-404 mock.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock(
            "GET",
            mockito::Matcher::Regex(r"/resolve/v2\.0/index\.json".into()),
        )
        .with_status(404)
        .create();

    let err = resolve_hf_vindex("hf://owner/repo@v2.0").expect_err("404 must error");
    assert!(
        err.to_string().contains("owner/repo"),
        "error must mention repo: {err}"
    );
}

#[test]
#[serial]
fn download_hf_weights_errors_when_no_repo_kind_has_index_json() {
    // `download_hf_weights` now uses index.json as the "does this repo
    // type exist?" probe. When the Model probe 404s on index.json
    // (and there's no longer a dataset fallback), the function
    // returns the "failed to fetch index.json" error rather than
    // silently succeeding.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("GET", mockito::Matcher::Any)
        .with_status(404)
        .create();

    let err = download_hf_weights("hf://owner/repo").expect_err("no index.json on either side");
    assert!(
        err.to_string().contains("failed to fetch index.json"),
        "got: {err}"
    );
}

#[test]
#[serial]
fn resolve_hf_model_with_progress_errors_when_info_fails() {
    // The model-side variant calls `repo.info()` first (which hits
    // /api/models/{repo}/revision/{rev}). A 500 there propagates as
    // `HF info failed for {hf_path}`.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock(
            "GET",
            mockito::Matcher::Regex(r"/api/models/owner/repo.*".into()),
        )
        .with_status(500)
        .with_body(r#"{"error": "boom"}"#)
        .create();

    let err = resolve_hf_model_with_progress("hf://owner/repo", |_| NoOpProgress)
        .expect_err("info failure must surface");
    assert!(
        err.to_string().contains("HF info failed"),
        "expected 'HF info failed' wrapper, got: {err}"
    );
}

#[test]
#[serial]
fn resolve_hf_vindex_with_progress_errors_when_index_json_404s() {
    // The progress variant fetches index.json first; when it's
    // missing the `ok_or_else` clause produces a clear error.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("GET", mockito::Matcher::Any)
        .with_status(404)
        .create();

    let err = resolve_hf_vindex_with_progress("hf://owner/repo", |_| NoOpProgress)
        .expect_err("404 on index.json must error");
    assert!(err.to_string().contains("failed to fetch index.json"));
}

// ── head_etag_and_size: header-parsing and dispatch ──────────────────
//
// The HEAD probe is the etag-pinning step that drives cache hits in
// `cached_snapshot_file`. Mockito returns specific header
// combinations — git-tracked file with `ETag`, LFS-redirected file
// with `X-Linked-Etag` + `X-Linked-Size`, missing-headers fail-soft —
// and we confirm the parser picks the right values per case.

#[test]
#[serial]
fn head_etag_and_size_prefers_x_linked_headers_on_redirect() {
    // LFS path: HF returns 302 + `X-Linked-Etag` (SHA256 oid) +
    // `X-Linked-Size`. The parser must prefer those over the plain
    // `ETag`/`Content-Length` (which would be S3's MD5 hash post-302).
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(302)
        .with_header("X-Linked-Etag", "\"linked-oid-abc\"")
        .with_header("X-Linked-Size", "1234")
        .with_header("ETag", "\"plain-md5\"")
        .with_header("Content-Length", "9999")
        .create();

    let result = head_etag_and_size(RepoKind::Dataset, "owner/repo", None, "blobs.bin").unwrap();
    assert_eq!(result, ("linked-oid-abc".to_string(), 1234));
}

#[test]
#[serial]
fn head_etag_and_size_falls_back_to_plain_etag_on_2xx() {
    // Git-tracked small files don't redirect — they just 200 with a
    // plain `ETag` (git blob SHA1) + `Content-Length`. Parser uses
    // those when the X-Linked-* headers are absent.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(200)
        .with_header("ETag", "W/\"git-blob-sha1\"")
        .with_header("Content-Length", "42")
        .create();

    let result = head_etag_and_size(RepoKind::Dataset, "owner/repo", None, "index.json").unwrap();
    // Weak-prefix `W/` is stripped by `strip_etag_quoting`.
    assert_eq!(result.0, "git-blob-sha1");
    assert_eq!(result.1, 42);
}

#[test]
#[serial]
fn head_etag_and_size_returns_none_on_4xx() {
    // 4xx (not redirection, not success) → None.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(404)
        .create();

    let result = head_etag_and_size(RepoKind::Dataset, "owner/repo", None, "missing.bin");
    assert!(result.is_none());
}

#[test]
#[serial]
fn head_etag_and_size_returns_none_when_etag_missing() {
    // 200 OK but no ETag/X-Linked-Etag → parser bails (cache cannot
    // be pinned without a content identifier).
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(200)
        .with_header("Content-Length", "100")
        .create();

    let result = head_etag_and_size(RepoKind::Dataset, "owner/repo", None, "f");
    assert!(result.is_none());
}

#[test]
#[serial]
fn head_etag_and_size_uses_revision_in_url() {
    // `revision = Some("v2")` puts `/resolve/v2/` in the URL instead
    // of `/resolve/main/`. Pin via a regex that requires `v2`.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock(
            "HEAD",
            mockito::Matcher::Regex(r"/resolve/v2/file\.bin$".into()),
        )
        .with_status(200)
        .with_header("ETag", "\"v2-etag\"")
        .with_header("Content-Length", "7")
        .create();

    let result =
        head_etag_and_size(RepoKind::Dataset, "owner/repo", Some("v2"), "file.bin").unwrap();
    assert_eq!(result.0, "v2-etag");
}

// ── cached_snapshot_file: cache directory traversal ──────────────────

/// Build an hf-hub-shaped cache layout under `hub_root`:
///   models--owner--name/
///     blobs/<etag>            ← `bytes`
///     snapshots/main/file.bin → blobs/<etag>  (we just write a
///                                              regular file, not
///                                              a symlink, since
///                                              the lookup walks
///                                              `entries.path()`
///                                              and tests
///                                              file presence
///                                              not symlink-ness)
fn make_hub_blob(
    hub_root: &std::path::Path,
    kind_prefix: &str,
    repo_id: &str,
    etag: &str,
    bytes: &[u8],
    snapshot_revision: Option<&str>,
    filename: &str,
) {
    let safe = repo_id.replace('/', "--");
    let repo_dir = hub_root.join(format!("{kind_prefix}{safe}"));
    let blobs = repo_dir.join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::write(blobs.join(etag), bytes).unwrap();
    if let Some(rev) = snapshot_revision {
        let snap = repo_dir.join("snapshots").join(rev);
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join(filename), bytes).unwrap();
    }
}

#[test]
#[serial]
fn cached_snapshot_file_returns_snapshot_path_when_present() {
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(200)
        .with_header("ETag", "\"abc123\"")
        .with_header("Content-Length", "5")
        .create();

    // Build a cache dir at $HF_HOME/hub matching what the function
    // expects. HfTestEnv set HF_HOME to a tempdir; reuse it.
    let hub_root: PathBuf = std::env::var("HF_HOME")
        .map(|p| PathBuf::from(p).join("hub"))
        .unwrap();
    std::fs::create_dir_all(&hub_root).unwrap();
    let bytes = b"hello";
    make_hub_blob(
        &hub_root,
        "datasets--",
        "owner/repo",
        "abc123",
        bytes,
        Some("main"),
        "file.bin",
    );

    let (path, size) =
        cached_snapshot_file(RepoKind::Dataset, "owner/repo", Some("main"), "file.bin").unwrap();
    assert_eq!(size, 5);
    assert!(path.ends_with("file.bin"));
}

#[test]
#[serial]
fn cached_snapshot_file_misses_when_the_pinned_revisions_snapshot_has_no_link() {
    // The blob's bytes are present (deduped from some other repo or
    // revision), but the PINNED revision's own snapshot dir has no
    // symlink for this filename yet. Must miss — not fall back to the
    // raw blob path or to some other revision's copy — so the caller
    // falls through to `download_with_progress`, which actually
    // creates this exact revision's symlink. Regression coverage for
    // the real bug this rewrite fixed: a `granite-4.1-3b` registry
    // pull reported `target.embedding.bin` as "cached" via the
    // now-removed blob-path fallback, no symlink was ever created
    // under the pinned revision's snapshot dir, and `larql serve`
    // failed opening the container with a bare, unhelpful IO error.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(200)
        .with_header("ETag", "\"deadbeef\"")
        .with_header("Content-Length", "4")
        .create();

    let hub_root: PathBuf = std::env::var("HF_HOME")
        .map(|p| PathBuf::from(p).join("hub"))
        .unwrap();
    std::fs::create_dir_all(&hub_root).unwrap();
    make_hub_blob(
        &hub_root,
        "datasets--",
        "owner/repo",
        "deadbeef",
        b"abcd",
        None, // no snapshot dir for any revision
        "f.bin",
    );

    let result = cached_snapshot_file(RepoKind::Dataset, "owner/repo", Some("pinned"), "f.bin");
    assert!(
        result.is_none(),
        "a bare blob with no symlink under the pinned revision must miss, not silently succeed"
    );
}

#[test]
fn cached_snapshot_file_misses_immediately_on_an_unpinned_revision() {
    // `revision: None` can name no single snapshot directory
    // unambiguously — always miss, unconditionally, before even
    // issuing the HEAD request. No mock server is set up at all: a
    // network call here would itself be the bug.
    let result = cached_snapshot_file(RepoKind::Dataset, "owner/repo", None, "f.bin");
    assert!(result.is_none());
}

#[test]
#[serial]
fn cached_snapshot_file_returns_none_on_size_mismatch() {
    // The HEAD reports size=10 but the on-disk blob is 4 bytes — the
    // defensive size check rejects the cache hit and returns None.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(200)
        .with_header("ETag", "\"sizemismatch\"")
        .with_header("Content-Length", "10")
        .create();

    let hub_root: PathBuf = std::env::var("HF_HOME")
        .map(|p| PathBuf::from(p).join("hub"))
        .unwrap();
    std::fs::create_dir_all(&hub_root).unwrap();
    make_hub_blob(
        &hub_root,
        "datasets--",
        "owner/repo",
        "sizemismatch",
        b"only4", // 5 bytes (still ≠ 10)
        Some("main"),
        "f.bin",
    );

    let result = cached_snapshot_file(RepoKind::Dataset, "owner/repo", Some("main"), "f.bin");
    assert!(result.is_none(), "size mismatch must abort cache hit");
}

#[test]
#[serial]
fn cached_snapshot_file_returns_none_when_blob_missing() {
    // HEAD returns valid headers but the blob doesn't exist on disk.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(200)
        .with_header("ETag", "\"never-cached\"")
        .with_header("Content-Length", "1")
        .create();

    // No blob written — straight cache miss.
    let result = cached_snapshot_file(RepoKind::Dataset, "owner/repo", Some("main"), "f.bin");
    assert!(result.is_none());
}

#[test]
#[serial]
fn cached_snapshot_file_works_for_model_prefix() {
    // Exercise the `models--` cache prefix path — existing tests
    // all use `datasets--`. Same logic, different prefix.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(200)
        .with_header("ETag", "\"model-etag\"")
        .with_header("Content-Length", "4")
        .create();

    let hub_root: PathBuf = std::env::var("HF_HOME")
        .map(|p| PathBuf::from(p).join("hub"))
        .unwrap();
    std::fs::create_dir_all(&hub_root).unwrap();
    make_hub_blob(
        &hub_root,
        "models--",
        "owner/repo",
        "model-etag",
        b"abcd",
        Some("main"),
        "config.json",
    );

    let (path, size) =
        cached_snapshot_file(RepoKind::Model, "owner/repo", Some("main"), "config.json").unwrap();
    assert_eq!(size, 4);
    assert!(path.ends_with("config.json"));
}

#[test]
#[serial]
fn cached_snapshot_file_ignores_an_unrelated_revisions_snapshot_dir() {
    // A DIFFERENT revision's snapshot dir has a file present (`noise`,
    // for a different filename even) — the pinned-revision-only
    // contract must not be satisfied by it. Only the pinned revision's
    // own `snapshots/v7/target.bin` symlink counts.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(200)
        .with_header("ETag", "\"rev-fallback\"")
        .with_header("Content-Length", "3")
        .create();

    let hub_root: PathBuf = std::env::var("HF_HOME")
        .map(|p| PathBuf::from(p).join("hub"))
        .unwrap();
    std::fs::create_dir_all(&hub_root).unwrap();
    let repo_dir = hub_root.join("datasets--owner--repo");
    let blobs = repo_dir.join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::write(blobs.join("rev-fallback"), b"abc").unwrap();
    // A different revision's snapshot dir, irrelevant to the request.
    let noise_snap = repo_dir.join("snapshots").join("noise");
    std::fs::create_dir_all(&noise_snap).unwrap();
    std::fs::write(noise_snap.join("other.bin"), b"abc").unwrap();
    // The pinned revision's own snapshot dir, with the file.
    let pinned_snap = repo_dir.join("snapshots").join("v7");
    std::fs::create_dir_all(&pinned_snap).unwrap();
    std::fs::write(pinned_snap.join("target.bin"), b"abc").unwrap();

    let (path, _) =
        cached_snapshot_file(RepoKind::Dataset, "owner/repo", Some("v7"), "target.bin").unwrap();
    assert!(path.to_string_lossy().contains("v7"));
}

#[test]
#[serial]
fn cached_snapshot_file_returns_none_when_blob_is_directory_not_file() {
    // Exercise the `!meta.is_file()` defensive branch — the blob
    // path resolves to a directory entry instead of a file.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(200)
        .with_header("ETag", "\"dir-as-blob\"")
        .with_header("Content-Length", "5")
        .create();

    let hub_root: PathBuf = std::env::var("HF_HOME")
        .map(|p| PathBuf::from(p).join("hub"))
        .unwrap();
    let blobs = hub_root.join("datasets--owner--repo").join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    // Create the blob path as a DIRECTORY, not a regular file.
    std::fs::create_dir_all(blobs.join("dir-as-blob")).unwrap();

    let result = cached_snapshot_file(RepoKind::Dataset, "owner/repo", Some("main"), "f.bin");
    assert!(result.is_none(), "blob-is-directory must miss");
}

// ── RepoKind variant tag direct tests ────────────────────────────────
//
// Production code only constructs RepoKind::Model (HF_PULL_REPO_KINDS
// dropped the dataset fallback). The Dataset variant is still
// referenced by the helper-level tests and remains in the enum for
// explicit callers. Cover the match arms directly.

#[test]
fn to_hub_type_maps_each_kind_to_hf_hub_repo_type() {
    // Dataset and Model variants both have their own match arm in
    // to_hub_type — Production only hits Model; this test pins
    // both branches.
    match RepoKind::Dataset.to_hub_type() {
        hf_hub::RepoType::Dataset => {}
        other => panic!("Dataset must map to RepoType::Dataset, got {other:?}"),
    }
    match RepoKind::Model.to_hub_type() {
        hf_hub::RepoType::Model => {}
        other => panic!("Model must map to RepoType::Model, got {other:?}"),
    }
}

#[test]
fn url_segment_matches_repo_kind_prefix() {
    assert_eq!(RepoKind::Dataset.url_segment(), "datasets/");
    assert_eq!(RepoKind::Model.url_segment(), "");
}

#[test]
fn cache_prefix_matches_repo_kind() {
    assert_eq!(RepoKind::Dataset.cache_prefix(), "datasets--");
    assert_eq!(RepoKind::Model.cache_prefix(), "models--");
}

#[test]
#[serial]
fn resolve_hf_vindex_success_via_broad_mocks() {
    // Happy path: index.json downloads, returns the snapshot dir.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let body = br#"{"version":2,"model":"owner/repo","family":"x"}"#;
    let idx = mock_hf_file_resolve(&mut server, r"index\.json", "idx", body);

    let dir = resolve_hf_vindex("hf://owner/repo").expect("success path");
    assert!(dir.exists(), "vindex dir must exist on disk");
    assert!(idx[0].matched(), "index.json mock must have been hit");
}

#[test]
#[serial]
fn download_hf_weights_success_via_broad_mocks() {
    // index.json fetches successfully, so the function enters the
    // weight-file loop and returns Ok. No fallback mocks here —
    // mockito's default response for unmatched requests is
    // sufficient; adding GET-Any fallback mocks intercepts our
    // specific index.json mock in mockito's matching order.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let body = br#"{"version":2}"#;
    let idx = mock_hf_file_resolve(&mut server, r"index\.json", "wt", body);

    download_hf_weights("hf://owner/repo").expect("success path");
    assert!(idx[0].matched(), "index.json mock must have been hit");
}

#[test]
#[serial]
fn resolve_hf_vindex_with_progress_success_via_broad_mocks() {
    // Exercise the with_progress success path. Same as
    // resolve_hf_vindex but routes through the cache-probe closure.
    // No fallback mocks — see comment on
    // `download_hf_weights_success_via_broad_mocks`.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let body = br#"{"version":2}"#;
    let idx = mock_hf_file_resolve(&mut server, r"index\.json", "wp", body);

    let dir =
        resolve_hf_vindex_with_progress("hf://owner/repo", |_| NoOpProgress).expect("success path");
    assert!(dir.exists());
    assert!(idx[0].matched(), "index.json mock must have been hit");
}

#[test]
#[serial]
fn a_v2_pull_also_fetches_every_file_its_index_requires() {
    // `norms.bin` is on no fixed download list; an inference-level index
    // that declares it requires it, so `larql list` will look for it — and
    // the pull must therefore fetch it, or "re-run `larql pull`" would
    // never finish the entry.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let body = br#"{"version":2,"model":"m","family":"f","num_layers":1,"hidden_size":4,
"intermediate_size":8,"vocab_size":2,"embed_scale":1.0,"layers":[],"down_top_k":1,
"extract_level":"inference","checksums":{"norms.bin":"n"}}"#;
    let idx = mock_hf_file_resolve(&mut server, r"index\.json", "dv", body);
    let norms = mock_hf_file_resolve(&mut server, r"norms\.bin", "dn", b"norms");

    let dir =
        resolve_hf_vindex_with_progress("hf://owner/repo", |_| NoOpProgress).expect("success path");
    assert!(idx[0].matched());
    assert!(norms[0].matched(), "the declared file must be requested");
    assert!(dir.join(NORMS_BIN).is_file());
}

#[test]
#[serial]
fn resolve_hf_vindex_with_progress_uses_cache_when_blob_present() {
    // When the cached_snapshot_file fast-path finds the blob on
    // disk, the function bypasses download_with_progress and goes
    // through the cache-hit branch (progress.init/update/finish
    // called with the [cached] tag). Build the cache + a matching
    // HEAD response so the cache short-circuit fires. The fast path
    // only ever fires for a PINNED revision (see cached_snapshot_file's
    // module docs) — an unpinned `hf://owner/repo` always misses it and
    // goes through `download_with_progress` instead, so this request
    // pins `@main` explicitly to exercise the cache-hit branch at all.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let body = br#"{"version":2}"#;
    let _head = server
        .mock("HEAD", mockito::Matcher::Regex(r"index\.json".into()))
        .with_status(200)
        .with_header("ETag", "\"cached-idx\"")
        .with_header("Content-Length", &body.len().to_string())
        .expect_at_least(1)
        .create();
    // Build the on-disk cache layout the function expects.
    let hub_root: PathBuf = std::env::var("HF_HOME")
        .map(|p| PathBuf::from(p).join("hub"))
        .unwrap();
    std::fs::create_dir_all(&hub_root).unwrap();
    make_hub_blob(
        &hub_root,
        "models--",
        "owner/repo",
        "cached-idx",
        body,
        Some("main"),
        INDEX_JSON,
    );
    // Other files (the rest of the metadata loop) return 404.
    let _fallback_get = server
        .mock("GET", mockito::Matcher::Any)
        .with_status(404)
        .create();
    let _fallback_head = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(404)
        .create();

    let dir = resolve_hf_vindex_with_progress("hf://owner/repo@main", |_| NoOpProgress)
        .expect("cache hit path must return Ok");
    assert!(dir.ends_with("main"), "expected snapshot dir under main");
}

#[test]
#[serial]
fn resolve_hf_model_with_progress_errors_when_info_returns_empty_siblings() {
    // Cover the `wanted.is_empty()` error branch — info() succeeds
    // but lists no files. hf-hub's info() endpoint is
    // /api/models/{repo}/revision/{rev} or similar; mock a 200
    // response anywhere on /api/models/... so the call lands but
    // returns no siblings.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _info = server
        .mock(
            "GET",
            mockito::Matcher::Regex(r"/api/models/owner/repo".into()),
        )
        .with_status(200)
        .with_header("Content-Type", "application/json")
        .with_body(r#"{"siblings":[],"sha":"abc"}"#)
        .create();

    let result = resolve_hf_model_with_progress("hf://owner/repo", |_| NoOpProgress);
    match result {
        Err(e) => {
            // Either the empty-siblings error fires or info parsing
            // fails first — both exercise the function's plumbing.
            let s = e.to_string();
            assert!(
                s.contains("no usable model files") || s.contains("HF info failed"),
                "expected siblings/info error, got: {s}"
            );
        }
        Ok(_) => panic!("must error on empty siblings or info failure"),
    }
}

#[test]
#[serial]
fn cached_snapshot_file_with_revision_falls_back_to_pinned_dir() {
    // Snapshot tree exists but doesn't have a directory matching the
    // requested revision under the iter — exercises the explicit
    // `snapshots.join(rev)` fallback path.
    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let _m = server
        .mock("HEAD", mockito::Matcher::Any)
        .with_status(200)
        .with_header("ETag", "\"rev-blob\"")
        .with_header("Content-Length", "3")
        .create();

    let hub_root: PathBuf = std::env::var("HF_HOME")
        .map(|p| PathBuf::from(p).join("hub"))
        .unwrap();
    std::fs::create_dir_all(&hub_root).unwrap();
    // Write blob + snapshot at the pinned revision.
    make_hub_blob(
        &hub_root,
        "datasets--",
        "owner/repo",
        "rev-blob",
        b"abc",
        Some("v3"),
        "f.bin",
    );

    let (path, size) =
        cached_snapshot_file(RepoKind::Dataset, "owner/repo", Some("v3"), "f.bin").unwrap();
    assert_eq!(size, 3);
    assert!(path.ends_with("f.bin"));
}
