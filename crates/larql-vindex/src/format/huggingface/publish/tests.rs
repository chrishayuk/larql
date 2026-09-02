//! Tests for [`super`].
//!
//! Split out of `mod.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

use super::*;
use serial_test::serial;
use std::fs;

/// Clear the test base-URL override so URL-builder tests see the
/// production default. Saved/restored around the assertion to
/// avoid leaking the change to other tests.
fn with_default_base<F: FnOnce()>(f: F) {
    let prev = std::env::var(protocol::TEST_BASE_ENV).ok();
    std::env::remove_var(protocol::TEST_BASE_ENV);
    f();
    if let Some(v) = prev {
        std::env::set_var(protocol::TEST_BASE_ENV, v);
    }
}

// ─── URL builders ──────────────────────────────────────────────

#[test]
#[serial]
fn hf_repo_url_model() {
    with_default_base(|| {
        assert_eq!(
            hf_repo_url("model", "org/repo"),
            "https://huggingface.co/org/repo"
        );
    });
}

#[test]
#[serial]
fn hf_repo_url_dataset() {
    with_default_base(|| {
        assert_eq!(
            hf_repo_url("dataset", "org/repo"),
            "https://huggingface.co/datasets/org/repo"
        );
    });
}

#[test]
#[serial]
fn hf_repo_url_unknown_type_falls_back_to_model() {
    // Unknown repo types should fall back to the model URL shape so a
    // typo doesn't silently route to a "datasets" 404.
    with_default_base(|| {
        assert_eq!(
            hf_repo_url("space", "org/repo"),
            "https://huggingface.co/org/repo"
        );
    });
}

#[test]
#[serial]
fn hf_api_url_model() {
    with_default_base(|| {
        assert_eq!(
            hf_api_url("model", "org/repo", "preupload/main"),
            "https://huggingface.co/api/models/org/repo/preupload/main"
        );
    });
}

#[test]
#[serial]
fn hf_api_url_dataset() {
    with_default_base(|| {
        assert_eq!(
            hf_api_url("dataset", "org/repo", "tree/main"),
            "https://huggingface.co/api/datasets/org/repo/tree/main"
        );
    });
}

// ─── PublishOptions ────────────────────────────────────────────

#[test]
fn publish_options_default_is_model_no_skip() {
    let opts = PublishOptions::default();
    assert!(!opts.skip_unchanged);
    assert_eq!(opts.repo_type, "model");
}

#[test]
fn publish_options_skip_unchanged_helper() {
    let opts = PublishOptions::skip_unchanged();
    assert!(opts.skip_unchanged);
    // Should keep the default repo_type — the helper only flips skip.
    assert_eq!(opts.repo_type, "model");
}

// ─── enumerate_publishable_files ───────────────────────────────

#[test]
fn enumerate_files_root_and_subdir() {
    let dir = tempfile::tempdir().unwrap();
    // Root files.
    fs::write(dir.path().join("index.json"), "{}").unwrap();
    fs::write(dir.path().join("gate_vectors.bin"), b"x").unwrap();
    // Subdirectory.
    fs::create_dir_all(dir.path().join("layers")).unwrap();
    fs::write(dir.path().join("layers/layer_00.weights"), b"y").unwrap();
    fs::write(dir.path().join("layers/layer_01.weights"), b"z").unwrap();

    let files = enumerate_publishable_files(dir.path()).unwrap();
    let names: Vec<&str> = files.iter().map(|(_, n)| n.as_str()).collect();

    // Sorted by repo path so the commit order is stable.
    assert_eq!(
        names,
        vec![
            "gate_vectors.bin",
            "index.json",
            "layers/layer_00.weights",
            "layers/layer_01.weights",
        ]
    );
    // Subdir paths use forward slashes regardless of platform.
    assert!(files.iter().any(|(_, n)| n.contains('/')));
}

#[test]
fn enumerate_files_skips_nested_subdirs() {
    // Only immediate subdirectories are walked. Files in `a/b/foo`
    // must not appear — HF dataset layouts are at most two levels.
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join("a/b")).unwrap();
    fs::write(dir.path().join("a/b/foo.bin"), b"deep").unwrap();
    fs::write(dir.path().join("a/top.bin"), b"shallow").unwrap();

    let files = enumerate_publishable_files(dir.path()).unwrap();
    let names: Vec<&str> = files.iter().map(|(_, n)| n.as_str()).collect();

    assert_eq!(names, vec!["a/top.bin"]);
}

#[test]
fn enumerate_files_empty_dir_returns_empty_vec() {
    let dir = tempfile::tempdir().unwrap();
    let files = enumerate_publishable_files(dir.path()).unwrap();
    assert!(files.is_empty());
}

// ─── publish_vindex_with_opts validation ───────────────────────

#[test]
fn publish_rejects_non_directory() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("not-a-dir.txt");
    fs::write(&file_path, "x").unwrap();

    let mut cb = SilentPublishCallbacks;
    let err = publish_vindex_with_opts(&file_path, "org/repo", &PublishOptions::default(), &mut cb)
        .expect_err("path is a file, not a directory");
    assert!(matches!(err, VindexError::NotADirectory(_)));
}

#[test]
fn publish_rejects_directory_without_index_json() {
    let dir = tempfile::tempdir().unwrap();
    // Directory exists but has no index.json — should fail before any
    // network call (no token required to reach this branch).
    let mut cb = SilentPublishCallbacks;
    let err = publish_vindex_with_opts(dir.path(), "org/repo", &PublishOptions::default(), &mut cb)
        .expect_err("missing index.json must error");
    match err {
        VindexError::Parse(msg) => assert!(
            msg.contains("not a vindex directory"),
            "unexpected error message: {msg}",
        ),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

// ─── get_hf_token ──────────────────────────────────────────────

#[test]
#[serial]
fn get_hf_token_reads_env_var() {
    // Process-wide env mutation must be serialised against the other
    // `get_hf_token_*` tests below — they share `HF_TOKEN` and `HOME`
    // and Windows scheduling has surfaced the race that Linux/macOS
    // happened to mask (this test sets HF_TOKEN to a sentinel and
    // `errors_when_no_source_present` then read the sentinel instead
    // of erroring on a missing token).
    let prev = std::env::var("HF_TOKEN").ok();
    std::env::set_var("HF_TOKEN", "sentinel-token-XYZ");
    let result = get_hf_token();
    // Restore before asserting so a panic doesn't leak the override.
    match prev {
        Some(v) => std::env::set_var("HF_TOKEN", v),
        None => std::env::remove_var("HF_TOKEN"),
    }
    assert_eq!(result.unwrap(), "sentinel-token-XYZ");
}

/// RAII guard for HF_TOKEN + HOME env vars, restored on drop.
struct HfTokenEnvGuard {
    prev_token: Option<String>,
    prev_hub_token: Option<String>,
    prev_home: Option<String>,
    _tmp: tempfile::TempDir,
}
impl HfTokenEnvGuard {
    /// Clear HF_TOKEN, point HOME at a tempdir so the
    /// file-based token lookup doesn't leak into the real
    /// user home. Returns the tempdir handle for callers to
    /// populate.
    fn new() -> Self {
        let prev_token = std::env::var("HF_TOKEN").ok();
        let prev_hub_token = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
        let prev_home = std::env::var("HOME").ok();
        let tmp = tempfile::tempdir().unwrap();
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
        std::env::set_var("HOME", tmp.path());
        Self {
            prev_token,
            prev_hub_token,
            prev_home,
            _tmp: tmp,
        }
    }
    fn home(&self) -> &std::path::Path {
        self._tmp.path()
    }
}
impl Drop for HfTokenEnvGuard {
    fn drop(&mut self) {
        match self.prev_token.take() {
            Some(v) => std::env::set_var("HF_TOKEN", v),
            None => std::env::remove_var("HF_TOKEN"),
        }
        match self.prev_hub_token.take() {
            Some(v) => std::env::set_var("HUGGING_FACE_HUB_TOKEN", v),
            None => std::env::remove_var("HUGGING_FACE_HUB_TOKEN"),
        }
        match self.prev_home.take() {
            Some(v) => std::env::set_var("HOME", v),
            None => std::env::remove_var("HOME"),
        }
    }
}

#[test]
#[serial]
fn get_hf_token_reads_legacy_huggingface_token_file() {
    let g = HfTokenEnvGuard::new();
    let token_path = g.home().join(".huggingface").join("token");
    fs::create_dir_all(token_path.parent().unwrap()).unwrap();
    fs::write(&token_path, "legacy-token-123\n").unwrap();

    let token = get_hf_token().expect("legacy token file must be read");
    // Trailing whitespace is stripped.
    assert_eq!(token, "legacy-token-123");
}

#[test]
#[serial]
fn get_hf_token_reads_cache_huggingface_token_file() {
    let g = HfTokenEnvGuard::new();
    let token_path = g.home().join(".cache").join("huggingface").join("token");
    fs::create_dir_all(token_path.parent().unwrap()).unwrap();
    fs::write(&token_path, "cache-token-456").unwrap();

    let token = get_hf_token().expect("cache token file must be read");
    assert_eq!(token, "cache-token-456");
}

#[test]
#[serial]
fn get_hf_token_errors_when_no_source_present() {
    let _g = HfTokenEnvGuard::new();
    let err = get_hf_token().expect_err("no env + no file → error");
    match err {
        VindexError::Parse(msg) => {
            assert!(msg.contains("HuggingFace token not found"), "got: {msg}")
        }
        other => panic!("expected Parse error, got {other:?}"),
    }
}

// ─── PublishCallbacks default impls ────────────────────────────

#[test]
fn silent_callbacks_default_methods_are_noop() {
    // Cover the default no-op impls of every PublishCallbacks
    // method via the SilentPublishCallbacks impl-by-default path.
    let mut cb = SilentPublishCallbacks;
    cb.on_start("org/repo");
    cb.on_file_start("a.bin", 100);
    cb.on_file_progress("a.bin", 50, 100);
    cb.on_file_done("a.bin");
    cb.on_file_skipped("a.bin", 100, "sha256-deadbeef");
    cb.on_complete("https://huggingface.co/org/repo");
}

// ─── publish_vindex thin wrapper ───────────────────────────────

#[test]
fn publish_vindex_wrapper_dispatches_to_with_opts() {
    // Pin that the no-options helper delegates: a missing
    // directory must surface the same NotADirectory error the
    // with_opts variant returns. Hits the wrapper body without
    // needing a working HF endpoint.
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("does-not-exist");
    let mut cb = SilentPublishCallbacks;
    let err = publish_vindex(&missing, "org/repo", &mut cb).expect_err("missing path must error");
    assert!(matches!(err, VindexError::NotADirectory(_)));
}

// ─── publish_vindex_with_opts happy path with mockito ──────────

/// Build a single-file vindex on disk with the smallest possible
/// `index.json` so the publish flow has something to enumerate.
/// Returns the directory path; caller is responsible for keeping
/// the tempdir alive.
fn make_minimal_vindex(dir: &std::path::Path) {
    fs::write(dir.join("index.json"), r#"{"version":2}"#).unwrap();
}

/// Mock the pinned-revision fetch every successful
/// `publish_vindex_with_opts` call now makes once uploads finish
/// (`GET /api/{models,datasets}/{repo_id}` → `{"sha": ...}`).
/// Returns the `Mock` so callers can assert it fired.
fn mock_repo_head_sha(
    server: &mut mockito::ServerGuard,
    repo_id: &str,
    sha: &str,
) -> mockito::Mock {
    server
        .mock("GET", format!("/api/models/{repo_id}").as_str())
        .with_status(200)
        .with_body(format!(r#"{{"sha":"{sha}"}}"#))
        .expect_at_least(1)
        .create()
}

/// RAII guard for an env-var: sets to `value`, restores on drop.
struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}
impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, prev }
    }
}
impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

#[test]
#[serial]
fn publish_vindex_with_opts_happy_path_invokes_callbacks() {
    // End-to-end mock: HF_TOKEN set, HF base pointed at mockito,
    // create_hf_repo returns 200, preupload returns inline mode,
    // commit returns 200. Verifies the function progresses through
    // every callback (on_start → on_file_start → on_file_done →
    // on_complete) and returns the canonical web URL.
    let mut server = mockito::Server::new();
    let _base = EnvGuard::set(protocol::TEST_BASE_ENV, &server.url());
    let _tok = EnvGuard::set("HF_TOKEN", "tok");

    // POST /api/repos/create → 200 (repo creation succeeds).
    let _create = server
        .mock("POST", "/api/repos/create")
        .with_status(200)
        .with_body("{}")
        .expect_at_least(1)
        .create();
    // POST /api/models/org/repo/preupload/main → returns "regular"
    // mode for the single small file. The upload handler then
    // inlines the bytes in the commit body.
    let _preupload = server
        .mock("POST", "/api/models/org/repo/preupload/main")
        .with_status(200)
        .with_body(r#"{"files":[{"path":"index.json","uploadMode":"regular"}]}"#)
        .expect_at_least(1)
        .create();
    // POST /api/models/org/repo/commit/main → 200 (commit accepted).
    let _commit = server
        .mock("POST", "/api/models/org/repo/commit/main")
        .with_status(200)
        .with_body("{}")
        .expect_at_least(1)
        .create();
    let _head = mock_repo_head_sha(&mut server, "org/repo", "deadbeef");

    // Build the vindex dir.
    let tmp = tempfile::tempdir().unwrap();
    make_minimal_vindex(tmp.path());

    // Capture callback firings.
    #[derive(Default)]
    struct Recorder {
        started: bool,
        files_started: Vec<String>,
        files_done: Vec<String>,
        completed_url: Option<String>,
    }
    impl PublishCallbacks for Recorder {
        fn on_start(&mut self, _repo: &str) {
            self.started = true;
        }
        fn on_file_start(&mut self, f: &str, _size: u64) {
            self.files_started.push(f.into());
        }
        fn on_file_done(&mut self, f: &str) {
            self.files_done.push(f.into());
        }
        fn on_complete(&mut self, url: &str) {
            self.completed_url = Some(url.into());
        }
    }
    let mut rec = Recorder::default();

    let result =
        publish_vindex_with_opts(tmp.path(), "org/repo", &PublishOptions::default(), &mut rec)
            .expect("happy path must return Ok");

    assert!(rec.started, "on_start must fire");
    assert_eq!(rec.files_started, vec!["index.json".to_string()]);
    assert_eq!(rec.files_done, vec!["index.json".to_string()]);
    assert_eq!(rec.completed_url.as_deref(), Some(result.url.as_str()));
    assert!(
        result.url.ends_with("/org/repo"),
        "model repo URL should end with /org/repo, got: {}",
        result.url
    );
    assert_eq!(result.revision, "deadbeef");
}

#[test]
#[serial]
fn publish_then_candidate_end_to_end_against_a_mocked_hf_endpoint() {
    // R3C's own acceptance gate, run for real rather than asserted
    // piecewise: publish against a mocked HF endpoint, let it return a
    // pinned commit, build a registry candidate from THAT exact
    // publish result (never a hand-typed stand-in revision), and prove
    // the candidate carries the real values and validates through the
    // real registry schema — the same `RegistryManifest::validate()`
    // `production_registry()` trusts, called inside `build_candidate`
    // itself.
    let mut server = mockito::Server::new();
    let _base = EnvGuard::set(protocol::TEST_BASE_ENV, &server.url());
    let _tok = EnvGuard::set("HF_TOKEN", "tok");

    let _create = server
        .mock("POST", "/api/repos/create")
        .with_status(200)
        .with_body("{}")
        .expect_at_least(1)
        .create();
    let _preupload = server
        .mock("POST", "/api/models/larql/granite-4.1-3b/preupload/main")
        .with_status(200)
        .with_body(r#"{"files":[{"path":"index.json","uploadMode":"regular"}]}"#)
        .expect_at_least(1)
        .create();
    let _commit = server
        .mock("POST", "/api/models/larql/granite-4.1-3b/commit/main")
        .with_status(200)
        .with_body("{}")
        .expect_at_least(1)
        .create();
    let _head = mock_repo_head_sha(
        &mut server,
        "larql/granite-4.1-3b",
        "1048a8eb2fec5812a698e76d7e603527d0475c17",
    );

    let tmp = tempfile::tempdir().unwrap();
    make_minimal_vindex(tmp.path());

    let published = publish_vindex_with_opts(
        tmp.path(),
        "larql/granite-4.1-3b",
        &PublishOptions::default(),
        &mut SilentPublishCallbacks,
    )
    .expect("mocked publish must succeed");
    assert_eq!(
        published.revision,
        "1048a8eb2fec5812a698e76d7e603527d0475c17"
    );

    let model = crate::registry::build_candidate(crate::registry::CandidateInputs {
        name: "granite-4.1-3b".to_string(),
        variant: "bf16".to_string(),
        // The published artifact's own repo/revision — "larql/granite-4.1-3b"
        // is the same string this test passed to `publish_vindex_with_opts`
        // above, `published.revision` is what the MOCKED HF endpoint
        // actually returned, not re-typed.
        artifact_repo: "larql/granite-4.1-3b".to_string(),
        artifact_revision: published.revision.clone(),
        abi: None,
        source_repo: "ibm-granite/granite-4.1-3b".to_string(),
        source_revision: "c0650403e44e78ec0262dab1c90914c65b196c4e".to_string(),
        attested_by: "chrishayuk".to_string(),
    })
    .expect("a candidate built from real inputs must pass real registry validation");

    let variant = model.variants.get("bf16").expect("bf16 variant present");
    assert_eq!(variant.artifact.repo, "larql/granite-4.1-3b");
    assert_eq!(
        variant.artifact.revision, "1048a8eb2fec5812a698e76d7e603527d0475c17",
        "the candidate's artifact.revision must be the pinned commit the \
         MOCKED HF endpoint actually returned, not a guessed or hand-typed value"
    );
    assert_eq!(variant.source.repo, "ibm-granite/granite-4.1-3b");
    assert_eq!(
        variant.source.revision,
        "c0650403e44e78ec0262dab1c90914c65b196c4e"
    );
    assert_eq!(
        variant.source.attestation,
        crate::registry::Attestation::HandAttested {
            by: "chrishayuk".to_string()
        }
    );

    // Missing provenance inputs fail rather than being guessed —
    // R3C's own invariant, proven against the same real publish result
    // rather than synthetic data.
    let refused = crate::registry::build_candidate(crate::registry::CandidateInputs {
        name: "granite-4.1-3b".to_string(),
        variant: "bf16".to_string(),
        artifact_repo: "larql/granite-4.1-3b".to_string(),
        artifact_revision: published.revision,
        abi: None,
        source_repo: "ibm-granite/granite-4.1-3b".to_string(),
        source_revision: "c0650403e44e78ec0262dab1c90914c65b196c4e".to_string(),
        attested_by: String::new(),
    });
    assert!(
        refused.is_err(),
        "an empty attestation must refuse, never silently build a candidate"
    );
}

#[test]
#[serial]
fn publish_fails_when_the_pinned_revision_cannot_be_fetched() {
    // Every file uploaded successfully, but the final "what's main's
    // HEAD now" call fails — this must fail the whole publish, not
    // return a result missing (or defaulting) its revision. A
    // caller pinning `RegistryArtifactRef::revision` from this
    // result needs a real fact or a real refusal, never a floating
    // fallback like "main".
    let mut server = mockito::Server::new();
    let _base = EnvGuard::set(protocol::TEST_BASE_ENV, &server.url());
    let _tok = EnvGuard::set("HF_TOKEN", "tok");

    let tmp = tempfile::tempdir().unwrap();
    make_minimal_vindex(tmp.path());

    let _create = server
        .mock("POST", "/api/repos/create")
        .with_status(200)
        .with_body("{}")
        .expect_at_least(1)
        .create();
    let _preupload = server
        .mock("POST", "/api/models/org/repo/preupload/main")
        .with_status(200)
        .with_body(r#"{"files":[{"path":"index.json","uploadMode":"regular"}]}"#)
        .expect_at_least(1)
        .create();
    let _commit = server
        .mock("POST", "/api/models/org/repo/commit/main")
        .with_status(200)
        .with_body("{}")
        .expect_at_least(1)
        .create();
    // No mock for GET /api/models/org/repo — the pinned-revision
    // fetch gets mockito's default (non-2xx) response.

    let mut cb = SilentPublishCallbacks;
    let err = publish_vindex_with_opts(tmp.path(), "org/repo", &PublishOptions::default(), &mut cb)
        .expect_err("a failed revision fetch must fail the whole publish");
    assert!(err.to_string().contains("org/repo"), "{err}");
}

#[test]
#[serial]
fn publish_vindex_with_opts_skip_unchanged_skips_matching_lfs_oid() {
    // skip_unchanged path: the tree endpoint returns a remote oid
    // that matches the local SHA256 of our file → on_file_skipped
    // fires and the upload endpoints are NOT called.
    let mut server = mockito::Server::new();
    let _base = EnvGuard::set(protocol::TEST_BASE_ENV, &server.url());
    let _tok = EnvGuard::set("HF_TOKEN", "tok");

    // Build the vindex dir.
    let tmp = tempfile::tempdir().unwrap();
    make_minimal_vindex(tmp.path());
    let local_sha = crate::format::checksums::sha256_file(&tmp.path().join("index.json"))
        .expect("sha must compute");

    // create_hf_repo + tree endpoint that reports our exact local SHA.
    let _create = server
        .mock("POST", "/api/repos/create")
        .with_status(200)
        .with_body("{}")
        .expect_at_least(1)
        .create();
    let tree_body = serde_json::json!([
        {"type":"file","path":"index.json","lfs":{"oid":local_sha}}
    ])
    .to_string();
    let _tree = server
        .mock("GET", "/api/models/org/repo/tree/main?recursive=true")
        .with_status(200)
        .with_body(tree_body)
        .expect_at_least(1)
        .create();
    // If the function uploads instead of skipping, this mock fires
    // and the test detects the bug via the skipped-files check
    // below. Don't actually mock the upload endpoints — they
    // shouldn't be called.
    let _head = mock_repo_head_sha(&mut server, "org/repo", "deadbeef");

    #[derive(Default)]
    struct Recorder {
        skipped: Vec<String>,
        uploaded: Vec<String>,
    }
    impl PublishCallbacks for Recorder {
        fn on_file_skipped(&mut self, f: &str, _: u64, _: &str) {
            self.skipped.push(f.into());
        }
        fn on_file_start(&mut self, f: &str, _: u64) {
            self.uploaded.push(f.into());
        }
    }
    let mut rec = Recorder::default();

    publish_vindex_with_opts(
        tmp.path(),
        "org/repo",
        &PublishOptions::skip_unchanged(),
        &mut rec,
    )
    .expect("skip-unchanged happy path must return Ok");

    assert_eq!(rec.skipped, vec!["index.json".to_string()]);
    assert!(
        rec.uploaded.is_empty(),
        "file with matching remote SHA must not be uploaded: {:?}",
        rec.uploaded
    );
}

#[test]
#[serial]
fn publish_prunes_remote_files_absent_from_the_vindex() {
    // The defect this pins: publish only ever added files, so a renamed
    // weight file left both generations in the repo and the loader chose
    // between them by name.
    let mut server = mockito::Server::new();
    let _base = EnvGuard::set(protocol::TEST_BASE_ENV, &server.url());
    let _tok = EnvGuard::set("HF_TOKEN", "tok");

    let tmp = tempfile::tempdir().unwrap();
    make_minimal_vindex(tmp.path());

    let _create = server
        .mock("POST", "/api/repos/create")
        .with_status(200)
        .with_body("{}")
        .expect_at_least(1)
        .create();
    // Remote carries the local file plus a superseded weight file, its
    // manifest sibling, and two entries that must survive.
    let _tree = server
        .mock("GET", "/api/models/org/repo/tree/main?recursive=true")
        .with_status(200)
        .with_body(
            serde_json::json!([
                {"type":"file","path":"index.json"},
                {"type":"file","path":"interleaved_q4k.bin","lfs":{"oid":"stale"}},
                {"type":"file","path":"interleaved_q4k_manifest.json"},
                {"type":"file","path":"README.md"},
                {"type":"file","path":".gitattributes"}
            ])
            .to_string(),
        )
        .expect_at_least(1)
        .create();
    // Upload of index.json, then the prune commit. Both POST the same
    // path, so one mock serves both and the callback records the split.
    let _preupload = server
        .mock("POST", "/api/models/org/repo/preupload/main")
        .with_status(200)
        .with_body(r#"{"files":[{"path":"index.json","uploadMode":"regular"}]}"#)
        .expect_at_least(1)
        .create();
    let _commit = server
        .mock("POST", "/api/models/org/repo/commit/main")
        .with_status(200)
        .with_body("{}")
        .expect_at_least(1)
        .create();
    let _head = mock_repo_head_sha(&mut server, "org/repo", "deadbeef");

    #[derive(Default)]
    struct Recorder {
        deleted: Vec<String>,
    }
    impl PublishCallbacks for Recorder {
        fn on_file_deleted(&mut self, f: &str) {
            self.deleted.push(f.into());
        }
    }
    let mut rec = Recorder::default();

    publish_vindex_with_opts(tmp.path(), "org/repo", &PublishOptions::default(), &mut rec)
        .expect("publish with prune must return Ok");

    rec.deleted.sort();
    assert_eq!(
        rec.deleted,
        vec![
            "interleaved_q4k.bin".to_string(),
            "interleaved_q4k_manifest.json".to_string()
        ],
        "only files absent from the vindex, and never README/dot-files"
    );
}

#[test]
#[serial]
fn publish_with_prune_disabled_deletes_nothing() {
    let mut server = mockito::Server::new();
    let _base = EnvGuard::set(protocol::TEST_BASE_ENV, &server.url());
    let _tok = EnvGuard::set("HF_TOKEN", "tok");

    let tmp = tempfile::tempdir().unwrap();
    make_minimal_vindex(tmp.path());

    let _create = server
        .mock("POST", "/api/repos/create")
        .with_status(200)
        .with_body("{}")
        .expect_at_least(1)
        .create();
    let _preupload = server
        .mock("POST", "/api/models/org/repo/preupload/main")
        .with_status(200)
        .with_body(r#"{"files":[{"path":"index.json","uploadMode":"regular"}]}"#)
        .expect_at_least(1)
        .create();
    let _commit = server
        .mock("POST", "/api/models/org/repo/commit/main")
        .with_status(200)
        .with_body("{}")
        .expect_at_least(1)
        .create();
    let _head = mock_repo_head_sha(&mut server, "org/repo", "deadbeef");

    #[derive(Default)]
    struct Recorder {
        deleted: Vec<String>,
    }
    impl PublishCallbacks for Recorder {
        fn on_file_deleted(&mut self, f: &str) {
            self.deleted.push(f.into());
        }
    }
    let mut rec = Recorder::default();

    let opts = PublishOptions {
        prune_remote: false,
        ..PublishOptions::default()
    };
    publish_vindex_with_opts(tmp.path(), "org/repo", &opts, &mut rec)
        .expect("publish must return Ok");
    assert!(rec.deleted.is_empty(), "--no-prune must delete nothing");
}
