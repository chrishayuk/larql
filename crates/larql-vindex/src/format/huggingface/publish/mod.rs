//! HuggingFace publish path — repo creation + per-file upload + LFS
//! pointer/upload protocol + callback hooks.
//!
//! Carved out of the monolithic `huggingface.rs` in the 2026-04-25
//! reorg, then split again 2026-05-09 into:
//!   - `mod.rs`     — public API (`publish_vindex*`, `PublishOptions`,
//!                    `PublishCallbacks`), URL helpers,
//!                    `enumerate_publishable_files`, `get_hf_token`,
//!                    tests
//!   - `remote.rs`  — `fetch_remote_lfs_oids`, `create_hf_repo`
//!   - `upload.rs`  — `upload_file_to_hf` + preupload + `upload_regular`
//!   - `lfs.rs`     — LFS protocol (batch / verify / commit) + streaming
//!                    PUT + `CountingReader`

mod lfs;
pub(super) mod protocol;
mod remote;
mod upload;

use std::path::{Path, PathBuf};

use crate::error::VindexError;
use crate::format::filenames::*;

use protocol::{hf_base, repo_type_plural, REPO_TYPE_DATASET, REPO_TYPE_MODEL};
use remote::{create_hf_repo, fetch_remote_lfs_oids};
use upload::upload_file_to_hf;

/// Options controlling [`publish_vindex_with_opts`]. Kept as a struct so
/// the signature can grow without breaking callers.
#[derive(Clone, Debug)]
pub struct PublishOptions {
    /// When true, skip uploading LFS-tracked files whose local SHA256
    /// already matches the remote `lfs.oid`. Small files (git-tracked
    /// json / manifest) are always re-uploaded — their text is tiny and
    /// the git blob SHA-1 format isn't directly derivable from the file
    /// content SHA256 without a separate hash.
    pub skip_unchanged: bool,
    /// HuggingFace repo type: `"model"` (default) or `"dataset"`.
    pub repo_type: String,
}

impl Default for PublishOptions {
    fn default() -> Self {
        Self {
            skip_unchanged: false,
            repo_type: REPO_TYPE_MODEL.into(),
        }
    }
}

impl PublishOptions {
    pub fn skip_unchanged() -> Self {
        Self {
            skip_unchanged: true,
            ..Self::default()
        }
    }
}

/// Returns the HF API base URL for a repo:
/// `{base}/api/{models|datasets}/{repo_id}`.
#[allow(dead_code)]
fn hf_api_url(repo_type: &str, repo_id: &str, path: &str) -> String {
    let base = hf_base();
    let plural = repo_type_plural(repo_type);
    format!("{base}/api/{plural}/{repo_id}/{path}")
}

/// Returns the web / git base URL for a repo.
/// Models: `{base}/{repo_id}`, datasets: `{base}/datasets/{repo_id}`.
pub(super) fn hf_repo_url(repo_type: &str, repo_id: &str) -> String {
    let base = hf_base();
    if repo_type == REPO_TYPE_DATASET {
        format!("{base}/datasets/{repo_id}")
    } else {
        format!("{base}/{repo_id}")
    }
}

/// Upload a local vindex directory to HuggingFace as a model repo
/// (the [`PublishOptions::default`] `repo_type`). Pass a customised
/// `PublishOptions` to [`publish_vindex_with_opts`] to publish under
/// the datasets namespace instead.
///
/// Equivalent to `publish_vindex_with_opts(dir, repo_id, &PublishOptions::default(), cb)`.
/// Requires HF_TOKEN environment variable or ~/.huggingface/token.
pub fn publish_vindex(
    vindex_dir: &Path,
    repo_id: &str,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<String, VindexError> {
    publish_vindex_with_opts(vindex_dir, repo_id, &PublishOptions::default(), callbacks)
}

/// Upload a vindex directory with explicit options. See [`PublishOptions`].
pub fn publish_vindex_with_opts(
    vindex_dir: &Path,
    repo_id: &str,
    opts: &PublishOptions,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<String, VindexError> {
    if !vindex_dir.is_dir() {
        return Err(VindexError::NotADirectory(vindex_dir.to_path_buf()));
    }
    let index_path = vindex_dir.join(INDEX_JSON);
    if !index_path.exists() {
        return Err(VindexError::Parse(format!(
            "not a vindex directory (no index.json): {}",
            vindex_dir.display()
        )));
    }

    let token = get_hf_token()?;
    let repo_type = opts.repo_type.as_str();
    callbacks.on_start(repo_id);
    create_hf_repo(repo_id, &token, repo_type)?;

    // Pull remote LFS index so we can skip unchanged files. Non-fatal
    // if the tree API errors (brand-new repo returns 404 here) — we just
    // fall back to "upload everything".
    let remote_lfs: std::collections::HashMap<String, String> = if opts.skip_unchanged {
        fetch_remote_lfs_oids(repo_id, &token, repo_type).unwrap_or_default()
    } else {
        std::collections::HashMap::new()
    };

    // Collect files from the root and any immediate subdirectories (e.g. layers/).
    let files = enumerate_publishable_files(vindex_dir)?;

    for (file_path, filename) in &files {
        let size = std::fs::metadata(file_path).map(|m| m.len()).unwrap_or(0);

        // Skip-if-unchanged: compare local SHA256 against remote lfs.oid.
        if opts.skip_unchanged {
            if let Some(remote_sha) = remote_lfs.get(filename) {
                if let Ok(local_sha) = crate::format::checksums::sha256_file(file_path) {
                    if local_sha == *remote_sha {
                        callbacks.on_file_skipped(filename, size, remote_sha);
                        continue;
                    }
                }
            }
        }

        callbacks.on_file_start(filename, size);
        upload_file_to_hf(repo_id, &token, file_path, filename, callbacks, repo_type)?;
        callbacks.on_file_done(filename);
    }

    let url = hf_repo_url(repo_type, repo_id);
    callbacks.on_complete(&url);
    Ok(url)
}

/// Enumerate publishable files in a vindex directory: every file at the
/// root plus every file in immediate subdirectories (e.g. `layers/`).
/// Result is sorted by repo path so commits are reproducible.
///
/// Returned tuples are `(absolute_path, repo_relative_path)` — the second
/// is what HuggingFace sees and is always forward-slash separated.
fn enumerate_publishable_files(vindex_dir: &Path) -> Result<Vec<(PathBuf, String)>, VindexError> {
    let mut files: Vec<(PathBuf, String)> = Vec::new();
    for entry in std::fs::read_dir(vindex_dir)?.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            files.push((path, name));
        } else if path.is_dir() {
            let dir_name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            for sub in std::fs::read_dir(&path)
                .ok()
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
            {
                let sub_path = sub.path();
                if sub_path.is_file() {
                    let sub_name = sub_path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    files.push((sub_path, format!("{dir_name}/{sub_name}")));
                }
            }
        }
    }
    files.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(files)
}

/// Callbacks for publish progress.
pub trait PublishCallbacks {
    fn on_start(&mut self, _repo: &str) {}
    fn on_file_start(&mut self, _filename: &str, _size: u64) {}
    /// Fired periodically during the upload with cumulative bytes sent
    /// for the current file. Default no-op. Implement to render a live
    /// progress bar; indicatif wrappers live in the CLI layer to stay
    /// version-agnostic here.
    fn on_file_progress(&mut self, _filename: &str, _bytes_sent: u64, _total_bytes: u64) {}
    fn on_file_done(&mut self, _filename: &str) {}
    /// Fired when [`PublishOptions::skip_unchanged`] matches the remote
    /// `lfs.oid` and the upload is skipped. Default no-op so existing
    /// callbacks don't need to change.
    fn on_file_skipped(&mut self, _filename: &str, _size: u64, _sha256: &str) {}
    fn on_complete(&mut self, _url: &str) {}
}

pub struct SilentPublishCallbacks;
impl PublishCallbacks for SilentPublishCallbacks {}

pub(in crate::format::huggingface) fn get_hf_token() -> Result<String, VindexError> {
    // Try environment variable first
    if let Ok(token) = std::env::var("HF_TOKEN") {
        return Ok(token);
    }

    // Try token file
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let token_path = PathBuf::from(&home).join(".huggingface").join("token");
    if token_path.exists() {
        let token = std::fs::read_to_string(&token_path)?;
        return Ok(token.trim().to_string());
    }

    // Try newer cache location
    let token_path = PathBuf::from(&home)
        .join(".cache")
        .join("huggingface")
        .join("token");
    if token_path.exists() {
        let token = std::fs::read_to_string(&token_path)?;
        return Ok(token.trim().to_string());
    }

    Err(VindexError::Parse(
        "HuggingFace token not found. Set HF_TOKEN or run `huggingface-cli login`.".into(),
    ))
}

#[cfg(test)]
mod tests {
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
        let err =
            publish_vindex_with_opts(&file_path, "org/repo", &PublishOptions::default(), &mut cb)
                .expect_err("path is a file, not a directory");
        assert!(matches!(err, VindexError::NotADirectory(_)));
    }

    #[test]
    fn publish_rejects_directory_without_index_json() {
        let dir = tempfile::tempdir().unwrap();
        // Directory exists but has no index.json — should fail before any
        // network call (no token required to reach this branch).
        let mut cb = SilentPublishCallbacks;
        let err =
            publish_vindex_with_opts(dir.path(), "org/repo", &PublishOptions::default(), &mut cb)
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
            let prev_home = std::env::var("HOME").ok();
            let tmp = tempfile::tempdir().unwrap();
            std::env::remove_var("HF_TOKEN");
            std::env::set_var("HOME", tmp.path());
            Self {
                prev_token,
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
        let err =
            publish_vindex(&missing, "org/repo", &mut cb).expect_err("missing path must error");
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

        let url =
            publish_vindex_with_opts(tmp.path(), "org/repo", &PublishOptions::default(), &mut rec)
                .expect("happy path must return Ok");

        assert!(rec.started, "on_start must fire");
        assert_eq!(rec.files_started, vec!["index.json".to_string()]);
        assert_eq!(rec.files_done, vec!["index.json".to_string()]);
        assert_eq!(rec.completed_url.as_deref(), Some(url.as_str()));
        assert!(
            url.ends_with("/org/repo"),
            "model repo URL should end with /org/repo, got: {url}"
        );
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
}
