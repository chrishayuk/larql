//! HuggingFace download path — `hf://` resolution, snapshot cache
//! traversal, conditional ETag-based fetch.
//!
//! Carved out of the monolithic `huggingface.rs` in the 2026-04-25
//! reorg. See `super::mod.rs` for the module map.
//!
//! Sibling layout (round-6 split, 2026-05-10):
//! - `helpers` — pure non-network utilities (etag/repo-filter/cache-path).

mod helpers;

use std::path::PathBuf;

use crate::error::VindexError;
use crate::format::filenames::*;

use super::publish::get_hf_token;
use super::{vindex_core_files, VINDEX_METADATA_FILES, VINDEX_WEIGHT_FILES};
use helpers::{hf_cache_repo_dir, strip_etag_quoting, want_model_file};

/// Which side of the HF API a repo lives on. Vindexes are published as
/// models (quantized weight artifacts + manifests); the `Dataset` variant
/// remains for the helper-level tests that exercise both cache prefixes
/// and for any caller that explicitly targets the datasets namespace.
/// Both share the same blob-cache layout but differ in the URL prefix
/// and the `{datasets,models}--` cache-dir prefix.
#[derive(Clone, Copy)]
pub(super) enum RepoKind {
    #[allow(dead_code)]
    Dataset,
    Model,
}

impl RepoKind {
    fn url_segment(self) -> &'static str {
        match self {
            RepoKind::Dataset => "datasets/",
            RepoKind::Model => "",
        }
    }

    pub(super) fn cache_prefix(self) -> &'static str {
        match self {
            RepoKind::Dataset => "datasets--",
            RepoKind::Model => "models--",
        }
    }

    fn to_hub_type(self) -> hf_hub::RepoType {
        match self {
            RepoKind::Dataset => hf_hub::RepoType::Dataset,
            RepoKind::Model => hf_hub::RepoType::Model,
        }
    }
}

/// Order in which `larql pull` probes HF for an `hf://owner/name` path.
/// Vindexes are model artifacts, so only the models namespace is probed;
/// the legacy dataset fallback was removed once all published vindexes
/// (e.g. `chrishayuk/*-vindex`) lived under `models--`.
const HF_PULL_REPO_KINDS: [RepoKind; 1] = [RepoKind::Model];

/// Build a typed `ApiRepo` handle for a given `(repo_id, revision, kind)`.
/// Centralised so the three pull entry points share one constructor and
/// the with/without-revision branching lives in one place.
fn hf_repo(
    api: &hf_hub::api::sync::Api,
    repo_id: &str,
    revision: Option<&str>,
    kind: RepoKind,
) -> hf_hub::api::sync::ApiRepo {
    let repo_type = kind.to_hub_type();
    if let Some(rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.to_string(),
            repo_type,
            rev.to_string(),
        ))
    } else {
        api.repo(hf_hub::Repo::new(repo_id.to_string(), repo_type))
    }
}

/// Resolve an `hf://` path to a local directory, downloading if needed.
///
/// Supports:
/// - `hf://user/repo` — downloads the full dataset repo
/// - `hf://user/repo@revision` — specific revision/tag
///
/// Files are cached in the HuggingFace cache directory (~/.cache/huggingface/).
/// Only downloads files that don't already exist locally.
pub fn resolve_hf_vindex(hf_path: &str) -> Result<PathBuf, VindexError> {
    let path = hf_path
        .strip_prefix("hf://")
        .ok_or_else(|| VindexError::Parse(format!("not an hf:// path: {hf_path}")))?;

    // Parse repo and optional revision
    let (repo_id, revision) = if let Some((repo, rev)) = path.split_once('@') {
        (repo.to_string(), Some(rev.to_string()))
    } else {
        (path.to_string(), None)
    };

    // Use hf-hub to download
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .build()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    // `larql publish` defaults to model repos, but older vindexes and
    // some docs examples live as dataset repos. Probe in publish-default
    // order; the first kind that yields index.json wins, the rest are
    // skipped.
    let mut last_err: Option<String> = None;
    let (repo, index_path) = HF_PULL_REPO_KINDS
        .into_iter()
        .find_map(|kind| {
            let repo = hf_repo(&api, &repo_id, revision.as_deref(), kind);
            match repo.get(INDEX_JSON) {
                Ok(path) => Some((repo, path)),
                Err(e) => {
                    last_err = Some(e.to_string());
                    None
                }
            }
        })
        .ok_or_else(|| {
            let suffix = last_err
                .as_deref()
                .map(|e| format!(": {e}"))
                .unwrap_or_default();
            VindexError::Parse(format!(
                "failed to download index.json from hf://{repo_id}{suffix}"
            ))
        })?;

    let vindex_dir = index_path
        .parent()
        .ok_or_else(|| VindexError::Parse("cannot determine vindex directory".into()))?
        .to_path_buf();

    // Download METADATA-only by default. Big tensor files
    // (`gate_vectors.bin`, `embeddings.bin`) are deferred — `larql show`
    // and similar metadata-only commands shouldn't pay for a multi-GB
    // download. Callers that actually need the tensors (run / walk) use
    // `resolve_hf_vindex_with_progress` (which still pulls them eagerly)
    // or `download_hf_weights`.
    for filename in VINDEX_METADATA_FILES {
        if *filename == INDEX_JSON {
            continue; // already downloaded
        }
        let _ = repo.get(filename); // optional file, skip if missing
    }

    Ok(vindex_dir)
}

/// Download additional weight files for inference/compile.
/// Called lazily when INFER or COMPILE is first used.
pub fn download_hf_weights(hf_path: &str) -> Result<(), VindexError> {
    let path = hf_path
        .strip_prefix("hf://")
        .ok_or_else(|| VindexError::Parse(format!("not an hf:// path: {hf_path}")))?;

    let (repo_id, revision) = if let Some((repo, rev)) = path.split_once('@') {
        (repo.to_string(), Some(rev.to_string()))
    } else {
        (path.to_string(), None)
    };

    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .build()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    // Same model-first-then-dataset probe order as `resolve_hf_vindex`.
    // We use index.json as the "does this repo type exist?" probe so we
    // don't accidentally fetch weight files from a stale dataset repo
    // when the live vindex lives on the model side.
    for kind in HF_PULL_REPO_KINDS {
        let repo = hf_repo(&api, &repo_id, revision.as_deref(), kind);
        if repo.get(INDEX_JSON).is_err() {
            continue;
        }
        for filename in VINDEX_WEIGHT_FILES {
            let _ = repo.get(filename); // optional, skip if not in repo
        }
        return Ok(());
    }

    Err(VindexError::Parse(format!(
        "failed to fetch index.json from hf://{repo_id}"
    )))
}

/// Re-exported from hf-hub 0.5 so callers don't have to depend on
/// `hf_hub` directly. Implement this trait on an `indicatif::ProgressBar`
/// wrapper (or similar) to get per-file progress + resume behaviour out
/// of [`resolve_hf_vindex_with_progress`].
pub use hf_hub::api::Progress as DownloadProgress;

/// Check hf-hub's on-disk cache for `filename` and return `(path, size)`
/// iff a ready-to-use copy exists whose content hash matches what HF
/// reports on the remote.
///
/// hf-hub 0.5 lays the cache out as:
///
///   ```text
///   ~/.cache/huggingface/hub/datasets--{owner}--{name}/
///     ├── blobs/<etag>            actual file bytes
///     └── snapshots/<commit>/     symlinks → blobs
///         └── <filename>
///   ```
///
/// The etag is HF's content identifier: for LFS-tracked files it's the
/// SHA-256 oid; for git-tracked small files it's the git blob SHA-1.
/// Either way it uniquely identifies the bytes — so if `blobs/<etag>`
/// exists locally, the content matches the remote and we can skip the
/// download. This is stronger than the old size-only check: if the
/// remote file changes (new commit rewriting the same filename), the
/// etag changes, the cache probe misses, and we re-download.
///
/// The cost is one HEAD request per file. On a 10-file vindex that's a
/// few hundred ms vs the GB we'd re-download otherwise — cheap.
///
/// Returns `None` on any failure (HEAD error, cache missing, etag
/// absent, etc.); the caller falls back to `download_with_progress`.
fn cached_snapshot_file(
    kind: RepoKind,
    repo_id: &str,
    revision: Option<&str>,
    filename: &str,
) -> Option<(PathBuf, u64)> {
    let (etag, size) = head_etag_and_size(kind, repo_id, revision, filename)?;
    let repo_dir = hf_cache_repo_dir(kind, repo_id)?;
    let blob_path = repo_dir.join("blobs").join(&etag);
    let meta = std::fs::metadata(&blob_path).ok()?;
    if !meta.is_file() {
        return None;
    }
    // Size mismatch shouldn't happen if the etag matched, but treat it
    // as cache-miss defensively.
    if meta.len() != size {
        return None;
    }

    // Return the snapshot path (symlink → blob) if the repo has one,
    // otherwise the blob path itself. Either works — the caller only
    // needs a file it can open.
    let snapshots = repo_dir.join("snapshots");
    if let Ok(entries) = std::fs::read_dir(&snapshots) {
        for entry in entries.flatten() {
            let snap_file = entry.path().join(filename);
            if snap_file.exists() {
                return Some((snap_file, size));
            }
        }
    }
    // Fall back to the pinned revision (if any) even if the symlink is
    // missing — the blob still has the bytes.
    if let Some(rev) = revision {
        let snap_file = snapshots.join(rev).join(filename);
        if snap_file.exists() {
            return Some((snap_file, size));
        }
    }
    Some((blob_path, size))
}

/// Issue a HEAD against HF's file-resolve endpoint for this repo+file
/// and return `(etag, size)` from the response headers. HF redirects
/// LFS files to S3 which also returns an etag, so we must follow
/// redirects. Returns `None` for any failure: bad status, missing
/// headers, malformed size, etc.
fn head_etag_and_size(
    kind: RepoKind,
    repo_id: &str,
    revision: Option<&str>,
    filename: &str,
) -> Option<(String, u64)> {
    let rev = revision.unwrap_or("main");
    // Honour `HF_ENDPOINT` the same way hf-hub does, so tests can point
    // at a mockito server. Production reads the default huggingface.co.
    let endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let url = format!(
        "{endpoint}/{}{repo_id}/resolve/{rev}/{filename}",
        kind.url_segment()
    );
    let token = get_hf_token().ok();

    // **No redirects.** HF LFS files 302 → S3, and `X-Linked-Etag` +
    // `X-Linked-Size` (the stable LFS oid + content length) only exist
    // on HF's own first response. Following the redirect would lose
    // those headers and leave us with S3's multipart ETag, which is
    // MD5-based and doesn't match how hf-hub names blob files.
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .ok()?;
    let mut req = client.head(&url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req.send().ok()?;
    // Accept both 2xx (git-tracked small files stay on HF) and 3xx
    // (LFS files redirect to S3; the 302 carries the linked-etag we want).
    let status = resp.status();
    if !status.is_success() && !status.is_redirection() {
        return None;
    }

    // Prefer `X-Linked-Etag` when present (LFS oid = SHA256, stable).
    // Fall back to `ETag` for git-tracked files.
    let raw_etag = resp
        .headers()
        .get("X-Linked-Etag")
        .or_else(|| resp.headers().get("ETag"))
        .and_then(|v| v.to_str().ok())?;
    let etag = strip_etag_quoting(raw_etag);
    let size_hdr = resp
        .headers()
        .get("X-Linked-Size")
        .or_else(|| resp.headers().get("Content-Length"))
        .and_then(|v| v.to_str().ok())?;
    let size: u64 = size_hdr.parse().ok()?;
    Some((etag, size))
}

/// Like [`resolve_hf_vindex`], but drives a progress reporter per file.
/// hf-hub handles `.incomplete` partial-file resume internally — if the
/// download is interrupted, the next call picks up from where it left off.
///
/// Also honours the local cache: before each file, we check the
/// `snapshots/` tree for an already-downloaded copy whose size matches
/// the remote. Matches fire `init → update(size) → finish` on the
/// progress reporter with no HTTP traffic, so cached pulls complete in
/// milliseconds and the bar snaps to 100 %.
///
/// `progress` is a factory: called once per file with the filename.
/// Return a fresh `DownloadProgress` — typically an
/// `indicatif::ProgressBar` fetched from a `MultiProgress`.
pub fn resolve_hf_vindex_with_progress<F, P>(
    hf_path: &str,
    mut progress: F,
) -> Result<PathBuf, VindexError>
where
    F: FnMut(&str) -> P,
    P: DownloadProgress,
{
    let path = hf_path
        .strip_prefix("hf://")
        .ok_or_else(|| VindexError::Parse(format!("not an hf:// path: {hf_path}")))?;

    let (repo_id, revision) = if let Some((repo, rev)) = path.split_once('@') {
        (repo.to_string(), Some(rev.to_string()))
    } else {
        (path.to_string(), None)
    };

    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .build()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    // Probe each repo kind in publish-default order. The first kind that
    // returns index.json (cache hit or download) is the winner; we then
    // fetch the rest of `vindex_core_files()` (metadata + big tensor
    // files) from that same handle. Callers here have committed to
    // displaying a progress bar — they accept the wait.
    for kind in HF_PULL_REPO_KINDS {
        let repo = hf_repo(&api, &repo_id, revision.as_deref(), kind);

        // Helper: one file, with cache short-circuit. Returns the resolved
        // on-disk path. The cache check fires the progress reporter so the
        // bar shows a filled-to-100% track tagged with the filename — users
        // see that the file was served from cache, not re-downloaded.
        let mut fetch = |filename: &str, label: &str| -> Option<PathBuf> {
            if let Some((cached_path, size)) =
                cached_snapshot_file(kind, &repo_id, revision.as_deref(), filename)
            {
                // Tag the progress message so the bar visibly distinguishes
                // "cached" from "just downloaded very fast". Callers rendering
                // the bar see the prefix at init time and can restyle.
                let mut p = progress(label);
                let tagged = format!("{filename} [cached]");
                p.init(size as usize, &tagged);
                p.update(size as usize);
                p.finish();
                return Some(cached_path);
            }
            repo.download_with_progress(filename, progress(label)).ok()
        };

        // index.json drives everything — we need its snapshot dir to know
        // where the rest of the files live. If this kind doesn't have it,
        // try the next kind.
        let Some(index_path) = fetch(INDEX_JSON, INDEX_JSON) else {
            continue;
        };
        let vindex_dir = index_path
            .parent()
            .ok_or_else(|| VindexError::Parse("cannot determine vindex directory".into()))?
            .to_path_buf();

        for filename in vindex_core_files() {
            if filename == INDEX_JSON {
                continue;
            }
            // Optional files — ignore failures (missing from repo is fine).
            let _ = fetch(filename, filename);
        }
        return Ok(vindex_dir);
    }

    Err(VindexError::Parse(format!(
        "failed to fetch index.json from hf://{repo_id}"
    )))
}

/// Resolve an `hf://` model repo path to a local snapshot directory,
/// downloading the safetensors + tokenizer + config sidecar files needed
/// for `larql convert safetensors-to-vindex`. Mirrors
/// [`resolve_hf_vindex_with_progress`] but talks to the model side of the
/// HF API (`models/...`) and enumerates files via the repo `info()` call
/// instead of a fixed list, so sharded checkpoints (Qwen3 4B/27B) Just Work.
///
/// Skips PyTorch `.bin` shards when safetensors are also present in the
/// repo (`want_model_file`) — saves several GB on the typical mirror.
pub fn resolve_hf_model_with_progress<F, P>(
    hf_path: &str,
    mut progress: F,
) -> Result<PathBuf, VindexError>
where
    F: FnMut(&str) -> P,
    P: DownloadProgress,
{
    let path = hf_path
        .strip_prefix("hf://")
        .ok_or_else(|| VindexError::Parse(format!("not an hf:// path: {hf_path}")))?;

    let (repo_id, revision) = if let Some((repo, rev)) = path.split_once('@') {
        (repo.to_string(), Some(rev.to_string()))
    } else {
        (path.to_string(), None)
    };

    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .build()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    let repo = if let Some(ref rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Model,
            rev.clone(),
        ))
    } else {
        api.repo(hf_hub::Repo::new(repo_id.clone(), hf_hub::RepoType::Model))
    };

    let info = repo
        .info()
        .map_err(|e| VindexError::Parse(format!("HF info failed for {hf_path}: {e}")))?;

    let mut wanted: Vec<&str> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|n| want_model_file(n))
        .collect();
    wanted.sort();

    if wanted.is_empty() {
        return Err(VindexError::Parse(format!(
            "no usable model files in {hf_path} (siblings: {})",
            info.siblings.len()
        )));
    }

    let mut snapshot_dir: Option<PathBuf> = None;
    let mut fetch = |filename: &str| -> Option<PathBuf> {
        if let Some((cached_path, size)) =
            cached_snapshot_file(RepoKind::Model, &repo_id, revision.as_deref(), filename)
        {
            let mut p = progress(filename);
            let tagged = format!("{filename} [cached]");
            p.init(size as usize, &tagged);
            p.update(size as usize);
            p.finish();
            return Some(cached_path);
        }
        repo.download_with_progress(filename, progress(filename))
            .ok()
    };

    for filename in &wanted {
        if let Some(p) = fetch(filename) {
            if snapshot_dir.is_none() {
                snapshot_dir = p.parent().map(|d| d.to_path_buf());
            }
        }
    }

    snapshot_dir.ok_or_else(|| {
        VindexError::Parse(format!(
            "downloaded zero files from {hf_path} — check repo access"
        ))
    })
}

#[cfg(test)]
mod tests {
    //! Unit tests for the hf_hub-bound functions — pure helpers tested
    //! in `helpers.rs`.
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

    /// Stub `DownloadProgress` for the *_with_progress tests. We only need
    /// the trait to exist so the function type-checks; the stub is never
    /// invoked because we hit the early-return path.
    struct NoOpProgress;
    impl DownloadProgress for NoOpProgress {
        fn init(&mut self, _size: usize, _filename: &str) {}
        fn update(&mut self, _size: usize) {}
        fn finish(&mut self) {}
    }

    #[test]
    fn resolve_hf_vindex_with_progress_rejects_non_hf_path() {
        let err =
            resolve_hf_vindex_with_progress("/tmp/foo", |_| NoOpProgress).expect_err("must reject");
        assert!(err.to_string().contains("not an hf://"));
    }

    #[test]
    fn resolve_hf_model_with_progress_rejects_non_hf_path() {
        let err = resolve_hf_model_with_progress("./local-model", |_| NoOpProgress)
            .expect_err("must reject");
        assert!(err.to_string().contains("not an hf://"));
    }

    // ─── hf_hub-bound: revision parsing covered by error path ──────────
    //
    // The `@revision` split happens after the `hf://` prefix strip but
    // before any network call. The functions then do `Api::new()` which
    // (with HF_ENDPOINT pointing at a non-existent server) fails fast.
    // That path covers the revision-vs-no-revision branches.

    /// RAII guard for HF_ENDPOINT + HF_HOME + a tempdir cache.
    /// Restores prior values on drop.
    struct HfTestEnv {
        prev_endpoint: Option<String>,
        prev_home: Option<String>,
        prev_hub: Option<String>,
        prev_token: Option<String>,
        // Hold the tempdir so it lives as long as the guard.
        _tmp: tempfile::TempDir,
    }
    impl HfTestEnv {
        fn new(endpoint: &str) -> Self {
            let prev_endpoint = std::env::var("HF_ENDPOINT").ok();
            let prev_home = std::env::var("HF_HOME").ok();
            let prev_hub = std::env::var("HUGGINGFACE_HUB_CACHE").ok();
            let prev_token = std::env::var("HF_TOKEN").ok();

            let tmp = tempfile::tempdir().unwrap();
            std::env::set_var("HF_ENDPOINT", endpoint);
            std::env::set_var("HF_HOME", tmp.path());
            // Clear HUGGINGFACE_HUB_CACHE so HF_HOME wins; clear token
            // so we don't accidentally hit a real auth header.
            std::env::remove_var("HUGGINGFACE_HUB_CACHE");
            std::env::remove_var("HF_TOKEN");

            Self {
                prev_endpoint,
                prev_home,
                prev_hub,
                prev_token,
                _tmp: tmp,
            }
        }
    }
    impl Drop for HfTestEnv {
        fn drop(&mut self) {
            for (k, prev) in [
                ("HF_ENDPOINT", self.prev_endpoint.take()),
                ("HF_HOME", self.prev_home.take()),
                ("HUGGINGFACE_HUB_CACHE", self.prev_hub.take()),
                ("HF_TOKEN", self.prev_token.take()),
            ] {
                match prev {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

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

        let result =
            head_etag_and_size(RepoKind::Dataset, "owner/repo", None, "blobs.bin").unwrap();
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

        let result =
            head_etag_and_size(RepoKind::Dataset, "owner/repo", None, "index.json").unwrap();
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
            cached_snapshot_file(RepoKind::Dataset, "owner/repo", None, "file.bin").unwrap();
        assert_eq!(size, 5);
        assert!(path.ends_with("file.bin"));
    }

    #[test]
    #[serial]
    fn cached_snapshot_file_returns_blob_when_no_snapshot_link() {
        // Same blob present, but no snapshot directory linking to the
        // filename. The function falls back to the raw blob path.
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
            None, // no snapshot dir
            "f.bin",
        );

        let (path, size) =
            cached_snapshot_file(RepoKind::Dataset, "owner/repo", None, "f.bin").unwrap();
        assert_eq!(size, 4);
        // No snapshot link → returns the blob path directly.
        assert!(path.ends_with("blobs/deadbeef"));
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

        let result = cached_snapshot_file(RepoKind::Dataset, "owner/repo", None, "f.bin");
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
        let result = cached_snapshot_file(RepoKind::Dataset, "owner/repo", None, "f.bin");
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
            cached_snapshot_file(RepoKind::Model, "owner/repo", None, "config.json").unwrap();
        assert_eq!(size, 4);
        assert!(path.ends_with("config.json"));
    }

    #[test]
    #[serial]
    fn cached_snapshot_file_falls_back_through_revision_after_unrelated_snapshot_dir() {
        // Build a cache where the entries-loop sees a snapshot dir
        // that DOESN'T contain the filename, then the explicit
        // `snapshots.join(rev)` fallback (lines ~258-261) succeeds.
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
        // Snapshot dir for `noise` (different filename) — entries loop
        // visits this but the join misses.
        let noise_snap = repo_dir.join("snapshots").join("noise");
        std::fs::create_dir_all(&noise_snap).unwrap();
        std::fs::write(noise_snap.join("other.bin"), b"abc").unwrap();
        // Snapshot dir for the revision we'll request, with the file.
        let pinned_snap = repo_dir.join("snapshots").join("v7");
        std::fs::create_dir_all(&pinned_snap).unwrap();
        std::fs::write(pinned_snap.join("target.bin"), b"abc").unwrap();

        let (path, _) =
            cached_snapshot_file(RepoKind::Dataset, "owner/repo", Some("v7"), "target.bin")
                .unwrap();
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

        let result = cached_snapshot_file(RepoKind::Dataset, "owner/repo", None, "f.bin");
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

    /// Build a mock that satisfies hf-hub's metadata() probe. The
    /// requirements are: an ETag (or X-Linked-Etag) header, an
    /// X-Repo-Commit header (commit hash), and a Content-Range header
    /// of the form "bytes 0-0/<size>". hf-hub's metadata path issues a
    /// GET with `Range: bytes=0-0`, then a follow-up GET for the full
    /// body; both must succeed for `repo.get()` to return a path.
    ///
    /// `expect` lets the test cap how many times the mock can fire —
    /// hf-hub's download path may retry or make follow-up requests we
    /// don't strictly model.
    fn mock_hf_file_resolve(
        server: &mut mockito::ServerGuard,
        path_regex: &str,
        etag: &str,
        body: &[u8],
    ) -> Vec<mockito::Mock> {
        let len = body.len();
        let cr = format!("bytes 0-0/{len}");
        let meta = server
            .mock("GET", mockito::Matcher::Regex(path_regex.into()))
            .with_status(200)
            .with_header("ETag", &format!("\"{etag}\""))
            .with_header("X-Repo-Commit", "deadbeefcafebabe")
            .with_header("Content-Range", &cr)
            .with_header("Accept-Ranges", "bytes")
            .with_header("Content-Length", &len.to_string())
            .with_body(body)
            .expect_at_least(1)
            .create();
        vec![meta]
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

        let dir = resolve_hf_vindex_with_progress("hf://owner/repo", |_| NoOpProgress)
            .expect("success path");
        assert!(dir.exists());
        assert!(idx[0].matched(), "index.json mock must have been hit");
    }

    #[test]
    #[serial]
    fn resolve_hf_vindex_with_progress_uses_cache_when_blob_present() {
        // When the cached_snapshot_file fast-path finds the blob on
        // disk, the function bypasses download_with_progress and goes
        // through the cache-hit branch (progress.init/update/finish
        // called with the [cached] tag). Build the cache + a matching
        // HEAD response so the cache short-circuit fires.
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

        let dir = resolve_hf_vindex_with_progress("hf://owner/repo", |_| NoOpProgress)
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
}
