//! HuggingFace download path — `hf://` resolution, snapshot cache
//! traversal, conditional ETag-based fetch.
//!
//! Carved out of the monolithic `huggingface.rs` in the 2026-04-25
//! reorg. See `super::mod.rs` for the module map.

use std::path::PathBuf;

use crate::error::VindexError;
use crate::format::filenames::*;

use super::publish::get_hf_token;
use super::{VINDEX_CORE_FILES, VINDEX_WEIGHT_FILES};

const HF_PULL_REPO_TYPES: [hf_hub::RepoType; 2] =
    [hf_hub::RepoType::Model, hf_hub::RepoType::Dataset];

fn hf_repo(
    api: &hf_hub::api::sync::Api,
    repo_id: &str,
    revision: Option<&str>,
    repo_type: hf_hub::RepoType,
) -> hf_hub::api::sync::ApiRepo {
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

fn repo_type_cache_prefix(repo_type: hf_hub::RepoType) -> &'static str {
    match repo_type {
        hf_hub::RepoType::Model => "models",
        hf_hub::RepoType::Dataset => "datasets",
        hf_hub::RepoType::Space => "spaces",
    }
}

fn repo_type_resolve_prefix(repo_type: hf_hub::RepoType) -> &'static str {
    match repo_type {
        hf_hub::RepoType::Model => "",
        hf_hub::RepoType::Dataset => "datasets/",
        hf_hub::RepoType::Space => "spaces/",
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
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    // Prefer model repos because `larql publish` defaults to `repo_type = "model"`,
    // but keep dataset fallback for older vindex repos and docs examples.
    let mut last_err = None;
    let (repo, index_path) = HF_PULL_REPO_TYPES
        .into_iter()
        .find_map(|repo_type| {
            let repo = hf_repo(&api, &repo_id, revision.as_deref(), repo_type);
            match repo.get(INDEX_JSON) {
                Ok(path) => Some((repo, path)),
                Err(e) => {
                    last_err = Some(e.to_string());
                    None
                }
            }
        })
        .ok_or_else(|| {
            VindexError::Parse(format!(
                "failed to download index.json from hf://{}{}",
                repo_id,
                last_err
                    .as_deref()
                    .map(|e| format!(": {e}"))
                    .unwrap_or_default()
            ))
        })?;

    let vindex_dir = index_path
        .parent()
        .ok_or_else(|| VindexError::Parse("cannot determine vindex directory".into()))?
        .to_path_buf();

    // Download core files (needed for browse)
    for filename in VINDEX_CORE_FILES {
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

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    for repo_type in HF_PULL_REPO_TYPES {
        let repo = hf_repo(&api, &repo_id, revision.as_deref(), repo_type);
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
    repo_id: &str,
    revision: Option<&str>,
    filename: &str,
    repo_type: hf_hub::RepoType,
) -> Option<(PathBuf, u64)> {
    let (etag, size) = head_etag_and_size(repo_id, revision, filename, repo_type)?;
    let repo_dir = hf_cache_repo_dir(repo_id, repo_type)?;
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
    repo_id: &str,
    revision: Option<&str>,
    filename: &str,
    repo_type: hf_hub::RepoType,
) -> Option<(String, u64)> {
    let rev = revision.unwrap_or("main");
    let prefix = repo_type_resolve_prefix(repo_type);
    let url = format!("https://huggingface.co/{prefix}{repo_id}/resolve/{rev}/{filename}");
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

/// Normalise an HTTP ETag header to the raw content hash hf-hub uses
/// as blob filenames. Handles:
///   * strong etag: `"abc123"` → `abc123`
///   * weak etag:   `W/"abc123"` → `abc123`
fn strip_etag_quoting(raw: &str) -> String {
    let trimmed = raw.trim();
    let no_weak = trimmed.strip_prefix("W/").unwrap_or(trimmed);
    no_weak.trim_matches('"').to_string()
}

/// Resolve the hf-hub cache directory for a dataset repo: the root of
/// `~/.cache/huggingface/hub/datasets--{owner}--{name}/`. Honours
/// `HF_HOME` and `HUGGINGFACE_HUB_CACHE` env overrides that hf-hub itself
/// respects.
fn hf_cache_repo_dir(repo_id: &str, repo_type: hf_hub::RepoType) -> Option<PathBuf> {
    let hub_root = if let Ok(hub) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        PathBuf::from(hub)
    } else if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home).join("hub")
    } else {
        let home = std::env::var("HOME").ok()?;
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub")
    };
    let safe = repo_id.replace('/', "--");
    Some(hub_root.join(format!("{}--{safe}", repo_type_cache_prefix(repo_type))))
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

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    for repo_type in HF_PULL_REPO_TYPES {
        let repo = hf_repo(&api, &repo_id, revision.as_deref(), repo_type);

        // Helper: one file, with cache short-circuit. Returns the resolved
        // on-disk path. The cache check fires the progress reporter so the
        // bar shows a filled-to-100% track tagged with the filename — users
        // see that the file was served from cache, not re-downloaded.
        let mut fetch = |filename: &str, label: &str| -> Option<PathBuf> {
            if let Some((cached_path, size)) =
                cached_snapshot_file(&repo_id, revision.as_deref(), filename, repo_type)
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
        // where the rest of the files live. Cache-hit or download.
        let Some(index_path) = fetch(INDEX_JSON, INDEX_JSON) else {
            continue;
        };
        let vindex_dir = index_path
            .parent()
            .ok_or_else(|| VindexError::Parse("cannot determine vindex directory".into()))?
            .to_path_buf();

        for filename in VINDEX_CORE_FILES {
            if *filename == INDEX_JSON {
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
