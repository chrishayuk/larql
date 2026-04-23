//! HuggingFace Hub integration — download and upload vindexes.
//!
//! Vindexes are stored as HuggingFace dataset repos. Each file in the vindex
//! directory maps 1:1 to a file in the repo. HuggingFace's CDN handles
//! distribution, caching, and access control.
//!
//! ```text
//! # Download a vindex
//! larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
//!
//! # Upload a vindex
//! larql publish gemma3-4b.vindex --repo chrishayuk/gemma-3-4b-it-vindex
//! ```

use std::path::{Path, PathBuf};

use crate::error::VindexError;

/// The files that make up a vindex, in priority order for lazy loading.
const VINDEX_CORE_FILES: &[&str] = &[
    "index.json",
    "tokenizer.json",
    "gate_vectors.bin",
    "embeddings.bin",
    "down_meta.bin",
    "down_meta.jsonl",
    "relation_clusters.json",
    "feature_labels.json",
];

const VINDEX_WEIGHT_FILES: &[&str] = &[
    "attn_weights.bin",
    "norms.bin",
    "up_weights.bin",
    "down_weights.bin",
    "lm_head.bin",
    "weight_manifest.json",
];

/// Resolve an `hf://` path to a local directory, downloading if needed.
///
/// Supports:
/// - `hf://user/repo` — downloads the full dataset repo
/// - `hf://user/repo@revision` — specific revision/tag
///
/// Files are cached in the HuggingFace cache directory (~/.cache/huggingface/).
/// Only downloads files that don't already exist locally.
pub fn resolve_hf_vindex(hf_path: &str) -> Result<PathBuf, VindexError> {
    let path = hf_path.strip_prefix("hf://")
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

    let repo = if let Some(ref rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
            rev.clone(),
        ))
    } else {
        api.repo(hf_hub::Repo::new(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
        ))
    };

    // Download index.json first (small, tells us what we need)
    let index_path = repo.get("index.json")
        .map_err(|e| VindexError::Parse(format!(
            "failed to download index.json from hf://{}: {e}", repo_id
        )))?;

    let vindex_dir = index_path.parent()
        .ok_or_else(|| VindexError::Parse("cannot determine vindex directory".into()))?
        .to_path_buf();

    // Download core files (needed for browse)
    for filename in VINDEX_CORE_FILES {
        if *filename == "index.json" {
            continue; // already downloaded
        }
        let _ = repo.get(filename); // optional file, skip if missing
    }

    Ok(vindex_dir)
}

/// Download additional weight files for inference/compile.
/// Called lazily when INFER or COMPILE is first used.
pub fn download_hf_weights(hf_path: &str) -> Result<(), VindexError> {
    let path = hf_path.strip_prefix("hf://")
        .ok_or_else(|| VindexError::Parse(format!("not an hf:// path: {hf_path}")))?;

    let (repo_id, revision) = if let Some((repo, rev)) = path.split_once('@') {
        (repo.to_string(), Some(rev.to_string()))
    } else {
        (path.to_string(), None)
    };

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    let repo = if let Some(ref rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
            rev.clone(),
        ))
    } else {
        api.repo(hf_hub::Repo::new(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
        ))
    };

    for filename in VINDEX_WEIGHT_FILES {
        let _ = repo.get(filename); // optional, skip if not in repo
    }

    Ok(())
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
) -> Option<(PathBuf, u64)> {
    let (etag, size) = head_etag_and_size(repo_id, revision, filename)?;
    let repo_dir = hf_cache_repo_dir(repo_id)?;
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
) -> Option<(String, u64)> {
    let rev = revision.unwrap_or("main");
    let url = format!(
        "https://huggingface.co/datasets/{repo_id}/resolve/{rev}/{filename}"
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
fn hf_cache_repo_dir(repo_id: &str) -> Option<PathBuf> {
    let hub_root = if let Ok(hub) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        PathBuf::from(hub)
    } else if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home).join("hub")
    } else {
        let home = std::env::var("HOME").ok()?;
        PathBuf::from(home).join(".cache").join("huggingface").join("hub")
    };
    let safe = repo_id.replace('/', "--");
    Some(hub_root.join(format!("datasets--{safe}")))
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

    let repo = if let Some(ref rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
            rev.clone(),
        ))
    } else {
        api.repo(hf_hub::Repo::new(repo_id.clone(), hf_hub::RepoType::Dataset))
    };

    // Helper: one file, with cache short-circuit. Returns the resolved
    // on-disk path. The cache check fires the progress reporter so the
    // bar shows a filled-to-100% track tagged with the filename — users
    // see that the file was served from cache, not re-downloaded.
    let mut fetch = |filename: &str, label: &str| -> Option<PathBuf> {
        if let Some((cached_path, size)) = cached_snapshot_file(&repo_id, revision.as_deref(), filename) {
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
        match repo.download_with_progress(filename, progress(label)) {
            Ok(path) => Some(path),
            Err(_) => None,
        }
    };

    // index.json drives everything — we need its snapshot dir to know
    // where the rest of the files live. Cache-hit or download.
    let index_path = fetch("index.json", "index.json").ok_or_else(|| {
        VindexError::Parse(format!(
            "failed to fetch index.json from hf://{repo_id}"
        ))
    })?;
    let vindex_dir = index_path
        .parent()
        .ok_or_else(|| VindexError::Parse("cannot determine vindex directory".into()))?
        .to_path_buf();

    for filename in VINDEX_CORE_FILES {
        if *filename == "index.json" {
            continue;
        }
        // Optional files — ignore failures (missing from repo is fine).
        let _ = fetch(filename, filename);
    }
    Ok(vindex_dir)
}

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
        Self { skip_unchanged: false, repo_type: "model".into() }
    }
}

impl PublishOptions {
    pub fn skip_unchanged() -> Self {
        Self { skip_unchanged: true, ..Self::default() }
    }
}

/// Returns the HF API base URL for a repo: `https://huggingface.co/api/{models|datasets}/{repo_id}`.
fn hf_api_url(repo_type: &str, repo_id: &str, path: &str) -> String {
    let plural = if repo_type == "dataset" { "datasets" } else { "models" };
    format!("https://huggingface.co/api/{plural}/{repo_id}/{path}")
}

/// Returns the web / git base URL for a repo.
/// Models: `https://huggingface.co/{repo_id}`, datasets: `https://huggingface.co/datasets/{repo_id}`.
fn hf_repo_url(repo_type: &str, repo_id: &str) -> String {
    if repo_type == "dataset" {
        format!("https://huggingface.co/datasets/{repo_id}")
    } else {
        format!("https://huggingface.co/{repo_id}")
    }
}

/// Upload a local vindex directory to HuggingFace as a dataset repo.
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
    let index_path = vindex_dir.join("index.json");
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

    let mut files: Vec<PathBuf> = std::fs::read_dir(vindex_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();

    for file_path in &files {
        let filename = file_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let size = std::fs::metadata(file_path).map(|m| m.len()).unwrap_or(0);

        // Skip-if-unchanged: compare local SHA256 against remote lfs.oid.
        if opts.skip_unchanged {
            if let Some(remote_sha) = remote_lfs.get(&filename) {
                if let Ok(local_sha) = crate::format::checksums::sha256_file(file_path) {
                    if local_sha == *remote_sha {
                        callbacks.on_file_skipped(&filename, size, remote_sha);
                        continue;
                    }
                }
            }
        }

        callbacks.on_file_start(&filename, size);
        upload_file_to_hf(repo_id, &token, file_path, &filename, callbacks, repo_type)?;
        callbacks.on_file_done(&filename);
    }

    let url = hf_repo_url(repo_type, repo_id);
    callbacks.on_complete(&url);
    Ok(url)
}

/// List remote files and return `filename → lfs.oid` for every LFS-tracked
/// file at the repo root. Files without an `lfs.oid` (git-tracked small
/// text) are omitted; callers skip only what's in the map.
fn fetch_remote_lfs_oids(
    repo_id: &str,
    token: &str,
    repo_type: &str,
) -> Result<std::collections::HashMap<String, String>, VindexError> {
    let plural = if repo_type == "dataset" { "datasets" } else { "models" };
    let url = format!("https://huggingface.co/api/{plural}/{repo_id}/tree/main?recursive=true");
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF tree fetch failed: {e}")))?;

    if !resp.status().is_success() {
        // 404 on a fresh repo → no remote files, can't skip anything.
        return Ok(std::collections::HashMap::new());
    }

    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF tree JSON: {e}")))?;
    let arr = match body.as_array() {
        Some(a) => a,
        None => return Ok(std::collections::HashMap::new()),
    };

    let mut out = std::collections::HashMap::new();
    for entry in arr {
        if entry.get("type").and_then(|v| v.as_str()) != Some("file") {
            continue;
        }
        let path = match entry.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => continue,
        };
        if let Some(lfs_oid) = entry
            .get("lfs")
            .and_then(|v| v.get("oid"))
            .and_then(|v| v.as_str())
        {
            out.insert(path.to_string(), lfs_oid.to_string());
        }
    }
    Ok(out)
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

// ═══════════════════════════════════════════════════════════════
// HuggingFace HTTP API helpers
// ═══════════════════════════════════════════════════════════════

fn get_hf_token() -> Result<String, VindexError> {
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
    let token_path = PathBuf::from(&home).join(".cache").join("huggingface").join("token");
    if token_path.exists() {
        let token = std::fs::read_to_string(&token_path)?;
        return Ok(token.trim().to_string());
    }

    Err(VindexError::Parse(
        "HuggingFace token not found. Set HF_TOKEN or run `huggingface-cli login`.".into()
    ))
}

fn create_hf_repo(repo_id: &str, token: &str, repo_type: &str) -> Result<(), VindexError> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post("https://huggingface.co/api/repos/create")
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({
            "name": repo_id.split('/').next_back().unwrap_or(repo_id),
            "type": repo_type,
            "private": false,
        }))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF API error: {e}")))?;

    // 409 = already exists, that's fine
    if resp.status().is_success() || resp.status().as_u16() == 409 {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!("HF repo create failed ({status}): {body}")))
    }
}

/// Counting `Read` adapter — increments a shared atomic on every read so
/// a poll thread can report upload progress without per-chunk syscalls.
struct CountingReader<R: std::io::Read> {
    inner: R,
    counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl<R: std::io::Read> std::io::Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.counter
            .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(n)
    }
}

/// Upload a single file to a HuggingFace dataset repo via the real HF
/// protocol:
///
///   1. **Preupload** — `POST /api/datasets/{repo}/preupload/main` with a
///      base64 sample of the first 512 bytes. HF decides `lfs` vs `regular`
///      based on size + `.gitattributes`.
///   2. **LFS batch** (LFS path only) — `POST {repo}.git/info/lfs/objects/batch`
///      returns a signed upload URL or tells us the file is already there.
///   3. **Streaming PUT** to the signed URL, ticking `on_file_progress` as
///      bytes flow. `CountingReader` + worker thread keeps the main thread
///      free to poll.
///   4. **Verify** — `POST {verify.href}` with `{oid, size}`.
///   5. **Commit** — `POST /api/datasets/{repo}/commit/main` as NDJSON with
///      a `lfsFile` (LFS) or `file` (regular, base64-inline) operation.
///
/// The old single-PUT "upload endpoint" this replaced was fictional — HF
/// never exposed `PUT /api/datasets/{repo}/upload/main/{file}`. Requests
/// to it 404 after the first few megabytes of body, which was the bug
/// that triggered this rewrite.
fn upload_file_to_hf(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
    callbacks: &mut dyn PublishCallbacks,
    repo_type: &str,
) -> Result<(), VindexError> {
    let size = std::fs::metadata(local_path)?.len();
    let sha256 = crate::format::checksums::sha256_file(local_path)?;

    let decision = preupload_decide(repo_id, token, remote_filename, local_path, size, repo_type)?;

    if decision.should_ignore {
        // HF's preupload told us the server would ignore this path
        // (matches `.gitignore` / similar). Skip silently.
        return Ok(());
    }

    match decision.mode.as_str() {
        "lfs" => upload_lfs(repo_id, token, local_path, remote_filename, size, &sha256, callbacks, repo_type),
        "regular" => upload_regular(repo_id, token, local_path, remote_filename, size, callbacks, repo_type),
        other => Err(VindexError::Parse(format!(
            "HF preupload returned unknown mode `{other}` for {remote_filename}"
        ))),
    }
}

struct PreuploadDecision {
    mode: String,
    should_ignore: bool,
}

/// Call `POST /api/datasets/{repo}/preupload/main` for a single file and
/// return whether HF wants it uploaded via LFS or inlined in a regular
/// commit. HF requires a base64 sample of the first ~512 bytes so it
/// can sniff the file's format (text vs binary, etc.).
fn preupload_decide(
    repo_id: &str,
    token: &str,
    remote_filename: &str,
    local_path: &Path,
    size: u64,
    repo_type: &str,
) -> Result<PreuploadDecision, VindexError> {
    use base64::Engine;
    use std::io::Read;

    // Read up to 512 bytes for the format-sniff sample. HF accepts a
    // smaller sample for small files without complaint.
    let mut sample_buf = vec![0u8; 512.min(size as usize)];
    if !sample_buf.is_empty() {
        let mut file = std::fs::File::open(local_path)?;
        file.read_exact(&mut sample_buf)?;
    }
    let sample_b64 = base64::prelude::BASE64_STANDARD.encode(&sample_buf);

    let plural = if repo_type == "dataset" { "datasets" } else { "models" };
    let url = format!("https://huggingface.co/api/{plural}/{repo_id}/preupload/main");
    let body = serde_json::json!({
        "files": [{
            "path":   remote_filename,
            "sample": sample_b64,
            "size":   size,
        }],
    });
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("preupload failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "preupload ({status}) for {remote_filename}: {body}"
        )));
    }
    let json: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("preupload JSON: {e}")))?;
    let files = json
        .get("files")
        .and_then(|v| v.as_array())
        .ok_or_else(|| VindexError::Parse("preupload response missing `files`".into()))?;
    let entry = files
        .first()
        .ok_or_else(|| VindexError::Parse("preupload response files[] empty".into()))?;
    let mode = entry
        .get("uploadMode")
        .and_then(|v| v.as_str())
        .unwrap_or("lfs")
        .to_string();
    let should_ignore = entry
        .get("shouldIgnore")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Ok(PreuploadDecision { mode, should_ignore })
}

/// LFS-mode upload: batch → PUT to signed URL → verify → commit pointer.
fn upload_lfs(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
    size: u64,
    sha256: &str,
    callbacks: &mut dyn PublishCallbacks,
    repo_type: &str,
) -> Result<(), VindexError> {
    let batch = lfs_batch_upload(repo_id, token, sha256, size, repo_type)?;

    // If the response has no upload action, the object is already present
    // on the LFS server — skip to verify (if present) + commit.
    if let Some(ref upload) = batch.upload {
        stream_put_with_progress(
            &upload.href,
            &upload.header,
            local_path,
            size,
            remote_filename,
            callbacks,
        )?;
    } else {
        // Still tick the bar to 100% so the UX matches the upload path.
        callbacks.on_file_progress(remote_filename, size, size);
    }

    if let Some(ref verify) = batch.verify {
        lfs_verify(&verify.href, &verify.header, token, sha256, size)?;
    }

    commit_lfs_file(repo_id, token, remote_filename, sha256, size, repo_type)
}

/// Small-file path: commit directly with the content inlined as base64
/// in the NDJSON commit body. HF's preupload flags tiny text files for
/// this path.
fn upload_regular(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
    size: u64,
    callbacks: &mut dyn PublishCallbacks,
    repo_type: &str,
) -> Result<(), VindexError> {
    use base64::Engine;
    let data = std::fs::read(local_path)?;
    // Fire start+end of the progress bar even though we don't stream —
    // keeps the UX consistent across file sizes.
    callbacks.on_file_progress(remote_filename, 0, size);
    let encoded = base64::prelude::BASE64_STANDARD.encode(&data);

    let plural = if repo_type == "dataset" { "datasets" } else { "models" };
    let url = format!("https://huggingface.co/api/{plural}/{repo_id}/commit/main");
    let mut ndjson = String::new();
    ndjson.push_str(&serde_json::to_string(&serde_json::json!({
        "key": "header",
        "value": {
            "summary": format!("Upload {remote_filename}"),
        },
    })).unwrap());
    ndjson.push('\n');
    ndjson.push_str(&serde_json::to_string(&serde_json::json!({
        "key": "file",
        "value": {
            "path":     remote_filename,
            "encoding": "base64",
            "content":  encoded,
        },
    })).unwrap());
    ndjson.push('\n');

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .map_err(|e| VindexError::Parse(format!("commit (regular) failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "commit (regular) {remote_filename} ({status}): {body}"
        )));
    }
    callbacks.on_file_progress(remote_filename, size, size);
    Ok(())
}

#[derive(Debug)]
struct LfsAction {
    href: String,
    header: std::collections::HashMap<String, String>,
}

#[derive(Debug)]
struct LfsBatchResponse {
    upload: Option<LfsAction>,
    verify: Option<LfsAction>,
}

/// POST to the LFS batch endpoint asking for an upload URL for one
/// object. Returns the upload + verify actions (either or both may be
/// absent — an absent `upload` means the object is already stored).
fn lfs_batch_upload(
    repo_id: &str,
    token: &str,
    sha256: &str,
    size: u64,
    repo_type: &str,
) -> Result<LfsBatchResponse, VindexError> {
    let url = format!("{}.git/info/lfs/objects/batch", hf_repo_url(repo_type, repo_id));
    let body = serde_json::json!({
        "operation":  "upload",
        "transfers":  ["basic"],
        "hash_algo":  "sha256",
        "objects":    [{"oid": sha256, "size": size}],
    });
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", "application/vnd.git-lfs+json")
        .header("Content-Type", "application/vnd.git-lfs+json")
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("LFS batch failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "LFS batch ({status}): {body}"
        )));
    }
    let json: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("LFS batch JSON: {e}")))?;
    let objects = json
        .get("objects")
        .and_then(|v| v.as_array())
        .ok_or_else(|| VindexError::Parse("LFS batch response missing `objects`".into()))?;
    let obj = objects
        .first()
        .ok_or_else(|| VindexError::Parse("LFS batch objects[] empty".into()))?;

    // Per-object error surfaced in-line rather than as an HTTP status.
    if let Some(err) = obj.get("error") {
        return Err(VindexError::Parse(format!(
            "LFS batch object error: {err}"
        )));
    }

    let actions = obj.get("actions");
    let parse_action = |key: &str| -> Option<LfsAction> {
        let a = actions?.get(key)?;
        let href = a.get("href").and_then(|v| v.as_str())?.to_string();
        let header: std::collections::HashMap<String, String> = a
            .get("header")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        Some(LfsAction { href, header })
    };
    Ok(LfsBatchResponse {
        upload: parse_action("upload"),
        verify: parse_action("verify"),
    })
}

/// PUT the file contents to the signed LFS URL, streaming through a
/// `CountingReader` so the worker thread can report progress.
fn stream_put_with_progress(
    href: &str,
    extra_headers: &std::collections::HashMap<String, String>,
    local_path: &Path,
    size: u64,
    remote_filename: &str,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<(), VindexError> {
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::TryRecvError;
    use std::time::Duration;

    let file = std::fs::File::open(local_path)?;
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let reader = CountingReader {
        inner: file,
        counter: counter.clone(),
    };
    let body = reqwest::blocking::Body::sized(reader, size);

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(3600))
        .build()
        .map_err(|e| VindexError::Parse(format!("HTTP client error: {e}")))?;

    // Build the request on the worker thread (reqwest's Body needs to
    // travel there). Include any signature headers the LFS server
    // requested — on AWS-backed buckets these carry the AWS sigv4 bits.
    let href_owned = href.to_string();
    let headers_owned: Vec<(String, String)> = extra_headers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let mut req = client.put(&href_owned);
        for (k, v) in &headers_owned {
            req = req.header(k.as_str(), v.as_str());
        }
        let result = req.body(body).send();
        let _ = tx.send(result);
    });

    loop {
        match rx.try_recv() {
            Ok(resp) => {
                let _ = handle.join();
                let resp = resp
                    .map_err(|e| VindexError::Parse(format!("LFS PUT failed: {e}")))?;
                if resp.status().is_success() {
                    callbacks.on_file_progress(remote_filename, size, size);
                    return Ok(());
                }
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                return Err(VindexError::Parse(format!(
                    "LFS PUT {remote_filename} ({status}): {body}"
                )));
            }
            Err(TryRecvError::Empty) => {
                let sent = counter.load(Ordering::Relaxed);
                callbacks.on_file_progress(remote_filename, sent, size);
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(TryRecvError::Disconnected) => {
                let _ = handle.join();
                return Err(VindexError::Parse(
                    "upload worker terminated unexpectedly".into(),
                ));
            }
        }
    }
}

/// POST `{oid, size}` to the verify URL the LFS batch returned. HF uses
/// this to confirm the object made it to storage intact before the
/// commit references it.
fn lfs_verify(
    href: &str,
    extra_headers: &std::collections::HashMap<String, String>,
    token: &str,
    sha256: &str,
    size: u64,
) -> Result<(), VindexError> {
    let body = serde_json::json!({"oid": sha256, "size": size});
    let client = reqwest::blocking::Client::new();
    let mut req = client
        .post(href)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", "application/vnd.git-lfs+json")
        .header("Content-Type", "application/vnd.git-lfs+json");
    for (k, v) in extra_headers {
        req = req.header(k.as_str(), v.as_str());
    }
    let resp = req
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("LFS verify failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!("LFS verify ({status}): {body}")));
    }
    Ok(())
}

/// Commit a single LFS pointer into the repo via NDJSON. HF's commit
/// API is one request per change set; we commit per file for simplicity
/// (batching every file into one commit is a future optimisation).
fn commit_lfs_file(
    repo_id: &str,
    token: &str,
    remote_filename: &str,
    sha256: &str,
    size: u64,
    repo_type: &str,
) -> Result<(), VindexError> {
    let plural = if repo_type == "dataset" { "datasets" } else { "models" };
    let url = format!("https://huggingface.co/api/{plural}/{repo_id}/commit/main");
    let mut ndjson = String::new();
    ndjson.push_str(&serde_json::to_string(&serde_json::json!({
        "key": "header",
        "value": {"summary": format!("Upload {remote_filename}")},
    })).unwrap());
    ndjson.push('\n');
    ndjson.push_str(&serde_json::to_string(&serde_json::json!({
        "key": "lfsFile",
        "value": {
            "path": remote_filename,
            "algo": "sha256",
            "oid":  sha256,
            "size": size,
        },
    })).unwrap());
    ndjson.push('\n');

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .map_err(|e| VindexError::Parse(format!("commit (LFS) failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "commit (LFS) {remote_filename} ({status}): {body}"
        )));
    }
    Ok(())
}

/// Check if a path is an hf:// reference.
pub fn is_hf_path(path: &str) -> bool {
    path.starts_with("hf://")
}

// ═══════════════════════════════════════════════════════════════
// Collections
// ═══════════════════════════════════════════════════════════════

/// One repo in a collection.
#[derive(Clone, Debug)]
pub struct CollectionItem {
    /// Repo id (`owner/name`). Full form including namespace.
    pub repo_id: String,
    /// `"model"` (vindex repos, default) or `"dataset"`.
    pub repo_type: String,
    /// Optional short note rendered on the collection card.
    pub note: Option<String>,
}

/// Ensure a collection titled `title` exists in `namespace`, then add
/// every item to it. Idempotent: re-runs reuse the slug (matched by
/// case-insensitive title) and treat HTTP 409 on add-item as success.
/// Returns the collection URL on success.
pub fn ensure_collection(
    namespace: &str,
    title: &str,
    description: Option<&str>,
    items: &[CollectionItem],
) -> Result<String, VindexError> {
    let token = get_hf_token()?;
    let slug = match find_collection_slug(namespace, title, &token)? {
        Some(existing) => existing,
        None => create_collection(namespace, title, description, &token)?,
    };
    for item in items {
        add_collection_item(&slug, item, &token)?;
    }
    Ok(format!("https://huggingface.co/collections/{slug}"))
}

fn find_collection_slug(
    namespace: &str,
    title: &str,
    token: &str,
) -> Result<Option<String>, VindexError> {
    let client = reqwest::blocking::Client::new();
    let url = format!("https://huggingface.co/api/users/{namespace}/collections?limit=100");
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collections list failed: {e}")))?;
    if !resp.status().is_success() {
        if resp.status().as_u16() == 404 {
            return Ok(None);
        }
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "HF collections list ({status}): {body}"
        )));
    }
    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF collections JSON: {e}")))?;
    let arr = match body.as_array() {
        Some(a) => a,
        None => return Ok(None),
    };
    let target = title.to_ascii_lowercase();
    for entry in arr {
        let entry_title = entry.get("title").and_then(|v| v.as_str()).unwrap_or("");
        if entry_title.to_ascii_lowercase() == target {
            if let Some(slug) = entry.get("slug").and_then(|v| v.as_str()) {
                return Ok(Some(slug.to_string()));
            }
        }
    }
    Ok(None)
}

fn create_collection(
    namespace: &str,
    title: &str,
    description: Option<&str>,
    token: &str,
) -> Result<String, VindexError> {
    let client = reqwest::blocking::Client::new();
    let mut body = serde_json::json!({
        "title": title,
        "namespace": namespace,
        "private": false,
    });
    if let Some(desc) = description {
        body["description"] = serde_json::Value::String(desc.to_string());
    }
    let resp = client
        .post("https://huggingface.co/api/collections")
        .header("Authorization", format!("Bearer {token}"))
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection create failed: {e}")))?;

    let status = resp.status();
    let body_text = resp.text().unwrap_or_default();

    // Happy path — new collection created.
    if status.is_success() {
        let json: serde_json::Value = serde_json::from_str(&body_text)
            .map_err(|e| VindexError::Parse(format!("HF collection JSON: {e}")))?;
        let slug = json
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| VindexError::Parse("HF collection response missing slug".into()))?;
        return Ok(slug.to_string());
    }

    // 409 Conflict — collection already exists. HF returns the existing
    // slug in the error body. We hit this when `find_collection_slug`
    // failed to find it (e.g. auth scope / list pagination issues) but
    // the collection does exist. Short-circuiting here is the robust
    // path regardless of why find missed it.
    if status.as_u16() == 409 {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body_text) {
            if let Some(slug) = json.get("slug").and_then(|v| v.as_str()) {
                return Ok(slug.to_string());
            }
        }
    }

    Err(VindexError::Parse(format!(
        "HF collection create ({status}): {body_text}"
    )))
}

fn add_collection_item(
    slug: &str,
    item: &CollectionItem,
    token: &str,
) -> Result<(), VindexError> {
    let client = reqwest::blocking::Client::new();
    // HF's collection API uses `/items` (plural) for POST-to-append.
    // The singular form is only valid as `PATCH/DELETE
    // /api/collections/{slug}/item/{item_id}` for editing an existing
    // entry. Got caught by this on the first real publish — the add
    // failed with 404 after the four repos had already uploaded fine.
    let url = format!("https://huggingface.co/api/collections/{slug}/items");
    let mut body = serde_json::json!({
        "item": {
            "type": item.repo_type,
            "id": item.repo_id,
        },
    });
    if let Some(note) = &item.note {
        body["note"] = serde_json::Value::String(note.clone());
    }
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection add-item failed: {e}")))?;
    if resp.status().is_success() || resp.status().as_u16() == 409 {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!(
            "HF collection add-item ({status}): {body}"
        )))
    }
}

/// Cheap HEAD probe — returns `Ok(true)` if the dataset repo exists and
/// is readable, `Ok(false)` on 404, `Err` on other failures. Auth is
/// optional; pass-through when available (lets callers see private
/// repos they own).
pub fn dataset_repo_exists(repo_id: &str) -> Result<bool, VindexError> {
    repo_exists(repo_id, "model")
}

pub fn repo_exists(repo_id: &str, repo_type: &str) -> Result<bool, VindexError> {
    let token = get_hf_token().ok();
    let plural = if repo_type == "dataset" { "datasets" } else { "models" };
    let url = format!("https://huggingface.co/api/{plural}/{repo_id}");
    let client = reqwest::blocking::Client::new();
    let mut req = client.head(&url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req
        .send()
        .map_err(|e| VindexError::Parse(format!("HF HEAD failed: {e}")))?;
    if resp.status().is_success() {
        Ok(true)
    } else if resp.status().as_u16() == 404 {
        Ok(false)
    } else {
        Err(VindexError::Parse(format!(
            "HF HEAD {repo_id}: {}",
            resp.status()
        )))
    }
}

/// Fetch a collection by slug (or full collection URL) and return its
/// items as `(type, id)` pairs — typically `("dataset", "owner/name")`.
pub fn fetch_collection_items(
    slug_or_url: &str,
) -> Result<Vec<(String, String)>, VindexError> {
    let slug = slug_or_url
        .trim_start_matches("https://huggingface.co/collections/")
        .trim_start_matches("http://huggingface.co/collections/")
        .trim_start_matches("hf://collections/")
        .trim_start_matches('/');
    let token = get_hf_token().ok();
    let url = format!("https://huggingface.co/api/collections/{slug}");
    let client = reqwest::blocking::Client::new();
    let mut req = client.get(&url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection fetch failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "HF collection fetch ({status}): {body}"
        )));
    }
    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF collection JSON: {e}")))?;
    let items = body
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| VindexError::Parse("collection response missing items".into()))?;
    let mut out = Vec::new();
    for item in items {
        let kind = match item.get("type").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let id = match item.get("id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        out.push((kind, id));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_hf_path() {
        assert!(is_hf_path("hf://chrishayuk/gemma-3-4b-it-vindex"));
        assert!(is_hf_path("hf://user/repo@v1.0"));
        assert!(!is_hf_path("./local.vindex"));
        assert!(!is_hf_path("/absolute/path"));
    }

    #[test]
    fn test_parse_hf_path() {
        let path = "hf://chrishayuk/gemma-3-4b-it-vindex@v2.0";
        let stripped = path.strip_prefix("hf://").unwrap();
        let (repo, rev) = stripped.split_once('@').unwrap();
        assert_eq!(repo, "chrishayuk/gemma-3-4b-it-vindex");
        assert_eq!(rev, "v2.0");
    }
}
