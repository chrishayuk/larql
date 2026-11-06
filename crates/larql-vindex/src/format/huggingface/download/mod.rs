//! HuggingFace download path — `hf://` resolution, snapshot cache
//! traversal, conditional ETag-based fetch.
//!
//! Carved out of the monolithic `huggingface.rs` in the 2026-04-25
//! reorg. See `super::mod.rs` for the module map.
//!
//! Sibling layout (round-6 split, 2026-05-10; tests split further
//! 2026-08-23 per the `format/generation.rs`+`format/generation_tests.rs`
//! pattern):
//! - `helpers`      — pure non-network utilities (etag/repo-filter/cache-path).
//! - `tests`        — hf_hub-plumbing tests (resolve/download/cache-walk).
//! - `tests_v3`     — VINDEX3 generation-aware completeness tests.
//! - `test_support` — mock/env-guard scaffolding shared by both test files.

mod helpers;

use std::path::PathBuf;

use crate::error::VindexError;
use crate::format::filenames::*;
use crate::format::generation::{detect_generation, ContainerGeneration};

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
///
/// # Why this only ever accepts the pinned revision's own snapshot dir
///
/// This used to also accept (a) any *other* revision's snapshot symlink
/// for the same filename, and (b) the bare blob path with no snapshot
/// symlink at all, on the reasoning that "the caller only needs a file
/// it can open." That reasoning is wrong for this caller specifically:
/// [`resolve_hf_vindex_with_progress`]'s V3 completeness loop discards
/// the returned `PathBuf` entirely (`fetch(...).ok_or_else(...)?` —
/// only the `Option`-ness is checked) and relies on every fetched file
/// actually landing under `vindex_dir` (the pinned revision's own
/// `snapshots/<revision>/` directory, established by `index.json`'s own
/// fetch) — that is what the VINDEX3 container loader opens files
/// relative to afterwards. A bare blob path, or a symlink living under
/// some *other* revision's snapshot dir, satisfies neither: the loop
/// reports success, but `vindex_dir` is left missing the file, and the
/// failure only surfaces later, opaquely, when the container is opened.
///
/// Concretely: `target.embedding.bin`'s blob was already resident
/// locally (deduped — identical BF16 embedding bytes from an earlier,
/// unrelated pull of a sibling container), so this function used to
/// report it "cached" via the removed fallback, the real
/// `download_with_progress` call that would have created the pinned
/// revision's own symlink never ran, and `larql serve` on the claimed
/// registry name failed with a bare `IO error: No such file or
/// directory` — nothing about that error named the missing symlink.
///
/// With no accepted fallback, an unpinned `revision: None` request
/// always misses here (there is no single directory a "None" revision
/// unambiguously names yet) and falls through to `download_with_progress`,
/// which resolves "current HEAD" and places every file consistently —
/// slightly less cache-optimized for the unpinned case, never silently
/// wrong.
fn cached_snapshot_file(
    kind: RepoKind,
    repo_id: &str,
    revision: Option<&str>,
    filename: &str,
) -> Option<(PathBuf, u64)> {
    let revision = revision?;
    let (etag, size) = head_etag_and_size(kind, repo_id, Some(revision), filename)?;
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

    let snap_file = repo_dir.join("snapshots").join(revision).join(filename);
    if snap_file.exists() {
        return Some((snap_file, size));
    }
    None
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
///
/// # Generation-aware since the vindex3-registry initiative's 2C rung
///
/// `index.json` is downloaded first regardless of generation — it is
/// the minimal control metadata every container declares itself with.
/// What happens next branches on what it says:
///
/// - **VINDEX2** — unchanged: [`vindex_core_files()`]'s fixed metadata
///   + big-tensor-file list, optional entries skipped silently (most
///   candidate files genuinely don't exist in most repos; that's the
///   list's whole design).
/// - **VINDEX3** — the repo's own file listing (`repo.info().siblings`)
///   *is* the required payload, downloaded in full. VINDEX3 has no
///   metadata/weight split the way VINDEX2 does — its payload IS its
///   structure (segments, representations, manifests, the M2 migration
///   rung's capability-snapshot side-channel) — so a hand-enumerated
///   "which `index.json` fields name a file" list would just be the
///   fixed-list bug this rung exists to fix, one layer removed: it
///   would silently miss whatever the format grows next, exactly like
///   the old list silently missed VINDEX3 entirely. A repo dedicated to
///   one vindex has no meaningful unrelated bulk to over-fetch (see
///   `docs/vindex3-registry-design.md` §10.5). Every listed file is
///   therefore required, not optional: a failed fetch is a hard error
///   naming the file, not a silently-skipped candidate.
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

        match detect_generation(&vindex_dir)? {
            ContainerGeneration::V3 => {
                let info = repo.info().map_err(|e| {
                    VindexError::Parse(format!("HF info failed for hf://{repo_id}: {e}"))
                })?;
                for sibling in &info.siblings {
                    if sibling.rfilename == INDEX_JSON {
                        continue;
                    }
                    fetch(&sibling.rfilename, &sibling.rfilename).ok_or_else(|| {
                        VindexError::Parse(format!(
                            "failed to download required VINDEX3 file '{}' from hf://{repo_id}",
                            sibling.rfilename
                        ))
                    })?;
                }
            }
            ContainerGeneration::V2 => {
                // The fixed core list, then whatever else the index requires
                // (`completeness::required_files`) — so a pull fetches every
                // file `larql list` will check for.
                let declared = crate::format::load::load_vindex_config_unchecked(&vindex_dir)
                    .map(|cfg| crate::format::completeness::required_files(&cfg))
                    .unwrap_or_default();
                let core = vindex_core_files();
                let extra = declared
                    .iter()
                    .map(String::as_str)
                    .filter(|f| !core.contains(f));
                for filename in core.iter().copied().chain(extra) {
                    if filename == INDEX_JSON {
                        continue;
                    }
                    // Optional files — ignore failures (missing from repo is fine).
                    let _ = fetch(filename, filename);
                }
            }
        }
        return Ok(vindex_dir);
    }

    Err(VindexError::Parse(format!(
        "failed to fetch index.json from hf://{repo_id}"
    )))
}

/// A `DownloadProgress` that reports nothing — for callers that need
/// [`resolve_hf_vindex_with_progress`]'s generation-aware completeness
/// but have no UI to drive (a background `serve`/server-side load, not
/// an interactive `pull`). Mirrors this crate's `SilentLoadCallbacks` /
/// `SilentPublishCallbacks` / `SilentBuildCallbacks` convention.
struct NoDownloadProgress;
impl DownloadProgress for NoDownloadProgress {
    fn init(&mut self, _size: usize, _filename: &str) {}
    fn update(&mut self, _size: usize) {}
    fn finish(&mut self) {}
}

/// [`resolve_hf_vindex_with_progress`] with no progress reporting.
///
/// Not [`resolve_hf_vindex`] — that function is deliberately
/// metadata-only for VINDEX2 (small files, cheap for `larql show`-style
/// callers). This one always fetches the complete, generation-aware
/// payload; use it wherever the caller actually needs a working
/// container, not just a peek at its metadata — the VINDEX3 registry's
/// `resolve_claimed` is the first such caller.
pub fn resolve_hf_vindex_complete(hf_path: &str) -> Result<PathBuf, VindexError> {
    resolve_hf_vindex_with_progress(hf_path, |_| NoDownloadProgress)
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
mod test_support;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_v3;
