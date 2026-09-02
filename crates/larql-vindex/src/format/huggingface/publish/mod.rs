//! HuggingFace publish path — repo creation + per-file upload + LFS
//! pointer/upload protocol + callback hooks.
//!
//! Carved out of the monolithic `huggingface.rs` in the 2026-04-25
//! reorg, then split again 2026-05-09, then again 2026-08-23 (each
//! implementation file's tests moved to a sibling `*_tests.rs`, per the
//! `format/generation.rs`+`format/generation_tests.rs` pattern) into:
//!   - `mod.rs`          — public API (`publish_vindex*`, `PublishOptions`,
//!                         `PublishResult`, `PublishCallbacks`), URL
//!                         helpers, `enumerate_publishable_files`,
//!                         `get_hf_token`
//!   - `tests.rs`        — tests for `mod.rs`
//!   - `remote.rs`       — `fetch_remote_lfs_oids`, `create_hf_repo`,
//!                         `fetch_repo_head_sha`
//!   - `remote_tests.rs` — tests for `remote.rs`
//!   - `upload.rs`       — `upload_file_to_hf` + preupload + `upload_regular`
//!   - `lfs.rs`          — LFS protocol (batch / verify / commit) +
//!                         streaming PUT + `CountingReader`

mod lfs;
pub(super) mod protocol;
mod remote;
#[cfg(test)]
mod remote_tests;
mod upload;

use std::path::{Path, PathBuf};

use crate::error::VindexError;
use crate::format::filenames::*;

use protocol::{hf_base, repo_type_plural, REPO_TYPE_DATASET, REPO_TYPE_MODEL};
use remote::{
    create_hf_repo, delete_remote_files, fetch_remote_file_paths, fetch_remote_lfs_oids,
    fetch_repo_head_sha, is_prune_exempt, update_repo_visibility,
};
use upload::upload_file_to_hf;

/// What a successful publish produced: where it's browsable, and the
/// exact commit its bytes landed at.
///
/// Publishing is not one atomic commit — each file lands as its own
/// commit against `main` (`upload_file_to_hf`'s per-file protocol) — so
/// `revision` is not "the commit this publish made" in a strict sense;
/// it's `main`'s HEAD immediately after the last file landed, fetched
/// with one extra API call once uploads finish
/// (`docs/vindex3-registry-publishing-design.md` §1/§6). That is the
/// only meaningful "pinned revision" a multi-commit upload can produce,
/// and it is what an official registry entry's `RegistryArtifactRef::revision`
/// needs — a caller no longer has to look this up and retype it by hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishResult {
    pub url: String,
    pub revision: String,
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
    /// Create the repo private. Vindex Factory (docs/vindex-factory.md
    /// §7/§8.3) publishes private, verifies the published bytes, and
    /// only then flips public via [`set_repo_visibility`] — "nothing
    /// goes public unverified". Default `false` preserves every
    /// existing caller's behaviour (`larql publish`/`larql hf publish`
    /// have always created public repos).
    pub private: bool,
    /// Delete remote files that no longer exist in the source vindex.
    ///
    /// Publishing only ever *added* files until 2026-08-07. When the Q6_K
    /// weight files were renamed (`interleaved_q4k.bin` →
    /// `interleaved_kquant.bin`) a republish left both generations in the
    /// repo, and `pick_bin` chose between them by name — so a repo could
    /// silently serve a stale weight file that the source vindex no longer
    /// contained. Mirroring the source is the intended meaning of "publish
    /// this vindex", so this defaults to `true`; see [`is_prune_exempt`]
    /// for what is never removed.
    pub prune_remote: bool,
}

impl Default for PublishOptions {
    fn default() -> Self {
        Self {
            skip_unchanged: false,
            repo_type: REPO_TYPE_MODEL.into(),
            private: false,
            prune_remote: true,
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
) -> Result<PublishResult, VindexError> {
    publish_vindex_with_opts(vindex_dir, repo_id, &PublishOptions::default(), callbacks)
}

/// Flip an already-published repo's visibility. The RELEASE step of
/// docs/vindex-factory.md §7: a build publishes PRIVATE
/// ([`PublishOptions::private`]), verifies the published bytes
/// (VERIFY-B, §8.2), and only then calls this to go public — "nothing
/// goes public unverified" (§8).
///
/// `repo_type` is `"model"` or `"dataset"`, matching
/// [`PublishOptions::repo_type`]. Requires `HF_TOKEN` or
/// `~/.huggingface/token`, same as publishing.
pub fn set_repo_visibility(
    repo_id: &str,
    repo_type: &str,
    private: bool,
) -> Result<(), VindexError> {
    let token = get_hf_token()?;
    update_repo_visibility(repo_id, &token, repo_type, private)
}

/// Upload a vindex directory with explicit options. See [`PublishOptions`].
pub fn publish_vindex_with_opts(
    vindex_dir: &Path,
    repo_id: &str,
    opts: &PublishOptions,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<PublishResult, VindexError> {
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
    create_hf_repo(repo_id, &token, repo_type, opts.private)?;

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

    // Mirror the source: anything on the Hub that the vindex no longer
    // contains is removed. Runs *after* the uploads so a failed upload
    // never leaves the repo with neither the old file nor the new one.
    if opts.prune_remote {
        let local: std::collections::HashSet<&str> =
            files.iter().map(|(_, name)| name.as_str()).collect();
        // A listing failure is not fatal — the upload already succeeded and
        // stale files are a tidiness problem, not a correctness one now that
        // the current generation is present.
        if let Ok(remote) = fetch_remote_file_paths(repo_id, &token, repo_type) {
            let stale: Vec<String> = remote
                .into_iter()
                .filter(|p| !local.contains(p.as_str()) && !is_prune_exempt(p))
                .collect();
            if !stale.is_empty() {
                delete_remote_files(repo_id, &token, repo_type, &stale)?;
                for p in &stale {
                    callbacks.on_file_deleted(p);
                }
            }
        }
    }

    // The pinned-revision step (design doc §1/§6): publishing is N
    // per-file commits, not one atomic commit, so there is no commit
    // sha to have captured along the way — the only meaningful
    // "revision this publish produced" is `main`'s HEAD now that every
    // file has landed. Fetched, not left to a caller to look up and
    // retype: a hand-typed pin is exactly the provenance bug this
    // exists to prevent.
    let revision = fetch_repo_head_sha(repo_id, &token, repo_type)?;
    let url = hf_repo_url(repo_type, repo_id);
    callbacks.on_complete(&url);
    Ok(PublishResult { url, revision })
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
    /// Fired before sleeping to retry a transient upload failure —
    /// `attempt` is the one about to be made, `reason` the status or
    /// transport error that triggered it. Default no-op; surface it, because
    /// a silent multi-minute backoff looks identical to a hung upload.
    fn on_retry(
        &mut self,
        _filename: &str,
        _attempt: u32,
        _max_attempts: u32,
        _reason: &str,
        _wait: std::time::Duration,
    ) {
    }
    /// Fired for each remote file deleted because it no longer exists in
    /// the source vindex. Default no-op.
    fn on_file_deleted(&mut self, _filename: &str) {}
    fn on_complete(&mut self, _url: &str) {}
}

pub struct SilentPublishCallbacks;
impl PublishCallbacks for SilentPublishCallbacks {}

pub(in crate::format::huggingface) fn get_hf_token() -> Result<String, VindexError> {
    super::token::resolve().ok_or_else(|| {
        VindexError::Parse(
            "HuggingFace token not found. Set HF_TOKEN or run `huggingface-cli login`.".into(),
        )
    })
}

#[cfg(test)]
mod tests;
