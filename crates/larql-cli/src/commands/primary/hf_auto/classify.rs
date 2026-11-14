//! What an `hf://` repo is: a published vindex, a raw checkpoint, or
//! neither — decided from which files the repo carries.
//!
//! The question is asked through [`RepoProbe`] so the decision is
//! testable without a network; the production probe is a range client
//! against the hub (`live_runtime.rs`).

use std::path::Path;

use larql_vindex::format::filenames::INDEX_JSON;

use super::super::cache::{HF_REPO_PREFIXES, HUB_REPO_SEP, HUB_SNAPSHOTS_DIR};

/// The file every Hugging Face model checkpoint carries.
pub const CHECKPOINT_CONFIG: &str = "config.json";

type BoxErr = Box<dyn std::error::Error>;

/// Questions about one repo at one revision, answered by the hub.
pub trait RepoProbe {
    /// Whether `file` exists in `repo` at `revision`.
    fn has_file(&self, repo: &str, revision: &str, file: &str) -> Result<bool, BoxErr>;

    /// The immutable commit `revision` resolves to, when the hub says.
    fn commit(&self, repo: &str, revision: &str) -> Result<Option<String>, BoxErr>;
}

/// What a repo is, for the purpose of running it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepoKind {
    /// Carries `index.json`: a published vindex, fetched as one.
    Vindex,
    /// Carries `config.json` and no `index.json`: a checkpoint to encode.
    Checkpoint,
    /// Carries neither, or the hub would not say (offline, gated, a
    /// dataset repo the model endpoint does not serve). The vindex path
    /// keeps the last word, so its existing behaviour and errors stand.
    Unrecognised,
}

/// Classify `repo@revision`. `index.json` wins: a repo that carries one
/// is a vindex whatever else it carries, exactly as before this existed.
/// A probe failure is [`RepoKind::Unrecognised`], never an error — the
/// vindex path that follows reports it in its own terms.
pub fn classify(probe: &dyn RepoProbe, repo: &str, revision: &str) -> RepoKind {
    match probe.has_file(repo, revision, INDEX_JSON) {
        Ok(true) => return RepoKind::Vindex,
        Ok(false) => {}
        Err(_) => return RepoKind::Unrecognised,
    }
    match probe.has_file(repo, revision, CHECKPOINT_CONFIG) {
        Ok(true) => RepoKind::Checkpoint,
        Ok(false) | Err(_) => RepoKind::Unrecognised,
    }
}

/// Whether the HF hub cache already holds a vindex snapshot for `repo` —
/// in which case the vindex path answers it from disk, as it always has,
/// and no probe is sent.
pub fn hub_has_vindex(hub: &Path, repo: &str) -> bool {
    let Some((owner, name)) = repo.split_once('/') else {
        return false;
    };
    HF_REPO_PREFIXES.iter().any(|prefix| {
        let snapshots = hub
            .join(format!("{prefix}{owner}{HUB_REPO_SEP}{name}"))
            .join(HUB_SNAPSHOTS_DIR);
        std::fs::read_dir(snapshots).is_ok_and(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|e| e.path().join(INDEX_JSON).is_file())
        })
    })
}
