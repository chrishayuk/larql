//! `larql pull` of an encoded checkpoint: the explicit refresh.
//!
//! An encode is cached at the commit its revision first resolved to, and
//! `larql run` answers from that cache with no network — it never asks
//! whether `main` has moved. `pull` is where the user asks: the revision is
//! re-resolved against the hub and
//!
//! - **unchanged** → nothing is encoded; the entry is already up to date;
//! - **moved** → the new commit is encoded through the same atomic store,
//!   `refs/<revision>` is repointed, and only then is the superseded
//!   snapshot pruned — unless another recorded revision still names it.
//!
//! Pruning keeps one snapshot per tracked revision, so a refresh does not
//! double the cache; the old commit stays reachable as
//! `hf://org/name@<old commit>`, which simply encodes again.
//!
//! A repo with nothing encoded takes [`HfAuto::resolve`] unchanged, so a
//! published vindex is still (re-)downloaded and a fresh checkpoint still
//! encoded on first pull.

use std::io::Write;

use larql_vindex::format::huggingface::range::{parse_spec, DEFAULT_REVISION};

use super::super::cache::{scan_hub_layout_at, CacheSource};
use super::resolve::{is_full_commit, shorthand, HfAuto, Resolution, VindexFetch};
use super::store::EncodedStore;
use super::HF_SCHEME;

type BoxErr = Box<dyn std::error::Error>;

impl HfAuto<'_> {
    /// What `larql pull hf://org/name[@revision]` does: refresh an encoded
    /// entry, or resolve exactly as `run` would when nothing is encoded.
    pub fn pull(
        &self,
        spec: &str,
        fetch_vindex: &mut VindexFetch<'_>,
        status: &mut dyn Write,
    ) -> Result<Resolution, BoxErr> {
        let parsed = parse_spec(spec)?;
        if !self.store.holds(&parsed.repo) {
            return self.resolve(spec, fetch_vindex, status);
        }
        self.refresh(&parsed.repo, &parsed.revision, status)
    }

    /// Re-resolve `repo@revision` and encode it if it moved.
    fn refresh(
        &self,
        repo: &str,
        revision: &str,
        status: &mut dyn Write,
    ) -> Result<Resolution, BoxErr> {
        let name = format!("{HF_SCHEME}{repo}@{revision}");
        // A full commit id is immutable: nothing to ask the hub.
        if is_full_commit(revision) {
            if let Some(path) = self.store.complete(repo, revision) {
                writeln!(status, "{name} is already up to date (commit {revision})")?;
                return Ok(Resolution::Encoded { path, fresh: false });
            }
        }
        let previous = self.store.read_ref(repo, revision);
        let commit = self.pin(repo, revision).map_err(|e| -> BoxErr {
            let cached = previous
                .as_deref()
                .map(|c| format!("; the cached commit {c} still runs offline"))
                .unwrap_or_default();
            format!("cannot refresh {name}: {e}{cached}").into()
        })?;

        if let Some(path) = self.store.complete(repo, &commit) {
            self.store.record_ref(repo, revision, &commit)?;
            writeln!(status, "{name} is already up to date (commit {commit})")?;
            return Ok(Resolution::Encoded { path, fresh: false });
        }

        match &previous {
            Some(old) => writeln!(
                status,
                "{name} moved: {old} → {commit}; encoding the new commit…"
            )?,
            None => writeln!(status, "{name} is not encoded yet; encoding it…")?,
        }
        let resolution = self.encode_commit(repo, revision, &commit, status)?;

        // Only after the new snapshot is published and the ref repointed:
        // an interrupted refresh leaves the old entry runnable.
        let superseded = previous.filter(|old| *old != commit);
        if let Some(old) = superseded.filter(|old| self.store.complete(repo, old).is_some()) {
            match self.store.prune(repo, &old)? {
                Some(_) => writeln!(status, "  pruned the superseded snapshot {old}")?,
                None => writeln!(
                    status,
                    "  kept snapshot {old}: another revision still points at it"
                )?,
            }
        }
        Ok(resolution)
    }
}

/// The `hf://` spec `larql pull <name>` refreshes when `name` is the
/// shorthand of an encoded entry, or `None` when no encoded entry answers
/// to it (the caller then treats `name` as it always has).
///
/// The revision is the one the entry tracks: `main` when it tracks `main`
/// (spelled without `@`), its single recorded revision otherwise, or its
/// single snapshot's commit when it was encoded by commit. An entry
/// tracking several revisions is ambiguous and refused, naming them.
pub fn pull_spec_for_shorthand(store: &EncodedStore, name: &str) -> Result<Option<String>, BoxErr> {
    let entries = scan_hub_layout_at(store.root(), CacheSource::Encoded)?;
    let repos: Vec<&str> = entries
        .iter()
        .map(|e| e.repo.as_str())
        .filter(|repo| shorthand(repo) == name)
        .collect();
    let repo = match repos.as_slice() {
        [] => return Ok(None),
        [repo] => *repo,
        many => {
            return Err(format!(
                "`{name}` matches several encoded checkpoints ({}); pull one by \
                 {HF_SCHEME}owner/name",
                many.join(", ")
            )
            .into())
        }
    };
    let refs = store.refs(repo);
    if refs
        .iter()
        .any(|(revision, _)| revision == DEFAULT_REVISION)
    {
        return Ok(Some(format!("{HF_SCHEME}{repo}")));
    }
    let snapshots = store.snapshots(repo);
    let revision = match (refs.as_slice(), snapshots.as_slice()) {
        ([(revision, _)], _) => revision.clone(),
        ([], [commit]) => commit.clone(),
        _ => {
            let tracked: Vec<&str> = if refs.is_empty() {
                snapshots.iter().map(String::as_str).collect()
            } else {
                refs.iter().map(|(r, _)| r.as_str()).collect()
            };
            return Err(format!(
                "`{name}` ({repo}) tracks several revisions ({}); pull one by \
                 {HF_SCHEME}{repo}@<revision>",
                tracked.join(", ")
            )
            .into());
        }
    };
    Ok(Some(format!("{HF_SCHEME}{repo}@{revision}")))
}
