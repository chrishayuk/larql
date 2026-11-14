//! The `hf://` decision: encoded cache, published vindex, or first-run
//! encode.
//!
//! Order, cheapest and most local first:
//!
//! 1. **Encoded cache** — `repo@revision` already encoded (the revision
//!    is a commit we hold, or a ref recorded when we encoded it). Answered
//!    from disk; no network.
//! 2. **HF hub cache holds a vindex for the repo** — the vindex path,
//!    unchanged; it answers from disk too.
//! 3. **Ask the hub** ([`classify`]): `index.json` → the vindex path,
//!    unchanged; `config.json` without `index.json` → encode; anything
//!    else (including a probe that failed) → the vindex path, so its
//!    behaviour and its errors stand exactly as before.
//!
//! An encode is pinned to the commit the revision resolves to, written
//! into a partial directory and renamed into place only on success (see
//! [`super::store`]).

use std::io::Write;
use std::path::{Path, PathBuf};

use larql_vindex::format::huggingface::range::parse_spec;

use super::classify::{classify, hub_has_vindex, RepoKind, RepoProbe};
use super::store::EncodedStore;
use super::HF_SCHEME;

type BoxErr = Box<dyn std::error::Error>;

/// Length of a full git commit id, the only revision spelling that is
/// already immutable without asking the hub.
const COMMIT_HEX_LEN: usize = 40;

/// Fetches a published vindex — `larql_vindex::resolve_hf_vindex`, or
/// `larql pull`'s progress-bar flavour of it.
pub type VindexFetch<'a> = dyn FnMut(&str) -> Result<PathBuf, BoxErr> + 'a;

/// Encodes one pinned `hf://org/name@<commit>` spec into a directory.
pub trait CheckpointEncoder {
    /// Plan, gate and encode `spec` into `out`, narrating to `status`.
    /// An inadmissible plan is an `Err` carrying the itemised refusal.
    fn encode(&self, spec: &str, out: &Path, status: &mut dyn Write) -> Result<(), BoxErr>;
}

/// Where an `hf://` spec ended up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// A published vindex, from the HF hub cache.
    Vindex(PathBuf),
    /// A checkpoint encoded to VINDEX3. `fresh` is true when this call
    /// did the encoding.
    Encoded { path: PathBuf, fresh: bool },
}

impl Resolution {
    pub fn into_path(self) -> PathBuf {
        match self {
            Self::Vindex(path) | Self::Encoded { path, .. } => path,
        }
    }
}

/// The resolver's collaborators, injected so every branch is testable
/// without a network or the user's caches.
pub struct HfAuto<'a> {
    pub store: &'a EncodedStore,
    /// HF hub cache root (`HF_HOME/hub`).
    pub hub: &'a Path,
    pub probe: &'a dyn RepoProbe,
    pub encoder: &'a dyn CheckpointEncoder,
}

impl HfAuto<'_> {
    /// Resolve `spec` (`hf://org/name[@revision]`) to a local directory.
    pub fn resolve(
        &self,
        spec: &str,
        fetch_vindex: &mut VindexFetch<'_>,
        status: &mut dyn Write,
    ) -> Result<Resolution, BoxErr> {
        let parsed = parse_spec(spec)?;
        let (repo, revision) = (parsed.repo.as_str(), parsed.revision.as_str());

        if let Some(path) = self.store.lookup(repo, revision) {
            return Ok(Resolution::Encoded { path, fresh: false });
        }
        if hub_has_vindex(self.hub, repo) {
            return fetch_vindex(spec).map(Resolution::Vindex);
        }
        match classify(self.probe, repo, revision) {
            RepoKind::Checkpoint => self.encode_checkpoint(repo, revision, status),
            RepoKind::Vindex | RepoKind::Unrecognised => fetch_vindex(spec).map(Resolution::Vindex),
        }
    }

    fn encode_checkpoint(
        &self,
        repo: &str,
        revision: &str,
        status: &mut dyn Write,
    ) -> Result<Resolution, BoxErr> {
        let commit = self.pin(repo, revision)?;
        if let Some(path) = self.store.complete(repo, &commit) {
            self.store.record_ref(repo, revision, &commit)?;
            return Ok(Resolution::Encoded { path, fresh: false });
        }
        writeln!(
            status,
            "{repo} is a Hugging Face checkpoint, not a vindex; \
             planning a VINDEX3 encode (first run only)…"
        )?;
        self.encode_commit(repo, revision, &commit, status)
    }

    /// Encode `repo` at the already-pinned `commit` into the store, then
    /// point `revision` at it. Atomic: the snapshot appears whole or not
    /// at all (see [`super::store`]).
    pub(super) fn encode_commit(
        &self,
        repo: &str,
        revision: &str,
        commit: &str,
        status: &mut dyn Write,
    ) -> Result<Resolution, BoxErr> {
        let dest = self.store.snapshot_dir(repo, commit)?;
        writeln!(status, "  commit {commit}")?;

        let partial = self.store.begin(repo)?;
        let pinned = format!("{HF_SCHEME}{repo}@{commit}");
        // On error `partial` drops here and its directory goes with it.
        self.encoder.encode(&pinned, partial.path(), status)?;
        let path = partial.commit(&dest)?;
        self.store.record_ref(repo, revision, commit)?;

        writeln!(status, "  cached at {}", path.display())?;
        writeln!(
            status,
            "  next time `larql run {}` runs it with no encode",
            shorthand(repo)
        )?;
        Ok(Resolution::Encoded { path, fresh: true })
    }

    /// The immutable commit to key the encode on.
    ///
    /// The hub's answer when it gives one. A revision that is itself a
    /// full commit id is accepted as-is when it does not. Anything else is
    /// refused: a container cached under a moving name would answer
    /// tomorrow's `main` with today's weights.
    pub(super) fn pin(&self, repo: &str, revision: &str) -> Result<String, BoxErr> {
        if let Some(commit) = self.probe.commit(repo, revision)? {
            return Ok(commit);
        }
        if is_full_commit(revision) {
            return Ok(revision.to_string());
        }
        Err(format!(
            "the hub named no commit for {HF_SCHEME}{repo}@{revision}, so the encode \
             cannot be cached under an immutable key; pass {HF_SCHEME}{repo}@<commit>"
        )
        .into())
    }
}

/// A 40-hex-digit commit id.
pub fn is_full_commit(revision: &str) -> bool {
    revision.len() == COMMIT_HEX_LEN && revision.chars().all(|c| c.is_ascii_hexdigit())
}

/// The cache shorthand an encoded repo answers to: its name half.
pub fn shorthand(repo: &str) -> &str {
    repo.rsplit_once('/').map_or(repo, |(_, name)| name)
}
