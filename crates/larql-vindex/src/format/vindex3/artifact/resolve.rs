//! Resolving one artifact argument to an inventory and a payload source.

use std::path::{Path, PathBuf};

use larql_models::inventory::{build_inventory, ArchitectureInventory};

use super::staging::StagingReport;
use crate::error::VindexError;
use crate::format::huggingface::metadata_checkpoint::{
    header_cache_dir, resolve_commit, stage_metadata_checkpoint, StagedCheckpoint,
};
use crate::format::huggingface::range::{is_hf_spec, parse_spec, HfRangeClient};
use crate::format::vindex3::encode::source::{
    staged_payload_bytes, ArtifactSource, RemoteArtifactSource, TensorSource,
};

/// Extension distinguishing a saved inventory JSON from a checkpoint dir.
pub const INVENTORY_EXT: &str = "json";

/// One artifact argument, resolved.
pub struct ResolvedArtifact {
    /// Name recorded in the container.
    pub name: String,
    pub inventory: ArchitectureInventory,
    /// Where this artifact's payloads come from. Opened lazily by
    /// [`Self::payloads`] so plan-only commands never construct one.
    origin: Origin,
}

enum Origin {
    /// Payloads sit beside the inventory, in the directory it records.
    Local,
    /// Payloads live in a repo; the headers are staged locally.
    ///
    /// Boxed because the local case carries nothing and this carries a
    /// client plus a whole staged manifest — unboxed, every
    /// `ResolvedArtifact` would be as large as the remote case, including
    /// the local ones this feature never touches.
    Remote(Box<RemoteOrigin>),
}

/// A repo artifact's client and the staging that produced its headers.
struct RemoteOrigin {
    client: HfRangeClient,
    staged: StagedCheckpoint,
    /// Set when the hub named no commit, so the caller can say that
    /// provenance records a revision NAME, which can move.
    unpinned: Option<String>,
}

/// A payload source plus, for a repo, the staging that produced it.
pub enum ArtifactPayloads {
    Local(ArtifactSource),
    Remote(Box<RemoteArtifactSource>),
}

impl ArtifactPayloads {
    pub fn as_source(&self) -> &dyn TensorSource {
        match self {
            Self::Local(source) => source,
            Self::Remote(source) => source.as_ref(),
        }
    }

    /// The repo source, when this artifact has one.
    pub fn remote(&self) -> Option<&RemoteArtifactSource> {
        match self {
            Self::Local(_) => None,
            Self::Remote(source) => Some(source.as_ref()),
        }
    }
}

impl ResolvedArtifact {
    /// The commit a repo artifact was pinned to, when the hub said which.
    pub fn commit(&self) -> Option<&str> {
        match &self.origin {
            Origin::Local => None,
            Origin::Remote(remote) => remote.staged.commit.as_deref(),
        }
    }

    /// The revision name this artifact fell back to because the hub named
    /// no commit — provenance that can move, and worth saying so.
    pub fn unpinned_revision(&self) -> Option<&str> {
        match &self.origin {
            Origin::Local => None,
            Origin::Remote(remote) => remote.unpinned.as_deref(),
        }
    }

    /// What staging cost, for a repo artifact. `None` for a local one,
    /// which staged nothing.
    /// What staging read to answer a question about this artifact, as
    /// JSON — or `None` for a local artifact, which staged nothing.
    /// Not an empty report: that would read as "staged, and it cost
    /// zero". The object's shape is
    /// [`super::staging::staging_report_json`].
    pub fn staging_json(&self) -> Option<serde_json::Value> {
        Some(super::staging::staging_report_json(
            &self.name,
            self.commit(),
            &self.staging()?,
        ))
    }

    pub fn staging(&self) -> Option<StagingReport> {
        let Origin::Remote(remote) = &self.origin else {
            return None;
        };
        let staged = &remote.staged;
        Some(StagingReport {
            header_bytes: staged.stub_bytes,
            metadata_bytes: staged.metadata_bytes,
            shards: staged.shards.len(),
            payload_bytes: staged_payload_bytes(staged),
            declared_total: staged.declared_total_size,
        })
    }

    /// Open this artifact's payload source.
    pub fn payloads(self) -> Result<ArtifactPayloads, VindexError> {
        match self.origin {
            Origin::Local => Ok(ArtifactPayloads::Local(ArtifactSource::open(Path::new(
                &self.inventory.path,
            ))?)),
            Origin::Remote(remote) => Ok(ArtifactPayloads::Remote(Box::new(
                RemoteArtifactSource::open(remote.client, &remote.staged)?,
            ))),
        }
    }
}

/// Whether an artifact argument names a repo rather than a local path.
pub fn is_remote_spec(spec: &Path) -> bool {
    is_hf_spec(&spec.to_string_lossy())
}

/// Resolve every artifact argument.
pub fn resolve_all(specs: &[PathBuf]) -> Result<Vec<ResolvedArtifact>, VindexError> {
    specs.iter().map(|spec| resolve(spec)).collect()
}

/// Resolve one artifact argument.
pub fn resolve(spec: &Path) -> Result<ResolvedArtifact, VindexError> {
    let text = spec.to_string_lossy();
    if is_hf_spec(&text) {
        return resolve_remote(&text);
    }
    Ok(ResolvedArtifact {
        name: local_name(spec),
        inventory: load_local(spec)?,
        origin: Origin::Local,
    })
}

/// A local artifact's name: the path stem, the rule encode has always used.
fn local_name(path: &Path) -> String {
    path.file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Load one local artifact: a `.json` file deserialises as a saved
/// inventory; anything else is inspected as a checkpoint directory.
fn load_local(path: &Path) -> Result<ArchitectureInventory, VindexError> {
    if path.extension().is_some_and(|ext| ext == INVENTORY_EXT) {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text)
            .map_err(|e| VindexError::Parse(format!("{}: {e}", path.display())))
    } else {
        Ok(build_inventory(path)?)
    }
}

/// Stage a repo's headers and inspect them.
///
/// The revision is resolved to a commit before anything is staged, and the
/// client re-pinned to it. `main` moves; headers read at one commit and
/// payloads read at another would address a different checkpoint with the
/// same offsets, which is the one failure this whole path could have that
/// still produces plausible bytes.
/// The immutable commit `spec` resolves to, **without staging it**.
///
/// One ranged request that reads the hub's commit header — the cheap
/// half of what [`resolve`] does before it fetches anything. It exists
/// so a caller holding a verdict cache can learn enough immutable
/// identity to look up an answer *before* paying for the headers that
/// would produce it. Measured on the public explorer, that is the
/// difference between a repeat plan costing ~90 MB and a full header
/// parse, and costing one HTTP round trip.
///
/// `None` — meaning "no persistent identity, do not cache" — for:
/// - a local path, which has no commit at all; and
/// - a repo the hub answers without a commit header, i.e. an unpinned
///   revision name, which can move.
///
/// Goes through the same [`resolve_commit`] that [`resolve`] itself
/// uses, so the commit a cache is keyed on and the commit a verdict
/// records cannot disagree by construction rather than by discipline.
pub fn resolve_pinned_commit(spec: &Path) -> Result<Option<String>, VindexError> {
    let text = spec.to_string_lossy();
    if !is_hf_spec(&text) {
        return Ok(None);
    }
    let parsed = parse_spec(&text)?;
    resolve_commit(&HfRangeClient::new(&parsed.repo, &parsed.revision)?)
}

/// [`resolve_pinned_commit`] for every spec, in order.
///
/// A caller may key a cache on the result only when **every** entry is
/// `Some`: one local path or one unpinned revision among the artifacts
/// makes the whole verdict uncacheable, the same rule
/// `SystemPlan::cache_key` applies after the fact.
pub fn resolve_pinned_commits(specs: &[PathBuf]) -> Result<Vec<Option<String>>, VindexError> {
    specs.iter().map(|s| resolve_pinned_commit(s)).collect()
}

fn resolve_remote(spec: &str) -> Result<ResolvedArtifact, VindexError> {
    let parsed = parse_spec(spec)?;
    let named = HfRangeClient::new(&parsed.repo, &parsed.revision)?;
    let commit = resolve_commit(&named)?;
    let (client, pin, unpinned) = match &commit {
        Some(sha) => (HfRangeClient::new(&parsed.repo, sha)?, sha.clone(), None),
        None => (
            named,
            parsed.revision.clone(),
            Some(parsed.revision.clone()),
        ),
    };

    let dir = header_cache_dir(&parsed.repo, &pin)?;
    let staged = stage_metadata_checkpoint(&client, &dir)?;
    let inventory = build_inventory(&staged.dir)?;
    Ok(ResolvedArtifact {
        name: remote_name(&parsed.repo),
        inventory,
        origin: Origin::Remote(Box::new(RemoteOrigin {
            client,
            staged,
            unpinned,
        })),
    })
}

/// A repo artifact's name: the repo's own name, without the owner.
fn remote_name(repo: &str) -> String {
    repo.rsplit('/').next().unwrap_or(repo).to_string()
}
