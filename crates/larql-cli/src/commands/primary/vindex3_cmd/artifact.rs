//! Resolving one `vindex3` artifact argument to an inventory and, when
//! the payloads are not local, a source that can stream them.
//!
//! Three spellings, one result:
//!
//! ```text
//! ./checkpoint            a directory of config.json + *.safetensors
//! ./inventory.json        a saved inventory
//! hf://org/name[@rev]     a repo — headers staged, payloads left there
//! ```
//!
//! The third is the one worth explaining. Admission — inventory, plan,
//! capability closure — reads safetensors *headers* and never a payload
//! byte, so a repo can be admitted from a few MB of staged headers.
//! Only if the plan is admissible does anything ask for a tensor, and
//! then it asks for that tensor's byte range and nothing else. A 328 GB
//! checkpoint is never on this disk, in whole or in part.

use std::path::{Path, PathBuf};

use larql_models::inventory::{build_inventory, ArchitectureInventory};
use larql_vindex::format::huggingface::metadata_checkpoint::{
    header_cache_dir, resolve_commit, stage_metadata_checkpoint, StagedCheckpoint,
};
use larql_vindex::format::huggingface::range::{is_hf_spec, parse_spec, HfRangeClient};
use larql_vindex::format::vindex3::encode::source::{
    staged_payload_bytes, ArtifactSource, RemoteArtifactSource, TensorSource,
};

use super::INVENTORY_EXT;

type CliError = Box<dyn std::error::Error>;

/// One artifact argument, resolved.
pub struct ResolvedArtifact {
    /// Name recorded in the container.
    pub name: String,
    pub inventory: ArchitectureInventory,
    /// Where this artifact's payloads come from. Opened lazily by
    /// [`Self::source`] so plan-only commands never construct one.
    origin: Origin,
}

enum Origin {
    /// Payloads sit beside the inventory, in the directory it records.
    Local,
    /// Payloads live in a repo; the headers are staged locally.
    Remote {
        client: HfRangeClient,
        staged: StagedCheckpoint,
    },
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
            Origin::Remote { staged, .. } => staged.commit.as_deref(),
        }
    }

    /// Open this artifact's payload source.
    pub fn payloads(self) -> Result<ArtifactPayloads, CliError> {
        match self.origin {
            Origin::Local => Ok(ArtifactPayloads::Local(ArtifactSource::open(Path::new(
                &self.inventory.path,
            ))?)),
            Origin::Remote { client, staged } => Ok(ArtifactPayloads::Remote(Box::new(
                RemoteArtifactSource::open(client, &staged)?,
            ))),
        }
    }
}

/// Whether an artifact argument names a repo rather than a local path.
pub fn is_remote_spec(spec: &Path) -> bool {
    is_hf_spec(&spec.to_string_lossy())
}

/// Resolve every artifact argument.
pub fn resolve_all(specs: &[PathBuf]) -> Result<Vec<ResolvedArtifact>, CliError> {
    specs.iter().map(|spec| resolve(spec)).collect()
}

/// Resolve one artifact argument.
pub fn resolve(spec: &Path) -> Result<ResolvedArtifact, CliError> {
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

/// A local artifact's name: the path stem, the rule `larql vindex3
/// encode` has always used.
fn local_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Load one local artifact: a `.json` file deserialises as a saved
/// inventory; anything else is inspected as a checkpoint directory.
fn load_local(path: &Path) -> Result<ArchitectureInventory, CliError> {
    if path.extension().is_some_and(|ext| ext == INVENTORY_EXT) {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    } else {
        Ok(build_inventory(path)?)
    }
}

/// Stage a repo's headers and inspect them.
///
/// The revision is resolved to a commit before anything is staged, and
/// the client re-pinned to it. `main` moves; headers read at one commit
/// and payloads read at another would address a different checkpoint with
/// the same offsets, which is the one failure mode this whole path could
/// have that produces plausible bytes.
fn resolve_remote(spec: &str) -> Result<ResolvedArtifact, CliError> {
    let parsed = parse_spec(spec)?;
    let named = HfRangeClient::new(&parsed.repo, &parsed.revision)?;
    let commit = resolve_commit(&named)?;
    let (client, pin) = match &commit {
        Some(sha) => (HfRangeClient::new(&parsed.repo, sha)?, sha.clone()),
        None => (named, parsed.revision.clone()),
    };
    if commit.is_none() {
        eprintln!(
            "warning: hf://{} did not report a commit for `{}` — \
             provenance records the revision name, which can move",
            parsed.repo, parsed.revision
        );
    }

    let dir = header_cache_dir(&parsed.repo, &pin)?;
    eprintln!("staging headers for hf://{} @ {pin}", parsed.repo);
    let staged = stage_metadata_checkpoint(&client, &dir)?;
    report_staging(&staged);

    let inventory = build_inventory(&staged.dir)?;
    Ok(ResolvedArtifact {
        name: remote_name(&parsed.repo),
        inventory,
        origin: Origin::Remote { client, staged },
    })
}

/// A repo artifact's name: the repo's own name, without the owner.
fn remote_name(repo: &str) -> String {
    repo.rsplit('/').next().unwrap_or(repo).to_string()
}

/// A byte count, unit stated.
///
/// Both units are given at GB scale and only there. That is where the
/// distinction bites — `GB` against `GiB` is a 7% difference, big enough
/// that two correct figures side by side read as a bug in one of them —
/// and it is also where the numbers this feature prints are compared
/// against each other. Below that, one unit stays readable: a staging
/// line quoting four figures in two units each is not clearer, it is
/// noise.
pub fn size(bytes: u64) -> String {
    const GB: f64 = 1e9;
    const GIB: f64 = (1u64 << 30) as f64;
    const MB: f64 = 1e6;
    let value = bytes as f64;
    if value >= GB {
        format!("{:.2} GB ({:.2} GiB)", value / GB, value / GIB)
    } else {
        format!("{:.2} MB", value / MB)
    }
}

/// What the staging cost, and what it stands in for.
///
/// Headers and metadata are quoted separately and then totalled. Quoting
/// the header figure alone would understate the transfer — a large
/// tokenizer can be several times the size of every shard header put
/// together — and the honest number is the one that makes "we did not
/// download the checkpoint" checkable.
///
/// The payload figure comes from the staged HEADERS, never from the shard
/// index's `metadata.total_size`. The two disagree whenever the source
/// model tied weights: HF computes `total_size` from deduplicated
/// parameter storage, so it declares a tied embedding once while the file
/// serialises it twice. granite-4.2-3b declares 6,805,672,960 bytes and
/// its own headers sum to 7,319,475,200 — short by exactly one
/// 513,802,240-byte member. Quoting the index would understate what a
/// range-read encode actually transfers, and a silent 7% gap between
/// "standing in for" and "fetched" reads like a units bug, so when they
/// disagree the difference is stated.
fn report_staging(staged: &StagedCheckpoint) {
    eprintln!(
        "staged {} ({} of headers over {} shard(s), {} of metadata)",
        size(staged.stub_bytes + staged.metadata_bytes),
        size(staged.stub_bytes),
        staged.shards.len(),
        size(staged.metadata_bytes),
    );
    let payload = match staged_payload_bytes(staged) {
        Ok(payload) => payload,
        // A census failure is not a reason to abort staging: the encode
        // reads the same headers and will fail with a better message.
        Err(err) => {
            eprintln!("warning: could not total the staged headers: {err}");
            return;
        }
    };
    eprintln!("  standing in for {} of tensor payload", size(payload));
    let Some(declared) = staged.declared_total_size else {
        return;
    };
    if declared != payload {
        eprintln!(
            "  note: the shard index declares {} — {} {} its own headers sum to \
             (tied weights counted once there, serialised twice here); the header \
             sum is what transfers",
            size(declared),
            size(declared.abs_diff(payload)),
            if declared < payload {
                "less than"
            } else {
                "more than"
            },
        );
    }
}
