//! The whole ingest: resolve, encode, snapshot capabilities.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use larql_models::inventory::ArchitectureInventory;

use super::resolve::{ArtifactPayloads, ResolvedArtifact};
use crate::error::VindexError;
use crate::format::vindex3::encode::checkpoint::snapshot_checkpoint_capabilities;
use crate::format::vindex3::encode::encode_system_from_sources;
use crate::format::vindex3::encode::source::TensorSource;
use crate::format::vindex3::plan::capability::Capability;

/// What one repo-backed artifact actually pulled over the wire.
///
/// The headline claim of an `hf://` encode is a ratio: bytes fetched
/// against bytes the checkpoint declares. Returning it makes the claim
/// checkable rather than asserted — and a ratio near 1.0 is the signal
/// that the representation plan asked for everything, which is worth
/// seeing rather than assuming.
pub struct RemoteTransfer {
    pub name: String,
    pub fetched: u64,
    pub declared: u64,
    pub tensors: u64,
}

impl RemoteTransfer {
    /// Fetched over declared, or `0.0` when nothing was declared.
    pub fn fraction(&self) -> f64 {
        if self.declared == 0 {
            0.0
        } else {
            self.fetched as f64 / self.declared as f64
        }
    }
}

/// What one encode moved and produced.
pub struct IngestOutcome {
    pub container: PathBuf,
    pub representations: usize,
    pub total_payload_bytes: u64,
    /// Tokenizer and HF metadata copied in, in the order found.
    pub capabilities: Vec<String>,
    /// One entry per repo-backed artifact; empty for a wholly local encode.
    pub transfers: Vec<RemoteTransfer>,
}

/// Encode resolved artifacts into `output` and snapshot their capabilities.
///
/// The whole ingest in one place, because `vindex encode` and
/// `larql vindex3 encode` must produce the SAME container. Two
/// orchestrations would be free to differ on the capability snapshot
/// alone, and a container that binds with token-ids only is not obviously
/// wrong — it just answers differently.
///
/// Opening a source reads no payload: the local one reads shard headers,
/// the remote one indexes headers already staged. The admission gate then
/// runs inside [`encode_system_from_sources`], before the first
/// `stream_payload`, so an inadmissible repo costs its headers and not one
/// tensor byte.
pub fn encode_from_specs(
    resolved: Vec<ResolvedArtifact>,
    output: &Path,
    capability: Option<Capability>,
) -> Result<IngestOutcome, VindexError> {
    let named: Vec<(String, ArchitectureInventory)> = resolved
        .iter()
        .map(|artifact| (artifact.name.clone(), artifact.inventory.clone()))
        .collect();
    let payloads: Vec<(String, ArtifactPayloads)> = resolved
        .into_iter()
        .map(|artifact| {
            let name = artifact.name.clone();
            artifact.payloads().map(|opened| (name, opened))
        })
        .collect::<Result<_, _>>()?;
    let sources: BTreeMap<&str, &dyn TensorSource> = payloads
        .iter()
        .map(|(name, opened)| (name.as_str(), opened.as_source()))
        .collect();

    let outcome = encode_system_from_sources(&named, Some(&sources), output, capability)?;

    let transfers = payloads
        .iter()
        .filter_map(|(name, entry)| {
            let remote = entry.remote()?;
            Some(RemoteTransfer {
                name: name.clone(),
                fetched: remote.fetched(),
                declared: remote.declared_bytes(),
                tensors: remote.tensors(),
            })
        })
        .collect();

    Ok(IngestOutcome {
        container: outcome.container,
        representations: outcome.representations,
        total_payload_bytes: outcome.total_payload_bytes,
        capabilities: snapshot_capabilities(&named, output)?,
        transfers,
    })
}

/// Tokenizer and HF metadata from the first artifact directory carrying
/// them.
///
/// The inventory records its source directory, so this covers
/// checkpoint-dir, saved-inventory and staged-header inputs alike. A
/// container without them binds with token-id capability only — which is
/// why a granite smoke test needed a manual copy before this existed.
fn snapshot_capabilities(
    named: &[(String, ArchitectureInventory)],
    output: &Path,
) -> Result<Vec<String>, VindexError> {
    for (_, inventory) in named {
        let copied = snapshot_checkpoint_capabilities(Path::new(&inventory.path), output)?;
        if !copied.is_empty() {
            return Ok(copied);
        }
    }
    Ok(Vec::new())
}
