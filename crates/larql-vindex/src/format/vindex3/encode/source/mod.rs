//! Streaming access to source tensor payloads.
//!
//! The inventory records what tensors exist (name, dtype, shape, bytes,
//! shard) from safetensors *headers*; encoding additionally needs each
//! payload's absolute offset. This module answers seek+stream requests —
//! payloads are never held in memory whole, a 50 GB decoder stack streams
//! through a fixed buffer.
//!
//! # Two sources, one offset arithmetic
//!
//! [`TensorSource`] has two implementations:
//!
//! ```text
//! ArtifactSource        a local checkpoint directory   — read(2) at offset
//! RemoteArtifactSource  a HuggingFace repo             — GET Range: bytes=…
//! ```
//!
//! What they do *not* have is two implementations of where a tensor
//! begins. A safetensors header states every payload's offset and length,
//! and both sources index one through [`index_shard_header`]. The remote
//! source reads its headers out of a staged header-only checkpoint
//! ([`crate::format::huggingface::metadata_checkpoint`]) — the same bytes
//! the hub would serve, already on disk because admission needed them
//! first. Agreement between two implementations of the same arithmetic
//! would not be evidence that either is right; there is only one.

mod remote;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::VindexError;

pub use remote::{index_staged_shards, staged_payload_bytes, RemoteArtifactSource};

/// Upper bound on a plausible safetensors header (mirrors the inventory
/// scanner's bound).
pub const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

/// Key safetensors reserves for free-form metadata inside the header.
const HEADER_METADATA_KEY: &str = "__metadata__";

/// Filename of the HF shard index.
const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";

/// Shard extension for the fallback scan.
const SAFETENSORS_EXT: &str = "safetensors";

/// Length of the safetensors length prefix.
const LENGTH_PREFIX_BYTES: u64 = 8;

/// Fixed streaming buffer size.
const STREAM_BUF: usize = 1 << 20;

/// Where one tensor's payload lives.
///
/// `shard` is stated in the source's own addressing: an absolute path for
/// a local directory, a repo-relative filename for a repo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadLocation {
    pub shard: PathBuf,
    /// Absolute offset of the first payload byte within the shard.
    pub offset: u64,
    pub len: u64,
}

/// A place tensor payloads can be streamed from.
///
/// Deliberately one method. The encoder does not branch on where bytes
/// come from, and nothing here exposes a path — a source that has no
/// local path must be as usable as one that does, or the abstraction is
/// only pretending.
pub trait TensorSource {
    /// Stream one tensor's payload into `write`, feeding every byte
    /// through `observe` (hashing) on the way. Returns bytes copied.
    ///
    /// Dyn parameters because callers pass through `write_segment`'s dyn
    /// callback.
    fn stream_payload(
        &self,
        name: &str,
        write: &mut dyn std::io::Write,
        observe: &mut dyn FnMut(&[u8]),
    ) -> Result<u64, VindexError>;
}

/// Payload locations for every tensor of one artifact directory.
pub struct ArtifactSource {
    locations: BTreeMap<String, PayloadLocation>,
}

impl ArtifactSource {
    /// Scan every `*.safetensors` shard in `dir` (via the HF index when
    /// present, else directory listing) and record payload locations.
    pub fn open(dir: &Path) -> Result<Self, VindexError> {
        let shards = discover_shards(dir)?;
        let mut locations = BTreeMap::new();
        for shard in shards {
            let (header, payload_base) = read_shard_header(&shard)?;
            index_shard_header(&header, payload_base, &shard, &mut locations)?;
        }
        Ok(Self { locations })
    }

    /// Location of one tensor by its full source name.
    pub fn locate(&self, name: &str) -> Result<&PayloadLocation, VindexError> {
        locate_in(&self.locations, name)
    }
}

impl TensorSource for ArtifactSource {
    fn stream_payload(
        &self,
        name: &str,
        write: &mut dyn std::io::Write,
        observe: &mut dyn FnMut(&[u8]),
    ) -> Result<u64, VindexError> {
        let location = self.locate(name)?;
        let mut file = std::fs::File::open(&location.shard)?;
        file.seek(SeekFrom::Start(location.offset))?;
        let mut remaining = location.len;
        let mut buffer = vec![0u8; STREAM_BUF];
        while remaining > 0 {
            let want = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..want]).map_err(|e| {
                VindexError::Parse(format!(
                    "short read streaming `{name}` from {}: {e}",
                    location.shard.display()
                ))
            })?;
            observe(&buffer[..want]);
            write.write_all(&buffer[..want])?;
            remaining -= want as u64;
        }
        Ok(location.len)
    }
}

/// Look one tensor up, or say why it is not there.
fn locate_in<'a>(
    locations: &'a BTreeMap<String, PayloadLocation>,
    name: &str,
) -> Result<&'a PayloadLocation, VindexError> {
    locations.get(name).ok_or_else(|| {
        VindexError::Parse(format!(
            "tensor `{name}` is in the inventory but not in any shard header — \
             the source directory changed since inspection"
        ))
    })
}

/// Shard filenames for a checkpoint dir (HF index preferred, sorted).
fn discover_shards(dir: &Path) -> Result<Vec<PathBuf>, VindexError> {
    let index_path = dir.join(SAFETENSORS_INDEX_FILE);
    if index_path.exists() {
        let text = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| VindexError::Parse(format!("parse {SAFETENSORS_INDEX_FILE}: {e}")))?;
        let mut files: Vec<String> = index["weight_map"]
            .as_object()
            .map(|m| {
                m.values()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        files.dedup();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == SAFETENSORS_EXT))
        .collect();
    files.sort();
    Ok(files)
}

/// Read one shard's header bytes and the absolute offset its payload
/// region begins at.
///
/// Reads the length prefix and the header it announces, and nothing else
/// — which is why this works unchanged on a header-only stub whose
/// payload region is not present at all.
fn read_shard_header(shard: &Path) -> Result<(Vec<u8>, u64), VindexError> {
    let mut file = std::fs::File::open(shard)?;
    let mut len_bytes = [0u8; LENGTH_PREFIX_BYTES as usize];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_BYTES {
        return Err(VindexError::Parse(format!(
            "{}: safetensors header claims {header_len} bytes — corrupt",
            shard.display()
        )));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    Ok((header_bytes, LENGTH_PREFIX_BYTES + header_len))
}

/// Record every tensor a shard header declares: absolute offset =
/// `payload_base` + the header's relative data offset.
///
/// `shard` is recorded verbatim into each location, so the caller decides
/// whether locations address a local file or a repo-relative name.
fn index_shard_header(
    header_bytes: &[u8],
    payload_base: u64,
    shard: &Path,
    out: &mut BTreeMap<String, PayloadLocation>,
) -> Result<(), VindexError> {
    let header: serde_json::Value = serde_json::from_slice(header_bytes)
        .map_err(|e| VindexError::Parse(format!("{}: header: {e}", shard.display())))?;
    let entries = header.as_object().ok_or_else(|| {
        VindexError::Parse(format!("{}: header is not an object", shard.display()))
    })?;
    for (name, desc) in entries {
        if name == HEADER_METADATA_KEY {
            continue;
        }
        let offsets = desc["data_offsets"].as_array().and_then(|offs| {
            match (offs.first()?.as_u64(), offs.get(1)?.as_u64()) {
                (Some(start), Some(end)) if end >= start => Some((start, end)),
                _ => None,
            }
        });
        let Some((start, end)) = offsets else {
            return Err(VindexError::Parse(format!(
                "{}: tensor `{name}` has no valid data_offsets",
                shard.display()
            )));
        };
        out.insert(
            name.clone(),
            PayloadLocation {
                shard: shard.to_path_buf(),
                offset: payload_base + start,
                len: end - start,
            },
        );
    }
    Ok(())
}
