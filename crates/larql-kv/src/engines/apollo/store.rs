//! On-disk Apollo store format.
//!
//! Mirrors the layout of `apollo-demo/apollo11_store/`:
//!
//! ```text
//! apollo11_store/
//! ├── manifest.json              # version, num_windows, crystal_layer, arch_config
//! ├── boundaries/
//! │   ├── window_000.npy         # shape (hidden,) f32 — single residual
//! │   ├── window_001.npy
//! │   └── ...
//! ├── boundary_residual.npy      # shape (1, 1, hidden) — most recent / active boundary
//! ├── window_token_lists.npz    # keyed by "0", "1", ... → u32 token arrays
//! └── entries.npz                # structured array of VecInjectEntry
//! ```
//!
//! Loading uses a handwritten `.npy` parser (see `npy.rs`) and a stored-only
//! `.npz` reader (see `npz.rs`). No `ndarray-npy` dependency because its
//! current release (0.10) pins ndarray 0.17 and our workspace is on 0.16.
//!
//! Every size and every file in a store is untrusted. Files must be regular
//! (`store_file.rs`) and within [`StoreLimits`], compared with their logical
//! length. Allocations sized from file contents are fallible and made after the
//! size is checked against that length. `manifest.num_windows` sizes nothing,
//! because the vector grows per file read.
//!
//! This does not make loading safe against memory exhaustion. A sparse file is
//! judged by its logical length, so a tiny file within the limits can still cost
//! real memory up to them, and small bounded allocations such as paths and JSON
//! values stay infallible. See `store_limits.rs`.

use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::entry::VecInjectEntry;
use super::npy;
use super::npz::{ArchiveError, StoredArchive};
use super::store_file::{try_reserve_for, StoreReader};
use super::store_limits::StoreLimits;

#[derive(Debug, Error)]
pub enum StoreLoadError {
    #[error("i/o error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("json parse error in manifest: {0}")]
    Json(#[from] serde_json::Error),
    #[error("npy parse error in {path}: {source}")]
    Npy {
        path: String,
        #[source]
        source: npy::NpyError,
    },
    #[error("archive error in {path}: {source}")]
    Archive {
        path: String,
        #[source]
        source: ArchiveError,
    },
    #[error("{path}: not a regular file ({file_type})")]
    NotRegularFile { path: String, file_type: String },
    #[error("{path}: {len} bytes is over the limit of {limit}")]
    FileTooLarge { path: String, len: u64, limit: u64 },
    #[error("{path}: brings the store to {total} bytes, over the limit of {limit}")]
    StoreTooLarge {
        path: String,
        total: u64,
        limit: u64,
    },
    #[error("{path}: allocation of {bytes} bytes refused")]
    AllocationRefused { path: String, bytes: u64 },
    #[error("store missing required file: {0}")]
    MissingFile(String),
    #[error("manifest mismatch: {0}")]
    ManifestMismatch(String),
    #[error("structured-dtype parse error in {path}: {reason}")]
    StructuredDtype { path: String, reason: String },
    #[error("window_token_lists archive invalid: {0}")]
    WindowArchive(String),
    /// A member's two declared sizes disagree.
    #[error(
        "{path}: member declares {declared} uncompressed bytes but occupies {stored} bytes; \
         only stored (uncompressed) .npz members are supported"
    )]
    MemberSizeInconsistent {
        path: String,
        declared: u64,
        stored: u64,
    },
    /// The archive's members claim more bytes than the file's logical length.
    #[error(
        "{path}: members claim {claimed} bytes in total but the archive is {archive_len} bytes"
    )]
    MemberExceedsArchive {
        path: String,
        claimed: u64,
        archive_len: u64,
    },
    /// A file or member read back a different length from the one measured.
    #[error("{path}: expected {declared} bytes but read back {read}")]
    MemberLengthMismatch {
        path: String,
        declared: u64,
        read: u64,
    },
}

/// Contents of `manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreManifest {
    pub version: u32,
    pub num_entries: usize,
    pub num_windows: usize,
    pub num_tokens: usize,
    pub entries_per_window: usize,
    pub crystal_layer: usize,
    pub window_size: usize,
    pub arch_config: ArchConfig,
    pub has_residuals: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchConfig {
    pub retrieval_layer: usize,
    pub query_head: usize,
    pub injection_layer: usize,
    pub inject_coefficient: f32,
}

impl Default for ArchConfig {
    fn default() -> Self {
        // Apollo 11 defaults on Gemma 3 4B.
        Self {
            retrieval_layer: 29,
            query_head: 4,
            injection_layer: 30,
            inject_coefficient: 10.0,
        }
    }
}

/// In-memory representation of a loaded Apollo store.
#[derive(Debug)]
pub struct ApolloStore {
    pub manifest: StoreManifest,
    /// One residual vector per window at `crystal_layer`. `boundaries[i]`
    /// is a flat `(hidden,)` Vec for window i.
    pub boundaries: Vec<Vec<f32>>,
    /// `(1, 1, hidden)` — most recent / active boundary residual.
    /// Flattened to Vec<f32>.
    pub boundary_residual: Option<Vec<f32>>,
    /// Per-window token ID lists. `window_tokens[i]` has `window_size`
    /// entries (the last window may be shorter).
    pub window_tokens: Vec<Vec<u32>>,
    /// All vec_inject entries (flattened across windows).
    pub entries: Vec<VecInjectEntry>,
}

impl ApolloStore {
    /// Load an Apollo store from a directory, within [`StoreLimits::default`].
    ///
    /// Malformed input returns an error rather than aborting, with the limits
    /// in `store_limits.rs`: treat this as bounding, not preventing, the memory
    /// an untrusted store can cost.
    pub fn load(path: &Path) -> Result<Self, StoreLoadError> {
        Self::load_with_limits(path, &StoreLimits::default())
    }

    /// [`Self::load`] with explicit limits, for a store larger than the defaults
    /// admit or a host with less memory to spare.
    pub fn load_with_limits(path: &Path, limits: &StoreLimits) -> Result<Self, StoreLoadError> {
        let mut reader = StoreReader::new(limits);
        let manifest = load_manifest(&mut reader, path)?;
        let boundaries = load_boundaries(&mut reader, path, manifest.num_windows)?;
        let boundary_residual = load_boundary_residual(&mut reader, path)?;
        let window_tokens = load_window_tokens(&mut reader, path)?;
        let entries = load_entries(&mut reader, path)?;

        if boundaries.len() != manifest.num_windows {
            return Err(StoreLoadError::ManifestMismatch(format!(
                "manifest.num_windows={} but loaded {} boundaries",
                manifest.num_windows,
                boundaries.len(),
            )));
        }
        if entries.len() != manifest.num_entries {
            return Err(StoreLoadError::ManifestMismatch(format!(
                "manifest.num_entries={} but loaded {} entries",
                manifest.num_entries,
                entries.len(),
            )));
        }
        // window_tokens indexes by window_id, and boundaries/entries assume
        // the same id space — a count mismatch means the archive and the
        // manifest disagree about which windows exist.
        if window_tokens.len() != manifest.num_windows {
            return Err(StoreLoadError::ManifestMismatch(format!(
                "manifest.num_windows={} but loaded {} window token lists",
                manifest.num_windows,
                window_tokens.len(),
            )));
        }

        Ok(Self {
            manifest,
            boundaries,
            boundary_residual,
            window_tokens,
            entries,
        })
    }

    pub fn total_bytes(&self) -> usize {
        let boundary_bytes: usize = self.boundaries.iter().map(|b| b.len() * 4).sum();
        let boundary_residual_bytes = self
            .boundary_residual
            .as_ref()
            .map(|b| b.len() * 4)
            .unwrap_or(0);
        let token_bytes: usize = self.window_tokens.iter().map(|w| w.len() * 4).sum();
        let entry_bytes = self.entries.len() * std::mem::size_of::<VecInjectEntry>();
        boundary_bytes + boundary_residual_bytes + token_bytes + entry_bytes
    }

    pub fn hidden_size(&self) -> usize {
        self.boundaries.first().map(|b| b.len()).unwrap_or(0)
    }
}

// ── internals ────────────────────────────────────────────────────────────

fn load_manifest(reader: &mut StoreReader, path: &Path) -> Result<StoreManifest, StoreLoadError> {
    let limit = reader.limits().max_manifest_bytes;
    let bytes = reader.read_file(&path.join("manifest.json"), limit)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn load_boundaries(
    reader: &mut StoreReader,
    path: &Path,
    num_windows: usize,
) -> Result<Vec<Vec<f32>>, StoreLoadError> {
    let dir = path.join("boundaries");
    let limit = reader.limits().max_array_file_bytes;
    // Grown per file: `num_windows` is unvalidated, and a missing file ends the
    // loop, so a huge claim costs one failed open rather than a huge reservation.
    let mut out = Vec::new();
    for i in 0..num_windows {
        let p = dir.join(format!("window_{:03}.npy", i));
        let path_text = p.display().to_string();
        let bytes = reader.read_file(&p, limit)?;
        let arr = npy::read_f32_1d(&bytes).map_err(|source| StoreLoadError::Npy {
            path: path_text.clone(),
            source,
        })?;
        try_reserve_for(&mut out, 1, &path_text)?;
        out.push(arr);
    }
    Ok(out)
}

/// `None` if the optional residual is absent or its contents are not a usable
/// array. A file that cannot be read as a regular file within the limits, or an
/// allocation refused, is a failure of the store and not an absent residual.
fn load_boundary_residual(
    reader: &mut StoreReader,
    path: &Path,
) -> Result<Option<Vec<f32>>, StoreLoadError> {
    let p = path.join("boundary_residual.npy");
    let limit = reader.limits().max_array_file_bytes;
    let bytes = match reader.read_file(&p, limit) {
        Ok(bytes) => bytes,
        Err(StoreLoadError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(other) => return Err(other),
    };
    match npy::read_f32_flat(&bytes) {
        Ok((flat, _shape)) => Ok(Some(flat)),
        Err(source @ npy::NpyError::AllocationRefused { .. }) => Err(StoreLoadError::Npy {
            path: p.display().to_string(),
            source,
        }),
        Err(_) => Ok(None),
    }
}

fn open_archive(reader: &mut StoreReader, path: &Path) -> Result<StoredArchive, StoreLoadError> {
    let limit = reader.limits().max_archive_bytes;
    StoredArchive::open(reader.open(path, limit)?)
}

fn load_window_tokens(
    reader: &mut StoreReader,
    path: &Path,
) -> Result<Vec<Vec<u32>>, StoreLoadError> {
    let p = path.join("window_token_lists.npz");
    let mut archive = open_archive(reader, &p)?;
    let archive_path = p.display().to_string();

    // Collect and numerically sort the members so returned Vec is indexable
    // by window_id. Member names are like "0.npy", "1.npy", ...
    let mut numbered: Vec<(usize, usize, String)> = Vec::new();
    for (position, name) in archive.names().enumerate() {
        if let Ok(id) = name.trim_end_matches(".npy").parse::<usize>() {
            try_reserve_for(&mut numbered, 1, &archive_path)?;
            numbered.push((id, position, name.to_string()));
        }
    }
    numbered.sort_by_key(|(id, _, _)| *id);

    // Invariant: the returned Vec is indexed by window_id, so member ids
    // must be exactly 0..len after the sort — a gapped archive (0, 1, 3)
    // or a duplicated id would otherwise be packed densely and silently
    // misalign every later window against boundaries/entries.
    for (expected_id, (id, _, name)) in numbered.iter().enumerate() {
        if *id != expected_id {
            return Err(StoreLoadError::WindowArchive(format!(
                "{}: member ids must be contiguous from 0 — expected id {expected_id}, \
                 found {id} ({name}); a gapped or duplicated archive would misalign \
                 window ids against boundaries/entries",
                p.display(),
            )));
        }
    }

    let mut out = Vec::new();
    try_reserve_for(&mut out, numbered.len(), &archive_path)?;
    for (_id, position, name) in numbered {
        let buf = archive.read_member(position)?;
        let arr = npy::read_u32_1d(&buf).map_err(|source| StoreLoadError::Npy {
            path: format!("{}::{}", p.display(), name),
            source,
        })?;
        out.push(arr);
    }
    Ok(out)
}

fn load_entries(
    reader: &mut StoreReader,
    path: &Path,
) -> Result<Vec<VecInjectEntry>, StoreLoadError> {
    let p = path.join("entries.npz");
    let mut archive = open_archive(reader, &p)?;

    // Find the first member whose name starts with "entries" (typically
    // "entries.npy" inside the zip).
    let position = archive
        .names()
        .position(|name| name.starts_with("entries"))
        .ok_or_else(|| StoreLoadError::MissingFile("entries.npz::entries".into()))?;
    let member_name = archive
        .names()
        .nth(position)
        .unwrap_or_default()
        .to_string();
    let bytes = archive.read_member(position)?;
    let member_path = format!("{}::{}", p.display(), member_name);

    parse_structured_entries_npy(&bytes).map_err(|error| match error {
        npy::NpyError::StructuredLayout(reason) => StoreLoadError::StructuredDtype {
            path: member_path,
            reason,
        },
        source => StoreLoadError::Npy {
            path: member_path,
            source,
        },
    })
}

/// Exact structured-dtype layout the row decoder below assumes. Invariant:
/// the decoder reads fields at fixed byte offsets in this order, so the
/// archive's descr must match name-for-name AND dtype-for-dtype in this
/// exact order — a reordered or retyped descr would otherwise decode into
/// garbage coefficients/ids without any error.
const ENTRY_DESCR_FIELDS: [(&str, &str); 5] = [
    ("token_id", "<u4"),
    ("coefficient", "<f4"),
    ("window_id", "<u2"),
    ("position_in_window", "<u2"),
    ("fact_id", "<u2"),
];

/// Extract every single-quoted token from a numpy structured descr, in
/// order. For `[('token_id', '<u4'), ...]` this yields alternating
/// field-name / dtype strings — numpy always emits single quotes here.
fn descr_quoted_tokens(descr: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = descr;
    while let Some(start) = rest.find('\'') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('\'') else { break };
        out.push(&after[..end]);
        rest = &after[end + 1..];
    }
    out
}

/// Validate that `descr` declares exactly [`ENTRY_DESCR_FIELDS`] — same
/// fields, same dtypes, same order. Presence-by-substring is not enough:
/// the decoder reads fixed offsets.
fn check_entry_descr(descr: &str) -> Result<(), npy::NpyError> {
    let tokens = descr_quoted_tokens(descr);
    let got: Vec<(&str, &str)> = tokens
        .chunks(2)
        .filter(|c| c.len() == 2)
        .map(|c| (c[0], c[1]))
        .collect();
    if got != ENTRY_DESCR_FIELDS {
        return Err(npy::NpyError::StructuredLayout(format!(
            "structured descr mismatch: got fields {got:?}, expected {ENTRY_DESCR_FIELDS:?} \
             (decoder reads fixed offsets; field order and dtypes must match exactly)"
        )));
    }
    Ok(())
}

/// Parse a .npy file containing a structured-dtype array of `VecInjectEntry`.
///
/// Expected dtype (from the Python side): see [`ENTRY_DESCR_FIELDS`].
/// Row size: 14 bytes, no padding (numpy packs structured dtypes tightly
/// when fields are already aligned).
fn parse_structured_entries_npy(bytes: &[u8]) -> Result<Vec<VecInjectEntry>, npy::NpyError> {
    let (header, data_off) = npy::parse_header(bytes)?;

    check_entry_descr(&header.descr)?;
    if header.shape.len() != 1 {
        return Err(npy::NpyError::StructuredLayout(format!(
            "expected 1D structured array, got shape {:?}",
            header.shape
        )));
    }

    const ROW_SIZE: usize = 4 + 4 + 2 + 2 + 2;
    let n = header.shape[0];
    let data = &bytes[data_off..];
    let expected = npy::checked_data_len(&header.shape, ROW_SIZE)?;
    if data.len() != expected {
        return Err(npy::NpyError::StructuredLayout(format!(
            "data size {} != expected {} ({} rows × {} bytes)",
            data.len(),
            expected,
            n,
            ROW_SIZE,
        )));
    }

    // `n` rows of `ROW_SIZE` bytes are present in `data`, as just checked.
    let mut out = npy::try_vec_with_capacity(n)?;
    let (rows, _) = data.as_chunks::<ROW_SIZE>();
    for row in rows {
        out.push(VecInjectEntry {
            token_id: u32::from_le_bytes([row[0], row[1], row[2], row[3]]),
            coefficient: f32::from_le_bytes([row[4], row[5], row[6], row[7]]),
            window_id: u16::from_le_bytes([row[8], row[9]]),
            position_in_window: u16::from_le_bytes([row[10], row[11]]),
            fact_id: u16::from_le_bytes([row[12], row[13]]),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
