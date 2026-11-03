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
//! Loading uses a handwritten `.npy` parser (see `npy.rs`) + the `zip` crate
//! for the `.npz` containers. No `ndarray-npy` dependency because its
//! current release (0.10) pins ndarray 0.17 and our workspace is on 0.16.
//!
//! Every size in a store is file-declared and untrusted. The loader allocates
//! only from what was checked against bytes present on disk: see
//! `ArchiveBudget` for `.npz` members and `npy::checked_data_len` for shapes.
//! `manifest.num_windows` sizes nothing, because the vector grows per file read.
//! Allocations inside `zip::ZipArchive::new` are outside this module's control.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::entry::VecInjectEntry;
use super::npy;

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
    #[error("zip parse error in {path}: {source}")]
    Zip {
        path: String,
        #[source]
        source: zip::result::ZipError,
    },
    #[error("store missing required file: {0}")]
    MissingFile(String),
    #[error("manifest mismatch: {0}")]
    ManifestMismatch(String),
    #[error("structured-dtype parse error in {path}: {reason}")]
    StructuredDtype { path: String, reason: String },
    #[error("window_token_lists archive invalid: {0}")]
    WindowArchive(String),
    /// A member's two declared sizes disagree. See `ArchiveBudget`.
    #[error(
        "{path}: member declares {declared} uncompressed bytes but occupies {stored} bytes; \
         only stored (uncompressed) .npz members are supported"
    )]
    MemberSizeInconsistent {
        path: String,
        declared: u64,
        stored: u64,
    },
    /// The archive's members claim more bytes than the file holds.
    /// See `ArchiveBudget`.
    #[error(
        "{path}: members claim {claimed} bytes in total but the archive is {archive_len} bytes"
    )]
    MemberExceedsArchive {
        path: String,
        claimed: u64,
        archive_len: u64,
    },
    /// A member read back a different length from the one it declared.
    #[error("{path}: member declares {declared} bytes but read back {read}")]
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
    /// Load an Apollo store from a directory.
    pub fn load(path: &Path) -> Result<Self, StoreLoadError> {
        let manifest = load_manifest(path)?;
        let boundaries = load_boundaries(path, manifest.num_windows)?;
        let boundary_residual = load_boundary_residual(path).ok();
        let window_tokens = load_window_tokens(path)?;
        let entries = load_entries(path)?;

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

fn read_file(path: &Path) -> Result<Vec<u8>, StoreLoadError> {
    std::fs::read(path).map_err(|source| StoreLoadError::Io {
        path: path.display().to_string(),
        source,
    })
}

fn load_manifest(path: &Path) -> Result<StoreManifest, StoreLoadError> {
    let bytes = read_file(&path.join("manifest.json"))?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn load_boundaries(path: &Path, num_windows: usize) -> Result<Vec<Vec<f32>>, StoreLoadError> {
    let dir = path.join("boundaries");
    // Grown per file: `num_windows` is unvalidated, and a missing file ends the
    // loop, so a huge claim costs one failed open rather than a huge reservation.
    let mut out = Vec::new();
    for i in 0..num_windows {
        let p = dir.join(format!("window_{:03}.npy", i));
        let bytes = read_file(&p)?;
        let arr = npy::read_f32_1d(&bytes).map_err(|source| StoreLoadError::Npy {
            path: p.display().to_string(),
            source,
        })?;
        out.push(arr);
    }
    Ok(out)
}

fn load_boundary_residual(path: &Path) -> Result<Vec<f32>, StoreLoadError> {
    let p = path.join("boundary_residual.npy");
    let bytes = read_file(&p)?;
    let (flat, _shape) = npy::read_f32_flat(&bytes).map_err(|source| StoreLoadError::Npy {
        path: p.display().to_string(),
        source,
    })?;
    Ok(flat)
}

/// Tracks what an archive's members claim against the file's real length.
///
/// The rule, for each member, before anything is allocated for it:
///
/// 1. Its declared uncompressed size must equal the size it occupies. This crate
///    enables no `zip` compression feature, so only stored members can be decoded
///    and for those the two are the same number. `numpy.savez` writes stored
///    members; `numpy.savez_compressed` output already fails with
///    `ZipError::UnsupportedArchive`.
/// 2. The bytes occupied, summed over the members read so far, must fit in the
///    file. Members of a real archive are disjoint regions of it, so this holds
///    for any honest writer. It also refuses directory records that share one
///    payload, which would otherwise decode to many times the file's length.
///
/// The file's length comes from the open handle, not from the archive, so
/// together these limit all member data to bytes that exist. There is no
/// tunable constant: a real store of any size passes.
///
/// Rule 1 also fails closed if a downstream build unifies in a `zip` compression
/// feature: compressed members are refused rather than admitted without a bound.
/// Accepting them would need its own limit, such as DEFLATE's 1032:1 ceiling.
#[derive(Debug)]
struct ArchiveBudget {
    archive_len: u64,
    claimed: u64,
}

impl ArchiveBudget {
    fn admit(&mut self, path: &str, declared: u64, stored: u64) -> Result<(), StoreLoadError> {
        if declared != stored {
            return Err(StoreLoadError::MemberSizeInconsistent {
                path: path.to_string(),
                declared,
                stored,
            });
        }
        let claimed = self.claimed.saturating_add(stored);
        if claimed > self.archive_len {
            return Err(StoreLoadError::MemberExceedsArchive {
                path: path.to_string(),
                claimed,
                archive_len: self.archive_len,
            });
        }
        self.claimed = claimed;
        Ok(())
    }
}

/// One byte past the declared size, read so that a member decoding longer than
/// it declared is detected rather than silently cut to length.
const READ_OVERRUN_PROBE_BYTES: u64 = 1;

fn open_archive(path: &Path) -> Result<(zip::ZipArchive<File>, ArchiveBudget), StoreLoadError> {
    let io_err = |source| StoreLoadError::Io {
        path: path.display().to_string(),
        source,
    };
    let file = File::open(path).map_err(io_err)?;
    let archive_len = file.metadata().map_err(io_err)?.len();
    let archive = zip::ZipArchive::new(file).map_err(|source| StoreLoadError::Zip {
        path: path.display().to_string(),
        source,
    })?;
    let budget = ArchiveBudget {
        archive_len,
        claimed: 0,
    };
    Ok((archive, budget))
}

/// Read one member, sized by what arrives and never by what it declares. Growth
/// inside `read_to_end` is fallible, so a refused allocation is an `Io` error
/// rather than an abort.
fn read_member(
    archive: &mut zip::ZipArchive<File>,
    budget: &mut ArchiveBudget,
    archive_path: &Path,
    name: &str,
) -> Result<Vec<u8>, StoreLoadError> {
    let member_path = format!("{}::{}", archive_path.display(), name);
    let entry = archive
        .by_name(name)
        .map_err(|source| StoreLoadError::Zip {
            path: member_path.clone(),
            source,
        })?;
    let declared = entry.size();
    budget.admit(&member_path, declared, entry.compressed_size())?;

    let mut buf = Vec::new();
    entry
        .take(declared.saturating_add(READ_OVERRUN_PROBE_BYTES))
        .read_to_end(&mut buf)
        .map_err(|source| StoreLoadError::Io {
            path: member_path.clone(),
            source,
        })?;
    check_member_read_len(&member_path, declared, buf.len())?;
    Ok(buf)
}

/// With rule 1 of [`ArchiveBudget`] holding, a stored member comes up short only
/// if its data runs off the end of the file and a forged CRC still matches.
fn check_member_read_len(path: &str, declared: u64, read: usize) -> Result<(), StoreLoadError> {
    let read = read as u64; // Widening: usize is at most 64 bits on supported targets.
    if read != declared {
        return Err(StoreLoadError::MemberLengthMismatch {
            path: path.to_string(),
            declared,
            read,
        });
    }
    Ok(())
}

fn load_window_tokens(path: &Path) -> Result<Vec<Vec<u32>>, StoreLoadError> {
    let p = path.join("window_token_lists.npz");
    let (mut archive, mut budget) = open_archive(&p)?;

    // `archive.len()` counts directory records the archive has already parsed
    // into memory, so this and the `out` below add a constant factor to that.
    // Collect and numerically sort the members so returned Vec is indexable
    // by window_id. Member names are like "0.npy", "1.npy", ...
    let mut numbered: Vec<(usize, String)> = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let name = archive
            .by_index(i)
            .map_err(|source| StoreLoadError::Zip {
                path: p.display().to_string(),
                source,
            })?
            .name()
            .to_string();
        let trimmed = name.trim_end_matches(".npy");
        if let Ok(id) = trimmed.parse::<usize>() {
            numbered.push((id, name));
        }
    }
    numbered.sort_by_key(|(i, _)| *i);

    // Invariant: the returned Vec is indexed by window_id, so member ids
    // must be exactly 0..len after the sort — a gapped archive (0, 1, 3)
    // or a duplicated id would otherwise be packed densely and silently
    // misalign every later window against boundaries/entries.
    for (expected_id, (id, name)) in numbered.iter().enumerate() {
        if *id != expected_id {
            return Err(StoreLoadError::WindowArchive(format!(
                "{}: member ids must be contiguous from 0 — expected id {expected_id}, \
                 found {id} ({name}); a gapped or duplicated archive would misalign \
                 window ids against boundaries/entries",
                p.display(),
            )));
        }
    }

    let mut out = Vec::with_capacity(numbered.len());
    for (_id, name) in numbered {
        let buf = read_member(&mut archive, &mut budget, &p, &name)?;
        let arr = npy::read_u32_1d(&buf).map_err(|source| StoreLoadError::Npy {
            path: format!("{}::{}", p.display(), name),
            source,
        })?;
        out.push(arr);
    }
    Ok(out)
}

fn load_entries(path: &Path) -> Result<Vec<VecInjectEntry>, StoreLoadError> {
    let p = path.join("entries.npz");
    let (mut archive, mut budget) = open_archive(&p)?;

    // Find the first member whose name starts with "entries" (typically
    // "entries.npy" inside the zip).
    let member_name = {
        let mut found: Option<String> = None;
        for i in 0..archive.len() {
            let n = archive
                .by_index(i)
                .map_err(|source| StoreLoadError::Zip {
                    path: p.display().to_string(),
                    source,
                })?
                .name()
                .to_string();
            if n.starts_with("entries") {
                found = Some(n);
                break;
            }
        }
        found.ok_or_else(|| StoreLoadError::MissingFile("entries.npz::entries".into()))?
    };

    let bytes = read_member(&mut archive, &mut budget, &p, &member_name)?;

    parse_structured_entries_npy(&bytes).map_err(|reason| StoreLoadError::StructuredDtype {
        path: format!("{}::{}", p.display(), member_name),
        reason,
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
fn check_entry_descr(descr: &str) -> Result<(), String> {
    let tokens = descr_quoted_tokens(descr);
    let got: Vec<(&str, &str)> = tokens
        .chunks(2)
        .filter(|c| c.len() == 2)
        .map(|c| (c[0], c[1]))
        .collect();
    if got != ENTRY_DESCR_FIELDS {
        return Err(format!(
            "structured descr mismatch: got fields {got:?}, expected {ENTRY_DESCR_FIELDS:?} \
             (decoder reads fixed offsets; field order and dtypes must match exactly)"
        ));
    }
    Ok(())
}

/// Parse a .npy file containing a structured-dtype array of `VecInjectEntry`.
///
/// Expected dtype (from the Python side): see [`ENTRY_DESCR_FIELDS`].
/// Row size: 14 bytes, no padding (numpy packs structured dtypes tightly
/// when fields are already aligned).
fn parse_structured_entries_npy(bytes: &[u8]) -> Result<Vec<VecInjectEntry>, String> {
    let (header, data_off) = npy::parse_header(bytes).map_err(|e| e.to_string())?;

    check_entry_descr(&header.descr)?;
    if header.shape.len() != 1 {
        return Err(format!(
            "expected 1D structured array, got shape {:?}",
            header.shape
        ));
    }

    const ROW_SIZE: usize = 4 + 4 + 2 + 2 + 2;
    let n = header.shape[0];
    let data = &bytes[data_off..];
    let expected = npy::checked_data_len(&header.shape, ROW_SIZE).map_err(|e| e.to_string())?;
    if data.len() != expected {
        return Err(format!(
            "data size {} != expected {} ({} rows × {} bytes)",
            data.len(),
            expected,
            n,
            ROW_SIZE,
        ));
    }

    // `n` rows of `ROW_SIZE` bytes are present in `data`, as just checked.
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * ROW_SIZE;
        out.push(VecInjectEntry {
            token_id: u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]),
            coefficient: f32::from_le_bytes([data[o + 4], data[o + 5], data[o + 6], data[o + 7]]),
            window_id: u16::from_le_bytes([data[o + 8], data[o + 9]]),
            position_in_window: u16::from_le_bytes([data[o + 10], data[o + 11]]),
            fact_id: u16::from_le_bytes([data[o + 12], data[o + 13]]),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
