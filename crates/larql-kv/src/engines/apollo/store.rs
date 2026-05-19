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
    let mut out = Vec::with_capacity(num_windows);
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

fn load_window_tokens(path: &Path) -> Result<Vec<Vec<u32>>, StoreLoadError> {
    let p = path.join("window_token_lists.npz");
    let file = std::fs::File::open(&p).map_err(|source| StoreLoadError::Io {
        path: p.display().to_string(),
        source,
    })?;
    let mut archive = zip::ZipArchive::new(file).map_err(|source| StoreLoadError::Zip {
        path: p.display().to_string(),
        source,
    })?;

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

    let mut out = Vec::with_capacity(numbered.len());
    for (_id, name) in numbered {
        let mut entry = archive
            .by_name(&name)
            .map_err(|source| StoreLoadError::Zip {
                path: format!("{}::{}", p.display(), name),
                source,
            })?;
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry
            .read_to_end(&mut buf)
            .map_err(|source| StoreLoadError::Io {
                path: format!("{}::{}", p.display(), name),
                source,
            })?;
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
    let file = std::fs::File::open(&p).map_err(|source| StoreLoadError::Io {
        path: p.display().to_string(),
        source,
    })?;
    let mut archive = zip::ZipArchive::new(file).map_err(|source| StoreLoadError::Zip {
        path: p.display().to_string(),
        source,
    })?;

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

    let mut entry = archive
        .by_name(&member_name)
        .map_err(|source| StoreLoadError::Zip {
            path: format!("{}::{}", p.display(), member_name),
            source,
        })?;
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry
        .read_to_end(&mut bytes)
        .map_err(|source| StoreLoadError::Io {
            path: member_name.clone(),
            source,
        })?;

    parse_structured_entries_npy(&bytes).map_err(|reason| StoreLoadError::StructuredDtype {
        path: format!("{}::{}", p.display(), member_name),
        reason,
    })
}

/// Parse a .npy file containing a structured-dtype array of `VecInjectEntry`.
///
/// Expected dtype (from the Python side):
///   (token_id: u32, coefficient: f32, window_id: u16,
///    position_in_window: u16, fact_id: u16)
/// Row size: 14 bytes, no padding (numpy packs structured dtypes tightly
/// when fields are already aligned).
fn parse_structured_entries_npy(bytes: &[u8]) -> Result<Vec<VecInjectEntry>, String> {
    let (header, data_off) = npy::parse_header(bytes).map_err(|e| e.to_string())?;

    for field in [
        "token_id",
        "coefficient",
        "window_id",
        "position_in_window",
        "fact_id",
    ] {
        if !header.descr.contains(field) {
            return Err(format!(
                "missing field '{field}' in descr: {}",
                header.descr
            ));
        }
    }
    if header.shape.len() != 1 {
        return Err(format!(
            "expected 1D structured array, got shape {:?}",
            header.shape
        ));
    }

    const ROW_SIZE: usize = 4 + 4 + 2 + 2 + 2;
    let n = header.shape[0];
    let data = &bytes[data_off..];
    let expected = n * ROW_SIZE;
    if data.len() != expected {
        return Err(format!(
            "data size {} != expected {} ({} rows × {} bytes)",
            data.len(),
            expected,
            n,
            ROW_SIZE,
        ));
    }

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
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;
    use zip::write::SimpleFileOptions;
    use zip::ZipWriter;

    // ── Synthetic .npy / .npz builders ────────────────────────────────────

    /// Build a minimal v1.0 .npy header for the given dtype and shape.
    fn npy_header(dtype: &str, shape: &[usize]) -> Vec<u8> {
        let shape_str = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", parts.join(", "))
        };
        let header =
            format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape_str}, }}");
        let mut padded = header.into_bytes();
        let total = 10 + padded.len();
        let pad_to = (total + 63) & !63;
        while 10 + padded.len() + 1 < pad_to {
            padded.push(b' ');
        }
        padded.push(b'\n');
        padded
    }

    fn synth_f32_npy(values: &[f32]) -> Vec<u8> {
        let header = npy_header("<f4", &[values.len()]);
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(&header);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn synth_f32_3d_npy(values: &[f32], shape: &[usize]) -> Vec<u8> {
        let header = npy_header("<f4", shape);
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(&header);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn synth_u32_npy(values: &[u32]) -> Vec<u8> {
        let header = npy_header("<u4", &[values.len()]);
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(&header);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Build a structured-dtype `.npy` blob with N VecInjectEntry rows.
    fn synth_structured_entries_npy(rows: &[VecInjectEntry]) -> Vec<u8> {
        let descr = "[('token_id', '<u4'), ('coefficient', '<f4'), \
                     ('window_id', '<u2'), ('position_in_window', '<u2'), \
                     ('fact_id', '<u2')]";
        let header = format!(
            "{{'descr': {descr}, 'fortran_order': False, 'shape': ({},), }}",
            rows.len()
        );
        let mut padded = header.into_bytes();
        let total = 10 + padded.len();
        let pad_to = (total + 63) & !63;
        while 10 + padded.len() + 1 < pad_to {
            padded.push(b' ');
        }
        padded.push(b'\n');
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(padded.len() as u16).to_le_bytes());
        out.extend_from_slice(&padded);
        for r in rows {
            out.extend_from_slice(&r.token_id.to_le_bytes());
            out.extend_from_slice(&r.coefficient.to_le_bytes());
            out.extend_from_slice(&r.window_id.to_le_bytes());
            out.extend_from_slice(&r.position_in_window.to_le_bytes());
            out.extend_from_slice(&r.fact_id.to_le_bytes());
        }
        out
    }

    fn synth_npz<I: IntoIterator<Item = (String, Vec<u8>)>>(members: I) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zw = ZipWriter::new(cursor);
            let opts: SimpleFileOptions =
                SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
            for (name, data) in members {
                zw.start_file(&name, opts).unwrap();
                zw.write_all(&data).unwrap();
            }
            zw.finish().unwrap();
        }
        buf
    }

    fn write_minimal_store(
        dir: &Path,
        num_windows: usize,
        hidden: usize,
        window_size: usize,
        entries: &[VecInjectEntry],
    ) {
        // manifest.json
        let manifest = StoreManifest {
            version: 1,
            num_entries: entries.len(),
            num_windows,
            num_tokens: num_windows * window_size,
            entries_per_window: entries.len().checked_div(num_windows).unwrap_or(0),
            crystal_layer: 29,
            window_size,
            arch_config: ArchConfig::default(),
            has_residuals: false,
        };
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        // boundaries/window_NNN.npy — single (hidden,) per file
        let bdir = dir.join("boundaries");
        std::fs::create_dir_all(&bdir).unwrap();
        for w in 0..num_windows {
            let vals: Vec<f32> = (0..hidden).map(|i| (w * 100 + i) as f32).collect();
            std::fs::write(
                bdir.join(format!("window_{:03}.npy", w)),
                synth_f32_npy(&vals),
            )
            .unwrap();
        }

        // window_token_lists.npz: members "0.npy", "1.npy", ...
        let token_members: Vec<(String, Vec<u8>)> = (0..num_windows)
            .map(|w| {
                let toks: Vec<u32> = (0..window_size as u32)
                    .map(|i| w as u32 * 1000 + i)
                    .collect();
                (format!("{}.npy", w), synth_u32_npy(&toks))
            })
            .collect();
        std::fs::write(dir.join("window_token_lists.npz"), synth_npz(token_members)).unwrap();

        // entries.npz: single member "entries.npy"
        let entry_blob = synth_structured_entries_npy(entries);
        std::fs::write(
            dir.join("entries.npz"),
            synth_npz(vec![("entries.npy".to_string(), entry_blob)]),
        )
        .unwrap();
    }

    // ── ArchConfig defaults ───────────────────────────────────────────────

    #[test]
    fn default_arch_config_matches_apollo11() {
        let cfg = ArchConfig::default();
        assert_eq!(cfg.retrieval_layer, 29);
        assert_eq!(cfg.query_head, 4);
        assert_eq!(cfg.injection_layer, 30);
        assert_eq!(cfg.inject_coefficient, 10.0);
    }

    // ── Load — happy path & high-level errors ─────────────────────────────

    #[test]
    fn load_missing_directory_errors() {
        let r = ApolloStore::load(Path::new("/tmp/apollo-does-not-exist-xyz"));
        assert!(matches!(r.unwrap_err(), StoreLoadError::Io { .. }));
    }

    #[test]
    fn load_minimal_synthetic_store_succeeds() {
        let dir = TempDir::new().unwrap();
        let entries = vec![
            VecInjectEntry {
                token_id: 42,
                coefficient: 1.0,
                window_id: 0,
                position_in_window: 0,
                fact_id: 0,
            },
            VecInjectEntry {
                token_id: 99,
                coefficient: 2.5,
                window_id: 1,
                position_in_window: 3,
                fact_id: 1,
            },
        ];
        write_minimal_store(dir.path(), 2, 4, 5, &entries);
        let store = ApolloStore::load(dir.path()).expect("load");
        assert_eq!(store.boundaries.len(), 2);
        assert_eq!(store.boundaries[0].len(), 4);
        assert_eq!(store.window_tokens.len(), 2);
        assert_eq!(store.window_tokens[0].len(), 5);
        assert_eq!(store.entries.len(), 2);
        assert_eq!(store.entries[0].token_id, 42);
        assert_eq!(store.entries[1].window_id, 1);
        assert_eq!(store.manifest.num_entries, 2);
        assert_eq!(store.manifest.num_windows, 2);
    }

    #[test]
    fn load_with_optional_boundary_residual() {
        let dir = TempDir::new().unwrap();
        write_minimal_store(dir.path(), 1, 4, 3, &[]);
        // Add the optional boundary_residual.npy.
        let res_blob = synth_f32_3d_npy(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        std::fs::write(dir.path().join("boundary_residual.npy"), res_blob).unwrap();
        let store = ApolloStore::load(dir.path()).unwrap();
        assert_eq!(store.boundary_residual, Some(vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn load_manifest_mismatch_on_boundary_count() {
        let dir = TempDir::new().unwrap();
        write_minimal_store(dir.path(), 2, 4, 3, &[]);
        // Tamper: rewrite manifest claiming 3 windows but only 2 boundaries on disk.
        let manifest_path = dir.path().join("manifest.json");
        let mut m: StoreManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        m.num_windows = 3;
        std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();
        // Loader will fail on missing boundaries/window_002.npy → Io
        let err = ApolloStore::load(dir.path()).unwrap_err();
        assert!(matches!(err, StoreLoadError::Io { .. }));
    }

    #[test]
    fn load_manifest_mismatch_on_entry_count() {
        let dir = TempDir::new().unwrap();
        let entries = vec![VecInjectEntry {
            token_id: 1,
            coefficient: 1.0,
            window_id: 0,
            position_in_window: 0,
            fact_id: 0,
        }];
        write_minimal_store(dir.path(), 1, 4, 3, &entries);
        let manifest_path = dir.path().join("manifest.json");
        let mut m: StoreManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        m.num_entries = 99; // disagrees with on-disk entries.npy (1)
        std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();
        let err = ApolloStore::load(dir.path()).unwrap_err();
        match err {
            StoreLoadError::ManifestMismatch(msg) => assert!(msg.contains("num_entries")),
            other => panic!("expected ManifestMismatch, got {other:?}"),
        }
    }

    /// Drive the npy error-wrap path in `load_boundaries` (lines 181-184):
    /// write a malformed window_000.npy so `npy::read_f32_1d` fails.
    #[test]
    fn load_boundaries_malformed_npy_returns_npy_error() {
        let dir = TempDir::new().unwrap();
        write_minimal_store(dir.path(), 1, 4, 3, &[]);
        // Overwrite the boundary file with garbage bytes.
        let bdir = dir.path().join("boundaries");
        std::fs::write(bdir.join("window_000.npy"), b"NOT_AN_NPY_BLOB").unwrap();
        let err = ApolloStore::load(dir.path()).unwrap_err();
        assert!(
            matches!(err, StoreLoadError::Npy { .. }),
            "expected Npy error, got {err:?}"
        );
    }

    /// Drive the `load_boundary_residual` npy error path (lines 193-196):
    /// optional file — when present and malformed, the wrap fires; when
    /// absent, `.ok()` in the caller swallows it (already covered).
    #[test]
    fn load_boundary_residual_malformed_npy_returns_none_via_ok() {
        let dir = TempDir::new().unwrap();
        write_minimal_store(dir.path(), 1, 4, 3, &[]);
        // Plant a malformed boundary_residual.npy.
        std::fs::write(dir.path().join("boundary_residual.npy"), b"bogus").unwrap();
        // Loader's `.ok()` on load_boundary_residual swallows the error;
        // the store loads successfully with `boundary_residual = None`.
        let store = ApolloStore::load(dir.path()).expect("store loads despite malformed residual");
        assert!(store.boundary_residual.is_none());
    }

    /// Drive the npz / zip error-wrap path in `load_window_tokens`
    /// (lines 206-209): write a non-zip file as window_token_lists.npz.
    #[test]
    fn load_window_tokens_malformed_npz_returns_zip_error() {
        let dir = TempDir::new().unwrap();
        write_minimal_store(dir.path(), 1, 4, 3, &[]);
        std::fs::write(dir.path().join("window_token_lists.npz"), b"NOT_A_ZIP").unwrap();
        let err = ApolloStore::load(dir.path()).unwrap_err();
        assert!(
            matches!(err, StoreLoadError::Zip { .. }),
            "expected Zip error on malformed npz, got {err:?}"
        );
    }

    #[test]
    fn load_invalid_manifest_json_returns_json_error() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("manifest.json"), b"not a json").unwrap();
        let err = ApolloStore::load(dir.path()).unwrap_err();
        assert!(matches!(err, StoreLoadError::Json(_)));
    }

    // ── In-memory accessors ───────────────────────────────────────────────

    #[test]
    fn total_bytes_sums_components() {
        let store = ApolloStore {
            manifest: dummy_manifest(2),
            boundaries: vec![vec![0.0; 4], vec![0.0; 4]],
            boundary_residual: Some(vec![0.0; 4]),
            window_tokens: vec![vec![0u32; 3], vec![0u32; 3]],
            entries: vec![
                VecInjectEntry {
                    token_id: 0,
                    coefficient: 0.0,
                    window_id: 0,
                    position_in_window: 0,
                    fact_id: 0,
                };
                5
            ],
        };
        let entry_size = std::mem::size_of::<VecInjectEntry>();
        let expected = 4 * 4 * 2 + 4 * 4 + 3 * 4 * 2 + 5 * entry_size;
        assert_eq!(store.total_bytes(), expected);
    }

    #[test]
    fn hidden_size_is_first_boundary_len() {
        let store = ApolloStore {
            manifest: dummy_manifest(1),
            boundaries: vec![vec![0.0; 16]],
            boundary_residual: None,
            window_tokens: vec![vec![]],
            entries: vec![],
        };
        assert_eq!(store.hidden_size(), 16);

        let empty = ApolloStore {
            manifest: dummy_manifest(0),
            boundaries: vec![],
            boundary_residual: None,
            window_tokens: vec![],
            entries: vec![],
        };
        assert_eq!(empty.hidden_size(), 0);
    }

    fn dummy_manifest(num_windows: usize) -> StoreManifest {
        StoreManifest {
            version: 1,
            num_entries: 0,
            num_windows,
            num_tokens: 0,
            entries_per_window: 0,
            crystal_layer: 29,
            window_size: 0,
            arch_config: ArchConfig::default(),
            has_residuals: false,
        }
    }

    // ── parse_structured_entries_npy direct tests ─────────────────────────

    #[test]
    fn structured_npy_zero_rows() {
        let blob = synth_structured_entries_npy(&[]);
        let parsed = parse_structured_entries_npy(&blob).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn structured_npy_three_rows_roundtrip() {
        let rows = vec![
            VecInjectEntry {
                token_id: 1,
                coefficient: 0.5,
                window_id: 10,
                position_in_window: 7,
                fact_id: 3,
            },
            VecInjectEntry {
                token_id: 2,
                coefficient: -1.5,
                window_id: 11,
                position_in_window: 0,
                fact_id: 4,
            },
            VecInjectEntry {
                token_id: u32::MAX,
                coefficient: 12345.0,
                window_id: u16::MAX,
                position_in_window: u16::MAX,
                fact_id: u16::MAX,
            },
        ];
        let blob = synth_structured_entries_npy(&rows);
        let parsed = parse_structured_entries_npy(&blob).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].token_id, 1);
        assert_eq!(parsed[0].coefficient, 0.5);
        assert_eq!(parsed[1].coefficient, -1.5);
        assert_eq!(parsed[2].token_id, u32::MAX);
        assert_eq!(parsed[2].fact_id, u16::MAX);
    }

    #[test]
    fn structured_npy_missing_field_errors() {
        // Build a structured npy whose descr is missing 'fact_id'.
        let descr = "[('token_id', '<u4'), ('coefficient', '<f4'), \
                     ('window_id', '<u2'), ('position_in_window', '<u2')]";
        let header = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': (1,), }}");
        let mut padded = header.into_bytes();
        let total = 10 + padded.len();
        let pad_to = (total + 63) & !63;
        while 10 + padded.len() + 1 < pad_to {
            padded.push(b' ');
        }
        padded.push(b'\n');
        let mut blob = Vec::new();
        blob.extend_from_slice(b"\x93NUMPY");
        blob.push(1);
        blob.push(0);
        blob.extend_from_slice(&(padded.len() as u16).to_le_bytes());
        blob.extend_from_slice(&padded);
        // 12 bytes: u32 + f32 + u16 + u16
        blob.extend_from_slice(&[0u8; 12]);
        let err = parse_structured_entries_npy(&blob).unwrap_err();
        assert!(err.contains("fact_id"));
    }

    #[test]
    fn structured_npy_2d_shape_errors() {
        let descr = "[('token_id', '<u4'), ('coefficient', '<f4'), \
                     ('window_id', '<u2'), ('position_in_window', '<u2'), \
                     ('fact_id', '<u2')]";
        let header = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': (1, 1), }}");
        let mut padded = header.into_bytes();
        let total = 10 + padded.len();
        let pad_to = (total + 63) & !63;
        while 10 + padded.len() + 1 < pad_to {
            padded.push(b' ');
        }
        padded.push(b'\n');
        let mut blob = Vec::new();
        blob.extend_from_slice(b"\x93NUMPY");
        blob.push(1);
        blob.push(0);
        blob.extend_from_slice(&(padded.len() as u16).to_le_bytes());
        blob.extend_from_slice(&padded);
        blob.extend_from_slice(&[0u8; 14]);
        let err = parse_structured_entries_npy(&blob).unwrap_err();
        assert!(err.contains("expected 1D"));
    }

    #[test]
    fn structured_npy_data_size_mismatch_errors() {
        // Build a blob declaring 2 rows but provide only 1 row of data.
        let descr = "[('token_id', '<u4'), ('coefficient', '<f4'), \
                     ('window_id', '<u2'), ('position_in_window', '<u2'), \
                     ('fact_id', '<u2')]";
        let header = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': (2,), }}");
        let mut padded = header.into_bytes();
        let total = 10 + padded.len();
        let pad_to = (total + 63) & !63;
        while 10 + padded.len() + 1 < pad_to {
            padded.push(b' ');
        }
        padded.push(b'\n');
        let mut blob = Vec::new();
        blob.extend_from_slice(b"\x93NUMPY");
        blob.push(1);
        blob.push(0);
        blob.extend_from_slice(&(padded.len() as u16).to_le_bytes());
        blob.extend_from_slice(&padded);
        blob.extend_from_slice(&[0u8; 14]); // only 1 row × 14 bytes
        let err = parse_structured_entries_npy(&blob).unwrap_err();
        assert!(err.contains("data size"));
    }

    #[test]
    fn structured_npy_bad_magic_errors() {
        let blob = b"not a valid npy".to_vec();
        let err = parse_structured_entries_npy(&blob).unwrap_err();
        assert!(!err.is_empty());
    }
}
