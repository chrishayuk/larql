//! `EXPORT WEIGHT "<tensor>" [ROWS (…)] [COLUMNS (…)] TO "<file>.npy" [AS BITS]`: one stored tensor, or a
//! selection of it, written as a NumPy array with a sidecar that says exactly where the numbers came from.
//!
//! The values are the container's stored bytes. A staged edit or an applied patch lives in the session, not in the
//! container, and is not part of an export; the sidecar says so (`"reads": "stored"`). Nothing is overwritten: an
//! output or sidecar that exists is refused before anything is written, and a failed write removes what it wrote.

use std::path::{Path, PathBuf};

use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::vindex3::represent::codec::codecs::float::FloatDtype;
use larql_vindex::format::vindex3::stored_tensor::{locate_stored_tensor, Selection, StoredTensor};
use sha2::{Digest, Sha256};

use super::npy::{NpyDescr, NpyWriter};
use crate::error::LqlError;
use crate::executor::Session;

const STATEMENT: &str = "EXPORT WEIGHT";
/// The extension an output must carry.
const NPY_SUFFIX: &str = ".npy";
/// Appended to the output's name to name the sidecar.
const SIDECAR_SUFFIX: &str = ".json";
/// The sidecar's schema id.
const SIDECAR_SCHEMA: &str = "larql-export-weight/v1";
/// What an export reads: the container's stored bytes, never session state.
const READS_STORED: &str = "stored";
/// The LARQL version that wrote the export (the workspace version).
const LARQL_VERSION: &str = env!("CARGO_PKG_VERSION");

fn refused(e: impl std::fmt::Display) -> LqlError {
    LqlError::Execution(format!("{STATEMENT} refused: {e}"))
}

/// A string as an LQL literal.
fn literal(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}

fn index_list(indices: &[usize]) -> String {
    indices
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The statement as LQL text, in its canonical form: what the sidecar records.
fn statement_text(
    tensor: &str,
    rows: Option<&[usize]>,
    columns: Option<&[usize]>,
    output: &str,
    bits: bool,
) -> String {
    let mut text = format!("{STATEMENT} {}", literal(tensor));
    if let Some(rows) = rows {
        text.push_str(&format!(" ROWS ({})", index_list(rows)));
    }
    if let Some(columns) = columns {
        text.push_str(&format!(" COLUMNS ({})", index_list(columns)));
    }
    text.push_str(&format!(" TO {}", literal(output)));
    if bits {
        text.push_str(" AS BITS");
    }
    text.push(';');
    text
}

/// The array type an export writes: f32 values, or the stored element bits.
fn descr(dtype: FloatDtype, bits: bool) -> NpyDescr {
    match (bits, dtype) {
        (false, _) => NpyDescr::F4,
        (true, FloatDtype::Bf16 | FloatDtype::F16) => NpyDescr::U2,
        (true, FloatDtype::F32) => NpyDescr::U4,
    }
}

/// Stream the selection into `writer`: stored bytes as they are, or widened exactly to f32.
fn write_body(
    stored: &StoredTensor,
    selection: &Selection,
    as_stored: bool,
    writer: &mut NpyWriter,
) -> Result<(), LqlError> {
    let mut sink = |chunk: &[u8]| -> std::io::Result<()> {
        if as_stored {
            return writer.write(chunk);
        }
        let values = stored.widen(chunk).map_err(std::io::Error::other)?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write(&bytes)
    };
    stored
        .read(selection, &mut sink)
        .map_err(|e| LqlError::exec("read the stored tensor", e))
}

/// Remove what a failed export wrote. A file that cannot be removed is reported, never ignored silently.
fn remove_partial(paths: &[&Path], cause: LqlError) -> LqlError {
    let leftovers: Vec<String> = paths
        .iter()
        .filter(|p| p.exists() && std::fs::remove_file(p).is_err())
        .map(|p| p.display().to_string())
        .collect();
    if leftovers.is_empty() {
        cause
    } else {
        LqlError::Execution(format!(
            "{cause}; could not remove the partial output {}",
            leftovers.join(", ")
        ))
    }
}

impl Session {
    pub(crate) fn exec_export_weight(
        &self,
        tensor: &str,
        rows: Option<&[usize]>,
        columns: Option<&[usize]>,
        output: &str,
        bits: bool,
    ) -> Result<Vec<String>, LqlError> {
        let root = self.v3_container_root(STATEMENT)?.to_path_buf();
        let named_npy = Path::new(output)
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.len() > NPY_SUFFIX.len() && name.ends_with(NPY_SUFFIX));
        if !named_npy {
            return Err(refused(format!(
                "the output `{output}` is not a {NPY_SUFFIX} file; name it `<file>{NPY_SUFFIX}`"
            )));
        }
        let npy_path = PathBuf::from(output);
        let sidecar_path = PathBuf::from(format!("{output}{SIDECAR_SUFFIX}"));
        for path in [&npy_path, &sidecar_path] {
            // `symlink_metadata` sees a dangling symlink too: nothing at the path may be replaced.
            if path.symlink_metadata().is_ok() {
                return Err(refused(format!(
                    "refusing to write `{}`: it already exists, and an export never overwrites",
                    path.display()
                )));
            }
        }

        let index_bytes = std::fs::read(root.join(INDEX_JSON))
            .map_err(|e| LqlError::exec("read container index", e))?;
        let index: larql_vindex::format::vindex3::Vindex3Index =
            serde_json::from_slice(&index_bytes)
                .map_err(|e| LqlError::exec("parse container index", e))?;
        let index_sha256 = format!("{:x}", Sha256::digest(&index_bytes));

        let stored = locate_stored_tensor(&root, tensor).map_err(refused)?;
        let selection = stored.select(rows, columns).map_err(refused)?;
        let descr = descr(stored.dtype, bits);
        let as_stored = bits || stored.dtype == FloatDtype::F32;

        let mut writer = NpyWriter::create_new(&npy_path, descr, &selection.shape)
            .map_err(|e| LqlError::exec(format!("create {}", npy_path.display()), e))?;
        if let Err(e) = write_body(&stored, &selection, as_stored, &mut writer) {
            drop(writer);
            return Err(remove_partial(&[&npy_path], e));
        }
        let npy_sha256 = match writer.finish() {
            Ok(digest) => digest,
            Err(e) => {
                let cause = LqlError::exec(format!("write {}", npy_path.display()), e);
                return Err(remove_partial(&[&npy_path], cause));
            }
        };

        let container = std::fs::canonicalize(&root).unwrap_or_else(|_| root.clone());
        let sidecar = serde_json::json!({
            "schema": SIDECAR_SCHEMA,
            "statement": statement_text(tensor, rows, columns, output, bits),
            "container": container.display().to_string(),
            "index_sha256": index_sha256,
            "model": index.model,
            "reads": READS_STORED,
            "representation": stored.representation,
            "canonical": stored.canonical,
            "tensor": stored.tensor,
            "stored_dtype": stored.dtype.label(),
            "exported_dtype": descr.label(),
            "shape": stored.shape,
            "rows": selection.rows,
            "columns": selection.columns,
            "exported_shape": selection.shape,
            "npy_sha256": npy_sha256,
            "larql_version": LARQL_VERSION,
        });
        let written = serde_json::to_vec_pretty(&sidecar)
            .map_err(std::io::Error::other)
            .and_then(|text| {
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&sidecar_path)?;
                file.write_all(&text)?;
                file.sync_all()
            });
        if let Err(e) = written {
            let cause = LqlError::exec(format!("write {}", sidecar_path.display()), e);
            return Err(remove_partial(&[&npy_path, &sidecar_path], cause));
        }

        Ok(vec![
            format!(
                "Exported {}::{} ({} {:?}) as {} {:?} to {}",
                stored.representation,
                stored.tensor,
                stored.dtype.label(),
                stored.shape,
                descr.label(),
                selection.shape,
                npy_path.display(),
            ),
            format!("  sha256:  {npy_sha256}"),
            format!("  sidecar: {}", sidecar_path.display()),
            "  reads the container's stored weights; edits staged or applied in this session are not included"
                .into(),
        ])
    }
}
