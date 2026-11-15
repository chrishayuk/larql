//! `EXPORT WEIGHT "<tensor>" [ROWS (…)] [COLUMNS (…)] TO "<file>.npy" [AS BITS]`: one stored tensor, or a
//! selection of it, written as a NumPy array with a sidecar that says exactly where the numbers came from.
//!
//! The values are the container's stored bytes. A staged edit or an applied patch lives in the session, not in the
//! container, and is not part of an export; the sidecar says so (`"reads": "stored"`). Nothing is overwritten: an
//! output or sidecar that exists is refused before anything is written, and a failed write removes what it wrote.

use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::represent::codec::codecs::float::FloatDtype;
use larql_vindex::format::vindex3::stored_tensor::{locate_stored_tensor, Selection, StoredTensor};

use super::npy::{npy_sha256, NpyDescr, NpyWriter};
use super::npy_output::{
    container_index, container_path, index_list, literal, output_paths, remove_partial,
    write_sidecar, LARQL_VERSION,
};
use crate::error::LqlError;
use crate::executor::Session;

const STATEMENT: &str = "EXPORT WEIGHT";
/// The sidecar's schema id.
const SIDECAR_SCHEMA: &str = "larql-export-weight/v1";
/// What an export reads: the container's stored bytes, never session state.
const READS_STORED: &str = "stored";

fn refused(e: impl std::fmt::Display) -> LqlError {
    super::npy_output::refused(STATEMENT, e)
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

/// The selection of one stored tensor an export reads, located and checked, with the container identity its
/// provenance records. `EXPORT WEIGHT` streams it into a file; [`Session::read_weight`] reads it into memory.
struct LocatedWeight {
    root: PathBuf,
    index: larql_vindex::format::vindex3::Vindex3Index,
    index_sha256: String,
    stored: StoredTensor,
    selection: Selection,
    descr: NpyDescr,
    as_stored: bool,
}

impl LocatedWeight {
    fn locate(
        root: &Path,
        tensor: &str,
        rows: Option<&[usize]>,
        columns: Option<&[usize]>,
        bits: bool,
    ) -> Result<Self, LqlError> {
        let (index, index_sha256) = container_index(root)?;
        let stored = locate_stored_tensor(root, tensor).map_err(refused)?;
        let selection = stored.select(rows, columns).map_err(refused)?;
        let descr = descr(stored.dtype, bits);
        let as_stored = bits || stored.dtype == FloatDtype::F32;
        Ok(Self {
            root: root.to_path_buf(),
            index,
            index_sha256,
            stored,
            selection,
            descr,
            as_stored,
        })
    }

    /// Stream the selection into `write`: stored bytes as they are, or widened exactly to f32.
    fn stream(&self, mut write: impl FnMut(&[u8]) -> std::io::Result<()>) -> Result<(), LqlError> {
        let stored = &self.stored;
        let as_stored = self.as_stored;
        let mut sink = |chunk: &[u8]| -> std::io::Result<()> {
            if as_stored {
                return write(chunk);
            }
            let values = stored.widen(chunk).map_err(std::io::Error::other)?;
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            write(&bytes)
        };
        stored
            .read(&self.selection, &mut sink)
            .map_err(|e| LqlError::exec("read the stored tensor", e))
    }

    /// The sidecar's fields: `statement` is the statement that wrote the file (null where none did).
    fn provenance(&self, statement: serde_json::Value, npy_sha256: &str) -> serde_json::Value {
        let stored = &self.stored;
        serde_json::json!({
            "schema": SIDECAR_SCHEMA,
            "statement": statement,
            "container": container_path(&self.root),
            "index_sha256": self.index_sha256,
            "model": self.index.model,
            "reads": READS_STORED,
            "representation": stored.representation,
            "canonical": stored.canonical,
            "tensor": stored.tensor,
            "stored_dtype": stored.dtype.label(),
            "exported_dtype": self.descr.label(),
            "shape": stored.shape,
            "rows": self.selection.rows,
            "columns": self.selection.columns,
            "exported_shape": self.selection.shape,
            "npy_sha256": npy_sha256,
            "larql_version": LARQL_VERSION,
        })
    }
}

/// One stored tensor, or a selection of it, read into memory: what `EXPORT WEIGHT` writes, without the file.
#[derive(Debug, Clone)]
pub struct WeightRead {
    /// The element type: f32 values, or the stored element bits.
    pub descr: NpyDescr,
    /// The selection's shape.
    pub shape: Vec<usize>,
    /// The elements, little-endian, C order.
    pub bytes: Vec<u8>,
    /// The fields `EXPORT WEIGHT`'s sidecar records for the same selection, with `statement` null and
    /// `npy_sha256` the digest of the `.npy` file the statement would write.
    pub provenance: serde_json::Value,
}

impl Session {
    /// The typed operation behind `EXPORT WEIGHT`: the selection of a stored tensor of the bound container, in
    /// memory, with its provenance. The values are the container's stored bytes, as the statement exports them.
    pub fn read_weight(
        &self,
        tensor: &str,
        rows: Option<&[usize]>,
        columns: Option<&[usize]>,
        bits: bool,
    ) -> Result<WeightRead, LqlError> {
        let root = self.v3_container_root(STATEMENT)?;
        let located = LocatedWeight::locate(root, tensor, rows, columns, bits)?;
        let mut bytes = Vec::new();
        located.stream(|chunk| {
            bytes.extend_from_slice(chunk);
            Ok(())
        })?;
        let npy_sha256 = npy_sha256(located.descr, &located.selection.shape, &bytes)
            .map_err(|e| LqlError::exec("describe the array", e))?;
        Ok(WeightRead {
            descr: located.descr,
            shape: located.selection.shape.clone(),
            provenance: located.provenance(serde_json::Value::Null, &npy_sha256),
            bytes,
        })
    }

    pub(crate) fn exec_export_weight(
        &self,
        tensor: &str,
        rows: Option<&[usize]>,
        columns: Option<&[usize]>,
        output: &str,
        bits: bool,
    ) -> Result<Vec<String>, LqlError> {
        let root = self.v3_container_root(STATEMENT)?.to_path_buf();
        let (npy_path, sidecar_path) = output_paths(STATEMENT, output)?;
        let located = LocatedWeight::locate(&root, tensor, rows, columns, bits)?;
        let (stored, selection, descr) = (&located.stored, &located.selection, located.descr);

        let mut writer = NpyWriter::create_new(&npy_path, descr, &selection.shape)
            .map_err(|e| LqlError::exec(format!("create {}", npy_path.display()), e))?;
        if let Err(e) = located.stream(|chunk| writer.write(chunk)) {
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

        let statement = statement_text(tensor, rows, columns, output, bits);
        let sidecar = located.provenance(statement.into(), &npy_sha256);
        write_sidecar(&sidecar_path, &sidecar, &[&npy_path])?;

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
