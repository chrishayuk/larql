//! Locate one stored tensor by name and read its stored elements, whole or a selection of rows and columns.

use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use super::error::StoredTensorError;
use super::listing::read_index;
use crate::format::vindex3::container_path;
use crate::format::vindex3::encode::segment::{read_exact_at, read_segment_header, SegmentTensor};
use crate::format::vindex3::opplan::exec::operands::widen;
use crate::format::vindex3::represent::codec::codecs::float::FloatDtype;
use crate::format::vindex3::weight_patch::{TensorName, WeightPatchError};

/// The float element types a stored tensor can be read as, by their segment label.
const READABLE_DTYPES: [FloatDtype; 3] = [FloatDtype::Bf16, FloatDtype::F16, FloatDtype::F32];

/// How many bytes a whole-tensor read hands its sink at a time; a multiple of every element width, so no element
/// is split across two chunks.
const READ_CHUNK_BYTES: usize = 1 << 20;

/// The axis `ROWS` selects along.
const ROW_AXIS: usize = 0;
/// The fewest axes a tensor needs before it has columns (the last axis).
const MIN_AXES_FOR_COLUMNS: usize = 2;

/// One stored tensor, located: where its bytes are and what they are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredTensor {
    /// The representation that holds it.
    pub representation: String,
    /// Whether that representation is canonical.
    pub canonical: bool,
    /// The name in the segment's tensor table (bare).
    pub tensor: String,
    pub dtype: FloatDtype,
    pub shape: Vec<usize>,
    segment: PathBuf,
    /// The tensor's first byte in the segment file.
    start: u64,
}

/// Which rows (first axis) and columns (last axis) of a tensor to read, in the order given, and the shape of what
/// is read. `None` selects the whole axis. Outside this crate it is made only by [`StoredTensor::select`], which checks
/// it against the tensor, so a read never addresses outside it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Selection {
    pub rows: Option<Vec<usize>>,
    pub columns: Option<Vec<usize>>,
    pub shape: Vec<usize>,
}

/// Find the tensor `name` names in the container at `root`: a bare name in the one canonical representation that
/// holds it, a qualified `<representation>::<tensor>` in the representation it names, canonical or derived. Only
/// the tensor tables are read.
pub fn locate_stored_tensor(root: &Path, name: &str) -> Result<StoredTensor, StoredTensorError> {
    let index = read_index(root)?;
    let wanted = TensorName::parse(name);
    let mut holders: Vec<(String, bool, PathBuf, u64, SegmentTensor)> = Vec::new();
    for (id, entry) in &index.representations {
        let canonical = entry.compiled_from.is_none();
        let in_scope = match wanted.representation {
            Some(representation) => representation == id,
            None => canonical,
        };
        if !in_scope {
            continue;
        }
        let segment = container_path::resolve(root, &entry.segment)?;
        let (header, payload_start) = read_segment_header(&segment)?;
        if let Some(t) = header.tensors.into_iter().find(|t| t.name == wanted.tensor) {
            holders.push((id.clone(), canonical, segment, payload_start, t));
        }
    }
    let (representation, canonical, segment, payload_start, table) = match holders.len() {
        0 if wanted.representation.is_some() => {
            return Err(StoredTensorError::UnknownQualified(name.to_string()))
        }
        0 => return Err(StoredTensorError::UnknownBare(name.to_string())),
        1 => holders.remove(0),
        _ => {
            return Err(StoredTensorError::Ambiguous(
                WeightPatchError::AmbiguousTensor {
                    tensor: name.to_string(),
                    holders: holders.into_iter().map(|h| h.0).collect(),
                },
            ))
        }
    };
    let dtype = READABLE_DTYPES
        .into_iter()
        .find(|d| d.label() == table.dtype)
        .ok_or_else(|| StoredTensorError::PackedDtype {
            tensor: name.to_string(),
            dtype: table.dtype.clone(),
        })?;
    let elements: u64 = table.shape.iter().map(|d| *d as u64).product();
    let expected = elements * dtype.width_bytes() as u64;
    if table.len != expected {
        return Err(StoredTensorError::Malformed {
            tensor: name.to_string(),
            why: format!(
                "its table says {} bytes; {:?} {} elements are {expected}",
                table.len,
                table.shape,
                dtype.label()
            ),
        });
    }
    Ok(StoredTensor {
        representation,
        canonical,
        tensor: table.name,
        dtype,
        shape: table.shape,
        segment,
        start: payload_start + table.offset,
    })
}

/// Check one axis's selection: non-empty, in range, each index once.
fn check_axis(
    tensor: &str,
    axis: &str,
    picked: &[usize],
    extent: usize,
) -> Result<(), StoredTensorError> {
    let refuse = |why: String| StoredTensorError::Selection {
        tensor: tensor.to_string(),
        why,
    };
    if picked.is_empty() {
        return Err(refuse(format!("an empty {axis} list")));
    }
    let mut seen = BTreeSet::new();
    for &i in picked {
        if i >= extent {
            return Err(refuse(format!("{axis} {i} is outside 0..{extent}")));
        }
        if !seen.insert(i) {
            return Err(refuse(format!("{axis} {i} is listed twice")));
        }
    }
    Ok(())
}

impl StoredTensor {
    /// How many elements the tensor holds.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// The bytes of one row (one index of the first axis).
    fn row_bytes(&self) -> usize {
        self.shape.iter().skip(1).product::<usize>() * self.dtype.width_bytes()
    }

    /// Check a selection against the tensor and give the shape it reads. Rows are the first axis; columns are the
    /// last axis, so a 1-D tensor (a norm gain) has rows and no columns.
    pub fn select(
        &self,
        rows: Option<&[usize]>,
        columns: Option<&[usize]>,
    ) -> Result<Selection, StoredTensorError> {
        let mut shape = self.shape.clone();
        if let Some(rows) = rows {
            let Some(&extent) = self.shape.get(ROW_AXIS) else {
                return Err(StoredTensorError::Selection {
                    tensor: self.tensor.clone(),
                    why: "a scalar has no rows".into(),
                });
            };
            check_axis(&self.tensor, "row", rows, extent)?;
            shape[ROW_AXIS] = rows.len();
        }
        if let Some(columns) = columns {
            if self.shape.len() < MIN_AXES_FOR_COLUMNS {
                return Err(StoredTensorError::Selection {
                    tensor: self.tensor.clone(),
                    why: format!(
                        "a tensor of shape {:?} has no columns; ROWS selects along its first axis",
                        self.shape
                    ),
                });
            }
            let last = self.shape.len() - 1;
            check_axis(&self.tensor, "column", columns, self.shape[last])?;
            shape[last] = columns.len();
        }
        Ok(Selection {
            rows: rows.map(<[usize]>::to_vec),
            columns: columns.map(<[usize]>::to_vec),
            shape,
        })
    }

    /// Read the selected elements as stored (little-endian, the tensor's own dtype), row-major in the selection's
    /// order, handing them to `sink` a piece at a time. Pieces never split an element. The whole tensor is never
    /// held in memory at once.
    pub fn read(
        &self,
        selection: &Selection,
        sink: &mut dyn FnMut(&[u8]) -> std::io::Result<()>,
    ) -> Result<(), StoredTensorError> {
        let file = File::open(&self.segment)?;
        if selection.rows.is_none() && selection.columns.is_none() {
            let total = self.element_count() * self.dtype.width_bytes();
            let mut buffer = vec![0u8; READ_CHUNK_BYTES.min(total)];
            let mut done = 0usize;
            while done < total {
                let take = READ_CHUNK_BYTES.min(total - done);
                read_exact_at(&file, &mut buffer[..take], self.start + done as u64)?;
                sink(&buffer[..take])?;
                done += take;
            }
            return Ok(());
        }
        let row_bytes = self.row_bytes();
        let all_rows: Vec<usize>;
        let rows: &[usize] = match &selection.rows {
            Some(rows) => rows,
            None => {
                all_rows = (0..self.shape[ROW_AXIS]).collect();
                &all_rows
            }
        };
        let width = self.dtype.width_bytes();
        let mut row = vec![0u8; row_bytes];
        let mut picked = Vec::new();
        for &r in rows {
            read_exact_at(&file, &mut row, self.start + (r * row_bytes) as u64)?;
            match &selection.columns {
                None => sink(&row)?,
                Some(columns) => {
                    let line_bytes = self.shape[self.shape.len() - 1] * width;
                    picked.clear();
                    for line in row.chunks_exact(line_bytes) {
                        for &c in columns {
                            picked.extend_from_slice(&line[c * width..(c + 1) * width]);
                        }
                    }
                    sink(&picked)?;
                }
            }
        }
        Ok(())
    }

    /// Stored elements of this tensor's dtype widened to f32. Exact: every BF16, F16 and F32 value is an f32.
    pub fn widen(&self, stored: &[u8]) -> Result<Vec<f32>, StoredTensorError> {
        Ok(widen(self.dtype.label(), stored, &self.tensor)?)
    }
}
