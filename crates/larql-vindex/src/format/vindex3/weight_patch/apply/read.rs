//! Reading one BF16 matrix of a container, and what the container says about it.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::super::error::WeightPatchError;
use super::locate::locate;
use super::{DeclaredPatch, ResultEdit, ResultPatch, BF16_BYTES};
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::graph::{ObjectKind, SystemGraph};
use crate::format::vindex3::index::Vindex3Index;

/// The `[rows, cols]` of BF16 matrix `tensor` of the container at `src`, from its segment header alone (no
/// payload is read). Refuses what [`read_bf16_tensor`] refuses.
pub fn tensor_shape(src: &Path, tensor: &str) -> Result<[usize; 2], WeightPatchError> {
    let raw = std::fs::read_to_string(src.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&raw)
        .map_err(|e| crate::error::VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    let probe = DeclaredPatch::Results(ResultPatch {
        tensor: tensor.to_string(),
        edits: Vec::new(),
    });
    let at = locate(src, &index, &probe)?;
    Ok([at.tensor.shape[0], at.tensor.shape[1]])
}

/// The kind of the graph object whose representation holds `tensor` (the token embedding, a decoder stack, ...), as the
/// container's own system graph declares it. A tensor's NAME says nothing reliable about this: families name their
/// embedding differently, and some containers hold it under a name no pattern would match. When the kind cannot be
/// determined the answer is an error, never a guess.
pub(in crate::format::vindex3::weight_patch) fn tensor_object_kind(
    src: &Path,
    tensor: &str,
) -> Result<ObjectKind, WeightPatchError> {
    let raw = std::fs::read_to_string(src.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&raw)
        .map_err(|e| crate::error::VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    let probe = DeclaredPatch::Results(ResultPatch {
        tensor: tensor.to_string(),
        edits: vec![ResultEdit {
            row: 0,
            col: 0,
            new_bits: 0,
        }],
    });
    let at = locate(src, &index, &probe)?;
    let undetermined = |why: String| {
        WeightPatchError::Structured(format!(
            "cannot tell what kind of object holds {tensor}: {why}"
        ))
    };
    let graph_name = index
        .system_graph
        .as_deref()
        .ok_or_else(|| undetermined("the container declares no system graph".into()))?;
    let graph_text = std::fs::read_to_string(src.join(graph_name))?;
    let graph: SystemGraph = serde_json::from_str(&graph_text)
        .map_err(|e| crate::error::VindexError::Parse(format!("parse {graph_name}: {e}")))?;
    graph
        .objects
        .iter()
        .find(|o| o.id == at.entry.object)
        .map(|o| o.kind)
        .ok_or_else(|| undetermined(format!("the graph has no object {}", at.entry.object)))
}

/// The BF16 bits of a whole matrix `tensor` of the container at `src`, row-major, with its `[rows, cols]`.
/// Refuses what a patch would: a tensor no canonical representation holds, one held twice, or one that
/// is not a BF16 matrix.
pub fn read_bf16_tensor(
    src: &Path,
    tensor: &str,
) -> Result<([usize; 2], Vec<u16>), WeightPatchError> {
    let raw = std::fs::read_to_string(src.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&raw)
        .map_err(|e| crate::error::VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    // One element is enough for `locate` to find, type-check and bound the tensor.
    let probe = DeclaredPatch::Results(ResultPatch {
        tensor: tensor.to_string(),
        edits: vec![ResultEdit {
            row: 0,
            col: 0,
            new_bits: 0,
        }],
    });
    let at = locate(src, &index, &probe)?;
    let [rows, cols] = [at.tensor.shape[0], at.tensor.shape[1]];
    let mut file = File::open(src.join(&at.entry.segment))?;
    file.seek(SeekFrom::Start(at.payload_start + at.tensor.offset))?;
    let mut bytes = vec![0u8; rows * cols * BF16_BYTES as usize];
    file.read_exact(&mut bytes)?;
    let bits = bytes
        .chunks_exact(BF16_BYTES as usize)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    Ok(([rows, cols], bits))
}
