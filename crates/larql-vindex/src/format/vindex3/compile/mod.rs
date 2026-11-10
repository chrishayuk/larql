//! `COMPILE INTO VINDEX` for a VINDEX3 container: bake an overlay's
//! operand edits into a **clean container** — no overlay required to
//! reproduce the composed behaviour.
//!
//! The bake follows the format's authority chain, not byte offsets:
//!
//! ```text
//! base objects + overlay operand edits
//!     → materialise effective logical tensors
//!     → rewrite ONLY the touched objects' segments
//!     → hard-link every untouched segment
//!     → new index (same graph, updated hashes)
//! ```
//!
//! Effective bytes are produced by the SAME resolver execution uses
//! ([`OperandSource`]), so the compiled container's stored bytes equal
//! the overlaid program's effective bytes by construction — the
//! equivalence gate then only has to confirm it.
//!
//! Refusals (fail closed, never silently drop):
//! - an overridden tensor whose stored dtype is not F32 — rewriting an
//!   edited tensor into a lossy representation is a representation-
//!   policy decision this rung does not take;
//! - an overridden object that carries alternate variants — the
//!   catalogue's other physical encodings would go stale;
//! - the legacy `.lyrw` bank layout (encode-layout containers only).

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::encode::segment::{read_segment_header, write_segment, PlannedTensor};
use super::encode::SYSTEM_GRAPH_JSON;
use super::index::Vindex3Index;
use super::inspect::inspect_container;
use super::opplan::exec::operands::{OperandOverrides, OperandSource, OperandStore};
use super::opplan::OperandRef;
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;

/// What one bake did — the executor's report material.
#[derive(Debug)]
pub struct BakeReport {
    pub rewritten_segments: usize,
    pub linked_segments: usize,
    pub rewritten_tensors: usize,
}

/// Safetensors dtype label for the widened representation rewritten
/// tensors are stored in.
const DTYPE_F32: &str = "F32";

/// Bake `overrides` over the container at `src` into a clean container
/// at `out`.
///
/// `out` must be a fresh destination — absent or an empty directory,
/// and neither the source nor nested with it
/// ([`crate::format::fresh_output`]). The bake is staged beside `out`
/// and promoted whole: a refusal or a failed bake leaves nothing there.
pub fn bake_container(
    src: &Path,
    overrides: &OperandOverrides,
    out: &Path,
) -> Result<BakeReport, VindexError> {
    bake_container_with(src, overrides, out, |_| Ok(()))
}

/// [`bake_container`], with `finish` run against the baked directory
/// before it is promoted to `out`.
///
/// A compiled container can carry files the bake does not derive (the
/// session's KNN store). Writing them here keeps the result whole: if
/// `finish` fails, nothing reaches `out`.
pub fn bake_container_with(
    src: &Path,
    overrides: &OperandOverrides,
    out: &Path,
    finish: impl FnOnce(&Path) -> Result<(), VindexError>,
) -> Result<BakeReport, VindexError> {
    crate::format::fresh_output::write_fresh_directory(src, out, |staging| {
        let report = bake_into(src, overrides, staging)?;
        finish(staging)?;
        Ok(report)
    })
}

/// The bake itself, writing into a directory that already exists and is
/// this call's alone.
fn bake_into(
    src: &Path,
    overrides: &OperandOverrides,
    out: &Path,
) -> Result<BakeReport, VindexError> {
    let raw_index = std::fs::read_to_string(src.join(INDEX_JSON))?;
    let mut index: Vindex3Index = serde_json::from_str(&raw_index)
        .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;

    let inspection = inspect_container(src, false)?;
    let store = OperandStore::open(src, &inspection)?;
    let source = OperandSource::overlaid(&store, overrides);

    let mut report = BakeReport {
        rewritten_segments: 0,
        linked_segments: 0,
        rewritten_tensors: 0,
    };

    let rep_ids: Vec<String> = index.representations.keys().cloned().collect();
    for rep_id in rep_ids {
        let entry = index.representations.get(&rep_id).unwrap().clone();
        let src_segment = src.join(&entry.segment);
        let out_segment = super::container_path::prepare_write(out, &entry.segment)?;

        let (header, payload_start) = read_segment_header(&src_segment)?;
        let overridden: BTreeSet<String> = header
            .tensors
            .iter()
            .filter(|t| {
                overrides.is_overridden(&OperandRef {
                    object: entry.object.clone(),
                    tensor: t.name.clone(),
                    dtype: t.dtype.clone(),
                    shape: t.shape.clone(),
                })
            })
            .map(|t| t.name.clone())
            .collect();

        if overridden.is_empty() {
            // Untouched object: the segment file is the same bytes —
            // hard-link where the filesystem allows, copy otherwise.
            if std::fs::hard_link(&src_segment, &out_segment).is_err() {
                std::fs::copy(&src_segment, &out_segment)?;
            }
            report.linked_segments += 1;
            continue;
        }

        if !index.variants.is_empty() {
            return Err(VindexError::Parse(format!(
                "the container carries alternate representation variants and object `{}` \
                 is edited — baking would leave the variant catalogue stale; a later rung \
                 rebuilds variants, this one refuses",
                entry.object
            )));
        }
        for t in &header.tensors {
            if overridden.contains(&t.name) && t.dtype != DTYPE_F32 {
                return Err(VindexError::Parse(format!(
                    "tensor `{}/{}` is stored as {} — baking an f32 edit into a non-f32 \
                     representation is a representation-policy decision a later rung takes; \
                     the compile was not performed",
                    entry.object, t.name, t.dtype
                )));
            }
        }

        // Rewrite this object's segment with effective bytes.
        let planned: Vec<PlannedTensor> = header
            .tensors
            .iter()
            .map(|t| PlannedTensor {
                relative_name: t.name.clone(),
                source_name: t.name.clone(),
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
                len: t.len,
            })
            .collect();
        let mut src_file = std::fs::File::open(&src_segment)?;
        let written = write_segment(&out_segment, &rep_id, planned, |name, w, tap| {
            let tensor = header
                .tensors
                .iter()
                .find(|t| t.name == name)
                .expect("planned from the same header");
            if overridden.contains(name) {
                let values = source.load(&OperandRef {
                    object: entry.object.clone(),
                    tensor: tensor.name.clone(),
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                })?;
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                w.write_all(&bytes)?;
                tap(&bytes);
                Ok(bytes.len() as u64)
            } else {
                src_file.seek(SeekFrom::Start(payload_start + tensor.offset))?;
                let mut remaining = tensor.len;
                let mut buf = vec![0u8; 1 << 20];
                while remaining > 0 {
                    let take = remaining.min(buf.len() as u64) as usize;
                    src_file.read_exact(&mut buf[..take])?;
                    w.write_all(&buf[..take])?;
                    tap(&buf[..take]);
                    remaining -= take as u64;
                }
                Ok(tensor.len)
            }
        })?;

        report.rewritten_segments += 1;
        report.rewritten_tensors += overridden.len();
        let entry_mut = index.representations.get_mut(&rep_id).unwrap();
        entry_mut.payload_sha256 = written.payload_sha256;
        entry_mut.segment_sha256 = written.segment_sha256;
        entry_mut.payload_bytes = written.payload_bytes;
        entry_mut.tensor_count = written.tensor_count;
    }

    // The programme manifest and capability files travel with the
    // container unchanged.
    for aux in [SYSTEM_GRAPH_JSON, "moe_manifest.json", "tokenizer.json"] {
        let from = src.join(aux);
        if from.exists() {
            std::fs::copy(&from, out.join(aux))?;
        }
    }

    // Index last: a crash mid-bake leaves a directory that is not yet
    // a container (the encode writer's ordering contract).
    let serialised = serde_json::to_string_pretty(&index)
        .map_err(|e| VindexError::Parse(format!("serialise {INDEX_JSON}: {e}")))?;
    std::fs::write(out.join(INDEX_JSON), serialised)?;

    Ok(report)
}
