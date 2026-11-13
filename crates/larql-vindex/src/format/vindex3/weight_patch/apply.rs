//! Apply a declared weight edit to a container and assemble the result.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::bf16;
use super::error::WeightPatchError;
use super::ledger::{
    DigestPair, EditEncoding, LedgerEdit, LedgerResult, LedgerSource, LedgerTarget, ResultDerived,
    SourceDerived, WeightPatchLedger, ROUNDING_RULE, WEIGHT_PATCHES_SCHEMA,
    WEIGHT_PATCHES_SCHEMA_V3, WEIGHT_PATCHES_SCHEMA_V4,
};
use super::op_target::{tensor_sha256, OperationPatch};
use super::MAX_PATCH_ELEMENTS;
use crate::format::filenames::{INDEX_JSON, WEIGHT_PATCHES_JSON};
use crate::format::vindex3::encode::segment::{
    read_exact_at, read_segment_header, write_segment, PlannedTensor, SegmentHeader, SegmentTensor,
};
use crate::format::vindex3::graph::{ObjectKind, SystemGraph};
use crate::format::vindex3::index::{ContainerAuthority, RepresentationEntry, Vindex3Index};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::represent::policy::{Protections, RolePolicy};
use crate::format::vindex3::represent::{compile_representation, RepresentSpec};

/// Representation compilers write this record beside the segments; it is
/// regenerated for the result, never carried from the source.
const CANDIDATE_JSON: &str = "candidate.json";
/// The one directory a container holds.
const SEGMENTS_DIR: &str = "segments";
const BF16_DTYPE: &str = "BF16";
const BF16_BYTES: u64 = 2;
const WRITER: &str = "larql-vindex weight_patch v1";
const COPY_BUFFER: usize = 1 << 20;

/// One element edit: add `delta` to the stored value at `(row, col)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightEdit {
    pub row: usize,
    pub col: usize,
    pub delta: f32,
}

/// A declared edit of one tensor. Several, of distinct tensors in one
/// canonical representation, compile into one container
/// ([`apply_weight_patches`]).
#[derive(Debug, Clone, PartialEq)]
pub struct WeightPatch {
    /// The tensor's name in the segment table, e.g. `26.mlp.down_proj.weight`.
    pub tensor: String,
    pub edits: Vec<WeightEdit>,
}

/// One element whose declared new value is the exact BF16 a structured op produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResultEdit {
    pub row: usize,
    pub col: usize,
    pub new_bits: u16,
}

/// A declared edit of one tensor as exact results, not deltas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultPatch {
    pub tensor: String,
    pub edits: Vec<ResultEdit>,
}

/// An edit of one tensor in any form. A set of these may mix forms, one tensor each.
#[derive(Debug, Clone, PartialEq)]
pub enum DeclaredPatch {
    Deltas(WeightPatch),
    Results(ResultPatch),
    /// The whole tensor compiled from structured ops, recorded by its operations (`weight-patches/v4`). It is
    /// not an element list, so [`MAX_PATCH_ELEMENTS`] does not bound it; the `MAX_STRUCTURED_*` bounds did.
    Operations(OperationPatch),
}

/// What a located patch resolved to: element edits, or a tensor compiled from its operations.
enum Resolved<'a> {
    Edits(Vec<LedgerEdit>),
    Operations(&'a OperationPatch),
}

impl DeclaredPatch {
    pub fn tensor(&self) -> &str {
        match self {
            DeclaredPatch::Deltas(p) => &p.tensor,
            DeclaredPatch::Results(p) => &p.tensor,
            DeclaredPatch::Operations(p) => &p.tensor,
        }
    }

    /// Listed elements: what counts against [`MAX_PATCH_ELEMENTS`]. An operation patch lists none.
    fn len(&self) -> usize {
        match self {
            DeclaredPatch::Deltas(p) => p.edits.len(),
            DeclaredPatch::Results(p) => p.edits.len(),
            DeclaredPatch::Operations(_) => 0,
        }
    }

    fn coords(&self) -> Vec<(usize, usize)> {
        match self {
            DeclaredPatch::Deltas(p) => p.edits.iter().map(|e| (e.row, e.col)).collect(),
            DeclaredPatch::Results(p) => p.edits.iter().map(|e| (e.row, e.col)).collect(),
            DeclaredPatch::Operations(_) => Vec::new(),
        }
    }
}

/// What an applied patch produced.
#[derive(Debug, Clone)]
pub struct PatchReport {
    /// Edited elements, across every tensor.
    pub elements: usize,
    /// The edited tensors, in the order given.
    pub tensors: Vec<String>,
    pub object: String,
    pub canonical: DigestPair,
    pub rederived: Vec<String>,
    pub ledger: WeightPatchLedger,
}

/// Where the edited tensor lives and what derives from it.
struct Located {
    canonical_id: String,
    entry: RepresentationEntry,
    header: SegmentHeader,
    payload_start: u64,
    tensor: SegmentTensor,
    derived: Vec<(String, RepresentationEntry)>,
}

/// Removes its directories on drop unless disarmed, so a failed patch leaves
/// nothing behind.
struct Scratch(Vec<PathBuf>);

impl Drop for Scratch {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_dir_all(p);
        }
    }
}

/// Apply `patch` to the container at `src`, writing a new container at `out`.
///
/// `out` must not exist. The result is complete and inspected, or nothing is
/// left at `out`.
pub fn apply_weight_patch(
    src: &Path,
    patch: &WeightPatch,
    out: &Path,
) -> Result<PatchReport, WeightPatchError> {
    apply_weight_patch_with_origins(src, patch, &[], out)
}

/// [`apply_weight_patch`], recording `origins` in the ledger: the digests of
/// the patch ops this edit was assembled from, so a container can be traced
/// back to the `.vlp` instructions that produced it.
pub fn apply_weight_patch_with_origins(
    src: &Path,
    patch: &WeightPatch,
    origins: &[String],
    out: &Path,
) -> Result<PatchReport, WeightPatchError> {
    apply_weight_patches(src, std::slice::from_ref(patch), origins, out)
}

/// Apply edits of several tensors as ONE patch: one new container, one
/// ledger, one re-derivation. The tensors must be distinct and held by one
/// canonical representation, and the elements across all of them count
/// against [`MAX_PATCH_ELEMENTS`]. Either every edit lands or nothing is
/// written.
pub fn apply_weight_patches(
    src: &Path,
    patches: &[WeightPatch],
    origins: &[String],
    out: &Path,
) -> Result<PatchReport, WeightPatchError> {
    let declared: Vec<DeclaredPatch> = patches.iter().cloned().map(DeclaredPatch::Deltas).collect();
    apply_declared_patches(src, &declared, origins, out)
}

/// [`apply_weight_patches`] for edits in any form: deltas, the exact results of structured ops, or a tensor
/// compiled from structured ops. The ledger is schema `v4` when any tensor is an operation target, else `v3` when
/// any edit is a result, else `v2`.
pub fn apply_declared_patches(
    src: &Path,
    patches: &[DeclaredPatch],
    origins: &[String],
    out: &Path,
) -> Result<PatchReport, WeightPatchError> {
    validate_all(patches)?;
    if out.exists() {
        return Err(WeightPatchError::OutputExists(out.to_path_buf()));
    }
    // `exists()` follows symlinks and says nothing about where a missing
    // path would land: a dangling link, or a destination inside the
    // source, gets past it. The shared boundary refuses both before the
    // stage directories are created.
    crate::format::fresh_output::check_fresh_destination(src, out)?;
    let raw = std::fs::read_to_string(src.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&raw)
        .map_err(|e| crate::error::VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    let inspection = inspect_container(src, false)?;
    if !inspection.is_coherent() {
        return Err(WeightPatchError::SourceIncoherent(format!(
            "{:?}",
            inspection.defects
        )));
    }
    require_patchable(&index)?;
    let edited = locate_all(src, &index, patches)?;
    let at = &edited[0].0;

    let stage_a = sibling(out, "a")?;
    let stage_b = sibling(out, "b")?;
    let mut scratch = Scratch(vec![stage_a.clone(), stage_b.clone()]);

    let written = write_stage(src, &index, &edited, &stage_a)?;
    let derived_encodings: Vec<String> = at
        .derived
        .iter()
        .map(|(_, e)| e.encoding.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let assembled = rederive(&stage_a, &stage_b, at, &derived_encodings, &index)?;

    let mut ledger = build_ledger(&index, &raw, &edited, &written, &assembled);
    ledger.origins = origins.to_vec();
    std::fs::write(
        assembled.join(WEIGHT_PATCHES_JSON),
        serde_json::to_string_pretty(&ledger)
            .map_err(|e| crate::error::VindexError::Parse(format!("serialise ledger: {e}")))?,
    )?;
    let mut final_index = read_index(&assembled)?;
    final_index.weight_patches = Some(WEIGHT_PATCHES_JSON.to_string());
    write_index(&assembled, &final_index)?;

    let inspected = inspect_container(&assembled, false)?;
    if !inspected.is_coherent() {
        return Err(WeightPatchError::Incoherent(format!(
            "{:?}",
            inspected.defects
        )));
    }
    std::fs::rename(&assembled, out)?;
    scratch.0.retain(|p| p != &assembled);
    Ok(PatchReport {
        elements: ledger.edit_count()
            + ledger
                .targets
                .iter()
                .filter_map(|t| t.changed_elements)
                .sum::<usize>(),
        tensors: ledger.targets.iter().map(|t| t.tensor.clone()).collect(),
        object: at.entry.object.clone(),
        canonical: ledger.result.canonical.clone(),
        rederived: ledger
            .result
            .derived
            .iter()
            .map(|d| d.representation.clone())
            .collect(),
        ledger,
    })
}

/// Check `patch` against the container at `src` without writing anything:
/// the tensor exists and is a BF16 matrix, every element is in bounds, every
/// stored value is finite and every patched value fits. Returns the edits as
/// they would be recorded, with the exact old and new bits.
///
/// This is what staging an edit in a session runs, so a bad edit is refused
/// when it is written down and not when it is compiled.
pub fn plan_weight_patch(
    src: &Path,
    patch: &WeightPatch,
) -> Result<Vec<LedgerEdit>, WeightPatchError> {
    plan_declared_patch(src, &DeclaredPatch::Deltas(patch.clone()))
}

/// [`plan_weight_patch`] for either form.
pub fn plan_declared_patch(
    src: &Path,
    patch: &DeclaredPatch,
) -> Result<Vec<LedgerEdit>, WeightPatchError> {
    validate(patch)?;
    let raw = std::fs::read_to_string(src.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&raw)
        .map_err(|e| crate::error::VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    require_patchable(&index)?;
    let at = locate(src, &index, patch)?;
    match resolve_edits(src, &at, patch)? {
        Resolved::Edits(edits) => Ok(edits),
        Resolved::Operations(_) => Ok(Vec::new()),
    }
}

/// What only the set can break, cheapest first: the element total within
/// the limit, then [`validate`] for each patch, and no tensor twice.
fn validate_all(patches: &[DeclaredPatch]) -> Result<(), WeightPatchError> {
    if patches.is_empty() {
        return Err(WeightPatchError::Empty);
    }
    // Element patches only: an operation patch lists no elements and is bounded by `MAX_STRUCTURED_*`.
    let count: usize = patches.iter().map(|p| p.len()).sum();
    if count > MAX_PATCH_ELEMENTS {
        return Err(WeightPatchError::TooManyElements {
            count,
            limit: MAX_PATCH_ELEMENTS,
        });
    }
    let mut names = BTreeSet::new();
    for p in patches {
        validate(p)?;
        if !names.insert(p.tensor()) {
            return Err(WeightPatchError::DuplicateTensor(p.tensor().to_string()));
        }
    }
    Ok(())
}

/// Locate and resolve every patch; refuse unless one canonical
/// representation holds them all, since the re-derivation and the result
/// digests belong to one representation.
fn locate_all<'a>(
    src: &Path,
    index: &Vindex3Index,
    patches: &'a [DeclaredPatch],
) -> Result<Vec<(Located, Resolved<'a>)>, WeightPatchError> {
    let mut out = Vec::with_capacity(patches.len());
    for p in patches {
        let at = locate(src, index, p)?;
        let edits = resolve_edits(src, &at, p)?;
        out.push((at, edits));
    }
    let first = out[0].0.canonical_id.clone();
    if let Some((other, _)) = out.iter().find(|(a, _)| a.canonical_id != first) {
        return Err(WeightPatchError::SpansRepresentations {
            first: (patches[0].tensor().to_string(), first),
            other: (other.tensor.name.clone(), other.canonical_id.clone()),
        });
    }
    Ok(out)
}

fn validate(patch: &DeclaredPatch) -> Result<(), WeightPatchError> {
    if let DeclaredPatch::Operations(p) = patch {
        // Nothing changed is refused unless every op is zero-valued: a declared sham compiles to the base's bytes.
        if p.operations.is_empty() || (p.changed_elements == 0 && !p.is_sham()?) {
            return Err(WeightPatchError::Empty);
        }
        if p.result.len() != p.shape[0] * p.shape[1] {
            return Err(WeightPatchError::Structured(format!(
                "the compiled `{}` has {} elements, not {} x {}",
                p.tensor,
                p.result.len(),
                p.shape[0],
                p.shape[1]
            )));
        }
        if let Some(i) = p.result.iter().position(|&b| !bf16::is_finite(b)) {
            let cols = p.shape[1];
            return Err(WeightPatchError::NonFiniteResult {
                row: i / cols,
                col: i % cols,
            });
        }
        return Ok(());
    }
    if patch.len() == 0 {
        return Err(WeightPatchError::Empty);
    }
    if patch.len() > MAX_PATCH_ELEMENTS {
        return Err(WeightPatchError::TooManyElements {
            count: patch.len(),
            limit: MAX_PATCH_ELEMENTS,
        });
    }
    if let DeclaredPatch::Deltas(p) = patch {
        for e in &p.edits {
            if !e.delta.is_finite() {
                return Err(WeightPatchError::NonFiniteDelta {
                    row: e.row,
                    col: e.col,
                });
            }
        }
    }
    let mut seen = BTreeSet::new();
    for (row, col) in patch.coords() {
        if !seen.insert((row, col)) {
            return Err(WeightPatchError::DuplicateElement { row, col });
        }
    }
    Ok(())
}

fn require_patchable(index: &Vindex3Index) -> Result<(), WeightPatchError> {
    if index.authority != ContainerAuthority::Canonical {
        return Err(WeightPatchError::NotCanonical(format!(
            "{:?}",
            index.authority
        )));
    }
    if index.weight_patches.is_some() {
        return Err(WeightPatchError::AlreadyPatched);
    }
    if !index.variants.is_empty() {
        return Err(WeightPatchError::HasVariants);
    }
    Ok(())
}

fn locate(
    src: &Path,
    index: &Vindex3Index,
    patch: &DeclaredPatch,
) -> Result<Located, WeightPatchError> {
    let name = patch.tensor();
    let mut holders = Vec::new();
    for (id, entry) in index
        .representations
        .iter()
        .filter(|(_, e)| e.compiled_from.is_none())
    {
        let (header, payload_start) = read_segment_header(&src.join(&entry.segment))?;
        if let Some(t) = header.tensors.iter().find(|t| t.name == name) {
            holders.push((
                id.clone(),
                entry.clone(),
                header.clone(),
                payload_start,
                t.clone(),
            ));
        }
    }
    let (canonical_id, entry, header, payload_start, tensor) = match holders.len() {
        0 => return Err(WeightPatchError::UnknownTensor(name.to_string())),
        1 => holders.remove(0),
        _ => {
            return Err(WeightPatchError::AmbiguousTensor {
                tensor: name.to_string(),
                holders: holders.into_iter().map(|h| h.0).collect(),
            })
        }
    };
    if tensor.dtype != BF16_DTYPE {
        return Err(WeightPatchError::UnsupportedDtype {
            tensor: name.to_string(),
            dtype: tensor.dtype,
        });
    }
    let (rows, cols) = match tensor.shape.as_slice() {
        [r, c] => (*r, *c),
        _ => {
            return Err(WeightPatchError::NotAMatrix {
                tensor: name.to_string(),
                shape: tensor.shape,
            })
        }
    };
    for (row, col) in patch.coords() {
        if row >= rows || col >= cols {
            return Err(WeightPatchError::OutOfBounds {
                tensor: name.to_string(),
                row,
                col,
                rows,
                cols,
            });
        }
    }
    let mut derived = Vec::new();
    for (id, e) in index
        .representations
        .iter()
        .filter(|(_, e)| e.compiled_from.is_some())
    {
        if e.compiled_from.as_deref() != Some(canonical_id.as_str()) {
            return Err(WeightPatchError::DerivedFromOther {
                derived: id.clone(),
                canonical: canonical_id,
            });
        }
        derived.push((id.clone(), e.clone()));
    }
    Ok(Located {
        canonical_id,
        entry,
        header,
        payload_start,
        tensor,
        derived,
    })
}

/// The ledger's edits: the stored bits read from the source, and the new
/// bits: computed by the declared rule for a delta, taken as declared for a
/// structured op's result.
fn resolve_edits<'a>(
    src: &Path,
    at: &Located,
    patch: &'a DeclaredPatch,
) -> Result<Resolved<'a>, WeightPatchError> {
    if let DeclaredPatch::Operations(p) = patch {
        if at.tensor.shape != p.shape {
            return Err(WeightPatchError::Structured(format!(
                "`{}` is {:?} in the container, but was compiled as {:?}",
                p.tensor, at.tensor.shape, p.shape
            )));
        }
        // The ops were resolved against a base; refuse to write them over another.
        let (_, base) = read_bf16_tensor(src, &p.tensor)?;
        if tensor_sha256(&base) != p.base_tensor_sha256 {
            return Err(WeightPatchError::Structured(format!(
                "`{}` differs from the base its operations were compiled against",
                p.tensor
            )));
        }
        return Ok(Resolved::Operations(p));
    }
    let file = File::open(src.join(&at.entry.segment))?;
    let cols = at.tensor.shape[1] as u64;
    let name = patch.tensor();
    let read_old = |row: usize, col: usize| -> Result<u16, WeightPatchError> {
        let mut stored = [0u8; BF16_BYTES as usize];
        let offset =
            at.payload_start + at.tensor.offset + (row as u64 * cols + col as u64) * BF16_BYTES;
        read_exact_at(&file, &mut stored, offset)?;
        let old_bits = u16::from_le_bytes(stored);
        if !bf16::is_finite(old_bits) {
            return Err(WeightPatchError::NonFiniteSource {
                tensor: name.to_string(),
                row,
                col,
            });
        }
        Ok(old_bits)
    };
    let mut out = Vec::with_capacity(patch.len());
    match patch {
        DeclaredPatch::Deltas(p) => {
            for e in &p.edits {
                let old_bits = read_old(e.row, e.col)?;
                let new_bits =
                    bf16::patched_bits(old_bits, e.delta).ok_or(WeightPatchError::Overflow {
                        row: e.row,
                        col: e.col,
                    })?;
                out.push(LedgerEdit {
                    row: e.row,
                    col: e.col,
                    delta: e.delta,
                    delta_bits: e.delta.to_bits(),
                    old_bits,
                    new_bits,
                    encoding: EditEncoding::Delta,
                });
            }
        }
        DeclaredPatch::Results(p) => {
            for e in &p.edits {
                let old_bits = read_old(e.row, e.col)?;
                if !bf16::is_finite(e.new_bits) {
                    return Err(WeightPatchError::NonFiniteResult {
                        row: e.row,
                        col: e.col,
                    });
                }
                out.push(LedgerEdit {
                    row: e.row,
                    col: e.col,
                    delta: 0.0,
                    delta_bits: 0,
                    old_bits,
                    new_bits: e.new_bits,
                    encoding: EditEncoding::ResultBits,
                });
            }
        }
        DeclaredPatch::Operations(_) => unreachable!("returned above"),
    }
    Ok(Resolved::Edits(out))
}

/// A scratch path beside `out`, on the same filesystem so the final rename is
/// atomic.
fn sibling(out: &Path, tag: &str) -> Result<PathBuf, WeightPatchError> {
    let name = out.file_name().ok_or_else(|| {
        crate::error::VindexError::Parse(format!("`{}` has no file name", out.display()))
    })?;
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Ok(parent.join(format!(
        ".{}.wsp-{}-{}-{tag}",
        name.to_string_lossy(),
        std::process::id(),
        stamp
    )))
}

fn read_index(root: &Path) -> Result<Vindex3Index, WeightPatchError> {
    let text = std::fs::read_to_string(root.join(INDEX_JSON))?;
    Ok(serde_json::from_str(&text)
        .map_err(|e| crate::error::VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?)
}

fn write_index(root: &Path, index: &Vindex3Index) -> Result<(), WeightPatchError> {
    let text = serde_json::to_string_pretty(index)
        .map_err(|e| crate::error::VindexError::Parse(format!("serialise {INDEX_JSON}: {e}")))?;
    std::fs::write(root.join(INDEX_JSON), text)?;
    Ok(())
}

/// Stage A: the source with its canonical segment rewritten and the derived
/// representations of that object removed. Untouched segments are hard-linked.
fn write_stage(
    src: &Path,
    index: &Vindex3Index,
    edited: &[(Located, Resolved<'_>)],
    stage: &Path,
) -> Result<crate::format::vindex3::encode::segment::WrittenSegment, WeightPatchError> {
    let at = &edited[0].0;
    std::fs::create_dir_all(stage.join(SEGMENTS_DIR))?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if entry.file_type()?.is_dir() {
            if name != SEGMENTS_DIR {
                return Err(WeightPatchError::UnexpectedEntry(name));
            }
        } else if ![INDEX_JSON, CANDIDATE_JSON, WEIGHT_PATCHES_JSON].contains(&name.as_str()) {
            std::fs::copy(entry.path(), stage.join(&name))?;
        }
    }
    let derived_segments: BTreeSet<&str> =
        at.derived.iter().map(|(_, e)| e.segment.as_str()).collect();
    for entry in std::fs::read_dir(src.join(SEGMENTS_DIR))? {
        let name = entry?.file_name().to_string_lossy().to_string();
        let rel = format!("{SEGMENTS_DIR}/{name}");
        if rel == at.entry.segment || derived_segments.contains(rel.as_str()) {
            continue;
        }
        let from = src.join(&rel);
        let to = crate::format::vindex3::container_path::prepare_write(stage, &rel)?;
        if std::fs::hard_link(&from, &to).is_err() {
            std::fs::copy(&from, &to)?;
        }
    }
    let written = write_patched_segment(src, stage, edited)?;
    let mut staged = index.clone();
    for (id, e) in &at.derived {
        staged.representations.remove(id);
        staged
            .segments
            .remove(e.segment.strip_suffix(".bin").unwrap_or(&e.segment));
    }
    staged.precision_map = None;
    let canonical = staged
        .representations
        .get_mut(&at.canonical_id)
        .expect("located above");
    canonical.payload_sha256 = written.payload_sha256.clone();
    canonical.segment_sha256 = written.segment_sha256.clone();
    canonical.payload_bytes = written.payload_bytes;
    canonical.tensor_count = written.tensor_count;
    write_index(stage, &staged)?;
    Ok(written)
}

fn write_patched_segment(
    src: &Path,
    stage: &Path,
    edited: &[(Located, Resolved<'_>)],
) -> Result<crate::format::vindex3::encode::segment::WrittenSegment, WeightPatchError> {
    let at = &edited[0].0;
    let by_tensor: BTreeMap<&str, (u64, &Resolved<'_>)> = edited
        .iter()
        .map(|(a, r)| (a.tensor.name.as_str(), (a.tensor.shape[1] as u64, r)))
        .collect();
    let planned: Vec<PlannedTensor> = at
        .header
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
    let mut source = File::open(src.join(&at.entry.segment))?;
    let dest = crate::format::vindex3::container_path::prepare_write(stage, &at.entry.segment)?;
    Ok(write_segment(
        &dest,
        &at.canonical_id,
        planned,
        |name, w, tap| {
            let t = at
                .header
                .tensors
                .iter()
                .find(|t| t.name == name)
                .expect("planned from this header");
            source.seek(SeekFrom::Start(at.payload_start + t.offset))?;
            if let Some(&(cols, resolved)) = by_tensor.get(name) {
                let mut bytes = vec![0u8; t.len as usize];
                source.read_exact(&mut bytes)?;
                match resolved {
                    Resolved::Edits(edits) => {
                        for e in edits {
                            let at_byte =
                                ((e.row as u64 * cols + e.col as u64) * BF16_BYTES) as usize;
                            bytes[at_byte..at_byte + 2].copy_from_slice(&e.new_bits.to_le_bytes());
                        }
                    }
                    Resolved::Operations(p) => {
                        for (chunk, bits) in
                            bytes.chunks_exact_mut(BF16_BYTES as usize).zip(&p.result)
                        {
                            chunk.copy_from_slice(&bits.to_le_bytes());
                        }
                    }
                }
                w.write_all(&bytes)?;
                tap(&bytes);
            } else {
                let mut remaining = t.len;
                let mut buf = vec![0u8; COPY_BUFFER];
                while remaining > 0 {
                    let take = remaining.min(buf.len() as u64) as usize;
                    source.read_exact(&mut buf[..take])?;
                    w.write_all(&buf[..take])?;
                    tap(&buf[..take]);
                    remaining -= take as u64;
                }
            }
            Ok(t.len)
        },
    )?)
}

/// Stage B: re-derive each derived encoding from the edited canonical bytes
/// with the same compiler, one after another. Returns the assembled directory.
fn rederive(
    stage_a: &Path,
    stage_b: &Path,
    at: &Located,
    encodings: &[String],
    source_index: &Vindex3Index,
) -> Result<PathBuf, WeightPatchError> {
    if encodings.is_empty() {
        std::fs::rename(stage_a, stage_b)?;
        return Ok(stage_b.to_path_buf());
    }
    let mut current = stage_a.to_path_buf();
    for (n, encoding) in encodings.iter().enumerate() {
        let next = if n + 1 == encodings.len() {
            stage_b.to_path_buf()
        } else {
            stage_b.with_extension(format!("step{n}"))
        };
        let spec = RepresentSpec {
            encoding: encoding.clone(),
            objects: vec![at.entry.object.clone()],
            roles: RolePolicy::default(),
            deployment: false,
            protect: Protections::default(),
        };
        compile_representation(&current, &next, &spec)?;
        current = next;
    }
    let rebuilt = read_index(&current)?;
    let same_map = serde_json::to_value(&source_index.precision_map).ok()
        == serde_json::to_value(&rebuilt.precision_map).ok();
    if !same_map {
        return Err(WeightPatchError::NotReproducible(
            "the precision map the default policy produces differs from the source's".to_string(),
        ));
    }
    for (id, old) in &at.derived {
        let new = rebuilt.representations.get(id).ok_or_else(|| {
            WeightPatchError::NotReproducible(format!("`{id}` was not re-derived"))
        })?;
        if new.codec != old.codec {
            return Err(WeightPatchError::NotReproducible(format!(
                "`{id}` was re-derived under a different codec"
            )));
        }
    }
    Ok(current)
}

fn digests(e: &RepresentationEntry) -> DigestPair {
    DigestPair {
        payload_sha256: e.payload_sha256.clone(),
        segment_sha256: e.segment_sha256.clone(),
    }
}

fn encoder_of(e: &RepresentationEntry) -> Option<serde_json::Value> {
    serde_json::to_value(&e.encoder)
        .ok()
        .filter(|v| !v.is_null())
}

fn build_ledger(
    source_index: &Vindex3Index,
    raw_source_index: &str,
    edited: &[(Located, Resolved<'_>)],
    written: &crate::format::vindex3::encode::segment::WrittenSegment,
    assembled: &Path,
) -> WeightPatchLedger {
    let rebuilt = read_index(assembled).expect("assembled above");
    let at = &edited[0].0;
    let canonical = DigestPair {
        payload_sha256: written.payload_sha256.clone(),
        segment_sha256: written.segment_sha256.clone(),
    };
    let any_operations = edited
        .iter()
        .any(|(_, r)| matches!(r, Resolved::Operations(_)));
    let any_result = edited.iter().any(|(_, r)| match r {
        Resolved::Edits(edits) => edits.iter().any(|e| e.encoding == EditEncoding::ResultBits),
        Resolved::Operations(_) => false,
    });
    WeightPatchLedger {
        schema: if any_operations {
            WEIGHT_PATCHES_SCHEMA_V4
        } else if any_result {
            WEIGHT_PATCHES_SCHEMA_V3
        } else {
            WEIGHT_PATCHES_SCHEMA
        }
        .to_string(),
        rounding: ROUNDING_RULE.to_string(),
        writer: WRITER.to_string(),
        source: LedgerSource {
            model: source_index.model.clone(),
            index_sha256: hex(&Sha256::digest(raw_source_index.as_bytes())),
            canonical: digests(&at.entry),
            derived: at
                .derived
                .iter()
                .map(|(id, e)| SourceDerived {
                    representation: id.clone(),
                    encoding: e.encoding.clone(),
                    digests: digests(e),
                    encoder: encoder_of(e),
                })
                .collect(),
        },
        targets: edited
            .iter()
            .map(|(a, resolved)| {
                let mut target = LedgerTarget {
                    object: a.entry.object.clone(),
                    representation: a.canonical_id.clone(),
                    tensor: a.tensor.name.clone(),
                    dtype: a.tensor.dtype.clone(),
                    shape: a.tensor.shape.clone(),
                    edits: Vec::new(),
                    operations: Vec::new(),
                    base_tensor_sha256: None,
                    result_tensor_sha256: None,
                    changed_elements: None,
                };
                match resolved {
                    Resolved::Edits(edits) => target.edits = edits.clone(),
                    Resolved::Operations(p) => {
                        target.operations = p.operations.clone();
                        target.base_tensor_sha256 = Some(p.base_tensor_sha256.clone());
                        target.result_tensor_sha256 = Some(tensor_sha256(&p.result));
                        target.changed_elements = Some(p.changed_elements);
                    }
                }
                target
            })
            .collect(),
        origins: Vec::new(),
        result: LedgerResult {
            derived: at
                .derived
                .iter()
                .filter_map(|(id, _)| rebuilt.representations.get(id).map(|e| (id, e)))
                .map(|(id, e)| ResultDerived {
                    representation: id.clone(),
                    encoding: e.encoding.clone(),
                    source_representation_digest: canonical.payload_sha256.clone(),
                    digests: digests(e),
                    encoder: encoder_of(e),
                })
                .collect(),
            canonical,
        },
    }
}

pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

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
pub(super) fn tensor_object_kind(src: &Path, tensor: &str) -> Result<ObjectKind, WeightPatchError> {
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
