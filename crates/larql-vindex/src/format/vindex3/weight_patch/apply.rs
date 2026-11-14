//! Apply a declared weight edit to a container and assemble the result.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::bf16;
use super::error::WeightPatchError;
use super::ledger::{DigestPair, LedgerEdit, WeightPatchLedger};
use super::op_target::OperationPatch;
use super::MAX_PATCH_ELEMENTS;
use crate::format::filenames::{INDEX_JSON, WEIGHT_PATCHES_JSON};
use crate::format::vindex3::encode::segment::{write_segment, PlannedTensor, WrittenSegment};
use crate::format::vindex3::index::{ContainerAuthority, Vindex3Index};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::represent::policy::{Protections, RolePolicy};
use crate::format::vindex3::represent::{compile_representation, RepresentSpec};

mod ledger_build;
mod locate;
mod materialise;
mod read;

use ledger_build::build_ledger;
use locate::{locate, resolve_edits, Located, Resolved};
use materialise::{check_source, patch_tensor_bytes, CheckedSource};

pub use materialise::{materialise_declared_patches, MaterialisedTensor};
pub(super) use read::tensor_object_kind;
pub use read::{read_bf16_tensor, tensor_shape};

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
    let CheckedSource {
        raw_index: raw,
        index,
        edited,
        carried,
    } = check_source(src, patches)?;
    let at = &edited[0].0;

    let stage_a = sibling(out, "a")?;
    let stage_b = sibling(out, "b")?;
    let mut scratch = Scratch(vec![stage_a.clone(), stage_b.clone()]);

    let written = write_stage(src, &index, &edited, &carried, &stage_a)?;
    let derived_encodings: Vec<String> = at
        .derived
        .iter()
        .map(|(_, e)| e.encoding.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let assembled = rederive(&stage_a, &stage_b, at, &derived_encodings, &index)?;
    carry_top_level_files(src, &carried, &assembled)?;

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

/// Copy into `assembled` every carried file of `src` it lacks ([`materialise::carried_files`]).
/// Stage A carries them, but the re-derivation compiles a fresh container from stage A and keeps only what a
/// representation compile knows about. A file such as the container's L0 knowledge (`knn_store.bin`) is part of
/// what the container means and must not be lost with the weights it sits beside.
fn carry_top_level_files(
    src: &Path,
    carried: &[String],
    assembled: &Path,
) -> Result<(), WeightPatchError> {
    for name in carried {
        let to = assembled.join(name);
        if !to.exists() {
            std::fs::copy(src.join(name), to)?;
        }
    }
    Ok(())
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
    carried: &[String],
    stage: &Path,
) -> Result<WrittenSegment, WeightPatchError> {
    let at = &edited[0].0;
    std::fs::create_dir_all(stage.join(SEGMENTS_DIR))?;
    for name in carried {
        std::fs::copy(src.join(name), stage.join(name))?;
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
) -> Result<WrittenSegment, WeightPatchError> {
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
                patch_tensor_bytes(&mut bytes, cols, resolved);
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

pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
