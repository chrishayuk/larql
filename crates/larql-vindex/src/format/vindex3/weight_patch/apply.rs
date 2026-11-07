//! Apply a declared weight edit to a container and assemble the result.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::bf16;
use super::error::WeightPatchError;
use super::ledger::{
    DigestPair, LedgerEdit, LedgerResult, LedgerSource, LedgerTarget, ResultDerived, SourceDerived,
    WeightPatchLedger, ROUNDING_RULE, WEIGHT_PATCHES_SCHEMA,
};
use super::MAX_PATCH_ELEMENTS;
use crate::format::filenames::{INDEX_JSON, WEIGHT_PATCHES_JSON};
use crate::format::vindex3::encode::segment::{
    read_exact_at, read_segment_header, write_segment, PlannedTensor, SegmentHeader, SegmentTensor,
};
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
    validate_all(patches)?;
    if out.exists() {
        return Err(WeightPatchError::OutputExists(out.to_path_buf()));
    }
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
        elements: ledger.edit_count(),
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
    validate(patch)?;
    let raw = std::fs::read_to_string(src.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&raw)
        .map_err(|e| crate::error::VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    require_patchable(&index)?;
    let at = locate(src, &index, patch)?;
    resolve_edits(src, &at, patch)
}

/// What only the set can break, cheapest first: the element total within
/// the limit, then [`validate`] for each patch, and no tensor twice.
fn validate_all(patches: &[WeightPatch]) -> Result<(), WeightPatchError> {
    if patches.is_empty() {
        return Err(WeightPatchError::Empty);
    }
    let count: usize = patches.iter().map(|p| p.edits.len()).sum();
    if count > MAX_PATCH_ELEMENTS {
        return Err(WeightPatchError::TooManyElements {
            count,
            limit: MAX_PATCH_ELEMENTS,
        });
    }
    let mut names = BTreeSet::new();
    for p in patches {
        validate(p)?;
        if !names.insert(p.tensor.as_str()) {
            return Err(WeightPatchError::DuplicateTensor(p.tensor.clone()));
        }
    }
    Ok(())
}

/// Locate and resolve every patch; refuse unless one canonical
/// representation holds them all, since the re-derivation and the result
/// digests belong to one representation.
fn locate_all(
    src: &Path,
    index: &Vindex3Index,
    patches: &[WeightPatch],
) -> Result<Vec<(Located, Vec<LedgerEdit>)>, WeightPatchError> {
    let mut out = Vec::with_capacity(patches.len());
    for p in patches {
        let at = locate(src, index, p)?;
        let edits = resolve_edits(src, &at, p)?;
        out.push((at, edits));
    }
    let first = out[0].0.canonical_id.clone();
    if let Some((other, _)) = out.iter().find(|(a, _)| a.canonical_id != first) {
        return Err(WeightPatchError::SpansRepresentations {
            first: (patches[0].tensor.clone(), first),
            other: (other.tensor.name.clone(), other.canonical_id.clone()),
        });
    }
    Ok(out)
}

fn validate(patch: &WeightPatch) -> Result<(), WeightPatchError> {
    if patch.edits.is_empty() {
        return Err(WeightPatchError::Empty);
    }
    if patch.edits.len() > MAX_PATCH_ELEMENTS {
        return Err(WeightPatchError::TooManyElements {
            count: patch.edits.len(),
            limit: MAX_PATCH_ELEMENTS,
        });
    }
    let mut seen = BTreeSet::new();
    for e in &patch.edits {
        if !e.delta.is_finite() {
            return Err(WeightPatchError::NonFiniteDelta {
                row: e.row,
                col: e.col,
            });
        }
        if !seen.insert((e.row, e.col)) {
            return Err(WeightPatchError::DuplicateElement {
                row: e.row,
                col: e.col,
            });
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
    patch: &WeightPatch,
) -> Result<Located, WeightPatchError> {
    let mut holders = Vec::new();
    for (id, entry) in index
        .representations
        .iter()
        .filter(|(_, e)| e.compiled_from.is_none())
    {
        let (header, payload_start) = read_segment_header(&src.join(&entry.segment))?;
        if let Some(t) = header.tensors.iter().find(|t| t.name == patch.tensor) {
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
        0 => return Err(WeightPatchError::UnknownTensor(patch.tensor.clone())),
        1 => holders.remove(0),
        _ => {
            return Err(WeightPatchError::AmbiguousTensor {
                tensor: patch.tensor.clone(),
                holders: holders.into_iter().map(|h| h.0).collect(),
            })
        }
    };
    if tensor.dtype != BF16_DTYPE {
        return Err(WeightPatchError::UnsupportedDtype {
            tensor: patch.tensor.clone(),
            dtype: tensor.dtype,
        });
    }
    let (rows, cols) = match tensor.shape.as_slice() {
        [r, c] => (*r, *c),
        _ => {
            return Err(WeightPatchError::NotAMatrix {
                tensor: patch.tensor.clone(),
                shape: tensor.shape,
            })
        }
    };
    for e in &patch.edits {
        if e.row >= rows || e.col >= cols {
            return Err(WeightPatchError::OutOfBounds {
                tensor: patch.tensor.clone(),
                row: e.row,
                col: e.col,
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

/// The ledger's edits: the stored bits read from the source, the patched
/// bits computed by the declared rule.
fn resolve_edits(
    src: &Path,
    at: &Located,
    patch: &WeightPatch,
) -> Result<Vec<LedgerEdit>, WeightPatchError> {
    let file = File::open(src.join(&at.entry.segment))?;
    let cols = at.tensor.shape[1] as u64;
    let mut out = Vec::with_capacity(patch.edits.len());
    for e in &patch.edits {
        let mut stored = [0u8; BF16_BYTES as usize];
        let offset =
            at.payload_start + at.tensor.offset + (e.row as u64 * cols + e.col as u64) * BF16_BYTES;
        read_exact_at(&file, &mut stored, offset)?;
        let old_bits = u16::from_le_bytes(stored);
        if !bf16::is_finite(old_bits) {
            return Err(WeightPatchError::NonFiniteSource {
                tensor: patch.tensor.clone(),
                row: e.row,
                col: e.col,
            });
        }
        let new_bits = bf16::patched_bits(old_bits, e.delta).ok_or(WeightPatchError::Overflow {
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
        });
    }
    Ok(out)
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
    edited: &[(Located, Vec<LedgerEdit>)],
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
        let to = stage.join(&rel);
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
    edited: &[(Located, Vec<LedgerEdit>)],
) -> Result<crate::format::vindex3::encode::segment::WrittenSegment, WeightPatchError> {
    let at = &edited[0].0;
    let by_tensor: BTreeMap<&str, (u64, &[LedgerEdit])> = edited
        .iter()
        .map(|(a, e)| {
            (
                a.tensor.name.as_str(),
                (a.tensor.shape[1] as u64, e.as_slice()),
            )
        })
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
    Ok(write_segment(
        &stage.join(&at.entry.segment),
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
            if let Some(&(cols, edits)) = by_tensor.get(name) {
                let mut bytes = vec![0u8; t.len as usize];
                source.read_exact(&mut bytes)?;
                for e in edits {
                    let at_byte = ((e.row as u64 * cols + e.col as u64) * BF16_BYTES) as usize;
                    bytes[at_byte..at_byte + 2].copy_from_slice(&e.new_bits.to_le_bytes());
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
    edited: &[(Located, Vec<LedgerEdit>)],
    written: &crate::format::vindex3::encode::segment::WrittenSegment,
    assembled: &Path,
) -> WeightPatchLedger {
    let rebuilt = read_index(assembled).expect("assembled above");
    let at = &edited[0].0;
    let canonical = DigestPair {
        payload_sha256: written.payload_sha256.clone(),
        segment_sha256: written.segment_sha256.clone(),
    };
    WeightPatchLedger {
        schema: WEIGHT_PATCHES_SCHEMA.to_string(),
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
            .map(|(a, edits)| LedgerTarget {
                object: a.entry.object.clone(),
                representation: a.canonical_id.clone(),
                tensor: a.tensor.name.clone(),
                dtype: a.tensor.dtype.clone(),
                shape: a.tensor.shape.clone(),
                edits: edits.clone(),
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
