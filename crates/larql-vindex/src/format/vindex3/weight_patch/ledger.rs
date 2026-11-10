//! The weight-patch ledger: how a container was derived from another by a
//! declared edit of stored weights.
//!
//! It names the source (model identity, exact index digest, and the digests
//! of the representations that were edited and re-derived), each edited
//! tensor with every edited element and its exact old and new bits, and the
//! digests of what resulted. The result's canonical digests are the identity
//! of the EDITED source: an independent verifier measures the re-derived
//! representation against that, not against the unedited upstream weights.
//!
//! [`check_ledger`] checks CONSISTENCY: that the ledger agrees with the
//! container's catalogue and with the bytes at the edited elements. It does
//! not verify that the patch is correct. A passing check means the ledger is
//! a faithful description of this container, which is a weaker claim than an
//! independent verifier's "this container implements the declared edit".
//!
//! Schema `v2` edits one or more tensors of ONE canonical representation
//! (`targets`). A `v1` ledger (one `target`, its `edits` beside it) is still
//! read, as the one-target `v2` it is equivalent to; writers emit `v2` only.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::index::Vindex3Index;

/// Current ledger schema. Bump on any breaking change.
pub const WEIGHT_PATCHES_SCHEMA: &str = "weight-patches/v2";
/// The previous schema: exactly one target. Still read; never written.
pub const WEIGHT_PATCHES_SCHEMA_V1: &str = "weight-patches/v1";
/// The rounding rule, spelled out so a verifier cannot guess it.
pub const ROUNDING_RULE: &str = "bf16-rne-of-exact-sum";
/// Bytes per stored bfloat16 element.
const BF16_BYTES: u64 = 2;

/// Payload and whole-segment digests of one representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DigestPair {
    pub payload_sha256: String,
    pub segment_sha256: String,
}

/// A derived representation as it stood before the edit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDerived {
    pub representation: String,
    pub encoding: String,
    #[serde(flatten)]
    pub digests: DigestPair,
    /// The encoder identity the source's pack records (it may differ in
    /// revision from the result's with identical bytes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder: Option<serde_json::Value>,
}

/// A derived representation as re-derived from the edited source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResultDerived {
    pub representation: String,
    pub encoding: String,
    /// The edited canonical payload digest this was derived from.
    pub source_representation_digest: String,
    #[serde(flatten)]
    pub digests: DigestPair,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerSource {
    /// The upstream model identity, unchanged by the patch.
    pub model: String,
    /// SHA-256 of the source container's `index.json`: which exact revision
    /// was modified.
    pub index_sha256: String,
    pub canonical: DigestPair,
    pub derived: Vec<SourceDerived>,
}

/// One edited tensor and every element edited in it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerTarget {
    pub object: String,
    /// The canonical representation id that was edited.
    pub representation: String,
    pub tensor: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub edits: Vec<LedgerEdit>,
}

/// One edited element, with the exact bits before and after.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEdit {
    pub row: usize,
    pub col: usize,
    pub delta: f32,
    /// The exact bits of `delta`, so the value is not a decimal's reading.
    pub delta_bits: u32,
    pub old_bits: u16,
    pub new_bits: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerResult {
    /// Digests of the EDITED canonical representation: the edited source.
    pub canonical: DigestPair,
    pub derived: Vec<ResultDerived>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightPatchLedger {
    pub schema: String,
    pub rounding: String,
    pub writer: String,
    pub source: LedgerSource,
    /// The edited tensors, by name, all in one canonical representation.
    pub targets: Vec<LedgerTarget>,
    /// Digests of the `.vlp` weight ops this edit was assembled from, when
    /// it came from patches; empty for an edit made directly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub origins: Vec<String>,
    pub result: LedgerResult,
}

impl WeightPatchLedger {
    /// Every edited element across all targets.
    pub fn edit_count(&self) -> usize {
        self.targets.iter().map(|t| t.edits.len()).sum()
    }
}

/// The `v1` shape: one target, its edits beside it.
#[derive(Deserialize)]
struct LedgerV1 {
    schema: String,
    rounding: String,
    writer: String,
    source: LedgerSource,
    target: TargetV1,
    edits: Vec<LedgerEdit>,
    #[serde(default)]
    origins: Vec<String>,
    result: LedgerResult,
}

#[derive(Deserialize)]
struct TargetV1 {
    object: String,
    representation: String,
    tensor: String,
    dtype: String,
    shape: Vec<usize>,
}

/// Read a ledger of either schema. A `v1` ledger becomes the one-target
/// ledger it describes, keeping its `schema` string so a reader can tell.
pub fn parse_ledger(text: &str) -> Result<WeightPatchLedger, String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    if value.get("schema").and_then(|s| s.as_str()) == Some(WEIGHT_PATCHES_SCHEMA_V1) {
        let v1: LedgerV1 = serde_json::from_value(value).map_err(|e| e.to_string())?;
        return Ok(WeightPatchLedger {
            schema: v1.schema,
            rounding: v1.rounding,
            writer: v1.writer,
            source: v1.source,
            targets: vec![LedgerTarget {
                object: v1.target.object,
                representation: v1.target.representation,
                tensor: v1.target.tensor,
                dtype: v1.target.dtype,
                shape: v1.target.shape,
                edits: v1.edits,
            }],
            origins: v1.origins,
            result: v1.result,
        });
    }
    serde_json::from_value(value).map_err(|e| e.to_string())
}

/// Defects of `ledger` against the container at `root`, empty when it is a
/// faithful description. See the module note: consistency, not correctness.
pub fn check_ledger(root: &Path, index: &Vindex3Index) -> Vec<String> {
    let Some(name) = index.weight_patches.as_deref() else {
        return Vec::new();
    };
    let mut defects = Vec::new();
    let text = match std::fs::read_to_string(root.join(name)) {
        Ok(t) => t,
        Err(e) => {
            return vec![format!(
                "weight-patch ledger `{name}` is named but unreadable: {e}"
            )]
        }
    };
    let ledger = match parse_ledger(&text) {
        Ok(l) => l,
        Err(e) => return vec![format!("weight-patch ledger `{name}` is malformed: {e}")],
    };
    if ledger.schema != WEIGHT_PATCHES_SCHEMA && ledger.schema != WEIGHT_PATCHES_SCHEMA_V1 {
        defects.push(format!(
            "weight-patch ledger schema `{}` is not `{WEIGHT_PATCHES_SCHEMA}` (or `{WEIGHT_PATCHES_SCHEMA_V1}`)",
            ledger.schema
        ));
    }
    if ledger.rounding != ROUNDING_RULE {
        defects.push(format!(
            "weight-patch ledger names rounding `{}`, not `{ROUNDING_RULE}`",
            ledger.rounding
        ));
    }
    if ledger.source.model != index.model {
        defects.push(format!(
            "ledger source model `{}` differs from the container's `{}`",
            ledger.source.model, index.model
        ));
    }
    let Some(first) = ledger.targets.first() else {
        defects.push("weight-patch ledger lists no edited tensor".to_string());
        return defects;
    };
    let rep = &first.representation;
    if let Some(other) = ledger.targets.iter().find(|t| &t.representation != rep) {
        defects.push(format!(
            "ledger edits tensors of two canonical representations, `{rep}` and `{}`",
            other.representation
        ));
    }
    let mut named = BTreeSet::new();
    for t in &ledger.targets {
        if !named.insert(t.tensor.as_str()) {
            defects.push(format!("ledger lists tensor `{}` twice", t.tensor));
        }
    }
    let Some(canonical) = index.representations.get(rep) else {
        defects.push(format!(
            "ledger edits representation `{rep}`, which the container does not hold"
        ));
        return defects;
    };
    if canonical.compiled_from.is_some() {
        defects.push(format!(
            "ledger edits `{rep}`, which is a derived representation, not a canonical one"
        ));
    }
    if canonical.payload_sha256 != ledger.result.canonical.payload_sha256
        || canonical.segment_sha256 != ledger.result.canonical.segment_sha256
    {
        defects.push(format!(
            "ledger result digests for `{rep}` differ from the catalogue's"
        ));
    }
    defects.extend(check_derived(index, rep, &ledger));
    for target in &ledger.targets {
        defects.extend(check_edits(root, &canonical.segment, target));
    }
    defects
}

fn check_derived(index: &Vindex3Index, canonical: &str, ledger: &WeightPatchLedger) -> Vec<String> {
    let mut defects = Vec::new();
    let held: BTreeSet<&str> = index
        .representations
        .iter()
        .filter(|(_, e)| e.compiled_from.as_deref() == Some(canonical))
        .map(|(id, _)| id.as_str())
        .collect();
    let listed: BTreeSet<&str> = ledger
        .result
        .derived
        .iter()
        .map(|d| d.representation.as_str())
        .collect();
    if held != listed {
        defects.push(format!(
            "ledger lists derived representations {listed:?} but the container holds {held:?} derived from `{canonical}`"
        ));
    }
    for d in &ledger.result.derived {
        let Some(entry) = index.representations.get(&d.representation) else {
            continue;
        };
        if entry.payload_sha256 != d.digests.payload_sha256
            || entry.segment_sha256 != d.digests.segment_sha256
        {
            defects.push(format!(
                "ledger digests for `{}` differ from the catalogue's",
                d.representation
            ));
        }
        if entry.source_representation_digest.as_deref()
            != Some(ledger.result.canonical.payload_sha256.as_str())
            || d.source_representation_digest != ledger.result.canonical.payload_sha256
        {
            defects.push(format!(
                "`{}` does not derive from the edited canonical payload the ledger names",
                d.representation
            ));
        }
    }
    defects
}

fn check_edits(root: &Path, segment: &str, target: &LedgerTarget) -> Vec<String> {
    if target.edits.is_empty() {
        // A writer never records a tensor it did not edit; a ledger that does
        // names an edit that is not there.
        return vec![format!(
            "ledger lists tensor `{}` with no edited element",
            target.tensor
        )];
    }
    let path = root.join(segment);
    let (header, payload_start) = match read_segment_header(&path) {
        Ok(h) => h,
        Err(e) => {
            return vec![format!(
                "cannot read `{segment}` to check the ledger's edits: {e}"
            )]
        }
    };
    let Some(tensor) = header.tensors.iter().find(|t| t.name == target.tensor) else {
        return vec![format!(
            "ledger edits tensor `{}`, which `{segment}` does not hold",
            target.tensor
        )];
    };
    let mut defects = Vec::new();
    if tensor.dtype != target.dtype || tensor.shape != target.shape {
        defects.push(format!(
            "ledger target `{}` dtype or shape differs from the segment's tensor table",
            target.tensor
        ));
    }
    let (rows, cols) = match tensor.shape.as_slice() {
        [r, c] => (*r, *c),
        _ => return vec![format!("ledger target `{}` is not a matrix", target.tensor)],
    };
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => return vec![format!("cannot open `{segment}`: {e}")],
    };
    let mut seen = BTreeMap::new();
    let name = &target.tensor;
    for e in &target.edits {
        if e.row >= rows || e.col >= cols {
            defects.push(format!(
                "ledger edit ({}, {}) is outside the {rows}x{cols} tensor `{name}`",
                e.row, e.col
            ));
            continue;
        }
        if seen.insert((e.row, e.col), ()).is_some() {
            defects.push(format!(
                "ledger edits element ({}, {}) of `{name}` twice",
                e.row, e.col
            ));
        }
        if e.delta.to_bits() != e.delta_bits {
            defects.push(format!(
                "ledger edit ({}, {}) of `{name}`: delta and delta_bits disagree",
                e.row, e.col
            ));
        }
        let at = payload_start
            + tensor.offset
            + (e.row as u64 * cols as u64 + e.col as u64) * BF16_BYTES;
        let mut stored = [0u8; BF16_BYTES as usize];
        if let Err(err) =
            crate::format::vindex3::encode::segment::read_exact_at(&file, &mut stored, at)
        {
            defects.push(format!(
                "cannot read element ({}, {}) of `{name}`: {err}",
                e.row, e.col
            ));
            continue;
        }
        if u16::from_le_bytes(stored) != e.new_bits {
            defects.push(format!(
                "element ({}, {}) of `{name}` holds {:#06x}, the ledger says {:#06x}",
                e.row,
                e.col,
                u16::from_le_bytes(stored),
                e.new_bits
            ));
        }
    }
    defects
}
