//! The weight-patch ledger: how a container was derived from another by a
//! declared edit of stored weights.
//!
//! It names the source (model identity, exact index digest, and the digests
//! of the representations that were edited and re-derived), the edited
//! tensor, every edited element with the exact old and new bits, and the
//! digests of what resulted. The result's canonical digests are the identity
//! of the EDITED source: an independent verifier measures the re-derived
//! representation against that, not against the unedited upstream weights.
//!
//! [`check_ledger`] checks CONSISTENCY: that the ledger agrees with the
//! container's catalogue and with the bytes at the edited elements. It does
//! not verify that the patch is correct. A passing check means the ledger is
//! a faithful description of this container, which is a weaker claim than an
//! independent verifier's "this container implements the declared edit".

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::index::Vindex3Index;

/// Current ledger schema. Bump on any breaking change.
pub const WEIGHT_PATCHES_SCHEMA: &str = "weight-patches/v1";
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerTarget {
    pub object: String,
    /// The canonical representation id that was edited.
    pub representation: String,
    pub tensor: String,
    pub dtype: String,
    pub shape: Vec<usize>,
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
    pub target: LedgerTarget,
    pub edits: Vec<LedgerEdit>,
    /// Digests of the `.vlp` weight ops this edit was assembled from, when
    /// it came from patches; empty for an edit made directly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub origins: Vec<String>,
    pub result: LedgerResult,
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
    let ledger: WeightPatchLedger = match serde_json::from_str(&text) {
        Ok(l) => l,
        Err(e) => return vec![format!("weight-patch ledger `{name}` is malformed: {e}")],
    };
    if ledger.schema != WEIGHT_PATCHES_SCHEMA {
        defects.push(format!(
            "weight-patch ledger schema `{}` is not `{WEIGHT_PATCHES_SCHEMA}`",
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
    let rep = &ledger.target.representation;
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
    defects.extend(check_edits(root, &canonical.segment, &ledger));
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

fn check_edits(root: &Path, segment: &str, ledger: &WeightPatchLedger) -> Vec<String> {
    let path = root.join(segment);
    let (header, payload_start) = match read_segment_header(&path) {
        Ok(h) => h,
        Err(e) => {
            return vec![format!(
                "cannot read `{segment}` to check the ledger's edits: {e}"
            )]
        }
    };
    let Some(tensor) = header
        .tensors
        .iter()
        .find(|t| t.name == ledger.target.tensor)
    else {
        return vec![format!(
            "ledger edits tensor `{}`, which `{segment}` does not hold",
            ledger.target.tensor
        )];
    };
    let mut defects = Vec::new();
    if tensor.dtype != ledger.target.dtype || tensor.shape != ledger.target.shape {
        defects.push(
            "ledger target dtype or shape differs from the segment's tensor table".to_string(),
        );
    }
    let (rows, cols) = match tensor.shape.as_slice() {
        [r, c] => (*r, *c),
        _ => return vec!["ledger target is not a matrix".to_string()],
    };
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => return vec![format!("cannot open `{segment}`: {e}")],
    };
    let mut seen = BTreeMap::new();
    for e in &ledger.edits {
        if e.row >= rows || e.col >= cols {
            defects.push(format!(
                "ledger edit ({}, {}) is outside the {rows}x{cols} tensor",
                e.row, e.col
            ));
            continue;
        }
        if seen.insert((e.row, e.col), ()).is_some() {
            defects.push(format!("ledger edits element ({}, {}) twice", e.row, e.col));
        }
        if e.delta.to_bits() != e.delta_bits {
            defects.push(format!(
                "ledger edit ({}, {}) delta and delta_bits disagree",
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
            defects.push(format!("cannot read element ({}, {}): {err}", e.row, e.col));
            continue;
        }
        if u16::from_le_bytes(stored) != e.new_bits {
            defects.push(format!(
                "element ({}, {}) holds {:#06x}, the ledger says {:#06x}",
                e.row,
                e.col,
                u16::from_le_bytes(stored),
                e.new_bits
            ));
        }
    }
    defects
}
