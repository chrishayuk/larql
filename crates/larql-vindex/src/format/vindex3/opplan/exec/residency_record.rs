//! A record of what each matrix operand was made resident as.
//!
//! A container's stored weights and the weights a run computes with are not always the same bytes: a loader that
//! is permitted to may manufacture a representation at load (`docs/cpu-load-time-q8-residency.md`). The source a
//! run was ASKED for does not say what it got. This records, per operand and at the one place operands are loaded,
//! what the loader read and what it left resident, so that something outside the engine can judge both the
//! derivation and the selection.
//!
//! **It only observes.** It reads the stored bytes and the resident bytes and digests them. It computes no
//! expected value and judges nothing: a check written here would be the engine marking its own work.
//!
//! **Off unless asked for.** The idle cost is one uncontended lock per operand loaded. While recording, every
//! operand's stored bytes are read a second time and digested, so do not compare load times from a recording run
//! with others.
//!
//! **Per store, not per process.** A recording is opened on one [`OperandStore`](super::operands::OperandStore)
//! ([`OperandStore::record_residency`](super::operands::OperandStore::record_residency)) and sees that store's
//! loads only, from whichever pool worker performs them. Two containers opened in one process keep two records.

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::backend::WeightFormat;
use super::operands::{OperandSource, RepresentationSource};
use super::weights::LoadedWeight;
use crate::format::vindex3::opplan::OperandRef;

/// The record format's name.
pub const SCHEMA: &str = "larql-residency-record/1";

/// The derivation rule of the loader's Q8 residency (`quantise::quantise_q8`): blocks of
/// [`super::quantise::Q8_BLOCK`] elements along a row, an f32 scale of `peak / 127` per block (1 for an all-zero
/// block), codes `round_half_away(w / scale)` clamped to ±127.
pub const RULE_Q8_LOAD: &str = "q8-load/1";
/// The resident bytes are the stored bytes, copied.
pub const RULE_STORED_BYTES: &str = "stored-bytes";
/// The resident bytes are the stored bytes, mapped in place: nothing was copied or converted.
pub const RULE_STORED_BYTES_MAPPED: &str = "stored-bytes-mapped";
/// The resident values are the stored elements widened exactly to binary32: a BF16 pattern `h` becomes the
/// binary32 with bits `h << 16`, and a stored binary32 is itself. Nothing is rounded.
pub const RULE_WIDEN_F32: &str = "widen-f32/1";
/// How the resident digest of a widened matrix is taken: its values as little-endian binary32, row-major.
pub const F32_DIGEST_LAYOUT: &str = "f32le-row-major";
/// A resident form this record names and does not digest: no rule is written down for it here.
pub const RULE_UNSPECIFIED: &str = "unspecified";

/// How the resident digest of a Q8 residency is taken: the codes, one byte each in row-major order, then the
/// scales as little-endian binary32, row by row and block by block. The code-sum index is derived and is not
/// part of it.
pub const Q8_DIGEST_LAYOUT: &str = "codes-i8-row-major||scales-f32le-row-then-block";

/// One operand as it was loaded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperandRecord {
    /// The order operands reached the record, from 0, assigned as each is appended. Loads run concurrently, so
    /// it is not reproducible.
    pub order: u64,
    pub object: String,
    pub tensor: String,
    pub shape: Vec<usize>,
    /// The dtype the container stores the tensor as; `None` for an operand carrying overlay edits, which has no
    /// stored bytes to read.
    pub stored_dtype: Option<String>,
    /// SHA-256 of the stored bytes the loader read; `None` as for `stored_dtype`.
    pub stored_sha256: Option<String>,
    /// The source the store was opened under: what the run was ASKED for.
    pub representation_source: &'static str,
    /// The format the selector asked the loader for.
    pub requested_format: String,
    /// The format left resident: what the run GOT.
    pub resident_format: String,
    /// Whether the resident bytes were manufactured in this process rather than read from the container.
    pub manufactured: bool,
    /// The rule that takes the stored bytes to the resident ones.
    pub rule: &'static str,
    /// SHA-256 of the resident bytes, in `resident_digest_layout`; `None` where the rule is
    /// [`RULE_STORED_BYTES_MAPPED`] (the bytes are the stored ones) or [`RULE_UNSPECIFIED`].
    pub resident_sha256: Option<String>,
    pub resident_digest_layout: Option<&'static str>,
}

fn sha256_hex(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A representation source as `--representation-source` spells it.
pub fn source_name(source: RepresentationSource) -> &'static str {
    match source {
        RepresentationSource::Auto => "auto",
        RepresentationSource::Stored => "stored",
        RepresentationSource::Transient => "transient",
    }
}

/// The rule and the resident digest of a loaded weight.
fn resident(loaded: &LoadedWeight) -> (&'static str, Option<String>, Option<&'static str>) {
    match loaded {
        LoadedWeight::Q8 { codes, scales, .. } => {
            let code_bytes: Vec<u8> = codes.iter().map(|c| *c as u8).collect();
            let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
            (
                RULE_Q8_LOAD,
                Some(sha256_hex(&[&code_bytes, &scale_bytes])),
                Some(Q8_DIGEST_LAYOUT),
            )
        }
        // The buffer is padded to a page; only its logical front is the tensor.
        LoadedWeight::Bf16(bytes) => (
            RULE_STORED_BYTES,
            Some(sha256_hex(&[&bytes.as_slice()[..bytes.logical_len()]])),
            Some("stored-bytes"),
        ),
        LoadedWeight::F32(values) => {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            (
                RULE_WIDEN_F32,
                Some(sha256_hex(&[&bytes])),
                Some(F32_DIGEST_LAYOUT),
            )
        }
        LoadedWeight::Mapped { .. } => (RULE_STORED_BYTES_MAPPED, None, None),
        _ => (RULE_UNSPECIFIED, None, None),
    }
}

/// Record one loaded operand. Called by the weight loader, the one place an operand becomes resident.
pub(super) fn note(
    store: OperandSource<'_>,
    operand: &OperandRef,
    requested: WeightFormat,
    loaded: &LoadedWeight,
) {
    if !store.store().records_residency() {
        return;
    }
    // An operand carrying overlay edits has no stored bytes that mean what it executes; `load_raw` refuses it.
    let stored = store.load_raw(operand).ok();
    let (rule, resident_sha256, resident_digest_layout) = resident(loaded);
    store.store().push_residency(OperandRecord {
        order: 0,
        object: operand.object.clone(),
        tensor: operand.tensor.clone(),
        shape: operand.shape.clone(),
        stored_dtype: stored.as_ref().map(|raw| raw.dtype.clone()),
        stored_sha256: stored.as_ref().map(|raw| sha256_hex(&[&raw.bytes])),
        representation_source: source_name(store.store().representation_source()),
        requested_format: format!("{requested:?}"),
        resident_format: format!("{:?}", loaded.format()),
        manufactured: matches!(rule, RULE_Q8_LOAD),
        rule,
        resident_sha256,
        resident_digest_layout,
    });
}

/// What decided the selection, for the whole process: recorded beside the operands so that a reader can judge
/// whether each choice was the one these inputs give.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SelectionInputs {
    /// The cache size the size policy compares a matrix's stored image against, in bytes.
    pub l2_bytes: usize,
    /// `LARQL_CPU_MAX_FORMAT`, `LARQL_CPU_ARITHMETIC` and `LARQL_KQUANT_EXEC` as set, or `None`.
    pub max_format: Option<String>,
    pub arithmetic: Option<String>,
    pub kquant_exec: Option<String>,
}

impl SelectionInputs {
    /// Read from this process.
    pub fn observe() -> Self {
        use super::cpu::physical::{
            compact_threshold_bytes, ARITHMETIC_ARM_ENV, KQUANT_EXEC_ENV, MAX_FORMAT_ENV,
        };
        let var = |name: &str| std::env::var(name).ok();
        Self {
            l2_bytes: compact_threshold_bytes(),
            max_format: var(MAX_FORMAT_ENV),
            arithmetic: var(ARITHMETIC_ARM_ENV),
            kquant_exec: var(KQUANT_EXEC_ENV),
        }
    }
}

/// The record as one document.
pub fn document(
    container: &str,
    container_index_sha256: &str,
    operands: &[OperandRecord],
) -> serde_json::Value {
    serde_json::json!({
        "schema": SCHEMA,
        "container": container,
        "container_index_sha256": container_index_sha256,
        "selection_inputs": SelectionInputs::observe(),
        "rules": {
            RULE_Q8_LOAD: {
                "block_elements": super::quantise::Q8_BLOCK,
                "scale": "f32 peak / 127 per block along a row; 1 for an all-zero block",
                "code": "round_half_away(w / scale), clamped to [-127, 127]",
                "digest_layout": Q8_DIGEST_LAYOUT,
            },
            RULE_WIDEN_F32: {
                "value": "each stored element widened exactly to binary32 (a BF16 pattern h is the binary32 h << 16)",
                "digest_layout": F32_DIGEST_LAYOUT,
            },
            RULE_STORED_BYTES: {
                "value": "the stored bytes, copied",
                "digest_layout": "stored-bytes",
            },
        },
        "operands": operands,
    })
}
