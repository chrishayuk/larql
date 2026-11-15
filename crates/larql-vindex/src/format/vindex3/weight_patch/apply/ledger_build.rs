//! The ledger of an applied patch: the source, every edit, and the resulting digests.

use std::path::Path;

use sha2::{Digest, Sha256};

use super::super::ledger::{
    DigestPair, EditEncoding, LedgerResult, LedgerSource, LedgerTarget, ResultDerived,
    SourceDerived, WeightPatchLedger, ROUNDING_RULE, WEIGHT_PATCHES_SCHEMA,
    WEIGHT_PATCHES_SCHEMA_V3, WEIGHT_PATCHES_SCHEMA_V4,
};
use super::super::op_target::tensor_sha256;
use super::locate::{Located, Resolved};
use super::{hex, read_index, WRITER};
use crate::format::vindex3::encode::segment::WrittenSegment;
use crate::format::vindex3::index::{RepresentationEntry, Vindex3Index};

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

pub(super) fn build_ledger(
    source_index: &Vindex3Index,
    raw_source_index: &str,
    edited: &[(Located, Resolved<'_>)],
    written: &WrittenSegment,
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
