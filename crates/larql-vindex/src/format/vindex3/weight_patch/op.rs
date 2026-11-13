//! The `.vlp` form of a weight edit: a portable, versioned instruction to
//! change stored weights, bound to the exact artifact it was written against.
//!
//! A patch file is an instruction; a compiled container is the result of
//! following it. They are different artifacts and are verified separately.
//! This op carries what a reader needs to refuse a mismatched target:
//!
//! - the **source binding**: the SHA-256 of the source `index.json` (which
//!   exact container revision) and the digest of the canonical weights being
//!   edited (which exact bytes). The upstream model name alone is not enough:
//!   two different containers of one model must not both accept an edit
//!   written for one of them.
//! - the **rounding rule**, spelled out, so a reader that applies it
//!   differently refuses and does not guess;
//! - a **digest over the op's own fields**, so a patch that was corrupted or
//!   edited by hand is refused at load. The digest detects accidental damage
//!   and casual edits; it is not a signature and does not stop someone who
//!   recomputes it. Whether the container the patch produces is correct is
//!   for the independent verifiers.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::error::WeightPatchError;
use super::ledger::ROUNDING_RULE;
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::index::Vindex3Index;

/// Version tag mixed into the op digest, so a change to what the digest
/// covers is a new digest and never a silent reinterpretation.
const DIGEST_DOMAIN: &str = "weight_delta/v1";

/// Which exact artifact and weights an edit was written against.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceBinding {
    /// Upstream model identity (informational: identity is the two digests).
    pub model: String,
    /// SHA-256 of the source container's `index.json`.
    pub index_sha256: String,
    /// Canonical representation id holding the tensor, e.g.
    /// `target.decoder_stack@BF16`.
    pub representation: String,
    /// Payload digest of that canonical representation.
    pub payload_sha256: String,
}

/// One edited element: the exact `f32` bits of the delta are authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeltaEdit {
    pub row: usize,
    pub col: usize,
    /// For readers; `delta_bits` decides.
    pub delta: f32,
    pub delta_bits: u32,
}

/// A weight edit as it appears in a `.vlp` patch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightDeltaOp {
    pub rounding: String,
    pub source: SourceBinding,
    pub tensor: String,
    pub edits: Vec<DeltaEdit>,
    /// SHA-256 over every other field; see the module note.
    pub digest: String,
}

impl WeightDeltaOp {
    /// Build an op and seal it with its digest.
    pub fn new(source: SourceBinding, tensor: String, edits: Vec<DeltaEdit>) -> Self {
        let mut op = Self {
            rounding: ROUNDING_RULE.to_string(),
            source,
            tensor,
            edits,
            digest: String::new(),
        };
        op.digest = op.compute_digest();
        op
    }

    /// The digest of this op's fields, independent of the stored `digest`.
    pub fn compute_digest(&self) -> String {
        let mut edits = self.edits.clone();
        edits.sort_by_key(|e| (e.row, e.col));
        let mut text = format!(
            "{DIGEST_DOMAIN}\nrounding={}\nmodel={}\nindex_sha256={}\nrepresentation={}\npayload_sha256={}\ntensor={}\n",
            self.rounding, self.source.model, self.source.index_sha256, self.source.representation,
            self.source.payload_sha256, self.tensor
        );
        for e in &edits {
            text.push_str(&format!("{},{},{}\n", e.row, e.col, e.delta_bits));
        }
        super::apply::hex(&Sha256::digest(text.as_bytes()))
    }

    /// Whether the op is intact and says what this build implements. Does not
    /// look at any container.
    pub fn check_intact(&self) -> Result<(), WeightPatchError> {
        if self.rounding != ROUNDING_RULE {
            return Err(WeightPatchError::PatchOp(format!(
                "names rounding `{}`; this build implements `{ROUNDING_RULE}`",
                self.rounding
            )));
        }
        if self.compute_digest() != self.digest {
            return Err(WeightPatchError::PatchOp(
                "its digest does not match its contents: the patch was corrupted or edited"
                    .to_string(),
            ));
        }
        if self.edits.iter().any(|e| e.delta.to_bits() != e.delta_bits) {
            return Err(WeightPatchError::PatchOp(
                "a delta disagrees with its recorded bits".to_string(),
            ));
        }
        Ok(())
    }

    /// Refuse unless `container` is the exact artifact this edit was written
    /// against.
    pub fn check_binding(&self, container: &Path) -> Result<(), WeightPatchError> {
        check_same_source(&self.source, container, &self.tensor)
    }

    /// The edits as library edits.
    pub fn weight_edits(&self) -> Vec<super::WeightEdit> {
        self.edits
            .iter()
            .map(|e| super::WeightEdit {
                row: e.row,
                col: e.col,
                delta: f32::from_bits(e.delta_bits),
            })
            .collect()
    }
}

/// Refuse unless `container` is the exact artifact `source` binds: the same index, representation and
/// payload of `tensor`. Shared by every stored-weight op.
pub(super) fn check_same_source(
    source: &SourceBinding,
    container: &Path,
    tensor: &str,
) -> Result<(), WeightPatchError> {
    let now = bind_source(container, tensor)?;
    if &now != source {
        return Err(WeightPatchError::WrongSource {
            expected: format!(
                "{} / {}",
                short(&source.index_sha256),
                short(&source.payload_sha256)
            ),
            found: format!(
                "{} / {}",
                short(&now.index_sha256),
                short(&now.payload_sha256)
            ),
        });
    }
    Ok(())
}

fn short(digest: &str) -> &str {
    digest.get(..12).unwrap_or(digest)
}

/// The binding of `container` for an edit of `tensor`: its exact index digest
/// and the digest of the canonical weights that hold the tensor.
pub fn bind_source(container: &Path, tensor: &str) -> Result<SourceBinding, WeightPatchError> {
    let raw = std::fs::read_to_string(container.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&raw)
        .map_err(|e| crate::error::VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    let mut holders = Vec::new();
    for (id, entry) in index
        .representations
        .iter()
        .filter(|(_, e)| e.compiled_from.is_none())
    {
        let (header, _) = read_segment_header(&container.join(&entry.segment))?;
        if header.tensors.iter().any(|t| t.name == tensor) {
            holders.push((id.clone(), entry.payload_sha256.clone()));
        }
    }
    match holders.len() {
        0 => Err(WeightPatchError::UnknownTensor(tensor.to_string())),
        1 => {
            let (representation, payload_sha256) = holders.remove(0);
            Ok(SourceBinding {
                model: index.model,
                index_sha256: super::apply::hex(&Sha256::digest(raw.as_bytes())),
                representation,
                payload_sha256,
            })
        }
        _ => Err(WeightPatchError::AmbiguousTensor {
            tensor: tensor.to_string(),
            holders: holders.into_iter().map(|h| h.0).collect(),
        }),
    }
}

/// Build the sealed `.vlp` op for `edits` on `tensor` of `container`.
pub fn record_op(
    container: &Path,
    tensor: &str,
    edits: &[super::WeightEdit],
) -> Result<WeightDeltaOp, WeightPatchError> {
    let source = bind_source(container, tensor)?;
    let edits = edits
        .iter()
        .map(|e| DeltaEdit {
            row: e.row,
            col: e.col,
            delta: e.delta,
            delta_bits: e.delta.to_bits(),
        })
        .collect();
    Ok(WeightDeltaOp::new(source, tensor.to_string(), edits))
}
