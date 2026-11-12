//! The `.vlp` form of a **structured** weight edit: `remove_direction`, declared by its operands, not by
//! the elements it changes.
//!
//! A [`super::WeightDeltaOp`] lists elements and f32 deltas. That cannot always carry what the
//! structured numerics compute: the BF16 result of `w - (alpha * u) * c` can be an element a single f32
//! delta added to the stored value never rounds to (a near-cancellation leaves a tiny nonzero where the
//! delta would have to name a value below the f32 spacing of the old one). This op declares the
//! operation instead:
//!
//! - the **operand** `u`, as stored bit patterns, and `alpha` as bits;
//! - the **columns** it is applied to and the **numerical profile** that fixes every rounding;
//! - the same **source binding** as a delta op: the exact container and weights it was written for.
//!
//! [`StructuredWeightOp::resolve`] runs the frozen interpreter ([`super::structured::run`]) over the
//! bound tensor and returns the exact BF16 results of the elements that change, ready for
//! [`super::apply_declared_patches`]. The compiled container then records those results, not deltas.
//!
//! Like the delta op, the digest detects accidental damage and casual edits and is not a signature.
//! Whether a container implements the declared op is for an independent verifier, which recomputes the
//! op by a second route and compares every element.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::apply::{read_bf16_tensor, ResultEdit, ResultPatch};
use super::error::WeightPatchError;
use super::op::{bind_source, SourceBinding};
use super::structured::{
    run, BaseTensor, OpKind, OperandBits, Profile, ScalarBits, StructuredOp, TensorRole,
};

/// Version tag mixed into the digest, so a change to what it covers is a new digest.
const DIGEST_DOMAIN: &str = "structured_op/remove_direction/v1";
/// Hex digits of one stored operand element: eight for f32 bits, sixteen for f64.
const F32_HEX_DIGITS: usize = 8;
const F64_HEX_DIGITS: usize = 16;
/// Tensors whose name contains this are the token embedding, which doubles as the tied output head. The
/// interpreter refuses to edit it.
const EMBEDDING_MARKER: &str = "embed_tokens";

/// A `remove_direction` op as it appears in a `.vlp` patch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredWeightOp {
    pub source: SourceBinding,
    pub tensor: String,
    /// The numerical profile id, such as `f64-seq-rne1/v1`.
    pub numerics: String,
    /// `u`, one fixed-width lowercase hex string per element: the stored bits, never a decimal.
    pub operand: Vec<String>,
    /// `alpha`, as stored bits in the profile's working width.
    pub alpha: String,
    /// Strictly increasing column indices.
    pub columns: Vec<usize>,
    /// Where the operand came from, if its writer says: a feature artifact id, a file hash, a note. The
    /// operand's bits are what the op computes with; this only makes its provenance separately identifiable
    /// rather than lost inside the op. It is covered by the digest and read by nothing else.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// SHA-256 over every other field; see the module note.
    pub digest: String,
}

impl StructuredWeightOp {
    /// Build an op bound to `container` and seal it. (Named `bind`, not `record`: it makes an instruction,
    /// it does not record a measurement, and the ingestion-closure guard counts `record` calls.) The operand and alpha are the profile's working width:
    /// f64 bits for the f64 profiles, f32 bits for the f32 one.
    pub fn bind(
        container: &Path,
        tensor: &str,
        numerics: &str,
        operand: &OperandBits,
        alpha: ScalarBits,
        columns: Vec<usize>,
    ) -> Result<Self, WeightPatchError> {
        let source = bind_source(container, tensor)?;
        let hex = |bits: &OperandBits| -> Vec<String> {
            match bits {
                OperandBits::F64(v) => v.iter().map(|b| format!("{b:016x}")).collect(),
                OperandBits::F32(v) => v.iter().map(|b| format!("{b:08x}")).collect(),
            }
        };
        let alpha = match alpha {
            ScalarBits::F64(b) => format!("{b:016x}"),
            ScalarBits::F32(b) => format!("{b:08x}"),
        };
        let mut op = Self {
            source,
            tensor: tensor.to_string(),
            numerics: numerics.to_string(),
            operand: hex(operand),
            alpha,
            columns,
            origin: None,
            digest: String::new(),
        };
        op.digest = op.compute_digest();
        Ok(op)
    }

    /// This op with its operand's origin recorded, re-sealed.
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = Some(origin.into());
        self.digest = self.compute_digest();
        self
    }

    /// The digest of this op's fields, independent of the stored `digest`.
    pub fn compute_digest(&self) -> String {
        let columns: Vec<String> = self.columns.iter().map(|c| c.to_string()).collect();
        let text =
            format!(
            "{DIGEST_DOMAIN}\nmodel={}\nindex_sha256={}\nrepresentation={}\npayload_sha256={}\n\
             tensor={}\nnumerics={}\nalpha={}\ncolumns={}\noperand={}\norigin={}\n",
            self.source.model,
            self.source.index_sha256,
            self.source.representation,
            self.source.payload_sha256,
            self.tensor,
            self.numerics,
            self.alpha,
            columns.join(","),
            self.operand.join(","),
            self.origin.as_deref().map_or("-".to_string(), |o| format!("+{o}")),
        );
        super::apply::hex(&Sha256::digest(text.as_bytes()))
    }

    /// Whether the op is intact, names a profile this build implements, and carries operands of that
    /// profile's width. Does not look at any container.
    pub fn check_intact(&self) -> Result<(), WeightPatchError> {
        if self.compute_digest() != self.digest {
            return Err(WeightPatchError::PatchOp(
                "its digest does not match its contents: the patch was corrupted or edited"
                    .to_string(),
            ));
        }
        self.operands().map(|_| ())
    }

    /// Refuse unless `container` is the exact artifact this op was written against.
    pub fn check_binding(&self, container: &Path) -> Result<(), WeightPatchError> {
        let now = bind_source(container, &self.tensor)?;
        if now != self.source {
            return Err(WeightPatchError::WrongSource {
                expected: format!(
                    "{} / {}",
                    short(&self.source.index_sha256),
                    short(&self.source.payload_sha256)
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

    /// The operand and alpha as the interpreter takes them.
    fn operands(&self) -> Result<(OperandBits, ScalarBits), WeightPatchError> {
        let profile = Profile::parse(Some(&self.numerics))
            .map_err(|r| WeightPatchError::PatchOp(r.to_string()))?;
        let width = if profile.works_in_f64() {
            F64_HEX_DIGITS
        } else {
            F32_HEX_DIGITS
        };
        let parse = |text: &str| -> Result<u64, WeightPatchError> {
            if text.len() != width
                || !text
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            {
                return Err(WeightPatchError::PatchOp(format!(
                    "`{text}` is not {width} lowercase hex digits, the width `{}` works in",
                    self.numerics
                )));
            }
            u64::from_str_radix(text, 16)
                .map_err(|e| WeightPatchError::PatchOp(format!("`{text}`: {e}")))
        };
        let bits: Vec<u64> = self
            .operand
            .iter()
            .map(|t| parse(t))
            .collect::<Result<_, _>>()?;
        let alpha = parse(&self.alpha)?;
        Ok(if profile.works_in_f64() {
            (OperandBits::F64(bits), ScalarBits::F64(alpha))
        } else {
            (
                OperandBits::F32(bits.into_iter().map(|b| b as u32).collect()),
                ScalarBits::F32(alpha as u32),
            )
        })
    }

    /// Run the op over the bound tensor of `container` and return the exact BF16 result of every element
    /// that changes. An op that changes nothing is refused: a patch edits at least one element.
    pub fn resolve(&self, container: &Path) -> Result<ResultPatch, WeightPatchError> {
        self.check_intact()?;
        self.check_binding(container)?;
        let (operand, alpha) = self.operands()?;
        let ([rows, cols], base) = read_bf16_tensor(container, &self.tensor)?;
        let op = StructuredOp {
            numerics: Some(self.numerics.clone()),
            shape: [rows, cols],
            kind: OpKind::RemoveDirection {
                u: operand,
                alpha,
                columns: self.columns.clone(),
            },
            fixups: Vec::new(),
        };
        let role = if self.tensor.contains(EMBEDDING_MARKER) {
            TensorRole::TokenEmbedding
        } else {
            TensorRole::Ordinary
        };
        let result = run(
            &op,
            &BaseTensor {
                shape: [rows, cols],
                bits: &base,
                role,
            },
        )
        .map_err(|e| WeightPatchError::Structured(e.to_string()))?;
        let edits: Vec<ResultEdit> = base
            .iter()
            .zip(&result)
            .enumerate()
            .filter(|(_, (old, new))| old != new)
            .map(|(i, (_, &new_bits))| ResultEdit {
                row: i / cols,
                col: i % cols,
                new_bits,
            })
            .collect();
        if edits.is_empty() {
            return Err(WeightPatchError::Empty);
        }
        Ok(ResultPatch {
            tensor: self.tensor.clone(),
            edits,
        })
    }
}

fn short(digest: &str) -> &str {
    digest.get(..12).unwrap_or(digest)
}
