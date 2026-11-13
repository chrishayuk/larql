//! The `.vlp` form of `add_low_rank_columns` (VLP numerics v2): a low-rank update confined to columns `J`,
//! declared by its operands, not by the elements it changes.
//!
//! The op adds `Σ_k a[r,k]·b[k,p]` to `W[r, J[p]]`, sequentially in `k` from the widened base, and an element
//! whose terms all have a zero factor keeps its base bits (`conformance/vlp-numerics/v2/SPEC.md`). It carries:
//!
//! - the **operands** `a` (`rows × rank`, row-major) and `b` (`rank × |J|`, row-major and compact), as stored
//!   bits in the profile's working width. There is no scalar: a producer with `α·u·vᵀ` folds `α` into its
//!   operands and may name where they came from in `origin`;
//! - the **columns** `J` (the input axis) and the **numerical profile** that fixes every rounding;
//! - the same **source binding** as every stored-weight op: the exact container and weights it was written for.
//!
//! [`LowRankColumnsOp::resolve`] runs the frozen interpreter over the bound tensor and returns the exact BF16
//! results of the elements that change. Its footprint, for the conflict rule, is the columns `J` over every
//! row. Like every op's digest, this one detects accidental damage and casual edits and is not a signature;
//! whether a container implements the op is for an independent verifier.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::apply::ResultEdit;
use super::apply::ResultPatch;
use super::error::WeightPatchError;
use super::op::{bind_source, SourceBinding};
use super::op_target::{resolve_operation_target, OperationPatch, StructuredOpRecord};
use super::structured::{OpKind, OperandBits};
use super::structured_op::parse_operand_hex;

/// Version tag mixed into the digest, so a change to what it covers is a new digest.
const DIGEST_DOMAIN: &str = "structured_op/add_low_rank_columns/v2";

/// An `add_low_rank_columns` op as it appears in a `.vlp` patch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LowRankColumnsOp {
    pub source: SourceBinding,
    pub tensor: String,
    /// The numerical profile id, such as `f64-seq-rne1/v1`.
    pub numerics: String,
    /// `K`, the number of terms.
    pub rank: usize,
    /// `a`, `rows × rank` row-major, one fixed-width lowercase hex string per element.
    pub a: Vec<String>,
    /// `b`, `rank × columns.len()` row-major (compact), one fixed-width lowercase hex string per element.
    pub b: Vec<String>,
    /// `J`: strictly increasing column indices into the input axis.
    pub columns: Vec<usize>,
    /// Where the operands came from, if their writer says (an `α`, `u`, `v` provenance, a file hash, a note).
    /// Covered by the digest and read by nothing else.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// SHA-256 over every other field; see the module note.
    pub digest: String,
}

impl LowRankColumnsOp {
    /// Build an op bound to `container` and seal it. `a` and `b` are the profile's working width: f64 bits for
    /// the f64 profiles, f32 bits for the f32 one.
    pub fn bind(
        container: &Path,
        tensor: &str,
        numerics: &str,
        rank: usize,
        a: &OperandBits,
        b: &OperandBits,
        columns: Vec<usize>,
    ) -> Result<Self, WeightPatchError> {
        let source = bind_source(container, tensor)?;
        let mut op = Self {
            source,
            tensor: tensor.to_string(),
            numerics: numerics.to_string(),
            rank,
            a: hex(a),
            b: hex(b),
            columns,
            origin: None,
            digest: String::new(),
        };
        op.digest = op.compute_digest();
        Ok(op)
    }

    /// This op with its operands' origin recorded, re-sealed.
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = Some(origin.into());
        self.digest = self.compute_digest();
        self
    }

    /// The digest of this op's fields, independent of the stored `digest`.
    pub fn compute_digest(&self) -> String {
        let columns: Vec<String> = self.columns.iter().map(|c| c.to_string()).collect();
        let text = format!(
            "{DIGEST_DOMAIN}\nmodel={}\nindex_sha256={}\nrepresentation={}\npayload_sha256={}\n\
             tensor={}\nnumerics={}\nrank={}\ncolumns={}\na={}\nb={}\norigin={}\n",
            self.source.model,
            self.source.index_sha256,
            self.source.representation,
            self.source.payload_sha256,
            self.tensor,
            self.numerics,
            self.rank,
            columns.join(","),
            self.a.join(","),
            self.b.join(","),
            self.origin
                .as_deref()
                .map_or("-".to_string(), |o| format!("+{o}")),
        );
        super::apply::hex(&Sha256::digest(text.as_bytes()))
    }

    /// Whether the op is intact, names a profile this build implements, carries operands of that profile's
    /// width, and is within the structured-op bounds that need no container (rank, operand size, work). Does not
    /// look at any container.
    pub fn check_intact(&self) -> Result<(), WeightPatchError> {
        if self.compute_digest() != self.digest {
            return Err(WeightPatchError::PatchOp(
                "its digest does not match its contents: the patch was corrupted or edited"
                    .to_string(),
            ));
        }
        super::op_target::check_cost(&StructuredOpRecord::LowRankColumns(self.clone()).cost())?;
        self.operands().map(|_| ())
    }

    /// The op as the interpreter takes it, with its operands decoded.
    pub(super) fn op_kind(&self) -> Result<OpKind, WeightPatchError> {
        let (a, b) = self.operands()?;
        Ok(OpKind::AddLowRankColumns {
            rank: self.rank,
            a,
            b,
            columns: self.columns.clone(),
        })
    }

    /// Refuse unless `container` is the exact artifact this op was written against.
    pub fn check_binding(&self, container: &Path) -> Result<(), WeightPatchError> {
        super::op::check_same_source(&self.source, container, &self.tensor)
    }

    fn operands(&self) -> Result<(OperandBits, OperandBits), WeightPatchError> {
        Ok((
            parse_operand_hex(&self.numerics, &self.a)?,
            parse_operand_hex(&self.numerics, &self.b)?,
        ))
    }

    /// Run the op over the bound tensor of `container`: the whole compiled tensor and how it is recorded as an
    /// operation target. An op that changes nothing is refused (a patch edits at least one element) unless it is
    /// zero-valued: then it is a declared sham and resolves to its base's bytes.
    pub fn resolve_tensor(&self, container: &Path) -> Result<OperationPatch, WeightPatchError> {
        self.check_intact()?;
        self.check_binding(container)?;
        resolve_operation_target(
            container,
            &self.tensor,
            &[StructuredOpRecord::LowRankColumns(self.clone())],
        )
    }

    /// [`Self::resolve_tensor`] as the exact BF16 result of every element that changes: the element form, for a
    /// tensor small enough to be listed.
    pub fn resolve(&self, container: &Path) -> Result<ResultPatch, WeightPatchError> {
        let resolved = self.resolve_tensor(container)?;
        let ([_, cols], base) = super::apply::read_bf16_tensor(container, &self.tensor)?;
        let edits: Vec<ResultEdit> = base
            .iter()
            .zip(&resolved.result)
            .enumerate()
            .filter(|(_, (old, new))| old != new)
            .map(|(i, (_, &new_bits))| ResultEdit {
                row: i / cols,
                col: i % cols,
                new_bits,
            })
            .collect();
        // The element form lists changes, so it cannot carry a sham (a zero-valued op); only the operation form can.
        if edits.is_empty() {
            return Err(WeightPatchError::Empty);
        }
        Ok(ResultPatch {
            tensor: self.tensor.clone(),
            edits,
        })
    }
}

fn hex(bits: &OperandBits) -> Vec<String> {
    match bits {
        OperandBits::F64(v) => v.iter().map(|b| format!("{b:016x}")).collect(),
        OperandBits::F32(v) => v.iter().map(|b| format!("{b:08x}")).collect(),
    }
}
