//! **Operation targets** (`weight-patches/v4`): a tensor edited by structured ops is recorded by the ops and their
//! exact operands, not by its elements. The contract is `docs/verification/weight-patches-v4.md`.
//!
//! A tensor becomes an operation target when an `add_low_rank_columns` op edits it; every structured op on it is
//! then one of its operations. The ops have disjoint column sets and each reads only its own columns, so the
//! compiled tensor is the base with each op's columns replaced by that op's result on the base (§4). The bounds of
//! §5 are checked before any arithmetic, so a compact record can never be an instruction to do unbounded work.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::any_op::WeightOpRef;
use super::apply::read_bf16_tensor;
use super::error::WeightPatchError;
use super::low_rank_op::LowRankColumnsOp;
use super::op::SourceBinding;
use super::structured::{run, BaseTensor, OpKind, OperandBits, ScalarBits, StructuredOp};
use super::structured_op::{tensor_role, StructuredWeightOp};
use super::{
    MAX_STRUCTURED_OPERAND_ELEMENTS, MAX_STRUCTURED_RANK, MAX_STRUCTURED_TENSOR_ELEMENTS,
    MAX_STRUCTURED_WORK,
};

const F64_SIGN: u64 = 1 << 63;
const F32_SIGN: u32 = 1 << 31;

/// Multiply-adds per element of a removal's footprint: the reduction `uᵀw` and the update `w − (αu)c`.
const REMOVAL_WORK_PER_ELEMENT: u64 = 2;

/// A structured op as an operation target records it: the `.vlp` op itself, under the same `op` tag a patch uses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum StructuredOpRecord {
    #[serde(rename = "structured_weight")]
    RemoveDirection(StructuredWeightOp),
    #[serde(rename = "low_rank_columns_weight")]
    LowRankColumns(LowRankColumnsOp),
}

/// One operation of an operation target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerOperation {
    #[serde(flatten)]
    pub op: StructuredOpRecord,
    /// SHA-256 of each operand's bits (little-endian, working width), by operand name.
    pub operands_sha256: BTreeMap<String, String>,
    /// Base elements whose bits this op changes.
    pub changed_elements: usize,
}

/// A tensor compiled from its structured ops: the whole result and how it is recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationPatch {
    pub tensor: String,
    pub shape: [usize; 2],
    /// The whole compiled tensor, row-major BF16 bits.
    pub result: Vec<u16>,
    pub base_tensor_sha256: String,
    pub operations: Vec<LedgerOperation>,
    pub changed_elements: usize,
}

impl OperationPatch {
    /// Whether every operation is zero-valued: the target is a declared sham, compiled to its base's bytes.
    pub fn is_sham(&self) -> Result<bool, WeightPatchError> {
        all_zero_valued(&self.operations)
    }
}

/// Whether every one of `operations` is zero-valued (false for none).
pub fn all_zero_valued(operations: &[LedgerOperation]) -> Result<bool, WeightPatchError> {
    if operations.is_empty() {
        return Ok(false);
    }
    for o in operations {
        if !o.op.is_zero_valued()? {
            return Ok(false);
        }
    }
    Ok(true)
}

impl StructuredOpRecord {
    /// The record of a structured op; `None` for a delta op.
    pub fn of(op: WeightOpRef<'_>) -> Option<Self> {
        match op {
            WeightOpRef::Delta(_) => None,
            WeightOpRef::Structured(o) => Some(Self::RemoveDirection(o.clone())),
            WeightOpRef::LowRankColumns(o) => Some(Self::LowRankColumns(o.clone())),
        }
    }

    pub fn as_ref(&self) -> WeightOpRef<'_> {
        match self {
            Self::RemoveDirection(o) => WeightOpRef::Structured(o),
            Self::LowRankColumns(o) => WeightOpRef::LowRankColumns(o),
        }
    }

    pub fn source(&self) -> &SourceBinding {
        match self {
            Self::RemoveDirection(o) => &o.source,
            Self::LowRankColumns(o) => &o.source,
        }
    }

    fn numerics(&self) -> &str {
        match self {
            Self::RemoveDirection(o) => &o.numerics,
            Self::LowRankColumns(o) => &o.numerics,
        }
    }

    /// The interpreter's form of the op, with its operands decoded.
    fn kind(&self) -> Result<OpKind, WeightPatchError> {
        match self {
            Self::RemoveDirection(o) => o.op_kind(),
            Self::LowRankColumns(o) => o.op_kind(),
        }
    }

    /// Each operand's SHA-256, by name (§3).
    pub fn operands_sha256(&self) -> Result<BTreeMap<String, String>, WeightPatchError> {
        let named: Vec<(&str, OperandBits)> = match self.kind()? {
            OpKind::RemoveDirection { u, alpha, .. } => vec![
                ("u", u),
                (
                    "alpha",
                    match alpha {
                        ScalarBits::F64(b) => OperandBits::F64(vec![b]),
                        ScalarBits::F32(b) => OperandBits::F32(vec![b]),
                    },
                ),
            ],
            OpKind::AddLowRankColumns { a, b, .. } => vec![("a", a), ("b", b)],
            OpKind::AddLowRank { .. } | OpKind::ZeroColumns { .. } => {
                unreachable!("no .vlp op declares this kind")
            }
        };
        Ok(named
            .into_iter()
            .map(|(name, bits)| (name.to_string(), operand_sha256(&bits)))
            .collect())
    }

    /// The op's cost from its declared operands alone (no container): rank, operand size and work.
    pub fn cost(&self) -> StructuredCost {
        let (rank, operand_elements) = match self {
            Self::RemoveDirection(o) => (0, o.operand.len() + 1),
            Self::LowRankColumns(o) => (o.rank, o.a.len() + o.b.len()),
        };
        StructuredCost {
            rank,
            operand_elements,
            work: self.work(),
            tensor_elements: 0,
        }
    }

    /// Whether the op is **zero-valued** (VLP numerics v2 §5): an `add_low_rank_columns` op in which no element
    /// has a term with both factors nonzero, i.e. for every `k` either column `k` of `a` or row `k` of `b` is all
    /// zero (`−0` included). Under the v2 identity rule it leaves its tensor byte-identical, so it is the declared
    /// form of a sham edit. Decided from the operands alone. A `remove_direction` op is never zero-valued. An op
    /// whose operand lengths do not fit its rank and columns is an error, not a verdict.
    pub fn is_zero_valued(&self) -> Result<bool, WeightPatchError> {
        Ok(match self.kind()? {
            OpKind::AddLowRankColumns {
                rank,
                a,
                b,
                columns,
            } => {
                let zero = |bits: &OperandBits, i: usize| match bits {
                    OperandBits::F64(v) => v[i] & !F64_SIGN == 0,
                    OperandBits::F32(v) => v[i] & !F32_SIGN == 0,
                };
                // The record may come from a ledger nobody has run: its operand lengths are checked here, not
                // trusted, so a short `a` or `b` is an error and never an index out of range.
                let n = columns.len();
                if rank == 0 || a.len() % rank != 0 || rank.checked_mul(n) != Some(b.len()) {
                    return Err(WeightPatchError::Structured(format!(
                        "a rank-{rank} op over {n} column(s) carries {} `a` and {} `b` operand(s); `a` must be \
                         rows x rank and `b` rank x columns",
                        a.len(),
                        b.len()
                    )));
                }
                let rows = a.len() / rank;
                (0..rank).all(|k| {
                    (0..rows).all(|r| zero(&a, r * rank + k)) || (0..n).all(|p| zero(&b, k * n + p))
                })
            }
            _ => false,
        })
    }

    /// Multiply-adds the op costs, from its declared operands (§5).
    pub fn work(&self) -> u64 {
        match self {
            // `a` is rows × K, so `len(a) × |J|` is rows × |J| × K.
            Self::LowRankColumns(o) => (o.a.len() as u64).saturating_mul(o.columns.len() as u64),
            Self::RemoveDirection(o) => (o.operand.len() as u64)
                .saturating_mul(o.columns.len() as u64)
                .saturating_mul(REMOVAL_WORK_PER_ELEMENT),
        }
    }
}

/// SHA-256 over an operand's bits as little-endian bytes of its working width.
pub fn operand_sha256(bits: &OperandBits) -> String {
    let mut h = Sha256::new();
    match bits {
        OperandBits::F64(v) => v.iter().for_each(|b| h.update(b.to_le_bytes())),
        OperandBits::F32(v) => v.iter().for_each(|b| h.update(b.to_le_bytes())),
    }
    super::apply::hex(&h.finalize())
}

/// SHA-256 over a tensor's stored payload: row-major BF16, little-endian.
pub fn tensor_sha256(bits: &[u16]) -> String {
    let mut h = Sha256::new();
    bits.iter().for_each(|b| h.update(b.to_le_bytes()));
    super::apply::hex(&h.finalize())
}

fn within(what: &'static str, value: u64, limit: u64) -> Result<(), WeightPatchError> {
    if value > limit {
        return Err(WeightPatchError::OverBound { what, value, limit });
    }
    Ok(())
}

/// The cost of a set of ops together: the largest rank and operand size of any one, the sum of their work.
pub fn cost_of(ops: &[WeightOpRef<'_>]) -> StructuredCost {
    ops.iter()
        .filter_map(|o| StructuredOpRecord::of(*o))
        .map(|r| r.cost())
        .fold(StructuredCost::default(), |acc, c| StructuredCost {
            rank: acc.rank.max(c.rank),
            operand_elements: acc.operand_elements.max(c.operand_elements),
            work: acc.work.saturating_add(c.work),
            tensor_elements: 0,
        })
}

/// Refuse a set of structured ops over the bounds that need no container (§5).
pub fn check_ops(ops: &[WeightOpRef<'_>]) -> Result<(), WeightPatchError> {
    check_cost(&cost_of(ops))
}

/// What a set of structured ops costs, in the terms of the §5 bounds. `rank` is the largest rank among them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructuredCost {
    pub rank: usize,
    /// The most operand elements any one op carries.
    pub operand_elements: usize,
    /// Multiply-adds of all of them together.
    pub work: u64,
    /// `rows × cols` of the tensor they edit; zero when not yet known.
    pub tensor_elements: u64,
}

/// Refuse a cost over any `MAX_STRUCTURED_*` bound (§5), naming the first bound it exceeds.
pub fn check_cost(cost: &StructuredCost) -> Result<(), WeightPatchError> {
    within("rank", cost.rank as u64, MAX_STRUCTURED_RANK as u64)?;
    within(
        "operand elements",
        cost.operand_elements as u64,
        MAX_STRUCTURED_OPERAND_ELEMENTS as u64,
    )?;
    within(
        "structured work (multiply-adds)",
        cost.work,
        MAX_STRUCTURED_WORK,
    )?;
    within(
        "tensor elements",
        cost.tensor_elements,
        MAX_STRUCTURED_TENSOR_ELEMENTS as u64,
    )
}

/// Compile `tensor` of `container` from its structured `ops` (§4). The caller has checked every op's integrity,
/// binding and footprint; this checks the bounds, runs each op on the base, and joins their columns. An op that
/// changes no element is refused, as at staging.
pub fn resolve_operation_target(
    container: &Path,
    tensor: &str,
    ops: &[StructuredOpRecord],
) -> Result<OperationPatch, WeightPatchError> {
    let refs: Vec<WeightOpRef<'_>> = ops.iter().map(StructuredOpRecord::as_ref).collect();
    let [rows, cols] = super::apply::tensor_shape(container, tensor)?;
    check_cost(&StructuredCost {
        tensor_elements: (rows as u64).saturating_mul(cols as u64),
        ..cost_of(&refs)
    })?;
    let (_, base) = read_bf16_tensor(container, tensor)?;
    let role = tensor_role(container, tensor)?;
    let mut result = base.clone();
    let mut operations = Vec::with_capacity(ops.len());
    for op in ops {
        let out = run(
            &StructuredOp {
                numerics: Some(op.numerics().to_string()),
                shape: [rows, cols],
                kind: op.kind()?,
                fixups: Vec::new(),
            },
            &BaseTensor {
                shape: [rows, cols],
                bits: &base,
                role,
            },
        )
        .map_err(|e| WeightPatchError::Structured(e.to_string()))?;
        let mut changed = 0;
        for (i, (&old, &new)) in base.iter().zip(&out).enumerate() {
            if old != new {
                result[i] = new;
                changed += 1;
            }
        }
        // A zero-valued op is the declared identity (a sham) and changes nothing by construction; any other op
        // that changes nothing is refused, as at staging.
        match (op.is_zero_valued()?, changed) {
            (false, 0) => return Err(WeightPatchError::Empty),
            (true, n) if n > 0 => {
                return Err(WeightPatchError::Structured(format!(
                    "a zero-valued op changed {n} element(s) of `{tensor}`; the v2 identity rule forbids it"
                )))
            }
            _ => {}
        }
        operations.push(LedgerOperation {
            operands_sha256: op.operands_sha256()?,
            op: op.clone(),
            changed_elements: changed,
        });
    }
    Ok(OperationPatch {
        tensor: tensor.to_string(),
        shape: [rows, cols],
        changed_elements: operations.iter().map(|o| o.changed_elements).sum(),
        base_tensor_sha256: tensor_sha256(&base),
        result,
        operations,
    })
}
