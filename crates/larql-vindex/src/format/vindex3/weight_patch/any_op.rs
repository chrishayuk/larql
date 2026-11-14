//! A stored-weight `.vlp` op of either form, and the rule for when two of them may be held together.
//!
//! The delta op ([`super::WeightDeltaOp`]) names elements; the structured ops ([`super::StructuredWeightOp`],
//! `remove_direction`, and [`super::LowRankColumnsOp`], `add_low_rank_columns`) name columns and an operation
//! over them. Two ops conflict by their **declared write footprint**, never by
//! the elements an op happens to change after rounding: an op that rounds back to a stored value still
//! declared its write, and a later op on that element must not silently compose with it. Footprint is the
//! only thing the rule looks at, so it can be read off the op without opening a container.
//!
//! - delta against delta: the same element;
//! - structured against structured, of either kind: the same column (a structured op writes down whole
//!   columns, and reads only its own);
//! - delta against structured, on one tensor: refused as a mixed form. Resolving the delta's rounding
//!   against the structured result is a composition rule this version does not define.

use std::path::Path;

use super::apply::{DeclaredPatch, ResultPatch, WeightPatch};
use super::error::WeightPatchError;
use super::low_rank_op::LowRankColumnsOp;
use super::op::WeightDeltaOp;
use super::op_target::{check_ops, resolve_operation_target, StructuredOpRecord};
use super::structured_op::StructuredWeightOp;

/// A stored-weight op, owned.
#[derive(Debug, Clone, PartialEq)]
pub enum WeightOp {
    Delta(WeightDeltaOp),
    Structured(StructuredWeightOp),
    LowRankColumns(LowRankColumnsOp),
}

/// A stored-weight op, borrowed.
#[derive(Debug, Clone, Copy)]
pub enum WeightOpRef<'a> {
    Delta(&'a WeightDeltaOp),
    Structured(&'a StructuredWeightOp),
    LowRankColumns(&'a LowRankColumnsOp),
}

impl WeightOp {
    pub fn as_ref(&self) -> WeightOpRef<'_> {
        match self {
            WeightOp::Delta(op) => WeightOpRef::Delta(op),
            WeightOp::Structured(op) => WeightOpRef::Structured(op),
            WeightOp::LowRankColumns(op) => WeightOpRef::LowRankColumns(op),
        }
    }
}

impl WeightOpRef<'_> {
    pub fn tensor(&self) -> &str {
        match self {
            WeightOpRef::Delta(op) => &op.tensor,
            WeightOpRef::Structured(op) => &op.tensor,
            WeightOpRef::LowRankColumns(op) => &op.tensor,
        }
    }

    pub fn digest(&self) -> &str {
        match self {
            WeightOpRef::Delta(op) => &op.digest,
            WeightOpRef::Structured(op) => &op.digest,
            WeightOpRef::LowRankColumns(op) => &op.digest,
        }
    }

    pub fn check_intact(&self) -> Result<(), WeightPatchError> {
        match self {
            WeightOpRef::Delta(op) => op.check_intact(),
            WeightOpRef::Structured(op) => op.check_intact(),
            WeightOpRef::LowRankColumns(op) => op.check_intact(),
        }
    }

    pub fn check_binding(&self, container: &Path) -> Result<(), WeightPatchError> {
        match self {
            WeightOpRef::Delta(op) => op.check_binding(container),
            WeightOpRef::Structured(op) => op.check_binding(container),
            WeightOpRef::LowRankColumns(op) => op.check_binding(container),
        }
    }

    /// The columns a structured op writes down, over every row; `None` for a delta op, which names elements.
    pub fn structured_columns(&self) -> Option<&[usize]> {
        match self {
            WeightOpRef::Delta(_) => None,
            WeightOpRef::Structured(op) => Some(&op.columns),
            WeightOpRef::LowRankColumns(op) => Some(&op.columns),
        }
    }

    /// Refuse unless `self` and `held` can be held together. `by` names where `held` came from.
    pub fn check_disjoint(&self, held: WeightOpRef<'_>, by: &str) -> Result<(), WeightPatchError> {
        if self.tensor() != held.tensor() {
            return Ok(());
        }
        let tensor = self.tensor().to_string();
        match (self, held) {
            (WeightOpRef::Delta(new), WeightOpRef::Delta(old)) => {
                let taken: std::collections::HashSet<(usize, usize)> =
                    old.edits.iter().map(|e| (e.row, e.col)).collect();
                match new.edits.iter().find(|e| taken.contains(&(e.row, e.col))) {
                    Some(e) => Err(WeightPatchError::Conflict {
                        tensor,
                        row: e.row,
                        col: e.col,
                        by: by.to_string(),
                    }),
                    None => Ok(()),
                }
            }
            _ => match (self.structured_columns(), held.structured_columns()) {
                (Some(new), Some(old)) => match new.iter().find(|c| old.contains(c)) {
                    Some(&col) => Err(WeightPatchError::ColumnConflict {
                        tensor,
                        col,
                        by: by.to_string(),
                    }),
                    None => Ok(()),
                },
                _ => Err(WeightPatchError::MixedForms {
                    tensor,
                    by: by.to_string(),
                }),
            },
        }
    }
}

/// The declared patches that compile `ops` against `container`, one per tensor in the order each tensor
/// was first edited, and the digests of the ops they came from.
///
/// Every op is checked for integrity and for its binding to `container` before anything is resolved. A
/// tensor takes delta edits or structured ops, never both; several structured ops on one tensor have
/// disjoint columns (checked here again, not trusted from staging) and each reads only its own columns, so
/// their results are independent and are joined. A tensor that any `add_low_rank_columns` op edits is
/// compiled whole and recorded by its operations (`weight-patches/v4`); a tensor edited only by
/// `remove_direction` ops keeps its element form. An op that resolves to nothing is refused by name.
pub fn declare_patches(
    container: &Path,
    ops: &[WeightOpRef<'_>],
) -> Result<(Vec<DeclaredPatch>, Vec<String>), WeightPatchError> {
    for (i, op) in ops.iter().enumerate() {
        op.check_intact()?;
        op.check_binding(container)?;
        for earlier in &ops[..i] {
            op.check_disjoint(*earlier, "this compile")?;
        }
    }
    check_ops(ops)?;
    let mut tensors: Vec<&str> = Vec::new();
    for op in ops {
        if !tensors.contains(&op.tensor()) {
            tensors.push(op.tensor());
        }
    }
    let mut patches = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let on: Vec<WeightOpRef<'_>> = ops
            .iter()
            .copied()
            .filter(|o| o.tensor() == tensor)
            .collect();
        patches.push(declare_tensor(container, tensor, &on)?);
    }
    let origins = ops.iter().map(|o| o.digest().to_string()).collect();
    Ok((patches, origins))
}

/// One tensor's patch from its ops, all of one form (checked by the caller).
fn declare_tensor(
    container: &Path,
    tensor: &str,
    ops: &[WeightOpRef<'_>],
) -> Result<DeclaredPatch, WeightPatchError> {
    if ops
        .iter()
        .any(|o| matches!(o, WeightOpRef::LowRankColumns(_)))
    {
        let records: Vec<StructuredOpRecord> = ops
            .iter()
            .filter_map(|o| StructuredOpRecord::of(*o))
            .collect();
        return resolve_operation_target(container, tensor, &records)
            .map(DeclaredPatch::Operations);
    }
    if let WeightOpRef::Delta(_) = ops[0] {
        let edits = ops
            .iter()
            .flat_map(|o| match o {
                WeightOpRef::Delta(d) => d.weight_edits(),
                _ => unreachable!("mixed forms refused above"),
            })
            .collect();
        return Ok(DeclaredPatch::Deltas(WeightPatch {
            tensor: tensor.to_string(),
            edits,
        }));
    }
    let mut edits = Vec::new();
    for op in ops {
        match op {
            WeightOpRef::Structured(st) => edits.extend(st.resolve(container)?.edits),
            _ => unreachable!("mixed forms refused above; low-rank handled above"),
        }
    }
    Ok(DeclaredPatch::Results(ResultPatch {
        tensor: tensor.to_string(),
        edits,
    }))
}
