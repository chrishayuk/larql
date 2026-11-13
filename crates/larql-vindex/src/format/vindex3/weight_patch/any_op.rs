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
/// tensor takes delta edits or structured results, never both; several structured ops on one tensor have
/// disjoint columns (checked here again, not trusted from staging) and each reads only its own columns, so
/// their results are independent and are joined. An op that resolves to nothing is refused by name.
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
    let mut patches: Vec<DeclaredPatch> = Vec::new();
    for op in ops {
        match op {
            WeightOpRef::Delta(d) => match patches.iter_mut().find(|p| p.tensor() == d.tensor) {
                Some(DeclaredPatch::Deltas(p)) => p.edits.extend(d.weight_edits()),
                Some(DeclaredPatch::Results(_)) => unreachable!("mixed forms refused above"),
                None => patches.push(DeclaredPatch::Deltas(WeightPatch {
                    tensor: d.tensor.clone(),
                    edits: d.weight_edits(),
                })),
            },
            WeightOpRef::Structured(_) | WeightOpRef::LowRankColumns(_) => {
                let resolved = match op {
                    WeightOpRef::Structured(st) => st.resolve(container)?,
                    WeightOpRef::LowRankColumns(lr) => lr.resolve(container)?,
                    WeightOpRef::Delta(_) => unreachable!("matched above"),
                };
                match patches.iter_mut().find(|p| p.tensor() == resolved.tensor) {
                    Some(DeclaredPatch::Results(p)) => p.edits.extend(resolved.edits),
                    Some(DeclaredPatch::Deltas(_)) => unreachable!("mixed forms refused above"),
                    None => patches.push(DeclaredPatch::Results(ResultPatch {
                        tensor: resolved.tensor,
                        edits: resolved.edits,
                    })),
                }
            }
        }
    }
    let origins = ops.iter().map(|o| o.digest().to_string()).collect();
    Ok((patches, origins))
}
