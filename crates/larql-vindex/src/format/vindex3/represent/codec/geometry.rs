//! What a shape means, decided once for every codec.

use std::ops::Range;

use super::error::CodecError;

/// `[rows, k]` as every codec in this module reads a shape.
///
/// The last axis is `k` — the axis blocks, groups and scales run along —
/// and every leading axis multiplies into `rows`. So a bias `[n]` is one
/// row of `n`, a matrix `[out, in]` is `out` rows of `in`, and an expert
/// bank `[experts, out, in]` is `experts * out` rows of `in`. One rule,
/// so a codec cannot read a bank one way and a matrix another, and a
/// scalar `[]` is a single element rather than a special case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowGeometry {
    pub rows: usize,
    pub k: usize,
}

impl RowGeometry {
    pub fn of(shape: &[usize], label: &str, tensor: &str) -> Result<Self, CodecError> {
        let Some((&k, leading)) = shape.split_last() else {
            return Ok(Self { rows: 1, k: 1 });
        };
        let rows = leading
            .iter()
            .try_fold(1usize, |acc, d| acc.checked_mul(*d))
            .ok_or_else(|| CodecError::Geometry {
                tensor: tensor.into(),
                label: label.into(),
                shape: shape.to_vec(),
                why: "the row count overflows".into(),
            })?;
        Ok(Self { rows, k })
    }

    /// `rows * k`, refusing overflow rather than wrapping into a plausible
    /// small number.
    pub fn elements(self, label: &str, tensor: &str) -> Result<usize, CodecError> {
        self.rows
            .checked_mul(self.k)
            .ok_or_else(|| CodecError::Geometry {
                tensor: tensor.into(),
                label: label.into(),
                shape: vec![self.rows, self.k],
                why: "the element count overflows".into(),
            })
    }

    /// Refuse a row range that reaches past what the tensor holds.
    pub fn check_rows(
        self,
        rows: &Range<usize>,
        label: &str,
        tensor: &str,
    ) -> Result<(), CodecError> {
        if rows.start > rows.end || rows.end > self.rows {
            return Err(CodecError::RowRange {
                tensor: tensor.into(),
                label: label.into(),
                start: rows.start,
                end: rows.end,
                rows: self.rows,
            });
        }
        Ok(())
    }

    /// Refuse a `k` that is not a whole number of `group` elements — the
    /// refusal every grouped codec makes, worded once. Returns the group
    /// count.
    pub fn check_group(self, group: usize, label: &str, tensor: &str) -> Result<usize, CodecError> {
        if group == 0 || !self.k.is_multiple_of(group) {
            return Err(CodecError::Geometry {
                tensor: tensor.into(),
                label: label.into(),
                shape: vec![self.rows, self.k],
                why: format!(
                    "k={} is not a whole number of {group}-element groups",
                    self.k
                ),
            });
        }
        Ok(self.k / group)
    }

    /// Refuse a destination that does not hold exactly `rows` rows.
    pub fn check_destination(
        self,
        rows: &Range<usize>,
        dst_len: usize,
        tensor: &str,
    ) -> Result<(), CodecError> {
        let need = rows.len() * self.k;
        if dst_len != need {
            return Err(CodecError::Destination {
                tensor: tensor.into(),
                need,
                have: dst_len,
            });
        }
        Ok(())
    }
}
