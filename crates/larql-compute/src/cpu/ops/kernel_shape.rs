//! Typed refusal for the CPU quantised kernels' operand shapes.
//!
//! Every Q*K matvec kernel used to zero-fill its output and return when a
//! caller handed it inconsistent slices: an output shorter than `rows`, an
//! activation that was not `cols` long, a column count that was not a
//! whole number of super-blocks, or a weight slab too short for the
//! geometry. Those guards exist because the hand-asm kernels take
//! `q8k_x.qs.as_ptr()` as a bare pointer with no bound of its own, so a
//! caller-side mismatch is a real out-of-bounds read, not a debug-only
//! assertion. But a zero vector is a plausible logit vector — the
//! wrong-but-plausible failure class dec-readiness §1 names — so the
//! kernels now refuse by name and leave `out` untouched instead.
//!
//! Zero dimensions are not a refusal: a matvec over no rows or no columns
//! is the empty product, and the kernels still write zeros for it.

use std::fmt;

/// Why a quantised kernel refused its operands. Every field is the value
/// the kernel saw, so the message names the mismatch rather than the rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelShapeError {
    /// The refusing kernel.
    pub kernel: &'static str,
    /// `out.len()` as handed in; must equal `rows`.
    pub out_len: usize,
    /// Output rows the caller asked for.
    pub rows: usize,
    /// Activation length (`q8k_x.qs.len()` or `x.len()`); must equal `cols`.
    pub x_len: usize,
    /// Weight columns the caller asked for; must be a whole number of
    /// super-blocks.
    pub cols: usize,
    /// Packed weight bytes handed in.
    pub weight_bytes: usize,
    /// Packed weight bytes `rows × cols` needs at this block layout.
    pub needed_bytes: usize,
}

impl KernelShapeError {
    /// Validate a `rows × cols` matvec over packed super-blocks of
    /// `block_bytes` per `block_elems` columns. `Ok` means every slice is
    /// exactly what the kernel will index: `out.len() == rows`,
    /// `x_len == cols`, `cols` a multiple of `block_elems`, and at least
    /// `rows` packed rows of weight bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn check(
        kernel: &'static str,
        out_len: usize,
        rows: usize,
        x_len: usize,
        cols: usize,
        weight_bytes: usize,
        block_elems: usize,
        block_bytes: usize,
    ) -> Result<(), Self> {
        let needed_bytes = (cols / block_elems) * block_bytes * rows;
        let ok = out_len == rows
            && x_len == cols
            && cols.is_multiple_of(block_elems)
            && weight_bytes >= needed_bytes;
        if ok {
            Ok(())
        } else {
            Err(Self {
                kernel,
                out_len,
                rows,
                x_len,
                cols,
                weight_bytes,
                needed_bytes,
            })
        }
    }
}

impl fmt::Display for KernelShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: refused operands: out.len() {} for {} rows, activation length {} for {} cols \
             (cols must be a whole number of super-blocks), {} weight bytes where {} are needed",
            self.kernel,
            self.out_len,
            self.rows,
            self.x_len,
            self.cols,
            self.weight_bytes,
            self.needed_bytes
        )
    }
}

impl std::error::Error for KernelShapeError {}

#[cfg(test)]
mod tests {
    use super::*;

    const ELEMS: usize = 256;
    const BYTES: usize = 144;

    #[test]
    fn a_consistent_shape_passes() {
        assert_eq!(
            KernelShapeError::check("k", 3, 3, 512, 512, 3 * 2 * BYTES, ELEMS, BYTES),
            Ok(())
        );
        // Zero dimensions are the empty product, not a refusal.
        assert_eq!(
            KernelShapeError::check("k", 0, 0, 0, 0, 0, ELEMS, BYTES),
            Ok(())
        );
        assert_eq!(
            KernelShapeError::check("k", 4, 4, 0, 0, 0, ELEMS, BYTES),
            Ok(())
        );
    }

    #[test]
    fn every_mismatch_is_refused_by_name() {
        let cases: [(usize, usize, usize, usize, usize); 4] = [
            (2, 3, 512, 512, 3 * 2 * BYTES),     // out shorter than rows
            (3, 3, 511, 512, 3 * 2 * BYTES),     // activation not cols long
            (3, 3, 500, 500, 3 * 2 * BYTES),     // cols not a super-block multiple
            (3, 3, 512, 512, 3 * 2 * BYTES - 1), // weight slab short
        ];
        for (out_len, rows, x_len, cols, wb) in cases {
            let err =
                KernelShapeError::check("q4k_test", out_len, rows, x_len, cols, wb, ELEMS, BYTES)
                    .expect_err("must refuse");
            assert_eq!(err.kernel, "q4k_test");
            assert_eq!(
                (err.out_len, err.rows, err.x_len, err.cols, err.weight_bytes),
                (out_len, rows, x_len, cols, wb)
            );
            let text = err.to_string();
            assert!(text.starts_with("q4k_test: refused operands"), "{text}");
            assert!(text.contains(&format!("{wb} weight bytes")), "{text}");
        }
    }

    #[test]
    fn needed_bytes_follows_the_block_layout() {
        let err = KernelShapeError::check("k", 5, 5, 768, 768, 0, ELEMS, 210).expect_err("short");
        assert_eq!(err.needed_bytes, 3 * 210 * 5);
    }
}
