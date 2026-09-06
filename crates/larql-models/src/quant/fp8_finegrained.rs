//! Fine-grained (block-wise) FP8 as frontier checkpoints ship it: E4M3
//! values with a **two-dimensional grid** of f32 scales, one per
//! `block_rows × block_cols` tile.
//!
//! This is the DeepSeek-V3 lineage convention — `quant_method: "fp8"`,
//! `weight_block_size: [128, 128]`, and a `*.weight_scale_inv` sibling
//! beside every quantised `*.weight`. GLM-5.3-Flash ships **95.8 % of its
//! 306 GiB in this format**: every routed expert, every shared expert,
//! the dense MLP estate and the MLA projections.
//!
//! # Not to be confused with [`super::fp4_block`]'s "FP8 block"
//!
//! That module's `FP8_BLOCK_BYTES` names LARQL's *own* vindex FP4 format
//! (exp 26): 256 contiguous E4M3 values plus one **E4M3** block scale, in
//! a packed 257-byte record, blocked one-dimensionally. This module
//! describes a *checkpoint* format that shares only the element codec:
//! the values live in an ordinary row-major matrix, the scales live in a
//! separate tensor, they are **f32**, and the tiling is two-dimensional.
//! Nothing here packs, and nothing here is 256-element.
//!
//! # Two properties worth stating, because both invite a wrong guess
//!
//! **The block shape is derived from the scale grid, not from the config.**
//! `quantization_config.weight_block_size` says `[128, 128]` on
//! GLM-5.3-Flash and it is *not* the authority: transformers'
//! `Fp8Dequantize._dequantize_one` computes `block_m = rows / scale_rows`
//! from the tensors themselves, precisely so one checkpoint can ship
//! several grids (its own comment cites MoE experts at `[1, 32]` beside
//! dense linears at `[128, 128]`). Reading the config value instead would
//! be right on this checkpoint and wrong by construction.
//!
//! **`weight_scale_inv` is MULTIPLIED, not divided by.** The name says
//! "inv" because it is the inverse of the quantisation scale; the
//! dequantiser applies it directly. Another declared name that promises a
//! computation the reference does not do.
//!
//! # Provenance
//!
//! Transcribed from `transformers.integrations.finegrained_fp8`
//! (5.16.1, `Fp8Dequantize._dequantize_one`), which is the loader
//! GLM-5.3-Flash's `config.json` selects via `quant_method: "fp8"`. The
//! checkpoint ships no code of its own, so upstream is the contract.

use super::fp8::e4m3_to_f32;

/// Why a `(weight, weight_scale_inv)` pair could not be read.
///
/// Every variant names both sides of the disagreement: a scale grid that
/// does not tile its weight is a checkpoint fact worth reporting, not a
/// condition to round away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fp8GridError {
    /// `rows × cols` disagrees with the number of E4M3 bytes.
    CodeCount { expected: usize, found: usize },
    /// `scale_rows × scale_cols` disagrees with the number of scales.
    ScaleCount { expected: usize, found: usize },
    /// The grid does not tile the matrix evenly.
    ///
    /// Refused rather than padded: a partial trailing tile would have to
    /// borrow a neighbour's scale, and there is no reading of the format
    /// under which that is correct.
    NotDivisible {
        rows: usize,
        cols: usize,
        scale_rows: usize,
        scale_cols: usize,
    },
    /// A zero-sized grid, which would divide by zero.
    EmptyGrid {
        scale_rows: usize,
        scale_cols: usize,
    },
}

impl std::fmt::Display for Fp8GridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CodeCount { expected, found } => write!(
                f,
                "fine-grained FP8: shape implies {expected} E4M3 codes, found {found}"
            ),
            Self::ScaleCount { expected, found } => write!(
                f,
                "fine-grained FP8: scale grid implies {expected} scales, found {found}"
            ),
            Self::NotDivisible {
                rows,
                cols,
                scale_rows,
                scale_cols,
            } => write!(
                f,
                "fine-grained FP8: weight ({rows}, {cols}) is not evenly tiled by its \
                 scale grid ({scale_rows}, {scale_cols}); a partial tile has no scale \
                 of its own and this build will not borrow a neighbour's"
            ),
            Self::EmptyGrid {
                scale_rows,
                scale_cols,
            } => write!(
                f,
                "fine-grained FP8: scale grid ({scale_rows}, {scale_cols}) has a zero axis"
            ),
        }
    }
}

impl std::error::Error for Fp8GridError {}

/// One quantised matrix's geometry: the weight's shape and the shape of
/// its scale grid, with the tile size **derived** from the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8Grid {
    pub rows: usize,
    pub cols: usize,
    pub scale_rows: usize,
    pub scale_cols: usize,
}

impl Fp8Grid {
    /// Check that the grid tiles the matrix, and report the tile size.
    ///
    /// This is the only place a block size is decided, and it is decided
    /// from the tensors — see the module docs on why
    /// `weight_block_size` is not consulted.
    pub fn tile(self) -> Result<(usize, usize), Fp8GridError> {
        let Self {
            rows,
            cols,
            scale_rows,
            scale_cols,
        } = self;
        if scale_rows == 0 || scale_cols == 0 {
            return Err(Fp8GridError::EmptyGrid {
                scale_rows,
                scale_cols,
            });
        }
        if rows % scale_rows != 0 || cols % scale_cols != 0 {
            return Err(Fp8GridError::NotDivisible {
                rows,
                cols,
                scale_rows,
                scale_cols,
            });
        }
        Ok((rows / scale_rows, cols / scale_cols))
    }

    /// Check the tile this tensor's grid implies against the one
    /// `quantization_config.weight_block_size` declares.
    ///
    /// This is what makes carrying the declared value worth more than
    /// using it. The derived tile stays the authority — a checkpoint may
    /// ship several grids under one declaration, which is legal and which
    /// transformers' own dequantiser accommodates — so a disagreement is
    /// reported to the caller to classify, not raised as an error here.
    ///
    /// Returns `Ok(())` when they agree, and the two tiles when they do
    /// not. A tensor whose grid cannot tile it at all returns that error
    /// instead: there is nothing to compare.
    pub fn check_declared_tile(
        self,
        declared: (usize, usize),
    ) -> Result<Result<(), TileDisagreement>, Fp8GridError> {
        let derived = self.tile()?;
        Ok(if derived == declared {
            Ok(())
        } else {
            Err(TileDisagreement { declared, derived })
        })
    }

    /// Elements in the weight matrix.
    pub fn elements(self) -> usize {
        self.rows * self.cols
    }

    /// Entries in the scale grid.
    pub fn scales(self) -> usize {
        self.scale_rows * self.scale_cols
    }
}

/// A tensor whose scale grid implies a different tile than the
/// checkpoint's `weight_block_size` declares.
///
/// Not an error: the derived tile is the authority and the declaration is
/// a summary that a mixed-grid checkpoint cannot state for every tensor.
/// It is worth surfacing because on a checkpoint where they are *meant*
/// to agree, a disagreement is the first sign of a mis-read shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileDisagreement {
    pub declared: (usize, usize),
    pub derived: (usize, usize),
}

impl std::fmt::Display for TileDisagreement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "declared weight_block_size {:?} but this tensor's scale grid implies {:?};              the derived tile is what is applied",
            self.declared, self.derived
        )
    }
}

/// Dequantise a fine-grained FP8 matrix into a caller-provided buffer.
///
/// `codes` is row-major `[rows, cols]` E4M3; `scales` is row-major
/// `[scale_rows, scale_cols]` f32; `out` receives row-major f32.
///
/// The arithmetic is a single f32 multiply per element, in the same order
/// and the same precision as the reference, so the result is **bit-exact**
/// against `Fp8Dequantize` at f32 rather than merely close.
pub fn dequantize_into(
    codes: &[u8],
    scales: &[f32],
    grid: Fp8Grid,
    out: &mut [f32],
) -> Result<(), Fp8GridError> {
    let (block_rows, block_cols) = grid.tile()?;
    if codes.len() != grid.elements() {
        return Err(Fp8GridError::CodeCount {
            expected: grid.elements(),
            found: codes.len(),
        });
    }
    if scales.len() != grid.scales() {
        return Err(Fp8GridError::ScaleCount {
            expected: grid.scales(),
            found: scales.len(),
        });
    }
    if out.len() != grid.elements() {
        return Err(Fp8GridError::CodeCount {
            expected: grid.elements(),
            found: out.len(),
        });
    }
    for r in 0..grid.rows {
        // Within a row the scale changes only every `block_cols`
        // elements, so the row's slice of the grid is hoisted once. The
        // row-dependent part is the grid ROW — which is what makes this
        // tiling two-dimensional and unlike the crate's other blocked
        // formats, whose scales never span rows.
        let scale_row = &scales[(r / block_rows) * grid.scale_cols..][..grid.scale_cols];
        let src = &codes[r * grid.cols..][..grid.cols];
        let dst = &mut out[r * grid.cols..][..grid.cols];
        for (c, (d, &code)) in dst.iter_mut().zip(src).enumerate() {
            *d = e4m3_to_f32(code) * scale_row[c / block_cols];
        }
    }
    Ok(())
}

/// [`dequantize_into`], allocating the output.
pub fn dequantize(codes: &[u8], scales: &[f32], grid: Fp8Grid) -> Result<Vec<f32>, Fp8GridError> {
    let mut out = vec![0.0f32; grid.elements()];
    dequantize_into(codes, scales, grid, &mut out)?;
    Ok(out)
}

/// The `*.weight_scale_inv` name a quantised `*.weight` pairs with.
///
/// Transcribed from `Fp8Dequantize._scale_pattern_for`: a key ending
/// `.weight` swaps that suffix, anything else gets `_scale_inv`
/// appended. Returned as a `String` because callers look it up in the
/// checkpoint's own tensor index — this function decides the *name*, and
/// never whether the sibling exists.
pub fn scale_sibling_name(weight: &str) -> String {
    match weight.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.weight_scale_inv"),
        None => format!("{weight}_scale_inv"),
    }
}

/// Whether a tensor name IS a scale sibling rather than a weight.
///
/// Used to keep scale tensors out of role classification: they are part
/// of the operand they accompany, not operands of their own, and a
/// classifier that saw them as unclassified tensors would report a
/// closure defect for every quantised matrix in the checkpoint (76,108
/// tensors on GLM-5.3-Flash, of which 37,000-odd are scales).
pub fn is_scale_sibling(name: &str) -> bool {
    name.ends_with("_scale_inv")
}

#[cfg(test)]
#[path = "tests/fp8_finegrained.rs"]
mod tests;
