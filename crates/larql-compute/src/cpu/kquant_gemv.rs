//! The CPU's K-quant gemv — the stored block, multiplied where it lies.
//!
//! **Not [`crate::pipeline::quant_format::QuantFormat::Q8_0`].** That
//! name already exists in this crate and means a different thing: int8
//! codes with an EXTERNAL per-block f32 scale array, block 32. The
//! formats here are ggml's, whose scale is INSIDE the block — Q8_0 is
//! one f16 then 32 codes in 34 bytes. Two layouts sharing a name is how
//! a reader ends up passing one to the other's kernel, so this module
//! says `ggml` in its types and keeps its distance.
//!
//! Until this existed, a K-quant operand had exactly one execution
//! route: decode the whole matrix to f32 and run an f32 gemv over the
//! result. That is correct and it is what PARETO-1's rung A was measured
//! with, but it prices a Q8_0 model at **four bytes per weight**. On
//! Qwen3.8-27B the profiler put 97.3% of a decode token in `Projection`,
//! moving 97.4 GB of widened f32 per token where the artifact itself
//! holds the same information in 24.4 GB.
//!
//! The arithmetic here is not a second opinion about the format. It is
//! the association [`larql_models::quant::ggml::dequantize_q8_0`]
//! documents — `(q as i8 as f32) * f16(d)` — applied one block at a time
//! instead of into a whole decoded matrix, and multiplied into the
//! accumulator in element order. That makes it match
//! dequantise-then-multiply **exactly** rather than approximately, and
//! the tests hold it to that oracle.
//!
//! # Why exactness is available here at all
//!
//! `d` is an f16 and `q` an int8, so `d * q` needs at most 11 + 8 bits
//! of significand and is therefore EXACT in f32. Folding the scale into
//! each weight before the multiply — rather than hoisting it out of the
//! block's dot product, which would be cheaper — is what lets this
//! kernel be bit-for-bit with the decoder instead of merely close. A
//! hoisted scale is a different program, and on a path whose whole
//! purpose is to leave kernel quality out of a behavioural curve, "a
//! different program" is the thing to avoid.
//!
//! # Threading
//!
//! The kernel threads itself, as the Q4_K and Q6_K kernels beside it do:
//! rows are independent, so a row-parallel schedule changes which thread
//! runs a row and nothing about what that row computes — every row's
//! accumulation is the serial loop in `row_dot`, in the same order.
//! A caller that threads on top (LARQL's plan executor) declares this
//! kernel library-owned and calls it once.

use larql_models::quant::half::f16_to_f32;

/// Elements described by one Q8_0 block.
pub const Q8_0_BLOCK_ELEMS: usize = 32;

/// Bytes one Q8_0 block occupies: an f16 scale then 32 signed codes.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Rows per parallel work unit — the granularity the Q4_K and Q6_K
/// kernels use, for the same reason: a bandwidth-bound row loop wants
/// few, large units rather than one task per row.
const CHUNK_ROWS: usize = 32;

/// `out[n] = W[n, k] · x[k]`, with `W` read straight from the stored
/// Q8_0 pack.
///
/// `None` — never a wrong answer — when the geometry does not describe
/// this pack: `k` off the block grid, or the stream the wrong length for
/// `[n, k]`. Silently accepting a short stream would read a neighbouring
/// row's codes as this row's and return a plausible vector.
pub fn q8_0_gemv(blocks: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
    if k == 0 || !k.is_multiple_of(Q8_0_BLOCK_ELEMS) || x.len() != k {
        return None;
    }
    let per_row = k / Q8_0_BLOCK_ELEMS;
    if blocks.len() != n * per_row * Q8_0_BLOCK_BYTES {
        return None;
    }

    let mut out = vec![0.0f32; n];
    {
        use rayon::prelude::*;
        out.par_chunks_mut(CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk, slots)| {
                for (i, slot) in slots.iter_mut().enumerate() {
                    *slot = row_dot(blocks, x, chunk * CHUNK_ROWS + i, per_row);
                }
            });
    }
    Some(out)
}

/// One row's dot in the decoder's association — `(q * d) * x`, element
/// by element, blocks in order. The whole arithmetic of the kernel is
/// here; [`q8_0_gemv`] only decides which thread runs it.
#[inline]
fn row_dot(blocks: &[u8], x: &[f32], row: usize, per_row: usize) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..per_row {
        let base = (row * per_row + b) * Q8_0_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[base], blocks[base + 1]]));
        let codes = &blocks[base + 2..base + Q8_0_BLOCK_BYTES];
        let xs = &x[b * Q8_0_BLOCK_ELEMS..][..Q8_0_BLOCK_ELEMS];
        for (j, &c) in codes.iter().enumerate() {
            // `d * q` first, then the activation — the decoder's
            // association, element by element. See the module note on
            // why this is exact and why hoisting `d` would not be.
            acc += ((c as i8 as f32) * d) * xs[j];
        }
    }
    acc
}

#[cfg(test)]
#[path = "kquant_gemv_tests.rs"]
mod tests;
