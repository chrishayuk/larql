//! Turning stored f32 weights into a lossy compact residency.
//!
//! Separate from [`super::weights`] because it is a different kind of
//! decision. That module BINDS what a checkpoint holds — bf16 stays bf16,
//! f32 stays f32, and nothing it does changes a value. This one changes
//! the model: the weights it produces are not the weights that were
//! stored, and every claim about them has to be made on logits, KL, a
//! trajectory and recurrent-state drift rather than on bytes.
//!
//! Keeping the two apart means a reader can tell, from the module a
//! format is loaded through, whether it is exact.

use super::weights::LoadedWeight;

/// Elements per f32 scale, along the input axis.
///
/// 64 because every Qwen3.8 `in_dim` (5120, 6144, 17408) is a multiple of
/// it, so no real matrix pays a ragged final block — and because at 8.5
/// bits/weight the scales are 6% of the format rather than the 12.5% a
/// 32-element block would cost.
pub const Q8_BLOCK: usize = 64;

/// The largest magnitude an int8 code may represent.
///
/// 127 and not 128: symmetric, so the negative extreme is unused rather
/// than giving one direction a level the other lacks.
const Q8_MAX: f32 = 127.0;

/// Elements sharing one precomputed weight-code sum.
///
/// **A materialised execution index, not model semantics.** An
/// asymmetric activation reconstructs as `x = c*s + m`, so a dot becomes
/// `s*SUM(q*c) + m*SUM(q)` — and `SUM(q)` depends only on the weight
/// block. Recomputing it every token costs a second `SDOT` and a second
/// integer reduction per block, on a path that is already compute-bound
/// at ~3.4x its own memory wall.
///
/// 16 because that is the finest activation block the kernels admit;
/// coarser activation blocks aggregate consecutive sums, exactly, in
/// i32.
pub const SUM_BLOCK: usize = 16;

/// The widest a `SUM_BLOCK` sum of int8 codes can be: `16 * 127 = 2032`,
/// so `i16` holds it EXACTLY and the index costs one bit per weight.
const _: () = assert!(SUM_BLOCK as i32 * 127 < i16::MAX as i32);

/// Sum every [`SUM_BLOCK`] consecutive codes of each row.
///
/// Blocks never straddle a row: a row's last block is short rather than
/// borrowing its neighbour's codes, matching how the scales are cut.
fn code_sums(codes: &[i8], in_dim: usize) -> Vec<i16> {
    let per_row = in_dim.div_ceil(SUM_BLOCK);
    let rows = codes.len() / in_dim.max(1);
    let mut sums = vec![0i16; rows * per_row];
    for r in 0..rows {
        for b in 0..per_row {
            let lo = r * in_dim + b * SUM_BLOCK;
            let hi = (lo + SUM_BLOCK).min((r + 1) * in_dim);
            sums[r * per_row + b] = codes[lo..hi].iter().map(|c| *c as i16).sum();
        }
    }
    sums
}

/// Quantise `[out, in_dim]` f32 weights to symmetric per-block int8.
///
/// `scale = max|w| / 127` over each block, `code = round(w / scale)`.
/// Deliberately the simplest rule that can be stated in one line: it is
/// the BASELINE a better quantiser has to beat, and it is measured on
/// logits rather than on reconstruction error, because reconstruction
/// error is not what a decode reads.
///
/// Blocks never straddle a row: the last block of a row is short rather
/// than borrowing the next row's weights, which would give a row a scale
/// derived partly from its neighbour.
pub(super) fn quantise_q8(values: &[f32], in_dim: usize, with_code_sums: bool) -> LoadedWeight {
    let per_row = in_dim.div_ceil(Q8_BLOCK);
    let rows = values.len() / in_dim.max(1);
    let mut codes = vec![0i8; values.len()];
    let mut scales = vec![0.0f32; rows * per_row];
    for r in 0..rows {
        for b in 0..per_row {
            let lo = r * in_dim + b * Q8_BLOCK;
            let hi = (lo + Q8_BLOCK).min((r + 1) * in_dim);
            let peak = values[lo..hi].iter().fold(0.0f32, |m, w| m.max(w.abs()));
            // An all-zero block would divide by zero; 1.0 keeps its codes
            // at zero and the block reconstructs exactly.
            let scale = if peak > 0.0 { peak / Q8_MAX } else { 1.0 };
            scales[r * per_row + b] = scale;
            for i in lo..hi {
                codes[i] = (values[i] / scale).round().clamp(-Q8_MAX, Q8_MAX) as i8;
            }
        }
    }
    // The execution index is built only where an arm will consume it.
    // Carrying it unconditionally would add ~1 bit/weight of residency
    // and traffic to the symmetric path, which has no use for it.
    let sums = if with_code_sums {
        code_sums(&codes, in_dim)
    } else {
        Vec::new()
    };
    LoadedWeight::Q8 {
        codes,
        scales,
        sums,
    }
}

/// The shipped Q8 quantiser, reachable from a test.
///
/// Tests call THIS rather than restating the rule: a test that quantised
/// its own way would agree with itself whatever the loader did.
#[cfg(test)]
pub fn quantise_q8_for_test(values: &[f32], in_dim: usize) -> LoadedWeight {
    quantise_q8(values, in_dim, false)
}

/// The shipped quantiser WITH the execution index, reachable from a test.
#[cfg(test)]
pub fn quantise_q8_indexed_for_test(values: &[f32], in_dim: usize) -> LoadedWeight {
    quantise_q8(values, in_dim, true)
}

/// Elements per f32 scale for [`quantise_q4`], along the input axis.
///
/// The same 64 as [`Q8_BLOCK`], and deliberately so: CPU-4Y priced Q4 at
/// 4.5 bits/weight with this blocking, and a quality arm that quietly
/// used a different block would be measuring a format the mechanics rung
/// never timed.
pub const Q4_BLOCK: usize = 64;

/// The largest magnitude an int4 code may represent.
///
/// 7 and not 8: symmetric, so the negative extreme `-8` is unused rather
/// than giving one direction a level the other lacks. This is also the
/// whole numerical story of the format — the step is `peak / 7` against
/// Q8's `peak / 127`, **18.1x coarser at the same block size**, which is
/// the quantity every quality gate on this representation is really
/// about.
const Q4_MAX: f32 = 7.0;

/// The bias that makes a signed code an unsigned nibble.
///
/// Codes are `-8..=7` and stored `+8` so a nibble is `0..=15`; a kernel
/// unbiases with one vector subtract rather than sign-extending from four
/// bits.
const Q4_BIAS: i32 = 8;

/// Quantise `[out, in_dim]` f32 weights to symmetric per-block int4,
/// two codes per byte.
///
/// **Byte `j` of a block holds elements `j` and `j + block/2`**, not `2j`
/// and `2j+1`. Adjacent packing would make one 16-byte load yield 32
/// INTERLEAVED elements and every kernel would spend its time undoing
/// that; half-block packing yields two contiguous runs that pair directly
/// with two runs of the activation.
///
/// Same rule as [`quantise_q8`] otherwise — `scale = max|w| / 7`,
/// `code = round(w / scale)` — because the point of this rung is to
/// measure what FOUR BITS costs, and changing the quantiser and the bit
/// width together would confound the two.
pub(super) fn quantise_q4(values: &[f32], in_dim: usize) -> LoadedWeight {
    let per_row = in_dim.div_ceil(Q4_BLOCK);
    let rows = values.len() / in_dim.max(1);
    let mut packed = vec![0u8; values.len() / 2];
    let mut scales = vec![0.0f32; rows * per_row];
    for r in 0..rows {
        for b in 0..per_row {
            let lo = b * Q4_BLOCK;
            let hi = (lo + Q4_BLOCK).min(in_dim);
            let src = &values[r * in_dim + lo..r * in_dim + hi];
            let peak = src.iter().fold(0.0f32, |m, w| m.max(w.abs()));
            // An all-zero block would divide by zero; 1.0 keeps its codes
            // at the bias and the block reconstructs exactly.
            let scale = if peak > 0.0 { peak / Q4_MAX } else { 1.0 };
            scales[r * per_row + b] = scale;
            let half = (hi - lo) / 2;
            let base = (r * in_dim + lo) / 2;
            let code = |v: f32| {
                ((v / scale).round().clamp(-(Q4_MAX + 1.0), Q4_MAX) as i32 + Q4_BIAS) as u8
            };
            for j in 0..half {
                packed[base + j] = code(src[j]) | (code(src[j + half]) << 4);
            }
        }
    }
    LoadedWeight::Q4 { packed, scales }
}

/// The shipped Q4 quantiser, reachable from a test.
///
/// Tests call THIS rather than restating the rule: a test that quantised
/// its own way would agree with itself whatever the loader did.
#[cfg(test)]
pub fn quantise_q4_for_test(values: &[f32], in_dim: usize) -> LoadedWeight {
    quantise_q4(values, in_dim)
}
