//! `WeightSlice::rows` — the integer arms of the truncation.
//!
//! The bf16/f32 arms are covered by `compact_consumption`; the integer
//! ones are not, and they are where the slicing is least obvious. Q4
//! packs two codes to the byte, so asking for `want` bytes rather than
//! `want / 2` would demand twice the matrix and refuse every legitimate
//! operand — a bug that fails closed and so reads as "Q4 is unsupported"
//! rather than as an arithmetic error.
//!
//! The Q8 sums index is the other trap. It is per-`SUM_BLOCK`, a
//! different geometry from the per-`block` scales, and it may legitimately
//! be empty where no arm consumes it. Cutting it to the scales' geometry
//! would pair a row with another row's sums and still return finite
//! numbers.

use super::super::backend::WeightSlice;
use super::super::cpu::projector::WeightRows;
use super::super::quantise::SUM_BLOCK;

const OUT_DIM: usize = 3;
const IN_DIM: usize = 64;
const BLOCK: usize = 32;

/// Q4 asks for HALF the element count, because two codes share a byte.
/// A packed stream exactly half the matrix must be accepted whole.
#[test]
fn a_q4_slice_is_cut_by_bytes_not_by_elements() {
    let packed = vec![0x77u8; OUT_DIM * IN_DIM / 2];
    let scales = vec![1.0f32; OUT_DIM * IN_DIM.div_ceil(BLOCK)];
    let slice = WeightSlice::Q4 {
        packed: &packed,
        scales: &scales,
        block: BLOCK,
    };

    let rows = slice
        .rows(OUT_DIM, IN_DIM)
        .expect("a packed stream of exactly want/2 bytes is the whole matrix");
    match rows {
        WeightRows::Q4 {
            packed: p,
            scales: s,
            block,
        } => {
            assert_eq!(p.len(), OUT_DIM * IN_DIM / 2, "Q4 is cut in BYTES");
            assert_eq!(s.len(), scales.len());
            assert_eq!(block, BLOCK);
        }
        other => panic!("a Q4 slice must yield Q4 rows, got {other:?}"),
    }
}

/// A short packed stream is refused, and the message reports the count
/// in ELEMENTS (bytes x 2) so the number is comparable with the
/// requested geometry rather than half of it.
#[test]
fn a_short_q4_slice_is_refused_in_element_terms() {
    let packed = vec![0u8; OUT_DIM * IN_DIM / 2 - 1];
    let scales = vec![1.0f32; OUT_DIM * IN_DIM.div_ceil(BLOCK)];
    let err = WeightSlice::Q4 {
        packed: &packed,
        scales: &scales,
        block: BLOCK,
    }
    .rows(OUT_DIM, IN_DIM)
    .expect_err("one byte short is still short");
    let msg = err.to_string();
    assert!(
        msg.contains(&(packed.len() * 2).to_string()),
        "the refusal must quote elements, not bytes: {msg}"
    );
}

/// Q4 scales are per-`block` along the input axis; too few is a refusal,
/// not a silently reused scale.
#[test]
fn a_q4_slice_with_too_few_scales_is_refused() {
    let packed = vec![0u8; OUT_DIM * IN_DIM / 2];
    let scales = vec![1.0f32; OUT_DIM * IN_DIM.div_ceil(BLOCK) - 1];
    assert!(
        WeightSlice::Q4 {
            packed: &packed,
            scales: &scales,
            block: BLOCK,
        }
        .rows(OUT_DIM, IN_DIM)
        .is_err(),
        "a scale short of the block geometry must refuse"
    );
}

/// Empty sums are legitimate — no arm consumes them — and must stay
/// empty rather than being cut to the scales' geometry.
#[test]
fn a_q8_slice_may_carry_no_sums_at_all() {
    let codes = vec![1i8; OUT_DIM * IN_DIM];
    let scales = vec![1.0f32; OUT_DIM * IN_DIM.div_ceil(BLOCK)];
    let rows = WeightSlice::Q8 {
        codes: &codes,
        scales: &scales,
        sums: &[],
        block: BLOCK,
    }
    .rows(OUT_DIM, IN_DIM)
    .expect("an empty sums index is not an error");
    match rows {
        WeightRows::Q8 { sums, .. } => {
            assert!(sums.is_empty(), "empty must survive as empty");
        }
        other => panic!("expected Q8 rows, got {other:?}"),
    }
}

/// A populated sums index is cut to its OWN geometry — per `SUM_BLOCK`,
/// not per `block`. Cutting it to the scales' length would hand a row
/// another row's sums and still return finite numbers.
#[test]
fn a_q8_sums_index_is_cut_to_its_own_geometry() {
    let codes = vec![1i8; OUT_DIM * IN_DIM];
    let scales = vec![1.0f32; OUT_DIM * IN_DIM.div_ceil(BLOCK)];
    let per_sum = IN_DIM.div_ceil(SUM_BLOCK);
    // Deliberately longer than needed: the cut is what is under test.
    let sums = vec![7i16; OUT_DIM * per_sum + 11];
    let rows = WeightSlice::Q8 {
        codes: &codes,
        scales: &scales,
        sums: &sums,
        block: BLOCK,
    }
    .rows(OUT_DIM, IN_DIM)
    .expect("a longer-than-needed index is truncated, not refused");
    match rows {
        WeightRows::Q8 { sums: cut, .. } => {
            assert_eq!(
                cut.len(),
                OUT_DIM * per_sum,
                "sums must be cut per SUM_BLOCK ({SUM_BLOCK}), not per block ({BLOCK})"
            );
            assert_ne!(
                cut.len(),
                scales.len(),
                "the two geometries must not coincide, or this proves nothing"
            );
        }
        other => panic!("expected Q8 rows, got {other:?}"),
    }
}

/// A short code stream is refused rather than yielding fewer rows than
/// the matrix has — the executor would otherwise partition a wrong total
/// across its workers.
#[test]
fn a_short_q8_slice_is_refused() {
    let codes = vec![1i8; OUT_DIM * IN_DIM - 1];
    let scales = vec![1.0f32; OUT_DIM * IN_DIM.div_ceil(BLOCK)];
    let err = WeightSlice::Q8 {
        codes: &codes,
        scales: &scales,
        sums: &[],
        block: BLOCK,
    }
    .rows(OUT_DIM, IN_DIM)
    .expect_err("one code short is short");
    assert!(
        err.to_string().contains(&codes.len().to_string()),
        "the refusal must quote what was actually resident"
    );
}

/// A sums index too SHORT for its own geometry is refused: a partial
/// index is the case that would pair rows with the wrong sums.
#[test]
fn a_q8_slice_with_a_partial_sums_index_is_refused() {
    let codes = vec![1i8; OUT_DIM * IN_DIM];
    let scales = vec![1.0f32; OUT_DIM * IN_DIM.div_ceil(BLOCK)];
    let sums = vec![0i16; OUT_DIM * IN_DIM.div_ceil(SUM_BLOCK) - 1];
    assert!(
        WeightSlice::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &sums,
            block: BLOCK,
        }
        .rows(OUT_DIM, IN_DIM)
        .is_err(),
        "a partially-sliced index must refuse, not pair rows wrongly"
    );
}

/// The integer arms name themselves. `compact_consumption` enumerates
/// the float and device representations; these two complete it, and the
/// name is what a reader debugs a refusal from.
#[test]
fn the_integer_representations_name_themselves() {
    let codes = [1i8; 4];
    let scales = [1.0f32; 1];
    let packed = [0u8; 2];
    assert_eq!(
        WeightSlice::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &[],
            block: BLOCK,
        }
        .representation(),
        "q8"
    );
    assert_eq!(
        WeightSlice::Q4 {
            packed: &packed,
            scales: &scales,
            block: BLOCK,
        }
        .representation(),
        "q4"
    );
}
