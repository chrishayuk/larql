//! Decoding a stored NVFP4 pack back to f32, through the format's own
//! layout and the reference decoder.
//!
//! This closes the loop the fidelity claim runs around: values are
//! quantised, framed into a pack, stored, and read back. If the framing
//! and the decode disagreed about where the scales begin, everything
//! downstream would still produce numbers — the decode would simply be
//! reading group scales as codes — and a quality run would report a
//! fidelity figure for a representation that never existed.
//!
//! The round trip is asserted to a tolerance rather than exactly,
//! because NVFP4 is lossy by construction and this test is about the
//! plumbing, not the format's accuracy. What must be exact is the
//! SHAPE: the same count of values, in the same order.

use crate::format::vindex3::opplan::exec::operands::decode_nvfp4_operand;
use crate::format::vindex3::represent::nvfp4_pack::{encode, PackLayout};

const ROWS: usize = 3;
/// A multiple of the 16-element group, as the format requires.
const K: usize = 32;

fn values() -> Vec<f32> {
    // A spread wide enough that the two scale levels both do work, and
    // deterministic so a failure is reproducible.
    (0..ROWS * K)
        .map(|i| {
            let x = (i % 17) as f32 - 8.0;
            x * 0.03125 * if i % 3 == 0 { 4.0 } else { 1.0 }
        })
        .collect()
}

#[test]
fn a_pack_round_trips_through_its_own_layout() {
    let original = values();
    let shape = [ROWS, K];
    let layout = PackLayout::derive(&shape, "round-trip").expect("a 16-multiple width is legal");
    let matrix = larql_models::quant::nvfp4::quantize(&original, ROWS, K).expect("quantise");
    let stored = encode(&matrix, &layout, "round-trip").expect("frame the pack");

    // The three regions plus the tensor scale, and nothing else.
    assert_eq!(
        stored.len(),
        layout.total_len,
        "the framed pack is exactly the layout's size"
    );

    let decoded = decode_nvfp4_operand(&stored, &shape, "round-trip").expect("decode");
    assert_eq!(
        decoded.len(),
        original.len(),
        "decode returns the shape it was given, not the shape it inferred"
    );

    // Lossy by construction — but every value must land near its
    // original, in its original position. A framing error would scatter
    // them, not merely round them.
    let worst = original
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let span = original.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        worst <= span * 0.25,
        "max deviation {worst} is too large for span {span} — this is a framing error, not rounding"
    );
}

/// The width is the format's business, not the caller's. A shape that is
/// not a whole number of groups cannot be framed, and the refusal
/// happens at layout derivation — before any bytes are written or read.
#[test]
fn a_width_that_is_not_a_whole_number_of_groups_cannot_be_laid_out() {
    let err =
        PackLayout::derive(&[ROWS, K + 1], "ragged").expect_err("a partial group must refuse");
    assert!(
        err.to_string().contains("ragged"),
        "the refusal names the tensor it refused: {err}"
    );
}

/// A payload shorter than its own layout is refused rather than decoded
/// from whatever bytes happen to be present.
#[test]
fn a_truncated_pack_refuses_rather_than_decoding_what_is_there() {
    let original = values();
    let shape = [ROWS, K];
    let layout = PackLayout::derive(&shape, "truncated").unwrap();
    let matrix = larql_models::quant::nvfp4::quantize(&original, ROWS, K).unwrap();
    let stored = encode(&matrix, &layout, "truncated").unwrap();

    let short = &stored[..stored.len() - 1];
    assert!(
        decode_nvfp4_operand(short, &shape, "truncated").is_err(),
        "one byte short is still short"
    );
}
