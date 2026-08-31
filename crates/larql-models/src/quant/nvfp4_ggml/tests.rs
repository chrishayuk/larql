//! The lowering is checked three ways: the exact bytes, the values they
//! denote, and what it refuses.
//!
//! The semantic test is the one that matters. Byte pinning catches a
//! changed arrangement; only decoding both sides catches an arrangement
//! that is self-consistently wrong.

use super::*;
use crate::quant::nvfp4::{quantize, NVFP4_GROUP_ELEMS};

/// One 64-element row: four groups, deliberately distinguishable so a
/// mis-grouping shows up as a wrong byte rather than a plausible one.
fn source_row() -> (Vec<u8>, Vec<u8>, f32) {
    let packed: Vec<u8> = (0..32u8)
        .map(|j| (((2 * j + 1) % 16) << 4) | ((2 * j) % 16))
        .collect();
    // Four different scale bytes; the last has its sign bit set so the
    // strip to UE4M3 is actually exercised rather than assumed.
    let scales: Vec<u8> = vec![0x3c, 0x40, 0x44, 0xc8];
    (packed, scales, 0.375)
}

#[test]
fn a_block_lowers_to_exactly_thirty_six_bytes_in_the_declared_order() {
    let (packed, scales, ts) = source_row();
    let out = repack_nvfp4(&packed, &scales, ts, 1, 64).expect("64 is one whole block");

    assert_eq!(out.blocks.len(), NVFP4_BLOCK_BYTES, "one block");
    assert_eq!(
        out.tensor_scale, ts,
        "the tensor scale is carried, not folded"
    );
    assert_eq!(
        &out.blocks[..4],
        &[0x3c, 0x40, 0x44, 0x48],
        "UE4M3 scales lead the block, sign stripped"
    );

    let mut want = Vec::new();
    for g in 0..4 {
        let src = &packed[g * 8..g * 8 + 8];
        let mut elems = [0u8; 16];
        for (j, b) in src.iter().enumerate() {
            elems[2 * j] = b & 0x0f;
            elems[2 * j + 1] = b >> 4;
        }
        for j in 0..8 {
            want.push(elems[j] | (elems[j + 8] << 4));
        }
    }
    assert_eq!(
        &out.blocks[4..],
        &want[..],
        "codes are re-nibbled, not re-ordered"
    );
}

/// **The invariant.** Decode the VINDEX pack and decode the GGML block:
/// they must denote the same numbers, because lowering moved bytes and
/// nothing else.
#[test]
fn the_lowered_block_denotes_exactly_what_the_source_denoted() {
    let values: Vec<f32> = (0..64).map(|i| ((i % 13) as f32 - 6.0) * 0.125).collect();
    let m = quantize(&values, 1, 64).expect("quantise");
    let out = repack_nvfp4(&m.packed, &m.scales, m.tensor_scale, 1, 64).unwrap();

    let e2m1 = |c: u8| -> f32 {
        let mag = match c & 0x7 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        };
        if c & 0x8 != 0 {
            -mag
        } else {
            mag
        }
    };
    let ue4m3 = |b: u8| -> f32 {
        let e = ((b >> 3) & 0x0f) as i32;
        let mant = (b & 0x07) as f32 / 8.0;
        if e == 0 {
            mant * 2f32.powi(-6)
        } else {
            (1.0 + mant) * 2f32.powi(e - 7)
        }
    };

    let mut got = vec![0f32; 64];
    let d = &out.blocks[..4];
    let qs = &out.blocks[4..];
    for g in 0..4 {
        let s = ue4m3(d[g]) * out.tensor_scale;
        for j in 0..8 {
            let byte = qs[g * 8 + j];
            got[g * 16 + j] = e2m1(byte & 0x0f) * s;
            got[g * 16 + j + 8] = e2m1(byte >> 4) * s;
        }
    }

    let mut want = vec![0f32; 64];
    crate::quant::nvfp4::dequantize_into(&m, 1, 64, &mut want).expect("reference decode");

    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert!(
            (a - b).abs() <= a.abs() * 1e-6 + 1e-9,
            "element {i}: VINDEX says {a}, GGML block says {b} — lowering changed a value"
        );
    }
}

#[test]
fn a_row_that_is_not_whole_blocks_refuses_rather_than_padding() {
    for cols in [NVFP4_GROUP_ELEMS, 32, 48] {
        let groups = cols / NVFP4_GROUP_ELEMS;
        let err = repack_nvfp4(&vec![0u8; groups * 8], &vec![0x3cu8; groups], 1.0, 1, cols)
            .expect_err("a partial block must refuse");
        assert!(
            err.to_string().contains("multiple of 64"),
            "the refusal must name the constraint: {err}"
        );
    }
    for cols in [64, 128] {
        let groups = cols / NVFP4_GROUP_ELEMS;
        assert!(
            repack_nvfp4(&vec![0u8; groups * 8], &vec![0x3cu8; groups], 1.0, 1, cols).is_ok(),
            "{cols} is a whole number of blocks"
        );
    }
}

#[test]
fn a_region_of_the_wrong_size_refuses() {
    let (packed, scales, ts) = source_row();
    assert!(
        repack_nvfp4(&packed[..31], &scales, ts, 1, 64).is_err(),
        "short codes"
    );
    assert!(
        repack_nvfp4(&packed, &scales[..3], ts, 1, 64).is_err(),
        "short scales"
    );
}

#[test]
fn stored_size_follows_the_block_geometry() {
    assert_eq!(ggml_nvfp4_bytes(64).unwrap(), 36);
    assert_eq!(ggml_nvfp4_bytes(128).unwrap(), 72);
    assert!(ggml_nvfp4_bytes(48).is_err(), "not a whole block");
}
