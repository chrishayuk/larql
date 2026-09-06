//! Fine-grained FP8 dequantisation.
//!
//! The load-bearing test is not here: it is
//! `scripts/glm_fp8_dequant_gate.py`, which compares this codec against
//! `transformers`' own `Fp8Dequantize` on a **real GLM-5.3-Flash tensor**
//! and requires bit-exact agreement. These are the unit properties that
//! make the failure of that gate diagnosable.

use super::*;

/// E4M3 `0x38` is exactly 1.0, so a tile of `0x38` reads back as its own
/// scale. Chosen because it makes the tiling visible in the values
/// themselves rather than needing a separate assertion about indices.
const ONE: u8 = 0x38;

fn grid(rows: usize, cols: usize, scale_rows: usize, scale_cols: usize) -> Fp8Grid {
    Fp8Grid {
        rows,
        cols,
        scale_rows,
        scale_cols,
    }
}

#[test]
fn e4m3_0x38_is_exactly_one() {
    // The premise every other test in this file leans on.
    assert_eq!(crate::quant::fp8::e4m3_to_f32(ONE), 1.0);
}

#[test]
fn the_tile_is_derived_from_the_grid_not_declared() {
    assert_eq!(grid(256, 512, 2, 4).tile(), Ok((128, 128)));
    // The same weight under a DIFFERENT grid yields a different tile —
    // which is the whole reason the config's `weight_block_size` is not
    // the authority.
    assert_eq!(grid(256, 512, 256, 16).tile(), Ok((1, 32)));
    assert_eq!(grid(256, 512, 1, 1).tile(), Ok((256, 512)));
}

#[test]
fn a_grid_that_does_not_tile_is_refused_not_rounded() {
    assert_eq!(
        grid(100, 128, 3, 1).tile(),
        Err(Fp8GridError::NotDivisible {
            rows: 100,
            cols: 128,
            scale_rows: 3,
            scale_cols: 1,
        })
    );
    assert_eq!(
        grid(128, 128, 0, 1).tile(),
        Err(Fp8GridError::EmptyGrid {
            scale_rows: 0,
            scale_cols: 1,
        })
    );
}

/// Each scale reaches exactly its own tile — the property a
/// one-dimensional blocking cannot express, and the one most likely to be
/// got wrong by reusing this crate's other blocked formats.
///
/// The grid is deliberately NON-SQUARE with UNEQUAL tile axes (4x6
/// weight, 2x3 grid, so 2x2 tiles), because a square fixture cannot tell
/// a row-major scale index from a transposed one.
#[test]
fn every_scale_covers_exactly_its_own_tile() {
    let g = grid(4, 6, 2, 3);
    assert_eq!(g.tile(), Ok((2, 2)));
    let codes = vec![ONE; g.elements()];
    let scales: Vec<f32> = (1..=6).map(|i| i as f32).collect();
    let out = dequantize(&codes, &scales, g).expect("dequantise");

    #[rustfmt::skip]
    let expected = vec![
        1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
        1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
        4.0, 4.0, 5.0, 5.0, 6.0, 6.0,
        4.0, 4.0, 5.0, 5.0, 6.0, 6.0,
    ];
    assert_eq!(out, expected);
}

/// A transposed scale grid must NOT read back the same values.
///
/// Without this, `every_scale_covers_exactly_its_own_tile` would pass for
/// an implementation that indexed the grid column-major — the fixture is
/// non-square precisely so this control can fire.
#[test]
fn a_transposed_scale_grid_reads_differently() {
    let g = grid(4, 6, 2, 3);
    let codes = vec![ONE; g.elements()];
    let row_major: Vec<f32> = (1..=6).map(|i| i as f32).collect();
    let col_major = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
    assert_ne!(
        dequantize(&codes, &row_major, g).unwrap(),
        dequantize(&codes, &col_major, g).unwrap(),
        "row-major and column-major scale grids must not agree, or the \
         index is not being read"
    );
}

/// The scale MULTIPLIES. A dequantiser that divided by `weight_scale_inv`
/// — which the name invites — would read 0.25 here, not 4.0.
#[test]
fn the_scale_is_multiplied_not_divided() {
    let g = grid(1, 1, 1, 1);
    assert_eq!(dequantize(&[ONE], &[4.0], g).unwrap(), vec![4.0]);
}

#[test]
fn a_length_disagreement_names_both_sides() {
    let g = grid(2, 2, 1, 1);
    assert_eq!(
        dequantize(&[ONE; 3], &[1.0], g),
        Err(Fp8GridError::CodeCount {
            expected: 4,
            found: 3
        })
    );
    assert_eq!(
        dequantize(&[ONE; 4], &[1.0, 2.0], g),
        Err(Fp8GridError::ScaleCount {
            expected: 1,
            found: 2
        })
    );
}

/// The sibling name follows the reference's own rule, including the
/// non-`.weight` branch.
#[test]
fn scale_sibling_names_follow_the_reference_rule() {
    assert_eq!(
        scale_sibling_name("model.layers.0.mlp.gate_proj.weight"),
        "model.layers.0.mlp.gate_proj.weight_scale_inv"
    );
    // Not a `.weight`: the suffix is appended, not substituted.
    assert_eq!(scale_sibling_name("some_tensor"), "some_tensor_scale_inv");
    // `.weight` only at the END — a middle occurrence must not be caught.
    assert_eq!(
        scale_sibling_name("a.weight.b"),
        "a.weight.b_scale_inv",
        "`.weight` is a suffix rule, not a substring rule"
    );
}

#[test]
fn scale_siblings_are_recognisable_without_their_weight() {
    assert!(is_scale_sibling("x.weight_scale_inv"));
    assert!(!is_scale_sibling("x.weight"));
    // Round-trip: every name this module generates is one it recognises.
    for n in ["a.weight", "b", "c.weight.d"] {
        assert!(is_scale_sibling(&scale_sibling_name(n)), "{n}");
    }
}

/// Real GLM-5.3-Flash geometry, at the two shapes the checkpoint
/// actually ships, so the divisibility contract is asserted against the
/// estate rather than against invented numbers.
#[test]
fn real_glm_geometries_tile_at_128_squared() {
    // `mlp.gate_proj.weight` on a dense layer: [12288, 4096] / [96, 32].
    assert_eq!(grid(12288, 4096, 96, 32).tile(), Ok((128, 128)));
    // `mlp.experts.{i}.down_proj.weight`: [4096, 2048] / [32, 16].
    assert_eq!(grid(4096, 2048, 32, 16).tile(), Ok((128, 128)));
    // `self_attn.q_a_proj.weight`: [1536, 4096] / [12, 32].
    assert_eq!(grid(1536, 4096, 12, 32).tile(), Ok((128, 128)));
}

/// The declared tile is CHECKED against the derived one, and the derived
/// one wins.
///
/// Both directions are asserted: agreement must read `Ok`, and a
/// disagreement must report both tiles rather than silently preferring
/// either. Without the second arm, "cross-checked" would be a claim with
/// no check behind it.
#[test]
fn the_declared_tile_is_checked_against_the_derived_one() {
    let g = grid(12288, 4096, 96, 32);
    assert_eq!(g.check_declared_tile((128, 128)), Ok(Ok(())));

    // A checkpoint declaring [128, 128] while shipping a [1, 32] grid for
    // this tensor — legal, and exactly the mixed-grid case the reference
    // accommodates. Reported, not resolved.
    let mixed = grid(4096, 2048, 4096, 64);
    assert_eq!(mixed.tile(), Ok((1, 32)));
    assert_eq!(
        mixed.check_declared_tile((128, 128)),
        Ok(Err(TileDisagreement {
            declared: (128, 128),
            derived: (1, 32),
        }))
    );

    // A grid that does not tile at all has nothing to compare.
    assert!(grid(100, 128, 3, 1)
        .check_declared_tile((128, 128))
        .is_err());
}

/// Every refusal renders both sides of the disagreement. A message that
/// named only one would leave the reader guessing which half is wrong.
#[test]
fn every_error_names_both_sides() {
    let rendered = |e: Fp8GridError| e.to_string();

    let not_div = rendered(Fp8GridError::NotDivisible {
        rows: 100,
        cols: 128,
        scale_rows: 3,
        scale_cols: 1,
    });
    for needle in ["100", "128", "3", "1", "not evenly tiled"] {
        assert!(not_div.contains(needle), "{not_div}");
    }
    assert!(
        not_div.contains("will not borrow"),
        "the refusal should say what it declines to do: {not_div}"
    );

    let codes = rendered(Fp8GridError::CodeCount {
        expected: 4,
        found: 3,
    });
    assert!(codes.contains('4') && codes.contains('3'), "{codes}");

    let scales = rendered(Fp8GridError::ScaleCount {
        expected: 1,
        found: 2,
    });
    assert!(scales.contains("scale grid"), "{scales}");

    let empty = rendered(Fp8GridError::EmptyGrid {
        scale_rows: 0,
        scale_cols: 1,
    });
    assert!(empty.contains("zero axis"), "{empty}");

    // It is an `Error`, so a caller can propagate it with `?`.
    let boxed: Box<dyn std::error::Error> = Box::new(Fp8GridError::EmptyGrid {
        scale_rows: 0,
        scale_cols: 0,
    });
    assert!(!boxed.to_string().is_empty());
}

/// A tile disagreement renders BOTH tiles and says which one is applied
/// — the derived one. A message that reported only the mismatch would
/// leave a reader unable to tell which number the arithmetic used.
#[test]
fn a_tile_disagreement_says_which_tile_wins() {
    let d = TileDisagreement {
        declared: (128, 128),
        derived: (1, 32),
    };
    let s = d.to_string();
    assert!(s.contains("(128, 128)"), "{s}");
    assert!(s.contains("(1, 32)"), "{s}");
    assert!(s.contains("derived tile is what is applied"), "{s}");
}

/// `elements` and `scales` are the two counts every length check is made
/// against, so they are pinned rather than left implicit.
#[test]
fn the_grid_reports_its_two_counts() {
    let g = grid(12288, 4096, 96, 32);
    assert_eq!(g.elements(), 12288 * 4096);
    assert_eq!(g.scales(), 96 * 32);
}

/// The allocating and in-place entry points agree, and the in-place one
/// refuses a buffer of the wrong length rather than writing part of it.
#[test]
fn dequantize_into_matches_and_checks_its_buffer() {
    let g = grid(4, 6, 2, 3);
    let codes = vec![ONE; g.elements()];
    let scales: Vec<f32> = (1..=6).map(|i| i as f32).collect();

    let mut out = vec![0.0f32; g.elements()];
    dequantize_into(&codes, &scales, g, &mut out).expect("in place");
    assert_eq!(out, dequantize(&codes, &scales, g).expect("allocating"));

    let mut short = vec![0.0f32; g.elements() - 1];
    assert!(dequantize_into(&codes, &scales, g, &mut short).is_err());
}
