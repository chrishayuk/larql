//! The fine-grained FP8 kernel, and the slab arithmetic around it.
//!
//! Why this needs its own file: FP8 is the only `WeightRows` variant
//! whose scale spans ROWS. Every other blocked format tiles along the
//! input axis alone, so a row partition can cut codes and scales at the
//! same index; here one scale row serves `block_rows` output rows, and a
//! partition that does not land on a tile boundary has to remember where
//! it started. That offset is the thing most likely to be got wrong, and
//! it is invisible unless a test partitions off a boundary.

use super::super::kernels::FusedFp8Block;
use super::super::projector::{DenseProjector, WeightRows};
use larql_models::quant::fp8_finegrained::{dequantize, Fp8Grid};

/// Deterministic E4M3 codes. `0x7F`/`0xFF` are NaN in E4M3 and are
/// skipped — a NaN would make every comparison below vacuously unequal.
fn codes(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let b = ((i as u32 * 37 + seed as u32 * 101) % 256) as u8;
            if b == 0x7F || b == 0xFF {
                0x38
            } else {
                b
            }
        })
        .collect()
}

fn scales(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.03125 * (1 + i % 7) as f32).collect()
}

fn activation(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.017).sin() * 0.5 + 0.1)
        .collect()
}

/// The reference: dequantise the whole matrix, then take ordinary dots.
///
/// Deliberately the SLOW definition — the kernel's job is to compute this
/// without materialising it, so the test must materialise it.
fn reference(codes: &[u8], scales: &[f32], grid: Fp8Grid, x: &[f32]) -> Vec<f32> {
    let w = dequantize(codes, scales, grid).expect("dequantise");
    (0..grid.rows)
        .map(|r| {
            w[r * grid.cols..(r + 1) * grid.cols]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn rows_of<'a>(c: &'a [u8], s: &'a [f32], g: Fp8Grid) -> WeightRows<'a> {
    let (block_rows, block_cols) = g.tile().expect("tile");
    WeightRows::Fp8Block {
        codes: c,
        scales: s,
        block_rows,
        block_cols,
        scale_cols: g.scale_cols,
        row_in_tile: 0,
    }
}

/// Non-square everywhere: 6 rows x 8 cols against a 3 x 2 grid, so tiles
/// are 2 x 4 and neither axis can stand in for the other.
fn fixture() -> (Fp8Grid, Vec<u8>, Vec<f32>, Vec<f32>) {
    let g = Fp8Grid {
        rows: 6,
        cols: 8,
        scale_rows: 3,
        scale_cols: 2,
    };
    (
        g,
        codes(g.elements(), 1),
        scales(g.scales()),
        activation(g.cols),
    )
}

#[test]
fn the_kernel_computes_what_the_format_denotes() {
    let (g, c, s, x) = fixture();
    let mut out = vec![0.0f32; g.rows];
    FusedFp8Block.project_rows(rows_of(&c, &s, g), &x, &mut out);
    let want = reference(&c, &s, g, &x);
    for (i, (a, b)) in out.iter().zip(&want).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6 * b.abs().max(1.0),
            "row {i}: kernel {a} vs dequantise-then-dot {b}"
        );
    }
}

/// **The row-spanning scale, made to matter.**
///
/// Every row of a tile must read the SAME scale row, and consecutive
/// tiles must read different ones. A kernel that indexed the grid per
/// row (`scales[o * scale_cols..]`) would pass a 1-row tile and fail
/// here.
#[test]
fn rows_within_a_tile_share_a_scale_row_and_across_tiles_do_not() {
    let (g, c, mut s, x) = fixture();
    // Make grid row 0 unmistakable.
    s[0] = 1.0;
    s[1] = 1.0;
    let mut out = vec![0.0f32; g.rows];
    FusedFp8Block.project_rows(rows_of(&c, &s, g), &x, &mut out);
    let want = reference(&c, &s, g, &x);
    assert_eq!(out.len(), want.len());
    for (i, (a, b)) in out.iter().zip(&want).enumerate() {
        assert!((a - b).abs() <= 1e-6 * b.abs().max(1.0), "row {i}");
    }

    // And the control: perturbing ONLY grid row 0 must move rows 0-1 and
    // leave rows 2-5 untouched, or the tiling is not being read.
    let (_, _, base_s, _) = fixture();
    let mut base = vec![0.0f32; g.rows];
    FusedFp8Block.project_rows(rows_of(&c, &base_s, g), &x, &mut base);
    assert_ne!(out[0], base[0], "row 0 must follow grid row 0");
    assert_ne!(out[1], base[1], "row 1 must follow grid row 0");
    for r in 2..g.rows {
        assert_eq!(out[r], base[r], "row {r} must NOT follow grid row 0");
    }
}

/// **A partition that does not land on a tile boundary still computes the
/// same answer.**
///
/// This is the test the `row_in_tile` field exists for. Rows are cut at
/// 1, 3 and 5 against a 2-row tile, so every slab but the first starts
/// mid-tile. Without the offset each slab would read its scale slice from
/// row 0 and produce plausible, wrong numbers.
#[test]
fn an_off_boundary_row_partition_agrees_with_the_whole() {
    let (g, c, s, x) = fixture();
    let whole = reference(&c, &s, g, &x);
    let full = rows_of(&c, &s, g);

    for cut in [1usize, 2, 3, 4, 5] {
        let a = full.slice_rows(g.cols, 0, cut);
        let b = full.slice_rows(g.cols, cut, g.rows - cut);
        let mut out = vec![0.0f32; g.rows];
        let (head, tail) = out.split_at_mut(cut);
        FusedFp8Block.project_rows(a, &x, head);
        FusedFp8Block.project_rows(b, &x, tail);
        for (i, (got, want)) in out.iter().zip(&whole).enumerate() {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "cut at {cut}, row {i}: {got} vs {want}"
            );
        }
    }
}

/// A slab reports the bytes it actually holds — codes plus the scales it
/// spans — so a residency ledger is not flattered by the metadata it
/// still has to read.
#[test]
fn a_slab_prices_its_scales_with_its_codes() {
    let (g, c, s, _) = fixture();
    let full = rows_of(&c, &s, g);
    assert_eq!(full.bytes(), g.elements() + g.scales() * 4);

    // A two-row slab starting at row 2 spans exactly one grid row.
    let slab = full.slice_rows(g.cols, 2, 2);
    assert_eq!(slab.bytes(), 2 * g.cols + g.scale_cols * 4);

    // A three-row slab starting at row 1 spans TWO grid rows, and must
    // say so: rows 1 and 2 sit in different tiles.
    let straddle = full.slice_rows(g.cols, 1, 3);
    assert_eq!(straddle.bytes(), 3 * g.cols + 2 * g.scale_cols * 4);
}

#[test]
fn a_slab_reports_its_row_count_from_the_code_stride() {
    let (g, c, s, _) = fixture();
    assert_eq!(rows_of(&c, &s, g).rows(g.cols), g.rows);
}

/// Real GLM geometry, so the kernel is exercised at the widths it will
/// actually see rather than only at a toy shape.
#[test]
fn a_real_glm_tile_geometry_agrees_with_the_reference() {
    // A 256 x 512 slice of a dense FFN projection: 2 x 4 tiles of 128.
    let g = Fp8Grid {
        rows: 256,
        cols: 512,
        scale_rows: 2,
        scale_cols: 4,
    };
    assert_eq!(g.tile(), Ok((128, 128)));
    let (c, s, x) = (
        codes(g.elements(), 9),
        scales(g.scales()),
        activation(g.cols),
    );
    let mut out = vec![0.0f32; g.rows];
    FusedFp8Block.project_rows(rows_of(&c, &s, g), &x, &mut out);
    let want = reference(&c, &s, g, &x);
    let worst = out
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0f32, f32::max);
    assert!(worst <= 1e-6, "worst relative disagreement {worst:e}");
}
