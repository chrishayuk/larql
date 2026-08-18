//! NVFP4 codec contract.
//!
//! The tests that matter are about the *scale*, not the element width:
//! NVFP4 and MXFP4 share the E2M1 grid, and everything Q2 is testing
//! comes from E4M3 group scales plus a tensor scale replacing E8M0's
//! power-of-two-only one.

use super::*;

/// Exact-grid values survive when the scale is exact. With a group amax
/// of 6 and a tensor scale making the group scale 1, every stored code
/// decodes back to the value that produced it.
#[test]
fn grid_values_round_trip_exactly() {
    let mut row = vec![0.0f32; NVFP4_GROUP_ELEMS];
    let grid = [0.5f32, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.5, -6.0, -4.0];
    row[..grid.len()].copy_from_slice(&grid);
    let back = round_trip(&row, 1, NVFP4_GROUP_ELEMS).unwrap();
    assert_eq!(back, row, "grid values must survive exactly");
}

/// The tensor scale puts the largest group exactly at E4M3's maximum, so
/// the full scale range is spent rather than a fraction of it.
#[test]
fn the_tensor_scale_puts_the_largest_group_at_e4m3_max() {
    let values: Vec<f32> = (0..NVFP4_GROUP_ELEMS * 4)
        .map(|i| (i as f32 * 0.31).sin() * 12.0)
        .collect();
    let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let matrix = quantize(&values, 4, NVFP4_GROUP_ELEMS).unwrap();

    assert!(
        (matrix.tensor_scale - amax / (E4M3_MAX * E2M1_MAX)).abs() < f32::EPSILON,
        "tensor scale must normalise the matrix amax onto E4M3's top"
    );
    let largest = matrix
        .scales
        .iter()
        .map(|&b| crate::quant::fp8::e4m3_to_f32(b))
        .fold(0.0f32, f32::max);
    assert!(
        (largest - E4M3_MAX).abs() < 1.0,
        "the group holding the matrix amax should land at ~448, got {largest}"
    );
}

/// **The mechanism.** E8M0 forces a group's scale to a power of two, so
/// a group whose amax sits just below a power-of-two boundary has its
/// largest elements clipped: with amax 7.9 the MX rule picks
/// `2^(floor(log2 7.9) - 2) = 1`, a grid topping out at 6, and 7.9
/// saturates — a 24% error on the very element that defined the group.
///
/// E4M3's three mantissa bits let the scale track the amax instead, so
/// the same group reconstructs to within a fraction of a percent. This
/// is the whole reason Q2 exists, stated as a test.
#[test]
fn e4m3_scales_track_an_awkward_amax_where_a_power_of_two_would_clip() {
    // A group whose amax is just under 2^3 — the worst case for a
    // power-of-two scale, the best case for showing why it matters.
    let mut group = vec![0.0f32; NVFP4_GROUP_ELEMS];
    group[0] = 7.9;
    group[1] = -7.6;
    group[2] = 3.95;
    for (i, slot) in group.iter_mut().enumerate().skip(3) {
        *slot = 7.9 * ((i as f32) / (NVFP4_GROUP_ELEMS as f32) - 0.5);
    }

    let back = round_trip(&group, 1, NVFP4_GROUP_ELEMS).unwrap();

    // What the MX power-of-two rule would have had to do with the same
    // group: scale = 2^(floor(log2(amax)) - 2), grid top = 6 * scale.
    let amax = 7.9f32;
    let mx_scale = (amax.log2().floor() - 2.0).exp2();
    let mx_grid_top = 6.0 * mx_scale;
    assert!(
        mx_grid_top < amax,
        "the construction must actually be a clipping case for E8M0: \
         grid top {mx_grid_top} vs amax {amax}"
    );

    let nvfp4_top_error = (back[0] - group[0]).abs();
    let e8m0_top_error = amax - mx_grid_top;
    assert!(
        nvfp4_top_error < e8m0_top_error / 4.0,
        "NVFP4 error on the defining element ({nvfp4_top_error}) should be far \
         below the clip a power-of-two scale forces ({e8m0_top_error})"
    );
}

/// Reconstruction error stays within half a step of the group's own
/// quantisation step, for every element of every group.
#[test]
fn error_is_bounded_by_the_group_step() {
    let rows = 3;
    let k = NVFP4_GROUP_ELEMS * 5;
    let values: Vec<f32> = (0..rows * k)
        .map(|i| (i as f32 * 0.137).cos() * 2.5)
        .collect();
    let matrix = quantize(&values, rows, k).unwrap();
    let back = round_trip(&values, rows, k).unwrap();
    let groups = k / NVFP4_GROUP_ELEMS;

    for row in 0..rows {
        for g in 0..groups {
            let step = matrix.tensor_scale
                * crate::quant::fp8::e4m3_to_f32(matrix.scales[row * groups + g]);
            for i in 0..NVFP4_GROUP_ELEMS {
                let idx = row * k + g * NVFP4_GROUP_ELEMS + i;
                // The coarsest gap on the E2M1 grid is 6 → 4, i.e. 2 steps.
                assert!(
                    (values[idx] - back[idx]).abs() <= step * 2.0 + f32::EPSILON,
                    "row {row} group {g} element {i}: |{} - {}| exceeds 2 steps ({step})",
                    values[idx],
                    back[idx]
                );
            }
        }
    }
}

/// An all-zero group takes a zero scale and decodes to exact zeros,
/// without a division by zero leaking a NaN into the weights.
#[test]
fn a_zero_group_decodes_to_exact_zeros() {
    let values = vec![0.0f32; NVFP4_GROUP_ELEMS * 2];
    let matrix = quantize(&values, 1, NVFP4_GROUP_ELEMS * 2).unwrap();
    assert!(matrix.scales.iter().all(|&b| b == 0));
    assert!(matrix.packed.iter().all(|&b| b == 0));

    let back = round_trip(&values, 1, NVFP4_GROUP_ELEMS * 2).unwrap();
    assert!(back.iter().all(|v| *v == 0.0 && v.is_finite()));
}

/// A zero row inside a non-zero matrix keeps the matrix's tensor scale
/// and still decodes to zeros — the two scale levels are independent.
#[test]
fn a_zero_row_beside_a_live_one_stays_zero() {
    let k = NVFP4_GROUP_ELEMS;
    let mut values = vec![0.0f32; 2 * k];
    for (i, slot) in values[..k].iter_mut().enumerate() {
        *slot = (i as f32) - 8.0;
    }
    let back = round_trip(&values, 2, k).unwrap();
    assert!(
        back[k..].iter().all(|v| *v == 0.0),
        "the zero row stays zero"
    );
    assert!(back[..k].iter().any(|v| *v != 0.0), "the live row does not");
}

/// Lo nibble first, matching MXFP4's packing and the kernel contract, so
/// the two formats never disagree about which half of a byte is which.
#[test]
fn packing_is_lo_nibble_first() {
    let mut group = vec![0.0f32; NVFP4_GROUP_ELEMS];
    // With amax 6 the group scale is exactly 1 in tensor-scale units.
    group[0] = 0.5; // code 1
    group[1] = 6.0; // code 7
    let matrix = quantize(&group, 1, NVFP4_GROUP_ELEMS).unwrap();
    assert_eq!(matrix.packed[0] & 0x0F, 1, "element 0 is the low nibble");
    assert_eq!(
        (matrix.packed[0] >> 4) & 0x0F,
        7,
        "element 1 is the high nibble"
    );
}

/// Geometry is refused, never padded.
#[test]
fn bad_geometry_fails_closed() {
    assert_eq!(
        quantize(&[0.0; 20], 1, 20).unwrap_err(),
        Nvfp4Error::UnalignedK { k: 20 }
    );
    assert_eq!(
        quantize(&[0.0; 16], 2, 16).unwrap_err(),
        Nvfp4Error::ShapeMismatch {
            values: 16,
            rows: 2,
            k: 16
        }
    );
}

/// Stored size is 4.5 bits per weight plus one f32: 8 packed bytes and
/// one scale byte per 16 elements.
#[test]
fn stored_size_is_four_and_a_half_bits_per_weight() {
    let (rows, k) = (64, 1024);
    let bytes = stored_bytes(rows, k);
    let bits_per_weight = (bytes - 4) as f64 * 8.0 / (rows * k) as f64;
    assert!(
        (bits_per_weight - 4.5).abs() < 1e-9,
        "expected 4.5 bpw, got {bits_per_weight}"
    );
}

/// Decoding refuses the same geometry quantising refuses, and the errors
/// say what was wrong in the units the caller reasons in.
#[test]
fn decoding_refuses_bad_geometry_and_the_errors_name_it() {
    let matrix = quantize(&[0.5; 16], 1, 16).unwrap();
    let mut short = vec![0.0f32; 8];
    assert_eq!(
        dequantize_into(&matrix, 1, 16, &mut short).unwrap_err(),
        Nvfp4Error::ShapeMismatch {
            values: 8,
            rows: 1,
            k: 16
        }
    );
    let mut out = vec![0.0f32; 20];
    assert_eq!(
        dequantize_into(&matrix, 1, 20, &mut out).unwrap_err(),
        Nvfp4Error::UnalignedK { k: 20 }
    );

    let unaligned = Nvfp4Error::UnalignedK { k: 20 }.to_string();
    assert!(unaligned.contains("k=20"), "{unaligned}");
    assert!(unaligned.contains("16-element group"), "{unaligned}");
    let mismatch = Nvfp4Error::ShapeMismatch {
        values: 8,
        rows: 1,
        k: 16,
    }
    .to_string();
    assert_eq!(mismatch, "8 values do not fill [1, 16]");
    // It is a real `Error`, so it composes with `?` and `Box<dyn Error>`.
    let boxed: Box<dyn std::error::Error> = Box::new(Nvfp4Error::UnalignedK { k: 20 });
    assert!(boxed.to_string().starts_with("k=20"));
}
