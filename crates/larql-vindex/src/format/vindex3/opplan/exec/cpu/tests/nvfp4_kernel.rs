//! The CPU NVFP4 kernel computes what the format denotes.
//!
//! Three computations of one row: the NEON kernel `FusedNvfp4` runs, its
//! portable definition, and the shared scalar kernel that is bit-exact
//! against decode-then-multiply. The first two sum a group in a different
//! association from the third, so they agree within float reassociation
//! — the standard `FusedQ8`/`FusedQ4` are held to — not bit for bit.
//! Every code and every finite E4M3 scale is exercised: a nibble order,
//! sign or scale-table mistake moves results far outside that tolerance.

use super::super::kernels::{e4m3_steps, nvfp4_row_dot_portable, FusedNvfp4};
use super::super::projector::{DenseProjector, WeightRows};
use larql_models::quant::nvfp4::{NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS};

/// A deterministic byte stream that visits every value.
fn bytes(n: usize, seed: u32, skip_nan_scales: bool) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = (state >> 24) as u8;
            // E4M3 has NaN at 0x7f/0xff; a scale there makes every
            // computation NaN, which says nothing about agreement.
            if skip_nan_scales && b & 0x7f == 0x7f {
                0x38
            } else {
                b
            }
        })
        .collect()
}

fn activations(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
        .collect()
}

/// Sum of |term| for one row: the scale a reassociation error is measured
/// against, so cancellation cannot make the tolerance vanish.
fn magnitude(packed: &[u8], scales: &[u8], tensor_scale: f32, x: &[f32]) -> f32 {
    let steps = e4m3_steps();
    let mut m = 0.0f32;
    for (g, &s) in scales.iter().enumerate() {
        for b in 0..NVFP4_GROUP_BYTES {
            let byte = packed[g * NVFP4_GROUP_BYTES + b];
            for (nib, e) in [(byte & 0x0f, 2 * b), (byte >> 4, 2 * b + 1)] {
                let v = larql_models::quant::fp4::e2m1_to_f32(nib);
                m += (tensor_scale * steps[s as usize] * v * x[g * NVFP4_GROUP_ELEMS + e]).abs();
            }
        }
    }
    m
}

#[test]
fn the_nvfp4_kernel_computes_what_the_format_denotes() {
    let tensor_scale = 0.0371_f32;
    for (n, k) in [(1, 16), (3, 32), (5, 48), (7, 112), (4, 2560), (3, 8192)] {
        let groups = k / NVFP4_GROUP_ELEMS;
        let packed = bytes(n * groups * NVFP4_GROUP_BYTES, (n * k) as u32, false);
        let scales = bytes(n * groups, (n + k) as u32, true);
        let x = activations(k);

        let mut got = vec![0.0f32; n];
        FusedNvfp4.project_rows(
            WeightRows::Nvfp4 {
                packed: &packed,
                scales: &scales,
                tensor_scale,
            },
            &x,
            &mut got,
        );
        let reference =
            larql_compute::cpu::nvfp4_gemv::nvfp4_gemv(&packed, &scales, tensor_scale, &x, n, k)
                .expect("reference geometry");

        for row in 0..n {
            let p = &packed[row * groups * NVFP4_GROUP_BYTES..][..groups * NVFP4_GROUP_BYTES];
            let s = &scales[row * groups..][..groups];
            let portable = nvfp4_row_dot_portable(p, s, tensor_scale, e4m3_steps(), &x);
            let tol = magnitude(p, s, tensor_scale, &x) * 1e-6 + 1e-6;
            for (what, value) in [("kernel", got[row]), ("portable", portable)] {
                assert!(
                    (value - reference[row]).abs() <= tol,
                    "[{n},{k}] row {row}: {what} {value} vs bit-exact reference {} (tol {tol})",
                    reference[row]
                );
            }
        }
    }
}

/// A single set nibble moves exactly the element it names, with the value
/// and sign the E2M1 table gives — nibble order and sign are not
/// reassociation-sized mistakes, so this pins them exactly.
#[test]
fn each_nibble_reaches_its_own_element_with_its_own_value() {
    let k = 16;
    let one = 0x38u8; // E4M3 1.0
    for elem in 0..k {
        for code in 0u8..16 {
            let mut packed = vec![0u8; NVFP4_GROUP_BYTES];
            packed[elem / 2] = if elem % 2 == 0 { code } else { code << 4 };
            let mut x = vec![0.0f32; k];
            x[elem] = 1.0;
            let mut got = [0.0f32; 1];
            FusedNvfp4.project_rows(
                WeightRows::Nvfp4 {
                    packed: &packed,
                    scales: &[one],
                    tensor_scale: 1.0,
                },
                &x,
                &mut got,
            );
            let want = larql_models::quant::fp4::e2m1_to_f32(code);
            assert_eq!(got[0], want, "element {elem}, code {code:#x}");
        }
    }
}
