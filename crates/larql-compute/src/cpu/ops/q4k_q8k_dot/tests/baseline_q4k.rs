use super::super::{q4k_q8k_matvec_scalar, quantize_x_to_q8k};
use super::*;
use crate::cpu::ops::q4_common::quantize_q4_k;

#[test]
fn baseline_dot_preserves_signed_extremes_and_lane_groups() {
    use std::arch::aarch64::*;
    let a = [
        -128, -128, -128, -128, 127, 127, 127, 127, -128, 127, -1, 1, 3, -5, 7, -11,
    ];
    let b = [
        -128, -128, -128, -128, 127, 127, 127, 127, 127, -128, 1, -1, -13, 17, -19, 23,
    ];
    let initial = [7, -31, 100, -200];
    let mut actual = [0; 4];
    // SAFETY: NEON is available and every load/store spans exactly 16 bytes.
    unsafe {
        vst1q_s32(
            actual.as_mut_ptr(),
            dot_acc_neon(
                vld1q_s32(initial.as_ptr()),
                vld1q_s8(a.as_ptr()),
                vld1q_s8(b.as_ptr()),
            ),
        );
    }
    for lane in 0..4 {
        let expected = initial[lane]
            + (lane * 4..lane * 4 + 4)
                .map(|i| i32::from(a[i]) * i32::from(b[i]))
                .sum::<i32>();
        assert_eq!(actual[lane], expected);
    }
}

#[test]
fn baseline_q4k_matches_scalar_including_odd_row_tail() {
    for (rows, cols) in [(1, 256), (3, 512), (8, 1024)] {
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.013).sin() * 1.7).collect();
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.007).cos() * 0.6)
            .collect();
        let w = quantize_q4_k(&w);
        let q8 = quantize_x_to_q8k(&x);
        let mut expected = vec![0.; rows];
        q4k_q8k_matvec_scalar(&mut expected, &q8, &w, rows, cols).unwrap();
        let mut single = vec![0.; rows];
        let mut paired = vec![0.; rows];
        // Exercise the baseline even on an SDOT-capable test machine.
        // SAFETY: operands have valid shapes and false requires only NEON.
        unsafe {
            q4k_q8k_matvec_neon_impl::<false>(&mut single, &q8, &w, rows, cols).unwrap();
            q4k_q8k_matvec_neon_2row_impl::<false>(&mut paired, &q8, &w, rows, cols).unwrap();
        }
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(bits(single), bits(expected.clone()));
        assert_eq!(bits(paired), bits(expected));
    }
}
