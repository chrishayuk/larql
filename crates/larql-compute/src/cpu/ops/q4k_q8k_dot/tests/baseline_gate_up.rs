use super::super::quantize_x_to_q8k;
use super::*;
use crate::cpu::ops::q4_common::quantize_q4_k;

#[test]
fn baseline_fused_gate_up_matches_separate_scalar_matvecs() {
    let (rows, cols) = (3, 512);
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.015).sin() * 1.4).collect();
    let g: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.011).cos() * 0.4)
        .collect();
    let u: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.013).sin() * 0.3)
        .collect();
    let (g, u) = (quantize_q4_k(&g), quantize_q4_k(&u));
    let q8 = quantize_x_to_q8k(&x);
    let (mut expected_g, mut expected_u) = (vec![0.; rows], vec![0.; rows]);
    q4k_q8k_matvec_scalar(&mut expected_g, &q8, &g, rows, cols).unwrap();
    q4k_q8k_matvec_scalar(&mut expected_u, &q8, &u, rows, cols).unwrap();
    let (mut actual_g, mut actual_u) = (vec![0.; rows], vec![0.; rows]);
    // SAFETY: valid operands; this specialization uses only NEON.
    unsafe {
        q4k_q8k_gate_up_neon_impl::<false>(&mut actual_g, &mut actual_u, &q8, &g, &u, rows, cols)
            .unwrap()
    };
    for (actual, expected) in [(actual_g, expected_g), (actual_u, expected_u)] {
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}
