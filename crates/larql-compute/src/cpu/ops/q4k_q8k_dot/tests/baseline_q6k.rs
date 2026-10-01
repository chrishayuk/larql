use super::super::quantize_x_to_q8k;
use super::*;
use crate::cpu::ops::q4_common::quantize_q6_k;

#[test]
fn baseline_q6k_matches_scalar_planar_layout() {
    for (rows, cols) in [(1, 256), (3, 512), (8, 1024)] {
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.0173).sin() * 1.7).collect();
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.027).cos() * 0.6)
            .collect();
        let w = quantize_q6_k(&w);
        let q8 = quantize_x_to_q8k(&x);
        let mut expected = vec![0.; rows];
        q6k_q8k_matvec_scalar(&mut expected, &q8, &w, rows, cols).unwrap();
        let mut actual = vec![0.; rows];
        // SAFETY: valid operands; this specialization uses only NEON.
        unsafe { q6k_q8k_matvec_neon_impl::<false>(&mut actual, &q8, &w, rows, cols).unwrap() };
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}
