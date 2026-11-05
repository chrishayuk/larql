//! Q4×Q8 matrix-vector multiply via C kernel.
//!
//! scores[N] = Q4[N, K] @ x[K]
//!
//! Internally quantizes x to Q8, then calls the C kernel with
//! ARM vdotq_s32 intrinsics. 0.95ms on 14.7MB matrix (M3 Max).

use super::q4_common::{q4_0_matvec_c, quantize_to_q8};

/// Q4 matvec: scores = Q4_matrix @ x.
/// Pre-quantizes x to Q8 internally.
pub fn dispatch(q4_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let (q8_x, q8_scales) = quantize_to_q8(x);
    dispatch_q8(q4_data, &q8_x, &q8_scales, num_rows, hidden)
}

/// Q4 matvec with pre-quantized Q8 input (avoids re-quantizing).
pub fn dispatch_q8(
    q4_data: &[u8],
    q8_x: &[i8],
    q8_scales: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Vec<f32> {
    // The C kernel assumes `hidden % 32 == 0` and reads exactly 18 bytes
    // per Q4_0 block × `num_rows` rows. Validate at the FFI boundary so a
    // malformed manifest fails on the Rust side with a clear message
    // rather than reading OOB inside `q4_0_matvec_c`.
    debug_assert_eq!(
        hidden % 32,
        0,
        "q4_0 matvec requires hidden to be a multiple of 32 (got {hidden})"
    );
    let blocks = hidden / 32;
    debug_assert_eq!(
        q4_data.len(),
        num_rows * blocks * 18,
        "q4_0 matvec: q4_data is {} bytes; expected {} \
         (num_rows={num_rows}, blocks={blocks}, 18 bytes/block)",
        q4_data.len(),
        num_rows * blocks * 18,
    );
    debug_assert_eq!(
        q8_x.len(),
        hidden,
        "q4_0 matvec: q8_x len {} != hidden {hidden}",
        q8_x.len()
    );
    debug_assert_eq!(
        q8_scales.len(),
        blocks,
        "q4_0 matvec: q8_scales len {} != hidden/32 = {blocks}",
        q8_scales.len()
    );

    let mut scores = vec![0.0f32; num_rows];
    let kernel: unsafe extern "C" fn(*const u8, *const i8, *const f32, *mut f32, usize, usize) =
        q4_0_matvec_c;
    #[cfg(target_arch = "aarch64")]
    let kernel = if std::arch::is_aarch64_feature_detected!("dotprod") {
        super::q4_common::q4_0_matvec_dotprod_c
    } else {
        kernel
    };
    // SAFETY: the validated operands match the C ABI dimensions; dispatch
    // selects the accelerated object only after detecting dotprod support.
    unsafe {
        kernel(
            q4_data.as_ptr(),
            q8_x.as_ptr(),
            q8_scales.as_ptr(),
            scores.as_mut_ptr(),
            num_rows,
            hidden,
        );
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::q4_common::quantize_q4_0;

    #[test]
    fn q4_matvec_produces_output() {
        let hidden = 256;
        let rows = 64;
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let matrix: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.001).cos())
            .collect();
        let q4 = quantize_q4_0(&matrix);
        let result = dispatch(&q4, &x, rows, hidden);
        assert_eq!(result.len(), rows);
        assert!(result.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn q4_matvec_zero_input() {
        let hidden = 256;
        let rows = 32;
        let x = vec![0.0f32; hidden];
        let matrix: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.001).cos())
            .collect();
        let q4 = quantize_q4_0(&matrix);
        let result = dispatch(&q4, &x, rows, hidden);
        assert!(result.iter().all(|&v| v.abs() < 0.01));
    }

    #[test]
    fn q4_c_baseline_matches_integer_reference_and_runtime_dispatch() {
        let (rows, hidden) = (3, 512);
        let weights: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.019).cos() * 0.7)
            .collect();
        let x: Vec<f32> = (0..hidden)
            .map(|i| (i as f32 * 0.013).sin() * 1.7)
            .collect();
        let q4 = quantize_q4_0(&weights);
        let (q8, scales) = quantize_to_q8(&x);
        let mut baseline = vec![0.0f32; rows];
        // SAFETY: all operands have the dimensions required by the C ABI.
        unsafe {
            q4_0_matvec_c(
                q4.as_ptr(),
                q8.as_ptr(),
                scales.as_ptr(),
                baseline.as_mut_ptr(),
                rows,
                hidden,
            );
        }
        for (row, actual) in baseline.iter().enumerate() {
            let mut expected = 0.0f32;
            for (b, scale) in scales.iter().enumerate() {
                let block = &q4[(row * scales.len() + b) * 18..][..18];
                let d =
                    super::super::q4_common::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let mut dot = 0i32;
                for j in 0..16 {
                    dot += ((block[2 + j] & 15) as i32 - 8) * q8[b * 32 + j] as i32;
                    dot += ((block[2 + j] >> 4) as i32 - 8) * q8[b * 32 + 16 + j] as i32;
                }
                expected += dot as f32 * (d * scale);
            }
            assert!(
                // C may contract its f32 multiply/add; the Rust oracle
                // rounds those operations separately.
                (actual - expected).abs() < 1e-6 * expected.abs().max(1.0),
                "row {row}: {actual} != {expected}"
            );
        }
        let dispatched = dispatch_q8(&q4, &q8, &scales, rows, hidden);
        assert_eq!(
            baseline.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dispatched.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}
