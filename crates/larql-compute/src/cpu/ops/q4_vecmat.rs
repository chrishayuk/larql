//! Q4 vector-matrix multiply via C kernel (scatter-accumulate).
//!
//! out[K] = activation[N] @ Q4[N, K]

use super::q4_common::q4_0_vecmat_c;

/// Elements in one Q4_0 block.
const Q4_0_BLOCK_ELEMENTS: usize = 32;
/// Bytes in one Q4_0 block: an f16 scale and 16 bytes of 4-bit values.
const Q4_0_BLOCK_BYTES: usize = 18;

/// Q4 vecmat: out = activation @ Q4_matrix.
///
/// # Panics
///
/// If `hidden` is not a multiple of 32, `activation` is not `intermediate`
/// long, or `q4_data` is not `intermediate * hidden / 32 * 18` bytes.
pub fn dispatch(
    activation: &[f32],
    q4_data: &[u8],
    intermediate: usize,
    hidden: usize,
) -> Vec<f32> {
    // The C kernel reads `intermediate` activations and, for each one that
    // is non-zero, `hidden / 32` blocks of 18 bytes. These are checked in
    // every build: this is a safe function, and with a short slice the
    // kernel would read past the end of it.
    assert_eq!(
        hidden % Q4_0_BLOCK_ELEMENTS,
        0,
        "q4_0 vecmat requires hidden to be a multiple of {Q4_0_BLOCK_ELEMENTS} (got {hidden})"
    );
    assert_eq!(
        activation.len(),
        intermediate,
        "q4_0 vecmat: activation has {} elements; expected intermediate = {intermediate}",
        activation.len()
    );
    let expected_bytes = intermediate * (hidden / Q4_0_BLOCK_ELEMENTS) * Q4_0_BLOCK_BYTES;
    assert_eq!(
        q4_data.len(),
        expected_bytes,
        "q4_0 vecmat: q4_data is {} bytes; expected {expected_bytes} \
         (intermediate={intermediate}, hidden={hidden})",
        q4_data.len()
    );

    let mut out = vec![0.0f32; hidden];
    // SAFETY: the lengths checked above are the ones the kernel reads, and
    // `out` holds the `hidden` elements it writes.
    unsafe {
        q4_0_vecmat_c(
            activation.as_ptr(),
            q4_data.as_ptr(),
            out.as_mut_ptr(),
            intermediate,
            hidden,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::q4_common::quantize_q4_0;

    #[test]
    fn q4_vecmat_produces_output() {
        let hidden = 256;
        let inter = 128;
        let act: Vec<f32> = (0..inter)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let matrix: Vec<f32> = (0..inter * hidden)
            .map(|i| (i as f32 * 0.001).cos())
            .collect();
        let q4 = quantize_q4_0(&matrix);
        let result = dispatch(&act, &q4, inter, hidden);
        assert_eq!(result.len(), hidden);
        assert!(result.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn q4_vecmat_zero_activation() {
        let hidden = 256;
        let inter = 64;
        let act = vec![0.0f32; inter];
        let matrix: Vec<f32> = (0..inter * hidden)
            .map(|i| (i as f32 * 0.001).cos())
            .collect();
        let q4 = quantize_q4_0(&matrix);
        let result = dispatch(&act, &q4, inter, hidden);
        assert!(result.iter().all(|&v| v.abs() < 0.01));
    }
}
