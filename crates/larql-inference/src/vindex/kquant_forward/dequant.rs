use ndarray::Array2;

/// Dequantise a row-major Q4_K or Q6_K matrix into a dense f32 `Array2`.
///
/// The on-disk layout (`rows x cols` elements) must be stored contiguously
/// row-major and padded to a multiple of 256 elements per the k-quant
/// super-block size. Unknown formats panic; callers have already dispatched on
/// format via `larql_vindex::quant::registry`.
pub(super) fn dequantize_matrix(
    bytes: &[u8],
    format: &str,
    rows: usize,
    cols: usize,
) -> Array2<f32> {
    let n = rows * cols;
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let padded = n.div_ceil(block) * block;
    let info = larql_vindex::quant::registry::lookup(format)
        .unwrap_or_else(|| panic!("unsupported quant format in vindex: {format}"));
    let floats =
        (info.dequantize)(bytes, padded).unwrap_or_else(|e| panic!("{format} dequant failed: {e}"));
    let truncated = if floats.len() > n {
        floats[..n].to_vec()
    } else {
        floats
    };
    Array2::from_shape_vec((rows, cols), truncated).expect("shape mismatch dequantising Q4K matrix")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q4_K format with `rows*cols` a multiple of the 256-element super-block:
    /// `padded == n`, dequant returns exactly `n` floats, so the truncate
    /// path takes the `else { floats }` arm at L25.
    #[test]
    fn dequantize_matrix_q4k_padded_path_keeps_full_buffer() {
        let rows = 1;
        let cols = 256;
        let f32_in: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.001).collect();
        let bytes = larql_compute::cpu::ops::q4_common::quantize_q4_k(&f32_in);
        let out = dequantize_matrix(&bytes, "Q4_K", rows, cols);
        assert_eq!(out.shape(), &[rows, cols]);
    }

    /// `rows*cols` not a multiple of 256 — `padded > n`, so the dequantiser
    /// returns more floats than needed and the truncate-to-`n` branch at
    /// L23 fires.
    #[test]
    fn dequantize_matrix_q4k_unpadded_path_truncates() {
        let rows = 1;
        let cols = 200; // not a multiple of 256 → padded = 256
                        // Quantiser still needs a 256-multiple input; pad with zeros.
        let mut padded = vec![0.0f32; 256];
        for (i, slot) in padded.iter_mut().take(cols).enumerate() {
            *slot = i as f32 * 0.01;
        }
        let bytes = larql_compute::cpu::ops::q4_common::quantize_q4_k(&padded);
        let out = dequantize_matrix(&bytes, "Q4_K", rows, cols);
        assert_eq!(out.shape(), &[rows, cols]);
    }

    #[test]
    #[should_panic(expected = "unsupported quant format")]
    fn dequantize_matrix_panics_on_unknown_format() {
        let _ = dequantize_matrix(&[0u8; 144], "no_such_format", 1, 256);
    }
}
