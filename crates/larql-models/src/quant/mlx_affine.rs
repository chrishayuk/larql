//! MLX affine quantization — packed U32 weights + per-group scales/biases.

/// Infer bits-per-weight from packed width and grouping.
pub fn infer_bits(packed_cols: usize, groups: usize, group_size: usize) -> Result<usize, String> {
    let packed_bits = packed_cols
        .checked_mul(32)
        .ok_or_else(|| "packed column count overflow".to_string())?;
    let cols = groups
        .checked_mul(group_size)
        .ok_or_else(|| "group shape overflow".to_string())?;

    if cols == 0 || packed_bits % cols != 0 {
        return Err(format!(
            "cannot infer MLX affine bits: packed_cols={packed_cols}, groups={groups}, group_size={group_size}"
        ));
    }

    let bits = packed_bits / cols;
    if bits == 0 || bits > 32 {
        return Err(format!("invalid MLX affine bit width: {bits}"));
    }
    Ok(bits)
}

/// Dequantize an MLX affine quantized 2D weight matrix stored as packed U32.
///
/// Layout matches `mlx.core.quantize(..., mode="affine")`:
/// - packed codes are concatenated little-endian within each row
/// - `scales` and `biases` are per-row, per-group
/// - dequantization is `bias + scale * code`
pub fn dequantize_u32_matrix_bytes(
    packed_bytes: &[u8],
    rows: usize,
    packed_cols: usize,
    scales: &[f32],
    biases: Option<&[f32]>,
    group_size: usize,
) -> Result<(Vec<f32>, usize), String> {
    if packed_bytes.len() % 4 != 0 {
        return Err("MLX affine packed weights must be U32-aligned".to_string());
    }

    let packed: Vec<u32> = packed_bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    if packed.len() != rows.saturating_mul(packed_cols) {
        return Err(format!(
            "MLX affine packed size mismatch: got {} words, expected {}",
            packed.len(),
            rows.saturating_mul(packed_cols)
        ));
    }

    if rows == 0 {
        return Ok((Vec::new(), 0));
    }

    if scales.len() % rows != 0 {
        return Err(format!(
            "MLX affine scales shape mismatch: {} values for {rows} rows",
            scales.len()
        ));
    }
    let groups = scales.len() / rows;
    let cols = groups
        .checked_mul(group_size)
        .ok_or_else(|| "MLX affine output shape overflow".to_string())?;
    let bits = infer_bits(packed_cols, groups, group_size)?;

    if let Some(biases) = biases {
        if biases.len() != scales.len() {
            return Err(format!(
                "MLX affine biases shape mismatch: {} values, expected {}",
                biases.len(),
                scales.len()
            ));
        }
    }

    let mask = if bits == 32 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };

    let mut out = vec![0.0; rows * cols];

    for row in 0..rows {
        let packed_row = &packed[row * packed_cols..(row + 1) * packed_cols];
        let row_scales = &scales[row * groups..(row + 1) * groups];
        let row_biases = biases.map(|b| &b[row * groups..(row + 1) * groups]);

        let mut out_col = 0usize;
        let mut acc = 0u64;
        let mut acc_bits = 0usize;

        for &word in packed_row {
            acc |= (word as u64) << acc_bits;
            acc_bits += 32;

            while acc_bits >= bits && out_col < cols {
                let group = out_col / group_size;
                let code = (acc & mask) as f32;
                let scale = row_scales[group];
                let bias = row_biases.map(|b| b[group]).unwrap_or(0.0);
                out[row * cols + out_col] = bias + scale * code;
                acc >>= bits;
                acc_bits -= bits;
                out_col += 1;
            }
        }

        if out_col != cols {
            return Err(format!(
                "MLX affine unpack ended early: row {row}, decoded {out_col}/{cols} values"
            ));
        }
    }

    Ok((out, cols))
}

#[cfg(test)]
mod tests {
    use super::dequantize_u32_matrix_bytes;

    fn pack_codes(codes: &[u32], bits: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut acc = 0u64;
        let mut acc_bits = 0usize;

        for &code in codes {
            acc |= (code as u64) << acc_bits;
            acc_bits += bits;

            while acc_bits >= 32 {
                out.push((acc & 0xFFFF_FFFF) as u32);
                acc >>= 32;
                acc_bits -= 32;
            }
        }

        if acc_bits > 0 {
            out.push(acc as u32);
        }

        out
    }

    fn approx_eq(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (idx, (&lhs, &rhs)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "mismatch at {idx}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn dequantizes_affine_u32_for_multiple_bit_widths() {
        for bits in [4usize, 5, 6, 8] {
            let rows = 2usize;
            let groups = 2usize;
            let group_size = 64usize;
            let cols = groups * group_size;
            let max_code = (1u32 << bits) - 1;

            let scales = vec![0.25, -0.5, 1.5, -0.125];
            let biases = vec![1.0, -3.0, 2.5, 7.0];

            let mut packed = Vec::new();
            let mut expected = Vec::new();

            for row in 0..rows {
                let codes: Vec<u32> = (0..cols)
                    .map(|col| ((row * cols + col) as u32) & max_code)
                    .collect();
                packed.extend(pack_codes(&codes, bits));

                for (col, code) in codes.into_iter().enumerate() {
                    let group = row * groups + (col / group_size);
                    expected.push(biases[group] + scales[group] * code as f32);
                }
            }

            let packed_bytes: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes()).collect();
            let (actual, actual_cols) = dequantize_u32_matrix_bytes(
                &packed_bytes,
                rows,
                cols * bits / 32,
                &scales,
                Some(&biases),
                group_size,
            )
            .unwrap();

            assert_eq!(actual_cols, cols);
            approx_eq(&actual, &expected);
        }
    }
}
